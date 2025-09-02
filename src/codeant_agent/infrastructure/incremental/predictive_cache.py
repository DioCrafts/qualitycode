"""
Sistema de cache predictivo para el Sistema de Análisis Incremental.

Este módulo implementa predicción de accesos futuros y cache warming
proactivo para optimizar el rendimiento.
"""

import asyncio
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import json

from ...domain.entities.incremental import (
    AccessPrediction, PredictionSource, CacheKey,
    CacheWarmingResult, CacheWarmingStatus
)
from ...domain.value_objects.incremental_metrics import (
    WarmingMetrics, PredictionAccuracy
)
from ...domain.services.incremental_service import PredictiveCacheService
from ...application.ports.incremental_ports import (
    PredictionEngineOutputPort, AnalysisEngineOutputPort,
    CacheStorageOutputPort, MetricsCollectorOutputPort
)
from .incremental_config import IncrementalConfig


class AccessPattern:
    """Patrón de acceso a cache."""
    
    def __init__(self):
        self.access_times: List[datetime] = []
        self.access_count: int = 0
        self.last_access: Optional[datetime] = None
        self.average_interval: Optional[timedelta] = None
        self.access_days: Set[int] = set()  # Días de la semana
        self.access_hours: Counter = Counter()  # Horas del día
        self.file_type: Optional[str] = None
        self.analysis_type: Optional[str] = None


class PredictiveCacheSystem(PredictiveCacheService, PredictionEngineOutputPort):
    """Sistema de cache predictivo con ML."""
    
    def __init__(
        self,
        config: IncrementalConfig,
        cache_storage: CacheStorageOutputPort,
        analysis_engine: AnalysisEngineOutputPort,
        metrics_collector: MetricsCollectorOutputPort
    ):
        self.config = config
        self.cache_storage = cache_storage
        self.analysis_engine = analysis_engine
        self.metrics_collector = metrics_collector
        
        # Historial de accesos
        self.access_history: List[Tuple[str, datetime, Dict[str, Any]]] = []
        
        # Patrones de acceso por archivo
        self.access_patterns: Dict[str, AccessPattern] = defaultdict(AccessPattern)
        
        # Modelo de predicción
        self.prediction_model: Optional[LinearRegression] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self._model_trained = False
        
        # Cache de predicciones
        self.prediction_cache: Dict[str, List[AccessPrediction]] = {}
        
        # Métricas de predicción
        self.prediction_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        # Cola de warming
        self.warming_queue: asyncio.Queue = asyncio.Queue()
        self._warming_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Inicializar el sistema predictivo."""
        # Cargar modelo si existe
        await self._load_model()
        
        # Iniciar worker de cache warming
        self._warming_task = asyncio.create_task(self._warming_worker())
        
        # Cargar historial si existe
        await self._load_access_history()
    
    async def shutdown(self):
        """Apagar el sistema predictivo."""
        # Detener worker
        if self._warming_task:
            self._warming_task.cancel()
            try:
                await self._warming_task
            except asyncio.CancelledError:
                pass
        
        # Guardar modelo y datos
        await self._save_model()
        await self._save_access_history()
    
    # Implementación de PredictiveCacheService
    
    async def predict_future_accesses(
        self,
        time_horizon: timedelta
    ) -> List[AccessPrediction]:
        """
        Predecir accesos futuros al cache.
        
        Args:
            time_horizon: Horizonte temporal de predicción
            
        Returns:
            Lista de predicciones de acceso
        """
        predictions = []
        current_time = datetime.now()
        end_time = current_time + time_horizon
        
        # Actualizar patrones con historial reciente
        await self._update_access_patterns()
        
        # Estrategia 1: Predicción basada en patrones temporales
        time_based = await self._predict_based_on_time_patterns(
            current_time, end_time
        )
        predictions.extend(time_based)
        
        # Estrategia 2: Predicción basada en secuencias
        sequence_based = await self._predict_based_on_sequences(
            current_time, end_time
        )
        predictions.extend(sequence_based)
        
        # Estrategia 3: Predicción basada en ML (si está entrenado)
        if self._model_trained:
            ml_based = await self._predict_with_ml_model(
                current_time, end_time
            )
            predictions.extend(ml_based)
        
        # Estrategia 4: Predicción basada en correlaciones
        correlation_based = await self._predict_based_on_correlations(
            current_time, end_time
        )
        predictions.extend(correlation_based)
        
        # Deduplicar y ordenar por confianza
        predictions = self._deduplicate_predictions(predictions)
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        
        # Registrar métricas
        self.prediction_metrics['total_predictions'] += len(predictions)
        
        return predictions
    
    async def warm_cache(
        self,
        predictions: List[AccessPrediction]
    ) -> int:
        """
        Calentar cache basado en predicciones.
        
        Args:
            predictions: Lista de predicciones
            
        Returns:
            Número de items calentados
        """
        warmed_count = 0
        
        # Filtrar predicciones por confianza
        high_confidence = [
            p for p in predictions
            if p.confidence >= self.config.min_confidence_for_warming
        ]
        
        # Ordenar por tiempo predicho
        high_confidence.sort(key=lambda p: p.predicted_access_time)
        
        # Calentar cache para cada predicción
        for prediction in high_confidence:
            # Verificar si ya está en cache
            cache_key = CacheKey(
                path=prediction.file_path,
                analysis_type=prediction.analysis_type,
                hash=None
            )
            
            existing = await self.cache_storage.retrieve_from_cache(
                self._cache_key_to_string(cache_key),
                level=self.config.default_warming_cache_level
            )
            
            if not existing:
                # Añadir a cola de warming
                await self.warming_queue.put(prediction)
                warmed_count += 1
        
        return warmed_count
    
    async def analyze_access_patterns(
        self,
        time_window: timedelta
    ) -> Dict[str, Any]:
        """
        Analizar patrones de acceso al cache.
        
        Args:
            time_window: Ventana temporal de análisis
            
        Returns:
            Análisis de patrones
        """
        cutoff_time = datetime.now() - time_window
        recent_accesses = [
            (key, time, metadata)
            for key, time, metadata in self.access_history
            if time >= cutoff_time
        ]
        
        analysis = {
            'total_accesses': len(recent_accesses),
            'unique_files': len(set(key for key, _, _ in recent_accesses)),
            'peak_hours': self._analyze_peak_hours(recent_accesses),
            'access_frequency': self._analyze_access_frequency(recent_accesses),
            'file_correlations': await self._analyze_file_correlations(recent_accesses),
            'prediction_accuracy': self._calculate_prediction_accuracy()
        }
        
        return analysis
    
    async def optimize_warming_strategy(self) -> Dict[str, Any]:
        """
        Optimizar estrategia de cache warming.
        
        Returns:
            Resultados de optimización
        """
        # Analizar rendimiento actual
        current_accuracy = self._calculate_prediction_accuracy()
        
        # Ajustar parámetros
        optimizations = {}
        
        # Optimizar umbral de confianza
        if current_accuracy < self.config.target_prediction_accuracy:
            # Aumentar umbral si hay muchos falsos positivos
            if self.prediction_metrics['false_positives'] > self.prediction_metrics['correct_predictions']:
                self.config.min_confidence_for_warming *= 1.1
                optimizations['confidence_threshold'] = 'increased'
        else:
            # Reducir umbral si la precisión es alta
            self.config.min_confidence_for_warming *= 0.95
            optimizations['confidence_threshold'] = 'decreased'
        
        # Optimizar ventana de predicción
        avg_lead_time = await self._calculate_average_lead_time()
        if avg_lead_time:
            self.config.default_prediction_window = int(avg_lead_time.total_seconds() / 60)
            optimizations['prediction_window'] = f'{self.config.default_prediction_window} minutes'
        
        # Re-entrenar modelo si es necesario
        if self._should_retrain_model():
            await self._train_prediction_model()
            optimizations['model_retrained'] = True
        
        return {
            'current_accuracy': current_accuracy,
            'optimizations': optimizations,
            'new_parameters': {
                'confidence_threshold': self.config.min_confidence_for_warming,
                'prediction_window': self.config.default_prediction_window
            }
        }
    
    async def evaluate_prediction_accuracy(
        self,
        time_window: timedelta
    ) -> float:
        """
        Evaluar precisión de predicciones.
        
        Args:
            time_window: Ventana temporal para evaluación
            
        Returns:
            Score de precisión (0.0 - 1.0)
        """
        cutoff_time = datetime.now() - time_window
        
        # Obtener predicciones realizadas en la ventana
        relevant_predictions = []
        for key, predictions in self.prediction_cache.items():
            for pred in predictions:
                if pred.timestamp >= cutoff_time:
                    relevant_predictions.append((key, pred))
        
        if not relevant_predictions:
            return 0.0
        
        # Verificar si se cumplieron las predicciones
        correct = 0
        for key, prediction in relevant_predictions:
            # Buscar acceso real cercano al tiempo predicho
            actual_access = self._find_actual_access(
                key,
                prediction.predicted_access_time,
                tolerance=timedelta(minutes=5)
            )
            
            if actual_access:
                correct += 1
                self.prediction_metrics['correct_predictions'] += 1
            else:
                self.prediction_metrics['false_positives'] += 1
        
        accuracy = correct / len(relevant_predictions)
        return accuracy
    
    # Implementación de PredictionEngineOutputPort
    
    async def get_historical_access_patterns(
        self,
        time_window: timedelta
    ) -> List[Dict[str, Any]]:
        """Obtener patrones históricos de acceso."""
        cutoff_time = datetime.now() - time_window
        patterns = []
        
        for key, pattern in self.access_patterns.items():
            if pattern.last_access and pattern.last_access >= cutoff_time:
                patterns.append({
                    'key': key,
                    'access_count': pattern.access_count,
                    'average_interval': pattern.average_interval.total_seconds() if pattern.average_interval else None,
                    'peak_hours': pattern.access_hours.most_common(3),
                    'access_days': list(pattern.access_days),
                    'file_type': pattern.file_type,
                    'analysis_type': pattern.analysis_type
                })
        
        return patterns
    
    async def train_prediction_model(
        self,
        training_data: List[Dict[str, Any]]
    ) -> Any:
        """Entrenar modelo de predicción."""
        if not training_data:
            training_data = await self._prepare_training_data()
        
        if len(training_data) < self.config.min_training_samples:
            return None
        
        # Preparar features y labels
        X, y = self._extract_features_and_labels(training_data)
        
        # Escalar features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Entrenar modelo
        self.prediction_model = LinearRegression()
        self.prediction_model.fit(X_scaled, y)
        
        self._model_trained = True
        
        # Guardar modelo
        await self._save_model()
        
        return self.prediction_model
    
    async def generate_predictions(
        self,
        model: Any,
        context: Dict[str, Any]
    ) -> List[AccessPrediction]:
        """Generar predicciones usando el modelo."""
        if not self._model_trained or not self.prediction_model:
            return []
        
        # Preparar features del contexto
        features = self._context_to_features(context)
        features_scaled = self.feature_scaler.transform([features])
        
        # Predecir
        prediction_value = self.prediction_model.predict(features_scaled)[0]
        
        # Convertir predicción a AccessPrediction
        prediction = AccessPrediction(
            file_path=context.get('file_path', ''),
            analysis_type=context.get('analysis_type', 'full'),
            predicted_access_time=datetime.now() + timedelta(minutes=prediction_value),
            confidence=0.7,  # Basado en score del modelo
            source=PredictionSource.ML_MODEL,
            timestamp=datetime.now()
        )
        
        return [prediction]
    
    async def evaluate_prediction_accuracy(
        self,
        predictions: List[AccessPrediction],
        actual_accesses: List[str]
    ) -> float:
        """Evaluar precisión de un conjunto de predicciones."""
        if not predictions:
            return 0.0
        
        correct = 0
        for prediction in predictions:
            pred_key = f"{prediction.file_path}:{prediction.analysis_type}"
            if pred_key in actual_accesses:
                correct += 1
        
        return correct / len(predictions)
    
    # Métodos auxiliares privados
    
    async def _warming_worker(self):
        """Worker para calentar cache de forma asíncrona."""
        while True:
            try:
                # Obtener predicción de la cola
                prediction = await self.warming_queue.get()
                
                # Realizar análisis y cachear
                result = await self._warm_single_cache_entry(prediction)
                
                # Registrar resultado
                if result.status == CacheWarmingStatus.SUCCESS:
                    await self.metrics_collector.record_cache_hit(
                        cache_level=self.config.default_warming_cache_level,
                        key=f"{prediction.file_path}:{prediction.analysis_type}"
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                pass
    
    async def _warm_single_cache_entry(
        self,
        prediction: AccessPrediction
    ) -> CacheWarmingResult:
        """Calentar una entrada específica del cache."""
        start_time = datetime.now()
        
        try:
            # Realizar análisis
            result = await self.analysis_engine.run_partial_analysis(
                file_path=Path(prediction.file_path),
                analysis_type=prediction.analysis_type,
                scope=None  # Scope completo
            )
            
            # Cachear resultado
            cache_key = f"{prediction.file_path}:{prediction.analysis_type}"
            success = await self.cache_storage.store_in_l1(
                key=cache_key,
                value=result,
                ttl=self.config.default_cache_ttl
            )
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return CacheWarmingResult(
                prediction=prediction,
                status=CacheWarmingStatus.SUCCESS if success else CacheWarmingStatus.FAILED,
                warming_time_ms=int(duration),
                cache_size_bytes=0  # TODO: calcular tamaño real
            )
            
        except Exception as e:
            return CacheWarmingResult(
                prediction=prediction,
                status=CacheWarmingStatus.FAILED,
                warming_time_ms=0,
                cache_size_bytes=0,
                error_message=str(e)
            )
    
    async def _update_access_patterns(self):
        """Actualizar patrones de acceso con historial reciente."""
        for key, access_time, metadata in self.access_history[-1000:]:
            pattern = self.access_patterns[key]
            
            # Actualizar tiempos
            pattern.access_times.append(access_time)
            pattern.access_count += 1
            pattern.last_access = access_time
            
            # Actualizar patrones temporales
            pattern.access_days.add(access_time.weekday())
            pattern.access_hours[access_time.hour] += 1
            
            # Calcular intervalo promedio
            if len(pattern.access_times) > 1:
                intervals = [
                    pattern.access_times[i+1] - pattern.access_times[i]
                    for i in range(len(pattern.access_times)-1)
                ]
                pattern.average_interval = sum(intervals, timedelta()) / len(intervals)
            
            # Extraer metadata
            if metadata:
                pattern.file_type = metadata.get('file_type')
                pattern.analysis_type = metadata.get('analysis_type')
    
    async def _predict_based_on_time_patterns(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[AccessPrediction]:
        """Predecir basado en patrones temporales."""
        predictions = []
        
        for key, pattern in self.access_patterns.items():
            if not pattern.average_interval or not pattern.last_access:
                continue
            
            # Predecir próximo acceso basado en intervalo promedio
            next_access = pattern.last_access + pattern.average_interval
            
            if start_time <= next_access <= end_time:
                # Calcular confianza basada en regularidad
                confidence = self._calculate_pattern_confidence(pattern)
                
                # Parsear key
                parts = key.split(':')
                file_path = parts[0] if parts else key
                analysis_type = parts[1] if len(parts) > 1 else 'full'
                
                prediction = AccessPrediction(
                    file_path=file_path,
                    analysis_type=analysis_type,
                    predicted_access_time=next_access,
                    confidence=confidence,
                    source=PredictionSource.TIME_PATTERN,
                    timestamp=datetime.now()
                )
                predictions.append(prediction)
        
        return predictions
    
    async def _predict_based_on_sequences(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[AccessPrediction]:
        """Predecir basado en secuencias de acceso."""
        predictions = []
        
        # Analizar secuencias en el historial
        sequences = self._extract_access_sequences()
        
        # Buscar patrones secuenciales
        for sequence in sequences:
            if len(sequence) < 2:
                continue
            
            # Verificar si la secuencia actual coincide
            recent_accesses = [key for key, _, _ in self.access_history[-len(sequence):]]
            
            if recent_accesses[:-1] == sequence[:-1]:
                # Predecir siguiente elemento
                next_key = sequence[-1]
                
                # Estimar tiempo basado en intervalos históricos
                avg_interval = timedelta(minutes=5)  # Default
                pattern = self.access_patterns.get(next_key)
                if pattern and pattern.average_interval:
                    avg_interval = pattern.average_interval
                
                predicted_time = datetime.now() + avg_interval
                
                if start_time <= predicted_time <= end_time:
                    parts = next_key.split(':')
                    prediction = AccessPrediction(
                        file_path=parts[0],
                        analysis_type=parts[1] if len(parts) > 1 else 'full',
                        predicted_access_time=predicted_time,
                        confidence=0.6,  # Confianza media para secuencias
                        source=PredictionSource.SEQUENCE_PATTERN,
                        timestamp=datetime.now()
                    )
                    predictions.append(prediction)
        
        return predictions
    
    async def _predict_with_ml_model(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[AccessPrediction]:
        """Predecir usando modelo de ML."""
        predictions = []
        
        # Generar contextos para predicción
        contexts = await self._generate_prediction_contexts()
        
        for context in contexts:
            pred_list = await self.generate_predictions(
                self.prediction_model,
                context
            )
            
            for pred in pred_list:
                if start_time <= pred.predicted_access_time <= end_time:
                    predictions.append(pred)
        
        return predictions
    
    async def _predict_based_on_correlations(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[AccessPrediction]:
        """Predecir basado en correlaciones entre archivos."""
        predictions = []
        
        # Analizar correlaciones recientes
        correlations = await self._analyze_file_correlations(
            self.access_history[-500:]
        )
        
        # Buscar accesos recientes
        recent_keys = [key for key, time, _ in self.access_history[-10:]
                      if (datetime.now() - time).total_seconds() < 300]
        
        for recent_key in recent_keys:
            if recent_key in correlations:
                for correlated_key, correlation_strength in correlations[recent_key]:
                    if correlation_strength > 0.5:
                        # Predecir acceso correlacionado
                        parts = correlated_key.split(':')
                        prediction = AccessPrediction(
                            file_path=parts[0],
                            analysis_type=parts[1] if len(parts) > 1 else 'full',
                            predicted_access_time=datetime.now() + timedelta(minutes=2),
                            confidence=correlation_strength * 0.8,
                            source=PredictionSource.USER_BEHAVIOR,
                            timestamp=datetime.now()
                        )
                        
                        if start_time <= prediction.predicted_access_time <= end_time:
                            predictions.append(prediction)
        
        return predictions
    
    def _deduplicate_predictions(
        self,
        predictions: List[AccessPrediction]
    ) -> List[AccessPrediction]:
        """Deduplicar predicciones manteniendo la de mayor confianza."""
        unique_predictions = {}
        
        for pred in predictions:
            key = f"{pred.file_path}:{pred.analysis_type}"
            
            if key not in unique_predictions or pred.confidence > unique_predictions[key].confidence:
                unique_predictions[key] = pred
        
        return list(unique_predictions.values())
    
    def _cache_key_to_string(self, key: CacheKey) -> str:
        """Convertir CacheKey a string."""
        return f"{key.path}:{key.analysis_type}"
    
    def _calculate_pattern_confidence(self, pattern: AccessPattern) -> float:
        """Calcular confianza basada en regularidad del patrón."""
        if pattern.access_count < 3:
            return 0.3
        
        # Calcular varianza en intervalos
        if len(pattern.access_times) < 2:
            return 0.3
        
        intervals = [
            (pattern.access_times[i+1] - pattern.access_times[i]).total_seconds()
            for i in range(len(pattern.access_times)-1)
        ]
        
        if not intervals:
            return 0.3
        
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Menor varianza = mayor confianza
        cv = std_interval / mean_interval if mean_interval > 0 else 1.0
        confidence = max(0.3, min(0.9, 1.0 - cv))
        
        # Ajustar por frecuencia
        if pattern.access_count > 10:
            confidence *= 1.1
        
        return min(confidence, 0.95)
    
    def _analyze_peak_hours(
        self,
        accesses: List[Tuple[str, datetime, Dict[str, Any]]]
    ) -> List[int]:
        """Analizar horas pico de acceso."""
        hour_counts = Counter()
        
        for _, access_time, _ in accesses:
            hour_counts[access_time.hour] += 1
        
        # Retornar top 3 horas
        return [hour for hour, _ in hour_counts.most_common(3)]
    
    def _analyze_access_frequency(
        self,
        accesses: List[Tuple[str, datetime, Dict[str, Any]]]
    ) -> Dict[str, int]:
        """Analizar frecuencia de acceso por archivo."""
        file_counts = Counter()
        
        for key, _, _ in accesses:
            parts = key.split(':')
            file_path = parts[0] if parts else key
            file_counts[file_path] += 1
        
        return dict(file_counts.most_common(10))
    
    async def _analyze_file_correlations(
        self,
        accesses: List[Tuple[str, datetime, Dict[str, Any]]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Analizar correlaciones entre accesos a archivos."""
        correlations = defaultdict(lambda: defaultdict(int))
        
        # Ventana deslizante para encontrar accesos cercanos
        window_size = timedelta(minutes=5)
        
        for i, (key1, time1, _) in enumerate(accesses):
            for j in range(i+1, min(i+10, len(accesses))):
                key2, time2, _ = accesses[j]
                
                if time2 - time1 <= window_size:
                    correlations[key1][key2] += 1
                    correlations[key2][key1] += 1
        
        # Normalizar correlaciones
        result = {}
        for key1, related in correlations.items():
            total = sum(related.values())
            if total > 0:
                result[key1] = [
                    (key2, count/total)
                    for key2, count in related.items()
                ]
                result[key1].sort(key=lambda x: x[1], reverse=True)
        
        return result
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calcular precisión global de predicciones."""
        total = (self.prediction_metrics['correct_predictions'] +
                self.prediction_metrics['false_positives'] +
                self.prediction_metrics['false_negatives'])
        
        if total == 0:
            return 0.0
        
        return self.prediction_metrics['correct_predictions'] / total
    
    async def _calculate_average_lead_time(self) -> Optional[timedelta]:
        """Calcular tiempo promedio de anticipación en predicciones."""
        lead_times = []
        
        for key, predictions in self.prediction_cache.items():
            for pred in predictions:
                actual = self._find_actual_access(
                    key,
                    pred.predicted_access_time,
                    tolerance=timedelta(minutes=10)
                )
                
                if actual:
                    lead_time = pred.predicted_access_time - actual
                    if lead_time > timedelta():
                        lead_times.append(lead_time)
        
        if not lead_times:
            return None
        
        return sum(lead_times, timedelta()) / len(lead_times)
    
    def _find_actual_access(
        self,
        key: str,
        predicted_time: datetime,
        tolerance: timedelta
    ) -> Optional[datetime]:
        """Buscar acceso real cerca del tiempo predicho."""
        for access_key, access_time, _ in self.access_history:
            if access_key == key:
                if abs(access_time - predicted_time) <= tolerance:
                    return access_time
        return None
    
    def _should_retrain_model(self) -> bool:
        """Determinar si el modelo necesita reentrenamiento."""
        # Reentrenar si:
        # 1. No está entrenado
        if not self._model_trained:
            return True
        
        # 2. Precisión ha caído significativamente
        current_accuracy = self._calculate_prediction_accuracy()
        if current_accuracy < self.config.min_acceptable_accuracy:
            return True
        
        # 3. Han pasado muchas predicciones desde el último entrenamiento
        if self.prediction_metrics['total_predictions'] > 1000:
            return True
        
        return False
    
    async def _prepare_training_data(self) -> List[Dict[str, Any]]:
        """Preparar datos de entrenamiento desde el historial."""
        training_data = []
        
        # Crear ejemplos de entrenamiento desde el historial
        for i in range(len(self.access_history) - 1):
            key, time, metadata = self.access_history[i]
            next_key, next_time, _ = self.access_history[i + 1]
            
            if key == next_key:
                # Ejemplo de re-acceso
                interval = (next_time - time).total_seconds() / 60  # minutos
                
                features = {
                    'hour': time.hour,
                    'weekday': time.weekday(),
                    'access_count': self.access_patterns[key].access_count,
                    'avg_interval': self.access_patterns[key].average_interval.total_seconds() / 60
                    if self.access_patterns[key].average_interval else 0,
                    'file_type': hash(metadata.get('file_type', '')) % 100,
                    'analysis_type': hash(metadata.get('analysis_type', '')) % 10
                }
                
                training_data.append({
                    'features': features,
                    'label': interval  # Tiempo hasta próximo acceso
                })
        
        return training_data
    
    def _extract_features_and_labels(
        self,
        training_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extraer features y labels para entrenamiento."""
        features = []
        labels = []
        
        for example in training_data:
            feature_vector = [
                example['features']['hour'],
                example['features']['weekday'],
                example['features']['access_count'],
                example['features']['avg_interval'],
                example['features']['file_type'],
                example['features']['analysis_type']
            ]
            features.append(feature_vector)
            labels.append(example['label'])
        
        return np.array(features), np.array(labels)
    
    def _context_to_features(self, context: Dict[str, Any]) -> List[float]:
        """Convertir contexto a vector de features."""
        current_time = datetime.now()
        key = f"{context.get('file_path', '')}:{context.get('analysis_type', 'full')}"
        pattern = self.access_patterns.get(key, AccessPattern())
        
        return [
            current_time.hour,
            current_time.weekday(),
            pattern.access_count,
            pattern.average_interval.total_seconds() / 60 if pattern.average_interval else 0,
            hash(context.get('file_type', '')) % 100,
            hash(context.get('analysis_type', 'full')) % 10
        ]
    
    async def _generate_prediction_contexts(self) -> List[Dict[str, Any]]:
        """Generar contextos para predicción con ML."""
        contexts = []
        
        # Generar contextos para archivos frecuentemente accedidos
        for key, pattern in self.access_patterns.items():
            if pattern.access_count > 5:
                parts = key.split(':')
                context = {
                    'file_path': parts[0],
                    'analysis_type': parts[1] if len(parts) > 1 else 'full',
                    'file_type': pattern.file_type,
                    'pattern': pattern
                }
                contexts.append(context)
        
        return contexts[:50]  # Limitar para evitar demasiadas predicciones
    
    def _extract_access_sequences(self) -> List[List[str]]:
        """Extraer secuencias de acceso del historial."""
        sequences = []
        
        # Buscar patrones repetidos de longitud 2-5
        for length in range(2, 6):
            for i in range(len(self.access_history) - length):
                sequence = [
                    key for key, _, _ in 
                    self.access_history[i:i+length]
                ]
                
                # Verificar si la secuencia se repite
                for j in range(i + length, len(self.access_history) - length):
                    candidate = [
                        key for key, _, _ in 
                        self.access_history[j:j+length]
                    ]
                    
                    if sequence == candidate:
                        sequences.append(sequence)
                        break
        
        return sequences
    
    async def _load_model(self):
        """Cargar modelo desde disco."""
        model_path = Path(self.config.model_storage_path) / "predictive_cache_model.pkl"
        scaler_path = Path(self.config.model_storage_path) / "feature_scaler.pkl"
        
        try:
            if model_path.exists() and scaler_path.exists():
                self.prediction_model = joblib.load(model_path)
                self.feature_scaler = joblib.load(scaler_path)
                self._model_trained = True
        except Exception:
            pass
    
    async def _save_model(self):
        """Guardar modelo a disco."""
        if not self._model_trained:
            return
        
        model_dir = Path(self.config.model_storage_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            joblib.dump(
                self.prediction_model,
                model_dir / "predictive_cache_model.pkl"
            )
            joblib.dump(
                self.feature_scaler,
                model_dir / "feature_scaler.pkl"
            )
        except Exception:
            pass
    
    async def _load_access_history(self):
        """Cargar historial de accesos."""
        history_path = Path(self.config.model_storage_path) / "access_history.json"
        
        try:
            if history_path.exists():
                with open(history_path, 'r') as f:
                    data = json.load(f)
                    
                self.access_history = [
                    (item['key'], datetime.fromisoformat(item['time']), item.get('metadata', {}))
                    for item in data
                ]
                
                # Reconstruir patrones
                await self._update_access_patterns()
        except Exception:
            pass
    
    async def _save_access_history(self):
        """Guardar historial de accesos."""
        history_path = Path(self.config.model_storage_path) / "access_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Guardar solo los últimos N accesos
            recent_history = self.access_history[-10000:]
            
            data = [
                {
                    'key': key,
                    'time': time.isoformat(),
                    'metadata': metadata
                }
                for key, time, metadata in recent_history
            ]
            
            with open(history_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass
    
    def record_access(self, key: str, metadata: Optional[Dict[str, Any]] = None):
        """Registrar un acceso al cache."""
        self.access_history.append((key, datetime.now(), metadata or {}))
        
        # Limitar tamaño del historial en memoria
        if len(self.access_history) > 50000:
            self.access_history = self.access_history[-25000:]

