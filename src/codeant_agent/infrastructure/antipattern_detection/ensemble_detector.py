"""
Detector ensemble que combina múltiples clasificadores para mayor precisión.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import statistics

from ...domain.entities.antipattern_analysis import (
    AntipatternType, AntipatternFeatures, AntipatternDetectionResult, EnsemblePrediction,
    ModelPrediction
)
from .classifiers.base_classifier import DetectedPattern

logger = logging.getLogger(__name__)


@dataclass
class ModelWeight:
    """Peso de un modelo en el ensemble."""
    model_name: str
    weight: float
    accuracy_score: float = 0.0
    confidence_calibration: float = 1.0


class EnsembleDetector:
    """Detector ensemble que combina predicciones de múltiples clasificadores."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Pesos de los modelos en el ensemble
        self.model_weights = self._initialize_model_weights()
        
        # Estrategias de voting disponibles
        self.voting_strategies = {
            "weighted_average": self._weighted_average_voting,
            "majority_vote": self._majority_voting,
            "consensus_threshold": self._consensus_threshold_voting,
            "confidence_weighted": self._confidence_weighted_voting
        }
        
        # Estrategia por defecto
        self.default_strategy = "confidence_weighted"
        
        # Histórico para mejora continua
        self.prediction_history = defaultdict(list)
    
    def _initialize_model_weights(self) -> Dict[str, ModelWeight]:
        """Inicializar pesos de modelos basado en performance esperada."""
        
        return {
            "security": ModelWeight(
                model_name="SecurityAntipatternClassifier",
                weight=1.2,  # Mayor peso por criticidad
                accuracy_score=0.85,
                confidence_calibration=0.95
            ),
            "performance": ModelWeight(
                model_name="PerformanceAntipatternClassifier", 
                weight=1.0,
                accuracy_score=0.80,
                confidence_calibration=1.0
            ),
            "architectural": ModelWeight(
                model_name="ArchitecturalAntipatternClassifier",
                weight=1.1,  # Mayor peso por impacto
                accuracy_score=0.78,
                confidence_calibration=1.05
            ),
            "design": ModelWeight(
                model_name="DesignAntipatternClassifier",
                weight=0.9,
                accuracy_score=0.82,
                confidence_calibration=1.0
            )
        }
    
    async def detect_ensemble_antipatterns(
        self, 
        features: AntipatternFeatures,
        individual_results: AntipatternDetectionResult
    ) -> List[DetectedPattern]:
        """Detectar antipatrones usando ensemble de clasificadores."""
        
        try:
            # Agrupar patrones por tipo
            patterns_by_type = self._group_patterns_by_type(individual_results.detected_antipatterns)
            
            # Aplicar ensemble voting
            ensemble_patterns = []
            
            for pattern_type, patterns in patterns_by_type.items():
                if len(patterns) > 1:  # Solo aplicar ensemble si hay múltiples predicciones
                    ensemble_pattern = await self._apply_ensemble_voting(
                        pattern_type, patterns, features
                    )
                    if ensemble_pattern:
                        ensemble_patterns.append(ensemble_pattern)
                else:
                    # Si solo hay una predicción, aplicar calibración simple
                    calibrated_pattern = await self._apply_single_model_calibration(
                        patterns[0], features
                    )
                    ensemble_patterns.append(calibrated_pattern)
            
            # Aplicar meta-validación
            validated_patterns = await self._apply_meta_validation(
                ensemble_patterns, features
            )
            
            # Registrar para mejora continua
            await self._record_ensemble_predictions(validated_patterns, features)
            
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Error in ensemble detection: {e}")
            # Fallback: retornar patrones originales
            return individual_results.detected_antipatterns
    
    def _group_patterns_by_type(
        self, 
        patterns: List[DetectedPattern]
    ) -> Dict[AntipatternType, List[DetectedPattern]]:
        """Agrupar patrones por tipo."""
        
        grouped = defaultdict(list)
        for pattern in patterns:
            grouped[pattern.pattern_type].append(pattern)
        
        return dict(grouped)
    
    async def _apply_ensemble_voting(
        self, 
        pattern_type: AntipatternType,
        patterns: List[DetectedPattern],
        features: AntipatternFeatures
    ) -> Optional[DetectedPattern]:
        """Aplicar voting ensemble a múltiples predicciones del mismo tipo."""
        
        try:
            # Obtener estrategia de voting
            strategy_name = self.config.get("voting_strategy", self.default_strategy)
            voting_strategy = self.voting_strategies.get(strategy_name, self._confidence_weighted_voting)
            
            # Aplicar estrategia
            ensemble_result = await voting_strategy(pattern_type, patterns, features)
            
            if ensemble_result:
                # Crear patrón ensemble
                ensemble_pattern = DetectedPattern(
                    pattern_type=pattern_type,
                    confidence=ensemble_result.confidence,
                    locations=self._merge_locations(patterns),
                    description=f"Ensemble detection: {ensemble_result.final_prediction.value}",
                    evidence=self._merge_evidence(patterns),
                    severity_indicators=self._merge_severity_indicators(patterns),
                    feature_importance=self._merge_feature_importance(patterns)
                )
                
                return ensemble_pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Error in ensemble voting for {pattern_type}: {e}")
            # Fallback: retornar el patrón con mayor confianza
            return max(patterns, key=lambda p: p.confidence) if patterns else None
    
    async def _confidence_weighted_voting(
        self, 
        pattern_type: AntipatternType,
        patterns: List[DetectedPattern],
        features: AntipatternFeatures
    ) -> Optional[EnsemblePrediction]:
        """Voting basado en confianza ponderada."""
        
        if not patterns:
            return None
        
        total_weighted_confidence = 0.0
        total_weight = 0.0
        model_predictions = []
        
        for pattern in patterns:
            # Obtener peso del modelo (simulado basado en evidencia)
            model_weight = self._estimate_model_weight(pattern, features)
            
            # Calcular contribución ponderada
            weighted_confidence = pattern.confidence * model_weight
            total_weighted_confidence += weighted_confidence
            total_weight += model_weight
            
            # Crear predicción del modelo
            model_pred = ModelPrediction(
                pattern_type=pattern_type,
                confidence=pattern.confidence,
                evidence=pattern.evidence,
                feature_importance=pattern.feature_importance or {}
            )
            model_predictions.append(model_pred)
        
        if total_weight == 0:
            return None
        
        # Calcular confianza final
        final_confidence = total_weighted_confidence / total_weight
        
        # Calcular consensus score
        consensus_score = self._calculate_consensus_score(patterns)
        
        # Calcular uncertainty
        uncertainty = self._calculate_ensemble_uncertainty(patterns, final_confidence)
        
        return EnsemblePrediction(
            final_prediction=pattern_type,
            confidence=final_confidence,
            model_predictions=model_predictions,
            consensus_score=consensus_score,
            uncertainty=uncertainty
        )
    
    async def _weighted_average_voting(
        self, 
        pattern_type: AntipatternType,
        patterns: List[DetectedPattern],
        features: AntipatternFeatures
    ) -> Optional[EnsemblePrediction]:
        """Voting por promedio ponderado simple."""
        
        if not patterns:
            return None
        
        # Pesos uniformes para simplicidad
        weights = [1.0] * len(patterns)
        confidences = [p.confidence for p in patterns]
        
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
        consensus_score = 1.0 - (max(confidences) - min(confidences))  # Proximidad de confianzas
        
        model_predictions = [
            ModelPrediction(
                pattern_type=pattern_type,
                confidence=p.confidence,
                evidence=p.evidence,
                feature_importance=p.feature_importance or {}
            )
            for p in patterns
        ]
        
        return EnsemblePrediction(
            final_prediction=pattern_type,
            confidence=weighted_confidence,
            model_predictions=model_predictions,
            consensus_score=consensus_score,
            uncertainty=1.0 - consensus_score
        )
    
    async def _majority_voting(
        self, 
        pattern_type: AntipatternType,
        patterns: List[DetectedPattern],
        features: AntipatternFeatures
    ) -> Optional[EnsemblePrediction]:
        """Voting por mayoría."""
        
        if not patterns:
            return None
        
        # En este caso, todos los patrones son del mismo tipo
        # Aplicar threshold de consenso
        threshold = self.config.get("majority_threshold", 0.6)
        
        valid_patterns = [p for p in patterns if p.confidence >= threshold]
        
        if len(valid_patterns) >= len(patterns) / 2:  # Mayoría
            avg_confidence = statistics.mean([p.confidence for p in valid_patterns])
            
            model_predictions = [
                ModelPrediction(
                    pattern_type=pattern_type,
                    confidence=p.confidence,
                    evidence=p.evidence,
                    feature_importance=p.feature_importance or {}
                )
                for p in valid_patterns
            ]
            
            return EnsemblePrediction(
                final_prediction=pattern_type,
                confidence=avg_confidence,
                model_predictions=model_predictions,
                consensus_score=len(valid_patterns) / len(patterns),
                uncertainty=0.2
            )
        
        return None
    
    async def _consensus_threshold_voting(
        self, 
        pattern_type: AntipatternType,
        patterns: List[DetectedPattern],
        features: AntipatternFeatures
    ) -> Optional[EnsemblePrediction]:
        """Voting basado en threshold de consenso."""
        
        if not patterns:
            return None
        
        consensus_threshold = self.config.get("consensus_threshold", 0.7)
        min_consensus_models = self.config.get("min_consensus_models", 2)
        
        high_confidence_patterns = [
            p for p in patterns if p.confidence >= consensus_threshold
        ]
        
        if len(high_confidence_patterns) >= min_consensus_models:
            # Hay suficiente consenso
            avg_confidence = statistics.mean([p.confidence for p in high_confidence_patterns])
            consensus_score = len(high_confidence_patterns) / len(patterns)
            
            model_predictions = [
                ModelPrediction(
                    pattern_type=pattern_type,
                    confidence=p.confidence,
                    evidence=p.evidence,
                    feature_importance=p.feature_importance or {}
                )
                for p in high_confidence_patterns
            ]
            
            return EnsemblePrediction(
                final_prediction=pattern_type,
                confidence=avg_confidence,
                model_predictions=model_predictions,
                consensus_score=consensus_score,
                uncertainty=1.0 - consensus_score
            )
        
        return None
    
    async def _apply_single_model_calibration(
        self, 
        pattern: DetectedPattern,
        features: AntipatternFeatures
    ) -> DetectedPattern:
        """Aplicar calibración a un solo modelo."""
        
        # Aplicar factor de calibración conservador
        calibration_factor = 0.9
        calibrated_confidence = pattern.confidence * calibration_factor
        
        # Crear patrón calibrado
        calibrated_pattern = DetectedPattern(
            pattern_type=pattern.pattern_type,
            confidence=calibrated_confidence,
            locations=pattern.locations,
            description=f"Single model (calibrated): {pattern.description}",
            evidence=pattern.evidence,
            severity_indicators=pattern.severity_indicators,
            feature_importance=pattern.feature_importance
        )
        
        return calibrated_pattern
    
    async def _apply_meta_validation(
        self, 
        patterns: List[DetectedPattern],
        features: AntipatternFeatures
    ) -> List[DetectedPattern]:
        """Aplicar validación meta-nivel."""
        
        validated_patterns = []
        
        for pattern in patterns:
            # Aplicar validaciones cruzadas
            is_valid = await self._validate_pattern_consistency(pattern, features)
            
            if is_valid:
                # Aplicar ajuste final de confianza
                final_confidence = await self._apply_final_confidence_adjustment(
                    pattern, features
                )
                
                pattern.confidence = final_confidence
                validated_patterns.append(pattern)
        
        return validated_patterns
    
    async def _validate_pattern_consistency(
        self, 
        pattern: DetectedPattern,
        features: AntipatternFeatures
    ) -> bool:
        """Validar consistencia del patrón detectado."""
        
        # Validación básica de consistencia
        
        # 1. Consistencia de evidencia
        if len(pattern.evidence) == 0 and pattern.confidence > 0.8:
            return False  # Alta confianza sin evidencia es sospechoso
        
        # 2. Consistencia con features
        if pattern.pattern_type == AntipatternType.SQL_INJECTION:
            if not features.has_sql_operations and not features.has_user_input:
                return False  # SQL injection sin SQL o input
        
        elif pattern.pattern_type == AntipatternType.GOD_OBJECT:
            if features.max_class_size < 100 and features.methods_count < 10:
                return False  # God object en clase pequeña
        
        elif pattern.pattern_type == AntipatternType.N_PLUS_ONE_QUERY:
            if not features.has_loops and not features.has_sql_operations:
                return False  # N+1 sin loops ni SQL
        
        # 3. Validación de threshold mínimo
        min_threshold = self.config.get("meta_validation_threshold", 0.4)
        if pattern.confidence < min_threshold:
            return False
        
        return True
    
    async def _apply_final_confidence_adjustment(
        self, 
        pattern: DetectedPattern,
        features: AntipatternFeatures
    ) -> float:
        """Aplicar ajuste final de confianza."""
        
        adjustment_factor = 1.0
        
        # Ajustar por consistencia de evidencia
        evidence_consistency = len(pattern.evidence) / 5.0  # Normalizar a ~5 evidencias
        evidence_factor = min(1.2, max(0.8, evidence_consistency))
        adjustment_factor *= evidence_factor
        
        # Ajustar por severity indicators
        if pattern.severity_indicators:
            high_severity_count = sum(
                1 for indicator in pattern.severity_indicators
                if "critical" in str(indicator.value).lower() or "high" in str(indicator.value).lower()
            )
            severity_factor = min(1.1, 1.0 + high_severity_count * 0.05)
            adjustment_factor *= severity_factor
        
        # Aplicar ajuste
        final_confidence = pattern.confidence * adjustment_factor
        return min(1.0, max(0.0, final_confidence))
    
    def _estimate_model_weight(
        self, 
        pattern: DetectedPattern,
        features: AntipatternFeatures
    ) -> float:
        """Estimar peso del modelo basado en el patrón y contexto."""
        
        base_weight = 1.0
        
        # Ajustar peso basado en cantidad de evidencia
        evidence_weight = min(2.0, len(pattern.evidence) / 3.0)
        
        # Ajustar peso basado en consistencia de feature importance
        importance_weight = 1.0
        if hasattr(pattern, 'feature_importance') and pattern.feature_importance:
            max_importance = max(pattern.feature_importance.values())
            importance_weight = min(1.5, max_importance * 2)
        
        return base_weight * evidence_weight * importance_weight
    
    def _calculate_consensus_score(self, patterns: List[DetectedPattern]) -> float:
        """Calcular score de consenso entre patrones."""
        
        if len(patterns) <= 1:
            return 1.0
        
        confidences = [p.confidence for p in patterns]
        
        # Calcular variabilidad de confianzas
        mean_confidence = statistics.mean(confidences)
        variance = statistics.variance(confidences) if len(confidences) > 1 else 0
        
        # Consenso alto = baja variabilidad
        consensus_score = max(0.0, 1.0 - variance * 2)  # Amplificar varianza
        
        return consensus_score
    
    def _calculate_ensemble_uncertainty(
        self, 
        patterns: List[DetectedPattern],
        final_confidence: float
    ) -> float:
        """Calcular incertidumbre del ensemble."""
        
        if len(patterns) <= 1:
            return 0.3  # Incertidumbre base para un solo modelo
        
        # Incertidumbre basada en dispersión de confianzas
        confidences = [p.confidence for p in patterns]
        confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0
        
        # Incertidumbre basada en confianza final
        confidence_uncertainty = 0.1 if final_confidence > 0.8 else 0.2
        
        # Combinar incertidumbres
        total_uncertainty = min(1.0, confidence_std + confidence_uncertainty)
        
        return total_uncertainty
    
    def _merge_locations(self, patterns: List[DetectedPattern]) -> List:
        """Merge ubicaciones de múltiples patrones."""
        
        all_locations = []
        for pattern in patterns:
            all_locations.extend(pattern.locations)
        
        # Eliminar duplicados (simplificado)
        unique_locations = []
        seen = set()
        for location in all_locations:
            loc_key = f"{location.line}:{location.column}"
            if loc_key not in seen:
                unique_locations.append(location)
                seen.add(loc_key)
        
        return unique_locations
    
    def _merge_evidence(self, patterns: List[DetectedPattern]) -> List[str]:
        """Merge evidencia de múltiples patrones."""
        
        all_evidence = []
        for pattern in patterns:
            all_evidence.extend(pattern.evidence)
        
        # Eliminar duplicados manteniendo orden
        unique_evidence = []
        seen = set()
        for evidence in all_evidence:
            if evidence not in seen:
                unique_evidence.append(evidence)
                seen.add(evidence)
        
        return unique_evidence[:10]  # Limitar a 10 evidencias más relevantes
    
    def _merge_severity_indicators(self, patterns: List[DetectedPattern]) -> List:
        """Merge indicadores de severidad."""
        
        all_indicators = []
        for pattern in patterns:
            all_indicators.extend(pattern.severity_indicators)
        
        # Eliminar duplicados (simplificado)
        unique_indicators = []
        seen = set()
        for indicator in all_indicators:
            indicator_key = f"{indicator.indicator_type}:{indicator.value}"
            if indicator_key not in seen:
                unique_indicators.append(indicator)
                seen.add(indicator_key)
        
        return unique_indicators
    
    def _merge_feature_importance(self, patterns: List[DetectedPattern]) -> Dict[str, float]:
        """Merge importancia de features."""
        
        merged_importance = defaultdict(list)
        
        for pattern in patterns:
            if hasattr(pattern, 'feature_importance') and pattern.feature_importance:
                for feature, importance in pattern.feature_importance.items():
                    merged_importance[feature].append(importance)
        
        # Promediar importancias
        final_importance = {}
        for feature, importances in merged_importance.items():
            final_importance[feature] = statistics.mean(importances)
        
        return final_importance
    
    async def _record_ensemble_predictions(
        self, 
        patterns: List[DetectedPattern],
        features: AntipatternFeatures
    ):
        """Registrar predicciones del ensemble para mejora continua."""
        
        for pattern in patterns:
            self.prediction_history[pattern.pattern_type].append({
                "confidence": pattern.confidence,
                "evidence_count": len(pattern.evidence),
                "feature_consistency": len(pattern.feature_importance or {}),
                "timestamp": None  # Se añadiría timestamp real
            })
            
            # Mantener solo las últimas 50 predicciones por tipo
            if len(self.prediction_history[pattern.pattern_type]) > 50:
                self.prediction_history[pattern.pattern_type] = self.prediction_history[pattern.pattern_type][-50:]
