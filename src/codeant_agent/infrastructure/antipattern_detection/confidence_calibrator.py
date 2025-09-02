"""
Calibrador de confianza para mejorar la precisión de las predicciones de antipatrones.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ...domain.entities.antipattern_analysis import (
    AntipatternType, AntipatternFeatures, AntipatternCategory
)
from .classifiers.base_classifier import DetectedPattern

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Datos para calibración de confianza."""
    pattern_type: AntipatternType
    raw_confidence: float
    calibrated_confidence: float
    feature_contributions: Dict[str, float]
    uncertainty: float = 0.0


class ConfidenceCalibrator:
    """Calibrador de confianza para predicciones de antipatrones."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Parámetros de calibración por tipo de patrón
        self.calibration_params = self._initialize_calibration_params()
        
        # Histórico de predicciones para mejorar calibración
        self.prediction_history = defaultdict(list)
        
        # Factores de corrección por categoría
        self.category_corrections = self._initialize_category_corrections()
    
    def _initialize_calibration_params(self) -> Dict[AntipatternType, Dict[str, float]]:
        """Inicializar parámetros de calibración por tipo de antipatrón."""
        
        return {
            # Parámetros para cada tipo de antipatrón
            AntipatternType.GOD_OBJECT: {
                "base_threshold": 0.6,
                "complexity_weight": 0.3,
                "size_weight": 0.4,
                "responsibility_weight": 0.3
            },
            AntipatternType.SQL_INJECTION: {
                "base_threshold": 0.5,  # Más conservador por ser crítico
                "security_weight": 0.6,
                "input_validation_weight": 0.4
            },
            AntipatternType.N_PLUS_ONE_QUERY: {
                "base_threshold": 0.7,
                "performance_weight": 0.5,
                "loop_weight": 0.5
            },
            AntipatternType.SPAGHETTI_CODE: {
                "base_threshold": 0.6,
                "complexity_weight": 0.4,
                "coupling_weight": 0.3,
                "nesting_weight": 0.3
            },
            AntipatternType.LARGE_CLASS: {
                "base_threshold": 0.6,
                "size_weight": 0.5,
                "method_count_weight": 0.3,
                "responsibility_weight": 0.2
            }
        }
    
    def _initialize_category_corrections(self) -> Dict[AntipatternCategory, float]:
        """Inicializar factores de corrección por categoría."""
        
        return {
            AntipatternCategory.SECURITY: 0.9,  # Más conservador
            AntipatternCategory.PERFORMANCE: 1.0,  # Balance normal
            AntipatternCategory.ARCHITECTURAL: 1.1,  # Menos estricto
            AntipatternCategory.DESIGN: 1.0,
            AntipatternCategory.MAINTAINABILITY: 1.05
        }
    
    async def calibrate_confidence(
        self, 
        pattern: DetectedPattern,
        features: Optional[AntipatternFeatures] = None
    ) -> float:
        """Calibrar la confianza de una predicción."""
        
        try:
            # Obtener confianza base
            raw_confidence = pattern.confidence
            
            # Aplicar calibración específica por tipo
            calibrated = await self._apply_pattern_specific_calibration(
                pattern, raw_confidence, features
            )
            
            # Aplicar corrección por categoría
            category_corrected = self._apply_category_correction(
                pattern.pattern_type, calibrated
            )
            
            # Aplicar factores contextuales
            context_adjusted = await self._apply_contextual_adjustments(
                pattern, category_corrected, features
            )
            
            # Aplicar suavizado para evitar sobre-confianza
            final_confidence = self._apply_confidence_smoothing(context_adjusted)
            
            # Registrar para histórico
            self._record_calibration(pattern.pattern_type, raw_confidence, final_confidence)
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Error calibrating confidence: {e}")
            # Fallback: aplicar factor conservador
            return pattern.confidence * 0.8
    
    async def _apply_pattern_specific_calibration(
        self, 
        pattern: DetectedPattern,
        raw_confidence: float,
        features: Optional[AntipatternFeatures]
    ) -> float:
        """Aplicar calibración específica por tipo de patrón."""
        
        pattern_params = self.calibration_params.get(pattern.pattern_type, {})
        base_threshold = pattern_params.get("base_threshold", 0.6)
        
        # Aplicar función de calibración sigmoidea
        calibrated = self._sigmoid_calibration(raw_confidence, base_threshold)
        
        # Ajustar basado en evidencia específica
        if features:
            evidence_adjustment = await self._calculate_evidence_adjustment(
                pattern, features, pattern_params
            )
            calibrated = min(1.0, calibrated * evidence_adjustment)
        
        return calibrated
    
    def _apply_category_correction(
        self, 
        pattern_type: AntipatternType, 
        confidence: float
    ) -> float:
        """Aplicar corrección por categoría de antipatrón."""
        
        category = self._get_pattern_category(pattern_type)
        correction_factor = self.category_corrections.get(category, 1.0)
        
        return min(1.0, confidence * correction_factor)
    
    async def _apply_contextual_adjustments(
        self, 
        pattern: DetectedPattern,
        confidence: float,
        features: Optional[AntipatternFeatures]
    ) -> float:
        """Aplicar ajustes contextuales."""
        
        adjustments = confidence
        
        if features:
            # Ajuste por tamaño de archivo
            if features.lines_of_code > 1000:
                # Archivos muy grandes pueden tener más falsos positivos
                adjustments *= 0.95
            elif features.lines_of_code < 50:
                # Archivos muy pequeños son menos propensos a antipatrones complejos
                adjustments *= 0.9
            
            # Ajuste por complejidad general
            if features.cyclomatic_complexity > 20:
                # Alta complejidad aumenta probabilidad de antipatrones
                adjustments *= 1.1
            elif features.cyclomatic_complexity < 2:
                # Baja complejidad reduce probabilidad
                adjustments *= 0.8
            
            # Ajuste por lenguaje
            language_factors = {
                'python': 1.0,
                'javascript': 0.95,  # JS puede tener patrones más dinámicos
                'typescript': 1.05,  # TypeScript es más estructurado
                'java': 1.1,        # Java tiende a tener antipatrones más claros
                'rust': 0.9         # Rust previene muchos antipatrones
            }
            
            lang_factor = language_factors.get(features.language.value, 1.0)
            adjustments *= lang_factor
        
        return min(1.0, max(0.0, adjustments))
    
    def _apply_confidence_smoothing(self, confidence: float) -> float:
        """Aplicar suavizado para prevenir sobre-confianza."""
        
        # Aplicar función de suavizado que reduce ligeramente las confianzas muy altas
        if confidence > 0.9:
            # Reducir confianzas extremadamente altas
            smoothed = 0.85 + (confidence - 0.9) * 0.5
        elif confidence > 0.8:
            # Reducir ligeramente confianzas altas
            smoothed = confidence * 0.95
        else:
            # Mantener confianzas medias/bajas
            smoothed = confidence
        
        return min(1.0, max(0.0, smoothed))
    
    async def _calculate_evidence_adjustment(
        self, 
        pattern: DetectedPattern,
        features: AntipatternFeatures,
        pattern_params: Dict[str, float]
    ) -> float:
        """Calcular ajuste basado en evidencia específica."""
        
        adjustment = 1.0
        
        # Ajustes específicos por tipo de patrón
        if pattern.pattern_type == AntipatternType.GOD_OBJECT:
            # Más evidencia = mayor confianza
            size_factor = min(2.0, features.max_class_size / 200.0) if features.max_class_size > 0 else 1.0
            method_factor = min(2.0, features.methods_count / 20.0) if features.methods_count > 0 else 1.0
            resp_factor = min(2.0, features.distinct_responsibilities / 5.0) if features.distinct_responsibilities > 0 else 1.0
            
            adjustment = (size_factor + method_factor + resp_factor) / 3.0
        
        elif pattern.pattern_type == AntipatternType.SQL_INJECTION:
            # Para seguridad, ser más conservador
            if features.has_sql_operations and features.has_user_input:
                adjustment = 1.2  # Alta evidencia
            elif features.has_sql_operations or features.has_user_input:
                adjustment = 0.9  # Evidencia parcial
            else:
                adjustment = 0.6  # Poca evidencia
        
        elif pattern.pattern_type == AntipatternType.N_PLUS_ONE_QUERY:
            # Evidencia directa es crucial
            if features.has_loops and features.has_sql_operations:
                adjustment = 1.3
            elif features.has_loops or features.has_sql_operations:
                adjustment = 0.8
            else:
                adjustment = 0.5
        
        return max(0.5, min(2.0, adjustment))
    
    def _sigmoid_calibration(self, confidence: float, threshold: float) -> float:
        """Aplicar calibración sigmoidea centrada en el threshold."""
        
        # Función sigmoidea que mapea [0,1] -> [0,1] con centro en threshold
        shifted = (confidence - threshold) * 4  # Amplificar para mejor separación
        sigmoid = 1 / (1 + math.exp(-shifted))
        
        return sigmoid
    
    def _get_pattern_category(self, pattern_type: AntipatternType) -> AntipatternCategory:
        """Obtener categoría de un tipo de antipatrón."""
        
        category_mapping = {
            AntipatternType.GOD_OBJECT: AntipatternCategory.ARCHITECTURAL,
            AntipatternType.BIG_BALL_OF_MUD: AntipatternCategory.ARCHITECTURAL,
            AntipatternType.SPAGHETTI_CODE: AntipatternCategory.ARCHITECTURAL,
            
            AntipatternType.SQL_INJECTION: AntipatternCategory.SECURITY,
            AntipatternType.HARDCODED_SECRETS: AntipatternCategory.SECURITY,
            AntipatternType.XSS_VULNERABILITY: AntipatternCategory.SECURITY,
            
            AntipatternType.N_PLUS_ONE_QUERY: AntipatternCategory.PERFORMANCE,
            AntipatternType.MEMORY_LEAK: AntipatternCategory.PERFORMANCE,
            AntipatternType.INEFFICIENT_ALGORITHM: AntipatternCategory.PERFORMANCE,
            
            AntipatternType.LARGE_CLASS: AntipatternCategory.DESIGN,
            AntipatternType.LONG_METHOD: AntipatternCategory.DESIGN,
            AntipatternType.FEATURE_ENVY: AntipatternCategory.DESIGN,
            AntipatternType.DATA_CLUMPS: AntipatternCategory.DESIGN,
        }
        
        return category_mapping.get(pattern_type, AntipatternCategory.DESIGN)
    
    def _record_calibration(
        self, 
        pattern_type: AntipatternType, 
        raw_confidence: float, 
        calibrated_confidence: float
    ):
        """Registrar calibración para mejora continua."""
        
        calibration_data = CalibrationData(
            pattern_type=pattern_type,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            feature_contributions={},
            uncertainty=abs(raw_confidence - calibrated_confidence)
        )
        
        self.prediction_history[pattern_type].append(calibration_data)
        
        # Mantener solo las últimas 100 predicciones por tipo
        if len(self.prediction_history[pattern_type]) > 100:
            self.prediction_history[pattern_type] = self.prediction_history[pattern_type][-100:]
    
    async def calculate_uncertainty(
        self, 
        pattern: DetectedPattern,
        features: Optional[AntipatternFeatures] = None
    ) -> float:
        """Calcular incertidumbre de una predicción."""
        
        try:
            uncertainty = 0.0
            
            # Incertidumbre basada en la confianza
            confidence = pattern.confidence
            if confidence < 0.6:
                uncertainty += (0.6 - confidence) * 0.5  # Alta incertidumbre para baja confianza
            elif confidence > 0.9:
                uncertainty += (confidence - 0.9) * 0.3  # Incertidumbre por sobre-confianza
            
            # Incertidumbre basada en evidencia
            evidence_count = len(pattern.evidence)
            if evidence_count < 2:
                uncertainty += 0.2  # Poca evidencia aumenta incertidumbre
            elif evidence_count > 8:
                uncertainty += 0.1  # Demasiada evidencia puede ser ruido
            
            # Incertidumbre basada en importancia de features
            if hasattr(pattern, 'feature_importance') and pattern.feature_importance:
                feature_variance = self._calculate_feature_importance_variance(
                    pattern.feature_importance
                )
                uncertainty += feature_variance * 0.1
            
            # Incertidumbre basada en histórico
            if pattern.pattern_type in self.prediction_history:
                historical_uncertainty = self._calculate_historical_uncertainty(pattern.pattern_type)
                uncertainty += historical_uncertainty * 0.1
            
            return min(1.0, max(0.0, uncertainty))
            
        except Exception as e:
            logger.error(f"Error calculating uncertainty: {e}")
            return 0.5  # Incertidumbre media como fallback
    
    def _calculate_feature_importance_variance(
        self, 
        feature_importance: Dict[str, float]
    ) -> float:
        """Calcular varianza en la importancia de features."""
        
        if not feature_importance:
            return 0.5
        
        values = list(feature_importance.values())
        if len(values) < 2:
            return 0.2
        
        mean_importance = sum(values) / len(values)
        variance = sum((v - mean_importance) ** 2 for v in values) / len(values)
        
        # Normalizar varianza
        return min(1.0, variance)
    
    def _calculate_historical_uncertainty(self, pattern_type: AntipatternType) -> float:
        """Calcular incertidumbre basada en histórico."""
        
        history = self.prediction_history.get(pattern_type, [])
        if len(history) < 5:
            return 0.3  # Poca historia = mayor incertidumbre
        
        # Calcular variabilidad en las calibraciones
        uncertainties = [data.uncertainty for data in history[-20:]]  # Últimas 20
        
        if not uncertainties:
            return 0.3
        
        mean_uncertainty = sum(uncertainties) / len(uncertainties)
        return min(0.5, mean_uncertainty)
    
    async def get_calibration_metrics(self) -> Dict[str, float]:
        """Obtener métricas de calibración del sistema."""
        
        metrics = {
            "total_predictions": sum(len(history) for history in self.prediction_history.values()),
            "patterns_calibrated": len(self.prediction_history),
            "average_uncertainty": 0.0,
            "calibration_drift": 0.0
        }
        
        if metrics["total_predictions"] > 0:
            all_uncertainties = []
            all_drifts = []
            
            for history in self.prediction_history.values():
                uncertainties = [data.uncertainty for data in history]
                all_uncertainties.extend(uncertainties)
                
                # Calcular drift (diferencia entre confianza raw y calibrada)
                drifts = [
                    abs(data.raw_confidence - data.calibrated_confidence)
                    for data in history
                ]
                all_drifts.extend(drifts)
            
            metrics["average_uncertainty"] = sum(all_uncertainties) / len(all_uncertainties)
            metrics["calibration_drift"] = sum(all_drifts) / len(all_drifts)
        
        return metrics
    
    async def suggest_threshold_adjustments(self) -> Dict[AntipatternType, float]:
        """Sugerir ajustes de threshold basados en histórico."""
        
        suggestions = {}
        
        for pattern_type, history in self.prediction_history.items():
            if len(history) < 10:
                continue  # Necesitamos más datos
            
            # Analizar distribución de confianzas calibradas
            calibrated_confidences = [data.calibrated_confidence for data in history[-50:]]
            
            # Sugerir threshold basado en percentil 70
            calibrated_confidences.sort()
            percentile_70_idx = int(len(calibrated_confidences) * 0.7)
            suggested_threshold = calibrated_confidences[percentile_70_idx]
            
            # Aplicar límites razonables
            suggested_threshold = max(0.5, min(0.9, suggested_threshold))
            
            suggestions[pattern_type] = suggested_threshold
        
        return suggestions
