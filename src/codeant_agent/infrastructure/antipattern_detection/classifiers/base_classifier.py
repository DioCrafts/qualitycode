"""
Clasificador base para detección de antipatrones.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

from ....domain.entities.antipattern_analysis import (
    AntipatternType, AntipatternFeatures, SeverityIndicator
)
from ....domain.value_objects.source_position import SourcePosition

logger = logging.getLogger(__name__)


@dataclass
class DetectedPattern:
    """Patrón detectado por un clasificador."""
    pattern_type: AntipatternType
    confidence: float
    locations: List[SourcePosition] = field(default_factory=list)
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    severity_indicators: List[SeverityIndicator] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)


class BaseAntipatternClassifier(ABC):
    """Clasificador base para detección de antipatrones."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.detection_history = []
        self.feature_weights = self._initialize_feature_weights()
    
    @abstractmethod
    async def detect_patterns(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> List[DetectedPattern]:
        """Detectar patrones de antipatrones."""
        pass
    
    @abstractmethod
    def _initialize_feature_weights(self) -> Dict[str, float]:
        """Inicializar pesos de features para este clasificador."""
        pass
    
    async def calculate_confidence_score(
        self, 
        features: AntipatternFeatures,
        pattern_indicators: Dict[str, float]
    ) -> float:
        """Calcular score de confianza basado en indicadores."""
        
        total_score = 0.0
        total_weight = 0.0
        
        for indicator, value in pattern_indicators.items():
            weight = self.feature_weights.get(indicator, 1.0)
            total_score += value * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalizar entre 0 y 1
        normalized_score = total_score / total_weight
        return min(1.0, max(0.0, normalized_score))
    
    def _extract_evidence_from_features(
        self, 
        features: AntipatternFeatures, 
        pattern_type: AntipatternType
    ) -> List[str]:
        """Extraer evidencia específica del patrón."""
        evidence = []
        
        # Evidencia general
        if features.lines_of_code > 0:
            evidence.append(f"File has {features.lines_of_code} lines of code")
        
        if features.cyclomatic_complexity > 0:
            evidence.append(f"Cyclomatic complexity: {features.cyclomatic_complexity:.1f}")
        
        return evidence
    
    def _identify_pattern_locations(
        self, 
        features: AntipatternFeatures, 
        pattern_type: AntipatternType
    ) -> List[SourcePosition]:
        """Identificar ubicaciones del patrón."""
        # Implementación básica - los clasificadores específicos pueden sobrescribir
        return [SourcePosition(line=1, column=1)]
    
    def _create_severity_indicators(
        self, 
        features: AntipatternFeatures, 
        confidence: float,
        pattern_type: AntipatternType
    ) -> List[SeverityIndicator]:
        """Crear indicadores de severidad."""
        indicators = []
        
        # Indicador de confianza
        indicators.append(SeverityIndicator(
            indicator_type="confidence",
            value=confidence,
            description=f"Detection confidence: {confidence:.2%}"
        ))
        
        # Indicador de complejidad
        if features.cyclomatic_complexity > 10:
            indicators.append(SeverityIndicator(
                indicator_type="complexity",
                value=features.cyclomatic_complexity,
                description=f"High cyclomatic complexity: {features.cyclomatic_complexity:.1f}"
            ))
        
        return indicators


class ClassifierError(Exception):
    """Excepción para errores en clasificadores."""
    pass
