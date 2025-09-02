"""
Módulo que define los puertos de la capa de aplicación para el aprendizaje y mejora continua.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from uuid import UUID

from codeant_agent.domain.entities.natural_rules.natural_rule import ExecutableRule, ExecutableRuleId
from codeant_agent.domain.value_objects.natural_rules.rule_metrics import (
    RuleFeedback, RulePerformanceMetrics
)


class FeedbackCollectorPort(ABC):
    """Puerto para el recolector de feedback."""
    
    @abstractmethod
    async def collect_feedback(
        self, rule_id: UUID, feedback: RuleFeedback
    ) -> bool:
        """Recolecta feedback para una regla.
        
        Args:
            rule_id: ID de la regla
            feedback: Feedback a recolectar
            
        Returns:
            True si el feedback se recolectó correctamente, False en caso contrario
        """
        pass
    
    @abstractmethod
    async def get_feedback(
        self, rule_id: UUID
    ) -> List[RuleFeedback]:
        """Obtiene el feedback para una regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Lista de feedback para la regla
        """
        pass


class RuleOptimizerPort(ABC):
    """Puerto para el optimizador de reglas."""
    
    @abstractmethod
    async def optimize_rule(
        self, rule_id: ExecutableRuleId
    ) -> Optional[ExecutableRule]:
        """Optimiza una regla.
        
        Args:
            rule_id: ID de la regla a optimizar
            
        Returns:
            Regla optimizada o None si no se pudo optimizar
        """
        pass
    
    @abstractmethod
    async def identify_optimizations(
        self, performance_data: List[RulePerformanceMetrics]
    ) -> List[str]:
        """Identifica posibles optimizaciones basadas en datos de rendimiento.
        
        Args:
            performance_data: Datos de rendimiento
            
        Returns:
            Lista de optimizaciones identificadas
        """
        pass


class PatternLearnerPort(ABC):
    """Puerto para el aprendizaje de patrones."""
    
    @abstractmethod
    async def learn_patterns(
        self, text_samples: List[str], language: str
    ) -> Dict[str, str]:
        """Aprende patrones a partir de muestras de texto.
        
        Args:
            text_samples: Muestras de texto
            language: Idioma de las muestras
            
        Returns:
            Diccionario con patrones aprendidos
        """
        pass
    
    @abstractmethod
    async def improve_pattern(
        self, pattern_name: str, feedback: List[Dict[str, str]]
    ) -> Optional[str]:
        """Mejora un patrón basándose en feedback.
        
        Args:
            pattern_name: Nombre del patrón a mejorar
            feedback: Feedback para el patrón
            
        Returns:
            Patrón mejorado o None si no se pudo mejorar
        """
        pass


class AccuracyMonitorPort(ABC):
    """Puerto para el monitor de precisión."""
    
    @abstractmethod
    async def track_accuracy(
        self, rule_id: UUID, true_positives: int, false_positives: int,
        true_negatives: int, false_negatives: int
    ) -> None:
        """Registra métricas de precisión para una regla.
        
        Args:
            rule_id: ID de la regla
            true_positives: Número de verdaderos positivos
            false_positives: Número de falsos positivos
            true_negatives: Número de verdaderos negativos
            false_negatives: Número de falsos negativos
        """
        pass
    
    @abstractmethod
    async def get_accuracy_metrics(
        self, rule_id: UUID
    ) -> Dict[str, float]:
        """Obtiene métricas de precisión para una regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Diccionario con métricas de precisión
        """
        pass
