"""
Módulo que define los servicios de dominio para el aprendizaje y mejora continua.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from uuid import UUID

from codeant_agent.domain.entities.natural_rules.natural_rule import ExecutableRule, ExecutableRuleId
from codeant_agent.domain.value_objects.natural_rules.rule_metrics import (
    RuleFeedback, RulePerformanceMetrics
)


class LearningService(ABC):
    """Servicio de dominio para el aprendizaje y mejora continua."""
    
    @abstractmethod
    async def learn_from_feedback(
        self, rule_id: UUID, feedback: RuleFeedback
    ) -> Dict[str, str]:
        """Aprende de feedback para mejorar reglas.
        
        Args:
            rule_id: ID de la regla
            feedback: Feedback recibido
            
        Returns:
            Diccionario con resultados del aprendizaje
        """
        pass
    
    @abstractmethod
    async def optimize_rule(
        self, rule_id: ExecutableRuleId
    ) -> Optional[ExecutableRule]:
        """Optimiza una regla basándose en métricas de rendimiento.
        
        Args:
            rule_id: ID de la regla a optimizar
            
        Returns:
            Regla optimizada o None si no se pudo optimizar
        """
        pass
    
    @abstractmethod
    async def analyze_feedback_patterns(
        self, rule_id: UUID
    ) -> Dict[str, str]:
        """Analiza patrones de feedback para una regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Diccionario con análisis de patrones
        """
        pass
    
    @abstractmethod
    async def generate_rule_improvements(
        self, rule_id: UUID, feedback_analysis: Dict[str, str]
    ) -> List[str]:
        """Genera mejoras para una regla basándose en análisis de feedback.
        
        Args:
            rule_id: ID de la regla
            feedback_analysis: Análisis de feedback
            
        Returns:
            Lista de mejoras sugeridas
        """
        pass


class PerformanceTrackingService(ABC):
    """Servicio de dominio para el seguimiento de rendimiento de reglas."""
    
    @abstractmethod
    async def track_rule_execution(
        self, rule_id: ExecutableRuleId, execution_time_ms: float, 
        memory_usage_kb: float, cpu_usage_percent: float
    ) -> None:
        """Registra métricas de ejecución de una regla.
        
        Args:
            rule_id: ID de la regla
            execution_time_ms: Tiempo de ejecución en milisegundos
            memory_usage_kb: Uso de memoria en kilobytes
            cpu_usage_percent: Uso de CPU en porcentaje
        """
        pass
    
    @abstractmethod
    async def get_performance_data(
        self, rule_id: ExecutableRuleId
    ) -> List[RulePerformanceMetrics]:
        """Obtiene datos de rendimiento de una regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Lista de métricas de rendimiento
        """
        pass
    
    @abstractmethod
    async def identify_performance_bottlenecks(
        self, rule_id: ExecutableRuleId
    ) -> List[str]:
        """Identifica cuellos de botella de rendimiento en una regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Lista de cuellos de botella identificados
        """
        pass
