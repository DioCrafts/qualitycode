"""
Módulo que define las interfaces de repositorio para las reglas en lenguaje natural.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from uuid import UUID

from codeant_agent.domain.entities.natural_rules.natural_rule import (
    ExecutableRule, ExecutableRuleId, NaturalRule
)
from codeant_agent.domain.value_objects.natural_rules.rule_metrics import (
    RuleFeedback, RulePerformanceMetrics, RuleUsageMetrics
)


class NaturalRuleRepository(ABC):
    """Interfaz para el repositorio de reglas en lenguaje natural."""
    
    @abstractmethod
    async def save(self, rule: NaturalRule) -> NaturalRule:
        """Guarda una regla en el repositorio.
        
        Args:
            rule: La regla a guardar
            
        Returns:
            La regla guardada con su ID actualizado si es nuevo
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, rule_id: UUID) -> Optional[NaturalRule]:
        """Busca una regla por su ID.
        
        Args:
            rule_id: ID de la regla a buscar
            
        Returns:
            La regla encontrada o None si no existe
        """
        pass
    
    @abstractmethod
    async def find_all(self) -> List[NaturalRule]:
        """Obtiene todas las reglas del repositorio.
        
        Returns:
            Lista de todas las reglas
        """
        pass
    
    @abstractmethod
    async def find_by_text(self, text: str) -> List[NaturalRule]:
        """Busca reglas por texto similar.
        
        Args:
            text: Texto a buscar
            
        Returns:
            Lista de reglas que coinciden con el texto
        """
        pass
    
    @abstractmethod
    async def delete(self, rule_id: UUID) -> bool:
        """Elimina una regla del repositorio.
        
        Args:
            rule_id: ID de la regla a eliminar
            
        Returns:
            True si la regla fue eliminada, False en caso contrario
        """
        pass


class ExecutableRuleRepository(ABC):
    """Interfaz para el repositorio de reglas ejecutables."""
    
    @abstractmethod
    async def save(self, rule: ExecutableRule) -> ExecutableRule:
        """Guarda una regla ejecutable en el repositorio.
        
        Args:
            rule: La regla ejecutable a guardar
            
        Returns:
            La regla ejecutable guardada
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, rule_id: ExecutableRuleId) -> Optional[ExecutableRule]:
        """Busca una regla ejecutable por su ID.
        
        Args:
            rule_id: ID de la regla ejecutable a buscar
            
        Returns:
            La regla ejecutable encontrada o None si no existe
        """
        pass
    
    @abstractmethod
    async def find_all(self) -> List[ExecutableRule]:
        """Obtiene todas las reglas ejecutables del repositorio.
        
        Returns:
            Lista de todas las reglas ejecutables
        """
        pass
    
    @abstractmethod
    async def find_by_category(self, category: str) -> List[ExecutableRule]:
        """Busca reglas ejecutables por categoría.
        
        Args:
            category: Categoría a buscar
            
        Returns:
            Lista de reglas ejecutables de la categoría especificada
        """
        pass
    
    @abstractmethod
    async def delete(self, rule_id: ExecutableRuleId) -> bool:
        """Elimina una regla ejecutable del repositorio.
        
        Args:
            rule_id: ID de la regla ejecutable a eliminar
            
        Returns:
            True si la regla ejecutable fue eliminada, False en caso contrario
        """
        pass


class RuleFeedbackRepository(ABC):
    """Interfaz para el repositorio de feedback de reglas."""
    
    @abstractmethod
    async def save_feedback(self, feedback: RuleFeedback) -> RuleFeedback:
        """Guarda feedback de una regla.
        
        Args:
            feedback: El feedback a guardar
            
        Returns:
            El feedback guardado
        """
        pass
    
    @abstractmethod
    async def find_feedback_by_rule_id(self, rule_id: UUID) -> List[RuleFeedback]:
        """Busca feedback por ID de regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Lista de feedback para la regla especificada
        """
        pass
    
    @abstractmethod
    async def get_rule_metrics(self, rule_id: UUID) -> Optional[RuleUsageMetrics]:
        """Obtiene métricas de uso de una regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Métricas de uso de la regla o None si no hay datos
        """
        pass
    
    @abstractmethod
    async def save_performance_metrics(self, metrics: RulePerformanceMetrics) -> None:
        """Guarda métricas de rendimiento de una regla.
        
        Args:
            metrics: Métricas de rendimiento a guardar
        """
        pass
    
    @abstractmethod
    async def get_performance_metrics(self, rule_id: UUID) -> List[RulePerformanceMetrics]:
        """Obtiene métricas de rendimiento de una regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Lista de métricas de rendimiento para la regla especificada
        """
        pass
