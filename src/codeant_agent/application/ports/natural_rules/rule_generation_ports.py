"""
Módulo que define los puertos de la capa de aplicación para la generación de reglas.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from codeant_agent.domain.entities.natural_rules.natural_rule import (
    ExecutableRule, RuleStructure, RuleValidationResult
)


class RuleGeneratorPort(ABC):
    """Puerto para el generador de reglas."""
    
    @abstractmethod
    async def generate_rule(
        self, rule_structure: RuleStructure, context: Dict[str, str] = None
    ) -> ExecutableRule:
        """Genera una regla ejecutable a partir de una estructura.
        
        Args:
            rule_structure: Estructura de la regla
            context: Contexto adicional para la generación
            
        Returns:
            Regla ejecutable generada
        """
        pass
    
    @abstractmethod
    async def determine_implementation_strategy(
        self, rule_structure: RuleStructure
    ) -> str:
        """Determina la mejor estrategia de implementación para una regla.
        
        Args:
            rule_structure: Estructura de la regla
            
        Returns:
            Nombre de la estrategia de implementación
        """
        pass


class CodeGeneratorPort(ABC):
    """Puerto para el generador de código."""
    
    @abstractmethod
    async def generate_code(
        self, rule_structure: RuleStructure, language: str = "python"
    ) -> str:
        """Genera código para una regla.
        
        Args:
            rule_structure: Estructura de la regla
            language: Lenguaje de programación para el código
            
        Returns:
            Código generado
        """
        pass
    
    @abstractmethod
    async def get_template(self, template_name: str) -> Optional[str]:
        """Obtiene una plantilla de código por nombre.
        
        Args:
            template_name: Nombre de la plantilla
            
        Returns:
            Plantilla de código o None si no existe
        """
        pass
    
    @abstractmethod
    async def render_template(
        self, template: str, variables: Dict[str, str]
    ) -> str:
        """Renderiza una plantilla con variables.
        
        Args:
            template: Plantilla a renderizar
            variables: Variables para la renderización
            
        Returns:
            Plantilla renderizada
        """
        pass


class RuleValidatorPort(ABC):
    """Puerto para el validador de reglas."""
    
    @abstractmethod
    async def validate_rule(
        self, rule: ExecutableRule
    ) -> RuleValidationResult:
        """Valida una regla ejecutable.
        
        Args:
            rule: Regla a validar
            
        Returns:
            Resultado de la validación
        """
        pass
    
    @abstractmethod
    async def validate_code(
        self, code: str, language: str
    ) -> List[str]:
        """Valida el código de una regla.
        
        Args:
            code: Código a validar
            language: Lenguaje de programación del código
            
        Returns:
            Lista de errores de validación (vacía si es válido)
        """
        pass
    
    @abstractmethod
    async def validate_structure(
        self, structure: RuleStructure
    ) -> List[str]:
        """Valida la estructura de una regla.
        
        Args:
            structure: Estructura a validar
            
        Returns:
            Lista de errores de validación (vacía si es válida)
        """
        pass
