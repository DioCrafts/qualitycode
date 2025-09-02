"""
Módulo que define los servicios de dominio para la generación de reglas.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from codeant_agent.domain.entities.natural_rules.natural_rule import (
    ExecutableRule, RuleStructure, RuleValidationResult
)


class RuleGenerationService(ABC):
    """Servicio de dominio para la generación de reglas ejecutables."""
    
    @abstractmethod
    async def generate_executable_rule(
        self, rule_structure: RuleStructure, context: Dict[str, str] = None
    ) -> ExecutableRule:
        """Genera una regla ejecutable a partir de una estructura de regla.
        
        Args:
            rule_structure: Estructura de la regla
            context: Contexto adicional para la generación
            
        Returns:
            Regla ejecutable generada
        """
        pass
    
    @abstractmethod
    async def validate_executable_rule(
        self, rule: ExecutableRule, context: Dict[str, str] = None
    ) -> RuleValidationResult:
        """Valida una regla ejecutable.
        
        Args:
            rule: Regla ejecutable a validar
            context: Contexto adicional para la validación
            
        Returns:
            Resultado de la validación
        """
        pass


class CodeGenerationService(ABC):
    """Servicio de dominio para la generación de código de reglas."""
    
    @abstractmethod
    async def generate_rule_code(
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
    async def format_code(self, code: str, language: str) -> str:
        """Formatea el código generado.
        
        Args:
            code: Código a formatear
            language: Lenguaje de programación del código
            
        Returns:
            Código formateado
        """
        pass
    
    @abstractmethod
    async def validate_code_syntax(self, code: str, language: str) -> List[str]:
        """Valida la sintaxis del código generado.
        
        Args:
            code: Código a validar
            language: Lenguaje de programación del código
            
        Returns:
            Lista de errores de sintaxis (vacía si es válido)
        """
        pass


class RuleTemplateService(ABC):
    """Servicio de dominio para la gestión de plantillas de reglas."""
    
    @abstractmethod
    async def get_template_for_intent(self, intent: str) -> Optional[str]:
        """Obtiene una plantilla para una intención.
        
        Args:
            intent: Intención para la que obtener la plantilla
            
        Returns:
            Plantilla para la intención o None si no existe
        """
        pass
    
    @abstractmethod
    async def render_template(self, template: str, variables: Dict[str, str]) -> str:
        """Renderiza una plantilla con variables.
        
        Args:
            template: Plantilla a renderizar
            variables: Variables para la renderización
            
        Returns:
            Plantilla renderizada
        """
        pass
    
    @abstractmethod
    async def register_template(self, intent: str, template: str) -> bool:
        """Registra una nueva plantilla para una intención.
        
        Args:
            intent: Intención para la que registrar la plantilla
            template: Plantilla a registrar
            
        Returns:
            True si la plantilla se registró correctamente, False en caso contrario
        """
        pass
