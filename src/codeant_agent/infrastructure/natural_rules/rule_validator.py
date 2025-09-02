"""
Módulo que implementa el validador de reglas para el sistema de reglas en lenguaje natural.
"""
import ast
import re
from typing import Dict, List, Optional

from codeant_agent.application.ports.natural_rules.rule_generation_ports import RuleValidatorPort
from codeant_agent.domain.entities.natural_rules.natural_rule import (
    ExecutableRule, RuleStructure, RuleValidationResult
)


class RuleValidator(RuleValidatorPort):
    """Implementación del validador de reglas."""
    
    def __init__(self):
        """Inicializa el validador de reglas."""
        pass
    
    async def validate_rule(
        self, rule: ExecutableRule
    ) -> RuleValidationResult:
        """Valida una regla ejecutable.
        
        Args:
            rule: Regla a validar
            
        Returns:
            Resultado de la validación
        """
        errors = []
        warnings = []
        
        # Validar código
        code_errors = await self.validate_code(
            rule.implementation.code, rule.implementation.language
        )
        errors.extend(code_errors)
        
        # Validar nombre de la regla
        name_errors = self._validate_rule_name(rule.rule_name)
        errors.extend(name_errors)
        
        # Validar descripción
        if not rule.description:
            warnings.append("Rule description is empty")
        elif len(rule.description) < 10:
            warnings.append("Rule description is too short")
        
        # Validar lenguajes aplicables
        if not rule.languages:
            warnings.append("No languages specified for the rule")
        
        # Validar configuración
        if rule.configuration:
            config_errors = self._validate_configuration(rule.configuration)
            errors.extend(config_errors)
        
        # Determinar si la regla es válida
        is_valid = len(errors) == 0
        
        return RuleValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
    
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
        errors = []
        
        # Validar según el lenguaje
        if language.lower() == "python":
            errors.extend(self._validate_python_code(code))
        else:
            errors.append(f"Validation for language '{language}' is not supported")
        
        return errors
    
    async def validate_structure(
        self, structure: RuleStructure
    ) -> List[str]:
        """Valida la estructura de una regla.
        
        Args:
            structure: Estructura a validar
            
        Returns:
            Lista de errores de validación (vacía si es válida)
        """
        errors = []
        
        # Validar análisis de intención
        if not structure.intent_analysis:
            errors.append("Missing intent analysis")
        elif structure.intent_analysis.primary_intent is None:
            errors.append("Missing primary intent")
        
        # Validar condiciones y acciones
        if not structure.conditions and not structure.actions:
            errors.append("Rule has no conditions or actions")
        
        # Validar umbrales
        for threshold in structure.thresholds:
            if threshold.value is None:
                errors.append(f"Threshold '{threshold.name}' has no value")
        
        return errors
    
    def _validate_python_code(self, code: str) -> List[str]:
        """Valida código Python.
        
        Args:
            code: Código Python a validar
            
        Returns:
            Lista de errores de validación (vacía si es válido)
        """
        errors = []
        
        # Verificar que el código no esté vacío
        if not code or code.isspace():
            errors.append("Code is empty")
            return errors
        
        # Verificar sintaxis Python
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")
            return errors
        
        # Verificar que el código define una función
        if not re.search(r"def\s+\w+\s*\(", code):
            errors.append("Code does not define a function")
        
        # Verificar que el código devuelve algo
        if not re.search(r"return\s+", code):
            errors.append("Function does not return a value")
        
        # Verificar que el código no contiene errores comunes
        if "print(" in code:
            errors.append("Code contains print statements, which should be avoided in rules")
        
        if "import " in code and not re.search(r"^import\s+", code, re.MULTILINE):
            errors.append("Import statements should be at the top of the file")
        
        return errors
    
    def _validate_rule_name(self, rule_name: str) -> List[str]:
        """Valida el nombre de una regla.
        
        Args:
            rule_name: Nombre de la regla a validar
            
        Returns:
            Lista de errores de validación (vacía si es válido)
        """
        errors = []
        
        # Verificar que el nombre no esté vacío
        if not rule_name:
            errors.append("Rule name is empty")
            return errors
        
        # Verificar formato del nombre
        if not re.match(r"^[a-z][a-z0-9_]*$", rule_name):
            errors.append("Rule name should start with a lowercase letter and contain only lowercase letters, numbers, and underscores")
        
        # Verificar longitud del nombre
        if len(rule_name) < 3:
            errors.append("Rule name is too short")
        elif len(rule_name) > 50:
            errors.append("Rule name is too long")
        
        return errors
    
    def _validate_configuration(self, configuration: Dict[str, str]) -> List[str]:
        """Valida la configuración de una regla.
        
        Args:
            configuration: Configuración a validar
            
        Returns:
            Lista de errores de validación (vacía si es válida)
        """
        errors = []
        
        # Verificar valores numéricos
        for key, value in configuration.items():
            if re.match(r"^(max|min|threshold|limit)\w*$", key):
                try:
                    float(value)
                except ValueError:
                    errors.append(f"Configuration parameter '{key}' should be a number")
        
        return errors
