"""
Modelos para condiciones de reglas del motor de reglas estáticas.

Este módulo define las estructuras de datos para las condiciones que deben
cumplirse para que una regla genere una violación o sugerencia.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from ...parsers.unified.unified_ast import UnifiedAST, UnifiedNode
from .pattern_models import PatternMatch


class ConditionType(str, Enum):
    """Tipos de condiciones para reglas."""
    NODE_COUNT = "node_count"
    ATTRIBUTE_VALUE = "attribute_value"
    CUSTOM_PREDICATE = "custom_predicate"
    CROSS_LANGUAGE_CHECK = "cross_language_check"
    COMPLEXITY_THRESHOLD = "complexity_threshold"
    SIZE_THRESHOLD = "size_threshold"
    DEPTH_THRESHOLD = "depth_threshold"
    NAMING_CONVENTION = "naming_convention"
    SECURITY_CHECK = "security_check"
    PERFORMANCE_CHECK = "performance_check"


@dataclass
class NodeCountCondition:
    """Condición basada en el número de nodos."""
    min_count: Optional[int] = None
    max_count: Optional[int] = None
    exact_count: Optional[int] = None
    
    def evaluate(self, nodes: List[UnifiedNode]) -> bool:
        """Evaluar la condición de conteo de nodos."""
        count = len(nodes)
        
        if self.exact_count is not None:
            return count == self.exact_count
        
        if self.min_count is not None and count < self.min_count:
            return False
        
        if self.max_count is not None and count > self.max_count:
            return False
        
        return True


@dataclass
class AttributeValueCondition:
    """Condición basada en valores de atributos."""
    attribute_name: str
    expected_value: Any
    operator: str = "equals"  # equals, not_equals, contains, regex, greater_than, less_than
    case_sensitive: bool = True
    
    def evaluate(self, node: UnifiedNode) -> bool:
        """Evaluar la condición de valor de atributo."""
        if not hasattr(node, 'attributes') or not node.attributes:
            return False
        
        if self.attribute_name not in node.attributes:
            return False
        
        actual_value = node.attributes[self.attribute_name]
        
        if self.operator == "equals":
            if self.case_sensitive:
                return actual_value == self.expected_value
            else:
                return str(actual_value).lower() == str(self.expected_value).lower()
        
        elif self.operator == "not_equals":
            if self.case_sensitive:
                return actual_value != self.expected_value
            else:
                return str(actual_value).lower() != str(self.expected_value).lower()
        
        elif self.operator == "contains":
            if self.case_sensitive:
                return str(self.expected_value) in str(actual_value)
            else:
                return str(self.expected_value).lower() in str(actual_value).lower()
        
        elif self.operator == "regex":
            flags = 0 if self.case_sensitive else re.IGNORECASE
            return bool(re.search(str(self.expected_value), str(actual_value), flags))
        
        elif self.operator == "greater_than":
            try:
                return float(actual_value) > float(self.expected_value)
            except (ValueError, TypeError):
                return False
        
        elif self.operator == "less_than":
            try:
                return float(actual_value) < float(self.expected_value)
            except (ValueError, TypeError):
                return False
        
        return False


@dataclass
class CustomPredicateCondition:
    """Condición basada en predicados personalizados."""
    predicate_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    negated: bool = False
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluar la condición de predicado personalizado."""
        # En una implementación real, esto se conectaría con un sistema de plugins
        # o un registro de predicados personalizados
        predicate_result = self._evaluate_predicate(context)
        return not predicate_result if self.negated else predicate_result
    
    def _evaluate_predicate(self, context: Dict[str, Any]) -> bool:
        """Evaluar el predicado específico."""
        # Predicados básicos predefinidos
        if self.predicate_name == "is_magic_number":
            return self._is_magic_number(context)
        elif self.predicate_name == "is_hardcoded_secret":
            return self._is_hardcoded_secret(context)
        elif self.predicate_name == "is_sql_injection_risk":
            return self._is_sql_injection_risk(context)
        elif self.predicate_name == "violates_naming_convention":
            return self._violates_naming_convention(context)
        elif self.predicate_name == "is_unused_variable":
            return self._is_unused_variable(context)
        elif self.predicate_name == "is_dead_code":
            return self._is_dead_code(context)
        else:
            # Predicado desconocido, asumir que es falso
            return False
    
    def _is_magic_number(self, context: Dict[str, Any]) -> bool:
        """Verificar si es un número mágico."""
        node = context.get('node')
        if not node or not node.value:
            return False
        
        # Números que no se consideran mágicos
        allowed_numbers = self.parameters.get('allowed_numbers', [0, 1, -1, 2])
        
        try:
            value = float(str(node.value))
            return value not in allowed_numbers and value != int(value)
        except (ValueError, TypeError):
            return False
    
    def _is_hardcoded_secret(self, context: Dict[str, Any]) -> bool:
        """Verificar si es un secreto hardcodeado."""
        node = context.get('node')
        if not node or not node.value:
            return False
        
        value_str = str(node.value).lower()
        secret_patterns = self.parameters.get('secret_patterns', [
            r'password\s*=\s*[\'"][^\'"]+[\'"]',
            r'api_?key\s*=\s*[\'"][^\'"]+[\'"]',
            r'secret\s*=\s*[\'"][^\'"]+[\'"]',
            r'token\s*=\s*[\'"][^\'"]+[\'"]'
        ])
        
        for pattern in secret_patterns:
            if re.search(pattern, value_str, re.IGNORECASE):
                return True
        
        return False
    
    def _is_sql_injection_risk(self, context: Dict[str, Any]) -> bool:
        """Verificar si hay riesgo de SQL injection."""
        node = context.get('node')
        if not node:
            return False
        
        # Verificar si es una llamada a función de base de datos
        if hasattr(node, 'name') and node.name:
            sql_functions = ['execute', 'query', 'cursor', 'fetchall', 'fetchone']
            if any(func in node.name.lower() for func in sql_functions):
                # Verificar si usa concatenación de strings
                return self._uses_string_concatenation(context)
        
        return False
    
    def _uses_string_concatenation(self, context: Dict[str, Any]) -> bool:
        """Verificar si usa concatenación de strings."""
        # Implementación simplificada
        return True  # En una implementación real, analizaría el AST
    
    def _violates_naming_convention(self, context: Dict[str, Any]) -> bool:
        """Verificar si viola la convención de nombres."""
        node = context.get('node')
        language = context.get('language', 'python')
        
        if not node or not node.name:
            return False
        
        name = node.name
        
        if language == 'python':
            # Python: snake_case para funciones y variables, PascalCase para clases
            if hasattr(node, 'node_type'):
                if 'class' in str(node.node_type).lower():
                    return not re.match(r'^[A-Z][a-zA-Z0-9]*$', name)
                else:
                    return not re.match(r'^[a-z_][a-z0-9_]*$', name)
        
        elif language in ['javascript', 'typescript']:
            # JavaScript/TypeScript: camelCase para funciones y variables, PascalCase para clases
            if hasattr(node, 'node_type'):
                if 'class' in str(node.node_type).lower():
                    return not re.match(r'^[A-Z][a-zA-Z0-9]*$', name)
                else:
                    return not re.match(r'^[a-z][a-zA-Z0-9]*$', name)
        
        return False
    
    def _is_unused_variable(self, context: Dict[str, Any]) -> bool:
        """Verificar si es una variable no utilizada."""
        # Implementación simplificada
        return False  # En una implementación real, analizaría el uso de variables
    
    def _is_dead_code(self, context: Dict[str, Any]) -> bool:
        """Verificar si es código muerto."""
        # Implementación simplificada
        return False  # En una implementación real, analizaría el flujo de control


@dataclass
class CrossLanguageCondition:
    """Condición de verificación cross-language."""
    target_languages: List[str]
    equivalence_type: str = "structural"  # structural, semantic, behavioral
    check_consistency: bool = True
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluar la condición cross-language."""
        current_language = context.get('language')
        if not current_language:
            return False
        
        if current_language not in self.target_languages:
            return False
        
        # En una implementación real, compararía con equivalentes en otros lenguajes
        return self._check_cross_language_equivalence(context)
    
    def _check_cross_language_equivalence(self, context: Dict[str, Any]) -> bool:
        """Verificar equivalencia cross-language."""
        # Implementación simplificada
        return True  # En una implementación real, analizaría patrones equivalentes


@dataclass
class ComplexityThresholdCondition:
    """Condición basada en umbrales de complejidad."""
    metric_name: str  # cyclomatic, cognitive, halstead, etc.
    threshold: float
    operator: str = "greater_than"  # greater_than, less_than, equals
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluar la condición de umbral de complejidad."""
        metrics = context.get('metrics', {})
        actual_value = metrics.get(self.metric_name, 0)
        
        if self.operator == "greater_than":
            return actual_value > self.threshold
        elif self.operator == "less_than":
            return actual_value < self.threshold
        elif self.operator == "equals":
            return actual_value == self.threshold
        
        return False


@dataclass
class RuleCondition:
    """Condición para una regla."""
    condition_type: ConditionType
    node_count: Optional[NodeCountCondition] = None
    attribute_value: Optional[AttributeValueCondition] = None
    custom_predicate: Optional[CustomPredicateCondition] = None
    cross_language: Optional[CrossLanguageCondition] = None
    complexity_threshold: Optional[ComplexityThresholdCondition] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validar que solo una condición esté presente."""
        conditions = [
            self.node_count,
            self.attribute_value,
            self.custom_predicate,
            self.cross_language,
            self.complexity_threshold
        ]
        
        present_conditions = [c for c in conditions if c is not None]
        if len(present_conditions) != 1:
            raise ValueError("Exactly one condition type must be specified")
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluar la condición."""
        if self.node_count:
            nodes = context.get('nodes', [])
            return self.node_count.evaluate(nodes)
        
        elif self.attribute_value:
            node = context.get('node')
            if not node:
                return False
            return self.attribute_value.evaluate(node)
        
        elif self.custom_predicate:
            return self.custom_predicate.evaluate(context)
        
        elif self.cross_language:
            return self.cross_language.evaluate(context)
        
        elif self.complexity_threshold:
            return self.complexity_threshold.evaluate(context)
        
        return False


@dataclass
class ConditionGroup:
    """Grupo de condiciones con lógica de combinación."""
    conditions: List[RuleCondition]
    combination_logic: str = "all"  # all, any, majority, none
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluar el grupo de condiciones."""
        if not self.conditions:
            return True
        
        results = [condition.evaluate(context) for condition in self.conditions]
        
        if self.combination_logic == "all":
            return all(results)
        elif self.combination_logic == "any":
            return any(results)
        elif self.combination_logic == "majority":
            return sum(results) > len(results) / 2
        elif self.combination_logic == "none":
            return not any(results)
        
        return False
