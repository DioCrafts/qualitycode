"""
Biblioteca de reglas built-in para el motor de reglas estáticas.

Este paquete contiene reglas predefinidas para análisis de código,
organizadas por categorías y lenguajes de programación.
"""

from .builtin_rules_library import BuiltinRulesLibrary, BuiltinRulesError, RuleTemplate

__all__ = [
    "BuiltinRulesLibrary",
    "BuiltinRulesError", 
    "RuleTemplate"
]
