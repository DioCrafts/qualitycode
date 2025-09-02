"""
Módulo que define los tipos de intención de reglas y sus características.
"""
from enum import Enum, auto
from typing import List, Optional


class RuleIntent(Enum):
    """Enumeración de intenciones posibles para una regla."""
    PROHIBIT = auto()    # "no debe", "must not"
    REQUIRE = auto()     # "debe", "must"
    RECOMMEND = auto()   # "debería", "should"
    LIMIT = auto()       # "no más de", "not more than"
    ENSURE = auto()      # "asegurar", "ensure"
    VALIDATE = auto()    # "validar", "validate"
    CHECK = auto()       # "verificar", "check"
    COUNT = auto()       # "contar", "count"
    MEASURE = auto()     # "medir", "measure"
    DETECT = auto()      # "detectar", "detect"
    UNKNOWN = auto()     # Intención desconocida
    
    def __str__(self) -> str:
        """Devuelve una representación en string de la intención."""
        intent_names = {
            RuleIntent.PROHIBIT: "prohibir",
            RuleIntent.REQUIRE: "requerir",
            RuleIntent.RECOMMEND: "recomendar",
            RuleIntent.LIMIT: "limitar",
            RuleIntent.ENSURE: "asegurar",
            RuleIntent.VALIDATE: "validar",
            RuleIntent.CHECK: "verificar",
            RuleIntent.COUNT: "contar",
            RuleIntent.MEASURE: "medir",
            RuleIntent.DETECT: "detectar",
            RuleIntent.UNKNOWN: "desconocido"
        }
        return intent_names.get(self, super().__str__())


class RuleDomain(Enum):
    """Enumeración de dominios posibles para una regla."""
    SECURITY = auto()
    PERFORMANCE = auto()
    MAINTAINABILITY = auto()
    BEST_PRACTICES = auto()
    NAMING = auto()
    STRUCTURE = auto()
    COMPLEXITY = auto()
    DOCUMENTATION = auto()
    TESTING = auto()
    ARCHITECTURE = auto()
    CUSTOM = auto()
    
    def __str__(self) -> str:
        """Devuelve una representación en string del dominio."""
        domain_names = {
            RuleDomain.SECURITY: "seguridad",
            RuleDomain.PERFORMANCE: "rendimiento",
            RuleDomain.MAINTAINABILITY: "mantenibilidad",
            RuleDomain.BEST_PRACTICES: "mejores prácticas",
            RuleDomain.NAMING: "nomenclatura",
            RuleDomain.STRUCTURE: "estructura",
            RuleDomain.COMPLEXITY: "complejidad",
            RuleDomain.DOCUMENTATION: "documentación",
            RuleDomain.TESTING: "pruebas",
            RuleDomain.ARCHITECTURE: "arquitectura",
            RuleDomain.CUSTOM: "personalizado"
        }
        return domain_names.get(self, super().__str__())


class RuleType(Enum):
    """Enumeración de tipos posibles para una regla."""
    CONSTRAINT = auto()      # Límites o restricciones
    QUALITY = auto()         # Medidas de calidad
    PATTERN = auto()         # Detección de patrones
    METRIC = auto()          # Cálculo de métricas
    VALIDATION = auto()      # Reglas de validación
    TRANSFORMATION = auto()  # Transformación de código
    DETECTION = auto()       # Detección de problemas
    
    def __str__(self) -> str:
        """Devuelve una representación en string del tipo de regla."""
        type_names = {
            RuleType.CONSTRAINT: "restricción",
            RuleType.QUALITY: "calidad",
            RuleType.PATTERN: "patrón",
            RuleType.METRIC: "métrica",
            RuleType.VALIDATION: "validación",
            RuleType.TRANSFORMATION: "transformación",
            RuleType.DETECTION: "detección"
        }
        return type_names.get(self, super().__str__())


class ConditionType(Enum):
    """Enumeración de tipos de condiciones para una regla."""
    IF = auto()
    WHEN = auto()
    THAT = auto()
    GREATER_THAN = auto()
    LESS_THAN = auto()
    EQUAL_TO = auto()
    CONTAINS = auto()
    NOT_CONTAINS = auto()
    IN_LOCATION = auto()
    OF_TYPE = auto()
    WITH_ATTRIBUTE = auto()
    
    def __str__(self) -> str:
        """Devuelve una representación en string del tipo de condición."""
        condition_names = {
            ConditionType.IF: "si",
            ConditionType.WHEN: "cuando",
            ConditionType.THAT: "que",
            ConditionType.GREATER_THAN: "mayor que",
            ConditionType.LESS_THAN: "menor que",
            ConditionType.EQUAL_TO: "igual a",
            ConditionType.CONTAINS: "contiene",
            ConditionType.NOT_CONTAINS: "no contiene",
            ConditionType.IN_LOCATION: "en ubicación",
            ConditionType.OF_TYPE: "de tipo",
            ConditionType.WITH_ATTRIBUTE: "con atributo"
        }
        return condition_names.get(self, super().__str__())


class ActionType(Enum):
    """Enumeración de tipos de acciones para una regla."""
    MUST_BE = auto()
    MUST_NOT_BE = auto()
    SHOULD = auto()
    REPORT = auto()
    SUGGEST = auto()
    WARN = auto()
    FAIL = auto()
    REFACTOR = auto()
    FIX = auto()
    COUNT = auto()
    MEASURE = auto()
    
    def __str__(self) -> str:
        """Devuelve una representación en string del tipo de acción."""
        action_names = {
            ActionType.MUST_BE: "debe ser",
            ActionType.MUST_NOT_BE: "no debe ser",
            ActionType.SHOULD: "debería",
            ActionType.REPORT: "reportar",
            ActionType.SUGGEST: "sugerir",
            ActionType.WARN: "advertir",
            ActionType.FAIL: "fallar",
            ActionType.REFACTOR: "refactorizar",
            ActionType.FIX: "corregir",
            ActionType.COUNT: "contar",
            ActionType.MEASURE: "medir"
        }
        return action_names.get(self, super().__str__())


class ActionSeverity(Enum):
    """Enumeración de niveles de severidad para acciones de reglas."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    
    def __str__(self) -> str:
        """Devuelve una representación en string de la severidad."""
        severity_names = {
            ActionSeverity.INFO: "información",
            ActionSeverity.WARNING: "advertencia",
            ActionSeverity.ERROR: "error",
            ActionSeverity.CRITICAL: "crítico"
        }
        return severity_names.get(self, super().__str__())
