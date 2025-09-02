"""
Módulo que define las entidades principales para el sistema de reglas en lenguaje natural.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from codeant_agent.domain.entities.natural_rules.language import Language
from codeant_agent.domain.entities.natural_rules.rule_intent import (
    ActionSeverity, ActionType, ConditionType, RuleDomain, RuleIntent, RuleType
)


@dataclass
class RuleCondition:
    """Condición para una regla en lenguaje natural."""
    condition_type: ConditionType
    condition_text: str
    parameters: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class RuleAction:
    """Acción para una regla en lenguaje natural."""
    action_type: ActionType
    action_text: str
    parameters: Dict[str, str] = field(default_factory=dict)
    severity: ActionSeverity = ActionSeverity.WARNING


@dataclass
class IntentAnalysis:
    """Análisis de intención para una regla en lenguaje natural."""
    primary_intent: RuleIntent
    secondary_intents: List[RuleIntent] = field(default_factory=list)
    domain: RuleDomain = RuleDomain.BEST_PRACTICES
    rule_type: RuleType = RuleType.CONSTRAINT
    conditions: List[RuleCondition] = field(default_factory=list)
    actions: List[RuleAction] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ElementType:
    """Tipo de elemento de código al que se aplica una regla."""
    name: str
    
    @classmethod
    def function(cls) -> 'ElementType':
        """Elemento de tipo función."""
        return cls("function")
    
    @classmethod
    def class_(cls) -> 'ElementType':
        """Elemento de tipo clase."""
        return cls("class")
    
    @classmethod
    def method(cls) -> 'ElementType':
        """Elemento de tipo método."""
        return cls("method")
    
    @classmethod
    def variable(cls) -> 'ElementType':
        """Elemento de tipo variable."""
        return cls("variable")
    
    @classmethod
    def parameter(cls) -> 'ElementType':
        """Elemento de tipo parámetro."""
        return cls("parameter")
    
    @classmethod
    def loop(cls) -> 'ElementType':
        """Elemento de tipo bucle."""
        return cls("loop")
    
    @classmethod
    def conditional(cls) -> 'ElementType':
        """Elemento de tipo condicional."""
        return cls("conditional")
    
    @classmethod
    def expression(cls) -> 'ElementType':
        """Elemento de tipo expresión."""
        return cls("expression")
    
    @classmethod
    def statement(cls) -> 'ElementType':
        """Elemento de tipo declaración."""
        return cls("statement")
    
    @classmethod
    def file(cls) -> 'ElementType':
        """Elemento de tipo archivo."""
        return cls("file")
    
    @classmethod
    def module(cls) -> 'ElementType':
        """Elemento de tipo módulo."""
        return cls("module")


@dataclass
class ElementFilter:
    """Filtro para elementos de código."""
    name: str
    value: str


@dataclass
class TargetElement:
    """Elemento objetivo al que se aplica una regla."""
    element_type: ElementType
    name: str
    attributes: Dict[str, str] = field(default_factory=dict)
    filters: List[ElementFilter] = field(default_factory=list)


@dataclass
class ThresholdOperator:
    """Operador para umbrales en reglas."""
    name: str
    symbol: str
    
    @classmethod
    def greater_than(cls) -> 'ThresholdOperator':
        """Operador mayor que."""
        return cls("greater_than", ">")
    
    @classmethod
    def greater_than_or_equal(cls) -> 'ThresholdOperator':
        """Operador mayor o igual que."""
        return cls("greater_than_or_equal", ">=")
    
    @classmethod
    def less_than(cls) -> 'ThresholdOperator':
        """Operador menor que."""
        return cls("less_than", "<")
    
    @classmethod
    def less_than_or_equal(cls) -> 'ThresholdOperator':
        """Operador menor o igual que."""
        return cls("less_than_or_equal", "<=")
    
    @classmethod
    def equal(cls) -> 'ThresholdOperator':
        """Operador igual a."""
        return cls("equal", "==")
    
    @classmethod
    def not_equal(cls) -> 'ThresholdOperator':
        """Operador no igual a."""
        return cls("not_equal", "!=")


@dataclass
class Threshold:
    """Umbral para una regla."""
    name: str
    value: float
    operator: ThresholdOperator
    unit: Optional[str] = None


@dataclass
class RuleScope:
    """Alcance de una regla."""
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    languages: Set[str] = field(default_factory=set)


@dataclass
class RuleExample:
    """Ejemplo de código para una regla."""
    code: str
    is_compliant: bool
    explanation: str


@dataclass
class RuleStructure:
    """Estructura de una regla generada a partir de lenguaje natural."""
    intent_analysis: IntentAnalysis
    target_element: Optional[TargetElement] = None
    conditions: List[RuleCondition] = field(default_factory=list)
    actions: List[RuleAction] = field(default_factory=list)
    thresholds: List[Threshold] = field(default_factory=list)
    scope: RuleScope = field(default_factory=RuleScope)
    description: str = ""
    examples: List[RuleExample] = field(default_factory=list)


@dataclass
class RuleValidationResult:
    """Resultado de la validación de una regla."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExecutableRuleId:
    """Identificador único para una regla ejecutable."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        """Devuelve una representación en string del ID."""
        return str(self.value)


@dataclass
class RuleImplementation:
    """Implementación de una regla ejecutable."""
    code: str
    language: str = "python"
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutableRule:
    """Regla ejecutable generada a partir de una regla en lenguaje natural."""
    id: ExecutableRuleId
    rule_name: str
    description: str
    implementation: RuleImplementation
    languages: List[str]
    category: RuleDomain
    severity: ActionSeverity
    configuration: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class NaturalRule:
    """Regla definida en lenguaje natural."""
    id: UUID = field(default_factory=uuid4)
    original_text: str = ""
    language: Language = Language.SPANISH
    preprocessed_text: str = ""
    intent_analysis: Optional[IntentAnalysis] = None
    rule_structure: Optional[RuleStructure] = None
    executable_rule: Optional[ExecutableRule] = None
    validation_result: Optional[RuleValidationResult] = None
    confidence_score: float = 0.0
    generation_time_ms: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_valid(self) -> bool:
        """Indica si la regla es válida."""
        return (self.validation_result is not None and 
                self.validation_result.is_valid and
                self.executable_rule is not None)
