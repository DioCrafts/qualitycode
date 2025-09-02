"""
Módulo que define los DTOs para las reglas en lenguaje natural.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import UUID


@dataclass
class ElementTypeDTO:
    """DTO para tipos de elementos."""
    name: str


@dataclass
class ElementFilterDTO:
    """DTO para filtros de elementos."""
    name: str
    value: str


@dataclass
class TargetElementDTO:
    """DTO para elementos objetivo."""
    element_type: str
    name: str
    attributes: Dict[str, str] = field(default_factory=dict)
    filters: List[ElementFilterDTO] = field(default_factory=list)


@dataclass
class ThresholdDTO:
    """DTO para umbrales."""
    name: str
    value: float
    operator: str
    unit: Optional[str] = None


@dataclass
class RuleScopeDTO:
    """DTO para alcances de reglas."""
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)


@dataclass
class RuleExampleDTO:
    """DTO para ejemplos de reglas."""
    code: str
    is_compliant: bool
    explanation: str


@dataclass
class RuleStructureDTO:
    """DTO para estructuras de reglas."""
    intent_analysis: Dict[str, str]
    target_element: Optional[TargetElementDTO] = None
    conditions: List[Dict[str, str]] = field(default_factory=list)
    actions: List[Dict[str, str]] = field(default_factory=list)
    thresholds: List[ThresholdDTO] = field(default_factory=list)
    scope: RuleScopeDTO = field(default_factory=RuleScopeDTO)
    description: str = ""
    examples: List[RuleExampleDTO] = field(default_factory=list)


@dataclass
class CreateNaturalRuleRequestDTO:
    """DTO para solicitudes de creación de reglas naturales."""
    text: str
    language: str
    context: Dict[str, str] = field(default_factory=dict)


@dataclass
class NaturalRuleResponseDTO:
    """DTO para respuestas de reglas naturales."""
    id: UUID
    original_text: str
    language: str
    preprocessed_text: str
    intent_analysis: Dict[str, str]
    rule_structure: Optional[Dict[str, str]] = None
    executable_rule: Optional[Dict[str, str]] = None
    is_valid: bool = False
    confidence_score: float = 0.0
    generation_time_ms: int = 0
    created_at: str = ""


@dataclass
class ExecutableRuleDTO:
    """DTO para reglas ejecutables."""
    id: str
    rule_name: str
    description: str
    implementation: Dict[str, str]
    languages: List[str]
    category: str
    severity: str
    configuration: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RuleValidationResultDTO:
    """DTO para resultados de validación de reglas."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
