"""
Módulo que define los DTOs para el procesamiento NLP.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProcessTextRequestDTO:
    """DTO para solicitudes de procesamiento de texto."""
    text: str
    language: str
    enable_entity_extraction: bool = True
    enable_intent_classification: bool = True
    enable_pattern_matching: bool = True
    enable_ambiguity_detection: bool = True


@dataclass
class ExtractedEntityDTO:
    """DTO para entidades extraídas."""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class PatternMatchDTO:
    """DTO para coincidencias de patrones."""
    pattern_name: str
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float
    captures: Dict[str, str] = field(default_factory=dict)


@dataclass
class AmbiguityDTO:
    """DTO para ambigüedades detectadas."""
    description: str
    ambiguous_text: str
    possible_interpretations: List[str]
    start_pos: int
    end_pos: int
    severity: float


@dataclass
class ProcessTextResponseDTO:
    """DTO para respuestas de procesamiento de texto."""
    preprocessed_text: str
    entities: List[ExtractedEntityDTO] = field(default_factory=list)
    pattern_matches: List[PatternMatchDTO] = field(default_factory=list)
    ambiguities: List[AmbiguityDTO] = field(default_factory=list)
    processing_time_ms: int = 0
    confidence_score: float = 0.0


@dataclass
class IntentAnalysisRequestDTO:
    """DTO para solicitudes de análisis de intención."""
    text: str
    language: str
    context: Dict[str, str] = field(default_factory=dict)


@dataclass
class RuleConditionDTO:
    """DTO para condiciones de reglas."""
    condition_type: str
    condition_text: str
    parameters: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class RuleActionDTO:
    """DTO para acciones de reglas."""
    action_type: str
    action_text: str
    parameters: Dict[str, str] = field(default_factory=dict)
    severity: str = "WARNING"


@dataclass
class IntentAnalysisResponseDTO:
    """DTO para respuestas de análisis de intención."""
    primary_intent: str
    secondary_intents: List[str] = field(default_factory=list)
    domain: str = "BEST_PRACTICES"
    rule_type: str = "CONSTRAINT"
    conditions: List[RuleConditionDTO] = field(default_factory=list)
    actions: List[RuleActionDTO] = field(default_factory=list)
    confidence: float = 0.0
