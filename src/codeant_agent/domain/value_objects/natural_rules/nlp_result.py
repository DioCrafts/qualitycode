"""
Módulo que define los objetos de valor relacionados con los resultados del procesamiento NLP.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ExtractedEntity:
    """Entidad extraída del texto mediante NLP."""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PatternMatch:
    """Coincidencia de un patrón en el texto."""
    pattern_name: str
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float
    captures: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Ambiguity:
    """Ambigüedad detectada en el texto."""
    description: str
    ambiguous_text: str
    possible_interpretations: List[str]
    start_pos: int
    end_pos: int
    severity: float  # 0.0-1.0, donde 1.0 es la más severa


@dataclass(frozen=True)
class NLPProcessingResult:
    """Resultado del procesamiento NLP de un texto."""
    preprocessed_text: str
    entities: List[ExtractedEntity] = field(default_factory=list)
    pattern_matches: List[PatternMatch] = field(default_factory=list)
    ambiguities: List[Ambiguity] = field(default_factory=list)
    processing_time_ms: int = 0
    confidence_score: float = 0.0


@dataclass(frozen=True)
class NLPConfig:
    """Configuración para el procesamiento NLP."""
    enable_entity_extraction: bool = True
    enable_intent_classification: bool = True
    enable_pattern_matching: bool = True
    confidence_threshold: float = 0.7
    max_rule_complexity: int = 10
    enable_ambiguity_detection: bool = True
    enable_context_awareness: bool = True
    custom_domain_vocabulary: Dict[str, List[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class TermMapping:
    """Mapeo de términos entre idiomas o conceptos."""
    source_term: str
    target_term: str
    context: Optional[str] = None
    
    def __hash__(self) -> int:
        """Devuelve el hash de este objeto."""
        return hash((self.source_term, self.target_term, self.context))


@dataclass(frozen=True)
class IntentPattern:
    """Patrón para detectar una intención en el texto."""
    pattern: str
    intent: str
    examples: List[str] = field(default_factory=list)
    confidence: float = 1.0
