"""
Value Objects para métricas y configuraciones del motor de explicaciones.

Este módulo define los value objects relacionados con métricas,
configuraciones y parámetros del motor de explicaciones.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class ExperienceLevel(Enum):
    """Niveles de experiencia del usuario."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class LearningStyle(Enum):
    """Estilos de aprendizaje."""
    VISUAL = "visual"
    TEXTUAL = "textual"
    INTERACTIVE = "interactive"
    EXAMPLE_BASED = "example_based"
    THEORY_FIRST = "theory_first"
    PRACTICE_FIRST = "practice_first"


class FocusArea(Enum):
    """Áreas de enfoque para las explicaciones."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    BEST_PRACTICES = "best_practices"
    CODE_QUALITY = "code_quality"
    ARCHITECTURE = "architecture"
    TESTING = "testing"


@dataclass(frozen=True)
class ExplanationEngineConfig:
    """Configuración del motor de explicaciones."""
    default_language: str
    default_audience: str
    explanation_depth: str
    include_examples: bool
    include_visualizations: bool
    enable_interactive_mode: bool
    personalization_enabled: bool
    learning_path_generation: bool
    max_explanation_length: int
    enable_multilingual: bool
    
    def __post_init__(self):
        """Validar configuración."""
        if self.max_explanation_length <= 0:
            raise ValueError("max_explanation_length debe ser mayor que 0")
        
        if self.max_explanation_length > 10000:
            raise ValueError("max_explanation_length no puede exceder 10000 caracteres")


@dataclass(frozen=True)
class PersonalizationContext:
    """Contexto de personalización del usuario."""
    experience_level: ExperienceLevel
    preferred_learning_style: LearningStyle
    known_technologies: List[str]
    role: str
    previous_interactions: List[Dict[str, Any]]
    
    def __post_init__(self):
        """Validar contexto de personalización."""
        if not self.known_technologies:
            raise ValueError("known_technologies no puede estar vacío")
        
        if not self.role:
            raise ValueError("role es requerido")


@dataclass(frozen=True)
class ContentGenerationRequest:
    """Request para generación de contenido."""
    template_key: str
    language: str
    audience: str
    context_variables: Dict[str, str]
    personalization_context: Optional[PersonalizationContext] = None
    
    def __post_init__(self):
        """Validar request de generación."""
        if not self.template_key:
            raise ValueError("template_key es requerido")
        
        if not self.language:
            raise ValueError("language es requerido")
        
        if not self.audience:
            raise ValueError("audience es requerido")


@dataclass(frozen=True)
class GeneratedContent:
    """Contenido generado por el motor."""
    id: str
    content: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validar contenido generado."""
        if not self.content.strip():
            raise ValueError("content no puede estar vacío")
        
        if not self.id:
            raise ValueError("id es requerido")


@dataclass(frozen=True)
class ContentMetadata:
    """Metadatos del contenido generado."""
    language: str
    audience: str
    generation_time: str
    template_used: str
    context_used: Dict[str, Any]
    word_count: int
    reading_time_minutes: float
    
    def __post_init__(self):
        """Validar metadatos."""
        if self.word_count < 0:
            raise ValueError("word_count no puede ser negativo")
        
        if self.reading_time_minutes < 0:
            raise ValueError("reading_time_minutes no puede ser negativo")


@dataclass(frozen=True)
class QualityMetrics:
    """Métricas de calidad de las explicaciones."""
    clarity_score: float
    completeness_score: float
    accuracy_score: float
    relevance_score: float
    overall_score: float
    
    def __post_init__(self):
        """Validar métricas de calidad."""
        scores = [
            self.clarity_score,
            self.completeness_score,
            self.accuracy_score,
            self.relevance_score,
            self.overall_score
        ]
        
        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Score debe estar entre 0.0 y 1.0, recibido: {score}")


@dataclass(frozen=True)
class PerformanceMetrics:
    """Métricas de performance del motor."""
    generation_time_ms: int
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    
    def __post_init__(self):
        """Validar métricas de performance."""
        if self.generation_time_ms < 0:
            raise ValueError("generation_time_ms no puede ser negativo")
        
        if self.memory_usage_mb < 0:
            raise ValueError("memory_usage_mb no puede ser negativo")
        
        if not 0.0 <= self.cpu_usage_percent <= 100.0:
            raise ValueError("cpu_usage_percent debe estar entre 0.0 y 100.0")
        
        if not 0.0 <= self.cache_hit_rate <= 1.0:
            raise ValueError("cache_hit_rate debe estar entre 0.0 y 1.0")


@dataclass(frozen=True)
class TemplateValidationResult:
    """Resultado de validación de plantilla."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    
    def __post_init__(self):
        """Validar resultado de validación."""
        if not isinstance(self.errors, list):
            raise ValueError("errors debe ser una lista")
        
        if not isinstance(self.warnings, list):
            raise ValueError("warnings debe ser una lista")
        
        if not isinstance(self.suggestions, list):
            raise ValueError("suggestions debe ser una lista")


@dataclass(frozen=True)
class LanguageDetectionResult:
    """Resultado de detección de idioma."""
    detected_language: str
    confidence: float
    alternative_languages: List[Dict[str, Any]]
    
    def __post_init__(self):
        """Validar resultado de detección."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence debe estar entre 0.0 y 1.0")
        
        if not self.detected_language:
            raise ValueError("detected_language es requerido")


@dataclass(frozen=True)
class AudienceAdaptationResult:
    """Resultado de adaptación por audiencia."""
    original_content: str
    adapted_content: str
    adaptations_applied: List[str]
    confidence: float
    
    def __post_init__(self):
        """Validar resultado de adaptación."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence debe estar entre 0.0 y 1.0")
        
        if not self.adaptations_applied:
            raise ValueError("adaptations_applied no puede estar vacío")
