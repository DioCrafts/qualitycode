"""
DTOs (Data Transfer Objects) para el Motor de Explicaciones.

Este módulo define los DTOs utilizados para transferir datos
entre las capas de la aplicación.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID


@dataclass
class ExplanationRequestDto:
    """DTO para request de explicación."""
    analysis_id: str
    language: str
    audience: str
    depth: str
    include_examples: bool = True
    include_visualizations: bool = True
    include_educational_content: bool = True
    focus_areas: List[str] = field(default_factory=list)
    personalization_context: Optional[Dict[str, Any]] = None


@dataclass
class InteractiveQueryDto:
    """DTO para consulta interactiva."""
    question: str
    explanation_id: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class ExplanationSectionDto:
    """DTO para sección de explicación."""
    id: str
    title: str
    content: str
    section_type: str
    importance: str
    interactive_elements: List[Dict[str, Any]] = field(default_factory=list)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class InteractiveElementDto:
    """DTO para elemento interactivo."""
    id: str
    element_type: str
    title: str
    content: str
    initial_state: str
    triggers: List[str]
    related_violation_id: Optional[str] = None


@dataclass
class VisualizationDto:
    """DTO para visualización."""
    id: str
    visualization_type: str
    title: str
    content: Dict[str, Any]
    description: str
    interactive: bool = True


@dataclass
class EducationalContentDto:
    """DTO para contenido educativo."""
    id: str
    title: str
    content: str
    content_type: str
    difficulty_level: str
    learning_objectives: List[str]
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class ActionItemDto:
    """DTO para item de acción."""
    id: str
    title: str
    description: str
    priority: str
    estimated_effort: str
    related_violation_id: Optional[str] = None


@dataclass
class ReferenceDto:
    """DTO para referencia."""
    id: str
    title: str
    url: str
    description: str
    reference_type: str


@dataclass
class ComprehensiveExplanationDto:
    """DTO para explicación comprehensiva."""
    id: str
    language: str
    audience: str
    summary: str
    detailed_sections: List[ExplanationSectionDto]
    visualizations: List[VisualizationDto] = field(default_factory=list)
    interactive_elements: List[InteractiveElementDto] = field(default_factory=list)
    educational_content: List[EducationalContentDto] = field(default_factory=list)
    action_items: List[ActionItemDto] = field(default_factory=list)
    glossary: Dict[str, str] = field(default_factory=dict)
    references: List[ReferenceDto] = field(default_factory=list)
    generation_time_ms: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InteractiveResponseDto:
    """DTO para respuesta interactiva."""
    response_text: str
    response_type: str
    confidence: float
    follow_up_questions: List[str] = field(default_factory=list)
    related_content: List[str] = field(default_factory=list)
    action_suggestions: List[str] = field(default_factory=list)


@dataclass
class LanguageDto:
    """DTO para idioma."""
    code: str
    name: str
    native_name: str
    supported: bool = True


@dataclass
class AudienceDto:
    """DTO para audiencia."""
    code: str
    name: str
    description: str
    characteristics: List[str] = field(default_factory=list)


@dataclass
class TemplateDto:
    """DTO para plantilla."""
    key: str
    language: str
    title: str
    description: str
    variables: List[str] = field(default_factory=list)
    content: str = ""


@dataclass
class QualityMetricsDto:
    """DTO para métricas de calidad."""
    clarity_score: float
    completeness_score: float
    accuracy_score: float
    relevance_score: float
    overall_score: float


@dataclass
class PerformanceMetricsDto:
    """DTO para métricas de performance."""
    generation_time_ms: int
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float


@dataclass
class PersonalizationContextDto:
    """DTO para contexto de personalización."""
    experience_level: str
    preferred_learning_style: str
    known_technologies: List[str]
    role: str
    previous_interactions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ContentGenerationRequestDto:
    """DTO para request de generación de contenido."""
    template_key: str
    language: str
    audience: str
    context_variables: Dict[str, str]
    personalization_context: Optional[PersonalizationContextDto] = None


@dataclass
class GeneratedContentDto:
    """DTO para contenido generado."""
    id: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class UserPreferencesDto:
    """DTO para preferencias del usuario."""
    user_id: str
    preferred_language: str
    preferred_audience: str
    preferred_depth: str
    include_examples: bool = True
    include_visualizations: bool = True
    include_educational_content: bool = True
    focus_areas: List[str] = field(default_factory=list)


@dataclass
class LearningProgressDto:
    """DTO para progreso de aprendizaje."""
    user_id: str
    concept: str
    mastery_level: float
    last_accessed: datetime
    interactions_count: int
    completion_percentage: float


@dataclass
class AnalyticsDataDto:
    """DTO para datos de analytics."""
    event_type: str
    timestamp: datetime
    user_id: Optional[str] = None
    explanation_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStatisticsDto:
    """DTO para estadísticas de caché."""
    total_entries: int
    hit_rate: float
    miss_rate: float
    memory_usage_mb: float
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None


@dataclass
class UsageStatisticsDto:
    """DTO para estadísticas de uso."""
    total_explanations_generated: int
    total_interactive_queries: int
    average_generation_time_ms: float
    most_used_language: str
    most_used_audience: str
    daily_usage: Dict[str, int] = field(default_factory=dict)
    weekly_usage: Dict[str, int] = field(default_factory=dict)
    monthly_usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class ErrorResponseDto:
    """DTO para respuesta de error."""
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SuccessResponseDto:
    """DTO para respuesta exitosa."""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
