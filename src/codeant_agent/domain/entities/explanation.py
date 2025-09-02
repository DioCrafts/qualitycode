"""
Entidades del dominio para el Motor de Explicaciones en Lenguaje Natural.

Este módulo define las entidades principales del motor de explicaciones,
siguiendo los principios de la arquitectura hexagonal y DDD.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4


class Language(Enum):
    """Idiomas soportados por el motor de explicaciones."""
    SPANISH = "es"
    ENGLISH = "en"
    PORTUGUESE = "pt"
    FRENCH = "fr"
    GERMAN = "de"


class Audience(Enum):
    """Audiencias objetivo para las explicaciones."""
    JUNIOR_DEVELOPER = "junior_developer"
    SENIOR_DEVELOPER = "senior_developer"
    TECHNICAL_LEAD = "technical_lead"
    SOFTWARE_ARCHITECT = "software_architect"
    PROJECT_MANAGER = "project_manager"
    QUALITY_ASSURANCE = "quality_assurance"
    SECURITY_TEAM = "security_team"
    BUSINESS_STAKEHOLDER = "business_stakeholder"


class ExplanationDepth(Enum):
    """Niveles de profundidad de las explicaciones."""
    BRIEF = "brief"  # 1-2 sentences
    STANDARD = "standard"  # 1-2 paragraphs
    DETAILED = "detailed"  # Multiple paragraphs with examples
    COMPREHENSIVE = "comprehensive"  # Full tutorial-style explanation


class SectionType(Enum):
    """Tipos de secciones en las explicaciones."""
    SUMMARY = "summary"
    ISSUES = "issues"
    METRICS = "metrics"
    ANTIPATTERNS = "antipatterns"
    RECOMMENDATIONS = "recommendations"
    SECURITY = "security"
    PERFORMANCE = "performance"
    EDUCATIONAL = "educational"
    EXAMPLES = "examples"


class SectionImportance(Enum):
    """Niveles de importancia de las secciones."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InteractiveElementType(Enum):
    """Tipos de elementos interactivos."""
    EXPANDABLE_SECTION = "expandable_section"
    QUESTION_ANSWER = "question_answer"
    CODE_COMPARISON = "code_comparison"
    STEP_BY_STEP_GUIDE = "step_by_step_guide"
    INTERACTIVE_TUTORIAL = "interactive_tutorial"
    PROGRESSIVE_DISCLOSURE = "progressive_disclosure"
    CONDITIONAL_CONTENT = "conditional_content"


class InteractiveState(Enum):
    """Estados de los elementos interactivos."""
    COLLAPSED = "collapsed"
    EXPANDED = "expanded"
    HIDDEN = "hidden"
    HIGHLIGHTED = "highlighted"


class InteractiveTrigger(Enum):
    """Triggers para elementos interactivos."""
    CLICK = "click"
    HOVER = "hover"
    SCROLL = "scroll"
    TIME_DELAY = "time_delay"
    USER_PROGRESS = "user_progress"


class ResponseType(Enum):
    """Tipos de respuesta del sistema interactivo."""
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    EXAMPLE = "example"
    GUIDANCE = "guidance"
    CLARIFICATION = "clarification"
    RECOMMENDATION = "recommendation"


@dataclass
class ExplanationId:
    """Identificador único para explicaciones."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass
class SectionId:
    """Identificador único para secciones."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass
class InteractiveElementId:
    """Identificador único para elementos interactivos."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass
class VisualizationId:
    """Identificador único para visualizaciones."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass
class ExplanationRequest:
    """Request para generar una explicación."""
    language: Language
    audience: Audience
    depth: ExplanationDepth
    include_examples: bool = True
    include_visualizations: bool = True
    include_educational_content: bool = True
    focus_areas: List[str] = field(default_factory=list)
    personalization_context: Optional[Dict[str, Any]] = None


@dataclass
class InteractiveElement:
    """Elemento interactivo en una explicación."""
    id: InteractiveElementId
    element_type: InteractiveElementType
    title: str
    content: str
    initial_state: InteractiveState
    triggers: List[InteractiveTrigger]
    related_violation_id: Optional[str] = None


@dataclass
class Visualization:
    """Visualización en una explicación."""
    id: VisualizationId
    visualization_type: str
    title: str
    content: Dict[str, Any]
    description: str
    interactive: bool = True


@dataclass
class EducationalContent:
    """Contenido educativo en una explicación."""
    id: str
    title: str
    content: str
    content_type: str
    difficulty_level: str
    learning_objectives: List[str]
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class ActionItem:
    """Item de acción recomendado."""
    id: str
    title: str
    description: str
    priority: str
    estimated_effort: str
    related_violation_id: Optional[str] = None


@dataclass
class Reference:
    """Referencia externa."""
    id: str
    title: str
    url: str
    description: str
    reference_type: str


@dataclass
class ExplanationSection:
    """Sección de una explicación."""
    id: SectionId
    title: str
    content: str
    section_type: SectionType
    importance: SectionImportance
    interactive_elements: List[InteractiveElement] = field(default_factory=list)
    visualizations: List[Visualization] = field(default_factory=list)


@dataclass
class ComprehensiveExplanation:
    """Explicación comprehensiva generada por el motor."""
    id: ExplanationId
    language: Language
    audience: Audience
    summary: str
    detailed_sections: List[ExplanationSection]
    visualizations: List[Visualization] = field(default_factory=list)
    interactive_elements: List[InteractiveElement] = field(default_factory=list)
    educational_content: List[EducationalContent] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)
    glossary: Dict[str, str] = field(default_factory=dict)
    references: List[Reference] = field(default_factory=list)
    generation_time_ms: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InteractiveResponse:
    """Respuesta del sistema interactivo."""
    response_text: str
    response_type: ResponseType
    confidence: float
    follow_up_questions: List[str] = field(default_factory=list)
    related_content: List[str] = field(default_factory=list)
    action_suggestions: List[str] = field(default_factory=list)


@dataclass
class ExplanationContext:
    """Contexto para explicaciones interactivas."""
    explanation_id: ExplanationId
    current_section: Optional[SectionId] = None
    user_interactions: List[Dict[str, Any]] = field(default_factory=list)
    session_data: Dict[str, Any] = field(default_factory=dict)