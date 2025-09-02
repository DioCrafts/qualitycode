"""
Configuración del Motor de Explicaciones.

Este módulo contiene la configuración y constantes utilizadas
por el motor de explicaciones en lenguaje natural.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum


class PerformanceTarget(Enum):
    """Targets de performance del motor."""
    MAX_GENERATION_TIME_MS = 3000  # 3 segundos máximo para generación
    MAX_INTERACTIVE_RESPONSE_TIME_MS = 1000  # 1 segundo máximo para respuesta interactiva
    MAX_CACHE_TTL_SECONDS = 3600  # 1 hora de TTL para caché
    MAX_CONTENT_LENGTH = 10000  # 10,000 caracteres máximo por contenido


class QualityThreshold(Enum):
    """Umbrales de calidad."""
    MIN_CONFIDENCE_SCORE = 0.7  # Mínimo 70% de confianza
    MIN_EXPLANATION_LENGTH = 100  # Mínimo 100 caracteres por explicación
    MAX_EXPLANATION_LENGTH = 50000  # Máximo 50,000 caracteres por explicación
    MIN_SECTIONS_COUNT = 3  # Mínimo 3 secciones por explicación


@dataclass
class ExplanationConfig:
    """Configuración principal del motor de explicaciones."""
    
    # Configuración de idiomas
    default_language: str = "es"
    supported_languages: List[str] = None
    
    # Configuración de audiencias
    default_audience: str = "senior_developer"
    supported_audiences: List[str] = None
    
    # Configuración de performance
    max_generation_time_ms: int = PerformanceTarget.MAX_GENERATION_TIME_MS.value
    max_interactive_response_time_ms: int = PerformanceTarget.MAX_INTERACTIVE_RESPONSE_TIME_MS.value
    max_cache_ttl_seconds: int = PerformanceTarget.MAX_CACHE_TTL_SECONDS.value
    
    # Configuración de calidad
    min_confidence_score: float = QualityThreshold.MIN_CONFIDENCE_SCORE.value
    min_explanation_length: int = QualityThreshold.MIN_EXPLANATION_LENGTH.value
    max_explanation_length: int = QualityThreshold.MAX_EXPLANATION_LENGTH.value
    min_sections_count: int = QualityThreshold.MIN_SECTIONS_COUNT.value
    
    # Configuración de contenido
    max_content_length: int = PerformanceTarget.MAX_CONTENT_LENGTH.value
    include_examples_by_default: bool = True
    include_visualizations_by_default: bool = True
    include_educational_content_by_default: bool = True
    
    # Configuración de caché
    enable_caching: bool = True
    cache_size_limit: int = 1000  # Máximo 1000 entradas en caché
    
    # Configuración de analytics
    enable_analytics: bool = True
    track_performance_metrics: bool = True
    track_user_interactions: bool = True
    
    # Configuración de personalización
    enable_personalization: bool = True
    max_personalization_context_size: int = 1000
    
    def __post_init__(self):
        """Inicializar valores por defecto."""
        if self.supported_languages is None:
            self.supported_languages = ["es", "en"]
        
        if self.supported_audiences is None:
            self.supported_audiences = [
                "junior_developer",
                "senior_developer", 
                "technical_lead",
                "software_architect",
                "project_manager",
                "quality_assurance",
                "security_team",
                "business_stakeholder"
            ]


@dataclass
class TemplateConfig:
    """Configuración de plantillas."""
    
    # Configuración de plantillas base
    template_directory: str = "templates"
    default_template_encoding: str = "utf-8"
    
    # Configuración de validación
    validate_templates_on_load: bool = True
    max_template_size: int = 10000  # 10KB máximo por plantilla
    allowed_template_variables: List[str] = None
    
    # Configuración de renderizado
    max_rendering_time_ms: int = 500  # 500ms máximo para renderizar plantilla
    enable_template_caching: bool = True
    
    def __post_init__(self):
        """Inicializar valores por defecto."""
        if self.allowed_template_variables is None:
            self.allowed_template_variables = [
                "total_issues", "critical_issues", "high_issues", "quality_score",
                "violation_title", "location", "severity", "category",
                "violation_description", "why_problematic", "how_to_fix",
                "concept_name", "concept_definition", "importance_explanation",
                "practical_examples", "best_practices", "learning_resources"
            ]


@dataclass
class LanguageConfig:
    """Configuración de idiomas."""
    
    # Configuración de detección
    enable_auto_detection: bool = True
    detection_confidence_threshold: float = 0.6
    fallback_language: str = "en"
    
    # Configuración de traducción
    enable_translation: bool = True
    translation_cache_size: int = 500
    max_translation_length: int = 5000
    
    # Configuración cultural
    enable_cultural_adaptation: bool = True
    cultural_adaptation_level: str = "medium"  # low, medium, high


@dataclass
class AudienceConfig:
    """Configuración de audiencias."""
    
    # Configuración de adaptación
    enable_audience_adaptation: bool = True
    adaptation_confidence_threshold: float = 0.8
    
    # Configuración de terminología
    enable_terminology_adaptation: bool = True
    terminology_simplification_level: str = "medium"  # low, medium, high
    
    # Configuración de tono
    enable_tone_adaptation: bool = True
    tone_formality_level: str = "professional"  # casual, professional, formal


@dataclass
class EducationalConfig:
    """Configuración de contenido educativo."""
    
    # Configuración de generación
    enable_educational_content: bool = True
    max_educational_items: int = 10
    educational_content_depth: str = "intermediate"  # basic, intermediate, advanced
    
    # Configuración de ejemplos
    enable_examples: bool = True
    max_examples_per_concept: int = 3
    example_complexity_level: str = "medium"  # simple, medium, complex
    
    # Configuración de rutas de aprendizaje
    enable_learning_paths: bool = True
    max_learning_path_length: int = 5
    learning_path_adaptation: bool = True


@dataclass
class InteractiveConfig:
    """Configuración de explicaciones interactivas."""
    
    # Configuración de Q&A
    enable_interactive_qa: bool = True
    max_qa_elements: int = 5
    qa_confidence_threshold: float = 0.7
    
    # Configuración de elementos expandibles
    enable_expandable_sections: bool = True
    max_expandable_sections: int = 8
    default_expansion_state: str = "collapsed"  # collapsed, expanded
    
    # Configuración de tutoriales
    enable_interactive_tutorials: bool = True
    max_tutorial_steps: int = 10
    tutorial_adaptation: bool = True


@dataclass
class MultimediaConfig:
    """Configuración de contenido multimedia."""
    
    # Configuración de visualizaciones
    enable_visualizations: bool = True
    max_visualizations: int = 5
    visualization_types: List[str] = None
    
    # Configuración de diagramas
    enable_diagrams: bool = True
    max_diagrams: int = 3
    diagram_types: List[str] = None
    
    # Configuración de comparaciones de código
    enable_code_comparisons: bool = True
    max_code_comparisons: int = 3
    code_highlighting: bool = True
    
    def __post_init__(self):
        """Inicializar valores por defecto."""
        if self.visualization_types is None:
            self.visualization_types = ["radar", "doughnut", "line", "bar"]
        
        if self.diagram_types is None:
            self.diagram_types = ["mermaid", "flowchart", "dependency"]


@dataclass
class CacheConfig:
    """Configuración de caché."""
    
    # Configuración de caché principal
    enable_explanation_cache: bool = True
    explanation_cache_ttl: int = 3600  # 1 hora
    explanation_cache_size: int = 1000
    
    # Configuración de caché de traducciones
    enable_translation_cache: bool = True
    translation_cache_ttl: int = 7200  # 2 horas
    translation_cache_size: int = 500
    
    # Configuración de caché de plantillas
    enable_template_cache: bool = True
    template_cache_ttl: int = 86400  # 24 horas
    template_cache_size: int = 100


@dataclass
class AnalyticsConfig:
    """Configuración de analytics."""
    
    # Configuración de métricas
    enable_performance_metrics: bool = True
    enable_quality_metrics: bool = True
    enable_usage_metrics: bool = True
    
    # Configuración de tracking
    track_generation_times: bool = True
    track_user_interactions: bool = True
    track_error_rates: bool = True
    
    # Configuración de reportes
    enable_performance_reports: bool = True
    report_generation_interval: int = 3600  # 1 hora
    max_report_history: int = 24  # 24 reportes


@dataclass
class SecurityConfig:
    """Configuración de seguridad."""
    
    # Configuración de validación
    validate_input_content: bool = True
    max_input_length: int = 100000  # 100KB máximo
    sanitize_output: bool = True
    
    # Configuración de rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    
    # Configuración de logging
    enable_security_logging: bool = True
    log_suspicious_activity: bool = True
    log_failed_requests: bool = True


@dataclass
class CompleteExplanationConfig:
    """Configuración completa del motor de explicaciones."""
    
    # Configuraciones principales
    explanation: ExplanationConfig = None
    template: TemplateConfig = None
    language: LanguageConfig = None
    audience: AudienceConfig = None
    educational: EducationalConfig = None
    interactive: InteractiveConfig = None
    multimedia: MultimediaConfig = None
    cache: CacheConfig = None
    analytics: AnalyticsConfig = None
    security: SecurityConfig = None
    
    def __post_init__(self):
        """Inicializar configuraciones por defecto."""
        if self.explanation is None:
            self.explanation = ExplanationConfig()
        
        if self.template is None:
            self.template = TemplateConfig()
        
        if self.language is None:
            self.language = LanguageConfig()
        
        if self.audience is None:
            self.audience = AudienceConfig()
        
        if self.educational is None:
            self.educational = EducationalConfig()
        
        if self.interactive is None:
            self.interactive = InteractiveConfig()
        
        if self.multimedia is None:
            self.multimedia = MultimediaConfig()
        
        if self.cache is None:
            self.cache = CacheConfig()
        
        if self.analytics is None:
            self.analytics = AnalyticsConfig()
        
        if self.security is None:
            self.security = SecurityConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario."""
        return {
            "explanation": self.explanation.__dict__,
            "template": self.template.__dict__,
            "language": self.language.__dict__,
            "audience": self.audience.__dict__,
            "educational": self.educational.__dict__,
            "interactive": self.interactive.__dict__,
            "multimedia": self.multimedia.__dict__,
            "cache": self.cache.__dict__,
            "analytics": self.analytics.__dict__,
            "security": self.security.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CompleteExplanationConfig':
        """Crear configuración desde diccionario."""
        config = cls()
        
        if "explanation" in config_dict:
            config.explanation = ExplanationConfig(**config_dict["explanation"])
        
        if "template" in config_dict:
            config.template = TemplateConfig(**config_dict["template"])
        
        if "language" in config_dict:
            config.language = LanguageConfig(**config_dict["language"])
        
        if "audience" in config_dict:
            config.audience = AudienceConfig(**config_dict["audience"])
        
        if "educational" in config_dict:
            config.educational = EducationalConfig(**config_dict["educational"])
        
        if "interactive" in config_dict:
            config.interactive = InteractiveConfig(**config_dict["interactive"])
        
        if "multimedia" in config_dict:
            config.multimedia = MultimediaConfig(**config_dict["multimedia"])
        
        if "cache" in config_dict:
            config.cache = CacheConfig(**config_dict["cache"])
        
        if "analytics" in config_dict:
            config.analytics = AnalyticsConfig(**config_dict["analytics"])
        
        if "security" in config_dict:
            config.security = SecurityConfig(**config_dict["security"])
        
        return config


# Configuración por defecto
DEFAULT_CONFIG = CompleteExplanationConfig()

# Configuraciones predefinidas
DEVELOPMENT_CONFIG = CompleteExplanationConfig(
    explanation=ExplanationConfig(
        max_generation_time_ms=5000,  # Más tiempo en desarrollo
        enable_caching=False,  # Sin caché en desarrollo
        enable_analytics=False  # Sin analytics en desarrollo
    ),
    security=SecurityConfig(
        enable_rate_limiting=False,  # Sin rate limiting en desarrollo
        validate_input_content=False  # Validación relajada en desarrollo
    )
)

PRODUCTION_CONFIG = CompleteExplanationConfig(
    explanation=ExplanationConfig(
        max_generation_time_ms=2000,  # Menos tiempo en producción
        enable_caching=True,
        enable_analytics=True
    ),
    security=SecurityConfig(
        enable_rate_limiting=True,
        validate_input_content=True,
        max_requests_per_minute=30  # Rate limiting más estricto
    )
)

TESTING_CONFIG = CompleteExplanationConfig(
    explanation=ExplanationConfig(
        max_generation_time_ms=10000,  # Tiempo generoso para tests
        enable_caching=False,
        enable_analytics=False
    ),
    template=TemplateConfig(
        validate_templates_on_load=False  # Sin validación en tests
    ),
    security=SecurityConfig(
        enable_rate_limiting=False,
        validate_input_content=False
    )
)
