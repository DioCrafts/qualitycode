"""
Puertos de la capa de aplicación para el Motor de Explicaciones.

Este módulo define las interfaces (puertos) que la capa de aplicación
utiliza para interactuar con el dominio y la infraestructura.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ...domain.entities.explanation import (
    ComprehensiveExplanation,
    ExplanationRequest,
    InteractiveResponse,
    ExplanationContext,
    Language,
    Audience
)
from ...domain.value_objects.explanation_metrics import (
    ContentGenerationRequest,
    GeneratedContent,
    PersonalizationContext,
    QualityMetrics,
    PerformanceMetrics
)


class ExplanationEnginePort(ABC):
    """Puerto principal del motor de explicaciones."""
    
    @abstractmethod
    async def generate_explanation(
        self, 
        analysis_result: Dict[str, Any],
        request: ExplanationRequest
    ) -> ComprehensiveExplanation:
        """Generar explicación comprehensiva."""
        pass
    
    @abstractmethod
    async def generate_interactive_response(
        self, 
        question: str, 
        context: ExplanationContext
    ) -> InteractiveResponse:
        """Generar respuesta interactiva."""
        pass
    
    @abstractmethod
    async def get_supported_languages(self) -> List[Language]:
        """Obtener idiomas soportados."""
        pass
    
    @abstractmethod
    async def get_supported_audiences(self) -> List[Audience]:
        """Obtener audiencias soportadas."""
        pass


class ContentGeneratorPort(ABC):
    """Puerto para generación de contenido."""
    
    @abstractmethod
    async def generate_content(
        self, 
        request: ContentGenerationRequest
    ) -> GeneratedContent:
        """Generar contenido explicativo."""
        pass
    
    @abstractmethod
    async def adapt_content_for_audience(
        self, 
        content: str, 
        audience: Audience,
        language: Language
    ) -> str:
        """Adaptar contenido para audiencia específica."""
        pass
    
    @abstractmethod
    async def validate_content(
        self, 
        content: str, 
        language: Language
    ) -> bool:
        """Validar calidad del contenido."""
        pass


class LanguageAdapterPort(ABC):
    """Puerto para adaptación de idiomas."""
    
    @abstractmethod
    async def translate_content(
        self, 
        content: str, 
        from_language: Language,
        to_language: Language
    ) -> str:
        """Traducir contenido entre idiomas."""
        pass
    
    @abstractmethod
    async def detect_language(self, content: str) -> Language:
        """Detectar idioma del contenido."""
        pass
    
    @abstractmethod
    async def adapt_cultural_context(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Adaptar contexto cultural."""
        pass


class AudienceAdapterPort(ABC):
    """Puerto para adaptación por audiencia."""
    
    @abstractmethod
    async def adapt_for_audience(
        self, 
        content: str, 
        audience: Audience,
        language: Language
    ) -> str:
        """Adaptar contenido para audiencia específica."""
        pass
    
    @abstractmethod
    async def get_audience_characteristics(
        self, 
        audience: Audience
    ) -> Dict[str, Any]:
        """Obtener características de la audiencia."""
        pass


class TemplateEnginePort(ABC):
    """Puerto para motor de plantillas."""
    
    @abstractmethod
    async def render_template(
        self, 
        template_key: str, 
        variables: Dict[str, str],
        language: Language
    ) -> str:
        """Renderizar plantilla con variables."""
        pass
    
    @abstractmethod
    async def get_template_variables(
        self, 
        template_key: str
    ) -> List[str]:
        """Obtener variables de una plantilla."""
        pass
    
    @abstractmethod
    async def validate_template(
        self, 
        template_key: str, 
        language: Language
    ) -> bool:
        """Validar plantilla."""
        pass


class EducationalContentPort(ABC):
    """Puerto para contenido educativo."""
    
    @abstractmethod
    async def generate_educational_content(
        self, 
        analysis_result: Dict[str, Any],
        request: ExplanationRequest
    ) -> List[Dict[str, Any]]:
        """Generar contenido educativo."""
        pass
    
    @abstractmethod
    async def generate_learning_path(
        self, 
        concept: str, 
        audience: Audience,
        language: Language
    ) -> List[Dict[str, Any]]:
        """Generar ruta de aprendizaje."""
        pass
    
    @abstractmethod
    async def generate_examples(
        self, 
        concept: str, 
        audience: Audience,
        language: Language
    ) -> List[Dict[str, Any]]:
        """Generar ejemplos educativos."""
        pass


class InteractiveExplainerPort(ABC):
    """Puerto para explicaciones interactivas."""
    
    @abstractmethod
    async def generate_interactive_elements(
        self, 
        analysis_result: Dict[str, Any],
        request: ExplanationRequest
    ) -> List[Dict[str, Any]]:
        """Generar elementos interactivos."""
        pass
    
    @abstractmethod
    async def handle_user_question(
        self, 
        question: str, 
        context: ExplanationContext
    ) -> InteractiveResponse:
        """Manejar pregunta del usuario."""
        pass
    
    @abstractmethod
    async def generate_qa_elements(
        self, 
        analysis_result: Dict[str, Any],
        request: ExplanationRequest
    ) -> List[Dict[str, Any]]:
        """Generar elementos de Q&A."""
        pass


class MultimediaGeneratorPort(ABC):
    """Puerto para generación multimedia."""
    
    @abstractmethod
    async def generate_visualizations(
        self, 
        analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generar visualizaciones."""
        pass
    
    @abstractmethod
    async def generate_diagrams(
        self, 
        analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generar diagramas."""
        pass
    
    @abstractmethod
    async def generate_code_comparisons(
        self, 
        original_code: str, 
        fixed_code: str,
        language: str
    ) -> Dict[str, Any]:
        """Generar comparaciones de código."""
        pass


class PersonalizationPort(ABC):
    """Puerto para personalización."""
    
    @abstractmethod
    async def personalize_content(
        self, 
        content: str, 
        context: PersonalizationContext
    ) -> str:
        """Personalizar contenido."""
        pass
    
    @abstractmethod
    async def get_user_preferences(
        self, 
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Obtener preferencias del usuario."""
        pass
    
    @abstractmethod
    async def update_user_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, Any]
    ) -> None:
        """Actualizar preferencias del usuario."""
        pass


class QualityAssessmentPort(ABC):
    """Puerto para evaluación de calidad."""
    
    @abstractmethod
    async def assess_explanation_quality(
        self, 
        explanation: ComprehensiveExplanation
    ) -> QualityMetrics:
        """Evaluar calidad de explicación."""
        pass
    
    @abstractmethod
    async def assess_content_quality(
        self, 
        content: str, 
        language: Language
    ) -> float:
        """Evaluar calidad de contenido."""
        pass


class PerformanceMonitoringPort(ABC):
    """Puerto para monitoreo de performance."""
    
    @abstractmethod
    async def measure_operation_time(
        self, 
        operation: str
    ) -> int:
        """Medir tiempo de operación."""
        pass
    
    @abstractmethod
    async def get_performance_metrics(
        self, 
        operation: str
    ) -> PerformanceMetrics:
        """Obtener métricas de performance."""
        pass
    
    @abstractmethod
    async def optimize_performance(
        self, 
        operation: str,
        target_time_ms: int
    ) -> Dict[str, Any]:
        """Optimizar performance."""
        pass


class CachePort(ABC):
    """Puerto para caché."""
    
    @abstractmethod
    async def get_cached_explanation(
        self, 
        cache_key: str
    ) -> Optional[ComprehensiveExplanation]:
        """Obtener explicación del caché."""
        pass
    
    @abstractmethod
    async def cache_explanation(
        self, 
        cache_key: str, 
        explanation: ComprehensiveExplanation,
        ttl_seconds: int = 3600
    ) -> None:
        """Guardar explicación en caché."""
        pass
    
    @abstractmethod
    async def invalidate_cache(self, cache_key: str) -> None:
        """Invalidar entrada del caché."""
        pass


class AnalyticsPort(ABC):
    """Puerto para analytics."""
    
    @abstractmethod
    async def track_explanation_generation(
        self, 
        explanation_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Rastrear generación de explicación."""
        pass
    
    @abstractmethod
    async def track_user_interaction(
        self, 
        user_id: str, 
        interaction_data: Dict[str, Any]
    ) -> None:
        """Rastrear interacción del usuario."""
        pass
    
    @abstractmethod
    async def get_usage_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de uso."""
        pass
