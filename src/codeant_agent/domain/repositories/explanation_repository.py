"""
Repositorios del dominio para el Motor de Explicaciones.

Este módulo define las interfaces de repositorios para el motor de explicaciones,
siguiendo los principios de inversión de dependencias.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID

from ..entities.explanation import (
    ComprehensiveExplanation,
    ExplanationId,
    ExplanationRequest,
    InteractiveResponse,
    ExplanationContext
)
from ..value_objects.explanation_metrics import (
    ContentGenerationRequest,
    GeneratedContent,
    TemplateValidationResult,
    LanguageDetectionResult,
    AudienceAdaptationResult
)


class ExplanationRepository(ABC):
    """Repositorio para explicaciones comprehensivas."""
    
    @abstractmethod
    async def save(self, explanation: ComprehensiveExplanation) -> None:
        """Guardar una explicación comprehensiva."""
        pass
    
    @abstractmethod
    async def find_by_id(self, explanation_id: ExplanationId) -> Optional[ComprehensiveExplanation]:
        """Buscar explicación por ID."""
        pass
    
    @abstractmethod
    async def find_by_analysis_id(self, analysis_id: str) -> List[ComprehensiveExplanation]:
        """Buscar explicaciones por ID de análisis."""
        pass
    
    @abstractmethod
    async def find_by_language_and_audience(
        self, 
        language: str, 
        audience: str
    ) -> List[ComprehensiveExplanation]:
        """Buscar explicaciones por idioma y audiencia."""
        pass
    
    @abstractmethod
    async def delete(self, explanation_id: ExplanationId) -> None:
        """Eliminar una explicación."""
        pass
    
    @abstractmethod
    async def count_by_language(self, language: str) -> int:
        """Contar explicaciones por idioma."""
        pass


class TemplateRepository(ABC):
    """Repositorio para plantillas de explicación."""
    
    @abstractmethod
    async def get_template(self, key: str, language: str) -> str:
        """Obtener plantilla por clave e idioma."""
        pass
    
    @abstractmethod
    async def get_all_templates(self, language: str) -> Dict[str, str]:
        """Obtener todas las plantillas para un idioma."""
        pass
    
    @abstractmethod
    async def save_template(self, key: str, language: str, template: str) -> None:
        """Guardar una plantilla."""
        pass
    
    @abstractmethod
    async def validate_template(self, template: str) -> TemplateValidationResult:
        """Validar una plantilla."""
        pass
    
    @abstractmethod
    async def get_template_variables(self, template: str) -> List[str]:
        """Obtener variables de una plantilla."""
        pass
    
    @abstractmethod
    async def exists(self, key: str, language: str) -> bool:
        """Verificar si existe una plantilla."""
        pass


class ContentRepository(ABC):
    """Repositorio para contenido generado."""
    
    @abstractmethod
    async def save_generated_content(self, content: GeneratedContent) -> None:
        """Guardar contenido generado."""
        pass
    
    @abstractmethod
    async def find_content_by_id(self, content_id: str) -> Optional[GeneratedContent]:
        """Buscar contenido por ID."""
        pass
    
    @abstractmethod
    async def find_content_by_template(
        self, 
        template_key: str, 
        language: str
    ) -> List[GeneratedContent]:
        """Buscar contenido por plantilla e idioma."""
        pass
    
    @abstractmethod
    async def delete_content(self, content_id: str) -> None:
        """Eliminar contenido."""
        pass
    
    @abstractmethod
    async def get_content_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de contenido."""
        pass


class InteractionRepository(ABC):
    """Repositorio para interacciones del usuario."""
    
    @abstractmethod
    async def save_interaction(
        self, 
        explanation_id: ExplanationId,
        interaction_data: Dict[str, Any]
    ) -> None:
        """Guardar interacción del usuario."""
        pass
    
    @abstractmethod
    async def get_interactions(
        self, 
        explanation_id: ExplanationId
    ) -> List[Dict[str, Any]]:
        """Obtener interacciones de una explicación."""
        pass
    
    @abstractmethod
    async def get_user_interaction_history(
        self, 
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Obtener historial de interacciones del usuario."""
        pass
    
    @abstractmethod
    async def save_interactive_response(
        self, 
        explanation_id: ExplanationId,
        response: InteractiveResponse
    ) -> None:
        """Guardar respuesta interactiva."""
        pass
    
    @abstractmethod
    async def get_interactive_responses(
        self, 
        explanation_id: ExplanationId
    ) -> List[InteractiveResponse]:
        """Obtener respuestas interactivas de una explicación."""
        pass


class PersonalizationRepository(ABC):
    """Repositorio para datos de personalización."""
    
    @abstractmethod
    async def save_user_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, Any]
    ) -> None:
        """Guardar preferencias del usuario."""
        pass
    
    @abstractmethod
    async def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtener preferencias del usuario."""
        pass
    
    @abstractmethod
    async def save_learning_progress(
        self, 
        user_id: str, 
        progress_data: Dict[str, Any]
    ) -> None:
        """Guardar progreso de aprendizaje."""
        pass
    
    @abstractmethod
    async def get_learning_progress(self, user_id: str) -> Dict[str, Any]:
        """Obtener progreso de aprendizaje."""
        pass
    
    @abstractmethod
    async def update_user_experience_level(
        self, 
        user_id: str, 
        experience_level: str
    ) -> None:
        """Actualizar nivel de experiencia del usuario."""
        pass


class CacheRepository(ABC):
    """Repositorio para caché de explicaciones."""
    
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
    
    @abstractmethod
    async def clear_cache(self) -> None:
        """Limpiar todo el caché."""
        pass
    
    @abstractmethod
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché."""
        pass


class AnalyticsRepository(ABC):
    """Repositorio para analytics y métricas."""
    
    @abstractmethod
    async def save_explanation_metrics(
        self, 
        explanation_id: ExplanationId,
        metrics: Dict[str, Any]
    ) -> None:
        """Guardar métricas de explicación."""
        pass
    
    @abstractmethod
    async def get_explanation_metrics(
        self, 
        explanation_id: ExplanationId
    ) -> Optional[Dict[str, Any]]:
        """Obtener métricas de explicación."""
        pass
    
    @abstractmethod
    async def save_performance_metrics(
        self, 
        operation: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Guardar métricas de performance."""
        pass
    
    @abstractmethod
    async def get_performance_metrics(
        self, 
        operation: str,
        time_range: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Obtener métricas de performance."""
        pass
    
    @abstractmethod
    async def get_usage_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de uso."""
        pass
