"""
Servicios del dominio para el Motor de Explicaciones.

Este módulo define los servicios de dominio que encapsulan la lógica de negocio
del motor de explicaciones, siguiendo los principios de DDD.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..entities.explanation import (
    ComprehensiveExplanation,
    ExplanationRequest,
    InteractiveResponse,
    ExplanationContext,
    Language,
    Audience
)
from ..value_objects.explanation_metrics import (
    ContentGenerationRequest,
    GeneratedContent,
    PersonalizationContext,
    QualityMetrics,
    PerformanceMetrics
)


class ContentGenerationService(ABC):
    """Servicio para generación de contenido explicativo."""
    
    @abstractmethod
    async def generate_content(
        self, 
        request: ContentGenerationRequest
    ) -> GeneratedContent:
        """Generar contenido basado en request."""
        pass
    
    @abstractmethod
    async def adapt_content_for_audience(
        self, 
        content: str, 
        audience: Audience,
        language: Language
    ) -> str:
        """Adaptar contenido para una audiencia específica."""
        pass
    
    @abstractmethod
    async def simplify_technical_terms(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Simplificar términos técnicos en el contenido."""
        pass
    
    @abstractmethod
    async def add_business_context(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Agregar contexto de negocio al contenido."""
        pass
    
    @abstractmethod
    async def emphasize_security_aspects(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Enfatizar aspectos de seguridad en el contenido."""
        pass


class LanguageAdaptationService(ABC):
    """Servicio para adaptación de idiomas."""
    
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
    async def translate_technical_terms(
        self, 
        terms: List[str], 
        to_language: Language
    ) -> Dict[str, str]:
        """Traducir términos técnicos."""
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
        """Adaptar contexto cultural del contenido."""
        pass
    
    @abstractmethod
    async def extract_technical_terms(self, content: str) -> List[str]:
        """Extraer términos técnicos del contenido."""
        pass


class AudienceAdaptationService(ABC):
    """Servicio para adaptación por audiencia."""
    
    @abstractmethod
    async def adapt_for_junior_developer(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Adaptar contenido para desarrollador junior."""
        pass
    
    @abstractmethod
    async def adapt_for_senior_developer(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Adaptar contenido para desarrollador senior."""
        pass
    
    @abstractmethod
    async def adapt_for_project_manager(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Adaptar contenido para project manager."""
        pass
    
    @abstractmethod
    async def adapt_for_security_team(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Adaptar contenido para equipo de seguridad."""
        pass
    
    @abstractmethod
    async def adapt_for_business_stakeholder(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Adaptar contenido para stakeholder de negocio."""
        pass


class EducationalContentService(ABC):
    """Servicio para contenido educativo."""
    
    @abstractmethod
    async def generate_learning_path(
        self, 
        concept: str, 
        audience: Audience,
        language: Language
    ) -> List[Dict[str, Any]]:
        """Generar ruta de aprendizaje para un concepto."""
        pass
    
    @abstractmethod
    async def generate_examples(
        self, 
        concept: str, 
        audience: Audience,
        language: Language
    ) -> List[Dict[str, Any]]:
        """Generar ejemplos para un concepto."""
        pass
    
    @abstractmethod
    async def explain_concept(
        self, 
        concept: str, 
        audience: Audience,
        language: Language
    ) -> str:
        """Explicar un concepto específico."""
        pass
    
    @abstractmethod
    async def identify_complex_concepts(
        self, 
        analysis_result: Dict[str, Any]
    ) -> List[str]:
        """Identificar conceptos complejos en el análisis."""
        pass
    
    @abstractmethod
    async def assess_difficulty_level(
        self, 
        concept: str
    ) -> str:
        """Evaluar nivel de dificultad de un concepto."""
        pass


class InteractiveExplanationService(ABC):
    """Servicio para explicaciones interactivas."""
    
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
    async def generate_expandable_sections(
        self, 
        analysis_result: Dict[str, Any],
        request: ExplanationRequest
    ) -> List[Dict[str, Any]]:
        """Generar secciones expandibles."""
        pass
    
    @abstractmethod
    async def generate_qa_elements(
        self, 
        analysis_result: Dict[str, Any],
        request: ExplanationRequest
    ) -> List[Dict[str, Any]]:
        """Generar elementos de Q&A."""
        pass
    
    @abstractmethod
    async def analyze_user_question(
        self, 
        question: str, 
        context: ExplanationContext
    ) -> Dict[str, Any]:
        """Analizar pregunta del usuario."""
        pass


class PersonalizationService(ABC):
    """Servicio para personalización."""
    
    @abstractmethod
    async def personalize_content(
        self, 
        content: str, 
        context: PersonalizationContext
    ) -> str:
        """Personalizar contenido basado en contexto."""
        pass
    
    @abstractmethod
    async def adapt_learning_style(
        self, 
        content: str, 
        learning_style: str
    ) -> str:
        """Adaptar contenido al estilo de aprendizaje."""
        pass
    
    @abstractmethod
    async def adjust_difficulty_level(
        self, 
        content: str, 
        experience_level: str
    ) -> str:
        """Ajustar nivel de dificultad del contenido."""
        pass
    
    @abstractmethod
    async def recommend_content(
        self, 
        user_id: str, 
        current_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Recomendar contenido basado en perfil del usuario."""
        pass
    
    @abstractmethod
    async def track_user_progress(
        self, 
        user_id: str, 
        interaction_data: Dict[str, Any]
    ) -> None:
        """Rastrear progreso del usuario."""
        pass


class QualityAssessmentService(ABC):
    """Servicio para evaluación de calidad."""
    
    @abstractmethod
    async def assess_explanation_quality(
        self, 
        explanation: ComprehensiveExplanation
    ) -> QualityMetrics:
        """Evaluar calidad de una explicación."""
        pass
    
    @abstractmethod
    async def assess_clarity(
        self, 
        content: str, 
        language: Language
    ) -> float:
        """Evaluar claridad del contenido."""
        pass
    
    @abstractmethod
    async def assess_completeness(
        self, 
        content: str, 
        expected_elements: List[str]
    ) -> float:
        """Evaluar completitud del contenido."""
        pass
    
    @abstractmethod
    async def assess_accuracy(
        self, 
        content: str, 
        reference_data: Dict[str, Any]
    ) -> float:
        """Evaluar precisión del contenido."""
        pass
    
    @abstractmethod
    async def assess_relevance(
        self, 
        content: str, 
        context: Dict[str, Any]
    ) -> float:
        """Evaluar relevancia del contenido."""
        pass


class PerformanceMonitoringService(ABC):
    """Servicio para monitoreo de performance."""
    
    @abstractmethod
    async def measure_generation_time(
        self, 
        operation: str
    ) -> int:
        """Medir tiempo de generación."""
        pass
    
    @abstractmethod
    async def measure_memory_usage(self) -> float:
        """Medir uso de memoria."""
        pass
    
    @abstractmethod
    async def measure_cpu_usage(self) -> float:
        """Medir uso de CPU."""
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
        """Optimizar performance de una operación."""
        pass
