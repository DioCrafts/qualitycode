"""
Casos de uso para el Motor de Explicaciones.

Este módulo implementa los casos de uso de la capa de aplicación,
orquestando las operaciones del motor de explicaciones.
"""

from typing import List, Optional, Dict, Any
import time
import logging

from ...domain.entities.explanation import (
    ComprehensiveExplanation,
    ExplanationRequest,
    InteractiveResponse,
    ExplanationContext,
    Language,
    Audience,
    ExplanationId
)
from ...domain.value_objects.explanation_metrics import (
    ContentGenerationRequest,
    GeneratedContent,
    PersonalizationContext,
    QualityMetrics,
    PerformanceMetrics
)
from ..ports.explanation_ports import (
    ExplanationEnginePort,
    ContentGeneratorPort,
    LanguageAdapterPort,
    AudienceAdapterPort,
    TemplateEnginePort,
    EducationalContentPort,
    InteractiveExplainerPort,
    MultimediaGeneratorPort,
    PersonalizationPort,
    QualityAssessmentPort,
    PerformanceMonitoringPort,
    CachePort,
    AnalyticsPort
)


logger = logging.getLogger(__name__)


class GenerateExplanationUseCase:
    """Caso de uso para generar explicaciones comprehensivas."""
    
    def __init__(
        self,
        explanation_engine: ExplanationEnginePort,
        content_generator: ContentGeneratorPort,
        language_adapter: LanguageAdapterPort,
        audience_adapter: AudienceAdapterPort,
        template_engine: TemplateEnginePort,
        educational_content: EducationalContentPort,
        interactive_explainer: InteractiveExplainerPort,
        multimedia_generator: MultimediaGeneratorPort,
        personalization: PersonalizationPort,
        quality_assessment: QualityAssessmentPort,
        performance_monitoring: PerformanceMonitoringPort,
        cache: CachePort,
        analytics: AnalyticsPort
    ):
        self.explanation_engine = explanation_engine
        self.content_generator = content_generator
        self.language_adapter = language_adapter
        self.audience_adapter = audience_adapter
        self.template_engine = template_engine
        self.educational_content = educational_content
        self.interactive_explainer = interactive_explainer
        self.multimedia_generator = multimedia_generator
        self.personalization = personalization
        self.quality_assessment = quality_assessment
        self.performance_monitoring = performance_monitoring
        self.cache = cache
        self.analytics = analytics
    
    async def execute(
        self, 
        analysis_result: Dict[str, Any],
        request: ExplanationRequest
    ) -> ComprehensiveExplanation:
        """Ejecutar generación de explicación."""
        start_time = time.time()
        
        try:
            # Verificar caché primero
            cache_key = self._generate_cache_key(analysis_result, request)
            cached_explanation = await self.cache.get_cached_explanation(cache_key)
            if cached_explanation:
                logger.info(f"Explicación encontrada en caché: {cache_key}")
                return cached_explanation
            
            # Generar explicación
            explanation = await self.explanation_engine.generate_explanation(
                analysis_result, request
            )
            
            # Evaluar calidad
            quality_metrics = await self.quality_assessment.assess_explanation_quality(
                explanation
            )
            
            # Personalizar si es necesario
            if request.personalization_context:
                explanation = await self._personalize_explanation(
                    explanation, request.personalization_context
                )
            
            # Guardar en caché
            await self.cache.cache_explanation(cache_key, explanation)
            
            # Rastrear analytics
            generation_time_ms = int((time.time() - start_time) * 1000)
            await self.analytics.track_explanation_generation(
                str(explanation.id),
                {
                    "generation_time_ms": generation_time_ms,
                    "language": request.language.value,
                    "audience": request.audience.value,
                    "quality_score": quality_metrics.overall_score,
                    "cache_hit": False
                }
            )
            
            logger.info(f"Explicación generada exitosamente en {generation_time_ms}ms")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generando explicación: {str(e)}")
            raise
    
    def _generate_cache_key(
        self, 
        analysis_result: Dict[str, Any], 
        request: ExplanationRequest
    ) -> str:
        """Generar clave de caché única."""
        # Implementar lógica de generación de clave de caché
        analysis_hash = hash(str(sorted(analysis_result.items())))
        return f"explanation_{analysis_hash}_{request.language.value}_{request.audience.value}"
    
    async def _personalize_explanation(
        self, 
        explanation: ComprehensiveExplanation,
        personalization_context: Dict[str, Any]
    ) -> ComprehensiveExplanation:
        """Personalizar explicación basada en contexto."""
        # Implementar lógica de personalización
        return explanation


class HandleInteractiveQueryUseCase:
    """Caso de uso para manejar consultas interactivas."""
    
    def __init__(
        self,
        interactive_explainer: InteractiveExplainerPort,
        performance_monitoring: PerformanceMonitoringPort,
        analytics: AnalyticsPort
    ):
        self.interactive_explainer = interactive_explainer
        self.performance_monitoring = performance_monitoring
        self.analytics = analytics
    
    async def execute(
        self, 
        question: str, 
        context: ExplanationContext
    ) -> InteractiveResponse:
        """Ejecutar manejo de consulta interactiva."""
        start_time = time.time()
        
        try:
            # Generar respuesta interactiva
            response = await self.interactive_explainer.handle_user_question(
                question, context
            )
            
            # Medir performance
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Rastrear analytics
            await self.analytics.track_user_interaction(
                str(context.explanation_id),
                {
                    "question": question,
                    "response_time_ms": response_time_ms,
                    "response_type": response.response_type.value,
                    "confidence": response.confidence
                }
            )
            
            logger.info(f"Consulta interactiva procesada en {response_time_ms}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error procesando consulta interactiva: {str(e)}")
            raise


class GenerateEducationalContentUseCase:
    """Caso de uso para generar contenido educativo."""
    
    def __init__(
        self,
        educational_content: EducationalContentPort,
        personalization: PersonalizationPort,
        analytics: AnalyticsPort
    ):
        self.educational_content = educational_content
        self.personalization = personalization
        self.analytics = analytics
    
    async def execute(
        self, 
        analysis_result: Dict[str, Any],
        request: ExplanationRequest
    ) -> List[Dict[str, Any]]:
        """Ejecutar generación de contenido educativo."""
        try:
            # Generar contenido educativo
            content = await self.educational_content.generate_educational_content(
                analysis_result, request
            )
            
            # Personalizar si es necesario
            if request.personalization_context:
                content = await self._personalize_educational_content(
                    content, request.personalization_context
                )
            
            # Rastrear analytics
            await self.analytics.track_user_interaction(
                "educational_content_generation",
                {
                    "content_items_count": len(content),
                    "language": request.language.value,
                    "audience": request.audience.value
                }
            )
            
            logger.info(f"Contenido educativo generado: {len(content)} elementos")
            return content
            
        except Exception as e:
            logger.error(f"Error generando contenido educativo: {str(e)}")
            raise
    
    async def _personalize_educational_content(
        self, 
        content: List[Dict[str, Any]],
        personalization_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Personalizar contenido educativo."""
        # Implementar lógica de personalización
        return content


class GenerateVisualizationsUseCase:
    """Caso de uso para generar visualizaciones."""
    
    def __init__(
        self,
        multimedia_generator: MultimediaGeneratorPort,
        performance_monitoring: PerformanceMonitoringPort,
        analytics: AnalyticsPort
    ):
        self.multimedia_generator = multimedia_generator
        self.performance_monitoring = performance_monitoring
        self.analytics = analytics
    
    async def execute(
        self, 
        analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Ejecutar generación de visualizaciones."""
        start_time = time.time()
        
        try:
            # Generar visualizaciones
            visualizations = await self.multimedia_generator.generate_visualizations(
                analysis_result
            )
            
            # Medir performance
            generation_time_ms = int((time.time() - start_time) * 1000)
            
            # Rastrear analytics
            await self.analytics.track_user_interaction(
                "visualization_generation",
                {
                    "visualizations_count": len(visualizations),
                    "generation_time_ms": generation_time_ms
                }
            )
            
            logger.info(f"Visualizaciones generadas: {len(visualizations)} en {generation_time_ms}ms")
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generando visualizaciones: {str(e)}")
            raise


class GetSupportedLanguagesUseCase:
    """Caso de uso para obtener idiomas soportados."""
    
    def __init__(self, explanation_engine: ExplanationEnginePort):
        self.explanation_engine = explanation_engine
    
    async def execute(self) -> List[Language]:
        """Ejecutar obtención de idiomas soportados."""
        try:
            languages = await self.explanation_engine.get_supported_languages()
            logger.info(f"Idiomas soportados obtenidos: {len(languages)}")
            return languages
        except Exception as e:
            logger.error(f"Error obteniendo idiomas soportados: {str(e)}")
            raise


class GetSupportedAudiencesUseCase:
    """Caso de uso para obtener audiencias soportadas."""
    
    def __init__(self, explanation_engine: ExplanationEnginePort):
        self.explanation_engine = explanation_engine
    
    async def execute(self) -> List[Audience]:
        """Ejecutar obtención de audiencias soportadas."""
        try:
            audiences = await self.explanation_engine.get_supported_audiences()
            logger.info(f"Audiencias soportadas obtenidas: {len(audiences)}")
            return audiences
        except Exception as e:
            logger.error(f"Error obteniendo audiencias soportadas: {str(e)}")
            raise


class AssessExplanationQualityUseCase:
    """Caso de uso para evaluar calidad de explicaciones."""
    
    def __init__(
        self,
        quality_assessment: QualityAssessmentPort,
        analytics: AnalyticsPort
    ):
        self.quality_assessment = quality_assessment
        self.analytics = analytics
    
    async def execute(
        self, 
        explanation: ComprehensiveExplanation
    ) -> QualityMetrics:
        """Ejecutar evaluación de calidad."""
        try:
            # Evaluar calidad
            quality_metrics = await self.quality_assessment.assess_explanation_quality(
                explanation
            )
            
            # Rastrear analytics
            await self.analytics.track_user_interaction(
                "quality_assessment",
                {
                    "explanation_id": str(explanation.id),
                    "overall_score": quality_metrics.overall_score,
                    "clarity_score": quality_metrics.clarity_score,
                    "completeness_score": quality_metrics.completeness_score,
                    "accuracy_score": quality_metrics.accuracy_score,
                    "relevance_score": quality_metrics.relevance_score
                }
            )
            
            logger.info(f"Calidad evaluada para explicación {explanation.id}: {quality_metrics.overall_score}")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error evaluando calidad: {str(e)}")
            raise
