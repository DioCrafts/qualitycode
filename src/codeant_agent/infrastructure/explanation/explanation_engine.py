"""
Implementación del Motor de Explicaciones en Lenguaje Natural.

Este módulo implementa el motor principal de explicaciones,
integrando todos los componentes del sistema.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4

from ...domain.entities.explanation import (
    ComprehensiveExplanation,
    ExplanationRequest,
    InteractiveResponse,
    ExplanationContext,
    Language,
    Audience,
    ExplanationId,
    ExplanationSection,
    SectionType,
    SectionImportance,
    InteractiveElement,
    InteractiveElementType,
    InteractiveState,
    InteractiveTrigger,
    Visualization,
    EducationalContent,
    ActionItem,
    Reference
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


class NaturalLanguageExplanationEngine(ExplanationEnginePort):
    """Motor principal de explicaciones en lenguaje natural."""
    
    def __init__(
        self,
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
    
    async def generate_explanation(
        self, 
        analysis_result: Dict[str, Any],
        request: ExplanationRequest
    ) -> ComprehensiveExplanation:
        """Generar explicación comprehensiva."""
        start_time = time.time()
        
        try:
            # Generar resumen
            summary = await self._generate_analysis_summary(analysis_result, request)
            
            # Generar secciones detalladas
            detailed_sections = await self._generate_detailed_sections(analysis_result, request)
            
            # Generar visualizaciones si está habilitado
            visualizations = []
            if request.include_visualizations:
                visualizations = await self.multimedia_generator.generate_visualizations(analysis_result)
            
            # Generar elementos interactivos si está habilitado
            interactive_elements = []
            if request.include_examples:
                interactive_elements = await self.interactive_explainer.generate_interactive_elements(
                    analysis_result, request
                )
            
            # Generar contenido educativo
            educational_content = []
            if request.include_educational_content:
                educational_content = await self.educational_content.generate_educational_content(
                    analysis_result, request
                )
            
            # Generar items de acción
            action_items = await self._generate_action_items(analysis_result, request)
            
            # Generar glosario
            glossary = await self._generate_glossary(analysis_result, request.language)
            
            # Generar referencias
            references = await self._generate_references(analysis_result)
            
            # Calcular tiempo de generación
            generation_time_ms = int((time.time() - start_time) * 1000)
            
            # Crear explicación comprehensiva
            explanation = ComprehensiveExplanation(
                id=ExplanationId(),
                language=request.language,
                audience=request.audience,
                summary=summary,
                detailed_sections=detailed_sections,
                visualizations=visualizations,
                interactive_elements=interactive_elements,
                educational_content=educational_content,
                action_items=action_items,
                glossary=glossary,
                references=references,
                generation_time_ms=generation_time_ms
            )
            
            logger.info(f"Explicación generada exitosamente en {generation_time_ms}ms")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generando explicación: {str(e)}")
            raise
    
    async def generate_interactive_response(
        self, 
        question: str, 
        context: ExplanationContext
    ) -> InteractiveResponse:
        """Generar respuesta interactiva."""
        try:
            response = await self.interactive_explainer.handle_user_question(question, context)
            logger.info(f"Respuesta interactiva generada para pregunta: {question[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error generando respuesta interactiva: {str(e)}")
            raise
    
    async def get_supported_languages(self) -> List[Language]:
        """Obtener idiomas soportados."""
        return list(Language)
    
    async def get_supported_audiences(self) -> List[Audience]:
        """Obtener audiencias soportadas."""
        return list(Audience)
    
    async def _generate_analysis_summary(
        self, 
        analysis_result: Dict[str, Any], 
        request: ExplanationRequest
    ) -> str:
        """Generar resumen del análisis."""
        try:
            # Obtener métricas básicas
            total_issues = len(analysis_result.get('violations', []))
            critical_issues = len([v for v in analysis_result.get('violations', []) 
                                 if v.get('severity') == 'critical'])
            high_issues = len([v for v in analysis_result.get('violations', []) 
                             if v.get('severity') == 'high'])
            quality_score = analysis_result.get('metrics', {}).get('overall_quality_score', 0.0)
            
            # Generar resumen basado en idioma y audiencia
            if request.language == Language.SPANISH:
                summary = self._generate_spanish_summary(
                    total_issues, critical_issues, high_issues, quality_score, request.audience
                )
            else:
                summary = self._generate_english_summary(
                    total_issues, critical_issues, high_issues, quality_score, request.audience
                )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generando resumen: {str(e)}")
            return "Error generando resumen del análisis."
    
    def _generate_spanish_summary(
        self, 
        total_issues: int, 
        critical_issues: int, 
        high_issues: int, 
        quality_score: float,
        audience: Audience
    ) -> str:
        """Generar resumen en español."""
        if audience == Audience.PROJECT_MANAGER:
            return (
                f"El análisis de código ha identificado {total_issues} problemas, "
                f"incluyendo {critical_issues} críticos y {high_issues} de alta prioridad. "
                f"La puntuación de calidad general es {quality_score:.1f}/100. "
                f"Se recomienda abordar los problemas críticos para reducir riesgos de negocio."
            )
        elif audience == Audience.JUNIOR_DEVELOPER:
            return (
                f"¡Hola! El análisis encontró {total_issues} áreas de mejora en tu código. "
                f"Hay {critical_issues} problemas críticos y {high_issues} de alta prioridad. "
                f"Tu puntuación de calidad es {quality_score:.1f}/100. "
                f"¡No te preocupes! Te ayudaremos a entender y solucionar cada problema paso a paso."
            )
        else:
            return (
                f"El análisis detectó {total_issues} violaciones de reglas de calidad, "
                f"con {critical_issues} críticas y {high_issues} de alta severidad. "
                f"La puntuación de calidad general es {quality_score:.1f}/100. "
                f"Se proporcionan recomendaciones detalladas para cada problema identificado."
            )
    
    def _generate_english_summary(
        self, 
        total_issues: int, 
        critical_issues: int, 
        high_issues: int, 
        quality_score: float,
        audience: Audience
    ) -> str:
        """Generar resumen en inglés."""
        if audience == Audience.PROJECT_MANAGER:
            return (
                f"Code analysis identified {total_issues} issues, "
                f"including {critical_issues} critical and {high_issues} high priority. "
                f"Overall quality score is {quality_score:.1f}/100. "
                f"Addressing critical issues is recommended to reduce business risks."
            )
        elif audience == Audience.JUNIOR_DEVELOPER:
            return (
                f"Hello! The analysis found {total_issues} areas for improvement in your code. "
                f"There are {critical_issues} critical issues and {high_issues} high priority ones. "
                f"Your quality score is {quality_score:.1f}/100. "
                f"Don't worry! We'll help you understand and fix each problem step by step."
            )
        else:
            return (
                f"Analysis detected {total_issues} quality rule violations, "
                f"with {critical_issues} critical and {high_issues} high severity. "
                f"Overall quality score is {quality_score:.1f}/100. "
                f"Detailed recommendations are provided for each identified issue."
            )
    
    async def _generate_detailed_sections(
        self, 
        analysis_result: Dict[str, Any], 
        request: ExplanationRequest
    ) -> List[ExplanationSection]:
        """Generar secciones detalladas."""
        sections = []
        
        # Sección de problemas
        violations = analysis_result.get('violations', [])
        if violations:
            issues_section = await self._generate_issues_section(violations, request)
            sections.append(issues_section)
        
        # Sección de métricas
        metrics_section = await self._generate_metrics_section(analysis_result, request)
        sections.append(metrics_section)
        
        # Sección de antipatrones
        antipatterns = analysis_result.get('antipattern_analysis')
        if antipatterns:
            antipatterns_section = await self._generate_antipatterns_section(antipatterns, request)
            sections.append(antipatterns_section)
        
        # Sección de recomendaciones
        recommendations_section = await self._generate_recommendations_section(analysis_result, request)
        sections.append(recommendations_section)
        
        return sections
    
    async def _generate_issues_section(
        self, 
        violations: List[Dict[str, Any]], 
        request: ExplanationRequest
    ) -> ExplanationSection:
        """Generar sección de problemas."""
        content = ""
        
        # Agrupar por severidad
        severity_groups = {}
        for violation in violations:
            severity = violation.get('severity', 'unknown')
            if severity not in severity_groups:
                severity_groups[severity] = []
            severity_groups[severity].append(violation)
        
        # Generar contenido para cada severidad
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in severity_groups:
                severity_content = await self._generate_severity_content(
                    severity, severity_groups[severity], request
                )
                content += severity_content
        
        title = "Problemas Detectados" if request.language == Language.SPANISH else "Detected Issues"
        
        return ExplanationSection(
            id=ExplanationId(),
            title=title,
            content=content,
            section_type=SectionType.ISSUES,
            importance=SectionImportance.HIGH
        )
    
    async def _generate_severity_content(
        self, 
        severity: str, 
        violations: List[Dict[str, Any]], 
        request: ExplanationRequest
    ) -> str:
        """Generar contenido para una severidad específica."""
        severity_names = {
            'critical': ('Críticos', 'Critical'),
            'high': ('Altos', 'High'),
            'medium': ('Medios', 'Medium'),
            'low': ('Bajos', 'Low')
        }
        
        severity_name = severity_names.get(severity, ('Desconocidos', 'Unknown'))
        severity_title = severity_name[0] if request.language == Language.SPANISH else severity_name[1]
        
        content = f"\n## Problemas {severity_title} ({len(violations)})\n\n" if request.language == Language.SPANISH else f"\n## {severity_title} Severity Issues ({len(violations)})\n\n"
        
        for i, violation in enumerate(violations, 1):
            violation_content = await self._generate_violation_content(violation, request)
            content += f"{i}. {violation_content}\n\n"
        
        return content
    
    async def _generate_violation_content(
        self, 
        violation: Dict[str, Any], 
        request: ExplanationRequest
    ) -> str:
        """Generar contenido para una violación específica."""
        message = violation.get('message', 'Problema no especificado')
        location = violation.get('location', {})
        start_line = location.get('start_line', 0)
        end_line = location.get('end_line', 0)
        
        if request.language == Language.SPANISH:
            location_text = f"**{message}** (Línea {start_line}-{end_line})"
        else:
            location_text = f"**{message}** (Line {start_line}-{end_line})"
        
        # Generar explicación detallada basada en audiencia
        detailed_explanation = await self._generate_detailed_violation_explanation(violation, request)
        
        return f"{location_text}\n\n{detailed_explanation}"
    
    async def _generate_detailed_violation_explanation(
        self, 
        violation: Dict[str, Any], 
        request: ExplanationRequest
    ) -> str:
        """Generar explicación detallada de violación."""
        rule_category = violation.get('rule_category', 'unknown')
        
        if request.audience == Audience.JUNIOR_DEVELOPER:
            return await self._generate_educational_explanation(violation, request)
        elif request.audience == Audience.PROJECT_MANAGER:
            return await self._generate_business_explanation(violation, request)
        else:
            return await self._generate_technical_explanation(violation, request)
    
    async def _generate_educational_explanation(
        self, 
        violation: Dict[str, Any], 
        request: ExplanationRequest
    ) -> str:
        """Generar explicación educativa."""
        if request.language == Language.SPANISH:
            return (
                "**¿Qué es este problema?**\n"
                "Este es un problema de calidad de código que puede afectar la mantenibilidad.\n\n"
                "**¿Por qué es problemático?**\n"
                "Puede hacer que el código sea más difícil de entender y mantener.\n\n"
                "**¿Cómo solucionarlo?**\n"
                "Sigue las mejores prácticas de programación y refactoriza el código."
            )
        else:
            return (
                "**What is this issue?**\n"
                "This is a code quality issue that can affect maintainability.\n\n"
                "**Why is it problematic?**\n"
                "It can make the code harder to understand and maintain.\n\n"
                "**How to fix it?**\n"
                "Follow programming best practices and refactor the code."
            )
    
    async def _generate_business_explanation(
        self, 
        violation: Dict[str, Any], 
        request: ExplanationRequest
    ) -> str:
        """Generar explicación orientada al negocio."""
        if request.language == Language.SPANISH:
            return (
                "**Impacto en el Negocio:**\n"
                "Este problema puede aumentar el tiempo de desarrollo y el riesgo de bugs.\n\n"
                "**Implicaciones de Costo:**\n"
                "Se estima un aumento del 20% en el tiempo de mantenimiento.\n\n"
                "**Acción Recomendada:**\n"
                "Priorizar la corrección en el próximo sprint para reducir costos futuros."
            )
        else:
            return (
                "**Business Impact:**\n"
                "This issue can increase development time and bug risk.\n\n"
                "**Cost Implications:**\n"
                "Estimated 20% increase in maintenance time.\n\n"
                "**Recommended Action:**\n"
                "Prioritize correction in the next sprint to reduce future costs."
            )
    
    async def _generate_technical_explanation(
        self, 
        violation: Dict[str, Any], 
        request: ExplanationRequest
    ) -> str:
        """Generar explicación técnica."""
        if request.language == Language.SPANISH:
            return (
                "**Descripción Técnica:**\n"
                "Violación de regla de calidad de código identificada por el analizador estático.\n\n"
                "**Solución Recomendada:**\n"
                "Refactorizar el código siguiendo las mejores prácticas establecidas.\n\n"
                "**Referencias:**\n"
                "Consultar la documentación de la regla para más detalles."
            )
        else:
            return (
                "**Technical Description:**\n"
                "Code quality rule violation identified by static analyzer.\n\n"
                "**Recommended Solution:**\n"
                "Refactor code following established best practices.\n\n"
                "**References:**\n"
                "Consult rule documentation for more details."
            )
    
    async def _generate_metrics_section(
        self, 
        analysis_result: Dict[str, Any], 
        request: ExplanationRequest
    ) -> ExplanationSection:
        """Generar sección de métricas."""
        metrics = analysis_result.get('metrics', {})
        
        if request.language == Language.SPANISH:
            title = "Métricas de Calidad"
            content = f"**Puntuación General:** {metrics.get('overall_quality_score', 0):.1f}/100\n\n"
            content += f"**Complejidad Ciclomática:** {metrics.get('complexity_metrics', {}).get('cyclomatic_complexity', 0)}\n\n"
            content += f"**Líneas de Código:** {metrics.get('code_metrics', {}).get('lines_of_code', 0)}\n\n"
        else:
            title = "Quality Metrics"
            content = f"**Overall Score:** {metrics.get('overall_quality_score', 0):.1f}/100\n\n"
            content += f"**Cyclomatic Complexity:** {metrics.get('complexity_metrics', {}).get('cyclomatic_complexity', 0)}\n\n"
            content += f"**Lines of Code:** {metrics.get('code_metrics', {}).get('lines_of_code', 0)}\n\n"
        
        return ExplanationSection(
            id=ExplanationId(),
            title=title,
            content=content,
            section_type=SectionType.METRICS,
            importance=SectionImportance.MEDIUM
        )
    
    async def _generate_antipatterns_section(
        self, 
        antipatterns: Dict[str, Any], 
        request: ExplanationRequest
    ) -> ExplanationSection:
        """Generar sección de antipatrones."""
        if request.language == Language.SPANISH:
            title = "Antipatrones Detectados"
            content = "Se han identificado varios antipatrones en el código que pueden afectar la calidad y mantenibilidad."
        else:
            title = "Detected Antipatterns"
            content = "Several antipatterns have been identified in the code that may affect quality and maintainability."
        
        return ExplanationSection(
            id=ExplanationId(),
            title=title,
            content=content,
            section_type=SectionType.ANTIPATTERNS,
            importance=SectionImportance.MEDIUM
        )
    
    async def _generate_recommendations_section(
        self, 
        analysis_result: Dict[str, Any], 
        request: ExplanationRequest
    ) -> ExplanationSection:
        """Generar sección de recomendaciones."""
        if request.language == Language.SPANISH:
            title = "Recomendaciones"
            content = "Basado en el análisis, se recomienda:\n\n"
            content += "1. Abordar los problemas críticos primero\n"
            content += "2. Refactorizar código complejo\n"
            content += "3. Implementar mejores prácticas de testing\n"
            content += "4. Revisar la arquitectura del código"
        else:
            title = "Recommendations"
            content = "Based on the analysis, it is recommended to:\n\n"
            content += "1. Address critical issues first\n"
            content += "2. Refactor complex code\n"
            content += "3. Implement better testing practices\n"
            content += "4. Review code architecture"
        
        return ExplanationSection(
            id=ExplanationId(),
            title=title,
            content=content,
            section_type=SectionType.RECOMMENDATIONS,
            importance=SectionImportance.HIGH
        )
    
    async def _generate_action_items(
        self, 
        analysis_result: Dict[str, Any], 
        request: ExplanationRequest
    ) -> List[ActionItem]:
        """Generar items de acción."""
        action_items = []
        
        violations = analysis_result.get('violations', [])
        critical_violations = [v for v in violations if v.get('severity') == 'critical']
        
        if critical_violations:
            if request.language == Language.SPANISH:
                action_items.append(ActionItem(
                    id=str(uuid4()),
                    title="Corregir Problemas Críticos",
                    description=f"Abordar {len(critical_violations)} problemas críticos identificados",
                    priority="Alta",
                    estimated_effort="2-4 horas"
                ))
            else:
                action_items.append(ActionItem(
                    id=str(uuid4()),
                    title="Fix Critical Issues",
                    description=f"Address {len(critical_violations)} critical issues identified",
                    priority="High",
                    estimated_effort="2-4 hours"
                ))
        
        return action_items
    
    async def _generate_glossary(
        self, 
        analysis_result: Dict[str, Any], 
        language: Language
    ) -> Dict[str, str]:
        """Generar glosario de términos."""
        if language == Language.SPANISH:
            return {
                "Complejidad Ciclomática": "Métrica que mide la complejidad de un programa",
                "Antipatrón": "Patrón de diseño que es contraproducente",
                "Refactoring": "Proceso de mejorar el código sin cambiar su funcionalidad"
            }
        else:
            return {
                "Cyclomatic Complexity": "Metric that measures program complexity",
                "Antipattern": "Design pattern that is counterproductive",
                "Refactoring": "Process of improving code without changing functionality"
            }
    
    async def _generate_references(self, analysis_result: Dict[str, Any]) -> List[Reference]:
        """Generar referencias."""
        return [
            Reference(
                id=str(uuid4()),
                title="Clean Code Principles",
                url="https://example.com/clean-code",
                description="Best practices for writing clean code",
                reference_type="book"
            )
        ]
