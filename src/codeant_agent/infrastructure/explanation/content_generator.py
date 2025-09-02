"""
Implementación del Generador de Contenido.

Este módulo implementa la generación de contenido explicativo
adaptado a diferentes audiencias e idiomas.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from uuid import uuid4

from ...domain.entities.explanation import Language, Audience
from ...domain.value_objects.explanation_metrics import (
    ContentGenerationRequest,
    GeneratedContent,
    ContentMetadata
)
from ..ports.explanation_ports import ContentGeneratorPort


logger = logging.getLogger(__name__)


class ContentGenerator(ContentGeneratorPort):
    """Generador de contenido explicativo."""
    
    def __init__(self):
        self.technical_terms_spanish = {
            "cyclomatic_complexity": "complejidad ciclomática",
            "code_smell": "código con mal olor",
            "technical_debt": "deuda técnica",
            "refactoring": "refactorización",
            "unit_test": "prueba unitaria",
            "integration_test": "prueba de integración",
            "code_coverage": "cobertura de código",
            "static_analysis": "análisis estático",
            "dependency_injection": "inyección de dependencias",
            "design_pattern": "patrón de diseño",
            "antipattern": "antipatrón",
            "maintainability": "mantenibilidad",
            "readability": "legibilidad",
            "performance": "rendimiento",
            "security_vulnerability": "vulnerabilidad de seguridad"
        }
        
        self.technical_terms_english = {
            "complejidad_ciclomática": "cyclomatic complexity",
            "código_con_mal_olor": "code smell",
            "deuda_técnica": "technical debt",
            "refactorización": "refactoring",
            "prueba_unitaria": "unit test",
            "prueba_de_integración": "integration test",
            "cobertura_de_código": "code coverage",
            "análisis_estático": "static analysis",
            "inyección_de_dependencias": "dependency injection",
            "patrón_de_diseño": "design pattern",
            "antipatrón": "antipattern",
            "mantenibilidad": "maintainability",
            "legibilidad": "readability",
            "rendimiento": "performance",
            "vulnerabilidad_de_seguridad": "security vulnerability"
        }
    
    async def generate_content(
        self, 
        request: ContentGenerationRequest
    ) -> GeneratedContent:
        """Generar contenido basado en request."""
        try:
            # Obtener plantilla base
            template = await self._get_template(request.template_key, request.language)
            
            # Renderizar plantilla con variables
            content = await self._render_template(template, request.context_variables)
            
            # Adaptar para audiencia
            adapted_content = await self.adapt_content_for_audience(
                content, 
                Audience(request.audience), 
                Language(request.language)
            )
            
            # Validar contenido
            validated_content = await self._validate_content(adapted_content)
            
            # Crear metadatos
            metadata = await self._create_metadata(request, validated_content)
            
            return GeneratedContent(
                id=str(uuid4()),
                content=validated_content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error generando contenido: {str(e)}")
            raise
    
    async def adapt_content_for_audience(
        self, 
        content: str, 
        audience: Audience,
        language: Language
    ) -> str:
        """Adaptar contenido para una audiencia específica."""
        try:
            if audience == Audience.JUNIOR_DEVELOPER:
                return await self.adapt_for_junior_developer(content, language)
            elif audience == Audience.SENIOR_DEVELOPER:
                return await self.adapt_for_senior_developer(content, language)
            elif audience == Audience.PROJECT_MANAGER:
                return await self.adapt_for_project_manager(content, language)
            elif audience == Audience.SECURITY_TEAM:
                return await self.adapt_for_security_team(content, language)
            elif audience == Audience.BUSINESS_STAKEHOLDER:
                return await self.adapt_for_business_stakeholder(content, language)
            else:
                return content
                
        except Exception as e:
            logger.error(f"Error adaptando contenido para audiencia: {str(e)}")
            return content
    
    async def adapt_for_junior_developer(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Adaptar contenido para desarrollador junior."""
        if language == Language.SPANISH:
            # Simplificar términos técnicos
            content = await self.simplify_technical_terms(content, language)
            
            # Agregar explicaciones educativas
            content = self._add_educational_explanations_spanish(content)
            
            # Usar tono más amigable
            content = self._make_tone_friendly_spanish(content)
            
        else:
            content = await self.simplify_technical_terms(content, language)
            content = self._add_educational_explanations_english(content)
            content = self._make_tone_friendly_english(content)
        
        return content
    
    async def adapt_for_senior_developer(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Adaptar contenido para desarrollador senior."""
        # Mantener términos técnicos
        # Agregar contexto arquitectónico
        if language == Language.SPANISH:
            content = self._add_architectural_context_spanish(content)
        else:
            content = self._add_architectural_context_english(content)
        
        return content
    
    async def adapt_for_project_manager(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Adaptar contenido para project manager."""
        if language == Language.SPANISH:
            content = await self.add_business_context(content, language)
            content = self._add_cost_implications_spanish(content)
            content = self._add_timeline_impact_spanish(content)
        else:
            content = await self.add_business_context(content, language)
            content = self._add_cost_implications_english(content)
            content = self._add_timeline_impact_english(content)
        
        return content
    
    async def adapt_for_security_team(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Adaptar contenido para equipo de seguridad."""
        if language == Language.SPANISH:
            content = await self.emphasize_security_aspects(content, language)
            content = self._add_security_implications_spanish(content)
            content = self._add_compliance_context_spanish(content)
        else:
            content = await self.emphasize_security_aspects(content, language)
            content = self._add_security_implications_english(content)
            content = self._add_compliance_context_english(content)
        
        return content
    
    async def adapt_for_business_stakeholder(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Adaptar contenido para stakeholder de negocio."""
        if language == Language.SPANISH:
            content = self._remove_technical_jargon_spanish(content)
            content = self._add_business_value_spanish(content)
            content = self._add_risk_assessment_spanish(content)
        else:
            content = self._remove_technical_jargon_english(content)
            content = self._add_business_value_english(content)
            content = self._add_risk_assessment_english(content)
        
        return content
    
    async def simplify_technical_terms(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Simplificar términos técnicos en el contenido."""
        if language == Language.SPANISH:
            terms_dict = self.technical_terms_spanish
        else:
            terms_dict = self.technical_terms_english
        
        simplified_content = content
        for technical_term, simple_term in terms_dict.items():
            # Reemplazar términos técnicos con explicaciones simples
            pattern = rf'\b{re.escape(technical_term)}\b'
            if language == Language.SPANISH:
                replacement = f"{simple_term} (término técnico que mide la complejidad del código)"
            else:
                replacement = f"{simple_term} (technical term that measures code complexity)"
            
            simplified_content = re.sub(pattern, replacement, simplified_content, flags=re.IGNORECASE)
        
        return simplified_content
    
    async def add_business_context(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Agregar contexto de negocio al contenido."""
        if language == Language.SPANISH:
            business_context = (
                "\n\n**Contexto de Negocio:**\n"
                "Este problema puede impactar la productividad del equipo y aumentar los costos de desarrollo. "
                "Se recomienda abordarlo en el próximo sprint para mantener la calidad del producto."
            )
        else:
            business_context = (
                "\n\n**Business Context:**\n"
                "This issue can impact team productivity and increase development costs. "
                "It's recommended to address it in the next sprint to maintain product quality."
            )
        
        return content + business_context
    
    async def emphasize_security_aspects(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Enfatizar aspectos de seguridad en el contenido."""
        if language == Language.SPANISH:
            security_context = (
                "\n\n**Aspectos de Seguridad:**\n"
                "Este problema puede crear vulnerabilidades de seguridad. "
                "Se recomienda una revisión inmediata por el equipo de seguridad."
            )
        else:
            security_context = (
                "\n\n**Security Aspects:**\n"
                "This issue can create security vulnerabilities. "
                "Immediate review by the security team is recommended."
            )
        
        return content + security_context
    
    async def _get_template(self, template_key: str, language: str) -> str:
        """Obtener plantilla por clave e idioma."""
        # Implementar lógica de obtención de plantillas
        templates = {
            "analysis_summary_es": "El análisis de código ha identificado {total_issues} problemas...",
            "analysis_summary_en": "Code analysis has identified {total_issues} issues...",
            "violation_explanation_es": "**Problema:** {message}\n**Ubicación:** {location}\n**Explicación:** {explanation}",
            "violation_explanation_en": "**Issue:** {message}\n**Location:** {location}\n**Explanation:** {explanation}"
        }
        
        key = f"{template_key}_{language}"
        return templates.get(key, "Template not found")
    
    async def _render_template(
        self, 
        template: str, 
        variables: Dict[str, str]
    ) -> str:
        """Renderizar plantilla con variables."""
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"Variable faltante en plantilla: {e}")
            return template
    
    async def _validate_content(self, content: str) -> str:
        """Validar contenido generado."""
        if not content or not content.strip():
            raise ValueError("El contenido no puede estar vacío")
        
        if len(content) > 10000:
            logger.warning("Contenido muy largo, truncando...")
            content = content[:10000] + "..."
        
        return content
    
    async def _create_metadata(
        self, 
        request: ContentGenerationRequest, 
        content: str
    ) -> Dict[str, Any]:
        """Crear metadatos del contenido."""
        word_count = len(content.split())
        reading_time_minutes = word_count / 200  # Asumiendo 200 palabras por minuto
        
        return {
            "language": request.language,
            "audience": request.audience,
            "generation_time": str(uuid4()),  # Placeholder
            "template_used": request.template_key,
            "context_used": request.context_variables,
            "word_count": word_count,
            "reading_time_minutes": reading_time_minutes
        }
    
    def _add_educational_explanations_spanish(self, content: str) -> str:
        """Agregar explicaciones educativas en español."""
        educational_additions = {
            "complejidad ciclomática": " (mide cuántas rutas diferentes puede tomar el código)",
            "refactorización": " (mejorar el código sin cambiar lo que hace)",
            "prueba unitaria": " (prueba que verifica una pequeña parte del código)"
        }
        
        for term, explanation in educational_additions.items():
            content = content.replace(term, term + explanation)
        
        return content
    
    def _add_educational_explanations_english(self, content: str) -> str:
        """Agregar explicaciones educativas en inglés."""
        educational_additions = {
            "cyclomatic complexity": " (measures how many different paths the code can take)",
            "refactoring": " (improving code without changing what it does)",
            "unit test": " (test that verifies a small part of the code)"
        }
        
        for term, explanation in educational_additions.items():
            content = content.replace(term, term + explanation)
        
        return content
    
    def _make_tone_friendly_spanish(self, content: str) -> str:
        """Hacer el tono más amigable en español."""
        friendly_replacements = {
            "El problema es": "¡Hola! El problema que encontramos es",
            "Se debe": "Te recomendamos que",
            "Es necesario": "Sería genial si pudieras"
        }
        
        for formal, friendly in friendly_replacements.items():
            content = content.replace(formal, friendly)
        
        return content
    
    def _make_tone_friendly_english(self, content: str) -> str:
        """Hacer el tono más amigable en inglés."""
        friendly_replacements = {
            "The issue is": "Hello! The issue we found is",
            "It should": "We recommend that you",
            "It is necessary": "It would be great if you could"
        }
        
        for formal, friendly in friendly_replacements.items():
            content = content.replace(formal, friendly)
        
        return content
    
    def _add_architectural_context_spanish(self, content: str) -> str:
        """Agregar contexto arquitectónico en español."""
        architectural_context = (
            "\n\n**Contexto Arquitectónico:**\n"
            "Considera el impacto en la arquitectura general del sistema y "
            "las implicaciones para otros componentes."
        )
        return content + architectural_context
    
    def _add_architectural_context_english(self, content: str) -> str:
        """Agregar contexto arquitectónico en inglés."""
        architectural_context = (
            "\n\n**Architectural Context:**\n"
            "Consider the impact on the overall system architecture and "
            "implications for other components."
        )
        return content + architectural_context
    
    def _add_cost_implications_spanish(self, content: str) -> str:
        """Agregar implicaciones de costo en español."""
        cost_context = (
            "\n\n**Implicaciones de Costo:**\n"
            "Tiempo estimado de corrección: 2-4 horas\n"
            "Impacto en el cronograma: Mínimo\n"
            "ROI de la corrección: Alto"
        )
        return content + cost_context
    
    def _add_cost_implications_english(self, content: str) -> str:
        """Agregar implicaciones de costo en inglés."""
        cost_context = (
            "\n\n**Cost Implications:**\n"
            "Estimated correction time: 2-4 hours\n"
            "Schedule impact: Minimal\n"
            "Correction ROI: High"
        )
        return content + cost_context
    
    def _add_timeline_impact_spanish(self, content: str) -> str:
        """Agregar impacto en cronograma en español."""
        timeline_context = (
            "\n\n**Impacto en Cronograma:**\n"
            "Puede ser abordado en el próximo sprint sin afectar las entregas planificadas."
        )
        return content + timeline_context
    
    def _add_timeline_impact_english(self, content: str) -> str:
        """Agregar impacto en cronograma en inglés."""
        timeline_context = (
            "\n\n**Timeline Impact:**\n"
            "Can be addressed in the next sprint without affecting planned deliveries."
        )
        return content + timeline_context
    
    def _add_security_implications_spanish(self, content: str) -> str:
        """Agregar implicaciones de seguridad en español."""
        security_context = (
            "\n\n**Implicaciones de Seguridad:**\n"
            "Nivel de riesgo: Medio\n"
            "Vectores de ataque potenciales: 2\n"
            "Recomendación: Revisión inmediata"
        )
        return content + security_context
    
    def _add_security_implications_english(self, content: str) -> str:
        """Agregar implicaciones de seguridad en inglés."""
        security_context = (
            "\n\n**Security Implications:**\n"
            "Risk level: Medium\n"
            "Potential attack vectors: 2\n"
            "Recommendation: Immediate review"
        )
        return content + security_context
    
    def _add_compliance_context_spanish(self, content: str) -> str:
        """Agregar contexto de cumplimiento en español."""
        compliance_context = (
            "\n\n**Contexto de Cumplimiento:**\n"
            "Verificar impacto en estándares de seguridad aplicables."
        )
        return content + compliance_context
    
    def _add_compliance_context_english(self, content: str) -> str:
        """Agregar contexto de cumplimiento en inglés."""
        compliance_context = (
            "\n\n**Compliance Context:**\n"
            "Verify impact on applicable security standards."
        )
        return content + compliance_context
    
    def _remove_technical_jargon_spanish(self, content: str) -> str:
        """Remover jerga técnica en español."""
        jargon_replacements = {
            "complejidad ciclomática": "complejidad del código",
            "refactorización": "mejora del código",
            "prueba unitaria": "prueba de código",
            "análisis estático": "revisión automática"
        }
        
        for jargon, simple in jargon_replacements.items():
            content = content.replace(jargon, simple)
        
        return content
    
    def _remove_technical_jargon_english(self, content: str) -> str:
        """Remover jerga técnica en inglés."""
        jargon_replacements = {
            "cyclomatic complexity": "code complexity",
            "refactoring": "code improvement",
            "unit test": "code test",
            "static analysis": "automatic review"
        }
        
        for jargon, simple in jargon_replacements.items():
            content = content.replace(jargon, simple)
        
        return content
    
    def _add_business_value_spanish(self, content: str) -> str:
        """Agregar valor de negocio en español."""
        business_value = (
            "\n\n**Valor de Negocio:**\n"
            "Mejorar la calidad del código reduce los costos de mantenimiento "
            "y aumenta la satisfacción del cliente."
        )
        return content + business_value
    
    def _add_business_value_english(self, content: str) -> str:
        """Agregar valor de negocio en inglés."""
        business_value = (
            "\n\n**Business Value:**\n"
            "Improving code quality reduces maintenance costs "
            "and increases customer satisfaction."
        )
        return content + business_value
    
    def _add_risk_assessment_spanish(self, content: str) -> str:
        """Agregar evaluación de riesgo en español."""
        risk_assessment = (
            "\n\n**Evaluación de Riesgo:**\n"
            "Riesgo actual: Bajo\n"
            "Riesgo si no se corrige: Medio\n"
            "Recomendación: Abordar en el próximo trimestre"
        )
        return content + risk_assessment
    
    def _add_risk_assessment_english(self, content: str) -> str:
        """Agregar evaluación de riesgo en inglés."""
        risk_assessment = (
            "\n\n**Risk Assessment:**\n"
            "Current risk: Low\n"
            "Risk if not corrected: Medium\n"
            "Recommendation: Address in next quarter"
        )
        return content + risk_assessment
