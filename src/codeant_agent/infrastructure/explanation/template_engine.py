"""
Implementación del Motor de Plantillas.

Este módulo implementa el sistema de plantillas para generar contenido
explicativo de manera consistente y reutilizable.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from uuid import uuid4

from ...domain.entities.explanation import Language
from ...domain.value_objects.explanation_metrics import TemplateValidationResult
from ..ports.explanation_ports import TemplateEnginePort


logger = logging.getLogger(__name__)


class ExplanationTemplateEngine(TemplateEnginePort):
    """Motor de plantillas para explicaciones."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.template_variables = self._initialize_template_variables()
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Inicializar plantillas base."""
        return {
            "analysis_summary": {
                "es": (
                    "## Resumen del Análisis de Código\n\n"
                    "El análisis de código ha identificado **{total_issues}** problemas en total, "
                    "incluyendo **{critical_issues}** críticos y **{high_issues}** de alta prioridad. "
                    "La puntuación de calidad general es **{quality_score}/100**.\n\n"
                    "{quality_rating_text}\n\n"
                    "{audience_specific_text}"
                ),
                "en": (
                    "## Code Analysis Summary\n\n"
                    "Code analysis has identified **{total_issues}** total issues, "
                    "including **{critical_issues}** critical and **{high_issues}** high priority. "
                    "Overall quality score is **{quality_score}/100**.\n\n"
                    "{quality_rating_text}\n\n"
                    "{audience_specific_text}"
                )
            },
            "violation_explanation": {
                "es": (
                    "### {violation_title}\n\n"
                    "**Ubicación:** {location}\n"
                    "**Severidad:** {severity}\n"
                    "**Categoría:** {category}\n\n"
                    "**Descripción:**\n{violation_description}\n\n"
                    "**¿Por qué es problemático?**\n{why_problematic}\n\n"
                    "**¿Cómo solucionarlo?**\n{how_to_fix}\n\n"
                    "{audience_specific_section}"
                ),
                "en": (
                    "### {violation_title}\n\n"
                    "**Location:** {location}\n"
                    "**Severity:** {severity}\n"
                    "**Category:** {category}\n\n"
                    "**Description:**\n{violation_description}\n\n"
                    "**Why is it problematic?**\n{why_problematic}\n\n"
                    "**How to fix it?**\n{how_to_fix}\n\n"
                    "{audience_specific_section}"
                )
            },
            "metrics_section": {
                "es": (
                    "## Métricas de Calidad\n\n"
                    "### Puntuación General\n"
                    "**{quality_score}/100** - {quality_rating}\n\n"
                    "### Métricas Detalladas\n"
                    "• **Complejidad Ciclomática:** {cyclomatic_complexity}\n"
                    "• **Líneas de Código:** {lines_of_code}\n"
                    "• **Funciones:** {function_count}\n"
                    "• **Clases:** {class_count}\n"
                    "• **Cobertura de Pruebas:** {test_coverage}%\n\n"
                    "{metrics_interpretation}"
                ),
                "en": (
                    "## Quality Metrics\n\n"
                    "### Overall Score\n"
                    "**{quality_score}/100** - {quality_rating}\n\n"
                    "### Detailed Metrics\n"
                    "• **Cyclomatic Complexity:** {cyclomatic_complexity}\n"
                    "• **Lines of Code:** {lines_of_code}\n"
                    "• **Functions:** {function_count}\n"
                    "• **Classes:** {class_count}\n"
                    "• **Test Coverage:** {test_coverage}%\n\n"
                    "{metrics_interpretation}"
                )
            },
            "recommendations_section": {
                "es": (
                    "## Recomendaciones\n\n"
                    "Basado en el análisis, se recomienda abordar los siguientes aspectos:\n\n"
                    "### Prioridad Alta\n"
                    "{high_priority_recommendations}\n\n"
                    "### Prioridad Media\n"
                    "{medium_priority_recommendations}\n\n"
                    "### Prioridad Baja\n"
                    "{low_priority_recommendations}\n\n"
                    "{implementation_guidance}"
                ),
                "en": (
                    "## Recommendations\n\n"
                    "Based on the analysis, it is recommended to address the following aspects:\n\n"
                    "### High Priority\n"
                    "{high_priority_recommendations}\n\n"
                    "### Medium Priority\n"
                    "{medium_priority_recommendations}\n\n"
                    "### Low Priority\n"
                    "{low_priority_recommendations}\n\n"
                    "{implementation_guidance}"
                )
            },
            "educational_content": {
                "es": (
                    "## Contenido Educativo\n\n"
                    "### ¿Qué es {concept_name}?\n"
                    "{concept_definition}\n\n"
                    "### ¿Por qué es importante?\n"
                    "{importance_explanation}\n\n"
                    "### Ejemplos Prácticos\n"
                    "{practical_examples}\n\n"
                    "### Mejores Prácticas\n"
                    "{best_practices}\n\n"
                    "### Recursos de Aprendizaje\n"
                    "{learning_resources}"
                ),
                "en": (
                    "## Educational Content\n\n"
                    "### What is {concept_name}?\n"
                    "{concept_definition}\n\n"
                    "### Why is it important?\n"
                    "{importance_explanation}\n\n"
                    "### Practical Examples\n"
                    "{practical_examples}\n\n"
                    "### Best Practices\n"
                    "{best_practices}\n\n"
                    "### Learning Resources\n"
                    "{learning_resources}"
                )
            },
            "interactive_qa": {
                "es": (
                    "## Preguntas y Respuestas\n\n"
                    "### {question}\n"
                    "{answer}\n\n"
                    "**Confianza:** {confidence}%\n"
                    "**Preguntas relacionadas:**\n{related_questions}\n\n"
                    "{follow_up_suggestions}"
                ),
                "en": (
                    "## Questions and Answers\n\n"
                    "### {question}\n"
                    "{answer}\n\n"
                    "**Confidence:** {confidence}%\n"
                    "**Related questions:**\n{related_questions}\n\n"
                    "{follow_up_suggestions}"
                )
            },
            "business_impact": {
                "es": (
                    "## Impacto en el Negocio\n\n"
                    "### Análisis de Costos\n"
                    "• **Tiempo estimado de corrección:** {estimated_time}\n"
                    "• **Costo estimado:** {estimated_cost}\n"
                    "• **ROI de la corrección:** {correction_roi}\n\n"
                    "### Impacto en el Cronograma\n"
                    "{timeline_impact}\n\n"
                    "### Riesgos de Negocio\n"
                    "{business_risks}\n\n"
                    "### Recomendaciones Estratégicas\n"
                    "{strategic_recommendations}"
                ),
                "en": (
                    "## Business Impact\n\n"
                    "### Cost Analysis\n"
                    "• **Estimated correction time:** {estimated_time}\n"
                    "• **Estimated cost:** {estimated_cost}\n"
                    "• **Correction ROI:** {correction_roi}\n\n"
                    "### Timeline Impact\n"
                    "{timeline_impact}\n\n"
                    "### Business Risks\n"
                    "{business_risks}\n\n"
                    "### Strategic Recommendations\n"
                    "{strategic_recommendations}"
                )
            },
            "security_analysis": {
                "es": (
                    "## Análisis de Seguridad\n\n"
                    "### Nivel de Riesgo\n"
                    "**{risk_level}** - {risk_description}\n\n"
                    "### Vectores de Ataque Identificados\n"
                    "{attack_vectors}\n\n"
                    "### Impacto Potencial\n"
                    "{potential_impact}\n\n"
                    "### Medidas de Mitigación\n"
                    "{mitigation_measures}\n\n"
                    "### Cumplimiento\n"
                    "{compliance_considerations}"
                ),
                "en": (
                    "## Security Analysis\n\n"
                    "### Risk Level\n"
                    "**{risk_level}** - {risk_description}\n\n"
                    "### Identified Attack Vectors\n"
                    "{attack_vectors}\n\n"
                    "### Potential Impact\n"
                    "{potential_impact}\n\n"
                    "### Mitigation Measures\n"
                    "{mitigation_measures}\n\n"
                    "### Compliance\n"
                    "{compliance_considerations}"
                )
            }
        }
    
    def _initialize_template_variables(self) -> Dict[str, List[str]]:
        """Inicializar variables de plantillas."""
        return {
            "analysis_summary": [
                "total_issues", "critical_issues", "high_issues", "quality_score",
                "quality_rating_text", "audience_specific_text"
            ],
            "violation_explanation": [
                "violation_title", "location", "severity", "category",
                "violation_description", "why_problematic", "how_to_fix",
                "audience_specific_section"
            ],
            "metrics_section": [
                "quality_score", "quality_rating", "cyclomatic_complexity",
                "lines_of_code", "function_count", "class_count", "test_coverage",
                "metrics_interpretation"
            ],
            "recommendations_section": [
                "high_priority_recommendations", "medium_priority_recommendations",
                "low_priority_recommendations", "implementation_guidance"
            ],
            "educational_content": [
                "concept_name", "concept_definition", "importance_explanation",
                "practical_examples", "best_practices", "learning_resources"
            ],
            "interactive_qa": [
                "question", "answer", "confidence", "related_questions",
                "follow_up_suggestions"
            ],
            "business_impact": [
                "estimated_time", "estimated_cost", "correction_roi",
                "timeline_impact", "business_risks", "strategic_recommendations"
            ],
            "security_analysis": [
                "risk_level", "risk_description", "attack_vectors",
                "potential_impact", "mitigation_measures", "compliance_considerations"
            ]
        }
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Inicializar reglas de validación."""
        return {
            "required_variables": True,
            "max_template_length": 10000,
            "min_template_length": 10,
            "allowed_variable_format": r"\{[a-zA-Z_][a-zA-Z0-9_]*\}",
            "forbidden_patterns": [
                r"<script.*?>.*?</script>",
                r"javascript:",
                r"on\w+\s*="
            ]
        }
    
    async def render_template(
        self, 
        template_key: str, 
        variables: Dict[str, str],
        language: Language
    ) -> str:
        """Renderizar plantilla con variables."""
        try:
            # Obtener plantilla
            template = await self._get_template(template_key, language)
            
            # Validar variables
            await self._validate_variables(template_key, variables)
            
            # Renderizar plantilla
            rendered_content = await self._render_template_content(template, variables)
            
            # Validar resultado
            await self._validate_rendered_content(rendered_content)
            
            logger.info(f"Plantilla {template_key} renderizada exitosamente para {language.value}")
            return rendered_content
            
        except Exception as e:
            logger.error(f"Error renderizando plantilla {template_key}: {str(e)}")
            raise
    
    async def get_template_variables(self, template_key: str) -> List[str]:
        """Obtener variables de una plantilla."""
        return self.template_variables.get(template_key, [])
    
    async def validate_template(
        self, 
        template_key: str, 
        language: Language
    ) -> bool:
        """Validar plantilla."""
        try:
            # Verificar que la plantilla existe
            if template_key not in self.templates:
                logger.error(f"Plantilla {template_key} no encontrada")
                return False
            
            if language.value not in self.templates[template_key]:
                logger.error(f"Plantilla {template_key} no disponible para idioma {language.value}")
                return False
            
            template = self.templates[template_key][language.value]
            
            # Validar longitud
            if len(template) < self.validation_rules["min_template_length"]:
                logger.error(f"Plantilla {template_key} muy corta")
                return False
            
            if len(template) > self.validation_rules["max_template_length"]:
                logger.error(f"Plantilla {template_key} muy larga")
                return False
            
            # Validar patrones prohibidos
            for pattern in self.validation_rules["forbidden_patterns"]:
                if re.search(pattern, template, re.IGNORECASE):
                    logger.error(f"Plantilla {template_key} contiene patrón prohibido: {pattern}")
                    return False
            
            # Validar formato de variables
            variable_pattern = self.validation_rules["allowed_variable_format"]
            variables_in_template = re.findall(variable_pattern, template)
            
            for variable in variables_in_template:
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", variable[1:-1]):
                    logger.error(f"Variable inválida en plantilla {template_key}: {variable}")
                    return False
            
            logger.info(f"Plantilla {template_key} validada exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error validando plantilla {template_key}: {str(e)}")
            return False
    
    async def _get_template(self, template_key: str, language: Language) -> str:
        """Obtener plantilla por clave e idioma."""
        if template_key not in self.templates:
            raise ValueError(f"Plantilla {template_key} no encontrada")
        
        if language.value not in self.templates[template_key]:
            raise ValueError(f"Plantilla {template_key} no disponible para idioma {language.value}")
        
        return self.templates[template_key][language.value]
    
    async def _validate_variables(self, template_key: str, variables: Dict[str, str]) -> None:
        """Validar variables de la plantilla."""
        expected_variables = self.template_variables.get(template_key, [])
        
        # Verificar variables requeridas
        missing_variables = []
        for expected_var in expected_variables:
            if expected_var not in variables:
                missing_variables.append(expected_var)
        
        if missing_variables:
            raise ValueError(f"Variables faltantes para plantilla {template_key}: {missing_variables}")
        
        # Validar que no hay variables extra
        extra_variables = []
        for provided_var in variables.keys():
            if provided_var not in expected_variables:
                extra_variables.append(provided_var)
        
        if extra_variables:
            logger.warning(f"Variables extra para plantilla {template_key}: {extra_variables}")
    
    async def _render_template_content(self, template: str, variables: Dict[str, str]) -> str:
        """Renderizar contenido de la plantilla."""
        try:
            # Reemplazar variables en la plantilla
            rendered_content = template
            
            for variable, value in variables.items():
                placeholder = f"{{{variable}}}"
                rendered_content = rendered_content.replace(placeholder, str(value))
            
            # Verificar que no quedan variables sin reemplazar
            remaining_variables = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", rendered_content)
            if remaining_variables:
                logger.warning(f"Variables no reemplazadas: {remaining_variables}")
            
            return rendered_content
            
        except Exception as e:
            logger.error(f"Error renderizando contenido de plantilla: {str(e)}")
            raise
    
    async def _validate_rendered_content(self, content: str) -> None:
        """Validar contenido renderizado."""
        # Validar longitud
        if len(content) < 10:
            raise ValueError("Contenido renderizado muy corto")
        
        if len(content) > 50000:
            logger.warning("Contenido renderizado muy largo, considerando truncar")
        
        # Validar patrones prohibidos
        for pattern in self.validation_rules["forbidden_patterns"]:
            if re.search(pattern, content, re.IGNORECASE):
                raise ValueError(f"Contenido renderizado contiene patrón prohibido: {pattern}")
    
    async def create_custom_template(
        self, 
        template_key: str, 
        language: Language,
        template_content: str,
        variables: List[str]
    ) -> bool:
        """Crear plantilla personalizada."""
        try:
            # Validar contenido de la plantilla
            validation_result = await self._validate_custom_template(template_content, variables)
            if not validation_result.is_valid:
                logger.error(f"Plantilla personalizada inválida: {validation_result.errors}")
                return False
            
            # Agregar plantilla
            if template_key not in self.templates:
                self.templates[template_key] = {}
            
            self.templates[template_key][language.value] = template_content
            self.template_variables[template_key] = variables
            
            logger.info(f"Plantilla personalizada {template_key} creada exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error creando plantilla personalizada: {str(e)}")
            return False
    
    async def _validate_custom_template(
        self, 
        template_content: str, 
        variables: List[str]
    ) -> TemplateValidationResult:
        """Validar plantilla personalizada."""
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # Validar longitud
            if len(template_content) < self.validation_rules["min_template_length"]:
                errors.append("Plantilla muy corta")
            
            if len(template_content) > self.validation_rules["max_template_length"]:
                errors.append("Plantilla muy larga")
            
            # Validar patrones prohibidos
            for pattern in self.validation_rules["forbidden_patterns"]:
                if re.search(pattern, template_content, re.IGNORECASE):
                    errors.append(f"Contiene patrón prohibido: {pattern}")
            
            # Validar variables
            variable_pattern = self.validation_rules["allowed_variable_format"]
            variables_in_template = re.findall(variable_pattern, template_content)
            
            # Verificar que todas las variables declaradas están en la plantilla
            for variable in variables:
                if f"{{{variable}}}" not in template_content:
                    warnings.append(f"Variable declarada no encontrada en plantilla: {variable}")
            
            # Verificar que todas las variables en la plantilla están declaradas
            for variable_match in variables_in_template:
                variable_name = variable_match[1:-1]  # Remover llaves
                if variable_name not in variables:
                    errors.append(f"Variable no declarada encontrada en plantilla: {variable_name}")
            
            # Sugerencias
            if len(variables) > 10:
                suggestions.append("Considerar dividir la plantilla en plantillas más pequeñas")
            
            if len(template_content) > 5000:
                suggestions.append("Considerar simplificar la plantilla para mejor legibilidad")
            
            return TemplateValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error validando plantilla personalizada: {str(e)}")
            return TemplateValidationResult(
                is_valid=False,
                errors=[f"Error de validación: {str(e)}"],
                warnings=[],
                suggestions=[]
            )
    
    async def get_available_templates(self, language: Language) -> List[str]:
        """Obtener plantillas disponibles para un idioma."""
        available_templates = []
        
        for template_key, template_languages in self.templates.items():
            if language.value in template_languages:
                available_templates.append(template_key)
        
        return available_templates
    
    async def delete_template(self, template_key: str, language: Language) -> bool:
        """Eliminar plantilla."""
        try:
            if template_key in self.templates and language.value in self.templates[template_key]:
                del self.templates[template_key][language.value]
                
                # Si no quedan idiomas para esta plantilla, eliminar completamente
                if not self.templates[template_key]:
                    del self.templates[template_key]
                    if template_key in self.template_variables:
                        del self.template_variables[template_key]
                
                logger.info(f"Plantilla {template_key} eliminada para {language.value}")
                return True
            else:
                logger.warning(f"Plantilla {template_key} no encontrada para {language.value}")
                return False
                
        except Exception as e:
            logger.error(f"Error eliminando plantilla: {str(e)}")
            return False
    
    async def update_template(
        self, 
        template_key: str, 
        language: Language,
        new_content: str,
        new_variables: List[str]
    ) -> bool:
        """Actualizar plantilla existente."""
        try:
            # Validar nueva plantilla
            validation_result = await self._validate_custom_template(new_content, new_variables)
            if not validation_result.is_valid:
                logger.error(f"Plantilla actualizada inválida: {validation_result.errors}")
                return False
            
            # Actualizar plantilla
            if template_key not in self.templates:
                self.templates[template_key] = {}
            
            self.templates[template_key][language.value] = new_content
            self.template_variables[template_key] = new_variables
            
            logger.info(f"Plantilla {template_key} actualizada exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando plantilla: {str(e)}")
            return False
