"""
Implementación del Sistema de Explicaciones Interactivas.

Este módulo implementa el sistema de explicaciones interactivas que permite
a los usuarios hacer preguntas y obtener respuestas contextuales.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from uuid import uuid4

from ...domain.entities.explanation import (
    Language, Audience, InteractiveResponse, ResponseType, ExplanationContext
)
from ..ports.explanation_ports import InteractiveExplainerPort


logger = logging.getLogger(__name__)


class InteractiveExplainer(InteractiveExplainerPort):
    """Sistema de explicaciones interactivas."""
    
    def __init__(self):
        self.question_patterns = self._initialize_question_patterns()
        self.response_templates = self._initialize_response_templates()
        self.context_analyzer = self._initialize_context_analyzer()
        self.dialogue_history = {}
    
    def _initialize_question_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Inicializar patrones de preguntas."""
        return {
            "what_is": {
                "es": [
                    r"qué es (.+)",
                    r"qué significa (.+)",
                    r"definir (.+)",
                    r"explicar (.+)",
                    r"cuál es (.+)"
                ],
                "en": [
                    r"what is (.+)",
                    r"what does (.+) mean",
                    r"define (.+)",
                    r"explain (.+)",
                    r"what's (.+)"
                ]
            },
            "why_bad": {
                "es": [
                    r"por qué es malo (.+)",
                    r"por qué es problemático (.+)",
                    r"cuál es el problema con (.+)",
                    r"por qué no debería (.+)",
                    r"qué tiene de malo (.+)"
                ],
                "en": [
                    r"why is (.+) bad",
                    r"why is (.+) problematic",
                    r"what's wrong with (.+)",
                    r"why shouldn't (.+)",
                    r"what's bad about (.+)"
                ]
            },
            "how_to_fix": {
                "es": [
                    r"cómo arreglar (.+)",
                    r"cómo solucionar (.+)",
                    r"cómo mejorar (.+)",
                    r"cómo corregir (.+)",
                    r"qué hacer con (.+)"
                ],
                "en": [
                    r"how to fix (.+)",
                    r"how to solve (.+)",
                    r"how to improve (.+)",
                    r"how to correct (.+)",
                    r"what to do with (.+)"
                ]
            },
            "show_example": {
                "es": [
                    r"mostrar ejemplo de (.+)",
                    r"ejemplo de (.+)",
                    r"cómo se ve (.+)",
                    r"demostrar (.+)",
                    r"ilustrar (.+)"
                ],
                "en": [
                    r"show example of (.+)",
                    r"example of (.+)",
                    r"how does (.+) look",
                    r"demonstrate (.+)",
                    r"illustrate (.+)"
                ]
            },
            "more_info": {
                "es": [
                    r"más información sobre (.+)",
                    r"detalles de (.+)",
                    r"más detalles (.+)",
                    r"información adicional (.+)",
                    r"ampliar (.+)"
                ],
                "en": [
                    r"more information about (.+)",
                    r"details about (.+)",
                    r"more details (.+)",
                    r"additional information (.+)",
                    r"expand on (.+)"
                ]
            }
        }
    
    def _initialize_response_templates(self) -> Dict[str, Dict[str, str]]:
        """Inicializar plantillas de respuesta."""
        return {
            "definition": {
                "es": (
                    "**¿Qué es {concept}?**\n\n"
                    "{definition}\n\n"
                    "**En términos simples:**\n"
                    "{simple_explanation}\n\n"
                    "**¿Por qué es importante?**\n"
                    "{importance}\n\n"
                    "**Ejemplo práctico:**\n"
                    "{example}"
                ),
                "en": (
                    "**What is {concept}?**\n\n"
                    "{definition}\n\n"
                    "**In simple terms:**\n"
                    "{simple_explanation}\n\n"
                    "**Why is it important?**\n"
                    "{importance}\n\n"
                    "**Practical example:**\n"
                    "{example}"
                )
            },
            "problem_explanation": {
                "es": (
                    "**¿Por qué es problemático {concept}?**\n\n"
                    "{problem_description}\n\n"
                    "**Impacto en el código:**\n"
                    "{code_impact}\n\n"
                    "**Consecuencias:**\n"
                    "{consequences}\n\n"
                    "**Ejemplo del problema:**\n"
                    "{problem_example}"
                ),
                "en": (
                    "**Why is {concept} problematic?**\n\n"
                    "{problem_description}\n\n"
                    "**Impact on code:**\n"
                    "{code_impact}\n\n"
                    "**Consequences:**\n"
                    "{consequences}\n\n"
                    "**Problem example:**\n"
                    "{problem_example}"
                )
            },
            "fix_guidance": {
                "es": (
                    "**¿Cómo solucionar {concept}?**\n\n"
                    "**Pasos recomendados:**\n"
                    "{steps}\n\n"
                    "**Ejemplo de solución:**\n"
                    "{solution_example}\n\n"
                    "**Consejos adicionales:**\n"
                    "{additional_tips}\n\n"
                    "**Verificación:**\n"
                    "{verification_steps}"
                ),
                "en": (
                    "**How to fix {concept}?**\n\n"
                    "**Recommended steps:**\n"
                    "{steps}\n\n"
                    "**Solution example:**\n"
                    "{solution_example}\n\n"
                    "**Additional tips:**\n"
                    "{additional_tips}\n\n"
                    "**Verification:**\n"
                    "{verification_steps}"
                )
            },
            "example": {
                "es": (
                    "**Ejemplo de {concept}:**\n\n"
                    "**❌ Código problemático:**\n"
                    "{bad_example}\n\n"
                    "**✅ Código mejorado:**\n"
                    "{good_example}\n\n"
                    "**Explicación de las diferencias:**\n"
                    "{differences_explanation}\n\n"
                    "**Beneficios del cambio:**\n"
                    "{benefits}"
                ),
                "en": (
                    "**Example of {concept}:**\n\n"
                    "**❌ Problematic code:**\n"
                    "{bad_example}\n\n"
                    "**✅ Improved code:**\n"
                    "{good_example}\n\n"
                    "**Explanation of differences:**\n"
                    "{differences_explanation}\n\n"
                    "**Benefits of the change:**\n"
                    "{benefits}"
                )
            },
            "detailed_info": {
                "es": (
                    "**Información detallada sobre {concept}:**\n\n"
                    "**Aspectos técnicos:**\n"
                    "{technical_aspects}\n\n"
                    "**Consideraciones avanzadas:**\n"
                    "{advanced_considerations}\n\n"
                    "**Mejores prácticas:**\n"
                    "{best_practices}\n\n"
                    "**Recursos adicionales:**\n"
                    "{additional_resources}"
                ),
                "en": (
                    "**Detailed information about {concept}:**\n\n"
                    "**Technical aspects:**\n"
                    "{technical_aspects}\n\n"
                    "**Advanced considerations:**\n"
                    "{advanced_considerations}\n\n"
                    "**Best practices:**\n"
                    "{best_practices}\n\n"
                    "**Additional resources:**\n"
                    "{additional_resources}"
                )
            }
        }
    
    def _initialize_context_analyzer(self) -> Dict[str, Any]:
        """Inicializar analizador de contexto."""
        return {
            "concept_keywords": {
                "es": {
                    "complejidad": "cyclomatic_complexity",
                    "código con mal olor": "code_smell",
                    "refactorización": "refactoring",
                    "patrón de diseño": "design_pattern",
                    "prueba unitaria": "unit_testing",
                    "mantenibilidad": "maintainability",
                    "legibilidad": "readability",
                    "rendimiento": "performance"
                },
                "en": {
                    "complexity": "cyclomatic_complexity",
                    "code smell": "code_smell",
                    "refactoring": "refactoring",
                    "design pattern": "design_pattern",
                    "unit test": "unit_testing",
                    "maintainability": "maintainability",
                    "readability": "readability",
                    "performance": "performance"
                }
            },
            "severity_keywords": {
                "es": {
                    "crítico": "critical",
                    "alto": "high",
                    "medio": "medium",
                    "bajo": "low"
                },
                "en": {
                    "critical": "critical",
                    "high": "high",
                    "medium": "medium",
                    "low": "low"
                }
            }
        }
    
    async def generate_interactive_elements(
        self, 
        analysis_result: Dict[str, Any],
        request: Any
    ) -> List[Dict[str, Any]]:
        """Generar elementos interactivos."""
        try:
            elements = []
            
            # Generar secciones expandibles
            expandable_sections = await self.generate_expandable_sections(analysis_result, request)
            elements.extend(expandable_sections)
            
            # Generar elementos de Q&A
            qa_elements = await self.generate_qa_elements(analysis_result, request)
            elements.extend(qa_elements)
            
            # Generar elementos de comparación de código
            comparison_elements = await self._generate_code_comparison_elements(analysis_result, request)
            elements.extend(comparison_elements)
            
            # Generar tutoriales interactivos
            tutorial_elements = await self._generate_tutorial_elements(analysis_result, request)
            elements.extend(tutorial_elements)
            
            logger.info(f"Elementos interactivos generados: {len(elements)}")
            return elements
            
        except Exception as e:
            logger.error(f"Error generando elementos interactivos: {str(e)}")
            return []
    
    async def handle_user_question(
        self, 
        question: str, 
        context: ExplanationContext
    ) -> InteractiveResponse:
        """Manejar pregunta del usuario."""
        try:
            # Analizar la pregunta
            question_analysis = await self.analyze_user_question(question, context)
            
            # Generar respuesta apropiada
            response = await self._generate_response(question_analysis, context)
            
            # Generar preguntas de seguimiento
            follow_up_questions = await self._generate_follow_up_questions(question_analysis, context)
            
            # Generar contenido relacionado
            related_content = await self._generate_related_content(question_analysis, context)
            
            # Generar sugerencias de acción
            action_suggestions = await self._generate_action_suggestions(question_analysis, context)
            
            # Crear respuesta interactiva
            interactive_response = InteractiveResponse(
                response_text=response,
                response_type=question_analysis.get('response_type', ResponseType.EXPLANATION),
                confidence=question_analysis.get('confidence', 0.8),
                follow_up_questions=follow_up_questions,
                related_content=related_content,
                action_suggestions=action_suggestions
            )
            
            # Guardar en historial de diálogo
            await self._save_dialogue_history(context.explanation_id, question, interactive_response)
            
            logger.info(f"Pregunta procesada: {question[:50]}...")
            return interactive_response
            
        except Exception as e:
            logger.error(f"Error manejando pregunta del usuario: {str(e)}")
            return InteractiveResponse(
                response_text="Lo siento, no pude procesar tu pregunta. Por favor, intenta reformularla.",
                response_type=ResponseType.CLARIFICATION,
                confidence=0.0
            )
    
    async def generate_expandable_sections(
        self, 
        analysis_result: Dict[str, Any],
        request: Any
    ) -> List[Dict[str, Any]]:
        """Generar secciones expandibles."""
        try:
            sections = []
            
            # Generar secciones para violaciones
            violations = analysis_result.get('violations', [])
            for violation in violations:
                section = await self._create_expandable_violation_section(violation, request)
                sections.append(section)
            
            # Generar secciones para métricas
            metrics = analysis_result.get('metrics', {})
            if metrics:
                metrics_section = await self._create_expandable_metrics_section(metrics, request)
                sections.append(metrics_section)
            
            return sections
            
        except Exception as e:
            logger.error(f"Error generando secciones expandibles: {str(e)}")
            return []
    
    async def generate_qa_elements(
        self, 
        analysis_result: Dict[str, Any],
        request: Any
    ) -> List[Dict[str, Any]]:
        """Generar elementos de Q&A."""
        try:
            qa_elements = []
            
            # Generar preguntas comunes basadas en el análisis
            common_questions = await self._generate_common_questions(analysis_result, request)
            
            for question_data in common_questions:
                qa_element = {
                    "id": str(uuid4()),
                    "element_type": "question_answer",
                    "title": question_data["question"],
                    "content": question_data["answer"],
                    "initial_state": "collapsed",
                    "triggers": ["click"],
                    "related_violation_id": question_data.get("related_violation_id")
                }
                qa_elements.append(qa_element)
            
            return qa_elements
            
        except Exception as e:
            logger.error(f"Error generando elementos Q&A: {str(e)}")
            return []
    
    async def analyze_user_question(
        self, 
        question: str, 
        context: ExplanationContext
    ) -> Dict[str, Any]:
        """Analizar pregunta del usuario."""
        try:
            # Detectar idioma de la pregunta
            language = await self._detect_question_language(question)
            
            # Identificar tipo de pregunta
            question_type = await self._identify_question_type(question, language)
            
            # Extraer conceptos mencionados
            concepts = await self._extract_concepts_from_question(question, language)
            
            # Calcular confianza
            confidence = await self._calculate_confidence(question, question_type, concepts)
            
            # Determinar tipo de respuesta
            response_type = await self._determine_response_type(question_type)
            
            return {
                "question": question,
                "language": language,
                "question_type": question_type,
                "concepts": concepts,
                "confidence": confidence,
                "response_type": response_type,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Error analizando pregunta: {str(e)}")
            return {
                "question": question,
                "language": "es",
                "question_type": "unknown",
                "concepts": [],
                "confidence": 0.0,
                "response_type": ResponseType.CLARIFICATION,
                "context": context
            }
    
    async def _detect_question_language(self, question: str) -> str:
        """Detectar idioma de la pregunta."""
        spanish_indicators = ["qué", "cómo", "por qué", "cuál", "dónde", "cuándo"]
        english_indicators = ["what", "how", "why", "which", "where", "when"]
        
        question_lower = question.lower()
        
        spanish_count = sum(1 for indicator in spanish_indicators if indicator in question_lower)
        english_count = sum(1 for indicator in english_indicators if indicator in question_lower)
        
        return "es" if spanish_count > english_count else "en"
    
    async def _identify_question_type(self, question: str, language: str) -> str:
        """Identificar tipo de pregunta."""
        patterns = self.question_patterns
        
        for question_type, lang_patterns in patterns.items():
            for pattern in lang_patterns.get(language, []):
                if re.search(pattern, question, re.IGNORECASE):
                    return question_type
        
        return "unknown"
    
    async def _extract_concepts_from_question(self, question: str, language: str) -> List[str]:
        """Extraer conceptos mencionados en la pregunta."""
        concepts = []
        concept_keywords = self.context_analyzer["concept_keywords"].get(language, {})
        
        question_lower = question.lower()
        
        for keyword, concept in concept_keywords.items():
            if keyword in question_lower:
                concepts.append(concept)
        
        return concepts
    
    async def _calculate_confidence(
        self, 
        question: str, 
        question_type: str, 
        concepts: List[str]
    ) -> float:
        """Calcular confianza en la respuesta."""
        confidence = 0.5  # Base confidence
        
        # Aumentar confianza si se identificó el tipo de pregunta
        if question_type != "unknown":
            confidence += 0.2
        
        # Aumentar confianza si se identificaron conceptos
        if concepts:
            confidence += 0.2
        
        # Aumentar confianza si la pregunta es clara
        if len(question.split()) >= 3:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _determine_response_type(self, question_type: str) -> ResponseType:
        """Determinar tipo de respuesta."""
        response_type_mapping = {
            "what_is": ResponseType.DEFINITION,
            "why_bad": ResponseType.EXPLANATION,
            "how_to_fix": ResponseType.GUIDANCE,
            "show_example": ResponseType.EXAMPLE,
            "more_info": ResponseType.EXPLANATION,
            "unknown": ResponseType.CLARIFICATION
        }
        
        return response_type_mapping.get(question_type, ResponseType.EXPLANATION)
    
    async def _generate_response(
        self, 
        question_analysis: Dict[str, Any], 
        context: ExplanationContext
    ) -> str:
        """Generar respuesta basada en el análisis de la pregunta."""
        try:
            question_type = question_analysis["question_type"]
            language = question_analysis["language"]
            concepts = question_analysis["concepts"]
            
            # Obtener plantilla de respuesta
            response_template = self.response_templates.get(question_type, {}).get(language, "")
            
            if not response_template:
                return await self._generate_fallback_response(question_analysis)
            
            # Generar contenido específico para la respuesta
            response_content = await self._generate_response_content(question_analysis, context)
            
            # Renderizar plantilla
            response = response_template.format(**response_content)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {str(e)}")
            return await self._generate_fallback_response(question_analysis)
    
    async def _generate_response_content(
        self, 
        question_analysis: Dict[str, Any], 
        context: ExplanationContext
    ) -> Dict[str, str]:
        """Generar contenido específico para la respuesta."""
        concepts = question_analysis["concepts"]
        language = question_analysis["language"]
        
        # Usar el primer concepto identificado
        main_concept = concepts[0] if concepts else "el problema"
        
        # Generar contenido basado en el concepto
        content = await self._get_concept_content(main_concept, language)
        
        return content
    
    async def _get_concept_content(self, concept: str, language: str) -> Dict[str, str]:
        """Obtener contenido para un concepto específico."""
        concept_content_db = {
            "cyclomatic_complexity": {
                "es": {
                    "concept": "complejidad ciclomática",
                    "definition": "Métrica que mide la complejidad de un programa contando el número de rutas independientes a través del código",
                    "simple_explanation": "Es una forma de medir qué tan complicado es tu código",
                    "importance": "Ayuda a identificar código que es difícil de entender y mantener",
                    "example": "Una función con muchas condiciones if/else tendrá alta complejidad ciclomática"
                },
                "en": {
                    "concept": "cyclomatic complexity",
                    "definition": "Metric that measures program complexity by counting the number of independent paths through the code",
                    "simple_explanation": "It's a way to measure how complicated your code is",
                    "importance": "Helps identify code that is hard to understand and maintain",
                    "example": "A function with many if/else conditions will have high cyclomatic complexity"
                }
            }
        }
        
        return concept_content_db.get(concept, {}).get(language, {
            "concept": concept,
            "definition": "Definición no disponible",
            "simple_explanation": "Explicación no disponible",
            "importance": "Importancia no disponible",
            "example": "Ejemplo no disponible"
        })
    
    async def _generate_fallback_response(self, question_analysis: Dict[str, Any]) -> str:
        """Generar respuesta de respaldo."""
        language = question_analysis["language"]
        
        if language == "es":
            return (
                "Lo siento, no tengo suficiente información para responder tu pregunta específicamente. "
                "¿Podrías reformular tu pregunta o ser más específico sobre qué te gustaría saber?"
            )
        else:
            return (
                "I'm sorry, I don't have enough information to answer your question specifically. "
                "Could you rephrase your question or be more specific about what you'd like to know?"
            )
    
    async def _generate_follow_up_questions(
        self, 
        question_analysis: Dict[str, Any], 
        context: ExplanationContext
    ) -> List[str]:
        """Generar preguntas de seguimiento."""
        language = question_analysis["language"]
        concepts = question_analysis["concepts"]
        
        follow_up_questions = []
        
        if language == "es":
            if "cyclomatic_complexity" in concepts:
                follow_up_questions.extend([
                    "¿Cómo puedo reducir la complejidad ciclomática?",
                    "¿Cuál es el valor máximo recomendado?",
                    "¿Qué herramientas puedo usar para medirla?"
                ])
            else:
                follow_up_questions.extend([
                    "¿Puedes darme más detalles?",
                    "¿Hay algún ejemplo que puedas mostrarme?",
                    "¿Cómo puedo aplicar esto en mi código?"
                ])
        else:
            if "cyclomatic_complexity" in concepts:
                follow_up_questions.extend([
                    "How can I reduce cyclomatic complexity?",
                    "What is the recommended maximum value?",
                    "What tools can I use to measure it?"
                ])
            else:
                follow_up_questions.extend([
                    "Can you give me more details?",
                    "Is there an example you can show me?",
                    "How can I apply this to my code?"
                ])
        
        return follow_up_questions[:3]  # Limitar a 3 preguntas
    
    async def _generate_related_content(
        self, 
        question_analysis: Dict[str, Any], 
        context: ExplanationContext
    ) -> List[str]:
        """Generar contenido relacionado."""
        concepts = question_analysis["concepts"]
        
        related_content = []
        
        # Mapear conceptos a contenido relacionado
        concept_relations = {
            "cyclomatic_complexity": ["refactoring", "code_smell", "maintainability"],
            "code_smell": ["refactoring", "clean_code", "maintainability"],
            "refactoring": ["design_patterns", "testing", "code_quality"]
        }
        
        for concept in concepts:
            related = concept_relations.get(concept, [])
            related_content.extend(related)
        
        return list(set(related_content))[:5]  # Limitar a 5 elementos únicos
    
    async def _generate_action_suggestions(
        self, 
        question_analysis: Dict[str, Any], 
        context: ExplanationContext
    ) -> List[str]:
        """Generar sugerencias de acción."""
        language = question_analysis["language"]
        concepts = question_analysis["concepts"]
        
        action_suggestions = []
        
        if language == "es":
            if "cyclomatic_complexity" in concepts:
                action_suggestions.extend([
                    "Revisar funciones con complejidad > 10",
                    "Dividir funciones complejas en funciones más pequeñas",
                    "Usar early returns para reducir anidamiento"
                ])
            else:
                action_suggestions.extend([
                    "Revisar el código relacionado",
                    "Aplicar las mejores prácticas",
                    "Considerar refactoring si es necesario"
                ])
        else:
            if "cyclomatic_complexity" in concepts:
                action_suggestions.extend([
                    "Review functions with complexity > 10",
                    "Break complex functions into smaller ones",
                    "Use early returns to reduce nesting"
                ])
            else:
                action_suggestions.extend([
                    "Review related code",
                    "Apply best practices",
                    "Consider refactoring if needed"
                ])
        
        return action_suggestions[:3]  # Limitar a 3 sugerencias
    
    async def _create_expandable_violation_section(
        self, 
        violation: Dict[str, Any], 
        request: Any
    ) -> Dict[str, Any]:
        """Crear sección expandible para una violación."""
        return {
            "id": str(uuid4()),
            "element_type": "expandable_section",
            "title": f"Detalles técnicos: {violation.get('message', 'Violación')}",
            "content": await self._generate_technical_details(violation, request),
            "initial_state": "collapsed",
            "triggers": ["click"],
            "related_violation_id": violation.get('id')
        }
    
    async def _create_expandable_metrics_section(
        self, 
        metrics: Dict[str, Any], 
        request: Any
    ) -> Dict[str, Any]:
        """Crear sección expandible para métricas."""
        return {
            "id": str(uuid4()),
            "element_type": "expandable_section",
            "title": "Análisis detallado de métricas",
            "content": await self._generate_metrics_details(metrics, request),
            "initial_state": "collapsed",
            "triggers": ["click"]
        }
    
    async def _generate_technical_details(
        self, 
        violation: Dict[str, Any], 
        request: Any
    ) -> str:
        """Generar detalles técnicos para una violación."""
        if request.language == Language.SPANISH:
            return (
                f"**Análisis técnico detallado:**\n\n"
                f"**Regla:** {violation.get('rule_id', 'N/A')}\n"
                f"**Categoría:** {violation.get('rule_category', 'N/A')}\n"
                f"**Severidad:** {violation.get('severity', 'N/A')}\n"
                f"**Líneas:** {violation.get('location', {}).get('start_line', 'N/A')}-{violation.get('location', {}).get('end_line', 'N/A')}\n\n"
                f"**Explicación técnica:**\n"
                f"Esta violación indica un problema en el diseño o implementación del código que puede afectar la mantenibilidad y calidad del software."
            )
        else:
            return (
                f"**Detailed technical analysis:**\n\n"
                f"**Rule:** {violation.get('rule_id', 'N/A')}\n"
                f"**Category:** {violation.get('rule_category', 'N/A')}\n"
                f"**Severity:** {violation.get('severity', 'N/A')}\n"
                f"**Lines:** {violation.get('location', {}).get('start_line', 'N/A')}-{violation.get('location', {}).get('end_line', 'N/A')}\n\n"
                f"**Technical explanation:**\n"
                f"This violation indicates a problem in the code design or implementation that may affect software maintainability and quality."
            )
    
    async def _generate_metrics_details(
        self, 
        metrics: Dict[str, Any], 
        request: Any
    ) -> str:
        """Generar detalles de métricas."""
        if request.language == Language.SPANISH:
            return (
                f"**Análisis detallado de métricas:**\n\n"
                f"**Puntuación general:** {metrics.get('overall_quality_score', 'N/A')}/100\n"
                f"**Complejidad ciclomática:** {metrics.get('complexity_metrics', {}).get('cyclomatic_complexity', 'N/A')}\n"
                f"**Líneas de código:** {metrics.get('code_metrics', {}).get('lines_of_code', 'N/A')}\n"
                f"**Funciones:** {metrics.get('code_metrics', {}).get('function_count', 'N/A')}\n"
                f"**Clases:** {metrics.get('code_metrics', {}).get('class_count', 'N/A')}\n\n"
                f"**Interpretación:**\n"
                f"Estas métricas proporcionan una visión general de la calidad y complejidad del código analizado."
            )
        else:
            return (
                f"**Detailed metrics analysis:**\n\n"
                f"**Overall score:** {metrics.get('overall_quality_score', 'N/A')}/100\n"
                f"**Cyclomatic complexity:** {metrics.get('complexity_metrics', {}).get('cyclomatic_complexity', 'N/A')}\n"
                f"**Lines of code:** {metrics.get('code_metrics', {}).get('lines_of_code', 'N/A')}\n"
                f"**Functions:** {metrics.get('code_metrics', {}).get('function_count', 'N/A')}\n"
                f"**Classes:** {metrics.get('code_metrics', {}).get('class_count', 'N/A')}\n\n"
                f"**Interpretation:**\n"
                f"These metrics provide an overview of the quality and complexity of the analyzed code."
            )
    
    async def _generate_common_questions(
        self, 
        analysis_result: Dict[str, Any], 
        request: Any
    ) -> List[Dict[str, Any]]:
        """Generar preguntas comunes basadas en el análisis."""
        questions = []
        
        violations = analysis_result.get('violations', [])
        if violations:
            if request.language == Language.SPANISH:
                questions.append({
                    "question": "¿Cuál es el problema más crítico encontrado?",
                    "answer": f"Se encontraron {len(violations)} problemas. Los más críticos son aquellos con severidad 'critical'.",
                    "related_violation_id": violations[0].get('id') if violations else None
                })
            else:
                questions.append({
                    "question": "What is the most critical issue found?",
                    "answer": f"Found {len(violations)} issues. The most critical ones are those with 'critical' severity.",
                    "related_violation_id": violations[0].get('id') if violations else None
                })
        
        return questions
    
    async def _generate_code_comparison_elements(
        self, 
        analysis_result: Dict[str, Any], 
        request: Any
    ) -> List[Dict[str, Any]]:
        """Generar elementos de comparación de código."""
        # Implementar lógica para generar comparaciones de código
        return []
    
    async def _generate_tutorial_elements(
        self, 
        analysis_result: Dict[str, Any], 
        request: Any
    ) -> List[Dict[str, Any]]:
        """Generar elementos de tutorial interactivo."""
        # Implementar lógica para generar tutoriales interactivos
        return []
    
    async def _save_dialogue_history(
        self, 
        explanation_id: str, 
        question: str, 
        response: InteractiveResponse
    ) -> None:
        """Guardar historial de diálogo."""
        if explanation_id not in self.dialogue_history:
            self.dialogue_history[explanation_id] = []
        
        self.dialogue_history[explanation_id].append({
            "question": question,
            "response": response.response_text,
            "timestamp": str(uuid4())  # Placeholder for timestamp
        })
        
        # Limitar historial a 10 interacciones por explicación
        if len(self.dialogue_history[explanation_id]) > 10:
            self.dialogue_history[explanation_id] = self.dialogue_history[explanation_id][-10:]
