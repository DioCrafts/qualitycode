"""
Implementación del Generador de Contenido Educativo.

Este módulo implementa la generación de contenido educativo adaptado
a diferentes niveles de experiencia y estilos de aprendizaje.
"""

import logging
from typing import Dict, List, Any, Optional
from uuid import uuid4

from ...domain.entities.explanation import Language, Audience
from ...domain.value_objects.explanation_metrics import ExperienceLevel, LearningStyle
from ..ports.explanation_ports import EducationalContentPort


logger = logging.getLogger(__name__)


class EducationalContentGenerator(EducationalContentPort):
    """Generador de contenido educativo."""
    
    def __init__(self):
        self.concept_database = self._initialize_concept_database()
        self.learning_paths = self._initialize_learning_paths()
        self.example_templates = self._initialize_example_templates()
        self.difficulty_levels = self._initialize_difficulty_levels()
    
    def _initialize_concept_database(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar base de datos de conceptos."""
        return {
            "cyclomatic_complexity": {
                "name_es": "Complejidad Ciclomática",
                "name_en": "Cyclomatic Complexity",
                "definition_es": "Métrica que mide la complejidad de un programa contando el número de rutas independientes a través del código",
                "definition_en": "Metric that measures program complexity by counting the number of independent paths through the code",
                "difficulty_level": "intermediate",
                "prerequisites": ["basic_programming", "control_structures"],
                "related_concepts": ["code_complexity", "maintainability", "testing"],
                "learning_objectives": [
                    "Entender qué es la complejidad ciclomática",
                    "Calcular la complejidad ciclomática de una función",
                    "Identificar funciones con alta complejidad",
                    "Aplicar técnicas de reducción de complejidad"
                ]
            },
            "code_smell": {
                "name_es": "Código con Mal Olor",
                "name_en": "Code Smell",
                "definition_es": "Indicadores superficiales en el código que sugieren un problema más profundo en el diseño",
                "definition_en": "Superficial indicators in code that suggest a deeper problem in design",
                "difficulty_level": "beginner",
                "prerequisites": ["basic_programming"],
                "related_concepts": ["refactoring", "clean_code", "maintainability"],
                "learning_objectives": [
                    "Identificar diferentes tipos de code smells",
                    "Entender el impacto de los code smells",
                    "Aplicar técnicas de refactoring para eliminar smells"
                ]
            },
            "refactoring": {
                "name_es": "Refactorización",
                "name_en": "Refactoring",
                "definition_es": "Proceso de mejorar el código existente sin cambiar su funcionalidad externa",
                "definition_en": "Process of improving existing code without changing its external functionality",
                "difficulty_level": "intermediate",
                "prerequisites": ["code_smell", "design_patterns"],
                "related_concepts": ["clean_code", "maintainability", "testing"],
                "learning_objectives": [
                    "Entender los principios de refactoring",
                    "Aplicar técnicas de refactoring comunes",
                    "Usar tests para validar refactoring",
                    "Identificar cuándo refactorizar"
                ]
            },
            "design_patterns": {
                "name_es": "Patrones de Diseño",
                "name_en": "Design Patterns",
                "definition_es": "Soluciones reutilizables a problemas comunes en el diseño de software",
                "definition_en": "Reusable solutions to common problems in software design",
                "difficulty_level": "advanced",
                "prerequisites": ["object_oriented_programming", "refactoring"],
                "related_concepts": ["architecture", "maintainability", "scalability"],
                "learning_objectives": [
                    "Conocer los patrones de diseño más comunes",
                    "Aplicar patrones apropiados en diferentes contextos",
                    "Identificar cuándo usar cada patrón",
                    "Evitar el sobre-uso de patrones"
                ]
            },
            "unit_testing": {
                "name_es": "Pruebas Unitarias",
                "name_en": "Unit Testing",
                "definition_es": "Práctica de escribir pruebas para verificar que unidades individuales de código funcionan correctamente",
                "definition_en": "Practice of writing tests to verify that individual units of code work correctly",
                "difficulty_level": "beginner",
                "prerequisites": ["basic_programming"],
                "related_concepts": ["test_driven_development", "code_coverage", "quality_assurance"],
                "learning_objectives": [
                    "Escribir pruebas unitarias efectivas",
                    "Entender los principios de testing",
                    "Usar frameworks de testing",
                    "Aplicar TDD (Test-Driven Development)"
                ]
            }
        }
    
    def _initialize_learning_paths(self) -> Dict[str, List[str]]:
        """Inicializar rutas de aprendizaje."""
        return {
            "beginner": [
                "basic_programming",
                "code_smell",
                "unit_testing",
                "simple_refactoring"
            ],
            "intermediate": [
                "cyclomatic_complexity",
                "refactoring",
                "code_coverage",
                "design_principles"
            ],
            "advanced": [
                "design_patterns",
                "architecture_patterns",
                "performance_optimization",
                "security_best_practices"
            ],
            "expert": [
                "system_design",
                "scalability_patterns",
                "advanced_testing",
                "code_quality_metrics"
            ]
        }
    
    def _initialize_example_templates(self) -> Dict[str, Dict[str, str]]:
        """Inicializar plantillas de ejemplos."""
        return {
            "cyclomatic_complexity": {
                "es": {
                    "bad_example": (
                        "```python\n"
                        "def procesar_datos(datos):\n"
                        "    if datos:\n"
                        "        if len(datos) > 0:\n"
                        "            if datos[0] == 'A':\n"
                        "                if datos[1] == 'B':\n"
                        "                    return 'AB'\n"
                        "                else:\n"
                        "                    return 'A'\n"
                        "            else:\n"
                        "                return 'Otro'\n"
                        "        else:\n"
                        "            return 'Vacío'\n"
                        "    else:\n"
                        "        return 'Nulo'\n"
                        "```\n"
                        "**Complejidad Ciclomática: 5** (muy alta)"
                    ),
                    "good_example": (
                        "```python\n"
                        "def procesar_datos(datos):\n"
                        "    if not datos or len(datos) == 0:\n"
                        "        return 'Sin datos'\n"
                        "    \n"
                        "    return procesar_tipo_especifico(datos)\n"
                        "\n"
                        "def procesar_tipo_especifico(datos):\n"
                        "    if datos[0] == 'A' and datos[1] == 'B':\n"
                        "        return 'AB'\n"
                        "    elif datos[0] == 'A':\n"
                        "        return 'A'\n"
                        "    else:\n"
                        "        return 'Otro'\n"
                        "```\n"
                        "**Complejidad Ciclomática: 2** (aceptable)"
                    )
                },
                "en": {
                    "bad_example": (
                        "```python\n"
                        "def process_data(data):\n"
                        "    if data:\n"
                        "        if len(data) > 0:\n"
                        "            if data[0] == 'A':\n"
                        "                if data[1] == 'B':\n"
                        "                    return 'AB'\n"
                        "                else:\n"
                        "                    return 'A'\n"
                        "            else:\n"
                        "                return 'Other'\n"
                        "        else:\n"
                        "            return 'Empty'\n"
                        "    else:\n"
                        "        return 'Null'\n"
                        "```\n"
                        "**Cyclomatic Complexity: 5** (very high)"
                    ),
                    "good_example": (
                        "```python\n"
                        "def process_data(data):\n"
                        "    if not data or len(data) == 0:\n"
                        "        return 'No data'\n"
                        "    \n"
                        "    return process_specific_type(data)\n"
                        "\n"
                        "def process_specific_type(data):\n"
                        "    if data[0] == 'A' and data[1] == 'B':\n"
                        "        return 'AB'\n"
                        "    elif data[0] == 'A':\n"
                        "        return 'A'\n"
                        "    else:\n"
                        "        return 'Other'\n"
                        "```\n"
                        "**Cyclomatic Complexity: 2** (acceptable)"
                    )
                }
            }
        }
    
    def _initialize_difficulty_levels(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar niveles de dificultad."""
        return {
            "beginner": {
                "description_es": "Conceptos básicos que cualquier programador debe conocer",
                "description_en": "Basic concepts that any programmer should know",
                "learning_time": "1-2 horas",
                "prerequisites": ["programación básica"],
                "teaching_approach": "explicación_simple_con_ejemplos"
            },
            "intermediate": {
                "description_es": "Conceptos intermedios para desarrolladores con experiencia",
                "description_en": "Intermediate concepts for experienced developers",
                "learning_time": "2-4 horas",
                "prerequisites": ["programación intermedia", "conceptos básicos"],
                "teaching_approach": "explicación_técnica_con_práctica"
            },
            "advanced": {
                "description_es": "Conceptos avanzados para desarrolladores senior",
                "description_en": "Advanced concepts for senior developers",
                "learning_time": "4-8 horas",
                "prerequisites": ["programación avanzada", "patrones de diseño"],
                "teaching_approach": "análisis_profundo_con_casos_reales"
            },
            "expert": {
                "description_es": "Conceptos de nivel experto para arquitectos y tech leads",
                "description_en": "Expert-level concepts for architects and tech leads",
                "learning_time": "8+ horas",
                "prerequisites": ["experiencia extensa", "conocimiento arquitectónico"],
                "teaching_approach": "análisis_arquitectónico_con_decisiones_complejas"
            }
        }
    
    async def generate_educational_content(
        self, 
        analysis_result: Dict[str, Any],
        request: Any
    ) -> List[Dict[str, Any]]:
        """Generar contenido educativo."""
        try:
            educational_content = []
            
            # Identificar conceptos complejos
            complex_concepts = await self.identify_complex_concepts(analysis_result)
            
            # Generar contenido para cada concepto
            for concept in complex_concepts:
                concept_content = await self._generate_concept_educational_content(
                    concept, request.audience, request.language
                )
                educational_content.append(concept_content)
            
            # Generar rutas de aprendizaje
            learning_paths = await self._generate_learning_paths_for_audience(request.audience)
            educational_content.extend(learning_paths)
            
            # Generar ejemplos prácticos
            practical_examples = await self._generate_practical_examples(
                analysis_result, request.audience, request.language
            )
            educational_content.extend(practical_examples)
            
            logger.info(f"Contenido educativo generado: {len(educational_content)} elementos")
            return educational_content
            
        except Exception as e:
            logger.error(f"Error generando contenido educativo: {str(e)}")
            return []
    
    async def generate_learning_path(
        self, 
        concept: str, 
        audience: Audience,
        language: Language
    ) -> List[Dict[str, Any]]:
        """Generar ruta de aprendizaje para un concepto."""
        try:
            concept_info = self.concept_database.get(concept, {})
            if not concept_info:
                logger.warning(f"Concepto {concept} no encontrado en la base de datos")
                return []
            
            # Determinar nivel de dificultad basado en audiencia
            difficulty_level = await self._determine_difficulty_for_audience(audience)
            
            # Generar ruta de aprendizaje
            learning_path = await self._create_learning_path(concept_info, difficulty_level, language)
            
            return learning_path
            
        except Exception as e:
            logger.error(f"Error generando ruta de aprendizaje: {str(e)}")
            return []
    
    async def generate_examples(
        self, 
        concept: str, 
        audience: Audience,
        language: Language
    ) -> List[Dict[str, Any]]:
        """Generar ejemplos para un concepto."""
        try:
            concept_info = self.concept_database.get(concept, {})
            if not concept_info:
                logger.warning(f"Concepto {concept} no encontrado en la base de datos")
                return []
            
            # Obtener ejemplos de las plantillas
            examples = await self._get_examples_from_templates(concept, language)
            
            # Adaptar ejemplos para la audiencia
            adapted_examples = await self._adapt_examples_for_audience(examples, audience, language)
            
            return adapted_examples
            
        except Exception as e:
            logger.error(f"Error generando ejemplos: {str(e)}")
            return []
    
    async def identify_complex_concepts(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Identificar conceptos complejos en el análisis."""
        try:
            complex_concepts = []
            
            # Analizar violaciones para identificar conceptos
            violations = analysis_result.get('violations', [])
            for violation in violations:
                rule_category = violation.get('rule_category', '')
                
                # Mapear categorías de reglas a conceptos
                concept_mapping = {
                    'complexity': 'cyclomatic_complexity',
                    'code_smell': 'code_smell',
                    'maintainability': 'refactoring',
                    'design': 'design_patterns',
                    'testing': 'unit_testing'
                }
                
                if rule_category in concept_mapping:
                    concept = concept_mapping[rule_category]
                    if concept not in complex_concepts:
                        complex_concepts.append(concept)
            
            # Analizar métricas para identificar conceptos adicionales
            metrics = analysis_result.get('metrics', {})
            complexity_metrics = metrics.get('complexity_metrics', {})
            
            if complexity_metrics.get('cyclomatic_complexity', 0) > 10:
                if 'cyclomatic_complexity' not in complex_concepts:
                    complex_concepts.append('cyclomatic_complexity')
            
            logger.info(f"Conceptos complejos identificados: {complex_concepts}")
            return complex_concepts
            
        except Exception as e:
            logger.error(f"Error identificando conceptos complejos: {str(e)}")
            return []
    
    async def _generate_concept_educational_content(
        self, 
        concept: str, 
        audience: Audience,
        language: Language
    ) -> Dict[str, Any]:
        """Generar contenido educativo para un concepto específico."""
        concept_info = self.concept_database.get(concept, {})
        
        if language == Language.SPANISH:
            name = concept_info.get('name_es', concept)
            definition = concept_info.get('definition_es', 'Definición no disponible')
        else:
            name = concept_info.get('name_en', concept)
            definition = concept_info.get('definition_en', 'Definition not available')
        
        # Generar explicación adaptada a la audiencia
        explanation = await self._generate_audience_specific_explanation(
            concept_info, audience, language
        )
        
        # Generar ejemplos
        examples = await self._get_examples_from_templates(concept, language)
        
        # Generar mejores prácticas
        best_practices = await self._generate_best_practices(concept, language)
        
        # Generar recursos de aprendizaje
        learning_resources = await self._generate_learning_resources(concept, language)
        
        return {
            "id": str(uuid4()),
            "title": name,
            "content": explanation,
            "content_type": "concept_explanation",
            "difficulty_level": concept_info.get('difficulty_level', 'intermediate'),
            "learning_objectives": concept_info.get('learning_objectives', []),
            "prerequisites": concept_info.get('prerequisites', []),
            "examples": examples,
            "best_practices": best_practices,
            "learning_resources": learning_resources
        }
    
    async def _generate_audience_specific_explanation(
        self, 
        concept_info: Dict[str, Any], 
        audience: Audience,
        language: Language
    ) -> str:
        """Generar explicación específica para la audiencia."""
        if audience == Audience.JUNIOR_DEVELOPER:
            return await self._generate_beginner_explanation(concept_info, language)
        elif audience == Audience.SENIOR_DEVELOPER:
            return await self._generate_advanced_explanation(concept_info, language)
        elif audience == Audience.PROJECT_MANAGER:
            return await self._generate_business_explanation(concept_info, language)
        else:
            return await self._generate_standard_explanation(concept_info, language)
    
    async def _generate_beginner_explanation(
        self, 
        concept_info: Dict[str, Any], 
        language: Language
    ) -> str:
        """Generar explicación para principiantes."""
        if language == Language.SPANISH:
            return (
                f"**¿Qué es {concept_info.get('name_es', 'este concepto')}?**\n\n"
                f"{concept_info.get('definition_es', 'Definición no disponible')}\n\n"
                "**¿Por qué es importante?**\n"
                "Este concepto te ayudará a escribir código más limpio y mantenible. "
                "Es una habilidad fundamental que todo programador debe desarrollar.\n\n"
                "**¿Cómo empezar?**\n"
                "Te recomendamos comenzar con ejemplos simples y practicar gradualmente "
                "con casos más complejos. ¡No te preocupes si no lo entiendes todo de inmediato!"
            )
        else:
            return (
                f"**What is {concept_info.get('name_en', 'this concept')}?**\n\n"
                f"{concept_info.get('definition_en', 'Definition not available')}\n\n"
                "**Why is it important?**\n"
                "This concept will help you write cleaner and more maintainable code. "
                "It's a fundamental skill that every programmer should develop.\n\n"
                "**How to get started?**\n"
                "We recommend starting with simple examples and gradually practicing "
                "with more complex cases. Don't worry if you don't understand everything immediately!"
            )
    
    async def _generate_advanced_explanation(
        self, 
        concept_info: Dict[str, Any], 
        language: Language
    ) -> str:
        """Generar explicación para desarrolladores avanzados."""
        if language == Language.SPANISH:
            return (
                f"**Análisis Técnico: {concept_info.get('name_es', 'Concepto')}**\n\n"
                f"{concept_info.get('definition_es', 'Definición no disponible')}\n\n"
                "**Consideraciones Técnicas:**\n"
                "• Impacto en el rendimiento del sistema\n"
                "• Implicaciones para la arquitectura\n"
                "• Trade-offs y alternativas\n"
                "• Mejores prácticas en contextos empresariales\n\n"
                "**Implementación Avanzada:**\n"
                "Considera el contexto específico de tu aplicación y las restricciones "
                "del sistema antes de implementar este concepto."
            )
        else:
            return (
                f"**Technical Analysis: {concept_info.get('name_en', 'Concept')}**\n\n"
                f"{concept_info.get('definition_en', 'Definition not available')}\n\n"
                "**Technical Considerations:**\n"
                "• Impact on system performance\n"
                "• Implications for architecture\n"
                "• Trade-offs and alternatives\n"
                "• Best practices in enterprise contexts\n\n"
                "**Advanced Implementation:**\n"
                "Consider the specific context of your application and system constraints "
                "before implementing this concept."
            )
    
    async def _generate_business_explanation(
        self, 
        concept_info: Dict[str, Any], 
        language: Language
    ) -> str:
        """Generar explicación orientada al negocio."""
        if language == Language.SPANISH:
            return (
                f"**Impacto de Negocio: {concept_info.get('name_es', 'Concepto')}**\n\n"
                f"{concept_info.get('definition_es', 'Definición no disponible')}\n\n"
                "**Implicaciones para el Negocio:**\n"
                "• **Tiempo de desarrollo:** Puede afectar la velocidad de entrega\n"
                "• **Costos de mantenimiento:** Impacta los costos a largo plazo\n"
                "• **Calidad del producto:** Mejora la satisfacción del cliente\n"
                "• **Riesgo del proyecto:** Reduce la probabilidad de problemas\n\n"
                "**Recomendación Estratégica:**\n"
                "Invertir en este concepto mejorará la eficiencia del equipo y "
                "reducirá los costos de mantenimiento a largo plazo."
            )
        else:
            return (
                f"**Business Impact: {concept_info.get('name_en', 'Concept')}**\n\n"
                f"{concept_info.get('definition_en', 'Definition not available')}\n\n"
                "**Business Implications:**\n"
                "• **Development time:** May affect delivery speed\n"
                "• **Maintenance costs:** Impacts long-term costs\n"
                "• **Product quality:** Improves customer satisfaction\n"
                "• **Project risk:** Reduces probability of issues\n\n"
                "**Strategic Recommendation:**\n"
                "Investing in this concept will improve team efficiency and "
                "reduce long-term maintenance costs."
            )
    
    async def _generate_standard_explanation(
        self, 
        concept_info: Dict[str, Any], 
        language: Language
    ) -> str:
        """Generar explicación estándar."""
        if language == Language.SPANISH:
            return (
                f"**{concept_info.get('name_es', 'Concepto')}**\n\n"
                f"{concept_info.get('definition_es', 'Definición no disponible')}\n\n"
                "**Aplicación Práctica:**\n"
                "Este concepto se aplica en el desarrollo de software para mejorar "
                "la calidad y mantenibilidad del código.\n\n"
                "**Beneficios:**\n"
                "• Mejora la legibilidad del código\n"
                "• Facilita el mantenimiento\n"
                "• Reduce la probabilidad de errores\n"
                "• Mejora la colaboración en equipo"
            )
        else:
            return (
                f"**{concept_info.get('name_en', 'Concept')}**\n\n"
                f"{concept_info.get('definition_en', 'Definition not available')}\n\n"
                "**Practical Application:**\n"
                "This concept is applied in software development to improve "
                "code quality and maintainability.\n\n"
                "**Benefits:**\n"
                "• Improves code readability\n"
                "• Facilitates maintenance\n"
                "• Reduces probability of errors\n"
                "• Improves team collaboration"
            )
    
    async def _get_examples_from_templates(
        self, 
        concept: str, 
        language: Language
    ) -> List[Dict[str, Any]]:
        """Obtener ejemplos de las plantillas."""
        examples = []
        
        concept_examples = self.example_templates.get(concept, {})
        lang_examples = concept_examples.get(language.value, {})
        
        if 'bad_example' in lang_examples:
            examples.append({
                "type": "bad_example",
                "title": "❌ Ejemplo Problemático" if language == Language.SPANISH else "❌ Problematic Example",
                "content": lang_examples['bad_example'],
                "explanation": "Este ejemplo muestra qué NO hacer" if language == Language.SPANISH else "This example shows what NOT to do"
            })
        
        if 'good_example' in lang_examples:
            examples.append({
                "type": "good_example",
                "title": "✅ Ejemplo Correcto" if language == Language.SPANISH else "✅ Correct Example",
                "content": lang_examples['good_example'],
                "explanation": "Este ejemplo muestra la forma correcta de hacerlo" if language == Language.SPANISH else "This example shows the correct way to do it"
            })
        
        return examples
    
    async def _adapt_examples_for_audience(
        self, 
        examples: List[Dict[str, Any]], 
        audience: Audience,
        language: Language
    ) -> List[Dict[str, Any]]:
        """Adaptar ejemplos para la audiencia."""
        adapted_examples = []
        
        for example in examples:
            adapted_example = example.copy()
            
            if audience == Audience.JUNIOR_DEVELOPER:
                # Agregar más explicaciones para principiantes
                if language == Language.SPANISH:
                    adapted_example["detailed_explanation"] = (
                        "Este ejemplo te muestra paso a paso cómo implementar este concepto. "
                        "Presta atención a cada línea y trata de entender por qué se hace de esta manera."
                    )
                else:
                    adapted_example["detailed_explanation"] = (
                        "This example shows you step by step how to implement this concept. "
                        "Pay attention to each line and try to understand why it's done this way."
                    )
            
            adapted_examples.append(adapted_example)
        
        return adapted_examples
    
    async def _generate_best_practices(self, concept: str, language: Language) -> List[str]:
        """Generar mejores prácticas para un concepto."""
        best_practices_db = {
            "cyclomatic_complexity": {
                "es": [
                    "Mantén la complejidad ciclomática por debajo de 10",
                    "Divide funciones complejas en funciones más pequeñas",
                    "Usa early returns para reducir anidamiento",
                    "Considera usar el patrón Strategy para lógica compleja"
                ],
                "en": [
                    "Keep cyclomatic complexity below 10",
                    "Break complex functions into smaller ones",
                    "Use early returns to reduce nesting",
                    "Consider using Strategy pattern for complex logic"
                ]
            },
            "code_smell": {
                "es": [
                    "Revisa regularmente el código en busca de smells",
                    "Refactoriza tan pronto como identifiques un smell",
                    "Usa herramientas de análisis estático",
                    "Mantén el código simple y legible"
                ],
                "en": [
                    "Regularly review code for smells",
                    "Refactor as soon as you identify a smell",
                    "Use static analysis tools",
                    "Keep code simple and readable"
                ]
            }
        }
        
        practices = best_practices_db.get(concept, {}).get(language.value, [])
        return practices
    
    async def _generate_learning_resources(self, concept: str, language: Language) -> List[Dict[str, str]]:
        """Generar recursos de aprendizaje."""
        resources_db = {
            "cyclomatic_complexity": {
                "es": [
                    {
                        "title": "Complejidad Ciclomática - Wikipedia",
                        "url": "https://es.wikipedia.org/wiki/Complejidad_ciclomática",
                        "type": "artículo"
                    },
                    {
                        "title": "Refactoring: Improving the Design of Existing Code",
                        "url": "https://martinfowler.com/books/refactoring.html",
                        "type": "libro"
                    }
                ],
                "en": [
                    {
                        "title": "Cyclomatic Complexity - Wikipedia",
                        "url": "https://en.wikipedia.org/wiki/Cyclomatic_complexity",
                        "type": "article"
                    },
                    {
                        "title": "Refactoring: Improving the Design of Existing Code",
                        "url": "https://martinfowler.com/books/refactoring.html",
                        "type": "book"
                    }
                ]
            }
        }
        
        resources = resources_db.get(concept, {}).get(language.value, [])
        return resources
    
    async def _determine_difficulty_for_audience(self, audience: Audience) -> str:
        """Determinar nivel de dificultad basado en audiencia."""
        difficulty_mapping = {
            Audience.JUNIOR_DEVELOPER: "beginner",
            Audience.SENIOR_DEVELOPER: "intermediate",
            Audience.TECHNICAL_LEAD: "advanced",
            Audience.SOFTWARE_ARCHITECT: "expert",
            Audience.PROJECT_MANAGER: "beginner",
            Audience.QUALITY_ASSURANCE: "intermediate",
            Audience.SECURITY_TEAM: "advanced",
            Audience.BUSINESS_STAKEHOLDER: "beginner"
        }
        
        return difficulty_mapping.get(audience, "intermediate")
    
    async def _create_learning_path(
        self, 
        concept_info: Dict[str, Any], 
        difficulty_level: str,
        language: Language
    ) -> List[Dict[str, Any]]:
        """Crear ruta de aprendizaje."""
        learning_path = []
        
        # Obtener prerequisitos
        prerequisites = concept_info.get('prerequisites', [])
        for prereq in prerequisites:
            learning_path.append({
                "step": len(learning_path) + 1,
                "title": f"Prerequisito: {prereq}",
                "type": "prerequisite",
                "estimated_time": "1-2 horas"
            })
        
        # Agregar concepto principal
        concept_name = concept_info.get('name_es' if language == Language.SPANISH else 'name_en', 'Concepto')
        learning_path.append({
            "step": len(learning_path) + 1,
            "title": f"Aprender: {concept_name}",
            "type": "main_concept",
            "estimated_time": self.difficulty_levels.get(difficulty_level, {}).get('learning_time', '2-4 horas')
        })
        
        # Agregar conceptos relacionados
        related_concepts = concept_info.get('related_concepts', [])
        for related in related_concepts[:3]:  # Limitar a 3 conceptos relacionados
            learning_path.append({
                "step": len(learning_path) + 1,
                "title": f"Explorar: {related}",
                "type": "related_concept",
                "estimated_time": "1-2 horas"
            })
        
        return learning_path
    
    async def _generate_learning_paths_for_audience(self, audience: Audience) -> List[Dict[str, Any]]:
        """Generar rutas de aprendizaje para la audiencia."""
        difficulty_level = await self._determine_difficulty_for_audience(audience)
        concepts = self.learning_paths.get(difficulty_level, [])
        
        learning_paths = []
        for concept in concepts:
            concept_info = self.concept_database.get(concept, {})
            if concept_info:
                learning_paths.append({
                    "id": str(uuid4()),
                    "title": f"Ruta de Aprendizaje: {concept_info.get('name_es', concept)}",
                    "content_type": "learning_path",
                    "difficulty_level": difficulty_level,
                    "concepts": concepts,
                    "estimated_total_time": f"{len(concepts) * 2}-{len(concepts) * 4} horas"
                })
        
        return learning_paths
    
    async def _generate_practical_examples(
        self, 
        analysis_result: Dict[str, Any], 
        audience: Audience,
        language: Language
    ) -> List[Dict[str, Any]]:
        """Generar ejemplos prácticos basados en el análisis."""
        examples = []
        
        # Generar ejemplos basados en las violaciones encontradas
        violations = analysis_result.get('violations', [])
        for violation in violations[:3]:  # Limitar a 3 ejemplos
            rule_category = violation.get('rule_category', '')
            
            if rule_category in self.concept_database:
                concept_examples = await self._get_examples_from_templates(rule_category, language)
                examples.extend(concept_examples)
        
        return examples
