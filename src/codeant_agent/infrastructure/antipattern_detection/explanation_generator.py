"""
Generador inteligente de explicaciones para antipatrones detectados.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from ...domain.entities.antipattern_analysis import (
    AntipatternExplanation, AntipatternType, AntipatternFeatures, 
    CodeExample, CodeHighlight, HighlightType, ExplanationConfig,
    ExplanationStyle, TargetAudience, VerbosityLevel
)
from ...domain.value_objects.programming_language import ProgrammingLanguage
from .classifiers.base_classifier import DetectedPattern

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Generador de explicaciones inteligente para antipatrones."""
    
    def __init__(self, config: Optional[ExplanationConfig] = None):
        self.config = config or ExplanationConfig()
        self.template_engine = ExplanationTemplateEngine()
        self.example_generator = CodeExampleGenerator()
        self.impact_analyzer = ImpactExplanationAnalyzer()
        
    async def generate_explanation(
        self, 
        pattern: DetectedPattern, 
        features: AntipatternFeatures
    ) -> AntipatternExplanation:
        """Generar explicación completa de un antipatrón."""
        
        try:
            # Generar explicación base
            explanation = AntipatternExplanation(
                pattern_type=pattern.pattern_type,
                summary=self._generate_summary(pattern),
                detailed_explanation=await self._generate_detailed_explanation(pattern, features),
                why_its_problematic=await self._explain_problems(pattern, features),
                potential_consequences=await self._list_consequences(pattern, features),
                how_to_fix=await self._generate_fix_instructions(pattern, features),
                confidence_explanation=self._explain_confidence(pattern),
                references=self._get_pattern_references(pattern.pattern_type)
            )
            
            # Generar ejemplos si está habilitado
            if self.config.include_examples:
                explanation.bad_example = await self.example_generator.generate_bad_example(
                    pattern.pattern_type, features.language
                )
                explanation.good_example = await self.example_generator.generate_good_example(
                    pattern.pattern_type, features.language
                )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation for {pattern.pattern_type}: {e}")
            return self._create_fallback_explanation(pattern)
    
    def _generate_summary(self, pattern: DetectedPattern) -> str:
        """Generar resumen del antipatrón."""
        
        confidence_level = self._get_confidence_level(pattern.confidence)
        pattern_name = self._get_pattern_display_name(pattern.pattern_type)
        
        base_templates = {
            AntipatternType.GOD_OBJECT: f"{pattern_name} detectado con confianza {confidence_level}. Esta clase tiene demasiadas responsabilidades y debería refactorizarse.",
            AntipatternType.SQL_INJECTION: f"Vulnerabilidad de {pattern_name} detectada con confianza {confidence_level}. La entrada del usuario se usa directamente en consultas SQL sin sanitización.",
            AntipatternType.N_PLUS_ONE_QUERY: f"Antipatrón {pattern_name} detectado con confianza {confidence_level}. Consultas de base de datos se ejecutan en bucles, causando problemas de rendimiento.",
            AntipatternType.LARGE_CLASS: f"{pattern_name} detectado con confianza {confidence_level}. La clase es demasiado grande y compleja para mantener eficazmente.",
            AntipatternType.SPAGHETTI_CODE: f"{pattern_name} detectado con confianza {confidence_level}. El código tiene un flujo de control complejo y enmarañado.",
            AntipatternType.MEMORY_LEAK: f"{pattern_name} potencial detectado con confianza {confidence_level}. Patrones que pueden causar acumulación de memoria.",
            AntipatternType.HARDCODED_SECRETS: f"{pattern_name} detectados con confianza {confidence_level}. Credenciales o secretos están hardcodeados en el código fuente."
        }
        
        return base_templates.get(
            pattern.pattern_type,
            f"Antipatrón {pattern_name} detectado con confianza {confidence_level}."
        )
    
    async def _generate_detailed_explanation(
        self, 
        pattern: DetectedPattern, 
        features: AntipatternFeatures
    ) -> str:
        """Generar explicación detallada."""
        
        explanations = {
            AntipatternType.GOD_OBJECT: await self._explain_god_object(pattern, features),
            AntipatternType.SQL_INJECTION: await self._explain_sql_injection(pattern, features),
            AntipatternType.N_PLUS_ONE_QUERY: await self._explain_n_plus_one(pattern, features),
            AntipatternType.LARGE_CLASS: await self._explain_large_class(pattern, features),
            AntipatternType.SPAGHETTI_CODE: await self._explain_spaghetti_code(pattern, features),
            AntipatternType.MEMORY_LEAK: await self._explain_memory_leak(pattern, features),
            AntipatternType.HARDCODED_SECRETS: await self._explain_hardcoded_secrets(pattern, features),
            AntipatternType.STRING_CONCATENATION_IN_LOOP: await self._explain_string_concat(pattern, features),
            AntipatternType.FEATURE_ENVY: await self._explain_feature_envy(pattern, features),
            AntipatternType.LONG_METHOD: await self._explain_long_method(pattern, features)
        }
        
        return explanations.get(
            pattern.pattern_type,
            f"El antipatrón {self._get_pattern_display_name(pattern.pattern_type)} ha sido detectado en su código. "
            f"Este patrón puede impactar negativamente la calidad y mantenibilidad del código."
        )
    
    async def _explain_problems(self, pattern: DetectedPattern, features: AntipatternFeatures) -> str:
        """Explicar por qué el patrón es problemático."""
        
        problems = {
            AntipatternType.GOD_OBJECT: "Los God Objects son problemáticos porque: 1) Violan el Principio de Responsabilidad Única, 2) Son difíciles de probar debido a múltiples dependencias, 3) Son difíciles de entender y modificar, 4) Crean acoplamiento fuerte con otras clases, 5) Hacen el código frágil y propenso a errores.",
            
            AntipatternType.SQL_INJECTION: "Las vulnerabilidades de SQL Injection son problemáticas porque: 1) Permiten acceso no autorizado a la base de datos, 2) Pueden llevar a filtraciones de datos, 3) Permiten manipulación o eliminación de datos, 4) Pueden permitir escalación de privilegios, 5) Violan requisitos de cumplimiento de seguridad.",
            
            AntipatternType.N_PLUS_ONE_QUERY: "Las consultas N+1 son problemáticas porque: 1) Causan degradación exponencial del rendimiento, 2) Aumentan la carga de la base de datos innecesariamente, 3) Llevan a timeouts con conjuntos de datos grandes, 4) Desperdician ancho de banda de red, 5) Crean cuellos de botella de escalabilidad.",
            
            AntipatternType.LARGE_CLASS: "Las clases grandes son problemáticas porque: 1) Violan el principio de responsabilidad única, 2) Son difíciles de entender y navegar, 3) Requieren más tiempo para hacer cambios, 4) Aumentan la probabilidad de conflictos de merge, 5) Dificultan las pruebas unitarias efectivas.",
            
            AntipatternType.SPAGHETTI_CODE: "El código espagueti es problemático porque: 1) Es extremadamente difícil de seguir y entender, 2) Los cambios pueden tener efectos secundarios impredecibles, 3) Es prácticamente imposible de probar comprehensivamente, 4) Aumenta significativamente el tiempo de desarrollo, 5) Hace que el debugging sea una pesadilla.",
            
            AntipatternType.MEMORY_LEAK: "Los memory leaks son problemáticos porque: 1) Causan degradación progresiva del rendimiento, 2) Pueden llevar a crashes de la aplicación, 3) Consumen recursos del sistema innecesariamente, 4) Afectan la estabilidad a largo plazo, 5) Pueden causar denial of service."
        }
        
        return problems.get(
            pattern.pattern_type,
            "Este antipatrón puede llevar a problemas de calidad del código, dificultades de mantenimiento y bugs potenciales."
        )
    
    async def _list_consequences(
        self, 
        pattern: DetectedPattern, 
        features: AntipatternFeatures
    ) -> List[str]:
        """Listar consecuencias potenciales."""
        
        consequences = {
            AntipatternType.GOD_OBJECT: [
                "Aumento del tiempo de desarrollo para nuevas características",
                "Mayor tasa de bugs debido a la complejidad",
                "Dificultad en pruebas unitarias",
                "Reducción de la reutilización de código",
                "Disminución de la productividad del equipo",
                "Mayores costos de mantenimiento"
            ],
            AntipatternType.SQL_INJECTION: [
                "Compromiso completo de la base de datos",
                "Robo o pérdida de datos",
                "Violaciones de cumplimiento regulatorio",
                "Responsabilidad legal",
                "Daño a la reputación",
                "Pérdidas financieras"
            ],
            AntipatternType.N_PLUS_ONE_QUERY: [
                "Aumento exponencial del tiempo de respuesta",
                "Sobrecarga del servidor de base de datos",
                "Experiencia de usuario deficiente",
                "Aumento de costos de infraestructura",
                "Timeouts de aplicación bajo carga"
            ],
            AntipatternType.MEMORY_LEAK: [
                "Degradación progresiva del rendimiento",
                "Crashes frecuentes de la aplicación",
                "Consumo excesivo de recursos del servidor",
                "Necesidad de reinicios frecuentes",
                "Inestabilidad del sistema"
            ]
        }
        
        return consequences.get(pattern.pattern_type, [
            "Reducción de la calidad del código",
            "Aumento de la carga de mantenimiento",
            "Bugs y problemas potenciales"
        ])
    
    async def _generate_fix_instructions(
        self, 
        pattern: DetectedPattern, 
        features: AntipatternFeatures
    ) -> str:
        """Generar instrucciones de solución."""
        
        fixes = {
            AntipatternType.GOD_OBJECT: "Para arreglar el God Object: 1) Identifica responsabilidades distintas dentro de la clase, 2) Extrae cada responsabilidad en una clase separada, 3) Usa composición o delegación para mantener funcionalidad, 4) Aplica el Principio de Responsabilidad Única, 5) Considera usar patrones como Strategy o Command para organizar el código refactorizado.",
            
            AntipatternType.SQL_INJECTION: "Para arreglar vulnerabilidades SQL Injection: 1) Usa consultas parametrizadas o prepared statements, 2) Implementa validación y sanitización de entrada, 3) Usa frameworks ORM que manejen parametrización automáticamente, 4) Aplica el principio de menor privilegio para acceso a base de datos, 5) Audita regularmente el código de construcción de consultas SQL.",
            
            AntipatternType.N_PLUS_ONE_QUERY: "Para arreglar consultas N+1: 1) Usa consultas JOIN para obtener datos relacionados en una sola consulta, 2) Implementa eager loading para entidades relacionadas, 3) Usa técnicas de batch loading, 4) Considera usar herramientas de optimización de consultas, 5) Añade monitoreo de consultas de base de datos para prevenir futuras ocurrencias.",
            
            AntipatternType.LARGE_CLASS: "Para arreglar clases grandes: 1) Identifica grupos de métodos relacionados, 2) Extrae grupos coherentes en clases separadas, 3) Usa composición para combinar las clases extraídas, 4) Aplica el principio de responsabilidad única, 5) Refactoriza gradualmente para mantener funcionalidad.",
            
            AntipatternType.SPAGHETTI_CODE: "Para arreglar código espagueti: 1) Identifica flujos de control principales, 2) Extrae lógica compleja en métodos pequeños y bien nombrados, 3) Reduce la profundidad de anidamiento, 4) Usa patrones como Strategy o State para simplificar lógica condicional, 5) Refactoriza incrementalmente con tests de regresión."
        }
        
        return fixes.get(
            pattern.pattern_type,
            "Considera refactorizar el código para seguir mejores prácticas y patrones de diseño. "
            "Revisa principios SOLID y patrones de diseño apropiados para tu situación específica."
        )
    
    def _explain_confidence(self, pattern: DetectedPattern) -> str:
        """Explicar el nivel de confianza."""
        
        confidence = pattern.confidence
        
        if confidence >= 0.9:
            return f"Confianza muy alta ({confidence:.1%}): Múltiples indicadores fuertes confirman la presencia de este antipatrón."
        elif confidence >= 0.7:
            return f"Confianza alta ({confidence:.1%}): Varios indicadores sugieren fuertemente la presencia de este antipatrón."
        elif confidence >= 0.5:
            return f"Confianza media ({confidence:.1%}): Algunos indicadores apuntan hacia este antipatrón, pero puede requerir revisión manual."
        else:
            return f"Confianza baja ({confidence:.1%}): Indicadores débiles sugieren posible presencia de este antipatrón."
    
    def _get_pattern_references(self, pattern_type: AntipatternType) -> List[str]:
        """Obtener referencias para el patrón."""
        
        references_db = {
            AntipatternType.GOD_OBJECT: [
                "Martin Fowler - Refactoring: Improving the Design of Existing Code",
                "SOLID Principles - Single Responsibility Principle",
                "Clean Code by Robert C. Martin - Chapter 10: Classes"
            ],
            AntipatternType.SQL_INJECTION: [
                "OWASP Top 10 - Injection Vulnerabilities",
                "NIST Secure Software Development Framework",
                "CWE-89: SQL Injection"
            ],
            AntipatternType.N_PLUS_ONE_QUERY: [
                "High Performance MySQL by Baron Schwartz",
                "Effective Java by Joshua Bloch - Item 67: Optimize judiciously",
                "Database Performance Tuning Handbook"
            ]
        }
        
        return references_db.get(pattern_type, [
            "Clean Code by Robert C. Martin",
            "Refactoring by Martin Fowler",
            "Design Patterns by Gang of Four"
        ])
    
    # Métodos de explicación específicos para cada antipatrón
    async def _explain_god_object(self, pattern: DetectedPattern, features: AntipatternFeatures) -> str:
        explanation = f"Se ha detectado un God Object en el archivo '{features.file_path.name}'. "
        
        if features.max_class_size > 0:
            explanation += f"La clase más grande tiene {features.max_class_size} líneas de código, "
        if features.methods_count > 0:
            explanation += f"contiene {features.methods_count} métodos, "
        if features.distinct_responsibilities > 1:
            explanation += f"y maneja {features.distinct_responsibilities} responsabilidades distintas. "
        
        explanation += "Esto viola el Principio de Responsabilidad Única y hace que la clase sea difícil de entender, probar y mantener."
        
        return explanation
    
    async def _explain_sql_injection(self, pattern: DetectedPattern, features: AntipatternFeatures) -> str:
        explanation = "Se ha detectado una potencial vulnerabilidad de SQL Injection. "
        explanation += "El código parece usar entrada del usuario directamente en consultas SQL sin parametrización adecuada. "
        explanation += "Esto permite que atacantes inyecten código SQL malicioso que puede leer, modificar o eliminar datos de la base de datos."
        
        return explanation
    
    async def _explain_n_plus_one(self, pattern: DetectedPattern, features: AntipatternFeatures) -> str:
        explanation = "Se ha detectado el antipatrón N+1 Query. "
        explanation += "Este patrón ocurre cuando el código ejecuta una consulta para obtener una lista de registros, "
        explanation += "y luego ejecuta N consultas adicionales para obtener datos relacionados para cada registro. "
        explanation += "Esto resulta en N+1 consultas de base de datos en lugar de una sola consulta optimizada."
        
        return explanation
    
    async def _explain_large_class(self, pattern: DetectedPattern, features: AntipatternFeatures) -> str:
        explanation = "Se ha detectado una clase demasiado grande. "
        
        if features.max_class_size > 0:
            explanation += f"La clase tiene {features.max_class_size} líneas de código, "
        if features.methods_count > 0:
            explanation += f"con {features.methods_count} métodos. "
        
        explanation += "Las clases grandes son difíciles de entender, mantener y probar eficazmente. "
        explanation += "Considera dividir la clase en componentes más pequeños y cohesivos."
        
        return explanation
    
    async def _explain_spaghetti_code(self, pattern: DetectedPattern, features: AntipatternFeatures) -> str:
        explanation = "Se ha detectado código espagueti con flujo de control complejo y enmarañado. "
        
        if features.cyclomatic_complexity > 0:
            explanation += f"La complejidad ciclomática es {features.cyclomatic_complexity:.1f}, "
        if features.nesting_depth > 0:
            explanation += f"con una profundidad de anidamiento de {features.nesting_depth} niveles. "
        
        explanation += "Este tipo de código es extremadamente difícil de seguir, entender y mantener."
        
        return explanation
    
    async def _explain_memory_leak(self, pattern: DetectedPattern, features: AntipatternFeatures) -> str:
        explanation = f"Se han detectado patrones que pueden causar memory leaks en {features.language.value}. "
        
        if features.has_recursive_calls:
            explanation += "Las llamadas recursivas sin límites apropiados pueden acumular memoria. "
        if features.has_loops:
            explanation += "Los bucles complejos pueden crear referencias circulares o no liberar recursos. "
        
        explanation += "Los memory leaks causan degradación progresiva del rendimiento y pueden llevar a crashes."
        
        return explanation
    
    async def _explain_hardcoded_secrets(self, pattern: DetectedPattern, features: AntipatternFeatures) -> str:
        explanation = "Se han detectado posibles secretos o credenciales hardcodeados en el código fuente. "
        explanation += "Esto es una grave vulnerabilidad de seguridad ya que los secretos pueden ser extraídos "
        explanation += "por cualquiera que tenga acceso al código. Los secretos deben almacenarse en variables de entorno "
        explanation += "o sistemas de gestión de secretos seguros."
        
        return explanation
    
    async def _explain_string_concat(self, pattern: DetectedPattern, features: AntipatternFeatures) -> str:
        explanation = "Se ha detectado concatenación de strings dentro de bucles. "
        explanation += "En muchos lenguajes, esto es ineficiente porque cada concatenación crea un nuevo objeto string, "
        explanation += "causando allocaciones de memoria innecesarias. Considera usar StringBuilder o técnicas similares."
        
        return explanation
    
    async def _explain_feature_envy(self, pattern: DetectedPattern, features: AntipatternFeatures) -> str:
        explanation = "Se ha detectado Feature Envy - una clase que usa excesivamente datos y métodos de otra clase. "
        
        if features.class_coupling > 0:
            explanation += f"El acoplamiento de clases es {features.class_coupling:.2f}, "
        if features.external_dependencies > 0:
            explanation += f"con {features.external_dependencies} dependencias externas. "
        
        explanation += "Esto sugiere que la funcionalidad podría estar en el lugar equivocado."
        
        return explanation
    
    async def _explain_long_method(self, pattern: DetectedPattern, features: AntipatternFeatures) -> str:
        explanation = "Se ha detectado un método demasiado largo. "
        
        if features.max_method_length > 0:
            explanation += f"El método más largo tiene {features.max_method_length} líneas. "
        
        explanation += "Los métodos largos son difíciles de entender, probar y mantener. "
        explanation += "Considera dividir el método en submétodos más pequeños y enfocados."
        
        return explanation
    
    # Métodos auxiliares
    def _get_confidence_level(self, confidence: float) -> str:
        """Convertir confianza numérica a texto."""
        if confidence >= 0.9:
            return "muy alta"
        elif confidence >= 0.7:
            return "alta"
        elif confidence >= 0.5:
            return "media"
        else:
            return "baja"
    
    def _get_pattern_display_name(self, pattern_type: AntipatternType) -> str:
        """Obtener nombre legible del patrón."""
        display_names = {
            AntipatternType.GOD_OBJECT: "God Object",
            AntipatternType.SQL_INJECTION: "SQL Injection",
            AntipatternType.N_PLUS_ONE_QUERY: "N+1 Query",
            AntipatternType.LARGE_CLASS: "Clase Grande",
            AntipatternType.LONG_METHOD: "Método Largo",
            AntipatternType.SPAGHETTI_CODE: "Código Espagueti",
            AntipatternType.MEMORY_LEAK: "Memory Leak",
            AntipatternType.HARDCODED_SECRETS: "Secretos Hardcodeados",
            AntipatternType.STRING_CONCATENATION_IN_LOOP: "Concatenación de Strings en Bucle",
            AntipatternType.FEATURE_ENVY: "Feature Envy",
            AntipatternType.BIG_BALL_OF_MUD: "Big Ball of Mud"
        }
        
        return display_names.get(pattern_type, pattern_type.value.replace('_', ' ').title())
    
    def _create_fallback_explanation(self, pattern: DetectedPattern) -> AntipatternExplanation:
        """Crear explicación básica en caso de error."""
        return AntipatternExplanation(
            pattern_type=pattern.pattern_type,
            summary=f"Antipatrón {self._get_pattern_display_name(pattern.pattern_type)} detectado.",
            detailed_explanation="Se ha detectado este antipatrón en el código.",
            why_its_problematic="Este patrón puede causar problemas de mantenibilidad.",
            potential_consequences=["Dificultad de mantenimiento", "Posibles bugs"],
            how_to_fix="Considera refactorizar siguiendo mejores prácticas.",
            confidence_explanation=f"Confianza: {pattern.confidence:.1%}"
        )


class ExplanationTemplateEngine:
    """Motor de plantillas para explicaciones."""
    
    def format_explanation(self, template: str, variables: Dict[str, str]) -> str:
        """Formatear explicación usando plantilla."""
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable in template: {e}")
            return template


class CodeExampleGenerator:
    """Generador de ejemplos de código."""
    
    async def generate_bad_example(
        self, 
        pattern_type: AntipatternType, 
        language: ProgrammingLanguage
    ) -> Optional[CodeExample]:
        """Generar ejemplo de código problemático."""
        
        examples = {
            (AntipatternType.GOD_OBJECT, ProgrammingLanguage.PYTHON): CodeExample(
                language=language,
                code="""class UserManager:
    def __init__(self):
        self.users = []
        self.emails = []
        self.logs = []
    
    def create_user(self, data): # Gestión de usuarios
        # 50+ líneas de validación
        pass
    
    def send_email(self, user): # Gestión de email
        # 30+ líneas de lógica de email
        pass
    
    def log_activity(self, action): # Logging
        # 20+ líneas de logging
        pass
    
    def calculate_analytics(self): # Analytics
        # 40+ líneas de cálculos
        pass
    
    def generate_reports(self): # Reports
        # 60+ líneas de reportes
        pass""",
                explanation="Esta clase maneja múltiples responsabilidades: gestión de usuarios, emails, logging, analytics y reportes.",
                highlights=[
                    CodeHighlight(2, 8, HighlightType.PROBLEM, "Múltiples responsabilidades en una sola clase"),
                    CodeHighlight(10, 12, HighlightType.PROBLEM, "Gestión de email debería estar separada"),
                    CodeHighlight(18, 20, HighlightType.PROBLEM, "Analytics no pertenece aquí")
                ]
            ),
            
            (AntipatternType.SQL_INJECTION, ProgrammingLanguage.PYTHON): CodeExample(
                language=language,
                code="""def get_user_by_id(user_id):
    # PELIGROSO: Concatenación directa de strings
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()

def search_users(name):
    # PELIGROSO: Entrada del usuario sin sanitizar
    query = "SELECT * FROM users WHERE name = '" + name + "'"
    return database.query(query)""",
                explanation="Estos métodos son vulnerables a SQL injection porque concatenan entrada del usuario directamente.",
                highlights=[
                    CodeHighlight(3, 3, HighlightType.PROBLEM, "Concatenación directa - vulnerable a injection"),
                    CodeHighlight(8, 8, HighlightType.PROBLEM, "Entrada sin sanitizar")
                ]
            )
        }
        
        return examples.get((pattern_type, language))
    
    async def generate_good_example(
        self, 
        pattern_type: AntipatternType, 
        language: ProgrammingLanguage
    ) -> Optional[CodeExample]:
        """Generar ejemplo de código mejorado."""
        
        examples = {
            (AntipatternType.GOD_OBJECT, ProgrammingLanguage.PYTHON): CodeExample(
                language=language,
                code="""class User:
    def __init__(self, data):
        self.data = data

class UserRepository:
    def create_user(self, user_data):
        # Lógica específica de persistencia
        pass

class EmailService:
    def send_welcome_email(self, user):
        # Lógica específica de email
        pass

class UserAnalytics:
    def track_user_creation(self, user):
        # Lógica específica de analytics
        pass

class UserManager:
    def __init__(self):
        self.repository = UserRepository()
        self.email_service = EmailService()
        self.analytics = UserAnalytics()
    
    def register_user(self, user_data):
        user = self.repository.create_user(user_data)
        self.email_service.send_welcome_email(user)
        self.analytics.track_user_creation(user)
        return user""",
                explanation="Cada clase tiene una responsabilidad específica y el UserManager coordina las operaciones.",
                highlights=[
                    CodeHighlight(5, 8, HighlightType.SOLUTION, "Responsabilidad única: persistencia"),
                    CodeHighlight(10, 13, HighlightType.SOLUTION, "Responsabilidad única: emails"),
                    CodeHighlight(20, 26, HighlightType.SOLUTION, "Coordinación sin lógica compleja")
                ]
            ),
            
            (AntipatternType.SQL_INJECTION, ProgrammingLanguage.PYTHON): CodeExample(
                language=language,
                code="""def get_user_by_id(user_id):
    # SEGURO: Usando consulta parametrizada
    query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()

def search_users(name):
    # SEGURO: Parámetros validados y sanitizados
    if not isinstance(name, str) or len(name) > 100:
        raise ValueError("Invalid name parameter")
    
    query = "SELECT * FROM users WHERE name = %s"
    return database.query(query, (name,))""",
                explanation="Estos métodos usan consultas parametrizadas y validación de entrada para prevenir SQL injection.",
                highlights=[
                    CodeHighlight(3, 4, HighlightType.SOLUTION, "Consulta parametrizada segura"),
                    CodeHighlight(9, 10, HighlightType.SOLUTION, "Validación de entrada"),
                    CodeHighlight(12, 13, HighlightType.SOLUTION, "Parámetros seguros")
                ]
            )
        }
        
        return examples.get((pattern_type, language))


class ImpactExplanationAnalyzer:
    """Analizador de impacto para explicaciones."""
    
    def analyze_business_impact(self, pattern_type: AntipatternType) -> str:
        """Analizar impacto en el negocio."""
        
        impacts = {
            AntipatternType.SQL_INJECTION: "Riesgo crítico de seguridad que puede resultar en filtraciones de datos y pérdidas financieras significativas.",
            AntipatternType.N_PLUS_ONE_QUERY: "Degradación del rendimiento que afecta la experiencia del usuario y puede requerir recursos adicionales de infraestructura.",
            AntipatternType.GOD_OBJECT: "Aumenta significativamente el costo y tiempo de desarrollo de nuevas características."
        }
        
        return impacts.get(pattern_type, "Puede impactar negativamente la productividad del desarrollo y la calidad del software.")
    
    def analyze_technical_impact(self, pattern_type: AntipatternType) -> str:
        """Analizar impacto técnico."""
        
        impacts = {
            AntipatternType.GOD_OBJECT: "Alta complejidad que dificulta el testing, debugging y mantenimiento del código.",
            AntipatternType.SPAGHETTI_CODE: "Código extremadamente difícil de entender, modificar y debuggear.",
            AntipatternType.MEMORY_LEAK: "Degradación progresiva del rendimiento que puede causar crashes de la aplicación."
        }
        
        return impacts.get(pattern_type, "Impacta negativamente la arquitectura y calidad técnica del software.")
