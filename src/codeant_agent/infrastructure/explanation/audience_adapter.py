"""
Implementación del Adaptador de Audiencia.

Este módulo implementa la adaptación de contenido para diferentes audiencias,
ajustando el tono, nivel técnico y enfoque según el perfil del usuario.
"""

import logging
from typing import Dict, List, Any, Optional
from uuid import uuid4

from ...domain.entities.explanation import Language, Audience
from ..ports.explanation_ports import AudienceAdapterPort


logger = logging.getLogger(__name__)


class AudienceAdapter(AudienceAdapterPort):
    """Adaptador para diferentes audiencias."""
    
    def __init__(self):
        self.audience_characteristics = self._initialize_audience_characteristics()
        self.adaptation_strategies = self._initialize_adaptation_strategies()
        self.terminology_levels = self._initialize_terminology_levels()
    
    def _initialize_audience_characteristics(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar características de cada audiencia."""
        return {
            "junior_developer": {
                "experience_level": "beginner",
                "technical_depth": "basic",
                "preferred_tone": "friendly_educational",
                "focus_areas": ["learning", "examples", "step_by_step"],
                "avoid_terms": ["enterprise", "scalability", "microservices"],
                "preferred_explanations": ["what", "why", "how"],
                "time_attention": "high",
                "complexity_threshold": "low"
            },
            "senior_developer": {
                "experience_level": "advanced",
                "technical_depth": "detailed",
                "preferred_tone": "professional_technical",
                "focus_areas": ["architecture", "best_practices", "performance"],
                "avoid_terms": [],
                "preferred_explanations": ["technical_details", "alternatives", "trade_offs"],
                "time_attention": "medium",
                "complexity_threshold": "high"
            },
            "technical_lead": {
                "experience_level": "expert",
                "technical_depth": "comprehensive",
                "preferred_tone": "authoritative_strategic",
                "focus_areas": ["architecture", "team_impact", "long_term"],
                "avoid_terms": [],
                "preferred_explanations": ["strategic_impact", "team_guidance", "architectural_decisions"],
                "time_attention": "low",
                "complexity_threshold": "very_high"
            },
            "software_architect": {
                "experience_level": "expert",
                "technical_depth": "architectural",
                "preferred_tone": "strategic_analytical",
                "focus_areas": ["system_design", "patterns", "scalability"],
                "avoid_terms": [],
                "preferred_explanations": ["architectural_impact", "design_patterns", "system_considerations"],
                "time_attention": "low",
                "complexity_threshold": "very_high"
            },
            "project_manager": {
                "experience_level": "intermediate",
                "technical_depth": "business_focused",
                "preferred_tone": "business_professional",
                "focus_areas": ["timeline", "cost", "risk", "team_impact"],
                "avoid_terms": ["cyclomatic_complexity", "dependency_injection", "design_patterns"],
                "preferred_explanations": ["business_impact", "cost_implications", "timeline_effects"],
                "time_attention": "medium",
                "complexity_threshold": "low"
            },
            "quality_assurance": {
                "experience_level": "intermediate",
                "technical_depth": "testing_focused",
                "preferred_tone": "analytical_detailed",
                "focus_areas": ["testing", "quality_metrics", "reproducibility"],
                "avoid_terms": [],
                "preferred_explanations": ["testing_implications", "quality_impact", "reproducibility"],
                "time_attention": "high",
                "complexity_threshold": "medium"
            },
            "security_team": {
                "experience_level": "advanced",
                "technical_depth": "security_focused",
                "preferred_tone": "security_conscious",
                "focus_areas": ["security", "vulnerabilities", "compliance"],
                "avoid_terms": [],
                "preferred_explanations": ["security_implications", "vulnerability_analysis", "compliance_impact"],
                "time_attention": "high",
                "complexity_threshold": "high"
            },
            "business_stakeholder": {
                "experience_level": "beginner",
                "technical_depth": "non_technical",
                "preferred_tone": "business_friendly",
                "focus_areas": ["business_value", "user_impact", "roi"],
                "avoid_terms": ["code", "function", "class", "method", "variable", "algorithm"],
                "preferred_explanations": ["business_value", "user_impact", "cost_benefit"],
                "time_attention": "low",
                "complexity_threshold": "very_low"
            }
        }
    
    def _initialize_adaptation_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar estrategias de adaptación."""
        return {
            "terminology": {
                "junior_developer": "simplify_and_explain",
                "senior_developer": "use_technical_terms",
                "technical_lead": "use_technical_terms_with_context",
                "software_architect": "use_architectural_terminology",
                "project_manager": "translate_to_business_terms",
                "quality_assurance": "use_quality_terminology",
                "security_team": "use_security_terminology",
                "business_stakeholder": "avoid_technical_terms"
            },
            "tone": {
                "junior_developer": "encouraging_and_supportive",
                "senior_developer": "professional_and_direct",
                "technical_lead": "authoritative_and_guidance",
                "software_architect": "analytical_and_strategic",
                "project_manager": "business_focused_and_practical",
                "quality_assurance": "detailed_and_analytical",
                "security_team": "security_conscious_and_urgent",
                "business_stakeholder": "business_friendly_and_clear"
            },
            "structure": {
                "junior_developer": "step_by_step_with_examples",
                "senior_developer": "technical_details_with_alternatives",
                "technical_lead": "strategic_overview_with_guidance",
                "software_architect": "architectural_analysis_with_patterns",
                "project_manager": "business_impact_with_timeline",
                "quality_assurance": "quality_analysis_with_testing",
                "security_team": "security_analysis_with_implications",
                "business_stakeholder": "business_value_with_roi"
            }
        }
    
    def _initialize_terminology_levels(self) -> Dict[str, Dict[str, str]]:
        """Inicializar niveles de terminología."""
        return {
            "technical_to_business": {
                "cyclomatic_complexity": "code complexity",
                "code_smell": "code quality issue",
                "technical_debt": "maintenance backlog",
                "refactoring": "code improvement",
                "unit_test": "automated test",
                "integration_test": "system test",
                "code_coverage": "test coverage",
                "static_analysis": "code review",
                "dependency_injection": "component connection",
                "design_pattern": "coding approach",
                "antipattern": "problematic approach",
                "maintainability": "ease of maintenance",
                "readability": "code clarity",
                "performance": "speed and efficiency",
                "security_vulnerability": "security risk"
            },
            "business_to_technical": {
                "code complexity": "cyclomatic complexity",
                "code quality issue": "code smell",
                "maintenance backlog": "technical debt",
                "code improvement": "refactoring",
                "automated test": "unit test",
                "system test": "integration test",
                "test coverage": "code coverage",
                "code review": "static analysis",
                "component connection": "dependency injection",
                "coding approach": "design pattern",
                "problematic approach": "antipattern",
                "ease of maintenance": "maintainability",
                "code clarity": "readability",
                "speed and efficiency": "performance",
                "security risk": "security vulnerability"
            }
        }
    
    async def adapt_for_audience(
        self, 
        content: str, 
        audience: Audience,
        language: Language
    ) -> str:
        """Adaptar contenido para audiencia específica."""
        try:
            audience_key = audience.value
            characteristics = self.audience_characteristics.get(audience_key, {})
            
            if not characteristics:
                logger.warning(f"Características no encontradas para audiencia: {audience_key}")
                return content
            
            # Aplicar adaptaciones
            adapted_content = content
            
            # 1. Adaptar terminología
            adapted_content = await self._adapt_terminology(adapted_content, audience_key, language)
            
            # 2. Adaptar tono
            adapted_content = await self._adapt_tone(adapted_content, audience_key, language)
            
            # 3. Adaptar estructura
            adapted_content = await self._adapt_structure(adapted_content, audience_key, language)
            
            # 4. Adaptar nivel de detalle
            adapted_content = await self._adapt_detail_level(adapted_content, audience_key, language)
            
            # 5. Agregar contexto específico de audiencia
            adapted_content = await self._add_audience_context(adapted_content, audience_key, language)
            
            logger.info(f"Contenido adaptado para audiencia: {audience_key}")
            return adapted_content
            
        except Exception as e:
            logger.error(f"Error adaptando contenido para audiencia: {str(e)}")
            return content
    
    async def get_audience_characteristics(
        self, 
        audience: Audience
    ) -> Dict[str, Any]:
        """Obtener características de la audiencia."""
        audience_key = audience.value
        return self.audience_characteristics.get(audience_key, {})
    
    async def _adapt_terminology(
        self, 
        content: str, 
        audience_key: str, 
        language: Language
    ) -> str:
        """Adaptar terminología según audiencia."""
        strategy = self.adaptation_strategies["terminology"].get(audience_key, "use_technical_terms")
        
        if strategy == "simplify_and_explain":
            return await self._simplify_terminology(content, language)
        elif strategy == "translate_to_business_terms":
            return await self._translate_to_business_terms(content, language)
        elif strategy == "avoid_technical_terms":
            return await self._avoid_technical_terms(content, language)
        else:
            return content  # Mantener terminología técnica
    
    async def _simplify_terminology(self, content: str, language: Language) -> str:
        """Simplificar terminología técnica."""
        if language == Language.SPANISH:
            simplifications = {
                "complejidad ciclomática": "complejidad del código (mide qué tan complejo es el código)",
                "refactorización": "mejorar el código sin cambiar lo que hace",
                "prueba unitaria": "prueba que verifica una pequeña parte del código",
                "análisis estático": "revisión automática del código",
                "inyección de dependencias": "forma de conectar diferentes partes del código",
                "patrón de diseño": "forma estándar de resolver un problema común",
                "antipatrón": "forma problemática de resolver un problema",
                "mantenibilidad": "qué tan fácil es mantener y modificar el código",
                "legibilidad": "qué tan fácil es leer y entender el código"
            }
        else:
            simplifications = {
                "cyclomatic complexity": "code complexity (measures how complex the code is)",
                "refactoring": "improving code without changing what it does",
                "unit test": "test that verifies a small part of the code",
                "static analysis": "automatic code review",
                "dependency injection": "way of connecting different parts of code",
                "design pattern": "standard way of solving a common problem",
                "antipattern": "problematic way of solving a problem",
                "maintainability": "how easy it is to maintain and modify code",
                "readability": "how easy it is to read and understand code"
            }
        
        adapted_content = content
        for technical, simple in simplifications.items():
            adapted_content = adapted_content.replace(technical, simple)
        
        return adapted_content
    
    async def _translate_to_business_terms(self, content: str, language: Language) -> str:
        """Traducir términos técnicos a términos de negocio."""
        if language == Language.SPANISH:
            translations = {
                "complejidad ciclomática": "complejidad del código",
                "código con mal olor": "problema de calidad de código",
                "deuda técnica": "acumulación de problemas de mantenimiento",
                "refactorización": "mejora del código",
                "prueba unitaria": "prueba automatizada",
                "prueba de integración": "prueba del sistema",
                "cobertura de código": "cobertura de pruebas",
                "análisis estático": "revisión de código",
                "inyección de dependencias": "conexión de componentes",
                "patrón de diseño": "enfoque de programación",
                "antipatrón": "enfoque problemático",
                "mantenibilidad": "facilidad de mantenimiento",
                "legibilidad": "claridad del código",
                "rendimiento": "velocidad y eficiencia",
                "vulnerabilidad de seguridad": "riesgo de seguridad"
            }
        else:
            translations = {
                "cyclomatic complexity": "code complexity",
                "code smell": "code quality issue",
                "technical debt": "maintenance backlog",
                "refactoring": "code improvement",
                "unit test": "automated test",
                "integration test": "system test",
                "code coverage": "test coverage",
                "static analysis": "code review",
                "dependency injection": "component connection",
                "design pattern": "coding approach",
                "antipattern": "problematic approach",
                "maintainability": "ease of maintenance",
                "readability": "code clarity",
                "performance": "speed and efficiency",
                "security vulnerability": "security risk"
            }
        
        adapted_content = content
        for technical, business in translations.items():
            adapted_content = adapted_content.replace(technical, business)
        
        return adapted_content
    
    async def _avoid_technical_terms(self, content: str, language: Language) -> str:
        """Evitar términos técnicos completamente."""
        if language == Language.SPANISH:
            replacements = {
                "código": "sistema",
                "función": "proceso",
                "clase": "componente",
                "método": "proceso",
                "variable": "dato",
                "algoritmo": "proceso",
                "programación": "desarrollo",
                "desarrollador": "especialista",
                "programador": "especialista",
                "software": "sistema",
                "aplicación": "sistema",
                "base de datos": "almacén de información",
                "API": "interfaz de conexión",
                "framework": "herramienta de desarrollo",
                "librería": "conjunto de herramientas"
            }
        else:
            replacements = {
                "code": "system",
                "function": "process",
                "class": "component",
                "method": "process",
                "variable": "data",
                "algorithm": "process",
                "programming": "development",
                "developer": "specialist",
                "programmer": "specialist",
                "software": "system",
                "application": "system",
                "database": "information store",
                "API": "connection interface",
                "framework": "development tool",
                "library": "tool set"
            }
        
        adapted_content = content
        for technical, business in replacements.items():
            adapted_content = adapted_content.replace(technical, business)
        
        return adapted_content
    
    async def _adapt_tone(
        self, 
        content: str, 
        audience_key: str, 
        language: Language
    ) -> str:
        """Adaptar tono según audiencia."""
        tone_strategy = self.adaptation_strategies["tone"].get(audience_key, "professional")
        
        if tone_strategy == "encouraging_and_supportive":
            return await self._make_encouraging_tone(content, language)
        elif tone_strategy == "business_focused_and_practical":
            return await self._make_business_tone(content, language)
        elif tone_strategy == "security_conscious_and_urgent":
            return await self._make_security_tone(content, language)
        else:
            return content  # Mantener tono profesional
    
    async def _make_encouraging_tone(self, content: str, language: Language) -> str:
        """Hacer el tono más alentador y de apoyo."""
        if language == Language.SPANISH:
            encouraging_phrases = {
                "El problema es": "¡Hola! El problema que encontramos es",
                "Se debe": "Te recomendamos que",
                "Es necesario": "Sería genial si pudieras",
                "Error": "Área de mejora",
                "Problema": "Oportunidad de aprendizaje",
                "Crítico": "Importante de abordar",
                "Mal": "Puede mejorarse"
            }
        else:
            encouraging_phrases = {
                "The issue is": "Hello! The issue we found is",
                "It should": "We recommend that you",
                "It is necessary": "It would be great if you could",
                "Error": "Area for improvement",
                "Problem": "Learning opportunity",
                "Critical": "Important to address",
                "Bad": "Can be improved"
            }
        
        adapted_content = content
        for formal, encouraging in encouraging_phrases.items():
            adapted_content = adapted_content.replace(formal, encouraging)
        
        return adapted_content
    
    async def _make_business_tone(self, content: str, language: Language) -> str:
        """Hacer el tono orientado al negocio."""
        if language == Language.SPANISH:
            business_phrases = {
                "El código": "El sistema",
                "desarrollador": "especialista",
                "programación": "desarrollo",
                "técnico": "operacional",
                "implementación": "despliegue",
                "funcionalidad": "característica del producto",
                "rendimiento": "eficiencia operacional"
            }
        else:
            business_phrases = {
                "The code": "The system",
                "developer": "specialist",
                "programming": "development",
                "technical": "operational",
                "implementation": "deployment",
                "functionality": "product feature",
                "performance": "operational efficiency"
            }
        
        adapted_content = content
        for technical, business in business_phrases.items():
            adapted_content = adapted_content.replace(technical, business)
        
        return adapted_content
    
    async def _make_security_tone(self, content: str, language: Language) -> str:
        """Hacer el tono consciente de la seguridad."""
        if language == Language.SPANISH:
            security_phrases = {
                "problema": "vulnerabilidad potencial",
                "error": "riesgo de seguridad",
                "crítico": "alto riesgo de seguridad",
                "importante": "requiere atención de seguridad",
                "recomendado": "recomendación de seguridad"
            }
        else:
            security_phrases = {
                "issue": "potential vulnerability",
                "error": "security risk",
                "critical": "high security risk",
                "important": "requires security attention",
                "recommended": "security recommendation"
            }
        
        adapted_content = content
        for general, security in security_phrases.items():
            adapted_content = adapted_content.replace(general, security)
        
        return adapted_content
    
    async def _adapt_structure(
        self, 
        content: str, 
        audience_key: str, 
        language: Language
    ) -> str:
        """Adaptar estructura del contenido."""
        structure_strategy = self.adaptation_strategies["structure"].get(audience_key, "standard")
        
        if structure_strategy == "step_by_step_with_examples":
            return await self._add_step_by_step_structure(content, language)
        elif structure_strategy == "business_impact_with_timeline":
            return await self._add_business_impact_structure(content, language)
        elif structure_strategy == "security_analysis_with_implications":
            return await self._add_security_analysis_structure(content, language)
        else:
            return content  # Mantener estructura estándar
    
    async def _add_step_by_step_structure(self, content: str, language: Language) -> str:
        """Agregar estructura paso a paso."""
        if language == Language.SPANISH:
            step_structure = (
                "\n\n**Pasos para solucionarlo:**\n"
                "1. **Identifica** el problema específico\n"
                "2. **Entiende** por qué ocurre\n"
                "3. **Aplica** la solución recomendada\n"
                "4. **Verifica** que funciona correctamente\n"
                "5. **Aprende** de esta experiencia para el futuro"
            )
        else:
            step_structure = (
                "\n\n**Steps to fix it:**\n"
                "1. **Identify** the specific problem\n"
                "2. **Understand** why it happens\n"
                "3. **Apply** the recommended solution\n"
                "4. **Verify** that it works correctly\n"
                "5. **Learn** from this experience for the future"
            )
        
        return content + step_structure
    
    async def _add_business_impact_structure(self, content: str, language: Language) -> str:
        """Agregar estructura de impacto de negocio."""
        if language == Language.SPANISH:
            business_structure = (
                "\n\n**Impacto en el Negocio:**\n"
                "• **Tiempo estimado de corrección:** 2-4 horas\n"
                "• **Impacto en el cronograma:** Mínimo\n"
                "• **Costo estimado:** $200-400\n"
                "• **ROI de la corrección:** Alto\n"
                "• **Riesgo si no se corrige:** Medio"
            )
        else:
            business_structure = (
                "\n\n**Business Impact:**\n"
                "• **Estimated correction time:** 2-4 hours\n"
                "• **Schedule impact:** Minimal\n"
                "• **Estimated cost:** $200-400\n"
                "• **Correction ROI:** High\n"
                "• **Risk if not corrected:** Medium"
            )
        
        return content + business_structure
    
    async def _add_security_analysis_structure(self, content: str, language: Language) -> str:
        """Agregar estructura de análisis de seguridad."""
        if language == Language.SPANISH:
            security_structure = (
                "\n\n**Análisis de Seguridad:**\n"
                "• **Nivel de riesgo:** Medio\n"
                "• **Vectores de ataque:** 2 identificados\n"
                "• **Impacto potencial:** Acceso no autorizado\n"
                "• **Recomendación:** Revisión inmediata\n"
                "• **Cumplimiento:** Verificar estándares aplicables"
            )
        else:
            security_structure = (
                "\n\n**Security Analysis:**\n"
                "• **Risk level:** Medium\n"
                "• **Attack vectors:** 2 identified\n"
                "• **Potential impact:** Unauthorized access\n"
                "• **Recommendation:** Immediate review\n"
                "• **Compliance:** Verify applicable standards"
            )
        
        return content + security_structure
    
    async def _adapt_detail_level(
        self, 
        content: str, 
        audience_key: str, 
        language: Language
    ) -> str:
        """Adaptar nivel de detalle según audiencia."""
        characteristics = self.audience_characteristics.get(audience_key, {})
        complexity_threshold = characteristics.get("complexity_threshold", "medium")
        
        if complexity_threshold == "very_low":
            return await self._simplify_to_basic_level(content, language)
        elif complexity_threshold == "low":
            return await self._simplify_to_intermediate_level(content, language)
        elif complexity_threshold == "very_high":
            return await self._add_advanced_details(content, language)
        else:
            return content  # Mantener nivel estándar
    
    async def _simplify_to_basic_level(self, content: str, language: Language) -> str:
        """Simplificar a nivel básico."""
        # Remover detalles técnicos complejos
        if language == Language.SPANISH:
            simplifications = {
                "arquitectura": "estructura",
                "patrón de diseño": "forma de hacer las cosas",
                "inyección de dependencias": "conexión de partes",
                "complejidad ciclomática": "complejidad",
                "análisis estático": "revisión automática"
            }
        else:
            simplifications = {
                "architecture": "structure",
                "design pattern": "way of doing things",
                "dependency injection": "connecting parts",
                "cyclomatic complexity": "complexity",
                "static analysis": "automatic review"
            }
        
        adapted_content = content
        for complex, simple in simplifications.items():
            adapted_content = adapted_content.replace(complex, simple)
        
        return adapted_content
    
    async def _simplify_to_intermediate_level(self, content: str, language: Language) -> str:
        """Simplificar a nivel intermedio."""
        # Mantener algunos detalles técnicos pero explicarlos
        return content  # Por ahora mantener como está
    
    async def _add_advanced_details(self, content: str, language: Language) -> str:
        """Agregar detalles avanzados."""
        if language == Language.SPANISH:
            advanced_details = (
                "\n\n**Detalles Técnicos Avanzados:**\n"
                "• **Consideraciones arquitectónicas:** Impacto en la arquitectura general\n"
                "• **Patrones aplicables:** Posibles patrones de diseño relevantes\n"
                "• **Alternativas de implementación:** Diferentes enfoques posibles\n"
                "• **Trade-offs:** Ventajas y desventajas de cada opción\n"
                "• **Escalabilidad:** Consideraciones para crecimiento futuro"
            )
        else:
            advanced_details = (
                "\n\n**Advanced Technical Details:**\n"
                "• **Architectural considerations:** Impact on overall architecture\n"
                "• **Applicable patterns:** Possible relevant design patterns\n"
                "• **Implementation alternatives:** Different possible approaches\n"
                "• **Trade-offs:** Advantages and disadvantages of each option\n"
                "• **Scalability:** Considerations for future growth"
            )
        
        return content + advanced_details
    
    async def _add_audience_context(
        self, 
        content: str, 
        audience_key: str, 
        language: Language
    ) -> str:
        """Agregar contexto específico de audiencia."""
        characteristics = self.audience_characteristics.get(audience_key, {})
        focus_areas = characteristics.get("focus_areas", [])
        
        if "learning" in focus_areas:
            content = await self._add_learning_context(content, language)
        elif "business_impact" in focus_areas:
            content = await self._add_business_context(content, language)
        elif "security" in focus_areas:
            content = await self._add_security_context(content, language)
        
        return content
    
    async def _add_learning_context(self, content: str, language: Language) -> str:
        """Agregar contexto de aprendizaje."""
        if language == Language.SPANISH:
            learning_context = (
                "\n\n**💡 Consejo de Aprendizaje:**\n"
                "Este es un concepto importante en programación. "
                "Te recomendamos practicar con ejemplos similares para consolidar tu comprensión."
            )
        else:
            learning_context = (
                "\n\n**💡 Learning Tip:**\n"
                "This is an important concept in programming. "
                "We recommend practicing with similar examples to consolidate your understanding."
            )
        
        return content + learning_context
    
    async def _add_business_context(self, content: str, language: Language) -> str:
        """Agregar contexto de negocio."""
        if language == Language.SPANISH:
            business_context = (
                "\n\n**📊 Impacto en el Negocio:**\n"
                "Abordar este problema mejorará la eficiencia del equipo y reducirá los costos de mantenimiento a largo plazo."
            )
        else:
            business_context = (
                "\n\n**📊 Business Impact:**\n"
                "Addressing this issue will improve team efficiency and reduce long-term maintenance costs."
            )
        
        return content + business_context
    
    async def _add_security_context(self, content: str, language: Language) -> str:
        """Agregar contexto de seguridad."""
        if language == Language.SPANISH:
            security_context = (
                "\n\n**🔒 Consideraciones de Seguridad:**\n"
                "Este problema puede crear vectores de ataque. Se recomienda una revisión inmediata por el equipo de seguridad."
            )
        else:
            security_context = (
                "\n\n**🔒 Security Considerations:**\n"
                "This issue can create attack vectors. Immediate review by the security team is recommended."
            )
        
        return content + security_context
