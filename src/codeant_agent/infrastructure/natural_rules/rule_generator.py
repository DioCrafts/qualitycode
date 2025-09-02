"""
Módulo que implementa el generador de reglas ejecutables para el sistema de reglas en lenguaje natural.
"""
import time
import uuid
from typing import Dict, List, Optional

from codeant_agent.application.ports.natural_rules.rule_generation_ports import RuleGeneratorPort
from codeant_agent.domain.entities.natural_rules.natural_rule import (
    ExecutableRule, ExecutableRuleId, RuleImplementation, RuleStructure
)
from codeant_agent.domain.entities.natural_rules.rule_intent import (
    ActionSeverity, RuleDomain, RuleIntent, RuleType
)


class RuleGenerator(RuleGeneratorPort):
    """Implementación del generador de reglas."""
    
    def __init__(self, code_generator, rule_template_engine):
        """Inicializa el generador de reglas.
        
        Args:
            code_generator: Generador de código
            rule_template_engine: Motor de plantillas de reglas
        """
        self.code_generator = code_generator
        self.rule_template_engine = rule_template_engine
        
        # Estrategias de implementación
        self.implementation_strategies = {
            RuleType.CONSTRAINT: "pattern",
            RuleType.QUALITY: "query",
            RuleType.PATTERN: "pattern",
            RuleType.METRIC: "procedural",
            RuleType.VALIDATION: "pattern",
            RuleType.TRANSFORMATION: "procedural",
            RuleType.DETECTION: "hybrid",
        }
    
    async def generate_rule(
        self, rule_structure: RuleStructure, context: Dict[str, str] = None
    ) -> ExecutableRule:
        """Genera una regla ejecutable a partir de una estructura.
        
        Args:
            rule_structure: Estructura de la regla
            context: Contexto adicional para la generación
            
        Returns:
            Regla ejecutable generada
        """
        # Determinar estrategia de implementación
        implementation_strategy = await self.determine_implementation_strategy(rule_structure)
        
        # Generar regla según la estrategia
        if implementation_strategy == "pattern":
            return await self._generate_pattern_rule(rule_structure, context)
        elif implementation_strategy == "query":
            return await self._generate_query_rule(rule_structure, context)
        elif implementation_strategy == "procedural":
            return await self._generate_procedural_rule(rule_structure, context)
        else:  # hybrid
            return await self._generate_hybrid_rule(rule_structure, context)
    
    async def determine_implementation_strategy(
        self, rule_structure: RuleStructure
    ) -> str:
        """Determina la mejor estrategia de implementación para una regla.
        
        Args:
            rule_structure: Estructura de la regla
            
        Returns:
            Nombre de la estrategia de implementación
        """
        # Determinar estrategia según el tipo de regla
        if rule_structure.rule_type in self.implementation_strategies:
            return self.implementation_strategies[rule_structure.rule_type]
        
        # Estrategia por defecto
        return "pattern"
    
    async def _generate_pattern_rule(
        self, rule_structure: RuleStructure, context: Dict[str, str] = None
    ) -> ExecutableRule:
        """Genera una regla basada en patrones.
        
        Args:
            rule_structure: Estructura de la regla
            context: Contexto adicional para la generación
            
        Returns:
            Regla ejecutable generada
        """
        # Generar código para la regla
        code = await self._generate_pattern_code(rule_structure)
        
        # Crear regla ejecutable
        return await self._create_executable_rule(
            rule_structure, code, "pattern", context
        )
    
    async def _generate_query_rule(
        self, rule_structure: RuleStructure, context: Dict[str, str] = None
    ) -> ExecutableRule:
        """Genera una regla basada en consultas.
        
        Args:
            rule_structure: Estructura de la regla
            context: Contexto adicional para la generación
            
        Returns:
            Regla ejecutable generada
        """
        # Generar código para la regla
        code = await self._generate_query_code(rule_structure)
        
        # Crear regla ejecutable
        return await self._create_executable_rule(
            rule_structure, code, "query", context
        )
    
    async def _generate_procedural_rule(
        self, rule_structure: RuleStructure, context: Dict[str, str] = None
    ) -> ExecutableRule:
        """Genera una regla procedimental.
        
        Args:
            rule_structure: Estructura de la regla
            context: Contexto adicional para la generación
            
        Returns:
            Regla ejecutable generada
        """
        # Generar código para la regla
        code = await self._generate_procedural_code(rule_structure)
        
        # Crear regla ejecutable
        return await self._create_executable_rule(
            rule_structure, code, "procedural", context
        )
    
    async def _generate_hybrid_rule(
        self, rule_structure: RuleStructure, context: Dict[str, str] = None
    ) -> ExecutableRule:
        """Genera una regla híbrida.
        
        Args:
            rule_structure: Estructura de la regla
            context: Contexto adicional para la generación
            
        Returns:
            Regla ejecutable generada
        """
        # Generar código para la regla
        code = await self._generate_hybrid_code(rule_structure)
        
        # Crear regla ejecutable
        return await self._create_executable_rule(
            rule_structure, code, "hybrid", context
        )
    
    async def _generate_pattern_code(self, rule_structure: RuleStructure) -> str:
        """Genera código para una regla basada en patrones.
        
        Args:
            rule_structure: Estructura de la regla
            
        Returns:
            Código generado
        """
        # Obtener plantilla según la intención primaria
        template_name = self._get_template_name_for_intent(
            rule_structure.intent_analysis.primary_intent
        )
        
        # Obtener plantilla
        template = await self.rule_template_engine.get_template(template_name)
        
        if not template:
            # Plantilla por defecto si no se encuentra la específica
            template = """
def analyze_{rule_name}(ast):
    violations = []
    
    # Find elements matching the pattern
    elements = find_elements(ast, "{element_type}")
    
    for element in elements:
        if check_condition(element, "{condition}"):
            violations.append(create_violation(
                element,
                "{message}",
                "{severity}"
            ))
    
    return violations
"""
        
        # Construir variables para la plantilla
        variables = await self._build_template_variables(rule_structure)
        
        # Renderizar plantilla
        return await self.rule_template_engine.render_template(template, variables)
    
    async def _generate_query_code(self, rule_structure: RuleStructure) -> str:
        """Genera código para una regla basada en consultas.
        
        Args:
            rule_structure: Estructura de la regla
            
        Returns:
            Código generado
        """
        # Plantilla para reglas basadas en consultas
        template = """
def analyze_{rule_name}(ast):
    violations = []
    
    # Execute query
    results = execute_query(ast, "{query}")
    
    for result in results:
        violations.append(create_violation(
            result,
            "{message}",
            "{severity}"
        ))
    
    return violations
"""
        
        # Construir variables para la plantilla
        variables = await self._build_template_variables(rule_structure)
        
        # Generar consulta
        variables["query"] = await self._generate_query(rule_structure)
        
        # Renderizar plantilla
        return await self.rule_template_engine.render_template(template, variables)
    
    async def _generate_procedural_code(self, rule_structure: RuleStructure) -> str:
        """Genera código para una regla procedimental.
        
        Args:
            rule_structure: Estructura de la regla
            
        Returns:
            Código generado
        """
        # Obtener plantilla según la intención primaria
        template_name = f"procedural_{self._get_template_name_for_intent(rule_structure.intent_analysis.primary_intent)}"
        
        # Obtener plantilla
        template = await self.rule_template_engine.get_template(template_name)
        
        if not template:
            # Plantilla por defecto si no se encuentra la específica
            template = """
def analyze_{rule_name}(ast):
    violations = []
    
    # Find target elements
    elements = find_elements(ast, "{element_type}")
    
    for element in elements:
        # Analyze element
        result = analyze_element(element, {threshold})
        
        if not result.is_compliant:
            violations.append(create_violation(
                element,
                "{message}",
                "{severity}"
            ))
    
    return violations
"""
        
        # Construir variables para la plantilla
        variables = await self._build_template_variables(rule_structure)
        
        # Renderizar plantilla
        return await self.rule_template_engine.render_template(template, variables)
    
    async def _generate_hybrid_code(self, rule_structure: RuleStructure) -> str:
        """Genera código para una regla híbrida.
        
        Args:
            rule_structure: Estructura de la regla
            
        Returns:
            Código generado
        """
        # Plantilla para reglas híbridas
        template = """
def analyze_{rule_name}(ast):
    violations = []
    
    # Find elements using pattern matching
    elements = find_elements_by_pattern(ast, "{pattern}")
    
    for element in elements:
        # Apply procedural analysis
        if not is_compliant(element, {threshold}):
            violations.append(create_violation(
                element,
                "{message}",
                "{severity}"
            ))
    
    return violations
"""
        
        # Construir variables para la plantilla
        variables = await self._build_template_variables(rule_structure)
        
        # Generar patrón
        variables["pattern"] = await self._generate_pattern(rule_structure)
        
        # Renderizar plantilla
        return await self.rule_template_engine.render_template(template, variables)
    
    async def _create_executable_rule(
        self, rule_structure: RuleStructure, code: str, 
        implementation_type: str, context: Dict[str, str] = None
    ) -> ExecutableRule:
        """Crea una regla ejecutable.
        
        Args:
            rule_structure: Estructura de la regla
            code: Código de la regla
            implementation_type: Tipo de implementación
            context: Contexto adicional
            
        Returns:
            Regla ejecutable
        """
        # Generar ID único
        rule_id = ExecutableRuleId()
        
        # Determinar nombre de la regla
        rule_name = await self._generate_rule_name(rule_structure)
        
        # Determinar descripción
        description = rule_structure.description
        if not description:
            description = await self._generate_rule_description(rule_structure)
        
        # Determinar lenguajes aplicables
        languages = []
        if rule_structure.scope.languages:
            languages = list(rule_structure.scope.languages)
        else:
            # Por defecto, aplicable a todos los lenguajes
            languages = ["python", "javascript", "typescript", "java", "csharp"]
        
        # Determinar categoría
        category = rule_structure.intent_analysis.domain
        
        # Determinar severidad
        severity = ActionSeverity.WARNING
        if rule_structure.actions:
            # Usar la severidad de la primera acción
            severity = rule_structure.actions[0].severity
        
        # Crear implementación
        implementation = RuleImplementation(
            code=code,
            language="python",
            parameters=self._extract_parameters(rule_structure)
        )
        
        # Crear configuración
        configuration = {}
        for threshold in rule_structure.thresholds:
            configuration[threshold.name] = str(threshold.value)
        
        # Crear metadatos
        metadata = {
            "implementation_type": implementation_type,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "intent": str(rule_structure.intent_analysis.primary_intent),
        }
        
        if context:
            metadata.update(context)
        
        return ExecutableRule(
            id=rule_id,
            rule_name=rule_name,
            description=description,
            implementation=implementation,
            languages=languages,
            category=category,
            severity=severity,
            configuration=configuration,
            metadata=metadata
        )
    
    async def _build_template_variables(
        self, rule_structure: RuleStructure
    ) -> Dict[str, str]:
        """Construye variables para una plantilla.
        
        Args:
            rule_structure: Estructura de la regla
            
        Returns:
            Diccionario con variables para la plantilla
        """
        variables = {}
        
        # Nombre de la regla
        variables["rule_name"] = await self._generate_rule_name(rule_structure)
        
        # Tipo de elemento
        element_type = "code"
        if rule_structure.target_element:
            element_type = rule_structure.target_element.element_type.name
        variables["element_type"] = element_type
        
        # Condición
        condition = ""
        if rule_structure.conditions:
            condition = rule_structure.conditions[0].condition_text
        variables["condition"] = condition
        
        # Mensaje
        message = await self._generate_rule_description(rule_structure)
        variables["message"] = message
        
        # Severidad
        severity = "WARNING"
        if rule_structure.actions:
            severity = str(rule_structure.actions[0].severity)
        variables["severity"] = severity
        
        # Umbral
        threshold = "0"
        if rule_structure.thresholds:
            threshold = str(rule_structure.thresholds[0].value)
        variables["threshold"] = threshold
        
        return variables
    
    def _get_template_name_for_intent(self, intent: RuleIntent) -> str:
        """Obtiene el nombre de plantilla para una intención.
        
        Args:
            intent: Intención
            
        Returns:
            Nombre de plantilla
        """
        # Mapeo de intenciones a nombres de plantillas
        intent_templates = {
            RuleIntent.PROHIBIT: "prohibition",
            RuleIntent.REQUIRE: "requirement",
            RuleIntent.RECOMMEND: "recommendation",
            RuleIntent.LIMIT: "limit",
            RuleIntent.ENSURE: "ensure",
            RuleIntent.VALIDATE: "validation",
            RuleIntent.CHECK: "check",
            RuleIntent.COUNT: "count",
            RuleIntent.MEASURE: "measure",
            RuleIntent.DETECT: "detection",
        }
        
        return intent_templates.get(intent, "generic")
    
    async def _generate_rule_name(self, rule_structure: RuleStructure) -> str:
        """Genera un nombre para una regla.
        
        Args:
            rule_structure: Estructura de la regla
            
        Returns:
            Nombre de la regla
        """
        # Prefijo según la intención
        prefix = self._get_template_name_for_intent(rule_structure.intent_analysis.primary_intent)
        
        # Sufijo según el elemento objetivo
        suffix = "code"
        if rule_structure.target_element:
            suffix = rule_structure.target_element.element_type.name.lower()
        
        # Generar nombre
        return f"{prefix}_{suffix}"
    
    async def _generate_rule_description(self, rule_structure: RuleStructure) -> str:
        """Genera una descripción para una regla.
        
        Args:
            rule_structure: Estructura de la regla
            
        Returns:
            Descripción de la regla
        """
        # Descripción según la intención y condiciones
        intent = str(rule_structure.intent_analysis.primary_intent)
        
        if rule_structure.conditions:
            condition = rule_structure.conditions[0].condition_text
            return f"{intent}: {condition}"
        
        return f"Rule implementing {intent}"
    
    async def _generate_query(self, rule_structure: RuleStructure) -> str:
        """Genera una consulta para una regla.
        
        Args:
            rule_structure: Estructura de la regla
            
        Returns:
            Consulta generada
        """
        # Consulta simple basada en condiciones
        if rule_structure.conditions:
            condition = rule_structure.conditions[0].condition_text
            return f"SELECT * WHERE {condition}"
        
        return "SELECT *"
    
    async def _generate_pattern(self, rule_structure: RuleStructure) -> str:
        """Genera un patrón para una regla.
        
        Args:
            rule_structure: Estructura de la regla
            
        Returns:
            Patrón generado
        """
        # Patrón simple basado en el elemento objetivo
        if rule_structure.target_element:
            element_type = rule_structure.target_element.element_type.name
            return f"{element_type}()"
        
        return "code()"
    
    def _extract_parameters(self, rule_structure: RuleStructure) -> Dict[str, str]:
        """Extrae parámetros de una estructura de regla.
        
        Args:
            rule_structure: Estructura de la regla
            
        Returns:
            Diccionario con parámetros
        """
        parameters = {}
        
        # Extraer umbrales como parámetros
        for threshold in rule_structure.thresholds:
            parameters[threshold.name] = str(threshold.value)
        
        # Extraer parámetros de condiciones
        for i, condition in enumerate(rule_structure.conditions):
            for key, value in condition.parameters.items():
                parameters[f"condition_{i}_{key}"] = value
        
        return parameters
