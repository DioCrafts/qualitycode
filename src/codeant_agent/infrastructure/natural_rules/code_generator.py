"""
Módulo que implementa el generador de código para el sistema de reglas en lenguaje natural.
"""
import re
from typing import Dict, List, Optional

from codeant_agent.application.ports.natural_rules.rule_generation_ports import CodeGeneratorPort
from codeant_agent.domain.entities.natural_rules.natural_rule import RuleStructure
from codeant_agent.domain.entities.natural_rules.rule_intent import RuleIntent


class CodeGenerator(CodeGeneratorPort):
    """Implementación del generador de código."""
    
    def __init__(self, template_engine):
        """Inicializa el generador de código.
        
        Args:
            template_engine: Motor de plantillas
        """
        self.template_engine = template_engine
        
        # Plantillas predefinidas por intención
        self.intent_templates = {
            RuleIntent.PROHIBIT: """
def check_prohibition_{element_type}(ast):
    \"\"\"
    Check that {element_type}s do not {prohibited_action}.
    
    Args:
        ast: Abstract Syntax Tree
        
    Returns:
        List of violations
    \"\"\"
    violations = []
    
    # Find all {element_type}s
    elements = find_elements(ast, "{element_type}")
    
    for element in elements:
        # Check if the element violates the prohibition
        if has_prohibited_action(element, "{prohibited_action}"):
            violations.append(create_violation(
                element,
                "The {element_type} must not {prohibited_action}",
                "{severity}"
            ))
    
    return violations
""",
            RuleIntent.REQUIRE: """
def check_requirement_{element_type}(ast):
    \"\"\"
    Check that {element_type}s {required_action}.
    
    Args:
        ast: Abstract Syntax Tree
        
    Returns:
        List of violations
    \"\"\"
    violations = []
    
    # Find all {element_type}s
    elements = find_elements(ast, "{element_type}")
    
    for element in elements:
        # Check if the element meets the requirement
        if not has_required_action(element, "{required_action}"):
            violations.append(create_violation(
                element,
                "The {element_type} must {required_action}",
                "{severity}"
            ))
    
    return violations
""",
            RuleIntent.LIMIT: """
def check_limit_{element_type}(ast):
    \"\"\"
    Check that {element_type}s do not exceed the limit of {threshold} {unit}.
    
    Args:
        ast: Abstract Syntax Tree
        
    Returns:
        List of violations
    \"\"\"
    violations = []
    
    # Find all {element_type}s
    elements = find_elements(ast, "{element_type}")
    
    for element in elements:
        # Measure the element
        value = measure_element(element, "{metric}")
        
        # Check if the element exceeds the limit
        if value > {threshold}:
            violations.append(create_violation(
                element,
                "The {element_type} has {metric} of " + str(value) + " which exceeds the limit of {threshold} {unit}",
                "{severity}"
            ))
    
    return violations
""",
        }
        
        # Plantillas para diferentes tipos de elementos
        self.element_templates = {
            "function": {
                "find_elements": """
def find_functions(ast):
    \"\"\"Find all functions in the AST.\"\"\"
    return ast.find_all("function_definition")
""",
                "measure_element": """
def measure_function_lines(function):
    \"\"\"Count the number of lines in a function.\"\"\"
    return function.end_line - function.start_line + 1
""",
            },
            "class": {
                "find_elements": """
def find_classes(ast):
    \"\"\"Find all classes in the AST.\"\"\"
    return ast.find_all("class_definition")
""",
                "measure_element": """
def measure_class_methods(class_node):
    \"\"\"Count the number of methods in a class.\"\"\"
    return len(class_node.find_all("function_definition"))
""",
            },
            "method": {
                "find_elements": """
def find_methods(ast):
    \"\"\"Find all methods in the AST.\"\"\"
    methods = []
    classes = ast.find_all("class_definition")
    for class_node in classes:
        methods.extend(class_node.find_all("function_definition"))
    return methods
""",
                "measure_element": """
def measure_method_parameters(method):
    \"\"\"Count the number of parameters in a method.\"\"\"
    params = method.find_first("parameters")
    if params:
        return len(params.find_all("parameter"))
    return 0
""",
            },
        }
    
    async def generate_code(
        self, rule_structure: RuleStructure, language: str = "python"
    ) -> str:
        """Genera código para una regla.
        
        Args:
            rule_structure: Estructura de la regla
            language: Lenguaje de programación para el código
            
        Returns:
            Código generado
        """
        # Determinar plantilla según la intención
        template = await self._get_template_for_intent(rule_structure.intent_analysis.primary_intent)
        
        # Construir variables para la plantilla
        variables = await self._build_template_variables(rule_structure)
        
        # Renderizar plantilla
        code = await self.template_engine.render_template(template, variables)
        
        # Formatear código según el lenguaje
        return await self.format_code(code, language)
    
    async def get_template(self, template_name: str) -> Optional[str]:
        """Obtiene una plantilla de código por nombre.
        
        Args:
            template_name: Nombre de la plantilla
            
        Returns:
            Plantilla de código o None si no existe
        """
        # Buscar en plantillas de intención
        for intent, template in self.intent_templates.items():
            if template_name == str(intent).lower():
                return template
        
        # Buscar en plantillas de elementos
        for element_type, templates in self.element_templates.items():
            if template_name == f"find_{element_type}s":
                return templates["find_elements"]
            elif template_name == f"measure_{element_type}":
                return templates["measure_element"]
        
        return None
    
    async def render_template(
        self, template: str, variables: Dict[str, str]
    ) -> str:
        """Renderiza una plantilla con variables.
        
        Args:
            template: Plantilla a renderizar
            variables: Variables para la renderización
            
        Returns:
            Plantilla renderizada
        """
        # Reemplazar variables en la plantilla
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", value)
        
        return result
    
    async def format_code(self, code: str, language: str) -> str:
        """Formatea el código generado.
        
        Args:
            code: Código a formatear
            language: Lenguaje de programación del código
            
        Returns:
            Código formateado
        """
        # Eliminar líneas en blanco múltiples
        formatted_code = re.sub(r'\n{3,}', '\n\n', code)
        
        # Eliminar espacios en blanco al final de las líneas
        formatted_code = re.sub(r'[ \t]+$', '', formatted_code, flags=re.MULTILINE)
        
        return formatted_code
    
    async def _get_template_for_intent(self, intent: RuleIntent) -> str:
        """Obtiene una plantilla para una intención.
        
        Args:
            intent: Intención para la que obtener la plantilla
            
        Returns:
            Plantilla para la intención
        """
        # Buscar plantilla específica para la intención
        if intent in self.intent_templates:
            return self.intent_templates[intent]
        
        # Plantilla genérica
        return """
def analyze_{rule_name}(ast):
    \"\"\"
    {description}
    
    Args:
        ast: Abstract Syntax Tree
        
    Returns:
        List of violations
    \"\"\"
    violations = []
    
    # Find relevant elements
    elements = find_elements(ast, "{element_type}")
    
    for element in elements:
        # Check the rule condition
        if not is_compliant(element):
            violations.append(create_violation(
                element,
                "{message}",
                "{severity}"
            ))
    
    return violations
"""
    
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
        rule_name = "custom_rule"
        if rule_structure.intent_analysis.primary_intent != RuleIntent.UNKNOWN:
            intent_name = str(rule_structure.intent_analysis.primary_intent).lower()
            rule_name = f"{intent_name}_rule"
        variables["rule_name"] = rule_name
        
        # Descripción
        variables["description"] = rule_structure.description or "Custom rule"
        
        # Tipo de elemento
        element_type = "code"
        if rule_structure.target_element:
            element_type = rule_structure.target_element.element_type.name.lower()
        variables["element_type"] = element_type
        
        # Mensaje
        variables["message"] = rule_structure.description or "Rule violation detected"
        
        # Severidad
        severity = "WARNING"
        if rule_structure.actions:
            severity = str(rule_structure.actions[0].severity)
        variables["severity"] = severity
        
        # Variables específicas según la intención
        if rule_structure.intent_analysis.primary_intent == RuleIntent.PROHIBIT:
            variables["prohibited_action"] = "violate the rule"
            if rule_structure.conditions:
                variables["prohibited_action"] = rule_structure.conditions[0].condition_text
        
        elif rule_structure.intent_analysis.primary_intent == RuleIntent.REQUIRE:
            variables["required_action"] = "follow the rule"
            if rule_structure.conditions:
                variables["required_action"] = rule_structure.conditions[0].condition_text
        
        elif rule_structure.intent_analysis.primary_intent == RuleIntent.LIMIT:
            variables["threshold"] = "0"
            variables["unit"] = "units"
            variables["metric"] = "value"
            
            if rule_structure.thresholds:
                variables["threshold"] = str(rule_structure.thresholds[0].value)
                if rule_structure.thresholds[0].unit:
                    variables["unit"] = rule_structure.thresholds[0].unit
            
            if element_type == "function":
                variables["metric"] = "lines"
            elif element_type == "method":
                variables["metric"] = "parameters"
            elif element_type == "class":
                variables["metric"] = "methods"
        
        return variables
