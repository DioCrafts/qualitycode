"""
Módulo que implementa el motor de plantillas para el sistema de reglas en lenguaje natural.
"""
import os
import re
from typing import Dict, Optional


class TemplateEngine:
    """Implementación del motor de plantillas."""
    
    def __init__(self, template_dir: Optional[str] = None):
        """Inicializa el motor de plantillas.
        
        Args:
            template_dir: Directorio de plantillas (opcional)
        """
        self.template_dir = template_dir
        self.templates = {}
        
        # Plantillas predefinidas
        self.templates["function_line_limit"] = """
def check_function_line_limit(ast):
    \"\"\"
    Check that functions do not exceed the line limit.
    
    Args:
        ast: Abstract Syntax Tree
        
    Returns:
        List of violations
    \"\"\"
    violations = []
    
    # Find all functions
    functions = find_functions(ast)
    
    for function in functions:
        # Count lines
        line_count = count_function_lines(function)
        
        # Check if the function exceeds the limit
        if line_count > {max_lines}:
            violations.append(create_violation(
                function,
                "Function '{function_name}' has {line_count} lines, exceeding the limit of {max_lines}",
                "{severity}"
            ))
    
    return violations
"""
        
        self.templates["method_parameter_limit"] = """
def check_method_parameter_limit(ast):
    \"\"\"
    Check that methods do not have too many parameters.
    
    Args:
        ast: Abstract Syntax Tree
        
    Returns:
        List of violations
    \"\"\"
    violations = []
    
    # Find all methods
    methods = find_methods(ast)
    
    for method in methods:
        # Count parameters
        param_count = count_method_parameters(method)
        
        # Check if the method exceeds the parameter limit
        if param_count > {max_params}:
            violations.append(create_violation(
                method,
                "Method '{method_name}' has {param_count} parameters, exceeding the limit of {max_params}",
                "{severity}"
            ))
    
    return violations
"""
        
        self.templates["class_method_limit"] = """
def check_class_method_limit(ast):
    \"\"\"
    Check that classes do not have too many methods.
    
    Args:
        ast: Abstract Syntax Tree
        
    Returns:
        List of violations
    \"\"\"
    violations = []
    
    # Find all classes
    classes = find_classes(ast)
    
    for class_node in classes:
        # Count methods
        method_count = count_class_methods(class_node)
        
        # Check if the class exceeds the method limit
        if method_count > {max_methods}:
            violations.append(create_violation(
                class_node,
                "Class '{class_name}' has {method_count} methods, exceeding the limit of {max_methods}",
                "{severity}"
            ))
    
    return violations
"""
        
        self.templates["nesting_depth_limit"] = """
def check_nesting_depth_limit(ast):
    \"\"\"
    Check that code does not have excessive nesting.
    
    Args:
        ast: Abstract Syntax Tree
        
    Returns:
        List of violations
    \"\"\"
    violations = []
    
    # Find all functions and methods
    functions = find_functions(ast)
    
    for function in functions:
        # Calculate max nesting depth
        max_depth = calculate_max_nesting_depth(function)
        
        # Check if the nesting depth exceeds the limit
        if max_depth > {max_depth}:
            violations.append(create_violation(
                function,
                "Function '{function_name}' has a nesting depth of {nesting_depth}, exceeding the limit of {max_depth}",
                "{severity}"
            ))
    
    return violations
"""
        
        self.templates["security_sensitive_data"] = """
def check_sensitive_data_handling(ast):
    \"\"\"
    Check that sensitive data is handled securely.
    
    Args:
        ast: Abstract Syntax Tree
        
    Returns:
        List of violations
    \"\"\"
    violations = []
    
    # Find variables that might contain sensitive data
    sensitive_vars = find_sensitive_variables(ast)
    
    for var in sensitive_vars:
        # Check if the variable is handled securely
        if not is_handled_securely(var):
            violations.append(create_violation(
                var,
                "Sensitive data in variable '{var_name}' is not handled securely",
                "ERROR"
            ))
    
    return violations
"""
        
        # Cargar plantillas desde el directorio si se proporciona
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)
    
    def _load_templates_from_dir(self, template_dir: str):
        """Carga plantillas desde un directorio.
        
        Args:
            template_dir: Directorio de plantillas
        """
        for filename in os.listdir(template_dir):
            if filename.endswith(".py") or filename.endswith(".template"):
                template_name = os.path.splitext(filename)[0]
                template_path = os.path.join(template_dir, filename)
                
                try:
                    with open(template_path, "r") as f:
                        self.templates[template_name] = f.read()
                except Exception:
                    # Ignorar errores al cargar plantillas
                    pass
    
    async def get_template(self, template_name: str) -> Optional[str]:
        """Obtiene una plantilla por nombre.
        
        Args:
            template_name: Nombre de la plantilla
            
        Returns:
            Plantilla o None si no existe
        """
        return self.templates.get(template_name)
    
    async def render_template(self, template: str, variables: Dict[str, str]) -> str:
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
            placeholder = "{" + key + "}"
            result = result.replace(placeholder, value)
        
        return result
    
    async def register_template(self, template_name: str, template: str) -> bool:
        """Registra una nueva plantilla.
        
        Args:
            template_name: Nombre de la plantilla
            template: Contenido de la plantilla
            
        Returns:
            True si la plantilla se registró correctamente, False en caso contrario
        """
        try:
            self.templates[template_name] = template
            return True
        except Exception:
            return False
