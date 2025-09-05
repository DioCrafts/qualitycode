"""
Parser real basado en Tree-sitter para análisis AST preciso.
"""
import tree_sitter
from tree_sitter import Language, Parser, Node
from typing import Dict, List, Set, Optional, Tuple, Any
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Importar los bindings de lenguajes específicos
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript


class NodeType(Enum):
    """Tipos de nodos AST que nos interesan para el análisis."""
    # Variables
    VARIABLE_DECLARATION = "variable_declaration"
    VARIABLE_REFERENCE = "variable_reference"
    
    # Funciones
    FUNCTION_DECLARATION = "function_declaration"
    FUNCTION_CALL = "function_call"
    
    # Clases
    CLASS_DECLARATION = "class_declaration"
    CLASS_INSTANTIATION = "class_instantiation"
    
    # Imports
    IMPORT_STATEMENT = "import_statement"
    IMPORT_FROM = "import_from"
    
    # Control de flujo
    IF_STATEMENT = "if_statement"
    WHILE_LOOP = "while_loop"
    FOR_LOOP = "for_loop"
    RETURN_STATEMENT = "return_statement"
    BREAK_STATEMENT = "break_statement"
    CONTINUE_STATEMENT = "continue_statement"
    
    # Asignaciones
    ASSIGNMENT = "assignment"
    
    # Parámetros
    PARAMETER = "parameter"


@dataclass
class Symbol:
    """Representa un símbolo (variable, función, clase, etc.) en el código."""
    name: str
    type: NodeType
    line: int
    column: int
    end_line: int
    end_column: int
    scope_level: int
    is_exported: bool = False
    is_used: bool = False
    definitions: List[Tuple[int, int]] = field(default_factory=list)  # (line, column)
    references: List[Tuple[int, int]] = field(default_factory=list)   # (line, column)
    
    def mark_as_used(self):
        """Marca el símbolo como usado."""
        self.is_used = True
        
    def add_reference(self, line: int, column: int):
        """Añade una referencia al símbolo."""
        self.references.append((line, column))
        self.mark_as_used()


@dataclass
class Scope:
    """Representa un ámbito (scope) en el código."""
    level: int
    parent: Optional['Scope'] = None
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    children: List['Scope'] = field(default_factory=list)
    
    def add_symbol(self, symbol: Symbol):
        """Añade un símbolo al scope."""
        self.symbols[symbol.name] = symbol
        
    def find_symbol(self, name: str) -> Optional[Symbol]:
        """Busca un símbolo en este scope o en scopes padres."""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.find_symbol(name)
        return None


class TreeSitterAnalyzer:
    """Analizador AST basado en Tree-sitter."""
    
    def __init__(self, language: str):
        self.language = language
        self.parser = Parser()
        self._setup_language()
        self.source_code: Optional[bytes] = None
        self.tree: Optional[Any] = None
        self.root_scope: Optional[Scope] = None
        self.current_scope: Optional[Scope] = None
        self.all_symbols: Dict[str, Symbol] = {}
        
    def _setup_language(self):
        """Configura el lenguaje para el parser."""
        if self.language == "python":
            PY_LANGUAGE = Language(tspython.language(), "python")
            self.parser.set_language(PY_LANGUAGE)
        elif self.language == "javascript":
            JS_LANGUAGE = Language(tsjavascript.language(), "javascript")
            self.parser.set_language(JS_LANGUAGE)
        elif self.language == "typescript":
            TS_LANGUAGE = Language(tstypescript.language(), "typescript")
            self.parser.set_language(TS_LANGUAGE)
        else:
            raise ValueError(f"Lenguaje no soportado: {self.language}")
            
    def parse(self, source_code: str) -> bool:
        """Parsea el código fuente y construye el AST."""
        try:
            self.source_code = source_code.encode('utf-8')
            self.tree = self.parser.parse(self.source_code)
            return True
        except Exception as e:
            print(f"Error parseando código: {e}")
            return False
            
    def analyze(self) -> Dict[str, Any]:
        """Analiza el AST para detectar código muerto."""
        if not self.tree:
            return {}
            
        # Inicializar scopes
        self.root_scope = Scope(level=0)
        self.current_scope = self.root_scope
        
        # Primera pasada: recolectar todas las definiciones
        self._collect_definitions(self.tree.root_node)
        
        # Segunda pasada: recolectar todos los usos
        self._collect_references(self.tree.root_node)
        
        # Analizar código muerto
        dead_code = self._analyze_dead_code()
        
        return dead_code
        
    def _collect_definitions(self, node: Node, depth: int = 0):
        """Recolecta todas las definiciones de símbolos."""
        node_type = node.type
        
        # Python
        if self.language == "python":
            if node_type == "function_definition":
                self._handle_python_function_definition(node)
            elif node_type == "class_definition":
                self._handle_python_class_definition(node)
            elif node_type == "assignment":
                self._handle_python_assignment(node)
            elif node_type == "import_statement" or node_type == "import_from_statement":
                self._handle_python_import(node)
            elif node_type in ["for_statement", "with_statement"]:
                # Estos crean nuevos scopes
                new_scope = Scope(level=self.current_scope.level + 1, parent=self.current_scope)
                self.current_scope.children.append(new_scope)
                old_scope = self.current_scope
                self.current_scope = new_scope
                
        # JavaScript/TypeScript
        elif self.language in ["javascript", "typescript"]:
            if node_type == "function_declaration":
                self._handle_js_function_declaration(node)
            elif node_type == "class_declaration":
                self._handle_js_class_declaration(node)
            elif node_type in ["variable_declaration", "lexical_declaration"]:
                self._handle_js_variable_declaration(node)
            elif node_type == "import_statement":
                self._handle_js_import(node)
                
        # Recursión para hijos
        for child in node.children:
            self._collect_definitions(child, depth + 1)
            
        # Restaurar scope si es necesario
        if self.language == "python" and node_type in ["for_statement", "with_statement", "function_definition", "class_definition"]:
            if hasattr(self, '_scope_changed'):
                self.current_scope = self.current_scope.parent
                delattr(self, '_scope_changed')
                
    def _collect_references(self, node: Node):
        """Recolecta todas las referencias a símbolos."""
        node_type = node.type
        
        # Python
        if self.language == "python":
            if node_type == "identifier" and self._is_reference_context(node):
                symbol_name = node.text.decode('utf-8')
                symbol = self.root_scope.find_symbol(symbol_name)
                if symbol:
                    line = node.start_point[0]
                    column = node.start_point[1]
                    symbol.add_reference(line, column)
                    
            elif node_type == "call":
                # Manejar llamadas a funciones
                function_node = node.child_by_field_name("function")
                if function_node and function_node.type == "identifier":
                    func_name = function_node.text.decode('utf-8')
                    symbol = self.root_scope.find_symbol(func_name)
                    if symbol:
                        line = function_node.start_point[0]
                        column = function_node.start_point[1]
                        symbol.add_reference(line, column)
                        
        # JavaScript/TypeScript
        elif self.language in ["javascript", "typescript"]:
            if node_type == "identifier" and self._is_js_reference_context(node):
                symbol_name = node.text.decode('utf-8')
                symbol = self.root_scope.find_symbol(symbol_name)
                if symbol:
                    line = node.start_point[0]
                    column = node.start_point[1]
                    symbol.add_reference(line, column)
                    
        # Recursión para hijos
        for child in node.children:
            self._collect_references(child)
            
    def _handle_python_function_definition(self, node: Node):
        """Maneja definiciones de funciones en Python."""
        name_node = node.child_by_field_name("name")
        if name_node:
            func_name = name_node.text.decode('utf-8')
            
            symbol = Symbol(
                name=func_name,
                type=NodeType.FUNCTION_DECLARATION,
                line=node.start_point[0],
                column=node.start_point[1],
                end_line=node.end_point[0],
                end_column=node.end_point[1],
                scope_level=self.current_scope.level
            )
            
            # Verificar si es exportada (__all__ o método público)
            if not func_name.startswith('_') or func_name.startswith('__') and func_name.endswith('__'):
                symbol.is_exported = True
                
            self.current_scope.add_symbol(symbol)
            self.all_symbols[func_name] = symbol
            
            # Manejar parámetros
            params_node = node.child_by_field_name("parameters")
            if params_node:
                self._handle_python_parameters(params_node)
                
        # Crear nuevo scope para el cuerpo de la función
        new_scope = Scope(level=self.current_scope.level + 1, parent=self.current_scope)
        self.current_scope.children.append(new_scope)
        self.current_scope = new_scope
        self._scope_changed = True
        
    def _handle_python_class_definition(self, node: Node):
        """Maneja definiciones de clases en Python."""
        name_node = node.child_by_field_name("name")
        if name_node:
            class_name = name_node.text.decode('utf-8')
            
            symbol = Symbol(
                name=class_name,
                type=NodeType.CLASS_DECLARATION,
                line=node.start_point[0],
                column=node.start_point[1],
                end_line=node.end_point[0],
                end_column=node.end_point[1],
                scope_level=self.current_scope.level
            )
            
            # Las clases generalmente son exportadas si no empiezan con _
            if not class_name.startswith('_'):
                symbol.is_exported = True
                
            self.current_scope.add_symbol(symbol)
            self.all_symbols[class_name] = symbol
            
        # Crear nuevo scope para el cuerpo de la clase
        new_scope = Scope(level=self.current_scope.level + 1, parent=self.current_scope)
        self.current_scope.children.append(new_scope)
        self.current_scope = new_scope
        self._scope_changed = True
        
    def _handle_python_assignment(self, node: Node):
        """Maneja asignaciones en Python."""
        left_node = node.child_by_field_name("left")
        if left_node and left_node.type == "identifier":
            var_name = left_node.text.decode('utf-8')
            
            # Solo crear símbolo si no existe ya
            existing = self.current_scope.find_symbol(var_name)
            if not existing:
                symbol = Symbol(
                    name=var_name,
                    type=NodeType.VARIABLE_DECLARATION,
                    line=node.start_point[0],
                    column=node.start_point[1],
                    end_line=node.end_point[0],
                    end_column=node.end_point[1],
                    scope_level=self.current_scope.level
                )
                
                self.current_scope.add_symbol(symbol)
                self.all_symbols[var_name] = symbol
                
    def _handle_python_import(self, node: Node):
        """Maneja imports en Python."""
        if node.type == "import_statement":
            # import module
            for child in node.children:
                if child.type == "dotted_name" or child.type == "identifier":
                    import_name = child.text.decode('utf-8')
                    symbol = Symbol(
                        name=import_name,
                        type=NodeType.IMPORT_STATEMENT,
                        line=node.start_point[0],
                        column=node.start_point[1],
                        end_line=node.end_point[0],
                        end_column=node.end_point[1],
                        scope_level=self.current_scope.level
                    )
                    self.current_scope.add_symbol(symbol)
                    self.all_symbols[import_name] = symbol
                    
        elif node.type == "import_from_statement":
            # from module import name
            for child in node.children:
                if child.type == "import_from_as_names" or child.type == "identifier":
                    if child.type == "identifier" and child.prev_sibling and child.prev_sibling.type == "import":
                        import_name = child.text.decode('utf-8')
                        symbol = Symbol(
                            name=import_name,
                            type=NodeType.IMPORT_FROM,
                            line=node.start_point[0],
                            column=node.start_point[1],
                            end_line=node.end_point[0],
                            end_column=node.end_point[1],
                            scope_level=self.current_scope.level
                        )
                        self.current_scope.add_symbol(symbol)
                        self.all_symbols[import_name] = symbol
                        
    def _handle_python_parameters(self, params_node: Node):
        """Maneja parámetros de funciones en Python."""
        for param in params_node.children:
            if param.type == "identifier":
                param_name = param.text.decode('utf-8')
                if param_name != "self":  # Ignorar self
                    symbol = Symbol(
                        name=param_name,
                        type=NodeType.PARAMETER,
                        line=param.start_point[0],
                        column=param.start_point[1],
                        end_line=param.end_point[0],
                        end_column=param.end_point[1],
                        scope_level=self.current_scope.level + 1  # En el scope de la función
                    )
                    # Los parámetros se añadirán al scope de la función
                    # Por ahora los marcamos como usados por defecto
                    symbol.mark_as_used()
                    
    def _handle_js_function_declaration(self, node: Node):
        """Maneja declaraciones de funciones en JavaScript/TypeScript."""
        name_node = node.child_by_field_name("name")
        if name_node:
            func_name = name_node.text.decode('utf-8')
            
            symbol = Symbol(
                name=func_name,
                type=NodeType.FUNCTION_DECLARATION,
                line=node.start_point[0],
                column=node.start_point[1],
                end_line=node.end_point[0],
                end_column=node.end_point[1],
                scope_level=self.current_scope.level
            )
            
            # Verificar si es exportada
            parent = node.parent
            if parent and parent.type == "export_statement":
                symbol.is_exported = True
                
            self.current_scope.add_symbol(symbol)
            self.all_symbols[func_name] = symbol
            
    def _handle_js_class_declaration(self, node: Node):
        """Maneja declaraciones de clases en JavaScript/TypeScript."""
        name_node = node.child_by_field_name("name")
        if name_node:
            class_name = name_node.text.decode('utf-8')
            
            symbol = Symbol(
                name=class_name,
                type=NodeType.CLASS_DECLARATION,
                line=node.start_point[0],
                column=node.start_point[1],
                end_line=node.end_point[0],
                end_column=node.end_point[1],
                scope_level=self.current_scope.level
            )
            
            # Verificar si es exportada
            parent = node.parent
            if parent and parent.type == "export_statement":
                symbol.is_exported = True
                
            self.current_scope.add_symbol(symbol)
            self.all_symbols[class_name] = symbol
            
    def _handle_js_variable_declaration(self, node: Node):
        """Maneja declaraciones de variables en JavaScript/TypeScript."""
        for declarator in node.children:
            if declarator.type == "variable_declarator":
                name_node = declarator.child_by_field_name("name")
                if name_node and name_node.type == "identifier":
                    var_name = name_node.text.decode('utf-8')
                    
                    symbol = Symbol(
                        name=var_name,
                        type=NodeType.VARIABLE_DECLARATION,
                        line=node.start_point[0],
                        column=node.start_point[1],
                        end_line=node.end_point[0],
                        end_column=node.end_point[1],
                        scope_level=self.current_scope.level
                    )
                    
                    self.current_scope.add_symbol(symbol)
                    self.all_symbols[var_name] = symbol
                    
    def _handle_js_import(self, node: Node):
        """Maneja imports en JavaScript/TypeScript."""
        # TODO: Implementar manejo de imports JS/TS
        pass
        
    def _is_reference_context(self, node: Node) -> bool:
        """Determina si un identificador es una referencia (no una definición) en Python."""
        parent = node.parent
        if not parent:
            return False
            
        # No es referencia si es parte de una definición
        if parent.type in ["function_definition", "class_definition"]:
            if parent.child_by_field_name("name") == node:
                return False
                
        # No es referencia si es el lado izquierdo de una asignación
        if parent.type == "assignment":
            if parent.child_by_field_name("left") == node:
                return False
                
        # No es referencia si es parte de un import
        if parent.type in ["import_statement", "import_from_statement"]:
            return False
            
        return True
        
    def _is_js_reference_context(self, node: Node) -> bool:
        """Determina si un identificador es una referencia en JavaScript/TypeScript."""
        parent = node.parent
        if not parent:
            return False
            
        # Similar lógica para JS/TS
        if parent.type in ["function_declaration", "class_declaration"]:
            if parent.child_by_field_name("name") == node:
                return False
                
        if parent.type == "variable_declarator":
            if parent.child_by_field_name("name") == node:
                return False
                
        return True
        
    def _analyze_dead_code(self) -> Dict[str, Any]:
        """Analiza el código muerto basándose en los símbolos recolectados."""
        dead_code = {
            "unused_variables": [],
            "unused_functions": [],
            "unused_classes": [],
            "unused_imports": [],
            "unused_parameters": [],
        }
        
        for symbol_name, symbol in self.all_symbols.items():
            # Ignorar símbolos exportados o especiales
            if symbol.is_exported:
                continue
                
            # Ignorar símbolos que empiezan con _ (convención de privado)
            if symbol_name.startswith('__') and symbol_name.endswith('__'):
                continue
                
            if not symbol.is_used:
                dead_item = {
                    "name": symbol_name,
                    "line": symbol.line + 1,  # Tree-sitter usa 0-based
                    "column": symbol.column,
                    "end_line": symbol.end_line + 1,
                    "end_column": symbol.end_column,
                }
                
                if symbol.type == NodeType.VARIABLE_DECLARATION:
                    dead_code["unused_variables"].append(dead_item)
                elif symbol.type == NodeType.FUNCTION_DECLARATION:
                    dead_code["unused_functions"].append(dead_item)
                elif symbol.type == NodeType.CLASS_DECLARATION:
                    dead_code["unused_classes"].append(dead_item)
                elif symbol.type in [NodeType.IMPORT_STATEMENT, NodeType.IMPORT_FROM]:
                    dead_code["unused_imports"].append(dead_item)
                elif symbol.type == NodeType.PARAMETER:
                    dead_code["unused_parameters"].append(dead_item)
                    
        return dead_code
