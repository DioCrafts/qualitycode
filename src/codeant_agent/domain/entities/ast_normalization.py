"""
Entidades para normalización cross-language de ASTs.

Este módulo define las entidades que representan el sistema
de normalización de Abstract Syntax Trees entre diferentes
lenguajes de programación.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from ..value_objects.programming_language import ProgrammingLanguage


class NodeType(Enum):
    """Tipos de nodos AST normalizados."""
    
    # Estructura del programa
    PROGRAM = "program"
    MODULE = "module"
    PACKAGE = "package"
    NAMESPACE = "namespace"
    
    # Declaraciones
    FUNCTION_DECLARATION = "function_declaration"
    CLASS_DECLARATION = "class_declaration"
    INTERFACE_DECLARATION = "interface_declaration"
    VARIABLE_DECLARATION = "variable_declaration"
    CONSTANT_DECLARATION = "constant_declaration"
    TYPE_DECLARATION = "type_declaration"
    ENUM_DECLARATION = "enum_declaration"
    STRUCT_DECLARATION = "struct_declaration"
    TRAIT_DECLARATION = "trait_declaration"
    IMPL_DECLARATION = "impl_declaration"
    
    # Declaraciones de importación/exportación
    IMPORT_DECLARATION = "import_declaration"
    EXPORT_DECLARATION = "export_declaration"
    USING_DECLARATION = "using_declaration"
    
    # Statements
    EXPRESSION_STATEMENT = "expression_statement"
    IF_STATEMENT = "if_statement"
    FOR_STATEMENT = "for_statement"
    WHILE_STATEMENT = "while_statement"
    DO_WHILE_STATEMENT = "do_while_statement"
    SWITCH_STATEMENT = "switch_statement"
    MATCH_STATEMENT = "match_statement"
    TRY_STATEMENT = "try_statement"
    THROW_STATEMENT = "throw_statement"
    RETURN_STATEMENT = "return_statement"
    BREAK_STATEMENT = "break_statement"
    CONTINUE_STATEMENT = "continue_statement"
    ASSERT_STATEMENT = "assert_statement"
    
    # Expressions
    BINARY_EXPRESSION = "binary_expression"
    UNARY_EXPRESSION = "unary_expression"
    CALL_EXPRESSION = "call_expression"
    MEMBER_EXPRESSION = "member_expression"
    ASSIGNMENT_EXPRESSION = "assignment_expression"
    CONDITIONAL_EXPRESSION = "conditional_expression"
    ARRAY_EXPRESSION = "array_expression"
    OBJECT_EXPRESSION = "object_expression"
    TUPLE_EXPRESSION = "tuple_expression"
    RANGE_EXPRESSION = "range_expression"
    SPREAD_EXPRESSION = "spread_expression"
    YIELD_EXPRESSION = "yield_expression"
    AWAIT_EXPRESSION = "await_expression"
    
    # Literals
    STRING_LITERAL = "string_literal"
    NUMBER_LITERAL = "number_literal"
    BOOLEAN_LITERAL = "boolean_literal"
    NULL_LITERAL = "null_literal"
    UNDEFINED_LITERAL = "undefined_literal"
    REGEX_LITERAL = "regex_literal"
    TEMPLATE_LITERAL = "template_literal"
    
    # Identificadores y referencias
    IDENTIFIER = "identifier"
    THIS_EXPRESSION = "this_expression"
    SUPER_EXPRESSION = "super_expression"
    NEW_EXPRESSION = "new_expression"
    
    # Comentarios y documentación
    COMMENT = "comment"
    DOCUMENTATION = "documentation"
    JSDOC = "jsdoc"
    RUSTDOC = "rustdoc"
    PYTHON_DOCSTRING = "python_docstring"
    
    # Tipos
    TYPE_ANNOTATION = "type_annotation"
    TYPE_REFERENCE = "type_reference"
    GENERIC_TYPE = "generic_type"
    UNION_TYPE = "union_type"
    INTERSECTION_TYPE = "intersection_type"
    OPTIONAL_TYPE = "optional_type"
    
    # Patrones
    PATTERN = "pattern"
    DESTRUCTURING_PATTERN = "destructuring_pattern"
    ARRAY_PATTERN = "array_pattern"
    OBJECT_PATTERN = "object_pattern"
    TUPLE_PATTERN = "tuple_pattern"
    
    # Específico del lenguaje (preservado para análisis detallado)
    LANGUAGE_SPECIFIC = "language_specific"


class NodeVisibility(Enum):
    """Visibilidad de nodos AST."""
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    INTERNAL = "internal"
    PACKAGE = "package"
    UNKNOWN = "unknown"


class NodeModifier(Enum):
    """Modificadores de nodos AST."""
    STATIC = "static"
    ABSTRACT = "abstract"
    FINAL = "final"
    CONST = "const"
    READONLY = "readonly"
    MUTABLE = "mutable"
    OVERRIDE = "override"
    VIRTUAL = "virtual"
    SEALED = "sealed"
    OPEN = "open"
    ASYNC = "async"
    GENERATOR = "generator"
    VOLATILE = "volatile"
    SYNCHRONIZED = "synchronized"
    TRANSIENT = "transient"
    NATIVE = "native"
    STRICTFP = "strictfp"
    DEFAULT = "default"


@dataclass
class SourcePosition:
    """Posición en el código fuente."""
    
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    start_byte: int
    end_byte: int
    
    def __post_init__(self) -> None:
        """Validar la posición."""
        if self.start_line < 0:
            raise ValueError("La línea de inicio no puede ser negativa")
        
        if self.start_column < 0:
            raise ValueError("La columna de inicio no puede ser negativa")
        
        if self.end_line < 0:
            raise ValueError("La línea de fin no puede ser negativa")
        
        if self.end_column < 0:
            raise ValueError("La columna de fin no puede ser negativa")
        
        if self.start_byte < 0:
            raise ValueError("El byte de inicio no puede ser negativo")
        
        if self.end_byte < 0:
            raise ValueError("El byte de fin no puede ser negativo")
        
        if self.end_line < self.start_line:
            raise ValueError("La línea de fin debe ser mayor o igual a la de inicio")
        
        if self.end_line == self.start_line and self.end_column < self.start_column:
            raise ValueError("En la misma línea, la columna de fin debe ser mayor o igual a la de inicio")
    
    @property
    def line_count(self) -> int:
        """Calcula el número de líneas que ocupa."""
        return self.end_line - self.start_line + 1
    
    @property
    def column_span(self) -> int:
        """Calcula el span de columnas."""
        if self.start_line == self.end_line:
            return self.end_column - self.start_column
        else:
            return self.end_column
    
    @property
    def byte_span(self) -> int:
        """Calcula el span de bytes."""
        return self.end_byte - self.start_byte
    
    def contains_position(self, line: int, column: int) -> bool:
        """Verifica si contiene una posición específica."""
        if line < self.start_line or line > self.end_line:
            return False
        
        if line == self.start_line and column < self.start_column:
            return False
        
        if line == self.end_line and column > self.end_column:
            return False
        
        return True
    
    def overlaps_with(self, other: 'SourcePosition') -> bool:
        """Verifica si se solapa con otra posición."""
        return not (
            self.end_line < other.start_line or
            other.end_line < self.start_line or
            (self.end_line == other.start_line and self.end_column < other.start_column) or
            (other.end_line == self.start_line and other.end_column < self.start_column)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la posición a diccionario."""
        return {
            "start_line": self.start_line,
            "start_column": self.start_column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "line_count": self.line_count,
            "column_span": self.column_span,
            "byte_span": self.byte_span,
        }
    
    def __str__(self) -> str:
        """Representación string de la posición."""
        if self.start_line == self.end_line:
            if self.start_column == self.end_column:
                return f"línea {self.start_line}, columna {self.start_column}"
            else:
                return f"línea {self.start_line}, columnas {self.start_column}-{self.end_column}"
        else:
            return f"líneas {self.start_line}-{self.end_line}"
    
    def __repr__(self) -> str:
        """Representación de debug de la posición."""
        return (
            f"SourcePosition("
            f"start_line={self.start_line}, "
            f"start_column={self.start_column}, "
            f"end_line={self.end_line}, "
            f"end_column={self.end_column}, "
            f"start_byte={self.start_byte}, "
            f"end_byte={self.end_byte}"
            f")"
        )


@dataclass
class SemanticInfo:
    """Información semántica de un nodo AST."""
    
    symbol_type: Optional[str] = None
    data_type: Optional[str] = None
    visibility: NodeVisibility = NodeVisibility.UNKNOWN
    modifiers: List[NodeModifier] = field(default_factory=list)
    scope_level: Optional[str] = None
    is_exported: bool = False
    is_imported: bool = False
    is_constant: bool = False
    is_mutable: bool = False
    is_optional: bool = False
    is_nullable: bool = False
    is_async: bool = False
    is_generator: bool = False
    is_abstract: bool = False
    is_final: bool = False
    is_static: bool = False
    is_virtual: bool = False
    is_override: bool = False
    is_sealed: bool = False
    is_open: bool = False
    is_native: bool = False
    is_deprecated: bool = False
    is_experimental: bool = False
    is_internal: bool = False
    is_public_api: bool = False
    is_test_only: bool = False
    is_debug_only: bool = False
    custom_flags: Dict[str, bool] = field(default_factory=dict)
    
    def has_modifier(self, modifier: NodeModifier) -> bool:
        """Verifica si tiene un modificador específico."""
        return modifier in self.modifiers
    
    def add_modifier(self, modifier: NodeModifier) -> None:
        """Agrega un modificador."""
        if modifier not in self.modifiers:
            self.modifiers.append(modifier)
    
    def remove_modifier(self, modifier: NodeModifier) -> None:
        """Remueve un modificador."""
        if modifier in self.modifiers:
            self.modifiers.remove(modifier)
    
    def set_flag(self, flag: str, value: bool = True) -> None:
        """Establece una bandera personalizada."""
        self.custom_flags[flag] = value
    
    def get_flag(self, flag: str, default: bool = False) -> bool:
        """Obtiene el valor de una bandera personalizada."""
        return self.custom_flags.get(flag, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la información semántica a diccionario."""
        return {
            "symbol_type": self.symbol_type,
            "data_type": self.data_type,
            "visibility": self.visibility.value,
            "modifiers": [modifier.value for modifier in self.modifiers],
            "scope_level": self.scope_level,
            "is_exported": self.is_exported,
            "is_imported": self.is_imported,
            "is_constant": self.is_constant,
            "is_mutable": self.is_mutable,
            "is_optional": self.is_optional,
            "is_nullable": self.is_nullable,
            "is_async": self.is_async,
            "is_generator": self.is_generator,
            "is_abstract": self.is_abstract,
            "is_final": self.is_final,
            "is_static": self.is_static,
            "is_virtual": self.is_virtual,
            "is_override": self.is_override,
            "is_sealed": self.is_sealed,
            "is_open": self.is_open,
            "is_native": self.is_native,
            "is_deprecated": self.is_deprecated,
            "is_experimental": self.is_experimental,
            "is_internal": self.is_internal,
            "is_public_api": self.is_public_api,
            "is_test_only": self.is_test_only,
            "is_debug_only": self.is_debug_only,
            "custom_flags": self.custom_flags,
        }
    
    def __str__(self) -> str:
        """Representación string de la información semántica."""
        parts = []
        
        if self.symbol_type:
            parts.append(self.symbol_type)
        
        if self.data_type:
            parts.append(f":{self.data_type}")
        
        if self.visibility != NodeVisibility.UNKNOWN:
            parts.append(self.visibility.value)
        
        if self.modifiers:
            parts.extend([mod.value for mod in self.modifiers])
        
        return " ".join(parts) if parts else "semantic_info"
    
    def __repr__(self) -> str:
        """Representación de debug de la información semántica."""
        return (
            f"SemanticInfo("
            f"symbol_type={self.symbol_type}, "
            f"data_type={self.data_type}, "
            f"visibility={self.visibility}, "
            f"modifiers={self.modifiers}, "
            f"scope_level={self.scope_level}"
            f")"
        )


@dataclass
class NormalizedNode:
    """Nodo AST normalizado."""
    
    node_type: NodeType
    position: SourcePosition
    name: Optional[str] = None
    value: Optional[str] = None
    children: List['NormalizedNode'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    semantic_info: Optional[SemanticInfo] = None
    original_node_type: Optional[str] = None
    language_specific_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validar el nodo normalizado."""
        if self.node_type is None:
            raise ValueError("El tipo de nodo no puede ser None")
        
        if self.position is None:
            raise ValueError("La posición no puede ser None")
    
    @property
    def child_count(self) -> int:
        """Obtiene el número de hijos."""
        return len(self.children)
    
    @property
    def is_leaf(self) -> bool:
        """Verifica si es un nodo hoja."""
        return self.child_count == 0
    
    @property
    def is_branch(self) -> bool:
        """Verifica si es un nodo rama."""
        return self.child_count > 0
    
    @property
    def depth(self) -> int:
        """Calcula la profundidad del nodo."""
        if self.is_leaf:
            return 0
        
        max_child_depth = max(child.depth for child in self.children)
        return max_child_depth + 1
    
    @property
    def total_nodes(self) -> int:
        """Calcula el número total de nodos en el subárbol."""
        if self.is_leaf:
            return 1
        
        return 1 + sum(child.total_nodes for child in self.children)
    
    def get_child_by_name(self, name: str) -> Optional['NormalizedNode']:
        """Obtiene un hijo por nombre."""
        for child in self.children:
            if child.name == name:
                return child
        return None
    
    def get_children_by_type(self, node_type: NodeType) -> List['NormalizedNode']:
        """Obtiene hijos por tipo."""
        return [child for child in self.children if child.node_type == node_type]
    
    def find_nodes_by_type(self, node_type: NodeType) -> List['NormalizedNode']:
        """Encuentra todos los nodos de un tipo específico en el subárbol."""
        result = []
        
        if self.node_type == node_type:
            result.append(self)
        
        for child in self.children:
            result.extend(child.find_nodes_by_type(node_type))
        
        return result
    
    def find_nodes_by_name(self, name: str) -> List['NormalizedNode']:
        """Encuentra todos los nodos con un nombre específico en el subárbol."""
        result = []
        
        if self.name == name:
            result.append(self)
        
        for child in self.children:
            result.extend(child.find_nodes_by_name(name))
        
        return result
    
    def get_node_at_position(self, line: int, column: int) -> Optional['NormalizedNode']:
        """Obtiene el nodo en una posición específica."""
        if not self.position.contains_position(line, column):
            return None
        
        # Buscar en los hijos
        for child in self.children:
            node_at_pos = child.get_node_at_position(line, column)
            if node_at_pos:
                return node_at_pos
        
        # Si no se encontró en los hijos, este nodo contiene la posición
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el nodo a diccionario."""
        return {
            "node_type": self.node_type.value,
            "name": self.name,
            "value": self.value,
            "position": self.position.to_dict(),
            "children": [child.to_dict() for child in self.children],
            "child_count": self.child_count,
            "is_leaf": self.is_leaf,
            "is_branch": self.is_branch,
            "depth": self.depth,
            "total_nodes": self.total_nodes,
            "attributes": self.attributes,
            "semantic_info": self.semantic_info.to_dict() if self.semantic_info else None,
            "original_node_type": self.original_node_type,
            "language_specific_data": self.language_specific_data,
        }
    
    def __str__(self) -> str:
        """Representación string del nodo."""
        node_str = self.node_type.value
        
        if self.name:
            node_str += f"({self.name})"
        
        if self.value:
            node_str += f"='{self.value}'"
        
        if self.children:
            node_str += f"[{self.child_count} children]"
        
        return node_str
    
    def __repr__(self) -> str:
        """Representación de debug del nodo."""
        return (
            f"NormalizedNode("
            f"node_type={self.node_type}, "
            f"name={self.name}, "
            f"value={self.value}, "
            f"position={self.position}, "
            f"children={self.children}, "
            f"semantic_info={self.semantic_info}"
            f")"
        )


@dataclass
class NormalizedAST:
    """AST normalizado cross-language."""
    
    root: NormalizedNode
    language: ProgrammingLanguage
    metadata: Dict[str, Any] = field(default_factory=dict)
    symbol_table: Optional[Dict[str, Any]] = None
    comments: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[Dict[str, Any]] = field(default_factory=list)
    exports: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    normalized_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validar el AST normalizado."""
        if self.root is None:
            raise ValueError("El nodo raíz no puede ser None")
        
        if self.language is None:
            raise ValueError("El lenguaje no puede ser None")
    
    @property
    def total_nodes(self) -> int:
        """Obtiene el número total de nodos."""
        return self.root.total_nodes
    
    @property
    def max_depth(self) -> int:
        """Obtiene la profundidad máxima del árbol."""
        return self.root.depth
    
    def find_nodes_by_type(self, node_type: NodeType) -> List[NormalizedNode]:
        """Encuentra todos los nodos de un tipo específico."""
        return self.root.find_nodes_by_type(node_type)
    
    def find_nodes_by_name(self, name: str) -> List[NormalizedNode]:
        """Encuentra todos los nodos con un nombre específico."""
        return self.root.find_nodes_by_name(name)
    
    def get_node_at_position(self, line: int, column: int) -> Optional[NormalizedNode]:
        """Obtiene el nodo en una posición específica."""
        return self.root.get_node_at_position(line, column)
    
    def get_functions(self) -> List[NormalizedNode]:
        """Obtiene todas las funciones declaradas."""
        return self.find_nodes_by_type(NodeType.FUNCTION_DECLARATION)
    
    def get_classes(self) -> List[NormalizedNode]:
        """Obtiene todas las clases declaradas."""
        return self.find_nodes_by_type(NodeType.CLASS_DECLARATION)
    
    def get_variables(self) -> List[NormalizedNode]:
        """Obtiene todas las variables declaradas."""
        return self.find_nodes_by_type(NodeType.VARIABLE_DECLARATION)
    
    def get_imports(self) -> List[NormalizedNode]:
        """Obtiene todas las importaciones."""
        return self.find_nodes_by_type(NodeType.IMPORT_DECLARATION)
    
    def get_exports(self) -> List[NormalizedNode]:
        """Obtiene todas las exportaciones."""
        return self.find_nodes_by_type(NodeType.EXPORT_DECLARATION)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el AST a diccionario."""
        return {
            "language": self.language.value,
            "language_name": self.language.get_name(),
            "root": self.root.to_dict(),
            "total_nodes": self.total_nodes,
            "max_depth": self.max_depth,
            "metadata": self.metadata,
            "symbol_table": self.symbol_table,
            "comments": self.comments,
            "imports": self.imports,
            "exports": self.exports,
            "dependencies": self.dependencies,
            "normalized_at": self.normalized_at.isoformat(),
        }
    
    def __str__(self) -> str:
        """Representación string del AST."""
        return f"NormalizedAST({self.language.get_name()}, {self.total_nodes} nodes, depth {self.max_depth})"
    
    def __repr__(self) -> str:
        """Representación de debug del AST."""
        return (
            f"NormalizedAST("
            f"root={self.root}, "
            f"language={self.language}, "
            f"total_nodes={self.total_nodes}, "
            f"max_depth={self.max_depth}, "
            f"normalized_at={self.normalized_at}"
            f")"
        )
