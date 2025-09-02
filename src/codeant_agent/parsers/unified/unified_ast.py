"""
Estructura Core del AST Unificado Cross-Language.

Este módulo define las estructuras de datos fundamentales para representar
código de múltiples lenguajes de programación en un formato unificado.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class ASTVersion(str, Enum):
    """Versiones del sistema AST unificado."""
    V1 = "1.0.0"


class ASTId(str):
    """Identificador único para un AST."""
    
    def __new__(cls, value: Optional[str] = None):
        if value is None:
            value = str(uuid.uuid4())
        return super().__new__(cls, value)


class NodeId(str):
    """Identificador único para un nodo del AST."""
    
    def __new__(cls, value: Optional[str] = None):
        if value is None:
            value = str(uuid.uuid4())
        return super().__new__(cls, value)


class BinaryOperator(str, Enum):
    """Operadores binarios unificados."""
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "%"
    POWER = "**"
    EQUAL = "=="
    NOT_EQUAL = "!="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    AND = "&&"
    OR = "||"
    BITWISE_AND = "&"
    BITWISE_OR = "|"
    BITWISE_XOR = "^"
    LEFT_SHIFT = "<<"
    RIGHT_SHIFT = ">>"
    ASSIGN = "="
    ADD_ASSIGN = "+="
    SUBTRACT_ASSIGN = "-="
    MULTIPLY_ASSIGN = "*="
    DIVIDE_ASSIGN = "/="
    MODULO_ASSIGN = "%="
    POWER_ASSIGN = "**="
    AND_ASSIGN = "&&="
    OR_ASSIGN = "||="
    BITWISE_AND_ASSIGN = "&="
    BITWISE_OR_ASSIGN = "|="
    BITWISE_XOR_ASSIGN = "^="
    LEFT_SHIFT_ASSIGN = "<<="
    RIGHT_SHIFT_ASSIGN = ">>="
    UNKNOWN = "unknown"


class UnaryOperator(str, Enum):
    """Operadores unarios unificados."""
    POSITIVE = "+"
    NEGATIVE = "-"
    NOT = "!"
    BITWISE_NOT = "~"
    INCREMENT = "++"
    DECREMENT = "--"
    DEREFERENCE = "*"
    ADDRESS_OF = "&"
    UNKNOWN = "unknown"


class Visibility(str, Enum):
    """Visibilidad de elementos."""
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    INTERNAL = "internal"
    PACKAGE = "package"


class CommentType(str, Enum):
    """Tipos de comentarios."""
    LINE = "line"
    BLOCK = "block"
    DOCUMENTATION = "documentation"


class DocumentationType(str, Enum):
    """Tipos de documentación."""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    PARAMETER = "parameter"
    RETURN = "return"
    THROWS = "throws"
    EXAMPLE = "example"
    DEPRECATED = "deprecated"


@dataclass
class UnifiedPosition:
    """Posición unificada en el código."""
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    start_byte: int
    end_byte: int
    file_path: Path


@dataclass
class UnifiedValue:
    """Valor unificado con información de tipo."""
    raw_value: str
    typed_value: 'TypedValue'
    normalized_value: str


@dataclass
class Parameter:
    """Parámetro de función unificado."""
    name: str
    parameter_type: 'UnifiedType'
    is_optional: bool = False
    default_value: Optional[UnifiedValue] = None
    is_variadic: bool = False


@dataclass
class UnifiedType:
    """Tipo unificado que puede representar tipos de múltiples lenguajes."""
    type_name: str
    is_primitive: bool = False
    is_composite: bool = False
    is_function: bool = False
    is_generic: bool = False
    is_reference: bool = False
    is_optional: bool = False
    is_mutable: bool = False
    size: Optional[int] = None
    signed: Optional[bool] = None
    element_type: Optional['UnifiedType'] = None
    type_parameters: List['UnifiedType'] = field(default_factory=list)
    properties: Dict[str, 'UnifiedType'] = field(default_factory=dict)
    parameters: List[Parameter] = field(default_factory=list)
    return_type: Optional['UnifiedType'] = None
    lifetime: Optional[str] = None
    is_async: bool = False
    is_generator: bool = False
    language_specific_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TypedValue:
    """Valor tipado unificado."""
    value_type: str
    string_value: Optional[str] = None
    integer_value: Optional[int] = None
    float_value: Optional[float] = None
    boolean_value: Optional[bool] = None
    array_value: Optional[List['TypedValue']] = None
    object_value: Optional[Dict[str, 'TypedValue']] = None
    function_value: Optional[Dict[str, Any]] = None


class UnifiedNodeType(str, Enum):
    """Tipos de nodos unificados."""
    # Program structure
    PROGRAM = "program"
    MODULE = "module"
    NAMESPACE = "namespace"
    PACKAGE = "package"
    
    # Declarations
    FUNCTION_DECLARATION = "function_declaration"
    CLASS_DECLARATION = "class_declaration"
    INTERFACE_DECLARATION = "interface_declaration"
    STRUCT_DECLARATION = "struct_declaration"
    ENUM_DECLARATION = "enum_declaration"
    TRAIT_DECLARATION = "trait_declaration"
    TYPE_DECLARATION = "type_declaration"
    VARIABLE_DECLARATION = "variable_declaration"
    
    # Statements
    EXPRESSION_STATEMENT = "expression_statement"
    IF_STATEMENT = "if_statement"
    FOR_STATEMENT = "for_statement"
    WHILE_STATEMENT = "while_statement"
    LOOP_STATEMENT = "loop_statement"
    RETURN_STATEMENT = "return_statement"
    BREAK_STATEMENT = "break_statement"
    CONTINUE_STATEMENT = "continue_statement"
    TRY_STATEMENT = "try_statement"
    THROW_STATEMENT = "throw_statement"
    MATCH_STATEMENT = "match_statement"
    
    # Expressions
    BINARY_EXPRESSION = "binary_expression"
    UNARY_EXPRESSION = "unary_expression"
    CALL_EXPRESSION = "call_expression"
    MEMBER_EXPRESSION = "member_expression"
    ASSIGNMENT_EXPRESSION = "assignment_expression"
    CONDITIONAL_EXPRESSION = "conditional_expression"
    ARRAY_EXPRESSION = "array_expression"
    OBJECT_EXPRESSION = "object_expression"
    LAMBDA_EXPRESSION = "lambda_expression"
    
    # Literals
    STRING_LITERAL = "string_literal"
    NUMBER_LITERAL = "number_literal"
    BOOLEAN_LITERAL = "boolean_literal"
    NULL_LITERAL = "null_literal"
    IDENTIFIER = "identifier"
    
    # Comments and documentation
    COMMENT = "comment"
    DOCUMENTATION = "documentation"
    
    # Language-specific nodes
    LANGUAGE_SPECIFIC = "language_specific"


class SemanticNodeType(str, Enum):
    """Tipos semánticos de nodos."""
    DECLARATION = "declaration"
    DEFINITION = "definition"
    REFERENCE = "reference"
    CALL = "call"
    ASSIGNMENT = "assignment"
    CONTROL_FLOW = "control_flow"
    DATA_STRUCTURE = "data_structure"
    TYPE_ANNOTATION = "type_annotation"
    IMPORT = "import"
    EXPORT = "export"
    LITERAL = "literal"
    OPERATOR = "operator"
    COMMENT = "comment"
    UNKNOWN = "unknown"


@dataclass
class UnifiedNode:
    """Nodo unificado del AST."""
    id: NodeId
    node_type: UnifiedNodeType
    semantic_type: SemanticNodeType
    name: Optional[str] = None
    value: Optional[UnifiedValue] = None
    position: Optional[UnifiedPosition] = None
    children: List['UnifiedNode'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    language_specific: Dict[str, Any] = field(default_factory=dict)
    cross_refs: List[str] = field(default_factory=list)
    
    # Additional metadata
    is_async: bool = False
    is_generator: bool = False
    visibility: Visibility = Visibility.PUBLIC
    is_abstract: bool = False
    is_constant: bool = False
    is_mutable: bool = False
    inheritance: List[str] = field(default_factory=list)
    comment_type: Optional[CommentType] = None
    documentation_type: Optional[DocumentationType] = None
    binary_operator: Optional[BinaryOperator] = None
    unary_operator: Optional[UnaryOperator] = None


@dataclass
class UnifiedSemanticInfo:
    """Información semántica unificada."""
    symbols: Dict[str, Any] = field(default_factory=dict)
    scopes: List[Dict[str, Any]] = field(default_factory=list)
    types: Dict[str, UnifiedType] = field(default_factory=dict)
    imports: List[Dict[str, Any]] = field(default_factory=list)
    exports: List[Dict[str, Any]] = field(default_factory=list)
    data_flow: Dict[str, Any] = field(default_factory=dict)
    control_flow: Dict[str, Any] = field(default_factory=dict)
    ownership_info: Optional[Dict[str, Any]] = None
    lifetime_info: Optional[Dict[str, Any]] = None
    trait_info: Optional[Dict[str, Any]] = None
    unsafe_info: Optional[Dict[str, Any]] = None


@dataclass
class CrossLanguageMapping:
    """Mapeo cross-language para conceptos equivalentes."""
    concept: str
    source_language: str
    target_language: str
    source_node_id: NodeId
    target_node_id: NodeId
    confidence: float
    mapping_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedASTMetadata:
    """Metadatos del AST unificado."""
    original_language: str
    parser_used: str
    unification_version: str
    node_count: int
    depth: int
    complexity_score: float
    semantic_features: List[str] = field(default_factory=list)
    cross_language_compatibility: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UnifiedAST:
    """AST unificado que puede representar código de múltiples lenguajes."""
    id: ASTId
    language: str
    file_path: Path
    root_node: UnifiedNode
    metadata: UnifiedASTMetadata
    semantic_info: UnifiedSemanticInfo
    cross_language_mappings: List[CrossLanguageMapping] = field(default_factory=list)
    version: ASTVersion = ASTVersion.V1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Post-inicialización para generar ID si no se proporciona."""
        if isinstance(self.id, str):
            self.id = ASTId(self.id)
        if isinstance(self.root_node.id, str):
            self.root_node.id = NodeId(self.root_node.id)
    
    def get_node_count(self) -> int:
        """Obtiene el número total de nodos en el AST."""
        return self._count_nodes_recursive(self.root_node)
    
    def get_depth(self) -> int:
        """Obtiene la profundidad máxima del AST."""
        return self._calculate_depth_recursive(self.root_node)
    
    def get_complexity_score(self) -> float:
        """Calcula un score de complejidad del AST."""
        return self._calculate_complexity_recursive(self.root_node)
    
    def find_nodes_by_type(self, node_type: UnifiedNodeType) -> List[UnifiedNode]:
        """Encuentra todos los nodos de un tipo específico."""
        nodes = []
        self._find_nodes_recursive(self.root_node, node_type, nodes)
        return nodes
    
    def find_nodes_by_semantic_type(self, semantic_type: SemanticNodeType) -> List[UnifiedNode]:
        """Encuentra todos los nodos de un tipo semántico específico."""
        nodes = []
        self._find_nodes_by_semantic_recursive(self.root_node, semantic_type, nodes)
        return nodes
    
    def find_node_by_id(self, node_id: NodeId) -> Optional[UnifiedNode]:
        """Encuentra un nodo por su ID."""
        return self._find_node_by_id_recursive(self.root_node, node_id)
    
    def get_node_path(self, node_id: NodeId) -> List[UnifiedNode]:
        """Obtiene la ruta desde la raíz hasta un nodo específico."""
        path = []
        self._get_node_path_recursive(self.root_node, node_id, path)
        return path
    
    def _count_nodes_recursive(self, node: UnifiedNode) -> int:
        """Cuenta nodos recursivamente."""
        count = 1
        for child in node.children:
            count += self._count_nodes_recursive(child)
        return count
    
    def _calculate_depth_recursive(self, node: UnifiedNode, current_depth: int = 0) -> int:
        """Calcula profundidad recursivamente."""
        if not node.children:
            return current_depth
        
        max_depth = current_depth
        for child in node.children:
            depth = self._calculate_depth_recursive(child, current_depth + 1)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_complexity_recursive(self, node: UnifiedNode) -> float:
        """Calcula complejidad recursivamente."""
        complexity = 1.0
        
        # Añadir complejidad basada en el tipo de nodo
        if node.node_type in [UnifiedNodeType.IF_STATEMENT, UnifiedNodeType.FOR_STATEMENT, 
                             UnifiedNodeType.WHILE_STATEMENT, UnifiedNodeType.LOOP_STATEMENT]:
            complexity += 2.0
        elif node.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
            complexity += 1.5
        elif node.node_type == UnifiedNodeType.CLASS_DECLARATION:
            complexity += 2.5
        
        # Añadir complejidad de los hijos
        for child in node.children:
            complexity += self._calculate_complexity_recursive(child)
        
        return complexity
    
    def _find_nodes_recursive(self, node: UnifiedNode, node_type: UnifiedNodeType, 
                             nodes: List[UnifiedNode]) -> None:
        """Encuentra nodos por tipo recursivamente."""
        if node.node_type == node_type:
            nodes.append(node)
        
        for child in node.children:
            self._find_nodes_recursive(child, node_type, nodes)
    
    def _find_nodes_by_semantic_recursive(self, node: UnifiedNode, semantic_type: SemanticNodeType,
                                         nodes: List[UnifiedNode]) -> None:
        """Encuentra nodos por tipo semántico recursivamente."""
        if node.semantic_type == semantic_type:
            nodes.append(node)
        
        for child in node.children:
            self._find_nodes_by_semantic_recursive(child, semantic_type, nodes)
    
    def _find_node_by_id_recursive(self, node: UnifiedNode, node_id: NodeId) -> Optional[UnifiedNode]:
        """Encuentra nodo por ID recursivamente."""
        if node.id == node_id:
            return node
        
        for child in node.children:
            result = self._find_node_by_id_recursive(child, node_id)
            if result:
                return result
        
        return None
    
    def _get_node_path_recursive(self, node: UnifiedNode, node_id: NodeId, 
                                path: List[UnifiedNode]) -> bool:
        """Obtiene la ruta a un nodo recursivamente."""
        path.append(node)
        
        if node.id == node_id:
            return True
        
        for child in node.children:
            if self._get_node_path_recursive(child, node_id, path):
                return True
        
        path.pop()
        return False
