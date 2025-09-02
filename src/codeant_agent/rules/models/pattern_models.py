"""
Modelos para el sistema de pattern matching del motor de reglas.

Este módulo define las estructuras de datos para el sistema de patrones AST,
incluyendo selectores de nodos, restricciones y coincidencias de patrones.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from ...parsers.unified.unified_ast import (
    UnifiedNode,
    UnifiedNodeType,
    SemanticNodeType,
    UnifiedPosition,
)


class PatternType(str, Enum):
    """Tipos de patrones AST."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    BEHAVIORAL = "behavioral"


class Quantifier(str, Enum):
    """Cuantificadores para restricciones de patrones."""
    ZERO_OR_ONE = "zero_or_one"  # ?
    ZERO_OR_MORE = "zero_or_more"  # *
    ONE_OR_MORE = "one_or_more"  # +
    EXACTLY_ONE = "exactly_one"  # {1}
    EXACTLY_N = "exactly_n"  # {n}
    BETWEEN = "between"  # {n,m}


class SiblingDirection(str, Enum):
    """Dirección de hermanos en el AST."""
    PREVIOUS = "previous"
    NEXT = "next"
    ANY = "any"


class ScopeType(str, Enum):
    """Tipos de scope para restricciones."""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    FILE = "file"
    NAMESPACE = "namespace"


class EquivalenceType(str, Enum):
    """Tipos de equivalencia cross-language."""
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    BEHAVIORAL = "behavioral"
    FUNCTIONAL = "functional"


@dataclass
class AttributeFilter:
    """Filtro de atributos para nodos."""
    name: str
    value_pattern: Optional[str] = None
    regex: bool = False
    case_sensitive: bool = True
    
    def matches(self, node: UnifiedNode) -> bool:
        """Verificar si el nodo coincide con el filtro."""
        if not hasattr(node, 'attributes') or not node.attributes:
            return False
        
        if self.name not in node.attributes:
            return False
        
        if self.value_pattern is None:
            return True
        
        attr_value = str(node.attributes[self.name])
        
        if self.regex:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            return bool(re.search(self.value_pattern, attr_value, flags))
        else:
            if self.case_sensitive:
                return attr_value == self.value_pattern
            else:
                return attr_value.lower() == self.value_pattern.lower()


@dataclass
class PositionConstraint:
    """Restricción de posición para nodos."""
    min_line: Optional[int] = None
    max_line: Optional[int] = None
    min_column: Optional[int] = None
    max_column: Optional[int] = None
    file_pattern: Optional[str] = None
    
    def matches(self, node: UnifiedNode) -> bool:
        """Verificar si el nodo coincide con la restricción de posición."""
        if not node.position:
            return True
        
        if self.min_line is not None and node.position.start_line < self.min_line:
            return False
        
        if self.max_line is not None and node.position.end_line > self.max_line:
            return False
        
        if self.min_column is not None and node.position.start_column < self.min_column:
            return False
        
        if self.max_column is not None and node.position.end_column > self.max_column:
            return False
        
        if self.file_pattern and node.position.file_path:
            return bool(re.search(self.file_pattern, str(node.position.file_path)))
        
        return True


@dataclass
class NodeSelector:
    """Selector de nodos para patrones AST."""
    node_type: Optional[UnifiedNodeType] = None
    semantic_type: Optional[SemanticNodeType] = None
    name_pattern: Optional[str] = None
    value_pattern: Optional[str] = None
    attribute_filters: Dict[str, AttributeFilter] = field(default_factory=dict)
    position_constraints: Optional[PositionConstraint] = None
    depth_range: Optional[tuple[int, int]] = None
    parent_selector: Optional['NodeSelector'] = None
    child_selectors: List['NodeSelector'] = field(default_factory=list)
    
    def matches(self, node: UnifiedNode, depth: int = 0) -> bool:
        """Verificar si el nodo coincide con el selector."""
        # Verificar tipo de nodo
        if self.node_type is not None and node.node_type != self.node_type:
            return False
        
        # Verificar tipo semántico
        if self.semantic_type is not None and node.semantic_type != self.semantic_type:
            return False
        
        # Verificar patrón de nombre
        if self.name_pattern and node.name:
            if not re.search(self.name_pattern, node.name):
                return False
        
        # Verificar patrón de valor
        if self.value_pattern and node.value:
            value_str = str(node.value.raw_value if hasattr(node.value, 'raw_value') else node.value)
            if not re.search(self.value_pattern, value_str):
                return False
        
        # Verificar filtros de atributos
        for attr_filter in self.attribute_filters.values():
            if not attr_filter.matches(node):
                return False
        
        # Verificar restricciones de posición
        if self.position_constraints and not self.position_constraints.matches(node):
            return False
        
        # Verificar rango de profundidad
        if self.depth_range:
            min_depth, max_depth = self.depth_range
            if depth < min_depth or depth > max_depth:
                return False
        
        return True


@dataclass
class PatternConstraint:
    """Restricción para patrones AST."""
    constraint_type: str
    selector: Optional[NodeSelector] = None
    quantifier: Quantifier = Quantifier.EXACTLY_ONE
    direction: SiblingDirection = SiblingDirection.ANY
    scope_type: Optional[ScopeType] = None
    attribute_name: Optional[str] = None
    attribute_value_pattern: Optional[str] = None
    predicate_name: Optional[str] = None
    predicate_parameters: List[Any] = field(default_factory=list)
    target_languages: List[str] = field(default_factory=list)
    equivalence_type: Optional[EquivalenceType] = None
    
    def __post_init__(self):
        """Validar la restricción según su tipo."""
        if self.constraint_type == "has_child" and not self.selector:
            raise ValueError("has_child constraint requires a selector")
        elif self.constraint_type == "has_parent" and not self.selector:
            raise ValueError("has_parent constraint requires a selector")
        elif self.constraint_type == "has_sibling" and not self.selector:
            raise ValueError("has_sibling constraint requires a selector")
        elif self.constraint_type == "in_scope" and not self.scope_type:
            raise ValueError("in_scope constraint requires a scope_type")
        elif self.constraint_type == "has_attribute" and not self.attribute_name:
            raise ValueError("has_attribute constraint requires an attribute_name")
        elif self.constraint_type == "custom_predicate" and not self.predicate_name:
            raise ValueError("custom_predicate constraint requires a predicate_name")
        elif self.constraint_type == "cross_language_equivalent" and not self.target_languages:
            raise ValueError("cross_language_equivalent constraint requires target_languages")


@dataclass
class CaptureGroup:
    """Grupo de captura para patrones."""
    name: str
    selector: NodeSelector
    optional: bool = False
    multiple: bool = False
    transform: Optional[str] = None  # Función de transformación
    
    def capture(self, nodes: List[UnifiedNode]) -> Dict[str, Any]:
        """Capturar nodos que coinciden con el selector."""
        captured = []
        
        for node in nodes:
            if self.selector.matches(node):
                captured.append(node)
        
        if not self.multiple and len(captured) > 1:
            captured = captured[:1]
        
        if not self.optional and not captured:
            return {}
        
        result = {self.name: captured}
        
        # Aplicar transformación si está definida
        if self.transform and captured:
            result[f"{self.name}_transformed"] = self.apply_transform(captured)
        
        return result
    
    def apply_transform(self, nodes: List[UnifiedNode]) -> List[Any]:
        """Aplicar transformación a los nodos capturados."""
        if not self.transform:
            return nodes
        
        # Transformaciones básicas predefinidas
        if self.transform == "names":
            return [node.name for node in nodes if node.name]
        elif self.transform == "values":
            return [node.value for node in nodes if node.value]
        elif self.transform == "positions":
            return [node.position for node in nodes if node.position]
        elif self.transform == "types":
            return [node.node_type for node in nodes]
        else:
            # Transformación personalizada (se implementaría con un sistema de plugins)
            return nodes


@dataclass
class ASTPattern:
    """Patrón AST para detección de reglas."""
    pattern_type: PatternType
    node_selector: NodeSelector
    constraints: List[PatternConstraint] = field(default_factory=list)
    capture_groups: Dict[str, CaptureGroup] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validar el patrón."""
        if not self.node_selector:
            raise ValueError("ASTPattern requires a node_selector")
        
        # Validar que los grupos de captura tengan nombres únicos
        capture_names = set()
        for group in self.capture_groups.values():
            if group.name in capture_names:
                raise ValueError(f"Duplicate capture group name: {group.name}")
            capture_names.add(group.name)


@dataclass
class PatternMatch:
    """Coincidencia de un patrón AST."""
    pattern: ASTPattern
    matched_nodes: List[UnifiedNode] = field(default_factory=list)
    captured_groups: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: UnifiedNode):
        """Añadir un nodo a la coincidencia."""
        self.matched_nodes.append(node)
    
    def add_capture(self, group_name: str, value: Any):
        """Añadir un valor capturado."""
        self.captured_groups[group_name] = value
    
    def get_captured_value(self, group_name: str, default: Any = None) -> Any:
        """Obtener un valor capturado."""
        return self.captured_groups.get(group_name, default)
    
    def get_first_node(self) -> Optional[UnifiedNode]:
        """Obtener el primer nodo coincidente."""
        return self.matched_nodes[0] if self.matched_nodes else None
    
    def get_node_count(self) -> int:
        """Obtener el número de nodos coincidentes."""
        return len(self.matched_nodes)
    
    def is_empty(self) -> bool:
        """Verificar si la coincidencia está vacía."""
        return len(self.matched_nodes) == 0


@dataclass
class PatternMatchResult:
    """Resultado de la búsqueda de patrones."""
    pattern: ASTPattern
    matches: List[PatternMatch] = field(default_factory=list)
    total_matches: int = 0
    execution_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def add_match(self, match: PatternMatch):
        """Añadir una coincidencia al resultado."""
        self.matches.append(match)
        self.total_matches += 1
    
    def get_all_nodes(self) -> List[UnifiedNode]:
        """Obtener todos los nodos de todas las coincidencias."""
        all_nodes = []
        for match in self.matches:
            all_nodes.extend(match.matched_nodes)
        return all_nodes
    
    def get_unique_nodes(self) -> List[UnifiedNode]:
        """Obtener nodos únicos de todas las coincidencias."""
        seen = set()
        unique_nodes = []
        
        for match in self.matches:
            for node in match.matched_nodes:
                node_id = getattr(node, 'id', id(node))
                if node_id not in seen:
                    seen.add(node_id)
                    unique_nodes.append(node)
        
        return unique_nodes
