"""
Entidades del dominio para análisis de dependencias.

Este módulo contiene las entidades que representan grafos de dependencias,
análisis de flujo de datos y análisis de reachability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum
from pathlib import Path

from ..value_objects.programming_language import ProgrammingLanguage
from .dead_code_analysis import SourceRange, SourcePosition


class NodeType(Enum):
    """Tipos de nodos en el grafo de control de flujo."""
    ENTRY = "entry"
    EXIT = "exit"
    STATEMENT = "statement"
    EXPRESSION = "expression"
    CONDITION = "condition"
    LOOP = "loop"
    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"
    THROW = "throw"
    CALL = "call"
    ASSIGNMENT = "assignment"


class EdgeType(Enum):
    """Tipos de aristas en el grafo de control de flujo."""
    SEQUENTIAL = "sequential"
    CONDITIONAL_TRUE = "conditional_true"
    CONDITIONAL_FALSE = "conditional_false"
    LOOP_ENTRY = "loop_entry"
    LOOP_BACK = "loop_back"
    EXCEPTION = "exception"
    CALL = "call"
    RETURN = "return"


class DependencyType(Enum):
    """Tipos de dependencia entre símbolos."""
    CALLS = "calls"
    REFERENCES = "references"
    INHERITS = "inherits"
    IMPORTS = "imports"
    USES_TYPE = "uses_type"
    CONTAINS = "contains"
    OVERRIDES = "overrides"
    IMPLEMENTS = "implements"


class SymbolType(Enum):
    """Tipos de símbolos en el código."""
    VARIABLE = "variable"
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    PROPERTY = "property"
    MODULE = "module"
    INTERFACE = "interface"
    ENUM = "enum"
    CONSTANT = "constant"
    NAMESPACE = "namespace"


@dataclass
class NodeId:
    """Identificador único de nodo."""
    value: int

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        return isinstance(other, NodeId) and self.value == other.value


@dataclass
class SymbolId:
    """Identificador único de símbolo."""
    module_path: str
    symbol_name: str
    symbol_type: SymbolType

    def __hash__(self) -> int:
        return hash((self.module_path, self.symbol_name, self.symbol_type))

    def __eq__(self, other) -> bool:
        return (isinstance(other, SymbolId) and
                self.module_path == other.module_path and
                self.symbol_name == other.symbol_name and
                self.symbol_type == other.symbol_type)

    def __str__(self) -> str:
        return f"{self.module_path}::{self.symbol_name}({self.symbol_type.value})"


@dataclass
class ModuleId:
    """Identificador único de módulo."""
    path: str

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other) -> bool:
        return isinstance(other, ModuleId) and self.path == other.path


@dataclass
class ControlFlowNode:
    """Nodo en el grafo de control de flujo."""
    id: NodeId
    node_type: NodeType
    position: SourceRange
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Verifica si es un nodo terminal."""
        return self.node_type in {NodeType.EXIT, NodeType.RETURN, NodeType.THROW}

    def is_conditional(self) -> bool:
        """Verifica si es un nodo condicional."""
        return self.node_type == NodeType.CONDITION


@dataclass
class ControlFlowEdge:
    """Arista en el grafo de control de flujo."""
    source: NodeId
    target: NodeId
    edge_type: EdgeType
    condition: Optional[str] = None
    probability: float = 1.0

    def is_conditional(self) -> bool:
        """Verifica si es una arista condicional."""
        return self.edge_type in {EdgeType.CONDITIONAL_TRUE, EdgeType.CONDITIONAL_FALSE}


@dataclass
class ControlFlowGraph:
    """Grafo de control de flujo."""
    nodes: Dict[NodeId, ControlFlowNode] = field(default_factory=dict)
    edges: List[ControlFlowEdge] = field(default_factory=list)
    entry_node: Optional[NodeId] = None
    exit_nodes: Set[NodeId] = field(default_factory=set)

    def add_node(self, node: ControlFlowNode) -> None:
        """Añade un nodo al grafo."""
        self.nodes[node.id] = node
        if node.node_type == NodeType.ENTRY:
            self.entry_node = node.id
        elif node.is_terminal():
            self.exit_nodes.add(node.id)

    def add_edge(self, edge: ControlFlowEdge) -> None:
        """Añade una arista al grafo."""
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("Los nodos de la arista deben existir en el grafo")
        self.edges.append(edge)

    def get_successors(self, node_id: NodeId) -> List[NodeId]:
        """Obtiene los sucesores de un nodo."""
        return [edge.target for edge in self.edges if edge.source == node_id]

    def get_predecessors(self, node_id: NodeId) -> List[NodeId]:
        """Obtiene los predecesores de un nodo."""
        return [edge.source for edge in self.edges if edge.target == node_id]

    def get_node(self, node_id: NodeId) -> Optional[ControlFlowNode]:
        """Obtiene un nodo por su ID."""
        return self.nodes.get(node_id)

    def get_all_nodes(self) -> List[NodeId]:
        """Obtiene todos los IDs de nodos."""
        return list(self.nodes.keys())

    def get_entry_node(self) -> Optional[NodeId]:
        """Obtiene el nodo de entrada."""
        return self.entry_node

    def is_reachable(self, from_node: NodeId, to_node: NodeId) -> bool:
        """Verifica si un nodo es alcanzable desde otro."""
        if from_node == to_node:
            return True

        visited = set()
        stack = [from_node]

        while stack:
            current = stack.pop()
            if current == to_node:
                return True
            
            if current in visited:
                continue
            
            visited.add(current)
            stack.extend(self.get_successors(current))

        return False


@dataclass
class SymbolInfo:
    """Información sobre un símbolo."""
    id: SymbolId
    name: str
    symbol_type: SymbolType
    location: SourceRange
    visibility: str = "private"
    is_exported: bool = False
    is_imported: bool = False
    module_id: Optional[ModuleId] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_qualified_name(self) -> str:
        """Obtiene el nombre calificado del símbolo."""
        if self.module_id:
            return f"{self.module_id.path}::{self.name}"
        return self.name


@dataclass
class Dependency:
    """Dependencia entre símbolos."""
    source: SymbolId
    target: SymbolId
    dependency_type: DependencyType
    location: SourceRange
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleInfo:
    """Información sobre un módulo."""
    id: ModuleId
    path: Path
    language: ProgrammingLanguage
    symbols: Set[SymbolId] = field(default_factory=set)
    exports: Set[SymbolId] = field(default_factory=set)
    imports: Set[SymbolId] = field(default_factory=set)
    dependencies: Set[ModuleId] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_symbol(self, symbol: SymbolId) -> None:
        """Añade un símbolo al módulo."""
        self.symbols.add(symbol)

    def add_export(self, symbol: SymbolId) -> None:
        """Añade un símbolo exportado."""
        self.exports.add(symbol)
        self.add_symbol(symbol)

    def add_import(self, symbol: SymbolId) -> None:
        """Añade un símbolo importado."""
        self.imports.add(symbol)


@dataclass
class DependencyGraph:
    """Grafo de dependencias entre símbolos."""
    symbols: Dict[SymbolId, SymbolInfo] = field(default_factory=dict)
    dependencies: List[Dependency] = field(default_factory=list)
    
    def add_symbol(self, symbol: SymbolInfo) -> None:
        """Añade un símbolo al grafo."""
        self.symbols[symbol.id] = symbol

    def add_dependency(self, dependency: Dependency) -> None:
        """Añade una dependencia al grafo."""
        if dependency.source not in self.symbols:
            raise ValueError(f"Símbolo fuente {dependency.source} no existe")
        if dependency.target not in self.symbols:
            raise ValueError(f"Símbolo objetivo {dependency.target} no existe")
        self.dependencies.append(dependency)

    def get_dependencies(self, symbol_id: SymbolId) -> List[SymbolId]:
        """Obtiene las dependencias de un símbolo."""
        return [dep.target for dep in self.dependencies if dep.source == symbol_id]

    def get_dependents(self, symbol_id: SymbolId) -> List[SymbolId]:
        """Obtiene los dependientes de un símbolo."""
        return [dep.source for dep in self.dependencies if dep.target == symbol_id]

    def is_reachable(self, from_symbol: SymbolId, to_symbol: SymbolId) -> bool:
        """Verifica si un símbolo es alcanzable desde otro."""
        if from_symbol == to_symbol:
            return True

        visited = set()
        stack = [from_symbol]

        while stack:
            current = stack.pop()
            if current == to_symbol:
                return True
            
            if current in visited:
                continue
            
            visited.add(current)
            stack.extend(self.get_dependencies(current))

        return False

    def find_cycles(self) -> List[List[SymbolId]]:
        """Encuentra ciclos en el grafo de dependencias."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(symbol: SymbolId) -> None:
            if symbol in rec_stack:
                # Encontrado un ciclo
                cycle_start = path.index(symbol)
                cycles.append(path[cycle_start:] + [symbol])
                return

            if symbol in visited:
                return

            visited.add(symbol)
            rec_stack.add(symbol)
            path.append(symbol)

            for dependency in self.get_dependencies(symbol):
                dfs(dependency)

            path.pop()
            rec_stack.remove(symbol)

        for symbol_id in self.symbols:
            if symbol_id not in visited:
                dfs(symbol_id)

        return cycles


@dataclass
class GlobalDependencyGraph:
    """Grafo global de dependencias para todo el proyecto."""
    modules: Dict[ModuleId, ModuleInfo] = field(default_factory=dict)
    symbols: Dict[SymbolId, SymbolInfo] = field(default_factory=dict)
    dependencies: Dict[SymbolId, List[SymbolId]] = field(default_factory=dict)
    reverse_dependencies: Dict[SymbolId, List[SymbolId]] = field(default_factory=dict)

    def add_symbol(self, module_id: ModuleId, symbol: SymbolInfo) -> None:
        """Añade un símbolo al grafo global."""
        self.symbols[symbol.id] = symbol
        
        if module_id not in self.modules:
            raise ValueError(f"Módulo {module_id} no existe")
        
        self.modules[module_id].add_symbol(symbol.id)

    def add_dependency(self, source: SymbolId, target: SymbolId) -> None:
        """Añade una dependencia al grafo global."""
        if source not in self.dependencies:
            self.dependencies[source] = []
        self.dependencies[source].append(target)

        if target not in self.reverse_dependencies:
            self.reverse_dependencies[target] = []
        self.reverse_dependencies[target].append(source)

    def get_dependencies(self, symbol_id: SymbolId) -> Optional[List[SymbolId]]:
        """Obtiene las dependencias de un símbolo."""
        return self.dependencies.get(symbol_id)

    def get_dependents(self, symbol_id: SymbolId) -> Optional[List[SymbolId]]:
        """Obtiene los dependientes de un símbolo."""
        return self.reverse_dependencies.get(symbol_id)

    def add_module(self, module: ModuleInfo) -> None:
        """Añade un módulo al grafo."""
        self.modules[module.id] = module


@dataclass
class LivenessInfo:
    """Información de liveness de variables."""
    live_in: Dict[NodeId, Set[str]] = field(default_factory=dict)
    live_out: Dict[NodeId, Set[str]] = field(default_factory=dict)

    def get_live_in(self, node_id: NodeId) -> Set[str]:
        """Obtiene variables live al entrar al nodo."""
        return self.live_in.get(node_id, set())

    def get_live_out(self, node_id: NodeId) -> Set[str]:
        """Obtiene variables live al salir del nodo."""
        return self.live_out.get(node_id, set())

    def is_live(self, node_id: NodeId, variable: str) -> bool:
        """Verifica si una variable está live en un nodo."""
        return variable in self.get_live_out(node_id)


@dataclass
class DefUseChain:
    """Cadena de definición-uso de una variable."""
    definition: NodeId
    uses: List[NodeId] = field(default_factory=list)
    variable_name: str = ""

    def has_uses(self) -> bool:
        """Verifica si la definición tiene usos."""
        return len(self.uses) > 0


@dataclass
class UsageLocation:
    """Ubicación de uso de un símbolo."""
    location: SourceRange
    usage_type: str  # call, reference, assignment, etc.
    context: Optional[str] = None


@dataclass
class UsageAnalysis:
    """Análisis de uso de un símbolo."""
    symbol_id: SymbolId
    total_usages: int = 0
    usage_locations: List[UsageLocation] = field(default_factory=list)
    is_exported: bool = False
    is_public_api: bool = False
    cross_module_usages: int = 0

    def is_unused(self) -> bool:
        """Verifica si el símbolo no está usado."""
        return self.total_usages == 0 and not self.is_public_api

    def is_only_locally_used(self) -> bool:
        """Verifica si solo se usa localmente."""
        return self.total_usages > 0 and self.cross_module_usages == 0
