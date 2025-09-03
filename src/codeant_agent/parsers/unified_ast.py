"""
Sistema AST Unificado Cross-Language.

Este módulo implementa una representación unificada de AST que permite
análisis cross-language, comparaciones semánticas y detección de patrones
similares entre diferentes lenguajes de programación.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UnifiedNodeType(Enum):
    """Tipos de nodos unificados cross-language."""
    # Estructurales
    MODULE = "module"
    PACKAGE = "package"
    NAMESPACE = "namespace"
    
    # Declaraciones
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    TRAIT = "trait"
    
    # Variables
    VARIABLE = "variable"
    CONSTANT = "constant"
    PARAMETER = "parameter"
    FIELD = "field"
    PROPERTY = "property"
    
    # Control de flujo
    IF_STATEMENT = "if_statement"
    FOR_LOOP = "for_loop"
    WHILE_LOOP = "while_loop"
    MATCH_STATEMENT = "match_statement"
    TRY_CATCH = "try_catch"
    RETURN = "return"
    
    # Expresiones
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    CALL = "call"
    ASSIGNMENT = "assignment"
    LITERAL = "literal"
    IDENTIFIER = "identifier"
    
    # Tipos
    TYPE_ANNOTATION = "type_annotation"
    GENERIC = "generic"
    UNION_TYPE = "union_type"
    INTERSECTION_TYPE = "intersection_type"
    
    # Imports/Exports
    IMPORT = "import"
    EXPORT = "export"
    
    # Otros
    COMMENT = "comment"
    DECORATOR = "decorator"
    ANNOTATION = "annotation"


class SemanticConcept(Enum):
    """Conceptos semánticos cross-language."""
    # Paradigmas
    OBJECT_ORIENTED = "object_oriented"
    FUNCTIONAL = "functional"
    PROCEDURAL = "procedural"
    
    # Patrones de diseño
    SINGLETON = "singleton"
    FACTORY = "factory"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    
    # Características
    ASYNC_OPERATION = "async_operation"
    ERROR_HANDLING = "error_handling"
    MEMORY_MANAGEMENT = "memory_management"
    TYPE_SAFETY = "type_safety"
    IMMUTABILITY = "immutability"
    
    # Operaciones
    ITERATION = "iteration"
    MAPPING = "mapping"
    FILTERING = "filtering"
    REDUCTION = "reduction"
    COMPOSITION = "composition"


@dataclass
class UnifiedType:
    """Representación unificada de tipos."""
    name: str
    category: str  # primitive, object, function, generic, etc.
    is_nullable: bool = False
    is_array: bool = False
    is_async: bool = False
    generics: List['UnifiedType'] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return {
            'name': self.name,
            'category': self.category,
            'is_nullable': self.is_nullable,
            'is_array': self.is_array,
            'is_async': self.is_async,
            'generics': [g.to_dict() for g in self.generics],
            'constraints': self.constraints
        }


@dataclass
class UnifiedNode:
    """Nodo unificado del AST."""
    node_type: UnifiedNodeType
    name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    start_col: int = 0
    end_col: int = 0
    
    # Información semántica
    semantic_concepts: Set[SemanticConcept] = field(default_factory=set)
    unified_type: Optional[UnifiedType] = None
    
    # Relaciones
    parent: Optional['UnifiedNode'] = None
    children: List['UnifiedNode'] = field(default_factory=list)
    
    # Metadatos
    source_language: Optional[str] = None
    original_node: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'UnifiedNode'):
        """Añade un nodo hijo."""
        child.parent = self
        self.children.append(child)
    
    def find_children_by_type(self, node_type: UnifiedNodeType) -> List['UnifiedNode']:
        """Encuentra todos los hijos de un tipo específico."""
        result = []
        for child in self.children:
            if child.node_type == node_type:
                result.append(child)
            result.extend(child.find_children_by_type(node_type))
        return result
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return {
            'node_type': self.node_type.value,
            'name': self.name,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'semantic_concepts': [c.value for c in self.semantic_concepts],
            'unified_type': self.unified_type.to_dict() if self.unified_type else None,
            'source_language': self.source_language,
            'children': [child.to_dict() for child in self.children],
            'metadata': self.metadata
        }


@dataclass
class UnifiedAST:
    """AST unificado completo."""
    root: UnifiedNode
    file_path: Path
    source_language: str
    
    # Índices para búsqueda rápida
    nodes_by_type: Dict[UnifiedNodeType, List[UnifiedNode]] = field(default_factory=dict)
    nodes_by_name: Dict[str, List[UnifiedNode]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Construye índices después de la inicialización."""
        self._build_indices()
    
    def _build_indices(self):
        """Construye índices para búsqueda eficiente."""
        self.nodes_by_type.clear()
        self.nodes_by_name.clear()
        self._index_node(self.root)
    
    def _index_node(self, node: UnifiedNode):
        """Indexa un nodo y sus hijos."""
        # Índice por tipo
        if node.node_type not in self.nodes_by_type:
            self.nodes_by_type[node.node_type] = []
        self.nodes_by_type[node.node_type].append(node)
        
        # Índice por nombre
        if node.name:
            if node.name not in self.nodes_by_name:
                self.nodes_by_name[node.name] = []
            self.nodes_by_name[node.name].append(node)
        
        # Indexar hijos
        for child in node.children:
            self._index_node(child)
    
    def find_by_type(self, node_type: UnifiedNodeType) -> List[UnifiedNode]:
        """Encuentra todos los nodos de un tipo."""
        return self.nodes_by_type.get(node_type, [])
    
    def find_by_name(self, name: str) -> List[UnifiedNode]:
        """Encuentra todos los nodos con un nombre."""
        return self.nodes_by_name.get(name, [])
    
    def find_by_concept(self, concept: SemanticConcept) -> List[UnifiedNode]:
        """Encuentra todos los nodos con un concepto semántico."""
        result = []
        self._find_by_concept_recursive(self.root, concept, result)
        return result
    
    def _find_by_concept_recursive(self, node: UnifiedNode, concept: SemanticConcept, result: List[UnifiedNode]):
        """Búsqueda recursiva por concepto semántico."""
        if concept in node.semantic_concepts:
            result.append(node)
        for child in node.children:
            self._find_by_concept_recursive(child, concept, result)
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return {
            'file_path': str(self.file_path),
            'source_language': self.source_language,
            'root': self.root.to_dict(),
            'statistics': {
                'total_nodes': sum(len(nodes) for nodes in self.nodes_by_type.values()),
                'node_types': {k.value: len(v) for k, v in self.nodes_by_type.items()},
                'unique_names': len(self.nodes_by_name)
            }
        }


class ASTUnifier(ABC):
    """Clase base para unificadores de AST específicos por lenguaje."""
    
    @abstractmethod
    def unify(self, source_ast: Any, file_path: Path) -> UnifiedAST:
        """Convierte un AST específico del lenguaje a UnifiedAST."""
        pass
    
    @abstractmethod
    def supported_language(self) -> str:
        """Retorna el lenguaje soportado."""
        pass


class CrossLanguageAnalyzer:
    """Analizador cross-language usando AST unificado."""
    
    def __init__(self):
        self.unifiers: Dict[str, ASTUnifier] = {}
    
    def register_unifier(self, language: str, unifier: ASTUnifier):
        """Registra un unificador para un lenguaje."""
        self.unifiers[language] = unifier
        logger.info(f"Unificador registrado para {language}")
    
    def analyze_similarity(self, ast1: UnifiedAST, ast2: UnifiedAST) -> float:
        """Calcula similitud estructural entre dos ASTs."""
        # Comparar tipos de nodos
        types1 = set(ast1.nodes_by_type.keys())
        types2 = set(ast2.nodes_by_type.keys())
        
        if not types1 or not types2:
            return 0.0
        
        # Jaccard similarity para tipos de nodos
        intersection = types1.intersection(types2)
        union = types1.union(types2)
        type_similarity = len(intersection) / len(union) if union else 0
        
        # Comparar estructura
        structure_similarity = self._compare_structure(ast1.root, ast2.root)
        
        # Comparar conceptos semánticos
        concepts1 = set()
        concepts2 = set()
        for concept in SemanticConcept:
            if ast1.find_by_concept(concept):
                concepts1.add(concept)
            if ast2.find_by_concept(concept):
                concepts2.add(concept)
        
        concept_similarity = 0
        if concepts1 or concepts2:
            concept_intersection = concepts1.intersection(concepts2)
            concept_union = concepts1.union(concepts2)
            concept_similarity = len(concept_intersection) / len(concept_union) if concept_union else 0
        
        # Promedio ponderado
        return (type_similarity * 0.3 + structure_similarity * 0.5 + concept_similarity * 0.2)
    
    def _compare_structure(self, node1: UnifiedNode, node2: UnifiedNode) -> float:
        """Compara la estructura de dos nodos."""
        if node1.node_type != node2.node_type:
            return 0.0
        
        # Comparar número de hijos
        if not node1.children and not node2.children:
            return 1.0
        
        if len(node1.children) == 0 or len(node2.children) == 0:
            return 0.0
        
        # Comparar hijos recursivamente
        max_children = max(len(node1.children), len(node2.children))
        min_children = min(len(node1.children), len(node2.children))
        
        similarity_sum = 0
        for i in range(min_children):
            similarity_sum += self._compare_structure(node1.children[i], node2.children[i])
        
        return similarity_sum / max_children
    
    def find_equivalent_concepts(self, ast1: UnifiedAST, ast2: UnifiedAST) -> List[tuple]:
        """Encuentra conceptos equivalentes entre dos ASTs."""
        equivalents = []
        
        # Buscar funciones con nombres similares
        for name1, nodes1 in ast1.nodes_by_name.items():
            if name1 in ast2.nodes_by_name:
                nodes2 = ast2.nodes_by_name[name1]
                for n1 in nodes1:
                    for n2 in nodes2:
                        if n1.node_type == n2.node_type:
                            equivalents.append((n1, n2, 'same_name_and_type'))
        
        # Buscar patrones similares
        for node_type in UnifiedNodeType:
            nodes1 = ast1.find_by_type(node_type)
            nodes2 = ast2.find_by_type(node_type)
            
            for n1 in nodes1:
                for n2 in nodes2:
                    similarity = self._calculate_node_similarity(n1, n2)
                    if similarity > 0.7:  # Umbral de similitud
                        equivalents.append((n1, n2, f'similar_{similarity:.2f}'))
        
        return equivalents
    
    def _calculate_node_similarity(self, node1: UnifiedNode, node2: UnifiedNode) -> float:
        """Calcula similitud entre dos nodos."""
        if node1.node_type != node2.node_type:
            return 0.0
        
        score = 0.5  # Base score por mismo tipo
        
        # Bonus por mismo nombre
        if node1.name and node2.name and node1.name == node2.name:
            score += 0.2
        
        # Bonus por conceptos semánticos compartidos
        shared_concepts = node1.semantic_concepts.intersection(node2.semantic_concepts)
        if shared_concepts:
            score += 0.3 * (len(shared_concepts) / max(len(node1.semantic_concepts), len(node2.semantic_concepts)))
        
        return min(score, 1.0)
    
    def suggest_translation(self, node: UnifiedNode, target_language: str) -> Optional[str]:
        """Sugiere una traducción de un nodo a otro lenguaje."""
        # Mapeo básico de conceptos entre lenguajes
        translations = {
            ('python', 'typescript'): {
                'def': 'function',
                'class': 'class',
                'if': 'if',
                'for': 'for',
                'while': 'while',
                'return': 'return',
                'import': 'import',
                'from': 'from',
                'None': 'null',
                'True': 'true',
                'False': 'false',
            },
            ('typescript', 'rust'): {
                'function': 'fn',
                'class': 'struct',
                'interface': 'trait',
                'let': 'let',
                'const': 'const',
                'if': 'if',
                'for': 'for',
                'while': 'while',
                'return': 'return',
                'import': 'use',
                'null': 'None',
                'undefined': 'None',
            },
            ('python', 'rust'): {
                'def': 'fn',
                'class': 'struct',
                'if': 'if',
                'for': 'for',
                'while': 'while',
                'return': 'return',
                'import': 'use',
                'None': 'None',
                'True': 'true',
                'False': 'false',
            }
        }
        
        # Obtener mapeo para el par de lenguajes
        key = (node.source_language, target_language)
        if key not in translations:
            # Intentar invertido
            reverse_key = (target_language, node.source_language)
            if reverse_key in translations:
                # Invertir el mapeo
                translations[key] = {v: k for k, v in translations[reverse_key].items()}
            else:
                return None
        
        mapping = translations[key]
        
        # Generar sugerencia básica
        if node.node_type == UnifiedNodeType.FUNCTION:
            if target_language == 'typescript':
                return f"function {node.name}() {{\n  // TODO: Implement\n}}"
            elif target_language == 'rust':
                return f"fn {node.name}() {{\n    // TODO: Implement\n}}"
            elif target_language == 'python':
                return f"def {node.name}():\n    # TODO: Implement\n    pass"
        
        return None


# Funciones de utilidad
def create_unified_ast(root_node: UnifiedNode, file_path: Path, language: str) -> UnifiedAST:
    """Crea un UnifiedAST."""
    return UnifiedAST(root=root_node, file_path=file_path, source_language=language)


def merge_asts(ast1: UnifiedAST, ast2: UnifiedAST) -> UnifiedAST:
    """Fusiona dos ASTs unificados."""
    # Crear nuevo root
    merged_root = UnifiedNode(
        node_type=UnifiedNodeType.MODULE,
        name="merged_module",
        source_language="mixed"
    )
    
    # Copiar nodos de ambos ASTs
    for child in ast1.root.children:
        merged_root.add_child(child)
    
    for child in ast2.root.children:
        merged_root.add_child(child)
    
    return UnifiedAST(
        root=merged_root,
        file_path=Path("merged.ast"),
        source_language="mixed"
    )


__all__ = [
    'UnifiedNodeType',
    'SemanticConcept',
    'UnifiedType',
    'UnifiedNode',
    'UnifiedAST',
    'ASTUnifier',
    'CrossLanguageAnalyzer',
    'create_unified_ast',
    'merge_asts'
]
