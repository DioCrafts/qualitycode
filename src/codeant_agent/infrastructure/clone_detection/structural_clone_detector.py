"""
Implementación del detector de clones estructurales.

Este módulo implementa la detección de clones estructurales (Type 2/3) usando
análisis AST, tree edit distance y comparación de estructuras sintácticas.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time
from pathlib import Path

from ...domain.entities.clone_analysis import (
    StructuralClone, CodeLocation, CloneId, CloneType,
    StructuralDifference, NodeMapping, SimilarityMetrics, SimilarityAlgorithm
)
from ...domain.entities.parse_result import ParseResult
from ...domain.entities.ast_normalization import NormalizedNode as UnifiedNode, NodeType as UnifiedNodeType, SourcePosition
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class ASTSubtree:
    """Representación de un subárbol AST para comparación."""
    root: UnifiedNode
    location: CodeLocation
    size_nodes: int
    depth: int
    node_types: Set[str]
    content: str = ""
    
    def get_signature(self) -> str:
        """Obtiene firma estructural del subárbol."""
        return f"{self.root.node_type.value}:{self.size_nodes}:{self.depth}"


@dataclass
class StructuralSimilarity:
    """Resultado de comparación estructural entre subárboles."""
    overall_similarity: float
    node_similarity: float
    structure_similarity: float
    differences: List[StructuralDifference]
    node_mapping: NodeMapping
    edit_distance: int = 0
    confidence: float = 0.0


@dataclass
class TreeEditOperation:
    """Operación de edición en árbol."""
    operation_type: str  # 'insert', 'delete', 'substitute', 'match'
    node1_id: Optional[str] = None
    node2_id: Optional[str] = None
    cost: float = 1.0
    node_type: Optional[str] = None


@dataclass
class StructuralCloneDetectionResult:
    """Resultado de la detección de clones estructurales."""
    structural_clones: List[StructuralClone]
    subtrees_analyzed: int
    comparison_count: int
    analysis_time_ms: int
    similarities_computed: Dict[str, StructuralSimilarity]


class ASTSubtreeExtractor:
    """Extractor de subárboles AST para análisis estructural."""
    
    def __init__(self, min_nodes: int = 10, max_depth: int = 50):
        self.min_nodes = min_nodes
        self.max_depth = max_depth
    
    def extract_subtrees(self, parse_result: ParseResult) -> List[ASTSubtree]:
        """
        Extrae subárboles significativos del AST.
        
        Args:
            parse_result: Resultado del parsing
            
        Returns:
            Lista de subárboles extraídos
        """
        subtrees = []
        
        if not hasattr(parse_result.tree, 'root_node'):
            logger.warning(f"No root_node found in parse result for {parse_result.file_path}")
            return subtrees
        
        # Convertir tree-sitter node a UnifiedNode si es necesario
        root_unified_node = self._convert_to_unified_node(parse_result.tree.root_node, parse_result)
        
        # Extraer recursivamente
        self._extract_recursive(root_unified_node, subtrees, parse_result, depth=0)
        
        return subtrees
    
    def _extract_recursive(self, node: UnifiedNode, subtrees: List[ASTSubtree], 
                          parse_result: ParseResult, depth: int = 0) -> None:
        """Extrae subárboles recursivamente."""
        if depth > self.max_depth:
            return
        
        # Calcular métricas del subárbol
        subtree_info = self._calculate_subtree_metrics(node)
        
        # Solo incluir subárboles de tamaño significativo
        if subtree_info['size_nodes'] >= self.min_nodes:
            location = self._get_node_location(node, parse_result.file_path)
            
            subtree = ASTSubtree(
                root=node,
                location=location,
                size_nodes=subtree_info['size_nodes'],
                depth=subtree_info['depth'],
                node_types=subtree_info['node_types'],
                content=self._get_node_content(node)
            )
            subtrees.append(subtree)
        
        # Continuar con hijos
        for child in node.children:
            self._extract_recursive(child, subtrees, parse_result, depth + 1)
    
    def _convert_to_unified_node(self, tree_sitter_node, parse_result: ParseResult) -> UnifiedNode:
        """Convierte tree-sitter node a UnifiedNode."""
        # Crear SourcePosition correcta
        position = SourcePosition(
            start_line=tree_sitter_node.start_point[0],
            start_column=tree_sitter_node.start_point[1],
            end_line=tree_sitter_node.end_point[0],
            end_column=tree_sitter_node.end_point[1],
            start_byte=tree_sitter_node.start_byte if hasattr(tree_sitter_node, 'start_byte') else 0,
            end_byte=tree_sitter_node.end_byte if hasattr(tree_sitter_node, 'end_byte') else 0
        )
        
        # Implementación básica - en una implementación real esto sería más sofisticado
        unified_node = UnifiedNode(
            node_type=UnifiedNodeType.LANGUAGE_SPECIFIC,
            position=position,
            children=[],
            value=tree_sitter_node.text.decode('utf-8') if tree_sitter_node.text else ""
        )
        
        # Mapear tipo de nodo
        if hasattr(tree_sitter_node, 'type'):
            node_type_mapping = {
                'function_definition': UnifiedNodeType.FUNCTION_DECLARATION,
                'class_definition': UnifiedNodeType.CLASS_DECLARATION,
                'if_statement': UnifiedNodeType.IF_STATEMENT,
                'for_statement': UnifiedNodeType.FOR_STATEMENT,
                'while_statement': UnifiedNodeType.WHILE_STATEMENT,
                'return_statement': UnifiedNodeType.RETURN_STATEMENT,
                'assignment': UnifiedNodeType.ASSIGNMENT_EXPRESSION,
            }
            unified_node.node_type = node_type_mapping.get(
                tree_sitter_node.type, 
                UnifiedNodeType.LANGUAGE_SPECIFIC
            )
        
        # Convertir hijos recursivamente
        if hasattr(tree_sitter_node, 'children'):
            for child in tree_sitter_node.children:
                unified_child = self._convert_to_unified_node(child, parse_result)
                unified_node.children.append(unified_child)
        
        return unified_node
    
    def _calculate_subtree_metrics(self, node: UnifiedNode) -> Dict[str, Any]:
        """Calcula métricas de un subárbol."""
        node_types = set()
        node_count = 0
        max_depth = 0
        
        def traverse(n: UnifiedNode, current_depth: int = 0) -> None:
            nonlocal node_count, max_depth
            node_count += 1
            max_depth = max(max_depth, current_depth)
            node_types.add(n.node_type.value)
            
            for child in n.children:
                traverse(child, current_depth + 1)
        
        traverse(node)
        
        return {
            'size_nodes': node_count,
            'depth': max_depth,
            'node_types': node_types
        }
    
    def _get_node_location(self, node: UnifiedNode, file_path: Path) -> CodeLocation:
        """Obtiene ubicación del nodo."""
        return CodeLocation(
            file_path=file_path,
            start_line=node.position.start_line + 1,  # 1-indexed
            end_line=node.position.end_line + 1,
            start_column=node.position.start_column,
            end_column=node.position.end_column
        )
    
    def _get_node_content(self, node: UnifiedNode) -> str:
        """Obtiene contenido textual del nodo."""
        return node.value[:200] if node.value else ""  # Primeros 200 caracteres


class TreeEditDistanceCalculator:
    """Calculador de distancia de edición entre árboles."""
    
    def __init__(self):
        self.operation_costs = {
            'insert': 1.0,
            'delete': 1.0,
            'substitute': 1.0,
            'match': 0.0
        }
    
    async def calculate_edit_distance(self, tree1: ASTSubtree, tree2: ASTSubtree) -> Tuple[int, List[TreeEditOperation]]:
        """
        Calcula distancia de edición entre dos árboles usando programación dinámica.
        
        Args:
            tree1: Primer subárbol
            tree2: Segundo subárbol
            
        Returns:
            Tupla con distancia y lista de operaciones
        """
        # Simplificación: usar size-based distance como proxy
        # En implementación completa sería algoritmo de Zhang-Shasha
        
        operations = []
        
        # Comparar raíces
        if tree1.root.node_type == tree2.root.node_type:
            operations.append(TreeEditOperation(
                operation_type='match',
                node1_id=str(id(tree1.root)),
                node2_id=str(id(tree2.root)),
                cost=0.0,
                node_type=tree1.root.node_type.value
            ))
        else:
            operations.append(TreeEditOperation(
                operation_type='substitute',
                node1_id=str(id(tree1.root)),
                node2_id=str(id(tree2.root)),
                cost=1.0,
                node_type=f"{tree1.root.node_type.value}->{tree2.root.node_type.value}"
            ))
        
        # Simplificar: distancia basada en diferencia de tamaño
        size_diff = abs(tree1.size_nodes - tree2.size_nodes)
        depth_diff = abs(tree1.depth - tree2.depth)
        
        distance = size_diff + depth_diff + (1 if tree1.root.node_type != tree2.root.node_type else 0)
        
        return distance, operations


class StructuralComparer:
    """Comparador de estructuras AST."""
    
    def __init__(self, edit_distance_calculator: TreeEditDistanceCalculator):
        self.edit_distance_calculator = edit_distance_calculator
    
    async def compare_subtrees(self, subtree1: ASTSubtree, subtree2: ASTSubtree) -> StructuralSimilarity:
        """
        Compara dos subárboles estructuralmente.
        
        Args:
            subtree1: Primer subárbol
            subtree2: Segundo subárbol
            
        Returns:
            StructuralSimilarity con métricas de similitud
        """
        # Comparación de nodos
        node_similarity = self._calculate_node_similarity(subtree1, subtree2)
        
        # Comparación estructural
        structure_similarity = self._calculate_structure_similarity(subtree1, subtree2)
        
        # Distancia de edición
        edit_distance, edit_operations = await self.edit_distance_calculator.calculate_edit_distance(
            subtree1, subtree2
        )
        
        # Similitud general (inversamente proporcional a la distancia)
        max_size = max(subtree1.size_nodes, subtree2.size_nodes)
        overall_similarity = 1.0 - (edit_distance / max_size) if max_size > 0 else 0.0
        overall_similarity = max(0.0, min(1.0, overall_similarity))  # Clamp [0, 1]
        
        # Mapeo de nodos y diferencias
        node_mapping = self._create_node_mapping(subtree1, subtree2, edit_operations)
        differences = self._identify_differences(subtree1, subtree2, edit_operations)
        
        # Confianza basada en tamaño y similitud
        confidence = self._calculate_confidence(subtree1, subtree2, overall_similarity)
        
        return StructuralSimilarity(
            overall_similarity=overall_similarity,
            node_similarity=node_similarity,
            structure_similarity=structure_similarity,
            differences=differences,
            node_mapping=node_mapping,
            edit_distance=edit_distance,
            confidence=confidence
        )
    
    def _calculate_node_similarity(self, tree1: ASTSubtree, tree2: ASTSubtree) -> float:
        """Calcula similitud basada en tipos de nodos."""
        types1 = tree1.node_types
        types2 = tree2.node_types
        
        if not types1 and not types2:
            return 1.0
        
        intersection = len(types1.intersection(types2))
        union = len(types1.union(types2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_structure_similarity(self, tree1: ASTSubtree, tree2: ASTSubtree) -> float:
        """Calcula similitud estructural."""
        # Similitud basada en tamaño y profundidad
        size_similarity = 1.0 - abs(tree1.size_nodes - tree2.size_nodes) / max(tree1.size_nodes, tree2.size_nodes)
        depth_similarity = 1.0 - abs(tree1.depth - tree2.depth) / max(tree1.depth, tree2.depth)
        
        return (size_similarity + depth_similarity) / 2.0
    
    def _create_node_mapping(self, tree1: ASTSubtree, tree2: ASTSubtree, 
                            operations: List[TreeEditOperation]) -> NodeMapping:
        """Crea mapeo entre nodos basado en operaciones de edición."""
        mapping = NodeMapping()
        
        for op in operations:
            if op.operation_type in ['match', 'substitute'] and op.node1_id and op.node2_id:
                mapping.add_mapping(op.node1_id, op.node2_id)
        
        return mapping
    
    def _identify_differences(self, tree1: ASTSubtree, tree2: ASTSubtree, 
                            operations: List[TreeEditOperation]) -> List[StructuralDifference]:
        """Identifica diferencias estructurales."""
        differences = []
        
        for op in operations:
            if op.operation_type == 'substitute':
                differences.append(StructuralDifference(
                    difference_type='node_type_change',
                    location=tree1.location,
                    original_content=tree1.root.node_type.value,
                    duplicate_content=tree2.root.node_type.value,
                    description=f"Node type changed from {tree1.root.node_type.value} to {tree2.root.node_type.value}",
                    impact_score=op.cost
                ))
            elif op.operation_type == 'insert':
                differences.append(StructuralDifference(
                    difference_type='node_inserted',
                    location=tree2.location,
                    duplicate_content=op.node_type or "unknown",
                    description=f"Node {op.node_type} was inserted",
                    impact_score=op.cost
                ))
            elif op.operation_type == 'delete':
                differences.append(StructuralDifference(
                    difference_type='node_deleted',
                    location=tree1.location,
                    original_content=op.node_type or "unknown",
                    description=f"Node {op.node_type} was deleted",
                    impact_score=op.cost
                ))
        
        return differences
    
    def _calculate_confidence(self, tree1: ASTSubtree, tree2: ASTSubtree, similarity: float) -> float:
        """Calcula confianza en la comparación."""
        # Confianza basada en tamaño de árboles y similitud
        min_size = min(tree1.size_nodes, tree2.size_nodes)
        size_factor = min(1.0, min_size / 20.0)  # Más confianza en árboles grandes
        
        return similarity * size_factor


class StructuralCloneDetector:
    """Detector principal de clones estructurales."""
    
    def __init__(self, min_similarity: float = 0.7):
        self.subtree_extractor = ASTSubtreeExtractor()
        self.edit_distance_calculator = TreeEditDistanceCalculator()
        self.structural_comparer = StructuralComparer(self.edit_distance_calculator)
        self.min_similarity = min_similarity
    
    async def detect_structural_clones(
        self, 
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> StructuralCloneDetectionResult:
        """
        Detecta clones estructurales en un archivo.
        
        Args:
            parse_result: Resultado del parsing
            config: Configuración opcional
            
        Returns:
            StructuralCloneDetectionResult con los clones encontrados
        """
        start_time = time.time()
        
        try:
            # Configurar parámetros
            min_similarity = config.get('min_similarity', self.min_similarity) if config else self.min_similarity
            
            logger.debug(f"Extrayendo subárboles de {parse_result.file_path}")
            
            # 1. Extraer subárboles
            subtrees = self.subtree_extractor.extract_subtrees(parse_result)
            logger.debug(f"Extraídos {len(subtrees)} subárboles")
            
            # 2. Comparar pares de subárboles
            structural_clones = []
            similarities_computed = {}
            comparison_count = 0
            
            for i in range(len(subtrees)):
                for j in range(i + 1, len(subtrees)):
                    subtree1 = subtrees[i]
                    subtree2 = subtrees[j]
                    
                    # Evitar comparar subárboles idénticos o muy pequeños
                    if (subtree1.location.start_line == subtree2.location.start_line or
                        subtree1.size_nodes < 5 or subtree2.size_nodes < 5):
                        continue
                    
                    comparison_count += 1
                    
                    # Comparar estructuralmente
                    similarity = await self.structural_comparer.compare_subtrees(subtree1, subtree2)
                    
                    similarity_key = f"{i}_{j}"
                    similarities_computed[similarity_key] = similarity
                    
                    # Si la similitud es suficiente, crear clone
                    if similarity.overall_similarity >= min_similarity:
                        structural_clone = StructuralClone(
                            id=CloneId(),
                            clone_type=CloneType.RENAMED if similarity.overall_similarity > 0.9 else CloneType.NEAR_MISS,
                            original_location=subtree1.location,
                            duplicate_location=subtree2.location,
                            similarity_score=similarity.overall_similarity,
                            confidence=similarity.confidence,
                            size_lines=subtree2.location.end_line - subtree2.location.start_line + 1,
                            size_tokens=subtree2.size_nodes,  # Aproximación
                            structural_similarity=similarity.structure_similarity,
                            differences=similarity.differences,
                            node_mapping=similarity.node_mapping
                        )
                        
                        structural_clones.append(structural_clone)
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Detección de clones estructurales completada para {parse_result.file_path}: "
                f"{len(structural_clones)} clones encontrados en {total_time}ms "
                f"({comparison_count} comparaciones)"
            )
            
            return StructuralCloneDetectionResult(
                structural_clones=structural_clones,
                subtrees_analyzed=len(subtrees),
                comparison_count=comparison_count,
                analysis_time_ms=total_time,
                similarities_computed=similarities_computed
            )
            
        except Exception as e:
            logger.error(f"Error detectando clones estructurales: {e}")
            raise
    
    async def detect_structural_clones_between_files(
        self,
        parse_results: List[ParseResult],
        config: Optional[Dict[str, Any]] = None
    ) -> List[StructuralClone]:
        """
        Detecta clones estructurales entre múltiples archivos.
        
        Args:
            parse_results: Lista de resultados de parsing
            config: Configuración opcional
            
        Returns:
            Lista de clones estructurales entre archivos
        """
        try:
            all_subtrees = []
            file_subtree_map = {}
            
            # Extraer subtrees de todos los archivos
            for i, parse_result in enumerate(parse_results):
                subtrees = self.subtree_extractor.extract_subtrees(parse_result)
                file_subtree_map[i] = subtrees
                all_subtrees.extend([(subtree, i) for subtree in subtrees])
            
            inter_file_clones = []
            min_similarity = config.get('min_similarity', self.min_similarity) if config else self.min_similarity
            
            # Comparar subtrees entre archivos diferentes
            for i in range(len(all_subtrees)):
                for j in range(i + 1, len(all_subtrees)):
                    subtree1, file1_idx = all_subtrees[i]
                    subtree2, file2_idx = all_subtrees[j]
                    
                    # Solo comparar si son de archivos diferentes
                    if file1_idx != file2_idx:
                        similarity = await self.structural_comparer.compare_subtrees(subtree1, subtree2)
                        
                        if similarity.overall_similarity >= min_similarity:
                            structural_clone = StructuralClone(
                                id=CloneId(),
                                clone_type=CloneType.RENAMED if similarity.overall_similarity > 0.9 else CloneType.NEAR_MISS,
                                original_location=subtree1.location,
                                duplicate_location=subtree2.location,
                                similarity_score=similarity.overall_similarity,
                                confidence=similarity.confidence,
                                size_lines=subtree2.location.end_line - subtree2.location.start_line + 1,
                                size_tokens=subtree2.size_nodes,
                                structural_similarity=similarity.structure_similarity,
                                differences=similarity.differences,
                                node_mapping=similarity.node_mapping
                            )
                            
                            inter_file_clones.append(structural_clone)
            
            return inter_file_clones
            
        except Exception as e:
            logger.error(f"Error detectando clones estructurales inter-archivo: {e}")
            raise
