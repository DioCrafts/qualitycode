"""
Implementación del detector de clones semánticos.

Este módulo implementa la detección de clones semánticos (Type 4) usando
análisis de comportamiento, flujo de datos, y similarity learning.
"""

import logging
import asyncio
import hashlib
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from pathlib import Path
import numpy as np
from collections import defaultdict

from ...domain.entities.clone_analysis import (
    SemanticClone, CodeLocation, CloneId, CloneType,
    SimilarityEvidence, SimilarityMetrics, SemanticUnitType
)
from ...domain.entities.parse_result import ParseResult
from ...domain.entities.ast_normalization import NormalizedNode as UnifiedNode, NodeType as UnifiedNodeType, SourcePosition
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class SemanticUnit:
    """Unidad semántica para análisis de clones."""
    id: str
    unit_type: SemanticUnitType
    ast_node: UnifiedNode
    location: CodeLocation
    semantic_info: 'SemanticInfo'
    behavior_signature: 'BehaviorSignature'
    data_flow_info: 'DataFlowInfo'
    
    def get_feature_vector(self) -> np.ndarray:
        """Obtiene vector de características para ML."""
        features = []
        
        # Características estructurales
        features.extend([
            len(self.ast_node.children),
            self._calculate_depth(),
            self._count_node_types(),
        ])
        
        # Características semánticas
        features.extend([
            len(self.semantic_info.variables),
            len(self.semantic_info.function_calls),
            len(self.semantic_info.external_dependencies),
        ])
        
        # Características de comportamiento
        features.extend([
            len(self.behavior_signature.input_types),
            len(self.behavior_signature.output_types),
            self.behavior_signature.complexity_score,
        ])
        
        return np.array(features, dtype=float)
    
    def _calculate_depth(self) -> int:
        """Calcula profundidad del AST."""
        def get_depth(node: UnifiedNode) -> int:
            if not node.children:
                return 1
            return 1 + max(get_depth(child) for child in node.children)
        
        return get_depth(self.ast_node)
    
    def _count_node_types(self) -> int:
        """Cuenta tipos de nodos únicos."""
        node_types = set()
        
        def collect_types(node: UnifiedNode):
            node_types.add(node.node_type.value)
            for child in node.children:
                collect_types(child)
        
        collect_types(self.ast_node)
        return len(node_types)


@dataclass
class SemanticInfo:
    """Información semántica extraída de una unidad."""
    variables: List[str] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)
    control_flow_patterns: List[str] = field(default_factory=list)
    data_types: Set[str] = field(default_factory=set)
    algorithms_used: List[str] = field(default_factory=list)
    
    def get_semantic_hash(self) -> str:
        """Obtiene hash de información semántica."""
        content = ":".join([
            "|".join(sorted(self.variables)),
            "|".join(sorted(self.function_calls)),
            "|".join(sorted(self.external_dependencies)),
            "|".join(sorted(self.control_flow_patterns)),
            "|".join(sorted(self.data_types)),
            "|".join(sorted(self.algorithms_used))
        ])
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class BehaviorSignature:
    """Firma de comportamiento de una función o método."""
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    io_patterns: List[str] = field(default_factory=list)
    error_handling: List[str] = field(default_factory=list)
    
    def calculate_similarity(self, other: 'BehaviorSignature') -> float:
        """Calcula similitud comportamental."""
        input_sim = self._jaccard_similarity(self.input_types, other.input_types)
        output_sim = self._jaccard_similarity(self.output_types, other.output_types)
        side_effects_sim = self._jaccard_similarity(self.side_effects, other.side_effects)
        io_sim = self._jaccard_similarity(self.io_patterns, other.io_patterns)
        
        # Similitud de complejidad
        max_complexity = max(self.complexity_score, other.complexity_score)
        complexity_sim = 1.0 - abs(self.complexity_score - other.complexity_score) / max_complexity if max_complexity > 0 else 1.0
        
        return (input_sim + output_sim + side_effects_sim + io_sim + complexity_sim) / 5.0
    
    def _jaccard_similarity(self, set1: List[str], set2: List[str]) -> float:
        """Calcula similitud de Jaccard."""
        s1, s2 = set(set1), set(set2)
        if not s1 and not s2:
            return 1.0
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        return intersection / union if union > 0 else 0.0


@dataclass
class DataFlowInfo:
    """Información de flujo de datos."""
    variables_read: Set[str] = field(default_factory=set)
    variables_written: Set[str] = field(default_factory=set)
    data_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    control_dependencies: List[str] = field(default_factory=list)
    memory_access_patterns: List[str] = field(default_factory=list)
    
    def calculate_similarity(self, other: 'DataFlowInfo') -> float:
        """Calcula similitud de flujo de datos."""
        read_sim = self._set_similarity(self.variables_read, other.variables_read)
        write_sim = self._set_similarity(self.variables_written, other.variables_written)
        
        # Similitud de dependencias de datos
        deps_sim = self._dependency_similarity(self.data_dependencies, other.data_dependencies)
        
        return (read_sim + write_sim + deps_sim) / 3.0
    
    def _set_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Similitud entre conjuntos."""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def _dependency_similarity(self, deps1: Dict[str, List[str]], deps2: Dict[str, List[str]]) -> float:
        """Similitud de dependencias."""
        if not deps1 and not deps2:
            return 1.0
        
        all_vars = set(deps1.keys()).union(set(deps2.keys()))
        if not all_vars:
            return 1.0
        
        similarities = []
        for var in all_vars:
            deps_var1 = set(deps1.get(var, []))
            deps_var2 = set(deps2.get(var, []))
            similarities.append(self._set_similarity(deps_var1, deps_var2))
        
        return sum(similarities) / len(similarities) if similarities else 0.0


@dataclass
class SemanticSimilarity:
    """Resultado de comparación semántica."""
    overall_similarity: float
    semantic_similarity: float
    behavioral_similarity: float
    data_flow_similarity: float
    ml_similarity: float = 0.0
    evidence: List[SimilarityEvidence] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class SemanticCloneDetectionResult:
    """Resultado de detección de clones semánticos."""
    semantic_clones: List[SemanticClone]
    semantic_units_analyzed: int
    comparison_count: int
    analysis_time_ms: int
    similarity_scores: Dict[str, float]


class SemanticAnalyzer:
    """Analizador de información semántica."""
    
    def analyze_semantic_unit(self, node: UnifiedNode, parse_result: ParseResult) -> SemanticInfo:
        """
        Analiza información semántica de una unidad de código.
        
        Args:
            node: Nodo AST a analizar
            parse_result: Contexto del parsing
            
        Returns:
            SemanticInfo extraída
        """
        info = SemanticInfo()
        
        # Extraer variables
        info.variables = self._extract_variables(node)
        
        # Extraer llamadas a funciones
        info.function_calls = self._extract_function_calls(node)
        
        # Extraer dependencias externas
        info.external_dependencies = self._extract_external_dependencies(node, parse_result.language)
        
        # Analizar patrones de control de flujo
        info.control_flow_patterns = self._analyze_control_flow(node)
        
        # Inferir tipos de datos
        info.data_types = self._infer_data_types(node, parse_result.language)
        
        # Detectar algoritmos comunes
        info.algorithms_used = self._detect_algorithms(node)
        
        return info
    
    def _extract_variables(self, node: UnifiedNode) -> List[str]:
        """Extrae nombres de variables."""
        variables = []
        
        def traverse(n: UnifiedNode):
            if n.node_type == UnifiedNodeType.IDENTIFIER:
                variables.append(n.value)
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return list(set(variables))  # Únicos
    
    def _extract_function_calls(self, node: UnifiedNode) -> List[str]:
        """Extrae llamadas a funciones."""
        function_calls = []
        
        def traverse(n: UnifiedNode):
            if n.node_type == UnifiedNodeType.FUNCTION_CALL:
                # Obtener nombre de la función del primer hijo
                if n.children:
                    function_calls.append(n.children[0].value)
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return list(set(function_calls))
    
    def _extract_external_dependencies(self, node: UnifiedNode, language: ProgrammingLanguage) -> List[str]:
        """Extrae dependencias externas (imports, includes)."""
        dependencies = []
        
        def traverse(n: UnifiedNode):
            if n.node_type == UnifiedNodeType.IMPORT_STATEMENT:
                # Extraer nombre del módulo importado
                for child in n.children:
                    if child.value and not child.value.startswith(('from', 'import')):
                        dependencies.append(child.value)
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return list(set(dependencies))
    
    def _analyze_control_flow(self, node: UnifiedNode) -> List[str]:
        """Analiza patrones de control de flujo."""
        patterns = []
        
        def traverse(n: UnifiedNode):
            pattern_map = {
                UnifiedNodeType.IF_STATEMENT: "conditional",
                UnifiedNodeType.FOR_STATEMENT: "loop_for",
                UnifiedNodeType.WHILE_STATEMENT: "loop_while",
                UnifiedNodeType.TRY_STATEMENT: "exception_handling",
                UnifiedNodeType.SWITCH_STATEMENT: "switch",
            }
            
            if n.node_type in pattern_map:
                patterns.append(pattern_map[n.node_type])
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return patterns
    
    def _infer_data_types(self, node: UnifiedNode, language: ProgrammingLanguage) -> Set[str]:
        """Infiere tipos de datos usados."""
        types = set()
        
        # Patrones por lenguaje
        if language == ProgrammingLanguage.PYTHON:
            type_patterns = {
                'int', 'float', 'str', 'list', 'dict', 'set', 'tuple', 'bool'
            }
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            type_patterns = {
                'number', 'string', 'boolean', 'object', 'array', 'function'
            }
        else:
            type_patterns = {'primitive', 'composite'}
        
        def traverse(n: UnifiedNode):
            if n.value in type_patterns:
                types.add(n.value)
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return types
    
    def _detect_algorithms(self, node: UnifiedNode) -> List[str]:
        """Detecta algoritmos comunes."""
        algorithms = []
        
        # Buscar patrones algorítmicos comunes
        def traverse(n: UnifiedNode):
            # Patrones de sorting
            if 'sort' in n.value.lower():
                algorithms.append('sorting')
            
            # Patrones de búsqueda
            if any(keyword in n.value.lower() for keyword in ['search', 'find', 'index']):
                algorithms.append('searching')
            
            # Patrones matemáticos
            if any(keyword in n.value.lower() for keyword in ['sum', 'avg', 'max', 'min']):
                algorithms.append('mathematical')
            
            # Patrones de iteración
            nested_loops = sum(1 for child in n.children 
                             if child.node_type in [UnifiedNodeType.FOR_STATEMENT, UnifiedNodeType.WHILE_STATEMENT])
            if nested_loops >= 2:
                algorithms.append('nested_iteration')
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return list(set(algorithms))


class BehaviorAnalyzer:
    """Analizador de comportamiento de funciones."""
    
    def analyze_behavior(self, node: UnifiedNode, semantic_info: SemanticInfo) -> BehaviorSignature:
        """
        Analiza comportamiento de una función o método.
        
        Args:
            node: Nodo AST de la función
            semantic_info: Información semántica
            
        Returns:
            BehaviorSignature del comportamiento
        """
        signature = BehaviorSignature()
        
        # Analizar tipos de entrada (parámetros)
        signature.input_types = self._extract_input_types(node)
        
        # Analizar tipos de salida (returns)
        signature.output_types = self._extract_output_types(node)
        
        # Detectar efectos secundarios
        signature.side_effects = self._detect_side_effects(node)
        
        # Calcular complejidad ciclomática
        signature.complexity_score = self._calculate_cyclomatic_complexity(node)
        
        # Analizar patrones I/O
        signature.io_patterns = self._analyze_io_patterns(node)
        
        # Analizar manejo de errores
        signature.error_handling = self._analyze_error_handling(node)
        
        return signature
    
    def _extract_input_types(self, node: UnifiedNode) -> List[str]:
        """Extrae tipos de parámetros de entrada."""
        input_types = []
        
        # Buscar nodo de parámetros
        for child in node.children:
            if child.node_type == UnifiedNodeType.PARAMETER_LIST:
                for param in child.children:
                    # Inferir tipo basado en valor por defecto o anotación
                    input_types.append(self._infer_parameter_type(param))
        
        return input_types
    
    def _extract_output_types(self, node: UnifiedNode) -> List[str]:
        """Extrae tipos de valores de retorno."""
        output_types = []
        
        def traverse(n: UnifiedNode):
            if n.node_type == UnifiedNodeType.RETURN_STATEMENT:
                if n.children:
                    output_types.append(self._infer_return_type(n.children[0]))
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return list(set(output_types)) if output_types else ['void']
    
    def _detect_side_effects(self, node: UnifiedNode) -> List[str]:
        """Detecta efectos secundarios."""
        side_effects = []
        
        def traverse(n: UnifiedNode):
            # Modificación de variables globales
            if n.node_type == UnifiedNodeType.ASSIGNMENT_EXPRESSION:
                side_effects.append('variable_modification')
            
            # I/O operations
            if 'print' in n.value.lower() or 'write' in n.value.lower():
                side_effects.append('io_output')
            
            if 'read' in n.value.lower() or 'input' in n.value.lower():
                side_effects.append('io_input')
            
            # File operations
            if any(keyword in n.value.lower() for keyword in ['open', 'file', 'write', 'read']):
                side_effects.append('file_operation')
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return list(set(side_effects))
    
    def _calculate_cyclomatic_complexity(self, node: UnifiedNode) -> float:
        """Calcula complejidad ciclomática aproximada."""
        complexity = 1.0  # Base complexity
        
        def traverse(n: UnifiedNode):
            nonlocal complexity
            
            # Decision points
            if n.node_type in [
                UnifiedNodeType.IF_STATEMENT,
                UnifiedNodeType.WHILE_STATEMENT,
                UnifiedNodeType.FOR_STATEMENT,
                UnifiedNodeType.CASE_CLAUSE
            ]:
                complexity += 1.0
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return complexity
    
    def _analyze_io_patterns(self, node: UnifiedNode) -> List[str]:
        """Analiza patrones de entrada/salida."""
        io_patterns = []
        
        def traverse(n: UnifiedNode):
            # Patrones de lectura secuencial
            if 'readline' in n.value.lower() or 'readlines' in n.value.lower():
                io_patterns.append('sequential_read')
            
            # Patrones de escritura
            if 'write' in n.value.lower():
                io_patterns.append('write_operation')
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return list(set(io_patterns))
    
    def _analyze_error_handling(self, node: UnifiedNode) -> List[str]:
        """Analiza manejo de errores."""
        error_patterns = []
        
        def traverse(n: UnifiedNode):
            if n.node_type == UnifiedNodeType.TRY_STATEMENT:
                error_patterns.append('try_catch')
            
            if 'raise' in n.value.lower() or 'throw' in n.value.lower():
                error_patterns.append('exception_raising')
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return list(set(error_patterns))
    
    def _infer_parameter_type(self, param_node: UnifiedNode) -> str:
        """Infiere tipo de parámetro."""
        # Implementación simplificada
        if param_node.value.isdigit():
            return 'number'
        elif '"' in param_node.value or "'" in param_node.value:
            return 'string'
        elif param_node.value in ['true', 'false', 'True', 'False']:
            return 'boolean'
        else:
            return 'unknown'
    
    def _infer_return_type(self, return_node: UnifiedNode) -> str:
        """Infiere tipo de valor de retorno."""
        return self._infer_parameter_type(return_node)


class SemanticCloneDetector:
    """Detector principal de clones semánticos."""
    
    def __init__(self, min_similarity: float = 0.6):
        self.semantic_analyzer = SemanticAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.min_similarity = min_similarity
    
    async def detect_semantic_clones(
        self, 
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> SemanticCloneDetectionResult:
        """
        Detecta clones semánticos en un archivo.
        
        Args:
            parse_result: Resultado del parsing
            config: Configuración opcional
            
        Returns:
            SemanticCloneDetectionResult con los clones encontrados
        """
        start_time = time.time()
        
        try:
            # Configurar parámetros
            min_similarity = config.get('min_similarity', self.min_similarity) if config else self.min_similarity
            
            logger.debug(f"Extrayendo unidades semánticas de {parse_result.file_path}")
            
            # 1. Extraer unidades semánticas (funciones, métodos)
            semantic_units = await self._extract_semantic_units(parse_result)
            logger.debug(f"Extraídas {len(semantic_units)} unidades semánticas")
            
            # 2. Comparar pares de unidades
            semantic_clones = []
            similarity_scores = {}
            comparison_count = 0
            
            for i in range(len(semantic_units)):
                for j in range(i + 1, len(semantic_units)):
                    unit1 = semantic_units[i]
                    unit2 = semantic_units[j]
                    
                    # Evitar comparar unidades idénticas
                    if unit1.location.start_line == unit2.location.start_line:
                        continue
                    
                    comparison_count += 1
                    
                    # Comparar semánticamente
                    similarity = await self._compare_semantic_units(unit1, unit2)
                    
                    similarity_key = f"{i}_{j}"
                    similarity_scores[similarity_key] = similarity.overall_similarity
                    
                    # Si la similitud es suficiente, crear clone
                    if similarity.overall_similarity >= min_similarity:
                        semantic_clone = SemanticClone(
                            id=CloneId(),
                            clone_type=CloneType.SEMANTIC,
                            original_location=unit1.location,
                            duplicate_location=unit2.location,
                            similarity_score=similarity.overall_similarity,
                            confidence=similarity.confidence,
                            size_lines=unit2.location.end_line - unit2.location.start_line + 1,
                            size_tokens=len(unit2.ast_node.children),  # Aproximación
                            semantic_similarity=similarity.semantic_similarity,
                            behavioral_similarity=similarity.behavioral_similarity,
                            data_flow_similarity=similarity.data_flow_similarity,
                            evidence=similarity.evidence
                        )
                        
                        semantic_clones.append(semantic_clone)
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Detección de clones semánticos completada para {parse_result.file_path}: "
                f"{len(semantic_clones)} clones encontrados en {total_time}ms "
                f"({comparison_count} comparaciones)"
            )
            
            return SemanticCloneDetectionResult(
                semantic_clones=semantic_clones,
                semantic_units_analyzed=len(semantic_units),
                comparison_count=comparison_count,
                analysis_time_ms=total_time,
                similarity_scores=similarity_scores
            )
            
        except Exception as e:
            logger.error(f"Error detectando clones semánticos: {e}")
            raise
    
    async def _extract_semantic_units(self, parse_result: ParseResult) -> List[SemanticUnit]:
        """Extrae unidades semánticas del código."""
        units = []
        unit_counter = 0
        
        def traverse_and_extract(node: UnifiedNode, parent_location: Optional[CodeLocation] = None):
            nonlocal unit_counter
            
            # Extraer funciones y métodos
            if node.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
                location = self._get_node_location(node, parse_result.file_path)
                
                # Analizar información semántica
                semantic_info = self.semantic_analyzer.analyze_semantic_unit(node, parse_result)
                
                # Analizar comportamiento
                behavior_signature = self.behavior_analyzer.analyze_behavior(node, semantic_info)
                
                # Crear información de flujo de datos (simplificada)
                data_flow_info = self._analyze_data_flow(node)
                
                unit = SemanticUnit(
                    id=f"unit_{unit_counter}",
                    unit_type=SemanticUnitType.FUNCTION,
                    ast_node=node,
                    location=location,
                    semantic_info=semantic_info,
                    behavior_signature=behavior_signature,
                    data_flow_info=data_flow_info
                )
                
                units.append(unit)
                unit_counter += 1
            
            # Continuar con hijos
            for child in node.children:
                traverse_and_extract(child, parent_location)
        
        # Iniciar extracción desde la raíz
        if hasattr(parse_result.tree, 'root_node'):
            root_unified = self._convert_tree_sitter_to_unified(parse_result.tree.root_node)
            traverse_and_extract(root_unified)
        
        return units
    
    async def _compare_semantic_units(self, unit1: SemanticUnit, unit2: SemanticUnit) -> SemanticSimilarity:
        """Compara dos unidades semánticas."""
        # Similitud semántica (basada en información extraída)
        semantic_sim = self._calculate_semantic_similarity(unit1.semantic_info, unit2.semantic_info)
        
        # Similitud comportamental
        behavioral_sim = unit1.behavior_signature.calculate_similarity(unit2.behavior_signature)
        
        # Similitud de flujo de datos
        data_flow_sim = unit1.data_flow_info.calculate_similarity(unit2.data_flow_info)
        
        # Similitud ML basada en features (simplificada)
        ml_sim = self._calculate_ml_similarity(unit1, unit2)
        
        # Similitud general combinada
        weights = {
            'semantic': 0.4,
            'behavioral': 0.3,
            'data_flow': 0.2,
            'ml': 0.1
        }
        
        overall_similarity = (
            semantic_sim * weights['semantic'] +
            behavioral_sim * weights['behavioral'] +
            data_flow_sim * weights['data_flow'] +
            ml_sim * weights['ml']
        )
        
        # Recopilar evidencia
        evidence = self._collect_similarity_evidence(unit1, unit2, semantic_sim, behavioral_sim, data_flow_sim)
        
        # Calcular confianza
        confidence = self._calculate_confidence(unit1, unit2, overall_similarity, evidence)
        
        return SemanticSimilarity(
            overall_similarity=overall_similarity,
            semantic_similarity=semantic_sim,
            behavioral_similarity=behavioral_sim,
            data_flow_similarity=data_flow_sim,
            ml_similarity=ml_sim,
            evidence=evidence,
            confidence=confidence
        )
    
    def _calculate_semantic_similarity(self, info1: SemanticInfo, info2: SemanticInfo) -> float:
        """Calcula similitud semántica entre dos unidades."""
        if info1.get_semantic_hash() == info2.get_semantic_hash():
            return 1.0
        
        # Similitud basada en componentes individuales
        var_sim = self._jaccard_similarity(info1.variables, info2.variables)
        func_sim = self._jaccard_similarity(info1.function_calls, info2.function_calls)
        deps_sim = self._jaccard_similarity(info1.external_dependencies, info2.external_dependencies)
        flow_sim = self._jaccard_similarity(info1.control_flow_patterns, info2.control_flow_patterns)
        algo_sim = self._jaccard_similarity(info1.algorithms_used, info2.algorithms_used)
        
        return (var_sim + func_sim + deps_sim + flow_sim + algo_sim) / 5.0
    
    def _calculate_ml_similarity(self, unit1: SemanticUnit, unit2: SemanticUnit) -> float:
        """Calcula similitud usando features ML."""
        try:
            features1 = unit1.get_feature_vector()
            features2 = unit2.get_feature_vector()
            
            # Similitud coseno entre vectores de características
            if np.linalg.norm(features1) == 0 or np.linalg.norm(features2) == 0:
                return 0.0
            
            cosine_sim = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
            return max(0.0, cosine_sim)  # Asegurar que no sea negativo
            
        except Exception as e:
            logger.debug(f"Error calculando similitud ML: {e}")
            return 0.0
    
    def _collect_similarity_evidence(self, unit1: SemanticUnit, unit2: SemanticUnit,
                                   semantic_sim: float, behavioral_sim: float, 
                                   data_flow_sim: float) -> List[SimilarityEvidence]:
        """Recopila evidencia de similitud."""
        evidence = []
        
        # Evidencia basada en tipos de entrada/salida
        if set(unit1.behavior_signature.input_types) == set(unit2.behavior_signature.input_types):
            evidence.append(SimilarityEvidence.SAME_INPUT_OUTPUT_TYPES)
        
        # Evidencia de control de flujo similar
        if behavioral_sim > 0.7:
            evidence.append(SimilarityEvidence.SIMILAR_CONTROL_FLOW)
        
        # Evidencia de flujo de datos similar
        if data_flow_sim > 0.7:
            evidence.append(SimilarityEvidence.SIMILAR_DATA_FLOW)
        
        # Evidencia de complejidad similar
        complexity_diff = abs(unit1.behavior_signature.complexity_score - unit2.behavior_signature.complexity_score)
        if complexity_diff < 2.0:
            evidence.append(SimilarityEvidence.SAME_ALGORITHMIC_COMPLEXITY)
        
        # Evidencia de dependencias similares
        if self._jaccard_similarity(unit1.semantic_info.external_dependencies, 
                                   unit2.semantic_info.external_dependencies) > 0.5:
            evidence.append(SimilarityEvidence.SAME_EXTERNAL_DEPENDENCIES)
        
        return evidence
    
    def _calculate_confidence(self, unit1: SemanticUnit, unit2: SemanticUnit, 
                            similarity: float, evidence: List[SimilarityEvidence]) -> float:
        """Calcula confianza en la similitud."""
        # Confianza base basada en similitud
        base_confidence = similarity
        
        # Boost por evidencia fuerte
        evidence_boost = len(evidence) * 0.1
        
        # Factor de tamaño (más confianza en funciones grandes)
        min_size = min(len(unit1.ast_node.children), len(unit2.ast_node.children))
        size_factor = min(1.0, min_size / 10.0)
        
        confidence = base_confidence + evidence_boost
        confidence *= size_factor
        
        return min(1.0, confidence)  # Cap at 1.0
    
    def _analyze_data_flow(self, node: UnifiedNode) -> DataFlowInfo:
        """Analiza flujo de datos (implementación simplificada)."""
        info = DataFlowInfo()
        
        def traverse(n: UnifiedNode):
            # Variables leídas
            if n.node_type == UnifiedNodeType.IDENTIFIER:
                info.variables_read.add(n.value)
            
            # Variables escritas
            if n.node_type == UnifiedNodeType.ASSIGNMENT_EXPRESSION:
                if n.children:
                    info.variables_written.add(n.children[0].value)
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return info
    
    def _convert_tree_sitter_to_unified(self, tree_sitter_node) -> UnifiedNode:
        """Convierte tree-sitter node a UnifiedNode (implementación básica)."""
        # Crear SourcePosition correcta
        position = SourcePosition(
            start_line=tree_sitter_node.start_point[0],
            start_column=tree_sitter_node.start_point[1],
            end_line=tree_sitter_node.end_point[0],
            end_column=tree_sitter_node.end_point[1],
            start_byte=tree_sitter_node.start_byte if hasattr(tree_sitter_node, 'start_byte') else 0,
            end_byte=tree_sitter_node.end_byte if hasattr(tree_sitter_node, 'end_byte') else 0
        )
        
        unified_node = UnifiedNode(
            node_type=UnifiedNodeType.LANGUAGE_SPECIFIC,
            position=position,
            children=[],
            value=tree_sitter_node.text.decode('utf-8') if tree_sitter_node.text else ""
        )
        
        # Mapear tipo de nodo (simplificado)
        type_mapping = {
            'function_definition': UnifiedNodeType.FUNCTION_DECLARATION,
            'class_definition': UnifiedNodeType.CLASS_DECLARATION,
            'if_statement': UnifiedNodeType.IF_STATEMENT,
            'for_statement': UnifiedNodeType.FOR_STATEMENT,
            'while_statement': UnifiedNodeType.WHILE_STATEMENT,
            'return_statement': UnifiedNodeType.RETURN_STATEMENT,
            'assignment': UnifiedNodeType.ASSIGNMENT_EXPRESSION,
        }
        
        if hasattr(tree_sitter_node, 'type'):
            unified_node.node_type = type_mapping.get(tree_sitter_node.type, UnifiedNodeType.LANGUAGE_SPECIFIC)
        
        # Convertir hijos
        if hasattr(tree_sitter_node, 'children'):
            for child in tree_sitter_node.children:
                unified_child = self._convert_tree_sitter_to_unified(child)
                unified_node.children.append(unified_child)
        
        return unified_node
    
    def _get_node_location(self, node: UnifiedNode, file_path: Path) -> CodeLocation:
        """Obtiene ubicación del nodo."""
        return CodeLocation(
            file_path=file_path,
            start_line=node.position.start_line + 1,
            end_line=node.position.end_line + 1,
            start_column=node.position.start_column,
            end_column=node.position.end_column
        )
    
    def _jaccard_similarity(self, list1: List[str], list2: List[str]) -> float:
        """Calcula similitud de Jaccard entre listas."""
        set1, set2 = set(list1), set(list2)
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
