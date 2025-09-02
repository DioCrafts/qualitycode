"""
Implementación del calculador de similitud multi-algoritmo.

Este módulo implementa múltiples algoritmos de similitud para comparar
bloques de código usando diferentes enfoques: string-based, tree-based, 
hash-based y semánticos.
"""

import logging
import asyncio
import hashlib
import math
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass
import time
from pathlib import Path
import numpy as np
from collections import Counter

from ...domain.entities.clone_analysis import (
    SimilarityMetrics, SimilarityAlgorithm, CodeBlock, SimilarityEvidence
)
from ...domain.entities.ast_normalization import NormalizedNode as UnifiedNode, NodeType as UnifiedNodeType
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Resultado de cálculo de similitud."""
    algorithm: SimilarityAlgorithm
    similarity_score: float
    confidence: float
    calculation_time_ms: int
    metadata: Dict[str, Any]
    
    def is_significant(self, threshold: float = 0.7) -> bool:
        """Verifica si la similitud es significativa."""
        return self.similarity_score >= threshold


@dataclass
class NGram:
    """Representación de un n-grama."""
    tokens: Tuple[str, ...]
    frequency: int = 1
    
    def __hash__(self) -> int:
        return hash(self.tokens)
    
    def __eq__(self, other) -> bool:
        return isinstance(other, NGram) and self.tokens == other.tokens


@dataclass
class TokenSequence:
    """Secuencia de tokens para análisis."""
    tokens: List[str]
    token_types: List[str]
    positions: List[Tuple[int, int]]  # (line, column) for each token
    
    def get_ngrams(self, n: int = 2) -> List[NGram]:
        """Obtiene n-gramas de la secuencia."""
        ngrams = []
        for i in range(len(self.tokens) - n + 1):
            ngram_tokens = tuple(self.tokens[i:i+n])
            ngrams.append(NGram(ngram_tokens))
        return ngrams
    
    def get_token_frequency(self) -> Dict[str, int]:
        """Obtiene frecuencia de tokens."""
        return Counter(self.tokens)


class StringSimilarityCalculator:
    """Calculador de similitud basado en strings."""
    
    def calculate_levenshtein_distance(self, str1: str, str2: str) -> int:
        """
        Calcula distancia de Levenshtein entre dos strings.
        
        Args:
            str1: Primer string
            str2: Segundo string
            
        Returns:
            Distancia de edición
        """
        if not str1:
            return len(str2)
        if not str2:
            return len(str1)
        
        # Programación dinámica
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Inicializar primera fila y columna
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Llenar matriz
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def calculate_levenshtein_similarity(self, str1: str, str2: str) -> float:
        """
        Calcula similitud basada en distancia de Levenshtein.
        
        Returns:
            Similitud entre 0.0 y 1.0
        """
        distance = self.calculate_levenshtein_distance(str1, str2)
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        return 1.0 - (distance / max_len)
    
    def calculate_lcs(self, str1: str, str2: str) -> int:
        """
        Calcula Longest Common Subsequence.
        
        Returns:
            Longitud de la subsecuencia común más larga
        """
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def calculate_lcs_similarity(self, str1: str, str2: str) -> float:
        """Calcula similitud basada en LCS."""
        lcs_length = self.calculate_lcs(str1, str2)
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        return lcs_length / max_len
    
    def calculate_jaccard_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Calcula similitud de Jaccard entre listas de tokens.
        
        Args:
            tokens1: Lista de tokens del primer elemento
            tokens2: Lista de tokens del segundo elemento
            
        Returns:
            Similitud de Jaccard (0.0 - 1.0)
        """
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_cosine_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Calcula similitud coseno entre vectores de frecuencia de tokens.
        
        Returns:
            Similitud coseno (0.0 - 1.0)
        """
        # Crear vectores de frecuencia
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        
        # Obtener vocabulario común
        all_tokens = set(freq1.keys()).union(set(freq2.keys()))
        
        if not all_tokens:
            return 1.0
        
        # Crear vectores
        vec1 = np.array([freq1.get(token, 0) for token in all_tokens])
        vec2 = np.array([freq2.get(token, 0) for token in all_tokens])
        
        # Calcular similitud coseno
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        return max(0.0, cosine_sim)  # Asegurar que no sea negativo


class TreeSimilarityCalculator:
    """Calculador de similitud basado en estructuras de árbol."""
    
    def calculate_ast_tree_similarity(self, tree1: UnifiedNode, tree2: UnifiedNode) -> float:
        """
        Calcula similitud entre árboles AST.
        
        Args:
            tree1: Primer árbol
            tree2: Segundo árbol
            
        Returns:
            Similitud estructural (0.0 - 1.0)
        """
        # Comparar estructura básica
        structure_sim = self._compare_tree_structure(tree1, tree2)
        
        # Comparar tipos de nodos
        node_types_sim = self._compare_node_types(tree1, tree2)
        
        # Combinar métricas
        return (structure_sim + node_types_sim) / 2.0
    
    def _compare_tree_structure(self, tree1: UnifiedNode, tree2: UnifiedNode) -> float:
        """Compara estructura de árboles."""
        # Métricas básicas de estructura
        depth1 = self._calculate_tree_depth(tree1)
        depth2 = self._calculate_tree_depth(tree2)
        
        nodes1 = self._count_nodes(tree1)
        nodes2 = self._count_nodes(tree2)
        
        # Similitud de profundidad
        depth_sim = 1.0 - abs(depth1 - depth2) / max(depth1, depth2) if max(depth1, depth2) > 0 else 1.0
        
        # Similitud de tamaño
        size_sim = 1.0 - abs(nodes1 - nodes2) / max(nodes1, nodes2) if max(nodes1, nodes2) > 0 else 1.0
        
        return (depth_sim + size_sim) / 2.0
    
    def _compare_node_types(self, tree1: UnifiedNode, tree2: UnifiedNode) -> float:
        """Compara distribución de tipos de nodos."""
        types1 = self._collect_node_types(tree1)
        types2 = self._collect_node_types(tree2)
        
        # Usar similitud de Jaccard para tipos de nodos
        if not types1 and not types2:
            return 1.0
        
        intersection = len(types1.intersection(types2))
        union = len(types1.union(types2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_tree_edit_distance(self, tree1: UnifiedNode, tree2: UnifiedNode) -> int:
        """
        Calcula distancia de edición entre árboles (aproximación).
        
        Returns:
            Número de operaciones de edición necesarias
        """
        # Implementación simplificada - en producción sería Zhang-Shasha
        diff_nodes = abs(self._count_nodes(tree1) - self._count_nodes(tree2))
        diff_depth = abs(self._calculate_tree_depth(tree1) - self._calculate_tree_depth(tree2))
        
        types1 = self._collect_node_types(tree1)
        types2 = self._collect_node_types(tree2)
        diff_types = len(types1.symmetric_difference(types2))
        
        return diff_nodes + diff_depth + diff_types
    
    def calculate_subtree_isomorphism(self, tree1: UnifiedNode, tree2: UnifiedNode) -> float:
        """
        Calcula isomorfismo de subárboles.
        
        Returns:
            Porcentaje de subárboles isomórficos (0.0 - 1.0)
        """
        # Extraer todos los subárboles
        subtrees1 = self._extract_subtrees(tree1)
        subtrees2 = self._extract_subtrees(tree2)
        
        # Contar isomorfismos
        isomorphic_count = 0
        total_comparisons = 0
        
        for st1 in subtrees1:
            for st2 in subtrees2:
                total_comparisons += 1
                if self._are_isomorphic(st1, st2):
                    isomorphic_count += 1
        
        return isomorphic_count / total_comparisons if total_comparisons > 0 else 0.0
    
    def _calculate_tree_depth(self, node: UnifiedNode) -> int:
        """Calcula profundidad del árbol."""
        if not node.children:
            return 1
        return 1 + max(self._calculate_tree_depth(child) for child in node.children)
    
    def _count_nodes(self, node: UnifiedNode) -> int:
        """Cuenta nodos en el árbol."""
        return 1 + sum(self._count_nodes(child) for child in node.children)
    
    def _collect_node_types(self, node: UnifiedNode) -> Set[str]:
        """Recopila tipos de nodos únicos."""
        types = {node.node_type.value}
        for child in node.children:
            types.update(self._collect_node_types(child))
        return types
    
    def _extract_subtrees(self, node: UnifiedNode, min_size: int = 2) -> List[UnifiedNode]:
        """Extrae subárboles significativos."""
        subtrees = []
        
        def traverse(n: UnifiedNode):
            if self._count_nodes(n) >= min_size:
                subtrees.append(n)
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return subtrees
    
    def _are_isomorphic(self, tree1: UnifiedNode, tree2: UnifiedNode) -> bool:
        """Verifica si dos árboles son isomórficos."""
        # Implementación simplificada
        if tree1.node_type != tree2.node_type:
            return False
        
        if len(tree1.children) != len(tree2.children):
            return False
        
        # Para simplificar, consideramos isomórficos si tienen misma estructura básica
        return self._count_nodes(tree1) == self._count_nodes(tree2)


class HashSimilarityCalculator:
    """Calculador de similitud basado en hashing."""
    
    def calculate_simhash(self, tokens: List[str], hash_size: int = 64) -> str:
        """
        Calcula SimHash para una lista de tokens.
        
        Args:
            tokens: Lista de tokens
            hash_size: Tamaño del hash en bits
            
        Returns:
            SimHash como string hexadecimal
        """
        feature_vector = [0] * hash_size
        
        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            
            for i in range(hash_size):
                if (token_hash >> i) & 1:
                    feature_vector[i] += 1
                else:
                    feature_vector[i] -= 1
        
        simhash = 0
        for i in range(hash_size):
            if feature_vector[i] > 0:
                simhash |= (1 << i)
        
        return format(simhash, f'0{hash_size//4}x')
    
    def calculate_simhash_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calcula similitud entre dos SimHashes.
        
        Returns:
            Similitud basada en distancia de Hamming (0.0 - 1.0)
        """
        # Convertir a enteros
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
        
        # Calcular distancia de Hamming
        xor_result = int1 ^ int2
        hamming_distance = bin(xor_result).count('1')
        
        # Convertir a similitud (64 es el tamaño típico de SimHash)
        max_distance = len(hash1) * 4  # 4 bits por carácter hex
        
        return 1.0 - (hamming_distance / max_distance) if max_distance > 0 else 1.0
    
    def calculate_minhash_signature(self, tokens: Set[str], num_hashes: int = 100) -> List[int]:
        """
        Calcula MinHash signature para un conjunto de tokens.
        
        Args:
            tokens: Conjunto de tokens únicos
            num_hashes: Número de funciones hash a usar
            
        Returns:
            Lista de valores mínimos de hash
        """
        if not tokens:
            return [0] * num_hashes
        
        signature = []
        
        for i in range(num_hashes):
            min_hash = float('inf')
            
            for token in tokens:
                # Usar diferentes semillas para simular diferentes funciones hash
                token_with_seed = f"{token}_{i}"
                hash_value = int(hashlib.md5(token_with_seed.encode()).hexdigest(), 16)
                min_hash = min(min_hash, hash_value)
            
            signature.append(int(min_hash) if min_hash != float('inf') else 0)
        
        return signature
    
    def calculate_minhash_similarity(self, signature1: List[int], signature2: List[int]) -> float:
        """
        Calcula similitud entre signatures MinHash.
        
        Returns:
            Estimación de similitud de Jaccard (0.0 - 1.0)
        """
        if len(signature1) != len(signature2):
            return 0.0
        
        if not signature1:
            return 1.0
        
        matches = sum(1 for s1, s2 in zip(signature1, signature2) if s1 == s2)
        return matches / len(signature1)


class SemanticSimilarityCalculator:
    """Calculador de similitud semántica."""
    
    def calculate_ngram_similarity(self, tokens1: List[str], tokens2: List[str], n: int = 2) -> float:
        """
        Calcula similitud basada en n-gramas.
        
        Args:
            tokens1: Tokens del primer elemento
            tokens2: Tokens del segundo elemento
            n: Tamaño de n-grama
            
        Returns:
            Similitud de n-gramas (0.0 - 1.0)
        """
        # Crear secuencias de tokens
        seq1 = TokenSequence(tokens1, [], [])
        seq2 = TokenSequence(tokens2, [], [])
        
        # Obtener n-gramas
        ngrams1 = set(tuple(ng.tokens) for ng in seq1.get_ngrams(n))
        ngrams2 = set(tuple(ng.tokens) for ng in seq2.get_ngrams(n))
        
        # Calcular similitud de Jaccard de n-gramas
        if not ngrams1 and not ngrams2:
            return 1.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_semantic_embedding_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Calcula similitud usando embeddings semánticos (simulado).
        
        En una implementación real, usaría modelos pre-entrenados como
        Word2Vec, GloVe, BERT, etc.
        
        Returns:
            Similitud semántica simulada (0.0 - 1.0)
        """
        # Simulación simple basada en tokens comunes y sinónimos
        
        # Diccionario de sinónimos simplificado
        synonyms = {
            'calculate': ['compute', 'process', 'determine'],
            'get': ['fetch', 'retrieve', 'obtain'],
            'set': ['assign', 'update', 'establish'],
            'create': ['make', 'build', 'generate'],
            'delete': ['remove', 'destroy', 'eliminate'],
            'find': ['search', 'locate', 'discover'],
        }
        
        # Expandir tokens con sinónimos
        expanded1 = self._expand_with_synonyms(tokens1, synonyms)
        expanded2 = self._expand_with_synonyms(tokens2, synonyms)
        
        # Calcular similitud de conjuntos expandidos
        intersection = len(expanded1.intersection(expanded2))
        union = len(expanded1.union(expanded2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_behavioral_similarity(self, control_patterns1: List[str], control_patterns2: List[str]) -> float:
        """
        Calcula similitud comportamental basada en patrones de control.
        
        Returns:
            Similitud comportamental (0.0 - 1.0)
        """
        # Normalizar patrones de control a conceptos equivalentes
        normalized1 = self._normalize_control_patterns(control_patterns1)
        normalized2 = self._normalize_control_patterns(control_patterns2)
        
        # Calcular similitud de Jaccard
        if not normalized1 and not normalized2:
            return 1.0
        
        intersection = len(normalized1.intersection(normalized2))
        union = len(normalized1.union(normalized2))
        
        return intersection / union if union > 0 else 0.0
    
    def _expand_with_synonyms(self, tokens: List[str], synonyms: Dict[str, List[str]]) -> Set[str]:
        """Expande tokens con sinónimos."""
        expanded = set(tokens)
        
        for token in tokens:
            token_lower = token.lower()
            if token_lower in synonyms:
                expanded.update(synonyms[token_lower])
            
            # Buscar también en valores de sinónimos
            for key, syn_list in synonyms.items():
                if token_lower in syn_list:
                    expanded.add(key)
                    expanded.update(syn_list)
        
        return expanded
    
    def _normalize_control_patterns(self, patterns: List[str]) -> Set[str]:
        """Normaliza patrones de control a conceptos canónicos."""
        pattern_mapping = {
            'if': 'conditional',
            'if_statement': 'conditional',
            'conditional': 'conditional',
            'for': 'loop',
            'while': 'loop',
            'for_statement': 'loop',
            'while_statement': 'loop',
            'loop': 'loop',
            'try': 'exception_handling',
            'catch': 'exception_handling',
            'except': 'exception_handling',
            'exception_handling': 'exception_handling',
            'switch': 'branching',
            'case': 'branching',
            'branching': 'branching',
        }
        
        normalized = set()
        for pattern in patterns:
            pattern_lower = pattern.lower()
            normalized.add(pattern_mapping.get(pattern_lower, pattern_lower))
        
        return normalized


class SimilarityCalculator:
    """Calculador principal de similitud multi-algoritmo."""
    
    def __init__(self):
        self.string_calculator = StringSimilarityCalculator()
        self.tree_calculator = TreeSimilarityCalculator()
        self.hash_calculator = HashSimilarityCalculator()
        self.semantic_calculator = SemanticSimilarityCalculator()
    
    async def calculate_similarity(
        self, 
        element1: Union[str, List[str], UnifiedNode, CodeBlock],
        element2: Union[str, List[str], UnifiedNode, CodeBlock],
        algorithms: List[SimilarityAlgorithm],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[SimilarityAlgorithm, SimilarityResult]:
        """
        Calcula similitud usando múltiples algoritmos.
        
        Args:
            element1: Primer elemento a comparar
            element2: Segundo elemento a comparar
            algorithms: Lista de algoritmos a usar
            config: Configuración opcional
            
        Returns:
            Diccionario con resultados por algoritmo
        """
        results = {}
        
        for algorithm in algorithms:
            start_time = time.time()
            
            try:
                if algorithm == SimilarityAlgorithm.LEVENSHTEIN:
                    similarity_score = await self._calculate_levenshtein(element1, element2)
                
                elif algorithm == SimilarityAlgorithm.JACCARD:
                    similarity_score = await self._calculate_jaccard(element1, element2)
                
                elif algorithm == SimilarityAlgorithm.COSINE:
                    similarity_score = await self._calculate_cosine(element1, element2)
                
                elif algorithm == SimilarityAlgorithm.LONGEST_COMMON_SUBSEQUENCE:
                    similarity_score = await self._calculate_lcs(element1, element2)
                
                elif algorithm == SimilarityAlgorithm.AST_TREE_MATCHING:
                    similarity_score = await self._calculate_ast_tree_matching(element1, element2)
                
                elif algorithm == SimilarityAlgorithm.TREE_EDIT_DISTANCE:
                    similarity_score = await self._calculate_tree_edit_distance(element1, element2)
                
                elif algorithm == SimilarityAlgorithm.SUBTREE_ISOMORPHISM:
                    similarity_score = await self._calculate_subtree_isomorphism(element1, element2)
                
                elif algorithm == SimilarityAlgorithm.SIMHASH:
                    similarity_score = await self._calculate_simhash(element1, element2)
                
                elif algorithm == SimilarityAlgorithm.MINHASH:
                    similarity_score = await self._calculate_minhash(element1, element2)
                
                elif algorithm == SimilarityAlgorithm.SEMANTIC_EMBEDDING:
                    similarity_score = await self._calculate_semantic_embedding(element1, element2)
                
                elif algorithm == SimilarityAlgorithm.BEHAVIORAL_SIMILARITY:
                    similarity_score = await self._calculate_behavioral_similarity(element1, element2)
                
                else:
                    logger.warning(f"Algoritmo no soportado: {algorithm}")
                    similarity_score = 0.0
                
                execution_time = int((time.time() - start_time) * 1000)
                
                # Calcular confianza basada en características del algoritmo y datos
                confidence = self._calculate_confidence(algorithm, similarity_score, element1, element2)
                
                results[algorithm] = SimilarityResult(
                    algorithm=algorithm,
                    similarity_score=similarity_score,
                    confidence=confidence,
                    calculation_time_ms=execution_time,
                    metadata={
                        "element1_type": type(element1).__name__,
                        "element2_type": type(element2).__name__,
                        "element1_size": self._get_element_size(element1),
                        "element2_size": self._get_element_size(element2),
                    }
                )
                
            except Exception as e:
                logger.error(f"Error calculando similitud con {algorithm}: {e}")
                results[algorithm] = SimilarityResult(
                    algorithm=algorithm,
                    similarity_score=0.0,
                    confidence=0.0,
                    calculation_time_ms=int((time.time() - start_time) * 1000),
                    metadata={"error": str(e)}
                )
        
        return results
    
    async def calculate_weighted_similarity(
        self,
        element1: Union[str, List[str], UnifiedNode, CodeBlock],
        element2: Union[str, List[str], UnifiedNode, CodeBlock],
        algorithm_weights: Dict[SimilarityAlgorithm, float],
        config: Optional[Dict[str, Any]] = None
    ) -> SimilarityResult:
        """
        Calcula similitud ponderada usando múltiples algoritmos.
        
        Args:
            element1: Primer elemento
            element2: Segundo elemento
            algorithm_weights: Pesos por algoritmo
            config: Configuración opcional
            
        Returns:
            SimilarityResult combinado
        """
        start_time = time.time()
        
        # Calcular similitud individual
        algorithms = list(algorithm_weights.keys())
        individual_results = await self.calculate_similarity(element1, element2, algorithms, config)
        
        # Combinar resultados con pesos
        weighted_sum = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        
        for algorithm, weight in algorithm_weights.items():
            if algorithm in individual_results:
                result = individual_results[algorithm]
                weighted_sum += result.similarity_score * weight
                total_weight += weight
                total_confidence += result.confidence * weight
        
        # Calcular similitud final
        final_similarity = weighted_sum / total_weight if total_weight > 0 else 0.0
        final_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
        
        return SimilarityResult(
            algorithm=SimilarityAlgorithm.JACCARD,  # Placeholder
            similarity_score=final_similarity,
            confidence=final_confidence,
            calculation_time_ms=int((time.time() - start_time) * 1000),
            metadata={
                "weighted_combination": True,
                "algorithms_used": [alg.value for alg in algorithms],
                "individual_results": {alg.value: result.similarity_score for alg, result in individual_results.items()}
            }
        )
    
    # Métodos auxiliares para cada algoritmo
    
    async def _calculate_levenshtein(self, elem1: Any, elem2: Any) -> float:
        """Calcula similitud Levenshtein."""
        str1 = self._to_string(elem1)
        str2 = self._to_string(elem2)
        return self.string_calculator.calculate_levenshtein_similarity(str1, str2)
    
    async def _calculate_jaccard(self, elem1: Any, elem2: Any) -> float:
        """Calcula similitud Jaccard."""
        tokens1 = self._to_tokens(elem1)
        tokens2 = self._to_tokens(elem2)
        return self.string_calculator.calculate_jaccard_similarity(tokens1, tokens2)
    
    async def _calculate_cosine(self, elem1: Any, elem2: Any) -> float:
        """Calcula similitud coseno."""
        tokens1 = self._to_tokens(elem1)
        tokens2 = self._to_tokens(elem2)
        return self.string_calculator.calculate_cosine_similarity(tokens1, tokens2)
    
    async def _calculate_lcs(self, elem1: Any, elem2: Any) -> float:
        """Calcula similitud LCS."""
        str1 = self._to_string(elem1)
        str2 = self._to_string(elem2)
        return self.string_calculator.calculate_lcs_similarity(str1, str2)
    
    async def _calculate_ast_tree_matching(self, elem1: Any, elem2: Any) -> float:
        """Calcula similitud AST."""
        tree1 = self._to_unified_node(elem1)
        tree2 = self._to_unified_node(elem2)
        if tree1 and tree2:
            return self.tree_calculator.calculate_ast_tree_similarity(tree1, tree2)
        return 0.0
    
    async def _calculate_tree_edit_distance(self, elem1: Any, elem2: Any) -> float:
        """Calcula similitud basada en distancia de edición de árboles."""
        tree1 = self._to_unified_node(elem1)
        tree2 = self._to_unified_node(elem2)
        if tree1 and tree2:
            distance = self.tree_calculator.calculate_tree_edit_distance(tree1, tree2)
            max_size = max(self.tree_calculator._count_nodes(tree1), self.tree_calculator._count_nodes(tree2))
            return 1.0 - (distance / max_size) if max_size > 0 else 1.0
        return 0.0
    
    async def _calculate_subtree_isomorphism(self, elem1: Any, elem2: Any) -> float:
        """Calcula similitud de isomorfismo de subárboles."""
        tree1 = self._to_unified_node(elem1)
        tree2 = self._to_unified_node(elem2)
        if tree1 and tree2:
            return self.tree_calculator.calculate_subtree_isomorphism(tree1, tree2)
        return 0.0
    
    async def _calculate_simhash(self, elem1: Any, elem2: Any) -> float:
        """Calcula similitud SimHash."""
        tokens1 = self._to_tokens(elem1)
        tokens2 = self._to_tokens(elem2)
        hash1 = self.hash_calculator.calculate_simhash(tokens1)
        hash2 = self.hash_calculator.calculate_simhash(tokens2)
        return self.hash_calculator.calculate_simhash_similarity(hash1, hash2)
    
    async def _calculate_minhash(self, elem1: Any, elem2: Any) -> float:
        """Calcula similitud MinHash."""
        tokens1 = set(self._to_tokens(elem1))
        tokens2 = set(self._to_tokens(elem2))
        sig1 = self.hash_calculator.calculate_minhash_signature(tokens1)
        sig2 = self.hash_calculator.calculate_minhash_signature(tokens2)
        return self.hash_calculator.calculate_minhash_similarity(sig1, sig2)
    
    async def _calculate_semantic_embedding(self, elem1: Any, elem2: Any) -> float:
        """Calcula similitud semántica."""
        tokens1 = self._to_tokens(elem1)
        tokens2 = self._to_tokens(elem2)
        return self.semantic_calculator.calculate_semantic_embedding_similarity(tokens1, tokens2)
    
    async def _calculate_behavioral_similarity(self, elem1: Any, elem2: Any) -> float:
        """Calcula similitud comportamental."""
        # Extraer patrones de control de los elementos
        patterns1 = self._extract_control_patterns(elem1)
        patterns2 = self._extract_control_patterns(elem2)
        return self.semantic_calculator.calculate_behavioral_similarity(patterns1, patterns2)
    
    # Métodos de conversión
    
    def _to_string(self, element: Any) -> str:
        """Convierte elemento a string."""
        if isinstance(element, str):
            return element
        elif isinstance(element, list):
            return " ".join(str(item) for item in element)
        elif isinstance(element, UnifiedNode):
            return element.value or ""
        elif hasattr(element, 'content'):
            return element.content
        else:
            return str(element)
    
    def _to_tokens(self, element: Any) -> List[str]:
        """Convierte elemento a lista de tokens."""
        if isinstance(element, list) and all(isinstance(item, str) for item in element):
            return element
        elif isinstance(element, str):
            # Tokenización simple
            return element.split()
        elif isinstance(element, UnifiedNode):
            return self._tokenize_node(element)
        elif hasattr(element, 'content'):
            return element.content.split()
        else:
            return str(element).split()
    
    def _to_unified_node(self, element: Any) -> Optional[UnifiedNode]:
        """Convierte elemento a UnifiedNode."""
        if isinstance(element, UnifiedNode):
            return element
        elif hasattr(element, 'ast_node') and isinstance(element.ast_node, UnifiedNode):
            return element.ast_node
        else:
            return None
    
    def _tokenize_node(self, node: UnifiedNode) -> List[str]:
        """Tokeniza un nodo UnifiedNode."""
        tokens = []
        
        def traverse(n: UnifiedNode):
            if n.value:
                tokens.extend(n.value.split())
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return tokens
    
    def _extract_control_patterns(self, element: Any) -> List[str]:
        """Extrae patrones de control de un elemento."""
        patterns = []
        
        if isinstance(element, UnifiedNode):
            def traverse(n: UnifiedNode):
                if n.node_type in [UnifiedNodeType.IF_STATEMENT, UnifiedNodeType.FOR_STATEMENT,
                                  UnifiedNodeType.WHILE_STATEMENT, UnifiedNodeType.TRY_STATEMENT]:
                    patterns.append(n.node_type.value)
                for child in n.children:
                    traverse(child)
            
            traverse(element)
        
        return patterns
    
    def _get_element_size(self, element: Any) -> int:
        """Obtiene tamaño del elemento."""
        if isinstance(element, str):
            return len(element)
        elif isinstance(element, list):
            return len(element)
        elif isinstance(element, UnifiedNode):
            return self.tree_calculator._count_nodes(element)
        else:
            return len(str(element))
    
    def _calculate_confidence(self, algorithm: SimilarityAlgorithm, similarity: float,
                            element1: Any, element2: Any) -> float:
        """Calcula confianza en el resultado."""
        base_confidence = 0.7  # Confianza base
        
        # Ajustar por tamaño de elementos (más confianza en elementos grandes)
        size1 = self._get_element_size(element1)
        size2 = self._get_element_size(element2)
        min_size = min(size1, size2)
        
        size_factor = min(1.0, min_size / 100.0)  # Normalizar a [0, 1]
        
        # Ajustar por algoritmo
        algorithm_confidence = {
            SimilarityAlgorithm.LEVENSHTEIN: 0.9,
            SimilarityAlgorithm.JACCARD: 0.8,
            SimilarityAlgorithm.COSINE: 0.8,
            SimilarityAlgorithm.AST_TREE_MATCHING: 0.9,
            SimilarityAlgorithm.SIMHASH: 0.7,
            SimilarityAlgorithm.SEMANTIC_EMBEDDING: 0.6,  # Simulado
        }.get(algorithm, 0.7)
        
        # Combinar factores
        final_confidence = base_confidence * algorithm_confidence * (0.5 + 0.5 * size_factor)
        
        return min(1.0, final_confidence)
