"""
Sistema de Pattern Matching Cross-Language.

Este módulo implementa el sistema de detección de patrones que funciona
a través de múltiples lenguajes de programación.
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

from .unified_ast import (
    UnifiedAST,
    UnifiedNode,
    UnifiedNodeType,
    SemanticNodeType,
    NodeId,
)

logger = logging.getLogger(__name__)


class PatternCategory(str, Enum):
    """Categorías de patrones."""
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    CREATIONAL = "creational"
    ARCHITECTURAL = "architectural"
    IDIOMATIC = "idiomatic"
    ANTI_PATTERN = "anti_pattern"


@dataclass
class Pattern:
    """Definición de un patrón."""
    id: str
    name: str
    description: str
    category: PatternCategory
    languages: List[str]
    template: 'PatternTemplate'
    variations: List['PatternVariation'] = field(default_factory=list)
    confidence_threshold: float = 0.7


@dataclass
class PatternTemplate:
    """Plantilla de un patrón."""
    structural: Optional['NodePattern'] = None
    semantic: Optional[str] = None
    behavioral: Optional[str] = None
    hybrid: Optional[Dict[str, Any]] = None


@dataclass
class NodePattern:
    """Patrón de nodos."""
    node_type: Optional[UnifiedNodeType] = None
    semantic_type: Optional[SemanticNodeType] = None
    name_pattern: Optional[str] = None
    value_pattern: Optional[str] = None
    children_patterns: List['NodePattern'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    quantifiers: 'PatternQuantifier' = field(default_factory=lambda: PatternQuantifier.EXACTLY(1))


@dataclass
class PatternQuantifier:
    """Cuantificador de patrones."""
    type: str
    min_count: int
    max_count: Optional[int] = None
    
    @classmethod
    def EXACTLY(cls, count: int):
        return cls("exactly", count, count)
    
    @classmethod
    def AT_LEAST(cls, count: int):
        return cls("at_least", count)
    
    @classmethod
    def AT_MOST(cls, count: int):
        return cls("at_most", 0, count)
    
    @classmethod
    def BETWEEN(cls, min_count: int, max_count: int):
        return cls("between", min_count, max_count)
    
    @classmethod
    def ZERO_OR_MORE(cls):
        return cls("zero_or_more", 0)
    
    @classmethod
    def ONE_OR_MORE(cls):
        return cls("one_or_more", 1)
    
    @classmethod
    def OPTIONAL(cls):
        return cls("optional", 0, 1)


@dataclass
class PatternVariation:
    """Variación de un patrón."""
    name: str
    description: str
    modifications: Dict[str, Any]
    confidence_adjustment: float = 0.0


@dataclass
class PatternMatch:
    """Coincidencia de un patrón."""
    pattern_id: str
    pattern_name: str
    matched_nodes: List[NodeId]
    confidence: float
    location: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossLanguagePatternMatch:
    """Coincidencia de patrón cross-language."""
    pattern_type: str
    language_a: str
    language_b: str
    match_a: PatternMatch
    match_b: PatternMatch
    similarity_score: float
    explanation: str


class PatternError(Exception):
    """Error en el sistema de pattern matching."""
    
    def __init__(self, message: str, pattern_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.pattern_id = pattern_id
        self.details = details or {}


class CrossLanguagePatternMatcher:
    """Matcher de patrones cross-language."""
    
    def __init__(self):
        self.pattern_library = self._load_pattern_library()
        self.similarity_calculator = SimilarityCalculator()
        self.pattern_cache = {}
    
    def _load_pattern_library(self) -> Dict[str, Pattern]:
        """Carga la biblioteca de patrones."""
        patterns = {}
        
        # Patrones estructurales básicos
        patterns["function_definition"] = Pattern(
            id="function_definition",
            name="Function Definition",
            description="Definición de función en cualquier lenguaje",
            category=PatternCategory.STRUCTURAL,
            languages=["python", "typescript", "javascript", "rust"],
            template=PatternTemplate(
                structural=NodePattern(
                    node_type=UnifiedNodeType.FUNCTION_DECLARATION,
                    semantic_type=SemanticNodeType.DEFINITION,
                )
            ),
            confidence_threshold=0.8
        )
        
        patterns["class_definition"] = Pattern(
            id="class_definition",
            name="Class Definition",
            description="Definición de clase en cualquier lenguaje",
            category=PatternCategory.STRUCTURAL,
            languages=["python", "typescript", "javascript", "rust"],
            template=PatternTemplate(
                structural=NodePattern(
                    node_type=UnifiedNodeType.CLASS_DECLARATION,
                    semantic_type=SemanticNodeType.DEFINITION,
                )
            ),
            confidence_threshold=0.8
        )
        
        patterns["loop_construct"] = Pattern(
            id="loop_construct",
            name="Loop Construct",
            description="Construcción de bucle en cualquier lenguaje",
            category=PatternCategory.STRUCTURAL,
            languages=["python", "typescript", "javascript", "rust"],
            template=PatternTemplate(
                structural=NodePattern(
                    node_type=UnifiedNodeType.FOR_STATEMENT,
                    semantic_type=SemanticNodeType.CONTROL_FLOW,
                )
            ),
            confidence_threshold=0.7
        )
        
        return patterns
    
    async def find_patterns(self, ast: UnifiedAST, pattern_ids: Optional[List[str]] = None) -> List[PatternMatch]:
        """Encuentra patrones en un AST."""
        matches = []
        
        if pattern_ids is None:
            pattern_ids = list(self.pattern_library.keys())
        
        for pattern_id in pattern_ids:
            if pattern_id in self.pattern_library:
                pattern = self.pattern_library[pattern_id]
                pattern_matches = await self.match_pattern(ast, pattern)
                matches.extend(pattern_matches)
        
        return matches
    
    async def find_similar_patterns_across_languages(self, asts: List[UnifiedAST]) -> List[CrossLanguagePatternMatch]:
        """Encuentra patrones similares entre diferentes lenguajes."""
        cross_language_matches = []
        
        # Encontrar patrones en cada AST
        ast_patterns = {}
        for ast in asts:
            patterns = await self.find_patterns(ast)
            ast_patterns[ast.language] = patterns
        
        # Comparar patrones entre lenguajes
        for lang1, patterns1 in ast_patterns.items():
            for lang2, patterns2 in ast_patterns.items():
                if lang1 != lang2:
                    similar_patterns = await self.find_similar_patterns(patterns1, patterns2)
                    for similar in similar_patterns:
                        cross_language_matches.append(CrossLanguagePatternMatch(
                            pattern_type=similar.pattern_type,
                            language_a=lang1,
                            language_b=lang2,
                            match_a=similar.match_a,
                            match_b=similar.match_b,
                            similarity_score=similar.similarity_score,
                            explanation=similar.explanation,
                        ))
        
        return cross_language_matches
    
    async def match_pattern(self, ast: UnifiedAST, pattern: Pattern) -> List[PatternMatch]:
        """Coincide un patrón específico en un AST."""
        matches = []
        
        if pattern.template.structural:
            structural_matches = await self.match_structural_pattern(ast, pattern.template.structural)
            matches.extend(structural_matches)
        
        if pattern.template.semantic:
            semantic_matches = await self.match_semantic_pattern(ast, pattern.template.semantic)
            matches.extend(semantic_matches)
        
        if pattern.template.behavioral:
            behavioral_matches = await self.match_behavioral_pattern(ast, pattern.template.behavioral)
            matches.extend(behavioral_matches)
        
        # Filtrar por umbral de confianza
        matches = [match for match in matches if match.confidence >= pattern.confidence_threshold]
        
        return matches
    
    async def match_structural_pattern(self, ast: UnifiedAST, node_pattern: NodePattern) -> List[PatternMatch]:
        """Coincide un patrón estructural."""
        matches = []
        
        def traverse(node: UnifiedNode):
            if self._node_matches_pattern(node, node_pattern):
                # Verificar cuantificadores para los hijos
                if self._check_children_quantifiers(node, node_pattern):
                    match = PatternMatch(
                        pattern_id="structural_pattern",
                        pattern_name="Structural Pattern",
                        matched_nodes=[node.id],
                        confidence=0.8,
                        location={
                            "file_path": str(ast.file_path),
                            "start_line": node.position.start_line if node.position else None,
                            "end_line": node.position.end_line if node.position else None,
                        }
                    )
                    matches.append(match)
            
            for child in node.children:
                traverse(child)
        
        traverse(ast.root_node)
        return matches
    
    async def match_semantic_pattern(self, ast: UnifiedAST, semantic_pattern: str) -> List[PatternMatch]:
        """Coincide un patrón semántico."""
        # Implementación básica - se expandirá
        return []
    
    async def match_behavioral_pattern(self, ast: UnifiedAST, behavioral_pattern: str) -> List[PatternMatch]:
        """Coincide un patrón comportamental."""
        # Implementación básica - se expandirá
        return []
    
    def _node_matches_pattern(self, node: UnifiedNode, pattern: NodePattern) -> bool:
        """Determina si un nodo coincide con un patrón."""
        # Verificar tipo de nodo
        if pattern.node_type and node.node_type != pattern.node_type:
            return False
        
        # Verificar tipo semántico
        if pattern.semantic_type and node.semantic_type != pattern.semantic_type:
            return False
        
        # Verificar patrón de nombre
        if pattern.name_pattern and node.name:
            if not self._matches_pattern(node.name, pattern.name_pattern):
                return False
        
        # Verificar patrón de valor
        if pattern.value_pattern and node.value:
            if not self._matches_pattern(str(node.value), pattern.value_pattern):
                return False
        
        # Verificar atributos
        for attr_name, attr_value in pattern.attributes.items():
            if attr_name not in node.attributes or node.attributes[attr_name] != attr_value:
                return False
        
        return True
    
    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Determina si un texto coincide con un patrón."""
        # Implementación básica - se expandirá con regex
        return pattern.lower() in text.lower()
    
    def _check_children_quantifiers(self, node: UnifiedNode, pattern: NodePattern) -> bool:
        """Verifica los cuantificadores de los hijos."""
        if not pattern.children_patterns:
            return True
        
        # Implementación básica - se expandirá
        return True
    
    async def find_similar_patterns(self, patterns1: List[PatternMatch], patterns2: List[PatternMatch]) -> List[Any]:
        """Encuentra patrones similares entre dos listas."""
        similar_patterns = []
        
        for pattern1 in patterns1:
            for pattern2 in patterns2:
                similarity = await self.similarity_calculator.calculate_similarity(pattern1, pattern2)
                if similarity > 0.7:  # Umbral de similitud
                    similar_patterns.append({
                        "pattern_type": pattern1.pattern_name,
                        "match_a": pattern1,
                        "match_b": pattern2,
                        "similarity_score": similarity,
                        "explanation": f"Similar patterns found with {similarity:.2f} similarity",
                    })
        
        return similar_patterns
    
    def get_pattern_library_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la biblioteca de patrones."""
        categories = {}
        languages = set()
        
        for pattern in self.pattern_library.values():
            # Contar por categoría
            category = pattern.category.value
            categories[category] = categories.get(category, 0) + 1
            
            # Recolectar lenguajes
            languages.update(pattern.languages)
        
        return {
            "total_patterns": len(self.pattern_library),
            "categories": categories,
            "supported_languages": list(languages),
            "cache_size": len(self.pattern_cache),
        }


class SimilarityCalculator:
    """Calculador de similitud entre patrones."""
    
    async def calculate_similarity(self, pattern1: PatternMatch, pattern2: PatternMatch) -> float:
        """Calcula la similitud entre dos patrones."""
        # Implementación básica - se expandirá con algoritmos más sofisticados
        if pattern1.pattern_name == pattern2.pattern_name:
            return 0.8
        else:
            return 0.3
    
    def calculate_structural_similarity(self, node1: UnifiedNode, node2: UnifiedNode) -> float:
        """Calcula similitud estructural entre nodos."""
        similarity = 0.0
        
        # Similitud de tipo de nodo
        if node1.node_type == node2.node_type:
            similarity += 0.3
        
        # Similitud de tipo semántico
        if node1.semantic_type == node2.semantic_type:
            similarity += 0.2
        
        # Similitud de nombre
        if node1.name and node2.name:
            if node1.name.lower() == node2.name.lower():
                similarity += 0.3
            elif self._calculate_name_similarity(node1.name, node2.name) > 0.7:
                similarity += 0.2
        
        # Similitud de estructura de hijos
        children_similarity = self._calculate_children_similarity(node1, node2)
        similarity += children_similarity * 0.2
        
        return min(similarity, 1.0)
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calcula similitud entre nombres."""
        # Implementación básica - se expandirá con algoritmos de distancia de edición
        if name1.lower() == name2.lower():
            return 1.0
        elif name1.lower() in name2.lower() or name2.lower() in name1.lower():
            return 0.7
        else:
            return 0.0
    
    def _calculate_children_similarity(self, node1: UnifiedNode, node2: UnifiedNode) -> float:
        """Calcula similitud entre estructuras de hijos."""
        if len(node1.children) == 0 and len(node2.children) == 0:
            return 1.0
        
        if len(node1.children) == 0 or len(node2.children) == 0:
            return 0.0
        
        # Comparar tipos de nodos hijos
        types1 = [child.node_type for child in node1.children]
        types2 = [child.node_type for child in node2.children]
        
        common_types = set(types1) & set(types2)
        total_types = set(types1) | set(types2)
        
        if len(total_types) == 0:
            return 0.0
        
        return len(common_types) / len(total_types)
