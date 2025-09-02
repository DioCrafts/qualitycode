"""
Motor de Comparación Cross-Language.

Este módulo implementa el sistema de comparación que permite analizar
similitudes y diferencias entre código de diferentes lenguajes.
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


class DifferenceType(str, Enum):
    """Tipos de diferencias."""
    STRUCTURAL_DIFFERENCE = "structural_difference"
    SEMANTIC_DIFFERENCE = "semantic_difference"
    SYNTACTIC_DIFFERENCE = "syntactic_difference"
    TYPE_DIFFERENCE = "type_difference"
    BEHAVIORAL_DIFFERENCE = "behavioral_difference"


class DifferenceImpact(str, Enum):
    """Impacto de las diferencias."""
    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass
class ComparisonResult:
    """Resultado de una comparación."""
    overall_similarity: float
    structural_comparison: 'StructuralComparison'
    semantic_comparison: 'SemanticComparison'
    behavioral_comparison: 'BehavioralComparison'
    differences: List['CodeDifference']
    equivalences: List['CodeEquivalence']


@dataclass
class StructuralComparison:
    """Comparación estructural."""
    similarity_score: float
    common_structures: List[str]
    unique_structures: Dict[str, List[str]]
    structural_metrics: Dict[str, float]


@dataclass
class SemanticComparison:
    """Comparación semántica."""
    similarity_score: float
    common_concepts: List[str]
    concept_mappings: Dict[str, str]
    semantic_metrics: Dict[str, float]


@dataclass
class BehavioralComparison:
    """Comparación comportamental."""
    similarity_score: float
    common_behaviors: List[str]
    behavior_differences: List[str]
    behavioral_metrics: Dict[str, float]


@dataclass
class CodeDifference:
    """Diferencia en el código."""
    difference_type: DifferenceType
    description: str
    location_a: NodeId
    location_b: NodeId
    impact: DifferenceImpact
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeEquivalence:
    """Equivalencia en el código."""
    concept: str
    implementation_a: NodeId
    implementation_b: NodeId
    confidence: float
    explanation: str


@dataclass
class CrossLanguageComparison:
    """Comparación cross-language."""
    pairwise_comparisons: Dict[tuple[str, str], ComparisonResult]
    concept_equivalences: List[CodeEquivalence]
    translation_difficulty: Dict[str, float]


class ComparisonError(Exception):
    """Error en el sistema de comparación."""
    
    def __init__(self, message: str, comparison_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.comparison_type = comparison_type
        self.details = details or {}


class ComparisonEngine:
    """Motor principal de comparación."""
    
    def __init__(self):
        self.structural_comparator = StructuralComparator()
        self.semantic_comparator = SemanticComparator()
        self.behavioral_comparator = BehavioralComparator()
        self.similarity_metrics = SimilarityMetrics()
    
    async def compare_code_fragments(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> ComparisonResult:
        """Compara dos fragmentos de código."""
        try:
            # Comparación estructural
            structural_similarity = await self.structural_comparator.compare(fragment_a, fragment_b)
            
            # Comparación semántica
            semantic_similarity = await self.semantic_comparator.compare(fragment_a, fragment_b)
            
            # Comparación comportamental
            behavioral_similarity = await self.behavioral_comparator.compare(fragment_a, fragment_b)
            
            # Calcular similitud general
            overall_similarity = self.similarity_metrics.calculate_weighted_similarity(
                structural_similarity.similarity_score,
                semantic_similarity.similarity_score,
                behavioral_similarity.similarity_score,
            )
            
            # Identificar diferencias
            differences = await self.identify_differences(fragment_a, fragment_b)
            
            # Identificar equivalencias
            equivalences = await self.identify_equivalences(fragment_a, fragment_b)
            
            return ComparisonResult(
                overall_similarity=overall_similarity,
                structural_comparison=structural_similarity,
                semantic_comparison=semantic_similarity,
                behavioral_comparison=behavioral_similarity,
                differences=differences,
                equivalences=equivalences,
            )
            
        except Exception as e:
            raise ComparisonError(f"Error comparing code fragments: {str(e)}")
    
    async def compare_across_languages(self, fragments: Dict[str, UnifiedNode]) -> CrossLanguageComparison:
        """Compara fragmentos de código entre diferentes lenguajes."""
        pairwise_comparisons = {}
        concept_equivalences = []
        translation_difficulty = {}
        
        languages = list(fragments.keys())
        
        # Comparar cada par de lenguajes
        for i in range(len(languages)):
            for j in range(i + 1, len(languages)):
                lang_a = languages[i]
                lang_b = languages[j]
                
                fragment_a = fragments[lang_a]
                fragment_b = fragments[lang_b]
                
                comparison = await self.compare_code_fragments(fragment_a, fragment_b)
                pairwise_comparisons[(lang_a, lang_b)] = comparison
                
                # Identificar equivalencias de conceptos
                concept_equivs = await self.identify_concept_equivalences(fragment_a, fragment_b, lang_a, lang_b)
                concept_equivalences.extend(concept_equivs)
                
                # Evaluar dificultad de traducción
                difficulty = self.assess_translation_difficulty(comparison)
                translation_difficulty[f"{lang_a}_to_{lang_b}"] = difficulty
        
        return CrossLanguageComparison(
            pairwise_comparisons=pairwise_comparisons,
            concept_equivalences=concept_equivalences,
            translation_difficulty=translation_difficulty,
        )
    
    async def identify_differences(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> List[CodeDifference]:
        """Identifica diferencias entre dos fragmentos."""
        differences = []
        
        # Diferencias estructurales
        structural_diffs = await self.structural_comparator.identify_differences(fragment_a, fragment_b)
        differences.extend(structural_diffs)
        
        # Diferencias semánticas
        semantic_diffs = await self.semantic_comparator.identify_differences(fragment_a, fragment_b)
        differences.extend(semantic_diffs)
        
        # Diferencias comportamentales
        behavioral_diffs = await self.behavioral_comparator.identify_differences(fragment_a, fragment_b)
        differences.extend(behavioral_diffs)
        
        return differences
    
    async def identify_equivalences(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> List[CodeEquivalence]:
        """Identifica equivalencias entre dos fragmentos."""
        equivalences = []
        
        # Equivalencias estructurales
        structural_equivs = await self.structural_comparator.identify_equivalences(fragment_a, fragment_b)
        equivalences.extend(structural_equivs)
        
        # Equivalencias semánticas
        semantic_equivs = await self.semantic_comparator.identify_equivalences(fragment_a, fragment_b)
        equivalences.extend(semantic_equivs)
        
        return equivalences
    
    async def identify_concept_equivalences(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode, 
                                          lang_a: str, lang_b: str) -> List[CodeEquivalence]:
        """Identifica equivalencias de conceptos entre lenguajes."""
        equivalences = []
        
        # Mapear conceptos comunes
        concept_mappings = {
            "function": ["function", "def", "fn"],
            "class": ["class", "struct", "trait"],
            "loop": ["for", "while", "loop"],
            "condition": ["if", "match", "switch"],
        }
        
        for concept, keywords in concept_mappings.items():
            if self._has_concept(fragment_a, keywords) and self._has_concept(fragment_b, keywords):
                equivalences.append(CodeEquivalence(
                    concept=concept,
                    implementation_a=fragment_a.id,
                    implementation_b=fragment_b.id,
                    confidence=0.8,
                    explanation=f"Both fragments implement {concept} concept",
                ))
        
        return equivalences
    
    def _has_concept(self, fragment: UnifiedNode, keywords: List[str]) -> bool:
        """Determina si un fragmento tiene un concepto específico."""
        def traverse(node: UnifiedNode):
            if node.name and any(keyword in node.name.lower() for keyword in keywords):
                return True
            
            for child in node.children:
                if traverse(child):
                    return True
            
            return False
        
        return traverse(fragment)
    
    def assess_translation_difficulty(self, comparison: ComparisonResult) -> float:
        """Evalúa la dificultad de traducción basada en la comparación."""
        # Factor de similitud estructural
        structural_factor = comparison.structural_comparison.similarity_score
        
        # Factor de similitud semántica
        semantic_factor = comparison.semantic_comparison.similarity_score
        
        # Factor de diferencias críticas
        critical_differences = sum(1 for diff in comparison.differences 
                                 if diff.impact == DifferenceImpact.CRITICAL)
        difference_factor = max(0, 1 - (critical_differences * 0.2))
        
        # Calcular dificultad (0 = fácil, 1 = difícil)
        difficulty = 1 - ((structural_factor + semantic_factor + difference_factor) / 3)
        
        return min(max(difficulty, 0.0), 1.0)


class StructuralComparator:
    """Comparador estructural."""
    
    async def compare(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> StructuralComparison:
        """Compara la estructura de dos fragmentos."""
        # Calcular similitud estructural
        similarity_score = self._calculate_structural_similarity(fragment_a, fragment_b)
        
        # Identificar estructuras comunes
        common_structures = self._find_common_structures(fragment_a, fragment_b)
        
        # Identificar estructuras únicas
        unique_structures = self._find_unique_structures(fragment_a, fragment_b)
        
        # Calcular métricas estructurales
        structural_metrics = self._calculate_structural_metrics(fragment_a, fragment_b)
        
        return StructuralComparison(
            similarity_score=similarity_score,
            common_structures=common_structures,
            unique_structures=unique_structures,
            structural_metrics=structural_metrics,
        )
    
    def _calculate_structural_similarity(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> float:
        """Calcula la similitud estructural entre dos fragmentos."""
        similarity = 0.0
        
        # Similitud de tipo de nodo
        if fragment_a.node_type == fragment_b.node_type:
            similarity += 0.4
        
        # Similitud de estructura de hijos
        children_similarity = self._compare_children_structure(fragment_a, fragment_b)
        similarity += children_similarity * 0.3
        
        # Similitud de profundidad
        depth_a = self._calculate_depth(fragment_a)
        depth_b = self._calculate_depth(fragment_b)
        depth_similarity = 1 - abs(depth_a - depth_b) / max(depth_a, depth_b, 1)
        similarity += depth_similarity * 0.2
        
        # Similitud de complejidad
        complexity_a = self._calculate_complexity(fragment_a)
        complexity_b = self._calculate_complexity(fragment_b)
        complexity_similarity = 1 - abs(complexity_a - complexity_b) / max(complexity_a, complexity_b, 1)
        similarity += complexity_similarity * 0.1
        
        return min(similarity, 1.0)
    
    def _compare_children_structure(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> float:
        """Compara la estructura de los hijos."""
        if len(fragment_a.children) == 0 and len(fragment_b.children) == 0:
            return 1.0
        
        if len(fragment_a.children) == 0 or len(fragment_b.children) == 0:
            return 0.0
        
        # Comparar tipos de nodos hijos
        types_a = [child.node_type for child in fragment_a.children]
        types_b = [child.node_type for child in fragment_b.children]
        
        common_types = set(types_a) & set(types_b)
        total_types = set(types_a) | set(types_b)
        
        if len(total_types) == 0:
            return 0.0
        
        return len(common_types) / len(total_types)
    
    def _calculate_depth(self, fragment: UnifiedNode) -> int:
        """Calcula la profundidad de un fragmento."""
        if not fragment.children:
            return 1
        
        max_depth = 1
        for child in fragment.children:
            child_depth = self._calculate_depth(child)
            max_depth = max(max_depth, child_depth + 1)
        
        return max_depth
    
    def _calculate_complexity(self, fragment: UnifiedNode) -> float:
        """Calcula la complejidad de un fragmento."""
        complexity = 1.0
        
        # Añadir complejidad basada en el tipo de nodo
        if fragment.node_type in [UnifiedNodeType.IF_STATEMENT, UnifiedNodeType.FOR_STATEMENT, 
                                 UnifiedNodeType.WHILE_STATEMENT, UnifiedNodeType.LOOP_STATEMENT]:
            complexity += 2.0
        elif fragment.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
            complexity += 1.5
        elif fragment.node_type == UnifiedNodeType.CLASS_DECLARATION:
            complexity += 2.5
        
        # Añadir complejidad de los hijos
        for child in fragment.children:
            complexity += self._calculate_complexity(child)
        
        return complexity
    
    def _find_common_structures(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> List[str]:
        """Encuentra estructuras comunes entre dos fragmentos."""
        structures_a = self._extract_structures(fragment_a)
        structures_b = self._extract_structures(fragment_b)
        
        return list(structures_a & structures_b)
    
    def _find_unique_structures(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> Dict[str, List[str]]:
        """Encuentra estructuras únicas en cada fragmento."""
        structures_a = self._extract_structures(fragment_a)
        structures_b = self._extract_structures(fragment_b)
        
        return {
            "fragment_a": list(structures_a - structures_b),
            "fragment_b": list(structures_b - structures_a),
        }
    
    def _extract_structures(self, fragment: UnifiedNode) -> Set[str]:
        """Extrae estructuras de un fragmento."""
        structures = set()
        
        def traverse(node: UnifiedNode):
            structures.add(node.node_type.value)
            
            for child in node.children:
                traverse(child)
        
        traverse(fragment)
        return structures
    
    def _calculate_structural_metrics(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> Dict[str, float]:
        """Calcula métricas estructurales."""
        return {
            "depth_difference": abs(self._calculate_depth(fragment_a) - self._calculate_depth(fragment_b)),
            "complexity_difference": abs(self._calculate_complexity(fragment_a) - self._calculate_complexity(fragment_b)),
            "children_count_difference": abs(len(fragment_a.children) - len(fragment_b.children)),
        }
    
    async def identify_differences(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> List[CodeDifference]:
        """Identifica diferencias estructurales."""
        differences = []
        
        # Diferencia de tipo de nodo
        if fragment_a.node_type != fragment_b.node_type:
            differences.append(CodeDifference(
                difference_type=DifferenceType.STRUCTURAL_DIFFERENCE,
                description=f"Different node types: {fragment_a.node_type} vs {fragment_b.node_type}",
                location_a=fragment_a.id,
                location_b=fragment_b.id,
                impact=DifferenceImpact.MODERATE,
            ))
        
        # Diferencia en número de hijos
        if len(fragment_a.children) != len(fragment_b.children):
            differences.append(CodeDifference(
                difference_type=DifferenceType.STRUCTURAL_DIFFERENCE,
                description=f"Different number of children: {len(fragment_a.children)} vs {len(fragment_b.children)}",
                location_a=fragment_a.id,
                location_b=fragment_b.id,
                impact=DifferenceImpact.MINOR,
            ))
        
        return differences
    
    async def identify_equivalences(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> List[CodeEquivalence]:
        """Identifica equivalencias estructurales."""
        equivalences = []
        
        # Equivalencia de tipo de nodo
        if fragment_a.node_type == fragment_b.node_type:
            equivalences.append(CodeEquivalence(
                concept="structural_type",
                implementation_a=fragment_a.id,
                implementation_b=fragment_b.id,
                confidence=0.9,
                explanation=f"Both fragments have the same structural type: {fragment_a.node_type}",
            ))
        
        return equivalences


class SemanticComparator:
    """Comparador semántico."""
    
    async def compare(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> SemanticComparison:
        """Compara la semántica de dos fragmentos."""
        # Calcular similitud semántica
        similarity_score = self._calculate_semantic_similarity(fragment_a, fragment_b)
        
        # Identificar conceptos comunes
        common_concepts = self._find_common_concepts(fragment_a, fragment_b)
        
        # Mapear conceptos
        concept_mappings = self._map_concepts(fragment_a, fragment_b)
        
        # Calcular métricas semánticas
        semantic_metrics = self._calculate_semantic_metrics(fragment_a, fragment_b)
        
        return SemanticComparison(
            similarity_score=similarity_score,
            common_concepts=common_concepts,
            concept_mappings=concept_mappings,
            semantic_metrics=semantic_metrics,
        )
    
    def _calculate_semantic_similarity(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> float:
        """Calcula la similitud semántica entre dos fragmentos."""
        similarity = 0.0
        
        # Similitud de tipo semántico
        if fragment_a.semantic_type == fragment_b.semantic_type:
            similarity += 0.5
        
        # Similitud de nombre
        if fragment_a.name and fragment_b.name:
            name_similarity = self._calculate_name_similarity(fragment_a.name, fragment_b.name)
            similarity += name_similarity * 0.3
        
        # Similitud de valor
        if fragment_a.value and fragment_b.value:
            value_similarity = self._calculate_value_similarity(fragment_a.value, fragment_b.value)
            similarity += value_similarity * 0.2
        
        return min(similarity, 1.0)
    
    def _calculate_name_similarity(self, name_a: str, name_b: str) -> float:
        """Calcula la similitud entre nombres."""
        if name_a.lower() == name_b.lower():
            return 1.0
        elif name_a.lower() in name_b.lower() or name_b.lower() in name_a.lower():
            return 0.7
        else:
            return 0.0
    
    def _calculate_value_similarity(self, value_a: Any, value_b: Any) -> float:
        """Calcula la similitud entre valores."""
        if str(value_a) == str(value_b):
            return 1.0
        else:
            return 0.0
    
    def _find_common_concepts(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> List[str]:
        """Encuentra conceptos comunes entre dos fragmentos."""
        concepts_a = self._extract_concepts(fragment_a)
        concepts_b = self._extract_concepts(fragment_b)
        
        return list(concepts_a & concepts_b)
    
    def _map_concepts(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> Dict[str, str]:
        """Mapea conceptos entre dos fragmentos."""
        # Implementación básica - se expandirá
        return {}
    
    def _extract_concepts(self, fragment: UnifiedNode) -> Set[str]:
        """Extrae conceptos de un fragmento."""
        concepts = set()
        
        def traverse(node: UnifiedNode):
            # Añadir concepto basado en el tipo semántico
            concepts.add(node.semantic_type.value)
            
            # Añadir concepto basado en el nombre
            if node.name:
                concepts.add(f"named_{node.name.lower()}")
            
            for child in node.children:
                traverse(child)
        
        traverse(fragment)
        return concepts
    
    def _calculate_semantic_metrics(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> Dict[str, float]:
        """Calcula métricas semánticas."""
        return {
            "concept_overlap": len(self._find_common_concepts(fragment_a, fragment_b)),
            "naming_similarity": self._calculate_name_similarity(fragment_a.name or "", fragment_b.name or ""),
        }
    
    async def identify_differences(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> List[CodeDifference]:
        """Identifica diferencias semánticas."""
        differences = []
        
        # Diferencia de tipo semántico
        if fragment_a.semantic_type != fragment_b.semantic_type:
            differences.append(CodeDifference(
                difference_type=DifferenceType.SEMANTIC_DIFFERENCE,
                description=f"Different semantic types: {fragment_a.semantic_type} vs {fragment_b.semantic_type}",
                location_a=fragment_a.id,
                location_b=fragment_b.id,
                impact=DifferenceImpact.MAJOR,
            ))
        
        return differences
    
    async def identify_equivalences(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> List[CodeEquivalence]:
        """Identifica equivalencias semánticas."""
        equivalences = []
        
        # Equivalencia de tipo semántico
        if fragment_a.semantic_type == fragment_b.semantic_type:
            equivalences.append(CodeEquivalence(
                concept="semantic_type",
                implementation_a=fragment_a.id,
                implementation_b=fragment_b.id,
                confidence=0.8,
                explanation=f"Both fragments have the same semantic type: {fragment_a.semantic_type}",
            ))
        
        return equivalences


class BehavioralComparator:
    """Comparador comportamental."""
    
    async def compare(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> BehavioralComparison:
        """Compara el comportamiento de dos fragmentos."""
        # Calcular similitud comportamental
        similarity_score = self._calculate_behavioral_similarity(fragment_a, fragment_b)
        
        # Identificar comportamientos comunes
        common_behaviors = self._find_common_behaviors(fragment_a, fragment_b)
        
        # Identificar diferencias de comportamiento
        behavior_differences = self._find_behavior_differences(fragment_a, fragment_b)
        
        # Calcular métricas comportamentales
        behavioral_metrics = self._calculate_behavioral_metrics(fragment_a, fragment_b)
        
        return BehavioralComparison(
            similarity_score=similarity_score,
            common_behaviors=common_behaviors,
            behavior_differences=behavior_differences,
            behavioral_metrics=behavioral_metrics,
        )
    
    def _calculate_behavioral_similarity(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> float:
        """Calcula la similitud comportamental entre dos fragmentos."""
        # Implementación básica - se expandirá
        return 0.5
    
    def _find_common_behaviors(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> List[str]:
        """Encuentra comportamientos comunes entre dos fragmentos."""
        # Implementación básica - se expandirá
        return []
    
    def _find_behavior_differences(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> List[str]:
        """Encuentra diferencias de comportamiento entre dos fragmentos."""
        # Implementación básica - se expandirá
        return []
    
    def _calculate_behavioral_metrics(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> Dict[str, float]:
        """Calcula métricas comportamentales."""
        # Implementación básica - se expandirá
        return {}
    
    async def identify_differences(self, fragment_a: UnifiedNode, fragment_b: UnifiedNode) -> List[CodeDifference]:
        """Identifica diferencias comportamentales."""
        # Implementación básica - se expandirá
        return []


class SimilarityMetrics:
    """Métricas de similitud."""
    
    def calculate_weighted_similarity(self, structural_similarity: float, semantic_similarity: float, 
                                    behavioral_similarity: float) -> float:
        """Calcula la similitud ponderada general."""
        # Pesos para cada tipo de similitud
        weights = {
            'structural': 0.4,
            'semantic': 0.4,
            'behavioral': 0.2,
        }
        
        weighted_sum = (
            structural_similarity * weights['structural'] +
            semantic_similarity * weights['semantic'] +
            behavioral_similarity * weights['behavioral']
        )
        
        return min(max(weighted_sum, 0.0), 1.0)
