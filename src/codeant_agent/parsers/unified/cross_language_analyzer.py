"""
Motor de Análisis Cross-Language.

Este módulo implementa el sistema de análisis que trasciende las barreras de lenguaje,
permitiendo comparaciones semánticas, detección de patrones similares y mapeo de conceptos
entre diferentes lenguajes de programación.
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


class ProgrammingConcept(str, Enum):
    """Conceptos de programación que pueden ser mapeados entre lenguajes."""
    FUNCTION_DEFINITION = "function_definition"
    CLASS_DEFINITION = "class_definition"
    LOOP_CONSTRUCT = "loop_construct"
    CONDITIONAL_STATEMENT = "conditional_statement"
    ERROR_HANDLING = "error_handling"
    ASYNCHRONOUS_OPERATION = "asynchronous_operation"
    DATA_STRUCTURE = "data_structure"
    MEMORY_MANAGEMENT = "memory_management"
    TYPE_SYSTEM = "type_system"
    MODULE_SYSTEM = "module_system"
    ITERATOR_PATTERN = "iterator_pattern"
    BUILDER_PATTERN = "builder_pattern"
    FACTORY_PATTERN = "factory_pattern"
    OBSERVER_PATTERN = "observer_pattern"
    SINGLETON_PATTERN = "singleton_pattern"
    STRATEGY_PATTERN = "strategy_pattern"
    DECORATOR_PATTERN = "decorator_pattern"
    ADAPTER_PATTERN = "adapter_pattern"
    FACADE_PATTERN = "facade_pattern"
    COMMAND_PATTERN = "command_pattern"


class PatternType(str, Enum):
    """Tipos de patrones que pueden ser detectados."""
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    CREATIONAL = "creational"
    ARCHITECTURAL = "architectural"
    IDIOMATIC = "idiomatic"
    ANTI_PATTERN = "anti_pattern"


@dataclass
class SimilarPattern:
    """Patrón similar encontrado entre lenguajes."""
    pattern_type: PatternType
    languages: List[str]
    nodes: List[NodeId]
    similarity_score: float
    description: str
    concept: ProgrammingConcept
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptMapping:
    """Mapeo de un concepto entre lenguajes."""
    concept: ProgrammingConcept
    implementations: Dict[str, List[NodeId]]
    equivalence_score: float
    description: str
    examples: Dict[str, str] = field(default_factory=dict)
    best_practices: List[str] = field(default_factory=list)


@dataclass
class TranslationSuggestion:
    """Sugerencia de traducción entre lenguajes."""
    from_language: str
    to_language: str
    source_node: NodeId
    suggested_translation: str
    confidence: float
    explanation: str
    code_example: str = ""
    alternatives: List[str] = field(default_factory=list)


@dataclass
class CrossLanguageAntiPattern:
    """Anti-patrón detectado en múltiples lenguajes."""
    name: str
    description: str
    languages: List[str]
    affected_nodes: List[NodeId]
    severity: str
    impact: str
    suggestions: List[str] = field(default_factory=list)


@dataclass
class BestPractice:
    """Mejor práctica identificada."""
    name: str
    description: str
    languages: List[str]
    examples: Dict[str, str] = field(default_factory=dict)
    benefits: List[str] = field(default_factory=list)
    implementation_notes: str = ""


@dataclass
class LanguageMigration:
    """Sugerencia de migración entre lenguajes."""
    from_language: str
    to_language: str
    reason: str
    complexity: str
    estimated_effort: str
    benefits: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)


@dataclass
class CrossLanguageAnalysis:
    """Resultado completo del análisis cross-language."""
    similar_patterns: List[SimilarPattern]
    concept_mappings: List[ConceptMapping]
    translation_suggestions: List[TranslationSuggestion]
    anti_patterns: List[CrossLanguageAntiPattern]
    best_practices: List[BestPractice]
    language_migrations: List[LanguageMigration]
    summary: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


class CrossLanguageAnalyzer:
    """Motor principal de análisis cross-language."""
    
    def __init__(self):
        self.pattern_matcher = CrossLanguagePatternMatcher()
        self.similarity_analyzer = SimilarityAnalyzer()
        self.concept_mapper = ConceptMapper()
        self.translation_engine = TranslationEngine()
        self.anti_pattern_detector = AntiPatternDetector()
        self.best_practice_identifier = BestPracticeIdentifier()
        self.migration_advisor = MigrationAdvisor()
    
    async def analyze_cross_language_patterns(self, asts: List[UnifiedAST]) -> CrossLanguageAnalysis:
        """Analiza patrones cross-language en múltiples ASTs."""
        start_time = datetime.now()
        
        analysis = CrossLanguageAnalysis(
            similar_patterns=[],
            concept_mappings=[],
            translation_suggestions=[],
            anti_patterns=[],
            best_practices=[],
            language_migrations=[],
        )
        
        try:
            # Encontrar patrones similares entre lenguajes
            analysis.similar_patterns = await self.find_similar_patterns(asts)
            
            # Mapear conceptos entre lenguajes
            analysis.concept_mappings = await self.concept_mapper.map_concepts(asts)
            
            # Generar sugerencias de traducción
            analysis.translation_suggestions = await self.translation_engine.suggest_translations(asts)
            
            # Detectar anti-patrones cross-language
            analysis.anti_patterns = await self.detect_cross_language_antipatterns(asts)
            
            # Identificar mejores prácticas
            analysis.best_practices = await self.best_practice_identifier.identify_best_practices(asts)
            
            # Sugerir migraciones entre lenguajes
            analysis.language_migrations = await self.migration_advisor.suggest_migrations(asts)
            
            # Generar resumen
            analysis.summary = self._generate_summary(analysis)
            
        except Exception as e:
            logger.error(f"Error en análisis cross-language: {e}")
            raise
        
        # Calcular tiempo de ejecución
        end_time = datetime.now()
        analysis.execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return analysis
    
    async def find_similar_patterns(self, asts: List[UnifiedAST]) -> List[SimilarPattern]:
        """Encuentra patrones similares entre diferentes lenguajes."""
        similar_patterns = []
        
        # Comparar cada par de ASTs
        for i in range(len(asts)):
            for j in range(i + 1, len(asts)):
                patterns = await self.similarity_analyzer.find_similarities(asts[i], asts[j])
                similar_patterns.extend(patterns)
        
        # Agrupar patrones similares
        grouped_patterns = self._group_similar_patterns(similar_patterns)
        
        return grouped_patterns
    
    async def detect_cross_language_antipatterns(self, asts: List[UnifiedAST]) -> List[CrossLanguageAntiPattern]:
        """Detecta anti-patrones que aparecen en múltiples lenguajes."""
        anti_patterns = []
        
        for ast in asts:
            patterns = await self.anti_pattern_detector.detect_antipatterns(ast)
            anti_patterns.extend(patterns)
        
        # Agrupar anti-patrones por tipo
        grouped_antipatterns = self._group_antipatterns(anti_patterns)
        
        return grouped_antipatterns
    
    def _group_similar_patterns(self, patterns: List[SimilarPattern]) -> List[SimilarPattern]:
        """Agrupa patrones similares para evitar duplicados."""
        grouped = {}
        
        for pattern in patterns:
            key = f"{pattern.pattern_type}_{pattern.concept}"
            if key not in grouped:
                grouped[key] = pattern
            else:
                # Combinar patrones similares
                existing = grouped[key]
                existing.languages.extend(pattern.languages)
                existing.nodes.extend(pattern.nodes)
                existing.similarity_score = max(existing.similarity_score, pattern.similarity_score)
                existing.confidence = max(existing.confidence, pattern.confidence)
        
        return list(grouped.values())
    
    def _group_antipatterns(self, anti_patterns: List[CrossLanguageAntiPattern]) -> List[CrossLanguageAntiPattern]:
        """Agrupa anti-patrones por nombre."""
        grouped = {}
        
        for anti_pattern in anti_patterns:
            if anti_pattern.name not in grouped:
                grouped[anti_pattern.name] = anti_pattern
            else:
                # Combinar anti-patrones del mismo tipo
                existing = grouped[anti_pattern.name]
                existing.languages.extend(anti_pattern.languages)
                existing.affected_nodes.extend(anti_pattern.affected_nodes)
        
        return list(grouped.values())
    
    def _generate_summary(self, analysis: CrossLanguageAnalysis) -> Dict[str, Any]:
        """Genera un resumen del análisis cross-language."""
        return {
            "total_patterns": len(analysis.similar_patterns),
            "total_concepts": len(analysis.concept_mappings),
            "total_translations": len(analysis.translation_suggestions),
            "total_antipatterns": len(analysis.anti_patterns),
            "total_best_practices": len(analysis.best_practices),
            "total_migrations": len(analysis.language_migrations),
            "languages_analyzed": list(set(
                lang for pattern in analysis.similar_patterns 
                for lang in pattern.languages
            )),
            "most_common_patterns": self._get_most_common_patterns(analysis.similar_patterns),
            "most_common_antipatterns": self._get_most_common_antipatterns(analysis.anti_patterns),
        }
    
    def _get_most_common_patterns(self, patterns: List[SimilarPattern]) -> List[Dict[str, Any]]:
        """Obtiene los patrones más comunes."""
        pattern_counts = {}
        for pattern in patterns:
            concept = pattern.concept.value
            pattern_counts[concept] = pattern_counts.get(concept, 0) + 1
        
        return [
            {"concept": concept, "count": count}
            for concept, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def _get_most_common_antipatterns(self, anti_patterns: List[CrossLanguageAntiPattern]) -> List[Dict[str, Any]]:
        """Obtiene los anti-patrones más comunes."""
        antipattern_counts = {}
        for anti_pattern in anti_patterns:
            antipattern_counts[anti_pattern.name] = antipattern_counts.get(anti_pattern.name, 0) + 1
        
        return [
            {"name": name, "count": count}
            for name, count in sorted(antipattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]


class CrossLanguagePatternMatcher:
    """Matcher de patrones cross-language."""
    
    def __init__(self):
        self.pattern_library = self._load_pattern_library()
    
    def _load_pattern_library(self) -> Dict[str, Any]:
        """Carga la biblioteca de patrones cross-language."""
        return {
            "function_definition": {
                "python": ["def", "async def"],
                "typescript": ["function", "const", "let"],
                "javascript": ["function", "const", "let"],
                "rust": ["fn"],
            },
            "class_definition": {
                "python": ["class"],
                "typescript": ["class"],
                "javascript": ["class"],
                "rust": ["struct", "enum", "trait"],
            },
            "loop_construct": {
                "python": ["for", "while"],
                "typescript": ["for", "while", "forEach"],
                "javascript": ["for", "while", "forEach"],
                "rust": ["for", "while", "loop"],
            },
        }
    
    async def find_patterns(self, ast: UnifiedAST, pattern_type: str) -> List[SimilarPattern]:
        """Encuentra patrones de un tipo específico en un AST."""
        patterns = []
        
        if pattern_type in self.pattern_library:
            pattern_def = self.pattern_library[pattern_type]
            
            # Buscar nodos que coincidan con el patrón
            matching_nodes = self._find_matching_nodes(ast, pattern_def)
            
            if matching_nodes:
                patterns.append(SimilarPattern(
                    pattern_type=PatternType.STRUCTURAL,
                    languages=[ast.language],
                    nodes=[node.id for node in matching_nodes],
                    similarity_score=0.8,
                    description=f"Found {pattern_type} pattern in {ast.language}",
                    concept=ProgrammingConcept(pattern_type.upper()),
                    confidence=0.7
                ))
        
        return patterns
    
    def _find_matching_nodes(self, ast: UnifiedAST, pattern_def: Dict[str, List[str]]) -> List[UnifiedNode]:
        """Encuentra nodos que coincidan con una definición de patrón."""
        matching_nodes = []
        
        def traverse(node: UnifiedNode):
            if self._node_matches_pattern(node, pattern_def):
                matching_nodes.append(node)
            
            for child in node.children:
                traverse(child)
        
        traverse(ast.root_node)
        return matching_nodes
    
    def _node_matches_pattern(self, node: UnifiedNode, pattern_def: Dict[str, List[str]]) -> bool:
        """Determina si un nodo coincide con un patrón."""
        # Implementación básica - se expandirá
        return False


class SimilarityAnalyzer:
    """Analizador de similitud entre ASTs."""
    
    async def find_similarities(self, ast1: UnifiedAST, ast2: UnifiedAST) -> List[SimilarPattern]:
        """Encuentra similitudes entre dos ASTs."""
        similarities = []
        
        # Comparar estructuras de nodos
        structural_similarities = self._compare_structural_similarities(ast1, ast2)
        similarities.extend(structural_similarities)
        
        # Comparar patrones semánticos
        semantic_similarities = self._compare_semantic_similarities(ast1, ast2)
        similarities.extend(semantic_similarities)
        
        return similarities
    
    def _compare_structural_similarities(self, ast1: UnifiedAST, ast2: UnifiedAST) -> List[SimilarPattern]:
        """Compara similitudes estructurales."""
        similarities = []
        
        # Comparar tipos de nodos
        node_types_1 = self._get_node_type_distribution(ast1)
        node_types_2 = self._get_node_type_distribution(ast2)
        
        common_types = set(node_types_1.keys()) & set(node_types_2.keys())
        
        for node_type in common_types:
            if node_types_1[node_type] > 0 and node_types_2[node_type] > 0:
                similarities.append(SimilarPattern(
                    pattern_type=PatternType.STRUCTURAL,
                    languages=[ast1.language, ast2.language],
                    nodes=[],  # Se llenaría con nodos específicos
                    similarity_score=0.6,
                    description=f"Common {node_type.value} usage",
                    concept=ProgrammingConcept.UNKNOWN,
                    confidence=0.5
                ))
        
        return similarities
    
    def _compare_semantic_similarities(self, ast1: UnifiedAST, ast2: UnifiedAST) -> List[SimilarPattern]:
        """Compara similitudes semánticas."""
        similarities = []
        
        # Comparar símbolos
        symbols_1 = set(ast1.semantic_info.symbols.keys())
        symbols_2 = set(ast2.semantic_info.symbols.keys())
        
        common_symbols = symbols_1 & symbols_2
        
        if common_symbols:
            similarities.append(SimilarPattern(
                pattern_type=PatternType.STRUCTURAL,
                languages=[ast1.language, ast2.language],
                nodes=[],
                similarity_score=0.7,
                description=f"Common symbols: {', '.join(list(common_symbols)[:3])}",
                concept=ProgrammingConcept.UNKNOWN,
                confidence=0.6
            ))
        
        return similarities
    
    def _get_node_type_distribution(self, ast: UnifiedAST) -> Dict[UnifiedNodeType, int]:
        """Obtiene la distribución de tipos de nodos en un AST."""
        distribution = {}
        
        def traverse(node: UnifiedNode):
            node_type = node.node_type
            distribution[node_type] = distribution.get(node_type, 0) + 1
            
            for child in node.children:
                traverse(child)
        
        traverse(ast.root_node)
        return distribution


class ConceptMapper:
    """Mapeador de conceptos entre lenguajes."""
    
    async def map_concepts(self, asts: List[UnifiedAST]) -> List[ConceptMapping]:
        """Mapea conceptos entre diferentes lenguajes."""
        concept_mappings = []
        
        # Mapear conceptos básicos
        basic_concepts = [
            ProgrammingConcept.FUNCTION_DEFINITION,
            ProgrammingConcept.CLASS_DEFINITION,
            ProgrammingConcept.LOOP_CONSTRUCT,
            ProgrammingConcept.CONDITIONAL_STATEMENT,
        ]
        
        for concept in basic_concepts:
            mapping = await self._map_concept(asts, concept)
            if mapping:
                concept_mappings.append(mapping)
        
        return concept_mappings
    
    async def _map_concept(self, asts: List[UnifiedAST], concept: ProgrammingConcept) -> Optional[ConceptMapping]:
        """Mapea un concepto específico entre lenguajes."""
        implementations = {}
        
        for ast in asts:
            nodes = self._find_concept_implementations(ast, concept)
            if nodes:
                implementations[ast.language] = [node.id for node in nodes]
        
        if len(implementations) >= 2:
            return ConceptMapping(
                concept=concept,
                implementations=implementations,
                equivalence_score=0.8,
                description=f"Concept {concept.value} found in {len(implementations)} languages"
            )
        
        return None
    
    def _find_concept_implementations(self, ast: UnifiedAST, concept: ProgrammingConcept) -> List[UnifiedNode]:
        """Encuentra implementaciones de un concepto en un AST."""
        implementations = []
        
        def traverse(node: UnifiedNode):
            if self._node_implements_concept(node, concept):
                implementations.append(node)
            
            for child in node.children:
                traverse(child)
        
        traverse(ast.root_node)
        return implementations
    
    def _node_implements_concept(self, node: UnifiedNode, concept: ProgrammingConcept) -> bool:
        """Determina si un nodo implementa un concepto específico."""
        concept_mapping = {
            ProgrammingConcept.FUNCTION_DEFINITION: UnifiedNodeType.FUNCTION_DECLARATION,
            ProgrammingConcept.CLASS_DEFINITION: UnifiedNodeType.CLASS_DECLARATION,
            ProgrammingConcept.LOOP_CONSTRUCT: [
                UnifiedNodeType.FOR_STATEMENT,
                UnifiedNodeType.WHILE_STATEMENT,
                UnifiedNodeType.LOOP_STATEMENT,
            ],
            ProgrammingConcept.CONDITIONAL_STATEMENT: UnifiedNodeType.IF_STATEMENT,
        }
        
        if concept in concept_mapping:
            expected_types = concept_mapping[concept]
            if isinstance(expected_types, list):
                return node.node_type in expected_types
            else:
                return node.node_type == expected_types
        
        return False


class TranslationEngine:
    """Motor de traducción entre lenguajes."""
    
    async def suggest_translations(self, asts: List[UnifiedAST]) -> List[TranslationSuggestion]:
        """Sugiere traducciones entre lenguajes."""
        suggestions = []
        
        # Generar sugerencias para cada par de lenguajes
        for i in range(len(asts)):
            for j in range(i + 1, len(asts)):
                ast1, ast2 = asts[i], asts[j]
                pair_suggestions = await self._generate_translation_suggestions(ast1, ast2)
                suggestions.extend(pair_suggestions)
        
        return suggestions
    
    async def _generate_translation_suggestions(self, ast1: UnifiedAST, ast2: UnifiedAST) -> List[TranslationSuggestion]:
        """Genera sugerencias de traducción entre dos ASTs."""
        suggestions = []
        
        # Encontrar funciones que podrían ser traducidas
        functions_1 = ast1.find_nodes_by_type(UnifiedNodeType.FUNCTION_DECLARATION)
        functions_2 = ast2.find_nodes_by_type(UnifiedNodeType.FUNCTION_DECLARATION)
        
        for func1 in functions_1[:3]:  # Limitar a 3 funciones para evitar demasiadas sugerencias
            suggestion = TranslationSuggestion(
                from_language=ast1.language,
                to_language=ast2.language,
                source_node=func1.id,
                suggested_translation=self._generate_function_translation(func1, ast2.language),
                confidence=0.6,
                explanation=f"Function '{func1.name}' could be translated to {ast2.language}",
                code_example=self._generate_code_example(func1, ast2.language)
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_function_translation(self, func_node: UnifiedNode, target_language: str) -> str:
        """Genera una traducción de función a un lenguaje específico."""
        # Implementación básica - se expandirá
        return f"// Function translation to {target_language}"
    
    def _generate_code_example(self, func_node: UnifiedNode, target_language: str) -> str:
        """Genera un ejemplo de código para la traducción."""
        # Implementación básica - se expandirá
        return f"// Code example for {target_language}"


class AntiPatternDetector:
    """Detector de anti-patrones."""
    
    async def detect_antipatterns(self, ast: UnifiedAST) -> List[CrossLanguageAntiPattern]:
        """Detecta anti-patrones en un AST."""
        anti_patterns = []
        
        # Detectar anti-patrones comunes
        anti_patterns.extend(self._detect_unused_variables(ast))
        anti_patterns.extend(self._detect_long_functions(ast))
        anti_patterns.extend(self._detect_duplicate_code(ast))
        
        return anti_patterns
    
    def _detect_unused_variables(self, ast: UnifiedAST) -> List[CrossLanguageAntiPattern]:
        """Detecta variables no utilizadas."""
        # Implementación básica
        return []
    
    def _detect_long_functions(self, ast: UnifiedAST) -> List[CrossLanguageAntiPattern]:
        """Detecta funciones muy largas."""
        # Implementación básica
        return []
    
    def _detect_duplicate_code(self, ast: UnifiedAST) -> List[CrossLanguageAntiPattern]:
        """Detecta código duplicado."""
        # Implementación básica
        return []


class BestPracticeIdentifier:
    """Identificador de mejores prácticas."""
    
    async def identify_best_practices(self, asts: List[UnifiedAST]) -> List[BestPractice]:
        """Identifica mejores prácticas en los ASTs."""
        best_practices = []
        
        # Identificar mejores prácticas comunes
        best_practices.extend(self._identify_naming_conventions(asts))
        best_practices.extend(self._identify_structure_patterns(asts))
        
        return best_practices
    
    def _identify_naming_conventions(self, asts: List[UnifiedAST]) -> List[BestPractice]:
        """Identifica convenciones de nomenclatura."""
        # Implementación básica
        return []
    
    def _identify_structure_patterns(self, asts: List[UnifiedAST]) -> List[BestPractice]:
        """Identifica patrones de estructura."""
        # Implementación básica
        return []


class MigrationAdvisor:
    """Asesor de migración entre lenguajes."""
    
    async def suggest_migrations(self, asts: List[UnifiedAST]) -> List[LanguageMigration]:
        """Sugiere migraciones entre lenguajes."""
        migrations = []
        
        # Analizar patrones para sugerir migraciones
        for i in range(len(asts)):
            for j in range(i + 1, len(asts)):
                migration = await self._analyze_migration_potential(asts[i], asts[j])
                if migration:
                    migrations.append(migration)
        
        return migrations
    
    async def _analyze_migration_potential(self, ast1: UnifiedAST, ast2: UnifiedAST) -> Optional[LanguageMigration]:
        """Analiza el potencial de migración entre dos ASTs."""
        # Implementación básica
        return None
