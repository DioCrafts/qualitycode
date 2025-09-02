"""
Sistema AST Unificado Cross-Language.

Este módulo implementa un sistema unificado que integra todos los parsers especializados
en una representación AST coherente y normalizada que permite análisis cross-language,
comparaciones semánticas entre lenguajes, y una base sólida para el motor de reglas
y análisis de IA del agente CodeAnt.
"""

from .unified_ast import (
    UnifiedAST,
    UnifiedNode,
    UnifiedNodeType,
    SemanticNodeType,
    UnifiedValue,
    TypedValue,
    UnifiedPosition,
    UnifiedType,
    Parameter,
    UnifiedASTMetadata,
    UnifiedSemanticInfo,
    CrossLanguageMapping,
    ASTId,
    NodeId,
    ASTVersion,
)
from .ast_unifier import (
    ASTUnifier,
    LanguageUnifier,
    UnificationConfig,
    UnificationError,
)
from .language_unifiers import (
    PythonUnifier,
    TypeScriptUnifier,
    JavaScriptUnifier,
    RustUnifier,
)
from .cross_language_analyzer import (
    CrossLanguageAnalyzer,
    CrossLanguageAnalysis,
    SimilarPattern,
    ConceptMapping,
    ProgrammingConcept,
    TranslationSuggestion,
    CrossLanguageAntiPattern,
    BestPractice,
    LanguageMigration,
)
from .query_engine import (
    UnifiedQueryEngine,
    UnifiedQuery,
    QueryType,
    QueryFilter,
    QueryProjection,
    QueryAggregation,
    QueryResult,
    CrossLanguageQueryResult,
    QueryError,
)
from .pattern_matcher import (
    CrossLanguagePatternMatcher,
    Pattern,
    PatternTemplate,
    NodePattern,
    PatternQuantifier,
    PatternMatch,
    CrossLanguagePatternMatch,
    PatternError,
)
from .comparison_engine import (
    ComparisonEngine,
    ComparisonResult,
    CodeDifference,
    DifferenceType,
    DifferenceImpact,
    CrossLanguageComparison,
    ComparisonError,
)
from .semantic_normalizer import (
    SemanticNormalizer,
    ConceptMapper,
    TypeUnifier,
    CrossLanguageMapper,
)

__all__ = [
    # Unified AST Core
    'UnifiedAST',
    'UnifiedNode',
    'UnifiedNodeType',
    'SemanticNodeType',
    'UnifiedValue',
    'TypedValue',
    'UnifiedPosition',
    'UnifiedType',
    'Parameter',
    'UnifiedASTMetadata',
    'UnifiedSemanticInfo',
    'CrossLanguageMapping',
    'ASTId',
    'NodeId',
    'ASTVersion',
    
    # AST Unifier
    'ASTUnifier',
    'LanguageUnifier',
    'UnificationConfig',
    'UnificationError',
    
    # Language Unifiers
    'PythonUnifier',
    'TypeScriptUnifier',
    'JavaScriptUnifier',
    'RustUnifier',
    
    # Cross-Language Analysis
    'CrossLanguageAnalyzer',
    'CrossLanguageAnalysis',
    'SimilarPattern',
    'ConceptMapping',
    'ProgrammingConcept',
    'TranslationSuggestion',
    'CrossLanguageAntiPattern',
    'BestPractice',
    'LanguageMigration',
    
    # Query Engine
    'UnifiedQueryEngine',
    'UnifiedQuery',
    'QueryType',
    'QueryFilter',
    'QueryProjection',
    'QueryAggregation',
    'QueryResult',
    'CrossLanguageQueryResult',
    'QueryError',
    
    # Pattern Matching
    'CrossLanguagePatternMatcher',
    'Pattern',
    'PatternTemplate',
    'NodePattern',
    'PatternQuantifier',
    'PatternMatch',
    'CrossLanguagePatternMatch',
    'PatternError',
    
    # Comparison Engine
    'ComparisonEngine',
    'ComparisonResult',
    'CodeDifference',
    'DifferenceType',
    'DifferenceImpact',
    'CrossLanguageComparison',
    'ComparisonError',
    
    # Semantic Normalization
    'SemanticNormalizer',
    'ConceptMapper',
    'TypeUnifier',
    'CrossLanguageMapper',
]
