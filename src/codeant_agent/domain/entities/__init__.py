"""
Entidades del dominio del agente CodeAnt.

Este módulo contiene todas las entidades del dominio que representan
los conceptos principales del negocio.
"""

from .project import Project
from .repository import Repository
from .file_index import FileIndex
from .user import User
from .parse_result import ParseResult, ParseRequest, ParseStatus, ParseWarning, ParseMetadata
from .language_detection import DetectionResult, DetectionContext, DetectionStrategy, DetectionConfidence
from .parse_cache import CachedParseResult, CacheConfig, CacheStats, CacheLevel, CacheStatus, EvictionPolicy
from .ast_query import QueryRequest, QueryResult, QueryMatch, QueryCapture, QueryType, QueryStatus
from .ast_normalization import (
    NormalizedAST, NormalizedNode, NodeType, NodeVisibility, NodeModifier,
    SourcePosition, SemanticInfo
)
from .parser_errors import (
    ParserError, ParseWarning as ParserParseWarning, ErrorContext, ErrorReport,
    ErrorSeverity, ErrorCategory, RecoveryStrategy
)
from .parser_config import (
    ParserConfig, LanguageSpecificConfig, ParserProfile,
    ParserMode, CacheStrategy, ParallelStrategy
)
from .dead_code_analysis import (
    DeadCodeAnalysis, ProjectDeadCodeAnalysis, UnusedVariable, UnusedFunction,
    UnusedClass, UnusedImport, UnreachableCode, DeadBranch, UnusedParameter,
    RedundantAssignment, DeadCodeStatistics, EntryPoint, CrossModuleIssue,
    UnusedReason, UnreachabilityReason, ImportType, Visibility, FunctionType,
    ScopeType, AssignmentType, RedundancyType, EntryPointType, SourcePosition,
    SourceRange, ScopeInfo, Parameter, SideEffect, BlockingCondition,
    ImportStatement
)
from .dependency_analysis import (
    ControlFlowGraph, ControlFlowNode, ControlFlowEdge, DependencyGraph,
    GlobalDependencyGraph, SymbolInfo, ModuleInfo, Dependency, LivenessInfo,
    DefUseChain, UsageLocation, UsageAnalysis, NodeId, SymbolId, ModuleId,
    NodeType, EdgeType, DependencyType, SymbolType
)
from .clone_analysis import (
    CloneAnalysis, ProjectCloneAnalysis, Clone, ExactClone, StructuralClone,
    SemanticClone, CrossLanguageClone, InterFileClone, CloneClass, CloneClassId,
    CloneId, DuplicationMetrics, RefactoringOpportunity, RefactoringStep,
    CodeChange, SimilarityMetrics, CloneDetectionConfig, CodeLocation, CodeBlock,
    CloneType, SimilarityAlgorithm, HashAlgorithm, RefactoringType, EstimatedEffort,
    RefactoringBenefit, SemanticUnitType, SimilarityEvidence, StructuralDifference,
    NodeMapping, ConceptMapping, TranslationEvidence, CloneClassMetrics,
    RefactoringPotential
)

__all__ = [
    # Entidades principales
    "Project",
    "Repository", 
    "FileIndex",
    "User",
    
    # Entidades de parsing
    "ParseResult",
    "ParseRequest",
    "ParseStatus",
    "ParseWarning",
    "ParseMetadata",
    
    # Entidades de detección de lenguajes
    "DetectionResult",
    "DetectionContext",
    "DetectionStrategy",
    "DetectionConfidence",
    
    # Entidades de cache
    "CachedParseResult",
    "CacheConfig",
    "CacheStats",
    "CacheLevel",
    "CacheStatus",
    "EvictionPolicy",
    
    # Entidades de queries AST
    "QueryRequest",
    "QueryResult",
    "QueryMatch",
    "QueryCapture",
    "QueryType",
    "QueryStatus",
    
    # Entidades de normalización AST
    "NormalizedAST",
    "NormalizedNode",
    "NodeType",
    "NodeVisibility",
    "NodeModifier",
    "SourcePosition",
    "SemanticInfo",
    
    # Entidades de manejo de errores
    "ParserError",
    "ParserParseWarning",
    "ErrorContext",
    "ErrorReport",
    "ErrorSeverity",
    "ErrorCategory",
    "RecoveryStrategy",
    
    # Entidades de configuración
    "ParserConfig",
    "LanguageSpecificConfig",
    "ParserProfile",
    "ParserMode",
    "CacheStrategy",
    "ParallelStrategy",
    
    # Entidades de análisis de código muerto
    "DeadCodeAnalysis",
    "ProjectDeadCodeAnalysis",
    "UnusedVariable",
    "UnusedFunction",
    "UnusedClass",
    "UnusedImport",
    "UnreachableCode",
    "DeadBranch",
    "UnusedParameter",
    "RedundantAssignment",
    "DeadCodeStatistics",
    "EntryPoint",
    "CrossModuleIssue",
    "UnusedReason",
    "UnreachabilityReason",
    "ImportType",
    "Visibility",
    "FunctionType",
    "ScopeType",
    "AssignmentType",
    "RedundancyType",
    "EntryPointType",
    "SourcePosition",
    "SourceRange",
    "ScopeInfo",
    "Parameter",
    "SideEffect",
    "BlockingCondition",
    "ImportStatement",
    
    # Entidades de análisis de dependencias
    "ControlFlowGraph",
    "ControlFlowNode",
    "ControlFlowEdge",
    "DependencyGraph",
    "GlobalDependencyGraph",
    "SymbolInfo",
    "ModuleInfo",
    "Dependency",
    "LivenessInfo",
    "DefUseChain",
    "UsageLocation",
    "UsageAnalysis",
    "NodeId",
    "SymbolId",
    "ModuleId",
    "NodeType",
    "EdgeType", 
    "DependencyType",
    "SymbolType",
    
    # Entidades de análisis de duplicación
    "CloneAnalysis",
    "ProjectCloneAnalysis", 
    "Clone",
    "ExactClone",
    "StructuralClone",
    "SemanticClone",
    "CrossLanguageClone",
    "InterFileClone",
    "CloneClass",
    "CloneClassId",
    "CloneId",
    "DuplicationMetrics",
    "RefactoringOpportunity",
    "RefactoringStep",
    "CodeChange",
    "SimilarityMetrics",
    "CloneDetectionConfig",
    "CodeLocation",
    "CodeBlock",
    "CloneType",
    "SimilarityAlgorithm",
    "HashAlgorithm",
    "RefactoringType",
    "EstimatedEffort",
    "RefactoringBenefit",
    "SemanticUnitType",
    "SimilarityEvidence",
    "StructuralDifference",
    "NodeMapping",
    "ConceptMapping",
    "TranslationEvidence",
    "CloneClassMetrics",
    "RefactoringPotential",
]
