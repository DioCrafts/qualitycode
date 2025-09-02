"""
Entidades del dominio del agente CodeAnt.

Este módulo contiene todas las entidades del dominio que representan
los conceptos principales del negocio.
"""

# No importamos directamente para evitar problemas de importación circular.
# En lugar de importar desde este archivo, las clases deben importarse directamente
# desde sus respectivos módulos para evitar problemas.
#
# Por ejemplo, en lugar de:
# from codeant_agent.domain.entities import Project
#
# Usar:
# from src.codeant_agent.domain.entities.project import Project

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
