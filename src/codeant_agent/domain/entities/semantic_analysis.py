"""
Entidades del dominio para análisis semántico avanzado.

Este módulo contiene todas las entidades para el sistema de embeddings
multi-nivel, búsqueda semántica y análisis de intención de código.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from enum import Enum
import uuid

from ..value_objects.programming_language import ProgrammingLanguage


class EmbeddingLevel(Enum):
    """Niveles de embeddings jerárquicos."""
    TOKEN = "token"
    EXPRESSION = "expression"
    STATEMENT = "statement"
    FUNCTION = "function"
    CLASS = "class"
    FILE = "file"
    PROJECT = "project"


class AggregationStrategy(Enum):
    """Estrategias de agregación para embeddings."""
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    ATTENTION = "attention"
    HIERARCHICAL = "hierarchical"
    GRAPH_CONVOLUTION = "graph_convolution"


class QueryIntent(Enum):
    """Tipos de intención de consulta."""
    FIND_SIMILAR_CODE = "find_similar_code"
    FIND_BY_FUNCTION = "find_by_function"
    FIND_BY_PATTERN = "find_by_pattern"
    FIND_BY_BEHAVIOR = "find_by_behavior"
    FIND_BY_PURPOSE = "find_by_purpose"
    FIND_IMPLEMENTATIONS = "find_implementations"
    FIND_ALTERNATIVES = "find_alternatives"


class IntentType(Enum):
    """Tipos de intención de código."""
    DATA_RETRIEVAL = "data_retrieval"
    DATA_MODIFICATION = "data_modification"
    DATA_TRANSFORMATION = "data_transformation"
    OBJECT_CREATION = "object_creation"
    OBJECT_DESTRUCTION = "object_destruction"
    VALIDATION = "validation"
    CALCULATION = "calculation"
    COMMUNICATION = "communication"
    ERROR_HANDLING = "error_handling"
    TESTING = "testing"
    LOGGING = "logging"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CONDITIONAL_LOGIC = "conditional_logic"
    ITERATION = "iteration"
    GENERAL = "general"


class ConceptType(Enum):
    """Tipos de conceptos de código."""
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    DESIGN_PATTERN = "design_pattern"
    PROGRAMMING_CONSTRUCT = "programming_construct"
    DOMAIN = "domain"
    FRAMEWORK = "framework"
    LIBRARY = "library"


class AbstractionLevel(Enum):
    """Niveles de abstracción."""
    LOW = "low"          # Hardware/system level
    MEDIUM = "medium"    # Framework/library level  
    HIGH = "high"        # Business/domain level
    VERY_HIGH = "very_high"  # Conceptual/abstract level


class PurposeType(Enum):
    """Tipos de propósito de código."""
    BUSINESS_LOGIC = "business_logic"
    DATA_ACCESS = "data_access"
    USER_INTERFACE = "user_interface"
    INFRASTRUCTURE = "infrastructure"
    UTILITY = "utility"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    APPLICATION = "application"
    SERVICE = "service"
    COMPONENT = "component"


@dataclass
class MultiLevelConfig:
    """Configuración para embeddings multi-nivel."""
    enable_token_embeddings: bool = True
    enable_expression_embeddings: bool = True
    enable_statement_embeddings: bool = True
    enable_function_embeddings: bool = True
    enable_class_embeddings: bool = True
    enable_file_embeddings: bool = True
    enable_project_embeddings: bool = False  # Costoso computacionalmente
    aggregation_strategy: AggregationStrategy = AggregationStrategy.HIERARCHICAL
    context_window_size: int = 10
    embedding_dimensions: Dict[EmbeddingLevel, int] = field(default_factory=lambda: {
        EmbeddingLevel.TOKEN: 256,
        EmbeddingLevel.EXPRESSION: 384,
        EmbeddingLevel.STATEMENT: 512,
        EmbeddingLevel.FUNCTION: 768,
        EmbeddingLevel.CLASS: 768,
        EmbeddingLevel.FILE: 1024,
        EmbeddingLevel.PROJECT: 1536
    })
    enable_hierarchical_attention: bool = True
    max_tokens_per_level: int = 1000


@dataclass
class TokenEmbedding:
    """Embedding de token individual."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token: str = ""
    token_type: str = "identifier"  # identifier, keyword, operator, literal
    embedding_vector: List[float] = field(default_factory=list)
    position: int = 0
    context_tokens: List[str] = field(default_factory=list)
    semantic_role: str = "unknown"  # variable, function_name, class_name, etc.
    
    def get_dimension(self) -> int:
        return len(self.embedding_vector)


@dataclass
class ExpressionEmbedding:
    """Embedding de expresión."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    expression: str = ""
    expression_type: str = "unknown"  # call, assignment, binary_op, etc.
    embedding_vector: List[float] = field(default_factory=list)
    token_embeddings: List[str] = field(default_factory=list)  # IDs de tokens
    start_position: int = 0
    end_position: int = 0
    complexity_score: float = 0.0
    semantic_features: Dict[str, Any] = field(default_factory=dict)
    
    def get_dimension(self) -> int:
        return len(self.embedding_vector)


@dataclass
class StatementEmbedding:
    """Embedding de statement."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""
    statement_type: str = "unknown"  # if, for, return, assignment, etc.
    embedding_vector: List[float] = field(default_factory=list)
    expression_embeddings: List[str] = field(default_factory=list)  # IDs
    control_flow_impact: str = "none"  # none, branching, loop, jump
    variables_used: List[str] = field(default_factory=list)
    variables_defined: List[str] = field(default_factory=list)
    
    def get_dimension(self) -> int:
        return len(self.embedding_vector)


@dataclass
class FunctionEmbedding:
    """Embedding de función."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    function_name: str = ""
    signature: str = ""
    embedding_vector: List[float] = field(default_factory=list)
    statement_embeddings: List[str] = field(default_factory=list)  # IDs
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    intent_analysis: Optional['CodeIntentAnalysis'] = None
    semantic_purpose: Optional[str] = None
    
    def get_dimension(self) -> int:
        return len(self.embedding_vector)
    
    def get_complexity_score(self) -> float:
        return self.complexity_metrics.get('cyclomatic_complexity', 0.0)


@dataclass
class ClassEmbedding:
    """Embedding de clase."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    class_name: str = ""
    embedding_vector: List[float] = field(default_factory=list)
    method_embeddings: List[str] = field(default_factory=list)  # IDs de métodos
    attribute_embeddings: List[str] = field(default_factory=list)  # IDs de atributos
    inheritance_chain: List[str] = field(default_factory=list)
    implemented_interfaces: List[str] = field(default_factory=list)
    design_patterns: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    cohesion_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_dimension(self) -> int:
        return len(self.embedding_vector)
    
    def get_method_count(self) -> int:
        return len(self.method_embeddings)


@dataclass
class FileEmbedding:
    """Embedding de archivo."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Path = Path()
    language: ProgrammingLanguage = ProgrammingLanguage.UNKNOWN
    embedding_vector: List[float] = field(default_factory=list)
    class_embeddings: List[str] = field(default_factory=list)  # IDs
    function_embeddings: List[str] = field(default_factory=list)  # IDs
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    file_purpose: Optional[str] = None
    architectural_role: Optional[str] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_dimension(self) -> int:
        return len(self.embedding_vector)
    
    def get_total_elements(self) -> int:
        return len(self.class_embeddings) + len(self.function_embeddings)


@dataclass
class ProjectEmbedding:
    """Embedding de proyecto."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_name: str = ""
    project_path: Path = Path()
    embedding_vector: List[float] = field(default_factory=list)
    file_embeddings: List[str] = field(default_factory=list)  # IDs
    architecture_patterns: List[str] = field(default_factory=list)
    technology_stack: List[str] = field(default_factory=list)
    domain_concepts: List[str] = field(default_factory=list)
    project_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def get_dimension(self) -> int:
        return len(self.embedding_vector)
    
    def get_file_count(self) -> int:
        return len(self.file_embeddings)


@dataclass
class MultiLevelEmbeddings:
    """Conjunto completo de embeddings multi-nivel."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Path = Path()
    language: ProgrammingLanguage = ProgrammingLanguage.UNKNOWN
    token_embeddings: Dict[str, TokenEmbedding] = field(default_factory=dict)
    expression_embeddings: Dict[str, ExpressionEmbedding] = field(default_factory=dict)
    statement_embeddings: Dict[str, StatementEmbedding] = field(default_factory=dict)
    function_embeddings: Dict[str, FunctionEmbedding] = field(default_factory=dict)
    class_embeddings: Dict[str, ClassEmbedding] = field(default_factory=dict)
    file_embedding: Optional[FileEmbedding] = None
    hierarchical_structure: Optional['HierarchicalStructure'] = None
    semantic_relationships: List['SemanticRelationship'] = field(default_factory=list)
    generation_time_ms: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_total_embeddings(self) -> int:
        return (len(self.token_embeddings) + len(self.expression_embeddings) + 
                len(self.statement_embeddings) + len(self.function_embeddings) + 
                len(self.class_embeddings) + (1 if self.file_embedding else 0))
    
    def get_embeddings_by_level(self, level: EmbeddingLevel) -> Dict[str, Any]:
        level_map = {
            EmbeddingLevel.TOKEN: self.token_embeddings,
            EmbeddingLevel.EXPRESSION: self.expression_embeddings,
            EmbeddingLevel.STATEMENT: self.statement_embeddings,
            EmbeddingLevel.FUNCTION: self.function_embeddings,
            EmbeddingLevel.CLASS: self.class_embeddings,
        }
        return level_map.get(level, {})


@dataclass
class ContextualEmbedding:
    """Embedding contextual con información de contexto."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_node_id: str = ""
    target_embedding: List[float] = field(default_factory=list)
    context_embeddings: List[List[float]] = field(default_factory=list)
    contextual_embedding: List[float] = field(default_factory=list)
    context_window_size: int = 0
    attention_weights: List[float] = field(default_factory=list)
    context_types: List[str] = field(default_factory=list)  # preceding, following, parent, child
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_dimension(self) -> int:
        return len(self.contextual_embedding)
    
    def get_attention_summary(self) -> Dict[str, float]:
        if not self.attention_weights or not self.context_types:
            return {}
        
        summary = {}
        for i, (weight, context_type) in enumerate(zip(self.attention_weights, self.context_types)):
            if context_type in summary:
                summary[context_type] += weight
            else:
                summary[context_type] = weight
        
        return summary


@dataclass
class HierarchicalStructure:
    """Estructura jerárquica de embeddings."""
    levels: Dict[EmbeddingLevel, 'LevelInfo'] = field(default_factory=dict)
    parent_child_relationships: List['HierarchicalRelationship'] = field(default_factory=list)
    aggregation_weights: Dict[EmbeddingLevel, float] = field(default_factory=dict)
    hierarchy_depth: int = 0
    
    def get_level_count(self) -> int:
        return len(self.levels)
    
    def get_total_relationships(self) -> int:
        return len(self.parent_child_relationships)


@dataclass
class LevelInfo:
    """Información de un nivel en la jerarquía."""
    level: EmbeddingLevel
    embedding_count: int = 0
    average_dimension: int = 0
    quality_score: float = 0.0
    aggregation_time_ms: int = 0
    
    def is_complete(self) -> bool:
        return self.embedding_count > 0 and self.average_dimension > 0


@dataclass
class HierarchicalRelationship:
    """Relación jerárquica entre elementos."""
    parent_id: str
    child_id: str
    parent_level: EmbeddingLevel
    child_level: EmbeddingLevel
    relationship_strength: float = 0.0
    relationship_type: str = "contains"  # contains, aggregates, composes
    
    def is_strong_relationship(self, threshold: float = 0.7) -> bool:
        return self.relationship_strength >= threshold


@dataclass
class SemanticRelationship:
    """Relación semántica entre elementos de código."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relationship_type: str = "similar"  # similar, calls, implements, extends, uses
    strength: float = 0.0
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    semantic_distance: float = 0.0
    contextual_relevance: float = 0.0
    
    def is_significant(self, threshold: float = 0.6) -> bool:
        return self.strength >= threshold and self.confidence >= threshold


@dataclass
class CodeConcept:
    """Concepto de código identificado."""
    concept_type: ConceptType
    name: str
    confidence: float = 0.0
    related_concepts: List[str] = field(default_factory=list)
    embedding_vector: Optional[List[float]] = None
    frequency: int = 1
    context: str = ""
    
    def is_confident(self, threshold: float = 0.7) -> bool:
        return self.confidence >= threshold


@dataclass
class DetectedIntent:
    """Intención detectada en el código."""
    intent_type: IntentType
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    context_clues: List[str] = field(default_factory=list)
    behavioral_patterns: List[str] = field(default_factory=list)
    
    def is_confident(self, threshold: float = 0.7) -> bool:
        return self.confidence >= threshold


@dataclass
class CodePurpose:
    """Propósito de código analizado."""
    purpose_type: PurposeType
    description: str = ""
    abstraction_level: AbstractionLevel = AbstractionLevel.MEDIUM
    domain_specificity: float = 0.0
    reusability_score: float = 0.0
    complexity_appropriateness: float = 0.0
    
    def get_quality_score(self) -> float:
        return (self.reusability_score + self.complexity_appropriateness) / 2.0


@dataclass
class CodeIntentAnalysis:
    """Análisis completo de intención de código."""
    code_id: str = ""
    detected_intents: List[DetectedIntent] = field(default_factory=list)
    primary_purpose: Optional[CodePurpose] = None
    behavioral_characteristics: List[str] = field(default_factory=list)
    domain_concepts: List[CodeConcept] = field(default_factory=list)
    confidence_scores: Dict[IntentType, float] = field(default_factory=dict)
    analysis_time_ms: int = 0
    
    def get_primary_intent(self) -> Optional[DetectedIntent]:
        if not self.detected_intents:
            return None
        return max(self.detected_intents, key=lambda x: x.confidence)
    
    def get_confident_intents(self, threshold: float = 0.7) -> List[DetectedIntent]:
        return [intent for intent in self.detected_intents if intent.is_confident(threshold)]


@dataclass
class ProcessedQuery:
    """Query procesada para búsqueda semántica."""
    original_query: str
    intent: QueryIntent
    concepts: List[CodeConcept] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    expanded_terms: List[str] = field(default_factory=list)
    query_embedding: Optional[List[float]] = None
    processing_time_ms: int = 0
    
    def has_embedding(self) -> bool:
        return self.query_embedding is not None and len(self.query_embedding) > 0


@dataclass
class SemanticSearchConfig:
    """Configuración para búsqueda semántica."""
    max_results: int = 50
    similarity_threshold: float = 0.7
    enable_query_expansion: bool = True
    enable_semantic_filtering: bool = True
    enable_cross_language_search: bool = False
    ranking_algorithm: str = "hybrid_ranking"  # cosine, euclidean, hybrid_ranking
    search_timeout_ms: int = 5000
    enable_intent_matching: bool = True
    context_awareness: bool = True


@dataclass
class SemanticFeature:
    """Característica semántica extraída."""
    feature_type: str = "general"  # complexity, abstraction, coupling, etc.
    value: float = 0.0
    description: str = ""
    confidence: float = 0.0
    extraction_method: str = "heuristic"  # heuristic, ml, rule_based
    
    def is_significant(self, threshold: float = 0.5) -> bool:
        return self.confidence >= threshold


@dataclass
class SimilarityExplanation:
    """Explicación de similitud entre códigos."""
    similarity_score: float = 0.0
    similarity_type: str = "semantic"
    explanation_text: str = ""
    key_factors: List[str] = field(default_factory=list)
    confidence: float = 0.0
    comparison_details: Dict[str, Any] = field(default_factory=dict)
    
    def is_reliable(self, threshold: float = 0.6) -> bool:
        return self.confidence >= threshold


@dataclass
class SemanticSearchResultItem:
    """Item de resultado de búsqueda semántica."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    code_snippet: str = ""
    language: ProgrammingLanguage = ProgrammingLanguage.UNKNOWN
    file_path: Optional[Path] = None
    function_name: Optional[str] = None
    cosine_similarity: float = 0.0
    semantic_similarity: float = 0.0
    structural_similarity: float = 0.0
    combined_similarity: float = 0.0
    semantic_features: List[SemanticFeature] = field(default_factory=list)
    explanation: Optional[SimilarityExplanation] = None
    confidence: float = 0.0
    intent_match_score: float = 0.0
    
    def is_high_quality_match(self, threshold: float = 0.8) -> bool:
        return (self.combined_similarity >= threshold and 
                self.confidence >= threshold * 0.9)
    
    def get_match_summary(self) -> Dict[str, float]:
        return {
            "cosine_similarity": self.cosine_similarity,
            "semantic_similarity": self.semantic_similarity,
            "structural_similarity": self.structural_similarity,
            "combined_similarity": self.combined_similarity,
            "confidence": self.confidence,
            "intent_match": self.intent_match_score
        }


@dataclass
class SemanticSearchResult:
    """Resultado completo de búsqueda semántica."""
    query: str
    processed_query: ProcessedQuery
    results: List[SemanticSearchResultItem] = field(default_factory=list)
    total_candidates: int = 0
    total_results: int = 0
    search_time_ms: int = 0
    languages_searched: List[ProgrammingLanguage] = field(default_factory=list)
    query_interpretation: str = ""
    search_strategy_used: str = "default"
    
    def get_best_match(self) -> Optional[SemanticSearchResultItem]:
        if not self.results:
            return None
        return max(self.results, key=lambda x: x.combined_similarity)
    
    def get_high_confidence_results(self, threshold: float = 0.8) -> List[SemanticSearchResultItem]:
        return [result for result in self.results if result.confidence >= threshold]
    
    def get_search_stats(self) -> Dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "returned_results": len(self.results),
            "search_time_ms": self.search_time_ms,
            "average_similarity": sum(r.combined_similarity for r in self.results) / len(self.results) if self.results else 0,
            "languages_searched": len(self.languages_searched),
            "query_complexity": len(self.processed_query.concepts)
        }


@dataclass
class IntentDetectionConfig:
    """Configuración para detección de intención."""
    enable_ml_intent_detection: bool = False  # ML no disponible aún
    enable_pattern_based_detection: bool = True
    enable_domain_analysis: bool = True
    confidence_threshold: float = 0.6
    max_intents_per_code: int = 5
    enable_multi_intent_detection: bool = True
    context_analysis_depth: int = 3


@dataclass
class KnowledgeGraphNode:
    """Nodo en el grafo de conocimiento."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: str = "function"  # function, class, module, concept, pattern
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def has_embedding(self) -> bool:
        return self.embedding is not None and len(self.embedding) > 0


@dataclass
class KnowledgeGraphEdge:
    """Arista en el grafo de conocimiento."""
    source_id: str = ""
    target_id: str = ""
    edge_type: str = "similar"  # similar, calls, implements, extends, uses
    weight: float = 0.0
    confidence: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def is_significant(self, threshold: float = 0.5) -> bool:
        return self.weight >= threshold and self.confidence >= threshold


@dataclass
class KnowledgeGraphResult:
    """Resultado de consulta al grafo de conocimiento."""
    query_type: str = ""
    nodes: List[KnowledgeGraphNode] = field(default_factory=list)
    edges: List[KnowledgeGraphEdge] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    query_time_ms: int = 0
    
    def get_node_count(self) -> int:
        return len(self.nodes)
    
    def get_edge_count(self) -> int:
        return len(self.edges)


@dataclass
class SemanticAnalysisSession:
    """Sesión de análisis semántico."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_name: str = ""
    embeddings_generated: int = 0
    searches_performed: int = 0
    intents_detected: int = 0
    total_processing_time_ms: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def get_duration_seconds(self) -> float:
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def get_processing_rate(self) -> float:
        duration = self.get_duration_seconds()
        return self.embeddings_generated / duration if duration > 0 else 0.0


@dataclass
class SemanticQualityMetrics:
    """Métricas de calidad del análisis semántico."""
    embedding_quality_score: float = 0.0
    search_precision: float = 0.0
    search_recall: float = 0.0
    intent_detection_accuracy: float = 0.0
    semantic_consistency: float = 0.0
    cross_language_coherence: float = 0.0
    
    def get_overall_quality(self) -> float:
        metrics = [
            self.embedding_quality_score,
            self.search_precision,
            self.search_recall,
            self.intent_detection_accuracy,
            self.semantic_consistency
        ]
        return sum(m for m in metrics if m > 0) / len([m for m in metrics if m > 0])


@dataclass
class SemanticAnalysisReport:
    """Reporte de análisis semántico."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    total_files_analyzed: int = 0
    total_embeddings_generated: int = 0
    embedding_levels_used: List[EmbeddingLevel] = field(default_factory=list)
    semantic_relationships_found: int = 0
    intents_detected: Dict[IntentType, int] = field(default_factory=dict)
    quality_metrics: SemanticQualityMetrics = field(default_factory=SemanticQualityMetrics)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_analysis_summary(self) -> str:
        return (f"Analyzed {self.total_files_analyzed} files, "
                f"generated {self.total_embeddings_generated} embeddings, "
                f"found {self.semantic_relationships_found} relationships, "
                f"overall quality: {self.quality_metrics.get_overall_quality():.2f}")


# Aliases para compatibilidad
EmbeddingId = str
NodeId = str
ConceptId = str
