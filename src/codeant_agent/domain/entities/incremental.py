"""
Entidades del dominio para el Sistema de Análisis Incremental.

Este módulo define las entidades principales del sistema de análisis incremental,
siguiendo los principios de la arquitectura hexagonal y DDD.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from pathlib import Path


class GranularityLevel(Enum):
    """Niveles de granularidad para detección de cambios."""
    FILE = "file"               # Cambios a nivel de archivo
    FUNCTION = "function"       # Cambios a nivel de función
    CLASS = "class"            # Cambios a nivel de clase
    STATEMENT = "statement"     # Cambios a nivel de sentencia
    EXPRESSION = "expression"   # Cambios a nivel de expresión
    TOKEN = "token"            # Cambios a nivel de token


class ChangeType(Enum):
    """Tipos de cambios detectados."""
    FILE_ADDED = "file_added"
    FILE_REMOVED = "file_removed"
    FILE_MODIFIED = "file_modified"
    FILE_MOVED = "file_moved"
    FILE_RENAMED = "file_renamed"
    FUNCTION_ADDED = "function_added"
    FUNCTION_REMOVED = "function_removed"
    FUNCTION_MODIFIED = "function_modified"
    FUNCTION_MOVED = "function_moved"
    FUNCTION_RENAMED = "function_renamed"
    CLASS_ADDED = "class_added"
    CLASS_REMOVED = "class_removed"
    CLASS_MODIFIED = "class_modified"
    STATEMENT_ADDED = "statement_added"
    STATEMENT_REMOVED = "statement_removed"
    STATEMENT_MODIFIED = "statement_modified"
    EXPRESSION_MODIFIED = "expression_modified"
    TOKEN_MODIFIED = "token_modified"


class CacheType(Enum):
    """Tipos de datos cacheables."""
    PARSED_AST = "parsed_ast"
    ANALYSIS_RESULT = "analysis_result"
    AI_EMBEDDING = "ai_embedding"
    RULE_RESULT = "rule_result"
    METRICS_RESULT = "metrics_result"
    DEPENDENCY_GRAPH = "dependency_graph"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    ANTIPATTERN_DETECTION = "antipattern_detection"


class CacheLevel(Enum):
    """Niveles de cache en el sistema multi-nivel."""
    L1 = "l1"  # Memory (hot cache)
    L2 = "l2"  # Redis/In-memory DB (warm cache)
    L3 = "l3"  # Disk/Object Storage (cold cache)


class EvictionPolicy(Enum):
    """Políticas de desalojo de cache."""
    LRU = "lru"        # Least Recently Used
    LFU = "lfu"        # Least Frequently Used
    FIFO = "fifo"      # First In First Out
    LIFO = "lifo"      # Last In First Out
    TTL = "ttl"        # Time To Live
    SIZE = "size"      # Size-based eviction


class ScopeReason(Enum):
    """Razones para incluir algo en el scope de análisis."""
    DIRECT_CHANGE = "direct_change"
    DEPENDENCY_CHANGE = "dependency_change"
    SEMANTIC_CHANGE = "semantic_change"
    CROSS_FILE_IMPACT = "cross_file_impact"
    RULE_SPECIFIC_REQUIREMENT = "rule_specific_requirement"


class PredictionSource(Enum):
    """Fuentes de predicción para cache warming."""
    HISTORICAL_PATTERN = "historical_pattern"
    TIME_PATTERN = "time_pattern"
    CO_MODIFICATION_PATTERN = "co_modification_pattern"
    USER_BEHAVIOR_PATTERN = "user_behavior_pattern"
    MACHINE_LEARNING = "machine_learning"
    DEPENDENCY_ANALYSIS = "dependency_analysis"


@dataclass
class ChangeSetId:
    """Identificador único para un conjunto de cambios."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass
class GranularChangeId:
    """Identificador único para un cambio granular."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass
class ChangeLocation:
    """Ubicación de un cambio en el código."""
    file_path: Path
    line_range: Optional[Tuple[int, int]] = None
    column_range: Optional[Tuple[int, int]] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    expression_type: Optional[str] = None


@dataclass
class FileChange:
    """Cambio detectado a nivel de archivo."""
    file_path: Path
    change_type: ChangeType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GranularChange:
    """Cambio granular detectado."""
    id: GranularChangeId
    change_type: ChangeType
    location: ChangeLocation
    content_hash: str
    affected_symbols: List[str] = field(default_factory=list)
    impact_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyImpact:
    """Impacto de cambios en las dependencias."""
    directly_affected_files: List[Path] = field(default_factory=list)
    transitively_affected_files: List[Path] = field(default_factory=list)
    affected_symbols: Dict[str, List[Path]] = field(default_factory=dict)
    impact_radius: int = 0
    estimated_analysis_scope: float = 0.0  # Porcentaje del proyecto a re-analizar


@dataclass
class FunctionIdentifier:
    """Identificador de función para análisis selectivo."""
    file_path: Path
    function_name: str
    line_number: int
    signature: str


@dataclass
class ClassIdentifier:
    """Identificador de clase para análisis selectivo."""
    file_path: Path
    class_name: str
    line_number: int
    methods: List[str] = field(default_factory=list)


@dataclass
class AnalysisScope:
    """Alcance del análisis incremental."""
    files_to_analyze: List[Path] = field(default_factory=list)
    functions_to_analyze: List[FunctionIdentifier] = field(default_factory=list)
    classes_to_analyze: List[ClassIdentifier] = field(default_factory=list)
    rules_to_execute: List[str] = field(default_factory=list)
    scope_justification: List[ScopeReason] = field(default_factory=list)
    estimated_analysis_time: timedelta = field(default_factory=lambda: timedelta(seconds=0))


@dataclass
class ChangeStatistics:
    """Estadísticas de cambios detectados."""
    total_files_changed: int = 0
    total_functions_changed: int = 0
    total_lines_added: int = 0
    total_lines_removed: int = 0
    total_lines_modified: int = 0
    change_complexity_score: float = 0.0
    estimated_impact_score: float = 0.0


@dataclass
class ChangeSet:
    """Conjunto de cambios detectados."""
    id: ChangeSetId
    file_changes: List[FileChange]
    granular_changes: List[GranularChange]
    dependency_impact: DependencyImpact
    analysis_scope: AnalysisScope
    change_statistics: ChangeStatistics
    detection_time_ms: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CacheKey:
    """Clave para el sistema de cache."""
    cache_type: CacheType
    primary_key: str
    secondary_keys: List[str] = field(default_factory=list)
    version: str = "1.0"
    language: Optional[str] = None
    
    def to_string(self) -> str:
        """Convertir a string único para uso como clave."""
        parts = [
            self.cache_type.value,
            self.primary_key,
            self.version
        ]
        if self.secondary_keys:
            parts.extend(self.secondary_keys)
        if self.language:
            parts.append(self.language)
        return ":".join(parts)
    
    @classmethod
    def for_ast(cls, file_path: Path, content_hash: str) -> 'CacheKey':
        """Crear clave para AST parseado."""
        return cls(
            cache_type=CacheType.PARSED_AST,
            primary_key=str(file_path),
            secondary_keys=[content_hash]
        )
    
    @classmethod
    def for_analysis_result(cls, file_path: Path, rules_hash: str, content_hash: str) -> 'CacheKey':
        """Crear clave para resultado de análisis."""
        return cls(
            cache_type=CacheType.ANALYSIS_RESULT,
            primary_key=str(file_path),
            secondary_keys=[rules_hash, content_hash]
        )
    
    @classmethod
    def for_embedding(cls, code_snippet: str, model_id: str, language: str) -> 'CacheKey':
        """Crear clave para embedding de IA."""
        import hashlib
        snippet_hash = hashlib.blake2b(code_snippet.encode()).hexdigest()
        return cls(
            cache_type=CacheType.AI_EMBEDDING,
            primary_key=snippet_hash,
            secondary_keys=[model_id],
            language=language
        )


@dataclass
class CacheEntry:
    """Entrada en el sistema de cache."""
    key: CacheKey
    value: Any
    size_bytes: int
    access_count: int = 0
    last_access_time: datetime = field(default_factory=datetime.utcnow)
    creation_time: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Verificar si la entrada ha expirado."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.utcnow() - self.creation_time).total_seconds()
        return age > self.ttl_seconds


@dataclass
class CachePerformance:
    """Métricas de rendimiento del cache."""
    ast_cache_hit: bool = False
    analysis_cache_hit: bool = False
    embedding_cache_hit: bool = False
    metrics_cache_hit: bool = False
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    total_accesses: int = 0
    hit_ratio: float = 0.0
    
    def merge(self, other: 'CachePerformance') -> None:
        """Combinar métricas con otra instancia."""
        self.l1_hits += other.l1_hits
        self.l2_hits += other.l2_hits
        self.l3_hits += other.l3_hits
        self.total_accesses += other.total_accesses
        
        total_hits = self.l1_hits + self.l2_hits + self.l3_hits
        if self.total_accesses > 0:
            self.hit_ratio = total_hits / self.total_accesses
        else:
            self.hit_ratio = 0.0


@dataclass
class InvalidationRequest:
    """Solicitud de invalidación de cache."""
    invalidation_type: str  # "key", "pattern", "dependencies"
    target: str  # Key específica, patrón o base key
    cascade_dependencies: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InvalidationResult:
    """Resultado de una operación de invalidación."""
    invalidated_keys: List[CacheKey]
    invalidation_time_ms: int
    cache_hit_ratio_impact: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReusableComponents:
    """Componentes que pueden ser reutilizados del cache."""
    can_reuse_ast: bool = True
    can_reuse_analysis: bool = True
    can_reuse_embeddings: bool = True
    can_reuse_metrics: bool = True
    reuse_confidence: float = 1.0


@dataclass
class AccessPrediction:
    """Predicción de acceso futuro para cache warming."""
    cache_key: CacheKey
    access_probability: float
    priority_score: float
    predicted_access_time: datetime
    prediction_source: PredictionSource
    confidence: float = 0.5


@dataclass
class CacheWarmingEntry:
    """Entrada calentada en el cache."""
    cache_key: CacheKey
    warming_time_ms: int
    success: bool
    data_size_bytes: int
    prediction_accuracy: float = 0.0


@dataclass
class CacheWarmingResult:
    """Resultado del proceso de cache warming."""
    predictions_made: int
    cache_entries_warmed: int
    warming_time_ms: int
    estimated_time_savings: int
    success_rate: float = 0.0


@dataclass
class IncrementalAnalysisResult:
    """Resultado del análisis incremental."""
    change_set_id: ChangeSetId
    files_analyzed: List[Path]
    cache_performance: CachePerformance
    analysis_results: Dict[Path, Any]  # Path -> AnalysisResult
    delta_results: Dict[Path, Any]    # Path -> DeltaAnalysisResult
    invalidated_cache_keys: List[CacheKey]
    time_saved_ms: int
    total_analysis_time_ms: int
    is_full_analysis_recommended: bool = False
    recommendation_reason: Optional[str] = None


@dataclass
class DeltaAnalysisResult:
    """Resultado del análisis delta de un archivo."""
    file_path: Path
    changes_processed: List[GranularChangeId]
    invalidated_results: List[str]
    new_results: List[str]
    delta_size_bytes: int
    processing_time_ms: int


@dataclass
class FileIncrementalResult:
    """Resultado incremental para un archivo específico."""
    file_path: Path
    analysis_result: Any  # AnalysisResult
    cache_performance: CachePerformance
    components_reused: ReusableComponents
    processing_time_ms: int = 0
