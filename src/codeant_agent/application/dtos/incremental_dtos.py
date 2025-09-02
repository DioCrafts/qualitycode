"""
Data Transfer Objects (DTOs) para el Sistema de Análisis Incremental.

Este módulo define los DTOs que se utilizan para transferir datos
entre las capas de la aplicación.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


# DTOs para Detección de Cambios

@dataclass
class DetectChangesRequestDTO:
    """DTO para solicitar detección de cambios."""
    repository_path: str
    from_commit: Optional[str] = None
    to_commit: Optional[str] = None
    granularity_level: str = "function"
    enable_granular_detection: bool = True
    calculate_impact: bool = True
    include_dependencies: bool = True


@dataclass
class DetectChangesResponseDTO:
    """DTO para respuesta de detección de cambios."""
    change_set_id: str
    total_files_changed: int
    total_changes: int
    impact_score: float
    detection_time_ms: int
    affected_dependencies: List[str] = field(default_factory=list)


# DTOs para Análisis Incremental

@dataclass
class AnalyzeIncrementalRequestDTO:
    """DTO para solicitar análisis incremental."""
    change_set_id: str
    analysis_types: List[str] = field(default_factory=lambda: ["complexity", "security", "quality"])
    use_cache: bool = True
    parallel_execution: bool = True
    project_size: int = 0


@dataclass
class AnalyzeIncrementalResponseDTO:
    """DTO para respuesta de análisis incremental."""
    analysis_id: str
    files_analyzed: int
    cache_hits: int
    cache_misses: int
    total_time_ms: int
    incremental_speedup: float
    results_summary: Dict[str, Any] = field(default_factory=dict)


# DTOs para Cache

@dataclass
class GetCachedAnalysisRequestDTO:
    """DTO para solicitar análisis cacheado."""
    file_path: str
    analysis_type: str
    content_hash: Optional[str] = None


@dataclass
class GetCachedAnalysisResponseDTO:
    """DTO para respuesta de análisis cacheado."""
    found: bool
    result: Optional[Any] = None
    cache_level: Optional[str] = None
    retrieval_time_ms: int = 0
    expiry_time: Optional[datetime] = None


@dataclass
class InvalidateCacheRequestDTO:
    """DTO para solicitar invalidación de cache."""
    pattern: Optional[str] = None
    file_paths: List[str] = field(default_factory=list)
    cache_levels: List[str] = field(default_factory=lambda: ["L1", "L2", "L3"])
    invalidate_dependencies: bool = True
    max_dependency_depth: Optional[int] = None
    reason: Optional[str] = None


@dataclass
class InvalidateCacheResponseDTO:
    """DTO para respuesta de invalidación de cache."""
    invalidated_count: int
    affected_files: List[str]
    execution_time_ms: int


# DTOs para Predicción de Cache

@dataclass
class PredictCacheNeedsRequestDTO:
    """DTO para solicitar predicción de necesidades de cache."""
    prediction_window_minutes: int = 30
    confidence_threshold: float = 0.7
    lookback_hours: int = 24
    include_patterns: bool = True


@dataclass
class PredictCacheNeedsResponseDTO:
    """DTO para respuesta de predicción de cache."""
    predictions: List[Dict[str, Any]]
    total_predictions: int
    average_confidence: float
    prediction_accuracy: float
    generation_time_ms: int


@dataclass
class WarmCacheRequestDTO:
    """DTO para solicitar cache warming."""
    predictions: Optional[List[Dict[str, Any]]] = None
    max_items: Optional[int] = None
    priority_threshold: float = 0.5
    parallel_warming: bool = True


@dataclass
class WarmCacheResponseDTO:
    """DTO para respuesta de cache warming."""
    warmed_items: int
    total_predictions: int
    cache_size_mb: float
    warming_time_ms: int
    success_rate: float = 1.0


# DTOs para Dependencias

@dataclass
class GetDependenciesRequestDTO:
    """DTO para solicitar dependencias."""
    file_path: str
    include_transitive: bool = False
    include_dependents: bool = False
    max_depth: Optional[int] = None


@dataclass
class GetDependenciesResponseDTO:
    """DTO para respuesta de dependencias."""
    file_path: str
    direct_dependencies: List[str]
    transitive_dependencies: List[str]
    dependent_files: List[str]
    total_dependencies: int
    circular_dependencies: List[List[str]] = field(default_factory=list)


# DTOs para Métricas

@dataclass
class GetCacheMetricsRequestDTO:
    """DTO para solicitar métricas del cache."""
    time_window_hours: int = 24
    include_details: bool = True


@dataclass
class GetCacheMetricsResponseDTO:
    """DTO para respuesta de métricas del cache."""
    hit_rate: float
    total_hits: int
    total_misses: int
    cache_size_mb: float
    eviction_count: int
    average_retrieval_time_ms: float
    cache_levels: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class GetIncrementalMetricsRequestDTO:
    """DTO para solicitar métricas incrementales."""
    time_window_hours: int = 24
    group_by: str = "hour"


@dataclass
class GetIncrementalMetricsResponseDTO:
    """DTO para respuesta de métricas incrementales."""
    total_analyses: int
    incremental_analyses: int
    average_speedup: float
    total_time_saved_ms: int
    cache_efficiency: float
    time_series: List[Dict[str, Any]] = field(default_factory=list)


# DTOs para Optimización

@dataclass
class OptimizeCacheRequestDTO:
    """DTO para solicitar optimización de cache."""
    optimize_distribution: bool = True
    optimize_warming_strategy: bool = True
    analyze_bottlenecks: bool = True
    target_hit_rate: float = 0.8


@dataclass
class OptimizeCacheResponseDTO:
    """DTO para respuesta de optimización de cache."""
    optimizations_performed: List[str]
    performance_improvement: float
    recommendations: List[Dict[str, Any]]
    execution_time_ms: int


# DTOs para Delta Processing

@dataclass
class ComputeDeltaRequestDTO:
    """DTO para solicitar cómputo de delta."""
    file_path: str
    old_version: str
    new_version: str
    include_semantic_diff: bool = True


@dataclass
class ComputeDeltaResponseDTO:
    """DTO para respuesta de cómputo de delta."""
    delta_id: str
    changes_count: int
    delta_size_bytes: int
    computation_time_ms: int
    semantic_changes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ApplyDeltaRequestDTO:
    """DTO para solicitar aplicación de delta."""
    base_analysis_id: str
    delta_id: str
    validate_result: bool = True


@dataclass
class ApplyDeltaResponseDTO:
    """DTO para respuesta de aplicación de delta."""
    result_id: str
    success: bool
    application_time_ms: int
    validation_passed: bool = True
    error_message: Optional[str] = None


# DTOs para Análisis de Impacto

@dataclass
class AnalyzeImpactRequestDTO:
    """DTO para solicitar análisis de impacto."""
    change_set_id: str
    impact_types: List[str] = field(default_factory=lambda: ["dependencies", "performance", "security"])
    max_depth: int = 3


@dataclass
class AnalyzeImpactResponseDTO:
    """DTO para respuesta de análisis de impacto."""
    impact_summary: Dict[str, float]
    affected_components: List[str]
    risk_level: str
    recommendations: List[str]
    analysis_time_ms: int


# DTOs para Configuración

@dataclass
class UpdateConfigRequestDTO:
    """DTO para actualizar configuración."""
    cache_config: Optional[Dict[str, Any]] = None
    analysis_config: Optional[Dict[str, Any]] = None
    prediction_config: Optional[Dict[str, Any]] = None


@dataclass
class UpdateConfigResponseDTO:
    """DTO para respuesta de actualización de configuración."""
    updated_sections: List[str]
    validation_errors: List[str]
    restart_required: bool


# DTOs para Estado del Sistema

@dataclass
class GetSystemStatusRequestDTO:
    """DTO para solicitar estado del sistema."""
    include_performance: bool = True
    include_resources: bool = True
    include_errors: bool = True


@dataclass
class GetSystemStatusResponseDTO:
    """DTO para respuesta de estado del sistema."""
    status: str
    uptime_seconds: int
    active_analyses: int
    cache_status: Dict[str, Any]
    resource_usage: Dict[str, float]
    recent_errors: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
