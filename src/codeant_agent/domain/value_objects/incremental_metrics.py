"""
Value Objects para métricas del Sistema de Análisis Incremental.

Este módulo define los value objects relacionados con métricas,
configuraciones y parámetros del sistema de análisis incremental.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class AnalysisMode(Enum):
    """Modos de análisis disponibles."""
    FULL = "full"                  # Análisis completo
    INCREMENTAL = "incremental"    # Análisis incremental
    DELTA = "delta"               # Solo cambios delta
    HYBRID = "hybrid"             # Híbrido según contexto


class CacheStrategy(Enum):
    """Estrategias de caché disponibles."""
    AGGRESSIVE = "aggressive"      # Cache todo lo posible
    CONSERVATIVE = "conservative"  # Cache solo elementos críticos
    ADAPTIVE = "adaptive"         # Ajusta según patrones de uso
    DISABLED = "disabled"         # Sin cache


class InvalidationStrategy(Enum):
    """Estrategias de invalidación de cache."""
    IMMEDIATE = "immediate"        # Invalidación inmediata
    LAZY = "lazy"                 # Invalidación diferida
    SCHEDULED = "scheduled"       # Invalidación programada
    MANUAL = "manual"            # Invalidación manual


@dataclass(frozen=True)
class ChangeDetectionConfig:
    """Configuración para detección de cambios."""
    enable_git_integration: bool = True
    enable_file_watching: bool = True
    enable_ast_diffing: bool = True
    enable_semantic_diffing: bool = True
    granularity_level: str = "function"
    change_aggregation_window_ms: int = 1000
    ignore_whitespace_changes: bool = True
    ignore_comment_changes: bool = True
    track_dependencies: bool = True
    max_change_set_size: int = 1000
    
    def __post_init__(self):
        """Validar configuración."""
        if self.change_aggregation_window_ms < 0:
            raise ValueError("change_aggregation_window_ms debe ser >= 0")
        
        if self.max_change_set_size <= 0:
            raise ValueError("max_change_set_size debe ser > 0")
        
        valid_granularities = ["file", "function", "class", "statement", "expression", "token"]
        if self.granularity_level not in valid_granularities:
            raise ValueError(f"granularity_level debe ser uno de: {valid_granularities}")


@dataclass(frozen=True)
class CacheConfig:
    """Configuración del sistema de cache."""
    l1_cache_size_mb: int = 512
    l2_cache_size_mb: int = 2048
    l3_cache_size_gb: int = 10
    enable_compression: bool = True
    enable_encryption: bool = False
    enable_predictive_loading: bool = True
    enable_cache_warming: bool = True
    cache_hit_ratio_target: float = 0.85
    eviction_policy: str = "lru"
    ttl_seconds: int = 3600
    
    def __post_init__(self):
        """Validar configuración."""
        if self.l1_cache_size_mb <= 0:
            raise ValueError("l1_cache_size_mb debe ser > 0")
        
        if self.l2_cache_size_mb <= 0:
            raise ValueError("l2_cache_size_mb debe ser > 0")
        
        if self.l3_cache_size_gb <= 0:
            raise ValueError("l3_cache_size_gb debe ser > 0")
        
        if not 0.0 <= self.cache_hit_ratio_target <= 1.0:
            raise ValueError("cache_hit_ratio_target debe estar entre 0.0 y 1.0")
        
        valid_policies = ["lru", "lfu", "fifo", "lifo", "ttl", "size"]
        if self.eviction_policy not in valid_policies:
            raise ValueError(f"eviction_policy debe ser uno de: {valid_policies}")


@dataclass(frozen=True)
class IncrementalConfig:
    """Configuración del análisis incremental."""
    enable_aggressive_caching: bool = True
    enable_predictive_analysis: bool = True
    enable_background_warming: bool = True
    max_incremental_scope_percentage: float = 0.3
    force_full_analysis_threshold: float = 0.5
    enable_delta_compression: bool = True
    parallel_incremental_analysis: bool = True
    incremental_batch_size: int = 10
    min_time_between_analyses_ms: int = 100
    
    def __post_init__(self):
        """Validar configuración."""
        if not 0.0 <= self.max_incremental_scope_percentage <= 1.0:
            raise ValueError("max_incremental_scope_percentage debe estar entre 0.0 y 1.0")
        
        if not 0.0 <= self.force_full_analysis_threshold <= 1.0:
            raise ValueError("force_full_analysis_threshold debe estar entre 0.0 y 1.0")
        
        if self.incremental_batch_size <= 0:
            raise ValueError("incremental_batch_size debe ser > 0")
        
        if self.min_time_between_analyses_ms < 0:
            raise ValueError("min_time_between_analyses_ms debe ser >= 0")


@dataclass(frozen=True)
class PredictiveConfig:
    """Configuración del sistema predictivo."""
    enable_ml_prediction: bool = True
    enable_pattern_based_prediction: bool = True
    enable_time_based_prediction: bool = True
    prediction_horizon_hours: int = 4
    warming_batch_size: int = 50
    max_warming_time_ms: int = 5000
    warming_priority_threshold: float = 0.7
    max_predictions_per_cycle: int = 100
    
    def __post_init__(self):
        """Validar configuración."""
        if self.prediction_horizon_hours <= 0:
            raise ValueError("prediction_horizon_hours debe ser > 0")
        
        if self.warming_batch_size <= 0:
            raise ValueError("warming_batch_size debe ser > 0")
        
        if self.max_warming_time_ms <= 0:
            raise ValueError("max_warming_time_ms debe ser > 0")
        
        if not 0.0 <= self.warming_priority_threshold <= 1.0:
            raise ValueError("warming_priority_threshold debe estar entre 0.0 y 1.0")
        
        if self.max_predictions_per_cycle <= 0:
            raise ValueError("max_predictions_per_cycle debe ser > 0")


@dataclass(frozen=True)
class ChangeMetrics:
    """Métricas de cambios detectados."""
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    functions_added: int = 0
    functions_modified: int = 0
    functions_deleted: int = 0
    lines_added: int = 0
    lines_modified: int = 0
    lines_deleted: int = 0
    complexity_delta: float = 0.0
    
    def __post_init__(self):
        """Validar métricas."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, int) and field_value < 0:
                raise ValueError(f"{field_name} no puede ser negativo")


@dataclass(frozen=True)
class CacheMetrics:
    """Métricas del sistema de cache."""
    total_entries: int = 0
    total_size_bytes: int = 0
    l1_entries: int = 0
    l1_size_bytes: int = 0
    l2_entries: int = 0
    l2_size_bytes: int = 0
    l3_entries: int = 0
    l3_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    
    def __post_init__(self):
        """Validar métricas."""
        for field_name, field_value in self.__dict__.items():
            if field_value < 0:
                raise ValueError(f"{field_name} no puede ser negativo")
    
    @property
    def hit_ratio(self) -> float:
        """Calcular ratio de aciertos."""
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return self.hit_count / total


@dataclass(frozen=True)
class IncrementalMetrics:
    """Métricas del análisis incremental."""
    files_analyzed: int = 0
    files_skipped: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    time_saved_ms: int = 0
    analysis_time_ms: int = 0
    delta_computations: int = 0
    full_computations: int = 0
    
    def __post_init__(self):
        """Validar métricas."""
        for field_name, field_value in self.__dict__.items():
            if field_value < 0:
                raise ValueError(f"{field_name} no puede ser negativo")
    
    @property
    def speedup_factor(self) -> float:
        """Calcular factor de aceleración."""
        if self.analysis_time_ms == 0:
            return 0.0
        baseline_time = self.analysis_time_ms + self.time_saved_ms
        return baseline_time / self.analysis_time_ms if self.analysis_time_ms > 0 else 1.0


@dataclass(frozen=True)
class DependencyMetrics:
    """Métricas de análisis de dependencias."""
    direct_dependencies: int = 0
    transitive_dependencies: int = 0
    circular_dependencies: int = 0
    dependency_depth: int = 0
    affected_files: int = 0
    impact_radius: int = 0
    
    def __post_init__(self):
        """Validar métricas."""
        for field_name, field_value in self.__dict__.items():
            if field_value < 0:
                raise ValueError(f"{field_name} no puede ser negativo")


@dataclass(frozen=True)
class PerformanceThresholds:
    """Umbrales de rendimiento del sistema."""
    max_change_detection_time_ms: int = 5000
    max_cache_lookup_time_ms: int = 10
    max_cache_warming_time_ms: int = 1000
    max_invalidation_time_ms: int = 100
    max_incremental_analysis_time_ms: int = 2000
    min_cache_hit_ratio: float = 0.8
    max_memory_usage_mb: int = 1024
    
    def __post_init__(self):
        """Validar umbrales."""
        for field_name, field_value in self.__dict__.items():
            if field_name.endswith("_ms") or field_name.endswith("_mb"):
                if field_value <= 0:
                    raise ValueError(f"{field_name} debe ser > 0")
        
        if not 0.0 <= self.min_cache_hit_ratio <= 1.0:
            raise ValueError("min_cache_hit_ratio debe estar entre 0.0 y 1.0")


@dataclass(frozen=True)
class ResourceUsage:
    """Uso de recursos del sistema."""
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    network_usage_kb: float = 0.0
    cache_memory_mb: float = 0.0
    
    def __post_init__(self):
        """Validar uso de recursos."""
        if not 0.0 <= self.cpu_usage_percent <= 100.0:
            raise ValueError("cpu_usage_percent debe estar entre 0.0 y 100.0")
        
        for field_name, field_value in self.__dict__.items():
            if field_name != "cpu_usage_percent" and field_value < 0:
                raise ValueError(f"{field_name} no puede ser negativo")


@dataclass(frozen=True)
class CostEstimation:
    """Estimación de costos computacionales."""
    parse_cost_ms: float = 0.0
    analysis_cost_ms: float = 0.0
    embedding_cost_ms: float = 0.0
    storage_cost_bytes: int = 0
    network_cost_bytes: int = 0
    total_cost_ms: float = 0.0
    
    def __post_init__(self):
        """Validar costos."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, (int, float)) and field_value < 0:
                raise ValueError(f"{field_name} no puede ser negativo")


@dataclass(frozen=True)
class QualityMetrics:
    """Métricas de calidad del análisis incremental."""
    analysis_accuracy: float = 1.0
    cache_consistency: float = 1.0
    delta_precision: float = 1.0
    dependency_accuracy: float = 1.0
    overall_quality: float = 1.0
    
    def __post_init__(self):
        """Validar métricas de calidad."""
        for field_name, field_value in self.__dict__.items():
            if not 0.0 <= field_value <= 1.0:
                raise ValueError(f"{field_name} debe estar entre 0.0 y 1.0")
