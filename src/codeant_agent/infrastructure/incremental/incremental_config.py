"""
Configuración para el Sistema de Análisis Incremental.

Este módulo define todas las configuraciones necesarias para el
funcionamiento del sistema de análisis incremental.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from ...domain.entities.incremental import CacheLevel


@dataclass
class IncrementalConfig:
    """Configuración principal del sistema incremental."""
    
    # Configuración de detección de cambios
    enable_granular_detection: bool = True
    default_granularity_level: str = "FUNCTION"  # FILE, FUNCTION, STATEMENT, EXPRESSION, TOKEN
    detect_semantic_changes: bool = True
    aggregate_changes_window_ms: int = 1000
    calculate_impact_score: bool = True
    max_changes_for_aggregation: int = 100
    
    # Configuración de cache multi-nivel
    l1_cache_size: int = 1000  # Número de items en memoria
    l1_item_max_size: int = 1024 * 1024  # 1MB
    l2_item_max_size: int = 10 * 1024 * 1024  # 10MB
    l3_cache_directory: str = ".cache/incremental"
    default_cache_ttl: int = 3600  # 1 hora en segundos
    min_cache_coverage_for_incremental: float = 0.5
    hot_item_threshold: int = 5  # Accesos en 5 minutos
    cold_item_threshold: int = 1800  # 30 minutos sin acceso
    
    # Configuración de análisis incremental
    incremental_threshold: float = 0.3  # Ratio máximo de cambios
    max_changes_for_incremental: int = 100
    min_project_size_for_incremental: int = 10
    max_parallel_analyses: int = 4
    analyze_transitive_dependencies: bool = True
    max_dependency_depth: int = 3
    reuse_unchanged_components: bool = True
    
    # Configuración de cache predictivo
    enable_predictive_caching: bool = True
    min_confidence_for_warming: float = 0.7
    default_prediction_window: int = 30  # minutos
    max_predictions_per_cycle: int = 50
    target_prediction_accuracy: float = 0.75
    min_training_samples: int = 100
    model_storage_path: str = ".cache/models"
    default_warming_cache_level: CacheLevel = CacheLevel.L1
    min_acceptable_accuracy: float = 0.5
    
    # Configuración de invalidación
    enable_cascade_invalidation: bool = True
    lazy_invalidation_delay_ms: int = 5000
    invalidation_batch_size: int = 100
    track_invalidation_reasons: bool = True
    
    # Configuración de dependencias
    dependency_cache_ttl: int = 7200  # 2 horas
    file_patterns_to_analyze: List[str] = field(default_factory=lambda: [
        "*.py", "*.js", "*.jsx", "*.ts", "*.tsx",
        "*.java", "*.cpp", "*.c", "*.h", "*.hpp"
    ])
    max_circular_dependencies_to_report: int = 10
    analyze_external_dependencies: bool = False
    
    # Configuración de métricas y monitoreo
    enable_metrics_collection: bool = True
    metrics_flush_interval: int = 60  # segundos
    max_metric_history: int = 10000
    performance_alert_threshold_ms: int = 5000
    
    # Configuración de optimización
    enable_auto_optimization: bool = True
    optimization_interval: int = 3600  # 1 hora
    cache_size_alert_threshold_mb: float = 1000.0
    memory_pressure_threshold: float = 0.8
    
    # Configuración de persistencia
    enable_state_persistence: bool = True
    state_save_interval: int = 300  # 5 minutos
    max_state_file_size_mb: float = 100.0
    compress_state_files: bool = True
    
    # Configuración de desarrollo/debug
    debug_mode: bool = False
    log_delta_operations: bool = False
    trace_cache_operations: bool = False
    profile_performance: bool = False
    
    def validate(self) -> List[str]:
        """
        Validar la configuración.
        
        Returns:
            Lista de errores de validación
        """
        errors = []
        
        # Validar tamaños de cache
        if self.l1_cache_size <= 0:
            errors.append("l1_cache_size must be positive")
        
        if self.l1_item_max_size >= self.l2_item_max_size:
            errors.append("l1_item_max_size should be less than l2_item_max_size")
        
        # Validar umbrales
        if not 0 <= self.incremental_threshold <= 1:
            errors.append("incremental_threshold must be between 0 and 1")
        
        if not 0 <= self.min_confidence_for_warming <= 1:
            errors.append("min_confidence_for_warming must be between 0 and 1")
        
        # Validar rutas
        cache_dir = Path(self.l3_cache_directory)
        if cache_dir.exists() and not cache_dir.is_dir():
            errors.append(f"{self.l3_cache_directory} exists but is not a directory")
        
        # Validar concurrencia
        if self.max_parallel_analyses < 1:
            errors.append("max_parallel_analyses must be at least 1")
        
        return errors
    
    def adjust_for_environment(self, available_memory_mb: float, cpu_count: int):
        """
        Ajustar configuración según recursos disponibles.
        
        Args:
            available_memory_mb: Memoria disponible en MB
            cpu_count: Número de CPUs disponibles
        """
        # Ajustar tamaño de cache L1 según memoria
        if available_memory_mb < 1000:
            self.l1_cache_size = min(self.l1_cache_size, 100)
            self.l1_item_max_size = 512 * 1024  # 512KB
        elif available_memory_mb < 2000:
            self.l1_cache_size = min(self.l1_cache_size, 500)
        
        # Ajustar paralelismo según CPUs
        self.max_parallel_analyses = min(
            self.max_parallel_analyses,
            max(1, cpu_count - 1)  # Dejar un CPU libre
        )
        
        # Ajustar predicción según recursos
        if available_memory_mb < 2000 or cpu_count < 4:
            self.enable_predictive_caching = False
        
        # Ajustar métricas según recursos
        if available_memory_mb < 1000:
            self.max_metric_history = 1000
            self.enable_auto_optimization = False
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'IncrementalConfig':
        """
        Crear configuración desde diccionario.
        
        Args:
            config_dict: Diccionario con valores de configuración
            
        Returns:
            Instancia de IncrementalConfig
        """
        # Convertir cache level string a enum si es necesario
        if 'default_warming_cache_level' in config_dict:
            level_str = config_dict['default_warming_cache_level']
            if isinstance(level_str, str):
                config_dict['default_warming_cache_level'] = CacheLevel[level_str]
        
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """
        Convertir configuración a diccionario.
        
        Returns:
            Diccionario con valores de configuración
        """
        config_dict = {}
        
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, CacheLevel):
                config_dict[field_name] = field_value.name
            elif isinstance(field_value, Path):
                config_dict[field_name] = str(field_value)
            else:
                config_dict[field_name] = field_value
        
        return config_dict


@dataclass
class CacheConfig:
    """Configuración específica para el sistema de cache."""
    
    # Niveles de cache
    l1_enabled: bool = True
    l2_enabled: bool = True
    l3_enabled: bool = True
    
    # Políticas de evicción
    l1_eviction_policy: str = "LRU"  # LRU, LFU, FIFO
    l2_eviction_policy: str = "LRU"
    l3_eviction_policy: str = "LRU"
    
    # Tamaños máximos
    l1_max_memory_mb: float = 100.0
    l2_max_memory_mb: float = 1000.0
    l3_max_disk_gb: float = 10.0
    
    # TTLs por nivel
    l1_default_ttl: int = 3600      # 1 hora
    l2_default_ttl: int = 86400     # 24 horas
    l3_default_ttl: int = 604800    # 7 días
    
    # Políticas de promoción
    promote_on_hit: bool = True
    promotion_threshold: int = 3  # Hits antes de promover
    
    # Compresión
    compress_l2_items: bool = False
    compress_l3_items: bool = True
    compression_threshold_bytes: int = 1024  # 1KB
    
    # Redis configuration (L2)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_connection_pool_size: int = 10


@dataclass 
class ChangeDetectionConfig:
    """Configuración específica para detección de cambios."""
    
    # Git configuration
    use_git_for_detection: bool = True
    git_diff_context_lines: int = 3
    ignore_whitespace_changes: bool = True
    
    # AST diffing
    ast_diff_algorithm: str = "tree_edit_distance"  # tree_edit_distance, hash_based
    max_ast_size_for_detailed_diff: int = 10000  # nodos
    
    # Granularidad
    min_change_size_tokens: int = 5
    group_related_changes: bool = True
    change_grouping_distance: int = 10  # líneas
    
    # Filtros
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "__pycache__", ".git", "node_modules", ".venv",
        "*.pyc", "*.pyo", "*.pyd", ".DS_Store"
    ])
    
    # Umbrales
    major_change_threshold: float = 0.3  # 30% del archivo
    refactoring_detection_threshold: float = 0.5  # similitud


@dataclass
class PredictiveConfig:
    """Configuración específica para cache predictivo."""
    
    # Modelos
    prediction_model: str = "linear_regression"  # linear_regression, random_forest, lstm
    feature_window_hours: int = 168  # 1 semana
    
    # Features
    use_time_features: bool = True
    use_sequence_features: bool = True
    use_correlation_features: bool = True
    use_user_features: bool = False
    
    # Training
    retrain_interval_hours: int = 24
    min_samples_for_training: int = 100
    train_test_split: float = 0.8
    
    # Predicción
    prediction_horizon_minutes: int = 30
    confidence_calculation: str = "model_probability"  # model_probability, heuristic
    
    # Warming
    max_concurrent_warming: int = 5
    warming_priority_queue_size: int = 100
    warming_timeout_seconds: int = 30


@dataclass
class DependencyConfig:
    """Configuración específica para análisis de dependencias."""
    
    # Resolución de imports
    python_path_additions: List[str] = field(default_factory=list)
    node_modules_paths: List[str] = field(default_factory=lambda: ["node_modules"])
    
    # Análisis
    analyze_stdlib_deps: bool = False
    analyze_third_party_deps: bool = True
    analyze_dynamic_imports: bool = False
    
    # Grafo
    max_graph_size: int = 10000  # nodos
    persist_graph: bool = True
    graph_storage_format: str = "networkx"  # networkx, graphml, json
    
    # Optimizaciones
    cache_import_resolutions: bool = True
    parallel_dependency_analysis: bool = True
    batch_size: int = 50


def load_config(config_path: Optional[Path] = None) -> IncrementalConfig:
    """
    Cargar configuración desde archivo o usar valores por defecto.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Configuración cargada
    """
    if config_path and config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return IncrementalConfig.from_dict(config_dict)
    
    return IncrementalConfig()


def save_config(config: IncrementalConfig, config_path: Path):
    """
    Guardar configuración a archivo.
    
    Args:
        config: Configuración a guardar
        config_path: Ruta donde guardar
    """
    import json
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

