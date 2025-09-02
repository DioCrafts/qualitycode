"""
Sistema de Análisis Incremental y Caching.

Este paquete implementa el sistema completo de análisis incremental con:
- Detección granular de cambios
- Cache multi-nivel inteligente
- Análisis incremental y delta processing
- Cache predictivo con ML
- Tracking de dependencias
- Sistema de métricas y optimización
"""

from .incremental_engine import IncrementalAnalysisEngine
from .change_detector import GranularChangeDetector
from .cache_manager import SmartCacheManager
from .dependency_tracker import DependencyTracker
from .predictive_cache import PredictiveCacheSystem
from .delta_processor import DeltaProcessor
from .incremental_config import (
    IncrementalConfig,
    CacheConfig,
    ChangeDetectionConfig,
    PredictiveConfig,
    DependencyConfig,
    load_config,
    save_config
)

# Versión del sistema incremental
__version__ = "1.0.0"

# Exportar clases principales
__all__ = [
    # Motor principal
    "IncrementalAnalysisEngine",
    
    # Componentes core
    "GranularChangeDetector",
    "SmartCacheManager", 
    "DependencyTracker",
    "PredictiveCacheSystem",
    "DeltaProcessor",
    
    # Configuración
    "IncrementalConfig",
    "CacheConfig",
    "ChangeDetectionConfig", 
    "PredictiveConfig",
    "DependencyConfig",
    "load_config",
    "save_config",
    
    # Factory functions
    "create_incremental_system",
    "create_default_config"
]


def create_incremental_system(
    config: IncrementalConfig = None,
    metrics_collector = None
) -> dict:
    """
    Factory function para crear el sistema incremental completo.
    
    Args:
        config: Configuración del sistema (usa defaults si es None)
        metrics_collector: Collector de métricas
        
    Returns:
        Diccionario con todos los componentes inicializados
    """
    if config is None:
        config = create_default_config()
    
    # Validar configuración
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {', '.join(errors)}")
    
    # Crear componentes
    components = {}
    
    # Detector de cambios
    components['change_detector'] = GranularChangeDetector(
        config=config,
        metrics_collector=metrics_collector
    )
    
    # Cache manager
    components['cache_manager'] = SmartCacheManager(
        config=config,
        metrics_collector=metrics_collector
    )
    
    # Dependency tracker
    components['dependency_tracker'] = DependencyTracker(
        config=config,
        metrics_collector=metrics_collector
    )
    
    # Delta processor
    components['delta_processor'] = DeltaProcessor(
        config=config,
        analysis_engine=None,  # Se debe inyectar después
        metrics_collector=metrics_collector
    )
    
    # Predictive cache
    components['predictive_cache'] = PredictiveCacheSystem(
        config=config,
        cache_storage=components['cache_manager'],
        analysis_engine=None,  # Se debe inyectar después
        metrics_collector=metrics_collector
    )
    
    # Motor principal
    components['engine'] = IncrementalAnalysisEngine(
        config=config,
        change_detection_service=components['change_detector'],
        dependency_analysis_service=components['dependency_tracker'],
        cache_repository=components['cache_manager'],
        analysis_result_repository=None,  # Se debe inyectar después
        analysis_engine_output=None,  # Se debe inyectar después
        metrics_collector=metrics_collector
    )
    
    return components


def create_default_config() -> IncrementalConfig:
    """
    Crear configuración por defecto optimizada.
    
    Returns:
        Configuración con valores por defecto
    """
    import os
    import psutil
    
    config = IncrementalConfig()
    
    # Ajustar según recursos disponibles
    try:
        # Obtener memoria disponible
        memory = psutil.virtual_memory()
        available_memory_mb = memory.available / (1024 * 1024)
        
        # Obtener número de CPUs
        cpu_count = os.cpu_count() or 1
        
        # Ajustar configuración
        config.adjust_for_environment(available_memory_mb, cpu_count)
        
    except Exception:
        # Si psutil no está disponible, usar valores conservadores
        pass
    
    return config


# Constantes útiles
DEFAULT_CACHE_TTL = 3600  # 1 hora
DEFAULT_PREDICTION_WINDOW = 30  # minutos
DEFAULT_DEPENDENCY_DEPTH = 3
MAX_INCREMENTAL_CHANGES = 100

