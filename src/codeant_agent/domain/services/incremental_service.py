"""
Servicios del dominio para el Sistema de Análisis Incremental.

Este módulo define los servicios de dominio que encapsulan la lógica de negocio
del sistema de análisis incremental, siguiendo los principios de DDD.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Set, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from ..entities.incremental import (
    ChangeSet, GranularChange, DependencyImpact, AnalysisScope,
    CacheKey, CacheEntry, CacheLevel, InvalidationRequest,
    AccessPrediction, ReusableComponents, FileChange,
    ChangeType, ChangeLocation, GranularityLevel
)
from ..value_objects.incremental_metrics import (
    ChangeDetectionConfig, CacheConfig, IncrementalConfig,
    PredictiveConfig, CacheMetrics, IncrementalMetrics
)


class ChangeDetectionService(ABC):
    """Servicio para detección de cambios en el código."""
    
    @abstractmethod
    async def detect_changes(
        self, 
        repository_path: Path,
        from_commit: Optional[str] = None,
        to_commit: Optional[str] = None
    ) -> ChangeSet:
        """Detectar cambios entre commits o estado actual."""
        pass
    
    @abstractmethod
    async def detect_file_changes(
        self, 
        file_path: Path,
        old_content: Optional[str] = None,
        new_content: Optional[str] = None
    ) -> List[FileChange]:
        """Detectar cambios en un archivo específico."""
        pass
    
    @abstractmethod
    async def detect_granular_changes(
        self, 
        file_change: FileChange,
        granularity: GranularityLevel
    ) -> List[GranularChange]:
        """Detectar cambios granulares dentro de un archivo."""
        pass
    
    @abstractmethod
    async def calculate_change_impact(
        self, 
        changes: List[GranularChange]
    ) -> float:
        """Calcular el impacto de un conjunto de cambios."""
        pass
    
    @abstractmethod
    async def aggregate_changes(
        self, 
        changes: List[GranularChange],
        window_ms: int
    ) -> List[GranularChange]:
        """Agregar cambios dentro de una ventana temporal."""
        pass


class DependencyAnalysisService(ABC):
    """Servicio para análisis de dependencias."""
    
    @abstractmethod
    async def analyze_dependencies(
        self, 
        file_path: Path
    ) -> Set[Path]:
        """Analizar dependencias directas de un archivo."""
        pass
    
    @abstractmethod
    async def analyze_dependency_impact(
        self, 
        changes: List[GranularChange]
    ) -> DependencyImpact:
        """Analizar impacto de cambios en las dependencias."""
        pass
    
    @abstractmethod
    async def calculate_transitive_impact(
        self, 
        file_path: Path,
        max_depth: Optional[int] = None
    ) -> Set[Path]:
        """Calcular impacto transitivo de cambios."""
        pass
    
    @abstractmethod
    async def find_circular_dependencies(
        self, 
        project_path: Path
    ) -> List[List[Path]]:
        """Encontrar dependencias circulares en el proyecto."""
        pass
    
    @abstractmethod
    async def build_dependency_graph(
        self, 
        project_path: Path
    ) -> Dict[Path, Set[Path]]:
        """Construir grafo de dependencias del proyecto."""
        pass


class CacheManagementService(ABC):
    """Servicio para gestión del cache multi-nivel."""
    
    @abstractmethod
    async def get_cached_item(
        self, 
        key: CacheKey
    ) -> Optional[Any]:
        """Obtener item del cache."""
        pass
    
    @abstractmethod
    async def cache_item(
        self, 
        key: CacheKey,
        item: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Cachear un item."""
        pass
    
    @abstractmethod
    async def invalidate_cache(
        self, 
        request: InvalidationRequest
    ) -> int:
        """Invalidar entradas del cache."""
        pass
    
    @abstractmethod
    async def promote_cache_entry(
        self, 
        key: CacheKey,
        to_level: CacheLevel
    ) -> bool:
        """Promover entrada a un nivel superior de cache."""
        pass
    
    @abstractmethod
    async def optimize_cache_distribution(self) -> Dict[str, Any]:
        """Optimizar distribución de datos entre niveles."""
        pass
    
    @abstractmethod
    async def get_cache_metrics(self) -> CacheMetrics:
        """Obtener métricas del sistema de cache."""
        pass


class IncrementalAnalysisService(ABC):
    """Servicio para análisis incremental."""
    
    @abstractmethod
    async def analyze_incremental(
        self, 
        change_set: ChangeSet,
        project_context: Dict[str, Any]
    ) -> Dict[Path, Any]:
        """Realizar análisis incremental basado en cambios."""
        pass
    
    @abstractmethod
    async def determine_analysis_scope(
        self, 
        change_set: ChangeSet
    ) -> AnalysisScope:
        """Determinar alcance del análisis necesario."""
        pass
    
    @abstractmethod
    async def identify_reusable_components(
        self, 
        file_path: Path,
        change_set: ChangeSet
    ) -> ReusableComponents:
        """Identificar componentes reutilizables del cache."""
        pass
    
    @abstractmethod
    async def perform_delta_analysis(
        self, 
        cached_result: Any,
        changes: List[GranularChange]
    ) -> Any:
        """Realizar análisis delta sobre resultado cacheado."""
        pass
    
    @abstractmethod
    async def should_use_incremental(
        self, 
        change_set: ChangeSet,
        project_size: int
    ) -> bool:
        """Determinar si usar análisis incremental o completo."""
        pass


class PredictiveCacheService(ABC):
    """Servicio para cache predictivo."""
    
    @abstractmethod
    async def predict_future_accesses(
        self, 
        time_horizon: timedelta
    ) -> List[AccessPrediction]:
        """Predecir accesos futuros al cache."""
        pass
    
    @abstractmethod
    async def warm_cache(
        self, 
        predictions: List[AccessPrediction]
    ) -> int:
        """Calentar cache basado en predicciones."""
        pass
    
    @abstractmethod
    async def analyze_access_patterns(
        self, 
        time_window: timedelta
    ) -> Dict[str, Any]:
        """Analizar patrones de acceso al cache."""
        pass
    
    @abstractmethod
    async def optimize_warming_strategy(self) -> Dict[str, Any]:
        """Optimizar estrategia de cache warming."""
        pass
    
    @abstractmethod
    async def evaluate_prediction_accuracy(
        self, 
        time_window: timedelta
    ) -> float:
        """Evaluar precisión de predicciones."""
        pass


class InvalidationService(ABC):
    """Servicio para invalidación de cache."""
    
    @abstractmethod
    async def create_invalidation_plan(
        self, 
        changes: List[GranularChange]
    ) -> List[InvalidationRequest]:
        """Crear plan de invalidación basado en cambios."""
        pass
    
    @abstractmethod
    async def execute_invalidation(
        self, 
        request: InvalidationRequest
    ) -> int:
        """Ejecutar invalidación de cache."""
        pass
    
    @abstractmethod
    async def invalidate_dependencies(
        self, 
        file_path: Path
    ) -> int:
        """Invalidar cache de dependencias."""
        pass
    
    @abstractmethod
    async def schedule_lazy_invalidation(
        self, 
        request: InvalidationRequest,
        delay: timedelta
    ) -> None:
        """Programar invalidación diferida."""
        pass
    
    @abstractmethod
    async def calculate_invalidation_impact(
        self, 
        request: InvalidationRequest
    ) -> Dict[str, Any]:
        """Calcular impacto de una invalidación."""
        pass


class DeltaProcessingService(ABC):
    """Servicio para procesamiento de deltas."""
    
    @abstractmethod
    async def compute_ast_delta(
        self, 
        old_ast: Any,
        new_ast: Any
    ) -> Dict[str, Any]:
        """Computar delta entre ASTs."""
        pass
    
    @abstractmethod
    async def apply_delta_to_analysis(
        self, 
        base_analysis: Any,
        delta: Dict[str, Any]
    ) -> Any:
        """Aplicar delta a un análisis base."""
        pass
    
    @abstractmethod
    async def compress_delta(
        self, 
        delta: Dict[str, Any]
    ) -> bytes:
        """Comprimir delta para almacenamiento eficiente."""
        pass
    
    @abstractmethod
    async def merge_deltas(
        self, 
        deltas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fusionar múltiples deltas."""
        pass
    
    @abstractmethod
    async def validate_delta_consistency(
        self, 
        base: Any,
        delta: Dict[str, Any],
        expected: Any
    ) -> bool:
        """Validar consistencia de aplicación de delta."""
        pass


class ResourceMonitoringService(ABC):
    """Servicio para monitoreo de recursos."""
    
    @abstractmethod
    async def get_current_resource_usage(self) -> Dict[str, float]:
        """Obtener uso actual de recursos."""
        pass
    
    @abstractmethod
    async def has_sufficient_resources(
        self, 
        operation: str
    ) -> bool:
        """Verificar si hay recursos suficientes para operación."""
        pass
    
    @abstractmethod
    async def predict_resource_usage(
        self, 
        operation: str,
        scope: AnalysisScope
    ) -> Dict[str, float]:
        """Predecir uso de recursos para operación."""
        pass
    
    @abstractmethod
    async def optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimizar asignación de recursos."""
        pass
    
    @abstractmethod
    async def get_resource_alerts(self) -> List[Dict[str, Any]]:
        """Obtener alertas de recursos."""
        pass


class PerformanceOptimizationService(ABC):
    """Servicio para optimización de rendimiento."""
    
    @abstractmethod
    async def analyze_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analizar cuellos de botella de rendimiento."""
        pass
    
    @abstractmethod
    async def optimize_analysis_pipeline(
        self, 
        change_set: ChangeSet
    ) -> Dict[str, Any]:
        """Optimizar pipeline de análisis."""
        pass
    
    @abstractmethod
    async def parallelize_analysis(
        self, 
        scope: AnalysisScope,
        max_workers: int
    ) -> Dict[str, Any]:
        """Paralelizar análisis para mejor rendimiento."""
        pass
    
    @abstractmethod
    async def profile_operation(
        self, 
        operation: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perfilar una operación específica."""
        pass
    
    @abstractmethod
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Obtener recomendaciones de optimización."""
        pass
