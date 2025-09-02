"""
Puertos de entrada y salida para el Sistema de Análisis Incremental.

Este módulo define las interfaces que permiten la comunicación entre
la capa de aplicación y las capas externas (presentación e infraestructura).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Set, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from ...domain.entities.incremental import (
    ChangeSet, GranularChange, DependencyImpact, AnalysisScope,
    CacheKey, CacheEntry, CacheLevel, InvalidationRequest,
    AccessPrediction, ReusableComponents, FileChange,
    IncrementalAnalysisResult, DeltaAnalysisResult,
    CachePerformance, ChangeType, PredictionSource
)
from ...domain.value_objects.incremental_metrics import (
    IncrementalMetrics, CacheMetrics, ChangeDetectionMetrics,
    AnalysisPerformanceMetrics, WarmingMetrics
)


# Puertos de Entrada (Input Ports)

class ChangeDetectionInputPort(ABC):
    """Puerto de entrada para detección de cambios."""
    
    @abstractmethod
    async def detect_repository_changes(
        self,
        repository_path: Path,
        from_commit: Optional[str] = None,
        to_commit: Optional[str] = None,
        granularity_level: str = "function"
    ) -> ChangeSet:
        """Detectar cambios en un repositorio."""
        pass
    
    @abstractmethod
    async def detect_file_changes(
        self,
        file_path: Path,
        content_before: Optional[str] = None,
        content_after: Optional[str] = None
    ) -> List[GranularChange]:
        """Detectar cambios granulares en un archivo."""
        pass
    
    @abstractmethod
    async def analyze_change_impact(
        self,
        changes: List[GranularChange]
    ) -> DependencyImpact:
        """Analizar el impacto de los cambios."""
        pass


class CacheManagementInputPort(ABC):
    """Puerto de entrada para gestión de cache."""
    
    @abstractmethod
    async def get_cached_analysis(
        self,
        file_path: Path,
        analysis_type: str
    ) -> Optional[Any]:
        """Obtener análisis cacheado."""
        pass
    
    @abstractmethod
    async def cache_analysis_result(
        self,
        file_path: Path,
        analysis_type: str,
        result: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Cachear resultado de análisis."""
        pass
    
    @abstractmethod
    async def invalidate_cache(
        self,
        invalidation_pattern: str,
        scope: Optional[str] = None
    ) -> int:
        """Invalidar entradas del cache."""
        pass
    
    @abstractmethod
    async def get_cache_statistics(self) -> CacheMetrics:
        """Obtener estadísticas del cache."""
        pass


class IncrementalAnalysisInputPort(ABC):
    """Puerto de entrada para análisis incremental."""
    
    @abstractmethod
    async def analyze_changes_incrementally(
        self,
        change_set: ChangeSet,
        analysis_types: List[str],
        use_cache: bool = True
    ) -> IncrementalAnalysisResult:
        """Analizar cambios de forma incremental."""
        pass
    
    @abstractmethod
    async def get_analysis_scope(
        self,
        changes: List[GranularChange]
    ) -> AnalysisScope:
        """Determinar el alcance del análisis."""
        pass
    
    @abstractmethod
    async def identify_reusable_results(
        self,
        file_path: Path,
        changes: List[GranularChange]
    ) -> ReusableComponents:
        """Identificar resultados reutilizables."""
        pass


class PredictiveCacheInputPort(ABC):
    """Puerto de entrada para cache predictivo."""
    
    @abstractmethod
    async def predict_cache_needs(
        self,
        time_horizon: int,
        confidence_threshold: float = 0.7
    ) -> List[AccessPrediction]:
        """Predecir necesidades futuras de cache."""
        pass
    
    @abstractmethod
    async def warm_cache_predictively(
        self,
        predictions: List[AccessPrediction],
        max_items: Optional[int] = None
    ) -> int:
        """Calentar cache de forma predictiva."""
        pass
    
    @abstractmethod
    async def get_warming_performance(self) -> WarmingMetrics:
        """Obtener métricas de cache warming."""
        pass


class DependencyTrackingInputPort(ABC):
    """Puerto de entrada para seguimiento de dependencias."""
    
    @abstractmethod
    async def get_file_dependencies(
        self,
        file_path: Path,
        include_transitive: bool = False
    ) -> Set[Path]:
        """Obtener dependencias de un archivo."""
        pass
    
    @abstractmethod
    async def get_dependent_files(
        self,
        file_path: Path,
        max_depth: Optional[int] = None
    ) -> Set[Path]:
        """Obtener archivos que dependen de uno dado."""
        pass
    
    @abstractmethod
    async def update_dependency_graph(
        self,
        file_path: Path,
        new_dependencies: Set[Path]
    ) -> bool:
        """Actualizar grafo de dependencias."""
        pass


# Puertos de Salida (Output Ports)

class ChangeDetectionOutputPort(ABC):
    """Puerto de salida para detección de cambios."""
    
    @abstractmethod
    async def get_git_changes(
        self,
        repository_path: Path,
        from_commit: Optional[str],
        to_commit: Optional[str]
    ) -> List[FileChange]:
        """Obtener cambios desde Git."""
        pass
    
    @abstractmethod
    async def parse_file_ast(
        self,
        file_path: Path,
        content: str
    ) -> Any:
        """Parsear AST de un archivo."""
        pass
    
    @abstractmethod
    async def compute_ast_diff(
        self,
        old_ast: Any,
        new_ast: Any
    ) -> List[Dict[str, Any]]:
        """Computar diferencias entre ASTs."""
        pass


class CacheStorageOutputPort(ABC):
    """Puerto de salida para almacenamiento en cache."""
    
    @abstractmethod
    async def store_in_l1(
        self,
        key: str,
        value: Any,
        ttl: int
    ) -> bool:
        """Almacenar en cache L1 (memoria)."""
        pass
    
    @abstractmethod
    async def store_in_l2(
        self,
        key: str,
        value: Any,
        ttl: int
    ) -> bool:
        """Almacenar en cache L2 (Redis)."""
        pass
    
    @abstractmethod
    async def store_in_l3(
        self,
        key: str,
        value: Any,
        ttl: int
    ) -> bool:
        """Almacenar en cache L3 (disco)."""
        pass
    
    @abstractmethod
    async def retrieve_from_cache(
        self,
        key: str,
        level: CacheLevel
    ) -> Optional[Any]:
        """Recuperar de cache específico."""
        pass
    
    @abstractmethod
    async def invalidate_by_pattern(
        self,
        pattern: str,
        levels: List[CacheLevel]
    ) -> int:
        """Invalidar por patrón en niveles específicos."""
        pass


class DependencyGraphOutputPort(ABC):
    """Puerto de salida para grafo de dependencias."""
    
    @abstractmethod
    async def save_dependency_graph(
        self,
        graph: Dict[Path, Set[Path]]
    ) -> bool:
        """Guardar grafo de dependencias."""
        pass
    
    @abstractmethod
    async def load_dependency_graph(self) -> Dict[Path, Set[Path]]:
        """Cargar grafo de dependencias."""
        pass
    
    @abstractmethod
    async def query_dependencies(
        self,
        file_path: Path,
        direction: str = "both"
    ) -> Dict[str, Set[Path]]:
        """Consultar dependencias."""
        pass


class AnalysisEngineOutputPort(ABC):
    """Puerto de salida para motor de análisis."""
    
    @abstractmethod
    async def run_partial_analysis(
        self,
        file_path: Path,
        analysis_type: str,
        scope: AnalysisScope
    ) -> Any:
        """Ejecutar análisis parcial."""
        pass
    
    @abstractmethod
    async def merge_analysis_results(
        self,
        base_result: Any,
        delta_result: Any,
        analysis_type: str
    ) -> Any:
        """Fusionar resultados de análisis."""
        pass
    
    @abstractmethod
    async def validate_analysis_result(
        self,
        result: Any,
        analysis_type: str
    ) -> bool:
        """Validar resultado de análisis."""
        pass


class MetricsCollectorOutputPort(ABC):
    """Puerto de salida para recolección de métricas."""
    
    @abstractmethod
    async def record_cache_hit(
        self,
        cache_level: CacheLevel,
        key: str
    ) -> None:
        """Registrar cache hit."""
        pass
    
    @abstractmethod
    async def record_cache_miss(
        self,
        key: str
    ) -> None:
        """Registrar cache miss."""
        pass
    
    @abstractmethod
    async def record_analysis_time(
        self,
        operation: str,
        duration_ms: int,
        incremental: bool
    ) -> None:
        """Registrar tiempo de análisis."""
        pass
    
    @abstractmethod
    async def record_invalidation_event(
        self,
        pattern: str,
        affected_count: int
    ) -> None:
        """Registrar evento de invalidación."""
        pass
    
    @abstractmethod
    async def get_metrics_summary(
        self,
        time_window: timedelta
    ) -> IncrementalMetrics:
        """Obtener resumen de métricas."""
        pass


class PredictionEngineOutputPort(ABC):
    """Puerto de salida para motor de predicción."""
    
    @abstractmethod
    async def get_historical_access_patterns(
        self,
        time_window: timedelta
    ) -> List[Dict[str, Any]]:
        """Obtener patrones históricos de acceso."""
        pass
    
    @abstractmethod
    async def train_prediction_model(
        self,
        training_data: List[Dict[str, Any]]
    ) -> Any:
        """Entrenar modelo de predicción."""
        pass
    
    @abstractmethod
    async def generate_predictions(
        self,
        model: Any,
        context: Dict[str, Any]
    ) -> List[AccessPrediction]:
        """Generar predicciones de acceso."""
        pass
    
    @abstractmethod
    async def evaluate_prediction_accuracy(
        self,
        predictions: List[AccessPrediction],
        actual_accesses: List[str]
    ) -> float:
        """Evaluar precisión de predicciones."""
        pass


class NotificationOutputPort(ABC):
    """Puerto de salida para notificaciones."""
    
    @abstractmethod
    async def notify_cache_invalidation(
        self,
        affected_files: List[Path],
        reason: str
    ) -> None:
        """Notificar invalidación de cache."""
        pass
    
    @abstractmethod
    async def notify_analysis_complete(
        self,
        result: IncrementalAnalysisResult
    ) -> None:
        """Notificar análisis completado."""
        pass
    
    @abstractmethod
    async def notify_performance_degradation(
        self,
        metrics: Dict[str, Any]
    ) -> None:
        """Notificar degradación de rendimiento."""
        pass
