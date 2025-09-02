"""
Repositorios del dominio para el Sistema de Análisis Incremental.

Este módulo define las interfaces de repositorios para el sistema de análisis
incremental, siguiendo los principios de inversión de dependencias.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
from pathlib import Path

from ..entities.incremental import (
    ChangeSet, ChangeSetId, CacheKey, CacheEntry, CacheLevel,
    InvalidationRequest, InvalidationResult, AccessPrediction,
    CacheWarmingResult, IncrementalAnalysisResult, FileChange,
    GranularChange, DependencyImpact, AnalysisScope
)
from ..value_objects.incremental_metrics import (
    CacheMetrics, IncrementalMetrics, ChangeMetrics
)


class ChangeSetRepository(ABC):
    """Repositorio para conjuntos de cambios."""
    
    @abstractmethod
    async def save(self, change_set: ChangeSet) -> None:
        """Guardar un conjunto de cambios."""
        pass
    
    @abstractmethod
    async def find_by_id(self, change_set_id: ChangeSetId) -> Optional[ChangeSet]:
        """Buscar conjunto de cambios por ID."""
        pass
    
    @abstractmethod
    async def find_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[ChangeSet]:
        """Buscar conjuntos de cambios por rango de fechas."""
        pass
    
    @abstractmethod
    async def find_by_file_path(self, file_path: Path) -> List[ChangeSet]:
        """Buscar conjuntos de cambios que afectan un archivo."""
        pass
    
    @abstractmethod
    async def get_latest(self, limit: int = 10) -> List[ChangeSet]:
        """Obtener los últimos conjuntos de cambios."""
        pass
    
    @abstractmethod
    async def delete_older_than(self, date: datetime) -> int:
        """Eliminar conjuntos de cambios más antiguos que una fecha."""
        pass


class CacheRepository(ABC):
    """Repositorio para el sistema de cache multi-nivel."""
    
    @abstractmethod
    async def get(self, key: CacheKey, level: Optional[CacheLevel] = None) -> Optional[CacheEntry]:
        """Obtener entrada del cache."""
        pass
    
    @abstractmethod
    async def set(
        self, 
        key: CacheKey, 
        entry: CacheEntry, 
        level: CacheLevel
    ) -> None:
        """Guardar entrada en el cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: CacheKey, level: Optional[CacheLevel] = None) -> bool:
        """Eliminar entrada del cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: CacheKey, level: Optional[CacheLevel] = None) -> bool:
        """Verificar si existe una entrada."""
        pass
    
    @abstractmethod
    async def get_metrics(self, level: Optional[CacheLevel] = None) -> CacheMetrics:
        """Obtener métricas del cache."""
        pass
    
    @abstractmethod
    async def clear(self, level: Optional[CacheLevel] = None) -> int:
        """Limpiar el cache."""
        pass
    
    @abstractmethod
    async def find_by_pattern(
        self, 
        pattern: str, 
        level: Optional[CacheLevel] = None
    ) -> List[CacheKey]:
        """Buscar claves por patrón."""
        pass
    
    @abstractmethod
    async def promote(self, key: CacheKey, from_level: CacheLevel, to_level: CacheLevel) -> bool:
        """Promover entrada entre niveles de cache."""
        pass
    
    @abstractmethod
    async def evict_lru(self, level: CacheLevel, count: int) -> List[CacheKey]:
        """Desalojar entradas menos recientemente usadas."""
        pass


class DependencyRepository(ABC):
    """Repositorio para el grafo de dependencias."""
    
    @abstractmethod
    async def save_dependencies(
        self, 
        file_path: Path, 
        dependencies: Set[Path]
    ) -> None:
        """Guardar dependencias de un archivo."""
        pass
    
    @abstractmethod
    async def get_dependencies(self, file_path: Path) -> Set[Path]:
        """Obtener dependencias directas de un archivo."""
        pass
    
    @abstractmethod
    async def get_dependents(self, file_path: Path) -> Set[Path]:
        """Obtener archivos que dependen de uno dado."""
        pass
    
    @abstractmethod
    async def get_transitive_dependencies(
        self, 
        file_path: Path, 
        max_depth: Optional[int] = None
    ) -> Set[Path]:
        """Obtener dependencias transitivas."""
        pass
    
    @abstractmethod
    async def get_transitive_dependents(
        self, 
        file_path: Path, 
        max_depth: Optional[int] = None
    ) -> Set[Path]:
        """Obtener dependientes transitivos."""
        pass
    
    @abstractmethod
    async def find_circular_dependencies(self) -> List[List[Path]]:
        """Encontrar dependencias circulares."""
        pass
    
    @abstractmethod
    async def update_dependency_graph(self, changes: List[FileChange]) -> None:
        """Actualizar grafo basado en cambios."""
        pass
    
    @abstractmethod
    async def calculate_impact_radius(self, file_path: Path) -> int:
        """Calcular radio de impacto de cambios en un archivo."""
        pass


class IncrementalAnalysisRepository(ABC):
    """Repositorio para resultados de análisis incremental."""
    
    @abstractmethod
    async def save_result(self, result: IncrementalAnalysisResult) -> None:
        """Guardar resultado de análisis incremental."""
        pass
    
    @abstractmethod
    async def find_by_change_set(
        self, 
        change_set_id: ChangeSetId
    ) -> Optional[IncrementalAnalysisResult]:
        """Buscar resultado por ID de conjunto de cambios."""
        pass
    
    @abstractmethod
    async def get_metrics(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> IncrementalMetrics:
        """Obtener métricas de análisis incremental."""
        pass
    
    @abstractmethod
    async def get_performance_history(
        self, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener historial de rendimiento."""
        pass
    
    @abstractmethod
    async def cleanup_old_results(self, days: int) -> int:
        """Limpiar resultados antiguos."""
        pass


class InvalidationRepository(ABC):
    """Repositorio para gestión de invalidaciones."""
    
    @abstractmethod
    async def record_invalidation(
        self, 
        request: InvalidationRequest, 
        result: InvalidationResult
    ) -> None:
        """Registrar una invalidación."""
        pass
    
    @abstractmethod
    async def get_invalidation_history(
        self, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener historial de invalidaciones."""
        pass
    
    @abstractmethod
    async def get_pending_invalidations(self) -> List[InvalidationRequest]:
        """Obtener invalidaciones pendientes."""
        pass
    
    @abstractmethod
    async def mark_invalidation_complete(
        self, 
        request_id: str
    ) -> None:
        """Marcar invalidación como completa."""
        pass
    
    @abstractmethod
    async def get_invalidation_impact_analysis(
        self, 
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analizar impacto de invalidaciones."""
        pass


class PredictionRepository(ABC):
    """Repositorio para predicciones de acceso."""
    
    @abstractmethod
    async def save_predictions(
        self, 
        predictions: List[AccessPrediction]
    ) -> None:
        """Guardar predicciones de acceso."""
        pass
    
    @abstractmethod
    async def get_active_predictions(self) -> List[AccessPrediction]:
        """Obtener predicciones activas."""
        pass
    
    @abstractmethod
    async def update_prediction_accuracy(
        self, 
        cache_key: CacheKey, 
        was_accessed: bool
    ) -> None:
        """Actualizar precisión de predicción."""
        pass
    
    @abstractmethod
    async def get_prediction_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de predicción."""
        pass
    
    @abstractmethod
    async def get_access_patterns(
        self, 
        time_window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Obtener patrones de acceso."""
        pass
    
    @abstractmethod
    async def save_warming_result(
        self, 
        result: CacheWarmingResult
    ) -> None:
        """Guardar resultado de cache warming."""
        pass


class AnalysisHistoryRepository(ABC):
    """Repositorio para historial de análisis."""
    
    @abstractmethod
    async def save_file_analysis(
        self, 
        file_path: Path, 
        analysis_data: Dict[str, Any]
    ) -> None:
        """Guardar análisis de archivo."""
        pass
    
    @abstractmethod
    async def get_file_history(
        self, 
        file_path: Path, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Obtener historial de análisis de un archivo."""
        pass
    
    @abstractmethod
    async def get_change_frequency(
        self, 
        file_path: Path, 
        days: int = 30
    ) -> float:
        """Obtener frecuencia de cambios de un archivo."""
        pass
    
    @abstractmethod
    async def get_hot_files(
        self, 
        limit: int = 20
    ) -> List[Tuple[Path, float]]:
        """Obtener archivos más frecuentemente modificados."""
        pass
    
    @abstractmethod
    async def get_co_change_patterns(
        self, 
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Obtener patrones de co-modificación."""
        pass


class ResourceMonitorRepository(ABC):
    """Repositorio para monitoreo de recursos."""
    
    @abstractmethod
    async def record_resource_usage(
        self, 
        timestamp: datetime, 
        usage: Dict[str, Any]
    ) -> None:
        """Registrar uso de recursos."""
        pass
    
    @abstractmethod
    async def get_current_usage(self) -> Dict[str, Any]:
        """Obtener uso actual de recursos."""
        pass
    
    @abstractmethod
    async def get_usage_history(
        self, 
        time_window_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Obtener historial de uso de recursos."""
        pass
    
    @abstractmethod
    async def has_sufficient_resources(
        self, 
        required: Dict[str, Any]
    ) -> bool:
        """Verificar si hay recursos suficientes."""
        pass
    
    @abstractmethod
    async def get_resource_alerts(self) -> List[Dict[str, Any]]:
        """Obtener alertas de recursos."""
        pass
