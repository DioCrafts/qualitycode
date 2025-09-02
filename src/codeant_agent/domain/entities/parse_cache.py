"""
Entidades para el sistema de cache del parser.

Este módulo define las entidades que representan el sistema
de cache para optimizar el parsing de código fuente.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum

from ..value_objects.programming_language import ProgrammingLanguage


class CacheLevel(Enum):
    """Niveles de cache."""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"


class CacheStatus(Enum):
    """Estado de un elemento en cache."""
    VALID = "valid"
    EXPIRED = "expired"
    STALE = "stale"
    INVALID = "invalid"


class EvictionPolicy(Enum):
    """Políticas de evicción de cache."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    HYBRID = "hybrid"     # Combinación de LRU y TTL
    SIZE_BASED = "size_based"  # Basado en tamaño


@dataclass
class CacheConfig:
    """Configuración del sistema de cache."""
    
    memory_cache_size: int = 1000  # Número máximo de elementos en memoria
    disk_cache_size_mb: int = 1024  # Tamaño máximo en disco (MB)
    ttl_seconds: int = 3600  # Tiempo de vida por defecto (1 hora)
    enable_disk_cache: bool = True
    enable_compression: bool = True
    eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID
    cleanup_interval_seconds: int = 300  # Intervalo de limpieza (5 minutos)
    max_file_size_mb: int = 100  # Tamaño máximo de archivo para cachear
    enable_metrics: bool = True
    enable_persistence: bool = True
    
    def __post_init__(self) -> None:
        """Validar la configuración del cache."""
        if self.memory_cache_size <= 0:
            raise ValueError("El tamaño del cache en memoria debe ser mayor a 0")
        
        if self.disk_cache_size_mb <= 0:
            raise ValueError("El tamaño del cache en disco debe ser mayor a 0")
        
        if self.ttl_seconds <= 0:
            raise ValueError("El TTL debe ser mayor a 0")
        
        if self.cleanup_interval_seconds <= 0:
            raise ValueError("El intervalo de limpieza debe ser mayor a 0")
        
        if self.max_file_size_mb <= 0:
            raise ValueError("El tamaño máximo de archivo debe ser mayor a 0")
    
    def get_ttl_timedelta(self) -> timedelta:
        """Obtiene el TTL como timedelta."""
        return timedelta(seconds=self.ttl_seconds)
    
    def get_cleanup_interval_timedelta(self) -> timedelta:
        """Obtiene el intervalo de limpieza como timedelta."""
        return timedelta(seconds=self.cleanup_interval_seconds)
    
    def get_disk_cache_size_bytes(self) -> int:
        """Obtiene el tamaño del cache en disco en bytes."""
        return self.disk_cache_size_mb * 1024 * 1024
    
    def get_max_file_size_bytes(self) -> int:
        """Obtiene el tamaño máximo de archivo en bytes."""
        return self.max_file_size_mb * 1024 * 1024


@dataclass
class CachedParseResult:
    """Resultado de parsing cacheado."""
    
    parse_result: Any  # ParseResult object
    cached_at: datetime
    last_accessed: datetime
    access_count: int = 0
    file_hash: str = ""
    file_size: int = 0
    language: ProgrammingLanguage = None
    cache_level: CacheLevel = CacheLevel.MEMORY
    compression_ratio: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validar el resultado cacheado."""
        if self.parse_result is None:
            raise ValueError("El resultado de parsing no puede ser None")
        
        if self.cached_at is None:
            raise ValueError("La fecha de cache no puede ser None")
        
        if self.last_accessed is None:
            raise ValueError("La fecha de último acceso no puede ser None")
        
        if self.access_count < 0:
            raise ValueError("El contador de accesos no puede ser negativo")
        
        if self.file_size < 0:
            raise ValueError("El tamaño del archivo no puede ser negativo")
        
        if self.language is None:
            raise ValueError("El lenguaje no puede ser None")
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Verifica si el elemento ha expirado."""
        now = datetime.utcnow()
        age = now - self.cached_at
        return age.total_seconds() > ttl_seconds
    
    def is_stale(self, max_age_seconds: int) -> bool:
        """Verifica si el elemento está obsoleto."""
        now = datetime.utcnow()
        age = now - self.cached_at
        return age.total_seconds() > max_age_seconds
    
    def get_age_seconds(self) -> float:
        """Obtiene la edad del elemento en segundos."""
        now = datetime.utcnow()
        age = now - self.cached_at
        return age.total_seconds()
    
    def get_time_since_last_access_seconds(self) -> float:
        """Obtiene el tiempo desde el último acceso en segundos."""
        now = datetime.utcnow()
        time_since = now - self.last_accessed
        return time_since.total_seconds()
    
    def touch(self) -> None:
        """Actualiza el timestamp de último acceso."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def get_cache_efficiency_score(self) -> float:
        """Calcula un score de eficiencia del cache."""
        # Factor de frecuencia de acceso
        frequency_score = min(self.access_count / 10.0, 1.0)
        
        # Factor de edad (elementos más nuevos tienen mejor score)
        age_score = max(0.0, 1.0 - (self.get_age_seconds() / 3600.0))
        
        # Factor de tamaño (archivos más pequeños son más eficientes)
        size_score = max(0.0, 1.0 - (self.file_size / (100 * 1024 * 1024)))  # 100MB como referencia
        
        # Score compuesto
        return (frequency_score * 0.4 + age_score * 0.4 + size_score * 0.2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el elemento cacheado a diccionario."""
        return {
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "language": self.language.value if self.language else None,
            "language_name": self.language.get_name() if self.language else None,
            "cache_level": self.cache_level.value,
            "cached_at": self.cached_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "age_seconds": self.get_age_seconds(),
            "time_since_last_access_seconds": self.get_time_since_last_access_seconds(),
            "compression_ratio": self.compression_ratio,
            "cache_efficiency_score": self.get_cache_efficiency_score(),
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        """Representación string del elemento cacheado."""
        age_hours = self.get_age_seconds() / 3600
        return f"CachedParseResult({self.language.get_name() if self.language else 'Unknown'}, {age_hours:.1f}h, {self.access_count} accesos)"
    
    def __repr__(self) -> str:
        """Representación de debug del elemento cacheado."""
        return (
            f"CachedParseResult("
            f"parse_result={type(self.parse_result).__name__}, "
            f"cached_at={self.cached_at}, "
            f"last_accessed={self.last_accessed}, "
            f"access_count={self.access_count}, "
            f"file_hash={self.file_hash}, "
            f"file_size={self.file_size}, "
            f"language={self.language}, "
            f"cache_level={self.cache_level}"
            f")"
        )


@dataclass
class CacheStats:
    """Estadísticas del sistema de cache."""
    
    memory_cache_size: int
    memory_cache_capacity: int
    disk_cache_size_mb: float
    disk_cache_capacity_mb: float
    total_items: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    compression_ratio: float
    avg_access_time_ms: float
    last_cleanup: datetime
    cache_efficiency: float
    
    def __post_init__(self) -> None:
        """Validar las estadísticas."""
        if self.memory_cache_size < 0:
            raise ValueError("El tamaño del cache en memoria no puede ser negativo")
        
        if self.memory_cache_capacity <= 0:
            raise ValueError("La capacidad del cache en memoria debe ser mayor a 0")
        
        if self.disk_cache_size_mb < 0:
            raise ValueError("El tamaño del cache en disco no puede ser negativo")
        
        if self.disk_cache_capacity_mb <= 0:
            raise ValueError("La capacidad del cache en disco debe ser mayor a 0")
        
        if self.total_items < 0:
            raise ValueError("El número total de elementos no puede ser negativo")
        
        if self.total_size_bytes < 0:
            raise ValueError("El tamaño total no puede ser negativo")
        
        if self.hit_count < 0:
            raise ValueError("El número de hits no puede ser negativo")
        
        if self.miss_count < 0:
            raise ValueError("El número de misses no puede ser negativo")
        
        if self.eviction_count < 0:
            raise ValueError("El número de evicciones no puede ser negativo")
        
        if not 0.0 <= self.compression_ratio <= 1.0:
            raise ValueError("El ratio de compresión debe estar entre 0.0 y 1.0")
        
        if self.avg_access_time_ms < 0:
            raise ValueError("El tiempo promedio de acceso no puede ser negativo")
        
        if not 0.0 <= self.cache_efficiency <= 1.0:
            raise ValueError("La eficiencia del cache debe estar entre 0.0 y 1.0")
    
    @property
    def hit_rate(self) -> float:
        """Calcula la tasa de hits del cache."""
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return 0.0
        return self.hit_count / total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calcula la tasa de misses del cache."""
        return 1.0 - self.hit_rate
    
    @property
    def memory_usage_percentage(self) -> float:
        """Calcula el porcentaje de uso de memoria."""
        if self.memory_cache_capacity == 0:
            return 0.0
        return (self.memory_cache_size / self.memory_cache_capacity) * 100
    
    @property
    def disk_usage_percentage(self) -> float:
        """Calcula el porcentaje de uso de disco."""
        if self.disk_cache_capacity_mb == 0:
            return 0.0
        return (self.disk_cache_size_mb / self.disk_cache_capacity_mb) * 100
    
    @property
    def total_requests(self) -> int:
        """Calcula el número total de requests."""
        return self.hit_count + self.miss_count
    
    def get_memory_efficiency(self) -> str:
        """Obtiene una descripción de la eficiencia de memoria."""
        usage_pct = self.memory_usage_percentage
        if usage_pct < 50:
            return "Excelente"
        elif usage_pct < 75:
            return "Buena"
        elif usage_pct < 90:
            return "Aceptable"
        else:
            return "Crítica"
    
    def get_disk_efficiency(self) -> str:
        """Obtiene una descripción de la eficiencia de disco."""
        usage_pct = self.disk_usage_percentage
        if usage_pct < 50:
            return "Excelente"
        elif usage_pct < 75:
            return "Buena"
        elif usage_pct < 90:
            return "Aceptable"
        else:
            return "Crítica"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte las estadísticas a diccionario."""
        return {
            "memory_cache": {
                "size": self.memory_cache_size,
                "capacity": self.memory_cache_capacity,
                "usage_percentage": self.memory_usage_percentage,
                "efficiency": self.get_memory_efficiency(),
            },
            "disk_cache": {
                "size_mb": self.disk_cache_size_mb,
                "capacity_mb": self.disk_cache_capacity_mb,
                "usage_percentage": self.disk_usage_percentage,
                "efficiency": self.get_disk_efficiency(),
            },
            "performance": {
                "total_items": self.total_items,
                "total_size_bytes": self.total_size_bytes,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "total_requests": self.total_requests,
                "hit_rate": self.hit_rate,
                "miss_rate": self.miss_rate,
                "eviction_count": self.eviction_count,
                "compression_ratio": self.compression_ratio,
                "avg_access_time_ms": self.avg_access_time_ms,
                "cache_efficiency": self.cache_efficiency,
            },
            "maintenance": {
                "last_cleanup": self.last_cleanup.isoformat(),
            },
        }
    
    def __str__(self) -> str:
        """Representación string de las estadísticas."""
        hit_rate_pct = self.hit_rate * 100
        memory_usage_pct = self.memory_usage_percentage
        disk_usage_pct = self.disk_usage_percentage
        
        return (
            f"CacheStats("
            f"hit_rate={hit_rate_pct:.1f}%, "
            f"memory={memory_usage_pct:.1f}%, "
            f"disk={disk_usage_pct:.1f}%, "
            f"items={self.total_items}"
            f")"
        )
    
    def __repr__(self) -> str:
        """Representación de debug de las estadísticas."""
        return (
            f"CacheStats("
            f"memory_cache_size={self.memory_cache_size}, "
            f"memory_cache_capacity={self.memory_cache_capacity}, "
            f"disk_cache_size_mb={self.disk_cache_size_mb}, "
            f"disk_cache_capacity_mb={self.disk_cache_capacity_mb}, "
            f"total_items={self.total_items}, "
            f"hit_count={self.hit_count}, "
            f"miss_count={self.miss_count}, "
            f"cache_efficiency={self.cache_efficiency}"
            f")"
        )
