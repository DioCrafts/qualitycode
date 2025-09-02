"""
Sistema de cache para el motor de reglas estáticas.

Este módulo implementa un sistema de cache inteligente para optimizar
la performance del motor de reglas, reduciendo la re-ejecución de reglas.
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..models.config_models import CacheConfig, CacheStrategy
from ..models.rule_models import RuleResult

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Excepción base para errores del cache."""
    pass


class CacheKeyError(CacheError):
    """Error relacionado con claves de cache."""
    pass


class CacheValueError(CacheError):
    """Error relacionado con valores de cache."""
    pass


@dataclass
class CacheEntry:
    """Entrada del cache."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: int = 3600
    compressed: bool = False
    encrypted: bool = False
    
    def is_expired(self) -> bool:
        """Verificar si la entrada ha expirado."""
        return time.time() - self.created_at > self.ttl_seconds
    
    def update_access(self) -> None:
        """Actualizar información de acceso."""
        self.accessed_at = time.time()
        self.access_count += 1
    
    def get_age_seconds(self) -> float:
        """Obtener la edad de la entrada en segundos."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Estadísticas del cache."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    error_count: int = 0
    hit_rate: float = 0.0
    average_entry_size_bytes: float = 0.0
    oldest_entry_age_seconds: float = 0.0
    newest_entry_age_seconds: float = 0.0


class MemoryCache:
    """Cache en memoria."""
    
    def __init__(self, max_size_mb: int = 100):
        """Inicializar cache en memoria."""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.entries: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self.lock = asyncio.Lock()
        
        logger.info(f"MemoryCache initialized with {max_size_mb}MB max size")
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache."""
        async with self.lock:
            if key not in self.entries:
                self.stats.miss_count += 1
                return None
            
            entry = self.entries[key]
            
            if entry.is_expired():
                del self.entries[key]
                self.stats.miss_count += 1
                self.stats.total_entries -= 1
                self.stats.total_size_bytes -= entry.size_bytes
                return None
            
            entry.update_access()
            self.stats.hit_count += 1
            self._update_hit_rate()
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Establecer valor en el cache."""
        async with self.lock:
            # Calcular tamaño del valor
            size_bytes = self._calculate_size(value)
            
            # Verificar si hay espacio suficiente
            if size_bytes > self.max_size_bytes:
                logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return
            
            # Evictar entradas si es necesario
            await self._evict_if_needed(size_bytes)
            
            # Crear entrada
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds
            )
            
            # Si la clave ya existe, actualizar estadísticas
            if key in self.entries:
                old_entry = self.entries[key]
                self.stats.total_size_bytes -= old_entry.size_bytes
            
            # Añadir entrada
            self.entries[key] = entry
            self.stats.total_entries += 1
            self.stats.total_size_bytes += size_bytes
            self._update_average_size()
    
    async def delete(self, key: str) -> bool:
        """Eliminar entrada del cache."""
        async with self.lock:
            if key not in self.entries:
                return False
            
            entry = self.entries[key]
            del self.entries[key]
            
            self.stats.total_entries -= 1
            self.stats.total_size_bytes -= entry.size_bytes
            self._update_average_size()
            
            return True
    
    async def clear(self) -> None:
        """Limpiar todo el cache."""
        async with self.lock:
            self.entries.clear()
            self.stats.total_entries = 0
            self.stats.total_size_bytes = 0
            self._update_average_size()
    
    async def get_stats(self) -> CacheStats:
        """Obtener estadísticas del cache."""
        async with self.lock:
            # Actualizar estadísticas de edad
            if self.entries:
                ages = [entry.get_age_seconds() for entry in self.entries.values()]
                self.stats.oldest_entry_age_seconds = min(ages)
                self.stats.newest_entry_age_seconds = max(ages)
            
            return self.stats
    
    async def _evict_if_needed(self, required_size: int) -> None:
        """Evictar entradas si es necesario para hacer espacio."""
        available_space = self.max_size_bytes - self.stats.total_size_bytes
        
        if available_space >= required_size:
            return
        
        # Ordenar entradas por último acceso (LRU)
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda x: x[1].accessed_at
        )
        
        # Evictar entradas hasta tener espacio suficiente
        space_needed = required_size - available_space
        space_freed = 0
        
        for key, entry in sorted_entries:
            if space_freed >= space_needed:
                break
            
            del self.entries[key]
            space_freed += entry.size_bytes
            self.stats.eviction_count += 1
            self.stats.total_entries -= 1
            self.stats.total_size_bytes -= entry.size_bytes
    
    def _calculate_size(self, value: Any) -> int:
        """Calcular tamaño aproximado de un valor."""
        try:
            # Serializar a JSON para obtener tamaño aproximado
            json_str = json.dumps(value, default=str)
            return len(json_str.encode('utf-8'))
        except Exception:
            # Fallback: estimación basada en tipo
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(v) for v in value.values())
            else:
                return 1024  # Estimación por defecto
    
    def _update_hit_rate(self) -> None:
        """Actualizar tasa de aciertos."""
        total_requests = self.stats.hit_count + self.stats.miss_count
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hit_count / total_requests
    
    def _update_average_size(self) -> None:
        """Actualizar tamaño promedio de entradas."""
        if self.stats.total_entries > 0:
            self.stats.average_entry_size_bytes = (
                self.stats.total_size_bytes / self.stats.total_entries
            )


class RuleCache:
    """
    Sistema de cache para resultados de reglas.
    
    Este cache optimiza la performance del motor de reglas almacenando
    resultados de ejecución de reglas para evitar re-ejecución.
    """
    
    def __init__(self, config: CacheConfig):
        """Inicializar el cache de reglas."""
        self.config = config
        self.cache_strategy = config.strategy
        
        # Cache en memoria (siempre disponible)
        self.memory_cache = MemoryCache(config.max_size_mb)
        
        # Otros caches (se inicializarán según la estrategia)
        self.redis_cache = None
        self.disk_cache = None
        
        # Estadísticas
        self.stats = CacheStats()
        
        logger.info(f"RuleCache initialized with strategy: {config.strategy}")
    
    async def initialize(self) -> None:
        """Inicializar el cache según la estrategia configurada."""
        try:
            if self.cache_strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID]:
                await self._initialize_redis_cache()
            
            if self.cache_strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
                await self._initialize_disk_cache()
            
            logger.info("RuleCache initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize RuleCache: {e}")
            # Fallback a cache en memoria
            self.cache_strategy = CacheStrategy.MEMORY
    
    async def get(self, key: str) -> Optional[List[RuleResult]]:
        """
        Obtener resultados de reglas del cache.
        
        Args:
            key: Clave del cache
            
        Returns:
            Lista de resultados de reglas o None si no está en cache
        """
        try:
            # Intentar cache en memoria primero
            if self.cache_strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
                result = await self.memory_cache.get(key)
                if result is not None:
                    self.stats.hit_count += 1
                    return result
            
            # Intentar Redis si está disponible
            if self.cache_strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID]:
                if self.redis_cache:
                    result = await self._get_from_redis(key)
                    if result is not None:
                        # Cachear en memoria para acceso rápido
                        if self.cache_strategy == CacheStrategy.HYBRID:
                            await self.memory_cache.set(key, result, self.config.ttl_seconds)
                        self.stats.hit_count += 1
                        return result
            
            # Intentar cache en disco si está disponible
            if self.cache_strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
                if self.disk_cache:
                    result = await self._get_from_disk(key)
                    if result is not None:
                        # Cachear en memoria para acceso rápido
                        if self.cache_strategy == CacheStrategy.HYBRID:
                            await self.memory_cache.set(key, result, self.config.ttl_seconds)
                        self.stats.hit_count += 1
                        return result
            
            self.stats.miss_count += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            self.stats.error_count += 1
            return None
    
    async def set(self, key: str, value: List[RuleResult]) -> None:
        """
        Establecer resultados de reglas en el cache.
        
        Args:
            key: Clave del cache
            value: Lista de resultados de reglas
        """
        try:
            # Cachear en memoria
            if self.cache_strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
                await self.memory_cache.set(key, value, self.config.ttl_seconds)
            
            # Cachear en Redis si está disponible
            if self.cache_strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID]:
                if self.redis_cache:
                    await self._set_in_redis(key, value)
            
            # Cachear en disco si está disponible
            if self.cache_strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
                if self.disk_cache:
                    await self._set_in_disk(key, value)
            
        except Exception as e:
            logger.error(f"Error setting in cache: {e}")
            self.stats.error_count += 1
    
    async def delete(self, key: str) -> bool:
        """
        Eliminar entrada del cache.
        
        Args:
            key: Clave a eliminar
            
        Returns:
            True si se eliminó, False si no existía
        """
        try:
            deleted = False
            
            # Eliminar de memoria
            if self.cache_strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
                deleted |= await self.memory_cache.delete(key)
            
            # Eliminar de Redis
            if self.cache_strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID]:
                if self.redis_cache:
                    deleted |= await self._delete_from_redis(key)
            
            # Eliminar de disco
            if self.cache_strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
                if self.disk_cache:
                    deleted |= await self._delete_from_disk(key)
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            self.stats.error_count += 1
            return False
    
    async def clear(self) -> None:
        """Limpiar todo el cache."""
        try:
            # Limpiar memoria
            if self.cache_strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
                await self.memory_cache.clear()
            
            # Limpiar Redis
            if self.cache_strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID]:
                if self.redis_cache:
                    await self._clear_redis()
            
            # Limpiar disco
            if self.cache_strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
                if self.disk_cache:
                    await self._clear_disk()
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            self.stats.error_count += 1
    
    async def get_stats(self) -> CacheStats:
        """Obtener estadísticas del cache."""
        try:
            # Combinar estadísticas de todos los caches
            memory_stats = await self.memory_cache.get_stats()
            
            combined_stats = CacheStats(
                total_entries=memory_stats.total_entries,
                total_size_bytes=memory_stats.total_size_bytes,
                hit_count=self.stats.hit_count + memory_stats.hit_count,
                miss_count=self.stats.miss_count + memory_stats.miss_count,
                eviction_count=memory_stats.eviction_count,
                error_count=self.stats.error_count + memory_stats.error_count
            )
            
            # Calcular tasa de aciertos
            total_requests = combined_stats.hit_count + combined_stats.miss_count
            if total_requests > 0:
                combined_stats.hit_rate = combined_stats.hit_count / total_requests
            
            return combined_stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return self.stats
    
    async def _initialize_redis_cache(self) -> None:
        """Inicializar cache de Redis."""
        if not self.config.redis_url:
            logger.warning("Redis URL not configured, skipping Redis cache")
            return
        
        try:
            # En una implementación real, aquí se conectaría a Redis
            # import redis.asyncio as redis
            # self.redis_cache = redis.from_url(self.config.redis_url)
            logger.info("Redis cache initialization would happen here")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
    
    async def _initialize_disk_cache(self) -> None:
        """Inicializar cache en disco."""
        if not self.config.disk_path:
            logger.warning("Disk path not configured, skipping disk cache")
            return
        
        try:
            # Crear directorio si no existe
            self.config.disk_path.mkdir(parents=True, exist_ok=True)
            self.disk_cache = True
            logger.info(f"Disk cache initialized at {self.config.disk_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize disk cache: {e}")
    
    async def _get_from_redis(self, key: str) -> Optional[List[RuleResult]]:
        """Obtener valor de Redis."""
        # Implementación simulada
        return None
    
    async def _set_in_redis(self, key: str, value: List[RuleResult]) -> None:
        """Establecer valor en Redis."""
        # Implementación simulada
        pass
    
    async def _delete_from_redis(self, key: str) -> bool:
        """Eliminar valor de Redis."""
        # Implementación simulada
        return False
    
    async def _clear_redis(self) -> None:
        """Limpiar cache de Redis."""
        # Implementación simulada
        pass
    
    async def _get_from_disk(self, key: str) -> Optional[List[RuleResult]]:
        """Obtener valor de disco."""
        # Implementación simulada
        return None
    
    async def _set_in_disk(self, key: str, value: List[RuleResult]) -> None:
        """Establecer valor en disco."""
        # Implementación simulada
        pass
    
    async def _delete_from_disk(self, key: str) -> bool:
        """Eliminar valor de disco."""
        # Implementación simulada
        return False
    
    async def _clear_disk(self) -> None:
        """Limpiar cache de disco."""
        # Implementación simulada
        pass
    
    async def shutdown(self) -> None:
        """Apagar el cache."""
        try:
            # Cerrar conexiones de Redis si están abiertas
            if self.redis_cache:
                # await self.redis_cache.close()
                pass
            
            logger.info("RuleCache shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during cache shutdown: {e}")
