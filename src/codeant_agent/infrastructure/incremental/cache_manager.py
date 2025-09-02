"""
Gestor de cache multi-nivel para el Sistema de Análisis Incremental.

Este módulo implementa un sistema de cache inteligente con múltiples niveles
(L1: memoria, L2: Redis, L3: disco) para optimizar el rendimiento.
"""

import asyncio
import time
import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Tuple, Set
from pathlib import Path
from datetime import datetime, timedelta
from collections import OrderedDict
import aiofiles
import sys

from ...domain.entities.incremental import (
    CacheKey, CacheEntry, CacheLevel, InvalidationRequest,
    InvalidationStrategy, AccessPrediction
)
from ...domain.value_objects.incremental_metrics import CacheMetrics
from ...domain.services.incremental_service import (
    CacheManagementService, InvalidationService
)
from ...domain.repositories.incremental_repository import CacheRepository
from ...application.ports.incremental_ports import (
    CacheStorageOutputPort, MetricsCollectorOutputPort
)
from .incremental_config import IncrementalConfig


class LRUCache:
    """Cache LRU en memoria (L1)."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[str, Tuple[Any, float, int]] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache."""
        if key in self.cache:
            # Mover al final (más reciente)
            value, expiry, size = self.cache.pop(key)
            
            # Verificar expiración
            if expiry > 0 and time.time() > expiry:
                self.misses += 1
                return None
            
            self.cache[key] = (value, expiry, size)
            self.hits += 1
            return value
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any, ttl: int = 0) -> bool:
        """Almacenar valor en cache."""
        try:
            # Calcular tamaño aproximado
            size = sys.getsizeof(value)
            
            # Calcular expiración
            expiry = time.time() + ttl if ttl > 0 else 0
            
            # Eliminar si ya existe
            if key in self.cache:
                self.cache.pop(key)
            
            # Verificar capacidad
            while len(self.cache) >= self.capacity:
                # Eliminar el más antiguo
                self.cache.popitem(last=False)
            
            # Añadir al final
            self.cache[key] = (value, expiry, size)
            return True
            
        except Exception:
            return False
    
    def invalidate(self, key: str) -> bool:
        """Invalidar entrada del cache."""
        if key in self.cache:
            self.cache.pop(key)
            return True
        return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidar entradas que coincidan con patrón."""
        import re
        regex = re.compile(pattern.replace('*', '.*'))
        
        keys_to_remove = [
            key for key in self.cache.keys()
            if regex.match(key)
        ]
        
        for key in keys_to_remove:
            self.cache.pop(key)
        
        return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        total_size = sum(size for _, _, size in self.cache.values())
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        
        return {
            'items': len(self.cache),
            'size_bytes': total_size,
            'size_mb': total_size / (1024 * 1024),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'capacity': self.capacity
        }
    
    def clear(self):
        """Limpiar todo el cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class SmartCacheManager(
    CacheManagementService,
    InvalidationService,
    CacheStorageOutputPort,
    CacheRepository
):
    """Gestor inteligente de cache multi-nivel."""
    
    def __init__(
        self,
        config: IncrementalConfig,
        metrics_collector: MetricsCollectorOutputPort
    ):
        self.config = config
        self.metrics_collector = metrics_collector
        
        # Inicializar caches
        self.l1_cache = LRUCache(config.l1_cache_size)
        self.l2_cache = None  # Redis (simplificado para esta implementación)
        self.l3_cache_dir = Path(config.l3_cache_directory)
        self.l3_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Estadísticas
        self.access_history: List[Tuple[str, datetime]] = []
        self.invalidation_history: List[Dict[str, Any]] = []
        
        # Cache de predicciones
        self.access_predictions: List[AccessPrediction] = []
    
    # Implementación de CacheManagementService
    
    async def get_cached_item(self, key: CacheKey) -> Optional[Any]:
        """Obtener item del cache."""
        key_str = self._cache_key_to_string(key)
        
        # Registrar acceso
        self.access_history.append((key_str, datetime.now()))
        
        # Buscar en L1
        value = self.l1_cache.get(key_str)
        if value is not None:
            await self.metrics_collector.record_cache_hit(
                CacheLevel.L1, key_str
            )
            return value
        
        # Buscar en L2 (Redis - simplificado)
        if self.l2_cache:
            value = await self._get_from_l2(key_str)
            if value is not None:
                # Promover a L1
                await self.promote_cache_entry(key, CacheLevel.L1)
                await self.metrics_collector.record_cache_hit(
                    CacheLevel.L2, key_str
                )
                return value
        
        # Buscar en L3 (disco)
        value = await self._get_from_l3(key_str)
        if value is not None:
            # Promover a L1
            await self.promote_cache_entry(key, CacheLevel.L1)
            await self.metrics_collector.record_cache_hit(
                CacheLevel.L3, key_str
            )
            return value
        
        # Cache miss
        await self.metrics_collector.record_cache_miss(key_str)
        return None
    
    async def cache_item(
        self,
        key: CacheKey,
        item: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Cachear un item."""
        key_str = self._cache_key_to_string(key)
        ttl = ttl or self.config.default_cache_ttl
        
        # Determinar nivel óptimo
        level = await self._determine_optimal_cache_level(key, item)
        
        # Almacenar según nivel
        if level == CacheLevel.L1:
            await self.store_in_l1(key_str, item, ttl)
        elif level == CacheLevel.L2:
            await self.store_in_l2(key_str, item, ttl)
        else:  # L3
            await self.store_in_l3(key_str, item, ttl)
    
    async def invalidate_cache(
        self,
        request: InvalidationRequest
    ) -> int:
        """Invalidar entradas del cache."""
        total_invalidated = 0
        
        # Registrar invalidación
        self.invalidation_history.append({
            'request': request,
            'timestamp': datetime.now()
        })
        
        # Invalidar por estrategia
        if request.strategy == InvalidationStrategy.EXACT:
            for key in request.keys:
                if await self._invalidate_exact(key):
                    total_invalidated += 1
                    
        elif request.strategy == InvalidationStrategy.PATTERN:
            for pattern in request.patterns:
                total_invalidated += await self._invalidate_pattern(pattern)
                
        elif request.strategy == InvalidationStrategy.TAG:
            for tag in request.tags:
                total_invalidated += await self._invalidate_tag(tag)
                
        elif request.strategy == InvalidationStrategy.AGE:
            total_invalidated = await self._invalidate_by_age(
                request.max_age_seconds
            )
        
        # Registrar evento
        await self.metrics_collector.record_invalidation_event(
            pattern=str(request),
            affected_count=total_invalidated
        )
        
        return total_invalidated
    
    async def promote_cache_entry(
        self,
        key: CacheKey,
        to_level: CacheLevel
    ) -> bool:
        """Promover entrada a un nivel superior de cache."""
        key_str = self._cache_key_to_string(key)
        
        # Buscar valor en niveles inferiores
        value = None
        from_level = None
        
        if to_level == CacheLevel.L1:
            # Buscar en L2 y L3
            if self.l2_cache:
                value = await self._get_from_l2(key_str)
                from_level = CacheLevel.L2
            
            if value is None:
                value = await self._get_from_l3(key_str)
                from_level = CacheLevel.L3
                
        elif to_level == CacheLevel.L2:
            # Buscar en L3
            value = await self._get_from_l3(key_str)
            from_level = CacheLevel.L3
        
        if value is None:
            return False
        
        # Promover al nivel deseado
        if to_level == CacheLevel.L1:
            self.l1_cache.put(key_str, value, self.config.default_cache_ttl)
        elif to_level == CacheLevel.L2 and self.l2_cache:
            await self._store_in_l2(key_str, value, self.config.default_cache_ttl)
        
        return True
    
    async def optimize_cache_distribution(self) -> Dict[str, Any]:
        """Optimizar distribución de datos entre niveles."""
        optimization_results = {
            'items_moved': 0,
            'space_freed': 0,
            'performance_improvement': 0
        }
        
        # Analizar patrones de acceso
        access_patterns = self._analyze_access_patterns()
        
        # Identificar items calientes
        hot_items = self._identify_hot_items(access_patterns)
        
        # Promover items calientes a L1
        for item_key in hot_items:
            if await self.promote_cache_entry(
                self._string_to_cache_key(item_key),
                CacheLevel.L1
            ):
                optimization_results['items_moved'] += 1
        
        # Degradar items fríos
        cold_items = self._identify_cold_items(access_patterns)
        for item_key in cold_items:
            if await self._demote_cache_entry(item_key):
                optimization_results['items_moved'] += 1
        
        # Limpiar items expirados
        expired_count = await self._clean_expired_items()
        optimization_results['space_freed'] = expired_count
        
        return optimization_results
    
    async def get_cache_metrics(self) -> CacheMetrics:
        """Obtener métricas del sistema de cache."""
        l1_stats = self.l1_cache.get_stats()
        
        # Calcular métricas globales
        total_hits = l1_stats['hits']  # + L2 + L3
        total_misses = l1_stats['misses']
        
        return CacheMetrics(
            hit_rate=l1_stats['hit_rate'],
            total_hits=total_hits,
            total_misses=total_misses,
            total_size_mb=l1_stats['size_mb'],
            eviction_count=0,  # TODO: implementar tracking de evictions
            average_retrieval_time_ms=0,  # TODO: implementar medición
            l1_size_mb=l1_stats['size_mb'],
            l1_items=l1_stats['items'],
            l1_hit_rate=l1_stats['hit_rate'],
            l2_size_mb=0,  # TODO: implementar para Redis
            l2_items=0,
            l2_hit_rate=0,
            l3_size_mb=await self._calculate_l3_size(),
            l3_items=await self._count_l3_items(),
            l3_hit_rate=0  # TODO: implementar tracking
        )
    
    # Implementación de InvalidationService
    
    async def create_invalidation_plan(
        self,
        changes: List[Any]
    ) -> List[InvalidationRequest]:
        """Crear plan de invalidación basado en cambios."""
        invalidation_requests = []
        
        # Agrupar cambios por archivo
        files_changed = set()
        for change in changes:
            if hasattr(change, 'location'):
                files_changed.add(change.location.file_path)
        
        # Crear requests de invalidación
        for file_path in files_changed:
            # Invalidar análisis del archivo
            request = InvalidationRequest(
                strategy=InvalidationStrategy.PATTERN,
                patterns=[f"{file_path}:*"],
                reason=f"File {file_path} changed",
                cascade=True
            )
            invalidation_requests.append(request)
        
        return invalidation_requests
    
    async def execute_invalidation(
        self,
        request: InvalidationRequest
    ) -> int:
        """Ejecutar invalidación de cache."""
        return await self.invalidate_cache(request)
    
    async def invalidate_dependencies(
        self,
        file_path: Path
    ) -> int:
        """Invalidar cache de dependencias."""
        pattern = f"*{file_path}*dependency*"
        request = InvalidationRequest(
            strategy=InvalidationStrategy.PATTERN,
            patterns=[pattern],
            reason=f"Dependencies of {file_path} invalidated",
            cascade=True
        )
        return await self.invalidate_cache(request)
    
    async def schedule_lazy_invalidation(
        self,
        request: InvalidationRequest,
        delay: timedelta
    ) -> None:
        """Programar invalidación diferida."""
        # En una implementación real, esto usaría un scheduler
        # Por ahora, ejecutamos inmediatamente
        await asyncio.sleep(delay.total_seconds())
        await self.execute_invalidation(request)
    
    async def calculate_invalidation_impact(
        self,
        request: InvalidationRequest
    ) -> Dict[str, Any]:
        """Calcular impacto de una invalidación."""
        impact = {
            'affected_items': 0,
            'space_to_free': 0,
            'dependent_items': 0
        }
        
        # Estimar items afectados
        if request.strategy == InvalidationStrategy.PATTERN:
            for pattern in request.patterns:
                # Contar items que coinciden
                count = self._count_matching_items(pattern)
                impact['affected_items'] += count
        
        # Estimar espacio a liberar
        impact['space_to_free'] = impact['affected_items'] * 1024  # Estimación
        
        return impact
    
    # Implementación de CacheStorageOutputPort
    
    async def store_in_l1(self, key: str, value: Any, ttl: int) -> bool:
        """Almacenar en cache L1 (memoria)."""
        return self.l1_cache.put(key, value, ttl)
    
    async def store_in_l2(self, key: str, value: Any, ttl: int) -> bool:
        """Almacenar en cache L2 (Redis)."""
        if self.l2_cache:
            return await self._store_in_l2(key, value, ttl)
        return False
    
    async def store_in_l3(self, key: str, value: Any, ttl: int) -> bool:
        """Almacenar en cache L3 (disco)."""
        try:
            file_path = self._get_l3_file_path(key)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Serializar y guardar
            data = {
                'value': value,
                'expiry': time.time() + ttl if ttl > 0 else 0,
                'created': time.time()
            }
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(pickle.dumps(data))
            
            return True
            
        except Exception:
            return False
    
    async def retrieve_from_cache(
        self,
        key: str,
        level: CacheLevel
    ) -> Optional[Any]:
        """Recuperar de cache específico."""
        if level == CacheLevel.L1:
            return self.l1_cache.get(key)
        elif level == CacheLevel.L2:
            return await self._get_from_l2(key)
        elif level == CacheLevel.L3:
            return await self._get_from_l3(key)
        return None
    
    async def invalidate_by_pattern(
        self,
        pattern: str,
        levels: List[CacheLevel]
    ) -> int:
        """Invalidar por patrón en niveles específicos."""
        total = 0
        
        if CacheLevel.L1 in levels:
            total += self.l1_cache.invalidate_pattern(pattern)
        
        if CacheLevel.L2 in levels and self.l2_cache:
            total += await self._invalidate_l2_pattern(pattern)
        
        if CacheLevel.L3 in levels:
            total += await self._invalidate_l3_pattern(pattern)
        
        return total
    
    # Implementación de CacheRepository
    
    async def save_cached_item(
        self,
        key: CacheKey,
        item: Any,
        ttl: int
    ) -> None:
        """Guardar item en cache."""
        await self.cache_item(key, item, ttl)
    
    async def get_cache_entry(self, key: CacheKey) -> Optional[CacheEntry]:
        """Obtener entrada de cache con metadatos."""
        value = await self.get_cached_item(key)
        if value is None:
            return None
        
        return CacheEntry(
            key=key,
            value=value,
            level=CacheLevel.L1,  # Simplificado
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            size_bytes=sys.getsizeof(value),
            ttl=self.config.default_cache_ttl
        )
    
    async def invalidate_cache_item(self, key: CacheKey) -> bool:
        """Invalidar item específico del cache."""
        key_str = self._cache_key_to_string(key)
        return await self._invalidate_exact(key_str)
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        metrics = await self.get_cache_metrics()
        return {
            'metrics': metrics,
            'access_history_size': len(self.access_history),
            'invalidation_history_size': len(self.invalidation_history)
        }
    
    # Métodos auxiliares privados
    
    def _cache_key_to_string(self, key: CacheKey) -> str:
        """Convertir CacheKey a string."""
        parts = [key.path, key.analysis_type]
        if key.hash:
            parts.append(key.hash)
        return ":".join(parts)
    
    def _string_to_cache_key(self, key_str: str) -> CacheKey:
        """Convertir string a CacheKey."""
        parts = key_str.split(":")
        return CacheKey(
            path=parts[0],
            analysis_type=parts[1] if len(parts) > 1 else "",
            hash=parts[2] if len(parts) > 2 else None
        )
    
    async def _determine_optimal_cache_level(
        self,
        key: CacheKey,
        item: Any
    ) -> CacheLevel:
        """Determinar nivel óptimo de cache para un item."""
        # Factores a considerar
        size = sys.getsizeof(item)
        access_frequency = await self._predict_access_frequency(key)
        
        # Reglas heurísticas
        if size < self.config.l1_item_max_size and access_frequency > 0.7:
            return CacheLevel.L1
        elif size < self.config.l2_item_max_size and access_frequency > 0.3:
            return CacheLevel.L2
        else:
            return CacheLevel.L3
    
    async def _predict_access_frequency(self, key: CacheKey) -> float:
        """Predecir frecuencia de acceso para una clave."""
        key_str = self._cache_key_to_string(key)
        
        # Contar accesos recientes
        recent_accesses = sum(
            1 for k, t in self.access_history[-1000:]
            if k == key_str and (datetime.now() - t).total_seconds() < 3600
        )
        
        # Normalizar a rango 0-1
        return min(recent_accesses / 10, 1.0)
    
    async def _get_from_l2(self, key: str) -> Optional[Any]:
        """Obtener valor de cache L2 (Redis)."""
        # Implementación simplificada - en producción usaría Redis
        return None
    
    async def _store_in_l2(self, key: str, value: Any, ttl: int) -> bool:
        """Almacenar en cache L2 (Redis)."""
        # Implementación simplificada - en producción usaría Redis
        return False
    
    async def _get_from_l3(self, key: str) -> Optional[Any]:
        """Obtener valor de cache L3 (disco)."""
        try:
            file_path = self._get_l3_file_path(key)
            if not file_path.exists():
                return None
            
            async with aiofiles.open(file_path, 'rb') as f:
                data = pickle.loads(await f.read())
            
            # Verificar expiración
            if data['expiry'] > 0 and time.time() > data['expiry']:
                # Eliminar archivo expirado
                file_path.unlink()
                return None
            
            return data['value']
            
        except Exception:
            return None
    
    def _get_l3_file_path(self, key: str) -> Path:
        """Obtener ruta de archivo para cache L3."""
        # Usar hash para evitar problemas con caracteres especiales
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.l3_cache_dir / f"{key_hash[:2]}" / f"{key_hash}.cache"
    
    async def _invalidate_exact(self, key: str) -> bool:
        """Invalidar clave exacta en todos los niveles."""
        invalidated = False
        
        # L1
        if self.l1_cache.invalidate(key):
            invalidated = True
        
        # L2
        if self.l2_cache and await self._invalidate_l2_key(key):
            invalidated = True
        
        # L3
        file_path = self._get_l3_file_path(key)
        if file_path.exists():
            file_path.unlink()
            invalidated = True
        
        return invalidated
    
    async def _invalidate_pattern(self, pattern: str) -> int:
        """Invalidar por patrón en todos los niveles."""
        total = 0
        
        # L1
        total += self.l1_cache.invalidate_pattern(pattern)
        
        # L2
        if self.l2_cache:
            total += await self._invalidate_l2_pattern(pattern)
        
        # L3
        total += await self._invalidate_l3_pattern(pattern)
        
        return total
    
    async def _invalidate_tag(self, tag: str) -> int:
        """Invalidar por tag."""
        # Implementación simplificada - requeriría tracking de tags
        return 0
    
    async def _invalidate_by_age(self, max_age_seconds: int) -> int:
        """Invalidar items más antiguos que edad especificada."""
        total = 0
        cutoff_time = time.time() - max_age_seconds
        
        # L3 (más fácil de implementar para archivos)
        for cache_file in self.l3_cache_dir.rglob("*.cache"):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
                total += 1
        
        return total
    
    async def _invalidate_l2_key(self, key: str) -> bool:
        """Invalidar clave en L2."""
        # Implementación para Redis
        return False
    
    async def _invalidate_l2_pattern(self, pattern: str) -> int:
        """Invalidar patrón en L2."""
        # Implementación para Redis
        return 0
    
    async def _invalidate_l3_pattern(self, pattern: str) -> int:
        """Invalidar patrón en L3."""
        import re
        regex = re.compile(pattern.replace('*', '.*'))
        
        count = 0
        for cache_file in self.l3_cache_dir.rglob("*.cache"):
            # Reconstruir clave desde archivo
            # (En producción, mantendríamos un índice)
            if regex.match(cache_file.stem):
                cache_file.unlink()
                count += 1
        
        return count
    
    def _analyze_access_patterns(self) -> Dict[str, List[datetime]]:
        """Analizar patrones de acceso."""
        patterns = {}
        for key, timestamp in self.access_history:
            if key not in patterns:
                patterns[key] = []
            patterns[key].append(timestamp)
        return patterns
    
    def _identify_hot_items(
        self,
        access_patterns: Dict[str, List[datetime]]
    ) -> List[str]:
        """Identificar items calientes (frecuentemente accedidos)."""
        now = datetime.now()
        hot_items = []
        
        for key, accesses in access_patterns.items():
            # Contar accesos recientes
            recent_accesses = sum(
                1 for access_time in accesses
                if (now - access_time).total_seconds() < 300  # últimos 5 minutos
            )
            
            if recent_accesses >= self.config.hot_item_threshold:
                hot_items.append(key)
        
        return hot_items
    
    def _identify_cold_items(
        self,
        access_patterns: Dict[str, List[datetime]]
    ) -> List[str]:
        """Identificar items fríos (raramente accedidos)."""
        now = datetime.now()
        cold_items = []
        
        for key, accesses in access_patterns.items():
            if not accesses:
                cold_items.append(key)
                continue
            
            # Verificar último acceso
            last_access = max(accesses)
            if (now - last_access).total_seconds() > self.config.cold_item_threshold:
                cold_items.append(key)
        
        return cold_items
    
    async def _demote_cache_entry(self, key: str) -> bool:
        """Degradar entrada de cache a nivel inferior."""
        # Buscar en L1
        value = self.l1_cache.get(key)
        if value is not None:
            # Mover a L3
            await self.store_in_l3(key, value, self.config.default_cache_ttl)
            self.l1_cache.invalidate(key)
            return True
        
        return False
    
    async def _clean_expired_items(self) -> int:
        """Limpiar items expirados de todos los niveles."""
        count = 0
        
        # L3 - verificar archivos
        for cache_file in self.l3_cache_dir.rglob("*.cache"):
            try:
                async with aiofiles.open(cache_file, 'rb') as f:
                    data = pickle.loads(await f.read())
                
                if data['expiry'] > 0 and time.time() > data['expiry']:
                    cache_file.unlink()
                    count += 1
                    
            except Exception:
                # Archivo corrupto, eliminar
                cache_file.unlink()
                count += 1
        
        return count
    
    async def _calculate_l3_size(self) -> float:
        """Calcular tamaño total del cache L3."""
        total_size = 0
        for cache_file in self.l3_cache_dir.rglob("*.cache"):
            total_size += cache_file.stat().st_size
        return total_size / (1024 * 1024)  # MB
    
    async def _count_l3_items(self) -> int:
        """Contar items en cache L3."""
        return sum(1 for _ in self.l3_cache_dir.rglob("*.cache"))
    
    def _count_matching_items(self, pattern: str) -> int:
        """Contar items que coinciden con patrón."""
        import re
        regex = re.compile(pattern.replace('*', '.*'))
        
        count = 0
        
        # L1
        for key in self.l1_cache.cache.keys():
            if regex.match(key):
                count += 1
        
        # L3
        for cache_file in self.l3_cache_dir.rglob("*.cache"):
            # Simplificado - en producción mantendríamos índice
            count += 1
        
        return count

