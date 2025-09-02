"""
Casos de uso para el Sistema de Análisis Incremental.

Este módulo implementa los casos de uso que orquestan las operaciones
del sistema de análisis incremental.
"""

from typing import List, Optional, Dict, Any, Set
from pathlib import Path
from datetime import datetime, timedelta

from ..ports.incremental_ports import (
    ChangeDetectionInputPort, CacheManagementInputPort,
    IncrementalAnalysisInputPort, PredictiveCacheInputPort,
    DependencyTrackingInputPort, ChangeDetectionOutputPort,
    CacheStorageOutputPort, DependencyGraphOutputPort,
    AnalysisEngineOutputPort, MetricsCollectorOutputPort,
    PredictionEngineOutputPort, NotificationOutputPort
)
from ..dtos.incremental_dtos import (
    DetectChangesRequestDTO, DetectChangesResponseDTO,
    AnalyzeIncrementalRequestDTO, AnalyzeIncrementalResponseDTO,
    GetCachedAnalysisRequestDTO, GetCachedAnalysisResponseDTO,
    InvalidateCacheRequestDTO, InvalidateCacheResponseDTO,
    PredictCacheNeedsRequestDTO, PredictCacheNeedsResponseDTO,
    WarmCacheRequestDTO, WarmCacheResponseDTO,
    GetDependenciesRequestDTO, GetDependenciesResponseDTO,
    GetCacheMetricsRequestDTO, GetCacheMetricsResponseDTO,
    OptimizeCacheRequestDTO, OptimizeCacheResponseDTO
)
from ...domain.entities.incremental import (
    ChangeSet, GranularChange, DependencyImpact,
    AnalysisScope, CacheKey, CacheLevel,
    AccessPrediction, IncrementalAnalysisResult
)
from ...domain.services.incremental_service import (
    ChangeDetectionService, DependencyAnalysisService,
    CacheManagementService, IncrementalAnalysisService,
    PredictiveCacheService, InvalidationService,
    DeltaProcessingService, PerformanceOptimizationService
)
from ...domain.repositories.incremental_repository import (
    ChangeSetRepository, CacheRepository,
    DependencyGraphRepository, AnalysisResultRepository
)


class DetectChangesUseCase:
    """Caso de uso para detectar cambios en el código."""
    
    def __init__(
        self,
        change_detection_service: ChangeDetectionService,
        change_detection_output: ChangeDetectionOutputPort,
        change_set_repository: ChangeSetRepository,
        metrics_collector: MetricsCollectorOutputPort
    ):
        self.change_detection_service = change_detection_service
        self.change_detection_output = change_detection_output
        self.change_set_repository = change_set_repository
        self.metrics_collector = metrics_collector
    
    async def execute(
        self, 
        request: DetectChangesRequestDTO
    ) -> DetectChangesResponseDTO:
        """Ejecutar detección de cambios."""
        start_time = datetime.now()
        
        try:
            # Detectar cambios en el repositorio
            change_set = await self.change_detection_service.detect_changes(
                repository_path=Path(request.repository_path),
                from_commit=request.from_commit,
                to_commit=request.to_commit
            )
            
            # Analizar cambios granulares si está habilitado
            if request.enable_granular_detection:
                for file_change in change_set.file_changes:
                    granular_changes = await self.change_detection_service.detect_granular_changes(
                        file_change=file_change,
                        granularity=request.granularity_level
                    )
                    file_change.granular_changes = granular_changes
            
            # Calcular impacto si está habilitado
            if request.calculate_impact:
                impact = await self.change_detection_service.calculate_change_impact(
                    changes=[gc for fc in change_set.file_changes for gc in fc.granular_changes]
                )
                change_set.total_impact = impact
            
            # Guardar change set
            await self.change_set_repository.save_change_set(change_set)
            
            # Registrar métricas
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            await self.metrics_collector.record_analysis_time(
                operation="detect_changes",
                duration_ms=duration_ms,
                incremental=False
            )
            
            return DetectChangesResponseDTO(
                change_set_id=change_set.id,
                total_files_changed=len(change_set.file_changes),
                total_changes=sum(len(fc.granular_changes) for fc in change_set.file_changes),
                impact_score=change_set.total_impact,
                detection_time_ms=duration_ms
            )
            
        except Exception as e:
            raise Exception(f"Error detecting changes: {str(e)}")


class AnalyzeIncrementalUseCase:
    """Caso de uso para análisis incremental."""
    
    def __init__(
        self,
        incremental_analysis_service: IncrementalAnalysisService,
        cache_management_service: CacheManagementService,
        dependency_analysis_service: DependencyAnalysisService,
        analysis_engine_output: AnalysisEngineOutputPort,
        cache_storage_output: CacheStorageOutputPort,
        analysis_result_repository: AnalysisResultRepository,
        metrics_collector: MetricsCollectorOutputPort,
        notification_output: NotificationOutputPort
    ):
        self.incremental_analysis_service = incremental_analysis_service
        self.cache_management_service = cache_management_service
        self.dependency_analysis_service = dependency_analysis_service
        self.analysis_engine_output = analysis_engine_output
        self.cache_storage_output = cache_storage_output
        self.analysis_result_repository = analysis_result_repository
        self.metrics_collector = metrics_collector
        self.notification_output = notification_output
    
    async def execute(
        self,
        request: AnalyzeIncrementalRequestDTO
    ) -> AnalyzeIncrementalResponseDTO:
        """Ejecutar análisis incremental."""
        start_time = datetime.now()
        cache_hits = 0
        cache_misses = 0
        
        try:
            # Obtener change set
            change_set = await self._get_change_set(request.change_set_id)
            
            # Determinar alcance del análisis
            scope = await self.incremental_analysis_service.determine_analysis_scope(
                change_set=change_set
            )
            
            # Verificar si usar análisis incremental
            use_incremental = await self.incremental_analysis_service.should_use_incremental(
                change_set=change_set,
                project_size=request.project_size
            )
            
            analysis_results = {}
            
            if use_incremental:
                # Análisis incremental
                for file_path in scope.affected_files:
                    # Buscar resultados en cache
                    cached_result = await self.cache_management_service.get_cached_item(
                        key=CacheKey(
                            path=str(file_path),
                            analysis_type="full",
                            hash=None  # Se calculará internamente
                        )
                    )
                    
                    if cached_result:
                        cache_hits += 1
                        # Identificar componentes reutilizables
                        reusable = await self.incremental_analysis_service.identify_reusable_components(
                            file_path=file_path,
                            change_set=change_set
                        )
                        
                        # Realizar análisis delta
                        file_changes = [fc for fc in change_set.file_changes if fc.file_path == file_path]
                        if file_changes:
                            granular_changes = file_changes[0].granular_changes
                            delta_result = await self.incremental_analysis_service.perform_delta_analysis(
                                cached_result=cached_result,
                                changes=granular_changes
                            )
                            analysis_results[file_path] = delta_result
                        else:
                            analysis_results[file_path] = cached_result
                    else:
                        cache_misses += 1
                        # Análisis completo del archivo
                        result = await self.analysis_engine_output.run_partial_analysis(
                            file_path=file_path,
                            analysis_type="full",
                            scope=scope
                        )
                        analysis_results[file_path] = result
                        
                        # Cachear resultado
                        await self.cache_storage_output.store_in_l1(
                            key=str(file_path),
                            value=result,
                            ttl=3600  # 1 hora
                        )
            else:
                # Análisis completo
                for file_path in scope.affected_files:
                    cache_misses += 1
                    result = await self.analysis_engine_output.run_partial_analysis(
                        file_path=file_path,
                        analysis_type="full",
                        scope=scope
                    )
                    analysis_results[file_path] = result
            
            # Crear resultado del análisis incremental
            result = IncrementalAnalysisResult(
                id=f"inc_{change_set.id}_{datetime.now().timestamp()}",
                change_set_id=change_set.id,
                affected_files=list(analysis_results.keys()),
                analysis_results=analysis_results,
                cache_performance=CachePerformance(
                    hits=cache_hits,
                    misses=cache_misses,
                    hit_rate=cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0,
                    bytes_saved=0  # Se calculará basado en los resultados
                ),
                reusable_components=ReusableComponents(
                    file_path="",  # Agregado
                    unchanged_functions=[],
                    unchanged_classes=[],
                    cached_analyses={}
                ),
                delta_results=DeltaAnalysisResult(
                    base_result={},
                    changes_applied=[],
                    final_result=analysis_results,
                    computation_time_ms=0
                ),
                total_time_ms=0,
                incremental_speedup=0,
                timestamp=datetime.now()
            )
            
            # Guardar resultado
            await self.analysis_result_repository.save_analysis_result(result)
            
            # Registrar métricas
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            result.total_time_ms = duration_ms
            
            await self.metrics_collector.record_analysis_time(
                operation="incremental_analysis",
                duration_ms=duration_ms,
                incremental=use_incremental
            )
            
            # Notificar finalización
            await self.notification_output.notify_analysis_complete(result)
            
            return AnalyzeIncrementalResponseDTO(
                analysis_id=result.id,
                files_analyzed=len(analysis_results),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                total_time_ms=duration_ms,
                incremental_speedup=result.incremental_speedup
            )
            
        except Exception as e:
            raise Exception(f"Error in incremental analysis: {str(e)}")
    
    async def _get_change_set(self, change_set_id: str) -> ChangeSet:
        """Obtener change set por ID."""
        # Implementar lógica para obtener change set
        return await self.change_set_repository.get_change_set_by_id(change_set_id)


class GetCachedAnalysisUseCase:
    """Caso de uso para obtener análisis cacheado."""
    
    def __init__(
        self,
        cache_management_service: CacheManagementService,
        cache_storage_output: CacheStorageOutputPort,
        metrics_collector: MetricsCollectorOutputPort
    ):
        self.cache_management_service = cache_management_service
        self.cache_storage_output = cache_storage_output
        self.metrics_collector = metrics_collector
    
    async def execute(
        self,
        request: GetCachedAnalysisRequestDTO
    ) -> GetCachedAnalysisResponseDTO:
        """Obtener análisis del cache."""
        start_time = datetime.now()
        
        try:
            # Crear clave de cache
            cache_key = CacheKey(
                path=request.file_path,
                analysis_type=request.analysis_type,
                hash=request.content_hash
            )
            
            # Buscar en cache
            cached_result = await self.cache_management_service.get_cached_item(
                key=cache_key
            )
            
            if cached_result:
                # Cache hit
                await self.metrics_collector.record_cache_hit(
                    cache_level=CacheLevel.L1,  # Por defecto
                    key=str(cache_key)
                )
                
                return GetCachedAnalysisResponseDTO(
                    found=True,
                    result=cached_result,
                    cache_level="L1",  # Simplificado
                    retrieval_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
                )
            else:
                # Cache miss
                await self.metrics_collector.record_cache_miss(
                    key=str(cache_key)
                )
                
                return GetCachedAnalysisResponseDTO(
                    found=False,
                    result=None,
                    cache_level=None,
                    retrieval_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
                )
                
        except Exception as e:
            raise Exception(f"Error retrieving cached analysis: {str(e)}")


class InvalidateCacheUseCase:
    """Caso de uso para invalidar cache."""
    
    def __init__(
        self,
        invalidation_service: InvalidationService,
        cache_storage_output: CacheStorageOutputPort,
        dependency_analysis_service: DependencyAnalysisService,
        metrics_collector: MetricsCollectorOutputPort,
        notification_output: NotificationOutputPort
    ):
        self.invalidation_service = invalidation_service
        self.cache_storage_output = cache_storage_output
        self.dependency_analysis_service = dependency_analysis_service
        self.metrics_collector = metrics_collector
        self.notification_output = notification_output
    
    async def execute(
        self,
        request: InvalidateCacheRequestDTO
    ) -> InvalidateCacheResponseDTO:
        """Invalidar entradas del cache."""
        start_time = datetime.now()
        
        try:
            affected_files = []
            total_invalidated = 0
            
            # Invalidar por patrón
            if request.pattern:
                levels = [CacheLevel[level] for level in request.cache_levels]
                invalidated = await self.cache_storage_output.invalidate_by_pattern(
                    pattern=request.pattern,
                    levels=levels
                )
                total_invalidated += invalidated
            
            # Invalidar dependencias si está habilitado
            if request.invalidate_dependencies and request.file_paths:
                for file_path in request.file_paths:
                    path = Path(file_path)
                    affected_files.append(path)
                    
                    # Obtener dependencias
                    dependencies = await self.dependency_analysis_service.calculate_transitive_impact(
                        file_path=path,
                        max_depth=request.max_dependency_depth
                    )
                    
                    # Invalidar cada dependencia
                    for dep in dependencies:
                        invalidated = await self.invalidation_service.invalidate_dependencies(
                            file_path=dep
                        )
                        total_invalidated += invalidated
                        affected_files.append(dep)
            
            # Registrar métricas
            await self.metrics_collector.record_invalidation_event(
                pattern=request.pattern or "manual",
                affected_count=total_invalidated
            )
            
            # Notificar invalidación
            if affected_files:
                await self.notification_output.notify_cache_invalidation(
                    affected_files=affected_files,
                    reason=request.reason or "Manual invalidation"
                )
            
            return InvalidateCacheResponseDTO(
                invalidated_count=total_invalidated,
                affected_files=[str(f) for f in affected_files],
                execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
            
        except Exception as e:
            raise Exception(f"Error invalidating cache: {str(e)}")


class PredictCacheNeedsUseCase:
    """Caso de uso para predecir necesidades de cache."""
    
    def __init__(
        self,
        predictive_cache_service: PredictiveCacheService,
        prediction_engine_output: PredictionEngineOutputPort,
        metrics_collector: MetricsCollectorOutputPort
    ):
        self.predictive_cache_service = predictive_cache_service
        self.prediction_engine_output = prediction_engine_output
        self.metrics_collector = metrics_collector
    
    async def execute(
        self,
        request: PredictCacheNeedsRequestDTO
    ) -> PredictCacheNeedsResponseDTO:
        """Predecir necesidades futuras de cache."""
        start_time = datetime.now()
        
        try:
            # Obtener patrones históricos
            historical_patterns = await self.prediction_engine_output.get_historical_access_patterns(
                time_window=timedelta(hours=request.lookback_hours)
            )
            
            # Generar predicciones
            predictions = await self.predictive_cache_service.predict_future_accesses(
                time_horizon=timedelta(minutes=request.prediction_window_minutes)
            )
            
            # Filtrar por umbral de confianza
            filtered_predictions = [
                p for p in predictions
                if p.confidence >= request.confidence_threshold
            ]
            
            # Evaluar precisión de predicciones anteriores
            accuracy = await self.predictive_cache_service.evaluate_prediction_accuracy(
                time_window=timedelta(hours=1)
            )
            
            return PredictCacheNeedsResponseDTO(
                predictions=[
                    {
                        "file_path": p.file_path,
                        "analysis_type": p.analysis_type,
                        "predicted_access_time": p.predicted_access_time.isoformat(),
                        "confidence": p.confidence,
                        "source": p.source.value
                    }
                    for p in filtered_predictions
                ],
                total_predictions=len(filtered_predictions),
                average_confidence=sum(p.confidence for p in filtered_predictions) / len(filtered_predictions) if filtered_predictions else 0,
                prediction_accuracy=accuracy,
                generation_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
            
        except Exception as e:
            raise Exception(f"Error predicting cache needs: {str(e)}")


class WarmCacheUseCase:
    """Caso de uso para calentar cache predictivamente."""
    
    def __init__(
        self,
        predictive_cache_service: PredictiveCacheService,
        analysis_engine_output: AnalysisEngineOutputPort,
        cache_storage_output: CacheStorageOutputPort,
        metrics_collector: MetricsCollectorOutputPort
    ):
        self.predictive_cache_service = predictive_cache_service
        self.analysis_engine_output = analysis_engine_output
        self.cache_storage_output = cache_storage_output
        self.metrics_collector = metrics_collector
    
    async def execute(
        self,
        request: WarmCacheRequestDTO
    ) -> WarmCacheResponseDTO:
        """Calentar cache basado en predicciones."""
        start_time = datetime.now()
        
        try:
            # Obtener predicciones si no se proporcionan
            if not request.predictions:
                predictions = await self.predictive_cache_service.predict_future_accesses(
                    time_horizon=timedelta(minutes=30)
                )
            else:
                # Convertir predicciones del request a objetos del dominio
                predictions = [
                    AccessPrediction(
                        file_path=p["file_path"],
                        analysis_type=p["analysis_type"],
                        predicted_access_time=datetime.fromisoformat(p["predicted_access_time"]),
                        confidence=p["confidence"],
                        source=PredictionSource.TIME_PATTERN  # Simplificado
                    )
                    for p in request.predictions
                ]
            
            # Limitar número de items si se especifica
            if request.max_items:
                predictions = predictions[:request.max_items]
            
            # Calentar cache
            warmed_count = await self.predictive_cache_service.warm_cache(
                predictions=predictions
            )
            
            # Obtener métricas de warming
            warming_metrics = await self.predictive_cache_service.analyze_access_patterns(
                time_window=timedelta(hours=1)
            )
            
            return WarmCacheResponseDTO(
                warmed_items=warmed_count,
                total_predictions=len(predictions),
                cache_size_mb=warming_metrics.get("cache_size_mb", 0),
                warming_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
            
        except Exception as e:
            raise Exception(f"Error warming cache: {str(e)}")


class GetDependenciesUseCase:
    """Caso de uso para obtener dependencias."""
    
    def __init__(
        self,
        dependency_analysis_service: DependencyAnalysisService,
        dependency_graph_output: DependencyGraphOutputPort
    ):
        self.dependency_analysis_service = dependency_analysis_service
        self.dependency_graph_output = dependency_graph_output
    
    async def execute(
        self,
        request: GetDependenciesRequestDTO
    ) -> GetDependenciesResponseDTO:
        """Obtener dependencias de archivos."""
        try:
            file_path = Path(request.file_path)
            
            # Obtener dependencias directas
            direct_deps = await self.dependency_analysis_service.analyze_dependencies(
                file_path=file_path
            )
            
            # Obtener dependencias transitivas si se solicita
            transitive_deps = set()
            if request.include_transitive:
                transitive_deps = await self.dependency_analysis_service.calculate_transitive_impact(
                    file_path=file_path,
                    max_depth=request.max_depth
                )
            
            # Obtener archivos dependientes si se solicita
            dependent_files = set()
            if request.include_dependents:
                dep_graph = await self.dependency_graph_output.query_dependencies(
                    file_path=file_path,
                    direction="dependents"
                )
                dependent_files = dep_graph.get("dependents", set())
            
            return GetDependenciesResponseDTO(
                file_path=str(file_path),
                direct_dependencies=[str(d) for d in direct_deps],
                transitive_dependencies=[str(d) for d in transitive_deps],
                dependent_files=[str(d) for d in dependent_files],
                total_dependencies=len(direct_deps) + len(transitive_deps)
            )
            
        except Exception as e:
            raise Exception(f"Error getting dependencies: {str(e)}")


class GetCacheMetricsUseCase:
    """Caso de uso para obtener métricas del cache."""
    
    def __init__(
        self,
        cache_management_service: CacheManagementService,
        metrics_collector: MetricsCollectorOutputPort
    ):
        self.cache_management_service = cache_management_service
        self.metrics_collector = metrics_collector
    
    async def execute(
        self,
        request: GetCacheMetricsRequestDTO
    ) -> GetCacheMetricsResponseDTO:
        """Obtener métricas del sistema de cache."""
        try:
            # Obtener métricas del servicio
            cache_metrics = await self.cache_management_service.get_cache_metrics()
            
            # Obtener métricas históricas
            time_window = timedelta(hours=request.time_window_hours)
            historical_metrics = await self.metrics_collector.get_metrics_summary(
                time_window=time_window
            )
            
            return GetCacheMetricsResponseDTO(
                hit_rate=cache_metrics.hit_rate,
                total_hits=cache_metrics.total_hits,
                total_misses=cache_metrics.total_misses,
                cache_size_mb=cache_metrics.total_size_mb,
                eviction_count=cache_metrics.eviction_count,
                average_retrieval_time_ms=cache_metrics.average_retrieval_time_ms,
                cache_levels={
                    "L1": {
                        "size_mb": cache_metrics.l1_size_mb,
                        "items": cache_metrics.l1_items,
                        "hit_rate": cache_metrics.l1_hit_rate
                    },
                    "L2": {
                        "size_mb": cache_metrics.l2_size_mb,
                        "items": cache_metrics.l2_items,
                        "hit_rate": cache_metrics.l2_hit_rate
                    },
                    "L3": {
                        "size_mb": cache_metrics.l3_size_mb,
                        "items": cache_metrics.l3_items,
                        "hit_rate": cache_metrics.l3_hit_rate
                    }
                }
            )
            
        except Exception as e:
            raise Exception(f"Error getting cache metrics: {str(e)}")


class OptimizeCacheUseCase:
    """Caso de uso para optimizar el cache."""
    
    def __init__(
        self,
        cache_management_service: CacheManagementService,
        performance_optimization_service: PerformanceOptimizationService,
        predictive_cache_service: PredictiveCacheService,
        notification_output: NotificationOutputPort
    ):
        self.cache_management_service = cache_management_service
        self.performance_optimization_service = performance_optimization_service
        self.predictive_cache_service = predictive_cache_service
        self.notification_output = notification_output
    
    async def execute(
        self,
        request: OptimizeCacheRequestDTO
    ) -> OptimizeCacheResponseDTO:
        """Optimizar el sistema de cache."""
        start_time = datetime.now()
        
        try:
            optimization_results = {}
            
            # Optimizar distribución de cache
            if request.optimize_distribution:
                distribution_result = await self.cache_management_service.optimize_cache_distribution()
                optimization_results["distribution"] = distribution_result
            
            # Optimizar estrategia de warming
            if request.optimize_warming_strategy:
                warming_result = await self.predictive_cache_service.optimize_warming_strategy()
                optimization_results["warming"] = warming_result
            
            # Analizar cuellos de botella
            if request.analyze_bottlenecks:
                bottlenecks = await self.performance_optimization_service.analyze_performance_bottlenecks()
                optimization_results["bottlenecks"] = bottlenecks
            
            # Obtener recomendaciones
            recommendations = await self.performance_optimization_service.get_optimization_recommendations()
            
            # Notificar si hay degradación de rendimiento
            if any(b.get("severity") == "high" for b in optimization_results.get("bottlenecks", [])):
                await self.notification_output.notify_performance_degradation(
                    metrics=optimization_results
                )
            
            return OptimizeCacheResponseDTO(
                optimizations_performed=list(optimization_results.keys()),
                performance_improvement=optimization_results.get("distribution", {}).get("improvement_percent", 0),
                recommendations=recommendations,
                execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
            
        except Exception as e:
            raise Exception(f"Error optimizing cache: {str(e)}")
