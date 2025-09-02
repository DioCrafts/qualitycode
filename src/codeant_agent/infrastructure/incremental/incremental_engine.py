"""
Motor principal del Sistema de Análisis Incremental.

Este módulo implementa el motor central que orquesta todos los componentes
del sistema de análisis incremental.
"""

import asyncio
from typing import List, Optional, Dict, Any, Set, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json

from ...domain.entities.incremental import (
    ChangeSet, GranularChange, DependencyImpact, AnalysisScope,
    CacheKey, CacheEntry, CacheLevel, InvalidationRequest,
    AccessPrediction, ReusableComponents, FileChange,
    IncrementalAnalysisResult, DeltaAnalysisResult,
    CachePerformance, ChangeType, ChangeLocation,
    GranularityLevel, InvalidationStrategy, PredictionSource
)
from ...domain.value_objects.incremental_metrics import (
    IncrementalMetrics, CacheMetrics, ChangeDetectionMetrics,
    AnalysisPerformanceMetrics, WarmingMetrics
)
from ...domain.services.incremental_service import (
    IncrementalAnalysisService, ChangeDetectionService,
    DependencyAnalysisService
)
from ...domain.repositories.incremental_repository import (
    ChangeSetRepository, CacheRepository,
    DependencyGraphRepository, AnalysisResultRepository
)
from ...application.ports.incremental_ports import (
    AnalysisEngineOutputPort, MetricsCollectorOutputPort
)
from .incremental_config import IncrementalConfig


class IncrementalAnalysisEngine(IncrementalAnalysisService):
    """Motor principal de análisis incremental."""
    
    def __init__(
        self,
        config: IncrementalConfig,
        change_detection_service: ChangeDetectionService,
        dependency_analysis_service: DependencyAnalysisService,
        cache_repository: CacheRepository,
        analysis_result_repository: AnalysisResultRepository,
        analysis_engine_output: AnalysisEngineOutputPort,
        metrics_collector: MetricsCollectorOutputPort
    ):
        self.config = config
        self.change_detection_service = change_detection_service
        self.dependency_analysis_service = dependency_analysis_service
        self.cache_repository = cache_repository
        self.analysis_result_repository = analysis_result_repository
        self.analysis_engine_output = analysis_engine_output
        self.metrics_collector = metrics_collector
        
        # Cache interno para optimización
        self._scope_cache: Dict[str, AnalysisScope] = {}
        self._reusability_cache: Dict[str, ReusableComponents] = {}
    
    async def analyze_incremental(
        self,
        change_set: ChangeSet,
        project_context: Dict[str, Any]
    ) -> Dict[Path, Any]:
        """
        Realizar análisis incremental basado en cambios.
        
        Args:
            change_set: Conjunto de cambios detectados
            project_context: Contexto del proyecto
            
        Returns:
            Resultados del análisis por archivo
        """
        start_time = datetime.now()
        results = {}
        
        try:
            # Determinar alcance del análisis
            scope = await self.determine_analysis_scope(change_set)
            
            # Decidir estrategia de análisis
            use_incremental = await self.should_use_incremental(
                change_set, 
                project_context.get("total_files", 0)
            )
            
            if use_incremental:
                # Análisis incremental paralelo
                tasks = []
                for file_path in scope.affected_files:
                    task = self._analyze_file_incremental(
                        file_path, change_set, scope
                    )
                    tasks.append(task)
                
                # Ejecutar análisis en paralelo
                if self.config.max_parallel_analyses > 0:
                    # Limitar concurrencia
                    semaphore = asyncio.Semaphore(self.config.max_parallel_analyses)
                    async def limited_task(task):
                        async with semaphore:
                            return await task
                    
                    tasks = [limited_task(task) for task in tasks]
                
                file_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Procesar resultados
                for file_path, result in zip(scope.affected_files, file_results):
                    if isinstance(result, Exception):
                        # Log error and fallback to full analysis
                        results[file_path] = await self._analyze_file_full(
                            file_path, scope
                        )
                    else:
                        results[file_path] = result
            else:
                # Análisis completo
                for file_path in scope.affected_files:
                    results[file_path] = await self._analyze_file_full(
                        file_path, scope
                    )
            
            # Registrar métricas
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            await self.metrics_collector.record_analysis_time(
                operation="incremental_analysis",
                duration_ms=duration_ms,
                incremental=use_incremental
            )
            
            return results
            
        except Exception as e:
            raise Exception(f"Error in incremental analysis: {str(e)}")
    
    async def determine_analysis_scope(
        self,
        change_set: ChangeSet
    ) -> AnalysisScope:
        """
        Determinar el alcance del análisis necesario.
        
        Args:
            change_set: Conjunto de cambios
            
        Returns:
            Alcance del análisis
        """
        # Check cache first
        cache_key = f"scope_{change_set.id}"
        if cache_key in self._scope_cache:
            return self._scope_cache[cache_key]
        
        affected_files = set()
        affected_functions = set()
        affected_classes = set()
        
        # Procesar cambios directos
        for file_change in change_set.file_changes:
            affected_files.add(Path(file_change.file_path))
            
            for granular_change in file_change.granular_changes:
                if granular_change.change_type in [
                    ChangeType.FUNCTION_MODIFIED,
                    ChangeType.FUNCTION_ADDED,
                    ChangeType.FUNCTION_REMOVED
                ]:
                    affected_functions.add(granular_change.symbol_name)
                elif granular_change.change_type in [
                    ChangeType.CLASS_MODIFIED,
                    ChangeType.CLASS_ADDED,
                    ChangeType.CLASS_REMOVED
                ]:
                    affected_classes.add(granular_change.symbol_name)
        
        # Analizar impacto en dependencias
        dependency_depth = 0
        if self.config.analyze_transitive_dependencies:
            for file_path in list(affected_files):
                dependencies = await self.dependency_analysis_service.calculate_transitive_impact(
                    file_path=file_path,
                    max_depth=self.config.max_dependency_depth
                )
                affected_files.update(dependencies)
                dependency_depth = max(
                    dependency_depth,
                    self._calculate_dependency_depth(file_path, dependencies)
                )
        
        scope = AnalysisScope(
            affected_files=list(affected_files),
            affected_functions=list(affected_functions),
            affected_classes=list(affected_classes),
            dependency_depth=dependency_depth,
            requires_full_analysis=self._requires_full_analysis(change_set),
            priority_files=self._identify_priority_files(affected_files)
        )
        
        # Cache result
        self._scope_cache[cache_key] = scope
        
        return scope
    
    async def identify_reusable_components(
        self,
        file_path: Path,
        change_set: ChangeSet
    ) -> ReusableComponents:
        """
        Identificar componentes reutilizables del cache.
        
        Args:
            file_path: Ruta del archivo
            change_set: Conjunto de cambios
            
        Returns:
            Componentes reutilizables
        """
        cache_key = f"reusable_{file_path}_{change_set.id}"
        if cache_key in self._reusability_cache:
            return self._reusability_cache[cache_key]
        
        # Encontrar cambios específicos del archivo
        file_changes = None
        for fc in change_set.file_changes:
            if Path(fc.file_path) == file_path:
                file_changes = fc
                break
        
        if not file_changes:
            # No hay cambios, todo es reutilizable
            return ReusableComponents(
                file_path=str(file_path),
                unchanged_functions=["*"],
                unchanged_classes=["*"],
                cached_analyses={}
            )
        
        # Identificar funciones y clases sin cambios
        changed_functions = set()
        changed_classes = set()
        
        for change in file_changes.granular_changes:
            if change.symbol_name:
                if change.change_type in [
                    ChangeType.FUNCTION_MODIFIED,
                    ChangeType.FUNCTION_ADDED,
                    ChangeType.FUNCTION_REMOVED
                ]:
                    changed_functions.add(change.symbol_name)
                elif change.change_type in [
                    ChangeType.CLASS_MODIFIED,
                    ChangeType.CLASS_ADDED,
                    ChangeType.CLASS_REMOVED
                ]:
                    changed_classes.add(change.symbol_name)
        
        # Obtener lista de todas las funciones y clases del archivo
        # (Esto requeriría parsing del archivo, simplificado aquí)
        all_functions = await self._get_all_functions(file_path)
        all_classes = await self._get_all_classes(file_path)
        
        unchanged_functions = [f for f in all_functions if f not in changed_functions]
        unchanged_classes = [c for c in all_classes if c not in changed_classes]
        
        # Buscar análisis cacheados
        cached_analyses = {}
        for analysis_type in ["complexity", "security", "quality"]:
            cache_key = CacheKey(
                path=str(file_path),
                analysis_type=analysis_type,
                hash=await self._compute_file_hash(file_path)
            )
            cached_result = await self.cache_repository.get_cached_item(cache_key)
            if cached_result:
                cached_analyses[analysis_type] = cached_result
        
        result = ReusableComponents(
            file_path=str(file_path),
            unchanged_functions=unchanged_functions,
            unchanged_classes=unchanged_classes,
            cached_analyses=cached_analyses
        )
        
        # Cache result
        self._reusability_cache[cache_key] = result
        
        return result
    
    async def perform_delta_analysis(
        self,
        cached_result: Any,
        changes: List[GranularChange]
    ) -> Any:
        """
        Realizar análisis delta sobre resultado cacheado.
        
        Args:
            cached_result: Resultado previamente cacheado
            changes: Cambios granulares a aplicar
            
        Returns:
            Resultado actualizado
        """
        start_time = datetime.now()
        
        # Clonar resultado para no modificar el original
        result = self._deep_copy(cached_result)
        
        # Aplicar cambios según tipo
        for change in changes:
            if change.change_type == ChangeType.FUNCTION_ADDED:
                # Analizar nueva función
                new_analysis = await self._analyze_function(
                    change.symbol_name,
                    change.new_content
                )
                self._merge_function_analysis(result, new_analysis)
                
            elif change.change_type == ChangeType.FUNCTION_REMOVED:
                # Eliminar análisis de función
                self._remove_function_analysis(result, change.symbol_name)
                
            elif change.change_type == ChangeType.FUNCTION_MODIFIED:
                # Re-analizar función modificada
                updated_analysis = await self._analyze_function(
                    change.symbol_name,
                    change.new_content
                )
                self._update_function_analysis(result, updated_analysis)
                
            elif change.change_type in [
                ChangeType.CLASS_ADDED,
                ChangeType.CLASS_REMOVED,
                ChangeType.CLASS_MODIFIED
            ]:
                # Manejar cambios en clases
                await self._handle_class_change(result, change)
        
        # Recalcular métricas agregadas
        self._recalculate_aggregate_metrics(result)
        
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Registrar el análisis delta
        delta_result = DeltaAnalysisResult(
            base_result=cached_result,
            changes_applied=changes,
            final_result=result,
            computation_time_ms=duration_ms
        )
        
        return result
    
    async def should_use_incremental(
        self,
        change_set: ChangeSet,
        project_size: int
    ) -> bool:
        """
        Determinar si usar análisis incremental o completo.
        
        Args:
            change_set: Conjunto de cambios
            project_size: Tamaño del proyecto en archivos
            
        Returns:
            True si se debe usar análisis incremental
        """
        # Calcular ratio de cambios
        total_changes = sum(
            len(fc.granular_changes) for fc in change_set.file_changes
        )
        change_ratio = len(change_set.file_changes) / max(project_size, 1)
        
        # Criterios para usar análisis incremental
        criteria = [
            # Ratio de cambios bajo
            change_ratio < self.config.incremental_threshold,
            # Número total de cambios manejable
            total_changes < self.config.max_changes_for_incremental,
            # Proyecto suficientemente grande
            project_size > self.config.min_project_size_for_incremental,
            # No hay cambios estructurales mayores
            not self._has_major_structural_changes(change_set),
            # Cache disponible
            await self._has_sufficient_cache_coverage(change_set)
        ]
        
        # Usar incremental si se cumplen la mayoría de criterios
        return sum(criteria) >= 3
    
    # Métodos auxiliares privados
    
    async def _analyze_file_incremental(
        self,
        file_path: Path,
        change_set: ChangeSet,
        scope: AnalysisScope
    ) -> Any:
        """Análisis incremental de un archivo."""
        # Buscar resultado cacheado
        cache_key = CacheKey(
            path=str(file_path),
            analysis_type="full",
            hash=await self._compute_file_hash(file_path)
        )
        
        cached_result = await self.cache_repository.get_cached_item(cache_key)
        
        if cached_result:
            # Identificar componentes reutilizables
            reusable = await self.identify_reusable_components(
                file_path, change_set
            )
            
            # Encontrar cambios del archivo
            file_changes = None
            for fc in change_set.file_changes:
                if Path(fc.file_path) == file_path:
                    file_changes = fc
                    break
            
            if file_changes and file_changes.granular_changes:
                # Aplicar análisis delta
                return await self.perform_delta_analysis(
                    cached_result,
                    file_changes.granular_changes
                )
            else:
                # No hay cambios, usar resultado cacheado
                return cached_result
        else:
            # No hay cache, análisis completo
            return await self._analyze_file_full(file_path, scope)
    
    async def _analyze_file_full(
        self,
        file_path: Path,
        scope: AnalysisScope
    ) -> Any:
        """Análisis completo de un archivo."""
        result = await self.analysis_engine_output.run_partial_analysis(
            file_path=file_path,
            analysis_type="full",
            scope=scope
        )
        
        # Cachear resultado
        cache_key = CacheKey(
            path=str(file_path),
            analysis_type="full",
            hash=await self._compute_file_hash(file_path)
        )
        
        await self.cache_repository.save_cached_item(
            key=cache_key,
            item=result,
            ttl=self.config.default_cache_ttl
        )
        
        return result
    
    def _calculate_dependency_depth(
        self,
        file_path: Path,
        dependencies: Set[Path]
    ) -> int:
        """Calcular profundidad de dependencias."""
        # Implementación simplificada
        return min(len(dependencies), self.config.max_dependency_depth)
    
    def _requires_full_analysis(self, change_set: ChangeSet) -> bool:
        """Determinar si se requiere análisis completo."""
        for file_change in change_set.file_changes:
            for change in file_change.granular_changes:
                if change.change_type in [
                    ChangeType.FILE_ADDED,
                    ChangeType.FILE_REMOVED,
                    ChangeType.MAJOR_REFACTORING
                ]:
                    return True
        return False
    
    def _identify_priority_files(
        self,
        affected_files: Set[Path]
    ) -> List[Path]:
        """Identificar archivos prioritarios para análisis."""
        # Implementación simplificada: archivos más grandes primero
        return sorted(
            affected_files,
            key=lambda p: p.stat().st_size if p.exists() else 0,
            reverse=True
        )[:10]
    
    async def _get_all_functions(self, file_path: Path) -> List[str]:
        """Obtener lista de todas las funciones en un archivo."""
        # Implementación simplificada
        return []
    
    async def _get_all_classes(self, file_path: Path) -> List[str]:
        """Obtener lista de todas las clases en un archivo."""
        # Implementación simplificada
        return []
    
    async def _compute_file_hash(self, file_path: Path) -> str:
        """Computar hash del contenido del archivo."""
        if not file_path.exists():
            return ""
        
        hasher = hashlib.blake2b()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _deep_copy(self, obj: Any) -> Any:
        """Copia profunda de un objeto."""
        import copy
        return copy.deepcopy(obj)
    
    async def _analyze_function(
        self,
        function_name: str,
        content: str
    ) -> Dict[str, Any]:
        """Analizar una función específica."""
        # Implementación simplificada
        return {
            "function": function_name,
            "complexity": 1,
            "lines": len(content.split('\n'))
        }
    
    def _merge_function_analysis(
        self,
        result: Dict[str, Any],
        new_analysis: Dict[str, Any]
    ) -> None:
        """Fusionar análisis de nueva función."""
        if "functions" not in result:
            result["functions"] = {}
        result["functions"][new_analysis["function"]] = new_analysis
    
    def _remove_function_analysis(
        self,
        result: Dict[str, Any],
        function_name: str
    ) -> None:
        """Eliminar análisis de función."""
        if "functions" in result and function_name in result["functions"]:
            del result["functions"][function_name]
    
    def _update_function_analysis(
        self,
        result: Dict[str, Any],
        updated_analysis: Dict[str, Any]
    ) -> None:
        """Actualizar análisis de función."""
        if "functions" not in result:
            result["functions"] = {}
        result["functions"][updated_analysis["function"]] = updated_analysis
    
    async def _handle_class_change(
        self,
        result: Dict[str, Any],
        change: GranularChange
    ) -> None:
        """Manejar cambios en clases."""
        # Implementación simplificada
        pass
    
    def _recalculate_aggregate_metrics(
        self,
        result: Dict[str, Any]
    ) -> None:
        """Recalcular métricas agregadas."""
        if "functions" in result:
            total_complexity = sum(
                f.get("complexity", 0) for f in result["functions"].values()
            )
            result["total_complexity"] = total_complexity
    
    def _has_major_structural_changes(
        self,
        change_set: ChangeSet
    ) -> bool:
        """Verificar si hay cambios estructurales mayores."""
        for file_change in change_set.file_changes:
            if file_change.change_type in [
                ChangeType.FILE_ADDED,
                ChangeType.FILE_REMOVED,
                ChangeType.MAJOR_REFACTORING
            ]:
                return True
        return False
    
    async def _has_sufficient_cache_coverage(
        self,
        change_set: ChangeSet
    ) -> bool:
        """Verificar si hay suficiente cobertura de cache."""
        total_files = len(change_set.file_changes)
        cached_files = 0
        
        for file_change in change_set.file_changes:
            cache_key = CacheKey(
                path=file_change.file_path,
                analysis_type="full",
                hash=None
            )
            if await self.cache_repository.get_cached_item(cache_key):
                cached_files += 1
        
        coverage = cached_files / max(total_files, 1)
        return coverage >= self.config.min_cache_coverage_for_incremental
