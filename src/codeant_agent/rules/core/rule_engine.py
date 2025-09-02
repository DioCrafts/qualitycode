"""
Motor principal de reglas estáticas.

Este módulo implementa el core del motor de reglas estáticas, proporcionando
la funcionalidad principal para ejecutar análisis de código con reglas
configurables y optimizadas.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from ...parsers.unified.unified_ast import UnifiedAST
from ...parsers.universal import ProgrammingLanguage
from ..models.rule_models import (
    Rule,
    RuleId,
    RuleCategory,
    RuleSeverity,
    AnalysisResult,
    ProjectAnalysisResult,
    ProjectConfig,
    ProjectMetrics,
    QualityGates,
)
from ..models.config_models import RulesEngineConfig
from .rule_registry import RuleRegistry
from .rule_executor import RuleExecutor
from .rule_cache import RuleCache
from .performance_optimizer import PerformanceOptimizer
from .result_aggregator import ResultAggregator
from .configuration_manager import ConfigurationManager

logger = logging.getLogger(__name__)


class RulesEngineError(Exception):
    """Excepción base para errores del motor de reglas."""
    pass


class AnalysisError(RulesEngineError):
    """Error durante el análisis de código."""
    pass


class ConfigurationError(RulesEngineError):
    """Error de configuración."""
    pass


@dataclass
class AnalysisContext:
    """Contexto para el análisis de código."""
    ast: UnifiedAST
    project_config: ProjectConfig
    rule_results: List[Any] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    cache_hits: int = 0
    execution_time_ms: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class RulesEngine:
    """
    Motor principal de reglas estáticas.
    
    Este motor proporciona la funcionalidad central para ejecutar análisis
    de código utilizando reglas configurables y optimizadas.
    """
    
    def __init__(self, config: Optional[RulesEngineConfig] = None):
        """Inicializar el motor de reglas."""
        self.config = config or RulesEngineConfig()
        
        # Componentes del motor
        self.rule_registry = RuleRegistry()
        self.rule_cache = RuleCache(self.config.cache_config)
        self.performance_optimizer = PerformanceOptimizer()
        self.result_aggregator = ResultAggregator()
        self.config_manager = ConfigurationManager()
        
        # Ejecutor de reglas
        self.rule_executor = RuleExecutor(
            rule_registry=self.rule_registry,
            rule_cache=self.rule_cache,
            performance_optimizer=self.performance_optimizer,
            config=self.config.executor_config
        )
        
        # Estado del motor
        self.is_initialized = False
        self.analysis_count = 0
        self.total_execution_time = 0.0
        
        logger.info("RulesEngine initialized with configuration")
    
    async def initialize(self) -> None:
        """Inicializar el motor de reglas."""
        if self.is_initialized:
            return
        
        try:
            # Inicializar componentes
            await self.rule_cache.initialize()
            await self.performance_optimizer.initialize()
            await self.result_aggregator.initialize()
            await self.config_manager.initialize()
            
            # Cargar reglas por defecto
            await self.load_default_rules()
            
            self.is_initialized = True
            logger.info("RulesEngine successfully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize RulesEngine: {e}")
            raise RulesEngineError(f"Initialization failed: {e}")
    
    async def analyze_code(self, unified_ast: UnifiedAST, project_config: Optional[ProjectConfig] = None) -> AnalysisResult:
        """
        Analizar código usando el AST unificado.
        
        Args:
            unified_ast: AST unificado del código a analizar
            project_config: Configuración del proyecto (opcional)
            
        Returns:
            Resultado del análisis con violaciones y métricas
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        analysis_context = AnalysisContext(
            ast=unified_ast,
            project_config=project_config or ProjectConfig(),
            start_time=datetime.now(timezone.utc)
        )
        
        try:
            # Obtener reglas aplicables
            applicable_rules = await self._get_applicable_rules(unified_ast, analysis_context.project_config)
            
            if not applicable_rules:
                logger.info(f"No applicable rules found for {unified_ast.file_path}")
                return self._create_empty_result(unified_ast)
            
            # Optimizar orden de ejecución
            optimized_rules = await self.performance_optimizer.optimize_execution_order(
                applicable_rules, unified_ast
            )
            
            # Ejecutar reglas
            rule_results = await self.rule_executor.execute_rules(
                optimized_rules, unified_ast, analysis_context.project_config
            )
            
            # Agregar resultados
            aggregated_results = await self.result_aggregator.aggregate_results(
                rule_results, analysis_context.project_config
            )
            
            # Crear resultado final
            analysis_result = self._create_analysis_result(
                unified_ast, aggregated_results, analysis_context, start_time
            )
            
            # Actualizar métricas
            self._update_engine_metrics(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis failed for {unified_ast.file_path}: {e}")
            return self._create_error_result(unified_ast, str(e))
        
        finally:
            analysis_context.end_time = datetime.now(timezone.utc)
            analysis_context.execution_time_ms = (time.time() - start_time) * 1000
    
    async def analyze_project(self, project_path: Path, project_config: Optional[ProjectConfig] = None) -> ProjectAnalysisResult:
        """
        Analizar un proyecto completo.
        
        Args:
            project_path: Ruta del proyecto a analizar
            project_config: Configuración del proyecto (opcional)
            
        Returns:
            Resultado del análisis del proyecto
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Cargar configuración del proyecto
            if project_config is None:
                project_config = await self.config_manager.load_project_config(project_path)
            
            # Descubrir archivos analizables
            files = await self._discover_analyzable_files(project_path, project_config)
            
            if not files:
                logger.info(f"No analyzable files found in {project_path}")
                return self._create_empty_project_result(project_path)
            
            # Analizar archivos en lotes paralelos
            batch_size = project_config.parallel_analysis_batch_size or 10
            file_results = []
            summary_metrics = ProjectMetrics()
            
            for batch in self._chunk_files(files, batch_size):
                batch_results = await self._analyze_file_batch(batch, project_config)
                
                for result in batch_results:
                    if result.success:
                        summary_metrics.aggregate(result.metrics)
                    file_results.append(result)
            
            # Crear resultado del proyecto
            project_result = self._create_project_result(
                project_path, file_results, summary_metrics, start_time
            )
            
            return project_result
            
        except Exception as e:
            logger.error(f"Project analysis failed for {project_path}: {e}")
            return self._create_error_project_result(project_path, str(e))
    
    async def register_rule(self, rule: Rule) -> None:
        """Registrar una nueva regla en el motor."""
        if not self.is_initialized:
            await self.initialize()
        
        await self.rule_registry.register_rule(rule)
        logger.info(f"Registered rule: {rule.id}")
    
    async def unregister_rule(self, rule_id: RuleId) -> None:
        """Desregistrar una regla del motor."""
        if not self.is_initialized:
            await self.initialize()
        
        await self.rule_registry.unregister_rule(rule_id)
        logger.info(f"Unregistered rule: {rule_id}")
    
    async def get_rule(self, rule_id: RuleId) -> Optional[Rule]:
        """Obtener una regla por su ID."""
        if not self.is_initialized:
            await self.initialize()
        
        return await self.rule_registry.get_rule(rule_id)
    
    async def get_rules_by_category(self, category: RuleCategory) -> List[Rule]:
        """Obtener reglas por categoría."""
        if not self.is_initialized:
            await self.initialize()
        
        return await self.rule_registry.get_rules_by_category(category)
    
    async def get_rules_by_language(self, language: ProgrammingLanguage) -> List[Rule]:
        """Obtener reglas por lenguaje de programación."""
        if not self.is_initialized:
            await self.initialize()
        
        return await self.rule_registry.get_rules_by_language(language)
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del motor."""
        cache_stats = await self.rule_cache.get_stats()
        optimizer_stats = await self.performance_optimizer.get_stats()
        
        return {
            'analysis_count': self.analysis_count,
            'total_execution_time_ms': self.total_execution_time,
            'average_execution_time_ms': self.total_execution_time / max(self.analysis_count, 1),
            'cache_stats': cache_stats,
            'optimizer_stats': optimizer_stats,
            'registered_rules': await self.rule_registry.get_rule_count(),
            'is_initialized': self.is_initialized
        }
    
    async def shutdown(self) -> None:
        """Apagar el motor de reglas."""
        try:
            await self.rule_cache.shutdown()
            await self.performance_optimizer.shutdown()
            await self.result_aggregator.shutdown()
            await self.config_manager.shutdown()
            
            self.is_initialized = False
            logger.info("RulesEngine shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _get_applicable_rules(self, ast: UnifiedAST, project_config: ProjectConfig) -> List[Rule]:
        """Obtener reglas aplicables para el análisis."""
        # Obtener reglas por lenguaje
        language = ProgrammingLanguage.from_string(ast.language)
        language_rules = await self.rule_registry.get_rules_by_language(language)
        
        # Filtrar por categorías habilitadas
        if project_config.enabled_categories:
            language_rules = [
                rule for rule in language_rules
                if rule.category in project_config.enabled_categories
            ]
        
        # Filtrar por umbral de severidad
        language_rules = [
            rule for rule in language_rules
            if self._is_severity_above_threshold(rule.severity, project_config.severity_threshold)
        ]
        
        # Aplicar overrides de configuración
        filtered_rules = []
        for rule in language_rules:
            effective_config = await self.config_manager.get_effective_rule_config(
                rule, project_config.project_path
            )
            
            if effective_config.enabled:
                filtered_rules.append(rule)
        
        return filtered_rules
    
    def _is_severity_above_threshold(self, rule_severity: RuleSeverity, threshold: RuleSeverity) -> bool:
        """Verificar si la severidad de la regla está por encima del umbral."""
        severity_order = {
            RuleSeverity.INFO: 0,
            RuleSeverity.LOW: 1,
            RuleSeverity.MEDIUM: 2,
            RuleSeverity.HIGH: 3,
            RuleSeverity.CRITICAL: 4
        }
        
        return severity_order[rule_severity] >= severity_order[threshold]
    
    async def _discover_analyzable_files(self, project_path: Path, project_config: ProjectConfig) -> List[Path]:
        """Descubrir archivos analizables en el proyecto."""
        # Implementación simplificada - en una implementación real,
        # esto usaría el sistema de parsers para detectar archivos
        analyzable_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.rs'
        }
        
        files = []
        for file_path in project_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in analyzable_extensions:
                # Verificar exclusiones
                if not self._is_file_excluded(file_path, project_config.exclusion_patterns):
                    files.append(file_path)
        
        return files
    
    def _is_file_excluded(self, file_path: Path, exclusion_patterns: List[str]) -> bool:
        """Verificar si un archivo está excluido por los patrones."""
        # Implementación simplificada
        file_str = str(file_path)
        for pattern in exclusion_patterns:
            if pattern in file_str:
                return True
        return False
    
    def _chunk_files(self, files: List[Path], batch_size: int) -> List[List[Path]]:
        """Dividir archivos en lotes."""
        return [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    
    async def _analyze_file_batch(self, files: List[Path], project_config: ProjectConfig) -> List[AnalysisResult]:
        """Analizar un lote de archivos."""
        # En una implementación real, esto usaría el sistema de parsers
        # para convertir archivos a ASTs unificados
        results = []
        
        for file_path in files:
            try:
                # Crear AST unificado (simulado)
                unified_ast = UnifiedAST(
                    id="temp-ast",
                    language="python",  # Detectar automáticamente
                    file_path=file_path,
                    root_node=None,  # Se crearía con el parser
                    metadata=None,
                    semantic_info=None
                )
                
                result = await self.analyze_code(unified_ast, project_config)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
                results.append(self._create_error_result(file_path, str(e)))
        
        return results
    
    def _create_analysis_result(self, ast: UnifiedAST, aggregated_results: Any, 
                               context: AnalysisContext, start_time: float) -> AnalysisResult:
        """Crear resultado de análisis."""
        execution_time = (time.time() - start_time) * 1000
        
        return AnalysisResult(
            file_path=ast.file_path,
            language=ast.language,
            violations=aggregated_results.violations,
            suggestions=aggregated_results.suggestions,
            metrics=aggregated_results.metrics,
            execution_time_ms=execution_time,
            rules_executed=len(context.rule_results),
            cache_hits=context.cache_hits,
            performance_stats=aggregated_results.performance_stats,
            success=True
        )
    
    def _create_empty_result(self, ast: UnifiedAST) -> AnalysisResult:
        """Crear resultado vacío."""
        return AnalysisResult(
            file_path=ast.file_path,
            language=ast.language,
            success=True
        )
    
    def _create_error_result(self, ast: Union[UnifiedAST, Path], error_message: str) -> AnalysisResult:
        """Crear resultado de error."""
        file_path = ast.file_path if hasattr(ast, 'file_path') else ast
        
        return AnalysisResult(
            file_path=file_path,
            language="unknown",
            success=False,
            error_message=error_message
        )
    
    def _create_project_result(self, project_path: Path, file_results: List[AnalysisResult],
                              summary_metrics: ProjectMetrics, start_time: float) -> ProjectAnalysisResult:
        """Crear resultado del proyecto."""
        execution_time = (time.time() - start_time) * 1000
        
        # Calcular métricas agregadas
        total_violations = sum(len(result.violations) for result in file_results)
        critical_violations = sum(
            len([v for v in result.violations if v.severity == RuleSeverity.CRITICAL])
            for result in file_results
        )
        high_violations = sum(
            len([v for v in result.violations if v.severity == RuleSeverity.HIGH])
            for result in file_results
        )
        
        # Calcular score de calidad
        quality_score = self._calculate_quality_score(summary_metrics)
        
        return ProjectAnalysisResult(
            project_path=project_path,
            file_results=file_results,
            summary_metrics=summary_metrics,
            total_violations=total_violations,
            critical_violations=critical_violations,
            high_violations=high_violations,
            quality_score=quality_score,
            execution_time_ms=execution_time,
            success=True
        )
    
    def _create_empty_project_result(self, project_path: Path) -> ProjectAnalysisResult:
        """Crear resultado vacío del proyecto."""
        return ProjectAnalysisResult(
            project_path=project_path,
            success=True
        )
    
    def _create_error_project_result(self, project_path: Path, error_message: str) -> ProjectAnalysisResult:
        """Crear resultado de error del proyecto."""
        return ProjectAnalysisResult(
            project_path=project_path,
            success=False,
            error_message=error_message
        )
    
    def _calculate_quality_score(self, metrics: ProjectMetrics) -> float:
        """Calcular score de calidad basado en métricas."""
        # Implementación simplificada
        base_score = 100.0
        
        # Penalizar por violaciones
        base_score -= metrics.critical_violations * 10
        base_score -= metrics.high_violations * 5
        base_score -= metrics.medium_violations * 2
        base_score -= metrics.low_violations * 1
        
        # Penalizar por deuda técnica
        base_score -= metrics.technical_debt_hours * 0.1
        
        return max(0.0, min(100.0, base_score))
    
    def _update_engine_metrics(self, result: AnalysisResult) -> None:
        """Actualizar métricas del motor."""
        self.analysis_count += 1
        self.total_execution_time += result.execution_time_ms
    
    async def load_default_rules(self) -> None:
        """Cargar reglas por defecto."""
        # En una implementación real, esto cargaría reglas desde archivos
        # o desde una biblioteca predefinida
        logger.info("Loading default rules...")
        
        # Por ahora, no cargamos reglas específicas
        # Las reglas se cargarían desde el sistema de reglas built-in
        pass
