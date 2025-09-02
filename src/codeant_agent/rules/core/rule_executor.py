"""
Ejecutor de reglas del motor de reglas estáticas.

Este módulo implementa el ejecutor de reglas con capacidades de paralelización
y optimización de performance para el análisis de código.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from ...parsers.unified.unified_ast import UnifiedAST
from ..models.rule_models import Rule, RuleResult, ProjectConfig
from ..models.config_models import ExecutorConfig
from .rule_registry import RuleRegistry
from .rule_cache import RuleCache
from .performance_optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)


class ExecutorError(Exception):
    """Excepción base para errores del ejecutor."""
    pass


class RuleExecutionError(ExecutorError):
    """Error durante la ejecución de una regla."""
    pass


class RuleTimeoutError(ExecutorError):
    """Error de timeout durante la ejecución de una regla."""
    pass


@dataclass
class ExecutionContext:
    """Contexto de ejecución para una regla."""
    rule: Rule
    ast: UnifiedAST
    project_config: ProjectConfig
    cache_key: str
    start_time: float
    timeout_ms: int


@dataclass
class ExecutionStats:
    """Estadísticas de ejecución."""
    total_rules: int = 0
    executed_rules: int = 0
    successful_rules: int = 0
    failed_rules: int = 0
    timed_out_rules: int = 0
    cache_hits: int = 0
    total_execution_time_ms: float = 0.0
    average_execution_time_ms: float = 0.0
    parallel_execution_time_ms: float = 0.0


class RuleExecutor:
    """
    Ejecutor de reglas con capacidades de paralelización.
    
    Este ejecutor proporciona funcionalidades para ejecutar reglas de análisis
    de manera eficiente, con soporte para paralelización, cache y optimización.
    """
    
    def __init__(self, rule_registry: RuleRegistry, rule_cache: RuleCache,
                 performance_optimizer: PerformanceOptimizer, config: ExecutorConfig):
        """Inicializar el ejecutor de reglas."""
        self.rule_registry = rule_registry
        self.rule_cache = rule_cache
        self.performance_optimizer = performance_optimizer
        self.config = config
        
        # Thread pool para ejecución paralela
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.max_concurrent_rules,
            thread_name_prefix="RuleExecutor"
        )
        
        # Estadísticas de ejecución
        self.execution_stats = ExecutionStats()
        
        # Semáforo para limitar concurrencia
        self.semaphore = asyncio.Semaphore(config.max_concurrent_rules)
        
        logger.info(f"RuleExecutor initialized with {config.max_concurrent_rules} max concurrent rules")
    
    async def execute_rules(self, rules: List[Rule], ast: UnifiedAST, 
                           project_config: ProjectConfig) -> List[RuleResult]:
        """
        Ejecutar un conjunto de reglas sobre un AST.
        
        Args:
            rules: Lista de reglas a ejecutar
            ast: AST unificado a analizar
            project_config: Configuración del proyecto
            
        Returns:
            Lista de resultados de ejecución de reglas
        """
        start_time = time.time()
        
        try:
            # Agrupar reglas por tipo de implementación
            rule_groups = self._group_rules_by_type(rules)
            
            # Ejecutar reglas en paralelo según el tipo
            all_results = []
            
            # Ejecutar reglas de patrones (más comunes)
            if rule_groups['pattern']:
                pattern_results = await self._execute_pattern_rules(
                    rule_groups['pattern'], ast, project_config
                )
                all_results.extend(pattern_results)
            
            # Ejecutar reglas de queries
            if rule_groups['query']:
                query_results = await self._execute_query_rules(
                    rule_groups['query'], ast, project_config
                )
                all_results.extend(query_results)
            
            # Ejecutar reglas procedurales
            if rule_groups['procedural']:
                procedural_results = await self._execute_procedural_rules(
                    rule_groups['procedural'], ast, project_config
                )
                all_results.extend(procedural_results)
            
            # Ejecutar reglas compuestas
            if rule_groups['composite']:
                composite_results = await self._execute_composite_rules(
                    rule_groups['composite'], ast, project_config
                )
                all_results.extend(composite_results)
            
            # Ejecutar reglas de machine learning
            if rule_groups['machine_learning']:
                ml_results = await self._execute_ml_rules(
                    rule_groups['machine_learning'], ast, project_config
                )
                all_results.extend(ml_results)
            
            # Actualizar estadísticas
            execution_time = (time.time() - start_time) * 1000
            self._update_execution_stats(len(rules), len(all_results), execution_time)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error executing rules: {e}")
            raise ExecutorError(f"Rule execution failed: {e}")
    
    def _group_rules_by_type(self, rules: List[Rule]) -> Dict[str, List[Rule]]:
        """Agrupar reglas por tipo de implementación."""
        groups = {
            'pattern': [],
            'query': [],
            'procedural': [],
            'composite': [],
            'machine_learning': []
        }
        
        for rule in rules:
            implementation = rule.implementation
            
            if implementation.pattern:
                groups['pattern'].append(rule)
            elif implementation.query:
                groups['query'].append(rule)
            elif implementation.procedural:
                groups['procedural'].append(rule)
            elif implementation.composite:
                groups['composite'].append(rule)
            elif implementation.machine_learning:
                groups['machine_learning'].append(rule)
        
        return groups
    
    async def _execute_pattern_rules(self, rules: List[Rule], ast: UnifiedAST,
                                   project_config: ProjectConfig) -> List[RuleResult]:
        """Ejecutar reglas basadas en patrones."""
        # Ejecutar en lotes para optimizar performance
        batch_size = self.config.batch_size
        all_results = []
        
        for i in range(0, len(rules), batch_size):
            batch = rules[i:i + batch_size]
            
            # Crear tareas para el lote
            tasks = []
            for rule in batch:
                task = self._execute_single_pattern_rule(rule, ast, project_config)
                tasks.append(task)
            
            # Ejecutar lote en paralelo
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Procesar resultados
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Pattern rule execution failed: {result}")
                    # Crear resultado de error
                    error_result = RuleResult(
                        rule_id="unknown",
                        success=False,
                        error_message=str(result)
                    )
                    all_results.append(error_result)
                else:
                    all_results.extend(result)
        
        return all_results
    
    async def _execute_single_pattern_rule(self, rule: Rule, ast: UnifiedAST,
                                         project_config: ProjectConfig) -> List[RuleResult]:
        """Ejecutar una sola regla de patrón."""
        start_time = time.time()
        
        try:
            # Generar clave de cache
            cache_key = self._generate_cache_key(rule, ast)
            
            # Verificar cache
            cached_result = await self.rule_cache.get(cache_key)
            if cached_result is not None:
                self.execution_stats.cache_hits += 1
                return cached_result
            
            # Crear contexto de ejecución
            context = ExecutionContext(
                rule=rule,
                ast=ast,
                project_config=project_config,
                cache_key=cache_key,
                start_time=start_time,
                timeout_ms=self.config.rule_timeout_ms
            )
            
            # Ejecutar regla con timeout
            result = await asyncio.wait_for(
                self._execute_pattern_rule_impl(context),
                timeout=self.config.rule_timeout_ms / 1000.0
            )
            
            # Cachear resultado exitoso
            if result and result[0].success:
                await self.rule_cache.set(cache_key, result)
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Rule {rule.id} timed out after {self.config.rule_timeout_ms}ms")
            self.execution_stats.timed_out_rules += 1
            
            return [RuleResult(
                rule_id=rule.id,
                success=False,
                error_message=f"Rule execution timed out after {self.config.rule_timeout_ms}ms"
            )]
        
        except Exception as e:
            logger.error(f"Error executing pattern rule {rule.id}: {e}")
            self.execution_stats.failed_rules += 1
            
            return [RuleResult(
                rule_id=rule.id,
                success=False,
                error_message=str(e)
            )]
    
    async def _execute_pattern_rule_impl(self, context: ExecutionContext) -> List[RuleResult]:
        """Implementación de ejecución de regla de patrón."""
        rule = context.rule
        ast = context.ast
        
        # En una implementación real, aquí se ejecutaría el pattern matching
        # usando el sistema de patrones AST definido en pattern_models.py
        
        # Simulación básica
        results = []
        
        # Buscar nodos que coincidan con el patrón
        if rule.implementation.pattern:
            # Aquí se implementaría la lógica de pattern matching
            # Por ahora, simulamos que no se encontraron coincidencias
            pass
        
        # Crear resultado
        execution_time = (time.time() - context.start_time) * 1000
        
        result = RuleResult(
            rule_id=rule.id,
            violations=[],  # Se llenarían con violaciones encontradas
            suggestions=[],  # Se llenarían con sugerencias generadas
            metrics={
                'execution_time_ms': execution_time,
                'memory_usage_mb': 0.0,
                'cache_hit': False,
                'violations_found': 0,
                'suggestions_generated': 0,
                'error_count': 0
            },
            success=True
        )
        
        results.append(result)
        return results
    
    async def _execute_query_rules(self, rules: List[Rule], ast: UnifiedAST,
                                 project_config: ProjectConfig) -> List[RuleResult]:
        """Ejecutar reglas basadas en queries."""
        # Implementación similar a pattern rules pero para queries
        results = []
        
        for rule in rules:
            try:
                # Ejecutar query unificada
                if rule.implementation.query:
                    # Aquí se implementaría la ejecución de queries
                    # usando el sistema de queries unificadas
                    pass
                
                result = RuleResult(
                    rule_id=rule.id,
                    success=True
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error executing query rule {rule.id}: {e}")
                results.append(RuleResult(
                    rule_id=rule.id,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    async def _execute_procedural_rules(self, rules: List[Rule], ast: UnifiedAST,
                                      project_config: ProjectConfig) -> List[RuleResult]:
        """Ejecutar reglas procedurales."""
        # Implementación para reglas procedurales
        results = []
        
        for rule in rules:
            try:
                # Ejecutar función de análisis procedural
                if rule.implementation.procedural:
                    # Aquí se implementaría la ejecución de funciones procedurales
                    pass
                
                result = RuleResult(
                    rule_id=rule.id,
                    success=True
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error executing procedural rule {rule.id}: {e}")
                results.append(RuleResult(
                    rule_id=rule.id,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    async def _execute_composite_rules(self, rules: List[Rule], ast: UnifiedAST,
                                     project_config: ProjectConfig) -> List[RuleResult]:
        """Ejecutar reglas compuestas."""
        # Implementación para reglas compuestas
        results = []
        
        for rule in rules:
            try:
                # Ejecutar sub-reglas y combinar resultados
                if rule.implementation.composite:
                    # Aquí se implementaría la lógica de composición
                    pass
                
                result = RuleResult(
                    rule_id=rule.id,
                    success=True
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error executing composite rule {rule.id}: {e}")
                results.append(RuleResult(
                    rule_id=rule.id,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    async def _execute_ml_rules(self, rules: List[Rule], ast: UnifiedAST,
                              project_config: ProjectConfig) -> List[RuleResult]:
        """Ejecutar reglas de machine learning."""
        # Implementación para reglas de ML
        results = []
        
        for rule in rules:
            try:
                # Ejecutar modelo de ML
                if rule.implementation.machine_learning:
                    # Aquí se implementaría la ejecución de modelos de ML
                    pass
                
                result = RuleResult(
                    rule_id=rule.id,
                    success=True
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error executing ML rule {rule.id}: {e}")
                results.append(RuleResult(
                    rule_id=rule.id,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    def _generate_cache_key(self, rule: Rule, ast: UnifiedAST) -> str:
        """Generar clave de cache para una regla y AST."""
        # Clave simple basada en ID de regla y hash del AST
        ast_hash = hash(str(ast.file_path) + str(ast.metadata.node_count))
        return f"{rule.id}:{ast_hash}"
    
    def _update_execution_stats(self, total_rules: int, executed_rules: int, 
                              execution_time_ms: float) -> None:
        """Actualizar estadísticas de ejecución."""
        self.execution_stats.total_rules += total_rules
        self.execution_stats.executed_rules += executed_rules
        self.execution_stats.total_execution_time_ms += execution_time_ms
        
        if self.execution_stats.executed_rules > 0:
            self.execution_stats.average_execution_time_ms = (
                self.execution_stats.total_execution_time_ms / 
                self.execution_stats.executed_rules
            )
    
    async def get_execution_stats(self) -> ExecutionStats:
        """Obtener estadísticas de ejecución."""
        return self.execution_stats
    
    async def reset_stats(self) -> None:
        """Resetear estadísticas de ejecución."""
        self.execution_stats = ExecutionStats()
    
    async def shutdown(self) -> None:
        """Apagar el ejecutor de reglas."""
        self.thread_pool.shutdown(wait=True)
        logger.info("RuleExecutor shutdown completed")
