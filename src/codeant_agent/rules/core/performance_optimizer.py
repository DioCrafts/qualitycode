"""
Optimizador de performance para el motor de reglas estáticas.

Este módulo implementa optimizaciones de performance para mejorar la
velocidad y eficiencia del motor de reglas.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ...parsers.unified.unified_ast import UnifiedAST
from ..models.rule_models import Rule, RuleCategory, RuleSeverity
from ..models.config_models import OptimizationStrategy

logger = logging.getLogger(__name__)


class OptimizerError(Exception):
    """Excepción base para errores del optimizador."""
    pass


@dataclass
class ExecutionStats:
    """Estadísticas de ejecución de una regla."""
    rule_id: str
    average_execution_time: float = 0.0
    success_rate: float = 1.0
    memory_usage: float = 0.0
    cache_hit_rate: float = 0.0
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    total_execution_time: float = 0.0
    total_memory_usage: float = 0.0
    total_cache_hits: int = 0
    total_executions: int = 0
    
    def update_stats(self, execution_time: float, memory_usage: float, 
                    cache_hit: bool, success: bool) -> None:
        """Actualizar estadísticas con una nueva ejecución."""
        self.execution_count += 1
        self.total_executions += 1
        self.total_execution_time += execution_time
        self.total_memory_usage += memory_usage
        
        if cache_hit:
            self.total_cache_hits += 1
        
        # Actualizar promedios
        self.average_execution_time = self.total_execution_time / self.total_executions
        self.memory_usage = self.total_memory_usage / self.total_executions
        self.cache_hit_rate = self.total_cache_hits / self.total_executions
        
        # Actualizar tasa de éxito
        if success:
            self.success_rate = (self.success_rate * (self.total_executions - 1) + 1.0) / self.total_executions
        else:
            self.success_rate = (self.success_rate * (self.total_executions - 1)) / self.total_executions
        
        self.last_executed = datetime.now(timezone.utc)


@dataclass
class OptimizationResult:
    """Resultado de la optimización."""
    optimized_rules: List[Rule]
    strategy_used: OptimizationStrategy
    estimated_execution_time: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceMonitor:
    """Monitor de recursos del sistema."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update_metrics(self, cpu: float, memory: float, disk: float, network: float) -> None:
        """Actualizar métricas de recursos."""
        self.cpu_usage = cpu
        self.memory_usage = memory
        self.disk_io = disk
        self.network_io = network
        self.last_updated = datetime.now(timezone.utc)


class PerformanceOptimizer:
    """
    Optimizador de performance para el motor de reglas.
    
    Este optimizador analiza patrones de ejecución y optimiza el orden
    de ejecución de reglas para maximizar la performance.
    """
    
    def __init__(self):
        """Inicializar el optimizador de performance."""
        # Historial de ejecución
        self.execution_history: Dict[str, ExecutionStats] = {}
        
        # Gráfico de dependencias entre reglas
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Monitor de recursos
        self.resource_monitor = ResourceMonitor()
        
        # Configuración de optimización
        self.optimization_enabled = True
        self.learning_rate = 0.1
        self.min_executions_for_optimization = 5
        
        # Lock para operaciones concurrentes
        self.lock = asyncio.Lock()
        
        logger.info("PerformanceOptimizer initialized")
    
    async def initialize(self) -> None:
        """Inicializar el optimizador."""
        try:
            # Cargar historial de ejecución desde persistencia
            await self._load_execution_history()
            
            # Inicializar monitor de recursos
            await self._initialize_resource_monitor()
            
            logger.info("PerformanceOptimizer initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize PerformanceOptimizer: {e}")
    
    async def optimize_execution_order(self, rules: List[Rule], ast: UnifiedAST) -> List[Rule]:
        """
        Optimizar el orden de ejecución de reglas.
        
        Args:
            rules: Lista de reglas a optimizar
            ast: AST unificado para análisis
            
        Returns:
            Lista de reglas optimizada
        """
        if not self.optimization_enabled or not rules:
            return rules
        
        try:
            # Determinar estrategia de optimización
            strategy = self._determine_optimization_strategy(rules, ast)
            
            # Aplicar optimización según la estrategia
            if strategy == OptimizationStrategy.FAST_FIRST:
                optimized_rules = await self._optimize_fast_first(rules)
            elif strategy == OptimizationStrategy.HIGH_IMPACT_FIRST:
                optimized_rules = await self._optimize_high_impact_first(rules)
            elif strategy == OptimizationStrategy.DEPENDENCY_BASED:
                optimized_rules = await self._optimize_dependency_based(rules)
            elif strategy == OptimizationStrategy.RESOURCE_AWARE:
                optimized_rules = await self._optimize_resource_aware(rules, ast)
            else:
                optimized_rules = rules
            
            # Crear resultado de optimización
            result = OptimizationResult(
                optimized_rules=optimized_rules,
                strategy_used=strategy,
                estimated_execution_time=self._estimate_execution_time(optimized_rules, ast),
                confidence=self._calculate_optimization_confidence(rules, strategy)
            )
            
            logger.info(f"Optimized {len(rules)} rules using {strategy} strategy")
            return optimized_rules
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return rules
    
    def _determine_optimization_strategy(self, rules: List[Rule], ast: UnifiedAST) -> OptimizationStrategy:
        """Determinar la estrategia de optimización más apropiada."""
        # Heurísticas para elegir estrategia
        if len(rules) < 10:
            return OptimizationStrategy.FAST_FIRST
        
        if ast.metadata.node_count < 1000:
            return OptimizationStrategy.HIGH_IMPACT_FIRST
        
        if len(rules) > 100:
            return OptimizationStrategy.DEPENDENCY_BASED
        
        if self.resource_monitor.cpu_usage > 80 or self.resource_monitor.memory_usage > 80:
            return OptimizationStrategy.RESOURCE_AWARE
        
        return OptimizationStrategy.FAST_FIRST
    
    async def _optimize_fast_first(self, rules: List[Rule]) -> List[Rule]:
        """Optimizar ordenando reglas rápidas primero."""
        # Ordenar por tiempo de ejecución promedio
        rule_stats = []
        
        for rule in rules:
            stats = self.execution_history.get(rule.id)
            if stats and stats.execution_count >= self.min_executions_for_optimization:
                rule_stats.append((rule, stats.average_execution_time))
            else:
                # Estimación por defecto basada en tipo de regla
                estimated_time = self._estimate_rule_execution_time(rule)
                rule_stats.append((rule, estimated_time))
        
        # Ordenar por tiempo de ejecución (ascendente)
        rule_stats.sort(key=lambda x: x[1])
        
        return [rule for rule, _ in rule_stats]
    
    async def _optimize_high_impact_first(self, rules: List[Rule]) -> List[Rule]:
        """Optimizar ordenando reglas de alto impacto primero."""
        # Calcular score de impacto para cada regla
        rule_scores = []
        
        for rule in rules:
            impact_score = self._calculate_impact_score(rule)
            rule_scores.append((rule, impact_score))
        
        # Ordenar por score de impacto (descendente)
        rule_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [rule for rule, _ in rule_scores]
    
    async def _optimize_dependency_based(self, rules: List[Rule]) -> List[Rule]:
        """Optimizar basado en dependencias entre reglas."""
        # Ordenamiento topológico basado en dependencias
        try:
            ordered_rules = self._topological_sort(rules)
            return ordered_rules
        except Exception as e:
            logger.warning(f"Dependency-based optimization failed: {e}")
            return rules
    
    async def _optimize_resource_aware(self, rules: List[Rule], ast: UnifiedAST) -> List[Rule]:
        """Optimizar considerando recursos del sistema."""
        # Ajustar orden basado en recursos disponibles
        available_memory = 100 - self.resource_monitor.memory_usage
        available_cpu = 100 - self.resource_monitor.cpu_usage
        
        # Priorizar reglas que usen menos recursos
        rule_scores = []
        
        for rule in rules:
            resource_score = self._calculate_resource_score(rule, available_memory, available_cpu)
            rule_scores.append((rule, resource_score))
        
        # Ordenar por score de recursos (descendente)
        rule_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [rule for rule, _ in rule_scores]
    
    def _calculate_impact_score(self, rule: Rule) -> float:
        """Calcular score de impacto de una regla."""
        base_score = 0.0
        
        # Impacto basado en severidad
        severity_weights = {
            RuleSeverity.CRITICAL: 1.0,
            RuleSeverity.HIGH: 0.8,
            RuleSeverity.MEDIUM: 0.6,
            RuleSeverity.LOW: 0.4,
            RuleSeverity.INFO: 0.2
        }
        base_score += severity_weights.get(rule.severity, 0.5)
        
        # Impacto basado en categoría
        category_weights = {
            RuleCategory.SECURITY: 1.0,
            RuleCategory.VULNERABILITY: 1.0,
            RuleCategory.BUG_PRONE: 0.9,
            RuleCategory.PERFORMANCE: 0.8,
            RuleCategory.BEST_PRACTICES: 0.7,
            RuleCategory.MAINTAINABILITY: 0.6
        }
        base_score += category_weights.get(rule.category, 0.5)
        
        # Impacto basado en historial de ejecución
        stats = self.execution_history.get(rule.id)
        if stats and stats.execution_count > 0:
            # Reglas que históricamente encuentran más violaciones
            violation_rate = 1.0 - stats.success_rate  # Asumimos que éxito = sin violaciones
            base_score += violation_rate * 0.5
        
        return base_score
    
    def _calculate_resource_score(self, rule: Rule, available_memory: float, 
                                available_cpu: float) -> float:
        """Calcular score de recursos para una regla."""
        stats = self.execution_history.get(rule.id)
        
        if stats:
            # Score basado en uso histórico de recursos
            memory_score = max(0, 1.0 - (stats.memory_usage / available_memory))
            cpu_score = max(0, 1.0 - (stats.average_execution_time / 1000))  # Normalizar a 1 segundo
            
            return (memory_score + cpu_score) / 2
        else:
            # Estimación basada en tipo de regla
            return self._estimate_resource_usage(rule, available_memory, available_cpu)
    
    def _estimate_resource_usage(self, rule: Rule, available_memory: float, 
                               available_cpu: float) -> float:
        """Estimar uso de recursos para una regla."""
        # Estimaciones basadas en tipo de implementación
        implementation = rule.implementation
        
        if implementation.pattern:
            return 0.8  # Patrones son generalmente eficientes
        elif implementation.query:
            return 0.6  # Queries pueden ser costosas
        elif implementation.procedural:
            return 0.7  # Procedural depende de la implementación
        elif implementation.composite:
            return 0.5  # Compuestas pueden ser costosas
        elif implementation.machine_learning:
            return 0.3  # ML puede ser muy costoso
        
        return 0.5  # Valor por defecto
    
    def _estimate_rule_execution_time(self, rule: Rule) -> float:
        """Estimar tiempo de ejecución de una regla."""
        # Estimaciones basadas en tipo de implementación
        implementation = rule.implementation
        
        if implementation.pattern:
            return 10.0  # 10ms para patrones
        elif implementation.query:
            return 50.0  # 50ms para queries
        elif implementation.procedural:
            return 30.0  # 30ms para procedural
        elif implementation.composite:
            return 100.0  # 100ms para compuestas
        elif implementation.machine_learning:
            return 500.0  # 500ms para ML
        
        return 25.0  # Valor por defecto
    
    def _topological_sort(self, rules: List[Rule]) -> List[Rule]:
        """Ordenamiento topológico basado en dependencias."""
        # Implementación simplificada
        # En una implementación real, analizaría dependencias reales entre reglas
        
        # Por ahora, ordenar por ID para consistencia
        return sorted(rules, key=lambda r: r.id)
    
    def _estimate_execution_time(self, rules: List[Rule], ast: UnifiedAST) -> float:
        """Estimar tiempo total de ejecución."""
        total_time = 0.0
        
        for rule in rules:
            stats = self.execution_history.get(rule.id)
            if stats and stats.execution_count > 0:
                # Ajustar por tamaño del AST
                size_factor = max(1.0, ast.metadata.node_count / 1000.0)
                total_time += stats.average_execution_time * size_factor
            else:
                total_time += self._estimate_rule_execution_time(rule)
        
        return total_time
    
    def _calculate_optimization_confidence(self, rules: List[Rule], 
                                         strategy: OptimizationStrategy) -> float:
        """Calcular confianza en la optimización."""
        # Confianza basada en cantidad de datos históricos
        rules_with_history = 0
        
        for rule in rules:
            stats = self.execution_history.get(rule.id)
            if stats and stats.execution_count >= self.min_executions_for_optimization:
                rules_with_history += 1
        
        base_confidence = rules_with_history / len(rules) if rules else 0.0
        
        # Ajustar por estrategia
        strategy_confidence = {
            OptimizationStrategy.FAST_FIRST: 0.9,
            OptimizationStrategy.HIGH_IMPACT_FIRST: 0.8,
            OptimizationStrategy.DEPENDENCY_BASED: 0.7,
            OptimizationStrategy.RESOURCE_AWARE: 0.6
        }
        
        strategy_multiplier = strategy_confidence.get(strategy, 0.5)
        
        return base_confidence * strategy_multiplier
    
    async def record_execution(self, rule_id: str, execution_time: float, 
                             memory_usage: float, cache_hit: bool, success: bool) -> None:
        """Registrar estadísticas de ejecución de una regla."""
        async with self.lock:
            if rule_id not in self.execution_history:
                self.execution_history[rule_id] = ExecutionStats(rule_id=rule_id)
            
            stats = self.execution_history[rule_id]
            stats.update_stats(execution_time, memory_usage, cache_hit, success)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del optimizador."""
        async with self.lock:
            return {
                'total_rules_tracked': len(self.execution_history),
                'optimization_enabled': self.optimization_enabled,
                'resource_monitor': {
                    'cpu_usage': self.resource_monitor.cpu_usage,
                    'memory_usage': self.resource_monitor.memory_usage,
                    'last_updated': self.resource_monitor.last_updated.isoformat()
                },
                'top_performing_rules': self._get_top_performing_rules(),
                'slowest_rules': self._get_slowest_rules()
            }
    
    def _get_top_performing_rules(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener reglas con mejor performance."""
        sorted_rules = sorted(
            self.execution_history.values(),
            key=lambda x: x.average_execution_time
        )[:limit]
        
        return [
            {
                'rule_id': stats.rule_id,
                'average_execution_time': stats.average_execution_time,
                'execution_count': stats.execution_count,
                'success_rate': stats.success_rate
            }
            for stats in sorted_rules
        ]
    
    def _get_slowest_rules(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener reglas más lentas."""
        sorted_rules = sorted(
            self.execution_history.values(),
            key=lambda x: x.average_execution_time,
            reverse=True
        )[:limit]
        
        return [
            {
                'rule_id': stats.rule_id,
                'average_execution_time': stats.average_execution_time,
                'execution_count': stats.execution_count,
                'success_rate': stats.success_rate
            }
            for stats in sorted_rules
        ]
    
    async def _load_execution_history(self) -> None:
        """Cargar historial de ejecución desde persistencia."""
        # En una implementación real, cargaría desde base de datos o archivo
        logger.info("Loading execution history...")
    
    async def _initialize_resource_monitor(self) -> None:
        """Inicializar monitor de recursos."""
        # En una implementación real, configuraría monitoreo de recursos del sistema
        logger.info("Initializing resource monitor...")
    
    async def shutdown(self) -> None:
        """Apagar el optimizador."""
        try:
            # Guardar historial de ejecución
            await self._save_execution_history()
            
            logger.info("PerformanceOptimizer shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during optimizer shutdown: {e}")
    
    async def _save_execution_history(self) -> None:
        """Guardar historial de ejecución a persistencia."""
        # En una implementación real, guardaría a base de datos o archivo
        logger.info("Saving execution history...")
