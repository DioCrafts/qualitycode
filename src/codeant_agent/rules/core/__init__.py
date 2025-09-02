"""
Componentes core del motor de reglas estáticas.

Este paquete contiene los componentes principales del motor de reglas,
incluyendo el engine principal, ejecutor, registro, cache y optimizador.
"""

from .rule_engine import RulesEngine, RulesEngineConfig, RulesEngineError
from .rule_registry import RuleRegistry, RuleRegistryError
from .rule_executor import RuleExecutor, ExecutorConfig, ExecutorError
from .rule_cache import RuleCache, CacheError
from .performance_optimizer import PerformanceOptimizer, OptimizerError
from .result_aggregator import ResultAggregator, AggregatorError
from .configuration_manager import ConfigurationManager, ConfigurationError

__all__ = [
    "RulesEngine",
    "RulesEngineConfig", 
    "RulesEngineError",
    "RuleRegistry",
    "RuleRegistryError",
    "RuleExecutor",
    "ExecutorConfig",
    "ExecutorError",
    "RuleCache",
    "CacheError",
    "PerformanceOptimizer",
    "OptimizerError",
    "ResultAggregator",
    "AggregatorError",
    "ConfigurationManager",
    "ConfigurationError"
]
