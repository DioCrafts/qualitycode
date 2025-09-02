"""
Modelos para configuración del motor de reglas estáticas.

Este módulo define las estructuras de datos para la configuración del motor
de reglas, incluyendo configuraciones globales, de proyecto y de reglas individuales.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone

from pydantic import BaseModel, Field, validator

from .rule_models import RuleCategory, RuleSeverity, RuleConfiguration

logger = logging.getLogger(__name__)


class OutputFormat(str, Enum):
    """Formatos de salida para reportes."""
    JSON = "json"
    XML = "xml"
    HTML = "html"
    MARKDOWN = "markdown"
    SARIF = "sarif"
    CONSOLE = "console"


class CacheStrategy(str, Enum):
    """Estrategias de cache."""
    NONE = "none"
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    HYBRID = "hybrid"


class OptimizationStrategy(str, Enum):
    """Estrategias de optimización de performance."""
    FAST_FIRST = "fast_first"
    HIGH_IMPACT_FIRST = "high_impact_first"
    DEPENDENCY_BASED = "dependency_based"
    RESOURCE_AWARE = "resource_aware"


@dataclass
class PerformanceSettings:
    """Configuración de performance del motor de reglas."""
    max_concurrent_rules: int = 10
    rule_timeout_ms: int = 5000
    memory_limit_mb: int = 1024
    enable_parallel_execution: bool = True
    batch_size: int = 50
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY
    cache_ttl_seconds: int = 3600
    enable_early_termination: bool = True
    max_cache_size_mb: int = 100


@dataclass
class LanguageConfig:
    """Configuración específica por lenguaje."""
    language: str
    enabled_rules: List[str] = field(default_factory=list)
    disabled_rules: List[str] = field(default_factory=list)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    naming_convention: str = "default"
    max_line_length: Optional[int] = None
    indentation_size: int = 4
    use_tabs: bool = False
    
    def __post_init__(self):
        """Validar configuración del lenguaje."""
        if self.max_line_length is not None and self.max_line_length <= 0:
            raise ValueError("max_line_length must be positive")
        if self.indentation_size <= 0:
            raise ValueError("indentation_size must be positive")


@dataclass
class QualityGates:
    """Puertas de calidad para el proyecto."""
    max_critical_violations: int = 0
    max_high_violations: int = 10
    max_medium_violations: int = 50
    max_low_violations: int = 100
    min_quality_score: float = 80.0
    min_maintainability_rating: str = "B"
    min_security_rating: str = "B"
    min_reliability_rating: str = "B"
    max_technical_debt_hours: float = 100.0
    fail_on_quality_gate: bool = True
    
    def __post_init__(self):
        """Validar puertas de calidad."""
        if self.min_quality_score < 0 or self.min_quality_score > 100:
            raise ValueError("min_quality_score must be between 0 and 100")
        if self.max_technical_debt_hours < 0:
            raise ValueError("max_technical_debt_hours must be non-negative")


@dataclass
class RuleOverride:
    """Sobrescritura de configuración para una regla específica."""
    rule_id: str
    enabled: Optional[bool] = None
    severity: Optional[RuleSeverity] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    custom_message: Optional[str] = None
    custom_suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EffectiveRuleConfig:
    """Configuración efectiva de una regla después de aplicar overrides."""
    rule_id: str
    enabled: bool
    severity: RuleSeverity
    parameters: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalRuleConfig:
    """Configuración global del motor de reglas."""
    default_severity_threshold: RuleSeverity = RuleSeverity.INFO
    enabled_categories: List[RuleCategory] = field(default_factory=list)
    disabled_rules: List[str] = field(default_factory=list)
    performance_settings: PerformanceSettings = field(default_factory=PerformanceSettings)
    output_format: OutputFormat = OutputFormat.JSON
    enable_auto_fix: bool = False
    enable_suggestions: bool = True
    enable_metrics: bool = True
    log_level: str = "INFO"
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validar configuración global."""
        if not self.enabled_categories:
            # Habilitar todas las categorías por defecto
            self.enabled_categories = list(RuleCategory)


@dataclass
class ProjectRuleConfig:
    """Configuración de reglas para un proyecto específico."""
    project_path: Path
    rule_overrides: Dict[str, RuleOverride] = field(default_factory=dict)
    custom_thresholds: Dict[str, float] = field(default_factory=dict)
    exclusion_patterns: List[str] = field(default_factory=list)
    language_specific_configs: Dict[str, LanguageConfig] = field(default_factory=dict)
    quality_gates: QualityGates = field(default_factory=QualityGates)
    parallel_analysis_batch_size: Optional[int] = None
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Validar configuración del proyecto."""
        if self.parallel_analysis_batch_size is not None and self.parallel_analysis_batch_size <= 0:
            raise ValueError("parallel_analysis_batch_size must be positive")


@dataclass
class CacheConfig:
    """Configuración del sistema de cache."""
    strategy: CacheStrategy = CacheStrategy.MEMORY
    ttl_seconds: int = 3600
    max_size_mb: int = 100
    redis_url: Optional[str] = None
    disk_path: Optional[Path] = None
    enable_compression: bool = True
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    
    def __post_init__(self):
        """Validar configuración de cache."""
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if self.max_size_mb <= 0:
            raise ValueError("max_size_mb must be positive")
        if self.strategy == CacheStrategy.REDIS and not self.redis_url:
            raise ValueError("redis_url is required for Redis cache strategy")
        if self.strategy == CacheStrategy.DISK and not self.disk_path:
            raise ValueError("disk_path is required for disk cache strategy")


@dataclass
class ExecutorConfig:
    """Configuración del ejecutor de reglas."""
    max_concurrent_rules: int = 10
    rule_timeout_ms: int = 5000
    enable_early_termination: bool = True
    batch_size: int = 50
    memory_limit_mb: int = 1024
    enable_metrics_collection: bool = True
    enable_error_recovery: bool = True
    max_retries: int = 3
    retry_delay_ms: int = 1000
    
    def __post_init__(self):
        """Validar configuración del ejecutor."""
        if self.max_concurrent_rules <= 0:
            raise ValueError("max_concurrent_rules must be positive")
        if self.rule_timeout_ms <= 0:
            raise ValueError("rule_timeout_ms must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay_ms < 0:
            raise ValueError("retry_delay_ms must be non-negative")


@dataclass
class RulesEngineConfig:
    """Configuración principal del motor de reglas."""
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)
    parallel_execution: bool = True
    enable_performance_optimization: bool = True
    enable_rule_metrics: bool = True
    enable_cross_language_analysis: bool = True
    enable_machine_learning: bool = False
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validar configuración del motor."""
        if not self.parallel_execution and self.executor_config.max_concurrent_rules > 1:
            logger.warning("Parallel execution disabled but max_concurrent_rules > 1")


@dataclass
class ConfigParameter:
    """Parámetro de configuración."""
    name: str
    parameter_type: str  # string, integer, float, boolean, array, object, enum
    default_value: Any
    description: str
    validation: Optional[Dict[str, Any]] = None
    enum_values: Optional[List[str]] = None
    required: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    
    def __post_init__(self):
        """Validar parámetro de configuración."""
        if self.parameter_type == "enum" and not self.enum_values:
            raise ValueError("enum_values is required for enum parameter type")
        if self.required and self.default_value is None:
            raise ValueError("default_value is required for required parameters")


@dataclass
class ParameterValidation:
    """Validación de parámetros."""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    custom_validator: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ThresholdConfig:
    """Configuración de umbrales."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    error_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    unit: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ExclusionPattern:
    """Patrón de exclusión."""
    pattern: str
    description: str
    pattern_type: str = "glob"  # glob, regex, path
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigFileFormat(str, Enum):
    """Formatos de archivo de configuración."""
    TOML = "toml"
    YAML = "yaml"
    JSON = "json"
    INI = "ini"


@dataclass
class ConfigFile:
    """Archivo de configuración."""
    path: Path
    format: ConfigFileFormat
    content: str
    last_modified: datetime
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validar archivo de configuración."""
        if not self.path.exists():
            self.is_valid = False
            self.errors.append(f"Config file does not exist: {self.path}")


@dataclass
class ConfigWatcher:
    """Observador de cambios en archivos de configuración."""
    config_path: Path
    callback: callable
    enabled: bool = True
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    check_interval_seconds: int = 30
    
    async def watch(self):
        """Observar cambios en el archivo de configuración."""
        while self.enabled:
            try:
                if self.config_path.exists():
                    current_mtime = self.config_path.stat().st_mtime
                    last_mtime = self.last_check.timestamp()
                    
                    if current_mtime > last_mtime:
                        await self.callback(self.config_path)
                        self.last_check = datetime.now(timezone.utc)
                
                await asyncio.sleep(self.check_interval_seconds)
            except Exception as e:
                logger.error(f"Error watching config file {self.config_path}: {e}")
                await asyncio.sleep(self.check_interval_seconds)


# Configuraciones predefinidas
DEFAULT_GLOBAL_CONFIG = GlobalRuleConfig(
    default_severity_threshold=RuleSeverity.INFO,
    enabled_categories=list(RuleCategory),
    performance_settings=PerformanceSettings(),
    output_format=OutputFormat.JSON,
    enable_auto_fix=False,
    enable_suggestions=True,
    enable_metrics=True,
    log_level="INFO"
)

DEFAULT_PROJECT_CONFIG = ProjectRuleConfig(
    project_path=Path("."),
    quality_gates=QualityGates(),
    parallel_analysis_batch_size=10
)

DEFAULT_CACHE_CONFIG = CacheConfig(
    strategy=CacheStrategy.MEMORY,
    ttl_seconds=3600,
    max_size_mb=100,
    enable_compression=True,
    enable_encryption=False
)

DEFAULT_EXECUTOR_CONFIG = ExecutorConfig(
    max_concurrent_rules=10,
    rule_timeout_ms=5000,
    enable_early_termination=True,
    batch_size=50,
    memory_limit_mb=1024,
    enable_metrics_collection=True,
    enable_error_recovery=True,
    max_retries=3,
    retry_delay_ms=1000
)
