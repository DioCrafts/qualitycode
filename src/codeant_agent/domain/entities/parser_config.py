"""
Entidades para configuración del parser universal.

Este módulo define las entidades que representan la configuración
del sistema de parsing universal.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from enum import Enum

from ..value_objects.programming_language import ProgrammingLanguage


class ParserMode(Enum):
    """Modos de operación del parser."""
    STRICT = "strict"
    RELAXED = "relaxed"
    RECOVERY = "recovery"
    FAST = "fast"
    COMPLETE = "complete"


class CacheStrategy(Enum):
    """Estrategias de cache."""
    NONE = "none"
    MEMORY_ONLY = "memory_only"
    DISK_ONLY = "disk_only"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"


class ParallelStrategy(Enum):
    """Estrategias de procesamiento paralelo."""
    SEQUENTIAL = "sequential"
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNC = "async"
    HYBRID = "hybrid"


@dataclass
class ParserConfig:
    """Configuración principal del parser universal."""
    
    # Configuración general
    mode: ParserMode = ParserMode.RELAXED
    enable_incremental_parsing: bool = True
    enable_error_recovery: bool = True
    enable_parallel_processing: bool = True
    enable_caching: bool = True
    
    # Límites de recursos
    max_file_size_mb: int = 100
    max_memory_usage_mb: int = 1024
    max_concurrent_parses: int = 10
    max_cache_size_mb: int = 512
    timeout_seconds: int = 300
    
    # Configuración de parsing
    parse_timeout_seconds: int = 60
    max_parse_errors: int = 100
    max_parse_warnings: int = 1000
    enable_syntax_validation: bool = True
    enable_semantic_validation: bool = False
    enable_type_checking: bool = False
    
    # Configuración de cache
    cache_strategy: CacheStrategy = CacheStrategy.HYBRID
    cache_ttl_seconds: int = 3600
    cache_cleanup_interval_seconds: int = 300
    enable_cache_compression: bool = True
    cache_eviction_policy: str = "lru"
    
    # Configuración de procesamiento paralelo
    parallel_strategy: ParallelStrategy = ParallelStrategy.HYBRID
    thread_pool_size: int = 4
    process_pool_size: int = 2
    chunk_size: int = 100
    enable_work_stealing: bool = True
    
    # Configuración de lenguajes
    supported_languages: Set[ProgrammingLanguage] = field(default_factory=set)
    experimental_languages: Set[ProgrammingLanguage] = field(default_factory=set)
    language_specific_configs: Dict[ProgrammingLanguage, Dict[str, Any]] = field(default_factory=dict)
    
    # Configuración de normalización
    enable_ast_normalization: bool = True
    preserve_original_syntax: bool = True
    normalize_node_types: bool = True
    normalize_semantic_info: bool = True
    
    # Configuración de queries
    enable_query_caching: bool = True
    max_query_results: int = 10000
    query_timeout_seconds: int = 30
    enable_query_optimization: bool = True
    
    # Configuración de logging y monitoreo
    enable_metrics: bool = True
    enable_performance_profiling: bool = False
    log_level: str = "INFO"
    enable_error_reporting: bool = True
    
    # Configuración avanzada
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validar la configuración del parser."""
        if self.max_file_size_mb <= 0:
            raise ValueError("El tamaño máximo de archivo debe ser mayor a 0")
        
        if self.max_memory_usage_mb <= 0:
            raise ValueError("El uso máximo de memoria debe ser mayor a 0")
        
        if self.max_concurrent_parses <= 0:
            raise ValueError("El número máximo de parses concurrentes debe ser mayor a 0")
        
        if self.max_cache_size_mb <= 0:
            raise ValueError("El tamaño máximo de cache debe ser mayor a 0")
        
        if self.timeout_seconds <= 0:
            raise ValueError("El timeout debe ser mayor a 0")
        
        if self.parse_timeout_seconds <= 0:
            raise ValueError("El timeout de parsing debe ser mayor a 0")
        
        if self.max_parse_errors < 0:
            raise ValueError("El número máximo de errores no puede ser negativo")
        
        if self.max_parse_warnings < 0:
            raise ValueError("El número máximo de advertencias no puede ser negativo")
        
        if self.cache_ttl_seconds <= 0:
            raise ValueError("El TTL del cache debe ser mayor a 0")
        
        if self.cache_cleanup_interval_seconds <= 0:
            raise ValueError("El intervalo de limpieza del cache debe ser mayor a 0")
        
        if self.thread_pool_size <= 0:
            raise ValueError("El tamaño del pool de threads debe ser mayor a 0")
        
        if self.process_pool_size <= 0:
            raise ValueError("El tamaño del pool de procesos debe ser mayor a 0")
        
        if self.chunk_size <= 0:
            raise ValueError("El tamaño de chunk debe ser mayor a 0")
        
        if self.max_query_results <= 0:
            raise ValueError("El número máximo de resultados de query debe ser mayor a 0")
        
        if self.query_timeout_seconds <= 0:
            raise ValueError("El timeout de query debe ser mayor a 0")
        
        # Inicializar lenguajes soportados por defecto
        if not self.supported_languages:
            self.supported_languages = {
                ProgrammingLanguage.PYTHON,
                ProgrammingLanguage.TYPESCRIPT,
                ProgrammingLanguage.JAVASCRIPT,
                ProgrammingLanguage.RUST,
                ProgrammingLanguage.JAVA,
                ProgrammingLanguage.GO,
                ProgrammingLanguage.CPP,
                ProgrammingLanguage.CSHARP,
            }
        
        if not self.experimental_languages:
            self.experimental_languages = {
                ProgrammingLanguage.C,
                ProgrammingLanguage.SCALA,
                ProgrammingLanguage.KOTLIN,
                ProgrammingLanguage.SWIFT,
                ProgrammingLanguage.PHP,
                ProgrammingLanguage.RUBY,
                ProgrammingLanguage.PERL,
            }
    
    def is_language_supported(self, language: ProgrammingLanguage) -> bool:
        """Verifica si un lenguaje está soportado."""
        return language in self.supported_languages
    
    def is_language_experimental(self, language: ProgrammingLanguage) -> bool:
        """Verifica si un lenguaje está en fase experimental."""
        return language in self.experimental_languages
    
    def get_language_config(self, language: ProgrammingLanguage) -> Dict[str, Any]:
        """Obtiene la configuración específica de un lenguaje."""
        return self.language_specific_configs.get(language, {})
    
    def set_language_config(self, language: ProgrammingLanguage, config: Dict[str, Any]) -> None:
        """Establece la configuración específica de un lenguaje."""
        self.language_specific_configs[language] = config
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Obtiene la configuración de cache."""
        return {
            "strategy": self.cache_strategy.value,
            "ttl_seconds": self.cache_ttl_seconds,
            "cleanup_interval_seconds": self.cache_cleanup_interval_seconds,
            "enable_compression": self.enable_cache_compression,
            "eviction_policy": self.cache_eviction_policy,
            "max_size_mb": self.max_cache_size_mb,
        }
    
    def get_parallel_config(self) -> Dict[str, Any]:
        """Obtiene la configuración de procesamiento paralelo."""
        return {
            "strategy": self.parallel_strategy.value,
            "thread_pool_size": self.thread_pool_size,
            "process_pool_size": self.process_pool_size,
            "chunk_size": self.chunk_size,
            "enable_work_stealing": self.enable_work_stealing,
            "max_concurrent": self.max_concurrent_parses,
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Obtiene la configuración de validación."""
        return {
            "enable_syntax_validation": self.enable_syntax_validation,
            "enable_semantic_validation": self.enable_semantic_validation,
            "enable_type_checking": self.enable_type_checking,
            "max_parse_errors": self.max_parse_errors,
            "max_parse_warnings": self.max_parse_warnings,
        }
    
    def get_query_config(self) -> Dict[str, Any]:
        """Obtiene la configuración de queries."""
        return {
            "enable_caching": self.enable_query_caching,
            "max_results": self.max_query_results,
            "timeout_seconds": self.query_timeout_seconds,
            "enable_optimization": self.enable_query_optimization,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a diccionario."""
        return {
            "general": {
                "mode": self.mode.value,
                "enable_incremental_parsing": self.enable_incremental_parsing,
                "enable_error_recovery": self.enable_error_recovery,
                "enable_parallel_processing": self.enable_parallel_processing,
                "enable_caching": self.enable_caching,
            },
            "resources": {
                "max_file_size_mb": self.max_file_size_mb,
                "max_memory_usage_mb": self.max_memory_usage_mb,
                "max_concurrent_parses": self.max_concurrent_parses,
                "max_cache_size_mb": self.max_cache_size_mb,
                "timeout_seconds": self.timeout_seconds,
            },
            "parsing": {
                "parse_timeout_seconds": self.parse_timeout_seconds,
                "max_parse_errors": self.max_parse_errors,
                "max_parse_warnings": self.max_parse_warnings,
                "enable_syntax_validation": self.enable_syntax_validation,
                "enable_semantic_validation": self.enable_semantic_validation,
                "enable_type_checking": self.enable_type_checking,
            },
            "cache": self.get_cache_config(),
            "parallel": self.get_parallel_config(),
            "validation": self.get_validation_config(),
            "query": self.get_query_config(),
            "languages": {
                "supported": [lang.value for lang in self.supported_languages],
                "experimental": [lang.value for lang in self.experimental_languages],
                "specific_configs": {
                    lang.value: config for lang, config in self.language_specific_configs.items()
                },
            },
            "normalization": {
                "enable_ast_normalization": self.enable_ast_normalization,
                "preserve_original_syntax": self.preserve_original_syntax,
                "normalize_node_types": self.normalize_node_types,
                "normalize_semantic_info": self.normalize_semantic_info,
            },
            "monitoring": {
                "enable_metrics": self.enable_metrics,
                "enable_performance_profiling": self.enable_performance_profiling,
                "log_level": self.log_level,
                "enable_error_reporting": self.enable_error_reporting,
            },
            "custom_settings": self.custom_settings,
        }
    
    def __str__(self) -> str:
        """Representación string de la configuración."""
        return f"ParserConfig({self.mode.value}, {len(self.supported_languages)} languages, {self.max_concurrent_parses} concurrent)"
    
    def __repr__(self) -> str:
        """Representación de debug de la configuración."""
        return (
            f"ParserConfig("
            f"mode={self.mode}, "
            f"supported_languages={self.supported_languages}, "
            f"max_file_size_mb={self.max_file_size_mb}, "
            f"max_concurrent_parses={self.max_concurrent_parses}"
            f")"
        )


@dataclass
class LanguageSpecificConfig:
    """Configuración específica de un lenguaje."""
    
    language: ProgrammingLanguage
    enable_parsing: bool = True
    enable_normalization: bool = True
    enable_queries: bool = True
    enable_caching: bool = True
    
    # Configuración específica del lenguaje
    parser_options: Dict[str, Any] = field(default_factory=dict)
    normalization_rules: Dict[str, Any] = field(default_factory=dict)
    query_patterns: List[str] = field(default_factory=list)
    cache_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Configuración de performance
    parse_timeout_seconds: Optional[int] = None
    max_file_size_mb: Optional[int] = None
    enable_optimizations: bool = True
    
    # Configuración de features
    supported_features: Set[str] = field(default_factory=set)
    experimental_features: Set[str] = field(default_factory=set)
    deprecated_features: Set[str] = field(default_factory=set)
    
    def __post_init__(self) -> None:
        """Validar la configuración específica del lenguaje."""
        if self.language is None:
            raise ValueError("El lenguaje no puede ser None")
        
        if self.parse_timeout_seconds is not None and self.parse_timeout_seconds <= 0:
            raise ValueError("El timeout de parsing debe ser mayor a 0")
        
        if self.max_file_size_mb is not None and self.max_file_size_mb <= 0:
            raise ValueError("El tamaño máximo de archivo debe ser mayor a 0")
    
    def is_feature_supported(self, feature: str) -> bool:
        """Verifica si una feature está soportada."""
        return feature in self.supported_features
    
    def is_feature_experimental(self, feature: str) -> bool:
        """Verifica si una feature está en fase experimental."""
        return feature in self.experimental_features
    
    def is_feature_deprecated(self, feature: str) -> bool:
        """Verifica si una feature está deprecada."""
        return feature in self.deprecated_features
    
    def add_supported_feature(self, feature: str) -> None:
        """Agrega una feature soportada."""
        self.supported_features.add(feature)
    
    def add_experimental_feature(self, feature: str) -> None:
        """Agrega una feature experimental."""
        self.experimental_features.add(feature)
    
    def add_deprecated_feature(self, feature: str) -> None:
        """Agrega una feature deprecada."""
        self.deprecated_features.add(feature)
    
    def get_parser_option(self, key: str, default: Any = None) -> Any:
        """Obtiene una opción del parser."""
        return self.parser_options.get(key, default)
    
    def set_parser_option(self, key: str, value: Any) -> None:
        """Establece una opción del parser."""
        self.parser_options[key] = value
    
    def get_normalization_rule(self, key: str, default: Any = None) -> Any:
        """Obtiene una regla de normalización."""
        return self.normalization_rules.get(key, default)
    
    def set_normalization_rule(self, key: str, value: Any) -> None:
        """Establece una regla de normalización."""
        self.normalization_rules[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a diccionario."""
        return {
            "language": self.language.value,
            "language_name": self.language.get_name(),
            "enable_parsing": self.enable_parsing,
            "enable_normalization": self.enable_normalization,
            "enable_queries": self.enable_queries,
            "enable_caching": self.enable_caching,
            "parser_options": self.parser_options,
            "normalization_rules": self.normalization_rules,
            "query_patterns": self.query_patterns,
            "cache_settings": self.cache_settings,
            "parse_timeout_seconds": self.parse_timeout_seconds,
            "max_file_size_mb": self.max_file_size_mb,
            "enable_optimizations": self.enable_optimizations,
            "supported_features": list(self.supported_features),
            "experimental_features": list(self.experimental_features),
            "deprecated_features": list(self.deprecated_features),
        }
    
    def __str__(self) -> str:
        """Representación string de la configuración."""
        features_count = len(self.supported_features)
        return f"LanguageSpecificConfig({self.language.get_name()}, {features_count} features)"
    
    def __repr__(self) -> str:
        """Representación de debug de la configuración."""
        return (
            f"LanguageSpecificConfig("
            f"language={self.language}, "
            f"enable_parsing={self.enable_parsing}, "
            f"enable_normalization={self.enable_normalization}, "
            f"enable_queries={self.enable_queries}, "
            f"supported_features={self.supported_features}"
            f")"
        )


@dataclass
class ParserProfile:
    """Perfil de configuración del parser."""
    
    name: str
    description: str
    config: ParserConfig
    language_configs: Dict[ProgrammingLanguage, LanguageSpecificConfig] = field(default_factory=dict)
    is_default: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validar el perfil del parser."""
        if not self.name.strip():
            raise ValueError("El nombre del perfil no puede estar vacío")
        
        if not self.description.strip():
            raise ValueError("La descripción del perfil no puede estar vacía")
        
        if self.config is None:
            raise ValueError("La configuración no puede ser None")
    
    def get_language_config(self, language: ProgrammingLanguage) -> Optional[LanguageSpecificConfig]:
        """Obtiene la configuración de un lenguaje específico."""
        return self.language_configs.get(language)
    
    def set_language_config(self, language: ProgrammingLanguage, config: LanguageSpecificConfig) -> None:
        """Establece la configuración de un lenguaje específico."""
        self.language_configs[language] = config
    
    def remove_language_config(self, language: ProgrammingLanguage) -> None:
        """Remueve la configuración de un lenguaje específico."""
        if language in self.language_configs:
            del self.language_configs[language]
    
    def get_supported_languages(self) -> Set[ProgrammingLanguage]:
        """Obtiene los lenguajes soportados por este perfil."""
        return set(self.language_configs.keys())
    
    def is_language_configured(self, language: ProgrammingLanguage) -> bool:
        """Verifica si un lenguaje está configurado en este perfil."""
        return language in self.language_configs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el perfil a diccionario."""
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config.to_dict(),
            "language_configs": {
                lang.value: config.to_dict() for lang, config in self.language_configs.items()
            },
            "is_default": self.is_default,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "supported_languages": [lang.value for lang in self.get_supported_languages()],
        }
    
    def __str__(self) -> str:
        """Representación string del perfil."""
        return f"ParserProfile({self.name}, {len(self.language_configs)} languages)"
    
    def __repr__(self) -> str:
        """Representación de debug del perfil."""
        return (
            f"ParserProfile("
            f"name='{self.name}', "
            f"description='{self.description}', "
            f"config={self.config}, "
            f"language_configs={self.language_configs}, "
            f"is_default={self.is_default}"
            f")"
        )
