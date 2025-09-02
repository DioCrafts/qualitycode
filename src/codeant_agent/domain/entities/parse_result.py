"""
Entidades del dominio para resultados de parsing.

Este módulo define las entidades que representan los resultados
del análisis sintáctico de código fuente.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from ..value_objects.programming_language import ProgrammingLanguage


class ParseStatus(Enum):
    """Estado del parsing."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CACHED = "cached"


class ParseWarning(Enum):
    """Tipos de advertencias del parsing."""
    SYNTAX_ERRORS_RECOVERED = "syntax_errors_recovered"
    PARTIAL_PARSE = "partial_parse"
    TIMEOUT_WARNING = "timeout_warning"
    LARGE_FILE = "large_file"
    UNSUPPORTED_FEATURE = "unsupported_feature"
    DEPRECATED_SYNTAX = "deprecated_syntax"


@dataclass
class ParseRequest:
    """Solicitud de parsing."""
    
    file_path: Optional[Path] = None
    content: Optional[str] = None
    language: Optional[ProgrammingLanguage] = None
    enable_cache: bool = True
    timeout_seconds: int = 30
    max_file_size_mb: int = 100
    enable_error_recovery: bool = True
    include_metadata: bool = True
    include_symbols: bool = True
    include_comments: bool = True
    
    def __post_init__(self) -> None:
        """Validar la solicitud de parsing."""
        if self.file_path is None and self.content is None:
            raise ValueError("Se debe proporcionar file_path o content")
        
        if self.content and len(self.content) > self.max_file_size_mb * 1024 * 1024:
            raise ValueError(f"El contenido excede el tamaño máximo de {self.max_file_size_mb}MB")
        
        if self.timeout_seconds <= 0:
            raise ValueError("El timeout debe ser mayor a 0")
        
        if self.max_file_size_mb <= 0:
            raise ValueError("El tamaño máximo de archivo debe ser mayor a 0")


@dataclass
class ParseMetadata:
    """Metadatos del resultado del parsing."""
    
    file_size_bytes: int
    line_count: int
    character_count: int
    encoding: str = "UTF-8"
    has_syntax_errors: bool = False
    complexity_estimate: float = 0.0
    parsed_at: datetime = field(default_factory=datetime.utcnow)
    parse_duration_ms: int = 0
    node_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    cache_hit: bool = False
    incremental_parse: bool = False
    
    def __post_init__(self) -> None:
        """Validar metadatos."""
        if self.file_size_bytes < 0:
            raise ValueError("El tamaño del archivo no puede ser negativo")
        
        if self.line_count < 0:
            raise ValueError("El número de líneas no puede ser negativo")
        
        if self.character_count < 0:
            raise ValueError("El número de caracteres no puede ser negativo")
        
        if self.complexity_estimate < 0:
            raise ValueError("La complejidad estimada no puede ser negativa")
        
        if self.parse_duration_ms < 0:
            raise ValueError("La duración del parsing no puede ser negativa")
        
        if self.node_count < 0:
            raise ValueError("El número de nodos no puede ser negativo")
        
        if self.error_count < 0:
            raise ValueError("El número de errores no puede ser negativo")
        
        if self.warning_count < 0:
            raise ValueError("El número de advertencias no puede ser negativo")


@dataclass
class ParseResult:
    """Resultado del parsing de código fuente."""
    
    tree: Any  # Tree-sitter Tree object
    language: ProgrammingLanguage
    status: ParseStatus
    metadata: ParseMetadata
    warnings: List[ParseWarning] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    file_path: Optional[Path] = None
    content_hash: Optional[str] = None
    ast_json: Optional[str] = None
    symbol_table: Optional[Dict[str, Any]] = None
    comments: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self) -> None:
        """Validar el resultado del parsing."""
        if self.tree is None:
            raise ValueError("El árbol AST no puede ser None")
        
        if self.language is None:
            raise ValueError("El lenguaje de programación no puede ser None")
        
        if self.status is None:
            raise ValueError("El estado del parsing no puede ser None")
        
        if self.metadata is None:
            raise ValueError("Los metadatos no pueden ser None")
    
    @property
    def is_successful(self) -> bool:
        """Verifica si el parsing fue exitoso."""
        return self.status == ParseStatus.SUCCESS
    
    @property
    def is_partial(self) -> bool:
        """Verifica si el parsing fue parcial."""
        return self.status == ParseStatus.PARTIAL
    
    @property
    def is_failed(self) -> bool:
        """Verifica si el parsing falló."""
        return self.status == ParseStatus.FAILED
    
    @property
    def is_cached(self) -> bool:
        """Verifica si el resultado proviene de caché."""
        return self.status == ParseStatus.CACHED
    
    @property
    def has_errors(self) -> bool:
        """Verifica si hay errores de parsing."""
        return len(self.errors) > 0 or self.metadata.error_count > 0
    
    @property
    def has_warnings(self) -> bool:
        """Verifica si hay advertencias de parsing."""
        return len(self.warnings) > 0 or self.metadata.warning_count > 0
    
    def get_error_summary(self) -> str:
        """Obtiene un resumen de los errores."""
        if not self.has_errors:
            return "No hay errores"
        
        error_count = self.metadata.error_count
        error_messages = len(self.errors)
        
        if error_count > 0 and error_messages > 0:
            return f"{error_count} errores de sintaxis, {error_messages} mensajes de error"
        elif error_count > 0:
            return f"{error_count} errores de sintaxis"
        else:
            return f"{error_messages} mensajes de error"
    
    def get_warning_summary(self) -> str:
        """Obtiene un resumen de las advertencias."""
        if not self.has_warnings:
            return "No hay advertencias"
        
        warning_count = self.metadata.warning_count
        warning_types = len(self.warnings)
        
        if warning_count > 0 and warning_types > 0:
            return f"{warning_count} advertencias, {warning_types} tipos de advertencia"
        elif warning_count > 0:
            return f"{warning_count} advertencias"
        else:
            return f"{warning_types} tipos de advertencia"
    
    def get_performance_summary(self) -> str:
        """Obtiene un resumen del rendimiento del parsing."""
        duration = self.metadata.parse_duration_ms
        size_mb = self.metadata.file_size_bytes / (1024 * 1024)
        lines_per_second = (self.metadata.line_count / (duration / 1000)) if duration > 0 else 0
        
        return (
            f"Parsing en {duration}ms, "
            f"archivo de {size_mb:.2f}MB, "
            f"{lines_per_second:.0f} líneas/segundo"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario."""
        return {
            "language": self.language.value,
            "status": self.status.value,
            "file_path": str(self.file_path) if self.file_path else None,
            "content_hash": self.content_hash,
            "metadata": {
                "file_size_bytes": self.metadata.file_size_bytes,
                "line_count": self.metadata.line_count,
                "character_count": self.metadata.character_count,
                "encoding": self.metadata.encoding,
                "has_syntax_errors": self.metadata.has_syntax_errors,
                "complexity_estimate": self.metadata.complexity_estimate,
                "parsed_at": self.metadata.parsed_at.isoformat(),
                "parse_duration_ms": self.metadata.parse_duration_ms,
                "node_count": self.metadata.node_count,
                "error_count": self.metadata.error_count,
                "warning_count": self.metadata.warning_count,
                "cache_hit": self.metadata.cache_hit,
                "incremental_parse": self.metadata.incremental_parse,
            },
            "warnings": [warning.value for warning in self.warnings],
            "errors": self.errors,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "is_successful": self.is_successful,
            "is_partial": self.is_partial,
            "is_failed": self.is_failed,
            "is_cached": self.is_cached,
        }
    
    def __str__(self) -> str:
        """Representación string del resultado."""
        status_str = self.status.value.upper()
        language_str = self.language.get_name()
        
        if self.file_path:
            return f"ParseResult({status_str}, {language_str}, {self.file_path.name})"
        else:
            return f"ParseResult({status_str}, {language_str}, content)"
    
    def __repr__(self) -> str:
        """Representación de debug del resultado."""
        return (
            f"ParseResult("
            f"tree={type(self.tree).__name__}, "
            f"language={self.language}, "
            f"status={self.status}, "
            f"file_path={self.file_path}, "
            f"metadata={self.metadata}"
            f")"
        )
