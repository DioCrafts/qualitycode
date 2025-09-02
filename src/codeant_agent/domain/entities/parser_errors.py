"""
Entidades para manejo de errores del parser.

Este módulo define las entidades que representan el sistema
de manejo de errores y recovery del parser universal.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from ..value_objects.programming_language import ProgrammingLanguage


class ErrorSeverity(Enum):
    """Niveles de severidad de errores."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Categorías de errores del parser."""
    SYNTAX_ERROR = "syntax_error"
    SEMANTIC_ERROR = "semantic_error"
    LEXICAL_ERROR = "lexical_error"
    PARSER_ERROR = "parser_error"
    LANGUAGE_DETECTION_ERROR = "language_detection_error"
    NORMALIZATION_ERROR = "normalization_error"
    QUERY_ERROR = "query_error"
    CACHE_ERROR = "cache_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    FILE_ERROR = "file_error"
    NETWORK_ERROR = "network_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Estrategias de recovery de errores."""
    SKIP_NODE = "skip_node"
    SKIP_LINE = "skip_line"
    SKIP_BLOCK = "skip_block"
    USE_DEFAULT = "use_default"
    FALLBACK_PARSER = "fallback_parser"
    PARTIAL_PARSE = "partial_parse"
    ERROR_RECOVERY = "error_recovery"
    RETRY = "retry"
    IGNORE = "ignore"
    ABORT = "abort"


@dataclass
class ParserError:
    """Error del parser universal."""
    
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    language: Optional[ProgrammingLanguage] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    byte_offset: Optional[int] = None
    context: Optional[str] = None
    original_error: Optional[Exception] = None
    recovery_strategy: Optional[RecoveryStrategy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validar el error del parser."""
        if not self.message.strip():
            raise ValueError("El mensaje de error no puede estar vacío")
        
        if self.category is None:
            raise ValueError("La categoría de error no puede ser None")
        
        if self.severity is None:
            raise ValueError("La severidad del error no puede ser None")
        
        if self.line_number is not None and self.line_number < 0:
            raise ValueError("El número de línea no puede ser negativo")
        
        if self.column_number is not None and self.column_number < 0:
            raise ValueError("El número de columna no puede ser negativo")
        
        if self.byte_offset is not None and self.byte_offset < 0:
            raise ValueError("El offset de byte no puede ser negativo")
    
    @property
    def has_location(self) -> bool:
        """Verifica si el error tiene información de ubicación."""
        return (
            self.line_number is not None or 
            self.column_number is not None or 
            self.byte_offset is not None
        )
    
    @property
    def is_recoverable(self) -> bool:
        """Verifica si el error es recuperable."""
        return self.recovery_strategy is not None and self.recovery_strategy != RecoveryStrategy.ABORT
    
    @property
    def is_critical(self) -> bool:
        """Verifica si el error es crítico."""
        return self.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]
    
    @property
    def is_warning(self) -> bool:
        """Verifica si el error es solo una advertencia."""
        return self.severity in [ErrorSeverity.INFO, ErrorSeverity.WARNING]
    
    def get_location_summary(self) -> str:
        """Obtiene un resumen de la ubicación del error."""
        if not self.has_location:
            return "ubicación desconocida"
        
        parts = []
        
        if self.file_path:
            parts.append(f"archivo: {self.file_path}")
        
        if self.line_number is not None:
            parts.append(f"línea: {self.line_number}")
        
        if self.column_number is not None:
            parts.append(f"columna: {self.column_number}")
        
        if self.byte_offset is not None:
            parts.append(f"byte: {self.byte_offset}")
        
        return ", ".join(parts)
    
    def get_error_summary(self) -> str:
        """Obtiene un resumen del error."""
        severity_str = self.severity.value.upper()
        category_str = self.category.value.replace('_', ' ').title()
        location_str = self.get_location_summary()
        
        return f"{severity_str} - {category_str} en {location_str}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el error a diccionario."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "language": self.language.value if self.language else None,
            "language_name": self.language.get_name() if self.language else None,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "byte_offset": self.byte_offset,
            "context": self.context,
            "original_error": str(self.original_error) if self.original_error else None,
            "recovery_strategy": self.recovery_strategy.value if self.recovery_strategy else None,
            "has_location": self.has_location,
            "is_recoverable": self.is_recoverable,
            "is_critical": self.is_critical,
            "is_warning": self.is_warning,
            "location_summary": self.get_location_summary(),
            "error_summary": self.get_error_summary(),
            "metadata": self.metadata,
            "occurred_at": self.occurred_at.isoformat(),
        }
    
    def __str__(self) -> str:
        """Representación string del error."""
        return self.get_error_summary()
    
    def __repr__(self) -> str:
        """Representación de debug del error."""
        return (
            f"ParserError("
            f"message='{self.message}', "
            f"category={self.category}, "
            f"severity={self.severity}, "
            f"language={self.language}, "
            f"file_path={self.file_path}, "
            f"line_number={self.line_number}, "
            f"column_number={self.column_number}"
            f")"
        )


@dataclass
class ParseWarning:
    """Advertencia del parser."""
    
    message: str
    category: ErrorCategory
    language: Optional[ProgrammingLanguage] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    context: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validar la advertencia."""
        if not self.message.strip():
            raise ValueError("El mensaje de advertencia no puede estar vacío")
        
        if self.category is None:
            raise ValueError("La categoría de advertencia no puede ser None")
        
        if self.line_number is not None and self.line_number < 0:
            raise ValueError("El número de línea no puede ser negativo")
        
        if self.column_number is not None and self.column_number < 0:
            raise ValueError("El número de columna no puede ser negativo")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la advertencia a diccionario."""
        return {
            "message": self.message,
            "category": self.category.value,
            "language": self.language.value if self.language else None,
            "language_name": self.language.get_name() if self.language else None,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "context": self.context,
            "suggestion": self.suggestion,
            "metadata": self.metadata,
            "occurred_at": self.occurred_at.isoformat(),
        }
    
    def __str__(self) -> str:
        """Representación string de la advertencia."""
        location = f" en {self.file_path}" if self.file_path else ""
        line_info = f" (línea {self.line_number})" if self.line_number else ""
        return f"Advertencia{location}{line_info}: {self.message}"
    
    def __repr__(self) -> str:
        """Representación de debug de la advertencia."""
        return (
            f"ParseWarning("
            f"message='{self.message}', "
            f"category={self.category}, "
            f"language={self.language}, "
            f"file_path={self.file_path}, "
            f"line_number={self.line_number}"
            f")"
        )


@dataclass
class ErrorContext:
    """Contexto de un error del parser."""
    
    file_path: Optional[str] = None
    content_preview: Optional[str] = None
    language: Optional[ProgrammingLanguage] = None
    parser_config: Optional[Dict[str, Any]] = None
    system_info: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_content_preview_lines(self, max_lines: int = 5) -> List[str]:
        """Obtiene las líneas de preview del contenido."""
        if not self.content_preview:
            return []
        
        lines = self.content_preview.split('\n')
        return lines[:max_lines]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el contexto a diccionario."""
        return {
            "file_path": self.file_path,
            "content_preview_lines": self.get_content_preview_lines(),
            "language": self.language.value if self.language else None,
            "language_name": self.language.get_name() if self.language else None,
            "parser_config": self.parser_config,
            "system_info": self.system_info,
            "user_context": self.user_context,
            "additional_data": self.additional_data,
        }
    
    def __str__(self) -> str:
        """Representación string del contexto."""
        if self.file_path:
            return f"ErrorContext({self.file_path})"
        elif self.language:
            return f"ErrorContext({self.language.get_name()})"
        else:
            return "ErrorContext(unknown)"
    
    def __repr__(self) -> str:
        """Representación de debug del contexto."""
        return (
            f"ErrorContext("
            f"file_path={self.file_path}, "
            f"language={self.language}, "
            f"parser_config={self.parser_config}, "
            f"system_info={self.system_info}"
            f")"
        )


@dataclass
class ErrorReport:
    """Reporte completo de errores del parser."""
    
    errors: List[ParserError] = field(default_factory=list)
    warnings: List[ParseWarning] = field(default_factory=list)
    context: Optional[ErrorContext] = None
    summary: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def error_count(self) -> int:
        """Obtiene el número de errores."""
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        """Obtiene el número de advertencias."""
        return len(self.warnings)
    
    @property
    def total_issues(self) -> int:
        """Obtiene el número total de problemas."""
        return self.error_count + self.warning_count
    
    @property
    def has_errors(self) -> bool:
        """Verifica si hay errores."""
        return self.error_count > 0
    
    @property
    def has_warnings(self) -> bool:
        """Verifica si hay advertencias."""
        return self.warning_count > 0
    
    @property
    def has_issues(self) -> bool:
        """Verifica si hay problemas."""
        return self.total_issues > 0
    
    @property
    def critical_error_count(self) -> int:
        """Obtiene el número de errores críticos."""
        return len([e for e in self.errors if e.is_critical])
    
    @property
    def recoverable_error_count(self) -> int:
        """Obtiene el número de errores recuperables."""
        return len([e for e in self.errors if e.is_recoverable])
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ParserError]:
        """Obtiene errores por categoría."""
        return [e for e in self.errors if e.category == category]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ParserError]:
        """Obtiene errores por severidad."""
        return [e for e in self.errors if e.severity == severity]
    
    def get_errors_by_language(self, language: ProgrammingLanguage) -> List[ParserError]:
        """Obtiene errores por lenguaje."""
        return [e for e in self.errors if e.language == language]
    
    def get_errors_by_file(self, file_path: str) -> List[ParserError]:
        """Obtiene errores por archivo."""
        return [e for e in self.errors if e.file_path == file_path]
    
    def add_error(self, error: ParserError) -> None:
        """Agrega un error al reporte."""
        self.errors.append(error)
    
    def add_warning(self, warning: ParseWarning) -> None:
        """Agrega una advertencia al reporte."""
        self.warnings.append(warning)
    
    def get_summary(self) -> str:
        """Genera un resumen del reporte."""
        if not self.has_issues:
            return "No hay problemas detectados"
        
        error_summary = f"{self.error_count} error{'es' if self.error_count != 1 else ''}"
        warning_summary = f"{self.warning_count} advertencia{'s' if self.warning_count != 1 else ''}"
        
        if self.has_errors and self.has_warnings:
            summary = f"{error_summary} y {warning_summary}"
        elif self.has_errors:
            summary = error_summary
        else:
            summary = warning_summary
        
        if self.critical_error_count > 0:
            summary += f" ({self.critical_error_count} crítico{'s' if self.critical_error_count != 1 else ''})"
        
        if self.recoverable_error_count > 0:
            summary += f" ({self.recoverable_error_count} recuperable{'s' if self.recoverable_error_count != 1 else ''})"
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el reporte a diccionario."""
        return {
            "errors": [error.to_dict() for error in self.errors],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "context": self.context.to_dict() if self.context else None,
            "summary": self.get_summary(),
            "statistics": {
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "total_issues": self.total_issues,
                "critical_error_count": self.critical_error_count,
                "recoverable_error_count": self.recoverable_error_count,
                "has_errors": self.has_errors,
                "has_warnings": self.has_warnings,
                "has_issues": self.has_issues,
            },
            "generated_at": self.generated_at.isoformat(),
        }
    
    def __str__(self) -> str:
        """Representación string del reporte."""
        return f"ErrorReport({self.get_summary()})"
    
    def __repr__(self) -> str:
        """Representación de debug del reporte."""
        return (
            f"ErrorReport("
            f"errors={self.errors}, "
            f"warnings={self.warnings}, "
            f"context={self.context}, "
            f"generated_at={self.generated_at}"
            f")"
        )
