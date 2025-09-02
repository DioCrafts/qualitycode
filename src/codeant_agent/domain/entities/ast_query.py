"""
Entidades para el sistema de queries sobre ASTs.

Este módulo define las entidades que representan el sistema
de consultas sobre Abstract Syntax Trees.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from ..value_objects.programming_language import ProgrammingLanguage


class QueryType(Enum):
    """Tipos de queries AST."""
    PATTERN_MATCH = "pattern_match"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    METRIC = "metric"
    CUSTOM = "custom"


class QueryStatus(Enum):
    """Estado de una query AST."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class QueryRequest:
    """Solicitud de query AST."""
    
    query_string: str
    language: ProgrammingLanguage
    query_type: QueryType = QueryType.PATTERN_MATCH
    timeout_seconds: int = 30
    max_results: int = 1000
    include_metadata: bool = True
    include_context: bool = True
    enable_caching: bool = True
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validar la solicitud de query."""
        if not self.query_string.strip():
            raise ValueError("La query string no puede estar vacía")
        
        if self.language is None:
            raise ValueError("El lenguaje de programación no puede ser None")
        
        if self.timeout_seconds <= 0:
            raise ValueError("El timeout debe ser mayor a 0")
        
        if self.max_results <= 0:
            raise ValueError("El número máximo de resultados debe ser mayor a 0")
    
    def get_query_hash(self) -> str:
        """Genera un hash único para la query."""
        import hashlib
        
        query_data = f"{self.query_string}:{self.language.value}:{self.query_type.value}"
        return hashlib.md5(query_data.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la solicitud a diccionario."""
        return {
            "query_string": self.query_string,
            "language": self.language.value,
            "language_name": self.language.get_name(),
            "query_type": self.query_type.value,
            "timeout_seconds": self.timeout_seconds,
            "max_results": self.max_results,
            "include_metadata": self.include_metadata,
            "include_context": self.include_context,
            "enable_caching": self.enable_caching,
            "custom_parameters": self.custom_parameters,
            "query_hash": self.get_query_hash(),
        }
    
    def __str__(self) -> str:
        """Representación string de la solicitud."""
        return f"QueryRequest({self.query_type.value}, {self.language.get_name()}, {len(self.query_string)} chars)"
    
    def __repr__(self) -> str:
        """Representación de debug de la solicitud."""
        return (
            f"QueryRequest("
            f"query_string='{self.query_string[:50]}...', "
            f"language={self.language}, "
            f"query_type={self.query_type}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"max_results={self.max_results}"
            f")"
        )


@dataclass
class QueryCapture:
    """Captura de una query AST."""
    
    node_type: str
    text: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    start_byte: int
    end_byte: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validar la captura."""
        if not self.node_type:
            raise ValueError("El tipo de nodo no puede estar vacío")
        
        if self.start_line < 0:
            raise ValueError("La línea de inicio no puede ser negativa")
        
        if self.start_column < 0:
            raise ValueError("La columna de inicio no puede ser negativa")
        
        if self.end_line < 0:
            raise ValueError("La línea de fin no puede ser negativa")
        
        if self.end_column < 0:
            raise ValueError("La columna de fin no puede ser negativa")
        
        if self.start_byte < 0:
            raise ValueError("El byte de inicio no puede ser negativo")
        
        if self.end_byte < 0:
            raise ValueError("El byte de fin no puede ser negativo")
        
        if self.end_line < self.start_line:
            raise ValueError("La línea de fin debe ser mayor o igual a la de inicio")
        
        if self.end_line == self.start_line and self.end_column < self.start_column:
            raise ValueError("En la misma línea, la columna de fin debe ser mayor o igual a la de inicio")
    
    @property
    def line_count(self) -> int:
        """Calcula el número de líneas que ocupa la captura."""
        return self.end_line - self.start_line + 1
    
    @property
    def column_span(self) -> int:
        """Calcula el span de columnas."""
        if self.start_line == self.end_line:
            return self.end_column - self.start_column
        else:
            return self.end_column
    
    @property
    def byte_span(self) -> int:
        """Calcula el span de bytes."""
        return self.end_byte - self.start_byte
    
    def get_position_summary(self) -> str:
        """Obtiene un resumen de la posición."""
        if self.start_line == self.end_line:
            if self.start_column == self.end_column:
                return f"línea {self.start_line}, columna {self.start_column}"
            else:
                return f"línea {self.start_line}, columnas {self.start_column}-{self.end_column}"
        else:
            return f"líneas {self.start_line}-{self.end_line}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la captura a diccionario."""
        return {
            "node_type": self.node_type,
            "text": self.text,
            "position": {
                "start_line": self.start_line,
                "start_column": self.start_column,
                "end_line": self.end_line,
                "end_column": self.end_column,
                "start_byte": self.start_byte,
                "end_byte": self.end_byte,
                "line_count": self.line_count,
                "column_span": self.column_span,
                "byte_span": self.byte_span,
                "summary": self.get_position_summary(),
            },
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        """Representación string de la captura."""
        return f"QueryCapture({self.node_type}, {self.get_position_summary()})"
    
    def __repr__(self) -> str:
        """Representación de debug de la captura."""
        return (
            f"QueryCapture("
            f"node_type='{self.node_type}', "
            f"text='{self.text[:30]}...', "
            f"start_line={self.start_line}, "
            f"start_column={self.start_column}, "
            f"end_line={self.end_line}, "
            f"end_column={self.end_column}"
            f")"
        )


@dataclass
class QueryMatch:
    """Coincidencia de una query AST."""
    
    pattern_index: int
    captures: Dict[str, QueryCapture]
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validar la coincidencia."""
        if self.pattern_index < 0:
            raise ValueError("El índice del patrón no puede ser negativo")
        
        if not self.captures:
            raise ValueError("Debe haber al menos una captura")
        
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("El score debe estar entre 0.0 y 1.0")
    
    @property
    def capture_count(self) -> int:
        """Obtiene el número de capturas."""
        return len(self.captures)
    
    @property
    def primary_capture(self) -> Optional[QueryCapture]:
        """Obtiene la captura principal (si existe)."""
        if "primary" in self.captures:
            return self.captures["primary"]
        elif "main" in self.captures:
            return self.captures["main"]
        elif "name" in self.captures:
            return self.captures["name"]
        elif self.captures:
            # Retornar la primera captura disponible
            return next(iter(self.captures.values()))
        return None
    
    def get_capture_by_name(self, name: str) -> Optional[QueryCapture]:
        """Obtiene una captura por nombre."""
        return self.captures.get(name)
    
    def get_captures_by_type(self, node_type: str) -> List[QueryCapture]:
        """Obtiene capturas por tipo de nodo."""
        return [capture for capture in self.captures.values() if capture.node_type == node_type]
    
    def get_text_summary(self) -> str:
        """Obtiene un resumen del texto capturado."""
        if not self.captures:
            return "Sin capturas"
        
        texts = [capture.text for capture in self.captures.values()]
        if len(texts) == 1:
            return texts[0]
        else:
            return f"{len(texts)} capturas: {', '.join(texts[:3])}{'...' if len(texts) > 3 else ''}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la coincidencia a diccionario."""
        return {
            "pattern_index": self.pattern_index,
            "captures": {
                name: capture.to_dict() for name, capture in self.captures.items()
            },
            "capture_count": self.capture_count,
            "score": self.score,
            "score_percentage": int(self.score * 100),
            "text_summary": self.get_text_summary(),
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        """Representación string de la coincidencia."""
        return f"QueryMatch(pattern_{self.pattern_index}, {self.capture_count} capturas, score={self.score:.2f})"
    
    def __repr__(self) -> str:
        """Representación de debug de la coincidencia."""
        return (
            f"QueryMatch("
            f"pattern_index={self.pattern_index}, "
            f"captures={self.captures}, "
            f"score={self.score}, "
            f"metadata={self.metadata}"
            f")"
        )


@dataclass
class QueryResult:
    """Resultado de una query AST."""
    
    query_request: QueryRequest
    matches: List[QueryMatch]
    status: QueryStatus
    execution_time_ms: int
    node_count_examined: int
    query_name: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    executed_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validar el resultado de la query."""
        if self.query_request is None:
            raise ValueError("La solicitud de query no puede ser None")
        
        if self.matches is None:
            raise ValueError("La lista de coincidencias no puede ser None")
        
        if self.status is None:
            raise ValueError("El estado de la query no puede ser None")
        
        if self.execution_time_ms < 0:
            raise ValueError("El tiempo de ejecución no puede ser negativo")
        
        if self.node_count_examined < 0:
            raise ValueError("El número de nodos examinados no puede ser negativo")
    
    @property
    def match_count(self) -> int:
        """Obtiene el número de coincidencias."""
        return len(self.matches)
    
    @property
    def is_successful(self) -> bool:
        """Verifica si la query fue exitosa."""
        return self.status == QueryStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Verifica si la query falló."""
        return self.status in [QueryStatus.FAILED, QueryStatus.TIMEOUT, QueryStatus.CANCELLED]
    
    @property
    def has_matches(self) -> bool:
        """Verifica si hay coincidencias."""
        return self.match_count > 0
    
    @property
    def has_warnings(self) -> bool:
        """Verifica si hay advertencias."""
        return len(self.warnings) > 0
    
    def get_matches_by_score(self, min_score: float = 0.0) -> List[QueryMatch]:
        """Obtiene coincidencias por score mínimo."""
        return [match for match in self.matches if match.score >= min_score]
    
    def get_matches_by_pattern(self, pattern_index: int) -> List[QueryMatch]:
        """Obtiene coincidencias por índice de patrón."""
        return [match for match in self.matches if match.pattern_index == pattern_index]
    
    def get_performance_summary(self) -> str:
        """Obtiene un resumen del rendimiento."""
        duration = self.execution_time_ms
        nodes_per_second = (self.node_count_examined / (duration / 1000)) if duration > 0 else 0
        
        return (
            f"Query ejecutada en {duration}ms, "
            f"{self.node_count_examined} nodos examinados, "
            f"{nodes_per_second:.0f} nodos/segundo, "
            f"{self.match_count} coincidencias encontradas"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario."""
        return {
            "query": self.query_request.to_dict(),
            "status": self.status.value,
            "matches": [match.to_dict() for match in self.matches],
            "match_count": self.match_count,
            "execution_time_ms": self.execution_time_ms,
            "node_count_examined": self.node_count_examined,
            "query_name": self.query_name,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "executed_at": self.executed_at.isoformat(),
            "performance_summary": self.get_performance_summary(),
            "is_successful": self.is_successful,
            "is_failed": self.is_failed,
            "has_matches": self.has_matches,
            "has_warnings": self.has_warnings,
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        """Representación string del resultado."""
        status_str = self.status.value.upper()
        return f"QueryResult({status_str}, {self.match_count} matches, {self.execution_time_ms}ms)"
    
    def __repr__(self) -> str:
        """Representación de debug del resultado."""
        return (
            f"QueryResult("
            f"query_request={self.query_request}, "
            f"matches={self.matches}, "
            f"status={self.status}, "
            f"execution_time_ms={self.execution_time_ms}, "
            f"node_count_examined={self.node_count_examined}"
            f")"
        )
