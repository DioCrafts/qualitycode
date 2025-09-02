"""
Sistema de manejo de errores para CodeAnt Agent.

Este módulo implementa:
- Errores específicos del dominio
- Result pattern para operaciones que pueden fallar
- Manejo centralizado de errores
- Trazabilidad de errores
"""

from typing import TypeVar, Generic, Union, Optional, Any, Dict
from enum import Enum
import traceback
from datetime import datetime

from .logging import get_logger

# Type variables para el Result pattern
T = TypeVar('T')
E = TypeVar('E', bound='BaseError')


class ErrorSeverity(Enum):
    """Niveles de severidad de errores."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categorías de errores."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    INFRASTRUCTURE = "infrastructure"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL = "internal"
    PARSING = "parsing"
    ANALYSIS = "analysis"


class BaseError(Exception):
    """Clase base para todos los errores de la aplicación."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.original_error = original_error
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        self.traceback = traceback.format_exc()
        
        # Log del error
        self._log_error()
    
    def _log_error(self) -> None:
        """Log del error con contexto completo."""
        logger = get_logger("error")
        log_data = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.original_error:
            log_data["original_error"] = str(self.original_error)
            log_data["original_error_type"] = self.original_error.__class__.__name__
        
        if self.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.error("Critical error occurred", **log_data)
        else:
            logger.warning("Error occurred", **log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el error a diccionario para serialización."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback if self.severity == ErrorSeverity.CRITICAL else None
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


# Errores específicos del dominio
class ValidationError(BaseError):
    """Error de validación de datos."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            details=details,
            **kwargs
        )


class AuthenticationError(BaseError):
    """Error de autenticación."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class AuthorizationError(BaseError):
    """Error de autorización."""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class NotFoundError(BaseError):
    """Error cuando no se encuentra un recurso."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        **kwargs
    ):
        message = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(
            message=message,
            category=ErrorCategory.NOT_FOUND,
            severity=ErrorSeverity.LOW,
            details={
                'resource_type': resource_type,
                'resource_id': resource_id
            },
            **kwargs
        )


class ConflictError(BaseError):
    """Error de conflicto (ej: recurso ya existe)."""
    
    def __init__(self, message: str = "Resource conflict", **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFLICT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class InfrastructureError(BaseError):
    """Error de infraestructura (DB, Redis, etc.)."""
    
    def __init__(self, message: str, service: str = None, **kwargs):
        details = {'service': service} if service else {}
        super().__init__(
            message=message,
            category=ErrorCategory.INFRASTRUCTURE,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class RepositoryError(InfrastructureError):
    """Error específico de repositorios de datos."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message=message, service="repository", **kwargs)


class ExternalServiceError(BaseError):
    """Error de servicio externo."""
    
    def __init__(
        self,
        message: str,
        service: str,
        status_code: Optional[int] = None,
        **kwargs
    ):
        details = {'service': service}
        if status_code:
            details['status_code'] = status_code
        
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class ParsingError(BaseError):
    """Error durante el parsing de código."""
    
    def __init__(
        self,
        message: str,
        language: str,
        file_path: Optional[str] = None,
        line: Optional[int] = None,
        **kwargs
    ):
        details = {'language': language}
        if file_path:
            details['file_path'] = file_path
        if line:
            details['line'] = line
        
        super().__init__(
            message=message,
            category=ErrorCategory.PARSING,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class AnalysisError(BaseError):
    """Error durante el análisis de código."""
    
    def __init__(
        self,
        message: str,
        analysis_type: str,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.ANALYSIS,
            severity=ErrorSeverity.MEDIUM,
            details={'analysis_type': analysis_type},
            **kwargs
        )


# Result pattern para operaciones que pueden fallar
class Result(Generic[T, E]):
    """
    Result pattern para manejar operaciones que pueden fallar.
    
    T: Tipo del valor de éxito
    E: Tipo del error (debe heredar de BaseError)
    """
    
    def __init__(self, success: bool, data: Optional[T] = None, error: Optional[E] = None):
        if success and error is not None:
            raise ValueError("Result cannot be successful and have an error")
        if not success and error is None:
            raise ValueError("Result must have an error when not successful")
        
        self.success = success
        self.data = data
        self.error = error
    
    @classmethod
    def success(cls, data: T) -> 'Result[T, E]':
        """Crea un resultado exitoso."""
        return cls(success=True, data=data)
    
    @classmethod
    def failure(cls, error: E) -> 'Result[T, E]':
        """Crea un resultado fallido."""
        return cls(success=False, error=error)
    
    def is_success(self) -> bool:
        """Retorna True si el resultado es exitoso."""
        return self.success
    
    def is_failure(self) -> bool:
        """Retorna True si el resultado es fallido."""
        return not self.success
    
    def unwrap(self) -> T:
        """
        Retorna el valor de éxito.
        
        Raises:
            ValueError: Si el resultado es fallido
        """
        if not self.success:
            raise ValueError(f"Cannot unwrap failed result: {self.error}")
        return self.data
    
    def unwrap_or(self, default: T) -> T:
        """Retorna el valor de éxito o un valor por defecto."""
        return self.data if self.success else default
    
    def map(self, func) -> 'Result':
        """Aplica una función al valor de éxito si existe."""
        if self.success:
            try:
                new_data = func(self.data)
                return Result.success(new_data)
            except Exception as e:
                return Result.failure(InternalError(f"Error in map function: {str(e)}"))
        return self
    
    def flat_map(self, func) -> 'Result':
        """Aplica una función que retorna un Result."""
        if self.success:
            try:
                return func(self.data)
            except Exception as e:
                return Result.failure(InternalError(f"Error in flat_map function: {str(e)}"))
        return self
    
    def __repr__(self) -> str:
        if self.success:
            return f"Result.success({self.data})"
        else:
            return f"Result.failure({self.error})"


# Error interno genérico
class InternalError(BaseError):
    """Error interno de la aplicación."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.INTERNAL,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


# Funciones de utilidad para manejo de errores
def handle_errors(func):
    """
    Decorator para manejo automático de errores.
    
    Convierte excepciones en Results fallidos.
    """
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return Result.success(result)
        except BaseError as e:
            return Result.failure(e)
        except Exception as e:
            # Convertir excepciones genéricas en InternalError
            internal_error = InternalError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                original_error=e
            )
            return Result.failure(internal_error)
    
    async def async_wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            return Result.success(result)
        except BaseError as e:
            return Result.failure(e)
        except Exception as e:
            internal_error = InternalError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                original_error=e
            )
            return Result.failure(internal_error)
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper


def ensure_result(func):
    """
    Decorator que asegura que una función retorne un Result.
    
    Útil para funciones que ya retornan Results pero queremos
    asegurar el tipo de retorno.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not isinstance(result, Result):
            raise TypeError(f"Function {func.__name__} must return a Result")
        return result
    
    async def async_wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        if not isinstance(result, Result):
            raise TypeError(f"Function {func.__name__} must return a Result")
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper


# Import asyncio para el decorator
import asyncio
