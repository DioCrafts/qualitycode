"""
Utilidades para manejo de errores y resultados.
"""
from typing import Generic, TypeVar, Union, Optional

T = TypeVar('T')
E = TypeVar('E', bound=Exception)

class BaseError(Exception):
    """Clase base para errores específicos de la aplicación."""
    pass

class ValidationError(BaseError):
    """Error de validación."""
    pass

class AuthenticationError(BaseError):
    """Error de autenticación."""
    pass

class AuthorizationError(BaseError):
    """Error de autorización."""
    pass

class NotFoundError(BaseError):
    """Error cuando un recurso no se encuentra."""
    pass

class ConflictError(BaseError):
    """Error cuando hay un conflicto (p.ej., recurso ya existe)."""
    pass

class ExternalServiceError(BaseError):
    """Error al comunicarse con un servicio externo."""
    pass

class Result(Generic[T, E]):
    """
    Clase de resultado que puede contener un valor exitoso o un error.
    """
    
    def __init__(self, success: bool, data: Optional[T] = None, error: Optional[E] = None):
        """
        Inicializar resultado.
        
        Args:
            success: Indica si la operación fue exitosa
            data: Datos en caso de éxito
            error: Error en caso de fallo
        """
        self.success = success
        self._data = data
        self._error = error
    
    @property
    def data(self) -> T:
        """
        Obtener datos del resultado.
        
        Returns:
            Los datos
            
        Raises:
            ValueError: Si no hay datos (resultado fallido)
        """
        if not self.success:
            raise ValueError("Cannot access data on failure result")
        return self._data
    
    @property
    def error(self) -> E:
        """
        Obtener error del resultado.
        
        Returns:
            El error
            
        Raises:
            ValueError: Si no hay error (resultado exitoso)
        """
        if self.success:
            raise ValueError("Cannot access error on success result")
        return self._error
    
    @staticmethod
    def success(data: T) -> 'Result[T, E]':
        """
        Crear un resultado exitoso.
        
        Args:
            data: Datos del resultado
            
        Returns:
            Resultado exitoso
        """
        return Result(True, data=data)
    
    @staticmethod
    def failure(error: E) -> 'Result[T, E]':
        """
        Crear un resultado fallido.
        
        Args:
            error: Error del resultado
            
        Returns:
            Resultado fallido
        """
        return Result(False, error=error)