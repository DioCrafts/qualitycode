"""Implementación del patrón Result para manejo de errores."""

from typing import TypeVar, Generic, Union, Optional, Callable
from dataclasses import dataclass


T = TypeVar('T')
E = TypeVar('E')


@dataclass(frozen=True)
class Result(Generic[T, E]):
    """
    Implementación del patrón Result para manejo funcional de errores.
    
    Un Result puede ser Success(data) o Failure(error).
    Esto permite manejar errores de forma explícita sin usar excepciones.
    """
    
    _data: Optional[T] = None
    _error: Optional[E] = None
    _is_success: bool = False
    
    def __post_init__(self):
        """Validar que solo uno de data o error esté presente."""
        if self._is_success and self._data is None:
            raise ValueError("Result de éxito debe tener data")
        if not self._is_success and self._error is None:
            raise ValueError("Result de error debe tener error")
        if self._is_success and self._error is not None:
            raise ValueError("Result de éxito no puede tener error")
        if not self._is_success and self._data is not None:
            raise ValueError("Result de error no puede tener data")
    
    @classmethod
    def success(cls, data: T) -> "Result[T, E]":
        """Crear un Result de éxito."""
        return cls(_data=data, _is_success=True)
    
    @classmethod
    def failure(cls, error: E) -> "Result[T, E]":
        """Crear un Result de error."""
        return cls(_error=error, _is_success=False)
    
    def is_success(self) -> bool:
        """Verificar si el Result es de éxito."""
        return self._is_success
    
    def is_failure(self) -> bool:
        """Verificar si el Result es de error."""
        return not self._is_success
    
    @property
    def data(self) -> T:
        """Obtener los datos (solo si es éxito)."""
        if not self.is_success():
            raise ValueError("No se puede obtener data de un Result de error")
        return self._data
    
    @property
    def error(self) -> E:
        """Obtener el error (solo si es error)."""
        if not self.is_failure():
            raise ValueError("No se puede obtener error de un Result de éxito")
        return self._error
    
    def map(self, func: Callable[[T], 'U']) -> "Result['U', E]":
        """
        Aplicar una función a los datos si es éxito.
        Si es error, devolver el mismo error.
        """
        if self.is_success():
            try:
                new_data = func(self.data)
                return Result.success(new_data)
            except Exception as e:
                return Result.failure(e)
        else:
            return Result.failure(self.error)
    
    def flat_map(self, func: Callable[[T], "Result['U', E]"]) -> "Result['U', E]":
        """
        Aplicar una función que devuelve Result a los datos si es éxito.
        Si es error, devolver el mismo error.
        """
        if self.is_success():
            try:
                return func(self.data)
            except Exception as e:
                return Result.failure(e)
        else:
            return Result.failure(self.error)
    
    def map_error(self, func: Callable[[E], 'F']) -> "Result[T, 'F']":
        """
        Aplicar una función al error si es error.
        Si es éxito, devolver los mismos datos.
        """
        if self.is_failure():
            try:
                new_error = func(self.error)
                return Result.failure(new_error)
            except Exception as e:
                return Result.failure(e)
        else:
            return Result.success(self.data)
    
    def unwrap(self) -> T:
        """
        Obtener los datos o lanzar excepción si es error.
        ¡Usar con cuidado!
        """
        if self.is_failure():
            if isinstance(self.error, Exception):
                raise self.error
            else:
                raise RuntimeError(f"Result falló con error: {self.error}")
        return self.data
    
    def unwrap_or(self, default: T) -> T:
        """Obtener los datos o un valor por defecto si es error."""
        if self.is_success():
            return self.data
        else:
            return default
    
    def unwrap_or_else(self, func: Callable[[E], T]) -> T:
        """Obtener los datos o ejecutar una función con el error."""
        if self.is_success():
            return self.data
        else:
            return func(self.error)
    
    def match(self, on_success: Callable[[T], 'U'], on_failure: Callable[[E], 'U']) -> 'U':
        """Pattern matching sobre el Result."""
        if self.is_success():
            return on_success(self.data)
        else:
            return on_failure(self.error)
    
    def __str__(self) -> str:
        """Representación en string."""
        if self.is_success():
            return f"Success({self.data})"
        else:
            return f"Failure({self.error})"
    
    def __repr__(self) -> str:
        """Representación detallada."""
        return self.__str__()


# Aliases para convenience
Success = Result.success
Failure = Result.failure


# Funciones helper
def wrap_exception(func: Callable[[], T]) -> Result[T, Exception]:
    """Envolver una función que puede lanzar excepciones en un Result."""
    try:
        return Success(func())
    except Exception as e:
        return Failure(e)


async def wrap_async_exception(func: Callable[[], T]) -> Result[T, Exception]:
    """Envolver una función async que puede lanzar excepciones en un Result."""
    try:
        result = await func()
        return Success(result)
    except Exception as e:
        return Failure(e)


def collect_results(results: list[Result[T, E]]) -> Result[list[T], E]:
    """
    Recopilar una lista de Results en un Result de lista.
    Si alguno es error, devuelve el primer error encontrado.
    """
    data_list = []
    for result in results:
        if result.is_failure():
            return Failure(result.error)
        data_list.append(result.data)
    
    return Success(data_list)


def partition_results(results: list[Result[T, E]]) -> tuple[list[T], list[E]]:
    """
    Separar una lista de Results en éxitos y errores.
    Devuelve (lista_de_datos, lista_de_errores).
    """
    successes = []
    failures = []
    
    for result in results:
        if result.is_success():
            successes.append(result.data)
        else:
            failures.append(result.error)
    
    return successes, failures
