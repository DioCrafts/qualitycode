"""
Utilidades varias para la aplicación.
"""

# Importar clases de error para disponibilidad global
from .error import (
    BaseError, ValidationError, AuthenticationError, 
    AuthorizationError, NotFoundError, ConflictError,
    ExternalServiceError, Result
)