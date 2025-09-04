"""
Utilidades para CodeAnt Agent.
"""

from .result import Result, Success, Failure
from .error import (
    BaseError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ConflictError,
    InfrastructureError,
    RepositoryError,
    ExternalServiceError,
    ParsingError,
    AnalysisError,
    InternalError
)

__all__ = [
    "Result",
    "Success", 
    "Failure",
    "BaseError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "ConflictError",
    "InfrastructureError",
    "RepositoryError",
    "ExternalServiceError",
    "ParsingError",
    "AnalysisError",
    "InternalError"
]
