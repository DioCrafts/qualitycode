"""
Custom exceptions for fix generation system.
"""
from typing import List, Optional, Any


class FixGenerationError(Exception):
    """Base exception for fix generation errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class RefactoringError(FixGenerationError):
    """Exception for refactoring-related errors."""
    pass


class ValidationError(FixGenerationError):
    """Exception for validation errors."""
    def __init__(self, message: str, errors: List[str] = None, warnings: List[str] = None):
        super().__init__(message)
        self.errors = errors or []
        self.warnings = warnings or []


class ApplicationError(FixGenerationError):
    """Exception for fix application errors."""
    pass


class ModelLoadError(FixGenerationError):
    """Exception for model loading errors."""
    pass


class GenerationTimeoutError(FixGenerationError):
    """Exception for generation timeout."""
    pass


class UnsupportedLanguageError(FixGenerationError):
    """Exception for unsupported programming languages."""
    pass


class NoValidCandidatesError(FixGenerationError):
    """Exception when no valid fix candidates are generated."""
    pass


class RefactoringNotPossibleError(RefactoringError):
    """Exception when refactoring is not possible."""
    pass


class UnsafeRefactoringError(RefactoringError):
    """Exception when refactoring is deemed unsafe."""
    def __init__(self, message: str, risks: List[str]):
        super().__init__(message)
        self.risks = risks
