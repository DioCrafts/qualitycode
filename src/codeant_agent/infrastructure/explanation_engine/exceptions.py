"""
Exceptions for the Explanation Engine.
"""
from typing import Optional, Dict, Any


class ExplanationError(Exception):
    """Base exception for explanation engine errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class TemplateError(ExplanationError):
    """Exception for template rendering errors."""
    pass


class LanguageAdapterError(ExplanationError):
    """Exception for language adaptation errors."""
    pass


class AudienceAdapterError(ExplanationError):
    """Exception for audience adaptation errors."""
    pass


class ContentGenerationError(ExplanationError):
    """Exception for content generation errors."""
    pass


class VisualizationError(ExplanationError):
    """Exception for visualization generation errors."""
    pass


class InteractiveElementError(ExplanationError):
    """Exception for interactive element errors."""
    pass


class QuestionAnswerError(ExplanationError):
    """Exception for question answering errors."""
    pass


class UnsupportedLanguageError(ExplanationError):
    """Exception for unsupported languages."""
    pass


class UnsupportedAudienceError(ExplanationError):
    """Exception for unsupported audiences."""
    pass


class ContentTooLongError(ExplanationError):
    """Exception for content exceeding maximum length."""
    pass
