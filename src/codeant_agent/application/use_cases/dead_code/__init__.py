"""
Casos de uso para análisis de código muerto.
"""

from .analyze_file_dead_code_use_case import AnalyzeFileDeadCodeUseCase
from .analyze_project_dead_code_use_case import AnalyzeProjectDeadCodeUseCase

__all__ = [
    'AnalyzeFileDeadCodeUseCase',
    'AnalyzeProjectDeadCodeUseCase',
]
