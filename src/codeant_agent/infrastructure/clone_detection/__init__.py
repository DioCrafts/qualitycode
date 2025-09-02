"""
Implementaciones de infraestructura para detección de código duplicado.

Este módulo contiene las implementaciones concretas de los detectores
de código duplicado y similitud.
"""

from .exact_clone_detector import ExactCloneDetector
from .structural_clone_detector import StructuralCloneDetector  
from .semantic_clone_detector import SemanticCloneDetector
from .cross_language_detector import CrossLanguageCloneDetector
from .similarity_calculator import SimilarityCalculator
from .refactoring_suggester import RefactoringSuggester
from .clone_detector import CloneDetector

__all__ = [
    "ExactCloneDetector",
    "StructuralCloneDetector",
    "SemanticCloneDetector", 
    "CrossLanguageCloneDetector",
    "SimilarityCalculator",
    "RefactoringSuggester",
    "CloneDetector",
]
