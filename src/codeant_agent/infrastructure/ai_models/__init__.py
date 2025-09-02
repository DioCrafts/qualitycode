"""
Implementaciones de infraestructura para modelos de IA.

Este módulo contiene las implementaciones concretas del sistema
de integración de IA, embeddings de código y análisis semántico.
"""

from .model_manager import AIModelManager, ModelCache, LoadedModelRegistry
from .embedding_engine import CodeEmbeddingEngine, BatchEmbeddingProcessor
from .code_preprocessor import CodePreprocessor, LanguagePreprocessor
from .vector_store import VectorStore, EmbeddingSearchEngine
from .inference_engine import InferenceEngine, ModelInferenceOptimizer
from .ai_analyzer import AICodeAnalyzer, PatternDetector, AnomalyDetector
from .similarity_engine import SemanticSimilarityEngine, CrossLanguageAnalyzer
from .ai_integration import AIIntegrationManager

__all__ = [
    "AIModelManager",
    "ModelCache",
    "LoadedModelRegistry",
    "CodeEmbeddingEngine",
    "BatchEmbeddingProcessor",
    "CodePreprocessor",
    "LanguagePreprocessor",
    "VectorStore",
    "EmbeddingSearchEngine",
    "InferenceEngine",
    "ModelInferenceOptimizer",
    "AICodeAnalyzer",
    "PatternDetector",
    "AnomalyDetector",
    "SemanticSimilarityEngine",
    "CrossLanguageAnalyzer",
    "AIIntegrationManager",
]
