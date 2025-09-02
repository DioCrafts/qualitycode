"""
Gestor de integración principal del sistema de IA.

Este módulo orquesta todos los componentes de IA y proporciona
una interfaz unificada para el análisis de código.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ...domain.entities.ai_models import (
    AIConfig, EmbeddingConfig, VectorStoreConfig, AIAnalysisConfig,
    AIIntegrationConfig, AISystemStatus, CodeEmbedding, AIAnalysisResult,
    SemanticSearchQuery, SemanticSearchResult, CrossLanguageSimilarity,
    AnalysisType
)
from ...domain.value_objects.programming_language import ProgrammingLanguage
from .model_manager import AIModelManager
from .embedding_engine import CodeEmbeddingEngine  
from .vector_store import VectorStore
from .inference_engine import InferenceEngine
from .code_preprocessor import CodePreprocessor
from .ai_analyzer import AICodeAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class AIIntegrationStatus:
    """Estado de integración del sistema de IA."""
    initialized: bool = False
    components_loaded: Dict[str, bool] = field(default_factory=dict)
    health_score: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def calculate_health_score(self) -> float:
        """Calcula score de salud del sistema."""
        if not self.components_loaded:
            return 0.0
        
        loaded_count = sum(1 for loaded in self.components_loaded.values() if loaded)
        total_count = len(self.components_loaded)
        
        base_score = loaded_count / total_count if total_count > 0 else 0.0
        
        # Penalizar por errores
        error_penalty = min(0.5, len(self.error_messages) * 0.1)
        warning_penalty = min(0.2, len(self.warnings) * 0.05)
        
        self.health_score = max(0.0, base_score - error_penalty - warning_penalty)
        self.last_health_check = datetime.now()
        
        return self.health_score


class AIIntegrationManager:
    """Gestor principal de integración de IA."""
    
    def __init__(self, config: Optional[AIIntegrationConfig] = None):
        """
        Inicializa el gestor de integración.
        
        Args:
            config: Configuración de integración
        """
        self.config = config or AIIntegrationConfig()
        self.status = AIIntegrationStatus()
        
        # Componentes principales
        self.model_manager: Optional[AIModelManager] = None
        self.embedding_engine: Optional[CodeEmbeddingEngine] = None
        self.vector_store: Optional[VectorStore] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.code_preprocessor: Optional[CodePreprocessor] = None
        self.ai_analyzer: Optional[AICodeAnalyzer] = None
        
        # Estado y estadísticas
        self.initialization_time: Optional[datetime] = None
        self.total_analyses: int = 0
        self.successful_analyses: int = 0
        
        logger.info("AIIntegrationManager creado")
    
    async def initialize(self) -> bool:
        """
        Inicializa todos los componentes de IA.
        
        Returns:
            True si la inicialización fue exitosa
        """
        logger.info("🚀 Iniciando integración del sistema de IA...")
        initialization_start = datetime.now()
        
        try:
            # Validar configuración
            config_issues = self.config.validate_config()
            if config_issues:
                for issue in config_issues:
                    self.status.error_messages.append(f"Config issue: {issue}")
                logger.warning(f"Problemas de configuración: {config_issues}")
            
            # Inicializar componentes en orden de dependencia
            await self._initialize_components()
            
            # Verificar inicialización
            self.status.initialized = all(self.status.components_loaded.values())
            
            if self.status.initialized:
                self.initialization_time = datetime.now()
                initialization_time = (initialization_time - initialization_start).total_seconds()
                
                logger.info(f"✅ Sistema de IA inicializado correctamente en {initialization_time:.2f}s")
                
                # Health check inicial
                await self.health_check()
                
                return True
            else:
                logger.error("❌ Fallo en la inicialización del sistema de IA")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error crítico en inicialización: {e}")
            self.status.error_messages.append(f"Critical initialization error: {e}")
            return False
    
    async def _initialize_components(self) -> None:
        """Inicializa componentes individuales."""
        
        # 1. Model Manager
        try:
            logger.info("Inicializando AIModelManager...")
            self.model_manager = AIModelManager(self.config.ai_config)
            await self.model_manager.initialize()
            self.status.components_loaded["model_manager"] = True
            logger.info("✅ AIModelManager inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando AIModelManager: {e}")
            self.status.components_loaded["model_manager"] = False
            self.status.error_messages.append(f"ModelManager error: {e}")
        
        # 2. Code Preprocessor
        try:
            logger.info("Inicializando CodePreprocessor...")
            self.code_preprocessor = CodePreprocessor()
            await self.code_preprocessor.initialize()
            self.status.components_loaded["code_preprocessor"] = True
            logger.info("✅ CodePreprocessor inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando CodePreprocessor: {e}")
            self.status.components_loaded["code_preprocessor"] = False
            self.status.error_messages.append(f"CodePreprocessor error: {e}")
        
        # 3. Inference Engine
        try:
            logger.info("Inicializando InferenceEngine...")
            device = self.config.ai_config.get_effective_device()
            self.inference_engine = InferenceEngine(device, self.config.ai_config)
            await self.inference_engine.initialize()
            self.status.components_loaded["inference_engine"] = True
            logger.info("✅ InferenceEngine inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando InferenceEngine: {e}")
            self.status.components_loaded["inference_engine"] = False
            self.status.error_messages.append(f"InferenceEngine error: {e}")
        
        # 4. Vector Store
        try:
            logger.info("Inicializando VectorStore...")
            self.vector_store = await VectorStore.create(self.config.vector_store_config)
            self.status.components_loaded["vector_store"] = True
            logger.info("✅ VectorStore inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando VectorStore: {e}")
            self.status.components_loaded["vector_store"] = False
            self.status.error_messages.append(f"VectorStore error: {e}")
        
        # 5. Embedding Engine (requiere model_manager)
        try:
            if self.model_manager:
                logger.info("Inicializando CodeEmbeddingEngine...")
                self.embedding_engine = CodeEmbeddingEngine(
                    self.model_manager, 
                    self.config.embedding_config
                )
                await self.embedding_engine.initialize()
                self.status.components_loaded["embedding_engine"] = True
                logger.info("✅ CodeEmbeddingEngine inicializado")
            else:
                raise Exception("Model manager not available")
        except Exception as e:
            logger.error(f"❌ Error inicializando CodeEmbeddingEngine: {e}")
            self.status.components_loaded["embedding_engine"] = False
            self.status.error_messages.append(f"EmbeddingEngine error: {e}")
        
        # 6. AI Analyzer (requiere todos los componentes anteriores)
        try:
            if all([self.model_manager, self.embedding_engine, self.vector_store, self.inference_engine]):
                logger.info("Inicializando AICodeAnalyzer...")
                self.ai_analyzer = AICodeAnalyzer(
                    self.model_manager,
                    self.embedding_engine,
                    self.vector_store,
                    self.inference_engine,
                    self.config.analysis_config
                )
                self.status.components_loaded["ai_analyzer"] = True
                logger.info("✅ AICodeAnalyzer inicializado")
            else:
                raise Exception("Required components not available")
        except Exception as e:
            logger.error(f"❌ Error inicializando AICodeAnalyzer: {e}")
            self.status.components_loaded["ai_analyzer"] = False
            self.status.error_messages.append(f"AIAnalyzer error: {e}")
        
        logger.info(f"Componentes inicializados: {sum(self.status.components_loaded.values())}/{len(self.status.components_loaded)}")
    
    async def analyze_code(
        self, 
        code: str, 
        language: ProgrammingLanguage,
        analysis_types: Optional[List[AnalysisType]] = None
    ) -> AIAnalysisResult:
        """
        Analiza código usando el sistema de IA completo.
        
        Args:
            code: Código fuente
            language: Lenguaje de programación  
            analysis_types: Tipos de análisis a realizar
            
        Returns:
            Resultado del análisis
        """
        if not self.status.initialized or not self.ai_analyzer:
            raise RuntimeError("AI system not properly initialized")
        
        try:
            self.total_analyses += 1
            
            # Ejecutar análisis
            result = await self.ai_analyzer.analyze_code(code, language, analysis_types)
            
            self.successful_analyses += 1
            
            # Almacenar embedding en vector store si se generó
            if result.embeddings and self.vector_store:
                for embedding in result.embeddings:
                    await self.vector_store.store_embedding(embedding)
            
            return result
            
        except Exception as e:
            logger.error(f"Error en análisis de código: {e}")
            # Crear resultado de error
            return AIAnalysisResult(
                code_snippet=code,
                language=language,
                analysis_type=analysis_types[0] if analysis_types else AnalysisType.SEMANTIC_SIMILARITY,
                model_used="error",
                confidence_scores={"error": 0.0}
            )
    
    async def generate_embedding(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> CodeEmbedding:
        """
        Genera embedding para código.
        
        Args:
            code: Código fuente
            language: Lenguaje de programación
            
        Returns:
            Embedding generado
        """
        if not self.embedding_engine:
            raise RuntimeError("Embedding engine not initialized")
        
        return await self.embedding_engine.generate_embedding(code, language)
    
    async def search_similar_code(
        self,
        query: SemanticSearchQuery
    ) -> SemanticSearchResult:
        """
        Busca código similar usando búsqueda semántica.
        
        Args:
            query: Query de búsqueda
            
        Returns:
            Resultado de búsqueda
        """
        if not self.vector_store or not self.vector_store.search_engine:
            raise RuntimeError("Vector store search engine not initialized")
        
        return await self.vector_store.search_engine.semantic_search(query)
    
    async def analyze_cross_language_similarity(
        self,
        code1: str,
        language1: ProgrammingLanguage,
        code2: str,
        language2: ProgrammingLanguage
    ) -> CrossLanguageSimilarity:
        """
        Analiza similitud entre códigos de diferentes lenguajes.
        
        Args:
            code1: Primer código
            language1: Lenguaje del primer código
            code2: Segundo código
            language2: Lenguaje del segundo código
            
        Returns:
            Análisis de similitud cross-language
        """
        if not self.ai_analyzer:
            raise RuntimeError("AI analyzer not initialized")
        
        return await self.ai_analyzer.analyze_cross_language_similarity(
            code1, language1, code2, language2
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Ejecuta health check completo del sistema.
        
        Returns:
            Reporte de salud
        """
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "system_initialized": self.status.initialized,
            "overall_health_score": 0.0,
            "component_status": {},
            "performance_metrics": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Health check por componente
            if self.model_manager:
                model_health = await self.model_manager.health_check()
                health_report["component_status"]["model_manager"] = model_health
            
            if self.vector_store:
                vector_health = await self.vector_store.health_check()
                health_report["component_status"]["vector_store"] = vector_health
            
            if self.inference_engine:
                inference_report = await self.inference_engine.get_performance_report()
                health_report["component_status"]["inference_engine"] = {
                    "status": "healthy" if inference_report["is_processing"] else "inactive",
                    "performance": inference_report["engine_stats"]
                }
            
            # Calcular salud general
            overall_health = self.status.calculate_health_score()
            health_report["overall_health_score"] = overall_health
            
            # Métricas de performance
            if self.total_analyses > 0:
                success_rate = self.successful_analyses / self.total_analyses
                health_report["performance_metrics"] = {
                    "total_analyses": self.total_analyses,
                    "success_rate": success_rate,
                    "initialization_time": self.initialization_time.isoformat() if self.initialization_time else None
                }
            
            # Identificar issues
            health_report["issues"] = self.status.error_messages.copy()
            
            # Generar recomendaciones
            recommendations = []
            
            if overall_health < 0.8:
                recommendations.append("System health below optimal - check component errors")
            
            if not self.status.components_loaded.get("vector_store", False):
                recommendations.append("Vector store not available - semantic search disabled")
            
            if self.total_analyses > 0 and (self.successful_analyses / self.total_analyses) < 0.9:
                recommendations.append("Low analysis success rate - check model compatibility")
            
            health_report["recommendations"] = recommendations
            
        except Exception as e:
            health_report["issues"].append(f"Health check error: {e}")
            logger.error(f"Error en health check: {e}")
        
        return health_report
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema."""
        stats = {
            "initialization": {
                "initialized": self.status.initialized,
                "initialization_time": self.initialization_time.isoformat() if self.initialization_time else None,
                "components_loaded": self.status.components_loaded
            },
            "usage": {
                "total_analyses": self.total_analyses,
                "successful_analyses": self.successful_analyses,
                "success_rate": self.successful_analyses / max(1, self.total_analyses)
            },
            "component_stats": {}
        }
        
        # Estadísticas por componente
        try:
            if self.model_manager:
                stats["component_stats"]["model_manager"] = await self.model_manager.get_performance_report()
            
            if self.embedding_engine:
                stats["component_stats"]["embedding_engine"] = await self.embedding_engine.get_embedding_stats()
            
            if self.vector_store:
                stats["component_stats"]["vector_store"] = await self.vector_store.get_stats()
            
            if self.inference_engine:
                stats["component_stats"]["inference_engine"] = await self.inference_engine.get_performance_report()
            
            if self.ai_analyzer:
                stats["component_stats"]["ai_analyzer"] = await self.ai_analyzer.get_analysis_stats()
        
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            stats["error"] = str(e)
        
        return stats
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Optimiza performance del sistema."""
        optimization_report = {
            "timestamp": datetime.now().isoformat(),
            "optimizations_applied": [],
            "performance_improvements": {},
            "recommendations": []
        }
        
        try:
            # Optimizar model manager
            if self.model_manager:
                model_optimization = await self.model_manager.optimize_memory_usage()
                optimization_report["optimizations_applied"].append("model_memory_optimization")
                optimization_report["performance_improvements"]["memory_freed_mb"] = model_optimization.get("memory_freed_mb", 0)
            
            # Optimizar embedding cache
            if self.embedding_engine:
                cache_optimization = await self.embedding_engine.optimize_cache()
                optimization_report["optimizations_applied"].append("embedding_cache_optimization")
                optimization_report["performance_improvements"]["cache_items_removed"] = cache_optimization.get("removed_count", 0)
            
            # Optimizar vector store
            if self.vector_store:
                vector_optimization = await self.vector_store.optimize_collections()
                optimization_report["optimizations_applied"].append("vector_store_optimization")
            
            # Generar recomendaciones adicionales
            if self.model_manager:
                model_recommendations = self.model_manager._calculate_model_score.__func__(self.model_manager, "recommendations", ProgrammingLanguage.UNKNOWN)
                # Esta línea es solo demostrativa - en implementación real usaríamos métodos específicos
            
            logger.info("Performance optimization completada")
            
        except Exception as e:
            optimization_report["error"] = str(e)
            logger.error(f"Error en optimización: {e}")
        
        return optimization_report
    
    async def shutdown(self) -> None:
        """Cierra todos los componentes del sistema."""
        logger.info("🔄 Cerrando sistema de IA...")
        
        try:
            if self.inference_engine:
                await self.inference_engine.shutdown()
                logger.info("✅ InferenceEngine cerrado")
            
            # Otros componentes no requieren shutdown explícito por ahora
            
            self.status.initialized = False
            self.status.components_loaded.clear()
            
            logger.info("✅ Sistema de IA cerrado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error cerrando sistema: {e}")
    
    def is_ready(self) -> bool:
        """Verifica si el sistema está listo para uso."""
        return (self.status.initialized and 
                self.ai_analyzer is not None and
                self.embedding_engine is not None)
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Obtiene lenguajes soportados por el sistema."""
        if self.code_preprocessor:
            return self.code_preprocessor.get_supported_languages()
        
        # Fallback a lenguajes básicos soportados
        return [
            ProgrammingLanguage.PYTHON,
            ProgrammingLanguage.JAVASCRIPT, 
            ProgrammingLanguage.TYPESCRIPT,
            ProgrammingLanguage.RUST
        ]
    
    def get_available_analysis_types(self) -> List[AnalysisType]:
        """Obtiene tipos de análisis disponibles."""
        return list(AnalysisType)
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.shutdown()
