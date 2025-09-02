"""
Gestor de integración del sistema semántico completo.

Este módulo orquesta todos los componentes del análisis semántico
y proporciona una interfaz unificada de alto nivel.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ...domain.entities.semantic_analysis import (
    MultiLevelConfig, SemanticSearchConfig, IntentDetectionConfig,
    MultiLevelEmbeddings, SemanticSearchResult, CodeIntentAnalysis,
    ContextualEmbedding, KnowledgeGraphResult, SemanticAnalysisSession,
    SemanticAnalysisReport, SemanticQualityMetrics, ProcessedQuery
)
from ...domain.entities.ai_models import AIIntegrationConfig, AnalysisType
from ...domain.value_objects.programming_language import ProgrammingLanguage
from ..ai_models.ai_integration import AIIntegrationManager
from .multilevel_embedding_engine import MultiLevelEmbeddingEngine
from .semantic_search_engine import SemanticSearchEngine
from .intent_detection_system import IntentDetectionSystem
from .contextual_analyzer import ContextualAnalyzer, AttentionConfig
from .knowledge_graph_builder import KnowledgeGraphBuilder, GraphBuildingConfig

logger = logging.getLogger(__name__)


@dataclass
class SemanticSystemConfig:
    """Configuración completa del sistema semántico."""
    multilevel_config: MultiLevelConfig = field(default_factory=MultiLevelConfig)
    search_config: SemanticSearchConfig = field(default_factory=SemanticSearchConfig)
    intent_config: IntentDetectionConfig = field(default_factory=IntentDetectionConfig)
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    graph_config: GraphBuildingConfig = field(default_factory=GraphBuildingConfig)
    ai_integration_config: AIIntegrationConfig = field(default_factory=AIIntegrationConfig)
    
    def validate_config(self) -> List[str]:
        """Valida la configuración completa."""
        issues = []
        
        # Validar configuración base de IA
        ai_issues = self.ai_integration_config.validate_config()
        issues.extend(ai_issues)
        
        # Validar configuración semántica
        if self.multilevel_config.context_window_size <= 0:
            issues.append("Context window size must be positive")
        
        if self.search_config.similarity_threshold < 0 or self.search_config.similarity_threshold > 1:
            issues.append("Similarity threshold must be between 0 and 1")
        
        if self.intent_config.confidence_threshold < 0 or self.intent_config.confidence_threshold > 1:
            issues.append("Intent confidence threshold must be between 0 and 1")
        
        return issues


@dataclass 
class SemanticSystemStatus:
    """Estado del sistema semántico."""
    initialized: bool = False
    components_loaded: Dict[str, bool] = field(default_factory=dict)
    active_sessions: int = 0
    total_embeddings_generated: int = 0
    total_searches_performed: int = 0
    total_intents_detected: int = 0
    knowledge_graph_size: int = 0
    system_health_score: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)


class SemanticIntegrationManager:
    """Gestor principal de integración semántica."""
    
    def __init__(self, config: Optional[SemanticSystemConfig] = None):
        """
        Inicializa el gestor de integración semántica.
        
        Args:
            config: Configuración del sistema semántico
        """
        self.config = config or SemanticSystemConfig()
        self.status = SemanticSystemStatus()
        
        # Componentes principales
        self.ai_integration_manager: Optional[AIIntegrationManager] = None
        self.multilevel_engine: Optional[MultiLevelEmbeddingEngine] = None
        self.semantic_search_engine: Optional[SemanticSearchEngine] = None
        self.intent_detection_system: Optional[IntentDetectionSystem] = None
        self.contextual_analyzer: Optional[ContextualAnalyzer] = None
        self.knowledge_graph_builder: Optional[KnowledgeGraphBuilder] = None
        
        # Sesiones activas
        self.active_sessions: Dict[str, SemanticAnalysisSession] = {}
        
        # Estadísticas globales
        self.global_stats = {
            'initialization_time_ms': 0,
            'total_analyses': 0,
            'successful_analyses': 0,
            'embeddings_generated': 0,
            'searches_performed': 0,
            'knowledge_graphs_built': 0
        }
        
        logger.info("SemanticIntegrationManager creado")
    
    async def initialize(self) -> bool:
        """
        Inicializa todos los componentes del sistema semántico.
        
        Returns:
            True si la inicialización fue exitosa
        """
        logger.info("🧠 Iniciando sistema de análisis semántico avanzado...")
        start_time = time.time()
        
        try:
            # Validar configuración
            config_issues = self.config.validate_config()
            if config_issues:
                logger.warning(f"Issues de configuración: {config_issues}")
            
            # Inicializar sistema base de IA
            await self._initialize_ai_system()
            
            # Inicializar componentes semánticos
            await self._initialize_semantic_components()
            
            # Verificar inicialización
            self.status.initialized = all(self.status.components_loaded.values())
            
            initialization_time = int((time.time() - start_time) * 1000)
            self.global_stats['initialization_time_ms'] = initialization_time
            
            if self.status.initialized:
                logger.info(f"✅ Sistema semántico inicializado en {initialization_time}ms")
                return True
            else:
                logger.error("❌ Fallo en inicialización del sistema semántico")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error crítico en inicialización semántica: {e}")
            return False
    
    async def _initialize_ai_system(self) -> None:
        """Inicializa el sistema base de IA."""
        try:
            logger.info("Inicializando sistema base de IA...")
            self.ai_integration_manager = AIIntegrationManager(self.config.ai_integration_config)
            
            success = await self.ai_integration_manager.initialize()
            self.status.components_loaded["ai_integration"] = success
            
            if success:
                logger.info("✅ Sistema base de IA inicializado")
            else:
                logger.warning("⚠️ Sistema base de IA en modo degradado")
                
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema de IA: {e}")
            self.status.components_loaded["ai_integration"] = False
    
    async def _initialize_semantic_components(self) -> None:
        """Inicializa componentes semánticos especializados."""
        
        # 1. MultiLevel Embedding Engine
        try:
            if self.ai_integration_manager and self.ai_integration_manager.embedding_engine:
                logger.info("Inicializando MultiLevelEmbeddingEngine...")
                self.multilevel_engine = MultiLevelEmbeddingEngine(
                    self.ai_integration_manager.model_manager,
                    self.ai_integration_manager.embedding_engine,
                    self.config.multilevel_config
                )
                self.status.components_loaded["multilevel_engine"] = True
                logger.info("✅ MultiLevelEmbeddingEngine inicializado")
            else:
                raise Exception("AI integration manager not available")
        except Exception as e:
            logger.error(f"❌ Error inicializando MultiLevelEmbeddingEngine: {e}")
            self.status.components_loaded["multilevel_engine"] = False
        
        # 2. Semantic Search Engine
        try:
            if (self.ai_integration_manager and 
                self.ai_integration_manager.embedding_engine and 
                self.ai_integration_manager.vector_store):
                
                logger.info("Inicializando SemanticSearchEngine...")
                self.semantic_search_engine = SemanticSearchEngine(
                    self.ai_integration_manager.embedding_engine,
                    self.ai_integration_manager.vector_store,
                    self.config.search_config
                )
                self.status.components_loaded["semantic_search"] = True
                logger.info("✅ SemanticSearchEngine inicializado")
            else:
                raise Exception("Required AI components not available")
        except Exception as e:
            logger.error(f"❌ Error inicializando SemanticSearchEngine: {e}")
            self.status.components_loaded["semantic_search"] = False
        
        # 3. Intent Detection System
        try:
            logger.info("Inicializando IntentDetectionSystem...")
            self.intent_detection_system = IntentDetectionSystem(self.config.intent_config)
            self.status.components_loaded["intent_detection"] = True
            logger.info("✅ IntentDetectionSystem inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando IntentDetectionSystem: {e}")
            self.status.components_loaded["intent_detection"] = False
        
        # 4. Contextual Analyzer
        try:
            logger.info("Inicializando ContextualAnalyzer...")
            self.contextual_analyzer = ContextualAnalyzer(self.config.attention_config)
            self.status.components_loaded["contextual_analyzer"] = True
            logger.info("✅ ContextualAnalyzer inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando ContextualAnalyzer: {e}")
            self.status.components_loaded["contextual_analyzer"] = False
        
        # 5. Knowledge Graph Builder
        try:
            logger.info("Inicializando KnowledgeGraphBuilder...")
            self.knowledge_graph_builder = KnowledgeGraphBuilder(self.config.graph_config)
            self.status.components_loaded["knowledge_graph"] = True
            logger.info("✅ KnowledgeGraphBuilder inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando KnowledgeGraphBuilder: {e}")
            self.status.components_loaded["knowledge_graph"] = False
        
        logger.info(f"Componentes semánticos inicializados: {sum(self.status.components_loaded.values())}/{len(self.status.components_loaded)}")
    
    async def analyze_code_semantically(
        self,
        code: str,
        language: ProgrammingLanguage,
        file_path: Optional[Path] = None,
        include_multilevel: bool = True,
        include_intent: bool = True,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Realiza análisis semántico completo de código.
        
        Args:
            code: Código fuente
            language: Lenguaje de programación
            file_path: Ruta del archivo
            include_multilevel: Incluir embeddings multi-nivel
            include_intent: Incluir detección de intención
            include_context: Incluir análisis contextual
            
        Returns:
            Resultado completo del análisis semántico
        """
        if not self.status.initialized:
            raise RuntimeError("Semantic system not initialized")
        
        start_time = time.time()
        session_id = f"semantic_analysis_{int(time.time())}"
        
        # Crear sesión
        session = SemanticAnalysisSession(
            session_id=session_id,
            session_name=f"Analysis of {file_path.name if file_path else 'code_snippet'}"
        )
        self.active_sessions[session_id] = session
        
        analysis_result = {
            "session_id": session_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "code_info": {
                "language": language.value,
                "file_path": str(file_path) if file_path else None,
                "code_length": len(code),
                "line_count": len(code.splitlines())
            },
            "multilevel_embeddings": None,
            "intent_analysis": None,
            "contextual_embeddings": [],
            "semantic_search_demo": None,
            "execution_time_ms": 0,
            "quality_metrics": {},
            "errors": []
        }
        
        try:
            # 1. Generar embeddings multi-nivel
            if include_multilevel and self.multilevel_engine:
                logger.info("Generando embeddings multi-nivel...")
                multilevel_embeddings = await self.multilevel_engine.generate_multilevel_embeddings(
                    code, language, file_path
                )
                
                analysis_result["multilevel_embeddings"] = {
                    "total_embeddings": multilevel_embeddings.get_total_embeddings(),
                    "levels_generated": list(multilevel_embeddings.get_embeddings_by_level(level).keys() 
                                           for level in multilevel_embeddings.get_embeddings_by_level.__code__.co_names),
                    "generation_time_ms": multilevel_embeddings.generation_time_ms,
                    "hierarchical_structure": bool(multilevel_embeddings.hierarchical_structure),
                    "semantic_relationships": len(multilevel_embeddings.semantic_relationships)
                }
                
                session.embeddings_generated += multilevel_embeddings.get_total_embeddings()
            
            # 2. Detección de intención
            if include_intent and self.intent_detection_system:
                logger.info("Detectando intención del código...")
                intent_analysis = await self.intent_detection_system.analyze_code_intent(
                    code, language, file_path=file_path
                )
                
                analysis_result["intent_analysis"] = {
                    "detected_intents": [
                        {
                            "intent_type": intent.intent_type.value,
                            "description": intent.description,
                            "confidence": intent.confidence,
                            "evidence": intent.evidence
                        }
                        for intent in intent_analysis.detected_intents
                    ],
                    "primary_purpose": (intent_analysis.primary_purpose.purpose_type.value 
                                      if intent_analysis.primary_purpose else None),
                    "behavioral_characteristics": intent_analysis.behavioral_characteristics,
                    "domain_concepts": [
                        {
                            "concept_type": concept.concept_type.value,
                            "name": concept.name,
                            "confidence": concept.confidence
                        }
                        for concept in intent_analysis.domain_concepts
                    ],
                    "confidence_scores": {k.value: v for k, v in intent_analysis.confidence_scores.items()}
                }
                
                session.intents_detected += len(intent_analysis.detected_intents)
            
            # 3. Análisis contextual (si hay múltiples funciones)
            if include_context and self.contextual_analyzer and include_multilevel:
                if 'multilevel_embeddings' in locals() and len(multilevel_embeddings.function_embeddings) > 1:
                    logger.info("Generando embeddings contextuales...")
                    
                    # Generar embedding contextual para la primera función
                    func_list = list(multilevel_embeddings.function_embeddings.values())
                    if func_list:
                        target_func = func_list[0]
                        context_funcs = func_list[1:]
                        
                        contextual_embedding = await self.contextual_analyzer.generate_contextual_embedding(
                            target_func.embedding_vector,
                            [f.embedding_vector for f in context_funcs],
                            ["similar_function"] * len(context_funcs)
                        )
                        
                        analysis_result["contextual_embeddings"] = [{
                            "target_function": target_func.function_name,
                            "context_window_size": contextual_embedding.context_window_size,
                            "attention_summary": contextual_embedding.get_attention_summary(),
                            "semantic_context": contextual_embedding.semantic_context
                        }]
            
            # 4. Demo de búsqueda semántica
            if self.semantic_search_engine:
                logger.info("Ejecutando demo de búsqueda semántica...")
                
                # Buscar código similar al analizado
                search_result = await self.semantic_search_engine.search_by_code_example(
                    code[:500],  # Primeros 500 caracteres
                    language
                )
                
                analysis_result["semantic_search_demo"] = {
                    "query": search_result.query,
                    "results_found": len(search_result.results),
                    "search_time_ms": search_result.search_time_ms,
                    "top_similarity": (search_result.results[0].combined_similarity 
                                     if search_result.results else 0.0),
                    "search_interpretation": search_result.query_interpretation
                }
                
                session.searches_performed += 1
            
            # Calcular métricas de calidad
            quality_metrics = await self._calculate_quality_metrics(analysis_result)
            analysis_result["quality_metrics"] = quality_metrics
            
        except Exception as e:
            logger.error(f"Error en análisis semántico: {e}")
            analysis_result["errors"].append(str(e))
        
        finally:
            # Finalizar sesión
            session.total_processing_time_ms = int((time.time() - start_time) * 1000)
            analysis_result["execution_time_ms"] = session.total_processing_time_ms
            
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Actualizar estadísticas globales
            self._update_global_stats(analysis_result)
        
        return analysis_result
    
    async def search_code_by_natural_language(
        self,
        query: str,
        target_languages: Optional[List[ProgrammingLanguage]] = None
    ) -> SemanticSearchResult:
        """
        Busca código usando lenguaje natural.
        
        Args:
            query: Query en lenguaje natural
            target_languages: Lenguajes objetivo
            
        Returns:
            Resultado de búsqueda semántica
        """
        if not self.semantic_search_engine:
            raise RuntimeError("Semantic search engine not initialized")
        
        return await self.semantic_search_engine.search_by_natural_language(
            query, target_languages
        )
    
    async def build_project_knowledge_graph(
        self,
        project_embeddings: List[MultiLevelEmbeddings]
    ) -> KnowledgeGraphResult:
        """
        Construye grafo de conocimiento para un proyecto.
        
        Args:
            project_embeddings: Lista de embeddings de archivos del proyecto
            
        Returns:
            Resultado del grafo de conocimiento
        """
        if not self.knowledge_graph_builder:
            raise RuntimeError("Knowledge graph builder not initialized")
        
        result = await self.knowledge_graph_builder.build_knowledge_graph(project_embeddings)
        
        # Actualizar estadísticas
        self.global_stats['knowledge_graphs_built'] += 1
        self.status.knowledge_graph_size = result.get_node_count()
        
        return result
    
    async def _calculate_quality_metrics(self, analysis_result: Dict[str, Any]) -> Dict[str, float]:
        """Calcula métricas de calidad del análisis."""
        metrics = {}
        
        # Métrica de completitud
        components_completed = sum(1 for key in ["multilevel_embeddings", "intent_analysis", "semantic_search_demo"] 
                                 if analysis_result.get(key) is not None)
        metrics["completeness_score"] = components_completed / 3.0
        
        # Métrica de confianza promedio
        if analysis_result.get("intent_analysis", {}).get("confidence_scores"):
            confidences = list(analysis_result["intent_analysis"]["confidence_scores"].values())
            metrics["average_confidence"] = sum(confidences) / len(confidences)
        else:
            metrics["average_confidence"] = 0.5
        
        # Métrica de riqueza semántica
        total_semantic_elements = 0
        if analysis_result.get("multilevel_embeddings"):
            total_semantic_elements += analysis_result["multilevel_embeddings"]["total_embeddings"]
        if analysis_result.get("intent_analysis", {}).get("detected_intents"):
            total_semantic_elements += len(analysis_result["intent_analysis"]["detected_intents"])
        
        metrics["semantic_richness"] = min(1.0, total_semantic_elements / 20.0)  # Normalizar a 20 elementos
        
        # Métrica de performance
        execution_time = analysis_result.get("execution_time_ms", 0)
        if execution_time > 0:
            # Score inverso de tiempo (mejor performance = score más alto)
            metrics["performance_score"] = max(0.1, 1.0 - (execution_time / 10000.0))  # 10s = score 0
        else:
            metrics["performance_score"] = 1.0
        
        # Score general
        metrics["overall_quality"] = (
            metrics["completeness_score"] * 0.3 +
            metrics["average_confidence"] * 0.3 +
            metrics["semantic_richness"] * 0.2 +
            metrics["performance_score"] * 0.2
        )
        
        return metrics
    
    def _update_global_stats(self, analysis_result: Dict[str, Any]) -> None:
        """Actualiza estadísticas globales."""
        self.global_stats['total_analyses'] += 1
        
        if not analysis_result.get("errors"):
            self.global_stats['successful_analyses'] += 1
        
        if analysis_result.get("multilevel_embeddings"):
            self.global_stats['embeddings_generated'] += analysis_result["multilevel_embeddings"]["total_embeddings"]
        
        if analysis_result.get("semantic_search_demo"):
            self.global_stats['searches_performed'] += 1
    
    async def health_check(self) -> Dict[str, Any]:
        """Ejecuta health check del sistema semántico."""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "system_initialized": self.status.initialized,
            "component_status": {},
            "performance_metrics": {},
            "system_statistics": {},
            "recommendations": []
        }
        
        try:
            # Health check de componentes
            health_report["component_status"] = self.status.components_loaded.copy()
            
            # Health check del sistema base de IA
            if self.ai_integration_manager:
                ai_health = await self.ai_integration_manager.health_check()
                health_report["component_status"]["ai_base_system"] = ai_health.get("overall_health_score", 0) > 0.7
            
            # Métricas de performance
            total_analyses = self.global_stats['total_analyses']
            if total_analyses > 0:
                success_rate = self.global_stats['successful_analyses'] / total_analyses
                
                health_report["performance_metrics"] = {
                    "total_analyses": total_analyses,
                    "success_rate": success_rate,
                    "embeddings_generated": self.global_stats['embeddings_generated'],
                    "searches_performed": self.global_stats['searches_performed'],
                    "knowledge_graphs_built": self.global_stats['knowledge_graphs_built']
                }
            
            # Estadísticas del sistema
            health_report["system_statistics"] = {
                "active_sessions": len(self.active_sessions),
                "components_loaded": sum(self.status.components_loaded.values()),
                "total_components": len(self.status.components_loaded),
                "initialization_time_ms": self.global_stats['initialization_time_ms']
            }
            
            # Calcular score de salud general
            component_health = sum(self.status.components_loaded.values()) / len(self.status.components_loaded)
            performance_health = (self.global_stats['successful_analyses'] / 
                                max(1, self.global_stats['total_analyses']))
            
            overall_health = (component_health + performance_health) / 2.0
            health_report["overall_health_score"] = overall_health
            
            # Generar recomendaciones
            if overall_health < 0.8:
                health_report["recommendations"].append("System health below optimal")
            
            if not self.status.components_loaded.get("ai_integration", False):
                health_report["recommendations"].append("AI integration system not available")
            
            if self.global_stats['total_analyses'] == 0:
                health_report["recommendations"].append("No analyses performed yet - try analyzing some code")
            
            if not self.status.components_loaded.get("knowledge_graph", False):
                health_report["recommendations"].append("Knowledge graph builder not available")
        
        except Exception as e:
            health_report["health_check_error"] = str(e)
            logger.error(f"Error en health check semántico: {e}")
        
        return health_report
    
    async def generate_semantic_report(self) -> SemanticAnalysisReport:
        """Genera reporte completo del análisis semántico."""
        # Calcular métricas de calidad
        quality_metrics = SemanticQualityMetrics()
        
        if self.global_stats['total_analyses'] > 0:
            quality_metrics.embedding_quality_score = 0.8  # Placeholder
            quality_metrics.search_precision = 0.75
            quality_metrics.search_recall = 0.7
            quality_metrics.intent_detection_accuracy = 0.8
            quality_metrics.semantic_consistency = 0.85
        
        # Obtener estadísticas de componentes
        component_stats = {}
        
        if self.multilevel_engine:
            component_stats["multilevel_engine"] = await self.multilevel_engine.get_generation_stats()
        
        if self.semantic_search_engine:
            component_stats["semantic_search"] = await self.semantic_search_engine.get_search_stats()
        
        if self.intent_detection_system:
            component_stats["intent_detection"] = await self.intent_detection_system.get_detection_stats()
        
        if self.knowledge_graph_builder:
            component_stats["knowledge_graph"] = await self.knowledge_graph_builder.get_graph_stats()
        
        # Generar recomendaciones
        recommendations = []
        
        if quality_metrics.get_overall_quality() < 0.8:
            recommendations.append("Consider tuning confidence thresholds for better quality")
        
        if self.global_stats['embeddings_generated'] < 100:
            recommendations.append("Analyze more code to improve semantic understanding")
        
        if self.status.knowledge_graph_size < 10:
            recommendations.append("Build larger knowledge graphs for better insights")
        
        report = SemanticAnalysisReport(
            total_files_analyzed=self.global_stats['total_analyses'],
            total_embeddings_generated=self.global_stats['embeddings_generated'],
            semantic_relationships_found=0,  # Placeholder
            quality_metrics=quality_metrics,
            performance_stats=component_stats,
            recommendations=recommendations
        )
        
        return report
    
    async def optimize_semantic_performance(self) -> Dict[str, Any]:
        """Optimiza performance del sistema semántico."""
        optimization_report = {
            "timestamp": datetime.now().isoformat(),
            "optimizations_applied": [],
            "performance_improvements": {},
            "recommendations": []
        }
        
        try:
            # Optimizar sistema base de IA
            if self.ai_integration_manager:
                ai_optimization = await self.ai_integration_manager.optimize_performance()
                optimization_report["optimizations_applied"].append("ai_system_optimization")
                optimization_report["performance_improvements"]["ai_system"] = ai_optimization
            
            # Optimizar caches de componentes semánticos
            if self.multilevel_engine and hasattr(self.multilevel_engine.token_embedder, 'token_cache'):
                cache_size = len(self.multilevel_engine.token_embedder.token_cache)
                if cache_size > 1000:  # Limpiar cache si está muy grande
                    self.multilevel_engine.token_embedder.token_cache.clear()
                    optimization_report["optimizations_applied"].append("token_cache_cleanup")
                    optimization_report["performance_improvements"]["tokens_cache_cleared"] = cache_size
            
            # Generar recomendaciones de optimización
            if self.global_stats['initialization_time_ms'] > 5000:
                optimization_report["recommendations"].append("Consider caching models to reduce initialization time")
            
            if self.status.knowledge_graph_size > 1000:
                optimization_report["recommendations"].append("Large knowledge graph - consider graph pruning")
            
            logger.info("Optimización semántica completada")
            
        except Exception as e:
            optimization_report["error"] = str(e)
            logger.error(f"Error en optimización semántica: {e}")
        
        return optimization_report
    
    async def shutdown(self) -> None:
        """Cierra el sistema semántico."""
        logger.info("🔄 Cerrando sistema semántico...")
        
        try:
            # Finalizar sesiones activas
            for session in self.active_sessions.values():
                session.completed_at = datetime.now()
            
            self.active_sessions.clear()
            
            # Cerrar sistema base de IA
            if self.ai_integration_manager:
                await self.ai_integration_manager.shutdown()
            
            self.status.initialized = False
            
            logger.info("✅ Sistema semántico cerrado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error cerrando sistema semántico: {e}")
    
    def is_ready(self) -> bool:
        """Verifica si el sistema está listo para uso."""
        essential_components = ["ai_integration", "multilevel_engine", "semantic_search", "intent_detection"]
        
        return (self.status.initialized and
                all(self.status.components_loaded.get(comp, False) for comp in essential_components))
    
    async def get_system_capabilities(self) -> Dict[str, Any]:
        """Obtiene capacidades del sistema."""
        capabilities = {
            "multilevel_embeddings": {
                "available": self.status.components_loaded.get("multilevel_engine", False),
                "supported_levels": ["token", "expression", "function", "class", "file"],
                "aggregation_strategies": ["mean", "weighted_mean", "attention", "hierarchical"]
            },
            "semantic_search": {
                "available": self.status.components_loaded.get("semantic_search", False),
                "natural_language_queries": True,
                "code_example_search": True,
                "cross_language_search": self.config.search_config.enable_cross_language_search
            },
            "intent_detection": {
                "available": self.status.components_loaded.get("intent_detection", False),
                "pattern_based": True,
                "ml_based": self.config.intent_config.enable_ml_intent_detection,
                "domain_analysis": self.config.intent_config.enable_domain_analysis
            },
            "contextual_analysis": {
                "available": self.status.components_loaded.get("contextual_analyzer", False),
                "attention_mechanism": True,
                "context_window_size": self.config.multilevel_config.context_window_size
            },
            "knowledge_graph": {
                "available": self.status.components_loaded.get("knowledge_graph", False),
                "concept_clustering": self.config.graph_config.enable_concept_clustering,
                "centrality_analysis": self.config.graph_config.enable_centrality_analysis
            }
        }
        
        return capabilities
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.shutdown()
