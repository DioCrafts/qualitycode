"""
Implementación del gestor de modelos de IA.

Este módulo implementa la gestión de modelos de IA incluyendo
carga, cache, lifecycle y optimización de memoria.
"""

import logging
import asyncio
import hashlib
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

from ...domain.entities.ai_models import (
    AIConfig, ModelConfig, LoadedModel, ModelType, DeviceType, ModelStatus,
    ModelPerformanceMetrics, AIModelCache, AISystemStatus
)
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)

# Fallback imports - intentar importar librerías de IA con fallbacks
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    AI_LIBRARIES_AVAILABLE = True
    logger.info("AI libraries (torch, transformers) disponibles")
except ImportError:
    # Mock classes para entornos sin AI libraries
    class MockTorch:
        def cuda(self): return self
        def is_available(self): return False
        def backends(self): return self
        def mps(self): return self
        def device(self, device_str): return device_str
        def no_grad(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class MockTransformers:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return MockModel(model_name)
    
    class MockModel:
        def __init__(self, name):
            self.name = name
            self.config = MockConfig()
        def to(self, device): return self
        def eval(self): return self
        def parameters(self): return []
        def buffers(self): return []
        def __call__(self, **kwargs): return MockOutput()
    
    class MockConfig:
        def __init__(self):
            self.model_name = "mock_model"
    
    class MockOutput:
        def __init__(self):
            self.last_hidden_state = MockTensor()
    
    class MockTensor:
        def __init__(self, shape=(1, 512, 768)):
            self.shape = shape
            self._data = [[0.1] * 768] * 512
        def __getitem__(self, key): return self
        def cpu(self): return self
        def numpy(self): return self
        def tolist(self): return self._data[0]
        def squeeze(self): return self
        def to(self, device): return self
        def unsqueeze(self, dim): return self
        def sum(self, dim): return self
        def max(self, dim): return (self, self)
    
    torch = MockTorch()
    nn = MockTorch()
    AutoModel = MockTransformers
    AutoTokenizer = MockTransformers
    AutoConfig = MockTransformers
    AI_LIBRARIES_AVAILABLE = False
    logger.warning("AI libraries no disponibles - usando mocks")


@dataclass
class ModelLoadingResult:
    """Resultado de carga de modelo."""
    success: bool
    model: Optional[LoadedModel] = None
    error_message: Optional[str] = None
    loading_time_ms: int = 0
    memory_allocated_mb: float = 0.0


@dataclass
class LoadedModelRegistry:
    """Registry de modelos cargados."""
    models: Dict[str, LoadedModel] = field(default_factory=dict)
    loading_queue: List[str] = field(default_factory=list)
    max_models: int = 3
    total_memory_mb: float = 0.0
    
    def can_load_model(self, estimated_memory_mb: float) -> bool:
        """Verifica si se puede cargar un modelo."""
        return (len(self.models) < self.max_models and 
                self.total_memory_mb + estimated_memory_mb < 8000)  # 8GB limit
    
    def get_lru_model(self) -> Optional[str]:
        """Obtiene modelo menos recientemente usado."""
        if not self.models:
            return None
        
        return min(self.models.keys(), key=lambda k: self.models[k].last_used_at)


class ModelCache:
    """Cache inteligente para modelos de IA."""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.cache: Dict[str, LoadedModel] = {}
        self.access_times: Dict[str, datetime] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.max_size_mb = config.model_cache_size_gb * 1024
        self.current_size_mb = 0.0
    
    async def get(self, model_id: str) -> Optional[LoadedModel]:
        """Obtiene modelo del cache."""
        if model_id in self.cache:
            self.access_times[model_id] = datetime.now()
            self.hit_count += 1
            return self.cache[model_id]
        
        self.miss_count += 1
        return None
    
    async def put(self, model_id: str, model: LoadedModel) -> bool:
        """Coloca modelo en cache."""
        # Verificar si hay espacio
        if not await self._ensure_space(model.memory_usage_mb):
            logger.warning(f"No hay espacio en cache para modelo {model_id}")
            return False
        
        self.cache[model_id] = model
        self.access_times[model_id] = datetime.now()
        self.current_size_mb += model.memory_usage_mb
        
        logger.debug(f"Modelo {model_id} añadido al cache ({self.current_size_mb:.1f}MB usado)")
        return True
    
    async def _ensure_space(self, required_mb: float) -> bool:
        """Asegura espacio en cache."""
        while self.current_size_mb + required_mb > self.max_size_mb:
            if not self.cache:
                return False  # Cache vacío pero aún no hay espacio
            
            # Evict LRU model
            lru_model_id = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            await self.evict(lru_model_id)
        
        return True
    
    async def evict(self, model_id: str) -> None:
        """Evict modelo del cache."""
        if model_id in self.cache:
            model = self.cache[model_id]
            self.current_size_mb -= model.memory_usage_mb
            del self.cache[model_id]
            del self.access_times[model_id]
            self.eviction_count += 1
            
            logger.debug(f"Modelo {model_id} evicted del cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "hit_rate": hit_rate,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "eviction_count": self.eviction_count,
            "current_size_mb": self.current_size_mb,
            "max_size_mb": self.max_size_mb,
            "utilization_percentage": (self.current_size_mb / self.max_size_mb) * 100.0,
            "models_cached": len(self.cache)
        }


class AIModelManager:
    """Gestor principal de modelos de IA."""
    
    def __init__(self, config: AIConfig):
        """
        Inicializa el gestor de modelos.
        
        Args:
            config: Configuración del sistema de IA
        """
        self.config = config
        self.model_cache = ModelCache(config)
        self.registry = LoadedModelRegistry(max_models=config.max_models_in_memory)
        self.device = self._initialize_device()
        self.model_configs: Dict[str, ModelConfig] = {}
        self.performance_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.loading_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent loads
        
        logger.info(f"AIModelManager inicializado con device: {self.device}")
    
    def _initialize_device(self) -> str:
        """Inicializa dispositivo para IA."""
        if not AI_LIBRARIES_AVAILABLE:
            return "cpu"
        
        if self.config.device_type == DeviceType.AUTO:
            try:
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except Exception:
                return "cpu"
        
        return self.config.device_type.value
    
    async def initialize(self) -> None:
        """Inicializa el gestor y carga configuraciones por defecto."""
        logger.info("Inicializando AIModelManager...")
        
        # Cargar configuraciones de modelos por defecto
        await self._load_default_model_configs()
        
        # Pre-cargar modelos esenciales si están disponibles
        if AI_LIBRARIES_AVAILABLE:
            await self._preload_essential_models()
        else:
            logger.info("AI libraries no disponibles - usando mode de simulación")
            await self._initialize_mock_models()
        
        logger.info("AIModelManager inicializado correctamente")
    
    async def _load_default_model_configs(self) -> None:
        """Carga configuraciones de modelos por defecto."""
        # CodeBERT configuration
        self.model_configs["microsoft/codebert-base"] = ModelConfig(
            model_id="microsoft/codebert-base",
            model_name="CodeBERT Base",
            model_type=ModelType.CODE_BERT,
            huggingface_repo="microsoft/codebert-base",
            supported_languages=[
                ProgrammingLanguage.PYTHON,
                ProgrammingLanguage.JAVASCRIPT,
                ProgrammingLanguage.TYPESCRIPT,
                ProgrammingLanguage.JAVA,
                ProgrammingLanguage.GO,
                ProgrammingLanguage.RUST,
            ],
            max_input_length=512,
            embedding_dimension=768,
            requires_preprocessing=True,
            memory_estimate_mb=500.0
        )
        
        # CodeT5 configuration
        self.model_configs["Salesforce/codet5-small"] = ModelConfig(
            model_id="Salesforce/codet5-small",
            model_name="CodeT5 Small",
            model_type=ModelType.CODE_T5,
            huggingface_repo="Salesforce/codet5-small",
            supported_languages=[
                ProgrammingLanguage.PYTHON,
                ProgrammingLanguage.JAVASCRIPT,
                ProgrammingLanguage.JAVA,
                ProgrammingLanguage.GO,
            ],
            max_input_length=512,
            embedding_dimension=512,
            requires_preprocessing=True,
            memory_estimate_mb=300.0
        )
        
        # GraphCodeBERT configuration
        self.model_configs["microsoft/graphcodebert-base"] = ModelConfig(
            model_id="microsoft/graphcodebert-base",
            model_name="GraphCodeBERT Base",
            model_type=ModelType.GRAPH_CODE_BERT,
            huggingface_repo="microsoft/graphcodebert-base",
            supported_languages=[
                ProgrammingLanguage.PYTHON,
                ProgrammingLanguage.JAVASCRIPT,
                ProgrammingLanguage.JAVA,
                ProgrammingLanguage.GO,
            ],
            max_input_length=512,
            embedding_dimension=768,
            requires_preprocessing=True,
            memory_estimate_mb=600.0
        )
        
        logger.info(f"Cargadas {len(self.model_configs)} configuraciones de modelos")
    
    async def _preload_essential_models(self) -> None:
        """Pre-carga modelos esenciales."""
        essential_models = ["microsoft/codebert-base"]  # Solo el esencial
        
        for model_id in essential_models:
            try:
                await self.load_model(model_id)
                logger.info(f"Modelo esencial {model_id} pre-cargado")
            except Exception as e:
                logger.warning(f"No se pudo pre-cargar modelo {model_id}: {e}")
    
    async def _initialize_mock_models(self) -> None:
        """Inicializa modelos mock para simulación."""
        for model_id, config in self.model_configs.items():
            mock_model = await self._create_mock_model(config)
            self.registry.models[model_id] = mock_model
            
            # Crear métricas por defecto
            self.performance_metrics[model_id] = ModelPerformanceMetrics(model_id=model_id)
        
        logger.info(f"Inicializados {len(self.registry.models)} modelos mock")
    
    async def load_model(self, model_id: str) -> LoadedModel:
        """
        Carga modelo por ID.
        
        Args:
            model_id: ID del modelo a cargar
            
        Returns:
            LoadedModel cargado
        """
        # Verificar cache primero
        cached_model = await self.model_cache.get(model_id)
        if cached_model:
            cached_model.update_usage()
            return cached_model
        
        # Verificar si ya está en registry
        if model_id in self.registry.models:
            model = self.registry.models[model_id]
            model.update_usage()
            return model
        
        # Cargar modelo nuevo
        async with self.loading_semaphore:
            return await self._load_model_internal(model_id)
    
    async def _load_model_internal(self, model_id: str) -> LoadedModel:
        """Carga modelo internamente."""
        start_time = time.time()
        
        if model_id not in self.model_configs:
            raise ValueError(f"Modelo no configurado: {model_id}")
        
        config = self.model_configs[model_id]
        
        try:
            logger.info(f"Cargando modelo {config.model_name}...")
            
            if AI_LIBRARIES_AVAILABLE:
                loaded_model = await self._load_real_model(config)
            else:
                loaded_model = await self._create_mock_model(config)
            
            # Añadir al registry
            self.registry.models[model_id] = loaded_model
            self.registry.total_memory_mb += loaded_model.memory_usage_mb
            
            # Añadir al cache
            await self.model_cache.put(model_id, loaded_model)
            
            # Inicializar métricas
            self.performance_metrics[model_id] = ModelPerformanceMetrics(model_id=model_id)
            
            loading_time = int((time.time() - start_time) * 1000)
            logger.info(f"Modelo {config.model_name} cargado en {loading_time}ms")
            
            return loaded_model
            
        except Exception as e:
            logger.error(f"Error cargando modelo {model_id}: {e}")
            raise
    
    async def _load_real_model(self, config: ModelConfig) -> LoadedModel:
        """Carga modelo real usando transformers."""
        try:
            # Cargar tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.huggingface_repo,
                cache_dir=self._get_cache_dir()
            )
            
            # Cargar modelo basado en tipo
            if config.model_type == ModelType.CODE_BERT:
                model = AutoModel.from_pretrained(
                    config.huggingface_repo,
                    cache_dir=self._get_cache_dir()
                )
            elif config.model_type == ModelType.CODE_T5:
                from transformers import T5Model
                model = T5Model.from_pretrained(
                    config.huggingface_repo,
                    cache_dir=self._get_cache_dir()
                )
            else:
                model = AutoModel.from_pretrained(
                    config.huggingface_repo,
                    cache_dir=self._get_cache_dir()
                )
            
            # Mover a device y modo eval
            model = model.to(self.device)
            model.eval()
            
            # Calcular uso de memoria
            memory_usage = self._calculate_model_memory_usage(model)
            
            return LoadedModel(
                model_id=config.model_id,
                model_type=config.model_type,
                model=model,
                tokenizer=tokenizer,
                config=config,
                status=ModelStatus.LOADED,
                memory_usage_mb=memory_usage
            )
            
        except Exception as e:
            logger.error(f"Error cargando modelo real {config.model_id}: {e}")
            # Fallback a mock si falla
            return await self._create_mock_model(config)
    
    async def _create_mock_model(self, config: ModelConfig) -> LoadedModel:
        """Crea modelo mock para simulación."""
        mock_model = MockModel(config.model_id)
        mock_tokenizer = MockTransformers()
        
        return LoadedModel(
            model_id=config.model_id,
            model_type=config.model_type,
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
            status=ModelStatus.LOADED,
            memory_usage_mb=config.memory_estimate_mb
        )
    
    def _calculate_model_memory_usage(self, model: Any) -> float:
        """Calcula uso de memoria del modelo."""
        if not AI_LIBRARIES_AVAILABLE:
            return 500.0  # Estimación por defecto
        
        try:
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            size_mb = (param_size + buffer_size) / 1024 / 1024
            return size_mb
            
        except Exception as e:
            logger.warning(f"Error calculando memoria del modelo: {e}")
            return 500.0  # Fallback
    
    def _get_cache_dir(self) -> Optional[Path]:
        """Obtiene directorio de cache para modelos."""
        cache_dir = Path.home() / ".cache" / "codeant_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    async def unload_model(self, model_id: str) -> bool:
        """
        Descarga modelo de memoria.
        
        Args:
            model_id: ID del modelo a descargar
            
        Returns:
            True si se descargó exitosamente
        """
        try:
            # Remover del cache
            await self.model_cache.evict(model_id)
            
            # Remover del registry
            if model_id in self.registry.models:
                model = self.registry.models[model_id]
                self.registry.total_memory_mb -= model.memory_usage_mb
                del self.registry.models[model_id]
            
            # Limpiar métricas
            if model_id in self.performance_metrics:
                del self.performance_metrics[model_id]
            
            logger.info(f"Modelo {model_id} descargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error descargando modelo {model_id}: {e}")
            return False
    
    async def get_model_list(self) -> List[Dict[str, Any]]:
        """Obtiene lista de modelos disponibles."""
        models_info = []
        
        for model_id, config in self.model_configs.items():
            is_loaded = model_id in self.registry.models
            
            model_info = {
                "model_id": model_id,
                "model_name": config.model_name,
                "model_type": config.model_type.value,
                "supported_languages": [lang.value for lang in config.supported_languages],
                "embedding_dimension": config.embedding_dimension,
                "memory_estimate_mb": config.memory_estimate_mb,
                "is_loaded": is_loaded,
                "status": self.registry.models[model_id].status.value if is_loaded else "not_loaded"
            }
            
            if is_loaded and model_id in self.performance_metrics:
                metrics = self.performance_metrics[model_id]
                model_info["performance"] = {
                    "total_inferences": metrics.total_inferences,
                    "average_time_ms": metrics.average_inference_time_ms,
                    "success_rate": metrics.get_success_rate(),
                    "throughput": metrics.get_average_throughput()
                }
            
            models_info.append(model_info)
        
        return models_info
    
    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimiza uso de memoria."""
        optimization_result = {
            "initial_memory_mb": self.registry.total_memory_mb,
            "models_evicted": 0,
            "memory_freed_mb": 0.0,
            "optimization_actions": []
        }
        
        # Identificar modelos poco usados
        current_time = datetime.now()
        threshold_minutes = 30.0
        
        models_to_evict = []
        for model_id, model in self.registry.models.items():
            minutes_since_use = (current_time - model.last_used_at).total_seconds() / 60.0
            if minutes_since_use > threshold_minutes and model.inference_count < 10:
                models_to_evict.append(model_id)
        
        # Evict modelos identificados
        for model_id in models_to_evict:
            model = self.registry.models[model_id]
            memory_freed = model.memory_usage_mb
            
            if await self.unload_model(model_id):
                optimization_result["models_evicted"] += 1
                optimization_result["memory_freed_mb"] += memory_freed
                optimization_result["optimization_actions"].append(f"Evicted unused model: {model_id}")
        
        optimization_result["final_memory_mb"] = self.registry.total_memory_mb
        
        logger.info(f"Optimización de memoria completada: {optimization_result}")
        return optimization_result
    
    async def get_system_status(self) -> AISystemStatus:
        """Obtiene estado del sistema de IA."""
        status = AISystemStatus()
        
        # Estado de modelos
        for model_id, model in self.registry.models.items():
            status.models_loaded[model_id] = model.status
        
        # Estado de componentes
        status.embedding_engine_status = "ready" if AI_LIBRARIES_AVAILABLE else "mock_mode"
        status.inference_engine_status = "ready" if AI_LIBRARIES_AVAILABLE else "mock_mode"
        status.vector_store_status = "disconnected"  # Se actualizará cuando se implemente VectorStore
        
        # Métricas de memoria
        status.total_memory_usage_mb = self.registry.total_memory_mb
        
        # Jobs (placeholder)
        status.active_jobs = 0
        status.completed_jobs = sum(metrics.total_inferences for metrics in self.performance_metrics.values())
        status.failed_jobs = sum(metrics.error_count for metrics in self.performance_metrics.values())
        
        # Calcular health score
        status.calculate_health_score()
        
        return status
    
    async def record_inference(self, model_id: str, inference_time_ms: int, success: bool) -> None:
        """Registra métricas de inferencia."""
        if model_id not in self.performance_metrics:
            self.performance_metrics[model_id] = ModelPerformanceMetrics(model_id=model_id)
        
        metrics = self.performance_metrics[model_id]
        metrics.update_inference_metrics(inference_time_ms, success)
        
        # Actualizar uso del modelo
        if model_id in self.registry.models:
            self.registry.models[model_id].update_usage()
    
    def get_best_model_for_language(self, language: ProgrammingLanguage) -> Optional[str]:
        """
        Obtiene mejor modelo para un lenguaje específico.
        
        Args:
            language: Lenguaje de programación
            
        Returns:
            ID del mejor modelo o None
        """
        # Filtrar modelos que soportan el lenguaje
        suitable_models = []
        
        for model_id, config in self.model_configs.items():
            if config.supports_language(language):
                # Calcular score del modelo
                score = self._calculate_model_score(model_id, language)
                suitable_models.append((model_id, score))
        
        if not suitable_models:
            # Fallback al modelo por defecto
            return "microsoft/codebert-base" if "microsoft/codebert-base" in self.model_configs else None
        
        # Retornar modelo con mejor score
        suitable_models.sort(key=lambda x: x[1], reverse=True)
        return suitable_models[0][0]
    
    def _calculate_model_score(self, model_id: str, language: ProgrammingLanguage) -> float:
        """Calcula score de modelo para lenguaje específico."""
        base_score = 50.0
        
        config = self.model_configs[model_id]
        
        # Score por soporte específico del lenguaje
        if language in config.supported_languages:
            base_score += 30.0
        
        # Score por tipo de modelo
        model_type_scores = {
            ModelType.CODE_BERT: 80.0,
            ModelType.GRAPH_CODE_BERT: 85.0,
            ModelType.CODE_T5: 75.0,
            ModelType.UNIX_CODER: 70.0
        }
        
        base_score += model_type_scores.get(config.model_type, 50.0)
        
        # Bonus si está cargado
        if model_id in self.registry.models:
            base_score += 20.0
        
        # Penalty por uso de memoria alto
        if config.memory_estimate_mb > 1000:
            base_score -= 10.0
        
        # Performance metrics si disponibles
        if model_id in self.performance_metrics:
            metrics = self.performance_metrics[model_id]
            if metrics.total_inferences > 0:
                # Bonus por buen performance
                if metrics.get_success_rate() > 0.95:
                    base_score += 15.0
                if metrics.get_average_throughput() > 10.0:
                    base_score += 10.0
        
        return base_score
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Genera reporte de performance."""
        report = {
            "system_overview": {
                "models_loaded": len(self.registry.models),
                "total_memory_mb": self.registry.total_memory_mb,
                "cache_hit_rate": self.model_cache.get_stats()["hit_rate"],
                "ai_libraries_available": AI_LIBRARIES_AVAILABLE
            },
            "model_performance": {},
            "cache_stats": self.model_cache.get_stats(),
            "recommendations": []
        }
        
        # Performance por modelo
        for model_id, metrics in self.performance_metrics.items():
            report["model_performance"][model_id] = {
                "total_inferences": metrics.total_inferences,
                "success_rate": metrics.get_success_rate(),
                "average_time_ms": metrics.average_inference_time_ms,
                "throughput": metrics.get_average_throughput(),
                "memory_usage_mb": metrics.memory_usage_mb
            }
        
        # Generar recomendaciones
        if self.model_cache.get_stats()["hit_rate"] < 0.7:
            report["recommendations"].append("Consider increasing model cache size")
        
        if self.registry.total_memory_mb > 6000:  # > 6GB
            report["recommendations"].append("High memory usage - consider model optimization")
        
        if not AI_LIBRARIES_AVAILABLE:
            report["recommendations"].append("Install AI libraries (torch, transformers) for full functionality")
        
        return report
    
    async def health_check(self) -> Dict[str, Any]:
        """Ejecuta health check del sistema."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "unknown",
            "component_status": {},
            "issues_detected": [],
            "recommendations": []
        }
        
        try:
            # Verificar modelos cargados
            loaded_models = len(self.registry.models)
            health_status["component_status"]["models_loaded"] = loaded_models
            
            # Verificar memoria
            memory_usage = self.registry.total_memory_mb
            memory_limit = self.config.model_cache_size_gb * 1024
            memory_ok = memory_usage < memory_limit * 0.9
            
            health_status["component_status"]["memory_usage_mb"] = memory_usage
            health_status["component_status"]["memory_ok"] = memory_ok
            
            # Verificar AI libraries
            health_status["component_status"]["ai_libraries"] = AI_LIBRARIES_AVAILABLE
            
            # Determinar salud general
            if loaded_models > 0 and memory_ok:
                health_status["overall_health"] = "healthy"
            elif loaded_models > 0:
                health_status["overall_health"] = "degraded"
            else:
                health_status["overall_health"] = "unhealthy"
            
            # Identificar issues
            if not memory_ok:
                health_status["issues_detected"].append("High memory usage")
            
            if not AI_LIBRARIES_AVAILABLE:
                health_status["issues_detected"].append("AI libraries not available - using mock mode")
            
            # Recomendaciones
            if memory_usage > memory_limit * 0.8:
                health_status["recommendations"].append("Consider optimizing memory usage")
            
            if loaded_models == 0:
                health_status["recommendations"].append("Load at least one AI model for functionality")
            
        except Exception as e:
            health_status["overall_health"] = "error"
            health_status["issues_detected"].append(f"Health check error: {e}")
        
        logger.debug(f"Health check completado: {health_status['overall_health']}")
        return health_status
