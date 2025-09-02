"""
Motor de inferencia optimizado para modelos de IA.

Este módulo implementa un motor de inferencia que optimiza
la ejecución de modelos de IA para análisis de código.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import concurrent.futures
from enum import Enum

from ...domain.entities.ai_models import (
    LoadedModel, AIConfig, ModelPerformanceMetrics, 
    EmbeddingConfig, PoolingStrategy
)
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)

# Fallback para librerías de IA
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    class MockTorch:
        def no_grad(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        
        class nn:
            @staticmethod
            def functional(): return MockF()
    
    class MockF:
        def softmax(self, x, dim=-1): return x
        def cosine_similarity(self, x1, x2): return 0.85
    
    torch = MockTorch()
    F = MockF()
    TORCH_AVAILABLE = False


class InferenceMode(Enum):
    """Modos de inferencia."""
    FAST = "fast"          # Rápido, menor precisión
    BALANCED = "balanced"  # Balance velocidad/precisión
    ACCURATE = "accurate"  # Máxima precisión


@dataclass
class InferenceRequest:
    """Petición de inferencia."""
    request_id: str
    model_id: str
    inputs: Dict[str, Any]
    mode: InferenceMode = InferenceMode.BALANCED
    priority: int = 5  # 1-10, donde 10 es máxima prioridad
    timeout_seconds: int = 30
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_age_seconds(self) -> float:
        """Obtiene edad de la petición en segundos."""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class InferenceResult:
    """Resultado de inferencia."""
    request_id: str
    model_id: str
    success: bool
    outputs: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    inference_time_ms: int = 0
    memory_used_mb: float = 0.0
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class InferenceStats:
    """Estadísticas del motor de inferencia."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_inference_time_ms: float = 0.0
    total_inference_time_ms: int = 0
    peak_memory_mb: float = 0.0
    throughput_per_second: float = 0.0
    
    def update_request_stats(self, inference_time_ms: int, success: bool, memory_mb: float) -> None:
        """Actualiza estadísticas de petición."""
        self.total_requests += 1
        self.total_inference_time_ms += inference_time_ms
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Actualizar promedio
        self.average_inference_time_ms = self.total_inference_time_ms / self.total_requests
        
        # Actualizar memoria pico
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        
        # Calcular throughput
        if self.total_inference_time_ms > 0:
            self.throughput_per_second = (self.total_requests * 1000.0) / self.total_inference_time_ms


class ModelInferenceOptimizer:
    """Optimizador de inferencia para modelos específicos."""
    
    def __init__(self, model: LoadedModel, config: AIConfig):
        self.model = model
        self.config = config
        self.optimization_cache: Dict[str, Any] = {}
        self.performance_profile = self._create_performance_profile()
    
    def _create_performance_profile(self) -> Dict[str, Any]:
        """Crea perfil de performance del modelo."""
        return {
            "model_type": self.model.model_type.value,
            "memory_usage_mb": self.model.memory_usage_mb,
            "optimal_batch_size": self._estimate_optimal_batch_size(),
            "supports_quantization": self._supports_quantization(),
            "preferred_sequence_length": self._get_preferred_sequence_length(),
            "device_optimizations": self._get_device_optimizations()
        }
    
    def _estimate_optimal_batch_size(self) -> int:
        """Estima tamaño óptimo de batch."""
        # Estimación basada en memoria del modelo
        if self.model.memory_usage_mb < 200:
            return 64
        elif self.model.memory_usage_mb < 500:
            return 32
        elif self.model.memory_usage_mb < 1000:
            return 16
        else:
            return 8
    
    def _supports_quantization(self) -> bool:
        """Verifica si el modelo soporta quantización."""
        return TORCH_AVAILABLE and self.config.enable_quantization
    
    def _get_preferred_sequence_length(self) -> int:
        """Obtiene longitud preferida de secuencia."""
        return min(self.model.config.max_input_length, 512)
    
    def _get_device_optimizations(self) -> Dict[str, bool]:
        """Obtiene optimizaciones de device."""
        return {
            "mixed_precision": TORCH_AVAILABLE and self.config.enable_quantization,
            "gradient_checkpointing": False,  # No necesario para inferencia
            "torch_compile": TORCH_AVAILABLE and hasattr(torch, 'compile'),
            "tensor_parallel": self.config.enable_model_parallelism
        }
    
    async def optimize_inputs(self, inputs: Dict[str, Any], mode: InferenceMode) -> Dict[str, Any]:
        """Optimiza inputs para inferencia."""
        optimized = inputs.copy()
        
        if mode == InferenceMode.FAST:
            # Reducir longitud para velocidad
            if 'max_length' in optimized:
                optimized['max_length'] = min(optimized['max_length'], 256)
            
            # Usar padding mínimo
            if 'padding' in optimized:
                optimized['padding'] = 'max_length'
        
        elif mode == InferenceMode.ACCURATE:
            # Usar configuración completa
            if 'max_length' not in optimized:
                optimized['max_length'] = self.performance_profile['preferred_sequence_length']
        
        return optimized
    
    async def post_process_outputs(self, outputs: Any, mode: InferenceMode) -> Dict[str, Any]:
        """Post-procesa outputs de inferencia."""
        result = {"raw_outputs": outputs}
        
        if TORCH_AVAILABLE and hasattr(outputs, 'last_hidden_state'):
            # Extraer hidden states
            hidden_states = outputs.last_hidden_state
            
            # Aplicar pooling por defecto (CLS token)
            if len(hidden_states.shape) == 3:  # batch_size, seq_len, hidden_size
                pooled_output = hidden_states[:, 0, :]  # CLS token
                result["pooled_output"] = pooled_output
                result["embedding"] = pooled_output.cpu().numpy().tolist()
            
            # Añadir atención si está disponible
            if hasattr(outputs, 'attentions') and outputs.attentions:
                result["attention_weights"] = outputs.attentions[-1].cpu().numpy().tolist()
        
        elif not TORCH_AVAILABLE:
            # Mock outputs
            result["embedding"] = [[0.1] * 768]
            result["pooled_output"] = "mock"
        
        return result
    
    def get_optimization_recommendations(self) -> List[str]:
        """Obtiene recomendaciones de optimización."""
        recommendations = []
        
        if self.model.memory_usage_mb > 1000:
            recommendations.append("Consider model quantization to reduce memory usage")
        
        if not TORCH_AVAILABLE:
            recommendations.append("Install PyTorch for better performance")
        
        if self.config.device_type.value == "cpu" and TORCH_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    recommendations.append("GPU available - consider using CUDA for faster inference")
            except ImportError:
                pass
        
        if self.performance_profile["optimal_batch_size"] > 32:
            recommendations.append("Model can handle larger batch sizes for better throughput")
        
        return recommendations


class InferenceQueue:
    """Cola de peticiones de inferencia con priorización."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: List[InferenceRequest] = []
        self.processing: Dict[str, InferenceRequest] = {}
        self.lock = asyncio.Lock()
    
    async def enqueue(self, request: InferenceRequest) -> bool:
        """Añade petición a la cola."""
        async with self.lock:
            if len(self.queue) >= self.max_size:
                # Remover petición de menor prioridad si hay espacio
                min_priority_idx = min(
                    range(len(self.queue)),
                    key=lambda i: self.queue[i].priority
                )
                
                if self.queue[min_priority_idx].priority < request.priority:
                    self.queue.pop(min_priority_idx)
                else:
                    return False  # Cola llena con prioridades más altas
            
            # Insertar ordenado por prioridad
            inserted = False
            for i, existing_request in enumerate(self.queue):
                if request.priority > existing_request.priority:
                    self.queue.insert(i, request)
                    inserted = True
                    break
            
            if not inserted:
                self.queue.append(request)
            
            return True
    
    async def dequeue(self) -> Optional[InferenceRequest]:
        """Obtiene siguiente petición de la cola."""
        async with self.lock:
            if self.queue:
                request = self.queue.pop(0)  # Pop de mayor prioridad
                self.processing[request.request_id] = request
                return request
            return None
    
    async def complete(self, request_id: str) -> None:
        """Marca petición como completada."""
        async with self.lock:
            if request_id in self.processing:
                del self.processing[request_id]
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la cola."""
        async with self.lock:
            return {
                "queue_size": len(self.queue),
                "processing_count": len(self.processing),
                "max_size": self.max_size,
                "utilization": (len(self.queue) + len(self.processing)) / self.max_size
            }


class InferenceEngine:
    """Motor de inferencia principal."""
    
    def __init__(self, device: str, config: Optional[AIConfig] = None):
        """
        Inicializa el motor de inferencia.
        
        Args:
            device: Dispositivo de inferencia
            config: Configuración de IA
        """
        self.device = device
        self.config = config or AIConfig()
        self.stats = InferenceStats()
        self.queue = InferenceQueue()
        self.optimizers: Dict[str, ModelInferenceOptimizer] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.inference_batch_size // 4  # Threads para paralelización
        )
        self.is_processing = False
        self._processing_task = None
        
        logger.info(f"InferenceEngine inicializado en device: {device}")
    
    async def initialize(self) -> None:
        """Inicializa el motor de inferencia."""
        logger.info("Inicializando InferenceEngine...")
        
        # Iniciar procesamiento de cola en background
        self._processing_task = asyncio.create_task(self._process_queue())
        self.is_processing = True
        
        logger.info("InferenceEngine inicializado correctamente")
    
    async def shutdown(self) -> None:
        """Cierra el motor de inferencia."""
        self.is_processing = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        self.executor.shutdown(wait=True)
        logger.info("InferenceEngine cerrado")
    
    def register_model_optimizer(self, model: LoadedModel) -> None:
        """Registra optimizador para un modelo."""
        optimizer = ModelInferenceOptimizer(model, self.config)
        self.optimizers[model.model_id] = optimizer
        logger.debug(f"Optimizador registrado para modelo {model.model_id}")
    
    async def inference_async(
        self,
        model: LoadedModel,
        inputs: Dict[str, Any],
        mode: InferenceMode = InferenceMode.BALANCED,
        priority: int = 5,
        timeout_seconds: int = 30
    ) -> InferenceResult:
        """
        Ejecuta inferencia asíncrona.
        
        Args:
            model: Modelo cargado
            inputs: Inputs para el modelo
            mode: Modo de inferencia
            priority: Prioridad de la petición
            timeout_seconds: Timeout
            
        Returns:
            Resultado de la inferencia
        """
        request_id = f"req_{int(time.time())}_{id(inputs)}"
        
        # Crear petición
        request = InferenceRequest(
            request_id=request_id,
            model_id=model.model_id,
            inputs=inputs,
            mode=mode,
            priority=priority,
            timeout_seconds=timeout_seconds
        )
        
        # Añadir a cola
        queued = await self.queue.enqueue(request)
        if not queued:
            return InferenceResult(
                request_id=request_id,
                model_id=model.model_id,
                success=False,
                error_message="Queue is full"
            )
        
        # Esperar procesamiento (en implementación real usaríamos callbacks/futures)
        start_time = time.time()
        while (time.time() - start_time) < timeout_seconds:
            # En implementación real, tendríamos un sistema de callbacks
            # Por ahora simulamos procesamiento inmediato
            result = await self._execute_inference(model, request)
            await self.queue.complete(request_id)
            return result
        
        # Timeout
        await self.queue.complete(request_id)
        return InferenceResult(
            request_id=request_id,
            model_id=model.model_id,
            success=False,
            error_message="Request timeout"
        )
    
    async def inference_batch(
        self,
        model: LoadedModel,
        batch_inputs: List[Dict[str, Any]],
        mode: InferenceMode = InferenceMode.BALANCED
    ) -> List[InferenceResult]:
        """
        Ejecuta inferencia en batch para mejor eficiencia.
        
        Args:
            model: Modelo cargado
            batch_inputs: Lista de inputs
            mode: Modo de inferencia
            
        Returns:
            Lista de resultados
        """
        if not batch_inputs:
            return []
        
        batch_start_time = time.time()
        results = []
        
        try:
            # Optimizar inputs si hay optimizador
            optimizer = self.optimizers.get(model.model_id)
            if optimizer:
                optimized_inputs = []
                for inputs in batch_inputs:
                    optimized = await optimizer.optimize_inputs(inputs, mode)
                    optimized_inputs.append(optimized)
            else:
                optimized_inputs = batch_inputs
            
            # Ejecutar inferencia en batch
            if TORCH_AVAILABLE:
                batch_result = await self._execute_batch_inference_torch(
                    model, optimized_inputs, mode
                )
            else:
                batch_result = await self._execute_batch_inference_mock(
                    model, optimized_inputs, mode
                )
            
            # Procesar resultados individuales
            for i, inputs in enumerate(batch_inputs):
                request_id = f"batch_{int(time.time())}_{i}"
                
                if i < len(batch_result):
                    result = InferenceResult(
                        request_id=request_id,
                        model_id=model.model_id,
                        success=True,
                        outputs=batch_result[i],
                        inference_time_ms=int((time.time() - batch_start_time) * 1000) // len(batch_inputs)
                    )
                else:
                    result = InferenceResult(
                        request_id=request_id,
                        model_id=model.model_id,
                        success=False,
                        error_message="Batch processing error"
                    )
                
                results.append(result)
        
        except Exception as e:
            logger.error(f"Error en batch inference: {e}")
            
            # Crear resultados de error para todos los inputs
            for i in range(len(batch_inputs)):
                result = InferenceResult(
                    request_id=f"batch_error_{i}",
                    model_id=model.model_id,
                    success=False,
                    error_message=str(e)
                )
                results.append(result)
        
        # Actualizar estadísticas
        batch_time = int((time.time() - batch_start_time) * 1000)
        for result in results:
            self.stats.update_request_stats(
                batch_time // len(results),
                result.success,
                model.memory_usage_mb
            )
        
        return results
    
    async def _execute_inference(self, model: LoadedModel, request: InferenceRequest) -> InferenceResult:
        """Ejecuta inferencia individual."""
        start_time = time.time()
        
        try:
            # Optimizar inputs
            optimizer = self.optimizers.get(model.model_id)
            if optimizer:
                optimized_inputs = await optimizer.optimize_inputs(request.inputs, request.mode)
            else:
                optimized_inputs = request.inputs
            
            # Ejecutar inferencia
            if TORCH_AVAILABLE:
                outputs = await self._execute_single_inference_torch(model, optimized_inputs)
            else:
                outputs = await self._execute_single_inference_mock(model, optimized_inputs)
            
            # Post-procesar si hay optimizador
            if optimizer:
                outputs = await optimizer.post_process_outputs(outputs, request.mode)
            
            inference_time = int((time.time() - start_time) * 1000)
            
            # Actualizar estadísticas
            self.stats.update_request_stats(inference_time, True, model.memory_usage_mb)
            
            return InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                success=True,
                outputs=outputs,
                inference_time_ms=inference_time,
                memory_used_mb=model.memory_usage_mb
            )
            
        except Exception as e:
            inference_time = int((time.time() - start_time) * 1000)
            self.stats.update_request_stats(inference_time, False, model.memory_usage_mb)
            
            return InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                success=False,
                error_message=str(e),
                inference_time_ms=inference_time
            )
    
    async def _execute_single_inference_torch(self, model: LoadedModel, inputs: Dict[str, Any]) -> Any:
        """Ejecuta inferencia individual con PyTorch."""
        if not TORCH_AVAILABLE:
            return await self._execute_single_inference_mock(model, inputs)
        
        # Preparar inputs para el modelo
        model_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, 'to'):
                model_inputs[key] = value.to(self.device)
            else:
                model_inputs[key] = value
        
        # Ejecutar inferencia
        with torch.no_grad():
            outputs = model.model(**model_inputs)
        
        return outputs
    
    async def _execute_single_inference_mock(self, model: LoadedModel, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta inferencia mock."""
        # Simular tiempo de procesamiento
        await asyncio.sleep(0.01)  # 10ms
        
        return {
            "embeddings": [[0.1] * 768],
            "logits": [[0.5, 0.3, 0.2]],
            "attention_mask": [[1, 1, 1, 0, 0]]
        }
    
    async def _execute_batch_inference_torch(
        self, 
        model: LoadedModel, 
        batch_inputs: List[Dict[str, Any]], 
        mode: InferenceMode
    ) -> List[Dict[str, Any]]:
        """Ejecuta inferencia en batch con PyTorch."""
        if not TORCH_AVAILABLE:
            return await self._execute_batch_inference_mock(model, batch_inputs, mode)
        
        # Combinar inputs en batch
        batched_inputs = {}
        
        # Asumiendo que todos los inputs tienen las mismas claves
        if batch_inputs:
            sample_input = batch_inputs[0]
            
            for key in sample_input:
                values = [inputs.get(key) for inputs in batch_inputs]
                
                # Si son tensors, hacer stack/cat
                if hasattr(values[0], 'shape'):
                    batched_inputs[key] = torch.stack(values).to(self.device)
                else:
                    batched_inputs[key] = values
        
        # Ejecutar inferencia
        with torch.no_grad():
            batch_outputs = model.model(**batched_inputs)
        
        # Descomponer batch en resultados individuales
        results = []
        batch_size = len(batch_inputs)
        
        for i in range(batch_size):
            individual_output = {}
            
            if hasattr(batch_outputs, 'last_hidden_state'):
                individual_output["hidden_states"] = batch_outputs.last_hidden_state[i].cpu().numpy().tolist()
                individual_output["embedding"] = batch_outputs.last_hidden_state[i, 0, :].cpu().numpy().tolist()
            
            results.append(individual_output)
        
        return results
    
    async def _execute_batch_inference_mock(
        self, 
        model: LoadedModel, 
        batch_inputs: List[Dict[str, Any]], 
        mode: InferenceMode
    ) -> List[Dict[str, Any]]:
        """Ejecuta inferencia batch mock."""
        # Simular tiempo de batch
        await asyncio.sleep(0.05)  # 50ms para el batch completo
        
        results = []
        for i in range(len(batch_inputs)):
            result = {
                "embeddings": [[0.1 + i * 0.01] * 768],
                "batch_index": i,
                "model_id": model.model_id
            }
            results.append(result)
        
        return results
    
    async def _process_queue(self) -> None:
        """Procesa cola de peticiones en background."""
        while self.is_processing:
            try:
                # En implementación real, procesaríamos la cola continuamente
                # Por ahora solo dormimos
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error procesando cola de inferencia: {e}")
                await asyncio.sleep(1)  # Esperar antes de reintentar
    
    def calculate_cosine_similarity(
        self, 
        vec1: List[float], 
        vec2: List[float]
    ) -> float:
        """
        Calcula similitud coseno entre vectores.
        
        Args:
            vec1: Primer vector
            vec2: Segundo vector
            
        Returns:
            Similitud coseno (0-1)
        """
        if TORCH_AVAILABLE:
            t1 = torch.tensor(vec1)
            t2 = torch.tensor(vec2)
            similarity = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0))
            return float(similarity.item())
        else:
            # Implementación manual
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
    
    def apply_pooling_strategy(
        self, 
        hidden_states: Any, 
        attention_mask: Optional[Any], 
        strategy: PoolingStrategy
    ) -> Any:
        """
        Aplica estrategia de pooling a hidden states.
        
        Args:
            hidden_states: Hidden states del modelo
            attention_mask: Máscara de atención
            strategy: Estrategia de pooling
            
        Returns:
            Tensor pooled
        """
        if not TORCH_AVAILABLE:
            return [[0.1] * 768]  # Mock
        
        if strategy == PoolingStrategy.CLS:
            return hidden_states[:, 0, :]
        
        elif strategy == PoolingStrategy.MEAN:
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(-1)
                return (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
            else:
                return hidden_states.mean(1)
        
        elif strategy == PoolingStrategy.MAX:
            return hidden_states.max(1)[0]
        
        elif strategy == PoolingStrategy.ATTENTION_WEIGHTED:
            # Implementación simplificada de attention pooling
            attention_scores = torch.softmax(hidden_states.mean(dim=-1), dim=-1)
            if attention_mask is not None:
                attention_scores = attention_scores * attention_mask
                attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)
            
            weighted = (hidden_states * attention_scores.unsqueeze(-1)).sum(dim=1)
            return weighted
        
        else:
            # Default a CLS
            return hidden_states[:, 0, :]
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Genera reporte de performance del motor."""
        queue_stats = await self.queue.get_queue_stats()
        
        return {
            "engine_stats": {
                "total_requests": self.stats.total_requests,
                "successful_requests": self.stats.successful_requests,
                "failed_requests": self.stats.failed_requests,
                "success_rate": (self.stats.successful_requests / max(1, self.stats.total_requests)) * 100,
                "average_inference_time_ms": self.stats.average_inference_time_ms,
                "throughput_per_second": self.stats.throughput_per_second,
                "peak_memory_mb": self.stats.peak_memory_mb
            },
            "queue_stats": queue_stats,
            "optimizer_count": len(self.optimizers),
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "is_processing": self.is_processing
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Obtiene recomendaciones de optimización general."""
        recommendations = []
        
        # Analizar estadísticas
        if self.stats.total_requests > 0:
            success_rate = self.stats.successful_requests / self.stats.total_requests
            
            if success_rate < 0.95:
                recommendations.append("High failure rate - check model compatibility and inputs")
            
            if self.stats.average_inference_time_ms > 1000:
                recommendations.append("High inference latency - consider model optimization")
            
            if self.stats.throughput_per_second < 1.0:
                recommendations.append("Low throughput - consider batch processing or hardware upgrade")
        
        # Recomendaciones de configuración
        if not TORCH_AVAILABLE:
            recommendations.append("Install PyTorch for better performance and functionality")
        
        if self.device == "cpu" and TORCH_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    recommendations.append("CUDA available - switch to GPU for faster inference")
            except ImportError:
                pass
        
        if len(self.optimizers) == 0:
            recommendations.append("No model optimizers registered - consider registering optimizers for better performance")
        
        return recommendations
