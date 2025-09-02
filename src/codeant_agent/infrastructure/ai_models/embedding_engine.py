"""
Motor de generación de embeddings de código.

Este módulo implementa la generación de embeddings semánticos
de código usando modelos de IA pre-entrenados.
"""

import logging
import asyncio
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ...domain.entities.ai_models import (
    CodeEmbedding, EmbeddingMetadata, EmbeddingConfig, BatchEmbeddingJob,
    PoolingStrategy, EmbeddingGenerationStats, LoadedModel
)
from ...domain.value_objects.programming_language import ProgrammingLanguage
from .model_manager import AIModelManager

logger = logging.getLogger(__name__)

# Fallback para numpy si no está disponible
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Mock numpy para entornos sin numpy
    class MockNumPy:
        def array(self, data): return MockArray(data)
        def linalg(self): return MockLinalg()
        def dot(self, a, b): return 0.5
    
    class MockArray:
        def __init__(self, data):
            self.data = data if isinstance(data, list) else [data]
        def tolist(self): return self.data
        def __truediv__(self, other): return self
        def __len__(self): return len(self.data)
    
    class MockLinalg:
        def norm(self, vector): return 1.0 if vector else 0.0
    
    np = MockNumPy()
    NUMPY_AVAILABLE = False
    logger.warning("NumPy no disponible - usando mock")

# Mock torch si no está disponible  
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    class MockTorch:
        def no_grad(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    torch = MockTorch()
    TORCH_AVAILABLE = False


@dataclass
class EmbeddingBatch:
    """Batch de códigos para procesar embeddings."""
    batch_id: str
    code_snippets: List[Tuple[str, ProgrammingLanguage]]
    model_id: str
    status: str = "pending"  # pending, processing, completed, failed
    embeddings: List[CodeEmbedding] = field(default_factory=list)
    processing_time_ms: int = 0
    created_at: datetime = field(default_factory=datetime.now)


class BatchEmbeddingProcessor:
    """Procesador de embeddings en batch para eficiencia."""
    
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.processing_queue: List[EmbeddingBatch] = []
        self.active_batches: Dict[str, EmbeddingBatch] = {}
    
    async def process_batch(
        self, 
        batch: EmbeddingBatch,
        embedding_engine: 'CodeEmbeddingEngine'
    ) -> EmbeddingBatch:
        """Procesa un batch de embeddings."""
        start_time = time.time()
        batch.status = "processing"
        self.active_batches[batch.batch_id] = batch
        
        try:
            # Procesar por grupos de lenguaje
            language_groups = self._group_by_language(batch.code_snippets)
            
            for language, snippets in language_groups.items():
                model_id = embedding_engine.config.get_model_for_language(language)
                model = await embedding_engine.model_manager.load_model(model_id)
                
                # Procesar snippets del mismo lenguaje juntos
                embeddings = await self._process_language_group(
                    snippets, language, model, embedding_engine
                )
                batch.embeddings.extend(embeddings)
            
            batch.status = "completed"
            batch.processing_time_ms = int((time.time() - start_time) * 1000)
            
        except Exception as e:
            batch.status = "failed"
            logger.error(f"Error procesando batch {batch.batch_id}: {e}")
        
        finally:
            if batch.batch_id in self.active_batches:
                del self.active_batches[batch.batch_id]
        
        return batch
    
    def _group_by_language(
        self, 
        code_snippets: List[Tuple[str, ProgrammingLanguage]]
    ) -> Dict[ProgrammingLanguage, List[Tuple[int, str]]]:
        """Agrupa snippets por lenguaje."""
        groups = {}
        
        for idx, (code, language) in enumerate(code_snippets):
            if language not in groups:
                groups[language] = []
            groups[language].append((idx, code))
        
        return groups
    
    async def _process_language_group(
        self,
        snippets: List[Tuple[int, str]],
        language: ProgrammingLanguage,
        model: LoadedModel,
        embedding_engine: 'CodeEmbeddingEngine'
    ) -> List[CodeEmbedding]:
        """Procesa grupo de mismo lenguaje."""
        embeddings = []
        
        # Procesar en sub-batches si es muy grande
        for i in range(0, len(snippets), self.max_batch_size):
            sub_batch = snippets[i:i + self.max_batch_size]
            sub_embeddings = await self._process_sub_batch(
                sub_batch, language, model, embedding_engine
            )
            embeddings.extend(sub_embeddings)
        
        return embeddings
    
    async def _process_sub_batch(
        self,
        sub_batch: List[Tuple[int, str]],
        language: ProgrammingLanguage,
        model: LoadedModel,
        embedding_engine: 'CodeEmbeddingEngine'
    ) -> List[CodeEmbedding]:
        """Procesa sub-batch de códigos."""
        embeddings = []
        
        try:
            # Preprocessar códigos
            preprocessed_codes = []
            for _, code in sub_batch:
                preprocessed = await embedding_engine.preprocessor.preprocess(code, language)
                preprocessed_codes.append(preprocessed)
            
            # Tokenizar batch
            inputs = model.tokenizer(
                preprocessed_codes,
                padding=True,
                truncation=True,
                max_length=embedding_engine.config.max_code_length,
                return_tensors="pt" if TORCH_AVAILABLE else "list"
            )
            
            # Generar embeddings
            if TORCH_AVAILABLE:
                # Mover a device
                device = embedding_engine.model_manager.device
                if hasattr(inputs, 'items'):
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.model(**inputs)
                    embeddings_tensor = self._apply_pooling(
                        outputs.last_hidden_state,
                        inputs.get('attention_mask'),
                        embedding_engine.config.pooling_strategy
                    )
                    embeddings_list = embeddings_tensor.cpu().numpy().tolist()
            else:
                # Mock mode
                embeddings_list = [[0.1] * 768 for _ in sub_batch]
            
            # Crear objetos CodeEmbedding
            for i, ((original_idx, original_code), embedding_vector) in enumerate(
                zip(sub_batch, embeddings_list)
            ):
                if embedding_engine.config.normalize_embeddings:
                    embedding_vector = self._normalize_vector(embedding_vector)
                
                code_embedding = CodeEmbedding(
                    code_snippet=original_code,
                    language=language,
                    model_id=model.model_id,
                    embedding_vector=embedding_vector,
                    metadata=EmbeddingMetadata(
                        code_length=len(original_code),
                        token_count=len(inputs['input_ids'][i]) if 'input_ids' in inputs else 50,
                        preprocessing_applied=True,
                        generation_time_ms=0,  # Se calculará después
                        model_version=model.config.model_name,
                        original_index=original_idx
                    )
                )
                
                embeddings.append(code_embedding)
        
        except Exception as e:
            logger.error(f"Error procesando sub-batch: {e}")
            # Crear embeddings mock en caso de error
            for original_idx, original_code in sub_batch:
                mock_embedding = self._create_mock_embedding(original_code, language, model)
                embeddings.append(mock_embedding)
        
        return embeddings
    
    def _apply_pooling(self, hidden_states: Any, attention_mask: Any, strategy: PoolingStrategy) -> Any:
        """Aplica estrategia de pooling."""
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
        else:
            # ATTENTION_WEIGHTED - simplificado
            return hidden_states.mean(1)
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normaliza vector a longitud unitaria."""
        if NUMPY_AVAILABLE:
            norm = np.linalg.norm(vector)
            if norm == 0:
                return vector
            return (np.array(vector) / norm).tolist()
        else:
            # Normalización simple sin numpy
            norm = sum(x**2 for x in vector) ** 0.5
            if norm == 0:
                return vector
            return [x / norm for x in vector]
    
    def _create_mock_embedding(
        self, 
        code: str, 
        language: ProgrammingLanguage,
        model: LoadedModel
    ) -> CodeEmbedding:
        """Crea embedding mock en caso de error."""
        # Usar hash para generar embedding consistente
        code_hash = hashlib.md5(code.encode()).hexdigest()
        embedding_vector = [float(int(code_hash[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
        embedding_vector.extend([0.1] * (768 - len(embedding_vector)))  # Pad to 768
        
        return CodeEmbedding(
            code_snippet=code,
            language=language,
            model_id=model.model_id,
            embedding_vector=embedding_vector,
            metadata=EmbeddingMetadata(
                code_length=len(code),
                token_count=len(code.split()),
                preprocessing_applied=False,
                generation_time_ms=1,
                model_version="mock"
            )
        )


class CodeEmbeddingEngine:
    """Motor principal de generación de embeddings de código."""
    
    def __init__(self, model_manager: AIModelManager, config: EmbeddingConfig):
        """
        Inicializa el motor de embeddings.
        
        Args:
            model_manager: Gestor de modelos de IA
            config: Configuración de embeddings
        """
        self.model_manager = model_manager
        self.config = config
        self.batch_processor = BatchEmbeddingProcessor(config.batch_size)
        self.stats = EmbeddingGenerationStats()
        self.cache: Dict[str, CodeEmbedding] = {}
        self.vector_store = None  # Se inicializará después
        
        # Importar preprocessor
        from .code_preprocessor import CodePreprocessor
        self.preprocessor = CodePreprocessor()
    
    async def initialize(self) -> None:
        """Inicializa el motor de embeddings."""
        logger.info("Inicializando CodeEmbeddingEngine...")
        
        # Inicializar preprocessor
        await self.preprocessor.initialize()
        
        # Pre-cargar modelo por defecto si está disponible
        try:
            await self.model_manager.load_model(self.config.default_model)
            logger.info(f"Modelo por defecto {self.config.default_model} cargado")
        except Exception as e:
            logger.warning(f"No se pudo cargar modelo por defecto: {e}")
        
        logger.info("CodeEmbeddingEngine inicializado correctamente")
    
    async def generate_embedding(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> CodeEmbedding:
        """
        Genera embedding para un fragmento de código.
        
        Args:
            code: Código fuente
            language: Lenguaje de programación
            
        Returns:
            CodeEmbedding generado
        """
        start_time = time.time()
        
        # Verificar cache primero
        if self.config.enable_caching:
            cache_key = self._generate_cache_key(code, language)
            if cache_key in self.cache:
                self.stats.cache_hit_rate = (self.stats.cache_hit_rate + 1.0) / 2.0
                return self.cache[cache_key]
        
        try:
            # Seleccionar modelo apropiado
            model_id = self.config.get_model_for_language(language)
            model = await self.model_manager.load_model(model_id)
            
            # Preprocessar código
            preprocessed_code = await self.preprocessor.preprocess(code, language)
            
            # Tokenizar
            inputs = model.tokenizer(
                preprocessed_code,
                padding=True,
                truncation=True,
                max_length=self.config.max_code_length,
                return_tensors="pt" if TORCH_AVAILABLE else "list"
            )
            
            # Generar embedding
            if TORCH_AVAILABLE:
                device = self.model_manager.device
                if hasattr(inputs, 'items'):
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.model(**inputs)
                    embedding_tensor = self._apply_pooling(
                        outputs.last_hidden_state,
                        inputs.get('attention_mask')
                    )
                    embedding_vector = embedding_tensor.squeeze().cpu().numpy().tolist()
            else:
                # Mock embedding
                embedding_vector = self._generate_mock_embedding(code)
            
            # Normalizar si está configurado
            if self.config.normalize_embeddings:
                embedding_vector = self._normalize_vector(embedding_vector)
            
            # Crear objeto embedding
            generation_time = int((time.time() - start_time) * 1000)
            
            embedding = CodeEmbedding(
                code_snippet=code,
                language=language,
                model_id=model_id,
                embedding_vector=embedding_vector,
                metadata=EmbeddingMetadata(
                    code_length=len(code),
                    token_count=len(inputs['input_ids'][0]) if 'input_ids' in inputs else len(code.split()),
                    preprocessing_applied=True,
                    generation_time_ms=generation_time,
                    model_version=model.config.model_name
                )
            )
            
            # Cache si está habilitado
            if self.config.enable_caching:
                self.cache[cache_key] = embedding
            
            # Actualizar estadísticas
            self.stats.update_stats(generation_time)
            
            # Registrar métricas en model manager
            await self.model_manager.record_inference(model_id, generation_time, True)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            
            # Registrar error
            error_time = int((time.time() - start_time) * 1000)
            model_id = self.config.get_model_for_language(language)
            await self.model_manager.record_inference(model_id, error_time, False)
            
            # Generar embedding fallback
            return self._generate_fallback_embedding(code, language)
    
    async def generate_batch_embeddings(
        self,
        code_snippets: List[Tuple[str, ProgrammingLanguage]]
    ) -> List[CodeEmbedding]:
        """
        Genera embeddings para múltiples códigos en batch.
        
        Args:
            code_snippets: Lista de tuplas (código, lenguaje)
            
        Returns:
            Lista de embeddings generados
        """
        if not code_snippets:
            return []
        
        logger.info(f"Generando embeddings para {len(code_snippets)} snippets")
        
        # Crear batch
        batch = EmbeddingBatch(
            batch_id=f"batch_{int(time.time())}_{len(code_snippets)}",
            code_snippets=code_snippets,
            model_id=self.config.default_model
        )
        
        # Procesar batch
        processed_batch = await self.batch_processor.process_batch(batch, self)
        
        # Actualizar estadísticas
        self.stats.batch_jobs_completed += 1
        self.stats.update_stats(processed_batch.processing_time_ms, len(code_snippets))
        
        logger.info(f"Batch {batch.batch_id} completado: {processed_batch.status}")
        
        return processed_batch.embeddings
    
    def _apply_pooling(self, hidden_states: Any, attention_mask: Any = None) -> Any:
        """Aplica estrategia de pooling configurada."""
        return self.batch_processor._apply_pooling(
            hidden_states, attention_mask, self.config.pooling_strategy
        )
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normaliza vector usando la implementación del batch processor."""
        return self.batch_processor._normalize_vector(vector)
    
    def _generate_cache_key(self, code: str, language: ProgrammingLanguage) -> str:
        """Genera clave de cache para código."""
        content = f"{code}_{language.value}_{self.config.pooling_strategy.value}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_mock_embedding(self, code: str) -> List[float]:
        """Genera embedding mock basado en hash del código."""
        code_hash = hashlib.md5(code.encode()).hexdigest()
        
        # Crear embedding consistente de 768 dimensiones
        embedding = []
        for i in range(0, len(code_hash), 2):
            if len(embedding) >= 768:
                break
            hex_val = code_hash[i:i+2]
            embedding.append(float(int(hex_val, 16)) / 255.0)
        
        # Pad to 768 dimensions
        while len(embedding) < 768:
            embedding.append(0.1)
        
        return embedding[:768]
    
    def _generate_fallback_embedding(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> CodeEmbedding:
        """Genera embedding fallback en caso de error."""
        embedding_vector = self._generate_mock_embedding(code)
        
        return CodeEmbedding(
            code_snippet=code,
            language=language,
            model_id="fallback",
            embedding_vector=embedding_vector,
            metadata=EmbeddingMetadata(
                code_length=len(code),
                token_count=len(code.split()),
                preprocessing_applied=False,
                generation_time_ms=1,
                model_version="fallback"
            )
        )
    
    async def search_similar_code(
        self,
        query_code: str,
        query_language: ProgrammingLanguage,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[CodeEmbedding, float]]:
        """
        Busca código similar usando embeddings.
        
        Args:
            query_code: Código de consulta
            query_language: Lenguaje del código
            limit: Máximo número de resultados
            similarity_threshold: Umbral mínimo de similitud
            
        Returns:
            Lista de tuplas (embedding, similitud)
        """
        # Generar embedding de consulta
        query_embedding = await self.generate_embedding(query_code, query_language)
        
        # Buscar en cache local (implementación básica)
        similar_embeddings = []
        
        for cached_embedding in self.cache.values():
            # Calcular similitud coseno
            similarity = self._calculate_cosine_similarity(
                query_embedding.embedding_vector,
                cached_embedding.embedding_vector
            )
            
            if similarity >= similarity_threshold:
                similar_embeddings.append((cached_embedding, similarity))
        
        # Ordenar por similitud
        similar_embeddings.sort(key=lambda x: x[1], reverse=True)
        
        return similar_embeddings[:limit]
    
    def _calculate_cosine_similarity(
        self, 
        vec1: List[float], 
        vec2: List[float]
    ) -> float:
        """Calcula similitud coseno entre vectores."""
        if NUMPY_AVAILABLE:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        else:
            # Implementación sin numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del motor de embeddings."""
        cache_stats = {
            "total_cached": len(self.cache),
            "cache_hit_rate": self.stats.cache_hit_rate,
            "cache_size_mb": len(str(self.cache)) / (1024 * 1024)  # Aproximación
        }
        
        generation_stats = {
            "total_generated": self.stats.total_embeddings_generated,
            "average_time_ms": self.stats.average_generation_time_ms,
            "embeddings_per_second": self.stats.embeddings_per_second,
            "batch_jobs": self.stats.batch_jobs_completed,
            "error_rate": self.stats.error_rate
        }
        
        return {
            "cache": cache_stats,
            "generation": generation_stats,
            "active_batches": len(self.batch_processor.active_batches),
            "queue_size": len(self.batch_processor.processing_queue),
            "libraries_available": {
                "torch": TORCH_AVAILABLE,
                "numpy": NUMPY_AVAILABLE
            }
        }
    
    async def clear_cache(self) -> int:
        """Limpia cache de embeddings."""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"Cache limpiado: {cache_size} embeddings removidos")
        return cache_size
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Optimiza cache removiendo embeddings antiguos."""
        optimization_result = {
            "initial_size": len(self.cache),
            "removed_count": 0,
            "final_size": 0
        }
        
        # En implementación real, removeríamos embeddings menos usados
        # Por ahora, implementación simple
        if len(self.cache) > 1000:  # Si cache muy grande
            # Remover 20% más antiguos
            items_to_remove = len(self.cache) // 5
            cache_items = list(self.cache.items())
            
            # Simular removal de items "antiguos"
            for i in range(items_to_remove):
                if cache_items:
                    key, _ = cache_items.pop(0)
                    if key in self.cache:
                        del self.cache[key]
                        optimization_result["removed_count"] += 1
        
        optimization_result["final_size"] = len(self.cache)
        
        logger.info(f"Cache optimizado: {optimization_result}")
        return optimization_result
