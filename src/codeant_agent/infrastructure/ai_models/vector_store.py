"""
Sistema de almacenamiento vectorial para embeddings de código.

Este módulo implementa la integración con Qdrant para almacenamiento
y búsqueda eficiente de embeddings de código.
"""

import logging
import asyncio
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ...domain.entities.ai_models import (
    CodeEmbedding, VectorStoreConfig, SearchResult, SimilarityMatch,
    SemanticSearchQuery, SemanticSearchResult
)
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)

# Fallback para Qdrant client si no está disponible
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter, 
        FieldCondition, MatchValue, SearchRequest,
        CollectionInfo, PayloadSchemaType
    )
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
    logger.info("Qdrant client disponible")
except ImportError:
    # Mock classes para entornos sin Qdrant
    class MockQdrantClient:
        def __init__(self, url=None, **kwargs):
            self.url = url
            self.collections = {}
            logger.info(f"MockQdrantClient inicializado para {url}")
        
        def create_collection(self, collection_name, vectors_config=None, **kwargs):
            self.collections[collection_name] = {
                'vectors_config': vectors_config,
                'points': {},
                'created_at': datetime.now()
            }
            return True
        
        def get_collection(self, collection_name):
            if collection_name in self.collections:
                return MockCollectionInfo(collection_name, self.collections[collection_name])
            raise Exception(f"Collection {collection_name} not found")
        
        def upsert(self, collection_name, points, **kwargs):
            if collection_name not in self.collections:
                raise Exception(f"Collection {collection_name} not found")
            
            for point in points:
                self.collections[collection_name]['points'][point.id] = {
                    'vector': point.vector,
                    'payload': point.payload
                }
            
            return True
        
        def search(self, collection_name, query_vector, limit=10, query_filter=None, with_payload=True, **kwargs):
            if collection_name not in self.collections:
                return []
            
            points = self.collections[collection_name]['points']
            results = []
            
            # Simulación simple de búsqueda
            for point_id, data in list(points.items())[:limit]:
                # Similitud mock basada en hash
                similarity = 0.8 + (abs(hash(point_id) % 20) / 100.0)
                
                result = MockSearchResult(
                    point_id, similarity, data['payload'] if with_payload else None
                )
                results.append(result)
            
            return sorted(results, key=lambda x: x.score, reverse=True)
        
        def delete_collection(self, collection_name):
            if collection_name in self.collections:
                del self.collections[collection_name]
            return True
        
        def get_collections(self):
            return [MockCollectionInfo(name, data) for name, data in self.collections.items()]
    
    class MockCollectionInfo:
        def __init__(self, name, data):
            self.name = name
            self.config = data
            self.vectors_count = len(data.get('points', {}))
    
    class MockSearchResult:
        def __init__(self, point_id, score, payload):
            self.id = point_id
            self.score = score
            self.payload = payload
    
    class MockDistance:
        COSINE = "cosine"
        EUCLIDEAN = "euclidean"
        DOT = "dot"
    
    class MockVectorParams:
        def __init__(self, size, distance):
            self.size = size
            self.distance = distance
    
    class MockPointStruct:
        def __init__(self, id, vector, payload=None):
            self.id = id
            self.vector = vector
            self.payload = payload or {}
    
    class MockFilter:
        def __init__(self, must=None, should=None):
            self.must = must or []
            self.should = should or []
    
    class MockFieldCondition:
        def __init__(self, key, match=None):
            self.key = key
            self.match = match
    
    class MockMatchValue:
        def __init__(self, value):
            self.value = value
    
    # Usar mocks
    QdrantClient = MockQdrantClient
    Distance = MockDistance()
    VectorParams = MockVectorParams
    PointStruct = MockPointStruct
    Filter = MockFilter
    FieldCondition = MockFieldCondition
    MatchValue = MockMatchValue
    UnexpectedResponse = Exception
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant no disponible - usando mocks")


@dataclass
class VectorStoreStats:
    """Estadísticas del vector store."""
    total_embeddings: int = 0
    total_searches: int = 0
    average_search_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    collections_count: int = 0
    storage_size_mb: float = 0.0
    
    def update_search_stats(self, search_time_ms: int) -> None:
        """Actualiza estadísticas de búsqueda."""
        self.total_searches += 1
        
        # Calcular promedio móvil
        if self.total_searches == 1:
            self.average_search_time_ms = search_time_ms
        else:
            self.average_search_time_ms = (
                (self.average_search_time_ms * (self.total_searches - 1) + search_time_ms) 
                / self.total_searches
            )


@dataclass 
class CollectionMetadata:
    """Metadatos de una colección."""
    name: str
    language: Optional[ProgrammingLanguage] = None
    vector_dimension: int = 768
    distance_metric: str = "cosine"
    embedding_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class EmbeddingSearchEngine:
    """Motor de búsqueda de embeddings."""
    
    def __init__(self, vector_store: 'VectorStore'):
        self.vector_store = vector_store
        self.search_cache: Dict[str, Tuple[List[SearchResult], datetime]] = {}
        self.cache_ttl_seconds = 300  # 5 minutos
    
    async def semantic_search(
        self,
        query: SemanticSearchQuery
    ) -> SemanticSearchResult:
        """
        Realiza búsqueda semántica.
        
        Args:
            query: Query de búsqueda semántica
            
        Returns:
            Resultado de búsqueda
        """
        start_time = time.time()
        
        # Verificar cache
        cache_key = self._generate_search_cache_key(query)
        cached_result = self._get_cached_search(cache_key)
        if cached_result:
            search_time = int((time.time() - start_time) * 1000)
            return SemanticSearchResult(
                query=query,
                matches=cached_result,
                search_time_ms=search_time,
                model_used="cached"
            )
        
        # Generar embedding si no está disponible
        if not query.query_embedding:
            # Aquí normalmente generaríamos el embedding del texto
            # Por ahora usamos un embedding mock
            query.query_embedding = self._generate_mock_embedding(query.query_text)
        
        # Determinar colección
        collection_name = self._get_collection_for_search(query.target_language)
        
        try:
            # Buscar en vector store
            search_results = await self.vector_store.search_similar_embeddings(
                query_vector=query.query_embedding,
                limit=query.max_results,
                language=query.target_language,
                filters=query.filters,
                similarity_threshold=query.similarity_threshold
            )
            
            # Convertir a SimilarityMatch
            matches = []
            for result in search_results:
                match = SimilarityMatch(
                    embedding_id=result["id"],
                    similarity_score=result["score"],
                    code_snippet=result["payload"].get("code_snippet", ""),
                    language=ProgrammingLanguage(result["payload"].get("language", "unknown")),
                    metadata=result["payload"],
                    match_type="semantic"
                )
                matches.append(match)
            
            # Cache resultado
            self._cache_search_result(cache_key, matches)
            
            search_time = int((time.time() - start_time) * 1000)
            
            return SemanticSearchResult(
                query=query,
                matches=matches,
                total_candidates=len(matches),
                search_time_ms=search_time,
                model_used="vector_search"
            )
            
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            return SemanticSearchResult(
                query=query,
                matches=[],
                search_time_ms=int((time.time() - start_time) * 1000),
                model_used="error"
            )
    
    async def find_similar_code(
        self,
        code_embedding: CodeEmbedding,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[SimilarityMatch]:
        """
        Encuentra código similar a un embedding dado.
        
        Args:
            code_embedding: Embedding de referencia
            limit: Máximo resultados
            similarity_threshold: Umbral de similitud
            
        Returns:
            Lista de matches similares
        """
        query = SemanticSearchQuery(
            query_text=code_embedding.code_snippet[:100] + "...",
            query_embedding=code_embedding.embedding_vector,
            target_language=code_embedding.language,
            similarity_threshold=similarity_threshold,
            max_results=limit
        )
        
        result = await self.semantic_search(query)
        return result.matches
    
    def _generate_search_cache_key(self, query: SemanticSearchQuery) -> str:
        """Genera clave de cache para búsqueda."""
        content = f"{query.query_text}_{query.target_language}_{query.similarity_threshold}_{query.max_results}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_search(self, cache_key: str) -> Optional[List[SimilarityMatch]]:
        """Obtiene resultado de cache si está válido."""
        if cache_key in self.search_cache:
            result, timestamp = self.search_cache[cache_key]
            
            # Verificar TTL
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl_seconds:
                return result
            else:
                # Cache expirado
                del self.search_cache[cache_key]
        
        return None
    
    def _cache_search_result(self, cache_key: str, matches: List[SimilarityMatch]) -> None:
        """Cache resultado de búsqueda."""
        self.search_cache[cache_key] = (matches, datetime.now())
        
        # Limpiar cache si está muy grande
        if len(self.search_cache) > 1000:
            # Remover 25% más antiguos
            sorted_items = sorted(
                self.search_cache.items(),
                key=lambda x: x[1][1]
            )
            
            items_to_remove = len(sorted_items) // 4
            for key, _ in sorted_items[:items_to_remove]:
                del self.search_cache[key]
    
    def _get_collection_for_search(self, language: Optional[ProgrammingLanguage]) -> str:
        """Obtiene colección apropiada para búsqueda."""
        if language:
            return self.vector_store.config.get_collection_name(language)
        return self.vector_store.config.default_collection
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Genera embedding mock para texto."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        embedding = []
        for i in range(0, len(text_hash), 2):
            if len(embedding) >= 768:
                break
            hex_val = text_hash[i:i+2]
            embedding.append(float(int(hex_val, 16)) / 255.0)
        
        # Pad to 768 dimensions
        while len(embedding) < 768:
            embedding.append(0.1)
        
        return embedding[:768]


class VectorStore:
    """Sistema de almacenamiento vectorial principal."""
    
    def __init__(self, config: VectorStoreConfig):
        """
        Inicializa el vector store.
        
        Args:
            config: Configuración del vector store
        """
        self.config = config
        self.client = None
        self.search_engine = None
        self.collections_metadata: Dict[str, CollectionMetadata] = {}
        self.stats = VectorStoreStats()
        self.is_connected = False
    
    @classmethod
    async def create(cls, config: VectorStoreConfig) -> 'VectorStore':
        """
        Crea e inicializa un VectorStore.
        
        Args:
            config: Configuración del vector store
            
        Returns:
            VectorStore inicializado
        """
        store = cls(config)
        await store.initialize()
        return store
    
    async def initialize(self) -> None:
        """Inicializa la conexión y configuración."""
        logger.info(f"Inicializando VectorStore con Qdrant en {self.config.qdrant_url}")
        
        try:
            # Crear cliente Qdrant
            self.client = QdrantClient(
                url=self.config.qdrant_url,
                timeout=self.config.timeout_seconds
            )
            
            # Verificar conexión
            if QDRANT_AVAILABLE:
                collections = self.client.get_collections()
                logger.info(f"Conectado a Qdrant - {len(collections)} colecciones encontradas")
            
            # Inicializar colecciones por defecto
            await self._initialize_default_collections()
            
            # Inicializar search engine
            self.search_engine = EmbeddingSearchEngine(self)
            
            self.is_connected = True
            logger.info("VectorStore inicializado correctamente")
            
        except Exception as e:
            logger.warning(f"Error conectando a Qdrant: {e}")
            logger.info("Continuando en modo mock...")
            
            # Fallback a cliente mock
            self.client = QdrantClient(url=self.config.qdrant_url)
            await self._initialize_default_collections()
            self.search_engine = EmbeddingSearchEngine(self)
            self.is_connected = False
    
    async def _initialize_default_collections(self) -> None:
        """Inicializa colecciones por defecto."""
        default_collections = [
            (self.config.default_collection, None),
            ("code_embeddings_python", ProgrammingLanguage.PYTHON),
            ("code_embeddings_javascript", ProgrammingLanguage.JAVASCRIPT),
            ("code_embeddings_typescript", ProgrammingLanguage.TYPESCRIPT),
            ("code_embeddings_rust", ProgrammingLanguage.RUST),
            ("code_embeddings_java", ProgrammingLanguage.JAVA),
            ("code_embeddings_go", ProgrammingLanguage.GO),
        ]
        
        for collection_name, language in default_collections:
            try:
                await self.create_collection(collection_name, language)
            except Exception as e:
                logger.debug(f"Colección {collection_name} posiblemente ya existe: {e}")
    
    async def create_collection(
        self, 
        collection_name: str, 
        language: Optional[ProgrammingLanguage] = None
    ) -> bool:
        """
        Crea una nueva colección.
        
        Args:
            collection_name: Nombre de la colección
            language: Lenguaje asociado (opcional)
            
        Returns:
            True si se creó exitosamente
        """
        try:
            # Crear colección en Qdrant
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_dimension,
                    distance=getattr(Distance, self.config.distance_metric.upper(), Distance.COSINE)
                )
            )
            
            # Guardar metadata
            self.collections_metadata[collection_name] = CollectionMetadata(
                name=collection_name,
                language=language,
                vector_dimension=self.config.vector_dimension,
                distance_metric=self.config.distance_metric
            )
            
            self.stats.collections_count += 1
            logger.debug(f"Colección {collection_name} creada exitosamente")
            return True
            
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug(f"Colección {collection_name} ya existe")
                # Añadir a metadata si no está
                if collection_name not in self.collections_metadata:
                    self.collections_metadata[collection_name] = CollectionMetadata(
                        name=collection_name,
                        language=language
                    )
                return True
            else:
                logger.error(f"Error creando colección {collection_name}: {e}")
                return False
    
    async def store_embedding(self, embedding: CodeEmbedding) -> bool:
        """
        Almacena un embedding en el vector store.
        
        Args:
            embedding: Embedding a almacenar
            
        Returns:
            True si se almacenó exitosamente
        """
        try:
            collection_name = self.config.get_collection_name(embedding.language)
            
            # Preparar payload
            payload = {
                "code_snippet": embedding.code_snippet,
                "language": embedding.language.value,
                "model_id": embedding.model_id,
                "code_length": embedding.metadata.code_length,
                "token_count": embedding.metadata.token_count,
                "generation_time_ms": embedding.metadata.generation_time_ms,
                "model_version": embedding.metadata.model_version,
                "created_at": embedding.created_at.isoformat()
            }
            
            # Añadir campos opcionales
            if embedding.metadata.file_path:
                payload["file_path"] = str(embedding.metadata.file_path)
            if embedding.metadata.function_name:
                payload["function_name"] = embedding.metadata.function_name
            if embedding.metadata.class_name:
                payload["class_name"] = embedding.metadata.class_name
            
            # Crear punto
            point = PointStruct(
                id=embedding.id,
                vector=embedding.embedding_vector,
                payload=payload
            )
            
            # Almacenar en Qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            # Actualizar estadísticas
            self.stats.total_embeddings += 1
            if collection_name in self.collections_metadata:
                self.collections_metadata[collection_name].embedding_count += 1
                self.collections_metadata[collection_name].last_updated = datetime.now()
            
            logger.debug(f"Embedding {embedding.id} almacenado en {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error almacenando embedding {embedding.id}: {e}")
            return False
    
    async def search_similar_embeddings(
        self,
        query_vector: List[float],
        limit: int = 10,
        language: Optional[ProgrammingLanguage] = None,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Busca embeddings similares.
        
        Args:
            query_vector: Vector de consulta
            limit: Número máximo de resultados
            language: Lenguaje objetivo (opcional)
            filters: Filtros adicionales
            similarity_threshold: Umbral mínimo de similitud
            
        Returns:
            Lista de resultados de búsqueda
        """
        start_time = time.time()
        
        try:
            # Determinar colección
            collection_name = (
                self.config.get_collection_name(language)
                if language
                else self.config.default_collection
            )
            
            # Construir filtros
            query_filter = None
            if filters or language:
                must_conditions = []
                
                if language:
                    must_conditions.append(
                        FieldCondition(
                            key="language",
                            match=MatchValue(value=language.value)
                        )
                    )
                
                if filters:
                    for key, value in filters.items():
                        must_conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )
                
                if must_conditions:
                    query_filter = Filter(must=must_conditions)
            
            # Buscar en Qdrant
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                score_threshold=similarity_threshold
            )
            
            # Convertir resultados
            search_results = []
            for point in results:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                }
                search_results.append(result)
            
            # Actualizar estadísticas
            search_time = int((time.time() - start_time) * 1000)
            self.stats.update_search_stats(search_time)
            
            logger.debug(f"Búsqueda completada: {len(search_results)} resultados en {search_time}ms")
            return search_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda similar: {e}")
            return []
    
    async def delete_embedding(self, embedding_id: str, language: Optional[ProgrammingLanguage] = None) -> bool:
        """
        Elimina un embedding.
        
        Args:
            embedding_id: ID del embedding
            language: Lenguaje (para determinar colección)
            
        Returns:
            True si se eliminó exitosamente
        """
        try:
            collection_name = (
                self.config.get_collection_name(language)
                if language
                else self.config.default_collection
            )
            
            # Eliminar de Qdrant
            self.client.delete(
                collection_name=collection_name,
                points_selector=[embedding_id]
            )
            
            # Actualizar estadísticas
            self.stats.total_embeddings = max(0, self.stats.total_embeddings - 1)
            if collection_name in self.collections_metadata:
                metadata = self.collections_metadata[collection_name]
                metadata.embedding_count = max(0, metadata.embedding_count - 1)
                metadata.last_updated = datetime.now()
            
            logger.debug(f"Embedding {embedding_id} eliminado de {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando embedding {embedding_id}: {e}")
            return False
    
    async def get_embedding_by_id(self, embedding_id: str, language: Optional[ProgrammingLanguage] = None) -> Optional[Dict]:
        """
        Obtiene un embedding por ID.
        
        Args:
            embedding_id: ID del embedding
            language: Lenguaje para determinar colección
            
        Returns:
            Datos del embedding o None
        """
        try:
            collection_name = (
                self.config.get_collection_name(language)
                if language
                else self.config.default_collection
            )
            
            # Obtener de Qdrant
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[embedding_id],
                with_payload=True,
                with_vectors=True
            )
            
            if points:
                point = points[0]
                return {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo embedding {embedding_id}: {e}")
            return None
    
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de una colección.
        
        Args:
            collection_name: Nombre de la colección
            
        Returns:
            Información de la colección
        """
        try:
            if QDRANT_AVAILABLE:
                info = self.client.get_collection(collection_name)
                
                return {
                    "name": collection_name,
                    "vectors_count": info.vectors_count,
                    "indexed_vectors_count": getattr(info, 'indexed_vectors_count', 0),
                    "points_count": getattr(info, 'points_count', 0),
                    "config": {
                        "distance": info.config.params.vectors.distance,
                        "vector_size": info.config.params.vectors.size
                    }
                }
            else:
                # Mock mode
                if collection_name in self.collections_metadata:
                    metadata = self.collections_metadata[collection_name]
                    return {
                        "name": collection_name,
                        "vectors_count": metadata.embedding_count,
                        "points_count": metadata.embedding_count,
                        "language": metadata.language.value if metadata.language else None,
                        "created_at": metadata.created_at.isoformat()
                    }
                return None
            
        except Exception as e:
            logger.error(f"Error obteniendo info de colección {collection_name}: {e}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del vector store."""
        collection_stats = {}
        
        for name, metadata in self.collections_metadata.items():
            collection_stats[name] = {
                "embedding_count": metadata.embedding_count,
                "language": metadata.language.value if metadata.language else None,
                "last_updated": metadata.last_updated.isoformat()
            }
        
        return {
            "connection_status": "connected" if self.is_connected else "mock_mode",
            "total_embeddings": self.stats.total_embeddings,
            "total_searches": self.stats.total_searches,
            "average_search_time_ms": self.stats.average_search_time_ms,
            "collections_count": self.stats.collections_count,
            "collections": collection_stats,
            "qdrant_available": QDRANT_AVAILABLE
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Ejecuta health check del vector store."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "qdrant_available": QDRANT_AVAILABLE,
            "connection_status": "unknown",
            "collections_accessible": 0,
            "total_embeddings": self.stats.total_embeddings,
            "issues": [],
            "recommendations": []
        }
        
        try:
            if QDRANT_AVAILABLE and self.client:
                # Verificar acceso a colecciones
                collections = self.client.get_collections()
                health_status["collections_accessible"] = len(collections)
                health_status["connection_status"] = "connected"
            else:
                health_status["connection_status"] = "mock_mode"
                health_status["collections_accessible"] = len(self.collections_metadata)
        
        except Exception as e:
            health_status["connection_status"] = "error"
            health_status["issues"].append(f"Connection error: {e}")
        
        # Analizar salud
        if health_status["connection_status"] == "error":
            health_status["recommendations"].append("Check Qdrant server status and configuration")
        
        if health_status["total_embeddings"] == 0:
            health_status["recommendations"].append("No embeddings stored - consider adding some code embeddings")
        
        if not QDRANT_AVAILABLE:
            health_status["recommendations"].append("Install qdrant-client for full functionality")
        
        return health_status
    
    async def optimize_collections(self) -> Dict[str, Any]:
        """Optimiza colecciones (implementación futura)."""
        return {
            "optimization_completed": True,
            "collections_optimized": len(self.collections_metadata),
            "space_saved_mb": 0.0,
            "note": "Optimization placeholder - not implemented yet"
        }
