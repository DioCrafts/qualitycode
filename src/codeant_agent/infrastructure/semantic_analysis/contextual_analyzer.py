"""
Analizador contextual para embeddings con atención.

Este módulo implementa análisis contextual avanzado usando
mecanismos de atención para generar embeddings contextuales.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from ...domain.entities.semantic_analysis import (
    ContextualEmbedding, MultiLevelEmbeddings, EmbeddingLevel
)
from ...domain.entities.ai_models import CodeEmbedding
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)

# Fallback para numpy si no está disponible
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    class MockNumPy:
        def array(self, data): return MockArray(data)
        def exp(self, data): return data
        def sum(self, data): return sum(data) if isinstance(data, list) else data
        def dot(self, a, b): return 0.5
    
    class MockArray:
        def __init__(self, data):
            self.data = data if isinstance(data, list) else [data]
        def tolist(self): return self.data
        def mean(self): return sum(self.data) / len(self.data) if self.data else 0.0
    
    np = MockNumPy()
    NUMPY_AVAILABLE = False


@dataclass
class AttentionConfig:
    """Configuración para mecanismo de atención."""
    attention_heads: int = 8
    key_dimension: int = 64
    value_dimension: int = 64
    dropout_rate: float = 0.1
    enable_self_attention: bool = True
    enable_cross_attention: bool = True
    temperature: float = 1.0


@dataclass
class ContextWindow:
    """Ventana de contexto para análisis."""
    target_embedding: List[float]
    preceding_embeddings: List[List[float]]
    following_embeddings: List[List[float]]
    parent_embeddings: List[List[float]]
    child_embeddings: List[List[float]]
    window_size: int
    
    def get_total_context_embeddings(self) -> int:
        return (len(self.preceding_embeddings) + len(self.following_embeddings) + 
                len(self.parent_embeddings) + len(self.child_embeddings))


class AttentionMechanism:
    """Mecanismo de atención para embeddings contextuales."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
    
    async def apply_self_attention(
        self,
        embeddings: List[List[float]],
        attention_mask: Optional[List[bool]] = None
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Aplica self-attention a una secuencia de embeddings.
        
        Args:
            embeddings: Lista de embeddings
            attention_mask: Máscara de atención opcional
            
        Returns:
            Tupla de (attended_embeddings, attention_weights)
        """
        if not embeddings:
            return [], []
        
        seq_len = len(embeddings)
        embed_dim = len(embeddings[0]) if embeddings else 0
        
        # Calcular attention scores (simplificado)
        attention_scores = []
        for i in range(seq_len):
            scores_for_i = []
            for j in range(seq_len):
                # Similitud dot-product entre embeddings
                score = self._calculate_attention_score(embeddings[i], embeddings[j])
                scores_for_i.append(score)
            attention_scores.append(scores_for_i)
        
        # Aplicar softmax a cada fila
        attention_weights = []
        for scores in attention_scores:
            weights = self._softmax(scores, self.config.temperature)
            attention_weights.append(weights)
        
        # Aplicar mask si está disponible
        if attention_mask:
            for i, mask_val in enumerate(attention_mask):
                if not mask_val and i < len(attention_weights):
                    attention_weights[i] = [0.0] * seq_len
        
        # Calcular attended embeddings
        attended_embeddings = []
        for i in range(seq_len):
            attended = [0.0] * embed_dim
            
            for j, weight in enumerate(attention_weights[i]):
                for k in range(embed_dim):
                    attended[k] += embeddings[j][k] * weight
            
            attended_embeddings.append(attended)
        
        return attended_embeddings, attention_weights
    
    async def apply_cross_attention(
        self,
        query_embeddings: List[List[float]],
        key_value_embeddings: List[List[float]]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Aplica cross-attention entre query y key-value embeddings.
        
        Args:
            query_embeddings: Embeddings de query
            key_value_embeddings: Embeddings de key-value
            
        Returns:
            Tupla de (attended_queries, attention_weights)
        """
        if not query_embeddings or not key_value_embeddings:
            return query_embeddings, []
        
        attended_queries = []
        all_attention_weights = []
        
        for query_emb in query_embeddings:
            # Calcular attention scores con todas las keys
            attention_scores = []
            for kv_emb in key_value_embeddings:
                score = self._calculate_attention_score(query_emb, kv_emb)
                attention_scores.append(score)
            
            # Softmax
            attention_weights = self._softmax(attention_scores, self.config.temperature)
            all_attention_weights.append(attention_weights)
            
            # Calcular attended query
            attended = [0.0] * len(query_emb)
            for i, (kv_emb, weight) in enumerate(zip(key_value_embeddings, attention_weights)):
                for j in range(len(attended)):
                    attended[j] += kv_emb[j] * weight
            
            attended_queries.append(attended)
        
        return attended_queries, all_attention_weights
    
    def _calculate_attention_score(self, query: List[float], key: List[float]) -> float:
        """Calcula score de atención entre query y key."""
        if NUMPY_AVAILABLE:
            return float(np.dot(query, key))
        else:
            # Dot product manual
            return sum(q * k for q, k in zip(query, key))
    
    def _softmax(self, scores: List[float], temperature: float = 1.0) -> List[float]:
        """Aplica softmax a scores."""
        if not scores:
            return []
        
        # Aplicar temperatura
        scaled_scores = [score / temperature for score in scores]
        
        if NUMPY_AVAILABLE:
            exp_scores = np.exp(scaled_scores)
            return (exp_scores / np.sum(exp_scores)).tolist()
        else:
            # Softmax manual con estabilidad numérica
            max_score = max(scaled_scores)
            exp_scores = [np.exp(score - max_score) for score in scaled_scores]
            sum_exp = sum(exp_scores)
            
            if sum_exp == 0:
                return [1.0 / len(scores)] * len(scores)  # Distribución uniforme
            
            return [exp_score / sum_exp for exp_score in exp_scores]


class ContextualAnalyzer:
    """Analizador principal de contexto."""
    
    def __init__(self, attention_config: Optional[AttentionConfig] = None):
        """
        Inicializa el analizador contextual.
        
        Args:
            attention_config: Configuración de atención
        """
        self.attention_config = attention_config or AttentionConfig()
        self.attention_mechanism = AttentionMechanism(self.attention_config)
        
        # Estadísticas
        self.context_stats = {
            'total_contextual_embeddings': 0,
            'average_context_window_size': 0.0,
            'average_generation_time_ms': 0.0
        }
    
    async def generate_contextual_embedding(
        self,
        target_embedding: List[float],
        context_embeddings: List[List[float]],
        context_types: List[str]
    ) -> ContextualEmbedding:
        """
        Genera embedding contextual usando atención.
        
        Args:
            target_embedding: Embedding objetivo
            context_embeddings: Embeddings de contexto
            context_types: Tipos de contexto
            
        Returns:
            Embedding contextual generado
        """
        start_time = time.time()
        
        try:
            # Preparar ventana de contexto
            context_window = self._prepare_context_window(
                target_embedding, context_embeddings, context_types
            )
            
            # Aplicar atención
            if self.attention_config.enable_self_attention:
                # Self-attention entre embeddings de contexto
                attended_context, context_attention = await self.attention_mechanism.apply_self_attention(
                    context_embeddings
                )
            else:
                attended_context = context_embeddings
                context_attention = [[1.0] * len(context_embeddings)] * len(context_embeddings)
            
            # Cross-attention entre target y contexto
            if self.attention_config.enable_cross_attention and attended_context:
                attended_target, cross_attention = await self.attention_mechanism.apply_cross_attention(
                    [target_embedding], attended_context
                )
                
                if attended_target:
                    contextual_vector = attended_target[0]
                    attention_weights = cross_attention[0] if cross_attention else []
                else:
                    contextual_vector = target_embedding
                    attention_weights = []
            else:
                # Fallback a promedio ponderado simple
                contextual_vector = self._simple_context_aggregation(
                    target_embedding, context_embeddings
                )
                attention_weights = [1.0 / len(context_embeddings)] * len(context_embeddings)
            
            generation_time = int((time.time() - start_time) * 1000)
            
            # Crear embedding contextual
            contextual_embedding = ContextualEmbedding(
                target_node_id=f"target_{hash(str(target_embedding))}",
                target_embedding=target_embedding,
                context_embeddings=context_embeddings,
                contextual_embedding=contextual_vector,
                context_window_size=len(context_embeddings),
                attention_weights=attention_weights,
                context_types=context_types,
                semantic_context={
                    "context_diversity": self._calculate_context_diversity(context_embeddings),
                    "attention_entropy": self._calculate_attention_entropy(attention_weights),
                    "generation_time_ms": generation_time
                }
            )
            
            # Actualizar estadísticas
            self._update_context_stats(contextual_embedding, generation_time)
            
            return contextual_embedding
            
        except Exception as e:
            logger.error(f"Error generando embedding contextual: {e}")
            
            # Fallback a embedding sin contexto
            return ContextualEmbedding(
                target_embedding=target_embedding,
                contextual_embedding=target_embedding,
                context_window_size=0,
                semantic_context={"error": str(e)}
            )
    
    def _prepare_context_window(
        self,
        target_embedding: List[float],
        context_embeddings: List[List[float]],
        context_types: List[str]
    ) -> ContextWindow:
        """Prepara ventana de contexto organizada por tipos."""
        # Organizar embeddings por tipo de contexto
        preceding = []
        following = []
        parent = []
        child = []
        
        for embedding, context_type in zip(context_embeddings, context_types):
            if context_type == "preceding":
                preceding.append(embedding)
            elif context_type == "following":
                following.append(embedding)
            elif context_type == "parent":
                parent.append(embedding)
            elif context_type == "child":
                child.append(embedding)
        
        return ContextWindow(
            target_embedding=target_embedding,
            preceding_embeddings=preceding,
            following_embeddings=following,
            parent_embeddings=parent,
            child_embeddings=child,
            window_size=len(context_embeddings)
        )
    
    def _simple_context_aggregation(
        self,
        target_embedding: List[float],
        context_embeddings: List[List[float]]
    ) -> List[float]:
        """Agregación simple de contexto sin atención."""
        if not context_embeddings:
            return target_embedding
        
        # Promedio ponderado: 70% target, 30% contexto
        target_weight = 0.7
        context_weight = 0.3 / len(context_embeddings)
        
        contextual = [val * target_weight for val in target_embedding]
        
        for context_emb in context_embeddings:
            for i, val in enumerate(context_emb):
                if i < len(contextual):
                    contextual[i] += val * context_weight
        
        return contextual
    
    def _calculate_context_diversity(self, context_embeddings: List[List[float]]) -> float:
        """Calcula diversidad del contexto."""
        if len(context_embeddings) < 2:
            return 0.0
        
        # Calcular similitudes entre todos los pares
        similarities = []
        for i in range(len(context_embeddings)):
            for j in range(i + 1, len(context_embeddings)):
                sim = self._cosine_similarity(context_embeddings[i], context_embeddings[j])
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        # Diversidad = 1 - similitud promedio
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity
    
    def _calculate_attention_entropy(self, attention_weights: List[float]) -> float:
        """Calcula entropía de los pesos de atención."""
        if not attention_weights:
            return 0.0
        
        # Calcular entropía de Shannon
        entropy = 0.0
        for weight in attention_weights:
            if weight > 0:
                entropy -= weight * np.log(weight) if NUMPY_AVAILABLE else weight * (weight ** 0.5)
        
        return entropy
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud coseno."""
        if NUMPY_AVAILABLE:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        else:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
    
    def _update_context_stats(self, contextual_embedding: ContextualEmbedding, generation_time: int) -> None:
        """Actualiza estadísticas contextuales."""
        self.context_stats['total_contextual_embeddings'] += 1
        
        # Actualizar promedio de window size
        total = self.context_stats['total_contextual_embeddings']
        current_avg_window = self.context_stats['average_context_window_size']
        self.context_stats['average_context_window_size'] = (
            (current_avg_window * (total - 1) + contextual_embedding.context_window_size) / total
        )
        
        # Actualizar promedio de tiempo
        current_avg_time = self.context_stats['average_generation_time_ms']
        self.context_stats['average_generation_time_ms'] = (
            (current_avg_time * (total - 1) + generation_time) / total
        )
    
    async def analyze_context_relationships(
        self,
        multilevel_embeddings: MultiLevelEmbeddings
    ) -> List[Dict[str, Any]]:
        """
        Analiza relaciones contextuales en embeddings multi-nivel.
        
        Args:
            multilevel_embeddings: Embeddings de múltiples niveles
            
        Returns:
            Lista de relaciones contextuales encontradas
        """
        relationships = []
        
        # Analizar relaciones función-función
        function_relationships = await self._analyze_function_relationships(
            multilevel_embeddings.function_embeddings
        )
        relationships.extend(function_relationships)
        
        # Analizar relaciones clase-método
        if multilevel_embeddings.class_embeddings and multilevel_embeddings.function_embeddings:
            class_method_relationships = await self._analyze_class_method_relationships(
                multilevel_embeddings.class_embeddings,
                multilevel_embeddings.function_embeddings
            )
            relationships.extend(class_method_relationships)
        
        return relationships
    
    async def _analyze_function_relationships(
        self,
        function_embeddings: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analiza relaciones entre funciones."""
        relationships = []
        func_list = list(function_embeddings.values())
        
        for i, func1 in enumerate(func_list):
            for func2 in func_list[i+1:]:
                # Calcular similitud
                similarity = self._cosine_similarity(
                    func1.embedding_vector,
                    func2.embedding_vector
                )
                
                if similarity > 0.7:  # Umbral de similitud
                    relationship = {
                        "type": "function_similarity",
                        "source": func1.function_name,
                        "target": func2.function_name,
                        "similarity_score": similarity,
                        "evidence": ["High embedding similarity"],
                        "relationship_strength": similarity
                    }
                    relationships.append(relationship)
        
        return relationships
    
    async def _analyze_class_method_relationships(
        self,
        class_embeddings: Dict[str, Any],
        function_embeddings: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analiza relaciones entre clases y métodos."""
        relationships = []
        
        # Para cada clase, encontrar métodos relacionados semánticamente
        for class_id, class_emb in class_embeddings.items():
            related_methods = []
            
            for func_id, func_emb in function_embeddings.items():
                similarity = self._cosine_similarity(
                    class_emb.embedding_vector,
                    func_emb.embedding_vector
                )
                
                if similarity > 0.6:  # Umbral para relación clase-método
                    related_methods.append({
                        "method_id": func_id,
                        "method_name": func_emb.function_name,
                        "similarity": similarity
                    })
            
            if related_methods:
                relationship = {
                    "type": "class_method_cohesion",
                    "class_name": class_emb.class_name,
                    "related_methods": related_methods,
                    "cohesion_score": sum(m["similarity"] for m in related_methods) / len(related_methods)
                }
                relationships.append(relationship)
        
        return relationships
    
    async def get_context_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del analizador contextual."""
        return {
            "total_contextual_embeddings": self.context_stats['total_contextual_embeddings'],
            "average_context_window_size": self.context_stats['average_context_window_size'],
            "average_generation_time_ms": self.context_stats['average_generation_time_ms'],
            "attention_config": {
                "attention_heads": self.attention_config.attention_heads,
                "enable_self_attention": self.attention_config.enable_self_attention,
                "enable_cross_attention": self.attention_config.enable_cross_attention,
                "temperature": self.attention_config.temperature
            }
        }
