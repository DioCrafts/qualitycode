"""
Motor de búsqueda semántica avanzado.

Este módulo implementa búsqueda semántica de código usando
lenguaje natural, análisis de intención y ranking inteligente.
"""

import logging
import asyncio
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from ...domain.entities.semantic_analysis import (
    SemanticSearchConfig, ProcessedQuery, SemanticSearchResult,
    SemanticSearchResultItem, QueryIntent, CodeConcept, ConceptType,
    SimilarityExplanation, SemanticFeature, MultiLevelEmbeddings
)
from ...domain.entities.ai_models import CodeEmbedding, SemanticSearchQuery
from ...domain.value_objects.programming_language import ProgrammingLanguage
from ..ai_models.embedding_engine import CodeEmbeddingEngine
from ..ai_models.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SearchContext:
    """Contexto de búsqueda."""
    target_languages: List[ProgrammingLanguage]
    file_patterns: List[str]
    project_constraints: Dict[str, Any]
    user_preferences: Dict[str, Any]
    
    def __post_init__(self):
        if not hasattr(self, 'target_languages'):
            self.target_languages = []
        if not hasattr(self, 'file_patterns'):
            self.file_patterns = []
        if not hasattr(self, 'project_constraints'):
            self.project_constraints = {}
        if not hasattr(self, 'user_preferences'):
            self.user_preferences = {}


class QueryProcessor:
    """Procesador de queries de lenguaje natural."""
    
    def __init__(self, embedding_engine: CodeEmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.intent_keywords = self._initialize_intent_keywords()
        self.concept_keywords = self._initialize_concept_keywords()
    
    def _initialize_intent_keywords(self) -> Dict[QueryIntent, List[str]]:
        """Inicializa keywords para detección de intención."""
        return {
            QueryIntent.FIND_SIMILAR_CODE: [
                "similar", "like", "equivalent", "comparable", "same", "matching"
            ],
            QueryIntent.FIND_BY_FUNCTION: [
                "function", "method", "procedure", "subroutine", "callable"
            ],
            QueryIntent.FIND_BY_PATTERN: [
                "pattern", "design pattern", "singleton", "factory", "observer", "template"
            ],
            QueryIntent.FIND_BY_BEHAVIOR: [
                "behavior", "does", "performs", "executes", "handles", "processes"
            ],
            QueryIntent.FIND_BY_PURPOSE: [
                "purpose", "goal", "objective", "intent", "reason", "why"
            ],
            QueryIntent.FIND_IMPLEMENTATIONS: [
                "implementation", "implements", "realization", "concrete", "actual"
            ],
            QueryIntent.FIND_ALTERNATIVES: [
                "alternative", "different", "other", "another", "variation", "variant"
            ]
        }
    
    def _initialize_concept_keywords(self) -> Dict[ConceptType, List[str]]:
        """Inicializa keywords para detección de conceptos."""
        return {
            ConceptType.ALGORITHM: [
                "sort", "search", "hash", "tree", "graph", "algorithm", "optimize"
            ],
            ConceptType.DATA_STRUCTURE: [
                "array", "list", "map", "set", "queue", "stack", "tree", "graph"
            ],
            ConceptType.DESIGN_PATTERN: [
                "singleton", "factory", "observer", "strategy", "decorator", "facade"
            ],
            ConceptType.DOMAIN: [
                "user", "payment", "order", "product", "customer", "invoice", "account"
            ],
            ConceptType.FRAMEWORK: [
                "react", "angular", "django", "flask", "spring", "express"
            ],
            ConceptType.LIBRARY: [
                "numpy", "pandas", "lodash", "axios", "requests", "json"
            ]
        }
    
    async def process_natural_language_query(self, query: str) -> ProcessedQuery:
        """
        Procesa query de lenguaje natural.
        
        Args:
            query: Query en lenguaje natural
            
        Returns:
            Query procesada con intención y conceptos
        """
        start_time = time.time()
        
        query_lower = query.lower()
        
        # Detectar intención
        detected_intent = self._detect_query_intent(query_lower)
        
        # Extraer conceptos
        concepts = self._extract_concepts(query_lower)
        
        # Expandir términos
        expanded_terms = self._expand_query_terms(query_lower)
        
        # Extraer constrains
        constraints = self._extract_constraints(query_lower)
        
        # Generar embedding del query
        query_embedding = await self._generate_query_embedding(query, detected_intent)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        processed = ProcessedQuery(
            original_query=query,
            intent=detected_intent,
            concepts=concepts,
            constraints=constraints,
            expanded_terms=expanded_terms,
            query_embedding=query_embedding,
            processing_time_ms=processing_time
        )
        
        return processed
    
    def _detect_query_intent(self, query: str) -> QueryIntent:
        """Detecta la intención del query."""
        # Buscar keywords de intención
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in query for keyword in keywords):
                return intent
        
        # Análisis por patrones comunes
        if any(word in query for word in ["how", "what", "where", "which"]):
            return QueryIntent.FIND_BY_PURPOSE
        
        if "like" in query or "similar" in query:
            return QueryIntent.FIND_SIMILAR_CODE
        
        if "implement" in query or "create" in query:
            return QueryIntent.FIND_IMPLEMENTATIONS
        
        # Default
        return QueryIntent.FIND_SIMILAR_CODE
    
    def _extract_concepts(self, query: str) -> List[CodeConcept]:
        """Extrae conceptos del query."""
        concepts = []
        
        for concept_type, keywords in self.concept_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    concept = CodeConcept(
                        concept_type=concept_type,
                        name=keyword,
                        confidence=0.8,
                        context=query
                    )
                    concepts.append(concept)
        
        return concepts
    
    def _expand_query_terms(self, query: str) -> List[str]:
        """Expande términos del query con sinónimos."""
        # Diccionario de sinónimos básico
        synonyms = {
            "function": ["method", "procedure", "subroutine"],
            "class": ["object", "type", "entity"],
            "variable": ["var", "field", "property"],
            "algorithm": ["algo", "procedure", "method"],
            "data": ["information", "content", "values"],
            "error": ["exception", "failure", "problem"],
            "test": ["testing", "validation", "verification"]
        }
        
        expanded = []
        words = query.split()
        
        for word in words:
            expanded.append(word)
            if word in synonyms:
                expanded.extend(synonyms[word])
        
        return list(set(expanded))  # Remover duplicados
    
    def _extract_constraints(self, query: str) -> List[str]:
        """Extrae constrains del query."""
        constraints = []
        
        # Detectar constrains de lenguaje
        languages = ["python", "javascript", "java", "rust", "go", "typescript"]
        for lang in languages:
            if lang in query:
                constraints.append(f"language:{lang}")
        
        # Detectar constrains de complejidad
        if any(word in query for word in ["simple", "basic", "easy"]):
            constraints.append("complexity:low")
        elif any(word in query for word in ["complex", "advanced", "sophisticated"]):
            constraints.append("complexity:high")
        
        # Detectar constrains de tamaño
        if any(word in query for word in ["short", "small", "brief"]):
            constraints.append("size:small")
        elif any(word in query for word in ["long", "large", "detailed"]):
            constraints.append("size:large")
        
        return constraints
    
    async def _generate_query_embedding(self, query: str, intent: QueryIntent) -> List[float]:
        """Genera embedding para el query."""
        # Generar embedding base
        try:
            embedding_result = await self.embedding_engine.generate_embedding(
                query, ProgrammingLanguage.PYTHON  # Default language for queries
            )
            return embedding_result.embedding_vector
        except Exception as e:
            logger.warning(f"Error generando embedding de query: {e}")
            # Fallback a embedding mock
            return self._generate_mock_query_embedding(query)
    
    def _generate_mock_query_embedding(self, query: str) -> List[float]:
        """Genera embedding mock para query."""
        import hashlib
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        embedding = []
        
        for i in range(0, len(query_hash), 2):
            if len(embedding) >= 768:
                break
            hex_val = query_hash[i:i+2]
            embedding.append(float(int(hex_val, 16)) / 255.0)
        
        # Pad to 768 dimensions
        while len(embedding) < 768:
            embedding.append(0.1)
        
        return embedding[:768]


class ResultRanker:
    """Ranking inteligente de resultados de búsqueda."""
    
    def __init__(self, config: SemanticSearchConfig):
        self.config = config
    
    async def rank_results(
        self,
        results: List[Dict[str, Any]],
        processed_query: ProcessedQuery
    ) -> List[SemanticSearchResultItem]:
        """
        Rankea resultados de búsqueda.
        
        Args:
            results: Resultados crudos de búsqueda
            processed_query: Query procesada
            
        Returns:
            Resultados rankeados
        """
        ranked_items = []
        
        for result in results:
            # Convertir a SemanticSearchResultItem
            item = await self._convert_to_result_item(result, processed_query)
            
            # Calcular scores de ranking
            item = await self._calculate_ranking_scores(item, processed_query)
            
            ranked_items.append(item)
        
        # Ordenar por combined_similarity
        ranked_items.sort(key=lambda x: x.combined_similarity, reverse=True)
        
        return ranked_items[:self.config.max_results]
    
    async def _convert_to_result_item(
        self, 
        result: Dict[str, Any], 
        query: ProcessedQuery
    ) -> SemanticSearchResultItem:
        """Convierte resultado crudo a SemanticSearchResultItem."""
        # Extraer información del resultado
        code_snippet = result.get("payload", {}).get("code_snippet", "")
        language_str = result.get("payload", {}).get("language", "unknown")
        
        # Convertir language string a enum
        try:
            language = ProgrammingLanguage(language_str)
        except ValueError:
            language = ProgrammingLanguage.UNKNOWN
        
        # Similitud básica desde el vector store
        cosine_similarity = result.get("score", 0.0)
        
        return SemanticSearchResultItem(
            id=result.get("id", "unknown"),
            code_snippet=code_snippet,
            language=language,
            cosine_similarity=cosine_similarity,
            semantic_similarity=cosine_similarity,  # Inicialmente igual
            structural_similarity=0.0,  # Se calculará después
            combined_similarity=cosine_similarity
        )
    
    async def _calculate_ranking_scores(
        self, 
        item: SemanticSearchResultItem, 
        query: ProcessedQuery
    ) -> SemanticSearchResultItem:
        """Calcula scores de ranking avanzados."""
        # Similitud semántica mejorada
        item.semantic_similarity = await self._calculate_semantic_similarity(item, query)
        
        # Similitud estructural
        item.structural_similarity = self._calculate_structural_similarity(item, query)
        
        # Score de matching de intención
        item.intent_match_score = self._calculate_intent_match(item, query)
        
        # Extraer características semánticas
        item.semantic_features = await self._extract_semantic_features(item)
        
        # Calcular confianza
        item.confidence = self._calculate_confidence(item)
        
        # Similitud combinada con pesos
        item.combined_similarity = self._calculate_combined_similarity(item, query)
        
        # Generar explicación
        item.explanation = self._generate_explanation(item, query)
        
        return item
    
    async def _calculate_semantic_similarity(
        self, 
        item: SemanticSearchResultItem, 
        query: ProcessedQuery
    ) -> float:
        """Calcula similitud semántica avanzada."""
        base_similarity = item.cosine_similarity
        
        # Boost por conceptos coincidentes
        concept_boost = 0.0
        if query.concepts:
            code_lower = item.code_snippet.lower()
            matching_concepts = sum(
                1 for concept in query.concepts 
                if concept.name.lower() in code_lower
            )
            concept_boost = (matching_concepts / len(query.concepts)) * 0.2
        
        # Boost por palabras clave del query
        keyword_boost = 0.0
        if query.expanded_terms:
            code_lower = item.code_snippet.lower()
            matching_terms = sum(
                1 for term in query.expanded_terms
                if term.lower() in code_lower
            )
            keyword_boost = (matching_terms / len(query.expanded_terms)) * 0.15
        
        return min(1.0, base_similarity + concept_boost + keyword_boost)
    
    def _calculate_structural_similarity(
        self, 
        item: SemanticSearchResultItem, 
        query: ProcessedQuery
    ) -> float:
        """Calcula similitud estructural."""
        # Análisis básico de estructura
        code = item.code_snippet
        
        # Contar estructuras de control
        control_structures = ['if', 'for', 'while', 'try', 'switch', 'case']
        structure_count = sum(code.lower().count(struct) for struct in control_structures)
        
        # Contar funciones/métodos
        function_count = sum(code.count(pattern) for pattern in ['def ', 'function ', 'fn '])
        
        # Contar clases
        class_count = sum(code.count(pattern) for pattern in ['class ', 'struct ', 'interface '])
        
        # Score basado en complejidad estructural
        if query.intent == QueryIntent.FIND_BY_PATTERN:
            # Para búsqueda de patrones, priorizar código con clases
            return min(1.0, (class_count * 0.4 + function_count * 0.3 + structure_count * 0.1))
        elif query.intent == QueryIntent.FIND_BY_FUNCTION:
            # Para búsqueda de funciones, priorizar código con funciones
            return min(1.0, (function_count * 0.5 + structure_count * 0.2))
        else:
            # Score general basado en balance
            return min(1.0, (function_count * 0.3 + class_count * 0.2 + structure_count * 0.2))
    
    def _calculate_intent_match(self, item: SemanticSearchResultItem, query: ProcessedQuery) -> float:
        """Calcula qué tan bien el resultado coincide con la intención."""
        code = item.code_snippet.lower()
        
        if query.intent == QueryIntent.FIND_BY_FUNCTION:
            # Buscar declaraciones de función
            return 1.0 if any(pattern in code for pattern in ['def ', 'function ', 'fn ']) else 0.3
        
        elif query.intent == QueryIntent.FIND_BY_PATTERN:
            # Buscar indicadores de patrones
            pattern_indicators = ['class', 'interface', 'extends', 'implements', 'singleton', 'factory']
            return min(1.0, sum(0.2 for indicator in pattern_indicators if indicator in code))
        
        elif query.intent == QueryIntent.FIND_BY_BEHAVIOR:
            # Buscar lógica de comportamiento
            behavior_indicators = ['if', 'for', 'while', 'switch', 'try', 'process', 'handle']
            return min(1.0, sum(0.15 for indicator in behavior_indicators if indicator in code))
        
        else:
            return 0.7  # Score neutral para otras intenciones
    
    async def _extract_semantic_features(self, item: SemanticSearchResultItem) -> List[SemanticFeature]:
        """Extrae características semánticas del resultado."""
        features = []
        code = item.code_snippet
        
        # Feature: Complejidad
        complexity_score = self._estimate_code_complexity(code)
        features.append(SemanticFeature(
            feature_type="complexity",
            value=complexity_score,
            description=f"Code complexity score: {complexity_score:.2f}",
            confidence=0.8
        ))
        
        # Feature: Abstracción
        abstraction_score = self._estimate_abstraction_level(code)
        features.append(SemanticFeature(
            feature_type="abstraction",
            value=abstraction_score,
            description=f"Code abstraction level: {abstraction_score:.2f}",
            confidence=0.7
        ))
        
        # Feature: Reusabilidad
        reusability_score = self._estimate_reusability(code)
        features.append(SemanticFeature(
            feature_type="reusability",
            value=reusability_score,
            description=f"Code reusability score: {reusability_score:.2f}",
            confidence=0.6
        ))
        
        return features
    
    def _calculate_confidence(self, item: SemanticSearchResultItem) -> float:
        """Calcula confianza general del resultado."""
        # Factores de confianza
        similarity_confidence = item.combined_similarity
        feature_confidence = (
            sum(f.confidence for f in item.semantic_features) / 
            len(item.semantic_features) if item.semantic_features else 0.5
        )
        
        # Confianza por longitud de código
        code_length = len(item.code_snippet)
        length_confidence = 1.0
        if code_length < 50:  # Muy corto
            length_confidence = 0.6
        elif code_length > 2000:  # Muy largo
            length_confidence = 0.8
        
        # Combinar factores
        overall_confidence = (
            similarity_confidence * 0.4 +
            feature_confidence * 0.3 +
            length_confidence * 0.3
        )
        
        return min(1.0, overall_confidence)
    
    def _calculate_combined_similarity(self, item: SemanticSearchResultItem, query: ProcessedQuery) -> float:
        """Calcula similitud combinada con pesos adaptativos."""
        # Pesos base
        cosine_weight = 0.4
        semantic_weight = 0.3
        structural_weight = 0.2
        intent_weight = 0.1
        
        # Ajustar pesos según la intención
        if query.intent == QueryIntent.FIND_BY_PATTERN:
            structural_weight = 0.4
            semantic_weight = 0.3
            cosine_weight = 0.2
        elif query.intent == QueryIntent.FIND_BY_BEHAVIOR:
            semantic_weight = 0.5
            intent_weight = 0.2
            cosine_weight = 0.2
        
        combined = (
            item.cosine_similarity * cosine_weight +
            item.semantic_similarity * semantic_weight +
            item.structural_similarity * structural_weight +
            item.intent_match_score * intent_weight
        )
        
        return min(1.0, combined)
    
    def _generate_explanation(self, item: SemanticSearchResultItem, query: ProcessedQuery) -> SimilarityExplanation:
        """Genera explicación de la similitud."""
        key_factors = []
        
        if item.cosine_similarity > 0.8:
            key_factors.append("High vector similarity")
        
        if item.semantic_similarity > 0.8:
            key_factors.append("Strong semantic match")
        
        if item.structural_similarity > 0.7:
            key_factors.append("Similar code structure")
        
        if item.intent_match_score > 0.8:
            key_factors.append("Intent alignment")
        
        # Verificar conceptos coincidentes
        code_lower = item.code_snippet.lower()
        matching_concepts = [
            concept.name for concept in query.concepts
            if concept.name.lower() in code_lower
        ]
        
        if matching_concepts:
            key_factors.append(f"Matching concepts: {', '.join(matching_concepts)}")
        
        if not key_factors:
            key_factors.append("Basic similarity detected")
        
        explanation_text = f"This code matches your query with {item.combined_similarity:.1%} similarity. "
        explanation_text += "Key factors: " + ", ".join(key_factors)
        
        return SimilarityExplanation(
            similarity_score=item.combined_similarity,
            similarity_type="semantic",
            explanation_text=explanation_text,
            key_factors=key_factors,
            confidence=item.confidence
        )
    
    def _estimate_code_complexity(self, code: str) -> float:
        """Estima complejidad del código (0-1)."""
        # Contar indicadores de complejidad
        complexity_indicators = ['if', 'for', 'while', 'try', 'nested', 'complex']
        complexity_count = sum(code.lower().count(indicator) for indicator in complexity_indicators)
        
        # Normalizar basado en longitud del código
        lines = len(code.splitlines())
        normalized_complexity = complexity_count / max(1, lines / 10)
        
        return min(1.0, normalized_complexity / 5.0)  # Cap at 1.0
    
    def _estimate_abstraction_level(self, code: str) -> float:
        """Estima nivel de abstracción (0-1)."""
        # Indicadores de alta abstracción
        high_abstraction = ['class', 'interface', 'abstract', 'template', 'generic']
        high_count = sum(code.lower().count(indicator) for indicator in high_abstraction)
        
        # Indicadores de baja abstracción
        low_abstraction = ['malloc', 'free', 'pointer', 'memory', 'byte', 'bit']
        low_count = sum(code.lower().count(indicator) for indicator in low_abstraction)
        
        # Score basado en balance
        if high_count > low_count:
            return min(1.0, (high_count - low_count) / 10.0 + 0.5)
        else:
            return max(0.0, 0.5 - (low_count - high_count) / 10.0)
    
    def _estimate_reusability(self, code: str) -> float:
        """Estima reusabilidad del código (0-1)."""
        base_score = 0.5
        
        # Factores positivos
        if 'def ' in code or 'function ' in code:
            base_score += 0.2  # Funciones son más reusables
        
        if 'class ' in code:
            base_score += 0.2  # Clases son reusables
        
        if any(word in code.lower() for word in ['parameter', 'argument', 'config']):
            base_score += 0.1  # Parametrización aumenta reusabilidad
        
        # Factores negativos
        if any(word in code.lower() for word in ['hardcode', 'specific', 'custom']):
            base_score -= 0.2
        
        if code.count('global ') > 0:
            base_score -= 0.1  # Variables globales reducen reusabilidad
        
        return max(0.0, min(1.0, base_score))


class SemanticSearchEngine:
    """Motor de búsqueda semántica principal."""
    
    def __init__(
        self,
        embedding_engine: CodeEmbeddingEngine,
        vector_store: VectorStore,
        config: Optional[SemanticSearchConfig] = None
    ):
        """
        Inicializa el motor de búsqueda semántica.
        
        Args:
            embedding_engine: Motor de embeddings
            vector_store: Almacén vectorial
            config: Configuración de búsqueda
        """
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.config = config or SemanticSearchConfig()
        
        # Componentes especializados
        self.query_processor = QueryProcessor(embedding_engine)
        self.result_ranker = ResultRanker(self.config)
        
        # Estadísticas
        self.search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'average_search_time_ms': 0.0,
            'average_results_returned': 0.0
        }
    
    async def search_by_natural_language(
        self,
        query: str,
        languages: Optional[List[ProgrammingLanguage]] = None,
        context: Optional[SearchContext] = None
    ) -> SemanticSearchResult:
        """
        Busca código usando lenguaje natural.
        
        Args:
            query: Query en lenguaje natural
            languages: Lenguajes objetivo
            context: Contexto de búsqueda
            
        Returns:
            Resultado de búsqueda semántica
        """
        start_time = time.time()
        
        try:
            # Procesar query
            processed_query = await self.query_processor.process_natural_language_query(query)
            
            # Determinar lenguajes objetivo
            target_languages = languages or [ProgrammingLanguage.PYTHON]
            
            # Buscar en vector store
            search_results = []
            total_candidates = 0
            
            for language in target_languages:
                if processed_query.has_embedding():
                    results = await self.vector_store.search_similar_embeddings(
                        query_vector=processed_query.query_embedding,
                        limit=self.config.max_results // len(target_languages),
                        language=language,
                        similarity_threshold=self.config.similarity_threshold
                    )
                    search_results.extend(results)
                    total_candidates += len(results)
            
            # Rankear resultados
            ranked_results = await self.result_ranker.rank_results(search_results, processed_query)
            
            # Aplicar filtros semánticos si está habilitado
            if self.config.enable_semantic_filtering:
                ranked_results = self._apply_semantic_filters(ranked_results, processed_query)
            
            search_time = int((time.time() - start_time) * 1000)
            
            # Crear resultado
            result = SemanticSearchResult(
                query=query,
                processed_query=processed_query,
                results=ranked_results,
                total_candidates=total_candidates,
                total_results=len(ranked_results),
                search_time_ms=search_time,
                languages_searched=target_languages,
                query_interpretation=self._generate_query_interpretation(processed_query),
                search_strategy_used=self.config.ranking_algorithm
            )
            
            # Actualizar estadísticas
            self._update_search_stats(search_time, len(ranked_results), True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            
            # Actualizar estadísticas de error
            search_time = int((time.time() - start_time) * 1000)
            self._update_search_stats(search_time, 0, False)
            
            # Retornar resultado vacío
            return SemanticSearchResult(
                query=query,
                processed_query=ProcessedQuery(
                    original_query=query,
                    intent=QueryIntent.FIND_SIMILAR_CODE
                ),
                results=[],
                search_time_ms=search_time
            )
    
    async def search_by_code_example(
        self,
        example_code: str,
        language: ProgrammingLanguage,
        similarity_threshold: float = 0.7
    ) -> SemanticSearchResult:
        """
        Busca código similar usando un ejemplo.
        
        Args:
            example_code: Código de ejemplo
            language: Lenguaje del código
            similarity_threshold: Umbral de similitud
            
        Returns:
            Resultado de búsqueda
        """
        start_time = time.time()
        
        try:
            # Generar embedding del código ejemplo
            example_embedding = await self.embedding_engine.generate_embedding(example_code, language)
            
            # Buscar similares
            search_results = await self.vector_store.search_similar_embeddings(
                query_vector=example_embedding.embedding_vector,
                limit=self.config.max_results,
                language=None,  # Buscar en todos los lenguajes
                similarity_threshold=similarity_threshold
            )
            
            # Crear processed query para el ejemplo
            processed_query = ProcessedQuery(
                original_query=example_code,
                intent=QueryIntent.FIND_SIMILAR_CODE,
                concepts=[],
                query_embedding=example_embedding.embedding_vector
            )
            
            # Rankear resultados
            ranked_results = await self.result_ranker.rank_results(search_results, processed_query)
            
            search_time = int((time.time() - start_time) * 1000)
            
            return SemanticSearchResult(
                query=f"Code example: {example_code[:50]}...",
                processed_query=processed_query,
                results=ranked_results,
                total_candidates=len(search_results),
                total_results=len(ranked_results),
                search_time_ms=search_time,
                languages_searched=[language],
                search_strategy_used="code_similarity"
            )
            
        except Exception as e:
            logger.error(f"Error en búsqueda por ejemplo: {e}")
            return SemanticSearchResult(
                query=example_code,
                processed_query=ProcessedQuery(
                    original_query=example_code,
                    intent=QueryIntent.FIND_SIMILAR_CODE
                ),
                results=[]
            )
    
    def _apply_semantic_filters(
        self, 
        results: List[SemanticSearchResultItem], 
        query: ProcessedQuery
    ) -> List[SemanticSearchResultItem]:
        """Aplica filtros semánticos a los resultados."""
        filtered = []
        
        for result in results:
            # Filtro por confianza mínima
            if result.confidence < 0.5:
                continue
            
            # Filtro por similitud mínima
            if result.combined_similarity < self.config.similarity_threshold:
                continue
            
            # Filtro por intención
            if query.intent != QueryIntent.FIND_SIMILAR_CODE:
                if result.intent_match_score < 0.5:
                    continue
            
            # Filtro por conceptos (si hay)
            if query.concepts:
                code_lower = result.code_snippet.lower()
                concept_match = any(
                    concept.name.lower() in code_lower 
                    for concept in query.concepts
                )
                if not concept_match:
                    continue
            
            filtered.append(result)
        
        return filtered
    
    def _generate_query_interpretation(self, query: ProcessedQuery) -> str:
        """Genera interpretación humana del query."""
        interpretation = f"Searching for code that {query.intent.value.replace('_', ' ')}"
        
        if query.concepts:
            concept_names = [c.name for c in query.concepts]
            interpretation += f" related to {', '.join(concept_names)}"
        
        if query.constraints:
            constraint_desc = []
            for constraint in query.constraints:
                if constraint.startswith("language:"):
                    lang = constraint.split(":")[1]
                    constraint_desc.append(f"in {lang}")
                elif constraint.startswith("complexity:"):
                    level = constraint.split(":")[1]
                    constraint_desc.append(f"with {level} complexity")
            
            if constraint_desc:
                interpretation += f" ({', '.join(constraint_desc)})"
        
        return interpretation
    
    def _update_search_stats(self, search_time_ms: int, results_count: int, success: bool) -> None:
        """Actualiza estadísticas de búsqueda."""
        self.search_stats['total_searches'] += 1
        
        if success:
            self.search_stats['successful_searches'] += 1
        
        # Actualizar promedio de tiempo
        total = self.search_stats['total_searches']
        current_avg = self.search_stats['average_search_time_ms']
        self.search_stats['average_search_time_ms'] = (
            (current_avg * (total - 1) + search_time_ms) / total
        )
        
        # Actualizar promedio de resultados
        current_avg_results = self.search_stats['average_results_returned']
        self.search_stats['average_results_returned'] = (
            (current_avg_results * (total - 1) + results_count) / total
        )
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de búsqueda."""
        success_rate = (
            self.search_stats['successful_searches'] / 
            max(1, self.search_stats['total_searches'])
        )
        
        return {
            "total_searches": self.search_stats['total_searches'],
            "successful_searches": self.search_stats['successful_searches'],
            "success_rate": success_rate,
            "average_search_time_ms": self.search_stats['average_search_time_ms'],
            "average_results_returned": self.search_stats['average_results_returned'],
            "config": {
                "max_results": self.config.max_results,
                "similarity_threshold": self.config.similarity_threshold,
                "ranking_algorithm": self.config.ranking_algorithm,
                "semantic_filtering_enabled": self.config.enable_semantic_filtering
            }
        }
