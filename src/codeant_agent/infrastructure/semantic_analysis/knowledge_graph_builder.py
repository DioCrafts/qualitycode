"""
Constructor de grafo de conocimiento de código.

Este módulo implementa la construcción de grafos de conocimiento
que representan relaciones semánticas entre elementos de código.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter

from ...domain.entities.semantic_analysis import (
    KnowledgeGraphNode, KnowledgeGraphEdge, KnowledgeGraphResult,
    MultiLevelEmbeddings, SemanticRelationship, CodeConcept
)
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class GraphBuildingConfig:
    """Configuración para construcción de grafo."""
    similarity_threshold: float = 0.7
    max_edges_per_node: int = 10
    enable_concept_clustering: bool = True
    enable_centrality_analysis: bool = True
    prune_weak_edges: bool = True
    edge_weight_threshold: float = 0.3


@dataclass
class GraphStatistics:
    """Estadísticas del grafo de conocimiento."""
    total_nodes: int = 0
    total_edges: int = 0
    average_degree: float = 0.0
    clustering_coefficient: float = 0.0
    diameter: int = 0
    connected_components: int = 0
    density: float = 0.0
    
    def calculate_basic_stats(self, nodes: List[KnowledgeGraphNode], edges: List[KnowledgeGraphEdge]) -> None:
        """Calcula estadísticas básicas."""
        self.total_nodes = len(nodes)
        self.total_edges = len(edges)
        
        if self.total_nodes > 0:
            self.average_degree = (2 * self.total_edges) / self.total_nodes
            max_possible_edges = self.total_nodes * (self.total_nodes - 1) / 2
            self.density = self.total_edges / max_possible_edges if max_possible_edges > 0 else 0.0


class RelationshipAnalyzer:
    """Analizador de relaciones semánticas."""
    
    def __init__(self, config: GraphBuildingConfig):
        self.config = config
    
    async def analyze_semantic_relationships(
        self,
        embeddings: MultiLevelEmbeddings
    ) -> List[SemanticRelationship]:
        """
        Analiza relaciones semánticas en embeddings.
        
        Args:
            embeddings: Embeddings multi-nivel
            
        Returns:
            Lista de relaciones semánticas
        """
        relationships = []
        
        # Relaciones función-función
        func_relationships = await self._analyze_function_function_relationships(
            embeddings.function_embeddings
        )
        relationships.extend(func_relationships)
        
        # Relaciones clase-función
        if embeddings.class_embeddings:
            class_func_relationships = await self._analyze_class_function_relationships(
                embeddings.class_embeddings,
                embeddings.function_embeddings
            )
            relationships.extend(class_func_relationships)
        
        # Relaciones basadas en nombres (calls, etc.)
        name_relationships = await self._analyze_name_based_relationships(embeddings)
        relationships.extend(name_relationships)
        
        return relationships
    
    async def _analyze_function_function_relationships(
        self,
        function_embeddings: Dict[str, Any]
    ) -> List[SemanticRelationship]:
        """Analiza relaciones entre funciones."""
        relationships = []
        func_list = list(function_embeddings.values())
        
        for i, func1 in enumerate(func_list):
            similarities = []
            
            for func2 in func_list[i+1:]:
                similarity = self._calculate_cosine_similarity(
                    func1.embedding_vector,
                    func2.embedding_vector
                )
                
                if similarity >= self.config.similarity_threshold:
                    similarities.append((func2, similarity))
            
            # Ordenar por similitud y tomar las mejores
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            for func2, similarity in similarities[:self.config.max_edges_per_node]:
                relationship = SemanticRelationship(
                    source_id=func1.id,
                    target_id=func2.id,
                    relationship_type="semantic_similarity",
                    strength=similarity,
                    confidence=similarity,
                    evidence=[f"Embedding similarity: {similarity:.3f}"],
                    semantic_distance=1.0 - similarity
                )
                relationships.append(relationship)
        
        return relationships
    
    async def _analyze_class_function_relationships(
        self,
        class_embeddings: Dict[str, Any],
        function_embeddings: Dict[str, Any]
    ) -> List[SemanticRelationship]:
        """Analiza relaciones entre clases y funciones."""
        relationships = []
        
        for class_id, class_emb in class_embeddings.items():
            # Encontrar funciones semánticamente relacionadas con la clase
            related_functions = []
            
            for func_id, func_emb in function_embeddings.items():
                similarity = self._calculate_cosine_similarity(
                    class_emb.embedding_vector,
                    func_emb.embedding_vector
                )
                
                if similarity >= 0.6:  # Umbral más bajo para relaciones clase-función
                    related_functions.append((func_emb, similarity))
            
            # Crear relaciones para las funciones más relacionadas
            related_functions.sort(key=lambda x: x[1], reverse=True)
            
            for func_emb, similarity in related_functions[:5]:  # Top 5 funciones relacionadas
                relationship = SemanticRelationship(
                    source_id=class_id,
                    target_id=func_emb.id,
                    relationship_type="class_function_semantic",
                    strength=similarity,
                    confidence=similarity * 0.9,  # Ligeramente menor confianza
                    evidence=[f"Class-function semantic similarity: {similarity:.3f}"],
                    contextual_relevance=similarity
                )
                relationships.append(relationship)
        
        return relationships
    
    async def _analyze_name_based_relationships(
        self,
        embeddings: MultiLevelEmbeddings
    ) -> List[SemanticRelationship]:
        """Analiza relaciones basadas en nombres (llamadas, herencia, etc.)."""
        relationships = []
        
        # Buscar patrones de llamadas a función en el código
        if embeddings.file_embedding:
            file_code = embeddings.file_embedding.file_path.read_text() if embeddings.file_embedding.file_path.exists() else ""
            
            for func_id, func_emb in embeddings.function_embeddings.items():
                # Buscar llamadas a esta función en el código
                func_name = func_emb.function_name
                call_pattern = f"{func_name}("
                
                if call_pattern in file_code:
                    # Encontrar desde qué otras funciones se llama
                    for other_func_id, other_func in embeddings.function_embeddings.items():
                        if other_func_id != func_id and call_pattern in other_func.signature:
                            relationship = SemanticRelationship(
                                source_id=other_func_id,
                                target_id=func_id,
                                relationship_type="function_call",
                                strength=0.9,
                                confidence=0.95,
                                evidence=[f"Function call detected: {func_name}"],
                                contextual_relevance=0.9
                            )
                            relationships.append(relationship)
        
        return relationships
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud coseno entre vectores."""
        if not vec1 or not vec2:
            return 0.0
        
        min_dim = min(len(vec1), len(vec2))
        v1 = vec1[:min_dim]
        v2 = vec2[:min_dim]
        
        if sum(v1) == 0 or sum(v2) == 0:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class KnowledgeGraphBuilder:
    """Constructor principal del grafo de conocimiento."""
    
    def __init__(self, config: Optional[GraphBuildingConfig] = None):
        """
        Inicializa el constructor de grafo.
        
        Args:
            config: Configuración de construcción
        """
        self.config = config or GraphBuildingConfig()
        self.relationship_analyzer = RelationshipAnalyzer(self.config)
        
        # Almacenamiento del grafo
        self.nodes: Dict[str, KnowledgeGraphNode] = {}
        self.edges: List[KnowledgeGraphEdge] = []
        self.node_index: Dict[str, str] = {}  # name -> node_id
        
        # Estadísticas
        self.stats = GraphStatistics()
    
    async def build_knowledge_graph(
        self,
        multilevel_embeddings_list: List[MultiLevelEmbeddings]
    ) -> KnowledgeGraphResult:
        """
        Construye grafo de conocimiento completo.
        
        Args:
            multilevel_embeddings_list: Lista de embeddings multi-nivel
            
        Returns:
            Resultado del grafo construido
        """
        logger.info(f"Construyendo knowledge graph con {len(multilevel_embeddings_list)} archivos")
        
        # Limpiar grafo existente
        self.nodes.clear()
        self.edges.clear()
        self.node_index.clear()
        
        try:
            # Añadir nodos desde embeddings
            for embeddings in multilevel_embeddings_list:
                await self._add_nodes_from_embeddings(embeddings)
            
            # Analizar y añadir relaciones
            for embeddings in multilevel_embeddings_list:
                relationships = await self.relationship_analyzer.analyze_semantic_relationships(embeddings)
                await self._add_relationships_to_graph(relationships)
            
            # Construir clusters de conceptos si está habilitado
            if self.config.enable_concept_clustering:
                await self._build_concept_clusters()
            
            # Analizar centralidad si está habilitado
            insights = []
            if self.config.enable_centrality_analysis:
                insights = await self._analyze_centrality()
            
            # Podar aristas débiles si está habilitado
            if self.config.prune_weak_edges:
                await self._prune_weak_edges()
            
            # Calcular estadísticas
            self.stats.calculate_basic_stats(list(self.nodes.values()), self.edges)
            
            # Crear resultado
            result = KnowledgeGraphResult(
                query_type="build_complete_graph",
                nodes=list(self.nodes.values()),
                edges=self.edges,
                insights=insights,
                metrics={
                    "total_nodes": self.stats.total_nodes,
                    "total_edges": self.stats.total_edges,
                    "average_degree": self.stats.average_degree,
                    "density": self.stats.density,
                    "clustering_coefficient": self.stats.clustering_coefficient
                }
            )
            
            logger.info(f"Knowledge graph construido: {self.stats.total_nodes} nodos, {self.stats.total_edges} aristas")
            
            return result
            
        except Exception as e:
            logger.error(f"Error construyendo knowledge graph: {e}")
            return KnowledgeGraphResult(
                query_type="build_complete_graph",
                insights=[f"Error: {e}"],
                metrics={"error": str(e)}
            )
    
    async def _add_nodes_from_embeddings(self, embeddings: MultiLevelEmbeddings) -> None:
        """Añade nodos desde embeddings multi-nivel."""
        # Añadir nodos de funciones
        for func_id, func_emb in embeddings.function_embeddings.items():
            node = KnowledgeGraphNode(
                id=func_id,
                node_type="function",
                embedding=func_emb.embedding_vector,
                metadata={
                    "function_name": func_emb.function_name,
                    "language": embeddings.language.value,
                    "file_path": str(embeddings.file_path),
                    "complexity": func_emb.get_complexity_score(),
                    "parameters": func_emb.parameters
                },
                properties={
                    "lines_of_code": func_emb.complexity_metrics.get('lines_of_code', 0),
                    "cyclomatic_complexity": func_emb.complexity_metrics.get('cyclomatic_complexity', 0),
                    "semantic_purpose": func_emb.semantic_purpose
                }
            )
            
            self.nodes[func_id] = node
            self.node_index[func_emb.function_name] = func_id
        
        # Añadir nodos de clases
        for class_id, class_emb in embeddings.class_embeddings.items():
            node = KnowledgeGraphNode(
                id=class_id,
                node_type="class",
                embedding=class_emb.embedding_vector,
                metadata={
                    "class_name": class_emb.class_name,
                    "language": embeddings.language.value,
                    "file_path": str(embeddings.file_path),
                    "method_count": class_emb.get_method_count()
                },
                properties={
                    "responsibilities": class_emb.responsibilities,
                    "design_patterns": class_emb.design_patterns,
                    "inheritance_chain": class_emb.inheritance_chain
                }
            )
            
            self.nodes[class_id] = node
            self.node_index[class_emb.class_name] = class_id
        
        # Añadir nodo de archivo
        if embeddings.file_embedding:
            file_emb = embeddings.file_embedding
            file_node = KnowledgeGraphNode(
                id=f"file_{file_emb.id}",
                node_type="file",
                embedding=file_emb.embedding_vector,
                metadata={
                    "file_name": file_emb.file_path.name,
                    "file_path": str(file_emb.file_path),
                    "language": embeddings.language.value,
                    "total_elements": file_emb.get_total_elements()
                },
                properties={
                    "file_purpose": file_emb.file_purpose,
                    "architectural_role": file_emb.architectural_role,
                    "imports": file_emb.imports,
                    "exports": file_emb.exports
                }
            )
            
            self.nodes[f"file_{file_emb.id}"] = file_node
    
    async def _add_relationships_to_graph(self, relationships: List[SemanticRelationship]) -> None:
        """Añade relaciones al grafo."""
        for relationship in relationships:
            # Verificar que ambos nodos existen
            if relationship.source_id in self.nodes and relationship.target_id in self.nodes:
                edge = KnowledgeGraphEdge(
                    source_id=relationship.source_id,
                    target_id=relationship.target_id,
                    edge_type=relationship.relationship_type,
                    weight=relationship.strength,
                    confidence=relationship.confidence,
                    properties={
                        "evidence": relationship.evidence,
                        "semantic_distance": relationship.semantic_distance,
                        "contextual_relevance": relationship.contextual_relevance
                    }
                )
                
                self.edges.append(edge)
    
    async def _build_concept_clusters(self) -> None:
        """Construye clusters de conceptos similares."""
        # Agrupar nodos por similitud semántica
        clusters = defaultdict(list)
        processed_nodes = set()
        
        for node_id, node in self.nodes.items():
            if node_id in processed_nodes or not node.has_embedding():
                continue
            
            # Encontrar nodos similares
            cluster = [node_id]
            processed_nodes.add(node_id)
            
            for other_id, other_node in self.nodes.items():
                if (other_id != node_id and 
                    other_id not in processed_nodes and 
                    other_node.has_embedding()):
                    
                    similarity = self._calculate_cosine_similarity(
                        node.embedding, other_node.embedding
                    )
                    
                    if similarity >= 0.8:  # Umbral alto para clustering
                        cluster.append(other_id)
                        processed_nodes.add(other_id)
            
            if len(cluster) > 1:
                clusters[f"cluster_{len(clusters)}"] = cluster
        
        # Añadir nodos de cluster
        for cluster_id, node_ids in clusters.items():
            if len(node_ids) > 1:
                # Crear embedding promedio del cluster
                cluster_embeddings = [
                    self.nodes[node_id].embedding 
                    for node_id in node_ids 
                    if self.nodes[node_id].has_embedding()
                ]
                
                if cluster_embeddings:
                    cluster_embedding = self._average_embeddings(cluster_embeddings)
                    
                    cluster_node = KnowledgeGraphNode(
                        id=cluster_id,
                        node_type="concept_cluster",
                        embedding=cluster_embedding,
                        metadata={
                            "cluster_size": len(node_ids),
                            "member_nodes": node_ids
                        },
                        properties={
                            "cluster_type": "semantic_similarity",
                            "average_similarity": self._calculate_cluster_cohesion(cluster_embeddings)
                        }
                    )
                    
                    self.nodes[cluster_id] = cluster_node
                    
                    # Añadir aristas de cluster a miembros
                    for node_id in node_ids:
                        edge = KnowledgeGraphEdge(
                            source_id=cluster_id,
                            target_id=node_id,
                            edge_type="contains",
                            weight=0.9,
                            confidence=0.9
                        )
                        self.edges.append(edge)
    
    async def _analyze_centrality(self) -> List[str]:
        """Analiza centralidad de nodos."""
        insights = []
        
        # Calcular degree centrality simple
        node_degrees = defaultdict(int)
        
        for edge in self.edges:
            node_degrees[edge.source_id] += 1
            node_degrees[edge.target_id] += 1
        
        if node_degrees:
            # Encontrar nodos más centrales
            max_degree = max(node_degrees.values())
            central_nodes = [
                (node_id, degree) for node_id, degree in node_degrees.items()
                if degree >= max_degree * 0.8  # Top 20%
            ]
            
            central_nodes.sort(key=lambda x: x[1], reverse=True)
            
            for node_id, degree in central_nodes[:5]:  # Top 5
                node_name = self.nodes[node_id].metadata.get('function_name', 
                           self.nodes[node_id].metadata.get('class_name', node_id))
                
                insights.append(
                    f"Central node: {node_name} (degree: {degree}, "
                    f"type: {self.nodes[node_id].node_type})"
                )
        
        # Análisis de componentes aislados
        isolated_nodes = [
            node_id for node_id in self.nodes.keys()
            if node_degrees[node_id] == 0
        ]
        
        if isolated_nodes:
            insights.append(f"Found {len(isolated_nodes)} isolated components")
        
        return insights
    
    async def _prune_weak_edges(self) -> None:
        """Poda aristas débiles del grafo."""
        initial_edge_count = len(self.edges)
        
        # Filtrar aristas por peso y confianza
        self.edges = [
            edge for edge in self.edges
            if (edge.weight >= self.config.edge_weight_threshold and
                edge.confidence >= self.config.edge_weight_threshold)
        ]
        
        pruned_count = initial_edge_count - len(self.edges)
        
        if pruned_count > 0:
            logger.debug(f"Podadas {pruned_count} aristas débiles del grafo")
    
    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Calcula promedio de embeddings."""
        if not embeddings:
            return []
        
        min_dim = min(len(emb) for emb in embeddings)
        normalized_embeddings = [emb[:min_dim] for emb in embeddings]
        
        averaged = [0.0] * min_dim
        for embedding in normalized_embeddings:
            for i, val in enumerate(embedding):
                averaged[i] += val
        
        return [val / len(normalized_embeddings) for val in averaged]
    
    def _calculate_cluster_cohesion(self, cluster_embeddings: List[List[float]]) -> float:
        """Calcula cohesión del cluster."""
        if len(cluster_embeddings) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(cluster_embeddings)):
            for j in range(i + 1, len(cluster_embeddings)):
                sim = self._calculate_cosine_similarity(
                    cluster_embeddings[i], cluster_embeddings[j]
                )
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud coseno."""
        return self.relationship_analyzer._calculate_cosine_similarity(vec1, vec2)
    
    async def query_graph(
        self,
        query_type: str,
        parameters: Dict[str, Any]
    ) -> KnowledgeGraphResult:
        """
        Consulta el grafo de conocimiento.
        
        Args:
            query_type: Tipo de consulta
            parameters: Parámetros de la consulta
            
        Returns:
            Resultado de la consulta
        """
        start_time = time.time()
        
        try:
            if query_type == "find_similar_functions":
                return await self._find_similar_functions_query(parameters)
            
            elif query_type == "find_central_nodes":
                return await self._find_central_nodes_query(parameters)
            
            elif query_type == "analyze_dependencies":
                return await self._analyze_dependencies_query(parameters)
            
            elif query_type == "find_clusters":
                return await self._find_clusters_query(parameters)
            
            else:
                return KnowledgeGraphResult(
                    query_type=query_type,
                    insights=[f"Unknown query type: {query_type}"]
                )
                
        except Exception as e:
            logger.error(f"Error en query del grafo: {e}")
            return KnowledgeGraphResult(
                query_type=query_type,
                insights=[f"Query error: {e}"]
            )
        
        finally:
            query_time = int((time.time() - start_time) * 1000)
            logger.debug(f"Graph query '{query_type}' completada en {query_time}ms")
    
    async def _find_similar_functions_query(self, params: Dict[str, Any]) -> KnowledgeGraphResult:
        """Encuentra funciones similares."""
        function_id = params.get("function_id")
        threshold = params.get("similarity_threshold", 0.7)
        
        if function_id not in self.nodes:
            return KnowledgeGraphResult(
                query_type="find_similar_functions",
                insights=["Function not found in graph"]
            )
        
        target_node = self.nodes[function_id]
        similar_nodes = []
        related_edges = []
        
        # Buscar funciones similares
        for node_id, node in self.nodes.items():
            if (node_id != function_id and 
                node.node_type == "function" and 
                node.has_embedding()):
                
                similarity = self._calculate_cosine_similarity(
                    target_node.embedding, node.embedding
                )
                
                if similarity >= threshold:
                    similar_nodes.append(node)
                    
                    # Crear arista de similitud
                    edge = KnowledgeGraphEdge(
                        source_id=function_id,
                        target_id=node_id,
                        edge_type="similar_to",
                        weight=similarity,
                        confidence=similarity
                    )
                    related_edges.append(edge)
        
        insights = [
            f"Found {len(similar_nodes)} similar functions to {target_node.metadata.get('function_name', function_id)}"
        ]
        
        if similar_nodes:
            avg_similarity = sum(
                edge.weight for edge in related_edges
            ) / len(related_edges)
            insights.append(f"Average similarity: {avg_similarity:.3f}")
        
        return KnowledgeGraphResult(
            query_type="find_similar_functions",
            nodes=[target_node] + similar_nodes,
            edges=related_edges,
            insights=insights
        )
    
    async def _find_central_nodes_query(self, params: Dict[str, Any]) -> KnowledgeGraphResult:
        """Encuentra nodos centrales."""
        top_k = params.get("top_k", 5)
        
        # Calcular degree centrality
        node_degrees = defaultdict(int)
        for edge in self.edges:
            node_degrees[edge.source_id] += 1
            node_degrees[edge.target_id] += 1
        
        # Ordenar por degree
        sorted_nodes = sorted(
            node_degrees.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        central_nodes = [self.nodes[node_id] for node_id, _ in sorted_nodes if node_id in self.nodes]
        
        insights = [
            f"Top {len(central_nodes)} central nodes by degree centrality:",
        ]
        
        for node_id, degree in sorted_nodes:
            node_name = self.nodes[node_id].metadata.get('function_name',
                       self.nodes[node_id].metadata.get('class_name', node_id))
            insights.append(f"  {node_name} (degree: {degree})")
        
        return KnowledgeGraphResult(
            query_type="find_central_nodes",
            nodes=central_nodes,
            insights=insights,
            metrics={"centrality_scores": dict(sorted_nodes)}
        )
    
    async def _analyze_dependencies_query(self, params: Dict[str, Any]) -> KnowledgeGraphResult:
        """Analiza dependencias de un nodo."""
        node_id = params.get("node_id")
        depth = params.get("depth", 2)
        
        if node_id not in self.nodes:
            return KnowledgeGraphResult(
                query_type="analyze_dependencies",
                insights=["Node not found in graph"]
            )
        
        # Encontrar dependencias (BFS limitado)
        visited = set()
        current_level = [node_id]
        dependency_nodes = [self.nodes[node_id]]
        dependency_edges = []
        
        for level in range(depth):
            next_level = []
            
            for current_node_id in current_level:
                if current_node_id in visited:
                    continue
                
                visited.add(current_node_id)
                
                # Buscar aristas salientes
                for edge in self.edges:
                    if edge.source_id == current_node_id and edge.target_id not in visited:
                        next_level.append(edge.target_id)
                        dependency_edges.append(edge)
                        
                        if edge.target_id in self.nodes:
                            dependency_nodes.append(self.nodes[edge.target_id])
            
            current_level = next_level
            if not current_level:
                break
        
        insights = [
            f"Dependency analysis for {self.nodes[node_id].metadata.get('function_name', node_id)}:",
            f"Found {len(dependency_nodes) - 1} dependencies at depth {depth}",
            f"Total dependency edges: {len(dependency_edges)}"
        ]
        
        return KnowledgeGraphResult(
            query_type="analyze_dependencies",
            nodes=dependency_nodes,
            edges=dependency_edges,
            insights=insights,
            metrics={
                "dependency_depth": depth,
                "total_dependencies": len(dependency_nodes) - 1
            }
        )
    
    async def _find_clusters_query(self, params: Dict[str, Any]) -> KnowledgeGraphResult:
        """Encuentra clusters en el grafo."""
        min_cluster_size = params.get("min_cluster_size", 2)
        
        # Buscar nodos de cluster existentes
        cluster_nodes = [
            node for node in self.nodes.values()
            if node.node_type == "concept_cluster"
        ]
        
        # Filtrar por tamaño mínimo
        valid_clusters = [
            node for node in cluster_nodes
            if node.metadata.get("cluster_size", 0) >= min_cluster_size
        ]
        
        insights = [
            f"Found {len(valid_clusters)} concept clusters",
            f"Total clustered nodes: {sum(node.metadata.get('cluster_size', 0) for node in valid_clusters)}"
        ]
        
        if valid_clusters:
            avg_cluster_size = sum(
                node.metadata.get("cluster_size", 0) for node in valid_clusters
            ) / len(valid_clusters)
            insights.append(f"Average cluster size: {avg_cluster_size:.1f}")
        
        return KnowledgeGraphResult(
            query_type="find_clusters",
            nodes=valid_clusters,
            insights=insights,
            metrics={
                "cluster_count": len(valid_clusters),
                "average_cluster_size": avg_cluster_size if valid_clusters else 0
            }
        )
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del grafo."""
        # Análisis de tipos de nodos
        node_type_counts = Counter(node.node_type for node in self.nodes.values())
        
        # Análisis de tipos de aristas
        edge_type_counts = Counter(edge.edge_type for edge in self.edges)
        
        # Análisis de pesos de aristas
        edge_weights = [edge.weight for edge in self.edges]
        avg_edge_weight = sum(edge_weights) / len(edge_weights) if edge_weights else 0.0
        
        return {
            "graph_size": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "average_degree": self.stats.average_degree,
                "density": self.stats.density
            },
            "node_types": dict(node_type_counts),
            "edge_types": dict(edge_type_counts),
            "edge_statistics": {
                "average_weight": avg_edge_weight,
                "min_weight": min(edge_weights) if edge_weights else 0,
                "max_weight": max(edge_weights) if edge_weights else 0
            },
            "config": {
                "similarity_threshold": self.config.similarity_threshold,
                "max_edges_per_node": self.config.max_edges_per_node,
                "concept_clustering_enabled": self.config.enable_concept_clustering
            }
        }
