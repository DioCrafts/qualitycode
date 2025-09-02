"""
Implementación del motor de clustering de issues.

Este módulo implementa el agrupamiento inteligente de issues similares
usando análisis de características, similarity metrics y algoritmos de clustering.
"""

import logging
import asyncio
import math
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from collections import defaultdict
# Intentar importar numpy, con fallback
try:
    import numpy as np
except ImportError:
    # Mock de numpy para entornos sin numpy
    class MockNumPy:
        def array(self, data): return data
        def linalg(self): return self
        def norm(self, data): return sum(x**2 for x in data)**0.5 if data else 0.0
        def dot(self, a, b): return sum(x*y for x,y in zip(a,b)) if len(a) == len(b) else 0.0
    
    np = MockNumPy()

from ...domain.entities.issue_management import (
    CategorizedIssue, IssueCluster, ClusterId, ClusterType, ClusterCentroid,
    CommonCharacteristics, FixStrategy, PriorityDistribution, ClusteringConfig,
    IssueFeatureVector, IssueCategory, IssueSeverity, PriorityLevel
)

logger = logging.getLogger(__name__)


@dataclass
class SimilarityMatrix:
    """Matriz de similitud entre issues."""
    matrix: List[List[float]]
    issue_indices: Dict[str, int]  # issue_id -> index
    
    def get_similarity(self, issue1_id: str, issue2_id: str) -> float:
        """Obtiene similitud entre dos issues."""
        idx1 = self.issue_indices.get(issue1_id, -1)
        idx2 = self.issue_indices.get(issue2_id, -1)
        
        if idx1 == -1 or idx2 == -1:
            return 0.0
        
        return self.matrix[idx1][idx2]
    
    def get_most_similar(self, issue_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Obtiene issues más similares."""
        idx = self.issue_indices.get(issue_id, -1)
        if idx == -1:
            return []
        
        similarities = []
        for other_id, other_idx in self.issue_indices.items():
            if other_id != issue_id:
                similarity = self.matrix[idx][other_idx]
                similarities.append((other_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]


@dataclass
class ClusteringResult:
    """Resultado del clustering."""
    clusters: List[IssueCluster]
    singleton_issues: List[CategorizedIssue]
    clustering_quality: float
    algorithm_used: str
    clustering_time_ms: int
    silhouette_score: float = 0.0


class FeatureExtractor:
    """Extractor de características para clustering."""
    
    def __init__(self):
        self.category_encoding = {category: i for i, category in enumerate(IssueCategory)}
        self.severity_encoding = {severity: i for i, severity in enumerate(IssueSeverity)}
    
    async def extract_features(self, issue: CategorizedIssue) -> IssueFeatureVector:
        """
        Extrae vector de características de un issue.
        
        Args:
            issue: Issue a analizar
            
        Returns:
            IssueFeatureVector con características
        """
        feature_vector = IssueFeatureVector(issue_id=issue.id)
        
        # 1. Características de categoría (one-hot encoding)
        category_vector = [0.0] * len(IssueCategory)
        category_vector[self.category_encoding[issue.primary_category]] = 1.0
        
        # Categorías secundarias con peso menor
        for secondary_cat in issue.secondary_categories:
            if secondary_cat in self.category_encoding:
                category_vector[self.category_encoding[secondary_cat]] = 0.5
        
        feature_vector.category_vector = category_vector
        
        # 2. Score de severidad
        if issue.original_issue:
            feature_vector.severity_score = self.severity_encoding.get(issue.original_issue.severity, 2) / 4.0
        
        # 3. Características de complejidad
        complexity_features = []
        if issue.original_issue and issue.original_issue.complexity_metrics:
            complexity = issue.original_issue.complexity_metrics
            complexity_features = [
                complexity.cyclomatic_complexity / 50.0,  # Normalizar
                complexity.cognitive_complexity / 30.0,
                complexity.nesting_depth / 10.0,
                complexity.function_length / 100.0
            ]
        else:
            complexity_features = [0.0, 0.0, 0.0, 0.0]
        
        feature_vector.complexity_features = complexity_features
        
        # 4. Características de ubicación
        location_features = []
        if issue.original_issue:
            # Profundidad del path
            path_depth = len(issue.original_issue.file_path.parts) / 10.0
            
            # Tipo de archivo (extensión)
            extension_encoding = {
                '.py': 0.1, '.js': 0.2, '.ts': 0.3, '.rs': 0.4,
                '.java': 0.5, '.cpp': 0.6, '.c': 0.7, '.go': 0.8
            }
            extension_score = extension_encoding.get(issue.original_issue.file_path.suffix, 0.0)
            
            # Línea en archivo (normalizada)
            line_position = issue.original_issue.location.start.line / 1000.0
            
            location_features = [path_depth, extension_score, line_position]
        else:
            location_features = [0.0, 0.0, 0.0]
        
        feature_vector.location_features = location_features
        
        # 5. Características de contexto
        context = issue.context_info
        context_features = [
            context.file_change_frequency,
            context.code_age_days / 365.0,  # Normalizar a años
            context.test_coverage_percentage / 100.0,
            context.dependency_count / 20.0,
            1.0 if context.module_criticality == "critical" else 0.5 if context.module_criticality == "important" else 0.0
        ]
        
        feature_vector.context_features = context_features
        
        # 6. Características textuales (TF-IDF simplificado)
        textual_features = await self._extract_textual_features(issue)
        feature_vector.textual_features = textual_features
        
        return feature_vector
    
    async def _extract_textual_features(self, issue: CategorizedIssue) -> List[float]:
        """Extrae características textuales usando TF-IDF simplificado."""
        if not issue.original_issue:
            return [0.0] * 10  # Vector vacío
        
        # Vocabulario común de términos técnicos
        vocabulary = [
            'error', 'performance', 'security', 'memory', 'complex', 
            'duplicate', 'test', 'null', 'exception', 'optimize'
        ]
        
        text = f"{issue.original_issue.rule_id} {issue.original_issue.message}".lower()
        
        # Calcular TF (term frequency) para cada término
        tf_scores = []
        for term in vocabulary:
            count = text.count(term)
            tf = count / max(1, len(text.split()))  # Normalizar por longitud
            tf_scores.append(tf)
        
        return tf_scores
    
    def extract_batch_features(self, issues: List[CategorizedIssue]) -> List[IssueFeatureVector]:
        """Extrae características de lote de issues."""
        return [asyncio.run(self.extract_features(issue)) for issue in issues]


class SimilarityCalculator:
    """Calculadora de similitud entre issues."""
    
    def calculate_similarity(self, feature1: IssueFeatureVector, feature2: IssueFeatureVector,
                           weights: Dict[str, float]) -> float:
        """
        Calcula similitud entre dos vectors de características.
        
        Args:
            feature1: Primer vector
            feature2: Segundo vector
            weights: Pesos por tipo de característica
            
        Returns:
            Score de similitud (0-1)
        """
        similarities = {}
        
        # Similitud de categorías (Jaccard)
        similarities["category"] = self._jaccard_similarity(feature1.category_vector, feature2.category_vector)
        
        # Similitud de severidad
        similarities["severity"] = 1.0 - abs(feature1.severity_score - feature2.severity_score)
        
        # Similitud de complejidad (coseno)
        similarities["complexity"] = self._cosine_similarity(feature1.complexity_features, feature2.complexity_features)
        
        # Similitud de ubicación
        similarities["location"] = self._euclidean_similarity(feature1.location_features, feature2.location_features)
        
        # Similitud de contexto
        similarities["context"] = self._cosine_similarity(feature1.context_features, feature2.context_features)
        
        # Similitud textual
        similarities["textual"] = self._cosine_similarity(feature1.textual_features, feature2.textual_features)
        
        # Calcular similitud ponderada
        weighted_similarity = sum(
            similarities[feature_type] * weights.get(feature_type, 0.1)
            for feature_type in similarities
        )
        
        return min(1.0, max(0.0, weighted_similarity))
    
    def _jaccard_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud de Jaccard para vectores binarios."""
        if len(vec1) != len(vec2):
            return 0.0
        
        intersection = sum(1 for a, b in zip(vec1, vec2) if a > 0 and b > 0)
        union = sum(1 for a, b in zip(vec1, vec2) if a > 0 or b > 0)
        
        return intersection / union if union > 0 else 1.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud coseno."""
        if len(vec1) != len(vec2):
            return 0.0
        
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = np.dot(v1, v2) / (norm1 * norm2)
            return max(0.0, cosine_sim)
        except Exception:
            return 0.0
    
    def _euclidean_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud basada en distancia euclidiana."""
        if len(vec1) != len(vec2):
            return 0.0
        
        try:
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
            # Convertir distancia a similitud (0-1)
            max_distance = math.sqrt(len(vec1))  # Máxima distancia posible
            similarity = 1.0 - (distance / max_distance)
            return max(0.0, similarity)
        except Exception:
            return 0.0


class HierarchicalClustering:
    """Algoritmo de clustering jerárquico."""
    
    def __init__(self, similarity_calculator: SimilarityCalculator):
        self.similarity_calculator = similarity_calculator
    
    async def cluster(self, issues: List[CategorizedIssue], feature_vectors: List[IssueFeatureVector],
                     config: ClusteringConfig) -> List[IssueCluster]:
        """
        Ejecuta clustering jerárquico.
        
        Args:
            issues: Lista de issues a agrupar
            feature_vectors: Vectores de características correspondientes
            config: Configuración de clustering
            
        Returns:
            Lista de clusters encontrados
        """
        if len(issues) < 2:
            return []
        
        # Calcular matriz de similitud
        similarity_matrix = await self._calculate_similarity_matrix(feature_vectors, config)
        
        # Inicializar clusters (cada issue en su propio cluster)
        clusters = [[i] for i in range(len(issues))]
        cluster_id_counter = 0
        result_clusters = []
        
        # Clustering aglomerativo
        while len(clusters) > 1:
            # Encontrar par de clusters más similares
            best_similarity = -1.0
            best_pair = None
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    similarity = self._calculate_cluster_similarity(
                        clusters[i], clusters[j], similarity_matrix
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_pair = (i, j)
            
            # Si la mejor similitud está por debajo del threshold, parar
            if best_similarity < config.similarity_threshold or best_pair is None:
                break
            
            # Fusionar clusters
            i, j = best_pair
            merged_cluster = clusters[i] + clusters[j]
            
            # Crear IssueCluster si cumple criterios de tamaño
            if config.min_cluster_size <= len(merged_cluster) <= config.max_cluster_size:
                cluster_issues = [issues[idx] for idx in merged_cluster]
                issue_cluster = await self._create_issue_cluster(cluster_issues, cluster_id_counter, best_similarity)
                result_clusters.append(issue_cluster)
                cluster_id_counter += 1
            
            # Actualizar lista de clusters
            new_clusters = []
            for k, cluster in enumerate(clusters):
                if k != i and k != j:
                    new_clusters.append(cluster)
            new_clusters.append(merged_cluster)
            clusters = new_clusters
        
        return result_clusters
    
    async def _calculate_similarity_matrix(self, feature_vectors: List[IssueFeatureVector],
                                         config: ClusteringConfig) -> List[List[float]]:
        """Calcula matriz de similitud completa."""
        n = len(feature_vectors)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.similarity_calculator.calculate_similarity(
                    feature_vectors[i], feature_vectors[j], config.feature_weights
                )
                matrix[i][j] = similarity
                matrix[j][i] = similarity  # Matriz simétrica
            matrix[i][i] = 1.0  # Similitud consigo mismo
        
        return matrix
    
    def _calculate_cluster_similarity(self, cluster1: List[int], cluster2: List[int],
                                    similarity_matrix: List[List[float]]) -> float:
        """Calcula similitud entre dos clusters (linkage promedio)."""
        if not cluster1 or not cluster2:
            return 0.0
        
        total_similarity = 0.0
        pair_count = 0
        
        for i in cluster1:
            for j in cluster2:
                total_similarity += similarity_matrix[i][j]
                pair_count += 1
        
        return total_similarity / pair_count if pair_count > 0 else 0.0
    
    async def _create_issue_cluster(self, cluster_issues: List[CategorizedIssue], 
                                  cluster_id: int, cohesion_score: float) -> IssueCluster:
        """Crea IssueCluster desde lista de issues."""
        # Calcular centroide
        centroid = await self._calculate_centroid(cluster_issues)
        
        # Analizar características comunes
        common_characteristics = await self._analyze_common_characteristics(cluster_issues)
        
        # Sugerir estrategia de fix
        fix_strategy = await self._suggest_fix_strategy(cluster_issues, common_characteristics)
        
        # Calcular distribución de prioridades
        priority_distribution = PriorityDistribution.from_issues(cluster_issues)
        
        # Estimar tiempo de fix en batch
        batch_fix_time = await self._estimate_batch_fix_time(cluster_issues)
        
        # Calcular ROI del cluster
        roi_score = await self._calculate_cluster_roi(cluster_issues, batch_fix_time)
        
        return IssueCluster(
            id=ClusterId(f"cluster_{cluster_id}"),
            cluster_type=ClusterType.SIMILAR,
            issues=cluster_issues,
            centroid=centroid,
            cohesion_score=cohesion_score,
            common_characteristics=common_characteristics,
            suggested_fix_strategy=fix_strategy,
            priority_distribution=priority_distribution,
            estimated_batch_fix_time=batch_fix_time,
            roi_score=roi_score
        )
    
    async def _calculate_centroid(self, cluster_issues: List[CategorizedIssue]) -> ClusterCentroid:
        """Calcula centroide del cluster."""
        centroid = ClusterCentroid()
        
        if not cluster_issues:
            return centroid
        
        # Pesos de categorías
        category_weights = defaultdict(float)
        for issue in cluster_issues:
            category_weights[issue.primary_category] += 1.0
            for secondary_cat in issue.secondary_categories:
                category_weights[secondary_cat] += 0.5
        
        # Normalizar pesos
        total_weight = sum(category_weights.values())
        if total_weight > 0:
            centroid.category_weights = {cat: weight / total_weight for cat, weight in category_weights.items()}
        
        # Severidad promedio
        severities = []
        for issue in cluster_issues:
            if issue.original_issue:
                severity_values = {
                    IssueSeverity.INFO: 1, IssueSeverity.LOW: 2, IssueSeverity.MEDIUM: 3,
                    IssueSeverity.HIGH: 4, IssueSeverity.CRITICAL: 5
                }
                severities.append(severity_values.get(issue.original_issue.severity, 3))
        
        centroid.average_severity = sum(severities) / len(severities) if severities else 3.0
        
        # Complejidad promedio
        complexities = []
        for issue in cluster_issues:
            if issue.original_issue and issue.original_issue.complexity_metrics:
                complexities.append(issue.original_issue.complexity_metrics.cyclomatic_complexity)
        
        centroid.average_complexity = sum(complexities) / len(complexities) if complexities else 1.0
        
        # Patrón de ubicación común
        file_paths = [str(issue.original_issue.file_path) for issue in cluster_issues if issue.original_issue]
        common_path_parts = self._find_common_path_pattern(file_paths)
        centroid.common_location_pattern = "/".join(common_path_parts)
        
        # Mensaje representativo
        messages = [issue.original_issue.message for issue in cluster_issues if issue.original_issue]
        centroid.representative_message = self._find_representative_message(messages)
        
        return centroid
    
    def _find_common_path_pattern(self, file_paths: List[str]) -> List[str]:
        """Encuentra patrón común en paths de archivos."""
        if not file_paths:
            return []
        
        # Dividir paths en partes
        path_parts = [path.split('/') for path in file_paths]
        
        # Encontrar prefijo común
        common_parts = []
        if path_parts:
            min_length = min(len(parts) for parts in path_parts)
            
            for i in range(min_length):
                current_part = path_parts[0][i]
                if all(parts[i] == current_part for parts in path_parts):
                    common_parts.append(current_part)
                else:
                    break
        
        return common_parts
    
    def _find_representative_message(self, messages: List[str]) -> str:
        """Encuentra mensaje representativo del cluster."""
        if not messages:
            return ""
        
        # Usar el mensaje más común o el más largo
        message_counts = defaultdict(int)
        for message in messages:
            message_counts[message] += 1
        
        # Si hay mensaje repetido, usar ese
        most_common = max(message_counts.items(), key=lambda x: x[1])
        if most_common[1] > 1:
            return most_common[0]
        
        # Sino, usar el más largo (más descriptivo)
        return max(messages, key=len)


class ClusterAnalyzer:
    """Analizador de características de clusters."""
    
    async def analyze_common_characteristics(self, cluster_issues: List[CategorizedIssue]) -> CommonCharacteristics:
        """Analiza características comunes del cluster."""
        characteristics = CommonCharacteristics()
        
        if not cluster_issues:
            return characteristics
        
        # Categoría dominante
        category_counts = defaultdict(int)
        for issue in cluster_issues:
            category_counts[issue.primary_category] += 1
        
        if category_counts:
            characteristics.dominant_category = max(category_counts.keys(), key=lambda k: category_counts[k])
        
        # Tags comunes
        tag_counts = defaultdict(int)
        for issue in cluster_issues:
            for tag in issue.tags:
                tag_counts[tag] += 1
        
        # Tags que aparecen en al menos 50% de issues
        threshold = len(cluster_issues) // 2
        characteristics.common_tags = [tag for tag, count in tag_counts.items() if count > threshold]
        
        # Patrones de archivos
        file_patterns = set()
        for issue in cluster_issues:
            if issue.original_issue:
                # Extraer patrones: extensión, directorio, prefijos
                file_path = issue.original_issue.file_path
                file_patterns.add(f"ext:{file_path.suffix}")
                if len(file_path.parts) > 1:
                    file_patterns.add(f"dir:{file_path.parts[-2]}")
        
        characteristics.file_patterns = list(file_patterns)
        
        # Patrones de reglas
        rule_patterns = set()
        for issue in cluster_issues:
            if issue.original_issue:
                rule_id = issue.original_issue.rule_id
                # Extraer prefijo de regla (antes del primer -)
                if '-' in rule_id:
                    prefix = rule_id.split('-')[0]
                    rule_patterns.add(f"rule_prefix:{prefix}")
                rule_patterns.add(f"rule_type:{rule_id}")
        
        characteristics.rule_patterns = list(rule_patterns)
        
        # Distribución de severidad
        severity_counts = defaultdict(int)
        for issue in cluster_issues:
            if issue.original_issue:
                severity_counts[issue.original_issue.severity] += 1
        
        characteristics.severity_distribution = dict(severity_counts)
        
        # Distribución de lenguajes
        language_counts = defaultdict(int)
        for issue in cluster_issues:
            if issue.original_issue:
                language_counts[issue.original_issue.language] += 1
        
        characteristics.language_distribution = dict(language_counts)
        
        # Causas raíz comunes (simplificado)
        characteristics.common_root_causes = await self._identify_root_causes(cluster_issues)
        
        return characteristics
    
    async def _identify_root_causes(self, cluster_issues: List[CategorizedIssue]) -> List[str]:
        """Identifica causas raíz comunes."""
        root_causes = []
        
        # Análisis de patrones en mensajes
        all_messages = [issue.original_issue.message.lower() for issue in cluster_issues if issue.original_issue]
        
        # Buscar palabras clave de causas comunes
        common_causes = {
            'lack of validation': ['validation', 'check', 'validate'],
            'missing error handling': ['error', 'exception', 'handle'],
            'high complexity': ['complex', 'nested', 'condition'],
            'poor design': ['design', 'pattern', 'structure'],
            'inadequate testing': ['test', 'coverage', 'untested'],
            'performance bottleneck': ['slow', 'performance', 'bottleneck'],
            'security vulnerability': ['security', 'vulnerable', 'exploit']
        }
        
        text_blob = " ".join(all_messages)
        
        for cause, keywords in common_causes.items():
            if sum(text_blob.count(keyword) for keyword in keywords) >= len(cluster_issues) // 2:
                root_causes.append(cause)
        
        return root_causes
    
    async def suggest_fix_strategy(self, cluster_issues: List[CategorizedIssue], 
                                 characteristics: CommonCharacteristics) -> FixStrategy:
        """Sugiere estrategia de fix para el cluster."""
        # Determinar si es aplicable fix en batch
        batch_applicable = len(cluster_issues) >= 3 and characteristics.dominant_category is not None
        
        # Determinar tipo de fix basado en categoría dominante
        if characteristics.dominant_category == IssueCategory.SECURITY:
            strategy_type = FixType.CODE_CHANGE
            approach = "Address security vulnerabilities systematically"
            risk_level = "high"
            
        elif characteristics.dominant_category == IssueCategory.PERFORMANCE:
            strategy_type = FixType.REFACTORING
            approach = "Optimize performance bottlenecks in batch"
            risk_level = "medium"
            
        elif characteristics.dominant_category == IssueCategory.MAINTAINABILITY:
            strategy_type = FixType.REFACTORING
            approach = "Apply consistent refactoring patterns"
            risk_level = "low"
            
        elif characteristics.dominant_category == IssueCategory.DOCUMENTATION:
            strategy_type = FixType.DOCUMENTATION
            approach = "Add documentation systematically"
            risk_level = "low"
            batch_applicable = True  # Documentación siempre es batch-friendly
            
        else:
            strategy_type = FixType.CODE_CHANGE
            approach = "Address issues individually"
            risk_level = "medium"
            batch_applicable = False
        
        # Calcular potencial de automatización
        automation_potential = 0.0
        if characteristics.dominant_category in [IssueCategory.CODE_STYLE, IssueCategory.DOCUMENTATION]:
            automation_potential = 0.8
        elif characteristics.dominant_category == IssueCategory.MAINTAINABILITY:
            automation_potential = 0.4
        
        # Estimar multiplicador de esfuerzo para batch
        effort_multiplier = 0.7 if batch_applicable and len(cluster_issues) >= 5 else 1.0
        
        return FixStrategy(
            strategy_type=strategy_type,
            description=f"Batch fix strategy for {len(cluster_issues)} {characteristics.dominant_category.value} issues",
            batch_applicable=batch_applicable,
            estimated_effort_multiplier=effort_multiplier,
            risk_level=risk_level,
            prerequisites=self._identify_fix_prerequisites(characteristics),
            automation_potential=automation_potential,
            recommended_approach=approach
        )
    
    def _identify_fix_prerequisites(self, characteristics: CommonCharacteristics) -> List[str]:
        """Identifica prerequisitos para fix."""
        prerequisites = []
        
        if characteristics.dominant_category == IssueCategory.SECURITY:
            prerequisites.extend([
                "Security review required",
                "Penetration testing after fix",
                "Security team approval"
            ])
        
        elif characteristics.dominant_category == IssueCategory.PERFORMANCE:
            prerequisites.extend([
                "Performance baseline measurement",
                "Load testing environment setup",
                "Monitoring setup"
            ])
        
        elif characteristics.dominant_category == IssueCategory.ARCHITECTURE:
            prerequisites.extend([
                "Architecture review",
                "Team design discussion",
                "Migration plan"
            ])
        
        # Prerequisites generales
        prerequisites.extend([
            "Backup current code",
            "Create test branch",
            "Run existing test suite"
        ])
        
        return prerequisites
    
    async def _estimate_batch_fix_time(self, cluster_issues: List[CategorizedIssue]) -> float:
        """Estima tiempo de fix en batch."""
        if not cluster_issues:
            return 0.0
        
        # Tiempo base: suma de tiempos individuales
        individual_times = [issue.metadata.estimated_fix_time_hours or 1.0 for issue in cluster_issues]
        total_individual_time = sum(individual_times)
        
        # Eficiencia de batch (descuento por similaridad)
        batch_efficiency = 0.3  # 30% de descuento base
        
        # Mayor eficiencia para clusters más cohesivos
        if len(cluster_issues) >= 5:
            batch_efficiency += 0.2
        
        # Mayor eficiencia para mismo tipo de fix
        same_category_issues = sum(1 for issue in cluster_issues 
                                 if issue.primary_category == cluster_issues[0].primary_category)
        if same_category_issues == len(cluster_issues):
            batch_efficiency += 0.15
        
        batch_time = total_individual_time * (1.0 - batch_efficiency)
        
        # Añadir overhead de coordinación
        coordination_overhead = len(cluster_issues) * 0.1  # 6 minutos por issue adicional
        
        return batch_time + coordination_overhead
    
    async def _calculate_cluster_roi(self, cluster_issues: List[CategorizedIssue], batch_fix_time: float) -> float:
        """Calcula ROI del cluster."""
        if batch_fix_time == 0:
            return 0.0
        
        # Beneficio: suma de prioridades de issues
        total_benefit = sum(issue.get_priority_score() for issue in cluster_issues)
        
        # ROI = beneficio / esfuerzo
        roi = total_benefit / batch_fix_time
        
        # Boost para clusters altamente cohesivos
        if len(set(issue.primary_category for issue in cluster_issues)) == 1:
            roi *= 1.2  # 20% boost para misma categoría
        
        return roi


class ClusteringEngine:
    """Motor principal de clustering."""
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Inicializa el motor de clustering.
        
        Args:
            config: Configuración de clustering
        """
        self.config = config or ClusteringConfig()
        self.feature_extractor = FeatureExtractor()
        self.similarity_calculator = SimilarityCalculator()
        self.hierarchical_clusterer = HierarchicalClustering(self.similarity_calculator)
        self.cluster_analyzer = ClusterAnalyzer()
    
    async def cluster_issues(self, issues: List[CategorizedIssue]) -> ClusteringResult:
        """
        Agrupa issues en clusters similares.
        
        Args:
            issues: Lista de issues categorizados
            
        Returns:
            ClusteringResult con clusters encontrados
        """
        start_time = time.time()
        
        logger.info(f"Iniciando clustering de {len(issues)} issues")
        
        if len(issues) < 2:
            logger.info("Insuficientes issues para clustering")
            return ClusteringResult(
                clusters=[],
                singleton_issues=issues,
                clustering_quality=1.0,
                algorithm_used="none",
                clustering_time_ms=0
            )
        
        try:
            # Extraer características
            feature_vectors = []
            for issue in issues:
                features = await self.feature_extractor.extract_features(issue)
                feature_vectors.append(features)
            
            # Aplicar algoritmo de clustering
            clusters = []
            if self.config.clustering_method == "hierarchical":
                clusters = await self.hierarchical_clusterer.cluster(issues, feature_vectors, self.config)
            
            # Identificar issues singleton (no agrupados)
            clustered_issue_ids = set()
            for cluster in clusters:
                for issue in cluster.issues:
                    clustered_issue_ids.add(issue.id.value)
            
            singleton_issues = [issue for issue in issues if issue.id.value not in clustered_issue_ids]
            
            # Calcular calidad del clustering
            clustering_quality = await self._calculate_clustering_quality(clusters, singleton_issues, feature_vectors)
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Clustering completado: {len(clusters)} clusters, {len(singleton_issues)} singletons, "
                f"calidad={clustering_quality:.3f} en {total_time}ms"
            )
            
            return ClusteringResult(
                clusters=clusters,
                singleton_issues=singleton_issues,
                clustering_quality=clustering_quality,
                algorithm_used=self.config.clustering_method,
                clustering_time_ms=total_time,
                silhouette_score=self._calculate_silhouette_score(clusters, feature_vectors)
            )
            
        except Exception as e:
            logger.error(f"Error en clustering: {e}")
            # Fallback: todos los issues como singletons
            return ClusteringResult(
                clusters=[],
                singleton_issues=issues,
                clustering_quality=0.0,
                algorithm_used="fallback",
                clustering_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _calculate_clustering_quality(self, clusters: List[IssueCluster], 
                                          singleton_issues: List[CategorizedIssue],
                                          feature_vectors: List[IssueFeatureVector]) -> float:
        """Calcula calidad del clustering."""
        if not clusters:
            return 0.0
        
        # Métricas de calidad
        total_issues = sum(len(cluster.issues) for cluster in clusters) + len(singleton_issues)
        clustered_percentage = sum(len(cluster.issues) for cluster in clusters) / total_issues
        
        # Cohesión promedio de clusters
        avg_cohesion = sum(cluster.cohesion_score for cluster in clusters) / len(clusters)
        
        # Penalizar demasiados singletons
        singleton_penalty = len(singleton_issues) / total_issues
        
        quality_score = (clustered_percentage * 0.5 + avg_cohesion * 0.5) * (1.0 - singleton_penalty * 0.3)
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_silhouette_score(self, clusters: List[IssueCluster], 
                                  feature_vectors: List[IssueFeatureVector]) -> float:
        """Calcula silhouette score para evaluar clustering."""
        # Implementación simplificada del silhouette score
        if len(clusters) < 2:
            return 0.0
        
        # En implementación completa, calcularía silhouette score real
        # Por simplicidad, usar cohesión promedio como proxy
        if clusters:
            return sum(cluster.cohesion_score for cluster in clusters) / len(clusters)
        
        return 0.0
    
    def get_clustering_summary(self, result: ClusteringResult) -> Dict[str, Any]:
        """
        Obtiene resumen del clustering.
        
        Returns:
            Diccionario con estadísticas de clustering
        """
        summary = {
            "total_clusters": len(result.clusters),
            "singleton_issues": len(result.singleton_issues),
            "clustering_quality": result.clustering_quality,
            "average_cluster_size": 0.0,
            "largest_cluster_size": 0,
            "cluster_types_distribution": {},
            "dominant_categories": {}
        }
        
        if result.clusters:
            cluster_sizes = [cluster.get_cluster_size() for cluster in result.clusters]
            summary["average_cluster_size"] = sum(cluster_sizes) / len(cluster_sizes)
            summary["largest_cluster_size"] = max(cluster_sizes)
            
            # Distribución de tipos de cluster
            type_counts = defaultdict(int)
            for cluster in result.clusters:
                type_counts[cluster.cluster_type] += 1
            summary["cluster_types_distribution"] = {ct.value: count for ct, count in type_counts.items()}
            
            # Categorías dominantes
            category_counts = defaultdict(int)
            for cluster in result.clusters:
                dominant_cat = cluster.get_dominant_category()
                category_counts[dominant_cat] += 1
            summary["dominant_categories"] = {cat.value: count for cat, count in category_counts.items()}
        
        return summary
