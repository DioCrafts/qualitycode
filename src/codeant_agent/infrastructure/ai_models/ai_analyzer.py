"""
Analizador de código con IA que integra todos los componentes.

Este módulo implementa el analizador principal que coordina
todos los componentes de IA para análisis avanzado de código.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ...domain.entities.ai_models import (
    AIAnalysisResult, AIAnalysisConfig, CodeEmbedding, AIPattern,
    CodeAnomaly, SemanticInsight, QualityPrediction, CrossLanguageSimilarity,
    AnalysisType, ModelType
)
from ...domain.value_objects.programming_language import ProgrammingLanguage
from .model_manager import AIModelManager
from .embedding_engine import CodeEmbeddingEngine
from .vector_store import VectorStore
from .inference_engine import InferenceEngine
from .code_preprocessor import CodePreprocessor

logger = logging.getLogger(__name__)


@dataclass
class AnalysisSession:
    """Sesión de análisis con IA."""
    session_id: str
    started_at: datetime = field(default_factory=datetime.now)
    analyses_completed: int = 0
    total_code_analyzed: int = 0
    active_analyses: Dict[str, 'AnalysisJob'] = field(default_factory=dict)
    
    def add_analysis(self, job: 'AnalysisJob') -> None:
        """Añade análisis a la sesión."""
        self.active_analyses[job.job_id] = job
    
    def complete_analysis(self, job_id: str) -> None:
        """Completa un análisis."""
        if job_id in self.active_analyses:
            del self.active_analyses[job_id]
            self.analyses_completed += 1


@dataclass
class AnalysisJob:
    """Job de análisis de código."""
    job_id: str
    code: str
    language: ProgrammingLanguage
    analysis_types: List[AnalysisType]
    priority: int = 5
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[AIAnalysisResult] = None
    created_at: datetime = field(default_factory=datetime.now)


class PatternDetector:
    """Detector de patrones de código usando IA."""
    
    def __init__(self, model_manager: AIModelManager, config: AIAnalysisConfig):
        self.model_manager = model_manager
        self.config = config
        self.pattern_library = self._initialize_pattern_library()
    
    def _initialize_pattern_library(self) -> Dict[str, Dict[str, Any]]:
        """Inicializa biblioteca de patrones conocidos."""
        return {
            "singleton": {
                "name": "Singleton Pattern",
                "type": "design_pattern",
                "keywords": ["getInstance", "instance", "private constructor", "static"],
                "confidence_threshold": 0.7,
                "description": "Ensures a class has only one instance"
            },
            "factory": {
                "name": "Factory Pattern", 
                "type": "design_pattern",
                "keywords": ["create", "factory", "build", "make"],
                "confidence_threshold": 0.6,
                "description": "Creates objects without specifying exact classes"
            },
            "observer": {
                "name": "Observer Pattern",
                "type": "design_pattern", 
                "keywords": ["notify", "observer", "subscribe", "listener", "event"],
                "confidence_threshold": 0.65,
                "description": "Defines one-to-many dependency between objects"
            },
            "god_class": {
                "name": "God Class",
                "type": "anti_pattern",
                "keywords": ["too many methods", "large class", "multiple responsibilities"],
                "confidence_threshold": 0.75,
                "description": "Class that does too much and has too many responsibilities"
            },
            "spaghetti_code": {
                "name": "Spaghetti Code",
                "type": "anti_pattern",
                "keywords": ["goto", "complex control flow", "nested conditions"],
                "confidence_threshold": 0.8,
                "description": "Code with complex and tangled control structure"
            }
        }
    
    async def detect_patterns(
        self, 
        code: str, 
        language: ProgrammingLanguage,
        embedding: Optional[CodeEmbedding] = None
    ) -> List[AIPattern]:
        """
        Detecta patrones en el código.
        
        Args:
            code: Código a analizar
            language: Lenguaje del código
            embedding: Embedding del código (opcional)
            
        Returns:
            Lista de patrones detectados
        """
        detected_patterns = []
        
        # Detección basada en keywords
        keyword_patterns = await self._detect_keyword_patterns(code, language)
        detected_patterns.extend(keyword_patterns)
        
        # Detección basada en estructura
        structural_patterns = await self._detect_structural_patterns(code, language)
        detected_patterns.extend(structural_patterns)
        
        # Detección basada en embedding (si está disponible)
        if embedding:
            embedding_patterns = await self._detect_embedding_patterns(embedding, language)
            detected_patterns.extend(embedding_patterns)
        
        # Filtrar por umbral de confianza
        filtered_patterns = [
            pattern for pattern in detected_patterns
            if pattern.confidence >= self.config.confidence_threshold
        ]
        
        return filtered_patterns
    
    async def _detect_keyword_patterns(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> List[AIPattern]:
        """Detecta patrones basándose en palabras clave."""
        patterns = []
        code_lower = code.lower()
        
        for pattern_id, pattern_info in self.pattern_library.items():
            keyword_matches = 0
            total_keywords = len(pattern_info["keywords"])
            
            for keyword in pattern_info["keywords"]:
                if keyword.lower() in code_lower:
                    keyword_matches += 1
            
            if keyword_matches > 0:
                confidence = (keyword_matches / total_keywords) * 0.8  # Max 0.8 para keyword-based
                
                if confidence >= pattern_info["confidence_threshold"]:
                    pattern = AIPattern(
                        pattern_name=pattern_info["name"],
                        pattern_type=pattern_info["type"],
                        confidence=confidence,
                        description=pattern_info["description"],
                        location_info=f"Keywords found: {keyword_matches}/{total_keywords}",
                        recommendations=self._get_pattern_recommendations(pattern_id, pattern_info["type"])
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_structural_patterns(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> List[AIPattern]:
        """Detecta patrones basándose en estructura del código."""
        patterns = []
        lines = code.splitlines()
        
        # Detectar God Class
        if language in [ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVA]:
            class_methods = self._count_methods_in_classes(code, language)
            for class_name, method_count in class_methods.items():
                if method_count > 20:  # Umbral para God Class
                    confidence = min(0.95, 0.6 + (method_count - 20) * 0.02)
                    
                    pattern = AIPattern(
                        pattern_name="God Class",
                        pattern_type="anti_pattern",
                        confidence=confidence,
                        description=f"Class {class_name} has too many methods ({method_count})",
                        location_info=f"Class: {class_name}",
                        recommendations=[
                            "Break down the class into smaller, focused classes",
                            "Apply Single Responsibility Principle",
                            "Extract related methods into separate classes"
                        ]
                    )
                    patterns.append(pattern)
        
        # Detectar código complejo (muchas condiciones anidadas)
        nesting_level = self._calculate_max_nesting_level(code)
        if nesting_level > 5:
            confidence = min(0.9, 0.5 + (nesting_level - 5) * 0.08)
            
            pattern = AIPattern(
                pattern_name="Deeply Nested Code",
                pattern_type="anti_pattern",
                confidence=confidence,
                description=f"Code has deeply nested structures (level {nesting_level})",
                location_info="Multiple locations",
                recommendations=[
                    "Extract methods to reduce nesting",
                    "Use early returns to flatten structure",
                    "Consider using guard clauses"
                ]
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_embedding_patterns(
        self, 
        embedding: CodeEmbedding, 
        language: ProgrammingLanguage
    ) -> List[AIPattern]:
        """Detecta patrones basándose en embeddings semánticos."""
        patterns = []
        
        # Esta es una implementación simplificada
        # En un sistema real, compararíamos con embeddings de patrones conocidos
        
        # Simular detección basada en embedding
        embedding_magnitude = sum(x*x for x in embedding.embedding_vector) ** 0.5
        
        if embedding_magnitude > 20.0:  # Umbral arbitrario
            pattern = AIPattern(
                pattern_name="Complex Code Structure",
                pattern_type="complexity_pattern",
                confidence=0.7,
                description="Code shows high semantic complexity based on embedding analysis",
                location_info="Entire code block",
                recommendations=[
                    "Consider breaking down into smaller functions",
                    "Review for potential simplification opportunities"
                ]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _count_methods_in_classes(self, code: str, language: ProgrammingLanguage) -> Dict[str, int]:
        """Cuenta métodos en clases."""
        method_counts = {}
        
        if language == ProgrammingLanguage.PYTHON:
            import re
            
            # Buscar clases
            class_pattern = r'class\s+(\w+)'
            method_pattern = r'def\s+(\w+)'
            
            classes = re.findall(class_pattern, code)
            methods = re.findall(method_pattern, code)
            
            # Estimación simple (en implementación real usaríamos AST)
            for class_name in classes:
                method_counts[class_name] = len(methods) // max(1, len(classes))
        
        return method_counts
    
    def _calculate_max_nesting_level(self, code: str) -> int:
        """Calcula nivel máximo de anidamiento."""
        max_level = 0
        current_level = 0
        
        for line in code.splitlines():
            stripped = line.strip()
            
            # Incrementar nivel por estructuras de control
            if any(stripped.startswith(keyword) for keyword in ['if', 'for', 'while', 'try', 'with']):
                current_level += 1
                max_level = max(max_level, current_level)
            
            # Decrementar por cierre de bloques (aproximación)
            elif stripped.startswith(('else', 'elif', 'except', 'finally')):
                pass  # Mismo nivel
            elif not stripped and current_level > 0:
                # Línea vacía podría indicar fin de bloque
                current_level = max(0, current_level - 1)
        
        return max_level
    
    def _get_pattern_recommendations(self, pattern_id: str, pattern_type: str) -> List[str]:
        """Obtiene recomendaciones para un patrón."""
        if pattern_type == "design_pattern":
            return [
                f"Good use of {pattern_id} pattern",
                "Consider documenting the pattern implementation",
                "Ensure pattern is necessary for current use case"
            ]
        elif pattern_type == "anti_pattern":
            return [
                f"Consider refactoring to avoid {pattern_id}",
                "Review design to improve maintainability",
                "Consider alternative architectural approaches"
            ]
        else:
            return ["Review implementation for potential improvements"]


class AnomalyDetector:
    """Detector de anomalías en código."""
    
    def __init__(self, model_manager: AIModelManager, config: AIAnalysisConfig):
        self.model_manager = model_manager
        self.config = config
        self.baseline_metrics = self._initialize_baseline_metrics()
    
    def _initialize_baseline_metrics(self) -> Dict[str, float]:
        """Inicializa métricas baseline para detección de anomalías."""
        return {
            "average_function_length": 15.0,
            "average_cyclomatic_complexity": 3.5,
            "average_nesting_depth": 2.0,
            "average_parameter_count": 3.0,
            "typical_comment_ratio": 0.15
        }
    
    async def detect_anomalies(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> List[CodeAnomaly]:
        """
        Detecta anomalías en el código.
        
        Args:
            code: Código a analizar
            language: Lenguaje del código
            
        Returns:
            Lista de anomalías detectadas
        """
        anomalies = []
        
        # Anomalías de tamaño
        size_anomalies = await self._detect_size_anomalies(code, language)
        anomalies.extend(size_anomalies)
        
        # Anomalías de complejidad
        complexity_anomalies = await self._detect_complexity_anomalies(code, language)
        anomalies.extend(complexity_anomalies)
        
        # Anomalías de estilo
        style_anomalies = await self._detect_style_anomalies(code, language)
        anomalies.extend(style_anomalies)
        
        # Filtrar por confianza
        filtered_anomalies = [
            anomaly for anomaly in anomalies
            if anomaly.confidence >= self.config.confidence_threshold
        ]
        
        return filtered_anomalies
    
    async def _detect_size_anomalies(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> List[CodeAnomaly]:
        """Detecta anomalías de tamaño."""
        anomalies = []
        lines = code.splitlines()
        
        # Función muy larga
        if len(lines) > 100:  # Umbral para función larga
            confidence = min(0.95, 0.6 + (len(lines) - 100) * 0.005)
            
            anomaly = CodeAnomaly(
                anomaly_type="excessive_length",
                severity="medium" if len(lines) < 200 else "high",
                confidence=confidence,
                description=f"Code block is unusually long ({len(lines)} lines)",
                affected_code=code[:200] + "..." if len(code) > 200 else code,
                potential_issues=[
                    "Reduced readability",
                    "Difficult to maintain",
                    "Possible violation of SRP"
                ],
                suggested_fixes=[
                    "Break down into smaller functions",
                    "Extract helper methods",
                    "Separate concerns"
                ]
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_complexity_anomalies(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> List[CodeAnomaly]:
        """Detecta anomalías de complejidad."""
        anomalies = []
        
        # Calcular complejidad ciclomática aproximada
        complexity = self._calculate_cyclomatic_complexity(code)
        
        if complexity > 10:  # Umbral de complejidad alta
            confidence = min(0.9, 0.5 + (complexity - 10) * 0.05)
            
            anomaly = CodeAnomaly(
                anomaly_type="complexity_spike",
                severity="medium" if complexity < 15 else "high",
                confidence=confidence,
                description=f"High cyclomatic complexity ({complexity})",
                affected_code=self._extract_complex_sections(code),
                potential_issues=[
                    "Hard to test",
                    "Prone to bugs", 
                    "Difficult to understand"
                ],
                suggested_fixes=[
                    "Simplify control flow",
                    "Extract conditions into methods",
                    "Use strategy pattern for complex logic"
                ]
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_style_anomalies(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> List[CodeAnomaly]:
        """Detecta anomalías de estilo."""
        anomalies = []
        
        # Falta de comentarios
        comment_ratio = self._calculate_comment_ratio(code, language)
        expected_ratio = self.baseline_metrics["typical_comment_ratio"]
        
        if comment_ratio < expected_ratio * 0.5:  # Menos del 50% esperado
            confidence = 0.7
            
            anomaly = CodeAnomaly(
                anomaly_type="style_deviation",
                severity="low",
                confidence=confidence,
                description=f"Unusually low comment ratio ({comment_ratio:.2f})",
                affected_code="Entire code block",
                potential_issues=[
                    "Reduced code maintainability",
                    "Harder for team members to understand"
                ],
                suggested_fixes=[
                    "Add explanatory comments",
                    "Document complex logic",
                    "Add function/class docstrings"
                ]
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calcula complejidad ciclomática aproximada."""
        complexity = 1  # Base complexity
        
        # Contar puntos de decisión
        decision_points = [
            'if', 'elif', 'else', 'for', 'while', 'try', 'except',
            'and', 'or', '?', 'case', 'switch'
        ]
        
        code_lower = code.lower()
        for point in decision_points:
            complexity += code_lower.count(point)
        
        return complexity
    
    def _calculate_comment_ratio(self, code: str, language: ProgrammingLanguage) -> float:
        """Calcula ratio de comentarios."""
        lines = code.splitlines()
        comment_lines = 0
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines == 0:
            return 0.0
        
        for line in lines:
            stripped = line.strip()
            if language == ProgrammingLanguage.PYTHON and stripped.startswith('#'):
                comment_lines += 1
            elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
                if stripped.startswith('//') or ('/*' in stripped and '*/' in stripped):
                    comment_lines += 1
        
        return comment_lines / total_lines
    
    def _extract_complex_sections(self, code: str) -> str:
        """Extrae secciones más complejas del código."""
        lines = code.splitlines()
        complex_lines = []
        
        for line in lines:
            # Identificar líneas complejas (con múltiples operadores de control)
            if sum(line.count(keyword) for keyword in ['if', 'for', 'while', 'and', 'or']) > 1:
                complex_lines.append(line)
        
        if complex_lines:
            return '\n'.join(complex_lines[:5])  # Primeras 5 líneas complejas
        
        return code[:200] + "..."  # Fallback


class SemanticAnalyzer:
    """Analizador semántico de código."""
    
    def __init__(self, model_manager: AIModelManager, config: AIAnalysisConfig):
        self.model_manager = model_manager
        self.config = config
    
    async def analyze(
        self, 
        code: str, 
        language: ProgrammingLanguage,
        embedding: Optional[CodeEmbedding] = None
    ) -> List[SemanticInsight]:
        """
        Realiza análisis semántico del código.
        
        Args:
            code: Código a analizar
            language: Lenguaje del código
            embedding: Embedding del código (opcional)
            
        Returns:
            Lista de insights semánticos
        """
        insights = []
        
        # Análisis de propósito de funciones
        function_insights = await self._analyze_function_purpose(code, language)
        insights.extend(function_insights)
        
        # Análisis de patrones de negocio
        business_insights = await self._analyze_business_logic(code, language)
        insights.extend(business_insights)
        
        # Análisis semántico basado en embedding
        if embedding:
            embedding_insights = await self._analyze_semantic_embedding(embedding, code)
            insights.extend(embedding_insights)
        
        return insights
    
    async def _analyze_function_purpose(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> List[SemanticInsight]:
        """Analiza el propósito de las funciones."""
        insights = []
        
        # Identificar funciones por patrón de nombres
        function_patterns = {
            'validation': ['validate', 'check', 'verify', 'is_valid'],
            'transformation': ['convert', 'transform', 'parse', 'format'],
            'data_access': ['get', 'fetch', 'load', 'retrieve', 'find'],
            'persistence': ['save', 'store', 'persist', 'write', 'update'],
            'calculation': ['calculate', 'compute', 'sum', 'total', 'count']
        }
        
        code_lower = code.lower()
        
        for purpose, patterns in function_patterns.items():
            for pattern in patterns:
                if pattern in code_lower:
                    insight = SemanticInsight(
                        insight_type="function_purpose",
                        description=f"Code appears to handle {purpose} operations",
                        confidence=0.7,
                        evidence=[f"Contains '{pattern}' pattern"],
                        related_concepts=[purpose, "functional_programming"],
                        abstraction_level="function"
                    )
                    insights.append(insight)
                    break  # Solo una vez por categoría
        
        return insights
    
    async def _analyze_business_logic(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> List[SemanticInsight]:
        """Analiza lógica de negocio."""
        insights = []
        
        # Detectar dominios de negocio comunes
        business_domains = {
            'authentication': ['login', 'password', 'auth', 'token', 'session'],
            'payment': ['payment', 'price', 'cost', 'billing', 'invoice'],
            'user_management': ['user', 'account', 'profile', 'registration'],
            'inventory': ['product', 'stock', 'inventory', 'catalog'],
            'reporting': ['report', 'analytics', 'metrics', 'dashboard']
        }
        
        code_lower = code.lower()
        
        for domain, keywords in business_domains.items():
            domain_score = sum(1 for keyword in keywords if keyword in code_lower)
            
            if domain_score >= 2:  # Al menos 2 keywords del dominio
                confidence = min(0.9, 0.5 + domain_score * 0.1)
                
                insight = SemanticInsight(
                    insight_type="business_logic",
                    description=f"Code implements {domain.replace('_', ' ')} functionality",
                    confidence=confidence,
                    evidence=[f"Domain keywords: {domain_score}/{len(keywords)}"],
                    related_concepts=[domain, "business_rules"],
                    abstraction_level="module"
                )
                insights.append(insight)
        
        return insights
    
    async def _analyze_semantic_embedding(
        self, 
        embedding: CodeEmbedding, 
        code: str
    ) -> List[SemanticInsight]:
        """Analiza semánticamente usando embeddings."""
        insights = []
        
        # Análisis simplificado basado en propiedades del embedding
        embedding_stats = self._calculate_embedding_statistics(embedding.embedding_vector)
        
        if embedding_stats['variance'] > 0.1:  # Alta varianza indica diversidad semántica
            insight = SemanticInsight(
                insight_type="semantic_diversity",
                description="Code shows high semantic diversity, indicating multiple concerns",
                confidence=0.6,
                evidence=[f"Embedding variance: {embedding_stats['variance']:.3f}"],
                related_concepts=["separation_of_concerns", "single_responsibility"],
                abstraction_level="function"
            )
            insights.append(insight)
        
        if embedding_stats['magnitude'] > 25.0:  # Alta magnitud indica complejidad
            insight = SemanticInsight(
                insight_type="semantic_complexity",
                description="Code demonstrates high semantic complexity",
                confidence=0.65,
                evidence=[f"Embedding magnitude: {embedding_stats['magnitude']:.2f}"],
                related_concepts=["complexity", "refactoring"],
                abstraction_level="function"
            )
            insights.append(insight)
        
        return insights
    
    def _calculate_embedding_statistics(self, embedding_vector: List[float]) -> Dict[str, float]:
        """Calcula estadísticas del vector de embedding."""
        if not embedding_vector:
            return {"variance": 0.0, "magnitude": 0.0, "mean": 0.0}
        
        mean = sum(embedding_vector) / len(embedding_vector)
        variance = sum((x - mean) ** 2 for x in embedding_vector) / len(embedding_vector)
        magnitude = sum(x ** 2 for x in embedding_vector) ** 0.5
        
        return {
            "variance": variance,
            "magnitude": magnitude,
            "mean": mean
        }


class AICodeAnalyzer:
    """Analizador principal de código con IA."""
    
    def __init__(
        self,
        model_manager: AIModelManager,
        embedding_engine: CodeEmbeddingEngine,
        vector_store: VectorStore,
        inference_engine: InferenceEngine,
        config: AIAnalysisConfig
    ):
        """
        Inicializa el analizador de código con IA.
        
        Args:
            model_manager: Gestor de modelos de IA
            embedding_engine: Motor de embeddings
            vector_store: Almacén vectorial
            inference_engine: Motor de inferencia
            config: Configuración de análisis
        """
        self.model_manager = model_manager
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.inference_engine = inference_engine
        self.config = config
        
        # Inicializar componentes especializados
        self.pattern_detector = PatternDetector(model_manager, config)
        self.anomaly_detector = AnomalyDetector(model_manager, config)
        self.semantic_analyzer = SemanticAnalyzer(model_manager, config)
        
        # Estadísticas y sesiones
        self.active_sessions: Dict[str, AnalysisSession] = {}
        self.analysis_history: List[AIAnalysisResult] = []
    
    async def analyze_code(
        self, 
        code: str, 
        language: ProgrammingLanguage,
        analysis_types: Optional[List[AnalysisType]] = None
    ) -> AIAnalysisResult:
        """
        Analiza código usando IA.
        
        Args:
            code: Código fuente a analizar
            language: Lenguaje de programación
            analysis_types: Tipos de análisis a realizar
            
        Returns:
            Resultado completo del análisis
        """
        start_time = time.time()
        
        # Usar todos los tipos de análisis si no se especifican
        if analysis_types is None:
            analysis_types = [
                AnalysisType.SEMANTIC_SIMILARITY,
                AnalysisType.PATTERN_DETECTION,
                AnalysisType.ANOMALY_DETECTION,
                AnalysisType.CODE_QUALITY_PREDICTION
            ]
        
        # Seleccionar modelo apropiado
        best_model_id = self.model_manager.get_best_model_for_language(language)
        if not best_model_id:
            best_model_id = "microsoft/codebert-base"  # Fallback
        
        # Inicializar resultado
        result = AIAnalysisResult(
            code_snippet=code,
            language=language,
            analysis_type=AnalysisType.SEMANTIC_SIMILARITY,  # Primary type
            model_used=best_model_id
        )
        
        try:
            # Generar embedding si es necesario
            embedding = None
            if any(t in analysis_types for t in [
                AnalysisType.SEMANTIC_SIMILARITY, 
                AnalysisType.PATTERN_DETECTION
            ]):
                embedding = await self.embedding_engine.generate_embedding(code, language)
                result.embeddings.append(embedding)
            
            # Ejecutar análisis según tipos solicitados
            if AnalysisType.PATTERN_DETECTION in analysis_types:
                patterns = await self.pattern_detector.detect_patterns(code, language, embedding)
                result.ai_detected_patterns.extend(patterns)
                result.confidence_scores["pattern_detection"] = self._calculate_avg_confidence(patterns)
            
            if AnalysisType.ANOMALY_DETECTION in analysis_types:
                anomalies = await self.anomaly_detector.detect_anomalies(code, language)
                result.anomalies.extend(anomalies)
                result.confidence_scores["anomaly_detection"] = self._calculate_avg_confidence(anomalies)
            
            # Análisis semántico
            semantic_insights = await self.semantic_analyzer.analyze(code, language, embedding)
            result.semantic_insights.extend(semantic_insights)
            result.confidence_scores["semantic_analysis"] = self._calculate_avg_confidence(semantic_insights)
            
            # Búsqueda de similaridad si tenemos embedding
            if embedding and AnalysisType.SEMANTIC_SIMILARITY in analysis_types:
                similar_matches = await self._find_similar_code(embedding, language)
                result.similarity_matches.extend(similar_matches)
                result.confidence_scores["similarity_search"] = 0.8  # Fixed confidence
            
            # Predicción de calidad
            if AnalysisType.CODE_QUALITY_PREDICTION in analysis_types:
                quality_predictions = await self._predict_code_quality(code, language, embedding)
                result.quality_predictions.extend(quality_predictions)
                result.confidence_scores["quality_prediction"] = self._calculate_avg_confidence(quality_predictions)
            
        except Exception as e:
            logger.error(f"Error en análisis de código: {e}")
            result.confidence_scores["error"] = 0.0
        
        # Finalizar resultado
        result.execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Guardar en historial
        self.analysis_history.append(result)
        
        # Mantener solo los últimos 1000 análisis
        if len(self.analysis_history) > 1000:
            self.analysis_history = self.analysis_history[-1000:]
        
        return result
    
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
        # Generar embeddings para ambos códigos
        embedding1 = await self.embedding_engine.generate_embedding(code1, language1)
        embedding2 = await self.embedding_engine.generate_embedding(code2, language2)
        
        # Calcular similitud semántica
        semantic_similarity = self.inference_engine.calculate_cosine_similarity(
            embedding1.embedding_vector,
            embedding2.embedding_vector
        )
        
        # Calcular similitud estructural (simplificado)
        structural_similarity = self._calculate_structural_similarity(code1, code2)
        
        # Calcular similitud funcional (basada en patrones)
        functional_similarity = await self._calculate_functional_similarity(
            code1, language1, code2, language2
        )
        
        # Similitud general
        overall_similarity = (semantic_similarity + structural_similarity + functional_similarity) / 3.0
        
        # Determinar modelo usado
        model_id = self.model_manager.get_best_model_for_language(language1) or "fallback"
        
        return CrossLanguageSimilarity(
            source_code=code1,
            source_language=language1,
            target_code=code2,
            target_language=language2,
            similarity_score=overall_similarity,
            semantic_similarity=semantic_similarity,
            structural_similarity=structural_similarity,
            functional_similarity=functional_similarity,
            model_used=model_id,
            analysis_confidence=0.75
        )
    
    async def _find_similar_code(
        self, 
        embedding: CodeEmbedding, 
        language: ProgrammingLanguage,
        limit: int = 5
    ) -> List[SimilarityMatch]:
        """Encuentra código similar usando el vector store."""
        try:
            similar_embeddings = await self.embedding_engine.search_similar_code(
                embedding.code_snippet,
                embedding.language,
                limit=limit,
                similarity_threshold=0.7
            )
            
            matches = []
            for similar_embedding, similarity_score in similar_embeddings:
                match = SimilarityMatch(
                    embedding_id=similar_embedding.id,
                    similarity_score=similarity_score,
                    code_snippet=similar_embedding.code_snippet,
                    language=similar_embedding.language,
                    metadata={
                        "model_id": similar_embedding.model_id,
                        "code_length": similar_embedding.metadata.code_length
                    },
                    match_type="semantic"
                )
                matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error buscando código similar: {e}")
            return []
    
    async def _predict_code_quality(
        self,
        code: str,
        language: ProgrammingLanguage,
        embedding: Optional[CodeEmbedding] = None
    ) -> List[QualityPrediction]:
        """Predice métricas de calidad del código."""
        predictions = []
        
        # Predicción de mantenibilidad
        maintainability_score = self._estimate_maintainability(code, language)
        maintainability_prediction = QualityPrediction(
            metric_name="maintainability",
            predicted_score=maintainability_score,
            confidence=0.75,
            model_reasoning="Based on code structure, comments, and complexity",
            contributing_factors=[
                f"Code length: {len(code.splitlines())} lines",
                f"Complexity level: {self._estimate_complexity_level(code)}",
                f"Comment ratio: {self.anomaly_detector._calculate_comment_ratio(code, language):.2%}"
            ],
            improvement_suggestions=self._get_maintainability_suggestions(code, maintainability_score)
        )
        predictions.append(maintainability_prediction)
        
        # Predicción de legibilidad
        readability_score = self._estimate_readability(code, language)
        readability_prediction = QualityPrediction(
            metric_name="readability",
            predicted_score=readability_score,
            confidence=0.7,
            model_reasoning="Based on naming conventions, structure, and documentation",
            contributing_factors=[
                f"Average line length: {self._calculate_avg_line_length(code):.1f}",
                f"Variable naming quality: {self._assess_naming_quality(code)}",
                f"Structure clarity: {self._assess_structure_clarity(code)}"
            ],
            improvement_suggestions=self._get_readability_suggestions(code, readability_score)
        )
        predictions.append(readability_prediction)
        
        return predictions
    
    def _calculate_avg_confidence(self, items: List[Any]) -> float:
        """Calcula confianza promedio de una lista de items con atributo confidence."""
        if not items:
            return 0.0
        
        confidences = [getattr(item, 'confidence', 0.0) for item in items]
        return sum(confidences) / len(confidences)
    
    def _calculate_structural_similarity(self, code1: str, code2: str) -> float:
        """Calcula similitud estructural entre códigos."""
        # Simplificado: compara características estructurales básicas
        lines1, lines2 = len(code1.splitlines()), len(code2.splitlines())
        length_similarity = 1.0 - abs(lines1 - lines2) / max(lines1, lines2, 1)
        
        # Compara patrones de indentación
        indent_pattern1 = self._extract_indentation_pattern(code1)
        indent_pattern2 = self._extract_indentation_pattern(code2)
        indent_similarity = self._compare_patterns(indent_pattern1, indent_pattern2)
        
        return (length_similarity + indent_similarity) / 2.0
    
    async def _calculate_functional_similarity(
        self,
        code1: str, language1: ProgrammingLanguage,
        code2: str, language2: ProgrammingLanguage
    ) -> float:
        """Calcula similitud funcional entre códigos."""
        # Detectar patrones en ambos códigos
        patterns1 = await self.pattern_detector.detect_patterns(code1, language1)
        patterns2 = await self.pattern_detector.detect_patterns(code2, language2)
        
        if not patterns1 and not patterns2:
            return 0.5  # Neutral
        
        # Comparar patrones detectados
        common_patterns = 0
        total_patterns = len(patterns1) + len(patterns2)
        
        for p1 in patterns1:
            for p2 in patterns2:
                if p1.pattern_name == p2.pattern_name:
                    common_patterns += 1
                    break
        
        if total_patterns == 0:
            return 0.5
        
        return (common_patterns * 2) / total_patterns
    
    def _estimate_maintainability(self, code: str, language: ProgrammingLanguage) -> float:
        """Estima score de mantenibilidad (0-100)."""
        base_score = 70.0
        
        # Factores positivos
        comment_ratio = self.anomaly_detector._calculate_comment_ratio(code, language)
        if comment_ratio > 0.15:
            base_score += 15.0
        
        # Factores negativos
        complexity = self.anomaly_detector._calculate_cyclomatic_complexity(code)
        if complexity > 10:
            base_score -= (complexity - 10) * 3.0
        
        line_count = len(code.splitlines())
        if line_count > 100:
            base_score -= (line_count - 100) * 0.1
        
        return max(0.0, min(100.0, base_score))
    
    def _estimate_readability(self, code: str, language: ProgrammingLanguage) -> float:
        """Estima score de legibilidad (0-100)."""
        base_score = 65.0
        
        # Factores de legibilidad
        avg_line_length = self._calculate_avg_line_length(code)
        if avg_line_length < 80:  # Líneas no muy largas
            base_score += 10.0
        elif avg_line_length > 120:
            base_score -= 15.0
        
        # Calidad de nombres
        naming_score = self._assess_naming_quality(code)
        base_score += naming_score * 20.0
        
        return max(0.0, min(100.0, base_score))
    
    def _calculate_avg_line_length(self, code: str) -> float:
        """Calcula longitud promedio de línea."""
        lines = [line for line in code.splitlines() if line.strip()]
        if not lines:
            return 0.0
        
        return sum(len(line) for line in lines) / len(lines)
    
    def _assess_naming_quality(self, code: str) -> float:
        """Evalúa calidad de nomenclatura (0-1)."""
        # Simplificado: busca nombres descriptivos vs nombres cortos
        import re
        
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        if not identifiers:
            return 0.5
        
        descriptive_count = sum(1 for name in identifiers if len(name) >= 4)
        return descriptive_count / len(identifiers)
    
    def _assess_structure_clarity(self, code: str) -> str:
        """Evalúa claridad estructural."""
        nesting_level = self.pattern_detector._calculate_max_nesting_level(code)
        
        if nesting_level <= 3:
            return "clear"
        elif nesting_level <= 5:
            return "moderate"
        else:
            return "complex"
    
    def _estimate_complexity_level(self, code: str) -> str:
        """Estima nivel de complejidad."""
        complexity = self.anomaly_detector._calculate_cyclomatic_complexity(code)
        
        if complexity <= 5:
            return "low"
        elif complexity <= 10:
            return "medium"
        else:
            return "high"
    
    def _get_maintainability_suggestions(self, code: str, score: float) -> List[str]:
        """Genera sugerencias para mejorar mantenibilidad."""
        suggestions = []
        
        if score < 50:
            suggestions.append("Consider major refactoring to improve maintainability")
            suggestions.append("Break down large functions into smaller ones")
            suggestions.append("Add comprehensive documentation")
        elif score < 70:
            suggestions.append("Add more comments to explain complex logic")
            suggestions.append("Reduce cyclomatic complexity")
        else:
            suggestions.append("Good maintainability - consider minor improvements")
        
        return suggestions
    
    def _get_readability_suggestions(self, code: str, score: float) -> List[str]:
        """Genera sugerencias para mejorar legibilidad."""
        suggestions = []
        
        avg_line_length = self._calculate_avg_line_length(code)
        if avg_line_length > 100:
            suggestions.append("Break long lines for better readability")
        
        naming_quality = self._assess_naming_quality(code)
        if naming_quality < 0.7:
            suggestions.append("Use more descriptive variable names")
        
        if score < 60:
            suggestions.append("Improve code formatting and structure")
            suggestions.append("Add whitespace for better visual separation")
        
        return suggestions
    
    def _extract_indentation_pattern(self, code: str) -> List[int]:
        """Extrae patrón de indentación."""
        pattern = []
        for line in code.splitlines():
            if line.strip():
                indent = len(line) - len(line.lstrip())
                pattern.append(indent)
        return pattern
    
    def _compare_patterns(self, pattern1: List[int], pattern2: List[int]) -> float:
        """Compara dos patrones de indentación."""
        if not pattern1 or not pattern2:
            return 0.0
        
        min_len = min(len(pattern1), len(pattern2))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for i in range(min_len) if pattern1[i] == pattern2[i])
        return matches / min_len
    
    async def get_analysis_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del analizador."""
        return {
            "total_analyses": len(self.analysis_history),
            "active_sessions": len(self.active_sessions),
            "average_analysis_time_ms": sum(r.execution_time_ms for r in self.analysis_history[-100:]) // max(1, min(100, len(self.analysis_history))),
            "pattern_detections": sum(len(r.ai_detected_patterns) for r in self.analysis_history[-100:]),
            "anomaly_detections": sum(len(r.anomalies) for r in self.analysis_history[-100:]),
            "semantic_insights": sum(len(r.semantic_insights) for r in self.analysis_history[-100:]),
            "supported_analysis_types": [t.value for t in AnalysisType],
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "batch_size": self.config.batch_analysis_size,
                "cross_language_enabled": self.config.enable_cross_language_analysis
            }
        }
