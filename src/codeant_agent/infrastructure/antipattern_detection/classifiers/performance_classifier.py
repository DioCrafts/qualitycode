"""
Clasificador especializado para antipatrones de performance.
"""

import logging
from typing import List, Dict, Any, Optional

from .base_classifier import BaseAntipatternClassifier, DetectedPattern, ClassifierError
from ....domain.entities.antipattern_analysis import (
    AntipatternType, AntipatternFeatures, SeverityIndicator, PerformanceImpact,
    AlgorithmicComplexity
)
from ....domain.value_objects.source_position import SourcePosition

logger = logging.getLogger(__name__)


class PerformanceAntipatternClassifier(BaseAntipatternClassifier):
    """Clasificador para antipatrones de performance."""
    
    def _initialize_feature_weights(self) -> Dict[str, float]:
        """Inicializar pesos de features de performance."""
        return {
            'nested_loops': 2.5,
            'algorithmic_complexity': 2.0,
            'database_in_loops': 3.0,
            'string_concat_in_loops': 2.0,
            'recursive_calls': 1.5,
            'inefficient_data_structures': 1.8,
            'missing_caching': 1.0,
            'memory_allocations': 1.5,
        }
    
    async def detect_patterns(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> List[DetectedPattern]:
        """Detectar antipatrones de performance."""
        
        detected_patterns = []
        
        try:
            # Detectar N+1 Query pattern
            n_plus_one_pattern = await self._detect_n_plus_one_queries(features, threshold)
            if n_plus_one_pattern:
                detected_patterns.append(n_plus_one_pattern)
            
            # Detectar algoritmos ineficientes
            inefficient_algo_pattern = await self._detect_inefficient_algorithms(features, threshold)
            if inefficient_algo_pattern:
                detected_patterns.append(inefficient_algo_pattern)
            
            # Detectar string concatenation in loops
            string_concat_pattern = await self._detect_string_concatenation_in_loops(features, threshold)
            if string_concat_pattern:
                detected_patterns.append(string_concat_pattern)
            
            # Detectar loops no optimizados
            unoptimized_loop_pattern = await self._detect_unoptimized_loops(features, threshold)
            if unoptimized_loop_pattern:
                detected_patterns.append(unoptimized_loop_pattern)
            
            # Detectar memory leaks potenciales
            memory_leak_pattern = await self._detect_memory_leaks(features, threshold)
            if memory_leak_pattern:
                detected_patterns.append(memory_leak_pattern)
                
        except Exception as e:
            logger.error(f"Error in performance pattern detection: {e}")
            raise ClassifierError(f"Performance classification failed: {e}")
        
        return detected_patterns
    
    async def _detect_n_plus_one_queries(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar patrón N+1 Query."""
        
        indicators = {}
        evidence = []
        
        # Verificar si hay loops
        if not features.has_loops:
            return None
        
        indicators['has_loops'] = 0.5
        evidence.append("Code contains loops")
        
        # Verificar operaciones de base de datos en loops
        if features.custom_features and features.custom_features.get('database_calls_in_loops', False):
            indicators['db_in_loops'] = 1.0
            evidence.append("Database operations detected inside loops")
        
        # Simular detección basada en heurísticas
        if features.has_loops and features.has_sql_operations:
            indicators['potential_n_plus_one'] = 0.7
            evidence.append("Loops combined with SQL operations suggest potential N+1 pattern")
        
        # Factor de complejidad
        if features.has_nested_loops:
            indicators['nested_complexity'] = 0.3
            evidence.append("Nested loops increase N+1 risk")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.N_PLUS_ONE_QUERY,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.N_PLUS_ONE_QUERY),
                description="N+1 Query antipattern detected - database queries in loops",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="performance_impact",
                        value=PerformanceImpact.HIGH,
                        description="N+1 queries cause exponential performance degradation"
                    ),
                    SeverityIndicator(
                        indicator_type="scalability_issue",
                        value=True,
                        description="Performance degrades significantly with data size"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_inefficient_algorithms(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar algoritmos ineficientes."""
        
        indicators = {}
        evidence = []
        
        # Evaluar complejidad algorítmica
        complexity_score = self._evaluate_complexity_score(features.algorithmic_complexity)
        if complexity_score > 0:
            indicators['algorithmic_complexity'] = complexity_score
            evidence.append(f"Algorithm has {features.algorithmic_complexity.value} complexity")
        
        # Verificar loops anidados
        if features.has_nested_loops:
            indicators['nested_loops'] = 0.8
            evidence.append("Code contains nested loops")
        
        # Verificar llamadas recursivas sin memoización
        if features.has_recursive_calls:
            indicators['recursive_calls'] = 0.6
            evidence.append("Recursive calls detected (may lack memoization)")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.INEFFICIENT_ALGORITHM,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.INEFFICIENT_ALGORITHM),
                description="Inefficient algorithm with poor time complexity detected",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="performance_impact",
                        value=PerformanceImpact.MEDIUM if complexity_score < 0.8 else PerformanceImpact.HIGH,
                        description="Poor algorithmic complexity affects performance"
                    ),
                    SeverityIndicator(
                        indicator_type="complexity_issue",
                        value=features.algorithmic_complexity,
                        description=f"Algorithm complexity: {features.algorithmic_complexity.value}"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_string_concatenation_in_loops(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar concatenación de strings en loops."""
        
        indicators = {}
        evidence = []
        
        # Verificar si hay loops
        if not features.has_loops:
            return None
        
        # Verificar concatenación de strings en loops
        if features.custom_features and features.custom_features.get('string_concatenation_in_loops', False):
            indicators['string_concat_in_loops'] = 1.0
            evidence.append("String concatenation detected in loops")
        
        # Heurística: loops con código complejo pueden tener concatenación
        if features.has_loops and features.cyclomatic_complexity > 5:
            indicators['complex_loops'] = 0.5
            evidence.append("Complex loops may contain inefficient string operations")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.STRING_CONCATENATION_IN_LOOP,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.STRING_CONCATENATION_IN_LOOP),
                description="String concatenation in loop detected - use string builder instead",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="performance_impact",
                        value=PerformanceImpact.MEDIUM,
                        description="String concatenation in loops creates many intermediate objects"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_unoptimized_loops(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar loops no optimizados."""
        
        indicators = {}
        evidence = []
        
        # Verificar si hay loops
        if not features.has_loops:
            return None
        
        # Loops con alta complejidad
        if features.cyclomatic_complexity > 8 and features.has_loops:
            indicators['complex_loops'] = 0.7
            evidence.append("Loops with high complexity detected")
        
        # Loops anidados profundos
        if features.has_nested_loops:
            indicators['deep_nesting'] = 0.8
            evidence.append("Deeply nested loops detected")
        
        # Loops con muchas operaciones
        if features.nesting_depth > 3:
            indicators['deep_nesting_overall'] = 0.6
            evidence.append("High nesting depth suggests complex loop structures")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.UNOPTIMIZED_LOOP,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.UNOPTIMIZED_LOOP),
                description="Unoptimized loops detected - consider refactoring",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="performance_impact",
                        value=PerformanceImpact.MEDIUM,
                        description="Unoptimized loops can cause performance bottlenecks"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_memory_leaks(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar posibles memory leaks."""
        
        indicators = {}
        evidence = []
        
        # Memory leaks más comunes en ciertos lenguajes
        if features.language.value == 'python':
            # En Python, memory leaks son menos comunes pero posibles
            if features.has_recursive_calls and features.cyclomatic_complexity > 10:
                indicators['recursive_complexity'] = 0.5
                evidence.append("Complex recursive calls may cause memory buildup")
        
        elif features.language.value in ['javascript', 'typescript']:
            # JavaScript memory leaks
            if features.has_loops and features.cyclomatic_complexity > 8:
                indicators['event_listeners'] = 0.6
                evidence.append("Complex loops may create uncleaned event listeners")
        
        # Heurística general: alta complejidad puede indicar manejo inadecuado de recursos
        if features.cyclomatic_complexity > 15:
            indicators['resource_management'] = 0.4
            evidence.append("High complexity may indicate poor resource management")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.MEMORY_LEAK,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.MEMORY_LEAK),
                description="Potential memory leak patterns detected",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="performance_impact",
                        value=PerformanceImpact.HIGH,
                        description="Memory leaks cause progressive performance degradation"
                    ),
                    SeverityIndicator(
                        indicator_type="resource_leak",
                        value=True,
                        description="Uncleaned resources consume system memory"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    def _evaluate_complexity_score(self, complexity: AlgorithmicComplexity) -> float:
        """Evaluar score de complejidad algorítmica."""
        
        complexity_scores = {
            AlgorithmicComplexity.CONSTANT: 0.1,
            AlgorithmicComplexity.LOGARITHMIC: 0.2,
            AlgorithmicComplexity.LINEAR: 0.3,
            AlgorithmicComplexity.LINEAR_LOGARITHMIC: 0.5,
            AlgorithmicComplexity.QUADRATIC: 0.8,
            AlgorithmicComplexity.CUBIC: 0.9,
            AlgorithmicComplexity.EXPONENTIAL: 1.0,
            AlgorithmicComplexity.FACTORIAL: 1.0,
            AlgorithmicComplexity.UNKNOWN: 0.0,
        }
        
        return complexity_scores.get(complexity, 0.0)
