"""
Clasificador especializado para antipatrones de diseño.
"""

import logging
from typing import List, Dict, Any, Optional

from .base_classifier import BaseAntipatternClassifier, DetectedPattern, ClassifierError
from ....domain.entities.antipattern_analysis import (
    AntipatternType, AntipatternFeatures, SeverityIndicator, ResponsibilityType
)
from ....domain.value_objects.source_position import SourcePosition

logger = logging.getLogger(__name__)


class DesignAntipatternClassifier(BaseAntipatternClassifier):
    """Clasificador para antipatrones de diseño."""
    
    def _initialize_feature_weights(self) -> Dict[str, float]:
        """Inicializar pesos de features de diseño."""
        return {
            'class_size': 2.0,
            'method_length': 1.8,
            'parameter_count': 1.5,
            'responsibility_count': 2.5,
            'coupling_level': 2.0,
            'cohesion_level': 1.5,
            'primitive_usage': 1.0,
            'data_clumping': 1.8,
        }
    
    async def detect_patterns(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> List[DetectedPattern]:
        """Detectar antipatrones de diseño."""
        
        detected_patterns = []
        
        try:
            # Detectar Large Class
            large_class_pattern = await self._detect_large_class(features, threshold)
            if large_class_pattern:
                detected_patterns.append(large_class_pattern)
            
            # Detectar Long Method
            long_method_pattern = await self._detect_long_method(features, threshold)
            if long_method_pattern:
                detected_patterns.append(long_method_pattern)
            
            # Detectar Long Parameter List
            long_params_pattern = await self._detect_long_parameter_list(features, threshold)
            if long_params_pattern:
                detected_patterns.append(long_params_pattern)
            
            # Detectar Feature Envy
            feature_envy_pattern = await self._detect_feature_envy(features, threshold)
            if feature_envy_pattern:
                detected_patterns.append(feature_envy_pattern)
            
            # Detectar Data Clumps
            data_clumps_pattern = await self._detect_data_clumps(features, threshold)
            if data_clumps_pattern:
                detected_patterns.append(data_clumps_pattern)
            
            # Detectar Primitive Obsession
            primitive_obsession_pattern = await self._detect_primitive_obsession(features, threshold)
            if primitive_obsession_pattern:
                detected_patterns.append(primitive_obsession_pattern)
                
        except Exception as e:
            logger.error(f"Error in design pattern detection: {e}")
            raise ClassifierError(f"Design classification failed: {e}")
        
        return detected_patterns
    
    async def _detect_large_class(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar Large Class antipattern."""
        
        indicators = {}
        evidence = []
        
        # Evaluar tamaño de la clase más grande
        if features.max_class_size > 0:
            # Normalizar: clases > 200 líneas son problemáticas
            size_score = min(1.0, features.max_class_size / 200.0)
            indicators['class_size'] = size_score
            evidence.append(f"Largest class has {features.max_class_size} lines")
        
        # Evaluar número de métodos
        if features.methods_count > 0:
            # Normalizar: clases > 20 métodos son problemáticas
            method_score = min(1.0, features.methods_count / 20.0)
            indicators['method_count'] = method_score
            evidence.append(f"Class has {features.methods_count} methods")
        
        # Evaluar complejidad total
        if features.cyclomatic_complexity > 10:
            complexity_score = min(1.0, features.cyclomatic_complexity / 50.0)
            indicators['complexity'] = complexity_score
            evidence.append(f"Class complexity: {features.cyclomatic_complexity:.1f}")
        
        # Evaluar responsabilidades múltiples
        if features.distinct_responsibilities > 1:
            resp_score = min(1.0, features.distinct_responsibilities / 5.0)
            indicators['multiple_responsibilities'] = resp_score
            evidence.append(f"Class handles {features.distinct_responsibilities} distinct responsibilities")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.LARGE_CLASS,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.LARGE_CLASS),
                description="Large Class antipattern detected - class has too many responsibilities",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="maintainability_impact",
                        value="high",
                        description="Large classes are difficult to understand and maintain"
                    ),
                    SeverityIndicator(
                        indicator_type="srp_violation",
                        value=True,
                        description="Violates Single Responsibility Principle"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_long_method(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar Long Method antipattern."""
        
        indicators = {}
        evidence = []
        
        # Evaluar longitud del método más largo
        if features.max_method_length > 0:
            # Normalizar: métodos > 50 líneas son problemáticos
            length_score = min(1.0, features.max_method_length / 50.0)
            indicators['method_length'] = length_score
            evidence.append(f"Longest method has {features.max_method_length} lines")
        
        # Evaluar complejidad ciclomática alta en métodos
        if features.cyclomatic_complexity > 5:
            # Para métodos individuales, complejidad > 10 es alta
            complexity_score = min(1.0, (features.cyclomatic_complexity - 5) / 15.0)
            indicators['method_complexity'] = complexity_score
            evidence.append(f"Methods have high cyclomatic complexity")
        
        # Evaluar profundidad de anidamiento
        if features.nesting_depth > 3:
            nesting_score = min(1.0, (features.nesting_depth - 3) / 5.0)
            indicators['nesting_depth'] = nesting_score
            evidence.append(f"High nesting depth: {features.nesting_depth}")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.LONG_METHOD,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.LONG_METHOD),
                description="Long Method antipattern detected - method is too long and complex",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="readability_impact",
                        value="high",
                        description="Long methods are difficult to read and understand"
                    ),
                    SeverityIndicator(
                        indicator_type="testing_difficulty",
                        value="high",
                        description="Long methods are hard to test effectively"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_long_parameter_list(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar Long Parameter List antipattern."""
        
        indicators = {}
        evidence = []
        
        # Heurística: funciones complejas pueden tener muchos parámetros
        if features.functions_count > 0 and features.cyclomatic_complexity > 8:
            # Estimar parámetros basado en complejidad
            param_score = min(1.0, features.cyclomatic_complexity / 20.0)
            indicators['estimated_parameters'] = param_score
            evidence.append("Functions with high complexity likely have many parameters")
        
        # Evaluar complejidad de interfaz
        if features.methods_count > 10:
            interface_score = min(1.0, features.methods_count / 25.0)
            indicators['interface_complexity'] = interface_score
            evidence.append("Many methods suggest complex interfaces")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.LONG_PARAMETER_LIST,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.LONG_PARAMETER_LIST),
                description="Long Parameter List antipattern detected",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="usability_impact",
                        value="medium",
                        description="Long parameter lists are hard to use and remember"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_feature_envy(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar Feature Envy antipattern."""
        
        indicators = {}
        evidence = []
        
        # Evaluar acoplamiento alto
        if features.class_coupling > 0.7:
            indicators['high_coupling'] = features.class_coupling
            evidence.append(f"High coupling score: {features.class_coupling:.2f}")
        
        # Evaluar dependencias externas
        if features.external_dependencies > 5:
            dep_score = min(1.0, features.external_dependencies / 10.0)
            indicators['external_dependencies'] = dep_score
            evidence.append(f"Many external dependencies: {features.external_dependencies}")
        
        # Evaluar imports excesivos
        if features.import_count > 10:
            import_score = min(1.0, features.import_count / 20.0)
            indicators['excessive_imports'] = import_score
            evidence.append(f"Excessive imports: {features.import_count}")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.FEATURE_ENVY,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.FEATURE_ENVY),
                description="Feature Envy antipattern detected - class uses external data too much",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="coupling_issue",
                        value="high",
                        description="High coupling indicates feature envy"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_data_clumps(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar Data Clumps antipattern."""
        
        indicators = {}
        evidence = []
        
        # Heurística: muchos parámetros y métodos sugiere data clumps
        if features.methods_count > 8 and features.cyclomatic_complexity > 10:
            clump_score = min(1.0, (features.methods_count * features.cyclomatic_complexity) / 100.0)
            indicators['potential_clumps'] = clump_score
            evidence.append("Multiple methods with similar complexity patterns")
        
        # Evaluar responsabilidades relacionadas
        data_access_types = [
            ResponsibilityType.DATA_ACCESS,
            ResponsibilityType.VALIDATION
        ]
        
        if any(resp_type in features.responsibility_types for resp_type in data_access_types):
            indicators['data_handling'] = 0.6
            evidence.append("Code handles multiple data-related responsibilities")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.DATA_CLUMPS,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.DATA_CLUMPS),
                description="Data Clumps antipattern detected - related data items appear together",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="design_issue",
                        value="medium",
                        description="Data clumps suggest missing abstractions"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_primitive_obsession(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar Primitive Obsession antipattern."""
        
        indicators = {}
        evidence = []
        
        # Heurística: mucha validación sugiere uso excesivo de primitivos
        if ResponsibilityType.VALIDATION in features.responsibility_types:
            indicators['excessive_validation'] = 0.7
            evidence.append("Excessive validation suggests primitive obsession")
        
        # Evaluar responsabilidades de procesamiento de datos
        if ResponsibilityType.DATA_ACCESS in features.responsibility_types:
            indicators['data_processing'] = 0.5
            evidence.append("Data processing code may overuse primitives")
        
        # Complejidad alta puede indicar manipulación manual de primitivos
        if features.cyclomatic_complexity > 12:
            complexity_score = min(1.0, (features.cyclomatic_complexity - 12) / 20.0)
            indicators['complex_primitive_handling'] = complexity_score
            evidence.append("High complexity may indicate primitive manipulation")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.PRIMITIVE_OBSESSION,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.PRIMITIVE_OBSESSION),
                description="Primitive Obsession antipattern detected - overuse of primitive types",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="design_issue",
                        value="medium",
                        description="Primitive obsession leads to scattered validation and logic"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
