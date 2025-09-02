"""
Clasificador especializado para antipatrones arquitectónicos.
"""

import logging
from typing import List, Dict, Any, Optional

from .base_classifier import BaseAntipatternClassifier, DetectedPattern, ClassifierError
from ....domain.entities.antipattern_analysis import (
    AntipatternType, AntipatternFeatures, SeverityIndicator
)
from ....domain.value_objects.source_position import SourcePosition

logger = logging.getLogger(__name__)


class ArchitecturalAntipatternClassifier(BaseAntipatternClassifier):
    """Clasificador para antipatrones arquitectónicos."""
    
    def _initialize_feature_weights(self) -> Dict[str, float]:
        """Inicializar pesos de features arquitectónicos."""
        return {
            'god_object_indicators': 3.0,
            'spaghetti_code_indicators': 2.5,
            'lava_flow_indicators': 2.0,
            'big_ball_of_mud': 2.8,
            'golden_hammer': 1.5,
            'dependency_complexity': 2.0,
            'architectural_violations': 2.5,
        }
    
    async def detect_patterns(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> List[DetectedPattern]:
        """Detectar antipatrones arquitectónicos."""
        
        detected_patterns = []
        
        try:
            # Detectar God Object
            god_object_pattern = await self._detect_god_object(features, threshold)
            if god_object_pattern:
                detected_patterns.append(god_object_pattern)
            
            # Detectar Spaghetti Code
            spaghetti_pattern = await self._detect_spaghetti_code(features, threshold)
            if spaghetti_pattern:
                detected_patterns.append(spaghetti_pattern)
            
            # Detectar Lava Flow
            lava_flow_pattern = await self._detect_lava_flow(features, threshold)
            if lava_flow_pattern:
                detected_patterns.append(lava_flow_pattern)
            
            # Detectar Big Ball of Mud
            big_ball_pattern = await self._detect_big_ball_of_mud(features, threshold)
            if big_ball_pattern:
                detected_patterns.append(big_ball_pattern)
            
            # Detectar Golden Hammer
            golden_hammer_pattern = await self._detect_golden_hammer(features, threshold)
            if golden_hammer_pattern:
                detected_patterns.append(golden_hammer_pattern)
                
        except Exception as e:
            logger.error(f"Error in architectural pattern detection: {e}")
            raise ClassifierError(f"Architectural classification failed: {e}")
        
        return detected_patterns
    
    async def _detect_god_object(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar God Object antipattern."""
        
        indicators = {}
        evidence = []
        
        # Evaluar tamaño de clase
        if features.max_class_size > 200:
            size_score = min(1.0, features.max_class_size / 500.0)
            indicators['class_size'] = size_score
            evidence.append(f"Extremely large class: {features.max_class_size} lines")
        
        # Evaluar número de métodos
        if features.methods_count > 20:
            method_score = min(1.0, features.methods_count / 40.0)
            indicators['method_count'] = method_score
            evidence.append(f"Too many methods: {features.methods_count}")
        
        # Evaluar responsabilidades múltiples
        if features.distinct_responsibilities > 3:
            resp_score = min(1.0, features.distinct_responsibilities / 8.0)
            indicators['multiple_responsibilities'] = resp_score
            evidence.append(f"Handles {features.distinct_responsibilities} distinct responsibilities")
        
        # Evaluar complejidad total
        if features.cyclomatic_complexity > 30:
            complexity_score = min(1.0, features.cyclomatic_complexity / 100.0)
            indicators['total_complexity'] = complexity_score
            evidence.append(f"Extremely high complexity: {features.cyclomatic_complexity:.1f}")
        
        # Evaluar acoplamiento alto
        if features.class_coupling > 0.8:
            indicators['high_coupling'] = features.class_coupling
            evidence.append(f"High coupling: {features.class_coupling:.2f}")
        
        # Factor de importaciones excesivas
        if features.import_count > 15:
            import_score = min(1.0, features.import_count / 30.0)
            indicators['excessive_imports'] = import_score
            evidence.append(f"Excessive imports: {features.import_count}")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.GOD_OBJECT,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.GOD_OBJECT),
                description="God Object antipattern detected - class has excessive responsibilities",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="architectural_violation",
                        value="critical",
                        description="God Object violates fundamental design principles"
                    ),
                    SeverityIndicator(
                        indicator_type="maintainability_impact",
                        value="critical",
                        description="Extremely difficult to maintain and understand"
                    ),
                    SeverityIndicator(
                        indicator_type="srp_violation",
                        value=True,
                        description="Severe violation of Single Responsibility Principle"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_spaghetti_code(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar Spaghetti Code antipattern."""
        
        indicators = {}
        evidence = []
        
        # Evaluar complejidad ciclomática alta
        if features.cyclomatic_complexity > 15:
            complexity_score = min(1.0, features.cyclomatic_complexity / 40.0)
            indicators['high_complexity'] = complexity_score
            evidence.append(f"High cyclomatic complexity: {features.cyclomatic_complexity:.1f}")
        
        # Evaluar profundidad de anidamiento
        if features.nesting_depth > 5:
            nesting_score = min(1.0, features.nesting_depth / 10.0)
            indicators['deep_nesting'] = nesting_score
            evidence.append(f"Deep nesting: {features.nesting_depth}")
        
        # Evaluar acoplamiento alto entre componentes
        if features.class_coupling > 0.7:
            indicators['tangled_dependencies'] = features.class_coupling
            evidence.append("High coupling indicates tangled dependencies")
        
        # Evaluar muchas dependencias externas
        if features.external_dependencies > 8:
            dep_score = min(1.0, features.external_dependencies / 15.0)
            indicators['external_dependencies'] = dep_score
            evidence.append(f"Many external dependencies: {features.external_dependencies}")
        
        # Evaluar múltiples responsabilidades entrelazadas
        if features.distinct_responsibilities > 4:
            tangle_score = min(1.0, features.distinct_responsibilities / 10.0)
            indicators['responsibility_tangle'] = tangle_score
            evidence.append("Multiple intertwined responsibilities")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.SPAGHETTI_CODE,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.SPAGHETTI_CODE),
                description="Spaghetti Code antipattern detected - tangled and complex control flow",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="architectural_violation",
                        value="high",
                        description="Spaghetti code makes system architecture unclear"
                    ),
                    SeverityIndicator(
                        indicator_type="complexity_issue",
                        value=features.cyclomatic_complexity,
                        description="Control flow is too complex to follow"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_lava_flow(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar Lava Flow antipattern."""
        
        indicators = {}
        evidence = []
        
        # Heurística: código con baja actividad de cambios pero alto tamaño
        if features.lines_of_code > 300 and features.cyclomatic_complexity < 5:
            # Código grande pero simple puede ser lava flow
            lava_score = min(1.0, features.lines_of_code / 1000.0)
            indicators['inactive_large_code'] = lava_score
            evidence.append("Large amount of code with low complexity may be dead/stale")
        
        # Evaluar muchas importaciones pero poca funcionalidad
        if features.import_count > 10 and features.functions_count < 3:
            unused_score = 0.6
            indicators['unused_imports'] = unused_score
            evidence.append("Many imports with few functions suggests unused code")
        
        # Heurística: archivos con muchas líneas pero pocos métodos activos
        if features.lines_of_code > 200 and features.methods_count < 5:
            stale_score = min(1.0, features.lines_of_code / 500.0)
            indicators['stale_code'] = stale_score
            evidence.append("Large files with few active methods")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.LAVA_FLOW,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.LAVA_FLOW),
                description="Lava Flow antipattern detected - dead code kept 'just in case'",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="maintenance_burden",
                        value="medium",
                        description="Dead code increases maintenance complexity"
                    ),
                    SeverityIndicator(
                        indicator_type="code_bloat",
                        value=features.lines_of_code,
                        description="Unused code bloats the codebase"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_big_ball_of_mud(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar Big Ball of Mud antipattern."""
        
        indicators = {}
        evidence = []
        
        # Combinar múltiples indicadores de mala arquitectura
        architectural_issues = 0
        
        # Alta complejidad
        if features.cyclomatic_complexity > 20:
            indicators['high_complexity'] = min(1.0, features.cyclomatic_complexity / 50.0)
            evidence.append(f"High overall complexity: {features.cyclomatic_complexity:.1f}")
            architectural_issues += 1
        
        # Múltiples responsabilidades
        if features.distinct_responsibilities > 5:
            indicators['multiple_responsibilities'] = min(1.0, features.distinct_responsibilities / 10.0)
            evidence.append(f"Too many responsibilities: {features.distinct_responsibilities}")
            architectural_issues += 1
        
        # Alto acoplamiento
        if features.class_coupling > 0.8:
            indicators['high_coupling'] = features.class_coupling
            evidence.append("Very high coupling between components")
            architectural_issues += 1
        
        # Clases muy grandes
        if features.max_class_size > 400:
            indicators['oversized_classes'] = min(1.0, features.max_class_size / 800.0)
            evidence.append(f"Oversized classes: {features.max_class_size} lines")
            architectural_issues += 1
        
        # Métodos muy largos
        if features.max_method_length > 80:
            indicators['oversized_methods'] = min(1.0, features.max_method_length / 150.0)
            evidence.append(f"Oversized methods: {features.max_method_length} lines")
            architectural_issues += 1
        
        # Solo detectar si hay múltiples problemas arquitectónicos
        if architectural_issues >= 3:
            indicators['architectural_issues_count'] = min(1.0, architectural_issues / 5.0)
            evidence.append(f"Multiple architectural issues detected: {architectural_issues}")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold and architectural_issues >= 3:
            return DetectedPattern(
                pattern_type=AntipatternType.BIG_BALL_OF_MUD,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.BIG_BALL_OF_MUD),
                description="Big Ball of Mud antipattern detected - system lacks clear architecture",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="architectural_violation",
                        value="critical",
                        description="System has no clear architectural structure"
                    ),
                    SeverityIndicator(
                        indicator_type="technical_debt",
                        value="high",
                        description="High technical debt due to poor architecture"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_golden_hammer(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar Golden Hammer antipattern."""
        
        indicators = {}
        evidence = []
        
        # Heurística: uso excesivo de patrones similares
        # Muchas funciones similares pueden indicar sobre-aplicación de un patrón
        if features.functions_count > 15 and features.classes_count < 3:
            overuse_score = min(1.0, features.functions_count / 25.0)
            indicators['pattern_overuse'] = overuse_score
            evidence.append("Many similar functions suggest overuse of one approach")
        
        # Importaciones repetitivas del mismo tipo
        if features.import_count > 12:
            # Simular detección de imports similares
            indicators['repetitive_imports'] = min(1.0, features.import_count / 20.0)
            evidence.append("Many imports may indicate overuse of specific libraries")
        
        # Complejidad distribuida uniformemente (sugiere aplicación rígida de patrón)
        if features.cyclomatic_complexity > 10 and features.methods_count > 8:
            uniformity_score = 0.5
            indicators['uniform_complexity'] = uniformity_score
            evidence.append("Uniform complexity patterns suggest rigid approach")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.GOLDEN_HAMMER,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.GOLDEN_HAMMER),
                description="Golden Hammer antipattern detected - overuse of familiar solution",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="design_flexibility",
                        value="low",
                        description="Overuse of one pattern reduces solution flexibility"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
