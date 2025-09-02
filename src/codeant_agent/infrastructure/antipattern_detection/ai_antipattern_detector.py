"""
Detector principal de antipatrones basado en IA que orquesta todo el sistema.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from ...domain.entities.antipattern_analysis import (
    AntipatternDetectionResult, AntipatternDetectionConfig, ProjectAntipatternAnalysis,
    AntipatternType, AntipatternCategory, AntipatternHotspot, AntipatternTrend,
    RemediationPriority, DetectedAntipattern, AntipatternSeverity,
    ImpactAnalysis, PerformanceImpact, SecurityRisk
)
from ...domain.entities.ai_models import AIAnalysisResult
from ...domain.value_objects.programming_language import ProgrammingLanguage
from ...domain.value_objects.source_position import SourcePosition
from ..ast_analysis.unified_ast import UnifiedAST

from .feature_extractor import AntipatternFeatureExtractor
from .classifiers import (
    SecurityAntipatternClassifier, PerformanceAntipatternClassifier,
    DesignAntipatternClassifier, ArchitecturalAntipatternClassifier
)
from .explanation_generator import ExplanationGenerator
from .confidence_calibrator import ConfidenceCalibrator
from .ensemble_detector import EnsembleDetector
from .contextual_analyzer import ContextualAnalyzer

logger = logging.getLogger(__name__)


class AntipatternDetectionError(Exception):
    """Excepción para errores en detección de antipatrones."""
    pass


class AIAntipatternDetector:
    """Detector principal de antipatrones basado en IA."""
    
    def __init__(self, config: Optional[AntipatternDetectionConfig] = None):
        self.config = config or AntipatternDetectionConfig()
        
        # Componentes del sistema
        self.feature_extractor = AntipatternFeatureExtractor()
        self.explanation_generator = ExplanationGenerator()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.ensemble_detector = EnsembleDetector()
        self.contextual_analyzer = ContextualAnalyzer()
        
        # Clasificadores especializados
        self.classifiers = {}
        self._initialize_classifiers()
        
        # Estadísticas del sistema
        self.detection_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "total_patterns_detected": 0,
            "avg_detection_time_ms": 0.0,
            "patterns_by_category": {category: 0 for category in AntipatternCategory},
            "patterns_by_type": {pattern_type: 0 for pattern_type in AntipatternType}
        }
    
    def _initialize_classifiers(self):
        """Inicializar clasificadores especializados."""
        
        try:
            self.classifiers = {
                AntipatternCategory.SECURITY: SecurityAntipatternClassifier(),
                AntipatternCategory.PERFORMANCE: PerformanceAntipatternClassifier(),
                AntipatternCategory.DESIGN: DesignAntipatternClassifier(),
                AntipatternCategory.ARCHITECTURAL: ArchitecturalAntipatternClassifier()
            }
        except Exception as e:
            logger.error(f"Error initializing classifiers: {e}")
            raise AntipatternDetectionError(f"Failed to initialize classifiers: {e}")
    
    async def detect_antipatterns(
        self, 
        ai_analysis: AIAnalysisResult, 
        unified_ast: UnifiedAST
    ) -> AntipatternDetectionResult:
        """Detectar antipatrones en un archivo."""
        
        start_time = time.time()
        
        try:
            # Extraer features para detección
            features = await self.feature_extractor.extract_features(ai_analysis, unified_ast)
            
            # Inicializar resultado
            result = AntipatternDetectionResult(
                file_path=unified_ast.file_path,
                language=unified_ast.language,
                detection_time_ms=0
            )
            
            # Ejecutar clasificadores especializados
            all_detected_patterns = []
            
            for category, classifier in self.classifiers.items():
                try:
                    threshold = self.config.category_specific_thresholds.get(
                        category, self.config.confidence_threshold
                    )
                    
                    patterns = await classifier.detect_patterns(features, threshold)
                    all_detected_patterns.extend(patterns)
                    
                    logger.debug(f"Classifier {category} detected {len(patterns)} patterns")
                    
                except Exception as e:
                    logger.error(f"Error in {category} classifier: {e}")
                    continue
            
            # Crear resultado preliminar
            preliminary_result = AntipatternDetectionResult(
                file_path=unified_ast.file_path,
                language=unified_ast.language,
                detected_antipatterns=all_detected_patterns,
                detection_time_ms=0
            )
            
            # Aplicar ensemble detection si está habilitado
            if self.config.enable_ensemble_detection and len(all_detected_patterns) > 0:
                ensemble_patterns = await self.ensemble_detector.detect_ensemble_antipatterns(
                    features, preliminary_result
                )
                result.detected_antipatterns = ensemble_patterns
            else:
                result.detected_antipatterns = all_detected_patterns
            
            # Aplicar contextual analysis si está habilitado
            if self.config.enable_contextual_analysis:
                result = await self.contextual_analyzer.refine_detections(result, unified_ast)
            
            # Calibrar confianzas
            calibrated_patterns = []
            for pattern in result.detected_antipatterns:
                try:
                    calibrated_confidence = await self.confidence_calibrator.calibrate_confidence(
                        pattern, features
                    )
                    pattern.confidence = calibrated_confidence
                    calibrated_patterns.append(pattern)
                except Exception as e:
                    logger.warning(f"Error calibrating confidence for {pattern.pattern_type}: {e}")
                    calibrated_patterns.append(pattern)
            
            result.detected_antipatterns = calibrated_patterns
            
            # Filtrar por threshold final
            final_patterns = [
                pattern for pattern in result.detected_antipatterns
                if pattern.confidence >= self.config.confidence_threshold
            ]
            result.detected_antipatterns = final_patterns
            
            # Limitar número de patrones si es necesario
            if len(final_patterns) > self.config.max_patterns_per_analysis:
                # Ordenar por confianza y tomar los mejores
                sorted_patterns = sorted(final_patterns, key=lambda p: p.confidence, reverse=True)
                result.detected_antipatterns = sorted_patterns[:self.config.max_patterns_per_analysis]
            
            # Convertir a DetectedAntipatterns y categorizar
            result.detected_antipatterns = await self._convert_to_detected_antipatterns(
                result.detected_antipatterns, features
            )
            
            # Categorizar por tipo
            result.architectural_issues = self._filter_by_category(
                result.detected_antipatterns, AntipatternCategory.ARCHITECTURAL
            )
            result.design_issues = self._filter_by_category(
                result.detected_antipatterns, AntipatternCategory.DESIGN
            )
            result.performance_issues = self._filter_by_category(
                result.detected_antipatterns, AntipatternCategory.PERFORMANCE
            )
            result.security_issues = self._filter_by_category(
                result.detected_antipatterns, AntipatternCategory.SECURITY
            )
            
            # Calcular confidence scores
            result.confidence_scores = {
                pattern.pattern_type: pattern.confidence 
                for pattern in result.detected_antipatterns
            }
            
            # Generar explicaciones si está habilitado
            if self.config.enable_explanation_generation:
                explanations = []
                for pattern in result.detected_antipatterns:
                    if hasattr(pattern, 'explanation') and pattern.explanation:
                        explanations.append(pattern.explanation)
                result.explanations = explanations
            
            # Calcular quality score
            result.quality_score = self._calculate_quality_score(result, features)
            
            # Finalizar timing
            detection_time_ms = int((time.time() - start_time) * 1000)
            result.detection_time_ms = detection_time_ms
            
            # Actualizar estadísticas
            self._update_detection_stats(result, detection_time_ms)
            
            logger.info(f"Detected {len(result.detected_antipatterns)} antipatterns in {detection_time_ms}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in antipattern detection: {e}")
            raise AntipatternDetectionError(f"Detection failed: {e}")
    
    async def detect_project_antipatterns(
        self, 
        project_analyses: List[AIAnalysisResult],
        project_asts: List[UnifiedAST]
    ) -> ProjectAntipatternAnalysis:
        """Detectar antipatrones a nivel de proyecto."""
        
        start_time = time.time()
        
        try:
            # Detectar antipatrones por archivo
            file_results = []
            
            for analysis, ast in zip(project_analyses, project_asts):
                try:
                    file_result = await self.detect_antipatterns(analysis, ast)
                    file_results.append(file_result)
                except Exception as e:
                    logger.error(f"Error analyzing {ast.file_path}: {e}")
                    continue
            
            # Detectar antipatrones a nivel de proyecto
            project_level_antipatterns = await self._detect_project_level_patterns(
                file_results, project_asts
            )
            
            # Análisis arquitectónico
            architectural_analysis = await self._analyze_project_architecture(
                file_results, project_asts
            )
            
            # Detectar antipatrones cross-file
            cross_file_antipatterns = await self._detect_cross_file_patterns(
                file_results, project_asts
            )
            
            # Identificar hotspots
            hotspots = await self._identify_antipattern_hotspots(file_results)
            
            # Analizar tendencias
            trends = await self._analyze_antipattern_trends(file_results)
            
            # Calcular prioridades de remediación
            remediation_priorities = await self._calculate_remediation_priorities(
                file_results, project_asts
            )
            
            # Calcular quality score general del proyecto
            overall_quality_score = self._calculate_project_quality_score(file_results)
            
            # Generar recomendaciones
            recommendations = self._generate_project_recommendations(
                file_results, hotspots, trends
            )
            
            detection_time_ms = int((time.time() - start_time) * 1000)
            
            project_analysis = ProjectAntipatternAnalysis(
                project_path=project_asts[0].file_path.parent if project_asts else None,
                file_antipatterns=file_results,
                project_level_antipatterns=project_level_antipatterns,
                architectural_analysis=architectural_analysis,
                cross_file_antipatterns=cross_file_antipatterns,
                hotspots=hotspots,
                trends=trends,
                remediation_priorities=remediation_priorities,
                detection_time_ms=detection_time_ms,
                overall_quality_score=overall_quality_score,
                recommendations=recommendations
            )
            
            logger.info(f"Project analysis completed in {detection_time_ms}ms")
            
            return project_analysis
            
        except Exception as e:
            logger.error(f"Error in project antipattern detection: {e}")
            raise AntipatternDetectionError(f"Project detection failed: {e}")
    
    async def _convert_to_detected_antipatterns(
        self, 
        patterns: List,
        features: Any
    ) -> List[DetectedAntipattern]:
        """Convertir patrones detectados a DetectedAntipattern."""
        
        detected_antipatterns = []
        
        for pattern in patterns:
            # Determinar severidad
            severity = self._determine_severity(pattern)
            
            # Crear análisis de impacto
            impact_analysis = await self._create_impact_analysis(pattern, features)
            
            # Generar explicación si está habilitado
            explanation = None
            if self.config.enable_explanation_generation:
                try:
                    explanation = await self.explanation_generator.generate_explanation(
                        pattern, features
                    )
                except Exception as e:
                    logger.warning(f"Error generating explanation: {e}")
            
            # Convertir a DetectedAntipattern
            antipattern = DetectedAntipattern(
                pattern_type=pattern.pattern_type,
                category=self._get_pattern_category(pattern.pattern_type),
                severity=severity,
                confidence=pattern.confidence,
                locations=pattern.locations,
                description=pattern.description,
                explanation=explanation,
                fix_suggestions=[],  # Se añadirían en una implementación completa
                impact_analysis=impact_analysis,
                detected_at=datetime.now(),
                evidence=pattern.evidence,
                severity_indicators=pattern.severity_indicators
            )
            
            detected_antipatterns.append(antipattern)
        
        return detected_antipatterns
    
    def _determine_severity(self, pattern) -> AntipatternSeverity:
        """Determinar severidad de un patrón."""
        
        # Basado en tipo de patrón y confianza
        critical_patterns = [
            AntipatternType.SQL_INJECTION,
            AntipatternType.HARDCODED_SECRETS,
            AntipatternType.XSS_VULNERABILITY
        ]
        
        high_patterns = [
            AntipatternType.GOD_OBJECT,
            AntipatternType.N_PLUS_ONE_QUERY,
            AntipatternType.MEMORY_LEAK,
            AntipatternType.BIG_BALL_OF_MUD
        ]
        
        if pattern.pattern_type in critical_patterns:
            return AntipatternSeverity.CRITICAL
        elif pattern.pattern_type in high_patterns:
            return AntipatternSeverity.HIGH if pattern.confidence > 0.8 else AntipatternSeverity.MEDIUM
        elif pattern.confidence > 0.9:
            return AntipatternSeverity.HIGH
        elif pattern.confidence > 0.7:
            return AntipatternSeverity.MEDIUM
        else:
            return AntipatternSeverity.LOW
    
    async def _create_impact_analysis(self, pattern, features) -> ImpactAnalysis:
        """Crear análisis de impacto."""
        
        # Mapeo de impactos por tipo de patrón
        impact_mapping = {
            AntipatternType.SQL_INJECTION: {
                "performance": PerformanceImpact.LOW,
                "security": SecurityRisk.CRITICAL,
                "maintainability": "Critical security vulnerability",
                "business": "High risk of data breach and legal issues",
                "debt_hours": 40.0
            },
            AntipatternType.GOD_OBJECT: {
                "performance": PerformanceImpact.MEDIUM,
                "security": SecurityRisk.LOW,
                "maintainability": "Very difficult to maintain and extend",
                "business": "Significantly slows down development",
                "debt_hours": 80.0
            },
            AntipatternType.N_PLUS_ONE_QUERY: {
                "performance": PerformanceImpact.HIGH,
                "security": SecurityRisk.LOW,
                "maintainability": "Moderate impact on code organization",
                "business": "Performance issues affect user experience",
                "debt_hours": 20.0
            }
        }
        
        default_impact = {
            "performance": PerformanceImpact.MEDIUM,
            "security": SecurityRisk.LOW,
            "maintainability": "Moderate maintenance impact",
            "business": "Minor impact on development productivity",
            "debt_hours": 15.0
        }
        
        impact_data = impact_mapping.get(pattern.pattern_type, default_impact)
        
        return ImpactAnalysis(
            performance_impact=impact_data["performance"],
            security_impact=impact_data["security"],
            maintainability_impact=impact_data["maintainability"],
            business_impact=impact_data["business"],
            technical_debt_hours=impact_data["debt_hours"],
            affected_components=[str(features.file_path.name)],
            risk_factors=pattern.evidence[:3]  # Top 3 evidencias como factores de riesgo
        )
    
    def _get_pattern_category(self, pattern_type: AntipatternType) -> AntipatternCategory:
        """Obtener categoría de un tipo de patrón."""
        
        category_mapping = {
            AntipatternType.GOD_OBJECT: AntipatternCategory.ARCHITECTURAL,
            AntipatternType.BIG_BALL_OF_MUD: AntipatternCategory.ARCHITECTURAL,
            AntipatternType.SPAGHETTI_CODE: AntipatternCategory.ARCHITECTURAL,
            AntipatternType.LAVA_FLOW: AntipatternCategory.ARCHITECTURAL,
            
            AntipatternType.SQL_INJECTION: AntipatternCategory.SECURITY,
            AntipatternType.HARDCODED_SECRETS: AntipatternCategory.SECURITY,
            AntipatternType.XSS_VULNERABILITY: AntipatternCategory.SECURITY,
            AntipatternType.WEAK_CRYPTOGRAPHY: AntipatternCategory.SECURITY,
            
            AntipatternType.N_PLUS_ONE_QUERY: AntipatternCategory.PERFORMANCE,
            AntipatternType.MEMORY_LEAK: AntipatternCategory.PERFORMANCE,
            AntipatternType.INEFFICIENT_ALGORITHM: AntipatternCategory.PERFORMANCE,
            AntipatternType.STRING_CONCATENATION_IN_LOOP: AntipatternCategory.PERFORMANCE,
            
            AntipatternType.LARGE_CLASS: AntipatternCategory.DESIGN,
            AntipatternType.LONG_METHOD: AntipatternCategory.DESIGN,
            AntipatternType.FEATURE_ENVY: AntipatternCategory.DESIGN,
            AntipatternType.DATA_CLUMPS: AntipatternCategory.DESIGN,
            AntipatternType.PRIMITIVE_OBSESSION: AntipatternCategory.DESIGN,
        }
        
        return category_mapping.get(pattern_type, AntipatternCategory.DESIGN)
    
    def _filter_by_category(
        self, 
        antipatterns: List[DetectedAntipattern],
        category: AntipatternCategory
    ) -> List[DetectedAntipattern]:
        """Filtrar antipatrones por categoría."""
        
        return [ap for ap in antipatterns if ap.category == category]
    
    def _calculate_quality_score(self, result: AntipatternDetectionResult, features) -> float:
        """Calcular score de calidad del archivo."""
        
        base_score = 1.0
        
        # Penalizar por antipatrones encontrados
        for antipattern in result.detected_antipatterns:
            penalty = 0.1
            
            # Penalidad mayor para antipatrones críticos
            if antipattern.severity == AntipatternSeverity.CRITICAL:
                penalty = 0.3
            elif antipattern.severity == AntipatternSeverity.HIGH:
                penalty = 0.2
            elif antipattern.severity == AntipatternSeverity.MEDIUM:
                penalty = 0.1
            else:
                penalty = 0.05
            
            # Ajustar penalidad por confianza
            adjusted_penalty = penalty * antipattern.confidence
            base_score -= adjusted_penalty
        
        # Bonificar por buenas prácticas (si no hay antipatrones)
        if len(result.detected_antipatterns) == 0:
            base_score = min(1.0, base_score + 0.1)
        
        return max(0.0, min(1.0, base_score))
    
    async def _detect_project_level_patterns(
        self, 
        file_results: List[AntipatternDetectionResult],
        project_asts: List[UnifiedAST]
    ) -> List[DetectedAntipattern]:
        """Detectar antipatrones a nivel de proyecto."""
        
        project_patterns = []
        
        # Contar antipatrones por tipo
        pattern_counts = {}
        for result in file_results:
            for antipattern in result.detected_antipatterns:
                pattern_type = antipattern.pattern_type
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        # Detectar patrones sistémicos
        total_files = len(file_results)
        
        for pattern_type, count in pattern_counts.items():
            prevalence = count / total_files if total_files > 0 else 0
            
            # Si un antipatrón aparece en >50% de archivos, es sistémico
            if prevalence > 0.5 and count > 2:
                systemic_pattern = DetectedAntipattern(
                    pattern_type=pattern_type,
                    category=self._get_pattern_category(pattern_type),
                    severity=AntipatternSeverity.HIGH,
                    confidence=min(0.9, prevalence * 1.2),
                    locations=[],  # A nivel de proyecto
                    description=f"Systemic antipattern: {pattern_type.value} found in {count}/{total_files} files",
                    detected_at=datetime.now(),
                    evidence=[f"Found in {count} out of {total_files} files ({prevalence:.1%})"]
                )
                
                project_patterns.append(systemic_pattern)
        
        return project_patterns
    
    async def _analyze_project_architecture(
        self, 
        file_results: List[AntipatternDetectionResult],
        project_asts: List[UnifiedAST]
    ) -> Dict[str, Any]:
        """Analizar arquitectura del proyecto."""
        
        # Análisis básico de arquitectura
        analysis = {
            "total_files": len(file_results),
            "languages_used": list(set(result.language for result in file_results)),
            "architectural_issues_count": sum(
                len(result.architectural_issues) for result in file_results
            ),
            "design_issues_count": sum(
                len(result.design_issues) for result in file_results
            ),
            "average_quality_score": 0.0,
            "complexity_distribution": {},
            "antipattern_density": 0.0
        }
        
        # Calcular score promedio
        if file_results:
            total_quality = sum(result.quality_score for result in file_results)
            analysis["average_quality_score"] = total_quality / len(file_results)
            
            # Calcular densidad de antipatrones
            total_antipatterns = sum(
                len(result.detected_antipatterns) for result in file_results
            )
            analysis["antipattern_density"] = total_antipatterns / len(file_results)
        
        return analysis
    
    async def _detect_cross_file_patterns(
        self, 
        file_results: List[AntipatternDetectionResult],
        project_asts: List[UnifiedAST]
    ) -> List[DetectedAntipattern]:
        """Detectar antipatrones que cruzan múltiples archivos."""
        
        cross_file_patterns = []
        
        # Ejemplo: Detectar duplicación de lógica similar entre archivos
        # (Implementación simplificada)
        
        security_files = []
        for result in file_results:
            if any(ap.category == AntipatternCategory.SECURITY for ap in result.detected_antipatterns):
                security_files.append(result)
        
        # Si hay vulnerabilidades de seguridad en múltiples archivos
        if len(security_files) > 2:
            cross_pattern = DetectedAntipattern(
                pattern_type=AntipatternType.CUSTOM,
                category=AntipatternCategory.SECURITY,
                severity=AntipatternSeverity.HIGH,
                confidence=0.8,
                locations=[],
                description=f"Cross-file security issues detected in {len(security_files)} files",
                detected_at=datetime.now(),
                evidence=[f"Security antipatterns found across {len(security_files)} files"]
            )
            cross_file_patterns.append(cross_pattern)
        
        return cross_file_patterns
    
    async def _identify_antipattern_hotspots(
        self, 
        file_results: List[AntipatternDetectionResult]
    ) -> List[AntipatternHotspot]:
        """Identificar hotspots de antipatrones."""
        
        hotspots = []
        
        for result in file_results:
            antipattern_count = len(result.detected_antipatterns)
            
            if antipattern_count > 3:  # Threshold para considerarlo hotspot
                # Calcular severity score
                severity_score = sum(
                    self._severity_to_number(ap.severity) * ap.confidence 
                    for ap in result.detected_antipatterns
                )
                
                pattern_types = [ap.pattern_type for ap in result.detected_antipatterns]
                
                hotspot = AntipatternHotspot(
                    location=SourcePosition(line=1, column=1),  # Representando todo el archivo
                    antipattern_count=antipattern_count,
                    severity_score=severity_score,
                    pattern_types=pattern_types,
                    description=f"High concentration of antipatterns in {result.file_path.name}"
                )
                
                hotspots.append(hotspot)
        
        # Ordenar por severity score descendente
        hotspots.sort(key=lambda h: h.severity_score, reverse=True)
        
        return hotspots[:10]  # Top 10 hotspots
    
    async def _analyze_antipattern_trends(
        self, 
        file_results: List[AntipatternDetectionResult]
    ) -> List[AntipatternTrend]:
        """Analizar tendencias de antipatrones."""
        
        trends = []
        
        # Contar frecuencia por tipo
        pattern_frequencies = {}
        pattern_locations = {}
        
        for result in file_results:
            for antipattern in result.detected_antipatterns:
                pattern_type = antipattern.pattern_type
                pattern_frequencies[pattern_type] = pattern_frequencies.get(pattern_type, 0) + 1
                
                if pattern_type not in pattern_locations:
                    pattern_locations[pattern_type] = []
                pattern_locations[pattern_type].extend(antipattern.locations)
        
        # Crear trends para patrones frecuentes
        for pattern_type, frequency in pattern_frequencies.items():
            if frequency > 1:  # Solo patrones que aparecen múltiples veces
                trend = AntipatternTrend(
                    pattern_type=pattern_type,
                    frequency=frequency,
                    trend_direction="stable",  # Simplificado - en implementación real se calcularía
                    locations=pattern_locations[pattern_type][:5]  # Top 5 ubicaciones
                )
                trends.append(trend)
        
        return trends
    
    async def _calculate_remediation_priorities(
        self, 
        file_results: List[AntipatternDetectionResult],
        project_asts: List[UnifiedAST]
    ) -> List[RemediationPriority]:
        """Calcular prioridades de remediación."""
        
        priorities = []
        
        # Recopilar todos los antipatrones
        all_antipatterns = []
        for result in file_results:
            all_antipatterns.extend(result.detected_antipatterns)
        
        # Calcular prioridad para cada antipatrón
        for antipattern in all_antipatterns:
            # Calcular score de prioridad
            priority_score = self._calculate_priority_score(antipattern)
            
            # Estimar esfuerzo
            effort_estimate = self._estimate_fix_effort(antipattern)
            
            # Analizar impacto
            business_impact = self._analyze_business_impact(antipattern)
            technical_impact = self._analyze_technical_impact(antipattern)
            
            priority = RemediationPriority(
                antipattern=antipattern,
                priority_score=priority_score,
                business_impact=business_impact,
                technical_impact=technical_impact,
                effort_estimate=effort_estimate,
                dependencies=[]  # Simplificado
            )
            
            priorities.append(priority)
        
        # Ordenar por prioridad descendente
        priorities.sort(key=lambda p: p.priority_score, reverse=True)
        
        return priorities[:20]  # Top 20 prioridades
    
    def _calculate_priority_score(self, antipattern: DetectedAntipattern) -> float:
        """Calcular score de prioridad."""
        
        severity_weight = self._severity_to_number(antipattern.severity) * 0.4
        confidence_weight = antipattern.confidence * 0.3
        
        # Peso por categoría (seguridad es más prioritaria)
        category_weights = {
            AntipatternCategory.SECURITY: 1.0,
            AntipatternCategory.PERFORMANCE: 0.8,
            AntipatternCategory.ARCHITECTURAL: 0.7,
            AntipatternCategory.DESIGN: 0.6
        }
        category_weight = category_weights.get(antipattern.category, 0.5) * 0.3
        
        return severity_weight + confidence_weight + category_weight
    
    def _estimate_fix_effort(self, antipattern: DetectedAntipattern) -> float:
        """Estimar esfuerzo de corrección en horas."""
        
        effort_mapping = {
            AntipatternType.SQL_INJECTION: 8.0,
            AntipatternType.HARDCODED_SECRETS: 4.0,
            AntipatternType.GOD_OBJECT: 40.0,
            AntipatternType.N_PLUS_ONE_QUERY: 12.0,
            AntipatternType.LARGE_CLASS: 20.0,
            AntipatternType.LONG_METHOD: 6.0,
        }
        
        base_effort = effort_mapping.get(antipattern.pattern_type, 10.0)
        
        # Ajustar por confianza (mayor confianza = estimación más precisa)
        confidence_factor = 0.8 + (antipattern.confidence * 0.4)
        
        return base_effort * confidence_factor
    
    def _analyze_business_impact(self, antipattern: DetectedAntipattern) -> str:
        """Analizar impacto en el negocio."""
        
        impact_mapping = {
            AntipatternType.SQL_INJECTION: "Critical security risk affecting customer data",
            AntipatternType.N_PLUS_ONE_QUERY: "Performance issues affecting user experience",
            AntipatternType.GOD_OBJECT: "Slows down feature development significantly",
            AntipatternType.MEMORY_LEAK: "System stability issues affecting availability"
        }
        
        return impact_mapping.get(
            antipattern.pattern_type,
            "Moderate impact on code quality and maintainability"
        )
    
    def _analyze_technical_impact(self, antipattern: DetectedAntipattern) -> str:
        """Analizar impacto técnico."""
        
        impact_mapping = {
            AntipatternType.GOD_OBJECT: "High coupling makes testing and changes difficult",
            AntipatternType.SPAGHETTI_CODE: "Code becomes unmaintainable and error-prone",
            AntipatternType.LARGE_CLASS: "Violates SRP, difficult to understand and modify"
        }
        
        return impact_mapping.get(
            antipattern.pattern_type,
            "General negative impact on code architecture"
        )
    
    def _severity_to_number(self, severity: AntipatternSeverity) -> float:
        """Convertir severidad a número."""
        
        mapping = {
            AntipatternSeverity.CRITICAL: 4.0,
            AntipatternSeverity.HIGH: 3.0,
            AntipatternSeverity.MEDIUM: 2.0,
            AntipatternSeverity.LOW: 1.0
        }
        
        return mapping.get(severity, 2.0)
    
    def _calculate_project_quality_score(
        self, 
        file_results: List[AntipatternDetectionResult]
    ) -> float:
        """Calcular score de calidad del proyecto."""
        
        if not file_results:
            return 0.0
        
        total_score = sum(result.quality_score for result in file_results)
        return total_score / len(file_results)
    
    def _generate_project_recommendations(
        self, 
        file_results: List[AntipatternDetectionResult],
        hotspots: List[AntipatternHotspot],
        trends: List[AntipatternTrend]
    ) -> List[str]:
        """Generar recomendaciones para el proyecto."""
        
        recommendations = []
        
        # Recomendaciones basadas en hotspots
        if len(hotspots) > 3:
            recommendations.append(
                f"Focus on {len(hotspots)} identified hotspots with high antipattern concentration"
            )
        
        # Recomendaciones basadas en tendencias
        if trends:
            most_frequent = max(trends, key=lambda t: t.frequency)
            recommendations.append(
                f"Address {most_frequent.pattern_type.value} pattern which appears {most_frequent.frequency} times"
            )
        
        # Recomendaciones basadas en categorías
        security_issues = sum(
            len(result.security_issues) for result in file_results
        )
        if security_issues > 0:
            recommendations.append(f"Critical: Address {security_issues} security vulnerabilities immediately")
        
        performance_issues = sum(
            len(result.performance_issues) for result in file_results
        )
        if performance_issues > 5:
            recommendations.append(f"Review {performance_issues} performance-related antipatterns")
        
        # Recomendación general
        total_antipatterns = sum(
            len(result.detected_antipatterns) for result in file_results
        )
        if total_antipatterns > 20:
            recommendations.append("Consider refactoring strategy for systematic improvement")
        
        return recommendations
    
    def _update_detection_stats(self, result: AntipatternDetectionResult, detection_time_ms: int):
        """Actualizar estadísticas de detección."""
        
        self.detection_stats["total_analyses"] += 1
        self.detection_stats["successful_analyses"] += 1
        
        # Actualizar tiempo promedio
        current_avg = self.detection_stats["avg_detection_time_ms"]
        total_analyses = self.detection_stats["total_analyses"]
        self.detection_stats["avg_detection_time_ms"] = (
            (current_avg * (total_analyses - 1) + detection_time_ms) / total_analyses
        )
        
        # Actualizar contadores por categoría y tipo
        for antipattern in result.detected_antipatterns:
            self.detection_stats["patterns_by_category"][antipattern.category] += 1
            self.detection_stats["patterns_by_type"][antipattern.pattern_type] += 1
        
        self.detection_stats["total_patterns_detected"] += len(result.detected_antipatterns)
    
    async def get_detection_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de detección."""
        
        return self.detection_stats.copy()
    
    async def reset_detection_stats(self):
        """Resetear estadísticas de detección."""
        
        self.detection_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "total_patterns_detected": 0,
            "avg_detection_time_ms": 0.0,
            "patterns_by_category": {category: 0 for category in AntipatternCategory},
            "patterns_by_type": {pattern_type: 0 for pattern_type in AntipatternType}
        }
