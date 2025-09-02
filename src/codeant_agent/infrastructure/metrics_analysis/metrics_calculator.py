"""
Calculadora principal de métricas de código (Orquestador).

Este módulo implementa el calculador principal que coordina todos los
analizadores de métricas para proporcionar análisis completo de calidad.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import time
from pathlib import Path

from ...domain.entities.code_metrics import (
    CodeMetrics, ProjectMetrics, MetricsConfig, ComplexityThresholds,
    TechnicalDebtEstimate, QualityDistribution, ComplexityHotspot
)
from ...domain.entities.parse_result import ParseResult
from ...domain.value_objects.programming_language import ProgrammingLanguage

from .complexity_analyzer import ComplexityAnalyzer, ComplexityAnalysisResult
from .halstead_calculator import HalsteadCalculator, HalsteadAnalysisResult
from .size_analyzer import SizeAnalyzer, SizeAnalysisResult
from .cohesion_analyzer import CohesionAnalyzer, CohesionAnalysisResult
from .coupling_analyzer import CouplingAnalyzer, CouplingAnalysisResult
from .quality_analyzer import QualityAnalyzer, QualityAnalysisResult
from .technical_debt_estimator import TechnicalDebtEstimator, TechnicalDebtAnalysisResult
from .quality_gates import QualityGateChecker, QualityGateEvaluation

logger = logging.getLogger(__name__)


@dataclass
class MetricsAnalysisSession:
    """Sesión de análisis de métricas."""
    session_id: str
    start_time: float
    files_analyzed: int = 0
    total_analysis_time_ms: int = 0
    errors_encountered: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Añade error a la sesión."""
        self.errors_encountered.append(error)
    
    def get_average_time_per_file(self) -> float:
        """Obtiene tiempo promedio por archivo."""
        return self.total_analysis_time_ms / self.files_analyzed if self.files_analyzed > 0 else 0.0


@dataclass
class MetricsCalculationResult:
    """Resultado completo del cálculo de métricas."""
    code_metrics: CodeMetrics
    calculation_success: bool = True
    components_calculated: Dict[str, bool] = field(default_factory=dict)
    component_times: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def add_component_result(self, component: str, success: bool, time_ms: int) -> None:
        """Añade resultado de componente."""
        self.components_calculated[component] = success
        self.component_times[component] = time_ms
    
    def add_warning(self, warning: str) -> None:
        """Añade warning."""
        self.warnings.append(warning)


class MetricsAggregator:
    """Agregador de métricas de proyecto."""
    
    def aggregate_project_metrics(self, file_metrics: List[CodeMetrics]) -> ProjectMetrics:
        """
        Agrega métricas de múltiples archivos en métricas de proyecto.
        
        Args:
            file_metrics: Lista de métricas de archivos individuales
            
        Returns:
            ProjectMetrics agregadas
        """
        if not file_metrics:
            return ProjectMetrics()
        
        # Agregar métricas de complejidad
        aggregated_complexity = self._aggregate_complexity_metrics(file_metrics)
        
        # Agregar métricas de Halstead
        aggregated_halstead = self._aggregate_halstead_metrics(file_metrics)
        
        # Agregar métricas de tamaño
        aggregated_size = self._aggregate_size_metrics(file_metrics)
        
        # Agregar métricas de cohesión
        project_cohesion = self._aggregate_cohesion_metrics(file_metrics)
        
        # Agregar métricas de acoplamiento
        project_coupling = self._aggregate_coupling_metrics(file_metrics)
        
        # Agregar métricas de calidad
        project_quality = self._aggregate_quality_metrics(file_metrics)
        
        # Identificar hotspots del proyecto
        hotspots = self._identify_project_hotspots(file_metrics)
        
        # Calcular distribución de calidad
        quality_distribution = self._calculate_quality_distribution(file_metrics)
        
        # Estimar deuda técnica del proyecto
        technical_debt = self._aggregate_technical_debt(file_metrics)
        
        # Calcular índice de mantenibilidad del proyecto
        project_maintainability = self._calculate_project_maintainability(file_metrics)
        
        return ProjectMetrics(
            project_path=file_metrics[0].file_path.parent if file_metrics else None,
            file_metrics=file_metrics,
            aggregated_complexity=aggregated_complexity,
            aggregated_halstead=aggregated_halstead,
            aggregated_size=aggregated_size,
            project_cohesion=project_cohesion,
            project_coupling=project_coupling,
            project_quality=project_quality,
            hotspots=hotspots,
            quality_distribution=quality_distribution,
            technical_debt_estimate=technical_debt,
            maintainability_index=project_maintainability
        )
    
    def _aggregate_complexity_metrics(self, file_metrics: List[CodeMetrics]) -> 'ComplexityMetrics':
        """Agrega métricas de complejidad."""
        from ...domain.entities.code_metrics import ComplexityMetrics
        
        total_cyclomatic = sum(m.complexity_metrics.cyclomatic_complexity for m in file_metrics)
        total_cognitive = sum(m.complexity_metrics.cognitive_complexity for m in file_metrics)
        max_nesting = max(m.complexity_metrics.max_nesting_depth for m in file_metrics)
        avg_nesting = sum(m.complexity_metrics.average_nesting_depth for m in file_metrics) / len(file_metrics)
        
        return ComplexityMetrics(
            cyclomatic_complexity=total_cyclomatic // len(file_metrics),
            cognitive_complexity=total_cognitive // len(file_metrics),
            max_nesting_depth=max_nesting,
            average_nesting_depth=avg_nesting,
            complexity_density=sum(m.complexity_metrics.complexity_density for m in file_metrics) / len(file_metrics)
        )
    
    def _aggregate_halstead_metrics(self, file_metrics: List[CodeMetrics]) -> 'HalsteadMetrics':
        """Agrega métricas de Halstead."""
        from ...domain.entities.code_metrics import HalsteadMetrics
        
        total_operators = sum(m.halstead_metrics.total_operators for m in file_metrics)
        total_operands = sum(m.halstead_metrics.total_operands for m in file_metrics)
        total_volume = sum(m.halstead_metrics.volume for m in file_metrics)
        avg_difficulty = sum(m.halstead_metrics.difficulty for m in file_metrics) / len(file_metrics)
        total_effort = sum(m.halstead_metrics.effort for m in file_metrics)
        
        aggregated = HalsteadMetrics(
            total_operators=total_operators,
            total_operands=total_operands,
            volume=total_volume,
            difficulty=avg_difficulty,
            effort=total_effort
        )
        
        # Recalcular métricas derivadas a nivel de proyecto
        aggregated.calculate_derived_metrics()
        
        return aggregated
    
    def _aggregate_size_metrics(self, file_metrics: List[CodeMetrics]) -> 'SizeMetrics':
        """Agrega métricas de tamaño."""
        from ...domain.entities.code_metrics import SizeMetrics
        
        aggregated = SizeMetrics(
            total_lines=sum(m.size_metrics.total_lines for m in file_metrics),
            logical_lines_of_code=sum(m.size_metrics.logical_lines_of_code for m in file_metrics),
            comment_lines=sum(m.size_metrics.comment_lines for m in file_metrics),
            blank_lines=sum(m.size_metrics.blank_lines for m in file_metrics),
            function_count=sum(m.size_metrics.function_count for m in file_metrics),
            class_count=sum(m.size_metrics.class_count for m in file_metrics),
            method_count=sum(m.size_metrics.method_count for m in file_metrics),
            max_function_length=max(m.size_metrics.max_function_length for m in file_metrics),
            max_class_length=max(m.size_metrics.max_class_length for m in file_metrics)
        )
        
        aggregated.calculate_derived_metrics()
        return aggregated
    
    def _aggregate_cohesion_metrics(self, file_metrics: List[CodeMetrics]) -> 'CohesionMetrics':
        """Agrega métricas de cohesión."""
        from ...domain.entities.code_metrics import CohesionMetrics
        
        files_with_classes = [m for m in file_metrics if m.cohesion_metrics.class_count > 0]
        
        if not files_with_classes:
            return CohesionMetrics()
        
        avg_lcom = sum(m.cohesion_metrics.average_lcom for m in files_with_classes) / len(files_with_classes)
        avg_tcc = sum(m.cohesion_metrics.average_tcc for m in files_with_classes) / len(files_with_classes)
        avg_lcc = sum(m.cohesion_metrics.average_lcc for m in files_with_classes) / len(files_with_classes)
        total_classes = sum(m.cohesion_metrics.class_count for m in file_metrics)
        
        return CohesionMetrics(
            average_lcom=avg_lcom,
            average_tcc=avg_tcc,
            average_lcc=avg_lcc,
            class_count=total_classes
        )
    
    def _aggregate_coupling_metrics(self, file_metrics: List[CodeMetrics]) -> 'CouplingMetrics':
        """Agrega métricas de acoplamiento."""
        from ...domain.entities.code_metrics import CouplingMetrics
        
        files_with_classes = [m for m in file_metrics if m.coupling_metrics.average_cbo > 0]
        
        if not files_with_classes:
            return CouplingMetrics()
        
        avg_cbo = sum(m.coupling_metrics.average_cbo for m in files_with_classes) / len(files_with_classes)
        avg_rfc = sum(m.coupling_metrics.average_rfc for m in files_with_classes) / len(files_with_classes)
        avg_dit = sum(m.coupling_metrics.average_dit for m in files_with_classes) / len(files_with_classes)
        avg_noc = sum(m.coupling_metrics.average_noc for m in files_with_classes) / len(files_with_classes)
        
        return CouplingMetrics(
            average_cbo=avg_cbo,
            average_rfc=avg_rfc,
            average_dit=avg_dit,
            average_noc=avg_noc,
            total_dependencies=sum(m.coupling_metrics.total_dependencies for m in file_metrics),
            circular_dependencies=sum(m.coupling_metrics.circular_dependencies for m in file_metrics),
            max_inheritance_depth=max(m.coupling_metrics.max_inheritance_depth for m in file_metrics)
        )
    
    def _aggregate_quality_metrics(self, file_metrics: List[CodeMetrics]) -> 'QualityMetrics':
        """Agrega métricas de calidad."""
        from ...domain.entities.code_metrics import QualityMetrics
        
        avg_maintainability = sum(m.quality_metrics.maintainability_index for m in file_metrics) / len(file_metrics)
        avg_testability = sum(m.quality_metrics.testability_score for m in file_metrics) / len(file_metrics)
        avg_readability = sum(m.quality_metrics.readability_score for m in file_metrics) / len(file_metrics)
        avg_reliability = sum(m.quality_metrics.reliability_score for m in file_metrics) / len(file_metrics)
        
        return QualityMetrics(
            maintainability_index=avg_maintainability,
            testability_score=avg_testability,
            readability_score=avg_readability,
            reliability_score=avg_reliability,
            code_smells_count=sum(len(m.get_all_smells()) for m in file_metrics)
        )
    
    def _identify_project_hotspots(self, file_metrics: List[CodeMetrics]) -> List[ComplexityHotspot]:
        """Identifica hotspots de complejidad del proyecto."""
        all_hotspots = []
        
        # Recopilar hotspots de todos los archivos
        for metrics in file_metrics:
            # Crear hotspots de funciones complejas
            for func_metric in metrics.function_metrics:
                if func_metric.cyclomatic_complexity > 15:
                    hotspot = ComplexityHotspot(
                        location=func_metric.location,
                        hotspot_type="function",
                        name=f"{metrics.file_path.name}::{func_metric.name}",
                        cyclomatic_complexity=func_metric.cyclomatic_complexity,
                        cognitive_complexity=func_metric.cognitive_complexity,
                        lines_of_code=func_metric.lines_of_code,
                        severity=self._determine_hotspot_severity(func_metric.cyclomatic_complexity),
                        impact_score=self._calculate_impact_score(func_metric),
                        suggested_actions=self._generate_hotspot_suggestions(func_metric)
                    )
                    all_hotspots.append(hotspot)
            
            # Crear hotspots de clases complejas
            for class_metric in metrics.class_metrics:
                if class_metric.weighted_methods_per_class > 30:
                    hotspot = ComplexityHotspot(
                        location=class_metric.location,
                        hotspot_type="class",
                        name=f"{metrics.file_path.name}::{class_metric.name}",
                        cyclomatic_complexity=class_metric.weighted_methods_per_class,
                        cognitive_complexity=0,  # No disponible a nivel de clase
                        lines_of_code=class_metric.lines_of_code,
                        severity=self._determine_class_hotspot_severity(class_metric.weighted_methods_per_class),
                        impact_score=self._calculate_class_impact_score(class_metric),
                        suggested_actions=self._generate_class_hotspot_suggestions(class_metric)
                    )
                    all_hotspots.append(hotspot)
        
        # Ordenar por impact score y retornar top hotspots
        all_hotspots.sort(key=lambda h: h.impact_score, reverse=True)
        return all_hotspots[:20]  # Top 20
    
    def _calculate_quality_distribution(self, file_metrics: List[CodeMetrics]) -> QualityDistribution:
        """Calcula distribución de calidad del proyecto."""
        distribution = QualityDistribution()
        
        for metrics in file_metrics:
            mi = metrics.quality_metrics.maintainability_index
            
            if mi >= 85:
                distribution.excellent_files += 1
            elif mi >= 70:
                distribution.good_files += 1
            elif mi >= 50:
                distribution.average_files += 1
            elif mi >= 25:
                distribution.poor_files += 1
            else:
                distribution.very_poor_files += 1
        
        return distribution
    
    def _aggregate_technical_debt(self, file_metrics: List[CodeMetrics]) -> TechnicalDebtEstimate:
        """Agrega deuda técnica del proyecto."""
        total_minutes = sum(m.technical_debt.total_minutes for m in file_metrics)
        all_smells = []
        
        for metrics in file_metrics:
            all_smells.extend(metrics.technical_debt.code_smells)
        
        # Crear estimación agregada
        estimate = TechnicalDebtEstimate(
            total_minutes=total_minutes,
            code_smells=all_smells
        )
        
        # Calcular métricas derivadas
        estimate.calculate_derived_metrics()
        
        # Calcular debt ratio a nivel de proyecto
        total_loc = sum(m.size_metrics.logical_lines_of_code for m in file_metrics)
        if total_loc > 0:
            estimate.debt_ratio = total_minutes / (total_loc * 0.5)
        
        return estimate
    
    def _calculate_project_maintainability(self, file_metrics: List[CodeMetrics]) -> float:
        """Calcula índice de mantenibilidad del proyecto."""
        if not file_metrics:
            return 0.0
        
        # Promedio ponderado por tamaño de archivo
        total_weighted_mi = 0.0
        total_weight = 0.0
        
        for metrics in file_metrics:
            weight = metrics.size_metrics.logical_lines_of_code
            total_weighted_mi += metrics.quality_metrics.maintainability_index * weight
            total_weight += weight
        
        return total_weighted_mi / total_weight if total_weight > 0 else 0.0
    
    # Métodos auxiliares para hotspots
    
    def _determine_hotspot_severity(self, complexity: int) -> 'CodeSmellSeverity':
        """Determina severidad de hotspot por complejidad."""
        from ...domain.entities.code_metrics import CodeSmellSeverity
        
        if complexity > 50:
            return CodeSmellSeverity.CRITICAL
        elif complexity > 30:
            return CodeSmellSeverity.HIGH
        elif complexity > 20:
            return CodeSmellSeverity.MEDIUM
        else:
            return CodeSmellSeverity.LOW
    
    def _determine_class_hotspot_severity(self, wmc: int) -> 'CodeSmellSeverity':
        """Determina severidad de hotspot de clase."""
        from ...domain.entities.code_metrics import CodeSmellSeverity
        
        if wmc > 100:
            return CodeSmellSeverity.CRITICAL
        elif wmc > 50:
            return CodeSmellSeverity.HIGH
        else:
            return CodeSmellSeverity.MEDIUM
    
    def _calculate_impact_score(self, func_metric) -> float:
        """Calcula score de impacto de función."""
        complexity_factor = min(100.0, func_metric.cyclomatic_complexity * 2.0)
        size_factor = min(50.0, func_metric.lines_of_code * 0.5)
        return complexity_factor + size_factor
    
    def _calculate_class_impact_score(self, class_metric) -> float:
        """Calcula score de impacto de clase."""
        complexity_factor = min(100.0, class_metric.weighted_methods_per_class * 1.5)
        size_factor = min(50.0, class_metric.lines_of_code * 0.1)
        return complexity_factor + size_factor
    
    def _generate_hotspot_suggestions(self, func_metric) -> List[str]:
        """Genera sugerencias para hotspot de función."""
        suggestions = []
        
        if func_metric.cyclomatic_complexity > 20:
            suggestions.extend([
                "Break down into smaller functions",
                "Extract complex conditional logic",
                "Use early returns to reduce nesting"
            ])
        
        if func_metric.lines_of_code > 50:
            suggestions.append("Extract sub-functionality into helper methods")
        
        if func_metric.parameter_count > 5:
            suggestions.append("Reduce number of parameters using parameter objects")
        
        return suggestions
    
    def _generate_class_hotspot_suggestions(self, class_metric) -> List[str]:
        """Genera sugerencias para hotspot de clase."""
        suggestions = []
        
        if class_metric.lines_of_code > 500:
            suggestions.extend([
                "Split class into smaller, focused classes",
                "Extract related functionality into separate classes"
            ])
        
        if class_metric.method_count > 20:
            suggestions.append("Reduce number of methods per class")
        
        return suggestions


class MetricsCalculator:
    """Calculadora principal de métricas que orquesta todos los analizadores."""
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Inicializa el calculador principal.
        
        Args:
            config: Configuración del calculador
        """
        self.config = config or MetricsConfig()
        
        # Inicializar analizadores especializados
        self.complexity_analyzer = ComplexityAnalyzer()
        self.halstead_calculator = HalsteadCalculator()
        self.size_analyzer = SizeAnalyzer()
        self.cohesion_analyzer = CohesionAnalyzer()
        self.coupling_analyzer = CouplingAnalyzer()
        self.quality_analyzer = QualityAnalyzer()
        self.technical_debt_estimator = TechnicalDebtEstimator()
        self.quality_gate_checker = QualityGateChecker()
        
        # Inicializar agregador
        self.metrics_aggregator = MetricsAggregator()
    
    async def calculate_metrics(self, parse_result: ParseResult) -> MetricsCalculationResult:
        """
        Calcula todas las métricas para un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            
        Returns:
            MetricsCalculationResult completo
        """
        start_time = time.time()
        
        try:
            logger.info(f"Iniciando cálculo de métricas para {parse_result.file_path}")
            
            # Inicializar métricas
            code_metrics = CodeMetrics(
                file_path=parse_result.file_path,
                language=parse_result.language
            )
            
            result = MetricsCalculationResult(code_metrics=code_metrics)
            
            # 1. Calcular métricas de complejidad
            if self.config.enable_cyclomatic_complexity or self.config.enable_cognitive_complexity:
                comp_start = time.time()
                try:
                    complexity_result = await self.complexity_analyzer.calculate_complexity(parse_result)
                    code_metrics.complexity_metrics = complexity_result.complexity_metrics
                    code_metrics.complexity_distribution = complexity_result.complexity_distribution
                    
                    result.add_component_result("complexity", True, int((time.time() - comp_start) * 1000))
                    logger.debug(f"Complejidad calculada: CC={complexity_result.complexity_metrics.cyclomatic_complexity}")
                except Exception as e:
                    logger.warning(f"Error calculando complejidad: {e}")
                    result.add_component_result("complexity", False, int((time.time() - comp_start) * 1000))
                    result.add_warning(f"Complexity calculation failed: {e}")
            
            # 2. Calcular métricas de Halstead
            if self.config.enable_halstead_metrics:
                halstead_start = time.time()
                try:
                    halstead_result = await self.halstead_calculator.calculate_halstead_metrics(parse_result)
                    code_metrics.halstead_metrics = halstead_result.metrics
                    
                    result.add_component_result("halstead", True, int((time.time() - halstead_start) * 1000))
                    logger.debug(f"Halstead calculado: V={halstead_result.metrics.volume:.1f}")
                except Exception as e:
                    logger.warning(f"Error calculando Halstead: {e}")
                    result.add_component_result("halstead", False, int((time.time() - halstead_start) * 1000))
                    result.add_warning(f"Halstead calculation failed: {e}")
            
            # 3. Calcular métricas de tamaño
            if self.config.enable_size_metrics:
                size_start = time.time()
                try:
                    size_result = await self.size_analyzer.calculate_size_metrics(parse_result)
                    code_metrics.size_metrics = size_result.size_metrics
                    
                    result.add_component_result("size", True, int((time.time() - size_start) * 1000))
                    logger.debug(f"Tamaño calculado: LOC={size_result.size_metrics.total_lines}")
                except Exception as e:
                    logger.warning(f"Error calculando tamaño: {e}")
                    result.add_component_result("size", False, int((time.time() - size_start) * 1000))
                    result.add_warning(f"Size calculation failed: {e}")
            
            # 4. Calcular métricas de cohesión
            if self.config.enable_cohesion_metrics:
                cohesion_start = time.time()
                try:
                    cohesion_result = await self.cohesion_analyzer.calculate_cohesion_metrics(parse_result)
                    code_metrics.cohesion_metrics = cohesion_result.cohesion_metrics
                    
                    result.add_component_result("cohesion", True, int((time.time() - cohesion_start) * 1000))
                    logger.debug(f"Cohesión calculada: LCOM avg={cohesion_result.cohesion_metrics.average_lcom:.3f}")
                except Exception as e:
                    logger.warning(f"Error calculando cohesión: {e}")
                    result.add_component_result("cohesion", False, int((time.time() - cohesion_start) * 1000))
                    result.add_warning(f"Cohesion calculation failed: {e}")
            
            # 5. Calcular métricas de acoplamiento
            if self.config.enable_coupling_metrics:
                coupling_start = time.time()
                try:
                    coupling_result = await self.coupling_analyzer.calculate_coupling_metrics(parse_result)
                    code_metrics.coupling_metrics = coupling_result.coupling_metrics
                    
                    result.add_component_result("coupling", True, int((time.time() - coupling_start) * 1000))
                    logger.debug(f"Acoplamiento calculado: CBO avg={coupling_result.coupling_metrics.average_cbo:.1f}")
                except Exception as e:
                    logger.warning(f"Error calculando acoplamiento: {e}")
                    result.add_component_result("coupling", False, int((time.time() - coupling_start) * 1000))
                    result.add_warning(f"Coupling calculation failed: {e}")
            
            # 6. Calcular métricas de calidad
            if self.config.enable_quality_metrics:
                quality_start = time.time()
                try:
                    quality_result = await self.quality_analyzer.calculate_quality_metrics(code_metrics, parse_result)
                    code_metrics.quality_metrics = quality_result.quality_metrics
                    
                    result.add_component_result("quality", True, int((time.time() - quality_start) * 1000))
                    logger.debug(f"Calidad calculada: MI={quality_result.quality_metrics.maintainability_index:.1f}")
                except Exception as e:
                    logger.warning(f"Error calculando calidad: {e}")
                    result.add_component_result("quality", False, int((time.time() - quality_start) * 1000))
                    result.add_warning(f"Quality calculation failed: {e}")
            
            # 7. Estimar deuda técnica
            if self.config.calculate_technical_debt:
                debt_start = time.time()
                try:
                    debt_result = await self.technical_debt_estimator.estimate_technical_debt(code_metrics, parse_result)
                    code_metrics.technical_debt = debt_result.debt_estimate
                    
                    result.add_component_result("technical_debt", True, int((time.time() - debt_start) * 1000))
                    logger.debug(f"Deuda técnica: {debt_result.debt_estimate.total_hours:.1f}h")
                except Exception as e:
                    logger.warning(f"Error estimando deuda técnica: {e}")
                    result.add_component_result("technical_debt", False, int((time.time() - debt_start) * 1000))
                    result.add_warning(f"Technical debt estimation failed: {e}")
            
            # 8. Evaluar quality gates
            if self.config.enable_quality_gates:
                gates_start = time.time()
                try:
                    gate_evaluation = await self.quality_gate_checker.check_quality_gates(code_metrics)
                    code_metrics.quality_metrics.quality_gate_status = gate_evaluation.overall_status
                    code_metrics.quality_metrics.quality_gate_results = gate_evaluation.gate_results
                    
                    result.add_component_result("quality_gates", True, int((time.time() - gates_start) * 1000))
                    logger.debug(f"Quality gates: {gate_evaluation.overall_status.value}")
                except Exception as e:
                    logger.warning(f"Error evaluando quality gates: {e}")
                    result.add_component_result("quality_gates", False, int((time.time() - gates_start) * 1000))
                    result.add_warning(f"Quality gates evaluation failed: {e}")
            
            # 9. Calcular score general de calidad
            code_metrics.overall_quality_score = code_metrics.calculate_overall_score()
            
            # 10. Registrar tiempo total
            code_metrics.calculation_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Cálculo de métricas completado para {parse_result.file_path}: "
                f"MI={code_metrics.quality_metrics.maintainability_index:.1f}, "
                f"CC={code_metrics.complexity_metrics.cyclomatic_complexity}, "
                f"LOC={code_metrics.size_metrics.logical_lines_of_code} en {code_metrics.calculation_time_ms}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error en cálculo de métricas para {parse_result.file_path}: {e}")
            result.calculation_success = False
            result.add_warning(f"Overall calculation failed: {e}")
            return result
    
    async def calculate_project_metrics(self, parse_results: List[ParseResult]) -> ProjectMetrics:
        """
        Calcula métricas para proyecto completo.
        
        Args:
            parse_results: Lista de resultados de parsing de todos los archivos
            
        Returns:
            ProjectMetrics completas del proyecto
        """
        start_time = time.time()
        
        try:
            logger.info(f"Iniciando cálculo de métricas de proyecto con {len(parse_results)} archivos")
            
            # Calcular métricas para cada archivo
            file_metrics = []
            session = MetricsAnalysisSession(
                session_id=f"session_{int(time.time())}",
                start_time=start_time
            )
            
            for parse_result in parse_results:
                try:
                    result = await self.calculate_metrics(parse_result)
                    if result.calculation_success:
                        file_metrics.append(result.code_metrics)
                        session.files_analyzed += 1
                        session.total_analysis_time_ms += result.code_metrics.calculation_time_ms
                    else:
                        session.add_error(f"Failed to analyze {parse_result.file_path}")
                        logger.warning(f"Fallo analizando {parse_result.file_path}")
                
                except Exception as e:
                    session.add_error(f"Error analyzing {parse_result.file_path}: {e}")
                    logger.warning(f"Error analizando {parse_result.file_path}: {e}")
            
            # Agregar métricas del proyecto
            project_metrics = self.metrics_aggregator.aggregate_project_metrics(file_metrics)
            project_metrics.calculation_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Cálculo de métricas de proyecto completado: "
                f"{len(file_metrics)}/{len(parse_results)} archivos analizados exitosamente, "
                f"MI proyecto={project_metrics.maintainability_index:.1f}, "
                f"Deuda técnica={project_metrics.technical_debt_estimate.total_hours:.1f}h "
                f"en {project_metrics.calculation_time_ms}ms"
            )
            
            return project_metrics
            
        except Exception as e:
            logger.error(f"Error en cálculo de métricas de proyecto: {e}")
            raise
    
    def get_metrics_summary(self, code_metrics: CodeMetrics) -> Dict[str, Any]:
        """
        Obtiene resumen de métricas.
        
        Returns:
            Diccionario con resumen ejecutivo
        """
        return {
            "file_info": {
                "path": str(code_metrics.file_path),
                "language": code_metrics.language.value,
                "analysis_time_ms": code_metrics.calculation_time_ms
            },
            "quality_overview": {
                "overall_score": code_metrics.overall_quality_score,
                "maintainability_index": code_metrics.quality_metrics.maintainability_index,
                "quality_grade": code_metrics.quality_metrics.get_overall_quality_grade(),
                "gate_status": code_metrics.quality_metrics.quality_gate_status.value
            },
            "complexity_summary": {
                "cyclomatic_complexity": code_metrics.complexity_metrics.cyclomatic_complexity,
                "cognitive_complexity": code_metrics.complexity_metrics.cognitive_complexity,
                "complexity_level": code_metrics.complexity_metrics.get_complexity_level(self.config.complexity_thresholds).value
            },
            "size_summary": {
                "total_lines": code_metrics.size_metrics.total_lines,
                "logical_lines": code_metrics.size_metrics.logical_lines_of_code,
                "function_count": code_metrics.size_metrics.function_count,
                "class_count": code_metrics.size_metrics.class_count
            },
            "technical_debt": {
                "total_hours": code_metrics.technical_debt.total_hours,
                "estimated_cost": code_metrics.technical_debt.estimated_cost,
                "sqale_rating": code_metrics.technical_debt.sqale_rating,
                "code_smells": len(code_metrics.technical_debt.code_smells)
            }
        }
    
    def generate_metrics_report(self, project_metrics: ProjectMetrics) -> Dict[str, Any]:
        """
        Genera reporte comprehensivo de métricas.
        
        Args:
            project_metrics: Métricas del proyecto
            
        Returns:
            Diccionario con reporte detallado
        """
        return {
            "project_summary": {
                "total_files": project_metrics.get_total_files(),
                "project_maintainability": project_metrics.maintainability_index,
                "total_loc": project_metrics.aggregated_size.total_lines,
                "total_sloc": project_metrics.aggregated_size.logical_lines_of_code,
                "analysis_time_ms": project_metrics.calculation_time_ms
            },
            "complexity_overview": {
                "average_cyclomatic": project_metrics.aggregated_complexity.cyclomatic_complexity,
                "average_cognitive": project_metrics.aggregated_complexity.cognitive_complexity,
                "max_nesting_depth": project_metrics.aggregated_complexity.max_nesting_depth,
                "complexity_distribution": {
                    "low": project_metrics.quality_distribution.excellent_files + project_metrics.quality_distribution.good_files,
                    "medium": project_metrics.quality_distribution.average_files,
                    "high": project_metrics.quality_distribution.poor_files + project_metrics.quality_distribution.very_poor_files
                }
            },
            "quality_overview": {
                "quality_distribution": {
                    "excellent": project_metrics.quality_distribution.excellent_files,
                    "good": project_metrics.quality_distribution.good_files,
                    "average": project_metrics.quality_distribution.average_files,
                    "poor": project_metrics.quality_distribution.poor_files,
                    "very_poor": project_metrics.quality_distribution.very_poor_files
                },
                "average_maintainability": project_metrics.project_quality.maintainability_index,
                "average_testability": project_metrics.project_quality.testability_score,
                "average_readability": project_metrics.project_quality.readability_score
            },
            "technical_debt_overview": {
                "total_debt_hours": project_metrics.technical_debt_estimate.total_hours,
                "total_debt_cost": project_metrics.technical_debt_estimate.estimated_cost,
                "sqale_rating": project_metrics.technical_debt_estimate.sqale_rating,
                "debt_ratio": project_metrics.technical_debt_estimate.debt_ratio,
                "top_smells": [
                    {
                        "type": smell.smell_type.value,
                        "severity": smell.severity.value,
                        "description": smell.description,
                        "fix_time_hours": smell.estimated_fix_time_minutes / 60.0
                    }
                    for smell in project_metrics.technical_debt_estimate.get_highest_priority_smells(5)
                ]
            },
            "hotspots": [
                {
                    "type": hotspot.hotspot_type,
                    "name": hotspot.name,
                    "complexity": hotspot.cyclomatic_complexity,
                    "lines": hotspot.lines_of_code,
                    "severity": hotspot.severity.value,
                    "impact_score": hotspot.impact_score,
                    "risk_level": hotspot.get_risk_level()
                }
                for hotspot in project_metrics.hotspots[:10]
            ],
            "recommendations": self._generate_project_recommendations(project_metrics)
        }
    
    def _generate_project_recommendations(self, project_metrics: ProjectMetrics) -> List[str]:
        """Genera recomendaciones para el proyecto."""
        recommendations = []
        
        # Recomendaciones basadas en mantenibilidad
        if project_metrics.maintainability_index < 50:
            recommendations.extend([
                "Project maintainability is low - consider comprehensive refactoring",
                "Focus on reducing complexity and improving code organization"
            ])
        
        # Recomendaciones basadas en deuda técnica
        if project_metrics.technical_debt_estimate.sqale_rating in ["D", "E"]:
            recommendations.extend([
                "High technical debt detected - prioritize debt reduction",
                "Address critical code smells first"
            ])
        
        # Recomendaciones basadas en distribución de calidad
        poor_files_percentage = project_metrics.quality_distribution.get_quality_percentage("poor") + \
                               project_metrics.quality_distribution.get_quality_percentage("very_poor")
        
        if poor_files_percentage > 20:
            recommendations.append(f"{poor_files_percentage:.1f}% of files have poor quality - focus refactoring efforts")
        
        # Recomendaciones basadas en hotspots
        if len(project_metrics.hotspots) > 10:
            recommendations.append("Multiple complexity hotspots detected - prioritize refactoring")
        
        # Recomendaciones basadas en métricas específicas
        if project_metrics.aggregated_complexity.cyclomatic_complexity > 15:
            recommendations.append("Average complexity is high - extract methods and simplify logic")
        
        if project_metrics.project_coupling.average_cbo > 8:
            recommendations.append("High coupling detected - apply dependency injection and interface segregation")
        
        return recommendations
    
    async def benchmark_against_industry_standards(self, project_metrics: ProjectMetrics) -> Dict[str, str]:
        """
        Compara métricas contra estándares de la industria.
        
        Returns:
            Diccionario con ratings por categoría
        """
        ratings = {}
        
        # Benchmark de complejidad
        avg_complexity = project_metrics.aggregated_complexity.cyclomatic_complexity
        if avg_complexity <= 5:
            ratings["complexity"] = "excellent"
        elif avg_complexity <= 10:
            ratings["complexity"] = "good"
        elif avg_complexity <= 20:
            ratings["complexity"] = "fair"
        else:
            ratings["complexity"] = "poor"
        
        # Benchmark de mantenibilidad
        maintainability = project_metrics.maintainability_index
        if maintainability >= 85:
            ratings["maintainability"] = "excellent"
        elif maintainability >= 70:
            ratings["maintainability"] = "good"
        elif maintainability >= 50:
            ratings["maintainability"] = "fair"
        else:
            ratings["maintainability"] = "poor"
        
        # Benchmark de deuda técnica
        debt_ratio = project_metrics.technical_debt_estimate.debt_ratio
        if debt_ratio <= 0.05:
            ratings["technical_debt"] = "excellent"
        elif debt_ratio <= 0.1:
            ratings["technical_debt"] = "good"
        elif debt_ratio <= 0.2:
            ratings["technical_debt"] = "fair"
        else:
            ratings["technical_debt"] = "poor"
        
        # Benchmark de tamaño
        avg_function_size = project_metrics.aggregated_size.average_function_length
        if avg_function_size <= 20:
            ratings["function_size"] = "excellent"
        elif avg_function_size <= 30:
            ratings["function_size"] = "good"
        elif avg_function_size <= 50:
            ratings["function_size"] = "fair"
        else:
            ratings["function_size"] = "poor"
        
        return ratings
