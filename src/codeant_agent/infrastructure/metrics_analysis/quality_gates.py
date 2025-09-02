"""
Implementación del sistema de Quality Gates.

Este módulo implementa la evaluación de quality gates basada en
umbrales configurables de métricas de calidad.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import time

from ...domain.entities.code_metrics import (
    QualityGateDefinition, QualityGateResult, QualityGateStatus,
    MetricType, CodeSmellSeverity, CodeMetrics
)
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class QualityGateEvaluation:
    """Evaluación completa de quality gates."""
    overall_status: QualityGateStatus
    gate_results: List[QualityGateResult]
    passed_gates: int
    failed_gates: int
    warning_gates: int
    success_percentage: float
    critical_failures: List[QualityGateResult]
    
    def is_passing(self) -> bool:
        """Verifica si la evaluación general está pasando."""
        return self.overall_status == QualityGateStatus.PASSED
    
    def get_failure_summary(self) -> str:
        """Obtiene resumen de fallos."""
        if self.is_passing():
            return "All quality gates passed"
        
        failed_gates = [result for result in self.gate_results if not result.is_passing()]
        return f"{len(failed_gates)} quality gates failed: {', '.join(gate.gate_name for gate in failed_gates[:3])}"


@dataclass
class QualityGateProfile:
    """Perfil de quality gates para un contexto específico."""
    profile_name: str
    description: str
    gates: List[QualityGateDefinition]
    strict_mode: bool = False
    
    def get_gates_by_type(self, metric_type: MetricType) -> List[QualityGateDefinition]:
        """Obtiene gates por tipo de métrica."""
        return [gate for gate in self.gates if gate.metric_type == metric_type]


class QualityGateProfileManager:
    """Gestor de perfiles de quality gates."""
    
    def __init__(self):
        self.profiles = self._initialize_default_profiles()
    
    def _initialize_default_profiles(self) -> Dict[str, QualityGateProfile]:
        """Inicializa perfiles por defecto."""
        profiles = {}
        
        # Perfil básico
        basic_gates = [
            QualityGateDefinition(
                name="Cyclomatic Complexity",
                metric_type=MetricType.COMPLEXITY,
                threshold_value=20.0,
                comparison_operator="<=",
                severity=CodeSmellSeverity.HIGH,
                description="Cyclomatic complexity should not exceed 20"
            ),
            QualityGateDefinition(
                name="Function Length",
                metric_type=MetricType.SIZE,
                threshold_value=50.0,
                comparison_operator="<=",
                severity=CodeSmellSeverity.MEDIUM,
                description="Functions should not exceed 50 lines"
            ),
            QualityGateDefinition(
                name="Maintainability Index",
                metric_type=MetricType.QUALITY,
                threshold_value=60.0,
                comparison_operator=">=",
                severity=CodeSmellSeverity.HIGH,
                description="Maintainability index should be at least 60"
            )
        ]
        
        profiles["basic"] = QualityGateProfile(
            profile_name="basic",
            description="Basic quality gates for general projects",
            gates=basic_gates
        )
        
        # Perfil estricto
        strict_gates = basic_gates + [
            QualityGateDefinition(
                name="Cognitive Complexity",
                metric_type=MetricType.COMPLEXITY,
                threshold_value=15.0,
                comparison_operator="<=",
                severity=CodeSmellSeverity.HIGH,
                description="Cognitive complexity should not exceed 15"
            ),
            QualityGateDefinition(
                name="Class Size",
                metric_type=MetricType.SIZE,
                threshold_value=500.0,
                comparison_operator="<=",
                severity=CodeSmellSeverity.MEDIUM,
                description="Classes should not exceed 500 lines"
            ),
            QualityGateDefinition(
                name="Coupling Between Objects",
                metric_type=MetricType.COUPLING,
                threshold_value=10.0,
                comparison_operator="<=",
                severity=CodeSmellSeverity.MEDIUM,
                description="CBO should not exceed 10"
            ),
            QualityGateDefinition(
                name="Comment Ratio",
                metric_type=MetricType.SIZE,
                threshold_value=0.1,
                comparison_operator=">=",
                severity=CodeSmellSeverity.LOW,
                description="Comment ratio should be at least 10%"
            )
        ]
        
        profiles["strict"] = QualityGateProfile(
            profile_name="strict",
            description="Strict quality gates for high-quality projects",
            gates=strict_gates,
            strict_mode=True
        )
        
        # Perfil enterprise
        enterprise_gates = strict_gates + [
            QualityGateDefinition(
                name="Technical Debt Ratio",
                metric_type=MetricType.TECHNICAL_DEBT,
                threshold_value=0.05,
                comparison_operator="<=",
                severity=CodeSmellSeverity.HIGH,
                description="Technical debt ratio should not exceed 5%"
            ),
            QualityGateDefinition(
                name="Halstead Difficulty",
                metric_type=MetricType.HALSTEAD,
                threshold_value=30.0,
                comparison_operator="<=",
                severity=CodeSmellSeverity.MEDIUM,
                description="Halstead difficulty should not exceed 30"
            ),
            QualityGateDefinition(
                name="Class Cohesion",
                metric_type=MetricType.COHESION,
                threshold_value=0.6,
                comparison_operator="<=",
                severity=CodeSmellSeverity.MEDIUM,
                description="LCOM should not exceed 0.6"
            )
        ]
        
        profiles["enterprise"] = QualityGateProfile(
            profile_name="enterprise",
            description="Enterprise-grade quality gates",
            gates=enterprise_gates,
            strict_mode=True
        )
        
        return profiles
    
    def get_profile(self, profile_name: str) -> QualityGateProfile:
        """Obtiene perfil por nombre."""
        return self.profiles.get(profile_name, self.profiles["basic"])
    
    def create_custom_profile(self, name: str, gates: List[QualityGateDefinition]) -> QualityGateProfile:
        """Crea perfil personalizado."""
        profile = QualityGateProfile(
            profile_name=name,
            description=f"Custom profile: {name}",
            gates=gates
        )
        self.profiles[name] = profile
        return profile


class QualityGateChecker:
    """Evaluador de quality gates."""
    
    def __init__(self):
        self.profile_manager = QualityGateProfileManager()
    
    async def check_quality_gates(
        self, 
        code_metrics: CodeMetrics,
        profile_name: str = "basic",
        config: Optional[Dict[str, Any]] = None
    ) -> QualityGateEvaluation:
        """
        Evalúa quality gates contra métricas de código.
        
        Args:
            code_metrics: Métricas de código calculadas
            profile_name: Nombre del perfil de gates a usar
            config: Configuración opcional
            
        Returns:
            QualityGateEvaluation completa
        """
        try:
            logger.debug(f"Evaluando quality gates con perfil '{profile_name}'")
            
            # Obtener perfil de gates
            profile = self.profile_manager.get_profile(profile_name)
            
            # Evaluar cada gate
            gate_results = []
            for gate_def in profile.gates:
                result = await self._evaluate_single_gate(gate_def, code_metrics)
                gate_results.append(result)
            
            # Calcular estadísticas
            passed_gates = sum(1 for result in gate_results if result.status == QualityGateStatus.PASSED)
            failed_gates = sum(1 for result in gate_results if result.status == QualityGateStatus.FAILED)
            warning_gates = sum(1 for result in gate_results if result.status == QualityGateStatus.WARNING)
            
            success_percentage = (passed_gates / len(gate_results)) * 100.0 if gate_results else 100.0
            
            # Determinar estado general
            overall_status = self._determine_overall_status(gate_results, profile)
            
            # Identificar fallos críticos
            critical_failures = [
                result for result in gate_results 
                if result.status == QualityGateStatus.FAILED and 
                self._is_critical_gate(result.gate_name, profile)
            ]
            
            evaluation = QualityGateEvaluation(
                overall_status=overall_status,
                gate_results=gate_results,
                passed_gates=passed_gates,
                failed_gates=failed_gates,
                warning_gates=warning_gates,
                success_percentage=success_percentage,
                critical_failures=critical_failures
            )
            
            logger.info(
                f"Quality gates evaluados: {passed_gates}/{len(gate_results)} passed "
                f"({success_percentage:.1f}%), status={overall_status.value}"
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluando quality gates: {e}")
            raise
    
    async def _evaluate_single_gate(self, gate_def: QualityGateDefinition, code_metrics: CodeMetrics) -> QualityGateResult:
        """Evalúa un single quality gate."""
        # Extraer valor actual de la métrica
        actual_value = self._extract_metric_value(gate_def, code_metrics)
        
        # Evaluar gate
        return gate_def.evaluate(actual_value)
    
    def _extract_metric_value(self, gate_def: QualityGateDefinition, code_metrics: CodeMetrics) -> float:
        """Extrae valor de métrica según el tipo de gate."""
        if gate_def.metric_type == MetricType.COMPLEXITY:
            if "cyclomatic" in gate_def.name.lower():
                return float(code_metrics.complexity_metrics.cyclomatic_complexity)
            elif "cognitive" in gate_def.name.lower():
                return float(code_metrics.complexity_metrics.cognitive_complexity)
            elif "nesting" in gate_def.name.lower():
                return float(code_metrics.complexity_metrics.max_nesting_depth)
            else:
                return float(code_metrics.complexity_metrics.cyclomatic_complexity)
        
        elif gate_def.metric_type == MetricType.SIZE:
            if "function" in gate_def.name.lower():
                return float(code_metrics.size_metrics.max_function_length)
            elif "class" in gate_def.name.lower():
                return float(code_metrics.size_metrics.max_class_length)
            elif "comment" in gate_def.name.lower():
                return code_metrics.size_metrics.get_comment_ratio()
            else:
                return float(code_metrics.size_metrics.logical_lines_of_code)
        
        elif gate_def.metric_type == MetricType.HALSTEAD:
            if "volume" in gate_def.name.lower():
                return code_metrics.halstead_metrics.volume
            elif "difficulty" in gate_def.name.lower():
                return code_metrics.halstead_metrics.difficulty
            elif "effort" in gate_def.name.lower():
                return code_metrics.halstead_metrics.effort
            else:
                return code_metrics.halstead_metrics.volume
        
        elif gate_def.metric_type == MetricType.COHESION:
            if "lcom" in gate_def.name.lower():
                return code_metrics.cohesion_metrics.average_lcom
            elif "tcc" in gate_def.name.lower():
                return code_metrics.cohesion_metrics.average_tcc
            else:
                return code_metrics.cohesion_metrics.average_lcom
        
        elif gate_def.metric_type == MetricType.COUPLING:
            if "cbo" in gate_def.name.lower():
                return code_metrics.coupling_metrics.average_cbo
            elif "rfc" in gate_def.name.lower():
                return code_metrics.coupling_metrics.average_rfc
            elif "dit" in gate_def.name.lower():
                return code_metrics.coupling_metrics.average_dit
            else:
                return code_metrics.coupling_metrics.average_cbo
        
        elif gate_def.metric_type == MetricType.QUALITY:
            if "maintainability" in gate_def.name.lower():
                return code_metrics.quality_metrics.maintainability_index
            elif "testability" in gate_def.name.lower():
                return code_metrics.quality_metrics.testability_score
            elif "readability" in gate_def.name.lower():
                return code_metrics.quality_metrics.readability_score
            else:
                return code_metrics.quality_metrics.maintainability_index
        
        elif gate_def.metric_type == MetricType.TECHNICAL_DEBT:
            if hasattr(code_metrics, 'technical_debt'):
                if "ratio" in gate_def.name.lower():
                    return code_metrics.technical_debt.debt_ratio
                elif "hours" in gate_def.name.lower():
                    return code_metrics.technical_debt.total_hours
                else:
                    return code_metrics.technical_debt.debt_ratio
        
        return 0.0  # Valor por defecto
    
    def _determine_overall_status(self, gate_results: List[QualityGateResult], 
                                profile: QualityGateProfile) -> QualityGateStatus:
        """Determina estado general basado en resultados individuales."""
        if not gate_results:
            return QualityGateStatus.NOT_EVALUATED
        
        failed_results = [r for r in gate_results if r.status == QualityGateStatus.FAILED]
        
        if not failed_results:
            return QualityGateStatus.PASSED
        
        # En modo estricto, cualquier fallo es fallo general
        if profile.strict_mode:
            return QualityGateStatus.FAILED
        
        # En modo normal, verificar severidad de fallos
        critical_failures = [
            r for r in failed_results 
            if self._is_critical_gate(r.gate_name, profile)
        ]
        
        if critical_failures:
            return QualityGateStatus.FAILED
        elif failed_results:
            return QualityGateStatus.WARNING
        else:
            return QualityGateStatus.PASSED
    
    def _is_critical_gate(self, gate_name: str, profile: QualityGateProfile) -> bool:
        """Verifica si un gate es crítico."""
        critical_gates = [
            "Cyclomatic Complexity",
            "Maintainability Index",
            "Technical Debt Ratio"
        ]
        
        return gate_name in critical_gates
    
    async def evaluate_gates_for_project(self, project_metrics: List[CodeMetrics], 
                                       profile_name: str = "basic") -> QualityGateEvaluation:
        """
        Evalúa quality gates para proyecto completo.
        
        Args:
            project_metrics: Lista de métricas de todos los archivos
            profile_name: Nombre del perfil a usar
            
        Returns:
            QualityGateEvaluation agregada del proyecto
        """
        if not project_metrics:
            return QualityGateEvaluation(
                overall_status=QualityGateStatus.NOT_EVALUATED,
                gate_results=[],
                passed_gates=0,
                failed_gates=0,
                warning_gates=0,
                success_percentage=0.0,
                critical_failures=[]
            )
        
        # Agregar métricas del proyecto
        aggregated_metrics = self._aggregate_project_metrics(project_metrics)
        
        # Evaluar gates contra métricas agregadas
        return await self.check_quality_gates(aggregated_metrics, profile_name)
    
    def _aggregate_project_metrics(self, project_metrics: List[CodeMetrics]) -> CodeMetrics:
        """Agrega métricas de proyecto."""
        from ...domain.entities.code_metrics import ComplexityMetrics, HalsteadMetrics, SizeMetrics, CohesionMetrics, CouplingMetrics, QualityMetrics
        
        if not project_metrics:
            return CodeMetrics()
        
        # Agregar métricas de complejidad
        total_complexity = sum(m.complexity_metrics.cyclomatic_complexity for m in project_metrics)
        total_cognitive = sum(m.complexity_metrics.cognitive_complexity for m in project_metrics)
        max_nesting = max(m.complexity_metrics.max_nesting_depth for m in project_metrics)
        
        aggregated_complexity = ComplexityMetrics(
            cyclomatic_complexity=total_complexity // len(project_metrics),
            cognitive_complexity=total_cognitive // len(project_metrics),
            max_nesting_depth=max_nesting
        )
        
        # Agregar métricas de Halstead
        total_volume = sum(m.halstead_metrics.volume for m in project_metrics)
        total_difficulty = sum(m.halstead_metrics.difficulty for m in project_metrics)
        
        aggregated_halstead = HalsteadMetrics(
            volume=total_volume / len(project_metrics),
            difficulty=total_difficulty / len(project_metrics)
        )
        
        # Agregar métricas de tamaño
        total_lines = sum(m.size_metrics.total_lines for m in project_metrics)
        total_sloc = sum(m.size_metrics.logical_lines_of_code for m in project_metrics)
        total_comment_lines = sum(m.size_metrics.comment_lines for m in project_metrics)
        max_func_length = max(m.size_metrics.max_function_length for m in project_metrics)
        max_class_length = max(m.size_metrics.max_class_length for m in project_metrics)
        
        aggregated_size = SizeMetrics(
            total_lines=total_lines,
            logical_lines_of_code=total_sloc,
            comment_lines=total_comment_lines,
            max_function_length=max_func_length,
            max_class_length=max_class_length
        )
        aggregated_size.calculate_derived_metrics()
        
        # Agregar métricas de cohesión
        avg_lcom = sum(m.cohesion_metrics.average_lcom for m in project_metrics) / len(project_metrics)
        avg_tcc = sum(m.cohesion_metrics.average_tcc for m in project_metrics) / len(project_metrics)
        
        aggregated_cohesion = CohesionMetrics(
            average_lcom=avg_lcom,
            average_tcc=avg_tcc,
            class_count=sum(m.cohesion_metrics.class_count for m in project_metrics)
        )
        
        # Agregar métricas de acoplamiento
        avg_cbo = sum(m.coupling_metrics.average_cbo for m in project_metrics) / len(project_metrics)
        avg_rfc = sum(m.coupling_metrics.average_rfc for m in project_metrics) / len(project_metrics)
        max_dit = max(m.coupling_metrics.max_inheritance_depth for m in project_metrics)
        
        aggregated_coupling = CouplingMetrics(
            average_cbo=avg_cbo,
            average_rfc=avg_rfc,
            max_inheritance_depth=max_dit
        )
        
        # Agregar métricas de calidad
        avg_maintainability = sum(m.quality_metrics.maintainability_index for m in project_metrics) / len(project_metrics)
        avg_testability = sum(m.quality_metrics.testability_score for m in project_metrics) / len(project_metrics)
        avg_readability = sum(m.quality_metrics.readability_score for m in project_metrics) / len(project_metrics)
        
        aggregated_quality = QualityMetrics(
            maintainability_index=avg_maintainability,
            testability_score=avg_testability,
            readability_score=avg_readability
        )
        
        # Crear métricas agregadas
        aggregated = CodeMetrics(
            complexity_metrics=aggregated_complexity,
            halstead_metrics=aggregated_halstead,
            size_metrics=aggregated_size,
            cohesion_metrics=aggregated_cohesion,
            coupling_metrics=aggregated_coupling,
            quality_metrics=aggregated_quality
        )
        
        return aggregated
    
    def get_quality_gate_recommendations(self, evaluation: QualityGateEvaluation) -> List[str]:
        """
        Genera recomendaciones basadas en evaluación de gates.
        
        Returns:
            Lista de recomendaciones específicas
        """
        recommendations = []
        
        if evaluation.is_passing():
            recommendations.append("All quality gates are passing - maintain current quality standards")
            return recommendations
        
        # Recomendaciones por gates fallidos
        for failed_gate in evaluation.critical_failures:
            if "Complexity" in failed_gate.gate_name:
                recommendations.extend([
                    "Reduce code complexity by extracting methods",
                    "Simplify conditional logic and nested structures"
                ])
            
            elif "Maintainability" in failed_gate.gate_name:
                recommendations.extend([
                    "Improve maintainability by reducing complexity and size",
                    "Add more documentation and comments"
                ])
            
            elif "Function Length" in failed_gate.gate_name or "Class Size" in failed_gate.gate_name:
                recommendations.extend([
                    "Break down large functions and classes",
                    "Apply Single Responsibility Principle"
                ])
            
            elif "Coupling" in failed_gate.gate_name:
                recommendations.extend([
                    "Reduce coupling by using dependency injection",
                    "Apply interface segregation principle"
                ])
            
            elif "Cohesion" in failed_gate.gate_name:
                recommendations.extend([
                    "Improve class cohesion by grouping related functionality",
                    "Split classes with multiple responsibilities"
                ])
        
        # Recomendaciones generales si hay muchos fallos
        if evaluation.failed_gates > len(evaluation.gate_results) * 0.5:
            recommendations.extend([
                "Consider comprehensive refactoring",
                "Implement incremental quality improvements",
                "Add more unit tests to prevent regressions"
            ])
        
        return list(set(recommendations))  # Eliminar duplicados
    
    def create_quality_report(self, evaluation: QualityGateEvaluation, 
                            code_metrics: CodeMetrics) -> Dict[str, Any]:
        """
        Crea reporte de calidad detallado.
        
        Returns:
            Diccionario con reporte comprehensivo
        """
        report = {
            "summary": {
                "overall_status": evaluation.overall_status.value,
                "success_percentage": evaluation.success_percentage,
                "gates_passed": evaluation.passed_gates,
                "gates_failed": evaluation.failed_gates,
                "gates_warning": evaluation.warning_gates,
                "quality_grade": code_metrics.quality_metrics.get_overall_quality_grade()
            },
            "gate_details": [
                {
                    "name": result.gate_name,
                    "status": result.status.value,
                    "actual_value": result.actual_value,
                    "threshold": result.threshold_value,
                    "message": result.message
                }
                for result in evaluation.gate_results
            ],
            "critical_failures": [
                {
                    "gate": failure.gate_name,
                    "actual": failure.actual_value,
                    "threshold": failure.threshold_value,
                    "severity": "critical"
                }
                for failure in evaluation.critical_failures
            ],
            "recommendations": self.get_quality_gate_recommendations(evaluation),
            "metrics_breakdown": {
                "maintainability_index": code_metrics.quality_metrics.maintainability_index,
                "complexity": code_metrics.complexity_metrics.cyclomatic_complexity,
                "size_loc": code_metrics.size_metrics.logical_lines_of_code,
                "technical_debt_hours": code_metrics.technical_debt.total_hours if hasattr(code_metrics, 'technical_debt') else 0.0
            }
        }
        
        return report
