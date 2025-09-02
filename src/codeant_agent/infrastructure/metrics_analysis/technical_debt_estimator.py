"""
Implementación del estimador de deuda técnica.

Este módulo implementa la estimación de deuda técnica basada en
code smells, complejidad, violaciones de calidad y métricas de mantenibilidad.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import time
import re

from ...domain.entities.code_metrics import (
    TechnicalDebtEstimate, CodeSmell, CodeSmellType, CodeSmellSeverity,
    CodeMetrics, ComplexityMetrics, HalsteadMetrics, SizeMetrics,
    CohesionMetrics, CouplingMetrics, FunctionMetrics, ClassMetrics
)
from ...domain.entities.parse_result import ParseResult
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class DebtRule:
    """Regla para cálculo de deuda técnica."""
    rule_name: str
    smell_type: CodeSmellType
    threshold_value: float
    debt_per_unit: float  # Minutos de deuda por unidad excedida
    severity_multiplier: Dict[CodeSmellSeverity, float]
    description: str = ""


@dataclass
class TechnicalDebtAnalysisResult:
    """Resultado del análisis de deuda técnica."""
    debt_estimate: TechnicalDebtEstimate
    debt_breakdown: Dict[str, float]
    prioritized_smells: List[CodeSmell]
    debt_trends: Dict[str, Any]
    analysis_time_ms: int


class CodeSmellDetector:
    """Detector de code smells."""
    
    def __init__(self):
        self.severity_multipliers = {
            CodeSmellSeverity.LOW: 0.5,
            CodeSmellSeverity.MEDIUM: 1.0,
            CodeSmellSeverity.HIGH: 2.0,
            CodeSmellSeverity.CRITICAL: 4.0
        }
    
    def detect_complexity_smells(self, complexity_metrics: ComplexityMetrics, 
                               function_metrics: List[FunctionMetrics]) -> List[CodeSmell]:
        """Detecta smells relacionados con complejidad."""
        smells = []
        
        # Complejidad ciclomática alta
        if complexity_metrics.cyclomatic_complexity > 20:
            severity = (CodeSmellSeverity.CRITICAL if complexity_metrics.cyclomatic_complexity > 50 
                       else CodeSmellSeverity.HIGH)
            
            smells.append(CodeSmell(
                smell_type=CodeSmellType.HIGH_COMPLEXITY,
                severity=severity,
                description=f"High cyclomatic complexity: {complexity_metrics.cyclomatic_complexity}",
                metric_value=complexity_metrics.cyclomatic_complexity,
                threshold_value=20,
                estimated_fix_time_minutes=self._calculate_complexity_fix_time(complexity_metrics.cyclomatic_complexity),
                impact_on_maintainability=min(1.0, complexity_metrics.cyclomatic_complexity / 50.0)
            ))
        
        # Complejidad cognitiva alta
        if complexity_metrics.cognitive_complexity > 25:
            smells.append(CodeSmell(
                smell_type=CodeSmellType.HIGH_COMPLEXITY,
                severity=CodeSmellSeverity.HIGH,
                description=f"High cognitive complexity: {complexity_metrics.cognitive_complexity}",
                metric_value=complexity_metrics.cognitive_complexity,
                threshold_value=25,
                estimated_fix_time_minutes=complexity_metrics.cognitive_complexity * 2.0,
                impact_on_maintainability=min(1.0, complexity_metrics.cognitive_complexity / 40.0)
            ))
        
        # Anidamiento profundo
        if complexity_metrics.max_nesting_depth > 6:
            smells.append(CodeSmell(
                smell_type=CodeSmellType.HIGH_COMPLEXITY,
                severity=CodeSmellSeverity.MEDIUM,
                description=f"Deep nesting: {complexity_metrics.max_nesting_depth} levels",
                metric_value=complexity_metrics.max_nesting_depth,
                threshold_value=6,
                estimated_fix_time_minutes=(complexity_metrics.max_nesting_depth - 6) * 10.0,
                impact_on_maintainability=0.3,
                suggestions=["Reduce nesting by using early returns", "Extract nested logic to methods"]
            ))
        
        # Funciones complejas individuales
        for func_metric in function_metrics:
            if func_metric.cyclomatic_complexity > 15:
                severity = (CodeSmellSeverity.HIGH if func_metric.cyclomatic_complexity > 30 
                           else CodeSmellSeverity.MEDIUM)
                
                smells.append(CodeSmell(
                    smell_type=CodeSmellType.HIGH_COMPLEXITY,
                    severity=severity,
                    description=f"Complex function '{func_metric.name}': CC={func_metric.cyclomatic_complexity}",
                    affected_function=func_metric.name,
                    metric_value=func_metric.cyclomatic_complexity,
                    threshold_value=15,
                    estimated_fix_time_minutes=func_metric.cyclomatic_complexity * 3.0,
                    impact_on_maintainability=0.4,
                    suggestions=["Break down into smaller functions", "Simplify conditional logic"]
                ))
        
        return smells
    
    def detect_size_smells(self, size_metrics: SizeMetrics, function_metrics: List[FunctionMetrics], 
                          class_metrics: List[ClassMetrics]) -> List[CodeSmell]:
        """Detecta smells relacionados con tamaño."""
        smells = []
        
        # Funciones largas
        for func_metric in function_metrics:
            if func_metric.lines_of_code > 50:
                severity = (CodeSmellSeverity.HIGH if func_metric.lines_of_code > 100 
                           else CodeSmellSeverity.MEDIUM)
                
                smells.append(CodeSmell(
                    smell_type=CodeSmellType.LONG_FUNCTION,
                    severity=severity,
                    description=f"Long function '{func_metric.name}': {func_metric.lines_of_code} LOC",
                    affected_function=func_metric.name,
                    metric_value=func_metric.lines_of_code,
                    threshold_value=50,
                    estimated_fix_time_minutes=(func_metric.lines_of_code - 50) * 0.5,
                    impact_on_maintainability=0.3,
                    suggestions=["Extract smaller functions", "Remove duplicated code"]
                ))
        
        # Clases grandes
        for class_metric in class_metrics:
            if class_metric.lines_of_code > 500:
                severity = (CodeSmellSeverity.CRITICAL if class_metric.lines_of_code > 1000 
                           else CodeSmellSeverity.HIGH)
                
                smells.append(CodeSmell(
                    smell_type=CodeSmellType.LARGE_CLASS,
                    severity=severity,
                    description=f"Large class '{class_metric.name}': {class_metric.lines_of_code} LOC",
                    affected_class=class_metric.name,
                    metric_value=class_metric.lines_of_code,
                    threshold_value=500,
                    estimated_fix_time_minutes=(class_metric.lines_of_code - 500) * 0.3,
                    impact_on_maintainability=0.5,
                    suggestions=["Split into smaller classes", "Apply Single Responsibility Principle"]
                ))
        
        # Funciones con muchos parámetros
        for func_metric in function_metrics:
            if func_metric.parameter_count > 7:
                smells.append(CodeSmell(
                    smell_type=CodeSmellType.MANY_PARAMETERS,
                    severity=CodeSmellSeverity.MEDIUM,
                    description=f"Function '{func_metric.name}' has {func_metric.parameter_count} parameters",
                    affected_function=func_metric.name,
                    metric_value=func_metric.parameter_count,
                    threshold_value=7,
                    estimated_fix_time_minutes=15.0,
                    impact_on_maintainability=0.2,
                    suggestions=["Use parameter objects", "Group related parameters"]
                ))
        
        return smells
    
    def detect_design_smells(self, cohesion_metrics: CohesionMetrics, coupling_metrics: CouplingMetrics,
                           class_metrics: List[ClassMetrics]) -> List[CodeSmell]:
        """Detecta smells relacionados con diseño."""
        smells = []
        
        # Baja cohesión
        if cohesion_metrics.average_lcom > 0.8:
            smells.append(CodeSmell(
                smell_type=CodeSmellType.LOW_COHESION,
                severity=CodeSmellSeverity.HIGH,
                description=f"Low class cohesion: LCOM={cohesion_metrics.average_lcom:.2f}",
                metric_value=cohesion_metrics.average_lcom,
                threshold_value=0.8,
                estimated_fix_time_minutes=45.0,
                impact_on_maintainability=0.6,
                suggestions=["Group related methods", "Split classes with multiple responsibilities"]
            ))
        
        # Alto acoplamiento
        if coupling_metrics.average_cbo > 10:
            smells.append(CodeSmell(
                smell_type=CodeSmellType.HIGH_COUPLING,
                severity=CodeSmellSeverity.MEDIUM,
                description=f"High coupling: CBO={coupling_metrics.average_cbo:.1f}",
                metric_value=coupling_metrics.average_cbo,
                threshold_value=10,
                estimated_fix_time_minutes=30.0,
                impact_on_maintainability=0.4,
                suggestions=["Use dependency injection", "Apply interface segregation"]
            ))
        
        # Herencia profunda
        if coupling_metrics.max_inheritance_depth > 6:
            smells.append(CodeSmell(
                smell_type=CodeSmellType.DEEP_INHERITANCE,
                severity=CodeSmellSeverity.MEDIUM,
                description=f"Deep inheritance: {coupling_metrics.max_inheritance_depth} levels",
                metric_value=coupling_metrics.max_inheritance_depth,
                threshold_value=6,
                estimated_fix_time_minutes=60.0,
                impact_on_maintainability=0.4,
                suggestions=["Use composition over inheritance", "Flatten inheritance hierarchy"]
            ))
        
        # God classes
        for class_metric in class_metrics:
            if class_metric.lines_of_code > 1000 and class_metric.method_count > 20:
                smells.append(CodeSmell(
                    smell_type=CodeSmellType.GOD_CLASS,
                    severity=CodeSmellSeverity.CRITICAL,
                    description=f"God class '{class_metric.name}': {class_metric.lines_of_code} LOC, {class_metric.method_count} methods",
                    affected_class=class_metric.name,
                    metric_value=class_metric.lines_of_code,
                    threshold_value=1000,
                    estimated_fix_time_minutes=180.0,
                    impact_on_maintainability=0.8,
                    suggestions=["Split into multiple focused classes", "Extract related functionality"]
                ))
        
        return smells
    
    def detect_maintainability_smells(self, parse_result: ParseResult, maintainability_index: float) -> List[CodeSmell]:
        """Detecta smells relacionados con mantenibilidad."""
        smells = []
        content = self._get_file_content(parse_result)
        
        # Índice de mantenibilidad bajo
        if maintainability_index < 25:
            smells.append(CodeSmell(
                smell_type=CodeSmellType.HIGH_COMPLEXITY,  # Usar como categoría general
                severity=CodeSmellSeverity.CRITICAL,
                description=f"Very low maintainability index: {maintainability_index:.1f}",
                metric_value=maintainability_index,
                threshold_value=25,
                estimated_fix_time_minutes=120.0,
                impact_on_maintainability=0.9,
                suggestions=["Comprehensive refactoring needed", "Reduce complexity and size"]
            ))
        
        # Magic numbers
        magic_numbers = re.findall(r'\b\d{2,}\b', content)
        if len(magic_numbers) > 10:
            smells.append(CodeSmell(
                smell_type=CodeSmellType.MAGIC_NUMBERS,
                severity=CodeSmellSeverity.MEDIUM,
                description=f"Magic numbers found: {len(magic_numbers)} instances",
                metric_value=len(magic_numbers),
                threshold_value=10,
                estimated_fix_time_minutes=len(magic_numbers) * 2.0,
                impact_on_maintainability=0.2,
                suggestions=["Replace magic numbers with named constants", "Use enums for related constants"]
            ))
        
        # TODO/FIXME comments
        todo_count = content.count('TODO') + content.count('FIXME') + content.count('HACK')
        if todo_count > 5:
            smells.append(CodeSmell(
                smell_type=CodeSmellType.HIGH_COMPLEXITY,  # Categoría general
                severity=CodeSmellSeverity.LOW,
                description=f"Technical debt markers: {todo_count} TODO/FIXME/HACK comments",
                metric_value=todo_count,
                threshold_value=5,
                estimated_fix_time_minutes=todo_count * 15.0,
                impact_on_maintainability=0.1,
                suggestions=["Address TODO items", "Fix FIXME issues", "Refactor HACK solutions"]
            ))
        
        return smells
    
    def _calculate_complexity_fix_time(self, complexity: int) -> float:
        """Calcula tiempo estimado para arreglar complejidad."""
        if complexity <= 10:
            return 0.0
        elif complexity <= 20:
            return (complexity - 10) * 5.0
        else:
            return 50.0 + (complexity - 20) * 10.0
    
    def _get_file_content(self, parse_result: ParseResult) -> str:
        """Obtiene contenido del archivo."""
        if hasattr(parse_result.tree, 'root_node') and hasattr(parse_result.tree.root_node, 'text'):
            return parse_result.tree.root_node.text.decode('utf-8')
        return ""


class DebtCalculationEngine:
    """Motor de cálculo de deuda técnica."""
    
    def __init__(self):
        self.hourly_rate = 75.0  # USD por hora de desarrollador
        self.debt_rules = self._initialize_debt_rules()
    
    def _initialize_debt_rules(self) -> List[DebtRule]:
        """Inicializa reglas de cálculo de deuda."""
        return [
            DebtRule(
                rule_name="High Cyclomatic Complexity",
                smell_type=CodeSmellType.HIGH_COMPLEXITY,
                threshold_value=10.0,
                debt_per_unit=5.0,  # 5 minutos por punto de complejidad extra
                severity_multiplier={
                    CodeSmellSeverity.LOW: 0.5,
                    CodeSmellSeverity.MEDIUM: 1.0,
                    CodeSmellSeverity.HIGH: 2.0,
                    CodeSmellSeverity.CRITICAL: 4.0
                }
            ),
            DebtRule(
                rule_name="Long Function",
                smell_type=CodeSmellType.LONG_FUNCTION,
                threshold_value=50.0,
                debt_per_unit=0.5,  # 0.5 minutos por línea extra
                severity_multiplier={
                    CodeSmellSeverity.LOW: 0.3,
                    CodeSmellSeverity.MEDIUM: 1.0,
                    CodeSmellSeverity.HIGH: 2.0,
                    CodeSmellSeverity.CRITICAL: 3.0
                }
            ),
            DebtRule(
                rule_name="Large Class",
                smell_type=CodeSmellType.LARGE_CLASS,
                threshold_value=500.0,
                debt_per_unit=0.3,  # 0.3 minutos por línea extra
                severity_multiplier={
                    CodeSmellSeverity.MEDIUM: 1.0,
                    CodeSmellSeverity.HIGH: 2.0,
                    CodeSmellSeverity.CRITICAL: 4.0
                }
            ),
            DebtRule(
                rule_name="Low Cohesion",
                smell_type=CodeSmellType.LOW_COHESION,
                threshold_value=0.8,
                debt_per_unit=45.0,  # 45 minutos por clase con baja cohesión
                severity_multiplier={
                    CodeSmellSeverity.MEDIUM: 1.0,
                    CodeSmellSeverity.HIGH: 1.5,
                    CodeSmellSeverity.CRITICAL: 2.0
                }
            ),
            DebtRule(
                rule_name="High Coupling",
                smell_type=CodeSmellType.HIGH_COUPLING,
                threshold_value=10.0,
                debt_per_unit=20.0,  # 20 minutos por clase altamente acoplada
                severity_multiplier={
                    CodeSmellSeverity.MEDIUM: 1.0,
                    CodeSmellSeverity.HIGH: 1.5,
                    CodeSmellSeverity.CRITICAL: 2.0
                }
            )
        ]
    
    def calculate_total_debt(self, all_smells: List[CodeSmell]) -> TechnicalDebtEstimate:
        """
        Calcula deuda técnica total.
        
        Args:
            all_smells: Lista de todos los code smells detectados
            
        Returns:
            TechnicalDebtEstimate completa
        """
        total_minutes = 0.0
        debt_by_category = {}
        
        for smell in all_smells:
            # Encontrar regla aplicable
            applicable_rule = self._find_applicable_rule(smell)
            
            if applicable_rule:
                # Calcular deuda basada en regla
                excess_value = max(0, smell.metric_value - smell.threshold_value)
                base_debt = excess_value * applicable_rule.debt_per_unit
                
                # Aplicar multiplicador de severidad
                severity_multiplier = applicable_rule.severity_multiplier.get(smell.severity, 1.0)
                smell_debt = base_debt * severity_multiplier
                
                total_minutes += smell_debt
                
                # Categorizar deuda
                category = smell.smell_type.value
                debt_by_category[category] = debt_by_category.get(category, 0.0) + smell_debt
            else:
                # Usar tiempo estimado del smell si no hay regla
                total_minutes += smell.estimated_fix_time_minutes
                category = smell.smell_type.value
                debt_by_category[category] = debt_by_category.get(category, 0.0) + smell.estimated_fix_time_minutes
        
        # Crear estimación
        estimate = TechnicalDebtEstimate(
            total_minutes=total_minutes,
            code_smells=all_smells,
            debt_ratio=0.0  # Calculado después con LOC
        )
        
        # Calcular métricas derivadas
        estimate.calculate_derived_metrics(self.hourly_rate)
        
        # Categorizar deuda
        estimate.maintainability_debt = debt_by_category.get('high_complexity', 0.0) + debt_by_category.get('long_function', 0.0)
        estimate.reliability_debt = debt_by_category.get('high_coupling', 0.0) + debt_by_category.get('low_cohesion', 0.0)
        estimate.security_debt = debt_by_category.get('magic_numbers', 0.0)  # Simplificación
        
        return estimate
    
    def _find_applicable_rule(self, smell: CodeSmell) -> Optional[DebtRule]:
        """Encuentra regla aplicable para un smell."""
        for rule in self.debt_rules:
            if rule.smell_type == smell.smell_type:
                return rule
        return None


class TechnicalDebtEstimator:
    """Estimador principal de deuda técnica."""
    
    def __init__(self):
        self.smell_detector = CodeSmellDetector()
        self.debt_calculator = DebtCalculationEngine()
    
    async def estimate_technical_debt(
        self, 
        code_metrics: CodeMetrics,
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> TechnicalDebtAnalysisResult:
        """
        Estima deuda técnica completa.
        
        Args:
            code_metrics: Métricas de código calculadas
            parse_result: Resultado del parsing original
            config: Configuración opcional
            
        Returns:
            TechnicalDebtAnalysisResult completo
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Estimando deuda técnica para {parse_result.file_path}")
            
            # Detectar todos los code smells
            all_smells = []
            
            # Smells de complejidad
            complexity_smells = self.smell_detector.detect_complexity_smells(
                code_metrics.complexity_metrics, 
                code_metrics.function_metrics
            )
            all_smells.extend(complexity_smells)
            
            # Smells de tamaño
            size_smells = self.smell_detector.detect_size_smells(
                code_metrics.size_metrics,
                code_metrics.function_metrics,
                code_metrics.class_metrics
            )
            all_smells.extend(size_smells)
            
            # Smells de diseño
            design_smells = self.smell_detector.detect_design_smells(
                code_metrics.cohesion_metrics,
                code_metrics.coupling_metrics,
                code_metrics.class_metrics
            )
            all_smells.extend(design_smells)
            
            # Smells de mantenibilidad
            maintainability_smells = self.smell_detector.detect_maintainability_smells(
                parse_result, 
                code_metrics.quality_metrics.maintainability_index
            )
            all_smells.extend(maintainability_smells)
            
            # Calcular deuda total
            debt_estimate = self.debt_calculator.calculate_total_debt(all_smells)
            
            # Calcular debt ratio basado en LOC
            if code_metrics.size_metrics.logical_lines_of_code > 0:
                debt_estimate.debt_ratio = debt_estimate.total_minutes / (code_metrics.size_metrics.logical_lines_of_code * 0.5)
            
            # Priorizar smells
            prioritized_smells = self._prioritize_smells(all_smells)
            
            # Calcular breakdown de deuda
            debt_breakdown = self._calculate_debt_breakdown(all_smells)
            
            # Analizar tendencias (simplificado)
            debt_trends = self._analyze_debt_trends(debt_estimate, code_metrics)
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Estimación de deuda técnica completada para {parse_result.file_path}: "
                f"{len(all_smells)} smells, {debt_estimate.total_hours:.1f}h, "
                f"${debt_estimate.estimated_cost:.0f} en {total_time}ms"
            )
            
            return TechnicalDebtAnalysisResult(
                debt_estimate=debt_estimate,
                debt_breakdown=debt_breakdown,
                prioritized_smells=prioritized_smells,
                debt_trends=debt_trends,
                analysis_time_ms=total_time
            )
            
        except Exception as e:
            logger.error(f"Error estimando deuda técnica: {e}")
            raise
    
    def _prioritize_smells(self, smells: List[CodeSmell]) -> List[CodeSmell]:
        """Prioriza smells por impacto y facilidad de arreglo."""
        return sorted(smells, key=lambda s: s.get_priority_score(), reverse=True)
    
    def _calculate_debt_breakdown(self, smells: List[CodeSmell]) -> Dict[str, float]:
        """Calcula breakdown de deuda por categoría."""
        breakdown = {}
        
        for smell in smells:
            category = smell.smell_type.value
            breakdown[category] = breakdown.get(category, 0.0) + smell.estimated_fix_time_minutes
        
        return breakdown
    
    def _analyze_debt_trends(self, debt_estimate: TechnicalDebtEstimate, code_metrics: CodeMetrics) -> Dict[str, Any]:
        """Analiza tendencias de deuda técnica."""
        return {
            "sqale_rating": debt_estimate.sqale_rating,
            "debt_density": debt_estimate.debt_ratio,
            "primary_debt_sources": self._identify_primary_debt_sources(debt_estimate),
            "improvement_opportunities": self._identify_improvement_opportunities(debt_estimate, code_metrics),
            "roi_recommendations": self._calculate_roi_recommendations(debt_estimate)
        }
    
    def _identify_primary_debt_sources(self, debt_estimate: TechnicalDebtEstimate) -> List[str]:
        """Identifica principales fuentes de deuda."""
        sources = []
        
        if debt_estimate.maintainability_debt > debt_estimate.total_minutes * 0.4:
            sources.append("complexity_and_maintainability")
        
        if debt_estimate.reliability_debt > debt_estimate.total_minutes * 0.3:
            sources.append("design_quality")
        
        critical_smells = debt_estimate.get_smells_by_severity(CodeSmellSeverity.CRITICAL)
        if len(critical_smells) > 3:
            sources.append("critical_issues")
        
        return sources
    
    def _identify_improvement_opportunities(self, debt_estimate: TechnicalDebtEstimate, 
                                          code_metrics: CodeMetrics) -> List[str]:
        """Identifica oportunidades de mejora."""
        opportunities = []
        
        # Oportunidades basadas en tipos de smells
        smell_types = {}
        for smell in debt_estimate.code_smells:
            smell_types[smell.smell_type] = smell_types.get(smell.smell_type, 0) + 1
        
        if smell_types.get(CodeSmellType.HIGH_COMPLEXITY, 0) > 2:
            opportunities.append("complexity_reduction")
        
        if smell_types.get(CodeSmellType.LONG_FUNCTION, 0) > 3:
            opportunities.append("function_extraction")
        
        if smell_types.get(CodeSmellType.LARGE_CLASS, 0) > 1:
            opportunities.append("class_decomposition")
        
        # Oportunidades basadas en métricas
        if code_metrics.quality_metrics.maintainability_index < 50:
            opportunities.append("comprehensive_refactoring")
        
        return opportunities
    
    def _calculate_roi_recommendations(self, debt_estimate: TechnicalDebtEstimate) -> List[Dict[str, Any]]:
        """Calcula recomendaciones ROI."""
        recommendations = []
        
        # Ordenar smells por ROI (impact / effort)
        high_roi_smells = []
        for smell in debt_estimate.code_smells:
            if smell.estimated_fix_time_minutes > 0:
                roi = smell.impact_on_maintainability / (smell.estimated_fix_time_minutes / 60.0)
                if roi > 0.5:  # ROI alto
                    high_roi_smells.append((smell, roi))
        
        # Crear recomendaciones
        high_roi_smells.sort(key=lambda x: x[1], reverse=True)
        
        for smell, roi in high_roi_smells[:5]:  # Top 5
            recommendations.append({
                "smell_type": smell.smell_type.value,
                "description": smell.description,
                "estimated_effort_hours": smell.estimated_fix_time_minutes / 60.0,
                "impact_score": smell.impact_on_maintainability,
                "roi_score": roi,
                "priority": "high" if roi > 1.0 else "medium"
            })
        
        return recommendations
