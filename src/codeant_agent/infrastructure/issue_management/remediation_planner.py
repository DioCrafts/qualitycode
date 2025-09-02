"""
Implementación del planificador de remediación y calculador de ROI.

Este módulo implementa la planificación inteligente de fixes, cálculo de ROI,
estimación de recursos y optimización de orden de ejecución.
"""

import logging
import asyncio
import math
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from datetime import datetime, timedelta
from collections import defaultdict

from ...domain.entities.issue_management import (
    IssueCluster, FixPlan, RemediationPlan, FixDependency, FixExecutionStep,
    SprintPlan, ROIAnalysis, FixBenefits, ResourceEstimate, QualityGoal,
    RemediationConfig, FixType, PriorityLevel, IssueCategory, ImplementationStep,
    TestingStrategy, RollbackPlan
)

logger = logging.getLogger(__name__)


@dataclass
class DependencyAnalysisResult:
    """Resultado del análisis de dependencias."""
    dependencies: List[FixDependency]
    dependency_graph: Dict[str, List[str]]
    circular_dependencies: List[List[str]]
    critical_path: List[str]
    parallelizable_groups: List[List[str]]


@dataclass
class ResourceRequirement:
    """Requerimiento de recursos para fix."""
    fix_id: str
    skill_requirements: Dict[str, float]  # skill -> hours needed
    tool_requirements: List[str]
    estimated_hours: float
    team_size_optimal: int
    external_dependencies: List[str]


class DependencyAnalyzer:
    """Analizador de dependencias entre fixes."""
    
    async def analyze_fix_dependencies(self, clusters: List[IssueCluster]) -> DependencyAnalysisResult:
        """
        Analiza dependencias entre fixes de clusters.
        
        Args:
            clusters: Lista de clusters con fix plans
            
        Returns:
            DependencyAnalysisResult completo
        """
        dependencies = []
        dependency_graph = defaultdict(list)
        
        # Crear fix plans para análisis
        fix_plans = []
        for cluster in clusters:
            fix_plan = await self._create_fix_plan_for_cluster(cluster)
            fix_plans.append(fix_plan)
        
        # Analizar dependencias entre fix plans
        for i, fix1 in enumerate(fix_plans):
            for j, fix2 in enumerate(fix_plans):
                if i != j:
                    dependency = await self._analyze_dependency_between_fixes(fix1, fix2)
                    if dependency:
                        dependencies.append(dependency)
                        dependency_graph[dependency.prerequisite_fix_id].append(dependency.dependent_fix_id)
        
        # Detectar dependencias circulares
        circular_deps = self._detect_circular_dependencies(dependency_graph)
        
        # Calcular critical path
        critical_path = self._calculate_critical_path(dependency_graph, fix_plans)
        
        # Identificar grupos paralelizables
        parallelizable_groups = self._identify_parallelizable_groups(dependency_graph, fix_plans)
        
        return DependencyAnalysisResult(
            dependencies=dependencies,
            dependency_graph=dict(dependency_graph),
            circular_dependencies=circular_deps,
            critical_path=critical_path,
            parallelizable_groups=parallelizable_groups
        )
    
    async def _create_fix_plan_for_cluster(self, cluster: IssueCluster) -> FixPlan:
        """Crea plan de fix para un cluster."""
        # Determinar tipo de fix basado en categoría dominante
        dominant_category = cluster.get_dominant_category()
        
        fix_type_mapping = {
            IssueCategory.SECURITY: FixType.CODE_CHANGE,
            IssueCategory.PERFORMANCE: FixType.REFACTORING,
            IssueCategory.MAINTAINABILITY: FixType.REFACTORING,
            IssueCategory.RELIABILITY: FixType.CODE_CHANGE,
            IssueCategory.DOCUMENTATION: FixType.DOCUMENTATION,
            IssueCategory.CODE_STYLE: FixType.CODE_CHANGE,
            IssueCategory.ARCHITECTURE: FixType.ARCHITECTURAL
        }
        
        fix_type = fix_type_mapping.get(dominant_category, FixType.CODE_CHANGE)
        
        # Crear plan de fix
        fix_plan = FixPlan(
            cluster_id=cluster.id,
            fix_type=fix_type,
            title=f"Fix {len(cluster.issues)} {dominant_category.value} issues",
            description=f"Batch fix for {dominant_category.value} issues in cluster {cluster.id.value[:8]}",
            affected_issues=[issue.id for issue in cluster.issues],
            estimated_effort_hours=cluster.estimated_batch_fix_time,
            confidence_level=cluster.cohesion_score,
            priority_score=cluster.get_average_priority(),
            implementation_steps=await self._generate_implementation_steps(cluster),
            testing_strategy=self._create_testing_strategy(cluster),
            rollback_plan=self._create_rollback_plan(cluster)
        )
        
        return fix_plan
    
    async def _analyze_dependency_between_fixes(self, fix1: FixPlan, fix2: FixPlan) -> Optional[FixDependency]:
        """Analiza si existe dependencia entre dos fixes."""
        # Dependencias basadas en tipo de fix
        dependency_rules = {
            # Architectural fixes deben ir antes que code changes
            (FixType.ARCHITECTURAL, FixType.CODE_CHANGE): "architectural_foundation",
            (FixType.ARCHITECTURAL, FixType.REFACTORING): "architectural_foundation",
            
            # Security fixes tienen prioridad sobre performance
            (FixType.CODE_CHANGE, FixType.REFACTORING): "security_first",
            
            # Documentation puede ir en paralelo con la mayoría
            # Refactoring debe ir antes que optimizaciones menores
        }
        
        dependency_type = dependency_rules.get((fix1.fix_type, fix2.fix_type))
        
        if dependency_type:
            return FixDependency(
                prerequisite_fix_id=fix1.id,
                dependent_fix_id=fix2.id,
                dependency_type="sequential",
                dependency_reason=dependency_type
            )
        
        # Dependencias basadas en archivos afectados
        if self._fixes_affect_same_files(fix1, fix2):
            # Si afectan los mismos archivos, deben ser secuenciales
            # El de mayor prioridad va primero
            if fix1.priority_score > fix2.priority_score:
                return FixDependency(
                    prerequisite_fix_id=fix1.id,
                    dependent_fix_id=fix2.id,
                    dependency_type="blocking",
                    dependency_reason="same_files_affected"
                )
            else:
                return FixDependency(
                    prerequisite_fix_id=fix2.id,
                    dependent_fix_id=fix1.id,
                    dependency_type="blocking",
                    dependency_reason="same_files_affected"
                )
        
        return None
    
    def _fixes_affect_same_files(self, fix1: FixPlan, fix2: FixPlan) -> bool:
        """Verifica si los fixes afectan los mismos archivos."""
        # Simplificación: asumir que clusters con issues en archivos similares se afectan
        # En implementación real, analizaría archivos específicos afectados
        return (len(fix1.affected_issues) > 2 and len(fix2.affected_issues) > 2 and
                abs(len(fix1.affected_issues) - len(fix2.affected_issues)) <= 2)
    
    def _detect_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detecta dependencias circulares."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Encontrado ciclo
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in dependency_graph.get(node, []):
                if dfs(neighbor, path[:]):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in dependency_graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def _calculate_critical_path(self, dependency_graph: Dict[str, List[str]], fix_plans: List[FixPlan]) -> List[str]:
        """Calcula critical path del proyecto."""
        # Simplificación: path con mayor suma de esfuerzo
        fix_efforts = {fix.id: fix.estimated_effort_hours for fix in fix_plans}
        
        def calculate_path_effort(path: List[str]) -> float:
            return sum(fix_efforts.get(fix_id, 0.0) for fix_id in path)
        
        # Encontrar path más largo (en términos de esfuerzo)
        longest_path = []
        max_effort = 0.0
        
        for start_node in dependency_graph:
            path = self._find_longest_path(start_node, dependency_graph, set())
            effort = calculate_path_effort(path)
            if effort > max_effort:
                max_effort = effort
                longest_path = path
        
        return longest_path
    
    def _find_longest_path(self, node: str, graph: Dict[str, List[str]], visited: Set[str]) -> List[str]:
        """Encuentra el path más largo desde un nodo."""
        if node in visited:
            return []
        
        visited.add(node)
        longest_subpath = []
        
        for neighbor in graph.get(node, []):
            subpath = self._find_longest_path(neighbor, graph, visited.copy())
            if len(subpath) > len(longest_subpath):
                longest_subpath = subpath
        
        return [node] + longest_subpath
    
    def _identify_parallelizable_groups(self, dependency_graph: Dict[str, List[str]], 
                                      fix_plans: List[FixPlan]) -> List[List[str]]:
        """Identifica grupos de fixes que se pueden ejecutar en paralelo."""
        all_fix_ids = {fix.id for fix in fix_plans}
        groups = []
        processed = set()
        
        for fix_id in all_fix_ids:
            if fix_id in processed:
                continue
            
            # Encontrar fixes que no tienen dependencias entre sí
            independent_group = [fix_id]
            processed.add(fix_id)
            
            for other_fix_id in all_fix_ids:
                if (other_fix_id not in processed and
                    not self._has_dependency_path(fix_id, other_fix_id, dependency_graph) and
                    not self._has_dependency_path(other_fix_id, fix_id, dependency_graph)):
                    independent_group.append(other_fix_id)
                    processed.add(other_fix_id)
            
            if len(independent_group) > 1:
                groups.append(independent_group)
        
        return groups
    
    def _has_dependency_path(self, from_fix: str, to_fix: str, graph: Dict[str, List[str]]) -> bool:
        """Verifica si existe path de dependencia entre fixes."""
        visited = set()
        
        def dfs(current: str) -> bool:
            if current == to_fix:
                return True
            if current in visited:
                return False
            
            visited.add(current)
            for neighbor in graph.get(current, []):
                if dfs(neighbor):
                    return True
            return False
        
        return dfs(from_fix)
    
    async def _generate_implementation_steps(self, cluster: IssueCluster) -> List[ImplementationStep]:
        """Genera pasos de implementación para cluster."""
        steps = []
        
        dominant_category = cluster.get_dominant_category()
        
        if dominant_category == IssueCategory.SECURITY:
            steps.extend([
                ImplementationStep(
                    step_number=1,
                    description="Conduct security assessment and create fix plan",
                    estimated_time_minutes=60,
                    skill_requirements=["security_expertise"],
                    validation_criteria=["Security review completed", "Threat model updated"]
                ),
                ImplementationStep(
                    step_number=2,
                    description="Implement security fixes",
                    estimated_time_minutes=int(cluster.estimated_batch_fix_time * 60 * 0.7),
                    skill_requirements=["secure_coding"],
                    validation_criteria=["Security tests pass", "No new vulnerabilities introduced"]
                ),
                ImplementationStep(
                    step_number=3,
                    description="Security testing and validation",
                    estimated_time_minutes=int(cluster.estimated_batch_fix_time * 60 * 0.3),
                    skill_requirements=["security_testing"],
                    validation_criteria=["Penetration testing completed", "Security scan passes"]
                )
            ])
        
        elif dominant_category == IssueCategory.PERFORMANCE:
            steps.extend([
                ImplementationStep(
                    step_number=1,
                    description="Establish performance baseline",
                    estimated_time_minutes=30,
                    tools_required=["profiling_tools", "monitoring"],
                    validation_criteria=["Baseline metrics captured"]
                ),
                ImplementationStep(
                    step_number=2,
                    description="Implement performance optimizations",
                    estimated_time_minutes=int(cluster.estimated_batch_fix_time * 60 * 0.8),
                    skill_requirements=["performance_optimization"],
                    validation_criteria=["Performance improvements measurable"]
                ),
                ImplementationStep(
                    step_number=3,
                    description="Performance testing and validation",
                    estimated_time_minutes=int(cluster.estimated_batch_fix_time * 60 * 0.2),
                    tools_required=["load_testing"],
                    validation_criteria=["Performance targets met", "No regression detected"]
                )
            ])
        
        elif dominant_category == IssueCategory.MAINTAINABILITY:
            steps.extend([
                ImplementationStep(
                    step_number=1,
                    description="Plan refactoring approach",
                    estimated_time_minutes=45,
                    skill_requirements=["software_design"],
                    validation_criteria=["Refactoring plan approved"]
                ),
                ImplementationStep(
                    step_number=2,
                    description="Execute refactoring",
                    estimated_time_minutes=int(cluster.estimated_batch_fix_time * 60 * 0.7),
                    skill_requirements=["refactoring"],
                    validation_criteria=["Code structure improved", "Complexity reduced"]
                ),
                ImplementationStep(
                    step_number=3,
                    description="Update tests and documentation",
                    estimated_time_minutes=int(cluster.estimated_batch_fix_time * 60 * 0.3),
                    validation_criteria=["Tests updated", "Documentation current"]
                )
            ])
        
        else:
            # Steps genéricos
            steps.extend([
                ImplementationStep(
                    step_number=1,
                    description=f"Analyze and plan {dominant_category.value} fixes",
                    estimated_time_minutes=30,
                    validation_criteria=["Fix plan documented"]
                ),
                ImplementationStep(
                    step_number=2,
                    description=f"Implement {dominant_category.value} fixes",
                    estimated_time_minutes=int(cluster.estimated_batch_fix_time * 60 * 0.8),
                    validation_criteria=["Issues resolved", "Quality improved"]
                ),
                ImplementationStep(
                    step_number=3,
                    description="Test and validate fixes",
                    estimated_time_minutes=int(cluster.estimated_batch_fix_time * 60 * 0.2),
                    validation_criteria=["All tests pass", "No regressions"]
                )
            ])
        
        return steps
    
    def _create_testing_strategy(self, cluster: IssueCluster) -> TestingStrategy:
        """Crea estrategia de testing para cluster."""
        dominant_category = cluster.get_dominant_category()
        
        # Configuración base
        strategy = TestingStrategy(
            unit_tests_required=True,
            regression_tests_required=True,
            estimated_test_effort_hours=cluster.estimated_batch_fix_time * 0.3
        )
        
        # Ajustes específicos por categoría
        if dominant_category == IssueCategory.SECURITY:
            strategy.security_tests_required = True
            strategy.estimated_test_effort_hours *= 1.5
            strategy.specific_test_cases = [
                "Security vulnerability tests",
                "Authentication bypass tests",
                "Input validation tests"
            ]
        
        elif dominant_category == IssueCategory.PERFORMANCE:
            strategy.performance_tests_required = True
            strategy.integration_tests_required = True
            strategy.specific_test_cases = [
                "Performance benchmark tests",
                "Load testing",
                "Memory usage tests"
            ]
        
        elif dominant_category == IssueCategory.RELIABILITY:
            strategy.integration_tests_required = True
            strategy.specific_test_cases = [
                "Error handling tests",
                "Edge case tests",
                "Failure recovery tests"
            ]
        
        return strategy
    
    def _create_rollback_plan(self, cluster: IssueCluster) -> RollbackPlan:
        """Crea plan de rollback para cluster."""
        dominant_category = cluster.get_dominant_category()
        
        # Configuración base
        plan = RollbackPlan(
            rollback_complexity="simple",
            rollback_time_minutes=30,
            rollback_steps=[
                "Revert code changes",
                "Restore previous configuration",
                "Validate system stability"
            ]
        )
        
        # Ajustes por categoría
        if dominant_category == IssueCategory.SECURITY:
            plan.rollback_complexity = "moderate"
            plan.rollback_time_minutes = 60
            plan.rollback_steps.extend([
                "Verify security posture",
                "Update security monitoring"
            ])
        
        elif dominant_category == IssueCategory.ARCHITECTURAL:
            plan.rollback_complexity = "complex"
            plan.rollback_time_minutes = 120
            plan.data_backup_required = True
            plan.rollback_risks = [
                "Data migration rollback required",
                "Service dependencies affected"
            ]
        
        elif dominant_category == IssueCategory.PERFORMANCE:
            plan.rollback_steps.extend([
                "Restore performance baseline",
                "Verify system performance"
            ])
        
        return plan


class ResourceEstimator:
    """Estimador de recursos para fixes."""
    
    def __init__(self):
        # Rates por tipo de skill (horas por punto de complejidad)
        self.skill_rates = {
            "junior_developer": 1.5,
            "senior_developer": 1.0,
            "security_expert": 0.8,
            "performance_engineer": 0.9,
            "architect": 0.7,
            "qa_engineer": 1.2
        }
        
        # Costo por hora por tipo de recurso (USD)
        self.hourly_rates = {
            "junior_developer": 50.0,
            "senior_developer": 80.0,
            "security_expert": 120.0,
            "performance_engineer": 100.0,
            "architect": 150.0,
            "qa_engineer": 70.0
        }
    
    async def estimate_resources(self, fix_plans: List[FixPlan]) -> ResourceEstimate:
        """
        Estima recursos necesarios para lista de fix plans.
        
        Args:
            fix_plans: Lista de planes de fix
            
        Returns:
            ResourceEstimate completa
        """
        total_hours = sum(plan.estimated_effort_hours for plan in fix_plans)
        
        # Calcular requerimientos de skills
        skill_requirements = defaultdict(float)
        tool_requirements = set()
        external_dependencies = set()
        
        for fix_plan in fix_plans:
            # Mapear tipo de fix a skills requeridos
            required_skills = self._map_fix_type_to_skills(fix_plan.fix_type)
            
            for skill in required_skills:
                skill_requirements[skill] += fix_plan.estimated_effort_hours
            
            # Recopilar tools de implementation steps
            for step in fix_plan.implementation_steps:
                tool_requirements.update(step.tools_required)
        
        # Calcular costo total
        total_cost = 0.0
        for skill, hours in skill_requirements.items():
            rate = self.hourly_rates.get(skill, 75.0)  # Default rate
            total_cost += hours * rate
        
        # Estimar timeline
        # Asumir 40 horas/semana por desarrollador
        max_parallel_work = max(skill_requirements.values()) if skill_requirements else total_hours
        timeline_weeks = math.ceil(max_parallel_work / 40.0)
        
        # Calcular team size óptimo
        team_size = max(1, math.ceil(total_hours / (timeline_weeks * 40.0)))
        
        return ResourceEstimate(
            total_hours=total_hours,
            total_cost=total_cost,
            skill_requirements=dict(skill_requirements),
            tool_requirements=list(tool_requirements),
            timeline_weeks=timeline_weeks,
            team_size_required=team_size,
            external_dependencies=list(external_dependencies)
        )
    
    def _map_fix_type_to_skills(self, fix_type: FixType) -> List[str]:
        """Mapea tipo de fix a skills requeridos."""
        skill_mapping = {
            FixType.CODE_CHANGE: ["senior_developer"],
            FixType.REFACTORING: ["senior_developer", "architect"],
            FixType.CONFIGURATION: ["senior_developer"],
            FixType.DOCUMENTATION: ["senior_developer"],
            FixType.ARCHITECTURAL: ["architect", "senior_developer"],
            FixType.PROCESS_CHANGE: ["architect"],
            FixType.TOOL_UPDATE: ["senior_developer"]
        }
        
        return skill_mapping.get(fix_type, ["senior_developer"])


class ROICalculator:
    """Calculador de ROI para fixes."""
    
    def __init__(self):
        self.hourly_rate = 75.0  # USD por hora desarrollador promedio
        self.annual_maintenance_cost_per_issue = 200.0  # USD por issue no resuelto
        self.bug_prevention_value = 500.0  # USD por bug evitado
        self.performance_improvement_value_per_percent = 1000.0  # USD por % de mejora
    
    async def calculate_roi_for_fixes(self, fix_plans: List[FixPlan]) -> List[ROIAnalysis]:
        """
        Calcula ROI para lista de fix plans.
        
        Args:
            fix_plans: Lista de planes de fix
            
        Returns:
            Lista de análisis de ROI
        """
        roi_analyses = []
        
        for fix_plan in fix_plans:
            roi_analysis = await self._calculate_single_fix_roi(fix_plan)
            roi_analyses.append(roi_analysis)
        
        return roi_analyses
    
    async def _calculate_single_fix_roi(self, fix_plan: FixPlan) -> ROIAnalysis:
        """Calcula ROI para un fix plan individual."""
        # Calcular inversión
        investment_cost = fix_plan.estimated_effort_hours * self.hourly_rate
        
        # Calcular beneficios
        benefits = await self._calculate_fix_benefits(fix_plan)
        
        # Calcular payback period
        annual_benefit = benefits.get_total_annual_value()
        payback_period_weeks = (investment_cost / (annual_benefit / 52.0)) if annual_benefit > 0 else 999.0
        
        # Calcular ROI
        if investment_cost > 0:
            roi_percentage = ((annual_benefit - investment_cost) / investment_cost) * 100.0
        else:
            roi_percentage = 0.0
        
        # NPV simplificado (3 años, 10% discount rate)
        discount_rate = 0.10
        npv = 0.0
        for year in range(1, 4):
            yearly_benefit = annual_benefit
            discounted_benefit = yearly_benefit / ((1 + discount_rate) ** year)
            npv += discounted_benefit
        npv -= investment_cost
        
        # ROI ajustado por riesgo
        risk_factor = 1.0 - (fix_plan.confidence_level * 0.2)  # Máximo 20% de descuento por riesgo
        risk_adjusted_roi = roi_percentage * risk_factor
        
        # Intervalo de confianza (± 25%)
        confidence_interval = (roi_percentage * 0.75, roi_percentage * 1.25)
        
        return ROIAnalysis(
            fix_plan_id=fix_plan.id,
            investment_hours=fix_plan.estimated_effort_hours,
            investment_cost=investment_cost,
            benefits=benefits,
            payback_period_weeks=payback_period_weeks,
            roi_score=roi_percentage,
            net_present_value=npv,
            risk_adjusted_roi=risk_adjusted_roi,
            confidence_interval=confidence_interval
        )
    
    async def _calculate_fix_benefits(self, fix_plan: FixPlan) -> FixBenefits:
        """Calcula beneficios de un fix."""
        benefits = FixBenefits()
        
        # Beneficios base por número de issues resueltos
        issue_count = len(fix_plan.affected_issues)
        
        # Ahorros en mantenimiento
        benefits.maintenance_cost_reduction = issue_count * self.annual_maintenance_cost_per_issue
        
        # Ganancia en productividad del desarrollador
        if fix_plan.fix_type in [FixType.REFACTORING, FixType.ARCHITECTURAL]:
            # Refactoring mejora productividad futura
            benefits.developer_productivity_gain = issue_count * 10.0  # 10 horas ahorradas por año por issue
        
        # Reducción de bugs estimada
        if fix_plan.fix_type in [FixType.CODE_CHANGE, FixType.REFACTORING]:
            benefits.bug_reduction_estimate = max(1, issue_count // 3)  # 1 bug evitado por cada 3 issues
        
        # Mejora de calidad (score 0-100)
        benefits.quality_improvement_score = min(100.0, issue_count * 5.0 + fix_plan.priority_score * 0.5)
        
        # Beneficios específicos por tipo de fix
        if fix_plan.fix_type == FixType.CODE_CHANGE:
            # Security y reliability fixes
            benefits.security_improvement_score = min(100.0, fix_plan.priority_score)
        
        elif fix_plan.fix_type == FixType.REFACTORING:
            # Performance y maintainability
            benefits.performance_improvement_percentage = min(20.0, issue_count * 2.0)
            benefits.developer_productivity_gain *= 1.5  # Refactoring es más valioso
        
        elif fix_plan.fix_type == FixType.DOCUMENTATION:
            # Mejora productividad y onboarding
            benefits.developer_productivity_gain = issue_count * 5.0
            benefits.user_satisfaction_improvement = 20.0
        
        return benefits


class FixStrategyGenerator:
    """Generador de estrategias de fix."""
    
    async def generate_fix_plan(self, cluster: IssueCluster) -> FixPlan:
        """
        Genera plan de fix para un cluster.
        
        Args:
            cluster: Cluster de issues
            
        Returns:
            FixPlan detallado
        """
        dominant_category = cluster.get_dominant_category()
        
        # Determinar tipo de fix
        fix_type = self._determine_fix_type(cluster, dominant_category)
        
        # Generar descripción
        description = self._generate_fix_description(cluster, fix_type)
        
        # Calcular effort y confidence
        estimated_effort = cluster.estimated_batch_fix_time
        confidence = cluster.cohesion_score * 0.8 + 0.2  # Base confidence de 20%
        
        # Generar pasos de implementación
        implementation_steps = await self._generate_implementation_steps(cluster, fix_type)
        
        # Crear estrategia de testing
        testing_strategy = self._create_testing_strategy(cluster, fix_type)
        
        # Crear plan de rollback
        rollback_plan = self._create_rollback_plan(cluster, fix_type)
        
        # Identificar prerequisites
        prerequisites = self._identify_prerequisites(cluster, fix_type)
        
        # Identificar riesgos
        risks = self._identify_risks(cluster, fix_type)
        
        # Calcular beneficios esperados
        expected_benefits = self._calculate_expected_benefits(cluster, fix_type)
        
        return FixPlan(
            cluster_id=cluster.id,
            fix_type=fix_type,
            title=f"Batch fix: {len(cluster.issues)} {dominant_category.value} issues",
            description=description,
            affected_issues=[issue.id for issue in cluster.issues],
            estimated_effort_hours=estimated_effort,
            confidence_level=confidence,
            priority_score=cluster.get_average_priority(),
            implementation_steps=implementation_steps,
            testing_strategy=testing_strategy,
            rollback_plan=rollback_plan,
            prerequisites=prerequisites,
            risks=risks,
            expected_benefits=expected_benefits
        )
    
    def _determine_fix_type(self, cluster: IssueCluster, dominant_category: IssueCategory) -> FixType:
        """Determina tipo de fix apropiado."""
        fix_type_mapping = {
            IssueCategory.SECURITY: FixType.CODE_CHANGE,
            IssueCategory.PERFORMANCE: FixType.REFACTORING,
            IssueCategory.MAINTAINABILITY: FixType.REFACTORING,
            IssueCategory.RELIABILITY: FixType.CODE_CHANGE,
            IssueCategory.DOCUMENTATION: FixType.DOCUMENTATION,
            IssueCategory.CODE_STYLE: FixType.CODE_CHANGE,
            IssueCategory.ARCHITECTURE: FixType.ARCHITECTURAL,
            IssueCategory.BEST_PRACTICES: FixType.REFACTORING
        }
        
        return fix_type_mapping.get(dominant_category, FixType.CODE_CHANGE)
    
    def _generate_fix_description(self, cluster: IssueCluster, fix_type: FixType) -> str:
        """Genera descripción del fix."""
        issue_count = len(cluster.issues)
        dominant_category = cluster.get_dominant_category()
        
        descriptions = {
            FixType.CODE_CHANGE: f"Implement code changes to resolve {issue_count} {dominant_category.value} issues",
            FixType.REFACTORING: f"Refactor code to address {issue_count} {dominant_category.value} concerns",
            FixType.DOCUMENTATION: f"Add comprehensive documentation for {issue_count} undocumented areas",
            FixType.ARCHITECTURAL: f"Redesign architecture to resolve {issue_count} structural issues",
            FixType.CONFIGURATION: f"Update configuration to fix {issue_count} configuration-related issues"
        }
        
        base_description = descriptions.get(fix_type, f"Address {issue_count} {dominant_category.value} issues")
        
        # Añadir contexto específico si hay características comunes
        if cluster.common_characteristics and cluster.common_characteristics.common_root_causes:
            root_cause = cluster.common_characteristics.common_root_causes[0]
            base_description += f" (root cause: {root_cause})"
        
        return base_description
    
    async def _generate_implementation_steps(self, cluster: IssueCluster, fix_type: FixType) -> List[ImplementationStep]:
        """Genera pasos de implementación específicos."""
        steps = []
        base_effort_minutes = int(cluster.estimated_batch_fix_time * 60)
        
        if fix_type == FixType.SECURITY:
            steps = [
                ImplementationStep(1, "Security assessment", base_effort_minutes // 4, ["security_expertise"]),
                ImplementationStep(2, "Implement security fixes", base_effort_minutes // 2, ["secure_coding"]),
                ImplementationStep(3, "Security testing", base_effort_minutes // 4, ["security_testing"])
            ]
        
        elif fix_type == FixType.REFACTORING:
            steps = [
                ImplementationStep(1, "Plan refactoring approach", base_effort_minutes // 5, ["software_design"]),
                ImplementationStep(2, "Execute refactoring", base_effort_minutes * 3 // 5, ["refactoring"]),
                ImplementationStep(3, "Update tests", base_effort_minutes // 5, ["testing"])
            ]
        
        elif fix_type == FixType.DOCUMENTATION:
            steps = [
                ImplementationStep(1, "Analyze documentation gaps", base_effort_minutes // 4),
                ImplementationStep(2, "Write documentation", base_effort_minutes * 3 // 4),
                ImplementationStep(3, "Review and publish", base_effort_minutes // 8)
            ]
        
        else:
            # Generic steps
            steps = [
                ImplementationStep(1, "Analysis and planning", base_effort_minutes // 4),
                ImplementationStep(2, "Implementation", base_effort_minutes // 2),
                ImplementationStep(3, "Testing and validation", base_effort_minutes // 4)
            ]
        
        return steps
    
    def _create_testing_strategy(self, cluster: IssueCluster, fix_type: FixType) -> TestingStrategy:
        """Crea estrategia de testing."""
        return TestingStrategy(
            unit_tests_required=True,
            integration_tests_required=fix_type in [FixType.REFACTORING, FixType.ARCHITECTURAL],
            regression_tests_required=True,
            performance_tests_required=fix_type == FixType.REFACTORING,
            security_tests_required=fix_type == FixType.CODE_CHANGE and 
                                   cluster.get_dominant_category() == IssueCategory.SECURITY,
            estimated_test_effort_hours=cluster.estimated_batch_fix_time * 0.4,
            test_coverage_target=80.0 if fix_type in [FixType.REFACTORING, FixType.ARCHITECTURAL] else 70.0
        )
    
    def _create_rollback_plan(self, cluster: IssueCluster, fix_type: FixType) -> RollbackPlan:
        """Crea plan de rollback."""
        complexity_mapping = {
            FixType.DOCUMENTATION: "simple",
            FixType.CODE_CHANGE: "simple",
            FixType.CONFIGURATION: "moderate",
            FixType.REFACTORING: "moderate",
            FixType.ARCHITECTURAL: "complex"
        }
        
        time_mapping = {
            FixType.DOCUMENTATION: 5,
            FixType.CODE_CHANGE: 15,
            FixType.CONFIGURATION: 30,
            FixType.REFACTORING: 45,
            FixType.ARCHITECTURAL: 120
        }
        
        return RollbackPlan(
            rollback_complexity=complexity_mapping.get(fix_type, "moderate"),
            rollback_time_minutes=time_mapping.get(fix_type, 30),
            rollback_steps=self._generate_rollback_steps(fix_type),
            data_backup_required=fix_type == FixType.ARCHITECTURAL,
            service_downtime_required=fix_type == FixType.ARCHITECTURAL
        )
    
    def _generate_rollback_steps(self, fix_type: FixType) -> List[str]:
        """Genera pasos de rollback."""
        base_steps = ["Revert code changes", "Run regression tests"]
        
        if fix_type == FixType.ARCHITECTURAL:
            base_steps.extend([
                "Restore database schema",
                "Update service configurations",
                "Verify system integration"
            ])
        elif fix_type == FixType.REFACTORING:
            base_steps.extend([
                "Verify API compatibility",
                "Check performance metrics"
            ])
        
        return base_steps
    
    def _identify_prerequisites(self, cluster: IssueCluster, fix_type: FixType) -> List[str]:
        """Identifica prerequisitos."""
        prerequisites = ["Code backup", "Test environment setup"]
        
        if fix_type == FixType.ARCHITECTURAL:
            prerequisites.extend([
                "Architecture review approval",
                "Migration plan documented",
                "Stakeholder approval"
            ])
        elif cluster.get_dominant_category() == IssueCategory.SECURITY:
            prerequisites.extend([
                "Security team consultation",
                "Threat assessment completed"
            ])
        
        return prerequisites
    
    def _identify_risks(self, cluster: IssueCluster, fix_type: FixType) -> List[str]:
        """Identifica riesgos del fix."""
        risks = ["Potential regression", "Testing effort underestimated"]
        
        if fix_type == FixType.ARCHITECTURAL:
            risks.extend([
                "System integration complexity",
                "Data migration risks",
                "Service dependencies affected"
            ])
        elif fix_type == FixType.REFACTORING:
            risks.extend([
                "Performance impact",
                "Behavioral changes"
            ])
        
        return risks
    
    def _calculate_expected_benefits(self, cluster: IssueCluster, fix_type: FixType) -> List[str]:
        """Calcula beneficios esperados."""
        benefits = []
        issue_count = len(cluster.issues)
        dominant_category = cluster.get_dominant_category()
        
        if dominant_category == IssueCategory.SECURITY:
            benefits.extend([
                "Reduced security vulnerabilities",
                "Improved compliance posture",
                "Lower security incident risk"
            ])
        
        elif dominant_category == IssueCategory.PERFORMANCE:
            benefits.extend([
                f"Performance improvement in {issue_count} areas",
                "Reduced resource usage",
                "Better user experience"
            ])
        
        elif dominant_category == IssueCategory.MAINTAINABILITY:
            benefits.extend([
                "Improved code maintainability",
                "Reduced technical debt",
                "Faster future development"
            ])
        
        # Beneficios generales
        benefits.extend([
            f"Resolution of {issue_count} code quality issues",
            "Improved overall code health",
            "Better team productivity"
        ])
        
        return benefits


class RemediationPlanner:
    """Planificador principal de remediación."""
    
    def __init__(self, config: Optional[RemediationConfig] = None):
        """
        Inicializa el planificador de remediación.
        
        Args:
            config: Configuración del planificador
        """
        self.config = config or RemediationConfig()
        self.dependency_analyzer = DependencyAnalyzer()
        self.fix_strategy_generator = FixStrategyGenerator()
        self.resource_estimator = ResourceEstimator()
        self.roi_calculator = ROICalculator()
    
    async def create_remediation_plan(self, clusters: List[IssueCluster]) -> RemediationPlan:
        """
        Crea plan completo de remediación.
        
        Args:
            clusters: Lista de clusters de issues
            
        Returns:
            RemediationPlan optimizado
        """
        start_time = time.time()
        
        logger.info(f"Creando plan de remediación para {len(clusters)} clusters")
        
        try:
            # 1. Generar fix plans para cada cluster
            fix_plans = []
            for cluster in clusters:
                fix_plan = await self.fix_strategy_generator.generate_fix_plan(cluster)
                fix_plans.append(fix_plan)
            
            # 2. Analizar dependencias entre fixes
            dependency_analysis = await self.dependency_analyzer.analyze_fix_dependencies(clusters)
            
            # 3. Estimar recursos necesarios
            resource_estimates = await self.resource_estimator.estimate_resources(fix_plans)
            
            # 4. Calcular ROI para cada fix
            roi_analyses = await self.roi_calculator.calculate_roi_for_fixes(fix_plans)
            
            # 5. Optimizar orden de ejecución
            execution_order = await self._optimize_execution_order(fix_plans, dependency_analysis.dependencies, roi_analyses)
            
            # 6. Crear planes de sprint
            sprint_plans = await self._create_sprint_plans(execution_order, resource_estimates)
            
            # 7. Calcular métricas del plan
            total_investment = sum(roi.investment_cost for roi in roi_analyses)
            expected_annual_savings = sum(roi.benefits.get_total_annual_value() for roi in roi_analyses)
            payback_period = (total_investment / (expected_annual_savings / 52.0)) if expected_annual_savings > 0 else 999.0
            
            # 8. Estimar mejora de calidad esperada
            expected_quality_improvement = await self._calculate_expected_quality_improvement(fix_plans)
            
            plan = RemediationPlan(
                title=f"Remediation plan for {sum(len(cluster.issues) for cluster in clusters)} issues",
                description=f"Comprehensive plan to address issues across {len(clusters)} clusters",
                fix_plans=fix_plans,
                dependencies=dependency_analysis.dependencies,
                resource_estimates=resource_estimates,
                roi_analyses=roi_analyses,
                execution_order=execution_order,
                sprint_plans=sprint_plans,
                total_estimated_effort_hours=resource_estimates.total_hours,
                expected_quality_improvement=expected_quality_improvement,
                total_investment=total_investment,
                expected_annual_savings=expected_annual_savings,
                payback_period_weeks=payback_period
            )
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Plan de remediación creado: {len(fix_plans)} fixes, "
                f"{len(sprint_plans)} sprints, ROI={((expected_annual_savings - total_investment) / total_investment * 100):.1f}% "
                f"en {total_time}ms"
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creando plan de remediación: {e}")
            raise
    
    async def _optimize_execution_order(self, fix_plans: List[FixPlan], dependencies: List[FixDependency],
                                      roi_analyses: List[ROIAnalysis]) -> List[FixExecutionStep]:
        """Optimiza orden de ejecución de fixes."""
        execution_steps = []
        completed_fixes = set()
        remaining_fixes = {fix.id: fix for fix in fix_plans}
        roi_map = {roi.fix_plan_id: roi for roi in roi_analyses}
        step_number = 1
        
        while remaining_fixes:
            # Encontrar fixes listos para ejecutar (sin dependencias pendientes)
            ready_fixes = []
            for fix_id, fix_plan in remaining_fixes.items():
                prerequisites_met = all(
                    dep.prerequisite_fix_id in completed_fixes
                    for dep in dependencies
                    if dep.dependent_fix_id == fix_id
                )
                
                if prerequisites_met:
                    ready_fixes.append((fix_id, fix_plan))
            
            if not ready_fixes:
                # Romper deadlock eligiendo el fix de mayor ROI
                best_fix = max(remaining_fixes.items(), key=lambda x: roi_map.get(x[0], ROIAnalysis("", 0, 0, FixBenefits(), 0, 0, 0, 0, (0, 0))).roi_score)
                ready_fixes = [best_fix]
            
            # Ordenar por ROI y prioridad
            ready_fixes.sort(key=lambda x: (
                roi_map.get(x[0], ROIAnalysis("", 0, 0, FixBenefits(), 0, 0, 0, 0, (0, 0))).roi_score,
                x[1].priority_score
            ), reverse=True)
            
            # Seleccionar fixes para este step (considerando paralelismo)
            step_fixes = []
            step_effort = 0.0
            
            for fix_id, fix_plan in ready_fixes:
                if (len(step_fixes) < self.config.max_parallel_fixes and
                    step_effort + fix_plan.estimated_effort_hours <= self.config.available_developer_hours_per_week):
                    step_fixes.append(fix_id)
                    step_effort += fix_plan.estimated_effort_hours
            
            if step_fixes:
                execution_step = FixExecutionStep(
                    step_number=step_number,
                    fix_ids=step_fixes,
                    estimated_duration_hours=step_effort,
                    parallel_execution=len(step_fixes) > 1,
                    prerequisites=self._get_step_prerequisites(step_fixes, dependencies),
                    risk_level=self._assess_step_risk_level(step_fixes, fix_plans),
                    validation_checkpoints=self._generate_validation_checkpoints(step_fixes, fix_plans)
                )
                
                execution_steps.append(execution_step)
                
                # Marcar fixes como completados
                for fix_id in step_fixes:
                    completed_fixes.add(fix_id)
                    del remaining_fixes[fix_id]
                
                step_number += 1
            else:
                break  # No se puede progresar más
        
        return execution_steps
    
    async def _create_sprint_plans(self, execution_order: List[FixExecutionStep], 
                                 resource_estimates: ResourceEstimate) -> List[SprintPlan]:
        """Crea planes de sprint."""
        sprint_plans = []
        current_sprint = 1
        current_sprint_steps = []
        current_sprint_effort = 0.0
        
        sprint_capacity = self.config.available_developer_hours_per_week * self.config.sprint_duration_weeks
        
        for step in execution_order:
            if current_sprint_effort + step.estimated_duration_hours > sprint_capacity:
                # Crear sprint plan actual
                if current_sprint_steps:
                    sprint_plan = await self._create_single_sprint_plan(
                        current_sprint, current_sprint_steps, current_sprint_effort
                    )
                    sprint_plans.append(sprint_plan)
                
                # Iniciar nuevo sprint
                current_sprint += 1
                current_sprint_steps = [step]
                current_sprint_effort = step.estimated_duration_hours
            else:
                current_sprint_steps.append(step)
                current_sprint_effort += step.estimated_duration_hours
        
        # Añadir último sprint si tiene contenido
        if current_sprint_steps:
            sprint_plan = await self._create_single_sprint_plan(
                current_sprint, current_sprint_steps, current_sprint_effort
            )
            sprint_plans.append(sprint_plan)
        
        return sprint_plans
    
    async def _create_single_sprint_plan(self, sprint_number: int, steps: List[FixExecutionStep],
                                       effort_hours: float) -> SprintPlan:
        """Crea plan de sprint individual."""
        # Calcular fecha de finalización
        start_date = datetime.now() + timedelta(weeks=(sprint_number - 1) * self.config.sprint_duration_weeks)
        completion_date = start_date + timedelta(weeks=self.config.sprint_duration_weeks)
        
        # Crear objetivos de calidad
        quality_goals = [
            QualityGoal(
                goal_name="Issue Resolution",
                metric_type="count",
                target_value=sum(len(step.fix_ids) for step in steps),
                measurement_method="Resolved issues count"
            ),
            QualityGoal(
                goal_name="Quality Improvement",
                metric_type="percentage",
                target_value=10.0,  # 10% improvement target
                measurement_method="Quality metrics comparison"
            )
        ]
        
        # Estrategias de mitigación de riesgo
        risk_strategies = [
            "Daily standup reviews",
            "Code review requirements",
            "Automated testing pipeline",
            "Incremental delivery approach"
        ]
        
        # Añadir estrategias específicas según tipos de fix
        fix_types = set()
        for step in steps:
            # En implementación real, se obtendría del fix_plan
            fix_types.add("general")
        
        if "security" in str(fix_types).lower():
            risk_strategies.append("Security review checkpoints")
        
        return SprintPlan(
            sprint_number=sprint_number,
            title=f"Sprint {sprint_number}: Quality Improvements",
            execution_steps=steps,
            total_effort_hours=effort_hours,
            team_capacity_hours=self.config.available_developer_hours_per_week * self.config.sprint_duration_weeks,
            expected_completion_date=completion_date,
            quality_goals=quality_goals,
            success_criteria=[
                "All planned fixes completed",
                "No critical regressions introduced",
                "Quality gates passed",
                "Team velocity maintained"
            ],
            risk_mitigation_strategies=risk_strategies
        )
    
    async def _calculate_expected_quality_improvement(self, fix_plans: List[FixPlan]) -> float:
        """Calcula mejora de calidad esperada."""
        total_improvement = 0.0
        
        for fix_plan in fix_plans:
            # Mejora basada en número de issues y prioridad
            issue_count = len(fix_plan.affected_issues)
            priority_weight = fix_plan.priority_score / 100.0
            
            fix_improvement = issue_count * priority_weight * 5.0  # 5 puntos por issue ponderado por prioridad
            total_improvement += fix_improvement
        
        # Normalizar a porcentaje
        return min(50.0, total_improvement)  # Cap en 50% improvement
    
    def _get_step_prerequisites(self, fix_ids: List[str], dependencies: List[FixDependency]) -> List[str]:
        """Obtiene prerequisitos para un step."""
        prerequisites = set()
        
        for fix_id in fix_ids:
            for dep in dependencies:
                if dep.dependent_fix_id == fix_id:
                    prerequisites.add(f"Complete fix {dep.prerequisite_fix_id}")
        
        return list(prerequisites)
    
    def _assess_step_risk_level(self, fix_ids: List[str], fix_plans: List[FixPlan]) -> str:
        """Evalúa nivel de riesgo del step."""
        # Obtener fix plans correspondientes
        step_fix_plans = [plan for plan in fix_plans if plan.id in fix_ids]
        
        if not step_fix_plans:
            return "low"
        
        # Calcular riesgo basado en tipos de fix y effort
        total_effort = sum(plan.estimated_effort_hours for plan in step_fix_plans)
        avg_confidence = sum(plan.confidence_level for plan in step_fix_plans) / len(step_fix_plans)
        
        # Determinar riesgo
        if total_effort > 20.0 or avg_confidence < 0.6:
            return "high"
        elif total_effort > 10.0 or avg_confidence < 0.8:
            return "medium"
        else:
            return "low"
    
    def _generate_validation_checkpoints(self, fix_ids: List[str], fix_plans: List[FixPlan]) -> List[str]:
        """Genera checkpoints de validación."""
        checkpoints = [
            "Pre-implementation review completed",
            "Implementation completed successfully",
            "Unit tests passing",
            "Integration tests passing",
            "Code review approved",
            "Quality gates passed",
            "No critical regressions detected"
        ]
        
        # Añadir checkpoints específicos
        step_fix_plans = [plan for plan in fix_plans if plan.id in fix_ids]
        
        if any(plan.fix_type == FixType.ARCHITECTURAL for plan in step_fix_plans):
            checkpoints.extend([
                "Architecture review completed",
                "Migration testing completed"
            ])
        
        if any(plan.fix_type == FixType.CODE_CHANGE for plan in step_fix_plans):
            checkpoints.append("Security scan completed")
        
        return checkpoints
