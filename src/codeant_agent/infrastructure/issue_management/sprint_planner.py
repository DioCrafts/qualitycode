"""
Implementación del planificador de sprints.

Este módulo implementa la planificación automática de sprints
basada en capacidad del equipo, dependencias y prioridades.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import time
from datetime import datetime, timedelta

from ...domain.entities.issue_management import (
    RemediationPlan, SprintPlan, FixExecutionStep, QualityGoal,
    RemediationConfig, FixPlan, PriorityLevel
)

logger = logging.getLogger(__name__)


@dataclass
class SprintCapacity:
    """Capacidad de sprint."""
    total_hours: float
    available_hours: float
    allocated_hours: float
    buffer_hours: float
    utilization_percentage: float
    
    def can_accommodate(self, additional_hours: float) -> bool:
        """Verifica si puede acomodar horas adicionales."""
        return self.allocated_hours + additional_hours <= self.available_hours
    
    def get_remaining_capacity(self) -> float:
        """Obtiene capacidad restante."""
        return max(0.0, self.available_hours - self.allocated_hours)


@dataclass
class SprintOptimizationResult:
    """Resultado de optimización de sprint."""
    optimized_sprints: List[SprintPlan]
    optimization_score: float
    capacity_utilization: List[float]
    quality_improvement_trajectory: List[float]
    optimization_time_ms: int


class SprintPlanner:
    """Planificador de sprints para remediación."""
    
    def __init__(self, config: Optional[RemediationConfig] = None):
        """
        Inicializa el planificador de sprints.
        
        Args:
            config: Configuración de remediación
        """
        self.config = config or RemediationConfig()
        self.sprint_capacity_hours = (
            self.config.available_developer_hours_per_week * 
            self.config.sprint_duration_weeks * 
            0.8  # 80% utilization target
        )
    
    async def create_optimized_sprint_plan(self, remediation_plan: RemediationPlan) -> SprintOptimizationResult:
        """
        Crea plan de sprints optimizado.
        
        Args:
            remediation_plan: Plan de remediación base
            
        Returns:
            SprintOptimizationResult optimizado
        """
        start_time = time.time()
        
        logger.info(f"Optimizando plan de sprints para {len(remediation_plan.fix_plans)} fixes")
        
        try:
            # 1. Analizar capacidad disponible
            sprint_capacities = await self._analyze_sprint_capacities(remediation_plan)
            
            # 2. Distribuir trabajo en sprints óptimamente
            optimized_sprints = await self._optimize_sprint_distribution(
                remediation_plan.execution_order, 
                sprint_capacities
            )
            
            # 3. Calcular métricas de optimización
            optimization_score = await self._calculate_optimization_score(optimized_sprints, sprint_capacities)
            
            # 4. Calcular utilización de capacidad por sprint
            capacity_utilization = [sprint.get_capacity_utilization() for sprint in optimized_sprints]
            
            # 5. Proyectar mejora de calidad a lo largo del tiempo
            quality_trajectory = await self._project_quality_improvement(optimized_sprints, remediation_plan)
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Optimización de sprints completada: {len(optimized_sprints)} sprints, "
                f"score={optimization_score:.2f}, utilización promedio={sum(capacity_utilization)/len(capacity_utilization):.1f}% "
                f"en {total_time}ms"
            )
            
            return SprintOptimizationResult(
                optimized_sprints=optimized_sprints,
                optimization_score=optimization_score,
                capacity_utilization=capacity_utilization,
                quality_improvement_trajectory=quality_trajectory,
                optimization_time_ms=total_time
            )
            
        except Exception as e:
            logger.error(f"Error optimizando sprints: {e}")
            raise
    
    async def _analyze_sprint_capacities(self, remediation_plan: RemediationPlan) -> List[SprintCapacity]:
        """Analiza capacidades de sprints."""
        total_effort = remediation_plan.total_estimated_effort_hours
        estimated_sprints = max(1, math.ceil(total_effort / self.sprint_capacity_hours))
        
        capacities = []
        for sprint_num in range(1, estimated_sprints + 1):
            # Calcular capacidad con buffer para riesgos
            buffer_percentage = 0.2  # 20% buffer
            available_hours = self.sprint_capacity_hours
            buffer_hours = available_hours * buffer_percentage
            
            capacity = SprintCapacity(
                total_hours=self.config.available_developer_hours_per_week * self.config.sprint_duration_weeks,
                available_hours=available_hours,
                allocated_hours=0.0,
                buffer_hours=buffer_hours,
                utilization_percentage=0.0
            )
            
            capacities.append(capacity)
        
        return capacities
    
    async def _optimize_sprint_distribution(self, execution_order: List[FixExecutionStep],
                                          capacities: List[SprintCapacity]) -> List[SprintPlan]:
        """Optimiza distribución de trabajo en sprints."""
        optimized_sprints = []
        current_sprint_index = 0
        current_sprint_steps = []
        
        for step in execution_order:
            # Verificar si el step cabe en el sprint actual
            if current_sprint_index < len(capacities):
                current_capacity = capacities[current_sprint_index]
                
                if current_capacity.can_accommodate(step.estimated_duration_hours):
                    # Añadir step al sprint actual
                    current_sprint_steps.append(step)
                    current_capacity.allocated_hours += step.estimated_duration_hours
                    current_capacity.utilization_percentage = (
                        current_capacity.allocated_hours / current_capacity.available_hours
                    ) * 100.0
                else:
                    # Crear sprint plan para sprint actual
                    if current_sprint_steps:
                        sprint_plan = await self._create_optimized_sprint_plan(
                            current_sprint_index + 1, current_sprint_steps, current_capacity
                        )
                        optimized_sprints.append(sprint_plan)
                    
                    # Mover al siguiente sprint
                    current_sprint_index += 1
                    current_sprint_steps = [step]
                    
                    # Asegurar que tenemos capacidad
                    if current_sprint_index >= len(capacities):
                        # Añadir capacidad adicional si es necesario
                        additional_capacity = SprintCapacity(
                            total_hours=self.sprint_capacity_hours / 0.8,  # Total capacity
                            available_hours=self.sprint_capacity_hours,
                            allocated_hours=step.estimated_duration_hours,
                            buffer_hours=self.sprint_capacity_hours * 0.2,
                            utilization_percentage=(step.estimated_duration_hours / self.sprint_capacity_hours) * 100.0
                        )
                        capacities.append(additional_capacity)
                    else:
                        capacities[current_sprint_index].allocated_hours = step.estimated_duration_hours
                        capacities[current_sprint_index].utilization_percentage = (
                            step.estimated_duration_hours / capacities[current_sprint_index].available_hours
                        ) * 100.0
        
        # Crear sprint plan final si queda trabajo
        if current_sprint_steps and current_sprint_index < len(capacities):
            sprint_plan = await self._create_optimized_sprint_plan(
                current_sprint_index + 1, current_sprint_steps, capacities[current_sprint_index]
            )
            optimized_sprints.append(sprint_plan)
        
        return optimized_sprints
    
    async def _create_optimized_sprint_plan(self, sprint_number: int, steps: List[FixExecutionStep],
                                         capacity: SprintCapacity) -> SprintPlan:
        """Crea plan de sprint optimizado."""
        # Calcular fechas
        start_date = datetime.now() + timedelta(weeks=(sprint_number - 1) * self.config.sprint_duration_weeks)
        completion_date = start_date + timedelta(weeks=self.config.sprint_duration_weeks)
        
        # Crear objetivos de calidad adaptativos
        quality_goals = await self._create_adaptive_quality_goals(steps, sprint_number)
        
        # Generar criterios de éxito específicos
        success_criteria = await self._generate_success_criteria(steps)
        
        # Estrategias de mitigación de riesgo
        risk_strategies = await self._generate_risk_mitigation_strategies(steps, capacity)
        
        return SprintPlan(
            sprint_number=sprint_number,
            title=f"Sprint {sprint_number}: Quality Remediation",
            execution_steps=steps,
            total_effort_hours=capacity.allocated_hours,
            team_capacity_hours=capacity.total_hours,
            expected_completion_date=completion_date,
            quality_goals=quality_goals,
            success_criteria=success_criteria,
            risk_mitigation_strategies=risk_strategies
        )
    
    async def _create_adaptive_quality_goals(self, steps: List[FixExecutionStep], sprint_number: int) -> List[QualityGoal]:
        """Crea objetivos de calidad adaptativos."""
        goals = []
        
        # Objetivo base: completar todos los fixes
        total_fixes = sum(len(step.fix_ids) for step in steps)
        goals.append(QualityGoal(
            goal_name="Fix Completion Rate",
            metric_type="percentage",
            target_value=100.0,
            current_value=0.0,
            improvement_percentage=100.0,
            measurement_method="Completed fixes / Total planned fixes"
        ))
        
        # Objetivo de mejora de calidad (adaptativo por sprint)
        base_improvement = 5.0  # 5% base
        sprint_multiplier = min(2.0, 1.0 + (sprint_number - 1) * 0.2)  # Incrementa con sprints
        quality_improvement_target = base_improvement * sprint_multiplier
        
        goals.append(QualityGoal(
            goal_name="Quality Metrics Improvement",
            metric_type="percentage",
            target_value=quality_improvement_target,
            current_value=0.0,
            improvement_percentage=quality_improvement_target,
            measurement_method="Quality index comparison before/after sprint"
        ))
        
        # Objetivo de reducción de deuda técnica
        if total_fixes > 5:
            debt_reduction_target = min(20.0, total_fixes * 2.0)
            goals.append(QualityGoal(
                goal_name="Technical Debt Reduction",
                metric_type="percentage", 
                target_value=debt_reduction_target,
                current_value=0.0,
                measurement_method="Technical debt hours before/after sprint"
            ))
        
        # Objetivo de velocity (para sprints posteriores)
        if sprint_number > 1:
            goals.append(QualityGoal(
                goal_name="Team Velocity Maintenance",
                metric_type="points",
                target_value=self.config.team_velocity_points,
                current_value=self.config.team_velocity_points * 0.9,  # Assume 90% current
                measurement_method="Story points completed per sprint"
            ))
        
        return goals
    
    async def _generate_success_criteria(self, steps: List[FixExecutionStep]) -> List[str]:
        """Genera criterios de éxito específicos."""
        criteria = [
            "All planned execution steps completed successfully",
            "No critical regressions introduced",
            "All quality gates pass",
            "Code review approval for all changes"
        ]
        
        # Criterios específicos basados en tipos de steps
        has_security_steps = any("security" in str(step.fix_ids).lower() for step in steps)
        has_performance_steps = any("performance" in str(step.fix_ids).lower() for step in steps)
        has_parallel_steps = any(step.parallel_execution for step in steps)
        
        if has_security_steps:
            criteria.extend([
                "Security review completed and approved",
                "Security tests pass",
                "No new security vulnerabilities introduced"
            ])
        
        if has_performance_steps:
            criteria.extend([
                "Performance benchmarks meet targets",
                "No performance regression detected",
                "Resource usage within acceptable limits"
            ])
        
        if has_parallel_steps:
            criteria.extend([
                "Parallel execution coordination successful",
                "No conflicts between parallel fixes",
                "Integration testing passes for all parallel changes"
            ])
        
        return criteria
    
    async def _generate_risk_mitigation_strategies(self, steps: List[FixExecutionStep], 
                                                 capacity: SprintCapacity) -> List[str]:
        """Genera estrategias de mitigación de riesgo."""
        strategies = [
            "Daily standup meetings to track progress",
            "Code review requirements for all changes",
            "Automated testing pipeline validation",
            "Incremental delivery with validation checkpoints"
        ]
        
        # Estrategias basadas en utilización
        if capacity.utilization_percentage > 90:
            strategies.extend([
                "Monitor team capacity closely - high utilization",
                "Have backup resources identified",
                "Consider scope reduction if needed"
            ])
        
        elif capacity.utilization_percentage < 60:
            strategies.extend([
                "Look for additional quality improvements",
                "Consider accelerating timeline",
                "Include technical debt reduction tasks"
            ])
        
        # Estrategias basadas en complejidad
        high_risk_steps = [step for step in steps if step.risk_level == "high"]
        if high_risk_steps:
            strategies.extend([
                "Extra review for high-risk changes",
                "Staged rollout for complex fixes",
                "Enhanced monitoring for critical changes"
            ])
        
        return strategies
    
    async def _calculate_optimization_score(self, sprints: List[SprintPlan], 
                                          capacities: List[SprintCapacity]) -> float:
        """Calcula score de optimización del plan."""
        if not sprints or not capacities:
            return 0.0
        
        # Métricas de optimización
        metrics = {
            "capacity_utilization": 0.0,
            "load_balancing": 0.0,
            "dependency_optimization": 0.0,
            "timeline_efficiency": 0.0
        }
        
        # 1. Utilización de capacidad (penalizar sobre/sub utilización)
        utilizations = [sprint.get_capacity_utilization() for sprint in sprints]
        target_utilization = 80.0
        
        avg_utilization = sum(utilizations) / len(utilizations)
        utilization_score = 100.0 - abs(avg_utilization - target_utilization)
        metrics["capacity_utilization"] = max(0.0, utilization_score)
        
        # 2. Balance de carga entre sprints
        utilization_variance = sum((u - avg_utilization) ** 2 for u in utilizations) / len(utilizations)
        balance_score = 100.0 - min(100.0, utilization_variance)
        metrics["load_balancing"] = balance_score
        
        # 3. Optimización de dependencias (sprints con menos bloqueos)
        dependency_score = 80.0  # Base score (simplificado)
        metrics["dependency_optimization"] = dependency_score
        
        # 4. Eficiencia de timeline
        timeline_efficiency = 100.0 - min(50.0, len(sprints) * 5.0)  # Penalizar sprints muy largos
        metrics["timeline_efficiency"] = max(50.0, timeline_efficiency)
        
        # Score general
        overall_score = sum(metrics.values()) / len(metrics)
        return overall_score
    
    async def _project_quality_improvement(self, sprints: List[SprintPlan], 
                                         remediation_plan: RemediationPlan) -> List[float]:
        """Proyecta mejora de calidad a lo largo de sprints."""
        trajectory = [0.0]  # Start at 0% improvement
        
        total_issues = sum(len(plan.affected_issues) for plan in remediation_plan.fix_plans)
        cumulative_issues_fixed = 0
        
        for sprint in sprints:
            # Estimar issues resueltos en este sprint
            sprint_fixes = sum(len(step.fix_ids) for step in sprint.execution_steps)
            cumulative_issues_fixed += sprint_fixes
            
            # Calcular mejora porcentual acumulativa
            improvement_percentage = (cumulative_issues_fixed / total_issues) * remediation_plan.expected_quality_improvement
            trajectory.append(improvement_percentage)
        
        return trajectory
    
    def generate_sprint_summary(self, optimization_result: SprintOptimizationResult) -> Dict[str, Any]:
        """
        Genera resumen del plan de sprints.
        
        Returns:
            Diccionario con resumen ejecutivo
        """
        sprints = optimization_result.optimized_sprints
        
        summary = {
            "sprint_overview": {
                "total_sprints": len(sprints),
                "total_duration_weeks": len(sprints) * self.config.sprint_duration_weeks,
                "optimization_score": optimization_result.optimization_score,
                "average_capacity_utilization": sum(optimization_result.capacity_utilization) / len(optimization_result.capacity_utilization) if optimization_result.capacity_utilization else 0.0
            },
            "capacity_analysis": {
                "peak_utilization": max(optimization_result.capacity_utilization) if optimization_result.capacity_utilization else 0.0,
                "minimum_utilization": min(optimization_result.capacity_utilization) if optimization_result.capacity_utilization else 0.0,
                "utilization_variance": self._calculate_variance(optimization_result.capacity_utilization),
                "overallocated_sprints": sum(1 for sprint in sprints if sprint.is_overallocated())
            },
            "quality_projections": {
                "final_quality_improvement": optimization_result.quality_improvement_trajectory[-1] if optimization_result.quality_improvement_trajectory else 0.0,
                "improvement_per_sprint": self._calculate_improvement_velocity(optimization_result.quality_improvement_trajectory),
                "projected_completion_date": sprints[-1].expected_completion_date if sprints else None
            },
            "risk_assessment": {
                "high_risk_sprints": sum(1 for sprint in sprints if self._assess_sprint_risk(sprint) == "high"),
                "parallel_execution_complexity": sum(1 for sprint in sprints if any(step.parallel_execution for step in sprint.execution_steps)),
                "resource_constraints": self._identify_resource_constraints(sprints)
            },
            "recommendations": self._generate_sprint_recommendations(sprints, optimization_result)
        }
        
        return summary
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calcula varianza de lista de valores."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _calculate_improvement_velocity(self, trajectory: List[float]) -> float:
        """Calcula velocidad de mejora por sprint."""
        if len(trajectory) < 2:
            return 0.0
        
        improvements = [trajectory[i] - trajectory[i-1] for i in range(1, len(trajectory))]
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def _assess_sprint_risk(self, sprint: SprintPlan) -> str:
        """Evalúa riesgo de un sprint."""
        risk_factors = 0
        
        # Factor: alta utilización
        if sprint.get_capacity_utilization() > 90:
            risk_factors += 2
        elif sprint.get_capacity_utilization() > 80:
            risk_factors += 1
        
        # Factor: muchos execution steps
        if len(sprint.execution_steps) > 5:
            risk_factors += 1
        
        # Factor: parallel execution
        if any(step.parallel_execution for step in sprint.execution_steps):
            risk_factors += 1
        
        # Factor: steps de alto riesgo
        high_risk_steps = sum(1 for step in sprint.execution_steps if step.risk_level == "high")
        risk_factors += high_risk_steps
        
        if risk_factors >= 4:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"
    
    def _identify_resource_constraints(self, sprints: List[SprintPlan]) -> List[str]:
        """Identifica limitaciones de recursos."""
        constraints = []
        
        # Sprints sobrecargados
        overallocated = sum(1 for sprint in sprints if sprint.is_overallocated())
        if overallocated > 0:
            constraints.append(f"{overallocated} sprints are overallocated")
        
        # Sprints muy largos
        long_sprints = sum(1 for sprint in sprints if len(sprint.execution_steps) > 6)
        if long_sprints > 0:
            constraints.append(f"{long_sprints} sprints have high complexity")
        
        # Timeline muy largo
        if len(sprints) > 8:
            constraints.append("Long timeline - consider parallel teams or scope reduction")
        
        return constraints
    
    def _generate_sprint_recommendations(self, sprints: List[SprintPlan], 
                                       optimization_result: SprintOptimizationResult) -> List[str]:
        """Genera recomendaciones para el plan de sprints."""
        recommendations = []
        
        # Recomendaciones basadas en optimización
        if optimization_result.optimization_score < 70:
            recommendations.append("Consider rebalancing workload across sprints")
        
        # Recomendaciones basadas en utilización
        avg_utilization = sum(optimization_result.capacity_utilization) / len(optimization_result.capacity_utilization)
        
        if avg_utilization > 85:
            recommendations.extend([
                "High team utilization - monitor burn-out risk",
                "Consider adding buffer time for unexpected issues"
            ])
        elif avg_utilization < 65:
            recommendations.extend([
                "Low utilization - opportunity for additional improvements",
                "Consider accelerating timeline or adding scope"
            ])
        
        # Recomendaciones basadas en trajectory
        if optimization_result.quality_improvement_trajectory:
            final_improvement = optimization_result.quality_improvement_trajectory[-1]
            if final_improvement < 15:
                recommendations.append("Consider adding more high-impact fixes to increase quality gains")
        
        # Recomendaciones de riesgo
        high_risk_sprints = sum(1 for sprint in sprints if self._assess_sprint_risk(sprint) == "high")
        if high_risk_sprints > len(sprints) // 3:
            recommendations.append("High percentage of risky sprints - consider spreading work more evenly")
        
        return recommendations
