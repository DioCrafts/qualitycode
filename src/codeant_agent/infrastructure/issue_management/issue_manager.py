"""
Gestor principal de issues - Orquestador del sistema completo.

Este módulo implementa el gestor principal que coordina todos los
componentes del sistema de categorización y priorización.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import time
from datetime import datetime
import math

from ...domain.entities.issue_management import (
    RawIssue, CategorizedIssue, IssueCluster, RemediationPlan,
    IssueAnalysisResult, AnalysisSummary, IssueManagerConfig,
    PriorityDistribution, IssueCategory, IssueSeverity, PriorityLevel
)

from .issue_categorizer import IssueCategorizer
from .priority_calculator import PriorityCalculator
from .clustering_engine import ClusteringEngine
from .business_analyzer import BusinessImpactAnalyzer
from .remediation_planner import RemediationPlanner
from .fix_generator import FixRecommendationEngine
from .sprint_planner import SprintPlanner

logger = logging.getLogger(__name__)


@dataclass
class ProcessingSession:
    """Sesión de procesamiento de issues."""
    session_id: str
    start_time: datetime
    issues_processed: int = 0
    issues_categorized: int = 0
    issues_prioritized: int = 0
    clusters_created: int = 0
    plans_generated: int = 0
    errors_encountered: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Añade error a la sesión."""
        self.errors_encountered.append(error)
    
    def get_success_rate(self) -> float:
        """Obtiene tasa de éxito."""
        return (self.issues_processed - len(self.errors_encountered)) / max(1, self.issues_processed)


class IssueManager:
    """Gestor principal del sistema de issues."""
    
    def __init__(self, config: Optional[IssueManagerConfig] = None):
        """
        Inicializa el gestor principal.
        
        Args:
            config: Configuración del gestor
        """
        self.config = config or IssueManagerConfig()
        
        # Inicializar componentes especializados
        self.issue_categorizer = IssueCategorizer(self.config.categorization_config)
        self.priority_calculator = PriorityCalculator(self.config.priority_config)
        self.clustering_engine = ClusteringEngine(self.config.clustering_config)
        self.business_analyzer = BusinessImpactAnalyzer()
        self.remediation_planner = RemediationPlanner(self.config.remediation_config)
        self.fix_recommendation_engine = FixRecommendationEngine()
        self.sprint_planner = SprintPlanner(self.config.remediation_config)
    
    async def process_issues(self, raw_issues: List[RawIssue]) -> IssueAnalysisResult:
        """
        Procesa lista completa de issues raw.
        
        Args:
            raw_issues: Lista de issues sin procesar
            
        Returns:
            IssueAnalysisResult completo con toda la información
        """
        start_time = time.time()
        session = ProcessingSession(
            session_id=f"session_{int(time.time())}",
            start_time=datetime.utcnow(),
            issues_processed=len(raw_issues)
        )
        
        try:
            logger.info(f"Iniciando procesamiento de {len(raw_issues)} issues")
            
            # 1. Categorizar issues
            logger.debug("Paso 1: Categorizando issues...")
            categorized_issues = await self._categorize_issues_with_error_handling(raw_issues, session)
            session.issues_categorized = len(categorized_issues)
            
            # 2. Calcular prioridades
            logger.debug("Paso 2: Calculando prioridades...")
            prioritized_issues = await self._calculate_priorities_with_error_handling(categorized_issues, session)
            session.issues_prioritized = len(prioritized_issues)
            
            # 3. Crear clusters de issues similares
            logger.debug("Paso 3: Creando clusters...")
            clusters = await self._create_clusters_with_error_handling(prioritized_issues, session)
            session.clusters_created = len(clusters)
            
            # 4. Crear plan de remediación
            logger.debug("Paso 4: Creando plan de remediación...")
            remediation_plan = await self._create_remediation_plan_with_error_handling(clusters, session)
            if remediation_plan:
                session.plans_generated = 1
            
            # 5. Calcular métricas y distribuciones
            priority_distribution = PriorityDistribution.from_issues(prioritized_issues)
            category_distribution = self._calculate_category_distribution(prioritized_issues)
            
            # 6. Crear resumen de análisis
            analysis_summary = await self._create_analysis_summary(
                prioritized_issues, clusters, remediation_plan, session
            )
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Procesamiento completado: {session.issues_categorized} categorizados, "
                f"{session.issues_prioritized} priorizados, {session.clusters_created} clusters, "
                f"tasa de éxito={session.get_success_rate():.2%} en {total_time}ms"
            )
            
            return IssueAnalysisResult(
                categorized_issues=prioritized_issues,
                issue_clusters=clusters,
                remediation_plan=remediation_plan,
                priority_distribution=priority_distribution,
                category_distribution=category_distribution,
                analysis_summary=analysis_summary,
                analysis_time_ms=total_time
            )
            
        except Exception as e:
            logger.error(f"Error en procesamiento de issues: {e}")
            session.add_error(f"Critical processing error: {e}")
            
            # Retornar resultado parcial
            return IssueAnalysisResult(
                categorized_issues=[],
                analysis_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _categorize_issues_with_error_handling(self, raw_issues: List[RawIssue], 
                                                   session: ProcessingSession) -> List[CategorizedIssue]:
        """Categoriza issues con manejo de errores."""
        try:
            return await self.issue_categorizer.categorize_issues(raw_issues)
        except Exception as e:
            logger.error(f"Error en categorización: {e}")
            session.add_error(f"Categorization failed: {e}")
            # Fallback: crear categorizaciones básicas
            return await self._create_fallback_categorizations(raw_issues)
    
    async def _calculate_priorities_with_error_handling(self, categorized_issues: List[CategorizedIssue],
                                                      session: ProcessingSession) -> List[CategorizedIssue]:
        """Calcula prioridades con manejo de errores."""
        try:
            priority_result = await self.priority_calculator.calculate_priorities(categorized_issues)
            return priority_result.prioritized_issues
        except Exception as e:
            logger.error(f"Error en priorización: {e}")
            session.add_error(f"Prioritization failed: {e}")
            # Fallback: usar prioridades básicas
            return await self._assign_fallback_priorities(categorized_issues)
    
    async def _create_clusters_with_error_handling(self, prioritized_issues: List[CategorizedIssue],
                                                 session: ProcessingSession) -> List[IssueCluster]:
        """Crea clusters con manejo de errores."""
        try:
            if self.config.clustering_config.enable_hierarchical_clustering:
                clustering_result = await self.clustering_engine.cluster_issues(prioritized_issues)
                return clustering_result.clusters
            else:
                logger.info("Clustering deshabilitado en configuración")
                return []
        except Exception as e:
            logger.error(f"Error en clustering: {e}")
            session.add_error(f"Clustering failed: {e}")
            return []
    
    async def _create_remediation_plan_with_error_handling(self, clusters: List[IssueCluster],
                                                         session: ProcessingSession) -> Optional[RemediationPlan]:
        """Crea plan de remediación con manejo de errores."""
        try:
            if clusters:
                return await self.remediation_planner.create_remediation_plan(clusters)
            else:
                logger.info("No hay clusters para crear plan de remediación")
                return None
        except Exception as e:
            logger.error(f"Error creando plan de remediación: {e}")
            session.add_error(f"Remediation planning failed: {e}")
            return None
    
    async def _create_fallback_categorizations(self, raw_issues: List[RawIssue]) -> List[CategorizedIssue]:
        """Crea categorizaciones fallback."""
        fallback_issues = []
        
        for raw_issue in raw_issues:
            # Categorización simple basada en keywords
            category = IssueCategory.MAINTAINABILITY  # Default
            
            message_lower = raw_issue.message.lower()
            if any(keyword in message_lower for keyword in ['security', 'vulnerability', 'exploit']):
                category = IssueCategory.SECURITY
            elif any(keyword in message_lower for keyword in ['performance', 'slow', 'memory']):
                category = IssueCategory.PERFORMANCE
            elif any(keyword in message_lower for keyword in ['error', 'exception', 'crash']):
                category = IssueCategory.RELIABILITY
            
            categorized = CategorizedIssue(
                original_issue=raw_issue,
                primary_category=category,
                tags=["fallback_categorization"],
                confidence_scores={category.value: 0.5}
            )
            
            fallback_issues.append(categorized)
        
        return fallback_issues
    
    async def _assign_fallback_priorities(self, categorized_issues: List[CategorizedIssue]) -> List[CategorizedIssue]:
        """Asigna prioridades fallback."""
        for issue in categorized_issues:
            # Prioridad simple basada en categoría y severidad
            priority_score = 50.0
            
            if issue.primary_category == IssueCategory.SECURITY:
                priority_score = 80.0
            elif issue.primary_category == IssueCategory.RELIABILITY:
                priority_score = 70.0
            elif issue.primary_category == IssueCategory.PERFORMANCE:
                priority_score = 60.0
            
            # Ajustar por severidad
            if issue.original_issue:
                severity_adjustments = {
                    IssueSeverity.CRITICAL: 20.0,
                    IssueSeverity.HIGH: 10.0,
                    IssueSeverity.MEDIUM: 0.0,
                    IssueSeverity.LOW: -10.0,
                    IssueSeverity.INFO: -20.0
                }
                priority_score += severity_adjustments.get(issue.original_issue.severity, 0.0)
            
            # Determinar level
            if priority_score >= 80:
                level = PriorityLevel.CRITICAL
            elif priority_score >= 65:
                level = PriorityLevel.HIGH
            elif priority_score >= 40:
                level = PriorityLevel.MEDIUM
            else:
                level = PriorityLevel.LOW
            
            # Crear prioridad básica
            from ...domain.entities.issue_management import IssuePriority, ImpactScore, UrgencyScore, BusinessValueScore, RiskScore
            
            issue.metadata.priority = IssuePriority(
                score=priority_score,
                level=level,
                impact=ImpactScore(score=priority_score),
                urgency=UrgencyScore(score=priority_score * 0.8),
                business_value=BusinessValueScore(score=priority_score * 0.6),
                risk=RiskScore(score=priority_score * 0.7),
                reasoning="Fallback priority calculation"
            )
        
        # Ordenar por prioridad
        categorized_issues.sort(key=lambda i: i.get_priority_score(), reverse=True)
        
        return categorized_issues
    
    def _calculate_category_distribution(self, issues: List[CategorizedIssue]) -> Dict[IssueCategory, int]:
        """Calcula distribución por categorías."""
        distribution = {}
        for issue in issues:
            category = issue.primary_category
            distribution[category] = distribution.get(category, 0) + 1
        
        return distribution
    
    async def _create_analysis_summary(self, issues: List[CategorizedIssue], clusters: List[IssueCluster],
                                     remediation_plan: Optional[RemediationPlan], 
                                     session: ProcessingSession) -> AnalysisSummary:
        """Crea resumen del análisis."""
        # Calcular métricas de calidad
        categorization_accuracy = session.get_success_rate()
        
        clustering_efficiency = 0.0
        if clusters and issues:
            clustered_issues = sum(len(cluster.issues) for cluster in clusters)
            clustering_efficiency = clustered_issues / len(issues)
        
        # Balance de distribución de prioridades
        priority_counts = {}
        for issue in issues:
            if issue.metadata.priority:
                level = issue.metadata.priority.level
                priority_counts[level] = priority_counts.get(level, 0) + 1
        
        # Calcular balance (penalizar distribuciones muy sesgadas)
        total_issues = len(issues)
        if total_issues > 0:
            high_priority_ratio = (priority_counts.get(PriorityLevel.CRITICAL, 0) + 
                                 priority_counts.get(PriorityLevel.HIGH, 0)) / total_issues
            
            # Balance ideal: 20-30% alta prioridad
            if 0.2 <= high_priority_ratio <= 0.3:
                priority_balance = 1.0
            else:
                priority_balance = 1.0 - abs(high_priority_ratio - 0.25) * 2.0
            
            priority_balance = max(0.0, priority_balance)
        else:
            priority_balance = 1.0
        
        # Estimar tiempo de remediación
        estimated_weeks = 0
        if remediation_plan and remediation_plan.sprint_plans:
            estimated_weeks = len(remediation_plan.sprint_plans) * self.config.remediation_config.sprint_duration_weeks
        elif issues:
            # Estimación simple
            total_effort = sum(issue.metadata.estimated_fix_time_hours or 2.0 for issue in issues)
            weekly_capacity = self.config.remediation_config.available_developer_hours_per_week
            estimated_weeks = math.ceil(total_effort / weekly_capacity)
        
        # Calcular mejora de calidad esperada
        expected_improvement = 0.0
        if remediation_plan:
            expected_improvement = remediation_plan.expected_quality_improvement
        else:
            # Estimación basada en prioridades
            high_priority_issues = sum(1 for issue in issues if issue.is_high_priority())
            expected_improvement = min(50.0, high_priority_issues * 3.0)
        
        # Top categorías
        category_counts = {}
        for issue in issues:
            category = issue.primary_category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generar recomendaciones
        recommendations = await self._generate_analysis_recommendations(issues, clusters, remediation_plan)
        
        summary = AnalysisSummary()
        summary.total_issues_analyzed = len(issues)
        summary.categorization_accuracy = categorization_accuracy
        summary.clustering_efficiency = clustering_efficiency
        summary.priority_distribution_balance = priority_balance
        summary.estimated_remediation_time_weeks = estimated_weeks
        summary.expected_quality_improvement = expected_improvement
        summary.top_categories = top_categories
        summary.recommendations = recommendations
        
        return summary
    
    async def _generate_analysis_recommendations(self, issues: List[CategorizedIssue], 
                                               clusters: List[IssueCluster],
                                               remediation_plan: Optional[RemediationPlan]) -> List[str]:
        """Genera recomendaciones del análisis."""
        recommendations = []
        
        # Recomendaciones basadas en distribución de issues
        if issues:
            # Análisis de categorías
            category_counts = {}
            for issue in issues:
                category_counts[issue.primary_category] = category_counts.get(issue.primary_category, 0) + 1
            
            # Recomendaciones por categoría dominante
            if category_counts.get(IssueCategory.SECURITY, 0) > len(issues) * 0.2:
                recommendations.append("High number of security issues - prioritize security review")
            
            if category_counts.get(IssueCategory.PERFORMANCE, 0) > len(issues) * 0.3:
                recommendations.append("Performance issues prevalent - consider performance audit")
            
            if category_counts.get(IssueCategory.MAINTAINABILITY, 0) > len(issues) * 0.4:
                recommendations.append("Many maintainability issues - invest in refactoring")
            
            # Recomendaciones basadas en prioridades
            high_priority_count = sum(1 for issue in issues if issue.is_high_priority())
            if high_priority_count > len(issues) * 0.4:
                recommendations.append("High percentage of critical issues - immediate action required")
            
            # Recomendaciones basadas en clustering
            if clusters:
                large_clusters = [c for c in clusters if len(c.issues) >= 5]
                if large_clusters:
                    recommendations.append(f"{len(large_clusters)} large issue clusters found - batch fixes recommended")
                
                avg_roi = sum(cluster.roi_score for cluster in clusters) / len(clusters)
                if avg_roi > 2.0:
                    recommendations.append("High ROI opportunities identified - prioritize cluster fixes")
            
            # Recomendaciones de timeline
            if remediation_plan and len(remediation_plan.sprint_plans) > 6:
                recommendations.append("Long remediation timeline - consider parallel teams or scope prioritization")
            
            # Recomendaciones generales
            if len(issues) > 100:
                recommendations.append("Large number of issues - consider systematic approach with automation")
            
            if not clusters and len(issues) > 10:
                recommendations.append("Issues not well clustered - may require individual attention")
        
        return recommendations
    
    async def analyze_business_impact(self, issues: List[CategorizedIssue]) -> Dict[str, Any]:
        """
        Analiza impacto de negocio de los issues.
        
        Args:
            issues: Lista de issues categorizados
            
        Returns:
            Análisis de impacto de negocio
        """
        return await self.business_analyzer.analyze_business_impact(issues)
    
    async def generate_comprehensive_report(self, analysis_result: IssueAnalysisResult) -> Dict[str, Any]:
        """
        Genera reporte comprehensivo del análisis.
        
        Args:
            analysis_result: Resultado del análisis de issues
            
        Returns:
            Diccionario con reporte detallado
        """
        report = {
            "executive_summary": await self._create_executive_summary(analysis_result),
            "issue_analysis": self._create_issue_analysis_section(analysis_result),
            "priority_analysis": self._create_priority_analysis_section(analysis_result),
            "clustering_analysis": self._create_clustering_analysis_section(analysis_result),
            "remediation_plan": self._create_remediation_plan_section(analysis_result),
            "business_impact": await self._create_business_impact_section(analysis_result),
            "recommendations": self._create_recommendations_section(analysis_result),
            "appendix": self._create_appendix(analysis_result)
        }
        
        return report
    
    async def _create_executive_summary(self, analysis_result: IssueAnalysisResult) -> Dict[str, Any]:
        """Crea resumen ejecutivo."""
        issues = analysis_result.categorized_issues
        
        summary = {
            "total_issues": len(issues),
            "analysis_quality_score": analysis_result.analysis_summary.get_analysis_quality_score() if analysis_result.analysis_summary else 0.0,
            "high_priority_issues": len(analysis_result.get_high_priority_issues()),
            "security_issues": len(analysis_result.get_issues_by_category(IssueCategory.SECURITY)),
            "estimated_remediation_weeks": analysis_result.analysis_summary.estimated_remediation_time_weeks if analysis_result.analysis_summary else 0,
            "expected_quality_improvement": analysis_result.analysis_summary.expected_quality_improvement if analysis_result.analysis_summary else 0.0,
            "key_insights": await self._generate_key_insights(analysis_result)
        }
        
        return summary
    
    async def _generate_key_insights(self, analysis_result: IssueAnalysisResult) -> List[str]:
        """Genera insights clave del análisis."""
        insights = []
        
        issues = analysis_result.categorized_issues
        if not issues:
            return ["No issues to analyze"]
        
        # Insight sobre distribución de categorías
        category_counts = analysis_result.category_distribution
        if category_counts:
            top_category = max(category_counts.keys(), key=lambda k: category_counts[k])
            top_count = category_counts[top_category]
            percentage = (top_count / len(issues)) * 100.0
            insights.append(f"{top_category.value} issues dominate ({percentage:.1f}% of total)")
        
        # Insight sobre prioridades
        high_priority_count = len(analysis_result.get_high_priority_issues())
        if high_priority_count > len(issues) * 0.3:
            insights.append(f"{high_priority_count} high-priority issues require immediate attention")
        
        # Insight sobre clustering
        if analysis_result.issue_clusters:
            avg_cluster_size = sum(len(cluster.issues) for cluster in analysis_result.issue_clusters) / len(analysis_result.issue_clusters)
            insights.append(f"Issues cluster well (avg size: {avg_cluster_size:.1f}) - batch fixes recommended")
        
        # Insight sobre ROI
        if analysis_result.remediation_plan:
            if analysis_result.remediation_plan.payback_period_weeks < 26:  # 6 meses
                insights.append("High ROI remediation opportunities with quick payback")
        
        return insights
    
    def _create_issue_analysis_section(self, analysis_result: IssueAnalysisResult) -> Dict[str, Any]:
        """Crea sección de análisis de issues."""
        return {
            "total_issues": analysis_result.get_total_issues(),
            "category_breakdown": {cat.value: count for cat, count in analysis_result.category_distribution.items()},
            "priority_breakdown": {
                "critical": analysis_result.priority_distribution.critical_count,
                "high": analysis_result.priority_distribution.high_count,
                "medium": analysis_result.priority_distribution.medium_count,
                "low": analysis_result.priority_distribution.low_count,
                "lowest": analysis_result.priority_distribution.lowest_count
            },
            "high_priority_percentage": analysis_result.priority_distribution.get_high_priority_percentage()
        }
    
    def _create_priority_analysis_section(self, analysis_result: IssueAnalysisResult) -> Dict[str, Any]:
        """Crea sección de análisis de prioridades."""
        issues = analysis_result.categorized_issues
        
        if not issues:
            return {"message": "No issues to analyze"}
        
        # Calcular estadísticas de prioridad
        priority_scores = [issue.get_priority_score() for issue in issues]
        
        return {
            "average_priority_score": sum(priority_scores) / len(priority_scores),
            "highest_priority_score": max(priority_scores),
            "lowest_priority_score": min(priority_scores),
            "priority_distribution": {
                "critical": analysis_result.priority_distribution.critical_count,
                "high": analysis_result.priority_distribution.high_count,
                "medium": analysis_result.priority_distribution.medium_count,
                "low": analysis_result.priority_distribution.low_count
            },
            "top_priority_issues": [
                {
                    "id": issue.id.value[:8],
                    "category": issue.primary_category.value,
                    "priority_score": issue.get_priority_score(),
                    "description": issue.get_display_title()
                }
                for issue in issues[:10]  # Top 10
            ]
        }
    
    def _create_clustering_analysis_section(self, analysis_result: IssueAnalysisResult) -> Dict[str, Any]:
        """Crea sección de análisis de clustering."""
        clusters = analysis_result.issue_clusters
        
        if not clusters:
            return {"message": "No clusters formed", "singleton_issues": analysis_result.get_total_issues()}
        
        return {
            "total_clusters": len(clusters),
            "clustered_issues": sum(len(cluster.issues) for cluster in clusters),
            "average_cluster_size": sum(len(cluster.issues) for cluster in clusters) / len(clusters),
            "largest_cluster_size": max(len(cluster.issues) for cluster in clusters),
            "cluster_details": [
                {
                    "cluster_id": cluster.id.value[:8],
                    "size": len(cluster.issues),
                    "dominant_category": cluster.get_dominant_category().value,
                    "cohesion_score": cluster.cohesion_score,
                    "roi_score": cluster.roi_score,
                    "estimated_fix_time": cluster.estimated_batch_fix_time
                }
                for cluster in clusters
            ]
        }
    
    def _create_remediation_plan_section(self, analysis_result: IssueAnalysisResult) -> Dict[str, Any]:
        """Crea sección del plan de remediación."""
        plan = analysis_result.remediation_plan
        
        if not plan:
            return {"message": "No remediation plan available"}
        
        return {
            "total_estimated_effort_hours": plan.total_estimated_effort_hours,
            "total_investment": plan.total_investment,
            "expected_annual_savings": plan.expected_annual_savings,
            "payback_period_weeks": plan.payback_period_weeks,
            "number_of_sprints": len(plan.sprint_plans),
            "expected_quality_improvement": plan.expected_quality_improvement,
            "quick_wins": [plan.get_quick_wins()[:5]],  # Top 5 quick wins
            "high_roi_fixes": [plan.get_high_roi_fixes()[:5]]  # Top 5 high ROI
        }
    
    async def _create_business_impact_section(self, analysis_result: IssueAnalysisResult) -> Dict[str, Any]:
        """Crea sección de impacto de negocio."""
        if not analysis_result.categorized_issues:
            return {"message": "No issues for business impact analysis"}
        
        business_impact = await self.analyze_business_impact(analysis_result.categorized_issues)
        
        return {
            "total_potential_revenue_impact": business_impact["total_potential_revenue_impact"],
            "total_potential_cost_savings": business_impact["total_potential_cost_savings"],
            "productivity_gain_hours_annual": business_impact["productivity_gain_hours_annual"],
            "business_recommendations": business_impact["business_priority_recommendations"],
            "quick_wins_count": len(business_impact["quick_wins"]),
            "strategic_investments_count": len(business_impact["strategic_investments"])
        }
    
    def _create_recommendations_section(self, analysis_result: IssueAnalysisResult) -> Dict[str, Any]:
        """Crea sección de recomendaciones."""
        recommendations = {
            "immediate_actions": [],
            "strategic_actions": [],
            "process_improvements": [],
            "tool_recommendations": []
        }
        
        issues = analysis_result.categorized_issues
        if not issues:
            return recommendations
        
        # Acciones inmediatas
        critical_issues = [issue for issue in issues 
                         if issue.metadata.priority and issue.metadata.priority.level == PriorityLevel.CRITICAL]
        if critical_issues:
            recommendations["immediate_actions"].append(f"Address {len(critical_issues)} critical issues immediately")
        
        security_issues = analysis_result.get_issues_by_category(IssueCategory.SECURITY)
        if security_issues:
            recommendations["immediate_actions"].append(f"Review and fix {len(security_issues)} security issues")
        
        # Acciones estratégicas
        if len(issues) > 50:
            recommendations["strategic_actions"].append("Implement systematic quality improvement program")
        
        if analysis_result.issue_clusters and len(analysis_result.issue_clusters) > 3:
            recommendations["strategic_actions"].append("Leverage batch fixing for clustered issues")
        
        # Mejoras de proceso
        maintainability_issues = analysis_result.get_issues_by_category(IssueCategory.MAINTAINABILITY)
        if len(maintainability_issues) > len(issues) * 0.4:
            recommendations["process_improvements"].append("Introduce code review guidelines for maintainability")
        
        # Recomendaciones de herramientas
        if analysis_result.get_issues_by_category(IssueCategory.CODE_STYLE):
            recommendations["tool_recommendations"].append("Implement automated code formatting tools")
        
        return recommendations
    
    def _create_appendix(self, analysis_result: IssueAnalysisResult) -> Dict[str, Any]:
        """Crea apéndice con detalles técnicos."""
        return {
            "analysis_metadata": {
                "analysis_time_ms": analysis_result.analysis_time_ms,
                "analyzed_at": analysis_result.analyzed_at.isoformat(),
                "configuration_used": {
                    "ml_enabled": self.config.enable_machine_learning,
                    "clustering_enabled": self.config.clustering_config.enable_hierarchical_clustering,
                    "performance_mode": self.config.performance_mode
                }
            },
            "technical_details": {
                "categorization_confidence": analysis_result.analysis_summary.categorization_accuracy if analysis_result.analysis_summary else 0.0,
                "clustering_efficiency": analysis_result.analysis_summary.clustering_efficiency if analysis_result.analysis_summary else 0.0,
                "processing_success_rate": analysis_result.analysis_summary.categorization_accuracy if analysis_result.analysis_summary else 0.0
            }
        }
    
    async def get_processing_statistics(self, analysis_result: IssueAnalysisResult) -> Dict[str, Any]:
        """
        Obtiene estadísticas de procesamiento.
        
        Returns:
            Diccionario con estadísticas detalladas
        """
        stats = {
            "processing_metrics": {
                "total_issues_processed": analysis_result.get_total_issues(),
                "categorization_success_rate": 100.0,  # Simplificado
                "prioritization_success_rate": 100.0,  # Simplificado
                "clustering_success_rate": 100.0 if analysis_result.issue_clusters else 0.0,
                "analysis_time_ms": analysis_result.analysis_time_ms
            },
            "quality_metrics": {
                "average_priority_score": 0.0,
                "high_priority_percentage": analysis_result.priority_distribution.get_high_priority_percentage(),
                "category_diversity": len(analysis_result.category_distribution),
                "clustering_quality": 0.0
            },
            "efficiency_metrics": {
                "issues_per_second": analysis_result.get_total_issues() / (analysis_result.analysis_time_ms / 1000.0) if analysis_result.analysis_time_ms > 0 else 0.0,
                "memory_efficiency": "good",  # Placeholder
                "cpu_efficiency": "good"      # Placeholder
            }
        }
        
        # Calcular average priority score
        issues = analysis_result.categorized_issues
        if issues:
            priority_scores = [issue.get_priority_score() for issue in issues]
            stats["quality_metrics"]["average_priority_score"] = sum(priority_scores) / len(priority_scores)
        
        # Calcular clustering quality
        if analysis_result.issue_clusters:
            avg_cohesion = sum(cluster.cohesion_score for cluster in analysis_result.issue_clusters) / len(analysis_result.issue_clusters)
            stats["quality_metrics"]["clustering_quality"] = avg_cohesion
        
        return stats
