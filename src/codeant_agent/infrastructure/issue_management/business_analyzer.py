"""
Implementación del analizador de impacto de negocio.

Este módulo implementa el análisis de impacto en el negocio,
evaluación de valor y assessment de riesgos empresariales.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import time

from ...domain.entities.issue_management import (
    CategorizedIssue, BusinessValueScore, IssueCategory, IssueSeverity,
    BusinessImpactLevel, PriorityLevel
)

logger = logging.getLogger(__name__)


@dataclass
class BusinessImpactAnalysis:
    """Análisis de impacto de negocio."""
    revenue_impact_estimate: float = 0.0  # USD anual
    cost_impact_estimate: float = 0.0     # USD anual
    productivity_impact_hours: float = 0.0  # Horas anuales
    customer_impact_score: float = 0.0   # 0-100
    competitive_impact_score: float = 0.0  # 0-100
    compliance_impact_score: float = 0.0   # 0-100
    reputation_impact_score: float = 0.0   # 0-100
    analysis_confidence: float = 0.7
    impact_timeline: str = "medium_term"  # "immediate", "short_term", "medium_term", "long_term"


class BusinessValueAssessor:
    """Evaluador de valor de negocio."""
    
    def __init__(self):
        # Configuración de valores de negocio
        self.revenue_per_user_annual = 1200.0  # USD promedio por usuario por año
        self.developer_cost_per_hour = 75.0
        self.customer_acquisition_cost = 500.0
        self.customer_retention_value = 2000.0
    
    async def assess_business_value(self, issue: CategorizedIssue) -> BusinessValueScore:
        """
        Evalúa valor de negocio de resolver un issue.
        
        Args:
            issue: Issue categorizado
            
        Returns:
            BusinessValueScore detallado
        """
        # Calcular diferentes componentes de valor
        revenue_impact = await self._calculate_revenue_impact(issue)
        cost_savings = await self._calculate_cost_savings(issue)
        risk_mitigation_value = await self._calculate_risk_mitigation_value(issue)
        productivity_improvement = await self._calculate_productivity_improvement(issue)
        customer_satisfaction_impact = await self._calculate_customer_satisfaction_impact(issue)
        competitive_advantage = await self._calculate_competitive_advantage(issue)
        
        # Score general ponderado
        overall_score = (
            revenue_impact * 0.25 +
            cost_savings * 0.25 +
            risk_mitigation_value * 0.2 +
            productivity_improvement * 0.15 +
            customer_satisfaction_impact * 0.1 +
            competitive_advantage * 0.05
        )
        
        # Generar razonamiento
        value_reasoning = self._generate_value_reasoning(
            issue, revenue_impact, cost_savings, productivity_improvement, risk_mitigation_value
        )
        
        return BusinessValueScore(
            score=overall_score,
            revenue_impact=revenue_impact,
            cost_savings=cost_savings,
            risk_mitigation_value=risk_mitigation_value,
            productivity_improvement=productivity_improvement,
            customer_satisfaction_impact=customer_satisfaction_impact,
            competitive_advantage=competitive_advantage,
            value_reasoning=value_reasoning
        )
    
    async def _calculate_revenue_impact(self, issue: CategorizedIssue) -> float:
        """Calcula impacto potencial en ingresos."""
        base_impact = 20.0
        
        # Impacto por categoría
        if issue.primary_category == IssueCategory.SECURITY:
            # Security issues pueden causar pérdida de confianza del cliente
            if issue.metadata.security_risk_score > 70:
                base_impact = 90.0  # Potencial pérdida significativa
            else:
                base_impact = 60.0
        
        elif issue.primary_category == IssueCategory.PERFORMANCE:
            # Performance issues afectan conversiones
            performance_impact = issue.metadata.performance_impact_percentage
            if performance_impact > 15:
                base_impact = 80.0  # Performance crítico
            elif performance_impact > 5:
                base_impact = 50.0
            else:
                base_impact = 30.0
        
        elif issue.primary_category == IssueCategory.USABILITY:
            base_impact = 70.0  # Usability afecta directamente conversiones
        
        elif issue.primary_category == IssueCategory.RELIABILITY:
            # Reliability issues pueden causar downtime
            base_impact = 65.0
        
        # Boost para módulos críticos
        if issue.context_info.module_criticality == "critical":
            base_impact *= 1.3
        
        return min(100.0, base_impact)
    
    async def _calculate_cost_savings(self, issue: CategorizedIssue) -> float:
        """Calcula ahorros de costo potenciales."""
        base_savings = 30.0
        
        # Ahorros por reducción de deuda técnica
        debt_savings = issue.metadata.technical_debt_contribution * 2.0
        base_savings += debt_savings
        
        # Ahorros específicos por categoría
        if issue.primary_category == IssueCategory.MAINTAINABILITY:
            # Mantenibilidad reduce costos futuros significativamente
            base_savings += 40.0
        
        elif issue.primary_category == IssueCategory.PERFORMANCE:
            # Performance improvements reducen costos de infraestructura
            base_savings += 35.0
        
        elif issue.primary_category == IssueCategory.RELIABILITY:
            # Reliability reduce costos de soporte y fixes
            base_savings += 45.0
        
        # Boost para archivos que cambian frecuentemente
        if issue.context_info.file_change_frequency > 0.7:
            base_savings += 20.0
        
        return min(100.0, base_savings)
    
    async def _calculate_risk_mitigation_value(self, issue: CategorizedIssue) -> float:
        """Calcula valor de mitigación de riesgo."""
        base_value = 25.0
        
        # Valor alto para mitigación de riesgos críticos
        if issue.primary_category == IssueCategory.SECURITY:
            base_value = 85.0
            if issue.metadata.security_risk_score > 80:
                base_value = 95.0
        
        elif issue.primary_category == IssueCategory.RELIABILITY:
            base_value = 70.0
        
        elif issue.primary_category == IssueCategory.PERFORMANCE:
            base_value = 55.0
        
        # Ajustar por business impact level
        business_multipliers = {
            BusinessImpactLevel.BLOCKING: 1.5,
            BusinessImpactLevel.SIGNIFICANT: 1.3,
            BusinessImpactLevel.MODERATE: 1.0,
            BusinessImpactLevel.MINOR: 0.8,
            BusinessImpactLevel.NEGLIGIBLE: 0.5
        }
        
        base_value *= business_multipliers.get(issue.metadata.business_impact_level, 1.0)
        
        return min(100.0, base_value)
    
    async def _calculate_productivity_improvement(self, issue: CategorizedIssue) -> float:
        """Calcula mejora de productividad."""
        base_improvement = 25.0
        
        # Mejora por categoría
        if issue.primary_category == IssueCategory.MAINTAINABILITY:
            base_improvement = 80.0  # Mantenibilidad mejora mucho la productividad
        
        elif issue.primary_category == IssueCategory.DOCUMENTATION:
            base_improvement = 60.0  # Documentación mejora onboarding y desarrollo
        
        elif issue.primary_category == IssueCategory.CODE_STYLE:
            base_improvement = 40.0  # Consistencia mejora velocidad
        
        elif issue.primary_category == IssueCategory.TESTABILITY:
            base_improvement = 70.0  # Testabilidad acelera desarrollo
        
        # Boost para archivos frecuentemente modificados
        if issue.context_info.file_change_frequency > 0.6:
            base_improvement += 25.0
        
        # Boost para complejidad alta
        if (issue.original_issue and issue.original_issue.complexity_metrics and
            issue.original_issue.complexity_metrics.cyclomatic_complexity > 15):
            base_improvement += 20.0
        
        return min(100.0, base_improvement)
    
    async def _calculate_customer_satisfaction_impact(self, issue: CategorizedIssue) -> float:
        """Calcula impacto en satisfacción del cliente."""
        base_impact = 15.0
        
        # Impacto directo por categoría
        customer_facing_categories = {
            IssueCategory.USABILITY: 85.0,
            IssueCategory.PERFORMANCE: 70.0,
            IssueCategory.RELIABILITY: 75.0,
            IssueCategory.SECURITY: 60.0  # Indirecto pero importante
        }
        
        base_impact = customer_facing_categories.get(issue.primary_category, 15.0)
        
        # Ajustar por severidad del issue
        if issue.original_issue:
            severity_multipliers = {
                IssueSeverity.CRITICAL: 1.5,
                IssueSeverity.HIGH: 1.2,
                IssueSeverity.MEDIUM: 1.0,
                IssueSeverity.LOW: 0.8
            }
            base_impact *= severity_multipliers.get(issue.original_issue.severity, 1.0)
        
        return min(100.0, base_impact)
    
    async def _calculate_competitive_advantage(self, issue: CategorizedIssue) -> float:
        """Calcula ventaja competitiva."""
        base_advantage = 10.0
        
        # Ventaja por mejoras de calidad
        if issue.primary_category in [IssueCategory.PERFORMANCE, IssueCategory.USABILITY]:
            base_advantage = 40.0
        
        elif issue.primary_category == IssueCategory.SECURITY:
            base_advantage = 35.0  # Seguridad es diferenciador
        
        elif issue.primary_category in [IssueCategory.MAINTAINABILITY, IssueCategory.ARCHITECTURE]:
            base_advantage = 25.0  # Permite desarrollo más rápido
        
        return base_advantage
    
    def _generate_value_reasoning(self, issue: CategorizedIssue, revenue: float, cost_savings: float,
                                productivity: float, risk_mitigation: float) -> str:
        """Genera razonamiento de valor de negocio."""
        reasons = []
        
        if revenue > 60:
            reasons.append("significant revenue protection")
        if cost_savings > 60:
            reasons.append("substantial cost reduction")
        if productivity > 60:
            reasons.append("major productivity gains")
        if risk_mitigation > 60:
            reasons.append("critical risk mitigation")
        
        if not reasons:
            reasons.append("limited but measurable business value")
        
        category_context = f"{issue.primary_category.value} improvement"
        return f"{category_context}: {', '.join(reasons)}"


class BusinessImpactAnalyzer:
    """Analizador principal de impacto de negocio."""
    
    def __init__(self):
        self.business_value_assessor = BusinessValueAssessor()
        
        # Métricas de negocio configurables
        self.business_metrics = {
            "average_deal_size": 10000.0,
            "customer_lifetime_value": 25000.0,
            "monthly_active_users": 50000,
            "conversion_rate": 0.03,
            "churn_rate": 0.05,
            "support_cost_per_ticket": 25.0
        }
    
    async def analyze_business_impact(self, issues: List[CategorizedIssue]) -> Dict[str, Any]:
        """
        Analiza impacto de negocio de lista de issues.
        
        Args:
            issues: Lista de issues categorizados
            
        Returns:
            Diccionario con análisis de impacto de negocio
        """
        analysis = {
            "total_potential_revenue_impact": 0.0,
            "total_potential_cost_savings": 0.0,
            "productivity_gain_hours_annual": 0.0,
            "customer_satisfaction_improvement": 0.0,
            "risk_mitigation_value": 0.0,
            "business_priority_recommendations": [],
            "roi_ranked_issues": [],
            "quick_wins": [],
            "strategic_investments": []
        }
        
        # Analizar cada issue
        for issue in issues:
            business_value = await self.business_value_assessor.assess_business_value(issue)
            
            # Acumular métricas
            analysis["total_potential_revenue_impact"] += business_value.revenue_impact * 100.0  # Escalar
            analysis["total_potential_cost_savings"] += business_value.cost_savings * 50.0  # Escalar
            analysis["productivity_gain_hours_annual"] += business_value.productivity_improvement * 10.0
            analysis["customer_satisfaction_improvement"] += business_value.customer_satisfaction_impact
            analysis["risk_mitigation_value"] += business_value.risk_mitigation_value * 25.0
        
        # Promediar métricas que deben ser promediadas
        if issues:
            analysis["customer_satisfaction_improvement"] /= len(issues)
        
        # Generar recomendaciones de prioridad de negocio
        analysis["business_priority_recommendations"] = await self._generate_business_recommendations(issues)
        
        # Ranking de issues por ROI potencial
        analysis["roi_ranked_issues"] = await self._rank_issues_by_roi(issues)
        
        # Identificar quick wins (bajo costo, alto valor)
        analysis["quick_wins"] = await self._identify_quick_wins(issues)
        
        # Identificar inversiones estratégicas (alto costo, muy alto valor)
        analysis["strategic_investments"] = await self._identify_strategic_investments(issues)
        
        return analysis
    
    async def _generate_business_recommendations(self, issues: List[CategorizedIssue]) -> List[str]:
        """Genera recomendaciones de prioridad de negocio."""
        recommendations = []
        
        # Analizar distribución de categorías
        category_counts = {}
        high_value_categories = {}
        
        for issue in issues:
            category = issue.primary_category
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Contar issues de alto valor de negocio
            if issue.metadata.business_impact_level in [BusinessImpactLevel.BLOCKING, BusinessImpactLevel.SIGNIFICANT]:
                high_value_categories[category] = high_value_categories.get(category, 0) + 1
        
        # Recomendaciones basadas en análisis
        if IssueCategory.SECURITY in high_value_categories and high_value_categories[IssueCategory.SECURITY] > 0:
            recommendations.append("Prioritize security issues due to high business risk")
        
        if IssueCategory.PERFORMANCE in category_counts and category_counts[IssueCategory.PERFORMANCE] > 3:
            recommendations.append("Address performance issues to improve user experience and conversions")
        
        if IssueCategory.MAINTAINABILITY in category_counts and category_counts[IssueCategory.MAINTAINABILITY] > 5:
            recommendations.append("Invest in maintainability to reduce long-term development costs")
        
        # Recomendaciones generales
        total_high_value = sum(high_value_categories.values())
        if total_high_value > len(issues) * 0.3:
            recommendations.append("High percentage of business-critical issues - consider immediate action")
        
        if len(issues) > 50:
            recommendations.append("Large number of issues - consider systematic approach with sprints")
        
        return recommendations
    
    async def _rank_issues_by_roi(self, issues: List[CategorizedIssue]) -> List[Dict[str, Any]]:
        """Rankea issues por ROI potencial."""
        roi_data = []
        
        for issue in issues:
            # Calcular ROI simplificado
            fix_cost = (issue.metadata.estimated_fix_time_hours or 2.0) * self.business_value_assessor.developer_cost_per_hour
            
            business_value = await self.business_value_assessor.assess_business_value(issue)
            potential_annual_value = (
                business_value.revenue_impact * 100.0 +
                business_value.cost_savings * 50.0 +
                business_value.productivity_improvement * 10.0
            )
            
            roi_percentage = ((potential_annual_value - fix_cost) / fix_cost * 100.0) if fix_cost > 0 else 0.0
            
            roi_data.append({
                "issue_id": issue.id.value,
                "category": issue.primary_category.value,
                "fix_cost": fix_cost,
                "annual_value": potential_annual_value,
                "roi_percentage": roi_percentage,
                "priority_score": issue.get_priority_score()
            })
        
        # Ordenar por ROI descendente
        roi_data.sort(key=lambda x: x["roi_percentage"], reverse=True)
        
        return roi_data[:20]  # Top 20
    
    async def _identify_quick_wins(self, issues: List[CategorizedIssue]) -> List[Dict[str, Any]]:
        """Identifica quick wins (bajo esfuerzo, alto valor)."""
        quick_wins = []
        
        for issue in issues:
            effort_hours = issue.metadata.estimated_fix_time_hours or 2.0
            priority_score = issue.get_priority_score()
            
            # Quick win: <= 4 horas effort, >= 60 priority score
            if effort_hours <= 4.0 and priority_score >= 60.0:
                business_value = await self.business_value_assessor.assess_business_value(issue)
                
                quick_wins.append({
                    "issue_id": issue.id.value,
                    "category": issue.primary_category.value,
                    "effort_hours": effort_hours,
                    "priority_score": priority_score,
                    "business_value_score": business_value.score,
                    "description": issue.get_display_title()
                })
        
        # Ordenar por ratio valor/esfuerzo
        quick_wins.sort(key=lambda x: x["business_value_score"] / x["effort_hours"], reverse=True)
        
        return quick_wins[:10]
    
    async def _identify_strategic_investments(self, issues: List[CategorizedIssue]) -> List[Dict[str, Any]]:
        """Identifica inversiones estratégicas (alto costo, muy alto valor)."""
        strategic = []
        
        for issue in issues:
            effort_hours = issue.metadata.estimated_fix_time_hours or 2.0
            priority_score = issue.get_priority_score()
            
            # Strategic investment: > 8 horas effort, >= 80 priority score
            if effort_hours > 8.0 and priority_score >= 80.0:
                business_value = await self.business_value_assessor.assess_business_value(issue)
                
                strategic.append({
                    "issue_id": issue.id.value,
                    "category": issue.primary_category.value,
                    "effort_hours": effort_hours,
                    "priority_score": priority_score,
                    "business_value_score": business_value.score,
                    "strategic_importance": self._assess_strategic_importance(issue),
                    "description": issue.get_display_title()
                })
        
        # Ordenar por importancia estratégica
        strategic.sort(key=lambda x: x["strategic_importance"], reverse=True)
        
        return strategic[:5]
    
    def _assess_strategic_importance(self, issue: CategorizedIssue) -> float:
        """Evalúa importancia estratégica."""
        importance = 50.0
        
        # Importancia por categoría
        if issue.primary_category == IssueCategory.ARCHITECTURE:
            importance = 90.0
        elif issue.primary_category == IssueCategory.SECURITY:
            importance = 85.0
        elif issue.primary_category == IssueCategory.PERFORMANCE:
            importance = 75.0
        
        # Boost para módulos críticos
        if issue.context_info.module_criticality == "critical":
            importance += 10.0
        
        # Boost para issues que afectan muchos componentes
        if issue.metadata.fix_complexity_score > 70:
            importance += 15.0
        
        return min(100.0, importance)
