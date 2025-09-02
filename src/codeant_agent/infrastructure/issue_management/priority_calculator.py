"""
Implementación del calculador de prioridades.

Este módulo implementa el cálculo inteligente de prioridades basado en
análisis de impacto, urgencia, valor de negocio y riesgo.
"""

import logging
import asyncio
import math
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from datetime import datetime, timedelta

from ...domain.entities.issue_management import (
    CategorizedIssue, IssuePriority, PriorityLevel, ImpactScore, UrgencyScore,
    BusinessValueScore, RiskScore, PriorityConfig, IssueCategory, IssueSeverity,
    BusinessImpactLevel
)
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class PriorityCalculationResult:
    """Resultado del cálculo de prioridades."""
    prioritized_issues: List[CategorizedIssue]
    priority_distribution: Dict[PriorityLevel, int]
    calculation_time_ms: int
    average_priority_score: float
    high_priority_percentage: float


class ImpactAnalyzer:
    """Analizador de impacto de issues."""
    
    def __init__(self):
        self.impact_weights = {
            "code_quality": 0.3,
            "user_experience": 0.25,
            "system_stability": 0.25,
            "business_operations": 0.2
        }
    
    async def calculate_impact(self, issue: CategorizedIssue) -> ImpactScore:
        """
        Calcula score de impacto completo.
        
        Args:
            issue: Issue categorizado
            
        Returns:
            ImpactScore detallado
        """
        # Calcular diferentes tipos de impacto
        code_impact = await self._calculate_code_impact(issue)
        user_impact = await self._calculate_user_impact(issue)
        system_impact = await self._calculate_system_impact(issue)
        business_impact = await self._calculate_business_impact(issue)
        
        # Calcular score general ponderado
        overall_score = (
            code_impact * self.impact_weights["code_quality"] +
            user_impact * self.impact_weights["user_experience"] +
            system_impact * self.impact_weights["system_stability"] +
            business_impact * self.impact_weights["business_operations"]
        )
        
        # Identificar componentes afectados
        affected_components = await self._identify_affected_components(issue)
        
        # Calcular blast radius
        blast_radius = self._calculate_blast_radius(issue, affected_components)
        
        # Generar razonamiento
        impact_reasoning = self._generate_impact_reasoning(issue, code_impact, user_impact, system_impact, business_impact)
        
        return ImpactScore(
            score=overall_score,
            code_impact=code_impact,
            user_impact=user_impact,
            system_impact=system_impact,
            business_impact=business_impact,
            affected_components=affected_components,
            blast_radius=blast_radius,
            impact_reasoning=impact_reasoning
        )
    
    async def _calculate_code_impact(self, issue: CategorizedIssue) -> float:
        """Calcula impacto en calidad del código."""
        score = 50.0  # Base score
        
        # Impacto por categoría
        category_impacts = {
            IssueCategory.SECURITY: 80.0,
            IssueCategory.PERFORMANCE: 70.0,
            IssueCategory.MAINTAINABILITY: 60.0,
            IssueCategory.RELIABILITY: 75.0,
            IssueCategory.ARCHITECTURE: 85.0,
            IssueCategory.CODE_STYLE: 30.0,
            IssueCategory.DOCUMENTATION: 25.0
        }
        
        score = category_impacts.get(issue.primary_category, 50.0)
        
        # Ajustar por severidad
        severity_multipliers = {
            IssueSeverity.CRITICAL: 1.5,
            IssueSeverity.HIGH: 1.2,
            IssueSeverity.MEDIUM: 1.0,
            IssueSeverity.LOW: 0.8,
            IssueSeverity.INFO: 0.5
        }
        
        if issue.original_issue:
            score *= severity_multipliers.get(issue.original_issue.severity, 1.0)
        
        # Ajustar por complejidad del código afectado
        if (issue.original_issue and issue.original_issue.complexity_metrics and 
            issue.original_issue.complexity_metrics.cyclomatic_complexity > 10):
            complexity_factor = 1.0 + (issue.original_issue.complexity_metrics.cyclomatic_complexity - 10) * 0.05
            score *= complexity_factor
        
        # Ajustar por criticidad del módulo
        if issue.context_info.module_criticality == "critical":
            score *= 1.3
        elif issue.context_info.module_criticality == "important":
            score *= 1.1
        
        return min(100.0, score)
    
    async def _calculate_user_impact(self, issue: CategorizedIssue) -> float:
        """Calcula impacto en usuarios."""
        base_score = 30.0
        
        # Impacto alto para issues de seguridad y performance
        if issue.primary_category == IssueCategory.SECURITY:
            base_score = 90.0
            if issue.metadata.security_risk_score > 70:
                base_score = 95.0
        
        elif issue.primary_category == IssueCategory.PERFORMANCE:
            base_score = 70.0
            if issue.metadata.performance_impact_percentage > 15:
                base_score = 85.0
        
        elif issue.primary_category == IssueCategory.USABILITY:
            base_score = 80.0
        
        elif issue.primary_category == IssueCategory.RELIABILITY:
            base_score = 75.0
        
        # Ajustar por business impact level
        business_multipliers = {
            BusinessImpactLevel.BLOCKING: 1.5,
            BusinessImpactLevel.SIGNIFICANT: 1.3,
            BusinessImpactLevel.MODERATE: 1.0,
            BusinessImpactLevel.MINOR: 0.8,
            BusinessImpactLevel.NEGLIGIBLE: 0.5
        }
        
        base_score *= business_multipliers.get(issue.metadata.business_impact_level, 1.0)
        
        return min(100.0, base_score)
    
    async def _calculate_system_impact(self, issue: CategorizedIssue) -> float:
        """Calcula impacto en estabilidad del sistema."""
        base_score = 40.0
        
        # Impacto alto para issues que pueden causar crashes
        if issue.primary_category == IssueCategory.RELIABILITY:
            if any(keyword in issue.original_issue.message.lower() 
                   for keyword in ['crash', 'fail', 'exception', 'null pointer']):
                base_score = 90.0
            else:
                base_score = 60.0
        
        elif issue.primary_category == IssueCategory.SECURITY:
            # Issues de seguridad pueden comprometer sistema
            base_score = 80.0
        
        elif issue.primary_category == IssueCategory.PERFORMANCE:
            # Performance issues pueden degradar sistema
            base_score = 55.0
        
        # Ajustar por criticidad del módulo
        if issue.context_info.module_criticality == "critical":
            base_score *= 1.4
        
        # Ajustar por número de dependencias
        dependency_factor = min(1.3, 1.0 + issue.context_info.dependency_count * 0.02)
        base_score *= dependency_factor
        
        return min(100.0, base_score)
    
    async def _calculate_business_impact(self, issue: CategorizedIssue) -> float:
        """Calcula impacto en operaciones de negocio."""
        base_score = 25.0
        
        # Mapeo directo desde business impact level
        business_scores = {
            BusinessImpactLevel.BLOCKING: 95.0,
            BusinessImpactLevel.SIGNIFICANT: 80.0,
            BusinessImpactLevel.MODERATE: 50.0,
            BusinessImpactLevel.MINOR: 25.0,
            BusinessImpactLevel.NEGLIGIBLE: 10.0
        }
        
        base_score = business_scores.get(issue.metadata.business_impact_level, 25.0)
        
        # Boost para issues de seguridad
        if issue.primary_category == IssueCategory.SECURITY:
            base_score = max(base_score, 70.0)
        
        # Boost para performance en módulos críticos
        if (issue.primary_category == IssueCategory.PERFORMANCE and 
            issue.context_info.module_criticality == "critical"):
            base_score = max(base_score, 60.0)
        
        return base_score
    
    async def _identify_affected_components(self, issue: CategorizedIssue) -> List[str]:
        """Identifica componentes afectados por el issue."""
        components = []
        
        if not issue.original_issue:
            return components
        
        # Componente basado en path del archivo
        file_path = issue.original_issue.file_path
        
        # Extraer módulo/paquete
        if len(file_path.parts) > 1:
            components.append(file_path.parts[-2])  # Directorio padre
        
        components.append(file_path.stem)  # Nombre del archivo
        
        # Componentes basados en categoría
        if issue.primary_category == IssueCategory.SECURITY:
            components.extend(["authentication", "authorization", "data_protection"])
        elif issue.primary_category == IssueCategory.PERFORMANCE:
            components.extend(["processing_engine", "data_access", "caching"])
        elif issue.primary_category == IssueCategory.MAINTAINABILITY:
            components.extend(["code_structure", "design_patterns"])
        
        return list(set(components))  # Eliminar duplicados
    
    def _calculate_blast_radius(self, issue: CategorizedIssue, components: List[str]) -> str:
        """Calcula radio de impacto."""
        if len(components) <= 1:
            return "file"
        elif len(components) <= 3:
            return "module"
        elif len(components) <= 6:
            return "component"
        else:
            return "system"
    
    def _generate_impact_reasoning(self, issue: CategorizedIssue, code_impact: float,
                                 user_impact: float, system_impact: float, business_impact: float) -> str:
        """Genera explicación del impacto."""
        reasons = []
        
        if code_impact > 70:
            reasons.append("high code quality impact")
        if user_impact > 70:
            reasons.append("significant user impact")
        if system_impact > 70:
            reasons.append("system stability concerns")
        if business_impact > 70:
            reasons.append("business operations affected")
        
        if not reasons:
            reasons.append("limited impact scope")
        
        return f"Impact factors: {', '.join(reasons)}"


class UrgencyCalculator:
    """Calculadora de urgencia de issues."""
    
    def __init__(self):
        self.urgency_factors = {
            "security_urgency": 0.4,
            "change_frequency": 0.3,
            "trend_analysis": 0.2,
            "temporal_context": 0.1
        }
    
    async def calculate_urgency(self, issue: CategorizedIssue) -> UrgencyScore:
        """
        Calcula score de urgencia.
        
        Args:
            issue: Issue categorizado
            
        Returns:
            UrgencyScore detallado
        """
        urgency_score = 0.0
        temporal_factors = []
        
        # 1. Urgencia por categoría y severidad
        category_urgency = self._calculate_category_urgency(issue)
        urgency_score += category_urgency
        
        # 2. Urgencia por frecuencia de cambios
        change_frequency = issue.context_info.file_change_frequency
        change_urgency = change_frequency * 20.0  # Escalar a 0-20
        urgency_score += change_urgency
        
        # 3. Urgencia por antigüedad del código
        recency_score = self._calculate_recency_urgency(issue)
        urgency_score += recency_score
        
        # 4. Urgencia por trend (si el problema está empeorando)
        trend_score = self._calculate_trend_urgency(issue)
        urgency_score += trend_score
        
        # 5. Verificar si afecta critical path
        is_critical_path = await self._affects_critical_path(issue)
        if is_critical_path:
            urgency_score += 25.0
            temporal_factors.append("affects_critical_path")
        
        # 6. Factores temporales adicionales
        temporal_factors.extend(self._analyze_temporal_factors(issue))
        
        # Normalizar score a 0-100
        final_urgency_score = min(100.0, urgency_score)
        
        # Generar razonamiento
        urgency_reasoning = self._generate_urgency_reasoning(
            issue, category_urgency, change_urgency, recency_score, trend_score, is_critical_path
        )
        
        return UrgencyScore(
            score=final_urgency_score,
            change_frequency=change_frequency,
            recency_score=recency_score,
            trend_score=trend_score,
            is_critical_path=is_critical_path,
            temporal_factors=temporal_factors,
            urgency_reasoning=urgency_reasoning
        )
    
    def _calculate_category_urgency(self, issue: CategorizedIssue) -> float:
        """Calcula urgencia basada en categoría."""
        category_urgencies = {
            IssueCategory.SECURITY: 40.0,
            IssueCategory.RELIABILITY: 30.0,
            IssueCategory.PERFORMANCE: 25.0,
            IssueCategory.MAINTAINABILITY: 15.0,
            IssueCategory.ARCHITECTURE: 20.0,
            IssueCategory.DOCUMENTATION: 10.0,
            IssueCategory.CODE_STYLE: 5.0
        }
        
        base_urgency = category_urgencies.get(issue.primary_category, 15.0)
        
        # Multiplicador por severidad
        if issue.original_issue:
            severity_multipliers = {
                IssueSeverity.CRITICAL: 2.0,
                IssueSeverity.HIGH: 1.5,
                IssueSeverity.MEDIUM: 1.0,
                IssueSeverity.LOW: 0.7,
                IssueSeverity.INFO: 0.3
            }
            base_urgency *= severity_multipliers.get(issue.original_issue.severity, 1.0)
        
        return base_urgency
    
    def _calculate_recency_urgency(self, issue: CategorizedIssue) -> float:
        """Calcula urgencia basada en antigüedad."""
        code_age_days = issue.context_info.code_age_days
        
        # Código muy nuevo (< 7 días) - alta urgencia para issues
        if code_age_days <= 7:
            return 15.0
        # Código reciente (< 30 días) - urgencia media
        elif code_age_days <= 30:
            return 10.0
        # Código viejo (> 180 días) - baja urgencia
        elif code_age_days > 180:
            return 2.0
        else:
            # Urgencia decrece con edad
            return max(2.0, 15.0 - (code_age_days - 7) * 0.1)
    
    def _calculate_trend_urgency(self, issue: CategorizedIssue) -> float:
        """Calcula urgencia basada en tendencias."""
        # Simplificación: urgencia basada en tipo de issue y contexto
        trend_score = 5.0
        
        # Issues en archivos que cambian frecuentemente tienen mayor urgencia
        if issue.context_info.file_change_frequency > 0.7:
            trend_score += 10.0
        
        # Issues en módulos críticos tienen urgencia de trend mayor
        if issue.context_info.module_criticality == "critical":
            trend_score += 8.0
        
        # Issues de complejidad creciente
        if (issue.original_issue and issue.original_issue.complexity_metrics and 
            issue.original_issue.complexity_metrics.cyclomatic_complexity > 20):
            trend_score += 7.0
        
        return trend_score
    
    async def _affects_critical_path(self, issue: CategorizedIssue) -> bool:
        """Verifica si el issue afecta critical path."""
        # Simplificación basada en módulo y categoría
        if issue.context_info.module_criticality == "critical":
            if issue.primary_category in [IssueCategory.SECURITY, IssueCategory.RELIABILITY]:
                return True
        
        # Issues de performance en módulos importantes
        if (issue.primary_category == IssueCategory.PERFORMANCE and 
            issue.context_info.module_criticality in ["critical", "important"]):
            return True
        
        return False
    
    def _analyze_temporal_factors(self, issue: CategorizedIssue) -> List[str]:
        """Analiza factores temporales."""
        factors = []
        
        if issue.context_info.code_age_days <= 7:
            factors.append("recently_introduced")
        
        if issue.context_info.file_change_frequency > 0.8:
            factors.append("high_change_frequency")
        
        if issue.metadata.business_impact_level in [BusinessImpactLevel.BLOCKING, BusinessImpactLevel.SIGNIFICANT]:
            factors.append("business_critical")
        
        if issue.primary_category == IssueCategory.SECURITY:
            factors.append("security_sensitive")
        
        return factors
    
    def _generate_urgency_reasoning(self, issue: CategorizedIssue, category_urgency: float,
                                  change_urgency: float, recency_score: float, trend_score: float,
                                  is_critical_path: bool) -> str:
        """Genera explicación de urgencia."""
        reasons = []
        
        if category_urgency > 25:
            reasons.append(f"{issue.primary_category.value} category urgency")
        
        if change_urgency > 10:
            reasons.append("high file change frequency")
        
        if recency_score > 10:
            reasons.append("recently modified code")
        
        if is_critical_path:
            reasons.append("affects critical system path")
        
        if not reasons:
            reasons.append("standard urgency factors")
        
        return f"Urgency factors: {', '.join(reasons)}"


class BusinessValueAssessor:
    """Evaluador de valor de negocio."""
    
    def __init__(self):
        self.value_factors = {
            "revenue_protection": 0.3,
            "cost_reduction": 0.25,
            "productivity_improvement": 0.2,
            "risk_mitigation": 0.15,
            "customer_satisfaction": 0.1
        }
    
    async def assess_business_value(self, issue: CategorizedIssue) -> BusinessValueScore:
        """
        Evalúa valor de negocio de resolver el issue.
        
        Args:
            issue: Issue categorizado
            
        Returns:
            BusinessValueScore detallado
        """
        # Calcular diferentes tipos de valor
        revenue_impact = self._calculate_revenue_impact(issue)
        cost_savings = self._calculate_cost_savings(issue)
        risk_mitigation = self._calculate_risk_mitigation_value(issue)
        productivity_improvement = self._calculate_productivity_improvement(issue)
        customer_satisfaction = self._calculate_customer_satisfaction_impact(issue)
        
        # Score general ponderado
        overall_score = (
            revenue_impact * self.value_factors["revenue_protection"] +
            cost_savings * self.value_factors["cost_reduction"] +
            productivity_improvement * self.value_factors["productivity_improvement"] +
            risk_mitigation * self.value_factors["risk_mitigation"] +
            customer_satisfaction * self.value_factors["customer_satisfaction"]
        )
        
        # Generar razonamiento
        value_reasoning = self._generate_value_reasoning(
            issue, revenue_impact, cost_savings, productivity_improvement, risk_mitigation
        )
        
        return BusinessValueScore(
            score=overall_score,
            revenue_impact=revenue_impact,
            cost_savings=cost_savings,
            risk_mitigation_value=risk_mitigation,
            productivity_improvement=productivity_improvement,
            customer_satisfaction_impact=customer_satisfaction,
            value_reasoning=value_reasoning
        )
    
    def _calculate_revenue_impact(self, issue: CategorizedIssue) -> float:
        """Calcula impacto en ingresos."""
        if issue.primary_category == IssueCategory.SECURITY:
            # Issues de seguridad pueden causar pérdidas significativas
            if issue.metadata.security_risk_score > 80:
                return 90.0
            elif issue.metadata.security_risk_score > 50:
                return 60.0
            else:
                return 30.0
        
        elif issue.primary_category == IssueCategory.PERFORMANCE:
            # Performance issues pueden afectar conversiones
            if issue.metadata.performance_impact_percentage > 20:
                return 70.0
            elif issue.metadata.performance_impact_percentage > 10:
                return 40.0
            else:
                return 20.0
        
        elif issue.primary_category == IssueCategory.USABILITY:
            return 50.0  # Usability afecta directamente ventas
        
        return 10.0  # Impacto mínimo por defecto
    
    def _calculate_cost_savings(self, issue: CategorizedIssue) -> float:
        """Calcula ahorros de costo potenciales."""
        base_savings = 20.0
        
        # Ahorros por reducción de deuda técnica
        debt_contribution = issue.metadata.technical_debt_contribution
        base_savings += debt_contribution * 2.0
        
        # Ahorros por reducción de tiempo de mantenimiento
        if issue.primary_category == IssueCategory.MAINTAINABILITY:
            base_savings += 30.0
        
        # Ahorros por prevención de bugs futuros
        if issue.primary_category == IssueCategory.RELIABILITY:
            base_savings += 25.0
        
        return min(100.0, base_savings)
    
    def _calculate_risk_mitigation_value(self, issue: CategorizedIssue) -> float:
        """Calcula valor de mitigación de riesgo."""
        risk_value = 0.0
        
        # Valor alto para mitigación de riesgos de seguridad
        if issue.primary_category == IssueCategory.SECURITY:
            risk_value = 80.0 + issue.metadata.security_risk_score * 0.2
        
        # Valor por mitigación de riesgos de confiabilidad
        elif issue.primary_category == IssueCategory.RELIABILITY:
            risk_value = 60.0
        
        # Valor por mitigación de riesgos de performance
        elif issue.primary_category == IssueCategory.PERFORMANCE:
            risk_value = 40.0
        
        else:
            risk_value = 20.0
        
        # Ajustar por riesgo de regresión
        risk_value -= issue.metadata.regression_risk_score * 0.3
        
        return max(0.0, min(100.0, risk_value))
    
    def _calculate_productivity_improvement(self, issue: CategorizedIssue) -> float:
        """Calcula mejora de productividad."""
        productivity_score = 20.0
        
        # Mejoras por categoría
        if issue.primary_category == IssueCategory.MAINTAINABILITY:
            productivity_score = 70.0
        elif issue.primary_category == IssueCategory.DOCUMENTATION:
            productivity_score = 50.0
        elif issue.primary_category == IssueCategory.CODE_STYLE:
            productivity_score = 30.0
        
        # Boost para módulos que cambian frecuentemente
        if issue.context_info.file_change_frequency > 0.6:
            productivity_score += 20.0
        
        return min(100.0, productivity_score)
    
    def _calculate_customer_satisfaction_impact(self, issue: CategorizedIssue) -> float:
        """Calcula impacto en satisfacción del cliente."""
        if issue.primary_category == IssueCategory.USABILITY:
            return 80.0
        elif issue.primary_category == IssueCategory.PERFORMANCE:
            return 60.0
        elif issue.primary_category == IssueCategory.RELIABILITY:
            return 70.0
        elif issue.primary_category == IssueCategory.SECURITY:
            return 50.0  # Security issues afectan confianza
        else:
            return 20.0
    
    def _generate_value_reasoning(self, issue: CategorizedIssue, revenue: float, cost_savings: float,
                                productivity: float, risk_mitigation: float) -> str:
        """Genera explicación de valor de negocio."""
        reasons = []
        
        if revenue > 50:
            reasons.append("revenue protection")
        if cost_savings > 50:
            reasons.append("cost reduction")
        if productivity > 50:
            reasons.append("productivity gains")
        if risk_mitigation > 50:
            reasons.append("risk mitigation")
        
        if not reasons:
            reasons.append("minimal business value")
        
        return f"Business value: {', '.join(reasons)}"


class RiskAnalyzer:
    """Analizador de riesgo asociado a issues."""
    
    async def calculate_risk(self, issue: CategorizedIssue) -> RiskScore:
        """
        Calcula score de riesgo total.
        
        Args:
            issue: Issue categorizado
            
        Returns:
            RiskScore detallado
        """
        # Calcular diferentes tipos de riesgo
        security_risk = self._calculate_security_risk(issue)
        stability_risk = self._calculate_stability_risk(issue)
        performance_risk = self._calculate_performance_risk(issue)
        maintenance_risk = self._calculate_maintenance_risk(issue)
        compliance_risk = self._calculate_compliance_risk(issue)
        reputation_risk = self._calculate_reputation_risk(issue)
        
        # Score general (máximo de los riesgos individuales)
        overall_risk = max(security_risk, stability_risk, performance_risk, 
                          maintenance_risk, compliance_risk, reputation_risk)
        
        # Generar razonamiento
        risk_reasoning = self._generate_risk_reasoning(
            issue, security_risk, stability_risk, performance_risk, maintenance_risk
        )
        
        return RiskScore(
            score=overall_risk,
            security_risk=security_risk,
            stability_risk=stability_risk,
            performance_risk=performance_risk,
            maintenance_risk=maintenance_risk,
            compliance_risk=compliance_risk,
            reputation_risk=reputation_risk,
            risk_reasoning=risk_reasoning
        )
    
    def _calculate_security_risk(self, issue: CategorizedIssue) -> float:
        """Calcula riesgo de seguridad."""
        if issue.primary_category == IssueCategory.SECURITY:
            return issue.metadata.security_risk_score
        else:
            return 0.0
    
    def _calculate_stability_risk(self, issue: CategorizedIssue) -> float:
        """Calcula riesgo de estabilidad."""
        base_risk = 10.0
        
        if issue.primary_category == IssueCategory.RELIABILITY:
            base_risk = 60.0
            
            # Incrementar por patrones de riesgo
            if issue.original_issue and any(keyword in issue.original_issue.message.lower() 
                                          for keyword in ['crash', 'exception', 'null']):
                base_risk += 20.0
        
        # Riesgo por complejidad alta
        if (issue.original_issue and issue.original_issue.complexity_metrics and 
            issue.original_issue.complexity_metrics.cyclomatic_complexity > 20):
            base_risk += 15.0
        
        return min(100.0, base_risk)
    
    def _calculate_performance_risk(self, issue: CategorizedIssue) -> float:
        """Calcula riesgo de performance."""
        if issue.primary_category == IssueCategory.PERFORMANCE:
            base_risk = issue.metadata.performance_impact_percentage * 2.0
            
            # Boost para memoria y recursos
            if issue.original_issue and 'memory' in issue.original_issue.message.lower():
                base_risk += 20.0
            
            return min(100.0, base_risk)
        
        return 0.0
    
    def _calculate_maintenance_risk(self, issue: CategorizedIssue) -> float:
        """Calcula riesgo de mantenimiento."""
        base_risk = issue.metadata.technical_debt_contribution
        
        # Incrementar por falta de tests
        if issue.context_info.test_coverage_percentage < 50:
            base_risk += 20.0
        
        # Incrementar por alta frecuencia de cambios
        base_risk += issue.context_info.file_change_frequency * 25.0
        
        return min(100.0, base_risk)
    
    def _calculate_compliance_risk(self, issue: CategorizedIssue) -> float:
        """Calcula riesgo de cumplimiento."""
        # Simplificación: mainly for security and documentation
        if issue.primary_category == IssueCategory.SECURITY:
            return 50.0
        elif issue.primary_category == IssueCategory.DOCUMENTATION:
            return 20.0
        else:
            return 5.0
    
    def _calculate_reputation_risk(self, issue: CategorizedIssue) -> float:
        """Calcula riesgo reputacional."""
        if issue.primary_category == IssueCategory.SECURITY:
            return 70.0
        elif issue.primary_category == IssueCategory.PERFORMANCE:
            return 40.0
        elif issue.primary_category == IssueCategory.USABILITY:
            return 50.0
        else:
            return 10.0
    
    def _generate_risk_reasoning(self, issue: CategorizedIssue, security: float, stability: float,
                               performance: float, maintenance: float) -> str:
        """Genera explicación de riesgo."""
        risks = []
        
        if security > 50:
            risks.append("security vulnerabilities")
        if stability > 50:
            risks.append("system stability concerns")
        if performance > 50:
            risks.append("performance degradation risk")
        if maintenance > 50:
            risks.append("maintenance burden increase")
        
        if not risks:
            risks.append("minimal risk factors")
        
        return f"Risk factors: {', '.join(risks)}"


class PriorityCalculator:
    """Calculadora principal de prioridades."""
    
    def __init__(self, config: Optional[PriorityConfig] = None):
        """
        Inicializa el calculador de prioridades.
        
        Args:
            config: Configuración del calculador
        """
        self.config = config or PriorityConfig()
        self.impact_analyzer = ImpactAnalyzer()
        self.urgency_calculator = UrgencyCalculator()
        self.business_value_assessor = BusinessValueAssessor()
        self.risk_analyzer = RiskAnalyzer()
    
    async def calculate_priorities(self, issues: List[CategorizedIssue]) -> PriorityCalculationResult:
        """
        Calcula prioridades para lista de issues.
        
        Args:
            issues: Lista de issues categorizados
            
        Returns:
            PriorityCalculationResult completo
        """
        start_time = time.time()
        
        logger.info(f"Iniciando cálculo de prioridades para {len(issues)} issues")
        
        # Calcular prioridad para cada issue
        prioritized_issues = []
        total_priority_score = 0.0
        
        for issue in issues:
            try:
                priority = await self._calculate_single_priority(issue)
                issue.metadata.priority = priority
                prioritized_issues.append(issue)
                total_priority_score += priority.score
            except Exception as e:
                logger.warning(f"Error calculando prioridad para issue {issue.id.value}: {e}")
                # Asignar prioridad por defecto
                default_priority = self._create_default_priority(issue)
                issue.metadata.priority = default_priority
                prioritized_issues.append(issue)
                total_priority_score += default_priority.score
        
        # Ordenar por score de prioridad (mayor a menor)
        prioritized_issues.sort(key=lambda i: i.get_priority_score(), reverse=True)
        
        # Calcular distribución de prioridades
        priority_distribution = self._calculate_priority_distribution(prioritized_issues)
        
        # Calcular estadísticas
        avg_priority_score = total_priority_score / len(prioritized_issues) if prioritized_issues else 0.0
        high_priority_count = sum(1 for issue in prioritized_issues if issue.is_high_priority())
        high_priority_percentage = (high_priority_count / len(prioritized_issues)) * 100.0 if prioritized_issues else 0.0
        
        total_time = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Cálculo de prioridades completado: {len(prioritized_issues)} issues priorizados, "
            f"promedio={avg_priority_score:.1f}, {high_priority_percentage:.1f}% alta prioridad en {total_time}ms"
        )
        
        return PriorityCalculationResult(
            prioritized_issues=prioritized_issues,
            priority_distribution=priority_distribution,
            calculation_time_ms=total_time,
            average_priority_score=avg_priority_score,
            high_priority_percentage=high_priority_percentage
        )
    
    async def _calculate_single_priority(self, issue: CategorizedIssue) -> IssuePriority:
        """Calcula prioridad de un issue individual."""
        # Calcular componentes de prioridad
        impact = await self.impact_analyzer.calculate_impact(issue)
        urgency = await self.urgency_calculator.calculate_urgency(issue)
        business_value = await self.business_value_assessor.assess_business_value(issue)
        risk = await self.risk_analyzer.calculate_risk(issue)
        
        # Calcular score ponderado
        base_score = (
            impact.score * self.config.impact_weight +
            urgency.score * self.config.urgency_weight +
            business_value.score * self.config.business_value_weight +
            risk.score * self.config.risk_weight
        )
        
        # Aplicar penalización por complejidad de fix
        complexity_penalty = issue.metadata.fix_complexity_score * self.config.complexity_penalty_factor
        
        # Aplicar factor de tiempo de fix
        time_penalty = (issue.metadata.estimated_fix_time_hours or 1.0) * self.config.fix_time_factor
        
        # Score final
        final_score = max(0.0, min(100.0, base_score - complexity_penalty - time_penalty))
        
        # Determinar nivel de prioridad
        priority_level = self._score_to_priority_level(final_score)
        
        # Generar razonamiento
        reasoning = self._generate_priority_reasoning(issue, impact, urgency, business_value, risk, final_score)
        
        # Breakdown de factores
        factors_breakdown = {
            "impact": impact.score,
            "urgency": urgency.score,
            "business_value": business_value.score,
            "risk": risk.score,
            "complexity_penalty": complexity_penalty,
            "time_penalty": time_penalty
        }
        
        return IssuePriority(
            score=final_score,
            level=priority_level,
            impact=impact,
            urgency=urgency,
            business_value=business_value,
            risk=risk,
            reasoning=reasoning,
            confidence=self._calculate_priority_confidence(issue, impact, urgency),
            factors_breakdown=factors_breakdown
        )
    
    def _score_to_priority_level(self, score: float) -> PriorityLevel:
        """Convierte score a nivel de prioridad."""
        if score >= 80.0:
            return PriorityLevel.CRITICAL
        elif score >= 65.0:
            return PriorityLevel.HIGH
        elif score >= 40.0:
            return PriorityLevel.MEDIUM
        elif score >= 20.0:
            return PriorityLevel.LOW
        else:
            return PriorityLevel.LOWEST
    
    def _calculate_priority_confidence(self, issue: CategorizedIssue, impact: ImpactScore, urgency: UrgencyScore) -> float:
        """Calcula confianza en la prioridad asignada."""
        base_confidence = 0.7
        
        # Mayor confianza para categorías bien definidas
        primary_cat_confidence = issue.confidence_scores.get(issue.primary_category.value, 0.5)
        base_confidence += primary_cat_confidence * 0.2
        
        # Mayor confianza cuando hay contexto rico
        if issue.context_info.module_criticality in ["critical", "important"]:
            base_confidence += 0.1
        
        if issue.context_info.test_coverage_percentage > 70:
            base_confidence += 0.05
        
        # Menor confianza para scores muy bajos o muy altos (puede ser outlier)
        if impact.score < 20 or impact.score > 90:
            base_confidence -= 0.1
        
        return min(1.0, max(0.3, base_confidence))
    
    def _generate_priority_reasoning(self, issue: CategorizedIssue, impact: ImpactScore,
                                   urgency: UrgencyScore, business_value: BusinessValueScore,
                                   risk: RiskScore, final_score: float) -> str:
        """Genera explicación de la prioridad."""
        reasoning_parts = []
        
        reasoning_parts.append(f"Priority score: {final_score:.1f}")
        reasoning_parts.append(f"Category: {issue.primary_category.value}")
        
        if impact.score > 60:
            reasoning_parts.append("high impact")
        if urgency.score > 60:
            reasoning_parts.append("urgent")
        if business_value.score > 60:
            reasoning_parts.append("valuable for business")
        if risk.score > 60:
            reasoning_parts.append("high risk")
        
        if issue.context_info.module_criticality == "critical":
            reasoning_parts.append("critical module")
        
        return "; ".join(reasoning_parts)
    
    def _create_default_priority(self, issue: CategorizedIssue) -> IssuePriority:
        """Crea prioridad por defecto cuando falla el cálculo."""
        # Prioridad basada en categoría y severidad
        base_score = 50.0
        
        if issue.primary_category == IssueCategory.SECURITY:
            base_score = 80.0
        elif issue.primary_category == IssueCategory.RELIABILITY:
            base_score = 65.0
        elif issue.primary_category == IssueCategory.PERFORMANCE:
            base_score = 60.0
        
        priority_level = self._score_to_priority_level(base_score)
        
        return IssuePriority(
            score=base_score,
            level=priority_level,
            impact=ImpactScore(score=base_score),
            urgency=UrgencyScore(score=base_score * 0.7),
            business_value=BusinessValueScore(score=base_score * 0.6),
            risk=RiskScore(score=base_score * 0.8),
            reasoning=f"Default priority for {issue.primary_category.value} issue",
            confidence=0.5
        )
    
    def _calculate_priority_distribution(self, issues: List[CategorizedIssue]) -> Dict[PriorityLevel, int]:
        """Calcula distribución de prioridades."""
        distribution = {level: 0 for level in PriorityLevel}
        
        for issue in issues:
            if issue.metadata.priority:
                distribution[issue.metadata.priority.level] += 1
        
        return distribution
