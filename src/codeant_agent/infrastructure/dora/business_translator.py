"""
Business metrics translator implementation.
"""

from decimal import Decimal
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from codeant_agent.domain.dora import (
    BusinessMetricsTranslatorService,
    BusinessMetrics,
    TechnicalDebtBusinessMetrics,
    SecurityBusinessMetrics,
    TeamProductivityMetrics,
    ROIAnalysis,
    ROIScenario,
    TrendDirection,
    RiskLevel,
    DORAMetrics,
    DORAPerformanceCategory
)
import logging

logger = logging.getLogger(__name__)


class BusinessMetricsTranslator(BusinessMetricsTranslatorService):
    """Translator for converting technical metrics to business value."""
    
    def __init__(
        self,
        default_hourly_rate: float = 150.0,
        currency: str = "USD"
    ):
        self.default_hourly_rate = default_hourly_rate
        self.currency = currency
        
        # Industry averages for calculations
        self.industry_averages = {
            "developer_productivity_loss_poor_quality": 0.25,  # 25% productivity loss
            "incident_cost_per_hour": 5000.0,  # Average cost of downtime
            "security_breach_average_cost": 4350000.0,  # IBM Cost of Data Breach Report 2023
            "developer_satisfaction_impact": 0.15,  # 15% productivity impact
            "time_to_market_delay_factor": 1.5  # 50% slower with poor quality
        }
    
    async def translate_to_business_metrics(
        self,
        technical_data: Dict[str, Any]
    ) -> BusinessMetrics:
        """Translate technical metrics to business metrics."""
        try:
            # Extract key technical metrics
            quality_score = technical_data.get("quality_metrics", {}).get("overall_score", 0)
            technical_debt_hours = technical_data.get("technical_debt", {}).get("total_hours", 0)
            security_vulnerabilities = technical_data.get("security_vulnerabilities", [])
            dora_metrics = technical_data.get("dora_metrics")
            
            # Calculate business metrics
            technical_debt = await self._calculate_technical_debt_metrics(
                technical_debt_hours,
                quality_score
            )
            
            security_metrics = await self._calculate_security_business_metrics(
                security_vulnerabilities,
                technical_data.get("compliance_status", {})
            )
            
            team_productivity = await self._calculate_team_productivity_metrics(
                quality_score,
                dora_metrics,
                technical_debt_hours
            )
            
            roi_analysis = await self._calculate_roi_analysis(
                quality_score,
                technical_debt_hours,
                team_productivity
            )
            
            quality_trend = self._determine_quality_trend(
                technical_data.get("historical_quality", [])
            )
            
            return BusinessMetrics(
                overall_quality_score=quality_score,
                quality_trend=quality_trend,
                technical_debt=technical_debt,
                security_metrics=security_metrics,
                dora_metrics=dora_metrics,
                team_productivity=team_productivity,
                roi_analysis=roi_analysis
            )
            
        except Exception as e:
            logger.error(f"Failed to translate to business metrics: {e}")
            raise
    
    async def calculate_technical_debt_cost(
        self,
        technical_debt_hours: float,
        hourly_rate: float = None
    ) -> Dict[str, Any]:
        """Calculate technical debt in monetary terms."""
        rate = hourly_rate or self.default_hourly_rate
        
        # Direct cost of fixing technical debt
        direct_cost = Decimal(str(technical_debt_hours * rate))
        
        # Calculate interest (ongoing cost of not fixing)
        # Assumes 2.5% monthly compound interest on technical debt
        monthly_interest_rate = 0.025
        monthly_interest = direct_cost * Decimal(str(monthly_interest_rate))
        yearly_interest = monthly_interest * 12
        
        # Calculate different payoff scenarios
        payoff_scenarios = []
        
        # Scenario 1: Fix immediately
        payoff_scenarios.append({
            "name": "Immediate Fix",
            "timeline_months": 1,
            "total_cost": float(direct_cost),
            "interest_saved": float(yearly_interest),
            "break_even_months": 0
        })
        
        # Scenario 2: Fix over 6 months
        six_month_cost = direct_cost
        six_month_interest = sum(
            monthly_interest * Decimal(str((1 + monthly_interest_rate) ** i))
            for i in range(6)
        )
        payoff_scenarios.append({
            "name": "6-Month Plan",
            "timeline_months": 6,
            "total_cost": float(six_month_cost + six_month_interest),
            "interest_saved": float(yearly_interest - six_month_interest),
            "break_even_months": 6
        })
        
        # Scenario 3: Fix over 12 months
        twelve_month_cost = direct_cost
        twelve_month_interest = sum(
            monthly_interest * Decimal(str((1 + monthly_interest_rate) ** i))
            for i in range(12)
        )
        payoff_scenarios.append({
            "name": "12-Month Plan",
            "timeline_months": 12,
            "total_cost": float(twelve_month_cost + twelve_month_interest),
            "interest_saved": 0,
            "break_even_months": 12
        })
        
        # Scenario 4: Do nothing (cost over 2 years)
        do_nothing_interest = sum(
            monthly_interest * Decimal(str((1 + monthly_interest_rate) ** i))
            for i in range(24)
        )
        payoff_scenarios.append({
            "name": "Do Nothing (2-year cost)",
            "timeline_months": 24,
            "total_cost": float(do_nothing_interest),
            "interest_saved": float(-do_nothing_interest),
            "break_even_months": float('inf')
        })
        
        return {
            "total_hours": technical_debt_hours,
            "hourly_rate": rate,
            "estimated_cost": float(direct_cost),
            "monthly_interest": float(monthly_interest),
            "yearly_interest": float(yearly_interest),
            "payoff_scenarios": payoff_scenarios,
            "currency": self.currency
        }
    
    async def calculate_roi_scenarios(
        self,
        current_metrics: Dict[str, Any],
        improvement_scenarios: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate ROI for different improvement scenarios."""
        roi_scenarios = []
        
        current_quality = current_metrics.get("quality_score", 70)
        current_productivity = current_metrics.get("team_productivity", 0.7)
        current_incident_rate = current_metrics.get("incident_rate", 0.15)
        
        for scenario in improvement_scenarios:
            investment = Decimal(str(scenario["investment_hours"] * self.default_hourly_rate))
            quality_improvement = scenario["expected_quality_improvement"]
            
            # Calculate expected returns
            
            # 1. Productivity gains
            new_quality = min(100, current_quality + quality_improvement)
            productivity_gain = (new_quality - current_quality) / 100 * 0.2  # 20% max gain
            
            # Annual productivity value (assuming 10 developers)
            annual_dev_hours = 10 * 2000  # 10 devs * 2000 hours/year
            productivity_value = Decimal(str(
                annual_dev_hours * self.default_hourly_rate * productivity_gain
            ))
            
            # 2. Reduced incident costs
            incident_reduction = quality_improvement / 100 * 0.5  # 50% reduction possible
            new_incident_rate = max(0.05, current_incident_rate * (1 - incident_reduction))
            
            # Annual incident cost savings
            incidents_per_year = 52 * current_incident_rate  # Weekly deployments
            incident_cost_savings = Decimal(str(
                incidents_per_year * incident_reduction *
                self.industry_averages["incident_cost_per_hour"] * 4  # 4 hour average
            ))
            
            # 3. Faster time to market
            time_to_market_improvement = quality_improvement / 100 * 0.3  # 30% max improvement
            
            # Estimated value of faster delivery (opportunity cost)
            opportunity_value = Decimal(str(investment)) * Decimal(str(time_to_market_improvement))
            
            # Total annual return
            total_annual_return = productivity_value + incident_cost_savings + opportunity_value
            
            # Calculate payback period
            if total_annual_return > 0:
                payback_months = int((investment / total_annual_return) * 12)
            else:
                payback_months = float('inf')
            
            # Calculate 3-year ROI
            three_year_return = total_annual_return * 3 - investment
            three_year_roi = float((three_year_return / investment) * 100) if investment > 0 else 0
            
            roi_scenarios.append({
                "scenario_name": scenario["name"],
                "investment_amount": float(investment),
                "expected_annual_return": float(total_annual_return),
                "three_year_roi_percentage": three_year_roi,
                "payback_period_months": payback_months,
                "breakdown": {
                    "productivity_gains": float(productivity_value),
                    "incident_reduction": float(incident_cost_savings),
                    "time_to_market": float(opportunity_value)
                },
                "confidence_level": self._calculate_confidence_level(quality_improvement)
            })
        
        return sorted(roi_scenarios, key=lambda x: x["three_year_roi_percentage"], reverse=True)
    
    async def assess_business_risks(
        self,
        security_vulnerabilities: List[Dict[str, Any]],
        quality_issues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess business risks from technical issues."""
        # Calculate security risk
        critical_vulns = len([v for v in security_vulnerabilities if v.get("severity") == "critical"])
        high_vulns = len([v for v in security_vulnerabilities if v.get("severity") == "high"])
        
        # Security risk score (0-10)
        security_risk_score = min(10, critical_vulns * 3 + high_vulns * 1)
        
        # Determine overall risk level
        if security_risk_score >= 8:
            overall_risk_level = RiskLevel.CRITICAL
        elif security_risk_score >= 6:
            overall_risk_level = RiskLevel.HIGH
        elif security_risk_score >= 4:
            overall_risk_level = RiskLevel.MEDIUM
        elif security_risk_score >= 2:
            overall_risk_level = RiskLevel.LOW
        else:
            overall_risk_level = RiskLevel.MINIMAL
        
        # Calculate potential financial impact
        breach_probability = min(0.5, security_risk_score / 20)  # Max 50% probability
        potential_breach_cost = Decimal(str(self.industry_averages["security_breach_average_cost"]))
        expected_loss = potential_breach_cost * Decimal(str(breach_probability))
        
        # Quality-related business risks
        critical_quality_issues = len([q for q in quality_issues if q.get("severity") == "critical"])
        
        # Reputation risk
        reputation_risk = "high" if critical_vulns > 0 or critical_quality_issues > 5 else "medium" if high_vulns > 5 else "low"
        
        # Compliance risk
        compliance_violations = len([v for v in security_vulnerabilities if v.get("compliance_impact", False)])
        compliance_risk = "high" if compliance_violations > 0 else "low"
        
        # Insurance implications
        insurance_implications = {
            "premium_impact": "increase" if overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else "stable",
            "coverage_concerns": critical_vulns > 0,
            "recommended_actions": self._get_insurance_recommendations(overall_risk_level)
        }
        
        return {
            "overall_risk_level": overall_risk_level,
            "security_risk_score": security_risk_score,
            "potential_financial_impact": float(expected_loss),
            "breach_probability": breach_probability,
            "reputation_risk": reputation_risk,
            "compliance_risk": compliance_risk,
            "insurance_implications": insurance_implications,
            "risk_factors": {
                "critical_vulnerabilities": critical_vulns,
                "high_vulnerabilities": high_vulns,
                "compliance_violations": compliance_violations,
                "critical_quality_issues": critical_quality_issues
            },
            "mitigation_priorities": self._get_mitigation_priorities(
                critical_vulns,
                high_vulns,
                critical_quality_issues
            )
        }
    
    async def _calculate_technical_debt_metrics(
        self,
        technical_debt_hours: float,
        quality_score: float
    ) -> TechnicalDebtBusinessMetrics:
        """Calculate technical debt business metrics."""
        # Base calculation
        debt_cost = await self.calculate_technical_debt_cost(technical_debt_hours)
        
        # Adjust for quality score impact
        quality_factor = (100 - quality_score) / 100
        adjusted_monthly_interest = Decimal(str(debt_cost["monthly_interest"])) * Decimal(str(1 + quality_factor))
        
        return TechnicalDebtBusinessMetrics(
            total_hours=technical_debt_hours,
            estimated_cost=Decimal(str(debt_cost["estimated_cost"])),
            monthly_interest=adjusted_monthly_interest,
            payoff_scenarios=debt_cost["payoff_scenarios"]
        )
    
    async def _calculate_security_business_metrics(
        self,
        vulnerabilities: List[Dict[str, Any]],
        compliance_status: Dict[str, Any]
    ) -> SecurityBusinessMetrics:
        """Calculate security business metrics."""
        risk_assessment = await self.assess_business_risks(vulnerabilities, [])
        
        return SecurityBusinessMetrics(
            overall_risk_level=risk_assessment["overall_risk_level"],
            potential_impact=Decimal(str(risk_assessment["potential_financial_impact"])),
            compliance_status=compliance_status,
            insurance_implications=risk_assessment["insurance_implications"]
        )
    
    async def _calculate_team_productivity_metrics(
        self,
        quality_score: float,
        dora_metrics: Optional[DORAMetrics],
        technical_debt_hours: float
    ) -> TeamProductivityMetrics:
        """Calculate team productivity metrics."""
        # Base productivity score from quality
        base_productivity = quality_score / 100
        
        # Velocity impact from technical debt
        # Assumes 1000 hours of debt = 10% velocity loss
        debt_impact = min(0.5, technical_debt_hours / 10000)
        velocity_impact = 1 - debt_impact
        
        # Quality impact on productivity
        quality_impact = base_productivity
        
        # Developer satisfaction estimate
        # Based on quality and DORA performance
        satisfaction_factors = []
        
        if quality_score >= 80:
            satisfaction_factors.append(0.9)
        elif quality_score >= 60:
            satisfaction_factors.append(0.7)
        else:
            satisfaction_factors.append(0.5)
        
        if dora_metrics and dora_metrics.performance_rating.overall_category == DORAPerformanceCategory.ELITE:
            satisfaction_factors.append(0.95)
        elif dora_metrics and dora_metrics.performance_rating.overall_category == DORAPerformanceCategory.HIGH:
            satisfaction_factors.append(0.8)
        else:
            satisfaction_factors.append(0.6)
        
        developer_satisfaction = sum(satisfaction_factors) / len(satisfaction_factors) if satisfaction_factors else 0.5
        
        # Time to market impact
        # Poor quality and high debt slow down delivery
        time_to_market_impact = velocity_impact * quality_impact
        
        # Overall productivity score
        overall_score = (
            velocity_impact * 0.3 +
            quality_impact * 0.3 +
            developer_satisfaction * 0.2 +
            time_to_market_impact * 0.2
        ) * 100
        
        return TeamProductivityMetrics(
            overall_score=overall_score,
            velocity_impact=velocity_impact,
            quality_impact=quality_impact,
            developer_satisfaction=developer_satisfaction,
            time_to_market_impact=time_to_market_impact
        )
    
    async def _calculate_roi_analysis(
        self,
        quality_score: float,
        technical_debt_hours: float,
        team_productivity: TeamProductivityMetrics
    ) -> ROIAnalysis:
        """Calculate ROI analysis for quality improvements."""
        # Current efficiency (0-1 scale)
        current_efficiency = team_productivity.overall_score / 100
        
        # Maximum realistic improvement
        max_quality = 95
        max_efficiency = 0.95
        
        # Improvement potential
        quality_gap = max_quality - quality_score
        improvement_potential = (quality_gap / (100 - quality_score)) * 100 if quality_score < 100 else 0
        
        # Define improvement scenarios
        scenarios = [
            {
                "name": "Quick Wins",
                "investment_hours": technical_debt_hours * 0.2,
                "expected_quality_improvement": quality_gap * 0.3
            },
            {
                "name": "Moderate Investment",
                "investment_hours": technical_debt_hours * 0.5,
                "expected_quality_improvement": quality_gap * 0.6
            },
            {
                "name": "Comprehensive Overhaul",
                "investment_hours": technical_debt_hours * 0.8,
                "expected_quality_improvement": quality_gap * 0.85
            }
        ]
        
        # Calculate ROI for each scenario
        current_metrics = {
            "quality_score": quality_score,
            "team_productivity": current_efficiency,
            "incident_rate": 0.15  # Assumed baseline
        }
        
        roi_scenarios = await self.calculate_roi_scenarios(current_metrics, scenarios)
        
        # Convert to ROIScenario entities
        investment_scenarios = []
        for scenario in roi_scenarios:
            investment_scenarios.append(ROIScenario(
                scenario_name=scenario["scenario_name"],
                investment_amount=Decimal(str(scenario["investment_amount"])),
                expected_return=Decimal(str(scenario["expected_annual_return"])),
                payback_period_months=scenario["payback_period_months"],
                confidence_level=scenario["confidence_level"]
            ))
        
        # Payback periods map
        payback_periods = {
            scenario["scenario_name"]: scenario["payback_period_months"]
            for scenario in roi_scenarios
        }
        
        return ROIAnalysis(
            current_efficiency=current_efficiency,
            improvement_potential=improvement_potential,
            investment_scenarios=investment_scenarios,
            payback_periods=payback_periods
        )
    
    def _determine_quality_trend(self, historical_quality: List[Dict[str, Any]]) -> TrendDirection:
        """Determine quality trend from historical data."""
        if len(historical_quality) < 2:
            return TrendDirection.STABLE
        
        # Get last 6 data points
        recent_history = historical_quality[-6:] if len(historical_quality) >= 6 else historical_quality
        
        # Extract quality scores
        scores = [h.get("quality_score", 0) for h in recent_history]
        
        # Calculate simple linear regression
        if len(set(scores)) == 1:  # All scores are the same
            return TrendDirection.STABLE
        
        # Calculate average change
        changes = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        avg_change = sum(changes) / len(changes) if changes else 0
        
        # Calculate volatility
        if len(changes) > 1:
            variance = sum((c - avg_change) ** 2 for c in changes) / len(changes)
            volatility = variance ** 0.5
        else:
            volatility = 0
        
        # Determine trend
        if volatility > abs(avg_change) * 2:  # High volatility
            return TrendDirection.VOLATILE
        elif avg_change > 1:
            return TrendDirection.IMPROVING
        elif avg_change < -1:
            return TrendDirection.DEGRADING
        else:
            return TrendDirection.STABLE
    
    def _calculate_confidence_level(self, quality_improvement: float) -> float:
        """Calculate confidence level for ROI predictions."""
        # Higher improvements have lower confidence
        if quality_improvement <= 10:
            return 0.9
        elif quality_improvement <= 20:
            return 0.8
        elif quality_improvement <= 30:
            return 0.7
        elif quality_improvement <= 40:
            return 0.6
        else:
            return 0.5
    
    def _get_insurance_recommendations(self, risk_level: RiskLevel) -> List[str]:
        """Get insurance recommendations based on risk level."""
        if risk_level == RiskLevel.CRITICAL:
            return [
                "Immediate security audit required",
                "Review cyber insurance coverage limits",
                "Implement incident response plan",
                "Consider additional breach coverage"
            ]
        elif risk_level == RiskLevel.HIGH:
            return [
                "Schedule security assessment",
                "Review current coverage adequacy",
                "Implement security training program"
            ]
        elif risk_level == RiskLevel.MEDIUM:
            return [
                "Monitor security metrics closely",
                "Maintain current coverage",
                "Plan security improvements"
            ]
        else:
            return [
                "Continue current security practices",
                "Annual coverage review sufficient"
            ]
    
    def _get_mitigation_priorities(
        self,
        critical_vulns: int,
        high_vulns: int,
        critical_quality_issues: int
    ) -> List[str]:
        """Get prioritized mitigation recommendations."""
        priorities = []
        
        if critical_vulns > 0:
            priorities.append(f"Fix {critical_vulns} critical security vulnerabilities immediately")
        
        if high_vulns > 5:
            priorities.append(f"Address {high_vulns} high severity vulnerabilities within 30 days")
        
        if critical_quality_issues > 5:
            priorities.append(f"Resolve {critical_quality_issues} critical quality issues affecting stability")
        
        if not priorities:
            priorities.append("Maintain current security posture")
        
        return priorities
