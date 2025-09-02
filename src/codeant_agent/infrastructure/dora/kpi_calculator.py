"""
KPI calculator implementation for organizations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
from statistics import mean, median

from codeant_agent.domain.dora import (
    KPICalculatorService,
    OrganizationKPIs,
    QualityKPIs,
    VelocityKPIs,
    SecurityKPIs,
    CostKPIs,
    TeamKPIs,
    TrendDirection,
    TimeRange
)
import logging

logger = logging.getLogger(__name__)


class KPICalculator(KPICalculatorService):
    """Calculator for organization-wide KPIs."""
    
    def __init__(
        self,
        project_repo: Any,  # ProjectRepository
        metrics_repo: Any,  # MetricsRepository
        security_repo: Any,  # SecurityRepository
        team_repo: Any      # TeamRepository
    ):
        self.project_repo = project_repo
        self.metrics_repo = metrics_repo
        self.security_repo = security_repo
        self.team_repo = team_repo
        
        # Default configuration
        self.default_hourly_rate = 150.0
        self.target_quality_score = 85.0
        self.target_security_score = 90.0
        self.target_productivity = 85.0
    
    async def calculate_organization_kpis(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> OrganizationKPIs:
        """Calculate all KPIs for an organization."""
        try:
            # Calculate all KPI categories in parallel
            results = await asyncio.gather(
                self.calculate_quality_kpis(organization_id, time_range),
                self.calculate_velocity_kpis(organization_id, time_range),
                self.calculate_security_kpis(organization_id, time_range),
                self.calculate_cost_kpis(organization_id, time_range),
                self.calculate_team_kpis(organization_id, time_range),
                return_exceptions=True
            )
            
            # Handle any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error calculating KPI category {i}: {result}")
                    raise result
            
            quality_kpis_dict, velocity_kpis_dict, security_kpis_dict, cost_kpis_dict, team_kpis_dict = results
            
            # Convert dictionaries to entities
            quality_kpis = QualityKPIs(**quality_kpis_dict)
            velocity_kpis = VelocityKPIs(**velocity_kpis_dict)
            security_kpis = SecurityKPIs(**security_kpis_dict)
            cost_kpis = CostKPIs(**cost_kpis_dict)
            team_kpis = TeamKPIs(**team_kpis_dict)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                quality_kpis,
                velocity_kpis,
                security_kpis,
                cost_kpis,
                team_kpis
            )
            
            return OrganizationKPIs(
                velocity=velocity_kpis,
                quality=quality_kpis,
                security=security_kpis,
                cost=cost_kpis,
                team=team_kpis,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate organization KPIs: {e}")
            raise
    
    async def calculate_quality_kpis(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate quality KPIs."""
        # Get all projects for the organization
        projects = await self.project_repo.get_organization_projects(organization_id)
        
        if not projects:
            return self._empty_quality_kpis()
        
        # Aggregate metrics across projects
        total_quality_score = 0.0
        total_weight = 0.0
        total_technical_debt_hours = 0.0
        total_issues = 0
        critical_issues = 0
        total_files = 0
        total_coverage = 0.0
        coverage_count = 0
        total_maintainability = 0.0
        
        for project in projects:
            # Get project metrics
            metrics = await self.metrics_repo.get_project_metrics(
                project["id"],
                time_range
            )
            
            if not metrics:
                continue
            
            # Weight by project size (lines of code)
            weight = metrics.get("lines_of_code", 1000) / 1000  # Normalize
            
            # Aggregate scores
            total_quality_score += metrics.get("quality_score", 0) * weight
            total_weight += weight
            
            # Aggregate other metrics
            total_technical_debt_hours += metrics.get("technical_debt_hours", 0)
            total_issues += metrics.get("total_issues", 0)
            critical_issues += metrics.get("critical_issues", 0)
            total_files += metrics.get("total_files", 0)
            
            # Coverage (only count if available)
            if metrics.get("code_coverage") is not None:
                total_coverage += metrics.get("code_coverage", 0)
                coverage_count += 1
            
            # Maintainability
            total_maintainability += metrics.get("maintainability_index", 50) * weight
        
        # Calculate weighted averages
        overall_quality_score = total_quality_score / total_weight if total_weight > 0 else 0
        maintainability_index = total_maintainability / total_weight if total_weight > 0 else 50
        code_coverage_percentage = total_coverage / coverage_count if coverage_count > 0 else 0
        debt_per_file = total_technical_debt_hours / total_files if total_files > 0 else 0
        
        # Calculate trend
        quality_trend = await self._calculate_quality_trend(organization_id, time_range)
        
        # Calculate issue resolution rate
        issue_resolution_rate = await self._calculate_issue_resolution_rate(
            organization_id,
            time_range
        )
        
        return {
            "overall_quality_score": overall_quality_score,
            "quality_trend": quality_trend,
            "technical_debt_hours": total_technical_debt_hours,
            "debt_per_file": debt_per_file,
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "issue_resolution_rate": issue_resolution_rate,
            "code_coverage_percentage": code_coverage_percentage,
            "maintainability_index": maintainability_index
        }
    
    async def calculate_velocity_kpis(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate velocity KPIs."""
        # Get velocity metrics from all projects
        projects = await self.project_repo.get_organization_projects(organization_id)
        
        if not projects:
            return self._empty_velocity_kpis()
        
        # Aggregate velocity metrics
        total_story_points = 0
        total_cycles = 0
        total_cycle_time = 0.0
        total_throughput = 0.0
        
        # DORA metrics aggregation
        deployment_frequencies = []
        lead_times = []
        failure_rates = []
        recovery_times = []
        
        for project in projects:
            # Get sprint/iteration metrics
            sprint_metrics = await self._get_sprint_metrics(project["id"], time_range)
            
            if sprint_metrics:
                total_story_points += sprint_metrics.get("completed_points", 0)
                total_cycles += sprint_metrics.get("sprint_count", 1)
                total_cycle_time += sprint_metrics.get("average_cycle_time", 0)
                total_throughput += sprint_metrics.get("weekly_throughput", 0)
            
            # Get DORA metrics
            dora_metrics = await self._get_project_dora_metrics(project["id"], time_range)
            
            if dora_metrics:
                deployment_frequencies.append(dora_metrics.get("deployment_frequency", 0))
                lead_times.append(dora_metrics.get("lead_time_hours", 0))
                failure_rates.append(dora_metrics.get("failure_rate", 0))
                recovery_times.append(dora_metrics.get("recovery_time_hours", 0))
        
        # Calculate averages
        story_points_per_sprint = total_story_points / total_cycles if total_cycles > 0 else 0
        cycle_time_days = total_cycle_time / len(projects) if projects else 0
        throughput_stories_per_week = total_throughput / len(projects) if projects else 0
        
        # DORA averages
        deployment_frequency = mean(deployment_frequencies) if deployment_frequencies else 0
        lead_time_hours = mean(lead_times) if lead_times else 0
        change_failure_rate = mean(failure_rates) if failure_rates else 0
        recovery_time_hours = mean(recovery_times) if recovery_times else 0
        
        return {
            "story_points_per_sprint": story_points_per_sprint,
            "cycle_time_days": cycle_time_days,
            "throughput_stories_per_week": throughput_stories_per_week,
            "deployment_frequency": deployment_frequency,
            "lead_time_hours": lead_time_hours,
            "change_failure_rate": change_failure_rate,
            "recovery_time_hours": recovery_time_hours
        }
    
    async def calculate_security_kpis(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate security KPIs."""
        # Get security data for all projects
        projects = await self.project_repo.get_organization_projects(organization_id)
        
        if not projects:
            return self._empty_security_kpis()
        
        # Aggregate security metrics
        total_critical_vulnerabilities = 0
        total_vulnerabilities = 0
        resolution_times = []
        compliance_scores = []
        security_debt_hours = 0.0
        threat_scores = []
        
        for project in projects:
            # Get vulnerability data
            vulnerabilities = await self.security_repo.get_project_vulnerabilities(
                project["id"],
                time_range
            )
            
            critical_vulns = len([v for v in vulnerabilities if v.get("severity") == "critical"])
            total_critical_vulnerabilities += critical_vulns
            total_vulnerabilities += len(vulnerabilities)
            
            # Get resolution times
            resolved_vulns = [v for v in vulnerabilities if v.get("resolved_at")]
            for vuln in resolved_vulns:
                resolution_time = (vuln["resolved_at"] - vuln["created_at"]).days
                resolution_times.append(resolution_time)
            
            # Get compliance data
            compliance_data = await self.security_repo.get_project_compliance(
                project["id"]
            )
            if compliance_data:
                compliance_scores.append(compliance_data.get("overall_score", 0))
            
            # Security debt
            security_debt = await self._calculate_security_debt(vulnerabilities)
            security_debt_hours += security_debt
            
            # Threat exposure
            threat_score = await self._calculate_threat_exposure(project["id"])
            threat_scores.append(threat_score)
        
        # Calculate overall security score
        security_score = self._calculate_security_score(
            total_critical_vulnerabilities,
            total_vulnerabilities,
            compliance_scores
        )
        
        # Calculate averages
        vulnerability_resolution_time_days = mean(resolution_times) if resolution_times else 0
        compliance_percentage = mean(compliance_scores) if compliance_scores else 0
        threat_exposure_score = mean(threat_scores) if threat_scores else 0
        
        return {
            "security_score": security_score,
            "critical_vulnerabilities": total_critical_vulnerabilities,
            "vulnerability_resolution_time_days": vulnerability_resolution_time_days,
            "compliance_percentage": compliance_percentage,
            "security_debt_hours": security_debt_hours,
            "threat_exposure_score": threat_exposure_score
        }
    
    async def calculate_cost_kpis(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate cost KPIs."""
        # Get quality and velocity KPIs first (for calculations)
        quality_kpis = await self.calculate_quality_kpis(organization_id, time_range)
        velocity_kpis = await self.calculate_velocity_kpis(organization_id, time_range)
        
        # Technical debt cost
        technical_debt_hours = quality_kpis["technical_debt_hours"]
        technical_debt_cost = Decimal(str(technical_debt_hours * self.default_hourly_rate))
        
        # Monthly maintenance cost estimate
        # Based on technical debt interest (2.5% monthly)
        monthly_maintenance_cost = technical_debt_cost * Decimal("0.025")
        
        # Incident cost calculation
        incident_rate = velocity_kpis["change_failure_rate"]
        deployments_per_month = velocity_kpis["deployment_frequency"] * 30
        incidents_per_month = deployments_per_month * incident_rate
        
        # Average incident cost (downtime + resolution)
        avg_incident_cost = Decimal("5000")  # $5k per incident
        incident_cost = Decimal(str(incidents_per_month)) * avg_incident_cost
        
        # Efficiency ratio (output value / input cost)
        # Simplified: story points delivered vs team cost
        team_size = await self._get_team_size(organization_id)
        monthly_team_cost = Decimal(str(team_size * 160 * self.default_hourly_rate))  # 160 hours/month
        
        story_points_per_month = velocity_kpis["story_points_per_sprint"] * 2  # Bi-weekly sprints
        cost_per_story_point = monthly_team_cost / Decimal(str(max(story_points_per_month, 1)))
        
        # Efficiency ratio: lower cost per story point = higher efficiency
        baseline_cost_per_point = Decimal("2000")  # Industry baseline
        efficiency_ratio = float(baseline_cost_per_point / cost_per_story_point) if cost_per_story_point > 0 else 0
        
        # Quality investment ROI
        quality_investment = await self._get_quality_investment(organization_id, time_range)
        quality_improvement = quality_kpis["overall_quality_score"] - 70  # Baseline
        quality_investment_roi = (quality_improvement / 10) * 100 if quality_investment > 0 else 0  # 10% ROI per quality point
        
        return {
            "technical_debt_cost": technical_debt_cost,
            "monthly_maintenance_cost": monthly_maintenance_cost,
            "incident_cost": incident_cost,
            "efficiency_ratio": efficiency_ratio,
            "cost_per_story_point": cost_per_story_point,
            "quality_investment_roi": quality_investment_roi
        }
    
    async def calculate_team_kpis(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate team KPIs."""
        # Get team data
        team_data = await self.team_repo.get_organization_team_data(
            organization_id,
            time_range
        )
        
        if not team_data:
            return self._empty_team_kpis()
        
        # Team satisfaction (from surveys)
        satisfaction_scores = team_data.get("satisfaction_scores", [])
        team_satisfaction_score = mean(satisfaction_scores) if satisfaction_scores else 3.5
        
        # Knowledge sharing metrics
        code_reviews = team_data.get("code_reviews", [])
        total_prs = team_data.get("total_pull_requests", 1)
        
        # Calculate review participation
        reviewers = set()
        for review in code_reviews:
            reviewers.update(review.get("reviewers", []))
        
        team_size = await self._get_team_size(organization_id)
        code_review_participation_rate = len(reviewers) / team_size if team_size > 0 else 0
        
        # Knowledge sharing index (based on reviews, documentation, etc.)
        knowledge_sharing_index = self._calculate_knowledge_sharing_index(
            team_data,
            team_size
        )
        
        # Pair programming percentage
        pair_programming_sessions = team_data.get("pair_programming_sessions", 0)
        total_development_sessions = team_data.get("total_dev_sessions", 100)
        pair_programming_percentage = (
            pair_programming_sessions / total_development_sessions * 100
            if total_development_sessions > 0 else 0
        )
        
        # Continuous learning
        training_hours = team_data.get("training_hours", {})
        continuous_learning_hours = sum(training_hours.values()) / team_size if team_size > 0 else 0
        
        # Burnout risk score (based on overtime, velocity variance, etc.)
        burnout_risk_score = await self._calculate_burnout_risk(
            team_data,
            organization_id
        )
        
        return {
            "team_satisfaction_score": team_satisfaction_score,
            "knowledge_sharing_index": knowledge_sharing_index,
            "code_review_participation_rate": code_review_participation_rate,
            "pair_programming_percentage": pair_programming_percentage,
            "continuous_learning_hours": continuous_learning_hours,
            "burnout_risk_score": burnout_risk_score
        }
    
    def _calculate_overall_score(
        self,
        quality: QualityKPIs,
        velocity: VelocityKPIs,
        security: SecurityKPIs,
        cost: CostKPIs,
        team: TeamKPIs
    ) -> float:
        """Calculate overall organization score."""
        # Weighted scoring
        weights = {
            "quality": 0.25,
            "velocity": 0.20,
            "security": 0.25,
            "cost": 0.15,
            "team": 0.15
        }
        
        # Normalize scores to 0-100 scale
        quality_score = quality.overall_quality_score
        
        # Velocity score based on DORA metrics
        velocity_score = self._calculate_velocity_score(velocity)
        
        security_score = security.security_score
        
        # Cost score (inverse of inefficiency)
        cost_score = min(100, cost.efficiency_ratio * 100)
        
        # Team score
        team_score = (
            team.team_satisfaction_score * 20 +  # Max 100 (5.0 * 20)
            team.knowledge_sharing_index +
            team.code_review_participation_rate * 100 +
            (100 - team.burnout_risk_score)
        ) / 4
        
        # Calculate weighted average
        overall_score = (
            quality_score * weights["quality"] +
            velocity_score * weights["velocity"] +
            security_score * weights["security"] +
            cost_score * weights["cost"] +
            team_score * weights["team"]
        )
        
        return overall_score
    
    def _calculate_velocity_score(self, velocity: VelocityKPIs) -> float:
        """Calculate velocity score from DORA metrics."""
        # Score based on DORA performance levels
        scores = {
            "deployment": self._score_deployment_frequency(velocity.deployment_frequency),
            "lead_time": self._score_lead_time(velocity.lead_time_hours),
            "failure_rate": self._score_failure_rate(velocity.change_failure_rate),
            "recovery": self._score_recovery_time(velocity.recovery_time_hours)
        }
        
        # Average of DORA scores (0-100 scale)
        return sum(scores.values()) / len(scores)
    
    def _score_deployment_frequency(self, freq: float) -> float:
        """Score deployment frequency (0-100)."""
        if freq >= 1:  # Daily or more
            return 100
        elif freq >= 1/7:  # Weekly
            return 75
        elif freq >= 1/30:  # Monthly
            return 50
        else:
            return 25
    
    def _score_lead_time(self, hours: float) -> float:
        """Score lead time (0-100)."""
        if hours <= 1:
            return 100
        elif hours <= 24:
            return 75
        elif hours <= 168:  # 1 week
            return 50
        else:
            return 25
    
    def _score_failure_rate(self, rate: float) -> float:
        """Score failure rate (0-100)."""
        if rate <= 0.15:
            return 100
        elif rate <= 0.30:
            return 75
        elif rate <= 0.45:
            return 50
        else:
            return 25
    
    def _score_recovery_time(self, hours: float) -> float:
        """Score recovery time (0-100)."""
        if hours <= 1:
            return 100
        elif hours <= 24:
            return 75
        elif hours <= 168:  # 1 week
            return 50
        else:
            return 25
    
    def _calculate_security_score(
        self,
        critical_vulns: int,
        total_vulns: int,
        compliance_scores: List[float]
    ) -> float:
        """Calculate overall security score."""
        # Vulnerability score (40% weight)
        vuln_penalty = critical_vulns * 10 + (total_vulns - critical_vulns) * 2
        vuln_score = max(0, 100 - vuln_penalty)
        
        # Compliance score (60% weight)
        compliance_score = mean(compliance_scores) if compliance_scores else 50
        
        return vuln_score * 0.4 + compliance_score * 0.6
    
    def _calculate_knowledge_sharing_index(
        self,
        team_data: Dict[str, Any],
        team_size: int
    ) -> float:
        """Calculate knowledge sharing index (0-100)."""
        factors = []
        
        # Code review coverage
        reviewed_prs = team_data.get("reviewed_prs", 0)
        total_prs = team_data.get("total_pull_requests", 1)
        review_coverage = reviewed_prs / total_prs if total_prs > 0 else 0
        factors.append(review_coverage * 100)
        
        # Documentation updates
        doc_updates = team_data.get("documentation_updates", 0)
        expected_updates = team_size * 2  # 2 per person per month
        doc_factor = min(100, (doc_updates / expected_updates * 100) if expected_updates > 0 else 0)
        factors.append(doc_factor)
        
        # Knowledge sharing sessions
        sessions = team_data.get("knowledge_sharing_sessions", 0)
        expected_sessions = 4  # Weekly
        session_factor = min(100, (sessions / expected_sessions * 100))
        factors.append(session_factor)
        
        return mean(factors) if factors else 0
    
    async def _calculate_burnout_risk(
        self,
        team_data: Dict[str, Any],
        organization_id: str
    ) -> float:
        """Calculate burnout risk score (0-100, higher is worse)."""
        risk_factors = []
        
        # Overtime factor
        overtime_hours = team_data.get("overtime_hours", {})
        if overtime_hours:
            avg_overtime = mean(overtime_hours.values())
            # More than 10 hours/week overtime is high risk
            overtime_risk = min(100, (avg_overtime / 10) * 100)
            risk_factors.append(overtime_risk)
        
        # Velocity variance (high variance = unsustainable pace)
        velocity_data = team_data.get("sprint_velocities", [])
        if len(velocity_data) > 2:
            velocity_variance = self._calculate_variance(velocity_data)
            avg_velocity = mean(velocity_data)
            cv = velocity_variance / avg_velocity if avg_velocity > 0 else 0
            # CV > 0.3 is high risk
            variance_risk = min(100, (cv / 0.3) * 100)
            risk_factors.append(variance_risk)
        
        # On-call burden
        on_call_incidents = team_data.get("on_call_incidents", 0)
        # More than 10 incidents/month is high risk
        on_call_risk = min(100, (on_call_incidents / 10) * 100)
        risk_factors.append(on_call_risk)
        
        # Work-life balance indicators
        after_hours_commits = team_data.get("after_hours_commits", 0)
        total_commits = team_data.get("total_commits", 1)
        after_hours_percentage = after_hours_commits / total_commits if total_commits > 0 else 0
        # More than 20% after hours is high risk
        balance_risk = min(100, (after_hours_percentage / 0.2) * 100)
        risk_factors.append(balance_risk)
        
        return mean(risk_factors) if risk_factors else 0
    
    def _calculate_variance(self, data: List[float]) -> float:
        """Calculate variance of a dataset."""
        if len(data) < 2:
            return 0
        
        avg = mean(data)
        return sum((x - avg) ** 2 for x in data) / len(data)
    
    # Helper methods for data retrieval
    async def _get_sprint_metrics(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> Optional[Dict[str, Any]]:
        """Get sprint metrics for a project."""
        # Stub implementation
        return {
            "completed_points": 45,
            "sprint_count": 2,
            "average_cycle_time": 5.5,
            "weekly_throughput": 8
        }
    
    async def _get_project_dora_metrics(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> Optional[Dict[str, Any]]:
        """Get DORA metrics for a project."""
        # Stub implementation
        return {
            "deployment_frequency": 0.5,  # Every 2 days
            "lead_time_hours": 48,
            "failure_rate": 0.15,
            "recovery_time_hours": 2
        }
    
    async def _calculate_quality_trend(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> TrendDirection:
        """Calculate quality trend for organization."""
        # Simplified - would compare with previous period
        return TrendDirection.IMPROVING
    
    async def _calculate_issue_resolution_rate(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> float:
        """Calculate issue resolution rate."""
        # Stub implementation
        return 0.85  # 85% resolution rate
    
    async def _calculate_security_debt(
        self,
        vulnerabilities: List[Dict[str, Any]]
    ) -> float:
        """Calculate security debt in hours."""
        debt_hours = 0.0
        
        for vuln in vulnerabilities:
            if vuln.get("severity") == "critical":
                debt_hours += 8  # 8 hours per critical
            elif vuln.get("severity") == "high":
                debt_hours += 4  # 4 hours per high
            elif vuln.get("severity") == "medium":
                debt_hours += 2  # 2 hours per medium
            else:
                debt_hours += 1  # 1 hour per low
        
        return debt_hours
    
    async def _calculate_threat_exposure(self, project_id: str) -> float:
        """Calculate threat exposure score (0-100)."""
        # Stub implementation
        return 25.0  # Low threat exposure
    
    async def _get_team_size(self, organization_id: str) -> int:
        """Get organization team size."""
        # Stub implementation
        return 10
    
    async def _get_quality_investment(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> float:
        """Get quality improvement investment."""
        # Stub implementation
        return 50000.0  # $50k investment
    
    # Empty KPI methods
    def _empty_quality_kpis(self) -> Dict[str, Any]:
        """Return empty quality KPIs."""
        return {
            "overall_quality_score": 0.0,
            "quality_trend": TrendDirection.STABLE,
            "technical_debt_hours": 0.0,
            "debt_per_file": 0.0,
            "total_issues": 0,
            "critical_issues": 0,
            "issue_resolution_rate": 0.0,
            "code_coverage_percentage": 0.0,
            "maintainability_index": 0.0
        }
    
    def _empty_velocity_kpis(self) -> Dict[str, Any]:
        """Return empty velocity KPIs."""
        return {
            "story_points_per_sprint": 0.0,
            "cycle_time_days": 0.0,
            "throughput_stories_per_week": 0.0,
            "deployment_frequency": 0.0,
            "lead_time_hours": 0.0,
            "change_failure_rate": 0.0,
            "recovery_time_hours": 0.0
        }
    
    def _empty_security_kpis(self) -> Dict[str, Any]:
        """Return empty security KPIs."""
        return {
            "security_score": 0.0,
            "critical_vulnerabilities": 0,
            "vulnerability_resolution_time_days": 0.0,
            "compliance_percentage": 0.0,
            "security_debt_hours": 0.0,
            "threat_exposure_score": 0.0
        }
    
    def _empty_cost_kpis(self) -> Dict[str, Any]:
        """Return empty cost KPIs."""
        return {
            "technical_debt_cost": Decimal("0"),
            "monthly_maintenance_cost": Decimal("0"),
            "incident_cost": Decimal("0"),
            "efficiency_ratio": 0.0,
            "cost_per_story_point": Decimal("0"),
            "quality_investment_roi": 0.0
        }
    
    def _empty_team_kpis(self) -> Dict[str, Any]:
        """Return empty team KPIs."""
        return {
            "team_satisfaction_score": 0.0,
            "knowledge_sharing_index": 0.0,
            "code_review_participation_rate": 0.0,
            "pair_programming_percentage": 0.0,
            "continuous_learning_hours": 0.0,
            "burnout_risk_score": 0.0
        }
