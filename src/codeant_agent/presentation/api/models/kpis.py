"""
API models for organization KPIs.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal

from codeant_agent.domain.dora import OrganizationKPIs, TimeRange


class QualityKPIsResponse(BaseModel):
    """Response model for quality KPIs."""
    overall_quality_score: float
    quality_trend: str
    technical_debt_hours: float
    debt_per_file: float
    total_issues: int
    critical_issues: int
    issue_resolution_rate: float
    code_coverage_percentage: float
    maintainability_index: float


class VelocityKPIsResponse(BaseModel):
    """Response model for velocity KPIs."""
    story_points_per_sprint: float
    cycle_time_days: float
    throughput_stories_per_week: float
    deployment_frequency: float
    lead_time_hours: float
    change_failure_rate: float
    recovery_time_hours: float


class SecurityKPIsResponse(BaseModel):
    """Response model for security KPIs."""
    security_score: float
    critical_vulnerabilities: int
    vulnerability_resolution_time_days: float
    compliance_percentage: float
    security_debt_hours: float
    threat_exposure_score: float


class CostKPIsResponse(BaseModel):
    """Response model for cost KPIs."""
    technical_debt_cost: float
    monthly_maintenance_cost: float
    incident_cost: float
    efficiency_ratio: float
    cost_per_story_point: float
    quality_investment_roi: float


class TeamKPIsResponse(BaseModel):
    """Response model for team KPIs."""
    team_satisfaction_score: float
    knowledge_sharing_index: float
    code_review_participation_rate: float
    pair_programming_percentage: float
    continuous_learning_hours: float
    burnout_risk_score: float


class OrganizationKPIsResponse(BaseModel):
    """Response model for complete organization KPIs."""
    organization_id: Optional[str] = None
    time_range: Optional[Dict[str, datetime]] = None
    velocity: VelocityKPIsResponse
    quality: QualityKPIsResponse
    security: SecurityKPIsResponse
    cost: CostKPIsResponse
    team: TeamKPIsResponse
    overall_score: float
    calculated_at: Optional[datetime] = None
    
    @classmethod
    def from_domain(cls, kpis: OrganizationKPIs) -> "OrganizationKPIsResponse":
        """Convert from domain model."""
        return cls(
            velocity=VelocityKPIsResponse(
                story_points_per_sprint=kpis.velocity.story_points_per_sprint,
                cycle_time_days=kpis.velocity.cycle_time_days,
                throughput_stories_per_week=kpis.velocity.throughput_stories_per_week,
                deployment_frequency=kpis.velocity.deployment_frequency,
                lead_time_hours=kpis.velocity.lead_time_hours,
                change_failure_rate=kpis.velocity.change_failure_rate,
                recovery_time_hours=kpis.velocity.recovery_time_hours
            ),
            quality=QualityKPIsResponse(
                overall_quality_score=kpis.quality.overall_quality_score,
                quality_trend=kpis.quality.quality_trend.value,
                technical_debt_hours=kpis.quality.technical_debt_hours,
                debt_per_file=kpis.quality.debt_per_file,
                total_issues=kpis.quality.total_issues,
                critical_issues=kpis.quality.critical_issues,
                issue_resolution_rate=kpis.quality.issue_resolution_rate,
                code_coverage_percentage=kpis.quality.code_coverage_percentage,
                maintainability_index=kpis.quality.maintainability_index
            ),
            security=SecurityKPIsResponse(
                security_score=kpis.security.security_score,
                critical_vulnerabilities=kpis.security.critical_vulnerabilities,
                vulnerability_resolution_time_days=kpis.security.vulnerability_resolution_time_days,
                compliance_percentage=kpis.security.compliance_percentage,
                security_debt_hours=kpis.security.security_debt_hours,
                threat_exposure_score=kpis.security.threat_exposure_score
            ),
            cost=CostKPIsResponse(
                technical_debt_cost=float(kpis.cost.technical_debt_cost),
                monthly_maintenance_cost=float(kpis.cost.monthly_maintenance_cost),
                incident_cost=float(kpis.cost.incident_cost),
                efficiency_ratio=kpis.cost.efficiency_ratio,
                cost_per_story_point=float(kpis.cost.cost_per_story_point),
                quality_investment_roi=kpis.cost.quality_investment_roi
            ),
            team=TeamKPIsResponse(
                team_satisfaction_score=kpis.team.team_satisfaction_score,
                knowledge_sharing_index=kpis.team.knowledge_sharing_index,
                code_review_participation_rate=kpis.team.code_review_participation_rate,
                pair_programming_percentage=kpis.team.pair_programming_percentage,
                continuous_learning_hours=kpis.team.continuous_learning_hours,
                burnout_risk_score=kpis.team.burnout_risk_score
            ),
            overall_score=kpis.overall_score,
            calculated_at=datetime.now()
        )


class KPICategoryResponse(BaseModel):
    """Response model for a specific KPI category."""
    category: str
    metrics: Dict[str, Any]
    time_range: TimeRange
    calculated_at: datetime


class KPITrendsResponse(BaseModel):
    """Response model for KPI trends."""
    organization_id: str
    metric: str
    period: str
    data_points: List[Dict[str, Any]]


class KPIComparisonResponse(BaseModel):
    """Response model for KPI comparison."""
    time_range: TimeRange
    organizations: List[Dict[str, Any]]
    comparisons: Dict[str, List[Dict[str, Any]]]
    rankings: Dict[str, List[Dict[str, Any]]]
    insights: List[str]


class KPIExportRequest(BaseModel):
    """Request model for KPI export."""
    organization_id: str
    start_date: datetime
    end_date: datetime
    categories: List[str] = Field(
        default=["quality", "velocity", "security", "cost", "team"]
    )
    format: str = Field(
        default="csv",
        description="Export format: csv, excel, json"
    )
    include_trends: bool = Field(default=False)
