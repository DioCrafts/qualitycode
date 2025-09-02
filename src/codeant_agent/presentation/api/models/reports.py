"""
API models for executive reports.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, time
from decimal import Decimal


class ExecutiveReportRequest(BaseModel):
    """Request model for executive report generation."""
    organization_id: str = Field(..., description="Organization ID")
    start_date: datetime = Field(..., description="Start date for report period")
    end_date: datetime = Field(..., description="End date for report period")
    report_type: str = Field(
        default="monthly",
        description="Type of report: monthly, quarterly, annual, custom"
    )
    language: Optional[str] = Field(
        default="en",
        description="Report language: en, es"
    )
    include_technical_appendix: bool = Field(
        default=False,
        description="Include detailed technical metrics"
    )
    async_generation: bool = Field(
        default=False,
        description="Generate report asynchronously"
    )


class KeyMetricResponse(BaseModel):
    """Response model for key metric."""
    name: str
    value: str
    trend: str
    business_impact: str
    target_value: Optional[str] = None


class CriticalIssueResponse(BaseModel):
    """Response model for critical issue."""
    issue_type: str
    description: str
    business_impact: str
    resolution_timeline: str
    required_investment: float


class StrategicRecommendationResponse(BaseModel):
    """Response model for strategic recommendation."""
    title: str
    description: str
    business_justification: str
    estimated_investment: float
    expected_roi: float
    timeline_weeks: int
    risk_level: str
    success_metrics: List[str]


class ExecutiveSummaryResponse(BaseModel):
    """Response model for executive summary."""
    summary_text: str
    key_metrics: List[KeyMetricResponse]
    critical_issues: List[CriticalIssueResponse]
    strategic_recommendations: List[StrategicRecommendationResponse]
    next_steps: List[Dict[str, Any]]


class ExecutiveReportResponse(BaseModel):
    """Response model for executive report."""
    report_id: str
    status: str
    message: Optional[str] = None
    organization_id: Optional[str] = None
    report_type: Optional[str] = None
    language: Optional[str] = None
    time_range: Optional[Dict[str, datetime]] = None
    executive_summary: Optional[ExecutiveSummaryResponse] = None
    business_metrics: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[StrategicRecommendationResponse]] = None
    generation_time_ms: Optional[int] = None
    generated_at: Optional[datetime] = None
    download_url: Optional[str] = None
    
    @classmethod
    def from_domain(cls, report: Any) -> "ExecutiveReportResponse":
        """Convert from domain model."""
        return cls(
            report_id=report.id,
            status="completed",
            organization_id=report.organization_id,
            report_type=report.report_type,
            language=report.language.value,
            time_range={
                "start_date": report.time_range.start_date,
                "end_date": report.time_range.end_date
            },
            executive_summary=ExecutiveSummaryResponse(
                summary_text=report.executive_summary.summary_text,
                key_metrics=[
                    KeyMetricResponse(
                        name=m.name,
                        value=m.value,
                        trend=m.trend.value,
                        business_impact=m.business_impact.value,
                        target_value=m.target_value
                    ) for m in report.executive_summary.key_metrics
                ],
                critical_issues=[
                    CriticalIssueResponse(
                        issue_type=i.issue_type,
                        description=i.description,
                        business_impact=i.business_impact,
                        resolution_timeline=i.resolution_timeline,
                        required_investment=float(i.required_investment)
                    ) for i in report.executive_summary.critical_issues
                ],
                strategic_recommendations=[
                    StrategicRecommendationResponse(
                        title=r.title,
                        description=r.description,
                        business_justification=r.business_justification,
                        estimated_investment=float(r.estimated_investment.cost_estimate),
                        expected_roi=r.expected_roi,
                        timeline_weeks=r.timeline.estimated_duration_weeks,
                        risk_level=r.risk_level.value,
                        success_metrics=r.success_metrics
                    ) for r in report.executive_summary.strategic_recommendations
                ],
                next_steps=[{
                    "step_number": s.step_number,
                    "description": s.description,
                    "responsible_party": s.responsible_party,
                    "timeline": s.timeline
                } for s in report.executive_summary.next_steps]
            ),
            generation_time_ms=report.generation_time_ms,
            generated_at=report.generated_at,
            download_url=f"/api/v1/reports/executive/download/{report.id}"
        )


class ScheduledReportRequest(BaseModel):
    """Request model for scheduled report."""
    organization_id: str
    frequency: str = Field(
        ...,
        description="Frequency: daily, weekly, monthly, quarterly"
    )
    day_of_week: Optional[int] = Field(
        None,
        ge=0,
        le=6,
        description="Day of week for weekly reports (0=Monday)"
    )
    day_of_month: Optional[int] = Field(
        None,
        ge=1,
        le=31,
        description="Day of month for monthly reports"
    )
    time: time = Field(
        default=time(9, 0),
        description="Time to generate report"
    )
    timezone: str = Field(
        default="UTC",
        description="Timezone for scheduling"
    )
    report_type: str = Field(default="monthly")
    language: str = Field(default="en")
    recipients: List[str] = Field(
        ...,
        description="Email addresses to send report to"
    )
    delivery_methods: List[str] = Field(
        default=["email"],
        description="Delivery methods: email, slack, teams, webhook"
    )
    enabled: bool = Field(default=True)


class ScheduledReportResponse(BaseModel):
    """Response model for scheduled report."""
    schedule_id: str
    status: str
    message: str


class ReportHistoryResponse(BaseModel):
    """Response model for report history."""
    reports: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int
