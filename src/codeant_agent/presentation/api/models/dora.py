"""
API models for DORA metrics.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal

from codeant_agent.domain.dora import (
    DORAMetrics,
    TrendDirection,
    DORAPerformanceCategory,
    BusinessImpact
)


class DORAMetricsRequest(BaseModel):
    """Request model for DORA metrics calculation."""
    project_id: str = Field(..., description="Project ID to calculate metrics for")
    start_date: datetime = Field(..., description="Start date for metrics calculation")
    end_date: datetime = Field(..., description="End date for metrics calculation")
    include_insights: bool = Field(default=True, description="Include AI-generated insights")
    include_benchmarks: bool = Field(default=True, description="Include industry benchmarks")


class DeploymentFrequencyResponse(BaseModel):
    """Response model for deployment frequency metric."""
    total_deployments: int
    deployments_per_day: float
    performance_category: str
    trend: str
    daily_stats: Dict[str, int]
    recommendations: List[str]


class LeadTimeMetricsResponse(BaseModel):
    """Response model for lead time metrics."""
    total_changes: int
    median_hours: float
    mean_hours: float
    p90_hours: float
    performance_category: str
    trend: str
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]


class ChangeFailureRateResponse(BaseModel):
    """Response model for change failure rate."""
    total_deployments: int
    failed_deployments: int
    failure_rate_percentage: float
    performance_category: str
    trend: str
    failure_patterns: List[Dict[str, Any]]
    recommendations: List[str]


class TimeToRecoveryResponse(BaseModel):
    """Response model for time to recovery."""
    total_incidents: int
    median_hours: float
    mean_hours: float
    p90_hours: float
    performance_category: str
    trend: str
    recovery_patterns: List[Dict[str, Any]]
    recommendations: List[str]


class DORAPerformanceRatingResponse(BaseModel):
    """Response model for overall DORA performance rating."""
    overall_category: str
    overall_score: float
    deployment_score: float
    lead_time_score: float
    failure_rate_score: float
    recovery_time_score: float
    strengths: List[str]
    improvement_areas: List[str]


class DORAInsightResponse(BaseModel):
    """Response model for DORA insight."""
    insight_type: str
    title: str
    description: str
    impact: str
    recommendations: List[str]


class DORAMetricsResponse(BaseModel):
    """Response model for complete DORA metrics."""
    project_id: str
    time_range: Dict[str, datetime]
    deployment_frequency: DeploymentFrequencyResponse
    lead_time_for_changes: LeadTimeMetricsResponse
    change_failure_rate: ChangeFailureRateResponse
    time_to_recovery: TimeToRecoveryResponse
    performance_rating: DORAPerformanceRatingResponse
    insights: List[DORAInsightResponse]
    calculation_time_ms: int
    calculated_at: datetime
    
    @classmethod
    def from_domain(cls, metrics: DORAMetrics) -> "DORAMetricsResponse":
        """Convert from domain model."""
        return cls(
            project_id=metrics.project_id,
            time_range={
                "start_date": metrics.time_range.start_date,
                "end_date": metrics.time_range.end_date
            },
            deployment_frequency=DeploymentFrequencyResponse(
                total_deployments=metrics.deployment_frequency.total_deployments,
                deployments_per_day=metrics.deployment_frequency.deployments_per_day,
                performance_category=metrics.deployment_frequency.performance_category.value,
                trend=metrics.deployment_frequency.trend.value,
                daily_stats=metrics.deployment_frequency.daily_deployments,
                recommendations=metrics.deployment_frequency.recommendations
            ),
            lead_time_for_changes=LeadTimeMetricsResponse(
                total_changes=metrics.lead_time_for_changes.total_changes,
                median_hours=metrics.lead_time_for_changes.stats.median,
                mean_hours=metrics.lead_time_for_changes.stats.mean,
                p90_hours=metrics.lead_time_for_changes.stats.p90,
                performance_category=metrics.lead_time_for_changes.performance_category.value,
                trend=metrics.lead_time_for_changes.trend.value,
                bottlenecks=[{
                    "stage": b.stage,
                    "average_time_hours": b.average_time_hours,
                    "percentage_of_total": b.percentage_of_total
                } for b in metrics.lead_time_for_changes.bottlenecks],
                recommendations=metrics.lead_time_for_changes.recommendations
            ),
            change_failure_rate=ChangeFailureRateResponse(
                total_deployments=metrics.change_failure_rate.total_deployments,
                failed_deployments=metrics.change_failure_rate.failed_deployments,
                failure_rate_percentage=metrics.change_failure_rate.failure_rate_percentage,
                performance_category=metrics.change_failure_rate.performance_category.value,
                trend=metrics.change_failure_rate.trend.value,
                failure_patterns=[{
                    "pattern_type": p.pattern_type,
                    "frequency": p.frequency,
                    "description": p.description
                } for p in metrics.change_failure_rate.failure_patterns],
                recommendations=metrics.change_failure_rate.recommendations
            ),
            time_to_recovery=TimeToRecoveryResponse(
                total_incidents=len(metrics.time_to_recovery.incidents),
                median_hours=metrics.time_to_recovery.stats.median_hours,
                mean_hours=metrics.time_to_recovery.stats.mean_hours,
                p90_hours=metrics.time_to_recovery.stats.p90_hours,
                performance_category=metrics.time_to_recovery.performance_category.value,
                trend=metrics.time_to_recovery.trend.value,
                recovery_patterns=[{
                    "pattern_type": p.pattern_type,
                    "average_recovery_hours": p.average_recovery_hours,
                    "frequency": p.frequency
                } for p in metrics.time_to_recovery.recovery_patterns],
                recommendations=metrics.time_to_recovery.recommendations
            ),
            performance_rating=DORAPerformanceRatingResponse(
                overall_category=metrics.performance_rating.overall_category.value,
                overall_score=metrics.performance_rating.overall_score,
                deployment_score=metrics.performance_rating.deployment_score,
                lead_time_score=metrics.performance_rating.lead_time_score,
                failure_rate_score=metrics.performance_rating.failure_rate_score,
                recovery_time_score=metrics.performance_rating.recovery_time_score,
                strengths=metrics.performance_rating.strengths,
                improvement_areas=metrics.performance_rating.improvement_areas
            ),
            insights=[
                DORAInsightResponse(
                    insight_type=i.insight_type,
                    title=i.title,
                    description=i.description,
                    impact=i.impact.value,
                    recommendations=i.recommendations
                ) for i in metrics.insights
            ],
            calculation_time_ms=metrics.calculation_time_ms,
            calculated_at=metrics.calculated_at
        )


class HistoricalDORAResponse(BaseModel):
    """Response model for historical DORA metrics."""
    project_id: str
    period: str
    data_points: List[Dict[str, Any]]


class DORAComparisonResponse(BaseModel):
    """Response model for DORA metrics comparison."""
    time_range: Dict[str, datetime]
    projects: List[Dict[str, Any]]
    rankings: Dict[str, List[Dict[str, Any]]]
    insights: List[str]
