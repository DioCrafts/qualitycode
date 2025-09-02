"""
DORA metrics and executive reporting entities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from decimal import Decimal


class TrendDirection(Enum):
    """Trend direction for metrics."""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"


class DORAPerformanceCategory(Enum):
    """DORA performance categories based on State of DevOps Report."""
    ELITE = "elite"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DeploymentPerformanceCategory(Enum):
    """Deployment frequency performance categories."""
    ELITE = "elite"      # Multiple deployments per day
    HIGH = "high"        # Between once per day and once per week
    MEDIUM = "medium"    # Between once per week and once per month
    LOW = "low"          # Less than once per month


class LeadTimePerformanceCategory(Enum):
    """Lead time performance categories."""
    ELITE = "elite"      # Less than 1 hour
    HIGH = "high"        # Between 1 hour and 1 day
    MEDIUM = "medium"    # Between 1 day and 1 week
    LOW = "low"          # More than 1 week


class FailureRatePerformanceCategory(Enum):
    """Change failure rate performance categories."""
    ELITE = "elite"      # 0-15% failure rate
    HIGH = "high"        # 16-30% failure rate
    MEDIUM = "medium"    # 31-45% failure rate
    LOW = "low"          # More than 45% failure rate


class RecoveryTimePerformanceCategory(Enum):
    """Time to recovery performance categories."""
    ELITE = "elite"      # Less than 1 hour
    HIGH = "high"        # Between 1 hour and 1 day
    MEDIUM = "medium"    # Between 1 day and 1 week
    LOW = "low"          # More than 1 week


class Language(Enum):
    """Supported languages for reports."""
    SPANISH = "es"
    ENGLISH = "en"


class ReportFormat(Enum):
    """Supported report export formats."""
    PDF = "pdf"
    POWERPOINT = "pptx"
    WORD = "docx"
    HTML = "html"
    JSON = "json"


class BusinessImpact(Enum):
    """Business impact levels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CRITICAL = "critical"


class RiskLevel(Enum):
    """Risk levels for various metrics."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class TimeRange:
    """Time range for metric calculations."""
    start_date: datetime
    end_date: datetime
    
    def duration_days(self) -> int:
        """Calculate duration in days."""
        return (self.end_date - self.start_date).days
    
    def duration_hours(self) -> float:
        """Calculate duration in hours."""
        return (self.end_date - self.start_date).total_seconds() / 3600


@dataclass
class DeploymentFrequencyStats:
    """Statistics for deployment frequency."""
    mean: float
    median: float
    std_dev: float
    min: float
    max: float


@dataclass
class DeploymentFrequency:
    """Deployment frequency metrics."""
    total_deployments: int
    deployments_per_day: float
    daily_deployments: Dict[str, int]  # Date -> count
    weekly_deployments: Dict[str, int]  # Week -> count
    monthly_deployments: Dict[str, int]  # Month -> count
    stats: DeploymentFrequencyStats
    performance_category: DeploymentPerformanceCategory
    trend: TrendDirection
    recommendations: List[str] = field(default_factory=list)


@dataclass
class LeadTimeStats:
    """Statistics for lead time metrics."""
    mean: float
    median: float
    p50: float
    p75: float
    p90: float
    p95: float
    std_dev: float
    min: float
    max: float


@dataclass
class LeadTimeEntry:
    """Individual lead time entry."""
    commit_id: str
    commit_timestamp: datetime
    deployment_timestamp: datetime
    lead_time_hours: float
    change_size: int
    change_complexity: float


@dataclass
class LeadTimeBottleneck:
    """Identified bottleneck in lead time."""
    stage: str
    average_time_hours: float
    percentage_of_total: float
    recommendations: List[str]


@dataclass
class LeadTimeMetrics:
    """Lead time for changes metrics."""
    total_changes: int
    lead_time_entries: List[LeadTimeEntry]
    stats: LeadTimeStats
    performance_category: LeadTimePerformanceCategory
    trend: TrendDirection
    bottlenecks: List[LeadTimeBottleneck] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FailurePattern:
    """Pattern identified in deployment failures."""
    pattern_type: str
    frequency: int
    description: str
    impact: str
    mitigation: str


@dataclass
class FailureRootCause:
    """Root cause analysis for failures."""
    cause_type: str
    description: str
    frequency: int
    impact_score: float
    remediation: str


@dataclass
class ChangeFailureRate:
    """Change failure rate metrics."""
    total_deployments: int
    failed_deployments: int
    failure_rate_percentage: float
    failure_patterns: List[FailurePattern]
    performance_category: FailureRatePerformanceCategory
    trend: TrendDirection
    root_causes: List[FailureRootCause] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RecoveryTimeStats:
    """Statistics for recovery time."""
    mean_hours: float
    median_hours: float
    p50_hours: float
    p75_hours: float
    p90_hours: float
    p95_hours: float
    std_dev_hours: float
    min_hours: float
    max_hours: float


@dataclass
class RecoveryIncident:
    """Individual recovery incident."""
    incident_id: str
    failure_timestamp: datetime
    recovery_timestamp: datetime
    recovery_time_hours: float
    severity: str
    root_cause: str
    resolution_method: str


@dataclass
class RecoveryPattern:
    """Pattern identified in recovery times."""
    pattern_type: str
    average_recovery_hours: float
    frequency: int
    recommendations: List[str]


@dataclass
class TimeToRecovery:
    """Time to recovery metrics."""
    incidents: List[RecoveryIncident]
    stats: RecoveryTimeStats
    performance_category: RecoveryTimePerformanceCategory
    trend: TrendDirection
    recovery_patterns: List[RecoveryPattern] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DORAPerformanceRating:
    """Overall DORA performance rating."""
    overall_category: DORAPerformanceCategory
    overall_score: float
    deployment_score: float
    lead_time_score: float
    failure_rate_score: float
    recovery_time_score: float
    strengths: List[str] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)


@dataclass
class DORAInsight:
    """Insight derived from DORA metrics."""
    insight_type: str
    title: str
    description: str
    impact: BusinessImpact
    recommendations: List[str]
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DORAMetricsTrends:
    """Trends for all DORA metrics."""
    deployment_frequency_trend: List[Dict[str, Any]]
    lead_time_trend: List[Dict[str, Any]]
    failure_rate_trend: List[Dict[str, Any]]
    recovery_time_trend: List[Dict[str, Any]]
    overall_trend: TrendDirection


@dataclass
class IndustryBenchmark:
    """Industry benchmark for a metric."""
    metric_name: str
    elite_threshold: float
    high_threshold: float
    medium_threshold: float
    low_threshold: float
    industry_average: float
    percentile_rank: float


@dataclass
class IndustryBenchmarks:
    """Industry benchmarks for all metrics."""
    deployment_frequency: IndustryBenchmark
    lead_time: IndustryBenchmark
    failure_rate: IndustryBenchmark
    recovery_time: IndustryBenchmark
    last_updated: datetime


@dataclass
class DORAMetrics:
    """Complete DORA metrics for a project/organization."""
    project_id: str
    time_range: TimeRange
    deployment_frequency: DeploymentFrequency
    lead_time_for_changes: LeadTimeMetrics
    change_failure_rate: ChangeFailureRate
    time_to_recovery: TimeToRecovery
    performance_rating: DORAPerformanceRating
    insights: List[DORAInsight] = field(default_factory=list)
    trends: Optional[DORAMetricsTrends] = None
    benchmarks: Optional[IndustryBenchmarks] = None
    calculation_time_ms: int = 0
    calculated_at: datetime = field(default_factory=datetime.now)


# Business Metrics Entities

@dataclass
class TechnicalDebtBusinessMetrics:
    """Technical debt translated to business metrics."""
    total_hours: float
    estimated_cost: Decimal
    monthly_interest: Decimal
    payoff_scenarios: List[Dict[str, Any]]


@dataclass
class SecurityBusinessMetrics:
    """Security metrics translated to business impact."""
    overall_risk_level: RiskLevel
    potential_impact: Decimal
    compliance_status: Dict[str, Any]
    insurance_implications: Dict[str, Any]


@dataclass
class TeamProductivityMetrics:
    """Team productivity metrics."""
    overall_score: float
    velocity_impact: float
    quality_impact: float
    developer_satisfaction: float
    time_to_market_impact: float


@dataclass
class ROIScenario:
    """ROI scenario for investments."""
    scenario_name: str
    investment_amount: Decimal
    expected_return: Decimal
    payback_period_months: int
    confidence_level: float


@dataclass
class ROIAnalysis:
    """ROI analysis for quality improvements."""
    current_efficiency: float
    improvement_potential: float
    investment_scenarios: List[ROIScenario]
    payback_periods: Dict[str, int]


@dataclass
class BusinessMetrics:
    """Technical metrics translated to business value."""
    overall_quality_score: float
    quality_trend: TrendDirection
    technical_debt: TechnicalDebtBusinessMetrics
    security_metrics: SecurityBusinessMetrics
    dora_metrics: DORAMetrics
    team_productivity: TeamProductivityMetrics
    roi_analysis: ROIAnalysis


# KPI Entities

@dataclass
class QualityKPIs:
    """Quality-related KPIs."""
    overall_quality_score: float
    quality_trend: TrendDirection
    technical_debt_hours: float
    debt_per_file: float
    total_issues: int
    critical_issues: int
    issue_resolution_rate: float
    code_coverage_percentage: float
    maintainability_index: float


@dataclass
class VelocityKPIs:
    """Development velocity KPIs."""
    story_points_per_sprint: float
    cycle_time_days: float
    throughput_stories_per_week: float
    deployment_frequency: float
    lead_time_hours: float
    change_failure_rate: float
    recovery_time_hours: float


@dataclass
class SecurityKPIs:
    """Security-related KPIs."""
    security_score: float
    critical_vulnerabilities: int
    vulnerability_resolution_time_days: float
    compliance_percentage: float
    security_debt_hours: float
    threat_exposure_score: float


@dataclass
class CostKPIs:
    """Cost-related KPIs."""
    technical_debt_cost: Decimal
    monthly_maintenance_cost: Decimal
    incident_cost: Decimal
    efficiency_ratio: float
    cost_per_story_point: Decimal
    quality_investment_roi: float


@dataclass
class TeamKPIs:
    """Team-related KPIs."""
    team_satisfaction_score: float
    knowledge_sharing_index: float
    code_review_participation_rate: float
    pair_programming_percentage: float
    continuous_learning_hours: float
    burnout_risk_score: float


@dataclass
class OrganizationKPIs:
    """Organization-wide KPIs."""
    velocity: VelocityKPIs
    quality: QualityKPIs
    security: SecurityKPIs
    cost: CostKPIs
    team: TeamKPIs
    overall_score: float


# Executive Report Entities

@dataclass
class KeyMetric:
    """Key metric for executive summary."""
    name: str
    value: str
    trend: TrendDirection
    business_impact: BusinessImpact
    target_value: Optional[str] = None


@dataclass
class CriticalIssue:
    """Critical issue for executive attention."""
    issue_type: str
    description: str
    business_impact: str
    resolution_timeline: str
    required_investment: Decimal


@dataclass
class EstimatedInvestment:
    """Estimated investment for recommendations."""
    development_hours: float
    cost_estimate: Decimal
    resource_requirements: List[str]
    external_dependencies: List[str]


@dataclass
class Timeline:
    """Timeline for recommendations."""
    estimated_duration_weeks: int
    milestones: List[Dict[str, Any]]
    dependencies: List[str]
    risk_factors: List[str]


@dataclass
class StrategicRecommendation:
    """Strategic recommendation for executives."""
    title: str
    description: str
    business_justification: str
    estimated_investment: EstimatedInvestment
    expected_roi: float
    timeline: Timeline
    risk_level: RiskLevel
    success_metrics: List[str]


@dataclass
class NextStep:
    """Next step in action plan."""
    step_number: int
    description: str
    responsible_party: str
    timeline: str
    dependencies: List[str]


@dataclass
class ExecutiveSummary:
    """Executive summary for reports."""
    summary_text: str
    key_metrics: List[KeyMetric]
    critical_issues: List[CriticalIssue]
    strategic_recommendations: List[StrategicRecommendation]
    next_steps: List[NextStep]


@dataclass
class ReportSection:
    """Section of an executive report."""
    section_type: str
    title: str
    content: str
    visualizations: List[str]  # IDs of visualizations
    data_tables: List[Dict[str, Any]]


@dataclass
class ReportVisualization:
    """Visualization for reports."""
    visualization_id: str
    visualization_type: str
    title: str
    data: Dict[str, Any]
    format: str  # SVG, PNG, etc.
    content: bytes


@dataclass
class ExecutiveReport:
    """Complete executive report."""
    id: str
    organization_id: str
    time_range: TimeRange
    report_type: str
    language: Language
    sections: List[ReportSection]
    visualizations: List[ReportVisualization]
    business_metrics: BusinessMetrics
    executive_summary: ExecutiveSummary
    recommendations: List[StrategicRecommendation]
    pdf_document: Optional[bytes] = None
    additional_formats: Dict[ReportFormat, bytes] = field(default_factory=dict)
    generation_time_ms: int = 0
    generated_at: datetime = field(default_factory=datetime.now)
