"""
Service interfaces for DORA metrics and executive reporting.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from .entities import (
    DORAMetrics,
    ExecutiveReport,
    BusinessMetrics,
    OrganizationKPIs,
    TimeRange,
    Language,
    TrendDirection,
    IndustryBenchmarks
)


class DORACalculatorService(ABC):
    """Service interface for DORA metrics calculation."""
    
    @abstractmethod
    async def calculate_dora_metrics(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> DORAMetrics:
        """Calculate all DORA metrics for a project."""
        pass
    
    @abstractmethod
    async def calculate_deployment_frequency(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate deployment frequency metric."""
        pass
    
    @abstractmethod
    async def calculate_lead_time(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate lead time for changes metric."""
        pass
    
    @abstractmethod
    async def calculate_change_failure_rate(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate change failure rate metric."""
        pass
    
    @abstractmethod
    async def calculate_time_to_recovery(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate time to recovery metric."""
        pass


class BusinessMetricsTranslatorService(ABC):
    """Service interface for translating technical metrics to business value."""
    
    @abstractmethod
    async def translate_to_business_metrics(
        self,
        technical_data: Dict[str, Any]
    ) -> BusinessMetrics:
        """Translate technical metrics to business metrics."""
        pass
    
    @abstractmethod
    async def calculate_technical_debt_cost(
        self,
        technical_debt_hours: float,
        hourly_rate: float = 150.0
    ) -> Dict[str, Any]:
        """Calculate technical debt in monetary terms."""
        pass
    
    @abstractmethod
    async def calculate_roi_scenarios(
        self,
        current_metrics: Dict[str, Any],
        improvement_scenarios: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate ROI for different improvement scenarios."""
        pass
    
    @abstractmethod
    async def assess_business_risks(
        self,
        security_vulnerabilities: List[Dict[str, Any]],
        quality_issues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess business risks from technical issues."""
        pass


class ExecutiveReportGeneratorService(ABC):
    """Service interface for executive report generation."""
    
    @abstractmethod
    async def generate_executive_report(
        self,
        organization_id: str,
        time_range: TimeRange,
        language: Language,
        report_type: str
    ) -> ExecutiveReport:
        """Generate complete executive report."""
        pass
    
    @abstractmethod
    async def generate_executive_summary(
        self,
        business_metrics: BusinessMetrics,
        language: Language
    ) -> Dict[str, Any]:
        """Generate executive summary."""
        pass
    
    @abstractmethod
    async def generate_strategic_recommendations(
        self,
        business_metrics: BusinessMetrics,
        organization_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate strategic recommendations."""
        pass
    
    @abstractmethod
    async def export_report(
        self,
        report: ExecutiveReport,
        format: str
    ) -> bytes:
        """Export report in specified format."""
        pass


class KPICalculatorService(ABC):
    """Service interface for KPI calculation."""
    
    @abstractmethod
    async def calculate_organization_kpis(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> OrganizationKPIs:
        """Calculate all KPIs for an organization."""
        pass
    
    @abstractmethod
    async def calculate_quality_kpis(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate quality KPIs."""
        pass
    
    @abstractmethod
    async def calculate_velocity_kpis(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate velocity KPIs."""
        pass
    
    @abstractmethod
    async def calculate_security_kpis(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate security KPIs."""
        pass
    
    @abstractmethod
    async def calculate_cost_kpis(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate cost KPIs."""
        pass
    
    @abstractmethod
    async def calculate_team_kpis(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate team KPIs."""
        pass


class TrendAnalyzerService(ABC):
    """Service interface for trend analysis."""
    
    @abstractmethod
    async def analyze_metric_trends(
        self,
        metric_history: List[Dict[str, Any]],
        window_days: int = 30
    ) -> TrendDirection:
        """Analyze trends in metric history."""
        pass
    
    @abstractmethod
    async def forecast_metrics(
        self,
        historical_data: List[Dict[str, Any]],
        forecast_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Forecast future metric values."""
        pass
    
    @abstractmethod
    async def identify_anomalies(
        self,
        metric_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify anomalies in metric history."""
        pass
    
    @abstractmethod
    async def calculate_correlation(
        self,
        metric1: List[float],
        metric2: List[float]
    ) -> float:
        """Calculate correlation between two metrics."""
        pass


class BenchmarkComparatorService(ABC):
    """Service interface for benchmark comparison."""
    
    @abstractmethod
    async def get_industry_benchmarks(
        self,
        industry: Optional[str] = None,
        company_size: Optional[str] = None
    ) -> IndustryBenchmarks:
        """Get industry benchmarks for comparison."""
        pass
    
    @abstractmethod
    async def compare_with_benchmarks(
        self,
        metrics: DORAMetrics,
        benchmarks: IndustryBenchmarks
    ) -> Dict[str, Any]:
        """Compare metrics with industry benchmarks."""
        pass
    
    @abstractmethod
    async def calculate_percentile_rank(
        self,
        metric_value: float,
        benchmark_distribution: List[float]
    ) -> float:
        """Calculate percentile rank for a metric value."""
        pass
    
    @abstractmethod
    async def get_improvement_recommendations(
        self,
        comparison_results: Dict[str, Any]
    ) -> List[str]:
        """Get recommendations based on benchmark comparison."""
        pass
