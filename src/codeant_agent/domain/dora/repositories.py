"""
Repository interfaces for DORA metrics and executive reporting.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any

from .entities import (
    DORAMetrics,
    ExecutiveReport,
    TimeRange,
    Language,
    ReportFormat
)


class DORAMetricsRepository(ABC):
    """Repository interface for DORA metrics."""
    
    @abstractmethod
    async def save_metrics(self, metrics: DORAMetrics) -> None:
        """Save DORA metrics."""
        pass
    
    @abstractmethod
    async def get_metrics(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> Optional[DORAMetrics]:
        """Get DORA metrics for a project and time range."""
        pass
    
    @abstractmethod
    async def get_historical_metrics(
        self,
        project_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[DORAMetrics]:
        """Get historical DORA metrics."""
        pass
    
    @abstractmethod
    async def get_organization_metrics(
        self,
        organization_id: str,
        time_range: TimeRange
    ) -> List[DORAMetrics]:
        """Get DORA metrics for all projects in an organization."""
        pass


class ExecutiveReportRepository(ABC):
    """Repository interface for executive reports."""
    
    @abstractmethod
    async def save_report(self, report: ExecutiveReport) -> None:
        """Save an executive report."""
        pass
    
    @abstractmethod
    async def get_report(self, report_id: str) -> Optional[ExecutiveReport]:
        """Get an executive report by ID."""
        pass
    
    @abstractmethod
    async def get_organization_reports(
        self,
        organization_id: str,
        limit: int = 10
    ) -> List[ExecutiveReport]:
        """Get recent reports for an organization."""
        pass
    
    @abstractmethod
    async def get_scheduled_reports(
        self,
        organization_id: str
    ) -> List[Dict[str, Any]]:
        """Get scheduled reports configuration."""
        pass
    
    @abstractmethod
    async def save_scheduled_report(
        self,
        organization_id: str,
        schedule_config: Dict[str, Any]
    ) -> None:
        """Save scheduled report configuration."""
        pass


class DeploymentRepository(ABC):
    """Repository interface for deployment data."""
    
    @abstractmethod
    async def get_deployments(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> List[Dict[str, Any]]:
        """Get deployments for a project in a time range."""
        pass
    
    @abstractmethod
    async def save_deployment(
        self,
        project_id: str,
        deployment_data: Dict[str, Any]
    ) -> None:
        """Save deployment data."""
        pass
    
    @abstractmethod
    async def get_deployment_for_commit(
        self,
        commit_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get deployment data for a specific commit."""
        pass


class IncidentRepository(ABC):
    """Repository interface for incident/failure data."""
    
    @abstractmethod
    async def get_incidents(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> List[Dict[str, Any]]:
        """Get incidents for a project in a time range."""
        pass
    
    @abstractmethod
    async def save_incident(
        self,
        project_id: str,
        incident_data: Dict[str, Any]
    ) -> None:
        """Save incident data."""
        pass
    
    @abstractmethod
    async def get_incident_resolution_time(
        self,
        incident_id: str
    ) -> Optional[float]:
        """Get resolution time for an incident in hours."""
        pass
    
    @abstractmethod
    async def get_failed_deployments(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> List[Dict[str, Any]]:
        """Get deployments that resulted in failures."""
        pass
