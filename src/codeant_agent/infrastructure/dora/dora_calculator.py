"""
DORA metrics calculator implementation.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from statistics import mean, median, stdev
import numpy as np
from collections import defaultdict

from codeant_agent.domain.dora import (
    DORACalculatorService,
    DORAMetrics,
    DeploymentFrequency,
    LeadTimeMetrics,
    ChangeFailureRate,
    TimeToRecovery,
    DORAPerformanceRating,
    DORAPerformanceCategory,
    DeploymentPerformanceCategory,
    LeadTimePerformanceCategory,
    FailureRatePerformanceCategory,
    RecoveryTimePerformanceCategory,
    TimeRange,
    TrendDirection,
    DeploymentFrequencyStats,
    LeadTimeStats,
    LeadTimeEntry,
    LeadTimeBottleneck,
    FailurePattern,
    FailureRootCause,
    RecoveryTimeStats,
    RecoveryIncident,
    RecoveryPattern,
    DORAInsight,
    DORAMetricsTrends,
    IndustryBenchmarks,
    IndustryBenchmark,
    BusinessImpact,
    DeploymentRepository,
    IncidentRepository
)
from codeant_agent.infrastructure.git import GitAnalyzer
import logging

logger = logging.getLogger(__name__)


class DORACalculator(DORACalculatorService):
    """Implementation of DORA metrics calculator."""
    
    def __init__(
        self,
        deployment_repo: DeploymentRepository,
        incident_repo: IncidentRepository,
        git_analyzer: GitAnalyzer
    ):
        self.deployment_repo = deployment_repo
        self.incident_repo = incident_repo
        self.git_analyzer = git_analyzer
        
        # Configuration
        self.business_hours_only = True
        self.exclude_weekends = True
        self.production_branches = ["main", "master", "production"]
        
    async def calculate_dora_metrics(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> DORAMetrics:
        """Calculate all DORA metrics for a project."""
        start_time = datetime.now()
        
        try:
            # Calculate individual metrics in parallel
            results = await asyncio.gather(
                self.calculate_deployment_frequency(project_id, time_range),
                self.calculate_lead_time(project_id, time_range),
                self.calculate_change_failure_rate(project_id, time_range),
                self.calculate_time_to_recovery(project_id, time_range),
                return_exceptions=True
            )
            
            # Handle any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error calculating metric {i}: {result}")
                    raise result
            
            deployment_freq, lead_time, failure_rate, recovery_time = results
            
            # Convert dict results to proper entities
            deployment_frequency = self._dict_to_deployment_frequency(deployment_freq)
            lead_time_metrics = self._dict_to_lead_time_metrics(lead_time)
            change_failure_rate = self._dict_to_change_failure_rate(failure_rate)
            time_to_recovery = self._dict_to_time_to_recovery(recovery_time)
            
            # Calculate overall performance rating
            performance_rating = self._calculate_performance_rating(
                deployment_frequency,
                lead_time_metrics,
                change_failure_rate,
                time_to_recovery
            )
            
            # Generate insights
            insights = await self._generate_insights(
                deployment_frequency,
                lead_time_metrics,
                change_failure_rate,
                time_to_recovery,
                project_id
            )
            
            # Calculate trends
            trends = await self._calculate_trends(project_id, time_range)
            
            # Get industry benchmarks
            benchmarks = await self._get_benchmarks()
            
            calculation_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return DORAMetrics(
                project_id=project_id,
                time_range=time_range,
                deployment_frequency=deployment_frequency,
                lead_time_for_changes=lead_time_metrics,
                change_failure_rate=change_failure_rate,
                time_to_recovery=time_to_recovery,
                performance_rating=performance_rating,
                insights=insights,
                trends=trends,
                benchmarks=benchmarks,
                calculation_time_ms=calculation_time_ms,
                calculated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate DORA metrics: {e}")
            raise
    
    async def calculate_deployment_frequency(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate deployment frequency metric."""
        # Get all deployments in the time range
        deployments = await self.deployment_repo.get_deployments(project_id, time_range)
        
        # Filter by production deployments
        production_deployments = [
            d for d in deployments
            if d.get("branch") in self.production_branches
            and d.get("status") == "success"
        ]
        
        # Calculate frequency metrics
        total_days = time_range.duration_days()
        deployment_count = len(production_deployments)
        deployments_per_day = deployment_count / total_days if total_days > 0 else 0
        
        # Group by time periods
        daily_deployments = self._group_deployments_by_day(production_deployments)
        weekly_deployments = self._group_deployments_by_week(production_deployments)
        monthly_deployments = self._group_deployments_by_month(production_deployments)
        
        # Calculate statistics
        daily_counts = list(daily_deployments.values())
        stats = {
            "mean": mean(daily_counts) if daily_counts else 0,
            "median": median(daily_counts) if daily_counts else 0,
            "std_dev": stdev(daily_counts) if len(daily_counts) > 1 else 0,
            "min": min(daily_counts) if daily_counts else 0,
            "max": max(daily_counts) if daily_counts else 0
        }
        
        # Determine performance category
        performance_category = self._categorize_deployment_frequency(deployments_per_day)
        
        # Calculate trend
        trend = self._calculate_deployment_trend(weekly_deployments)
        
        # Generate recommendations
        recommendations = self._generate_deployment_recommendations(
            performance_category,
            stats,
            deployment_count
        )
        
        return {
            "total_deployments": deployment_count,
            "deployments_per_day": deployments_per_day,
            "daily_deployments": daily_deployments,
            "weekly_deployments": weekly_deployments,
            "monthly_deployments": monthly_deployments,
            "stats": stats,
            "performance_category": performance_category,
            "trend": trend,
            "recommendations": recommendations
        }
    
    async def calculate_lead_time(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate lead time for changes metric."""
        # Get commits and deployments
        commits = await self.git_analyzer.get_commits_in_range(project_id, time_range)
        deployments = await self.deployment_repo.get_deployments(project_id, time_range)
        
        lead_time_entries = []
        
        # Calculate lead time for each commit
        for commit in commits:
            deployment = await self._find_deployment_for_commit(commit, deployments)
            if deployment:
                lead_time_hours = (
                    deployment["deployed_at"] - commit["committed_at"]
                ).total_seconds() / 3600
                
                change_size = await self._calculate_change_size(commit)
                change_complexity = await self._calculate_change_complexity(commit, project_id)
                
                lead_time_entries.append({
                    "commit_id": commit["id"],
                    "commit_timestamp": commit["committed_at"],
                    "deployment_timestamp": deployment["deployed_at"],
                    "lead_time_hours": lead_time_hours,
                    "change_size": change_size,
                    "change_complexity": change_complexity
                })
        
        if not lead_time_entries:
            return self._empty_lead_time_metrics()
        
        # Calculate statistics
        lead_times = [entry["lead_time_hours"] for entry in lead_time_entries]
        stats = self._calculate_lead_time_stats(lead_times)
        
        # Categorize performance
        performance_category = self._categorize_lead_time_performance(stats["median"])
        
        # Identify bottlenecks
        bottlenecks = await self._identify_lead_time_bottlenecks(project_id, time_range)
        
        # Calculate trend
        trend = self._calculate_lead_time_trend(lead_time_entries)
        
        # Generate recommendations
        recommendations = self._generate_lead_time_recommendations(
            performance_category,
            stats,
            bottlenecks
        )
        
        return {
            "total_changes": len(lead_time_entries),
            "lead_time_entries": lead_time_entries,
            "stats": stats,
            "performance_category": performance_category,
            "trend": trend,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations
        }
    
    async def calculate_change_failure_rate(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate change failure rate metric."""
        # Get all deployments
        deployments = await self.deployment_repo.get_deployments(project_id, time_range)
        
        # Get failed deployments
        failed_deployments = await self.incident_repo.get_failed_deployments(
            project_id,
            time_range
        )
        
        # Calculate failure rate
        total_deployments = len(deployments)
        failed_count = len(failed_deployments)
        failure_rate = failed_count / total_deployments if total_deployments > 0 else 0
        
        # Analyze failure patterns
        failure_patterns = await self._analyze_failure_patterns(failed_deployments)
        
        # Identify root causes
        root_causes = await self._identify_failure_root_causes(failed_deployments)
        
        # Categorize performance
        performance_category = self._categorize_failure_rate_performance(failure_rate)
        
        # Calculate trend
        trend = await self._calculate_failure_rate_trend(project_id, time_range)
        
        # Generate recommendations
        recommendations = self._generate_failure_rate_recommendations(
            performance_category,
            failure_patterns,
            root_causes
        )
        
        return {
            "total_deployments": total_deployments,
            "failed_deployments": failed_count,
            "failure_rate_percentage": failure_rate * 100,
            "failure_patterns": failure_patterns,
            "performance_category": performance_category,
            "trend": trend,
            "root_causes": root_causes,
            "recommendations": recommendations
        }
    
    async def calculate_time_to_recovery(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Calculate time to recovery metric."""
        # Get incidents
        incidents = await self.incident_repo.get_incidents(project_id, time_range)
        
        recovery_incidents = []
        
        # Calculate recovery time for each incident
        for incident in incidents:
            if incident.get("resolved_at"):
                recovery_time_hours = (
                    incident["resolved_at"] - incident["created_at"]
                ).total_seconds() / 3600
                
                recovery_incidents.append({
                    "incident_id": incident["id"],
                    "failure_timestamp": incident["created_at"],
                    "recovery_timestamp": incident["resolved_at"],
                    "recovery_time_hours": recovery_time_hours,
                    "severity": incident.get("severity", "unknown"),
                    "root_cause": incident.get("root_cause", "unknown"),
                    "resolution_method": incident.get("resolution_method", "unknown")
                })
        
        if not recovery_incidents:
            return self._empty_recovery_metrics()
        
        # Calculate statistics
        recovery_times = [inc["recovery_time_hours"] for inc in recovery_incidents]
        stats = self._calculate_recovery_stats(recovery_times)
        
        # Categorize performance
        performance_category = self._categorize_recovery_performance(stats["median_hours"])
        
        # Analyze recovery patterns
        recovery_patterns = self._analyze_recovery_patterns(recovery_incidents)
        
        # Calculate trend
        trend = self._calculate_recovery_trend(recovery_incidents)
        
        # Generate recommendations
        recommendations = self._generate_recovery_recommendations(
            performance_category,
            recovery_patterns,
            stats
        )
        
        return {
            "incidents": recovery_incidents,
            "stats": stats,
            "performance_category": performance_category,
            "trend": trend,
            "recovery_patterns": recovery_patterns,
            "recommendations": recommendations
        }
    
    def _calculate_performance_rating(
        self,
        deployment_freq: DeploymentFrequency,
        lead_time: LeadTimeMetrics,
        failure_rate: ChangeFailureRate,
        recovery_time: TimeToRecovery
    ) -> DORAPerformanceRating:
        """Calculate overall DORA performance rating."""
        # Score each metric (4 = Elite, 3 = High, 2 = Medium, 1 = Low)
        scores = {
            "deployment": self._score_category(deployment_freq.performance_category),
            "lead_time": self._score_category(lead_time.performance_category),
            "failure_rate": self._score_category(failure_rate.performance_category),
            "recovery": self._score_category(recovery_time.performance_category)
        }
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores)
        
        # Determine overall category
        if overall_score >= 3.5:
            overall_category = DORAPerformanceCategory.ELITE
        elif overall_score >= 2.5:
            overall_category = DORAPerformanceCategory.HIGH
        elif overall_score >= 1.5:
            overall_category = DORAPerformanceCategory.MEDIUM
        else:
            overall_category = DORAPerformanceCategory.LOW
        
        # Identify strengths and improvement areas
        strengths = []
        improvement_areas = []
        
        for metric, score in scores.items():
            if score >= 3:
                strengths.append(f"Strong {metric} performance")
            elif score <= 2:
                improvement_areas.append(f"Improve {metric} performance")
        
        return DORAPerformanceRating(
            overall_category=overall_category,
            overall_score=overall_score,
            deployment_score=float(scores["deployment"]),
            lead_time_score=float(scores["lead_time"]),
            failure_rate_score=float(scores["failure_rate"]),
            recovery_time_score=float(scores["recovery"]),
            strengths=strengths,
            improvement_areas=improvement_areas
        )
    
    def _score_category(self, category: Any) -> int:
        """Convert performance category to numeric score."""
        category_name = category.value if hasattr(category, 'value') else str(category)
        
        if "elite" in category_name.lower():
            return 4
        elif "high" in category_name.lower():
            return 3
        elif "medium" in category_name.lower():
            return 2
        else:
            return 1
    
    def _categorize_deployment_frequency(self, deployments_per_day: float) -> str:
        """Categorize deployment frequency performance."""
        if deployments_per_day >= 1:  # Multiple per day
            return "elite"
        elif deployments_per_day >= 1/7:  # Between daily and weekly
            return "high"
        elif deployments_per_day >= 1/30:  # Between weekly and monthly
            return "medium"
        else:
            return "low"
    
    def _categorize_lead_time_performance(self, median_hours: float) -> str:
        """Categorize lead time performance."""
        if median_hours < 1:  # Less than 1 hour
            return "elite"
        elif median_hours < 24:  # Less than 1 day
            return "high"
        elif median_hours < 168:  # Less than 1 week
            return "medium"
        else:
            return "low"
    
    def _categorize_failure_rate_performance(self, failure_rate: float) -> str:
        """Categorize failure rate performance."""
        if failure_rate <= 0.15:  # 0-15%
            return "elite"
        elif failure_rate <= 0.30:  # 16-30%
            return "high"
        elif failure_rate <= 0.45:  # 31-45%
            return "medium"
        else:
            return "low"
    
    def _categorize_recovery_performance(self, median_hours: float) -> str:
        """Categorize recovery time performance."""
        if median_hours < 1:  # Less than 1 hour
            return "elite"
        elif median_hours < 24:  # Less than 1 day
            return "high"
        elif median_hours < 168:  # Less than 1 week
            return "medium"
        else:
            return "low"
    
    def _calculate_lead_time_stats(self, lead_times: List[float]) -> Dict[str, float]:
        """Calculate lead time statistics."""
        if not lead_times:
            return self._empty_lead_time_stats()
        
        sorted_times = sorted(lead_times)
        
        return {
            "mean": mean(lead_times),
            "median": median(lead_times),
            "p50": np.percentile(sorted_times, 50),
            "p75": np.percentile(sorted_times, 75),
            "p90": np.percentile(sorted_times, 90),
            "p95": np.percentile(sorted_times, 95),
            "std_dev": stdev(lead_times) if len(lead_times) > 1 else 0,
            "min": min(lead_times),
            "max": max(lead_times)
        }
    
    def _calculate_recovery_stats(self, recovery_times: List[float]) -> Dict[str, float]:
        """Calculate recovery time statistics."""
        if not recovery_times:
            return self._empty_recovery_stats()
        
        sorted_times = sorted(recovery_times)
        
        return {
            "mean_hours": mean(recovery_times),
            "median_hours": median(recovery_times),
            "p50_hours": np.percentile(sorted_times, 50),
            "p75_hours": np.percentile(sorted_times, 75),
            "p90_hours": np.percentile(sorted_times, 90),
            "p95_hours": np.percentile(sorted_times, 95),
            "std_dev_hours": stdev(recovery_times) if len(recovery_times) > 1 else 0,
            "min_hours": min(recovery_times),
            "max_hours": max(recovery_times)
        }
    
    def _group_deployments_by_day(self, deployments: List[Dict]) -> Dict[str, int]:
        """Group deployments by day."""
        daily_counts = defaultdict(int)
        
        for deployment in deployments:
            date_str = deployment["deployed_at"].strftime("%Y-%m-%d")
            daily_counts[date_str] += 1
        
        return dict(daily_counts)
    
    def _group_deployments_by_week(self, deployments: List[Dict]) -> Dict[str, int]:
        """Group deployments by week."""
        weekly_counts = defaultdict(int)
        
        for deployment in deployments:
            week_str = deployment["deployed_at"].strftime("%Y-W%U")
            weekly_counts[week_str] += 1
        
        return dict(weekly_counts)
    
    def _group_deployments_by_month(self, deployments: List[Dict]) -> Dict[str, int]:
        """Group deployments by month."""
        monthly_counts = defaultdict(int)
        
        for deployment in deployments:
            month_str = deployment["deployed_at"].strftime("%Y-%m")
            monthly_counts[month_str] += 1
        
        return dict(monthly_counts)
    
    def _calculate_deployment_trend(self, weekly_deployments: Dict[str, int]) -> str:
        """Calculate deployment frequency trend."""
        if len(weekly_deployments) < 2:
            return "stable"
        
        weeks = sorted(weekly_deployments.keys())
        values = [weekly_deployments[week] for week in weeks]
        
        # Simple trend calculation using linear regression
        x = list(range(len(values)))
        y = values
        
        if len(set(y)) == 1:  # All values are the same
            return "stable"
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend based on slope and variance
        variance = np.var(y)
        mean_value = np.mean(y)
        cv = variance / mean_value if mean_value > 0 else 0
        
        if cv > 0.5:  # High variance
            return "volatile"
        elif slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "degrading"
        else:
            return "stable"
    
    # Helper methods for entity conversion
    def _dict_to_deployment_frequency(self, data: Dict[str, Any]) -> DeploymentFrequency:
        """Convert dictionary to DeploymentFrequency entity."""
        stats = DeploymentFrequencyStats(**data["stats"])
        
        return DeploymentFrequency(
            total_deployments=data["total_deployments"],
            deployments_per_day=data["deployments_per_day"],
            daily_deployments=data["daily_deployments"],
            weekly_deployments=data["weekly_deployments"],
            monthly_deployments=data["monthly_deployments"],
            stats=stats,
            performance_category=DeploymentPerformanceCategory(data["performance_category"]),
            trend=TrendDirection(data["trend"]),
            recommendations=data["recommendations"]
        )
    
    def _dict_to_lead_time_metrics(self, data: Dict[str, Any]) -> LeadTimeMetrics:
        """Convert dictionary to LeadTimeMetrics entity."""
        stats = LeadTimeStats(**data["stats"])
        
        entries = [
            LeadTimeEntry(**entry) for entry in data["lead_time_entries"]
        ]
        
        bottlenecks = [
            LeadTimeBottleneck(**bottleneck) for bottleneck in data["bottlenecks"]
        ]
        
        return LeadTimeMetrics(
            total_changes=data["total_changes"],
            lead_time_entries=entries,
            stats=stats,
            performance_category=LeadTimePerformanceCategory(data["performance_category"]),
            trend=TrendDirection(data["trend"]),
            bottlenecks=bottlenecks,
            recommendations=data["recommendations"]
        )
    
    def _dict_to_change_failure_rate(self, data: Dict[str, Any]) -> ChangeFailureRate:
        """Convert dictionary to ChangeFailureRate entity."""
        patterns = [
            FailurePattern(**pattern) for pattern in data["failure_patterns"]
        ]
        
        root_causes = [
            FailureRootCause(**cause) for cause in data["root_causes"]
        ]
        
        return ChangeFailureRate(
            total_deployments=data["total_deployments"],
            failed_deployments=data["failed_deployments"],
            failure_rate_percentage=data["failure_rate_percentage"],
            failure_patterns=patterns,
            performance_category=FailureRatePerformanceCategory(data["performance_category"]),
            trend=TrendDirection(data["trend"]),
            root_causes=root_causes,
            recommendations=data["recommendations"]
        )
    
    def _dict_to_time_to_recovery(self, data: Dict[str, Any]) -> TimeToRecovery:
        """Convert dictionary to TimeToRecovery entity."""
        stats = RecoveryTimeStats(**data["stats"])
        
        incidents = [
            RecoveryIncident(**incident) for incident in data["incidents"]
        ]
        
        patterns = [
            RecoveryPattern(**pattern) for pattern in data["recovery_patterns"]
        ]
        
        return TimeToRecovery(
            incidents=incidents,
            stats=stats,
            performance_category=RecoveryTimePerformanceCategory(data["performance_category"]),
            trend=TrendDirection(data["trend"]),
            recovery_patterns=patterns,
            recommendations=data["recommendations"]
        )
    
    # Empty metrics helpers
    def _empty_lead_time_metrics(self) -> Dict[str, Any]:
        """Return empty lead time metrics."""
        return {
            "total_changes": 0,
            "lead_time_entries": [],
            "stats": self._empty_lead_time_stats(),
            "performance_category": "low",
            "trend": "stable",
            "bottlenecks": [],
            "recommendations": ["No data available for lead time analysis"]
        }
    
    def _empty_lead_time_stats(self) -> Dict[str, float]:
        """Return empty lead time statistics."""
        return {
            "mean": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0
        }
    
    def _empty_recovery_metrics(self) -> Dict[str, Any]:
        """Return empty recovery metrics."""
        return {
            "incidents": [],
            "stats": self._empty_recovery_stats(),
            "performance_category": "low",
            "trend": "stable",
            "recovery_patterns": [],
            "recommendations": ["No incident data available for recovery analysis"]
        }
    
    def _empty_recovery_stats(self) -> Dict[str, float]:
        """Return empty recovery statistics."""
        return {
            "mean_hours": 0.0,
            "median_hours": 0.0,
            "p50_hours": 0.0,
            "p75_hours": 0.0,
            "p90_hours": 0.0,
            "p95_hours": 0.0,
            "std_dev_hours": 0.0,
            "min_hours": 0.0,
            "max_hours": 0.0
        }
    
    # Stub implementations for complex methods
    async def _find_deployment_for_commit(
        self,
        commit: Dict[str, Any],
        deployments: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find deployment for a commit."""
        # Simple implementation - find first deployment after commit
        commit_time = commit["committed_at"]
        
        for deployment in sorted(deployments, key=lambda d: d["deployed_at"]):
            if deployment["deployed_at"] > commit_time:
                # Check if commit is included in deployment
                if commit["id"] in deployment.get("commit_ids", []):
                    return deployment
        
        return None
    
    async def _calculate_change_size(self, commit: Dict[str, Any]) -> int:
        """Calculate size of changes in a commit."""
        # Simplified - return lines changed
        return commit.get("lines_added", 0) + commit.get("lines_deleted", 0)
    
    async def _calculate_change_complexity(
        self,
        commit: Dict[str, Any],
        project_id: str
    ) -> float:
        """Calculate complexity of changes."""
        # Simplified - based on files changed and change size
        files_changed = len(commit.get("files", []))
        change_size = await self._calculate_change_size(commit)
        
        # Simple complexity score
        complexity = (files_changed * 0.3) + (change_size * 0.001)
        return min(complexity, 10.0)  # Cap at 10
    
    async def _identify_lead_time_bottlenecks(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> List[Dict[str, Any]]:
        """Identify bottlenecks in lead time."""
        # Simplified implementation
        return [
            {
                "stage": "Code Review",
                "average_time_hours": 4.5,
                "percentage_of_total": 35.0,
                "recommendations": ["Implement automated code review checks", "Set SLA for review response time"]
            },
            {
                "stage": "Testing",
                "average_time_hours": 2.0,
                "percentage_of_total": 15.0,
                "recommendations": ["Increase test automation coverage", "Parallelize test execution"]
            }
        ]
    
    def _calculate_lead_time_trend(self, entries: List[Dict[str, Any]]) -> str:
        """Calculate lead time trend."""
        if len(entries) < 2:
            return "stable"
        
        # Sort by commit timestamp
        sorted_entries = sorted(entries, key=lambda e: e["commit_timestamp"])
        
        # Get lead times over time
        lead_times = [e["lead_time_hours"] for e in sorted_entries]
        
        # Calculate trend using simple linear regression
        x = list(range(len(lead_times)))
        y = lead_times
        
        if len(set(y)) == 1:
            return "stable"
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Negative slope means improving (shorter lead times)
        if slope < -0.5:
            return "improving"
        elif slope > 0.5:
            return "degrading"
        else:
            return "stable"
    
    async def _analyze_failure_patterns(
        self,
        failed_deployments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in deployment failures."""
        # Simplified pattern analysis
        patterns = []
        
        # Time-based patterns
        failure_by_hour = defaultdict(int)
        for deployment in failed_deployments:
            hour = deployment["deployed_at"].hour
            failure_by_hour[hour] += 1
        
        # Find peak failure hours
        if failure_by_hour:
            peak_hour = max(failure_by_hour, key=failure_by_hour.get)
            if failure_by_hour[peak_hour] > len(failed_deployments) * 0.2:
                patterns.append({
                    "pattern_type": "time_based",
                    "frequency": failure_by_hour[peak_hour],
                    "description": f"High failure rate during hour {peak_hour}:00",
                    "impact": "medium",
                    "mitigation": "Avoid deployments during peak hours or increase monitoring"
                })
        
        # Size-based patterns
        large_changes = [d for d in failed_deployments if d.get("change_size", 0) > 1000]
        if len(large_changes) > len(failed_deployments) * 0.3:
            patterns.append({
                "pattern_type": "size_based",
                "frequency": len(large_changes),
                "description": "Large changes have higher failure rate",
                "impact": "high",
                "mitigation": "Break down large changes into smaller, incremental deployments"
            })
        
        return patterns
    
    async def _identify_failure_root_causes(
        self,
        failed_deployments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify root causes of failures."""
        # Simplified root cause analysis
        root_causes = []
        
        # Categorize failures
        cause_counts = defaultdict(int)
        for deployment in failed_deployments:
            cause = deployment.get("failure_reason", "unknown")
            cause_counts[cause] += 1
        
        # Convert to root cause list
        for cause, count in cause_counts.items():
            impact_score = count / len(failed_deployments) * 10
            
            root_causes.append({
                "cause_type": cause,
                "description": f"Failure due to {cause}",
                "frequency": count,
                "impact_score": min(impact_score, 10.0),
                "remediation": self._get_remediation_for_cause(cause)
            })
        
        return sorted(root_causes, key=lambda x: x["impact_score"], reverse=True)
    
    def _get_remediation_for_cause(self, cause: str) -> str:
        """Get remediation suggestion for a failure cause."""
        remediation_map = {
            "test_failure": "Improve test coverage and reliability",
            "deployment_error": "Review deployment scripts and add validation",
            "configuration_error": "Implement configuration validation and testing",
            "dependency_issue": "Add dependency checks and version pinning",
            "performance_degradation": "Implement performance testing in CI/CD",
            "unknown": "Add better error tracking and monitoring"
        }
        
        return remediation_map.get(cause, "Investigate and add specific remediation")
    
    async def _calculate_failure_rate_trend(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> str:
        """Calculate failure rate trend."""
        # Get historical failure rates
        # Simplified - return stable for now
        return "stable"
    
    def _analyze_recovery_patterns(
        self,
        incidents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in recovery times."""
        patterns = []
        
        # Severity-based patterns
        severity_times = defaultdict(list)
        for incident in incidents:
            severity = incident["severity"]
            severity_times[severity].append(incident["recovery_time_hours"])
        
        for severity, times in severity_times.items():
            avg_time = mean(times)
            patterns.append({
                "pattern_type": f"{severity}_severity",
                "average_recovery_hours": avg_time,
                "frequency": len(times),
                "recommendations": self._get_recovery_recommendations_for_severity(severity, avg_time)
            })
        
        return sorted(patterns, key=lambda x: x["average_recovery_hours"], reverse=True)
    
    def _get_recovery_recommendations_for_severity(
        self,
        severity: str,
        avg_hours: float
    ) -> List[str]:
        """Get recovery recommendations based on severity."""
        if severity == "critical" and avg_hours > 1:
            return [
                "Implement automated rollback procedures",
                "Create runbooks for critical incidents",
                "Set up dedicated incident response team"
            ]
        elif severity == "high" and avg_hours > 4:
            return [
                "Improve monitoring and alerting",
                "Document common resolution procedures"
            ]
        else:
            return ["Continue monitoring recovery times"]
    
    def _calculate_recovery_trend(self, incidents: List[Dict[str, Any]]) -> str:
        """Calculate recovery time trend."""
        if len(incidents) < 2:
            return "stable"
        
        # Sort by incident time
        sorted_incidents = sorted(incidents, key=lambda i: i["failure_timestamp"])
        
        # Get recovery times over time
        recovery_times = [i["recovery_time_hours"] for i in sorted_incidents]
        
        # Calculate trend
        x = list(range(len(recovery_times)))
        y = recovery_times
        
        if len(set(y)) == 1:
            return "stable"
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Negative slope means improving (shorter recovery times)
        if slope < -0.1:
            return "improving"
        elif slope > 0.1:
            return "degrading"
        else:
            return "stable"
    
    # Recommendation generators
    def _generate_deployment_recommendations(
        self,
        category: str,
        stats: Dict[str, float],
        deployment_count: int
    ) -> List[str]:
        """Generate deployment frequency recommendations."""
        recommendations = []
        
        if category == "low":
            recommendations.extend([
                "Implement continuous integration to enable more frequent deployments",
                "Break down large releases into smaller, incremental changes",
                "Automate deployment processes to reduce manual overhead"
            ])
        elif category == "medium":
            recommendations.extend([
                "Work towards daily deployments by improving automation",
                "Implement feature flags to decouple deployment from release",
                "Reduce batch size of changes"
            ])
        elif category == "high":
            recommendations.extend([
                "Consider implementing continuous deployment for faster delivery",
                "Monitor deployment success rate closely"
            ])
        else:  # elite
            recommendations.extend([
                "Maintain current deployment frequency",
                "Focus on maintaining quality while deploying frequently"
            ])
        
        # Add specific recommendations based on stats
        if stats["std_dev"] > stats["mean"] * 0.5:
            recommendations.append("Reduce deployment frequency variability for more predictable delivery")
        
        return recommendations
    
    def _generate_lead_time_recommendations(
        self,
        category: str,
        stats: Dict[str, float],
        bottlenecks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate lead time recommendations."""
        recommendations = []
        
        if category in ["low", "medium"]:
            recommendations.extend([
                "Implement automated testing to reduce manual verification time",
                "Improve CI/CD pipeline efficiency",
                "Consider trunk-based development to reduce merge conflicts"
            ])
        
        # Add bottleneck-specific recommendations
        for bottleneck in bottlenecks[:2]:  # Top 2 bottlenecks
            recommendations.extend(bottleneck["recommendations"])
        
        # Add percentile-based recommendations
        if stats["p90"] > stats["median"] * 3:
            recommendations.append("Investigate and address outliers causing very long lead times")
        
        return recommendations
    
    def _generate_failure_rate_recommendations(
        self,
        category: str,
        patterns: List[Dict[str, Any]],
        root_causes: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate failure rate recommendations."""
        recommendations = []
        
        if category in ["low", "medium"]:
            recommendations.extend([
                "Implement comprehensive automated testing",
                "Add deployment verification steps",
                "Improve rollback procedures"
            ])
        
        # Add pattern-specific recommendations
        for pattern in patterns[:2]:
            recommendations.append(pattern["mitigation"])
        
        # Add root cause remediations
        for cause in root_causes[:2]:
            recommendations.append(cause["remediation"])
        
        return recommendations
    
    def _generate_recovery_recommendations(
        self,
        category: str,
        patterns: List[Dict[str, Any]],
        stats: Dict[str, float]
    ) -> List[str]:
        """Generate recovery time recommendations."""
        recommendations = []
        
        if category in ["low", "medium"]:
            recommendations.extend([
                "Implement automated rollback procedures",
                "Create and maintain incident runbooks",
                "Improve monitoring and alerting systems"
            ])
        
        # Add pattern-specific recommendations
        for pattern in patterns:
            recommendations.extend(pattern["recommendations"])
        
        # Add stats-based recommendations
        if stats["p95_hours"] > 24:
            recommendations.append("Focus on reducing recovery time for critical incidents")
        
        return recommendations
    
    # Insight generation
    async def _generate_insights(
        self,
        deployment_freq: DeploymentFrequency,
        lead_time: LeadTimeMetrics,
        failure_rate: ChangeFailureRate,
        recovery_time: TimeToRecovery,
        project_id: str
    ) -> List[DORAInsight]:
        """Generate insights from DORA metrics."""
        insights = []
        
        # Deployment frequency insights
        if deployment_freq.performance_category == DeploymentPerformanceCategory.ELITE:
            insights.append(DORAInsight(
                insight_type="achievement",
                title="Elite Deployment Frequency",
                description="Your team deploys multiple times per day, achieving elite performance",
                impact=BusinessImpact.POSITIVE,
                recommendations=["Share deployment practices with other teams"],
                supporting_data={"deployments_per_day": deployment_freq.deployments_per_day}
            ))
        elif deployment_freq.performance_category == DeploymentPerformanceCategory.LOW:
            insights.append(DORAInsight(
                insight_type="improvement_opportunity",
                title="Low Deployment Frequency",
                description="Infrequent deployments may lead to larger, riskier changes",
                impact=BusinessImpact.NEGATIVE,
                recommendations=deployment_freq.recommendations[:2],
                supporting_data={"current_frequency": deployment_freq.deployments_per_day}
            ))
        
        # Lead time insights
        if lead_time.stats.p90 > lead_time.stats.median * 3:
            insights.append(DORAInsight(
                insight_type="bottleneck",
                title="High Lead Time Variability",
                description="Some changes take significantly longer than others to deploy",
                impact=BusinessImpact.NEGATIVE,
                recommendations=["Investigate outliers", "Standardize deployment process"],
                supporting_data={
                    "median_hours": lead_time.stats.median,
                    "p90_hours": lead_time.stats.p90
                }
            ))
        
        # Failure rate insights
        if failure_rate.failure_rate_percentage > 30:
            insights.append(DORAInsight(
                insight_type="risk",
                title="High Change Failure Rate",
                description=f"{failure_rate.failure_rate_percentage:.1f}% of deployments fail, indicating quality issues",
                impact=BusinessImpact.CRITICAL,
                recommendations=failure_rate.recommendations[:3],
                supporting_data={
                    "failure_rate": failure_rate.failure_rate_percentage,
                    "failed_deployments": failure_rate.failed_deployments
                }
            ))
        
        # Recovery time insights
        if recovery_time.performance_category == RecoveryTimePerformanceCategory.ELITE:
            insights.append(DORAInsight(
                insight_type="achievement",
                title="Excellent Recovery Time",
                description="Your team recovers from incidents very quickly",
                impact=BusinessImpact.POSITIVE,
                recommendations=["Document recovery procedures for knowledge sharing"],
                supporting_data={"median_recovery_hours": recovery_time.stats.median_hours}
            ))
        
        # Cross-metric insights
        if (deployment_freq.performance_category in [DeploymentPerformanceCategory.ELITE, DeploymentPerformanceCategory.HIGH] and
            failure_rate.performance_category in [FailureRatePerformanceCategory.LOW, FailureRatePerformanceCategory.MEDIUM]):
            insights.append(DORAInsight(
                insight_type="imbalance",
                title="Speed vs Stability Imbalance",
                description="High deployment frequency but also high failure rate suggests need for better quality controls",
                impact=BusinessImpact.NEGATIVE,
                recommendations=[
                    "Strengthen automated testing",
                    "Implement progressive rollouts",
                    "Add deployment verification steps"
                ],
                supporting_data={
                    "deployment_category": deployment_freq.performance_category.value,
                    "failure_category": failure_rate.performance_category.value
                }
            ))
        
        return insights
    
    # Trend calculation
    async def _calculate_trends(
        self,
        project_id: str,
        time_range: TimeRange
    ) -> Optional[DORAMetricsTrends]:
        """Calculate trends for all DORA metrics."""
        # Simplified implementation - would need historical data
        return DORAMetricsTrends(
            deployment_frequency_trend=[],
            lead_time_trend=[],
            failure_rate_trend=[],
            recovery_time_trend=[],
            overall_trend=TrendDirection.STABLE
        )
    
    # Benchmark retrieval
    async def _get_benchmarks(self) -> Optional[IndustryBenchmarks]:
        """Get industry benchmarks."""
        # Based on 2023 State of DevOps Report
        return IndustryBenchmarks(
            deployment_frequency=IndustryBenchmark(
                metric_name="deployment_frequency",
                elite_threshold=1.0,  # Multiple per day
                high_threshold=1/7,   # Daily to weekly
                medium_threshold=1/30,  # Weekly to monthly
                low_threshold=0,      # Less than monthly
                industry_average=1/14,  # Bi-weekly
                percentile_rank=0.0   # Would be calculated based on actual performance
            ),
            lead_time=IndustryBenchmark(
                metric_name="lead_time_hours",
                elite_threshold=1.0,   # Less than 1 hour
                high_threshold=24.0,   # Less than 1 day
                medium_threshold=168.0,  # Less than 1 week
                low_threshold=720.0,   # Less than 1 month
                industry_average=96.0,  # 4 days
                percentile_rank=0.0
            ),
            failure_rate=IndustryBenchmark(
                metric_name="change_failure_rate",
                elite_threshold=0.15,  # 0-15%
                high_threshold=0.30,   # 16-30%
                medium_threshold=0.45, # 31-45%
                low_threshold=1.0,     # >45%
                industry_average=0.25,
                percentile_rank=0.0
            ),
            recovery_time=IndustryBenchmark(
                metric_name="time_to_recovery_hours",
                elite_threshold=1.0,   # Less than 1 hour
                high_threshold=24.0,   # Less than 1 day
                medium_threshold=168.0,  # Less than 1 week
                low_threshold=720.0,   # Less than 1 month
                industry_average=48.0,  # 2 days
                percentile_rank=0.0
            ),
            last_updated=datetime.now()
        )
