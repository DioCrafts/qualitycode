"""
Use case for calculating DORA metrics.
"""

from typing import Optional

from codeant_agent.domain.dora import (
    DORACalculatorService,
    DORAMetrics,
    TimeRange
)
import logging

logger = logging.getLogger(__name__)


class CalculateDORAMetricsUseCase:
    """Use case for calculating DORA metrics."""
    
    def __init__(self, dora_calculator: DORACalculatorService):
        self.dora_calculator = dora_calculator
    
    async def execute(
        self,
        project_id: str,
        time_range: TimeRange,
        include_insights: bool = True,
        include_benchmarks: bool = True
    ) -> DORAMetrics:
        """
        Calculate DORA metrics for a project.
        
        Args:
            project_id: ID of the project
            time_range: Time range for calculation
            include_insights: Whether to include AI insights
            include_benchmarks: Whether to include industry benchmarks
            
        Returns:
            Complete DORA metrics
        """
        logger.info(
            f"Calculating DORA metrics for project {project_id} "
            f"from {time_range.start_date} to {time_range.end_date}"
        )
        
        try:
            # Calculate metrics
            metrics = await self.dora_calculator.calculate_dora_metrics(
                project_id,
                time_range
            )
            
            # Optionally remove insights/benchmarks if not requested
            if not include_insights:
                metrics.insights = []
            
            if not include_benchmarks:
                metrics.benchmarks = None
            
            logger.info(
                f"Successfully calculated DORA metrics for project {project_id}. "
                f"Performance: {metrics.performance_rating.overall_category.value}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate DORA metrics: {e}")
            raise
