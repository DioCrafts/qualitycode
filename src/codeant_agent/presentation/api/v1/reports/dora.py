"""
DORA metrics API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from datetime import datetime, timedelta

from codeant_agent.application.use_cases.dora import (
    CalculateDORAMetricsUseCase,
    GetHistoricalDORAMetricsUseCase,
    CompareDORAMetricsUseCase
)
from codeant_agent.presentation.api.dependencies import (
    get_current_user,
    get_dora_calculator,
    get_dora_repository
)
from codeant_agent.presentation.api.models.dora import (
    DORAMetricsResponse,
    DORAMetricsRequest,
    HistoricalDORAResponse,
    DORAComparisonResponse
)
from codeant_agent.domain.dora import TimeRange

router = APIRouter(prefix="/dora", tags=["DORA Metrics"])


@router.post("/calculate", response_model=DORAMetricsResponse)
async def calculate_dora_metrics(
    request: DORAMetricsRequest,
    current_user=Depends(get_current_user),
    dora_calculator=Depends(get_dora_calculator)
):
    """
    Calculate DORA metrics for a project or organization.
    
    Calculates the four key DORA metrics:
    - Deployment Frequency
    - Lead Time for Changes
    - Change Failure Rate
    - Time to Recovery
    """
    try:
        # Create time range
        time_range = TimeRange(
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Execute use case
        use_case = CalculateDORAMetricsUseCase(dora_calculator)
        metrics = await use_case.execute(
            project_id=request.project_id,
            time_range=time_range,
            include_insights=request.include_insights,
            include_benchmarks=request.include_benchmarks
        )
        
        return DORAMetricsResponse.from_domain(metrics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/{project_id}", response_model=DORAMetricsResponse)
async def get_project_dora_metrics(
    project_id: str,
    start_date: Optional[datetime] = Query(None, description="Start date for metrics calculation"),
    end_date: Optional[datetime] = Query(None, description="End date for metrics calculation"),
    current_user=Depends(get_current_user),
    dora_calculator=Depends(get_dora_calculator)
):
    """
    Get DORA metrics for a specific project.
    
    If dates are not provided, defaults to last 30 days.
    """
    # Default to last 30 days if dates not provided
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    time_range = TimeRange(start_date=start_date, end_date=end_date)
    
    try:
        use_case = CalculateDORAMetricsUseCase(dora_calculator)
        metrics = await use_case.execute(
            project_id=project_id,
            time_range=time_range,
            include_insights=True,
            include_benchmarks=True
        )
        
        return DORAMetricsResponse.from_domain(metrics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organization/{organization_id}", response_model=list[DORAMetricsResponse])
async def get_organization_dora_metrics(
    organization_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    aggregate: bool = Query(False, description="Aggregate metrics across all projects"),
    current_user=Depends(get_current_user),
    dora_calculator=Depends(get_dora_calculator),
    dora_repository=Depends(get_dora_repository)
):
    """
    Get DORA metrics for all projects in an organization.
    
    Can return individual project metrics or aggregated organization metrics.
    """
    # Default to last 30 days
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    time_range = TimeRange(start_date=start_date, end_date=end_date)
    
    try:
        # Get organization metrics
        metrics_list = await dora_repository.get_organization_metrics(
            organization_id,
            time_range
        )
        
        if aggregate:
            # TODO: Implement aggregation logic
            raise HTTPException(
                status_code=501,
                detail="Aggregation not yet implemented"
            )
        
        return [DORAMetricsResponse.from_domain(m) for m in metrics_list]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{project_id}", response_model=HistoricalDORAResponse)
async def get_historical_dora_metrics(
    project_id: str,
    period: str = Query("monthly", regex="^(daily|weekly|monthly|quarterly)$"),
    periods: int = Query(6, ge=1, le=24, description="Number of periods to retrieve"),
    current_user=Depends(get_current_user),
    dora_repository=Depends(get_dora_repository)
):
    """
    Get historical DORA metrics for trend analysis.
    
    Returns metrics grouped by the specified period.
    """
    try:
        # Calculate date range based on period
        end_date = datetime.now()
        
        if period == "daily":
            start_date = end_date - timedelta(days=periods)
        elif period == "weekly":
            start_date = end_date - timedelta(weeks=periods)
        elif period == "monthly":
            start_date = end_date - timedelta(days=periods * 30)
        else:  # quarterly
            start_date = end_date - timedelta(days=periods * 90)
        
        # Get historical metrics
        use_case = GetHistoricalDORAMetricsUseCase(dora_repository)
        historical_data = await use_case.execute(
            project_id=project_id,
            start_date=start_date,
            end_date=end_date,
            period=period
        )
        
        return HistoricalDORAResponse(
            project_id=project_id,
            period=period,
            data_points=historical_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=DORAComparisonResponse)
async def compare_dora_metrics(
    project_ids: list[str],
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user=Depends(get_current_user),
    dora_calculator=Depends(get_dora_calculator)
):
    """
    Compare DORA metrics across multiple projects.
    
    Useful for benchmarking and identifying best practices.
    """
    if len(project_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 projects required for comparison"
        )
    
    if len(project_ids) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 projects allowed for comparison"
        )
    
    # Default to last 30 days
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    time_range = TimeRange(start_date=start_date, end_date=end_date)
    
    try:
        use_case = CompareDORAMetricsUseCase(dora_calculator)
        comparison = await use_case.execute(
            project_ids=project_ids,
            time_range=time_range
        )
        
        return DORAComparisonResponse(
            time_range=time_range,
            projects=comparison["projects"],
            rankings=comparison["rankings"],
            insights=comparison["insights"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmarks", response_model=dict)
async def get_dora_benchmarks(
    industry: Optional[str] = Query(None, description="Filter by industry"),
    company_size: Optional[str] = Query(None, regex="^(small|medium|large|enterprise)$"),
    current_user=Depends(get_current_user)
):
    """
    Get industry DORA benchmarks for comparison.
    
    Returns elite, high, medium, and low performance thresholds.
    """
    # TODO: Implement benchmark service
    return {
        "deployment_frequency": {
            "elite": "Multiple times per day",
            "high": "Between once per day and once per week",
            "medium": "Between once per week and once per month",
            "low": "Between once per month and once every six months"
        },
        "lead_time": {
            "elite": "Less than one hour",
            "high": "Between one hour and one day",
            "medium": "Between one day and one week",
            "low": "More than one week"
        },
        "change_failure_rate": {
            "elite": "0-15%",
            "high": "16-30%",
            "medium": "31-45%",
            "low": "More than 45%"
        },
        "time_to_recovery": {
            "elite": "Less than one hour",
            "high": "Between one hour and one day",
            "medium": "Between one day and one week",
            "low": "More than one week"
        },
        "source": "2023 State of DevOps Report",
        "last_updated": "2023-10-01"
    }
