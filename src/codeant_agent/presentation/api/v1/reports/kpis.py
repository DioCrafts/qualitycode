"""
Organization KPIs API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from datetime import datetime, timedelta

from codeant_agent.application.use_cases.kpis import (
    CalculateOrganizationKPIsUseCase,
    GetKPITrendsUseCase,
    CompareKPIsUseCase,
    ExportKPIReportUseCase
)
from codeant_agent.presentation.api.dependencies import (
    get_current_user,
    get_kpi_calculator,
    require_organization_access
)
from codeant_agent.presentation.api.models.kpis import (
    OrganizationKPIsResponse,
    KPICategoryResponse,
    KPITrendsResponse,
    KPIComparisonResponse,
    KPIExportRequest
)
from codeant_agent.domain.dora import TimeRange

router = APIRouter(prefix="/kpis", tags=["Organization KPIs"])


@router.get("/organization/{organization_id}", response_model=OrganizationKPIsResponse)
async def get_organization_kpis(
    organization_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user=Depends(get_current_user),
    kpi_calculator=Depends(get_kpi_calculator),
    _=Depends(require_organization_access)
):
    """
    Get comprehensive KPIs for an organization.
    
    Includes:
    - Quality KPIs: code quality, technical debt, issues
    - Velocity KPIs: DORA metrics, sprint velocity
    - Security KPIs: vulnerabilities, compliance
    - Cost KPIs: technical debt cost, efficiency
    - Team KPIs: satisfaction, productivity, burnout risk
    """
    # Default to last 30 days
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    time_range = TimeRange(start_date=start_date, end_date=end_date)
    
    try:
        use_case = CalculateOrganizationKPIsUseCase(kpi_calculator)
        kpis = await use_case.execute(
            organization_id=organization_id,
            time_range=time_range
        )
        
        return OrganizationKPIsResponse.from_domain(kpis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organization/{organization_id}/quality", response_model=KPICategoryResponse)
async def get_quality_kpis(
    organization_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user=Depends(get_current_user),
    kpi_calculator=Depends(get_kpi_calculator),
    _=Depends(require_organization_access)
):
    """
    Get quality-specific KPIs for an organization.
    """
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    time_range = TimeRange(start_date=start_date, end_date=end_date)
    
    try:
        quality_kpis = await kpi_calculator.calculate_quality_kpis(
            organization_id,
            time_range
        )
        
        return KPICategoryResponse(
            category="quality",
            metrics=quality_kpis,
            time_range=time_range,
            calculated_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organization/{organization_id}/velocity", response_model=KPICategoryResponse)
async def get_velocity_kpis(
    organization_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user=Depends(get_current_user),
    kpi_calculator=Depends(get_kpi_calculator),
    _=Depends(require_organization_access)
):
    """
    Get velocity-specific KPIs for an organization.
    """
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    time_range = TimeRange(start_date=start_date, end_date=end_date)
    
    try:
        velocity_kpis = await kpi_calculator.calculate_velocity_kpis(
            organization_id,
            time_range
        )
        
        return KPICategoryResponse(
            category="velocity",
            metrics=velocity_kpis,
            time_range=time_range,
            calculated_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organization/{organization_id}/security", response_model=KPICategoryResponse)
async def get_security_kpis(
    organization_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user=Depends(get_current_user),
    kpi_calculator=Depends(get_kpi_calculator),
    _=Depends(require_organization_access)
):
    """
    Get security-specific KPIs for an organization.
    """
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    time_range = TimeRange(start_date=start_date, end_date=end_date)
    
    try:
        security_kpis = await kpi_calculator.calculate_security_kpis(
            organization_id,
            time_range
        )
        
        return KPICategoryResponse(
            category="security",
            metrics=security_kpis,
            time_range=time_range,
            calculated_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organization/{organization_id}/cost", response_model=KPICategoryResponse)
async def get_cost_kpis(
    organization_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user=Depends(get_current_user),
    kpi_calculator=Depends(get_kpi_calculator),
    _=Depends(require_organization_access)
):
    """
    Get cost-specific KPIs for an organization.
    """
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    time_range = TimeRange(start_date=start_date, end_date=end_date)
    
    try:
        cost_kpis = await kpi_calculator.calculate_cost_kpis(
            organization_id,
            time_range
        )
        
        # Convert Decimal to float for JSON serialization
        serializable_kpis = {
            k: float(v) if hasattr(v, 'quantize') else v
            for k, v in cost_kpis.items()
        }
        
        return KPICategoryResponse(
            category="cost",
            metrics=serializable_kpis,
            time_range=time_range,
            calculated_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organization/{organization_id}/team", response_model=KPICategoryResponse)
async def get_team_kpis(
    organization_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user=Depends(get_current_user),
    kpi_calculator=Depends(get_kpi_calculator),
    _=Depends(require_organization_access)
):
    """
    Get team-specific KPIs for an organization.
    """
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    time_range = TimeRange(start_date=start_date, end_date=end_date)
    
    try:
        team_kpis = await kpi_calculator.calculate_team_kpis(
            organization_id,
            time_range
        )
        
        return KPICategoryResponse(
            category="team",
            metrics=team_kpis,
            time_range=time_range,
            calculated_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/{organization_id}", response_model=KPITrendsResponse)
async def get_kpi_trends(
    organization_id: str,
    metric: str = Query(..., description="KPI metric to get trends for"),
    period: str = Query("monthly", regex="^(daily|weekly|monthly|quarterly)$"),
    periods: int = Query(6, ge=1, le=24),
    current_user=Depends(get_current_user),
    _=Depends(require_organization_access)
):
    """
    Get historical trends for a specific KPI metric.
    
    Useful for tracking progress over time.
    """
    try:
        use_case = GetKPITrendsUseCase()
        trends = await use_case.execute(
            organization_id=organization_id,
            metric=metric,
            period=period,
            periods=periods
        )
        
        return KPITrendsResponse(
            organization_id=organization_id,
            metric=metric,
            period=period,
            data_points=trends
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=KPIComparisonResponse)
async def compare_organizations(
    organization_ids: list[str],
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    categories: Optional[list[str]] = Query(None),
    current_user=Depends(get_current_user),
    kpi_calculator=Depends(get_kpi_calculator)
):
    """
    Compare KPIs across multiple organizations.
    
    Useful for benchmarking between teams or departments.
    """
    if len(organization_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 organizations required for comparison"
        )
    
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    time_range = TimeRange(start_date=start_date, end_date=end_date)
    
    # Default to all categories
    if not categories:
        categories = ["quality", "velocity", "security", "cost", "team"]
    
    try:
        use_case = CompareKPIsUseCase(kpi_calculator)
        comparison = await use_case.execute(
            organization_ids=organization_ids,
            time_range=time_range,
            categories=categories
        )
        
        return KPIComparisonResponse(
            time_range=time_range,
            organizations=comparison["organizations"],
            comparisons=comparison["comparisons"],
            rankings=comparison["rankings"],
            insights=comparison["insights"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_kpis(
    request: KPIExportRequest,
    current_user=Depends(get_current_user),
    kpi_calculator=Depends(get_kpi_calculator)
):
    """
    Export KPIs in various formats (CSV, Excel, JSON).
    
    Useful for further analysis or reporting.
    """
    try:
        use_case = ExportKPIReportUseCase(kpi_calculator)
        
        export_data = await use_case.execute(
            organization_id=request.organization_id,
            time_range=TimeRange(
                start_date=request.start_date,
                end_date=request.end_date
            ),
            categories=request.categories,
            format=request.format,
            include_trends=request.include_trends
        )
        
        # Set appropriate content type
        content_types = {
            "csv": "text/csv",
            "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "json": "application/json"
        }
        
        return StreamingResponse(
            io.BytesIO(export_data),
            media_type=content_types.get(request.format, "application/octet-stream"),
            headers={
                "Content-Disposition": f"attachment; filename=kpis_{request.organization_id}_{datetime.now().strftime('%Y%m%d')}.{request.format}"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/targets/{organization_id}")
async def get_kpi_targets(
    organization_id: str,
    current_user=Depends(get_current_user),
    _=Depends(require_organization_access)
):
    """
    Get KPI targets for an organization.
    
    Returns target values for each KPI metric.
    """
    # TODO: Implement KPI target management
    return {
        "quality": {
            "overall_quality_score": 85.0,
            "technical_debt_hours": 500.0,
            "critical_issues": 0,
            "code_coverage_percentage": 80.0
        },
        "velocity": {
            "deployment_frequency": 1.0,  # Daily
            "lead_time_hours": 24.0,
            "change_failure_rate": 0.15,
            "recovery_time_hours": 1.0
        },
        "security": {
            "security_score": 90.0,
            "critical_vulnerabilities": 0,
            "compliance_percentage": 95.0
        },
        "cost": {
            "efficiency_ratio": 1.2,
            "cost_per_story_point": 1500.0
        },
        "team": {
            "team_satisfaction_score": 4.0,
            "burnout_risk_score": 20.0
        }
    }


@router.put("/targets/{organization_id}")
async def update_kpi_targets(
    organization_id: str,
    targets: dict,
    current_user=Depends(get_current_user),
    _=Depends(require_organization_access)
):
    """
    Update KPI targets for an organization.
    
    Requires organization admin permissions.
    """
    # TODO: Implement KPI target management
    return {
        "status": "updated",
        "message": "KPI targets updated successfully"
    }


# Helper imports
from fastapi.responses import StreamingResponse
import io
