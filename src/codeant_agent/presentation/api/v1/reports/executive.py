"""
Executive reports API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional
from datetime import datetime, timedelta
import io

from codeant_agent.application.use_cases.reports import (
    GenerateExecutiveReportUseCase,
    ScheduleReportUseCase,
    GetReportHistoryUseCase
)
from codeant_agent.presentation.api.dependencies import (
    get_current_user,
    get_report_generator,
    get_report_repository,
    require_organization_access
)
from codeant_agent.presentation.api.models.reports import (
    ExecutiveReportRequest,
    ExecutiveReportResponse,
    ScheduledReportRequest,
    ScheduledReportResponse,
    ReportHistoryResponse
)
from codeant_agent.domain.dora import Language, TimeRange

router = APIRouter(prefix="/executive", tags=["Executive Reports"])


@router.post("/generate", response_model=ExecutiveReportResponse)
async def generate_executive_report(
    request: ExecutiveReportRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    report_generator=Depends(get_report_generator),
    _=Depends(require_organization_access)
):
    """
    Generate an executive report for an organization.
    
    The report includes:
    - Executive summary with key findings
    - DORA metrics performance
    - Financial impact analysis
    - Strategic recommendations
    - Next steps action plan
    """
    try:
        # Create time range
        time_range = TimeRange(
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Determine language
        language = Language(request.language) if request.language else Language.ENGLISH
        
        # Execute use case
        use_case = GenerateExecutiveReportUseCase(report_generator)
        
        if request.async_generation:
            # Generate report asynchronously
            report_id = await use_case.execute_async(
                organization_id=request.organization_id,
                time_range=time_range,
                language=language,
                report_type=request.report_type,
                include_technical_appendix=request.include_technical_appendix,
                background_tasks=background_tasks
            )
            
            return ExecutiveReportResponse(
                report_id=report_id,
                status="generating",
                message="Report generation started. Check status endpoint for progress."
            )
        else:
            # Generate report synchronously
            report = await use_case.execute(
                organization_id=request.organization_id,
                time_range=time_range,
                language=language,
                report_type=request.report_type,
                include_technical_appendix=request.include_technical_appendix
            )
            
            return ExecutiveReportResponse.from_domain(report)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{report_id}")
async def download_report(
    report_id: str,
    format: str = Query("pdf", regex="^(pdf|html|json)$"),
    current_user=Depends(get_current_user),
    report_repository=Depends(get_report_repository),
    report_generator=Depends(get_report_generator)
):
    """
    Download a generated executive report.
    
    Supports PDF, HTML, and JSON formats.
    """
    try:
        # Get report from repository
        report = await report_repository.get_report(report_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Check access permissions
        if not await _check_report_access(current_user, report):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get report in requested format
        if format == "pdf":
            if not report.pdf_document:
                raise HTTPException(
                    status_code=404,
                    detail="PDF not available for this report"
                )
            
            return StreamingResponse(
                io.BytesIO(report.pdf_document),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=executive_report_{report_id}.pdf"
                }
            )
        
        elif format == "html":
            html_content = await report_generator.export_report(report, "html")
            
            return StreamingResponse(
                io.BytesIO(html_content),
                media_type="text/html",
                headers={
                    "Content-Disposition": f"inline; filename=executive_report_{report_id}.html"
                }
            )
        
        else:  # json
            json_content = await report_generator.export_report(report, "json")
            
            return StreamingResponse(
                io.BytesIO(json_content),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=executive_report_{report_id}.json"
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{report_id}")
async def get_report_status(
    report_id: str,
    current_user=Depends(get_current_user),
    report_repository=Depends(get_report_repository)
):
    """
    Get the status of a report generation request.
    
    Useful for tracking async report generation.
    """
    try:
        # Check if report exists
        report = await report_repository.get_report(report_id)
        
        if not report:
            # Check async job status
            job_status = await report_repository.get_job_status(report_id)
            
            if not job_status:
                raise HTTPException(status_code=404, detail="Report not found")
            
            return {
                "report_id": report_id,
                "status": job_status["status"],
                "progress": job_status.get("progress", 0),
                "message": job_status.get("message", ""),
                "created_at": job_status.get("created_at"),
                "estimated_completion": job_status.get("estimated_completion")
            }
        
        # Report is complete
        return {
            "report_id": report_id,
            "status": "completed",
            "progress": 100,
            "message": "Report generated successfully",
            "created_at": report.generated_at,
            "download_url": f"/api/v1/reports/executive/download/{report_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=ReportHistoryResponse)
async def get_report_history(
    organization_id: Optional[str] = Query(None),
    report_type: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
    report_repository=Depends(get_report_repository)
):
    """
    Get history of generated reports.
    
    Can filter by organization and report type.
    """
    try:
        use_case = GetReportHistoryUseCase(report_repository)
        
        reports, total = await use_case.execute(
            user_id=current_user.id,
            organization_id=organization_id,
            report_type=report_type,
            limit=limit,
            offset=offset
        )
        
        return ReportHistoryResponse(
            reports=[_report_to_summary(r) for r in reports],
            total=total,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule", response_model=ScheduledReportResponse)
async def schedule_report(
    request: ScheduledReportRequest,
    current_user=Depends(get_current_user),
    report_repository=Depends(get_report_repository),
    _=Depends(require_organization_access)
):
    """
    Schedule recurring executive reports.
    
    Reports can be scheduled:
    - Daily
    - Weekly
    - Monthly
    - Quarterly
    
    And delivered via:
    - Email
    - Slack
    - Teams
    - Webhook
    """
    try:
        use_case = ScheduleReportUseCase(report_repository)
        
        schedule_id = await use_case.execute(
            organization_id=request.organization_id,
            schedule_config={
                "frequency": request.frequency,
                "day_of_week": request.day_of_week,
                "day_of_month": request.day_of_month,
                "time": request.time,
                "timezone": request.timezone,
                "report_type": request.report_type,
                "language": request.language,
                "recipients": request.recipients,
                "delivery_methods": request.delivery_methods,
                "enabled": request.enabled
            },
            created_by=current_user.id
        )
        
        return ScheduledReportResponse(
            schedule_id=schedule_id,
            status="created",
            message="Report schedule created successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedules", response_model=list[dict])
async def get_scheduled_reports(
    organization_id: str,
    current_user=Depends(get_current_user),
    report_repository=Depends(get_report_repository),
    _=Depends(require_organization_access)
):
    """
    Get all scheduled reports for an organization.
    """
    try:
        schedules = await report_repository.get_scheduled_reports(organization_id)
        
        return [
            {
                "schedule_id": s["id"],
                "frequency": s["frequency"],
                "report_type": s["report_type"],
                "language": s["language"],
                "recipients": s["recipients"],
                "delivery_methods": s["delivery_methods"],
                "enabled": s["enabled"],
                "last_run": s.get("last_run"),
                "next_run": s.get("next_run"),
                "created_at": s["created_at"],
                "created_by": s["created_by"]
            }
            for s in schedules
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/schedules/{schedule_id}")
async def update_scheduled_report(
    schedule_id: str,
    request: ScheduledReportRequest,
    current_user=Depends(get_current_user),
    report_repository=Depends(get_report_repository)
):
    """
    Update a scheduled report configuration.
    """
    try:
        # Get existing schedule
        schedule = await report_repository.get_schedule(schedule_id)
        
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        # Check permissions
        if not await _check_schedule_access(current_user, schedule):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update schedule
        await report_repository.update_schedule(
            schedule_id,
            {
                "frequency": request.frequency,
                "day_of_week": request.day_of_week,
                "day_of_month": request.day_of_month,
                "time": request.time,
                "timezone": request.timezone,
                "report_type": request.report_type,
                "language": request.language,
                "recipients": request.recipients,
                "delivery_methods": request.delivery_methods,
                "enabled": request.enabled,
                "updated_by": current_user.id,
                "updated_at": datetime.now()
            }
        )
        
        return {"status": "updated", "message": "Schedule updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/schedules/{schedule_id}")
async def delete_scheduled_report(
    schedule_id: str,
    current_user=Depends(get_current_user),
    report_repository=Depends(get_report_repository)
):
    """
    Delete a scheduled report.
    """
    try:
        # Get existing schedule
        schedule = await report_repository.get_schedule(schedule_id)
        
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        # Check permissions
        if not await _check_schedule_access(current_user, schedule):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete schedule
        await report_repository.delete_schedule(schedule_id)
        
        return {"status": "deleted", "message": "Schedule deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
async def _check_report_access(user, report) -> bool:
    """Check if user has access to report."""
    # TODO: Implement proper access control
    return True


async def _check_schedule_access(user, schedule) -> bool:
    """Check if user has access to schedule."""
    # TODO: Implement proper access control
    return True


def _report_to_summary(report) -> dict:
    """Convert report to summary for history."""
    return {
        "report_id": report.id,
        "organization_id": report.organization_id,
        "report_type": report.report_type,
        "language": report.language.value,
        "time_range": {
            "start_date": report.time_range.start_date.isoformat(),
            "end_date": report.time_range.end_date.isoformat()
        },
        "generated_at": report.generated_at.isoformat(),
        "generation_time_ms": report.generation_time_ms,
        "key_metrics": {
            "quality_score": report.business_metrics.overall_quality_score,
            "technical_debt": float(report.business_metrics.technical_debt.estimated_cost),
            "security_risk": report.business_metrics.security_metrics.overall_risk_level.value,
            "team_productivity": report.business_metrics.team_productivity.overall_score
        },
        "has_pdf": report.pdf_document is not None,
        "formats_available": list(report.additional_formats.keys())
    }
