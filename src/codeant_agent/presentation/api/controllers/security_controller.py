"""
Controlador REST para análisis de seguridad (Fase 23).
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List

from codeant_agent.application.use_cases.security_use_cases import (
    SecurityContext,
    RunSecurityScanUseCase,
    RunComplianceCheckUseCase,
)
from codeant_agent.application.dtos.security_dtos import (
    SecurityScanRequest,
    ComplianceCheckRequest,
)
from codeant_agent.infrastructure.security.static_scanner import SimpleStaticSecurityScanner
from codeant_agent.infrastructure.security.compliance_checker import SimpleComplianceChecker


router = APIRouter(prefix="/security", tags=["security"])

scanner = SimpleStaticSecurityScanner()
compliance = SimpleComplianceChecker()
ctx = SecurityContext(scanner=scanner, compliance=compliance)


class SecurityScanBody(BaseModel):
    project_path: str = Field(...)
    files: List[str] = Field(default_factory=list)
    enable_static: bool = True
    enable_dynamic: bool = False
    min_severity: int = 2


class ComplianceBody(BaseModel):
    framework: str
    vulnerabilities: List[dict]


@router.post("/scan")
async def run_scan(body: SecurityScanBody):
    result = await RunSecurityScanUseCase(ctx).execute(SecurityScanRequest(**body.dict()))
    return result.__dict__


@router.post("/compliance")
async def run_compliance(body: ComplianceBody):
    result = await RunComplianceCheckUseCase(ctx).execute(ComplianceCheckRequest(**body.dict()))
    return result.__dict__


