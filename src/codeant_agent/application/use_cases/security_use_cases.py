"""
Casos de uso para el sistema de seguridad (Fase 23).
"""

from dataclasses import dataclass
from typing import List

from ..dtos.security_dtos import (
    SecurityScanRequest,
    SecurityScanResponse,
    ComplianceCheckRequest,
    ComplianceCheckResponse,
)
from ..ports.security_ports import StaticSecurityScannerPort, ComplianceCheckerPort


@dataclass
class SecurityContext:
    scanner: StaticSecurityScannerPort
    compliance: ComplianceCheckerPort


class RunSecurityScanUseCase:
    def __init__(self, ctx: SecurityContext) -> None:
        self.ctx = ctx

    async def execute(self, request: SecurityScanRequest) -> SecurityScanResponse:
        vulns = await self.ctx.scanner.scan_files(request.project_path, request.files)
        # Filtrar por severidad mínima
        filtered = [v for v in vulns if int(v.get("severity", 0)) >= request.min_severity]

        def count(level: int) -> int:
            return sum(1 for v in filtered if int(v.get("severity", 0)) == level)

        return SecurityScanResponse(
            vulnerabilities=filtered,
            total=len(filtered),
            critical=count(5),
            high=count(4),
            medium=count(3),
            low=count(2),
            info=count(1),
        )


class RunComplianceCheckUseCase:
    def __init__(self, ctx: SecurityContext) -> None:
        self.ctx = ctx

    async def execute(self, request: ComplianceCheckRequest) -> ComplianceCheckResponse:
        result = await self.ctx.compliance.check(request.framework, request.vulnerabilities)
        return ComplianceCheckResponse(
            framework=result["framework"],
            overall_status=result["overall_status"],
            score=float(result["score"]),
            violations=list(result.get("violations", [])),
        )


