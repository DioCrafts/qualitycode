"""
DTOs para el sistema de seguridad (Fase 23).
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class SecurityScanRequest:
    project_path: str
    files: List[str]
    enable_static: bool = True
    enable_dynamic: bool = False
    min_severity: int = 2  # Low+


@dataclass
class SecurityScanResponse:
    vulnerabilities: List[Dict[str, Any]]
    total: int
    critical: int
    high: int
    medium: int
    low: int
    info: int


@dataclass
class ComplianceCheckRequest:
    framework: str  # e.g., "OWASP"
    vulnerabilities: List[Dict[str, Any]]


@dataclass
class ComplianceCheckResponse:
    framework: str
    overall_status: str
    score: float
    violations: List[str]


