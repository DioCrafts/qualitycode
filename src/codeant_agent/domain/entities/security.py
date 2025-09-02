"""
Entidades de dominio para el sistema de seguridad (Fase 23).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional
from uuid import uuid4


class SecuritySeverity(int, Enum):
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1


@dataclass
class Vulnerability:
    id: str
    category: str
    title: str
    description: str
    severity: SecuritySeverity
    file_path: str
    line: Optional[int] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    evidence: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def new(category: str, title: str, description: str, severity: SecuritySeverity,
            file_path: str, line: Optional[int] = None, evidence: Optional[List[str]] = None,
            cwe_id: Optional[str] = None, cvss_score: Optional[float] = None) -> "Vulnerability":
        return Vulnerability(
            id=str(uuid4()),
            category=category,
            title=title,
            description=description,
            severity=severity,
            file_path=file_path,
            line=line,
            cwe_id=cwe_id,
            cvss_score=cvss_score,
            evidence=evidence or [],
        )


class ComplianceLevel(str, Enum):
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"


@dataclass
class ComplianceStatus:
    framework: str
    overall_status: ComplianceLevel
    score: float
    violations: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)


