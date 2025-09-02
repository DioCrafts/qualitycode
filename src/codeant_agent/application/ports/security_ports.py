"""
Puertos para el sistema de seguridad (Fase 23).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class StaticSecurityScannerPort(ABC):
    @abstractmethod
    async def scan_files(self, project_path: str, files: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class ComplianceCheckerPort(ABC):
    @abstractmethod
    async def check(self, framework: str, vulnerabilities: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError


