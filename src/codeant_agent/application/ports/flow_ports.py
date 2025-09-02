"""
Puertos para análisis de flujo (Fase 24).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class CFGBuilderPort(ABC):
    @abstractmethod
    async def build_cfg(self, project_path: str, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError


class DataFlowAnalyzerPort(ABC):
    @abstractmethod
    async def analyze(self, project_path: str, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError


class TaintAnalyzerPort(ABC):
    @abstractmethod
    async def analyze(self, project_path: str, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError


