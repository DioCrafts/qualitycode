"""
DTOs para análisis de flujo (Fase 24).
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class BuildCFGRequest:
    project_path: str
    file_path: str


@dataclass
class BuildCFGResponse:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


@dataclass
class DataFlowRequest:
    project_path: str
    file_path: str


@dataclass
class DataFlowResponse:
    variables: List[str]
    dependencies: List[Dict[str, Any]]


@dataclass
class TaintAnalysisRequest:
    project_path: str
    file_path: str


@dataclass
class TaintAnalysisResponse:
    issues: List[Dict[str, Any]]


