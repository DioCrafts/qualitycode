"""
Entidades de dominio para análisis de flujo (Fase 24): CFG, Data Flow y Taint.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class CFGNode:
    id: str
    label: str
    line: Optional[int] = None


@dataclass
class CFGEdge:
    source: str
    target: str
    edge_type: str = "sequential"  # sequential|conditional|exception


@dataclass
class ControlFlowGraph:
    nodes: List[CFGNode] = field(default_factory=list)
    edges: List[CFGEdge] = field(default_factory=list)


@dataclass
class DataFlowDependency:
    variable: str
    definition_line: int
    use_line: int
    dependency_type: str  # true|anti|output|input


@dataclass
class DataFlowResult:
    variables: List[str]
    dependencies: List[DataFlowDependency]


@dataclass
class TaintIssue:
    source_line: int
    sink_line: int
    source: str
    sink: str
    vulnerability_type: str
    severity: int


@dataclass
class TaintAnalysisResult:
    issues: List[TaintIssue]


