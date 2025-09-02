"""
Casos de uso para análisis de flujo (Fase 24).
"""

from dataclasses import dataclass

from ..dtos.flow_dtos import (
    BuildCFGRequest, BuildCFGResponse,
    DataFlowRequest, DataFlowResponse,
    TaintAnalysisRequest, TaintAnalysisResponse,
)
from ..ports.flow_ports import CFGBuilderPort, DataFlowAnalyzerPort, TaintAnalyzerPort


@dataclass
class FlowContext:
    cfg_builder: CFGBuilderPort
    dataflow: DataFlowAnalyzerPort
    taint: TaintAnalyzerPort


class BuildCFGUseCase:
    def __init__(self, ctx: FlowContext) -> None:
        self.ctx = ctx

    async def execute(self, request: BuildCFGRequest) -> BuildCFGResponse:
        result = await self.ctx.cfg_builder.build_cfg(request.project_path, request.file_path)
        return BuildCFGResponse(nodes=result["nodes"], edges=result["edges"])


class RunDataFlowUseCase:
    def __init__(self, ctx: FlowContext) -> None:
        self.ctx = ctx

    async def execute(self, request: DataFlowRequest) -> DataFlowResponse:
        result = await self.ctx.dataflow.analyze(request.project_path, request.file_path)
        return DataFlowResponse(variables=result["variables"], dependencies=result["dependencies"])


class RunTaintAnalysisUseCase:
    def __init__(self, ctx: FlowContext) -> None:
        self.ctx = ctx

    async def execute(self, request: TaintAnalysisRequest) -> TaintAnalysisResponse:
        result = await self.ctx.taint.analyze(request.project_path, request.file_path)
        return TaintAnalysisResponse(issues=result["issues"])


