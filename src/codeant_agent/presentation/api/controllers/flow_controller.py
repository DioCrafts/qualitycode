"""
Controlador REST para análisis de flujo (Fase 24).
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from codeant_agent.application.use_cases.flow_use_cases import (
    FlowContext,
    BuildCFGUseCase,
    RunDataFlowUseCase,
    RunTaintAnalysisUseCase,
)
from codeant_agent.application.dtos.flow_dtos import (
    BuildCFGRequest,
    DataFlowRequest,
    TaintAnalysisRequest,
)
from codeant_agent.infrastructure.flow.cfg_builder import SimpleCFGBuilder
from codeant_agent.infrastructure.flow.dataflow_analyzer import SimpleDataFlowAnalyzer
from codeant_agent.infrastructure.flow.taint_analyzer import SimpleTaintAnalyzer


router = APIRouter(prefix="/flow", tags=["flow"])

ctx = FlowContext(
    cfg_builder=SimpleCFGBuilder(),
    dataflow=SimpleDataFlowAnalyzer(),
    taint=SimpleTaintAnalyzer(),
)


class CFGBody(BaseModel):
    project_path: str = Field(...)
    file_path: str = Field(...)


@router.post("/cfg")
async def build_cfg(body: CFGBody):
    res = await BuildCFGUseCase(ctx).execute(BuildCFGRequest(**body.dict()))
    return res.__dict__


@router.post("/dataflow")
async def run_dataflow(body: CFGBody):
    res = await RunDataFlowUseCase(ctx).execute(DataFlowRequest(**body.dict()))
    return res.__dict__


@router.post("/taint")
async def run_taint(body: CFGBody):
    res = await RunTaintAnalysisUseCase(ctx).execute(TaintAnalysisRequest(**body.dict()))
    return res.__dict__


