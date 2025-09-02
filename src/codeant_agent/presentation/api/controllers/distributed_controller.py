"""
Controlador REST para el sistema de procesamiento distribuido (Fase 22).
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

from codeant_agent.application.use_cases.distributed_use_cases import (
    DistributedContext,
    ScheduleJobUseCase,
    RegisterWorkerUseCase,
    AssignJobUseCase,
    DistributedMetricsUseCase,
)
from codeant_agent.application.dtos.distributed_dtos import (
    ScheduleJobRequest,
    RegisterWorkerRequest,
)
from codeant_agent.infrastructure.distributed.in_memory_adapters import (
    InMemoryJobQueue,
    InMemoryWorkerRegistry,
    RoundRobinLoadBalancer,
)


router = APIRouter(prefix="/distributed", tags=["distributed"])


# Instancias simples para MVP (en memoria)
job_queue = InMemoryJobQueue()
worker_registry = InMemoryWorkerRegistry()
load_balancer = RoundRobinLoadBalancer()
ctx = DistributedContext(job_queue=job_queue, worker_registry=worker_registry, load_balancer=load_balancer)


class ScheduleJobBody(BaseModel):
    project_path: str = Field(...)
    files: List[str] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)
    priority: int = 3
    job_type: str = "full_analysis"
    resource_cpu_cores: int = 2
    resource_memory_mb: int = 2048
    gpu_required: bool = False


class RegisterWorkerBody(BaseModel):
    worker_id: str
    specializations: Optional[List[str]] = None
    cpu_cores: int
    memory_mb: int
    has_gpu: bool = False


@router.post("/jobs/schedule")
async def schedule_job(body: ScheduleJobBody):
    use_case = ScheduleJobUseCase(ctx)
    result = await use_case.execute(ScheduleJobRequest(**body.dict()))
    return result.__dict__


@router.post("/workers/register")
async def register_worker(body: RegisterWorkerBody):
    use_case = RegisterWorkerUseCase(ctx)
    result = await use_case.execute(RegisterWorkerRequest(**body.dict()))
    return result.__dict__


@router.post("/jobs/assign")
async def assign_job():
    use_case = AssignJobUseCase(ctx)
    result = await use_case.execute()
    return result.__dict__ if result else {"status": "no_job_available"}


@router.get("/metrics")
async def metrics():
    use_case = DistributedMetricsUseCase(ctx)
    result = await use_case.execute()
    return result.__dict__


