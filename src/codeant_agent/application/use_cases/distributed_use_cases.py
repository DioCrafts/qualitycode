"""
Casos de uso para el sistema de procesamiento distribuido.
"""

from typing import Optional
from dataclasses import dataclass

from ..dtos.distributed_dtos import (
    ScheduleJobRequest,
    ScheduleJobResponse,
    RegisterWorkerRequest,
    RegisterWorkerResponse,
    JobStatusResponse,
    DistributedMetricsResponse,
)
from ..ports.distributed_ports import JobQueuePort, WorkerRegistryPort, LoadBalancerPort
from ...domain.entities.distributed import (
    Job,
    JobType,
    JobPriority,
    ResourceRequirements,
    Worker,
    WorkerHealthStatus,
    WorkerSpecialization,
)


@dataclass
class DistributedContext:
    job_queue: JobQueuePort
    worker_registry: WorkerRegistryPort
    load_balancer: LoadBalancerPort


class ScheduleJobUseCase:
    def __init__(self, ctx: DistributedContext) -> None:
        self.ctx = ctx

    async def execute(self, request: ScheduleJobRequest) -> ScheduleJobResponse:
        job = Job.new(
            job_type=JobType(request.job_type),
            files=request.files,
            rules=request.rules,
            priority=JobPriority(request.priority),
            resource_requirements=ResourceRequirements(
                cpu_cores=request.resource_cpu_cores,
                memory_mb=request.resource_memory_mb,
                gpu_required=request.gpu_required,
            ),
        )
        await self.ctx.job_queue.enqueue(job)
        return ScheduleJobResponse(job_id=job.id, status="queued")


class RegisterWorkerUseCase:
    def __init__(self, ctx: DistributedContext) -> None:
        self.ctx = ctx

    async def execute(self, request: RegisterWorkerRequest) -> RegisterWorkerResponse:
        worker = Worker(
            id=request.worker_id,
            specializations=[WorkerSpecialization.GENERAL_PURPOSE],
            total_cpu_cores=request.cpu_cores,
            available_cpu_cores=request.cpu_cores,
            total_memory_mb=request.memory_mb,
            available_memory_mb=request.memory_mb,
            has_gpu=request.has_gpu,
            current_load=0.0,
            health_status=WorkerHealthStatus.HEALTHY,
        )
        await self.ctx.worker_registry.register_worker(worker)
        return RegisterWorkerResponse(worker_id=worker.id, status="registered")


class AssignJobUseCase:
    def __init__(self, ctx: DistributedContext) -> None:
        self.ctx = ctx

    async def execute(self) -> Optional[JobStatusResponse]:
        job = await self.ctx.job_queue.dequeue()
        if not job:
            return None
        workers = await self.ctx.worker_registry.list_workers()
        worker = await self.ctx.load_balancer.select_worker(job, workers)
        if not worker:
            # Re-enqueue if no worker available
            await self.ctx.job_queue.enqueue(job)
            return None
        job.status = "running"  # type: ignore
        job.assigned_worker = worker.id
        # Update worker resources simplificado
        worker.available_cpu_cores -= job.resource_requirements.cpu_cores
        worker.available_memory_mb -= job.resource_requirements.memory_mb
        worker.assigned_jobs.append(job.id)
        return JobStatusResponse(job_id=job.id, status=job.status, assigned_worker=worker.id)


class DistributedMetricsUseCase:
    def __init__(self, ctx: DistributedContext) -> None:
        self.ctx = ctx

    async def execute(self) -> DistributedMetricsResponse:
        queued = await self.ctx.job_queue.size()
        workers = await self.ctx.worker_registry.list_workers()
        running = sum(len(w.assigned_jobs) for w in workers)
        return DistributedMetricsResponse(
            active_workers=len(workers),
            queued_jobs=queued,
            running_jobs=running,
            completed_jobs=0,
            failed_jobs=0,
            average_queue_time_ms=0.0,
            average_job_duration_ms=0.0,
            load_distribution={w.id: len(w.assigned_jobs) for w in workers},
        )


