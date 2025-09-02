"""
DTOs para el sistema de procesamiento distribuido (Fase 22).
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ScheduleJobRequest:
    project_path: str
    files: List[str]
    rules: List[str]
    priority: int = 3
    job_type: str = "full_analysis"
    resource_cpu_cores: int = 2
    resource_memory_mb: int = 2048
    gpu_required: bool = False


@dataclass
class ScheduleJobResponse:
    job_id: str
    status: str


@dataclass
class RegisterWorkerRequest:
    worker_id: str
    specializations: List[str]
    cpu_cores: int
    memory_mb: int
    has_gpu: bool = False


@dataclass
class RegisterWorkerResponse:
    worker_id: str
    status: str


@dataclass
class JobStatusResponse:
    job_id: str
    status: str
    assigned_worker: Optional[str]


@dataclass
class DistributedMetricsResponse:
    active_workers: int
    queued_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    average_queue_time_ms: float
    average_job_duration_ms: float
    load_distribution: Dict[str, Any]


