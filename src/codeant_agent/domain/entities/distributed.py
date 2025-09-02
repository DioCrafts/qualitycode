"""
Entidades de dominio para el sistema de procesamiento distribuido (Fase 22).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import uuid4


class JobType(str, Enum):
    FULL_ANALYSIS = "full_analysis"
    PARTIAL_ANALYSIS = "partial_analysis"
    INCREMENTAL_ANALYSIS = "incremental_analysis"
    RULE_EXECUTION = "rule_execution"
    EMBEDDING_GENERATION = "embedding_generation"
    METRICS_CALCULATION = "metrics_calculation"
    CACHE_WARMING = "cache_warming"


class JobPriority(int, Enum):
    BACKGROUND = 1
    LOW = 2
    NORMAL = 3
    HIGH = 4
    CRITICAL = 5


class JobStatus(str, Enum):
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass(frozen=True)
class ResourceRequirements:
    cpu_cores: int
    memory_mb: int
    gpu_required: bool = False


@dataclass
class JobProgress:
    files_processed: int = 0
    total_files: int = 0
    rules_executed: int = 0
    total_rules: int = 0
    percentage_complete: float = 0.0
    eta_seconds: Optional[int] = None


@dataclass
class Job:
    id: str
    job_type: JobType
    files: List[str]
    rules: List[str]
    priority: JobPriority
    resource_requirements: ResourceRequirements
    dependencies: List[str] = field(default_factory=list)
    parent_job_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: JobStatus = JobStatus.QUEUED
    assigned_worker: Optional[str] = None
    progress: JobProgress = field(default_factory=JobProgress)

    @staticmethod
    def new(
        job_type: JobType,
        files: List[str],
        rules: List[str],
        priority: JobPriority,
        resource_requirements: ResourceRequirements,
        parent_job_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ) -> "Job":
        return Job(
            id=str(uuid4()),
            job_type=job_type,
            files=files,
            rules=rules,
            priority=priority,
            resource_requirements=resource_requirements,
            dependencies=dependencies or [],
            parent_job_id=parent_job_id,
        )


class WorkerHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class WorkerSpecialization(str, Enum):
    PARSING = "parsing"
    RULE_EXECUTION = "rule_execution"
    AI_ANALYSIS = "ai_analysis"
    EMBEDDING_GENERATION = "embedding_generation"
    METRICS_CALCULATION = "metrics_calculation"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    GENERAL_PURPOSE = "general_purpose"


@dataclass
class WorkerPerformanceStats:
    jobs_completed: int = 0
    jobs_failed: int = 0
    average_job_duration_ms: float = 0.0
    throughput_jobs_per_hour: float = 0.0
    cpu_utilization_avg: float = 0.0
    memory_utilization_avg: float = 0.0
    error_rate: float = 0.0


@dataclass
class Worker:
    id: str
    specializations: List[WorkerSpecialization]
    total_cpu_cores: int
    available_cpu_cores: int
    total_memory_mb: int
    available_memory_mb: int
    has_gpu: bool
    current_load: float
    health_status: WorkerHealthStatus
    assigned_jobs: List[str] = field(default_factory=list)
    performance_stats: WorkerPerformanceStats = field(default_factory=WorkerPerformanceStats)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)

    def is_suitable_for(self, job: Job) -> bool:
        if self.health_status != WorkerHealthStatus.HEALTHY:
            return False
        if self.available_cpu_cores < job.resource_requirements.cpu_cores:
            return False
        if self.available_memory_mb < job.resource_requirements.memory_mb:
            return False
        if job.resource_requirements.gpu_required and not self.has_gpu:
            return False
        return True


