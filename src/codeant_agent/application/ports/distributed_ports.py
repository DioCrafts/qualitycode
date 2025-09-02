"""
Puertos (interfaces) para interacción con el mundo exterior del sistema distribuido.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ...domain.entities.distributed import Job, Worker


class JobQueuePort(ABC):
    @abstractmethod
    async def enqueue(self, job: Job) -> None:
        raise NotImplementedError

    @abstractmethod
    async def dequeue(self) -> Optional[Job]:
        raise NotImplementedError

    @abstractmethod
    async def size(self) -> int:
        raise NotImplementedError


class WorkerRegistryPort(ABC):
    @abstractmethod
    async def register_worker(self, worker: Worker) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_workers(self) -> List[Worker]:
        raise NotImplementedError

    @abstractmethod
    async def get(self, worker_id: str) -> Optional[Worker]:
        raise NotImplementedError


class LoadBalancerPort(ABC):
    @abstractmethod
    async def select_worker(self, job: Job, workers: List[Worker]) -> Optional[Worker]:
        raise NotImplementedError


class ResultAggregatorPort(ABC):
    @abstractmethod
    async def aggregate(self, partial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError


