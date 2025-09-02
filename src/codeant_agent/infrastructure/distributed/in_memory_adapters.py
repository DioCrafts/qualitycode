"""
Adaptadores en memoria para el sistema de procesamiento distribuido (MVP).
"""

import asyncio
from typing import List, Optional, Dict, Any
from collections import deque

from ...application.ports.distributed_ports import (
    JobQueuePort,
    WorkerRegistryPort,
    LoadBalancerPort,
)
from ...domain.entities.distributed import Job, Worker


class InMemoryJobQueue(JobQueuePort):
    def __init__(self) -> None:
        self._queue: deque[Job] = deque()
        self._lock = asyncio.Lock()

    async def enqueue(self, job: Job) -> None:
        async with self._lock:
            self._queue.append(job)

    async def dequeue(self) -> Optional[Job]:
        async with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    async def size(self) -> int:
        async with self._lock:
            return len(self._queue)


class InMemoryWorkerRegistry(WorkerRegistryPort):
    def __init__(self) -> None:
        self._workers: Dict[str, Worker] = {}
        self._lock = asyncio.Lock()

    async def register_worker(self, worker: Worker) -> None:
        async with self._lock:
            self._workers[worker.id] = worker

    async def list_workers(self) -> List[Worker]:
        async with self._lock:
            return list(self._workers.values())

    async def get(self, worker_id: str) -> Optional[Worker]:
        async with self._lock:
            return self._workers.get(worker_id)


class RoundRobinLoadBalancer(LoadBalancerPort):
    def __init__(self) -> None:
        self._index = 0
        self._lock = asyncio.Lock()

    async def select_worker(self, job: Job, workers: List[Worker]) -> Optional[Worker]:
        # Filtrar por elegibilidad básica
        eligible = [w for w in workers if w.is_suitable_for(job)]
        if not eligible:
            return None
        async with self._lock:
            worker = eligible[self._index % len(eligible)]
            self._index += 1
            return worker


