"""Endpoints de health checks."""

import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, status, HTTPException
from pydantic import BaseModel

from codeant_agent.infrastructure.database.connection_pool import DatabaseConnectionPool


class ServiceHealth(BaseModel):
    """Estado de salud de un servicio."""
    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    response_time_ms: float
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Respuesta de health check."""
    status: str  # "healthy", "unhealthy", "degraded"
    timestamp: datetime
    version: str
    uptime_seconds: int
    services: list[ServiceHealth]


class ReadinessResponse(BaseModel):
    """Respuesta de readiness check."""
    ready: bool
    services: list[ServiceHealth]


class LivenessResponse(BaseModel):
    """Respuesta de liveness check."""
    alive: bool
    timestamp: datetime


class HealthChecker:
    """Verificador de salud de servicios."""
    
    def __init__(self, database_pool: Optional[DatabaseConnectionPool] = None):
        """Inicializar el verificador."""
        self.database_pool = database_pool
        self.start_time = time.time()
    
    async def check_database_health(self) -> ServiceHealth:
        """Verificar salud de la base de datos."""
        start_time = time.time()
        
        try:
            if not self.database_pool:
                return ServiceHealth(
                    name="database",
                    status="unhealthy",
                    response_time_ms=0,
                    details={"error": "Database pool not initialized"}
                )
            
            # Intentar obtener estadísticas del pool
            stats_result = await self.database_pool.get_pool_stats()
            
            if stats_result.is_failure():
                return ServiceHealth(
                    name="database",
                    status="unhealthy",
                    response_time_ms=(time.time() - start_time) * 1000,
                    details={"error": str(stats_result.error)}
                )
            
            stats = stats_result.data
            response_time = (time.time() - start_time) * 1000
            
            # Determinar estado basado en las estadísticas
            status_str = "healthy"
            if stats["checked_out"] >= stats["pool_size"] * 0.9:
                status_str = "degraded"  # Alto uso del pool
            
            return ServiceHealth(
                name="database",
                status=status_str,
                response_time_ms=response_time,
                details={
                    "pool_size": stats["pool_size"],
                    "checked_out": stats["checked_out"],
                    "checked_in": stats["checked_in"],
                    "overflow": stats["overflow"],
                    "invalid": stats["invalid"]
                }
            )
            
        except Exception as e:
            return ServiceHealth(
                name="database",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
    
    async def check_cache_health(self) -> ServiceHealth:
        """Verificar salud del cache (Redis)."""
        start_time = time.time()
        
        try:
            # TODO: Implementar verificación de Redis cuando esté disponible
            return ServiceHealth(
                name="cache",
                status="healthy",
                response_time_ms=(time.time() - start_time) * 1000,
                details={"note": "Cache verification not implemented yet"}
            )
            
        except Exception as e:
            return ServiceHealth(
                name="cache",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
    
    async def check_overall_health(self) -> HealthResponse:
        """Verificar salud general del sistema."""
        # Verificar todos los servicios
        database_health = await self.check_database_health()
        cache_health = await self.check_cache_health()
        
        services = [database_health, cache_health]
        
        # Determinar estado general
        unhealthy_services = [s for s in services if s.status == "unhealthy"]
        degraded_services = [s for s in services if s.status == "degraded"]
        
        if unhealthy_services:
            overall_status = "unhealthy"
        elif degraded_services:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime_seconds=int(time.time() - self.start_time),
            services=services
        )
    
    async def check_readiness(self) -> ReadinessResponse:
        """Verificar si la aplicación está lista para recibir tráfico."""
        # Verificar servicios críticos
        database_health = await self.check_database_health()
        
        # La aplicación está lista si la base de datos está disponible
        ready = database_health.status != "unhealthy"
        
        return ReadinessResponse(
            ready=ready,
            services=[database_health]
        )
    
    async def check_liveness(self) -> LivenessResponse:
        """Verificar si la aplicación está viva."""
        # Liveness simple - si podemos responder, estamos vivos
        return LivenessResponse(
            alive=True,
            timestamp=datetime.utcnow()
        )


def create_health_routes(database_pool: Optional[DatabaseConnectionPool] = None) -> APIRouter:
    """Crear las rutas de health checks."""
    router = APIRouter()
    health_checker = HealthChecker(database_pool)
    
    @router.get(
        "/",
        response_model=HealthResponse,
        summary="Health Check",
        description="Verificar el estado de salud general de la aplicación y sus servicios"
    )
    async def health_check():
        """Health check general."""
        health = await health_checker.check_overall_health()
        
        # Retornar código de estado apropiado
        if health.status == "unhealthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=health.dict()
            )
        elif health.status == "degraded":
            raise HTTPException(
                status_code=status.HTTP_200_OK,  # OK pero con advertencia
                detail=health.dict()
            )
        
        return health
    
    @router.get(
        "/ready",
        response_model=ReadinessResponse,
        summary="Readiness Check",
        description="Verificar si la aplicación está lista para recibir tráfico"
    )
    async def readiness_check():
        """Readiness check para Kubernetes."""
        readiness = await health_checker.check_readiness()
        
        if not readiness.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=readiness.dict()
            )
        
        return readiness
    
    @router.get(
        "/live",
        response_model=LivenessResponse,
        summary="Liveness Check",
        description="Verificar si la aplicación está viva"
    )
    async def liveness_check():
        """Liveness check para Kubernetes."""
        return await health_checker.check_liveness()
    
    @router.get(
        "/version",
        summary="Version Info",
        description="Obtener información de versión de la aplicación"
    )
    async def version_info():
        """Información de versión."""
        return {
            "version": "1.0.0",
            "build_date": "2024-01-01",
            "git_commit": "unknown",  # TODO: Obtener del build
            "environment": "development"  # TODO: Obtener de configuración
        }
    
    @router.get(
        "/metrics",
        summary="Basic Metrics",
        description="Métricas básicas de la aplicación"
    )
    async def basic_metrics():
        """Métricas básicas."""
        uptime = time.time() - health_checker.start_time
        
        # TODO: Agregar métricas más detalladas cuando tengamos Prometheus
        return {
            "uptime_seconds": int(uptime),
            "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            "memory_usage": "unknown",  # TODO: Implementar
            "cpu_usage": "unknown",     # TODO: Implementar
            "active_connections": "unknown"  # TODO: Implementar
        }
    
    return router
