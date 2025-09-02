"""
Sistema de health checks avanzado.

Este módulo implementa:
- Health checks para todas las dependencias
- Health checks de negocio
- Health checks de sistema
- Liveness y readiness probes
- Health check endpoints
"""

import asyncio
import time
import psutil
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ...config.settings import get_settings
from ...utils.logging import get_logger
from ...infrastructure.database.connection_pool import DatabaseConnectionPool

# Importaciones para health checks
try:
    import redis.asyncio as redis  # type: ignore
except ImportError:
    redis = None

try:
    import httpx  # type: ignore
except ImportError:
    httpx = None


class HealthStatus(Enum):
    """Estados de salud del sistema."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Resultado de un health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


@dataclass
class HealthReport:
    """Reporte completo de salud del sistema."""
    overall_status: HealthStatus
    checks: Dict[str, HealthCheckResult] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: str = ""
    environment: str = ""


class HealthService:
    """Servicio de health checks avanzado."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.checks: Dict[str, Callable] = {}
        self._last_check_time: Optional[datetime] = None
        self._cache_duration = timedelta(seconds=30)  # Cache por 30 segundos
        self._cached_report: Optional[HealthReport] = None
    
    def register_check(self, name: str, check_func: Callable, critical: bool = True) -> None:
        """
        Registra una función de health check.
        
        Args:
            name: Nombre del health check
            check_func: Función que retorna HealthCheckResult o bool
            critical: Si es True, el check es crítico para la salud general
        """
        self.checks[name] = check_func
        self.logger.info(f"Health check registrado: {name} (critical: {critical})")
    
    async def run_checks(self, force_refresh: bool = False) -> HealthReport:
        """
        Ejecuta todos los health checks registrados.
        
        Args:
            force_refresh: Si es True, ignora el cache
            
        Returns:
            Reporte completo de salud del sistema
        """
        # Verificar cache
        if not force_refresh and self._cached_report and self._last_check_time:
            if datetime.utcnow() - self._last_check_time < self._cache_duration:
                return self._cached_report
        
        results = {}
        start_time = time.time()
        
        # Ejecutar checks en paralelo
        tasks = []
        for name, check_func in self.checks.items():
            task = asyncio.create_task(self._run_single_check(name, check_func))
            tasks.append(task)
        
        # Esperar todos los checks
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        for i, (name, check_func) in enumerate(self.checks.items()):
            result = check_results[i]
            if isinstance(result, Exception):
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Error ejecutando check: {str(result)}",
                    error=str(result),
                    duration_ms=0.0
                )
            else:
                results[name] = result
        
        # Determinar estado general
        overall_status = self._determine_overall_status(results)
        
        # Crear reporte
        report = HealthReport(
            overall_status=overall_status,
            checks=results,
            version=self.settings.version,
            environment=self.settings.environment
        )
        
        # Actualizar cache
        self._cached_report = report
        self._last_check_time = datetime.utcnow()
        
        total_duration = time.time() - start_time
        self.logger.info(
            f"Health checks completados en {total_duration:.2f}s",
            overall_status=overall_status.value,
            checks_count=len(results)
        )
        
        return report
    
    async def _run_single_check(self, name: str, check_func: Callable) -> HealthCheckResult:
        """Ejecuta un health check individual."""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            duration = (time.time() - start_time) * 1000
            
            # Si el resultado es un HealthCheckResult, usarlo directamente
            if isinstance(result, HealthCheckResult):
                result.duration_ms = duration
                result.timestamp = datetime.utcnow()
                return result
            
            # Si es un bool, convertirlo a HealthCheckResult
            elif isinstance(result, bool):
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="Check passed" if result else "Check failed",
                    duration_ms=duration
                )
            
            # Si es un dict, convertirlo a HealthCheckResult
            elif isinstance(result, dict):
                status = HealthStatus(result.get("status", "unknown"))
                return HealthCheckResult(
                    name=name,
                    status=status,
                    message=result.get("message", ""),
                    details=result.get("details", {}),
                    duration_ms=duration
                )
            
            else:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Resultado inesperado: {type(result)}",
                    duration_ms=duration
                )
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.logger.exception(f"Error en health check {name}")
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Error ejecutando check: {str(e)}",
                error=str(e),
                duration_ms=duration
            )
    
    def _determine_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Determina el estado general basado en los resultados."""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results.values()]
        
        # Si hay algún error crítico, el sistema está unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        
        # Si hay algún estado degradado, el sistema está degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # Si todos están healthy, el sistema está healthy
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        
        # En cualquier otro caso, unknown
        return HealthStatus.UNKNOWN
    
    def get_liveness_status(self) -> bool:
        """Retorna True si el sistema está vivo (liveness probe)."""
        if not self._cached_report:
            return True  # Si no hay reporte, asumir que está vivo
        
        # Para liveness, solo verificamos que el proceso esté corriendo
        return True
    
    def get_readiness_status(self) -> bool:
        """Retorna True si el sistema está listo para recibir tráfico (readiness probe)."""
        if not self._cached_report:
            return False  # Si no hay reporte, no está listo
        
        # Para readiness, verificamos que los checks críticos estén healthy
        critical_checks = [
            "database",
            "redis",
            "basic"
        ]
        
        for check_name in critical_checks:
            if check_name in self._cached_report.checks:
                check_result = self._cached_report.checks[check_name]
                if check_result.status != HealthStatus.HEALTHY:
                    return False
        
        return True
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Retorna un resumen de la salud del sistema."""
        if not self._cached_report:
            return {
                "status": "unknown",
                "timestamp": datetime.utcnow().isoformat(),
                "checks_count": 0
            }
        
        healthy_checks = sum(
            1 for result in self._cached_report.checks.values()
            if result.status == HealthStatus.HEALTHY
        )
        
        return {
            "status": self._cached_report.overall_status.value,
            "timestamp": self._cached_report.timestamp.isoformat(),
            "checks_count": len(self._cached_report.checks),
            "healthy_checks": healthy_checks,
            "version": self._cached_report.version,
            "environment": self._cached_report.environment
        }


# Health checks específicos
async def database_health_check() -> HealthCheckResult:
    """Health check para la base de datos."""
    try:
        # Verificar conexión a la base de datos
        pool = DatabaseConnectionPool()
        async with pool.get_connection() as conn:
            # Ejecutar query simple
            result = await conn.execute("SELECT 1")
            await result.fetchone()
        
        return HealthCheckResult(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection successful",
            details={
                "pool_size": pool.pool.size,
                "checked_out": pool.pool.checkedout()
            }
        )
    except Exception as e:
        return HealthCheckResult(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {str(e)}",
            error=str(e)
        )


async def redis_health_check() -> HealthCheckResult:
    """Health check para Redis."""
    try:
        if redis is None:
            # Si redis no está instalado, considerar como no crítico
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis client not available; skipping",
            )
        
        settings = get_settings()
        r = redis.from_url(settings.redis.url)
        
        # Verificar conexión
        await r.ping()
        
        # Verificar información del servidor
        info = await r.info()
        
        return HealthCheckResult(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Redis connection successful",
            details={
                "version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown")
            }
        )
    except Exception as e:
        return HealthCheckResult(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            message=f"Redis connection failed: {str(e)}",
            error=str(e)
        )


def system_health_check() -> HealthCheckResult:
    """Health check para el sistema."""
    try:
        # CPU usage
        cpu_percent = float(psutil.cpu_percent(interval=1))
        
        # Memory usage
        memory = psutil.virtual_memory()
        mem_percent = float(getattr(memory, 'percent', 0.0))
        mem_available = float(getattr(memory, 'available', 0))
        
        # Disk usage
        disk_usage = psutil.disk_usage('/')
        disk_percent = float(getattr(disk_usage, 'percent', 0.0))
        disk_free = float(getattr(disk_usage, 'free', 0))
        
        # Determinar estado basado en thresholds
        status = HealthStatus.HEALTHY
        issues = []
        
        if cpu_percent > 90:
            status = HealthStatus.DEGRADED
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if mem_percent > 90:
            status = HealthStatus.DEGRADED
            issues.append(f"High memory usage: {mem_percent:.1f}%")
        
        if disk_percent > 90:
            status = HealthStatus.DEGRADED
            issues.append(f"High disk usage: {disk_percent:.1f}%")
        
        message = "System healthy" if not issues else "; ".join(issues)
        
        return HealthCheckResult(
            name="system",
            status=status,
            message=message,
            details={
                "cpu_percent": cpu_percent,
                "memory_percent": mem_percent,
                "disk_percent": disk_percent,
                "memory_available_gb": mem_available / (1024**3),
                "disk_free_gb": disk_free / (1024**3)
            }
        )
    except Exception as e:
        return HealthCheckResult(
            name="system",
            status=HealthStatus.UNHEALTHY,
            message=f"System check failed: {str(e)}",
            error=str(e)
        )


def basic_health_check() -> HealthCheckResult:
    """Health check básico que siempre retorna True."""
    return HealthCheckResult(
        name="basic",
        status=HealthStatus.HEALTHY,
        message="Basic health check passed"
    )


async def external_services_health_check() -> HealthCheckResult:
    """Health check para servicios externos."""
    try:
        if httpx is None:
            return HealthCheckResult(
                name="external_services",
                status=HealthStatus.HEALTHY,
                message="httpx not available; skipping"
            )
        
        settings = get_settings()
        issues = []
        
        # Verificar servicios externos si están configurados
        services_to_check = []
        
        if settings.telemetry.elasticsearch_url:
            services_to_check.append(("elasticsearch", settings.telemetry.elasticsearch_url))
        
        if settings.telemetry.alertmanager_url:
            services_to_check.append(("alertmanager", settings.telemetry.alertmanager_url))
        
        if not services_to_check:
            return HealthCheckResult(
                name="external_services",
                status=HealthStatus.HEALTHY,
                message="No external services configured"
            )
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for service_name, url in services_to_check:
                try:
                    response = await client.get(f"{url}/health")
                    if response.status_code >= 400:
                        issues.append(f"{service_name}: HTTP {response.status_code}")
                except Exception as e:
                    issues.append(f"{service_name}: {str(e)}")
        
        if issues:
            return HealthCheckResult(
                name="external_services",
                status=HealthStatus.DEGRADED,
                message=f"External services issues: {'; '.join(issues)}",
                details={"issues": issues}
            )
        else:
            return HealthCheckResult(
                name="external_services",
                status=HealthStatus.HEALTHY,
                message="All external services healthy"
            )
            
    except Exception as e:
        return HealthCheckResult(
            name="external_services",
            status=HealthStatus.UNHEALTHY,
            message=f"External services check failed: {str(e)}",
            error=str(e)
        )


# Instancia global
_health_service: Optional[HealthService] = None


def get_health_service() -> HealthService:
    """Retorna la instancia global del servicio de health checks."""
    global _health_service
    if _health_service is None:
        _health_service = HealthService()
        
        # Registrar health checks básicos
        _health_service.register_check("basic", basic_health_check, critical=True)
        _health_service.register_check("database", database_health_check, critical=True)
        _health_service.register_check("redis", redis_health_check, critical=False)
        _health_service.register_check("system", system_health_check, critical=False)
        _health_service.register_check("external_services", external_services_health_check, critical=False)
    
    return _health_service


def initialize_health_checks() -> None:
    """Inicializa el sistema de health checks."""
    get_health_service()
