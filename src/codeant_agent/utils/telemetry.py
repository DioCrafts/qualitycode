"""
Sistema de métricas y observabilidad para CodeAnt Agent.

Este módulo implementa:
- Métricas Prometheus
- Health checks
- Trazabilidad de requests
- Métricas de negocio
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import time
from contextlib import contextmanager
from functools import wraps

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    multiprocess,
    start_http_server
)

from .logging import get_logger
from ..config.settings import get_settings


class MetricsRegistry:
    """Registro centralizado de métricas Prometheus."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Inicializa todas las métricas del sistema."""
        
        # Métricas HTTP
        self.metrics['http_requests_total'] = Counter(
            'http_requests_total',
            'Total de requests HTTP',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.metrics['http_request_duration_seconds'] = Histogram(
            'http_request_duration_seconds',
            'Duración de requests HTTP',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Métricas de base de datos
        self.metrics['db_connections_active'] = Gauge(
            'db_connections_active',
            'Conexiones activas a la base de datos',
            ['database'],
            registry=self.registry
        )
        
        self.metrics['db_query_duration_seconds'] = Histogram(
            'db_query_duration_seconds',
            'Duración de queries de base de datos',
            ['database', 'query_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            registry=self.registry
        )
        
        # Métricas de análisis de código
        self.metrics['code_analysis_total'] = Counter(
            'code_analysis_total',
            'Total de análisis de código realizados',
            ['language', 'analysis_type', 'status'],
            registry=self.registry
        )
        
        self.metrics['code_analysis_duration_seconds'] = Histogram(
            'code_analysis_duration_seconds',
            'Duración de análisis de código',
            ['language', 'analysis_type'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        # Métricas de memoria y CPU
        self.metrics['memory_usage_bytes'] = Gauge(
            'memory_usage_bytes',
            'Uso de memoria en bytes',
            ['type'],
            registry=self.registry
        )
        
        self.metrics['cpu_usage_percent'] = Gauge(
            'cpu_usage_percent',
            'Uso de CPU en porcentaje',
            registry=self.registry
        )
        
        # Métricas de cache
        self.metrics['cache_hits_total'] = Counter(
            'cache_hits_total',
            'Total de hits en cache',
            ['cache_type'],
            registry=self.registry
        )
        
        self.metrics['cache_misses_total'] = Counter(
            'cache_misses_total',
            'Total de misses en cache',
            ['cache_type'],
            registry=self.registry
        )
        
        # Métricas de errores
        self.metrics['errors_total'] = Counter(
            'errors_total',
            'Total de errores',
            ['error_type', 'severity'],
            registry=self.registry
        )
        
        # Métricas de negocio
        self.metrics['projects_analyzed_total'] = Counter(
            'projects_analyzed_total',
            'Total de proyectos analizados',
            ['project_type', 'status'],
            registry=self.registry
        )
        
        self.metrics['issues_found_total'] = Counter(
            'issues_found_total',
            'Total de issues encontrados',
            ['issue_type', 'severity', 'language'],
            registry=self.registry
        )
    
    def get_metric(self, name: str):
        """Obtiene una métrica por nombre."""
        return self.metrics.get(name)
    
    def get_metrics_text(self) -> str:
        """Retorna las métricas en formato texto para Prometheus."""
        return generate_latest(self.registry)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Retorna las métricas como diccionario para debugging."""
        metrics_data = {}
        for name, metric in self.metrics.items():
            if hasattr(metric, '_value'):
                metrics_data[name] = metric._value.get()
            elif hasattr(metric, '_sum'):
                metrics_data[name] = {
                    'sum': metric._sum.get(),
                    'count': metric._count.get()
                }
        return metrics_data


class HealthChecker:
    """Sistema de health checks para la aplicación."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.logger = get_logger("health")
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """
        Registra una función de health check.
        
        Args:
            name: Nombre del health check
            check_func: Función que retorna True si está saludable
        """
        self.checks[name] = check_func
        self.logger.info(f"Health check registrado: {name}")
    
    async def run_checks(self) -> Dict[str, Dict[str, Any]]:
        """
        Ejecuta todos los health checks registrados.
        
        Returns:
            Diccionario con el estado de cada check
        """
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(check_func):
                    is_healthy = await check_func()
                else:
                    is_healthy = check_func()
                
                duration = time.time() - start_time
                
                results[name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'duration_ms': round(duration * 1000, 2),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                if is_healthy:
                    self.logger.debug(f"Health check {name} pasó", duration_ms=duration * 1000)
                else:
                    self.logger.warning(f"Health check {name} falló")
                    
            except Exception as e:
                self.logger.exception(f"Error en health check {name}")
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'duration_ms': 0,
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        return results
    
    def get_overall_status(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Determina el estado general basado en los resultados."""
        if not results:
            return 'unknown'
        
        statuses = [result['status'] for result in results.values()]
        
        if 'error' in statuses:
            return 'error'
        elif 'unhealthy' in statuses:
            return 'unhealthy'
        elif all(status == 'healthy' for status in statuses):
            return 'healthy'
        else:
            return 'degraded'


class TelemetryMiddleware:
    """Middleware para capturar métricas automáticamente."""
    
    def __init__(self, app, metrics_registry: MetricsRegistry):
        self.app = app
        self.metrics = metrics_registry
        self.logger = get_logger("telemetry")
    
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        method = scope.get('method', 'UNKNOWN')
        path = scope.get('path', 'UNKNOWN')
        
        # Capturar request
        self.metrics.metrics['http_requests_total'].labels(
            method=method,
            endpoint=path,
            status_code='unknown'
        ).inc()
        
        # Wrapper para capturar la respuesta
        async def send_wrapper(message):
            if message['type'] == 'http.response.start':
                status_code = str(message.get('status', 500))
                self.metrics.metrics['http_requests_total'].labels(
                    method=method,
                    endpoint=path,
                    status_code=status_code
                ).inc()
            
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            # Capturar errores
            self.metrics.metrics['errors_total'].labels(
                error_type='http_request',
                severity='high'
            ).inc()
            raise
        finally:
            # Capturar duración
            duration = time.time() - start_time
            self.metrics.metrics['http_request_duration_seconds'].labels(
                method=method,
                endpoint=path
            ).observe(duration)


class PerformanceMonitor:
    """Monitor de performance para operaciones específicas."""
    
    def __init__(self, metrics_registry: MetricsRegistry):
        self.metrics = metrics_registry
        self.logger = get_logger("performance")
    
    @contextmanager
    def measure_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager para medir la duración de una operación.
        
        Args:
            operation_name: Nombre de la operación
            labels: Labels adicionales para la métrica
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.logger.debug(
                f"Operación {operation_name} completada",
                duration_seconds=duration,
                labels=labels
            )
    
    def time_function(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Decorator para medir el tiempo de ejecución de una función.
        
        Args:
            metric_name: Nombre de la métrica de duración
            labels: Labels adicionales para la métrica
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    metric = self.metrics.get_metric(metric_name)
                    if metric and labels:
                        metric.labels(**labels).observe(duration)
                    elif metric:
                        metric.observe(duration)
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    metric = self.metrics.get_metric(metric_name)
                    if metric and labels:
                        metric.labels(**labels).observe(duration)
                    elif metric:
                        metric.observe(duration)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return wrapper
        
        return decorator


# Instancias globales
metrics_registry = MetricsRegistry()
health_checker = HealthChecker()
performance_monitor = PerformanceMonitor(metrics_registry)


def get_metrics_registry() -> MetricsRegistry:
    """Retorna el registro de métricas global."""
    return metrics_registry


def get_health_checker() -> HealthChecker:
    """Retorna el health checker global."""
    return health_checker


def get_performance_monitor() -> PerformanceMonitor:
    """Retorna el monitor de performance global."""
    return performance_monitor


# Health checks básicos
def basic_health_check() -> bool:
    """Health check básico que siempre retorna True."""
    return True


def memory_health_check() -> bool:
    """Health check de memoria."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent < 90  # Considerar saludable si usa menos del 90%
    except ImportError:
        # Si no hay psutil, asumir que está bien
        return True


def register_basic_health_checks():
    """Registra los health checks básicos del sistema."""
    health_checker.register_check("basic", basic_health_check)
    health_checker.register_check("memory", memory_health_check)


# Inicializar health checks básicos
register_basic_health_checks()


# Import asyncio para async/await
import asyncio
