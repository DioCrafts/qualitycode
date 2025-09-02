"""
Sistema de métricas con Prometheus.

Este módulo implementa:
- Métricas Prometheus personalizadas
- Métricas de negocio
- Métricas de sistema
- Exportación de métricas
- Custom collectors
"""

import asyncio
import time
import psutil
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess,
    start_http_server, REGISTRY
)
from prometheus_client.metrics_core import Metric
from prometheus_client.samples import Sample

from ...config.settings import get_settings
from ...utils.logging import get_logger


@dataclass
class MetricDefinition:
    """Definición de una métrica."""
    name: str
    help_text: str
    type: str  # counter, gauge, histogram, summary
    labels: List[str]
    buckets: Optional[List[float]] = None


class MetricsService:
    """Servicio de métricas con Prometheus."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self.custom_collectors: Dict[str, Callable] = {}
        self._initialized = False
        
        # Definir métricas del sistema
        self._metric_definitions = [
            # HTTP Metrics
            MetricDefinition(
                "http_requests_total",
                "Total de requests HTTP",
                "counter",
                ["method", "endpoint", "status_code", "version"]
            ),
            MetricDefinition(
                "http_request_duration_seconds",
                "Duración de requests HTTP",
                "histogram",
                ["method", "endpoint", "version"],
                [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            ),
            MetricDefinition(
                "http_requests_in_flight",
                "Requests HTTP en proceso",
                "gauge",
                ["version"]
            ),
            
            # Database Metrics
            MetricDefinition(
                "database_connections_active",
                "Conexiones activas a la base de datos",
                "gauge",
                ["database", "pool_name"]
            ),
            MetricDefinition(
                "database_queries_total",
                "Total de queries de base de datos",
                "counter",
                ["database", "query_type", "status"]
            ),
            MetricDefinition(
                "database_query_duration_seconds",
                "Duración de queries de base de datos",
                "histogram",
                ["database", "query_type"],
                [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
            ),
            
            # Analysis Metrics
            MetricDefinition(
                "code_analysis_total",
                "Total de análisis de código realizados",
                "counter",
                ["language", "analysis_type", "status", "project_id"]
            ),
            MetricDefinition(
                "code_analysis_duration_seconds",
                "Duración de análisis de código",
                "histogram",
                ["language", "analysis_type"],
                [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0]
            ),
            MetricDefinition(
                "code_analysis_files_processed",
                "Archivos procesados en análisis",
                "counter",
                ["language", "analysis_type", "status"]
            ),
            MetricDefinition(
                "code_analysis_issues_found",
                "Issues encontrados en análisis",
                "counter",
                ["language", "issue_type", "severity", "analysis_type"]
            ),
            
            # Business Metrics
            MetricDefinition(
                "projects_total",
                "Total de proyectos",
                "gauge",
                ["status", "visibility"]
            ),
            MetricDefinition(
                "users_active",
                "Usuarios activos",
                "gauge",
                ["role", "status"]
            ),
            MetricDefinition(
                "organizations_total",
                "Total de organizaciones",
                "gauge",
                ["plan_type", "status"]
            ),
            
            # System Metrics
            MetricDefinition(
                "system_cpu_usage_percent",
                "Uso de CPU del sistema",
                "gauge",
                ["core"]
            ),
            MetricDefinition(
                "system_memory_usage_bytes",
                "Uso de memoria del sistema",
                "gauge",
                ["type"]
            ),
            MetricDefinition(
                "system_disk_usage_bytes",
                "Uso de disco del sistema",
                "gauge",
                ["device", "mountpoint"]
            ),
            MetricDefinition(
                "system_network_bytes_total",
                "Bytes de red del sistema",
                "counter",
                ["interface", "direction"]
            ),
            
            # Cache Metrics
            MetricDefinition(
                "cache_hits_total",
                "Total de hits en cache",
                "counter",
                ["cache_type", "key_pattern"]
            ),
            MetricDefinition(
                "cache_misses_total",
                "Total de misses en cache",
                "counter",
                ["cache_type", "key_pattern"]
            ),
            MetricDefinition(
                "cache_size_bytes",
                "Tamaño del cache en bytes",
                "gauge",
                ["cache_type"]
            ),
            
            # Error Metrics
            MetricDefinition(
                "errors_total",
                "Total de errores",
                "counter",
                ["error_type", "severity", "module"]
            ),
            MetricDefinition(
                "error_rate",
                "Tasa de errores",
                "gauge",
                ["error_type", "severity"]
            ),
        ]
    
    def initialize(self) -> None:
        """Inicializa el sistema de métricas."""
        if self._initialized:
            return
        
        if not self.settings.telemetry.prometheus_enabled:
            self.logger.info("Métricas Prometheus deshabilitadas")
            return
        
        try:
            # Crear métricas basadas en definiciones
            self._create_metrics()
            
            # Registrar collectors personalizados
            self._register_custom_collectors()
            
            # Iniciar servidor HTTP para métricas
            self._start_metrics_server()
            
            # Iniciar métricas de sistema
            self._start_system_metrics_collector()
            
            self._initialized = True
            self.logger.info("Sistema de métricas inicializado correctamente")
            
        except Exception as e:
            self.logger.error(f"Error inicializando métricas: {e}")
    
    def _create_metrics(self) -> None:
        """Crea todas las métricas basadas en las definiciones."""
        for definition in self._metric_definitions:
            try:
                if definition.type == "counter":
                    metric = Counter(
                        definition.name,
                        definition.help_text,
                        definition.labels,
                        registry=self.registry
                    )
                elif definition.type == "gauge":
                    metric = Gauge(
                        definition.name,
                        definition.help_text,
                        definition.labels,
                        registry=self.registry
                    )
                elif definition.type == "histogram":
                    metric = Histogram(
                        definition.name,
                        definition.help_text,
                        definition.labels,
                        buckets=definition.buckets or [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                        registry=self.registry
                    )
                elif definition.type == "summary":
                    metric = Summary(
                        definition.name,
                        definition.help_text,
                        definition.labels,
                        registry=self.registry
                    )
                else:
                    self.logger.warning(f"Tipo de métrica no soportado: {definition.type}")
                    continue
                
                self.metrics[definition.name] = metric
                self.logger.debug(f"Métrica creada: {definition.name}")
                
            except Exception as e:
                self.logger.error(f"Error creando métrica {definition.name}: {e}")
    
    def _register_custom_collectors(self) -> None:
        """Registra collectors personalizados."""
        # Collector de información del sistema
        system_info = Info(
            "system_info",
            "Información del sistema",
            registry=self.registry
        )
        system_info.info({
            "version": self.settings.version,
            "environment": self.settings.environment,
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            "platform": psutil.sys.platform,
        })
        
        # Collector de métricas de proceso
        process_info = Info(
            "process_info",
            "Información del proceso",
            registry=self.registry
        )
        process_info.info({
            "pid": str(psutil.Process().pid),
            "start_time": str(psutil.Process().create_time()),
        })
    
    def _start_metrics_server(self) -> None:
        """Inicia el servidor HTTP para métricas."""
        try:
            start_http_server(
                self.settings.telemetry.prometheus_port,
                registry=self.registry
            )
            self.logger.info(
                f"Servidor de métricas iniciado en puerto {self.settings.telemetry.prometheus_port}"
            )
        except Exception as e:
            self.logger.error(f"Error iniciando servidor de métricas: {e}")
    
    def _start_system_metrics_collector(self) -> None:
        """Inicia el colector de métricas del sistema."""
        asyncio.create_task(self._collect_system_metrics())
    
    async def _collect_system_metrics(self) -> None:
        """Recolecta métricas del sistema periódicamente."""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.set_gauge("system_cpu_usage_percent", cpu_percent, {"core": "total"})
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.set_gauge("system_memory_usage_bytes", memory.used, {"type": "used"})
                self.set_gauge("system_memory_usage_bytes", memory.available, {"type": "available"})
                self.set_gauge("system_memory_usage_bytes", memory.total, {"type": "total"})
                
                # Disk usage
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        self.set_gauge(
                            "system_disk_usage_bytes",
                            usage.used,
                            {"device": partition.device, "mountpoint": partition.mountpoint}
                        )
                    except PermissionError:
                        continue
                
                # Network usage
                net_io = psutil.net_io_counters()
                self.inc_counter(
                    "system_network_bytes_total",
                    {"interface": "total", "direction": "bytes_sent"},
                    net_io.bytes_sent
                )
                self.inc_counter(
                    "system_network_bytes_total",
                    {"interface": "total", "direction": "bytes_recv"},
                    net_io.bytes_recv
                )
                
                await asyncio.sleep(30)  # Recolectar cada 30 segundos
                
            except Exception as e:
                self.logger.error(f"Error recolectando métricas del sistema: {e}")
                await asyncio.sleep(60)  # Esperar más tiempo en caso de error
    
    def get_metric(self, name: str):
        """Obtiene una métrica por nombre."""
        return self.metrics.get(name)
    
    def inc_counter(self, name: str, labels: Dict[str, str], value: float = 1.0) -> None:
        """Incrementa un contador."""
        metric = self.get_metric(name)
        if metric and hasattr(metric, 'labels'):
            metric.labels(**labels).inc(value)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str]) -> None:
        """Establece el valor de un gauge."""
        metric = self.get_metric(name)
        if metric and hasattr(metric, 'labels'):
            metric.labels(**labels).set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str]) -> None:
        """Observa un valor en un histograma."""
        metric = self.get_metric(name)
        if metric and hasattr(metric, 'labels'):
            metric.labels(**labels).observe(value)
    
    def time_function(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator para medir el tiempo de ejecución de una función."""
        def decorator(func: Callable):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.observe_histogram(metric_name, duration, labels or {})
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.observe_histogram(metric_name, duration, labels or {})
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    @contextmanager
    def measure_operation(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager para medir operaciones."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe_histogram(metric_name, duration, labels or {})
    
    def get_metrics_text(self) -> str:
        """Retorna las métricas en formato texto para Prometheus."""
        return generate_latest(self.registry)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Retorna las métricas como diccionario para debugging."""
        metrics_data = {}
        for name, metric in self.metrics.items():
            try:
                if hasattr(metric, '_value'):
                    metrics_data[name] = metric._value.get()
                elif hasattr(metric, '_sum'):
                    metrics_data[name] = {
                        'sum': metric._sum.get(),
                        'count': metric._count.get()
                    }
            except Exception as e:
                self.logger.warning(f"Error obteniendo métrica {name}: {e}")
        return metrics_data
    
    def reset_metrics(self) -> None:
        """Resetea todas las métricas (útil para testing)."""
        for metric in self.metrics.values():
            if hasattr(metric, '_value'):
                metric._value.set(0)
            elif hasattr(metric, '_sum'):
                metric._sum.set(0)
                metric._count.set(0)


# Instancia global
_metrics_service: Optional[MetricsService] = None


def get_metrics_service() -> MetricsService:
    """Retorna la instancia global del servicio de métricas."""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
        _metrics_service.initialize()
    return _metrics_service


def initialize_metrics() -> None:
    """Inicializa el sistema de métricas."""
    get_metrics_service()
