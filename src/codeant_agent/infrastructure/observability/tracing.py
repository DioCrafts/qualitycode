"""
Sistema de distributed tracing con OpenTelemetry.

Este módulo implementa:
- Configuración de OpenTelemetry
- Integración con Jaeger
- Span creation y management
- Context propagation
- Performance tracing
"""

import asyncio
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from functools import wraps
import time
import uuid

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from ...config.settings import get_settings
from ...utils.logging import get_logger


class TracingService:
    """Servicio de distributed tracing con OpenTelemetry."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer: Optional[trace.Tracer] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Inicializa el sistema de tracing."""
        if self._initialized:
            return
        
        if not self.settings.telemetry.otel_enabled:
            self.logger.info("OpenTelemetry tracing deshabilitado")
            return
        
        try:
            # Crear resource con información del servicio
            resource = Resource.create({
                "service.name": self.settings.telemetry.jaeger_service_name,
                "service.version": self.settings.version,
                "deployment.environment": self.settings.environment,
            })
            
            # Crear tracer provider
            self.tracer_provider = TracerProvider(
                resource=resource,
                sampler=self._create_sampler()
            )
            
            # Configurar exporters
            self._setup_exporters()
            
            # Establecer como tracer global
            trace.set_tracer_provider(self.tracer_provider)
            
            # Crear tracer
            self.tracer = trace.get_tracer(
                self.settings.telemetry.jaeger_service_name
            )
            
            # Instrumentar frameworks
            self._instrument_frameworks()
            
            self._initialized = True
            self.logger.info("Sistema de tracing inicializado correctamente")
            
        except Exception as e:
            self.logger.error(f"Error inicializando tracing: {e}")
            # Fallback a tracer no-op
            self.tracer = trace.get_tracer("noop")
    
    def _create_sampler(self):
        """Crea el sampler basado en la configuración."""
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        
        sampling_rate = self.settings.telemetry.trace_sampling_rate
        return TraceIdRatioBased(sampling_rate)
    
    def _setup_exporters(self) -> None:
        """Configura los exporters de traces."""
        if not self.tracer_provider:
            return
        
        # Exporter para Jaeger
        if self.settings.telemetry.jaeger_enabled:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
            self.logger.info("Jaeger exporter configurado")
        
        # Exporter para OTLP
        if self.settings.telemetry.otel_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.settings.telemetry.otel_endpoint
            )
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            self.logger.info("OTLP exporter configurado")
    
    def _instrument_frameworks(self) -> None:
        """Instrumenta frameworks y librerías."""
        try:
            # FastAPI instrumentation se hace en el middleware
            # SQLAlchemy instrumentation
            SQLAlchemyInstrumentor().instrument()
            
            # HTTPX instrumentation
            HTTPXClientInstrumentor().instrument()
            
            self.logger.info("Frameworks instrumentados correctamente")
        except Exception as e:
            self.logger.warning(f"Error instrumentando frameworks: {e}")
    
    def get_tracer(self) -> trace.Tracer:
        """Retorna el tracer configurado."""
        if not self._initialized:
            self.initialize()
        return self.tracer or trace.get_tracer("noop")
    
    def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> trace.Span:
        """Crea un nuevo span."""
        tracer = self.get_tracer()
        span = tracer.start_span(name, attributes=attributes or {})
        return span
    
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager para crear spans automáticamente."""
        tracer = self.get_tracer()
        with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
            yield span
    
    def trace_function(self, name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
        """Decorator para trazar funciones."""
        def decorator(func: Callable):
            func_name = name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.span(func_name, attributes) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("function.success", True)
                        return result
                    except Exception as e:
                        span.set_attribute("function.success", False)
                        span.set_attribute("function.error", str(e))
                        span.record_exception(e)
                        raise
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.span(func_name, attributes) as span:
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("function.success", True)
                        return result
                    except Exception as e:
                        span.set_attribute("function.success", False)
                        span.set_attribute("function.error", str(e))
                        span.record_exception(e)
                        raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Agrega un evento al span actual."""
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.add_event(name, attributes or {})
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Establece un atributo en el span actual."""
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute(key, value)
    
    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """Establece el status del span actual."""
        from opentelemetry.trace import Status, StatusCode
        
        current_span = trace.get_current_span()
        if current_span.is_recording():
            status_code = StatusCode.OK if status == "ok" else StatusCode.ERROR
            current_span.set_status(Status(status_code, description))
    
    def inject_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inyecta el contexto de tracing en headers HTTP."""
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        
        propagator = TraceContextTextMapPropagator()
        propagator.inject(headers)
        return headers
    
    def extract_context(self, headers: Dict[str, str]) -> None:
        """Extrae el contexto de tracing de headers HTTP."""
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        
        propagator = TraceContextTextMapPropagator()
        context = propagator.extract(headers)
        trace.set_span_in_context(trace.get_current_span(), context)
    
    def shutdown(self) -> None:
        """Cierra el sistema de tracing."""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
            self.logger.info("Sistema de tracing cerrado")


# Instancia global
_tracing_service: Optional[TracingService] = None


def get_tracing_service() -> TracingService:
    """Retorna la instancia global del servicio de tracing."""
    global _tracing_service
    if _tracing_service is None:
        _tracing_service = TracingService()
        _tracing_service.initialize()
    return _tracing_service


def initialize_tracing() -> None:
    """Inicializa el sistema de tracing."""
    service = get_tracing_service()
    # Garantiza invocación explícita para tests/mocks
    try:
        service.initialize()
    except Exception:
        pass


def shutdown_tracing() -> None:
    """Cierra el sistema de tracing."""
    global _tracing_service
    if _tracing_service:
        _tracing_service.shutdown()
        _tracing_service = None
