"""
Aplicación principal de CodeAnt Agent.

Este módulo configura y ejecuta la aplicación FastAPI con:
- Configuración de la aplicación
- Middleware de logging y métricas
- Endpoints de health check y métricas
- Configuración de CORS y seguridad
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config.settings import get_settings, Settings
from .utils.logging import setup_logging, get_logger, RequestContextMiddleware
from .utils.telemetry import (
    get_metrics_registry,
    get_health_checker,
    TelemetryMiddleware
)
from .utils.error import BaseError, ValidationError
from .presentation.api.routers import projects, analysis


# Logger global
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager para el ciclo de vida de la aplicación.
    
    Configura logging, métricas y otros servicios al inicio
    y los limpia al final.
    """
    # Startup
    settings = get_settings()
    logger.info("Iniciando CodeAnt Agent", version=settings.version)
    
    # Configurar logging
    setup_logging(
        log_level=settings.logging.level,
        log_format=settings.logging.format,
        log_file=settings.logging.file_path
    )
    
    logger.info("Logging configurado", level=settings.logging.level)
    
    # Verificar configuración crítica
    if settings.is_production():
        logger.warning("Ejecutando en modo PRODUCCIÓN")
        if settings.debug:
            raise ValueError("Debug no puede estar habilitado en producción")
    
    yield
    
    # Shutdown
    logger.info("Cerrando CodeAnt Agent")


def create_app() -> FastAPI:
    """
    Crea y configura la aplicación FastAPI.
    
    Returns:
        Aplicación FastAPI configurada
    """
    settings = get_settings()
    app = FastAPI(
        title="CodeAnt Agent",
        description="Agente inteligente de análisis de código con arquitectura hexagonal",
        version=settings.version,
        docs_url="/docs" if not settings.is_production() else None,
        redoc_url="/redoc" if not settings.is_production() else None,
        openapi_url="/openapi.json" if not settings.is_production() else None,
        lifespan=lifespan
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development() else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware personalizado
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(TelemetryMiddleware, metrics_registry=get_metrics_registry())
    
    # Configurar exception handlers
    setup_exception_handlers(app)
    
    # Configurar endpoints
    setup_endpoints(app)
    
    # Incluir routers de API
    app.include_router(projects.router)
    app.include_router(analysis.router)
    
    logger.info("Aplicación FastAPI creada y configurada")
    return app


def setup_exception_handlers(app: FastAPI) -> None:
    """Configura los manejadores de excepciones personalizados."""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Maneja errores de validación de request."""
        logger.warning(
            "Error de validación en request",
            path=request.url.path,
            errors=exc.errors()
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "message": "Los datos del request no son válidos",
                "details": exc.errors(),
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Maneja excepciones HTTP."""
        logger.warning(
            "Excepción HTTP",
            status_code=exc.status_code,
            path=request.url.path,
            detail=exc.detail
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(BaseError)
    async def domain_exception_handler(request: Request, exc: BaseError):
        """Maneja errores del dominio."""
        logger.error(
            "Error del dominio",
            error_type=exc.__class__.__name__,
            message=exc.message,
            category=exc.category.value,
            severity=exc.severity.value,
            path=request.url.path
        )
        
        # Determinar status code basado en la categoría
        status_code = 500  # Default
        if exc.category.value == "validation":
            status_code = 400
        elif exc.category.value == "authentication":
            status_code = 401
        elif exc.category.value == "authorization":
            status_code = 403
        elif exc.category.value == "not_found":
            status_code = 404
        elif exc.category.value == "conflict":
            status_code = 409
        
        return JSONResponse(
            status_code=status_code,
            content=exc.to_dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Maneja excepciones generales no capturadas."""
        logger.exception(
            "Excepción no manejada",
            error_type=exc.__class__.__name__,
            error=str(exc),
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "Ha ocurrido un error interno del servidor",
                "error_type": exc.__class__.__name__,
                "path": str(request.url.path)
            }
        )


def setup_endpoints(app: FastAPI) -> None:
    """Configura los endpoints básicos de la aplicación."""
    
    @app.get("/")
    async def root():
        """Endpoint raíz con información básica de la aplicación."""
        settings = get_settings()
        return {
            "name": "CodeAnt Agent",
            "version": settings.version,
            "description": "Agente inteligente de análisis de código",
            "status": "running",
            "environment": settings.environment
        }
    
    @app.options("/")
    async def root_options():
        """Endpoint OPTIONS para CORS preflight."""
        return {"message": "OK"}
    
    @app.get("/health")
    async def health_check():
        """Endpoint de health check."""
        health_checker = get_health_checker()
        results = await health_checker.run_checks()
        overall_status = health_checker.get_overall_status(results)
        
        status_code = 200 if overall_status == "healthy" else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": overall_status,
                "timestamp": results.get("basic", {}).get("timestamp"),
                "checks": results
            }
        )
    
    @app.get("/health/live")
    async def liveness_check():
        """Endpoint de liveness check para Kubernetes."""
        return {"status": "alive"}
    
    @app.get("/health/ready")
    async def readiness_check():
        """Endpoint de readiness check para Kubernetes."""
        health_checker = get_health_checker()
        results = await health_checker.run_checks()
        overall_status = health_checker.get_overall_status(results)
        
        if overall_status in ["healthy", "degraded"]:
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
    
    @app.get("/metrics")
    async def metrics():
        """Endpoint de métricas Prometheus."""
        settings = get_settings()
        if not settings.features.enable_telemetry:
            raise HTTPException(status_code=404, detail="Telemetry disabled")
        
        metrics_registry = get_metrics_registry()
        return PlainTextResponse(
            content=metrics_registry.get_metrics_text(),
            media_type="text/plain"
        )
    
    @app.get("/info")
    async def info():
        """Endpoint con información detallada de la aplicación."""
        settings = get_settings()
        return {
            "name": "CodeAnt Agent",
            "version": settings.version,
            "description": "Agente inteligente de análisis de código con arquitectura hexagonal",
            "environment": settings.environment,
            "debug": settings.debug,
            "features": {
                "ai_analysis": settings.ai.enable_ai_analysis,
                "telemetry": settings.features.enable_telemetry,
                "experimental": settings.features.enable_experimental,
                "health_checks": settings.features.enable_health_checks
            },
            "config": {
                "database": {
                    "max_connections": settings.database.max_connections,
                    "min_connections": settings.database.min_connections
                },
                "server": {
                    "host": settings.server.host,
                    "port": settings.server.port,
                    "workers": settings.server.workers
                },
                "logging": {
                    "level": settings.logging.level,
                    "format": settings.logging.format
                }
            }
        }
    
    @app.get("/config")
    async def config():
        """Endpoint para ver la configuración actual (solo en desarrollo)."""
        settings = get_settings()
        if not settings.is_development():
            raise HTTPException(status_code=404, detail="Not available in production")
        
        return {
            "environment": settings.environment,
            "debug": settings.debug,
            "database": {
                "url": settings.database.url,
                "max_connections": settings.database.max_connections,
                "min_connections": settings.database.min_connections
            },
            "redis": {
                "url": settings.redis.url,
                "max_connections": settings.redis.max_connections
            },
            "server": {
                "host": settings.server.host,
                "port": settings.server.port,
                "workers": settings.server.workers,
                "reload": settings.server.reload
            },
            "logging": {
                "level": settings.logging.level,
                "format": settings.logging.format
            },
            "ai": {
                "enable_ai_analysis": settings.ai.enable_ai_analysis,
                "max_tokens": settings.ai.max_tokens
            },
            "security": {
                "enable_rate_limiting": settings.security.enable_rate_limiting,
                "max_requests_per_minute": settings.security.max_requests_per_minute
            }
        }


# Crear la aplicación
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Iniciando servidor de desarrollo")
    settings = get_settings()
    
    uvicorn.run(
        "codeant_agent.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        log_level=settings.server.log_level,
        workers=1 if settings.server.reload else settings.server.workers
    )
