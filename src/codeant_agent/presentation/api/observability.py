"""
Endpoints de observabilidad para la API.

Este módulo implementa:
- Endpoints de métricas Prometheus
- Endpoints de health checks
- Endpoints de alertas
- Endpoints de logs
- Endpoints de tracing
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import PlainTextResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ...infrastructure.observability.metrics import get_metrics_service
from ...infrastructure.observability.health import get_health_service, HealthStatus
from ...infrastructure.observability.alerting import get_alerting_service, AlertSeverity
from ...infrastructure.observability.log_aggregation import get_log_aggregation_service, LogLevel
from ...infrastructure.observability.tracing import get_tracing_service
from ...presentation.middlewares.auth_middleware import get_current_user
from ...domain.entities.user import User
from ...utils.logging import get_logger

logger = get_logger(__name__)

observability_router = APIRouter(prefix="/observability", tags=["Observability"])


@observability_router.get("/metrics")
async def get_metrics() -> Response:
    """Endpoint para métricas Prometheus."""
    try:
        metrics_service = get_metrics_service()
        metrics_text = metrics_service.get_metrics_text()
        return Response(
            content=metrics_text,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo métricas"
        )


@observability_router.get("/metrics/json")
async def get_metrics_json() -> Dict[str, Any]:
    """Endpoint para métricas en formato JSON."""
    try:
        metrics_service = get_metrics_service()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics_service.get_metrics_dict()
        }
    except Exception as e:
        logger.error(f"Error obteniendo métricas JSON: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo métricas"
        )


@observability_router.get("/health")
async def get_health() -> Dict[str, Any]:
    """Endpoint de health check general."""
    try:
        health_service = get_health_service()
        report = await health_service.run_checks()
        
        return {
            "status": report.overall_status.value,
            "timestamp": report.timestamp.isoformat(),
            "version": report.version,
            "environment": report.environment,
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "duration_ms": check.duration_ms,
                    "timestamp": check.timestamp.isoformat(),
                    "details": check.details,
                    "error": check.error
                }
                for name, check in report.checks.items()
            }
        }
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error en health check"
        )


@observability_router.get("/health/live")
async def get_liveness() -> Dict[str, Any]:
    """Endpoint de liveness probe."""
    try:
        health_service = get_health_service()
        is_alive = health_service.get_liveness_status()
        
        if is_alive:
            return {
                "status": "alive",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not alive"
            )
    except Exception as e:
        logger.error(f"Error en liveness check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not alive"
        )


@observability_router.get("/health/ready")
async def get_readiness() -> Dict[str, Any]:
    """Endpoint de readiness probe."""
    try:
        health_service = get_health_service()
        is_ready = health_service.get_readiness_status()
        
        if is_ready:
            return {
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )
    except Exception as e:
        logger.error(f"Error en readiness check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@observability_router.get("/health/summary")
async def get_health_summary() -> Dict[str, Any]:
    """Endpoint de resumen de health."""
    try:
        health_service = get_health_service()
        return health_service.get_health_summary()
    except Exception as e:
        logger.error(f"Error obteniendo health summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo health summary"
        )


@observability_router.get("/alerts")
async def get_alerts(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Endpoint para obtener alertas activas."""
    try:
        alerting_service = get_alerting_service()
        active_alerts = alerting_service.get_active_alerts()
        
        return {
            "alerts": [
                {
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "description": alert.description,
                    "status": alert.status.value,
                    "starts_at": alert.starts_at.isoformat(),
                    "ends_at": alert.ends_at.isoformat() if alert.ends_at else None,
                    "labels": alert.labels,
                    "annotations": alert.annotations
                }
                for alert in active_alerts
            ],
            "count": len(active_alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error obteniendo alertas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo alertas"
        )


@observability_router.get("/alerts/history")
async def get_alert_history(
    limit: int = 100,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Endpoint para obtener historial de alertas."""
    try:
        alerting_service = get_alerting_service()
        history = alerting_service.get_alert_history(limit)
        
        return {
            "alerts": [
                {
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "description": alert.description,
                    "status": alert.status.value,
                    "starts_at": alert.starts_at.isoformat(),
                    "ends_at": alert.ends_at.isoformat() if alert.ends_at else None,
                    "labels": alert.labels,
                    "annotations": alert.annotations
                }
                for alert in history
            ],
            "count": len(history),
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error obteniendo historial de alertas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo historial de alertas"
        )


@observability_router.post("/alerts/{alert_name}/acknowledge")
async def acknowledge_alert(
    alert_name: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Endpoint para reconocer una alerta."""
    try:
        alerting_service = get_alerting_service()
        success = alerting_service.acknowledge_alert(alert_name)
        
        if success:
            return {
                "message": f"Alerta {alert_name} reconocida",
                "alert_name": alert_name,
                "acknowledged_by": current_user.username.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alerta {alert_name} no encontrada"
            )
    except Exception as e:
        logger.error(f"Error reconociendo alerta {alert_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reconociendo alerta"
        )


@observability_router.get("/alerts/summary")
async def get_alert_summary(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Endpoint para obtener resumen de alertas."""
    try:
        alerting_service = get_alerting_service()
        return alerting_service.get_alert_summary()
    except Exception as e:
        logger.error(f"Error obteniendo resumen de alertas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo resumen de alertas"
        )


@observability_router.get("/logs/search")
async def search_logs(
    query: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    level: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Endpoint para buscar logs."""
    try:
        log_service = get_log_aggregation_service()
        
        # Convertir nivel de log
        log_level = None
        if level:
            try:
                log_level = LogLevel(level.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Nivel de log inválido: {level}"
                )
        
        logs = await log_service.search_logs(
            query=query,
            start_time=start_time,
            end_time=end_time,
            level=log_level,
            limit=limit
        )
        
        return {
            "logs": logs,
            "count": len(logs),
            "query": query,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "level": level,
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error buscando logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error buscando logs"
        )


@observability_router.get("/logs/status")
async def get_log_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Endpoint para obtener estado del servicio de logs."""
    try:
        log_service = get_log_aggregation_service()
        return log_service.get_service_status()
    except Exception as e:
        logger.error(f"Error obteniendo estado de logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo estado de logs"
        )


@observability_router.get("/tracing/status")
async def get_tracing_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Endpoint para obtener estado del servicio de tracing."""
    try:
        tracing_service = get_tracing_service()
        return {
            "enabled": tracing_service._initialized,
            "service_name": tracing_service.settings.telemetry.jaeger_service_name,
            "jaeger_enabled": tracing_service.settings.telemetry.jaeger_enabled,
            "otel_enabled": tracing_service.settings.telemetry.otel_enabled,
            "sampling_rate": tracing_service.settings.telemetry.trace_sampling_rate
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado de tracing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo estado de tracing"
        )


@observability_router.get("/status")
async def get_observability_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Endpoint para obtener estado general de observabilidad."""
    try:
        # Obtener estado de todos los servicios
        metrics_service = get_metrics_service()
        health_service = get_health_service()
        alerting_service = get_alerting_service()
        log_service = get_log_aggregation_service()
        tracing_service = get_tracing_service()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "metrics": {
                    "enabled": metrics_service._initialized,
                    "prometheus_enabled": metrics_service.settings.telemetry.prometheus_enabled,
                    "port": metrics_service.settings.telemetry.prometheus_port
                },
                "health": {
                    "checks_count": len(health_service.checks),
                    "last_check": health_service._last_check_time.isoformat() if health_service._last_check_time else None
                },
                "alerting": {
                    "enabled": alerting_service._running,
                    "rules_count": len(alerting_service.rules),
                    "active_alerts": len(alerting_service.active_alerts)
                },
                "logs": log_service.get_service_status(),
                "tracing": {
                    "enabled": tracing_service._initialized,
                    "jaeger_enabled": tracing_service.settings.telemetry.jaeger_enabled,
                    "otel_enabled": tracing_service.settings.telemetry.otel_enabled
                }
            },
            "configuration": {
                "environment": metrics_service.settings.environment,
                "version": metrics_service.settings.version,
                "telemetry_enabled": metrics_service.settings.features.enable_telemetry
            }
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado de observabilidad: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo estado de observabilidad"
        )
