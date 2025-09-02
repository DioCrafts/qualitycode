"""
Controlador de integración para el sistema incremental.

Este módulo proporciona una interfaz unificada para el sistema incremental,
integrando con el sistema principal de CodeAnt.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ...infrastructure.integration.incremental_integration_service import IncrementalIntegrationService
from ...utils.logging import get_logger
from ...utils.error import BaseError, ValidationError

logger = get_logger(__name__)

# Router para las rutas de integración incremental
router = APIRouter(prefix="/incremental-integration", tags=["incremental-integration"])


class SessionStartRequestModel(BaseModel):
    """Modelo de request para iniciar sesión."""
    project_path: str = Field(..., description="Ruta del proyecto a analizar")
    session_config: Optional[Dict[str, Any]] = Field(None, description="Configuración de la sesión")


class AnalysisRequestModel(BaseModel):
    """Modelo de request para análisis."""
    force_full_analysis: bool = Field(False, description="Forzar análisis completo")
    include_metrics: bool = Field(True, description="Incluir métricas detalladas")


class ChangeDetectionRequestModel(BaseModel):
    """Modelo de request para detección de cambios."""
    previous_commit: Optional[str] = Field(None, description="Commit anterior")
    current_commit: Optional[str] = Field(None, description="Commit actual")


class CachePredictionRequestModel(BaseModel):
    """Modelo de request para predicción de caché."""
    analysis_history_days: int = Field(7, description="Días de historial a considerar")
    prediction_horizon_hours: int = Field(24, description="Horizonte de predicción en horas")


class SessionResponseModel(BaseModel):
    """Modelo de response para sesión."""
    session_id: str
    project_path: str
    start_time: str
    analysis_count: int
    change_count: int
    last_analysis_id: Optional[str]
    is_active: bool
    performance_metrics: Optional[Dict[str, Any]]


class SessionSummaryModel(BaseModel):
    """Modelo de resumen de sesión."""
    session_id: str
    project_path: str
    start_time: str
    end_time: str
    duration_seconds: float
    analysis_count: int
    change_count: int
    last_analysis_id: Optional[str]
    performance_metrics: Optional[Dict[str, Any]]


# Dependency injection
def get_incremental_integration_service() -> IncrementalIntegrationService:
    """Obtiene la instancia del servicio de integración incremental."""
    # TODO: Implementar inyección de dependencias real
    # Por ahora, creamos una instancia temporal
    from ...application.use_cases.incremental_use_cases import (
        IncrementalAnalysisUseCase,
        CacheManagementUseCase,
        ChangeDetectionUseCase,
        PredictiveCacheUseCase
    )
    
    return IncrementalIntegrationService(
        incremental_analysis_use_case=IncrementalAnalysisUseCase(),
        cache_management_use_case=CacheManagementUseCase(),
        change_detection_use_case=ChangeDetectionUseCase(),
        predictive_cache_use_case=PredictiveCacheUseCase()
    )


@router.post("/sessions", response_model=Dict[str, str])
async def start_analysis_session(
    request: SessionStartRequestModel,
    service: IncrementalIntegrationService = Depends(get_incremental_integration_service)
):
    """
    Inicia una nueva sesión de análisis incremental.
    
    Args:
        request: Datos del request de sesión
        service: Servicio de integración incremental
        
    Returns:
        ID de la sesión iniciada
    """
    try:
        logger.info(
            "Iniciando sesión de análisis incremental",
            project_path=request.project_path
        )
        
        session_id = await service.start_analysis_session(
            project_path=request.project_path,
            session_config=request.session_config
        )
        
        return {"session_id": session_id}
        
    except ValidationError as e:
        logger.warning("Error de validación al iniciar sesión", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio al iniciar sesión", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado al iniciar sesión", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/sessions/{session_id}/analyze")
async def perform_incremental_analysis(
    session_id: str,
    request: AnalysisRequestModel,
    service: IncrementalIntegrationService = Depends(get_incremental_integration_service)
):
    """
    Realiza análisis incremental en una sesión activa.
    
    Args:
        session_id: ID de la sesión
        request: Datos del request de análisis
        service: Servicio de integración incremental
        
    Returns:
        Resultado del análisis incremental
    """
    try:
        logger.info(
            "Realizando análisis incremental",
            session_id=session_id,
            force_full=request.force_full_analysis
        )
        
        result = await service.perform_incremental_analysis(
            session_id=session_id,
            force_full_analysis=request.force_full_analysis,
            include_metrics=request.include_metrics
        )
        
        return result.to_dict()
        
    except ValidationError as e:
        logger.warning("Error de validación en análisis incremental", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio en análisis incremental", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado en análisis incremental", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/sessions/{session_id}/detect-changes")
async def detect_changes(
    session_id: str,
    request: ChangeDetectionRequestModel,
    service: IncrementalIntegrationService = Depends(get_incremental_integration_service)
):
    """
    Detecta cambios en el código de una sesión.
    
    Args:
        session_id: ID de la sesión
        request: Datos del request de detección
        service: Servicio de integración incremental
        
    Returns:
        Resultado de la detección de cambios
    """
    try:
        logger.info(
            "Detectando cambios",
            session_id=session_id,
            previous_commit=request.previous_commit
        )
        
        result = await service.detect_changes(
            session_id=session_id,
            previous_commit=request.previous_commit,
            current_commit=request.current_commit
        )
        
        return result.to_dict()
        
    except ValidationError as e:
        logger.warning("Error de validación en detección de cambios", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio en detección de cambios", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado en detección de cambios", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/sessions/{session_id}/cache/status")
async def get_cache_status(
    session_id: str,
    service: IncrementalIntegrationService = Depends(get_incremental_integration_service)
):
    """
    Obtiene el estado del caché para una sesión.
    
    Args:
        session_id: ID de la sesión
        service: Servicio de integración incremental
        
    Returns:
        Estado del caché
    """
    try:
        logger.info("Obteniendo estado del caché", session_id=session_id)
        
        cache_status = await service.get_cache_status(session_id)
        
        return cache_status
        
    except ValidationError as e:
        logger.warning("Error de validación en estado del caché", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio en estado del caché", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado en estado del caché", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/sessions/{session_id}/cache/predict")
async def predict_cache_usage(
    session_id: str,
    request: CachePredictionRequestModel,
    service: IncrementalIntegrationService = Depends(get_incremental_integration_service)
):
    """
    Predice el uso del caché para una sesión.
    
    Args:
        session_id: ID de la sesión
        request: Datos del request de predicción
        service: Servicio de integración incremental
        
    Returns:
        Predicción del uso del caché
    """
    try:
        logger.info(
            "Prediciendo uso del caché",
            session_id=session_id,
            history_days=request.analysis_history_days
        )
        
        result = await service.predict_cache_usage(
            session_id=session_id,
            analysis_history_days=request.analysis_history_days,
            prediction_horizon_hours=request.prediction_horizon_hours
        )
        
        return result.to_dict()
        
    except ValidationError as e:
        logger.warning("Error de validación en predicción de caché", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio en predicción de caché", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado en predicción de caché", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.delete("/sessions/{session_id}", response_model=SessionSummaryModel)
async def end_analysis_session(
    session_id: str,
    service: IncrementalIntegrationService = Depends(get_incremental_integration_service)
):
    """
    Finaliza una sesión de análisis incremental.
    
    Args:
        session_id: ID de la sesión
        service: Servicio de integración incremental
        
    Returns:
        Resumen de la sesión
    """
    try:
        logger.info("Finalizando sesión de análisis incremental", session_id=session_id)
        
        session_summary = await service.end_analysis_session(session_id)
        
        return SessionSummaryModel(**session_summary)
        
    except ValidationError as e:
        logger.warning("Error de validación al finalizar sesión", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio al finalizar sesión", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado al finalizar sesión", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/sessions", response_model=List[SessionResponseModel])
async def get_active_sessions(
    service: IncrementalIntegrationService = Depends(get_incremental_integration_service)
):
    """
    Obtiene la lista de sesiones activas.
    
    Args:
        service: Servicio de integración incremental
        
    Returns:
        Lista de sesiones activas
    """
    try:
        logger.info("Obteniendo sesiones activas")
        
        sessions = await service.get_active_sessions()
        
        return [SessionResponseModel(**session) for session in sessions]
        
    except Exception as e:
        logger.exception("Error obteniendo sesiones activas", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/sessions/cleanup")
async def cleanup_expired_sessions(
    max_duration_hours: int = 24,
    service: IncrementalIntegrationService = Depends(get_incremental_integration_service)
):
    """
    Limpia sesiones expiradas.
    
    Args:
        max_duration_hours: Duración máxima en horas
        service: Servicio de integración incremental
        
    Returns:
        Número de sesiones limpiadas
    """
    try:
        logger.info(
            "Limpiando sesiones expiradas",
            max_duration_hours=max_duration_hours
        )
        
        cleaned_count = await service.cleanup_expired_sessions(max_duration_hours)
        
        return {"cleaned_sessions": cleaned_count}
        
    except Exception as e:
        logger.exception("Error limpiando sesiones expiradas", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/metrics")
async def get_system_metrics(
    service: IncrementalIntegrationService = Depends(get_incremental_integration_service)
):
    """
    Obtiene métricas del sistema incremental.
    
    Args:
        service: Servicio de integración incremental
        
    Returns:
        Métricas del sistema
    """
    try:
        logger.info("Obteniendo métricas del sistema incremental")
        
        metrics = await service.get_system_metrics()
        
        return metrics
        
    except Exception as e:
        logger.exception("Error obteniendo métricas del sistema", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/sessions/{session_id}")
async def get_session_info(
    session_id: str,
    service: IncrementalIntegrationService = Depends(get_incremental_integration_service)
):
    """
    Obtiene información de una sesión específica.
    
    Args:
        session_id: ID de la sesión
        service: Servicio de integración incremental
        
    Returns:
        Información de la sesión
    """
    try:
        logger.info("Obteniendo información de sesión", session_id=session_id)
        
        # Obtener todas las sesiones activas
        sessions = await service.get_active_sessions()
        
        # Buscar la sesión específica
        session_info = next(
            (session for session in sessions if session["session_id"] == session_id),
            None
        )
        
        if not session_info:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        return session_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error obteniendo información de sesión", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")
