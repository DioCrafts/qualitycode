"""
Controlador para el sistema de análisis incremental.

Este módulo maneja las peticiones HTTP relacionadas con:
- Análisis incremental de código
- Gestión de caché inteligente
- Detección de cambios granulares
- Predicción de caché
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ...application.use_cases.incremental_use_cases import (
    IncrementalAnalysisUseCase,
    CacheManagementUseCase,
    ChangeDetectionUseCase,
    PredictiveCacheUseCase
)
from ...application.dtos.incremental_dtos import (
    IncrementalAnalysisRequest,
    IncrementalAnalysisResponse,
    CacheStatusResponse,
    ChangeDetectionRequest,
    ChangeDetectionResponse,
    PredictiveCacheRequest,
    PredictiveCacheResponse
)
from ...utils.logging import get_logger
from ...utils.error import BaseError, ValidationError

logger = get_logger(__name__)

# Router para las rutas del sistema incremental
router = APIRouter(prefix="/incremental", tags=["incremental"])


class IncrementalAnalysisRequestModel(BaseModel):
    """Modelo de request para análisis incremental."""
    project_path: str = Field(..., description="Ruta del proyecto a analizar")
    previous_analysis_id: Optional[str] = Field(None, description="ID del análisis anterior")
    force_full_analysis: bool = Field(False, description="Forzar análisis completo")
    include_metrics: bool = Field(True, description="Incluir métricas detalladas")
    cache_strategy: str = Field("smart", description="Estrategia de caché a usar")


class ChangeDetectionRequestModel(BaseModel):
    """Modelo de request para detección de cambios."""
    project_path: str = Field(..., description="Ruta del proyecto")
    previous_commit: Optional[str] = Field(None, description="Commit anterior")
    current_commit: Optional[str] = Field(None, description="Commit actual")
    file_patterns: Optional[List[str]] = Field(None, description="Patrones de archivos a incluir")


class PredictiveCacheRequestModel(BaseModel):
    """Modelo de request para predicción de caché."""
    project_path: str = Field(..., description="Ruta del proyecto")
    analysis_history_days: int = Field(7, description="Días de historial a considerar")
    prediction_horizon_hours: int = Field(24, description="Horizonte de predicción en horas")


class CacheWarmupRequestModel(BaseModel):
    """Modelo de request para calentamiento de caché."""
    project_path: str = Field(..., description="Ruta del proyecto")
    priority_files: Optional[List[str]] = Field(None, description="Archivos prioritarios")
    warmup_strategy: str = Field("predictive", description="Estrategia de calentamiento")


# Dependency injection
def get_incremental_analysis_use_case() -> IncrementalAnalysisUseCase:
    """Obtiene la instancia del caso de uso de análisis incremental."""
    # TODO: Implementar inyección de dependencias real
    return IncrementalAnalysisUseCase()


def get_cache_management_use_case() -> CacheManagementUseCase:
    """Obtiene la instancia del caso de uso de gestión de caché."""
    # TODO: Implementar inyección de dependencias real
    return CacheManagementUseCase()


def get_change_detection_use_case() -> ChangeDetectionUseCase:
    """Obtiene la instancia del caso de uso de detección de cambios."""
    # TODO: Implementar inyección de dependencias real
    return ChangeDetectionUseCase()


def get_predictive_cache_use_case() -> PredictiveCacheUseCase:
    """Obtiene la instancia del caso de uso de caché predictivo."""
    # TODO: Implementar inyección de dependencias real
    return PredictiveCacheUseCase()


@router.post("/analyze", response_model=IncrementalAnalysisResponse)
async def analyze_incremental(
    request: IncrementalAnalysisRequestModel,
    background_tasks: BackgroundTasks,
    use_case: IncrementalAnalysisUseCase = Depends(get_incremental_analysis_use_case)
):
    """
    Realiza análisis incremental del código.
    
    Args:
        request: Datos del request de análisis
        background_tasks: Tareas en segundo plano
        use_case: Caso de uso de análisis incremental
        
    Returns:
        Resultado del análisis incremental
    """
    try:
        logger.info(
            "Iniciando análisis incremental",
            project_path=request.project_path,
            force_full=request.force_full_analysis
        )
        
        # Convertir request model a DTO
        analysis_request = IncrementalAnalysisRequest(
            project_path=request.project_path,
            previous_analysis_id=request.previous_analysis_id,
            force_full_analysis=request.force_full_analysis,
            include_metrics=request.include_metrics,
            cache_strategy=request.cache_strategy
        )
        
        # Ejecutar análisis
        result = await use_case.execute(analysis_request)
        
        # Programar tareas en segundo plano si es necesario
        if result.cache_warmup_recommended:
            background_tasks.add_task(
                _warmup_cache_background,
                request.project_path,
                result.analysis_id
            )
        
        logger.info(
            "Análisis incremental completado",
            analysis_id=result.analysis_id,
            files_analyzed=result.files_analyzed,
            cache_hit_rate=result.cache_hit_rate
        )
        
        return result
        
    except ValidationError as e:
        logger.warning("Error de validación en análisis incremental", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio en análisis incremental", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado en análisis incremental", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/detect-changes", response_model=ChangeDetectionResponse)
async def detect_changes(
    request: ChangeDetectionRequestModel,
    use_case: ChangeDetectionUseCase = Depends(get_change_detection_use_case)
):
    """
    Detecta cambios en el código.
    
    Args:
        request: Datos del request de detección
        use_case: Caso de uso de detección de cambios
        
    Returns:
        Resultado de la detección de cambios
    """
    try:
        logger.info(
            "Iniciando detección de cambios",
            project_path=request.project_path,
            previous_commit=request.previous_commit
        )
        
        # Convertir request model a DTO
        detection_request = ChangeDetectionRequest(
            project_path=request.project_path,
            previous_commit=request.previous_commit,
            current_commit=request.current_commit,
            file_patterns=request.file_patterns
        )
        
        # Ejecutar detección
        result = await use_case.execute(detection_request)
        
        logger.info(
            "Detección de cambios completada",
            files_changed=len(result.changed_files),
            functions_changed=len(result.changed_functions)
        )
        
        return result
        
    except ValidationError as e:
        logger.warning("Error de validación en detección de cambios", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio en detección de cambios", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado en detección de cambios", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/cache/status", response_model=CacheStatusResponse)
async def get_cache_status(
    project_path: str,
    use_case: CacheManagementUseCase = Depends(get_cache_management_use_case)
):
    """
    Obtiene el estado del caché.
    
    Args:
        project_path: Ruta del proyecto
        use_case: Caso de uso de gestión de caché
        
    Returns:
        Estado del caché
    """
    try:
        logger.info("Obteniendo estado del caché", project_path=project_path)
        
        result = await use_case.get_cache_status(project_path)
        
        return result
        
    except ValidationError as e:
        logger.warning("Error de validación en estado del caché", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio en estado del caché", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado en estado del caché", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/cache/clear")
async def clear_cache(
    project_path: str,
    cache_level: Optional[str] = None,
    use_case: CacheManagementUseCase = Depends(get_cache_management_use_case)
):
    """
    Limpia el caché.
    
    Args:
        project_path: Ruta del proyecto
        cache_level: Nivel de caché a limpiar (opcional)
        use_case: Caso de uso de gestión de caché
        
    Returns:
        Resultado de la operación
    """
    try:
        logger.info(
            "Limpiando caché",
            project_path=project_path,
            cache_level=cache_level
        )
        
        result = await use_case.clear_cache(project_path, cache_level)
        
        logger.info("Caché limpiado exitosamente", project_path=project_path)
        
        return {"message": "Caché limpiado exitosamente", "details": result}
        
    except ValidationError as e:
        logger.warning("Error de validación en limpieza de caché", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio en limpieza de caché", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado en limpieza de caché", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/cache/predict", response_model=PredictiveCacheResponse)
async def predict_cache_usage(
    request: PredictiveCacheRequestModel,
    use_case: PredictiveCacheUseCase = Depends(get_predictive_cache_use_case)
):
    """
    Predice el uso del caché.
    
    Args:
        request: Datos del request de predicción
        use_case: Caso de uso de caché predictivo
        
    Returns:
        Predicción del uso del caché
    """
    try:
        logger.info(
            "Iniciando predicción de caché",
            project_path=request.project_path,
            history_days=request.analysis_history_days
        )
        
        # Convertir request model a DTO
        prediction_request = PredictiveCacheRequest(
            project_path=request.project_path,
            analysis_history_days=request.analysis_history_days,
            prediction_horizon_hours=request.prediction_horizon_hours
        )
        
        # Ejecutar predicción
        result = await use_case.execute(prediction_request)
        
        logger.info(
            "Predicción de caché completada",
            predicted_files=len(result.predicted_files),
            confidence_score=result.confidence_score
        )
        
        return result
        
    except ValidationError as e:
        logger.warning("Error de validación en predicción de caché", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio en predicción de caché", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado en predicción de caché", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/cache/warmup")
async def warmup_cache(
    request: CacheWarmupRequestModel,
    background_tasks: BackgroundTasks,
    use_case: CacheManagementUseCase = Depends(get_cache_management_use_case)
):
    """
    Calienta el caché.
    
    Args:
        request: Datos del request de calentamiento
        background_tasks: Tareas en segundo plano
        use_case: Caso de uso de gestión de caché
        
    Returns:
        Resultado de la operación
    """
    try:
        logger.info(
            "Iniciando calentamiento de caché",
            project_path=request.project_path,
            strategy=request.warmup_strategy
        )
        
        # Programar calentamiento en segundo plano
        background_tasks.add_task(
            _warmup_cache_background,
            request.project_path,
            request.priority_files,
            request.warmup_strategy
        )
        
        return {"message": "Calentamiento de caché iniciado en segundo plano"}
        
    except ValidationError as e:
        logger.warning("Error de validación en calentamiento de caché", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio en calentamiento de caché", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado en calentamiento de caché", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/metrics")
async def get_incremental_metrics(
    project_path: str,
    use_case: IncrementalAnalysisUseCase = Depends(get_incremental_analysis_use_case)
):
    """
    Obtiene métricas del sistema incremental.
    
    Args:
        project_path: Ruta del proyecto
        use_case: Caso de uso de análisis incremental
        
    Returns:
        Métricas del sistema incremental
    """
    try:
        logger.info("Obteniendo métricas incrementales", project_path=project_path)
        
        metrics = await use_case.get_metrics(project_path)
        
        return metrics
        
    except ValidationError as e:
        logger.warning("Error de validación en métricas incrementales", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error("Error del dominio en métricas incrementales", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado en métricas incrementales", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor")


# Funciones auxiliares para tareas en segundo plano
async def _warmup_cache_background(
    project_path: str,
    analysis_id: Optional[str] = None,
    priority_files: Optional[List[str]] = None,
    strategy: str = "predictive"
):
    """
    Calienta el caché en segundo plano.
    
    Args:
        project_path: Ruta del proyecto
        analysis_id: ID del análisis (opcional)
        priority_files: Archivos prioritarios (opcional)
        strategy: Estrategia de calentamiento
    """
    try:
        logger.info(
            "Iniciando calentamiento de caché en segundo plano",
            project_path=project_path,
            strategy=strategy
        )
        
        # TODO: Implementar lógica de calentamiento real
        # use_case = get_cache_management_use_case()
        # await use_case.warmup_cache(project_path, priority_files, strategy)
        
        logger.info("Calentamiento de caché completado", project_path=project_path)
        
    except Exception as e:
        logger.exception(
            "Error en calentamiento de caché en segundo plano",
            project_path=project_path,
            error=str(e)
        )
