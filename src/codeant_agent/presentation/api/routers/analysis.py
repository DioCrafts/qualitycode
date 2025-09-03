"""
Router para endpoints de análisis de código.
"""
from typing import Dict, Any, Optional
import random
import asyncio
from datetime import datetime, timedelta
import time
import uuid

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from pydantic import BaseModel, Field

from ....domain.value_objects.project_id import ProjectId
from ....utils.logging import get_logger
from ....utils.error import Result

# Import del caso de uso real
from ....application.use_cases.analyze_project_use_case import (
    AnalyzeProjectUseCase,
    AnalyzeProjectRequest as DomainAnalyzeRequest,
    AnalysisResults
)

# Import de las dependencias necesarias
from ....infrastructure.parsers.parser_factory import ParserFactory
from ....infrastructure.dead_code_analysis.dead_code_engine import DeadCodeEngine
from ..dependencies import get_project_repository

logger = get_logger(__name__)

# Crear router
router = APIRouter(prefix="/api", tags=["analysis"])

# Almacenamiento en memoria para análisis 
_IN_MEMORY_ANALYSES = {}


# DTOs para requests y responses
class AnalysisConfigDTO(BaseModel):
    """DTO para configuración de análisis."""
    forceFullAnalysis: bool = Field(default=False, description="Si forzar análisis completo")
    includeMetrics: bool = Field(default=True, description="Si incluir métricas detalladas")


class RunAnalysisRequestDTO(BaseModel):
    """DTO para solicitud de análisis."""
    projectId: str = Field(..., description="ID del proyecto a analizar")
    config: Optional[AnalysisConfigDTO] = Field(default_factory=AnalysisConfigDTO, description="Configuración del análisis")


class AnalysisResponseDTO(BaseModel):
    """DTO para respuesta de análisis."""
    id: str
    project_id: str
    status: str
    created_at: str
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float
    config: dict
    estimated_completion_time: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# Servicio simulado
class AnalysisSimulator:
    """Simulador de análisis de código para desarrollo."""
    
    @staticmethod
    async def start_analysis(project_id: str, config: Dict[str, Any], background_tasks: BackgroundTasks) -> AnalysisResponseDTO:
        """
        Iniciar un análisis simulado.
        """
        logger.info(f"Iniciando análisis simulado para el proyecto {project_id}")
        
        # Generar ID único para el análisis
        analysis_id = f"analysis-{uuid.uuid4()}"
        
        # Crear resultado simulado
        now = datetime.now()
        estimated_completion = now + timedelta(minutes=random.randint(1, 10))
        
        analysis = {
            "id": analysis_id,
            "project_id": project_id,
            "status": "PENDING",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "progress": 0.0,
            "config": config,
            "estimated_completion_time": estimated_completion.isoformat(),
            "start_time": time.time(),
            "result": {
                "total_violations": 0,
                "critical_violations": 0,
                "high_violations": 0,
                "files_analyzed": 0,
                "quality_score": 0
            }
        }
        
        # Guardar en memoria
        _IN_MEMORY_ANALYSES[analysis_id] = analysis
        
        # Programar tarea en segundo plano para simular progreso
        background_tasks.add_task(
            AnalysisSimulator.simulate_analysis_progress,
            analysis_id
        )
        
        # Convertir a DTO
        return AnalysisResponseDTO(**analysis)
    
    @staticmethod
    def simulate_analysis_progress(analysis_id: str):
        """
        Simular el progreso del análisis en segundo plano.
        
        Esta función NO es async porque se ejecuta como una tarea en segundo plano
        y debe usar time.sleep en lugar de asyncio.sleep para no bloquear.
        """
        import time
        
        if analysis_id not in _IN_MEMORY_ANALYSES:
            logger.error(f"No se encontró el análisis {analysis_id}")
            return
            
        # Cambiar estado a IN_PROGRESS
        _IN_MEMORY_ANALYSES[analysis_id]["status"] = "IN_PROGRESS"
        
        # Simular progreso gradual
        total_time = random.uniform(15, 30)  # Entre 15 y 30 segundos para la simulación
        start_time = _IN_MEMORY_ANALYSES[analysis_id]["start_time"]
        total_files = random.randint(20, 100)
        
        # Bucle de simulación con 10 pasos
        steps = 10
        for step in range(steps):
            # Calcular progreso actual (0.1, 0.2, ..., 1.0)
            progress = min((step + 1) / steps, 1.0)
            
            # Actualizar progreso
            _IN_MEMORY_ANALYSES[analysis_id]["progress"] = progress
            _IN_MEMORY_ANALYSES[analysis_id]["updated_at"] = datetime.now().isoformat()
            
            # Generar métricas incrementalmente
            result = _IN_MEMORY_ANALYSES[analysis_id]["result"]
            files_analyzed = int(total_files * progress)
            
            result["files_analyzed"] = files_analyzed
            result["total_violations"] = int(files_analyzed * random.uniform(0.1, 0.5))
            result["critical_violations"] = int(result["total_violations"] * random.uniform(0.05, 0.15))
            result["high_violations"] = int(result["total_violations"] * random.uniform(0.2, 0.4))
            result["quality_score"] = int(100 - (result["total_violations"] / files_analyzed if files_analyzed > 0 else 0) * 25)
            
            # Esperar antes de la siguiente actualización (dividir el tiempo total en pasos)
            if step < steps - 1:  # No esperar en el último paso
                time.sleep(total_time / steps)
        
        # Análisis completado
        _IN_MEMORY_ANALYSES[analysis_id]["status"] = "COMPLETED"
        _IN_MEMORY_ANALYSES[analysis_id]["completed_at"] = datetime.now().isoformat()
        _IN_MEMORY_ANALYSES[analysis_id]["progress"] = 1.0
        logger.info(f"Análisis {analysis_id} completado")


# Dependencias
def get_analyze_use_case():
    """Obtener instancia del caso de uso de análisis."""
    from ..dependencies import get_analyze_project_use_case
    return get_analyze_project_use_case()


# Endpoints
@router.post("/analysis/run", response_model=AnalysisResponseDTO)
async def run_analysis(
    request: RunAnalysisRequestDTO,
    background_tasks: BackgroundTasks,
    analyze_use_case: AnalyzeProjectUseCase = Depends(get_analyze_use_case)
):
    """
    Iniciar un análisis de código.
    
    Args:
        request: Datos de la solicitud de análisis
        background_tasks: Tareas en segundo plano
        analyze_use_case: Caso de uso de análisis
        
    Returns:
        Información sobre el análisis iniciado
    """
    try:
        logger.info(f"Iniciando análisis para proyecto {request.projectId}")
        
        # Crear request del dominio
        domain_request = DomainAnalyzeRequest(
            project_id=request.projectId,
            config=request.config.dict() if request.config else {},
            force_full_analysis=request.config.forceFullAnalysis if request.config else False,
            include_metrics=request.config.includeMetrics if request.config else True
        )
        
        # Crear ID único para el análisis
        analysis_id = f"analysis-{uuid.uuid4()}"
        
        # Crear respuesta inicial
        initial_response = AnalysisResponseDTO(
            id=analysis_id,
            project_id=request.projectId,
            status="PENDING",
            created_at=datetime.now().isoformat(),
            progress=0.0,
            config=request.config.dict() if request.config else {},
            estimated_completion_time=(datetime.now() + timedelta(minutes=5)).isoformat()
        )
        
        # Guardar en memoria
        _IN_MEMORY_ANALYSES[analysis_id] = initial_response.dict()
        
        # Ejecutar análisis en segundo plano
        background_tasks.add_task(
            execute_real_analysis,
            analysis_id,
            domain_request,
            analyze_use_case
        )
        
        logger.info(f"Análisis {analysis_id} iniciado para proyecto {request.projectId}")
        
        return initial_response
        
    except Exception as e:
        logger.exception(f"Error al iniciar análisis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al iniciar análisis: {str(e)}"
        )


async def execute_real_analysis(
    analysis_id: str,
    request: DomainAnalyzeRequest,
    use_case: AnalyzeProjectUseCase
):
    """
    Ejecutar el análisis real en segundo plano.
    """
    try:
        logger.info(f"Ejecutando análisis real {analysis_id}")
        
        # Actualizar estado a IN_PROGRESS
        if analysis_id in _IN_MEMORY_ANALYSES:
            _IN_MEMORY_ANALYSES[analysis_id]["status"] = "IN_PROGRESS"
            _IN_MEMORY_ANALYSES[analysis_id]["updated_at"] = datetime.now().isoformat()
        
        # Ejecutar el análisis real
        result = await use_case.execute(request)
        
        if result.success:
            analysis_results = result.data
            
            # Actualizar los resultados en memoria
            if analysis_id in _IN_MEMORY_ANALYSES:
                _IN_MEMORY_ANALYSES[analysis_id].update({
                    "status": analysis_results.status,
                    "completed_at": analysis_results.completed_at,
                    "progress": analysis_results.progress,
                    "result": {
                        "total_violations": analysis_results.total_violations,
                        "critical_violations": analysis_results.critical_violations,
                        "high_violations": analysis_results.high_violations,
                        "files_analyzed": analysis_results.files_analyzed,
                        "quality_score": analysis_results.quality_score,
                        "complexity_metrics": analysis_results.complexity_metrics,
                        "quality_metrics": analysis_results.quality_metrics,
                        "dead_code_results": analysis_results.dead_code_results,
                        "security_results": analysis_results.security_results,
                        "duplicate_results": analysis_results.duplicate_results
                    }
                })
                
            logger.info(f"Análisis {analysis_id} completado exitosamente")
        else:
            # En caso de error, marcar como fallido
            if analysis_id in _IN_MEMORY_ANALYSES:
                _IN_MEMORY_ANALYSES[analysis_id].update({
                    "status": "FAILED",
                    "completed_at": datetime.now().isoformat(),
                    "progress": 0.0,
                    "error": str(result.error)
                })
                
            logger.error(f"Análisis {analysis_id} falló: {result.error}")
            
    except Exception as e:
        logger.exception(f"Error ejecutando análisis {analysis_id}: {str(e)}")
        
        # Marcar como fallido
        if analysis_id in _IN_MEMORY_ANALYSES:
            _IN_MEMORY_ANALYSES[analysis_id].update({
                "status": "FAILED",
                "completed_at": datetime.now().isoformat(),
                "progress": 0.0,
                "error": str(e)
            })


@router.get("/projects/{project_id}/analysis/latest", response_model=Optional[AnalysisResponseDTO])
async def get_latest_analysis(project_id: str):
    """
    Obtener el análisis más reciente de un proyecto.
    
    Args:
        project_id: ID del proyecto
        
    Returns:
        El análisis más reciente o None si no hay análisis
    """
    try:
        logger.info(f"Obteniendo último análisis para proyecto {project_id}")
        
        # Buscar análisis del proyecto ordenados por fecha de creación
        project_analyses = [
            analysis for analysis_id, analysis in _IN_MEMORY_ANALYSES.items()
            if analysis["project_id"] == project_id
        ]
        
        if not project_analyses:
            logger.info(f"No hay análisis para el proyecto {project_id}")
            return None
            
        # Ordenar por fecha de creación (más reciente primero)
        sorted_analyses = sorted(
            project_analyses,
            key=lambda a: a["created_at"],
            reverse=True
        )
        
        latest = sorted_analyses[0]
        logger.info(f"Análisis más reciente para proyecto {project_id}: {latest['id']}")
        
        return AnalysisResponseDTO(**latest)
        
    except Exception as e:
        logger.exception(f"Error al obtener último análisis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener último análisis: {str(e)}"
        )
