"""
Caso de uso para obtener proyectos.
"""
from typing import List, Optional
from dataclasses import dataclass

from src.codeant_agent.domain.entities.project import Project
from src.codeant_agent.domain.repositories.project_repository import ProjectRepository
from src.codeant_agent.domain.value_objects.project_id import ProjectId
from src.codeant_agent.utils.error import Result, BaseError
from src.codeant_agent.utils.logging import get_logger

logger = get_logger(__name__)


class GetProjectsError(BaseError):
    """Error al obtener proyectos."""
    pass


@dataclass
class GetProjectsRequest:
    """Request para obtener proyectos."""
    skip: int = 0
    limit: int = 100
    organization_id: Optional[str] = None
    status: Optional[str] = None


@dataclass
class GetProjectsResponse:
    """Response de obtener proyectos."""
    projects: List[Project]
    total: int
    skip: int
    limit: int


class GetProjectsUseCase:
    """
    Caso de uso para obtener proyectos.
    
    Este caso de uso se encarga de:
    1. Validar los parámetros de entrada
    2. Obtener proyectos del repositorio
    3. Aplicar filtros si es necesario
    4. Devolver la lista paginada de proyectos
    """
    
    def __init__(self, project_repository: ProjectRepository):
        """
        Inicializar el caso de uso.
        
        Args:
            project_repository: Repositorio de proyectos
        """
        self.project_repository = project_repository
    
    async def execute(self, skip: int = 0, limit: int = 100) -> Result[List[Project], Exception]:
        """
        Ejecutar el caso de uso.
        
        Args:
            skip: Número de proyectos a saltar
            limit: Número máximo de proyectos a devolver
            
        Returns:
            Result con la lista de proyectos o error
        """
        try:
            logger.info(f"Obteniendo proyectos", skip=skip, limit=limit)
            
            # 1. Validar parámetros
            validation_result = self._validate_parameters(skip, limit)
            if not validation_result.success:
                return validation_result
            
            # 2. Obtener proyectos del repositorio
            projects_result = await self.project_repository.find_all(skip=skip, limit=limit)
            if not projects_result.success:
                return Result.failure(GetProjectsError(f"Error al obtener proyectos: {projects_result.error}"))
            
            projects = projects_result.data
            logger.info(f"Proyectos obtenidos exitosamente", count=len(projects))
            
            return Result.success(projects)
            
        except Exception as e:
            logger.error(f"Error al obtener proyectos: {str(e)}")
            return Result.failure(GetProjectsError(f"Error al obtener proyectos: {str(e)}"))
    
    async def get_by_id(self, project_id: ProjectId) -> Result[Project, Exception]:
        """
        Obtener un proyecto por ID.
        
        Args:
            project_id: ID del proyecto
            
        Returns:
            Result con el proyecto o error
        """
        try:
            logger.info(f"Obteniendo proyecto por ID", project_id=str(project_id))
            
            result = await self.project_repository.find_by_id(project_id)
            if not result.success:
                return Result.failure(GetProjectsError(f"Error al obtener proyecto: {result.error}"))
            
            if not result.data:
                return Result.failure(GetProjectsError(f"Proyecto con ID {project_id} no encontrado"))
            
            project = result.data
            logger.info(f"Proyecto obtenido exitosamente", project_id=str(project_id))
            
            return Result.success(project)
            
        except Exception as e:
            logger.error(f"Error al obtener proyecto por ID: {str(e)}")
            return Result.failure(GetProjectsError(f"Error al obtener proyecto: {str(e)}"))
    
    def _validate_parameters(self, skip: int, limit: int) -> Result[None, Exception]:
        """Validar los parámetros de entrada."""
        if skip < 0:
            return Result.failure(GetProjectsError("El parámetro 'skip' debe ser mayor o igual a 0"))
        
        if limit <= 0:
            return Result.failure(GetProjectsError("El parámetro 'limit' debe ser mayor a 0"))
        
        if limit > 1000:
            return Result.failure(GetProjectsError("El parámetro 'limit' no puede ser mayor a 1000"))
        
        return Result.success(None)
