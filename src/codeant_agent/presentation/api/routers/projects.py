"""
Router para endpoints de gestión de proyectos.
"""
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from ....domain.entities.project import Project, ProjectSettings, ProjectMetadata
from ....domain.value_objects.project_id import ProjectId
from ....domain.value_objects.repository_type import RepositoryType, ProjectStatus
from ....application.use_cases.create_project_use_case import (
    CreateProjectUseCase,
    CreateProjectRequest,
    CreateProjectResponse
)
from ....application.use_cases.get_projects_use_case import GetProjectsUseCase
from ....infrastructure.repositories.postgresql_project_repository import PostgreSQLProjectRepository
from ....infrastructure.vcs.git_handler import GitHandler
from ....utils.error import Result
from ....utils.logging import get_logger

logger = get_logger(__name__)

# Crear router - usando /api en lugar de /api/v1 para que coincida con las llamadas del frontend
router = APIRouter(prefix="/api", tags=["projects"])


# DTOs para requests y responses
class CreateProjectRequestDTO(BaseModel):
    """DTO para crear un proyecto."""
    name: str = Field(..., min_length=1, max_length=255, description="Nombre del proyecto")
    slug: str = Field(..., min_length=3, max_length=100, description="Slug único del proyecto")
    description: Optional[str] = Field(None, max_length=1000, description="Descripción del proyecto")
    repository_url: str = Field(..., description="URL del repositorio")
    repository_type: RepositoryType = Field(default=RepositoryType.GIT, description="Tipo de repositorio")
    default_branch: str = Field(default="main", description="Rama principal")
    settings: Optional[dict] = Field(default_factory=dict, description="Configuración del proyecto")
    credentials: Optional[dict] = Field(None, description="Credenciales para el repositorio")


class ProjectResponseDTO(BaseModel):
    """DTO para respuesta de proyecto."""
    id: str
    name: str
    slug: str
    description: Optional[str]
    repository_url: str
    repository_type: str
    default_branch: str
    status: str
    created_at: str
    updated_at: str
    metadata: dict
    settings: dict

    @classmethod
    def from_entity(cls, project: Project) -> "ProjectResponseDTO":
        """Crear DTO desde entidad de dominio."""
        return cls(
            id=str(project.id),
            name=project.name,
            slug=project.slug,
            description=project.description,
            repository_url=getattr(project, 'repository_url', ''),
            repository_type=getattr(project, 'repository_type', RepositoryType.GIT).value,
            default_branch=getattr(project, 'default_branch', 'main'),
            status=project.status.value,
            created_at=project.created_at.isoformat(),
            updated_at=project.updated_at.isoformat(),
            metadata={
                'stars': project.metadata.stars,
                'forks': project.metadata.forks,
                'language_stats': project.metadata.language_stats,
                'topics': project.metadata.topics,
                'license': project.metadata.license,
                'homepage': project.metadata.homepage
            },
            settings={
                'analysis_config': project.settings.analysis_config,
                'ignore_patterns': project.settings.ignore_patterns,
                'include_patterns': project.settings.include_patterns,
                'max_file_size_mb': project.settings.max_file_size_mb,
                'enable_incremental_analysis': project.settings.enable_incremental_analysis
            }
        )


# Dependencias
def get_project_repository():
    """Obtener repositorio de proyectos."""
    # En un entorno real, esto vendría de un contenedor de dependencias
    # Por ahora, creamos una instancia simple con una implementación Mock
    class MockProjectRepository:
        async def save(self, project):
            logger.info(f"Guardar proyecto simulado: {project.name}")
            return Result.success(project)
            
        async def find_by_id(self, project_id):
            logger.info(f"Buscar proyecto simulado por ID: {project_id}")
            # Crear proyecto de ejemplo
            return Result.success(None)
            
        async def find_all(self, skip=0, limit=100):
            logger.info(f"Listar proyectos simulados: skip={skip}, limit={limit}")
            # Devolver lista vacía
            return Result.success([])
            
    return MockProjectRepository()


def get_git_handler():
    """Obtener handler de Git."""
    # Crear un handler simulado
    class MockGitHandler:
        async def clone_repository(self, url, local_path, branch='main', credentials=None):
            logger.info(f"Clonar repositorio simulado: {url} -> {local_path}")
            return Result.success({
                "url": url,
                "local_path": local_path,
                "branch": branch,
                "commit_hash": "1234567890abcdef",
                "commit_date": "2025-09-02T00:00:00Z"
            })
    
    return MockGitHandler()


def get_create_project_use_case(
    project_repo=Depends(get_project_repository),
    git_handler=Depends(get_git_handler)
) -> CreateProjectUseCase:
    """Obtener caso de uso para crear proyectos."""
    return CreateProjectUseCase(
        project_repository=project_repo,
        repository_repository=None,  # TODO: Implementar
        git_handler=git_handler,
        base_repository_path="/tmp/repositories"
    )


def get_projects_use_case(
    project_repo=Depends(get_project_repository)
) -> GetProjectsUseCase:
    """Obtener caso de uso para obtener proyectos."""
    return GetProjectsUseCase(project_repository=project_repo)


# Endpoints
@router.get("/projects", response_model=List[ProjectResponseDTO])
async def get_projects(
    skip: int = 0,
    limit: int = 100,
    use_case: GetProjectsUseCase = Depends(get_projects_use_case)
):
    """
    Obtener lista de proyectos.
    
    Args:
        skip: Número de proyectos a saltar
        limit: Número máximo de proyectos a devolver
        use_case: Caso de uso para obtener proyectos
    
    Returns:
        Lista de proyectos
    """
    try:
        logger.info("Obteniendo lista de proyectos", skip=skip, limit=limit)
        
        result = await use_case.execute(skip=skip, limit=limit)
        
        if not result.success:
            logger.error("Error obteniendo proyectos", error=str(result.error))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error obteniendo proyectos: {result.error}"
            )
        
        projects = result.data
        logger.info("Proyectos obtenidos exitosamente", count=len(projects))
        
        return [ProjectResponseDTO.from_entity(project) for project in projects]
        
    except Exception as e:
        logger.exception("Error inesperado obteniendo proyectos")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )


@router.post("/projects", response_model=ProjectResponseDTO, status_code=status.HTTP_201_CREATED)
async def create_project(
    request: CreateProjectRequestDTO,
    use_case: CreateProjectUseCase = Depends(get_create_project_use_case)
):
    """
    Crear un nuevo proyecto.
    
    Args:
        request: Datos del proyecto a crear
        use_case: Caso de uso para crear proyectos
    
    Returns:
        Proyecto creado
    """
    try:
        logger.info("Creando nuevo proyecto", name=request.name, slug=request.slug)
        
        # Convertir DTO a request del caso de uso
        create_request = CreateProjectRequest(
            name=request.name,
            repository_url=request.repository_url,
            description=request.description,
            repository_type=request.repository_type,
            default_branch=request.default_branch,
            settings=ProjectSettings(**request.settings) if request.settings else None,
            credentials=request.credentials
        )
        
        result = await use_case.execute(create_request)
        
        if not result.success:
            logger.error("Error creando proyecto", error=str(result.error))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error creando proyecto: {result.error}"
            )
        
        project = result.data.project
        logger.info("Proyecto creado exitosamente", project_id=str(project.id))
        
        return ProjectResponseDTO.from_entity(project)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error inesperado creando proyecto")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )


@router.get("/projects/{project_id}", response_model=ProjectResponseDTO)
async def get_project(
    project_id: str,
    use_case: GetProjectsUseCase = Depends(get_projects_use_case)
):
    """
    Obtener un proyecto específico por ID.
    
    Args:
        project_id: ID del proyecto
        use_case: Caso de uso para obtener proyectos
    
    Returns:
        Proyecto encontrado
    """
    try:
        logger.info("Obteniendo proyecto", project_id=project_id)
        
        # Convertir string a ProjectId
        try:
            project_uuid = ProjectId(project_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ID de proyecto inválido"
            )
        
        result = await use_case.get_by_id(project_uuid)
        
        if not result.success:
            if "not found" in str(result.error).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Proyecto no encontrado"
                )
            else:
                logger.error("Error obteniendo proyecto", error=str(result.error))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error obteniendo proyecto: {result.error}"
                )
        
        project = result.data
        logger.info("Proyecto obtenido exitosamente", project_id=project_id)
        
        return ProjectResponseDTO.from_entity(project)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error inesperado obteniendo proyecto")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    use_case: GetProjectsUseCase = Depends(get_projects_use_case)
):
    """
    Eliminar un proyecto.
    
    Args:
        project_id: ID del proyecto a eliminar
        use_case: Caso de uso para obtener proyectos
    """
    try:
        logger.info("Eliminando proyecto", project_id=project_id)
        
        # Convertir string a ProjectId
        try:
            project_uuid = ProjectId(project_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ID de proyecto inválido"
            )
        
        # TODO: Implementar caso de uso para eliminar proyectos
        # Por ahora, solo logueamos la intención
        logger.warning("Eliminación de proyecto no implementada aún", project_id=project_id)
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Eliminación de proyectos no implementada aún"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error inesperado eliminando proyecto")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )
