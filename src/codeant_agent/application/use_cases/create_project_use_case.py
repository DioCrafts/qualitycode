"""
Caso de uso para crear proyectos.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from codeant_agent.domain.entities.project import Project, ProjectSettings
from codeant_agent.domain.value_objects.repository_type import RepositoryType
from codeant_agent.domain.repositories.project_repository import ProjectRepository
from codeant_agent.domain.repositories.repository_repository import RepositoryRepository
from codeant_agent.infrastructure.vcs.git_handler import GitHandler
from codeant_agent.utils.error import Result, BaseError, ValidationError
from codeant_agent.utils.logging import get_logger

logger = get_logger(__name__)


class CreateProjectError(BaseError):
    """Error al crear un proyecto."""
    pass


class ProjectAlreadyExistsError(CreateProjectError):
    """Error cuando el proyecto ya existe."""
    pass


@dataclass
class CreateProjectRequest:
    """Request para crear un proyecto."""
    name: str
    repository_url: str
    description: Optional[str] = None
    repository_type: RepositoryType = RepositoryType.GIT
    default_branch: str = "main"
    settings: Optional[ProjectSettings] = None
    credentials: Optional[dict] = None


@dataclass
class CreateProjectResponse:
    """Response de la creación de un proyecto."""
    project: Project
    repository_created: bool
    repository_path: Optional[str] = None


class CreateProjectUseCase:
    """
    Caso de uso para crear un nuevo proyecto.
    
    Este caso de uso se encarga de:
    1. Validar los datos de entrada
    2. Verificar que el proyecto no exista
    3. Crear la entidad Project
    4. Clonar el repositorio si es necesario
    5. Guardar el proyecto en el repositorio
    """
    
    def __init__(
        self,
        project_repository: ProjectRepository,
        repository_repository: RepositoryRepository,
        git_handler: GitHandler,
        base_repository_path: str = "/tmp/repositories"
    ):
        """
        Inicializar el caso de uso.
        
        Args:
            project_repository: Repositorio de proyectos
            repository_repository: Repositorio de repositorios
            git_handler: Handler de Git
            base_repository_path: Ruta base para los repositorios
        """
        self.project_repository = project_repository
        self.repository_repository = repository_repository
        self.git_handler = git_handler
        self.base_repository_path = base_repository_path
    
    async def execute(self, request: CreateProjectRequest) -> Result[CreateProjectResponse, Exception]:
        """
        Ejecutar el caso de uso.
        
        Args:
            request: Datos para crear el proyecto
            
        Returns:
            Result con la respuesta de creación o error
        """
        try:
            logger.info(f"Creando proyecto: {request.name}")
            
            # 1. Validar request
            validation_result = self._validate_request(request)
            if not validation_result.success:
                return validation_result
            
            # 2. Verificar que el proyecto no exista
            existing_project_result = await self._check_existing_project(request)
            if not existing_project_result.success:
                return existing_project_result
            
            if existing_project_result.data:
                return Result.failure(
                    ProjectAlreadyExistsError(f"Ya existe un proyecto con la URL: {request.repository_url}")
                )
            
            # 3. Crear la entidad Project
            project = self._create_project_entity(request)
            
            # 4. Clonar el repositorio si es Git
            repository_created = False
            repository_path = None
            
            if request.repository_type == RepositoryType.GIT:
                clone_result = await self._clone_repository(project, request)
                if not clone_result.success:
                    return clone_result
                
                repository_created = True
                repository_path = clone_result.data.local_path
            
            # 5. Guardar el proyecto
            save_result = await self.project_repository.save(project)
            if not save_result.success:
                return Result.failure(CreateProjectError(f"Error al guardar proyecto: {save_result.error}"))
            
            logger.info(f"Proyecto creado exitosamente: {project.id}")
            
            response = CreateProjectResponse(
                project=project,
                repository_created=repository_created,
                repository_path=repository_path
            )
            
            return Result.success(response)
            
        except Exception as e:
            logger.error(f"Error al crear proyecto {request.name}: {str(e)}")
            return Result.failure(CreateProjectError(f"Error al crear proyecto: {str(e)}"))
    
    def _validate_request(self, request: CreateProjectRequest) -> Result[None, Exception]:
        """Validar los datos de la request."""
        if not request.name or not request.name.strip():
            return Result.failure(ValidationError("El nombre del proyecto es requerido"))
        
        if not request.repository_url or not request.repository_url.strip():
            return Result.failure(ValidationError("La URL del repositorio es requerida"))
        
        if not request.default_branch or not request.default_branch.strip():
            return Result.failure(ValidationError("La rama por defecto es requerida"))
        
        return Result.success(None)
    
    async def _check_existing_project(self, request: CreateProjectRequest) -> Result[Optional[Project], Exception]:
        """Verificar si ya existe un proyecto con la misma URL."""
        try:
            existing_result = await self.project_repository.find_by_repository_url(request.repository_url)
            if not existing_result.success:
                return Result.failure(CreateProjectError(f"Error al verificar proyecto existente: {existing_result.error}"))
            
            return Result.success(existing_result.data)
            
        except Exception as e:
            return Result.failure(CreateProjectError(f"Error al verificar proyecto existente: {str(e)}"))
    
    def _create_project_entity(self, request: CreateProjectRequest) -> Project:
        """Crear la entidad Project."""
        settings = request.settings or ProjectSettings()
        
        return Project.create(
            name=request.name.strip(),
            repository_url=request.repository_url.strip(),
            description=request.description.strip() if request.description else None,
            repository_type=request.repository_type,
            default_branch=request.default_branch.strip()
        )
    
    async def _clone_repository(self, project: Project, request: CreateProjectRequest) -> Result[Repository, Exception]:
        """Clonar el repositorio Git."""
        try:
            # Crear ruta para el repositorio
            repo_name = self._extract_repo_name(request.repository_url)
            repo_path = f"{self.base_repository_path}/{project.id.value}/{repo_name}"
            
            # Clonar el repositorio
            clone_result = await self.git_handler.clone_repository(
                url=request.repository_url,
                target_path=repo_path,
                credentials=request.credentials
            )
            
            if not clone_result.success:
                return Result.failure(CreateProjectError(f"Error al clonar repositorio: {clone_result.error}"))
            
            repository = clone_result.data
            repository.project_id = project.id
            
            # Guardar el repositorio
            save_repo_result = await self.repository_repository.save(repository)
            if not save_repo_result.success:
                return Result.failure(CreateProjectError(f"Error al guardar repositorio: {save_repo_result.error}"))
            
            return Result.success(repository)
            
        except Exception as e:
            return Result.failure(CreateProjectError(f"Error al clonar repositorio: {str(e)}"))
    
    def _extract_repo_name(self, repository_url: str) -> str:
        """Extraer el nombre del repositorio de la URL."""
        # Remover extensión .git si existe
        if repository_url.endswith('.git'):
            repository_url = repository_url[:-4]
        
        # Obtener la última parte de la URL
        parts = repository_url.split('/')
        return parts[-1] if parts else "repository"
