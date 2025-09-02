"""
Caso de uso para sincronizar proyectos.
"""
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from codeant_agent.domain.entities.project import Project
from codeant_agent.domain.entities.repository import Repository
from codeant_agent.domain.value_objects.project_id import ProjectId
from codeant_agent.domain.value_objects.repository_type import SyncStatus
from codeant_agent.domain.repositories.project_repository import ProjectRepository
from codeant_agent.domain.repositories.repository_repository import RepositoryRepository
from codeant_agent.infrastructure.vcs.git_handler import GitHandler, CommitInfo
from codeant_agent.utils.error import Result, BaseError, NotFoundError
from codeant_agent.utils.logging import get_logger

logger = get_logger(__name__)


class SyncProjectError(BaseError):
    """Error al sincronizar un proyecto."""
    pass


class ProjectNotFoundError(SyncProjectError):
    """Error cuando el proyecto no existe."""
    pass


@dataclass
class SyncProjectRequest:
    """Request para sincronizar un proyecto."""
    project_id: ProjectId
    force: bool = False


@dataclass
class SyncProjectResponse:
    """Response de la sincronización de un proyecto."""
    project: Project
    repository: Repository
    new_commits: List[CommitInfo]
    sync_status: SyncStatus
    sync_duration_seconds: float


class SyncProjectUseCase:
    """
    Caso de uso para sincronizar un proyecto.
    
    Este caso de uso se encarga de:
    1. Obtener el proyecto y su repositorio
    2. Verificar que el proyecto esté activo
    3. Sincronizar el repositorio Git
    4. Actualizar el estado de sincronización
    5. Retornar información sobre los cambios
    """
    
    def __init__(
        self,
        project_repository: ProjectRepository,
        repository_repository: RepositoryRepository,
        git_handler: GitHandler
    ):
        """
        Inicializar el caso de uso.
        
        Args:
            project_repository: Repositorio de proyectos
            repository_repository: Repositorio de repositorios
            git_handler: Handler de Git
        """
        self.project_repository = project_repository
        self.repository_repository = repository_repository
        self.git_handler = git_handler
    
    async def execute(self, request: SyncProjectRequest) -> Result[SyncProjectResponse, Exception]:
        """
        Ejecutar el caso de uso.
        
        Args:
            request: Datos para sincronizar el proyecto
            
        Returns:
            Result con la respuesta de sincronización o error
        """
        try:
            logger.info(f"Sincronizando proyecto: {request.project_id}")
            start_time = datetime.utcnow()
            
            # 1. Obtener el proyecto
            project_result = await self._get_project(request.project_id)
            if not project_result.success:
                return project_result
            
            project = project_result.data
            
            # 2. Verificar que el proyecto esté activo
            if not project.is_active():
                return Result.failure(
                    SyncProjectError(f"El proyecto {project.name} no está activo")
                )
            
            # 3. Obtener el repositorio
            repository_result = await self._get_repository(project.id)
            if not repository_result.success:
                return repository_result
            
            repository = repository_result.data
            
            # 4. Actualizar estado de sincronización
            repository.update_sync_status(SyncStatus.IN_PROGRESS)
            await self.repository_repository.save(repository)
            
            # 5. Sincronizar el repositorio
            sync_result = await self._sync_repository(repository)
            if not sync_result.success:
                # Marcar como fallido
                repository.update_sync_status(SyncStatus.FAILED)
                await self.repository_repository.save(repository)
                return sync_result
            
            new_commits = sync_result.data
            
            # 6. Actualizar información del repositorio
            await self._update_repository_info(repository)
            
            # 7. Marcar como completado
            repository.update_sync_status(SyncStatus.COMPLETED)
            await self.repository_repository.save(repository)
            
            # 8. Actualizar el proyecto
            project.mark_as_analyzed()
            await self.project_repository.save(project)
            
            sync_duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Proyecto sincronizado exitosamente: {project.id} ({len(new_commits)} commits nuevos)")
            
            response = SyncProjectResponse(
                project=project,
                repository=repository,
                new_commits=new_commits,
                sync_status=SyncStatus.COMPLETED,
                sync_duration_seconds=sync_duration
            )
            
            return Result.success(response)
            
        except Exception as e:
            logger.error(f"Error al sincronizar proyecto {request.project_id}: {str(e)}")
            return Result.failure(SyncProjectError(f"Error al sincronizar proyecto: {str(e)}"))
    
    async def _get_project(self, project_id: ProjectId) -> Result[Project, Exception]:
        """Obtener el proyecto por ID."""
        try:
            project_result = await self.project_repository.find_by_id(project_id)
            if not project_result.success:
                return Result.failure(SyncProjectError(f"Error al obtener proyecto: {project_result.error}"))
            
            if not project_result.data:
                return Result.failure(ProjectNotFoundError(f"Proyecto no encontrado: {project_id}"))
            
            return Result.success(project_result.data)
            
        except Exception as e:
            return Result.failure(SyncProjectError(f"Error al obtener proyecto: {str(e)}"))
    
    async def _get_repository(self, project_id: ProjectId) -> Result[Repository, Exception]:
        """Obtener el repositorio del proyecto."""
        try:
            repository_result = await self.repository_repository.find_by_project_id(project_id)
            if not repository_result.success:
                return Result.failure(SyncProjectError(f"Error al obtener repositorio: {repository_result.error}"))
            
            if not repository_result.data:
                return Result.failure(SyncProjectError(f"Repositorio no encontrado para el proyecto: {project_id}"))
            
            return Result.success(repository_result.data)
            
        except Exception as e:
            return Result.failure(SyncProjectError(f"Error al obtener repositorio: {str(e)}"))
    
    async def _sync_repository(self, repository: Repository) -> Result[List[CommitInfo], Exception]:
        """Sincronizar el repositorio Git."""
        try:
            # Obtener actualizaciones del repositorio
            fetch_result = await self.git_handler.fetch_updates(repository)
            if not fetch_result.success:
                return Result.failure(SyncProjectError(f"Error al obtener actualizaciones: {fetch_result.error}"))
            
            new_commits = fetch_result.data
            
            # Actualizar el commit actual si hay nuevos commits
            if new_commits:
                latest_commit = new_commits[0].hash
                repository.update_current_commit(latest_commit)
            
            return Result.success(new_commits)
            
        except Exception as e:
            return Result.failure(SyncProjectError(f"Error al sincronizar repositorio: {str(e)}"))
    
    async def _update_repository_info(self, repository: Repository) -> None:
        """Actualizar información del repositorio."""
        try:
            # Obtener información de ramas
            branches_result = await self.git_handler.list_branches(repository)
            if branches_result.success:
                repository.update_branches(branches_result.data)
            
            # Calcular estadísticas del repositorio
            await self._calculate_repository_stats(repository)
            
        except Exception as e:
            logger.warning(f"Error al actualizar información del repositorio: {str(e)}")
    
    async def _calculate_repository_stats(self, repository: Repository) -> None:
        """Calcular estadísticas del repositorio."""
        try:
            import os
            
            # Calcular tamaño total del repositorio
            total_size = 0
            file_count = 0
            
            for root, dirs, files in os.walk(repository.local_path):
                # Ignorar directorio .git
                if '.git' in dirs:
                    dirs.remove('.git')
                
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        file_count += 1
                    except OSError:
                        continue
            
            repository.update_stats(total_size, file_count)
            
        except Exception as e:
            logger.warning(f"Error al calcular estadísticas del repositorio: {str(e)}")
