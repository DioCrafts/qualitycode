"""
Repositorio PostgreSQL para repositorios.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from codeant_agent.domain.entities import Repository, Branch, Tag
from codeant_agent.domain.repositories import RepositoryRepository
from codeant_agent.domain.value_objects import RepositoryId, RepositoryType, SyncStatus
from codeant_agent.infrastructure.database.models import Repository as RepositoryModel
from codeant_agent.infrastructure.database.models import RepositoryBranch as RepositoryBranchModel
from codeant_agent.infrastructure.database.models import Project as ProjectModel
from codeant_agent.utils.error import Result, BaseError
from codeant_agent.utils.logging import get_logger

logger = get_logger(__name__)


class PostgreSQLRepositoryRepository(RepositoryRepository):
    """
    Implementación PostgreSQL del repositorio de repositorios.
    
    Maneja todas las operaciones CRUD y búsquedas para repositorios
    usando SQLAlchemy con PostgreSQL.
    """
    
    def __init__(self, session_factory):
        """
        Inicializar el repositorio.
        
        Args:
            session_factory: Factory para crear sesiones de SQLAlchemy
        """
        self.session_factory = session_factory
    
    async def save(self, repository: Repository) -> Result[Repository, Exception]:
        """Guardar un repositorio en la base de datos."""
        try:
            async with self.session_factory() as session:
                # Verificar si el repositorio ya existe
                existing_repository = await session.execute(
                    select(RepositoryModel).where(RepositoryModel.id == repository.id.value)
                )
                existing_repository = existing_repository.scalar_one_or_none()
                
                if existing_repository:
                    # Actualizar repositorio existente
                    await session.execute(
                        update(RepositoryModel)
                        .where(RepositoryModel.id == repository.id.value)
                        .values(
                            name=repository.name,
                            url=repository.url,
                            type=repository.type,
                            default_branch=repository.default_branch,
                            sync_status=repository.sync_status,
                            size_bytes=repository.size_bytes,
                            file_count=repository.file_count,
                            language_stats=repository.language_stats,
                            settings=repository.settings,
                            updated_at=datetime.utcnow()
                        )
                    )
                    logger.info(f"Repositorio actualizado: {repository.id}")
                else:
                    # Crear nuevo repositorio
                    repository_model = RepositoryModel(
                        id=repository.id.value,
                        project_id=repository.project_id.value,
                        name=repository.name,
                        url=repository.url,
                        type=repository.type,
                        default_branch=repository.default_branch,
                        sync_status=repository.sync_status,
                        size_bytes=repository.size_bytes,
                        file_count=repository.file_count,
                        language_stats=repository.language_stats,
                        settings=repository.settings
                    )
                    session.add(repository_model)
                    logger.info(f"Repositorio creado: {repository.id}")
                
                await session.commit()
                return Result.success(repository)
                
        except Exception as e:
            logger.error(f"Error al guardar repositorio {repository.id}: {str(e)}")
            return Result.failure(BaseError(f"Error al guardar repositorio: {str(e)}"))
    
    async def find_by_id(self, repository_id: RepositoryId) -> Result[Optional[Repository], Exception]:
        """Buscar un repositorio por ID."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(RepositoryModel)
                    .options(selectinload(RepositoryModel.project))
                    .options(selectinload(RepositoryModel.branches))
                    .where(RepositoryModel.id == repository_id.value)
                )
                repository_model = result.scalar_one_or_none()
                
                if repository_model:
                    repository = self._map_to_domain(repository_model)
                    return Result.success(repository)
                else:
                    return Result.success(None)
                    
        except Exception as e:
            logger.error(f"Error al buscar repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar repositorio: {str(e)}"))
    
    async def find_by_project_and_name(self, project_id: str, name: str) -> Result[Optional[Repository], Exception]:
        """Buscar un repositorio por proyecto y nombre."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(RepositoryModel)
                    .options(selectinload(RepositoryModel.project))
                    .options(selectinload(RepositoryModel.branches))
                    .where(
                        and_(
                            RepositoryModel.project_id == project_id,
                            RepositoryModel.name == name
                        )
                    )
                )
                repository_model = result.scalar_one_or_none()
                
                if repository_model:
                    repository = self._map_to_domain(repository_model)
                    return Result.success(repository)
                else:
                    return Result.success(None)
                    
        except Exception as e:
            logger.error(f"Error al buscar repositorio por proyecto {project_id} y nombre {name}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar repositorio: {str(e)}"))
    
    async def find_by_project(self, project_id: str, limit: int = 100, offset: int = 0) -> Result[List[Repository], Exception]:
        """Buscar repositorios por proyecto."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(RepositoryModel)
                    .options(selectinload(RepositoryModel.project))
                    .options(selectinload(RepositoryModel.branches))
                    .where(RepositoryModel.project_id == project_id)
                    .order_by(RepositoryModel.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                repository_models = result.scalars().all()
                
                repositories = [self._map_to_domain(model) for model in repository_models]
                return Result.success(repositories)
                
        except Exception as e:
            logger.error(f"Error al buscar repositorios de proyecto {project_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar repositorios: {str(e)}"))
    
    async def find_by_sync_status(self, sync_status: SyncStatus, limit: int = 100, offset: int = 0) -> Result[List[Repository], Exception]:
        """Buscar repositorios por estado de sincronización."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(RepositoryModel)
                    .options(selectinload(RepositoryModel.project))
                    .options(selectinload(RepositoryModel.branches))
                    .where(RepositoryModel.sync_status == sync_status)
                    .order_by(RepositoryModel.last_sync_at.asc())
                    .limit(limit)
                    .offset(offset)
                )
                repository_models = result.scalars().all()
                
                repositories = [self._map_to_domain(model) for model in repository_models]
                return Result.success(repositories)
                
        except Exception as e:
            logger.error(f"Error al buscar repositorios con estado {sync_status}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar repositorios: {str(e)}"))
    
    async def find_by_url(self, url: str) -> Result[Optional[Repository], Exception]:
        """Buscar un repositorio por URL."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(RepositoryModel)
                    .options(selectinload(RepositoryModel.project))
                    .options(selectinload(RepositoryModel.branches))
                    .where(RepositoryModel.url == url)
                )
                repository_model = result.scalar_one_or_none()
                
                if repository_model:
                    repository = self._map_to_domain(repository_model)
                    return Result.success(repository)
                else:
                    return Result.success(None)
                    
        except Exception as e:
            logger.error(f"Error al buscar repositorio por URL {url}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar repositorio: {str(e)}"))
    
    async def search(self, query: str, project_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> Result[List[Repository], Exception]:
        """Buscar repositorios por texto."""
        try:
            async with self.session_factory() as session:
                # Construir consulta base
                base_query = select(RepositoryModel).options(
                    selectinload(RepositoryModel.project),
                    selectinload(RepositoryModel.branches)
                )
                
                # Agregar filtros
                conditions = [
                    or_(
                        RepositoryModel.name.ilike(f"%{query}%"),
                        RepositoryModel.url.ilike(f"%{query}%")
                    )
                ]
                
                if project_id:
                    conditions.append(RepositoryModel.project_id == project_id)
                
                result = await session.execute(
                    base_query
                    .where(and_(*conditions))
                    .order_by(RepositoryModel.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                repository_models = result.scalars().all()
                
                repositories = [self._map_to_domain(model) for model in repository_models]
                return Result.success(repositories)
                
        except Exception as e:
            logger.error(f"Error al buscar repositorios con query '{query}': {str(e)}")
            return Result.failure(BaseError(f"Error al buscar repositorios: {str(e)}"))
    
    async def count_by_project(self, project_id: str) -> Result[int, Exception]:
        """Contar repositorios de un proyecto."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(RepositoryModel.id)
                    .where(RepositoryModel.project_id == project_id)
                )
                count = len(result.scalars().all())
                return Result.success(count)
                
        except Exception as e:
            logger.error(f"Error al contar repositorios de proyecto {project_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al contar repositorios: {str(e)}"))
    
    async def delete(self, repository_id: RepositoryId) -> Result[bool, Exception]:
        """Eliminar un repositorio."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    delete(RepositoryModel).where(RepositoryModel.id == repository_id.value)
                )
                await session.commit()
                
                deleted = result.rowcount > 0
                if deleted:
                    logger.info(f"Repositorio eliminado: {repository_id}")
                
                return Result.success(deleted)
                
        except Exception as e:
            logger.error(f"Error al eliminar repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al eliminar repositorio: {str(e)}"))
    
    async def update_sync_status(self, repository_id: RepositoryId, sync_status: SyncStatus) -> Result[bool, Exception]:
        """Actualizar el estado de sincronización de un repositorio."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    update(RepositoryModel)
                    .where(RepositoryModel.id == repository_id.value)
                    .values(
                        sync_status=sync_status,
                        last_sync_at=datetime.utcnow() if sync_status == SyncStatus.SYNCED else None,
                        updated_at=datetime.utcnow()
                    )
                )
                await session.commit()
                
                updated = result.rowcount > 0
                if updated:
                    logger.info(f"Estado de sincronización actualizado: {repository_id} -> {sync_status}")
                
                return Result.success(updated)
                
        except Exception as e:
            logger.error(f"Error al actualizar estado de sincronización de repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al actualizar estado de sincronización: {str(e)}"))
    
    async def update_commit_info(self, repository_id: RepositoryId, commit_hash: str, commit_message: str, 
                                commit_author: str, commit_date: datetime) -> Result[bool, Exception]:
        """Actualizar información del último commit."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    update(RepositoryModel)
                    .where(RepositoryModel.id == repository_id.value)
                    .values(
                        last_commit_hash=commit_hash,
                        last_commit_message=commit_message,
                        last_commit_author=commit_author,
                        last_commit_date=commit_date,
                        updated_at=datetime.utcnow()
                    )
                )
                await session.commit()
                
                updated = result.rowcount > 0
                if updated:
                    logger.info(f"Información de commit actualizada: {repository_id} -> {commit_hash}")
                
                return Result.success(updated)
                
        except Exception as e:
            logger.error(f"Error al actualizar información de commit de repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al actualizar información de commit: {str(e)}"))
    
    async def update_stats(self, repository_id: RepositoryId, size_bytes: int, file_count: int, 
                          language_stats: Dict[str, Any]) -> Result[bool, Exception]:
        """Actualizar estadísticas del repositorio."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    update(RepositoryModel)
                    .where(RepositoryModel.id == repository_id.value)
                    .values(
                        size_bytes=size_bytes,
                        file_count=file_count,
                        language_stats=language_stats,
                        updated_at=datetime.utcnow()
                    )
                )
                await session.commit()
                
                updated = result.rowcount > 0
                if updated:
                    logger.info(f"Estadísticas de repositorio actualizadas: {repository_id}")
                
                return Result.success(updated)
                
        except Exception as e:
            logger.error(f"Error al actualizar estadísticas de repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al actualizar estadísticas: {str(e)}"))
    
    def _map_to_domain(self, model: RepositoryModel) -> Repository:
        """Mapear modelo SQLAlchemy a entidad de dominio."""
        from codeant_agent.domain.value_objects import RepositoryId, ProjectId
        
        # Mapear ramas
        branches = []
        for branch_model in model.branches:
            branch = Branch(
                name=branch_model.name,
                commit_hash=branch_model.commit_hash,
                commit_message=branch_model.commit_message,
                commit_author=branch_model.commit_author,
                commit_date=branch_model.commit_date,
                is_default=branch_model.is_default,
                is_protected=branch_model.is_protected
            )
            branches.append(branch)
        
        # Por ahora no mapeamos tags ya que no están en el modelo
        tags = []
        
        return Repository(
            id=RepositoryId(str(model.id)),
            project_id=ProjectId(str(model.project_id)),
            name=model.name,
            url=model.url,
            type=model.type,
            default_branch=model.default_branch,
            sync_status=model.sync_status,
            size_bytes=model.size_bytes,
            file_count=model.file_count,
            language_stats=model.language_stats or {},
            settings=model.settings or {},
            branches=branches,
            tags=tags,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
