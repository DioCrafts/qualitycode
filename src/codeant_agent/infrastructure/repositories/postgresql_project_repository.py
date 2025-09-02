"""
Repositorio PostgreSQL para proyectos.
"""
from dataclasses import asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.codeant_agent.domain.entities.project import Project, ProjectSettings, ProjectMetadata
from src.codeant_agent.domain.repositories.project_repository import ProjectRepository
from src.codeant_agent.domain.value_objects.project_id import ProjectId
from src.codeant_agent.domain.value_objects.repository_type import ProjectStatus
from src.codeant_agent.infrastructure.database.models import Project as ProjectModel
from src.codeant_agent.infrastructure.database.models import Organization as OrganizationModel
from src.codeant_agent.infrastructure.database.models import User as UserModel
from src.codeant_agent.utils.error import Result, BaseError
from src.codeant_agent.utils.logging import get_logger

logger = get_logger(__name__)


class PostgreSQLProjectRepository(ProjectRepository):
    """
    Implementación PostgreSQL del repositorio de proyectos.
    
    Maneja todas las operaciones CRUD y búsquedas para proyectos
    usando SQLAlchemy con PostgreSQL.
    """
    
    def __init__(self, session_factory):
        """
        Inicializar el repositorio.
        
        Args:
            session_factory: Factory para crear sesiones de SQLAlchemy
        """
        self.session_factory = session_factory
    
    async def save(self, project: Project) -> Result[Project, Exception]:
        """Guardar un proyecto en la base de datos."""
        try:
            async with self.session_factory() as session:
                # Verificar si el proyecto ya existe
                existing_project = await session.execute(
                    select(ProjectModel).where(ProjectModel.id == project.id.value)
                )
                existing_project = existing_project.scalar_one_or_none()
                
                if existing_project:
                    # Actualizar proyecto existente
                    await session.execute(
                        update(ProjectModel)
                        .where(ProjectModel.id == project.id.value)
                        .values(
                            name=project.name,
                            description=project.description,
                            slug=project.slug,
                            status=project.status,
                            settings=asdict(project.settings),
                            metadata_json=asdict(project.metadata),
                            updated_at=datetime.utcnow()
                        )
                    )
                    logger.info(f"Proyecto actualizado: {project.id}")
                else:
                    # Crear nuevo proyecto
                    project_model = ProjectModel(
                        id=project.id.value,
                        organization_id=project.organization_id.value,
                        name=project.name,
                        description=project.description,
                        slug=project.slug,
                        status=project.status,
                        settings=asdict(project.settings),
                        metadata_json=asdict(project.metadata),
                        created_by=project.created_by.value
                    )
                    session.add(project_model)
                    logger.info(f"Proyecto creado: {project.id}")
                
                await session.commit()
                return Result.success(project)
                
        except Exception as e:
            logger.error(f"Error al guardar proyecto {project.id}: {str(e)}")
            return Result.failure(BaseError(f"Error al guardar proyecto: {str(e)}"))
    
    async def find_by_id(self, project_id: ProjectId) -> Result[Optional[Project], Exception]:
        """Buscar un proyecto por ID."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(ProjectModel)
                    .options(selectinload(ProjectModel.organization))
                    .where(ProjectModel.id == project_id.value)
                )
                project_model = result.scalar_one_or_none()
                
                if project_model:
                    project = self._map_to_domain(project_model)
                    return Result.success(project)
                else:
                    return Result.success(None)
                    
        except Exception as e:
            logger.error(f"Error al buscar proyecto {project_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar proyecto: {str(e)}"))
    
    async def find_by_organization_and_slug(self, organization_id: str, slug: str) -> Result[Optional[Project], Exception]:
        """Buscar un proyecto por organización y slug."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(ProjectModel)
                    .options(selectinload(ProjectModel.organization))
                    .where(
                        and_(
                            ProjectModel.organization_id == organization_id,
                            ProjectModel.slug == slug
                        )
                    )
                )
                project_model = result.scalar_one_or_none()
                
                if project_model:
                    project = self._map_to_domain(project_model)
                    return Result.success(project)
                else:
                    return Result.success(None)
                    
        except Exception as e:
            logger.error(f"Error al buscar proyecto por org {organization_id} y slug {slug}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar proyecto: {str(e)}"))
    
    async def find_by_organization(self, organization_id: str, limit: int = 100, offset: int = 0) -> Result[List[Project], Exception]:
        """Buscar proyectos por organización."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(ProjectModel)
                    .options(selectinload(ProjectModel.organization))
                    .where(ProjectModel.organization_id == organization_id)
                    .order_by(ProjectModel.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                project_models = result.scalars().all()
                
                projects = [self._map_to_domain(model) for model in project_models]
                return Result.success(projects)
                
        except Exception as e:
            logger.error(f"Error al buscar proyectos de organización {organization_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar proyectos: {str(e)}"))
    
    async def find_by_status(self, status: ProjectStatus, limit: int = 100, offset: int = 0) -> Result[List[Project], Exception]:
        """Buscar proyectos por estado."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(ProjectModel)
                    .options(selectinload(ProjectModel.organization))
                    .where(ProjectModel.status == status)
                    .order_by(ProjectModel.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                project_models = result.scalars().all()
                
                projects = [self._map_to_domain(model) for model in project_models]
                return Result.success(projects)
                
        except Exception as e:
            logger.error(f"Error al buscar proyectos con estado {status}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar proyectos: {str(e)}"))
    
    async def search(self, query: str, organization_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> Result[List[Project], Exception]:
        """Buscar proyectos por texto."""
        try:
            async with self.session_factory() as session:
                # Construir consulta base
                base_query = select(ProjectModel).options(selectinload(ProjectModel.organization))
                
                # Agregar filtros
                conditions = [
                    or_(
                        ProjectModel.name.ilike(f"%{query}%"),
                        ProjectModel.description.ilike(f"%{query}%"),
                        ProjectModel.slug.ilike(f"%{query}%")
                    )
                ]
                
                if organization_id:
                    conditions.append(ProjectModel.organization_id == organization_id)
                
                result = await session.execute(
                    base_query
                    .where(and_(*conditions))
                    .order_by(ProjectModel.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                project_models = result.scalars().all()
                
                projects = [self._map_to_domain(model) for model in project_models]
                return Result.success(projects)
                
        except Exception as e:
            logger.error(f"Error al buscar proyectos con query '{query}': {str(e)}")
            return Result.failure(BaseError(f"Error al buscar proyectos: {str(e)}"))
    
    async def count_by_organization(self, organization_id: str) -> Result[int, Exception]:
        """Contar proyectos de una organización."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(ProjectModel.id)
                    .where(ProjectModel.organization_id == organization_id)
                )
                count = len(result.scalars().all())
                return Result.success(count)
                
        except Exception as e:
            logger.error(f"Error al contar proyectos de organización {organization_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al contar proyectos: {str(e)}"))
    
    async def delete(self, project_id: ProjectId) -> Result[bool, Exception]:
        """Eliminar un proyecto."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    delete(ProjectModel).where(ProjectModel.id == project_id.value)
                )
                await session.commit()
                
                deleted = result.rowcount > 0
                if deleted:
                    logger.info(f"Proyecto eliminado: {project_id}")
                
                return Result.success(deleted)
                
        except Exception as e:
            logger.error(f"Error al eliminar proyecto {project_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al eliminar proyecto: {str(e)}"))
    
    async def update_status(self, project_id: ProjectId, status: ProjectStatus) -> Result[bool, Exception]:
        """Actualizar el estado de un proyecto."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    update(ProjectModel)
                    .where(ProjectModel.id == project_id.value)
                    .values(
                        status=status,
                        updated_at=datetime.utcnow()
                    )
                )
                await session.commit()
                
                updated = result.rowcount > 0
                if updated:
                    logger.info(f"Estado de proyecto actualizado: {project_id} -> {status}")
                
                return Result.success(updated)
                
        except Exception as e:
            logger.error(f"Error al actualizar estado de proyecto {project_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al actualizar estado: {str(e)}"))
    
    async def update_settings(self, project_id: ProjectId, settings: ProjectSettings) -> Result[bool, Exception]:
        """Actualizar la configuración de un proyecto."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    update(ProjectModel)
                    .where(ProjectModel.id == project_id.value)
                    .values(
                        settings=settings.dict(),
                        updated_at=datetime.utcnow()
                    )
                )
                await session.commit()
                
                updated = result.rowcount > 0
                if updated:
                    logger.info(f"Configuración de proyecto actualizada: {project_id}")
                
                return Result.success(updated)
                
        except Exception as e:
            logger.error(f"Error al actualizar configuración de proyecto {project_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al actualizar configuración: {str(e)}"))
    
    async def update_metadata(self, project_id: ProjectId, metadata: ProjectMetadata) -> Result[bool, Exception]:
        """Actualizar los metadatos de un proyecto."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    update(ProjectModel)
                    .where(ProjectModel.id == project_id.value)
                    .values(
                        metadata_json=metadata.dict(),
                        updated_at=datetime.utcnow()
                    )
                )
                await session.commit()
                
                updated = result.rowcount > 0
                if updated:
                    logger.info(f"Metadatos de proyecto actualizados: {project_id}")
                
                return Result.success(updated)
                
        except Exception as e:
            logger.error(f"Error al actualizar metadatos de proyecto {project_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al actualizar metadatos: {str(e)}"))
    
    def _map_to_domain(self, model: ProjectModel) -> Project:
        """Mapear modelo SQLAlchemy a entidad de dominio."""
        from codeant_agent.domain.value_objects import ProjectId, OrganizationId, UserId
        
        settings = ProjectSettings(**model.settings) if model.settings else ProjectSettings()
        metadata = ProjectMetadata(**model.metadata_json) if model.metadata_json else ProjectMetadata()
        
        return Project(
            id=ProjectId(str(model.id)),
            organization_id=OrganizationId(model.organization_id),
            name=model.name,
            description=model.description,
            slug=model.slug,
            status=model.status,
            settings=settings,
            metadata=metadata,
            created_by=UserId(model.created_by),
            created_at=model.created_at,
            updated_at=model.updated_at
        )
