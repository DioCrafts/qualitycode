"""
Repositorio PostgreSQL para índice de archivos.
"""
from dataclasses import asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from codeant_agent.domain.entities import FileIndex, FileMetadata
from codeant_agent.domain.repositories import FileIndexRepository
from codeant_agent.domain.value_objects import FileId, RepositoryId
from codeant_agent.infrastructure.database.models import FileIndex as FileIndexModel
from codeant_agent.infrastructure.database.models import Repository as RepositoryModel
from codeant_agent.utils.error import Result, BaseError
from codeant_agent.utils.logging import get_logger

logger = get_logger(__name__)


class PostgreSQLFileIndexRepository(FileIndexRepository):
    """
    Implementación PostgreSQL del repositorio de índice de archivos.
    
    Maneja todas las operaciones CRUD y búsquedas para archivos indexados
    usando SQLAlchemy con PostgreSQL.
    """
    
    def __init__(self, session_factory):
        """
        Inicializar el repositorio.
        
        Args:
            session_factory: Factory para crear sesiones de SQLAlchemy
        """
        self.session_factory = session_factory
    
    async def save(self, file_index: FileIndex) -> Result[FileIndex, Exception]:
        """Guardar un archivo indexado en la base de datos."""
        try:
            async with self.session_factory() as session:
                # Verificar si el archivo ya existe
                existing_file = await session.execute(
                    select(FileIndexModel).where(
                        and_(
                            FileIndexModel.repository_id == file_index.repository_id.value,
                            FileIndexModel.file_path == file_index.file_path,
                            FileIndexModel.commit_hash == file_index.commit_hash
                        )
                    )
                )
                existing_file = existing_file.scalar_one_or_none()
                
                if existing_file:
                    # Actualizar archivo existente
                    await session.execute(
                        update(FileIndexModel)
                        .where(FileIndexModel.id == existing_file.id)
                        .values(
                            file_name=file_index.file_name,
                            file_extension=file_index.file_extension,
                            language=file_index.language,
                            mime_type=file_index.mime_type,
                            size_bytes=file_index.size_bytes,
                            line_count=file_index.line_count,
                            branch_name=file_index.branch_name,
                            is_binary=file_index.is_binary,
                            is_ignored=file_index.is_ignored,
                            metadata_json=asdict(file_index.metadata),
                            updated_at=datetime.utcnow()
                        )
                    )
                    logger.info(f"Archivo indexado actualizado: {file_index.file_path}")
                else:
                    # Crear nuevo archivo indexado
                    file_model = FileIndexModel(
                        repository_id=file_index.repository_id.value,
                        file_path=file_index.file_path,
                        file_name=file_index.file_name,
                        file_extension=file_index.file_extension,
                        language=file_index.language,
                        mime_type=file_index.mime_type,
                        size_bytes=file_index.size_bytes,
                        line_count=file_index.line_count,
                        commit_hash=file_index.commit_hash,
                        branch_name=file_index.branch_name,
                        is_binary=file_index.is_binary,
                        is_ignored=file_index.is_ignored,
                        metadata_json=asdict(file_index.metadata)
                    )
                    session.add(file_model)
                    logger.info(f"Archivo indexado creado: {file_index.file_path}")
                
                await session.commit()
                return Result.success(file_index)
                
        except Exception as e:
            logger.error(f"Error al guardar archivo indexado {file_index.file_path}: {str(e)}")
            return Result.failure(BaseError(f"Error al guardar archivo indexado: {str(e)}"))
    
    async def find_by_id(self, file_id: FileId) -> Result[Optional[FileIndex], Exception]:
        """Buscar un archivo indexado por ID."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(FileIndexModel)
                    .options(selectinload(FileIndexModel.repository))
                    .where(FileIndexModel.id == file_id.value)
                )
                file_model = result.scalar_one_or_none()
                
                if file_model:
                    file_index = self._map_to_domain(file_model)
                    return Result.success(file_index)
                else:
                    return Result.success(None)
                    
        except Exception as e:
            logger.error(f"Error al buscar archivo indexado {file_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar archivo indexado: {str(e)}"))
    
    async def find_by_repository_and_path(self, repository_id: RepositoryId, file_path: str, 
                                         commit_hash: str) -> Result[Optional[FileIndex], Exception]:
        """Buscar un archivo por repositorio, ruta y commit."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(FileIndexModel)
                    .options(selectinload(FileIndexModel.repository))
                    .where(
                        and_(
                            FileIndexModel.repository_id == repository_id.value,
                            FileIndexModel.file_path == file_path,
                            FileIndexModel.commit_hash == commit_hash
                        )
                    )
                )
                file_model = result.scalar_one_or_none()
                
                if file_model:
                    file_index = self._map_to_domain(file_model)
                    return Result.success(file_index)
                else:
                    return Result.success(None)
                    
        except Exception as e:
            logger.error(f"Error al buscar archivo {file_path} en repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar archivo: {str(e)}"))
    
    async def find_by_repository(self, repository_id: RepositoryId, commit_hash: Optional[str] = None,
                                branch_name: Optional[str] = None, limit: int = 100, offset: int = 0) -> Result[List[FileIndex], Exception]:
        """Buscar archivos por repositorio."""
        try:
            async with self.session_factory() as session:
                # Construir consulta base
                base_query = select(FileIndexModel).options(selectinload(FileIndexModel.repository))
                conditions = [FileIndexModel.repository_id == repository_id.value]
                
                if commit_hash:
                    conditions.append(FileIndexModel.commit_hash == commit_hash)
                
                if branch_name:
                    conditions.append(FileIndexModel.branch_name == branch_name)
                
                result = await session.execute(
                    base_query
                    .where(and_(*conditions))
                    .order_by(FileIndexModel.file_path)
                    .limit(limit)
                    .offset(offset)
                )
                file_models = result.scalars().all()
                
                file_indexes = [self._map_to_domain(model) for model in file_models]
                return Result.success(file_indexes)
                
        except Exception as e:
            logger.error(f"Error al buscar archivos de repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar archivos: {str(e)}"))
    
    async def find_by_language(self, repository_id: RepositoryId, language: str, 
                              commit_hash: Optional[str] = None, limit: int = 100, offset: int = 0) -> Result[List[FileIndex], Exception]:
        """Buscar archivos por lenguaje de programación."""
        try:
            async with self.session_factory() as session:
                # Construir consulta base
                base_query = select(FileIndexModel).options(selectinload(FileIndexModel.repository))
                conditions = [
                    FileIndexModel.repository_id == repository_id.value,
                    FileIndexModel.language == language
                ]
                
                if commit_hash:
                    conditions.append(FileIndexModel.commit_hash == commit_hash)
                
                result = await session.execute(
                    base_query
                    .where(and_(*conditions))
                    .order_by(FileIndexModel.file_path)
                    .limit(limit)
                    .offset(offset)
                )
                file_models = result.scalars().all()
                
                file_indexes = [self._map_to_domain(model) for model in file_models]
                return Result.success(file_indexes)
                
        except Exception as e:
            logger.error(f"Error al buscar archivos de lenguaje {language} en repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar archivos: {str(e)}"))
    
    async def find_by_extension(self, repository_id: RepositoryId, extension: str, 
                               commit_hash: Optional[str] = None, limit: int = 100, offset: int = 0) -> Result[List[FileIndex], Exception]:
        """Buscar archivos por extensión."""
        try:
            async with self.session_factory() as session:
                # Construir consulta base
                base_query = select(FileIndexModel).options(selectinload(FileIndexModel.repository))
                conditions = [
                    FileIndexModel.repository_id == repository_id.value,
                    FileIndexModel.file_extension == extension
                ]
                
                if commit_hash:
                    conditions.append(FileIndexModel.commit_hash == commit_hash)
                
                result = await session.execute(
                    base_query
                    .where(and_(*conditions))
                    .order_by(FileIndexModel.file_path)
                    .limit(limit)
                    .offset(offset)
                )
                file_models = result.scalars().all()
                
                file_indexes = [self._map_to_domain(model) for model in file_models]
                return Result.success(file_indexes)
                
        except Exception as e:
            logger.error(f"Error al buscar archivos con extensión {extension} en repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar archivos: {str(e)}"))
    
    async def search_by_filename(self, repository_id: RepositoryId, filename: str, 
                                commit_hash: Optional[str] = None, limit: int = 100, offset: int = 0) -> Result[List[FileIndex], Exception]:
        """Buscar archivos por nombre de archivo."""
        try:
            async with self.session_factory() as session:
                # Construir consulta base
                base_query = select(FileIndexModel).options(selectinload(FileIndexModel.repository))
                conditions = [
                    FileIndexModel.repository_id == repository_id.value,
                    or_(
                        FileIndexModel.file_name.ilike(f"%{filename}%"),
                        FileIndexModel.file_path.ilike(f"%{filename}%")
                    )
                ]
                
                if commit_hash:
                    conditions.append(FileIndexModel.commit_hash == commit_hash)
                
                result = await session.execute(
                    base_query
                    .where(and_(*conditions))
                    .order_by(FileIndexModel.file_path)
                    .limit(limit)
                    .offset(offset)
                )
                file_models = result.scalars().all()
                
                file_indexes = [self._map_to_domain(model) for model in file_models]
                return Result.success(file_indexes)
                
        except Exception as e:
            logger.error(f"Error al buscar archivos con nombre {filename} en repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al buscar archivos: {str(e)}"))
    
    async def get_language_stats(self, repository_id: RepositoryId, commit_hash: Optional[str] = None) -> Result[Dict[str, Any], Exception]:
        """Obtener estadísticas de lenguajes de programación."""
        try:
            async with self.session_factory() as session:
                # Construir consulta base
                base_query = select(
                    FileIndexModel.language,
                    func.count(FileIndexModel.id).label('file_count'),
                    func.sum(FileIndexModel.size_bytes).label('total_size'),
                    func.sum(FileIndexModel.line_count).label('total_lines')
                )
                conditions = [FileIndexModel.repository_id == repository_id.value]
                
                if commit_hash:
                    conditions.append(FileIndexModel.commit_hash == commit_hash)
                
                result = await session.execute(
                    base_query
                    .where(and_(*conditions))
                    .group_by(FileIndexModel.language)
                    .order_by(func.count(FileIndexModel.id).desc())
                )
                rows = result.fetchall()
                
                stats = {}
                for row in rows:
                    if row.language:  # Ignorar archivos sin lenguaje detectado
                        stats[row.language] = {
                            'file_count': row.file_count,
                            'total_size': row.total_size or 0,
                            'total_lines': row.total_lines or 0
                        }
                
                return Result.success(stats)
                
        except Exception as e:
            logger.error(f"Error al obtener estadísticas de lenguajes para repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al obtener estadísticas: {str(e)}"))
    
    async def count_by_repository(self, repository_id: RepositoryId, commit_hash: Optional[str] = None) -> Result[int, Exception]:
        """Contar archivos de un repositorio."""
        try:
            async with self.session_factory() as session:
                # Construir consulta base
                base_query = select(func.count(FileIndexModel.id))
                conditions = [FileIndexModel.repository_id == repository_id.value]
                
                if commit_hash:
                    conditions.append(FileIndexModel.commit_hash == commit_hash)
                
                result = await session.execute(
                    base_query.where(and_(*conditions))
                )
                count = result.scalar()
                
                return Result.success(count or 0)
                
        except Exception as e:
            logger.error(f"Error al contar archivos de repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al contar archivos: {str(e)}"))
    
    async def delete_by_repository_and_commit(self, repository_id: RepositoryId, commit_hash: str) -> Result[int, Exception]:
        """Eliminar archivos de un repositorio y commit específico."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    delete(FileIndexModel).where(
                        and_(
                            FileIndexModel.repository_id == repository_id.value,
                            FileIndexModel.commit_hash == commit_hash
                        )
                    )
                )
                await session.commit()
                
                deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.info(f"Eliminados {deleted_count} archivos del repositorio {repository_id} commit {commit_hash}")
                
                return Result.success(deleted_count)
                
        except Exception as e:
            logger.error(f"Error al eliminar archivos del repositorio {repository_id} commit {commit_hash}: {str(e)}")
            return Result.failure(BaseError(f"Error al eliminar archivos: {str(e)}"))
    
    async def delete_by_repository(self, repository_id: RepositoryId) -> Result[int, Exception]:
        """Eliminar todos los archivos de un repositorio."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    delete(FileIndexModel).where(FileIndexModel.repository_id == repository_id.value)
                )
                await session.commit()
                
                deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.info(f"Eliminados {deleted_count} archivos del repositorio {repository_id}")
                
                return Result.success(deleted_count)
                
        except Exception as e:
            logger.error(f"Error al eliminar archivos del repositorio {repository_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al eliminar archivos: {str(e)}"))
    
    async def update_metadata(self, file_id: FileId, metadata: FileMetadata) -> Result[bool, Exception]:
        """Actualizar metadatos de un archivo."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    update(FileIndexModel)
                    .where(FileIndexModel.id == file_id.value)
                    .values(
                        metadata_json=asdict(metadata),
                        updated_at=datetime.utcnow()
                    )
                )
                await session.commit()
                
                updated = result.rowcount > 0
                if updated:
                    logger.info(f"Metadatos de archivo actualizados: {file_id}")
                
                return Result.success(updated)
                
        except Exception as e:
            logger.error(f"Error al actualizar metadatos de archivo {file_id}: {str(e)}")
            return Result.failure(BaseError(f"Error al actualizar metadatos: {str(e)}"))
    
    def _map_to_domain(self, model: FileIndexModel) -> FileIndex:
        """Mapear modelo SQLAlchemy a entidad de dominio."""
        from codeant_agent.domain.value_objects import FileId, RepositoryId
        
        metadata = FileMetadata(**model.metadata_json) if model.metadata_json else FileMetadata()
        
        return FileIndex(
            id=FileId(str(model.id)),
            repository_id=RepositoryId(str(model.repository_id)),
            file_path=model.file_path,
            file_name=model.file_name,
            file_extension=model.file_extension,
            language=model.language,
            mime_type=model.mime_type,
            size_bytes=model.size_bytes,
            line_count=model.line_count,
            commit_hash=model.commit_hash,
            branch_name=model.branch_name,
            is_binary=model.is_binary,
            is_ignored=model.is_ignored,
            metadata=metadata,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
