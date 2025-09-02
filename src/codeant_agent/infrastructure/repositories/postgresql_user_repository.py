"""Implementación PostgreSQL del repositorio de usuarios."""

from typing import List, Optional
from datetime import datetime

from sqlalchemy import and_, or_, func, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from codeant_agent.domain.entities.user import User
from codeant_agent.domain.repositories.user_repository import UserRepository
from codeant_agent.domain.value_objects import UserId, OrganizationId, Email, Username
from codeant_agent.infrastructure.database.models import User as UserModel, OrganizationMember
from codeant_agent.utils.result import Result
from codeant_agent.utils.error import (
    RepositoryError,
    NotFoundError,
    ValidationError
)


class PostgreSQLUserRepository(UserRepository):
    """Implementación PostgreSQL del repositorio de usuarios."""

    def __init__(self, session_factory):
        """Inicializar el repositorio."""
        self.session_factory = session_factory

    async def save(self, user: User) -> Result[User, Exception]:
        """Guardar un usuario."""
        try:
            async with self.session_factory() as session:
                # Buscar si ya existe
                stmt = select(UserModel).where(UserModel.id == user.id.value)
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    # Actualizar usuario existente
                    existing.email = user.email.value
                    existing.username = user.username.value
                    existing.full_name = user.full_name
                    existing.password_hash = user.password_hash
                    existing.is_active = user.is_active
                    existing.is_verified = user.is_verified
                    existing.roles = user.roles
                    existing.permissions = user.permissions
                    existing.last_login_at = user.last_login_at
                    existing.updated_at = datetime.utcnow()
                else:
                    # Crear nuevo usuario
                    user_model = UserModel(
                        id=user.id.value,
                        email=user.email.value,
                        username=user.username.value,
                        full_name=user.full_name,
                        password_hash=user.password_hash,
                        is_active=user.is_active,
                        is_verified=user.is_verified,
                        roles=user.roles,
                        permissions=user.permissions,
                        last_login_at=user.last_login_at,
                        created_at=user.created_at,
                        updated_at=user.updated_at
                    )
                    session.add(user_model)

                await session.commit()
                return Result.success(user)

        except Exception as e:
            return Result.failure(RepositoryError(f"Error guardando usuario: {str(e)}"))

    async def find_by_id(self, user_id: UserId) -> Result[Optional[User], Exception]:
        """Encontrar un usuario por su ID."""
        try:
            async with self.session_factory() as session:
                stmt = select(UserModel).where(UserModel.id == user_id.value)
                result = await session.execute(stmt)
                user_model = result.scalar_one_or_none()

                if user_model:
                    user = self._map_to_domain(user_model)
                    return Result.success(user)
                
                return Result.success(None)

        except Exception as e:
            return Result.failure(RepositoryError(f"Error buscando usuario por ID: {str(e)}"))

    async def find_by_email(self, email: Email) -> Result[Optional[User], Exception]:
        """Encontrar un usuario por su email."""
        try:
            async with self.session_factory() as session:
                stmt = select(UserModel).where(UserModel.email == email.value)
                result = await session.execute(stmt)
                user_model = result.scalar_one_or_none()

                if user_model:
                    user = self._map_to_domain(user_model)
                    return Result.success(user)
                
                return Result.success(None)

        except Exception as e:
            return Result.failure(RepositoryError(f"Error buscando usuario por email: {str(e)}"))

    async def find_by_username(self, username: Username) -> Result[Optional[User], Exception]:
        """Encontrar un usuario por su username."""
        try:
            async with self.session_factory() as session:
                stmt = select(UserModel).where(UserModel.username == username.value)
                result = await session.execute(stmt)
                user_model = result.scalar_one_or_none()

                if user_model:
                    user = self._map_to_domain(user_model)
                    return Result.success(user)
                
                return Result.success(None)

        except Exception as e:
            return Result.failure(RepositoryError(f"Error buscando usuario por username: {str(e)}"))

    async def find_by_organization(
        self, 
        organization_id: OrganizationId,
        skip: int = 0,
        limit: int = 100
    ) -> Result[List[User], Exception]:
        """Encontrar usuarios por organización."""
        try:
            async with self.session_factory() as session:
                stmt = (
                    select(UserModel)
                    .join(OrganizationMember, UserModel.id == OrganizationMember.user_id)
                    .where(OrganizationMember.organization_id == organization_id.value)
                    .order_by(UserModel.created_at.desc())
                    .offset(skip)
                    .limit(limit)
                )
                result = await session.execute(stmt)
                user_models = result.scalars().all()

                users = [self._map_to_domain(model) for model in user_models]
                return Result.success(users)

        except Exception as e:
            return Result.failure(RepositoryError(f"Error buscando usuarios por organización: {str(e)}"))

    async def list_all(
        self, 
        skip: int = 0, 
        limit: int = 100,
        only_active: bool = True
    ) -> Result[List[User], Exception]:
        """Listar todos los usuarios."""
        try:
            async with self.session_factory() as session:
                stmt = select(UserModel)
                
                if only_active:
                    stmt = stmt.where(UserModel.is_active == True)
                
                stmt = stmt.order_by(UserModel.created_at.desc()).offset(skip).limit(limit)
                
                result = await session.execute(stmt)
                user_models = result.scalars().all()

                users = [self._map_to_domain(model) for model in user_models]
                return Result.success(users)

        except Exception as e:
            return Result.failure(RepositoryError(f"Error listando usuarios: {str(e)}"))

    async def delete(self, user_id: UserId) -> Result[bool, Exception]:
        """Eliminar un usuario."""
        try:
            async with self.session_factory() as session:
                stmt = delete(UserModel).where(UserModel.id == user_id.value)
                result = await session.execute(stmt)
                
                await session.commit()
                return Result.success(result.rowcount > 0)

        except Exception as e:
            return Result.failure(RepositoryError(f"Error eliminando usuario: {str(e)}"))

    async def exists_by_email(self, email: Email) -> Result[bool, Exception]:
        """Verificar si existe un usuario con el email dado."""
        try:
            async with self.session_factory() as session:
                stmt = select(func.count(UserModel.id)).where(UserModel.email == email.value)
                result = await session.execute(stmt)
                count = result.scalar()
                
                return Result.success(count > 0)

        except Exception as e:
            return Result.failure(RepositoryError(f"Error verificando email existente: {str(e)}"))

    async def exists_by_username(self, username: Username) -> Result[bool, Exception]:
        """Verificar si existe un usuario con el username dado."""
        try:
            async with self.session_factory() as session:
                stmt = select(func.count(UserModel.id)).where(UserModel.username == username.value)
                result = await session.execute(stmt)
                count = result.scalar()
                
                return Result.success(count > 0)

        except Exception as e:
            return Result.failure(RepositoryError(f"Error verificando username existente: {str(e)}"))

    async def update_last_login(self, user_id: UserId) -> Result[bool, Exception]:
        """Actualizar el timestamp del último login."""
        try:
            async with self.session_factory() as session:
                stmt = (
                    update(UserModel)
                    .where(UserModel.id == user_id.value)
                    .values(
                        last_login_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                )
                result = await session.execute(stmt)
                
                await session.commit()
                return Result.success(result.rowcount > 0)

        except Exception as e:
            return Result.failure(RepositoryError(f"Error actualizando último login: {str(e)}"))

    async def count_by_organization(self, organization_id: OrganizationId) -> Result[int, Exception]:
        """Contar usuarios por organización."""
        try:
            async with self.session_factory() as session:
                stmt = (
                    select(func.count(UserModel.id))
                    .join(OrganizationMember, UserModel.id == OrganizationMember.user_id)
                    .where(OrganizationMember.organization_id == organization_id.value)
                )
                result = await session.execute(stmt)
                count = result.scalar()
                
                return Result.success(count)

        except Exception as e:
            return Result.failure(RepositoryError(f"Error contando usuarios por organización: {str(e)}"))

    async def search_users(
        self,
        query: str,
        organization_id: Optional[OrganizationId] = None,
        skip: int = 0,
        limit: int = 100
    ) -> Result[List[User], Exception]:
        """Buscar usuarios por texto."""
        try:
            async with self.session_factory() as session:
                # Construir query base
                stmt = select(UserModel)
                
                # Filtrar por organización si se especifica
                if organization_id:
                    stmt = stmt.join(OrganizationMember, UserModel.id == OrganizationMember.user_id)
                    stmt = stmt.where(OrganizationMember.organization_id == organization_id.value)
                
                # Agregar filtro de búsqueda
                search_filter = or_(
                    UserModel.full_name.ilike(f"%{query}%"),
                    UserModel.email.ilike(f"%{query}%"),
                    UserModel.username.ilike(f"%{query}%")
                )
                stmt = stmt.where(search_filter)
                
                # Ordenar y paginar
                stmt = stmt.order_by(UserModel.full_name).offset(skip).limit(limit)
                
                result = await session.execute(stmt)
                user_models = result.scalars().all()
                
                users = [self._map_to_domain(model) for model in user_models]
                return Result.success(users)

        except Exception as e:
            return Result.failure(RepositoryError(f"Error buscando usuarios: {str(e)}"))

    def _map_to_domain(self, model: UserModel) -> User:
        """Mapear un modelo SQLAlchemy a la entidad de dominio."""
        # Por ahora, asumimos que no tenemos organización específica
        # TODO: Implementar lógica para obtener la organización principal del usuario
        organization_id = None
        
        return User(
            id=UserId(model.id),
            organization_id=organization_id,
            email=Email(model.email),
            username=Username(model.username),
            full_name=model.full_name,
            password_hash=model.password_hash,
            is_active=model.is_active,
            is_verified=model.is_verified,
            roles=model.roles or [],
            permissions=model.permissions or [],
            last_login_at=model.last_login_at,
            created_at=model.created_at,
            updated_at=model.updated_at
        )