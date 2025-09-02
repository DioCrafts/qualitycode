"""Interfaz del repositorio de usuarios."""

from abc import ABC, abstractmethod
from typing import List, Optional

from codeant_agent.domain.entities.user import User
from codeant_agent.domain.value_objects import UserId, OrganizationId, Email, Username
from codeant_agent.utils.result import Result


class UserRepository(ABC):
    """Interfaz para el repositorio de usuarios."""

    @abstractmethod
    async def save(self, user: User) -> Result[User, Exception]:
        """Guardar un usuario."""
        pass

    @abstractmethod
    async def find_by_id(self, user_id: UserId) -> Result[Optional[User], Exception]:
        """Encontrar un usuario por su ID."""
        pass

    @abstractmethod
    async def find_by_email(self, email: Email) -> Result[Optional[User], Exception]:
        """Encontrar un usuario por su email."""
        pass

    @abstractmethod
    async def find_by_username(self, username: Username) -> Result[Optional[User], Exception]:
        """Encontrar un usuario por su username."""
        pass

    @abstractmethod
    async def find_by_organization(
        self, 
        organization_id: OrganizationId,
        skip: int = 0,
        limit: int = 100
    ) -> Result[List[User], Exception]:
        """Encontrar usuarios por organización."""
        pass

    @abstractmethod
    async def list_all(
        self, 
        skip: int = 0, 
        limit: int = 100,
        only_active: bool = True
    ) -> Result[List[User], Exception]:
        """Listar todos los usuarios."""
        pass

    @abstractmethod
    async def delete(self, user_id: UserId) -> Result[bool, Exception]:
        """Eliminar un usuario."""
        pass

    @abstractmethod
    async def exists_by_email(self, email: Email) -> Result[bool, Exception]:
        """Verificar si existe un usuario con el email dado."""
        pass

    @abstractmethod
    async def exists_by_username(self, username: Username) -> Result[bool, Exception]:
        """Verificar si existe un usuario con el username dado."""
        pass

    @abstractmethod
    async def update_last_login(self, user_id: UserId) -> Result[bool, Exception]:
        """Actualizar el timestamp del último login."""
        pass

    @abstractmethod
    async def count_by_organization(self, organization_id: OrganizationId) -> Result[int, Exception]:
        """Contar usuarios por organización."""
        pass

    @abstractmethod
    async def search_users(
        self,
        query: str,
        organization_id: Optional[OrganizationId] = None,
        skip: int = 0,
        limit: int = 100
    ) -> Result[List[User], Exception]:
        """Buscar usuarios por texto."""
        pass
