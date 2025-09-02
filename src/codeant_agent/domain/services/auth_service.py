"""Servicio de dominio para autenticación."""

from abc import ABC, abstractmethod
from typing import Optional, List
from datetime import datetime, timedelta

from codeant_agent.domain.entities.user import User
from codeant_agent.domain.value_objects import UserId, Email, Username
from codeant_agent.utils.result import Result


class AuthDomainService(ABC):
    """Servicio de dominio para operaciones de autenticación."""

    @abstractmethod
    def hash_password(self, password: str) -> str:
        """Hashear una contraseña."""
        pass

    @abstractmethod
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verificar una contraseña contra su hash."""
        pass

    @abstractmethod
    def validate_password_strength(self, password: str) -> Result[bool, str]:
        """Validar la fortaleza de una contraseña."""
        pass

    @abstractmethod
    def generate_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Generar un JWT token para el usuario."""
        pass

    @abstractmethod
    def verify_token(self, token: str) -> Result[dict, str]:
        """Verificar y decodificar un JWT token."""
        pass

    @abstractmethod
    def generate_refresh_token(self, user_id: UserId) -> str:
        """Generar un refresh token."""
        pass

    @abstractmethod
    def verify_refresh_token(self, token: str) -> Result[UserId, str]:
        """Verificar un refresh token."""
        pass


class AuthorizationDomainService:
    """Servicio de dominio para autorización."""

    # Definición de roles y sus permisos
    ROLE_PERMISSIONS = {
        "super_admin": [
            "system:admin",
            "organization:create",
            "organization:read",
            "organization:update", 
            "organization:delete",
            "organization:manage_users",
            "project:create",
            "project:read",
            "project:update",
            "project:delete",
            "project:analyze",
            "project:configure",
            "analysis:create",
            "analysis:read",
            "analysis:delete",
            "user:create",
            "user:read",
            "user:update",
            "user:delete",
        ],
        "organization_admin": [
            "organization:read",
            "organization:update",
            "organization:manage_users",
            "project:create",
            "project:read",
            "project:update",
            "project:delete",
            "project:analyze",
            "project:configure",
            "analysis:create",
            "analysis:read",
            "analysis:delete",
            "user:read",
            "user:update",
        ],
        "project_maintainer": [
            "project:read",
            "project:update",
            "project:analyze",
            "project:configure",
            "analysis:create",
            "analysis:read",
            "analysis:delete",
        ],
        "developer": [
            "project:read",
            "project:analyze",
            "analysis:create",
            "analysis:read",
        ],
        "viewer": [
            "project:read",
            "analysis:read",
        ]
    }

    def get_permissions_for_role(self, role: str) -> List[str]:
        """Obtener permisos para un rol específico."""
        return self.ROLE_PERMISSIONS.get(role, [])

    def user_has_permission(self, user: User, permission: str) -> bool:
        """Verificar si un usuario tiene un permiso específico."""
        # Verificar permisos directos
        if user.has_permission(permission):
            return True
        
        # Verificar permisos a través de roles
        for role in user.roles:
            if permission in self.get_permissions_for_role(role):
                return True
        
        return False

    def user_has_role(self, user: User, role: str) -> bool:
        """Verificar si un usuario tiene un rol específico."""
        return user.has_role(role)

    def user_can_access_project(self, user: User, project_organization_id: str) -> bool:
        """Verificar si un usuario puede acceder a un proyecto."""
        # Super admin puede acceder a todo
        if self.user_has_role(user, "super_admin"):
            return True
        
        # Admin de organización puede acceder a proyectos de su organización
        if (self.user_has_role(user, "organization_admin") and
            user.organization_id and 
            str(user.organization_id.value) == project_organization_id):
            return True
        
        # Otros roles necesitan permisos específicos de proyecto
        return self.user_has_permission(user, "project:read")

    def user_can_manage_organization(self, user: User, organization_id: str) -> bool:
        """Verificar si un usuario puede gestionar una organización."""
        # Super admin puede gestionar todas las organizaciones
        if self.user_has_role(user, "super_admin"):
            return True
        
        # Admin de organización solo puede gestionar su propia organización
        if (self.user_has_role(user, "organization_admin") and
            user.organization_id and 
            str(user.organization_id.value) == organization_id):
            return True
        
        return False

    def get_available_roles(self) -> List[str]:
        """Obtener lista de roles disponibles."""
        return list(self.ROLE_PERMISSIONS.keys())

    def is_valid_role(self, role: str) -> bool:
        """Verificar si un rol es válido."""
        return role in self.ROLE_PERMISSIONS

    def get_highest_role(self, user: User) -> Optional[str]:
        """Obtener el rol más alto del usuario."""
        role_hierarchy = [
            "super_admin",
            "organization_admin", 
            "project_maintainer",
            "developer",
            "viewer"
        ]
        
        for role in role_hierarchy:
            if user.has_role(role):
                return role
        
        return None
