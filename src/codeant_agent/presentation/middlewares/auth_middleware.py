"""Middleware de autenticación y autorización."""

from typing import Optional, List
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from codeant_agent.domain.repositories.user_repository import UserRepository
from codeant_agent.domain.services.auth_service import AuthDomainService, AuthorizationDomainService
from codeant_agent.domain.value_objects import UserId
from codeant_agent.application.dtos.auth_dtos import CurrentUser


class AuthMiddleware:
    """Middleware para autenticación y autorización."""

    def __init__(
        self,
        user_repository: UserRepository,
        auth_service: AuthDomainService,
        authorization_service: AuthorizationDomainService
    ):
        """Inicializar el middleware."""
        self.user_repository = user_repository
        self.auth_service = auth_service
        self.authorization_service = authorization_service
        self.security = HTTPBearer()

    async def get_current_user(
        self, 
        credentials: HTTPAuthorizationCredentials
    ) -> CurrentUser:
        """Obtener el usuario actual desde el token."""
        token = credentials.credentials
        
        # Verificar token
        token_result = self.auth_service.verify_token(token)
        if token_result.is_failure():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token_data = token_result.data
        user_id_str = token_data.get("sub")
        
        if not user_id_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            user_id = UserId.from_str(user_id_str)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Buscar usuario
        user_result = await self.user_repository.find_by_id(user_id)
        if user_result.is_failure() or not user_result.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Usuario no encontrado",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user = user_result.data

        # Verificar que el usuario esté activo
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Cuenta desactivada",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return CurrentUser(
            id=str(user.id.value),
            email=user.email.value,
            username=user.username.value,
            full_name=user.full_name,
            organization_id=str(user.organization_id.value) if user.organization_id else None,
            roles=user.roles,
            permissions=user.permissions
        )

    async def get_optional_current_user(
        self, 
        credentials: Optional[HTTPAuthorizationCredentials]
    ) -> Optional[CurrentUser]:
        """Obtener el usuario actual de forma opcional."""
        if not credentials:
            return None
        
        try:
            return await self.get_current_user(credentials)
        except HTTPException:
            return None

    def require_permission(self, permission: str):
        """Decorator para requerir un permiso específico."""
        async def permission_checker(
            current_user: CurrentUser = None
        ) -> CurrentUser:
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Autenticación requerida"
                )

            # Buscar usuario completo
            user_id = UserId.from_str(current_user.id)
            user_result = await self.user_repository.find_by_id(user_id)
            
            if user_result.is_failure() or not user_result.data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Usuario no encontrado"
                )

            user = user_result.data

            # Verificar permiso
            if not self.authorization_service.user_has_permission(user, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permiso requerido: {permission}"
                )

            return current_user

        return permission_checker

    def require_role(self, role: str):
        """Decorator para requerir un rol específico."""
        async def role_checker(
            current_user: CurrentUser = None
        ) -> CurrentUser:
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Autenticación requerida"
                )

            # Buscar usuario completo
            user_id = UserId.from_str(current_user.id)
            user_result = await self.user_repository.find_by_id(user_id)
            
            if user_result.is_failure() or not user_result.data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Usuario no encontrado"
                )

            user = user_result.data

            # Verificar rol
            if not self.authorization_service.user_has_role(user, role):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Rol requerido: {role}"
                )

            return current_user

        return role_checker

    def require_roles(self, roles: List[str]):
        """Decorator para requerir uno de varios roles."""
        async def roles_checker(
            current_user: CurrentUser = None
        ) -> CurrentUser:
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Autenticación requerida"
                )

            # Buscar usuario completo
            user_id = UserId.from_str(current_user.id)
            user_result = await self.user_repository.find_by_id(user_id)
            
            if user_result.is_failure() or not user_result.data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Usuario no encontrado"
                )

            user = user_result.data

            # Verificar que tenga al menos uno de los roles
            has_role = any(
                self.authorization_service.user_has_role(user, role) 
                for role in roles
            )

            if not has_role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Uno de estos roles es requerido: {', '.join(roles)}"
                )

            return current_user

        return roles_checker

    def require_organization_access(self, organization_id: str):
        """Decorator para requerir acceso a una organización específica."""
        async def org_checker(
            current_user: CurrentUser = None
        ) -> CurrentUser:
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Autenticación requerida"
                )

            # Buscar usuario completo
            user_id = UserId.from_str(current_user.id)
            user_result = await self.user_repository.find_by_id(user_id)
            
            if user_result.is_failure() or not user_result.data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Usuario no encontrado"
                )

            user = user_result.data

            # Verificar acceso a organización
            if not self.authorization_service.user_can_manage_organization(user, organization_id):
                # Verificar al menos acceso de lectura
                if not (current_user.organization_id == organization_id or 
                       self.authorization_service.user_has_role(user, "super_admin")):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Acceso denegado a esta organización"
                    )

            return current_user

        return org_checker


# Funciones helper para usar en FastAPI dependencies
async def get_current_user_dependency(
    credentials: HTTPAuthorizationCredentials,
    auth_middleware: AuthMiddleware
) -> CurrentUser:
    """Dependency para obtener el usuario actual."""
    return await auth_middleware.get_current_user(credentials)


async def get_optional_current_user_dependency(
    credentials: Optional[HTTPAuthorizationCredentials],
    auth_middleware: AuthMiddleware
) -> Optional[CurrentUser]:
    """Dependency para obtener el usuario actual de forma opcional."""
    return await auth_middleware.get_optional_current_user(credentials)


# Funciones de autorización específicas
def require_admin():
    """Requerir rol de administrador."""
    def admin_checker(current_user: CurrentUser) -> CurrentUser:
        if "super_admin" not in current_user.roles and "organization_admin" not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permisos de administrador requeridos"
            )
        return current_user
    
    return admin_checker


def require_project_access():
    """Requerir acceso a proyectos."""
    def project_checker(current_user: CurrentUser) -> CurrentUser:
        # Verificar que tenga permisos básicos de proyecto
        allowed_roles = ["super_admin", "organization_admin", "project_maintainer", "developer", "viewer"]
        has_role = any(role in current_user.roles for role in allowed_roles)
        
        if not has_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Acceso a proyectos requerido"
            )
        
        return current_user
    
    return project_checker
