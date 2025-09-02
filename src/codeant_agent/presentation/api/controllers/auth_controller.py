"""Controlador de autenticación."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from codeant_agent.application.dtos.auth_dtos import (
    RegisterUserRequest,
    LoginUserRequest,
    RefreshTokenRequest,
    AuthResponse,
    TokenResponse,
    UserResponse,
    CurrentUser
)
from codeant_agent.application.use_cases.auth.register_user import (
    RegisterUserUseCase,
    RegisterUserCommand
)
from codeant_agent.application.use_cases.auth.login_user import (
    LoginUserUseCase,
    LoginUserCommand
)
from codeant_agent.application.use_cases.auth.refresh_token import (
    RefreshTokenUseCase,
    RefreshTokenCommand
)
from codeant_agent.domain.repositories.user_repository import UserRepository
from codeant_agent.domain.services.auth_service import AuthDomainService
from codeant_agent.utils.error import ValidationError, AuthenticationError, ConflictError


security = HTTPBearer()
router = APIRouter(prefix="/auth", tags=["authentication"])


class AuthController:
    """Controlador para operaciones de autenticación."""

    def __init__(
        self,
        user_repository: UserRepository,
        auth_service: AuthDomainService
    ):
        """Inicializar el controlador."""
        self.user_repository = user_repository
        self.auth_service = auth_service
        self.register_use_case = RegisterUserUseCase(user_repository, auth_service)
        self.login_use_case = LoginUserUseCase(user_repository, auth_service)
        self.refresh_use_case = RefreshTokenUseCase(user_repository, auth_service)

    async def register(self, request: RegisterUserRequest) -> AuthResponse:
        """Registrar un nuevo usuario."""
        command = RegisterUserCommand(
            email=request.email,
            username=request.username,
            full_name=request.full_name,
            password=request.password,
            organization_id=request.organization_id
        )

        result = await self.register_use_case.execute(command)
        
        if result.is_failure():
            error = result.error
            if isinstance(error, ValidationError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(error)
                )
            elif isinstance(error, ConflictError):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(error)
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Error interno del servidor"
                )

        response_data = result.data
        
        # Crear tokens para el usuario recién registrado
        # Buscar el usuario completo para generar tokens
        user_result = await self.user_repository.find_by_id(
            response_data.user_id
        )
        
        if user_result.is_failure() or not user_result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error generando tokens"
            )

        user = user_result.data
        tokens = self.auth_service.create_tokens(user)

        return AuthResponse(
            user=UserResponse(
                id=response_data.user_id,
                email=response_data.email,
                username=response_data.username,
                full_name=response_data.full_name,
                is_active=response_data.is_active,
                is_verified=False,
                roles=response_data.roles,
                permissions=[],
                last_login_at=None,
                created_at=user.created_at,
                updated_at=user.updated_at
            ),
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_in=tokens["expires_in"]
        )

    async def login(self, request: LoginUserRequest) -> AuthResponse:
        """Autenticar un usuario."""
        command = LoginUserCommand(
            identifier=request.identifier,
            password=request.password
        )

        result = await self.login_use_case.execute(command)
        
        if result.is_failure():
            error = result.error
            if isinstance(error, (ValidationError, AuthenticationError)):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=str(error)
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Error interno del servidor"
                )

        response_data = result.data

        return AuthResponse(
            user=UserResponse(
                id=response_data.user_id,
                email=response_data.email,
                username=response_data.username,
                full_name=response_data.full_name,
                is_active=True,
                is_verified=True,
                roles=response_data.roles,
                permissions=response_data.permissions,
                last_login_at=None,  # Se actualiza en el caso de uso
                created_at=response_data.created_at if hasattr(response_data, 'created_at') else None,
                updated_at=response_data.updated_at if hasattr(response_data, 'updated_at') else None
            ),
            access_token=response_data.access_token,
            refresh_token=response_data.refresh_token,
            token_type=response_data.token_type,
            expires_in=response_data.expires_in
        )

    async def refresh_token(self, request: RefreshTokenRequest) -> TokenResponse:
        """Refrescar un token de acceso."""
        command = RefreshTokenCommand(
            refresh_token=request.refresh_token
        )

        result = await self.refresh_use_case.execute(command)
        
        if result.is_failure():
            error = result.error
            if isinstance(error, (ValidationError, AuthenticationError)):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=str(error)
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Error interno del servidor"
                )

        response_data = result.data

        return TokenResponse(
            access_token=response_data.access_token,
            refresh_token=response_data.refresh_token,
            token_type=response_data.token_type,
            expires_in=response_data.expires_in
        )

    async def get_current_user(
        self, 
        credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
    ) -> CurrentUser:
        """Obtener información del usuario actual."""
        token = credentials.credentials
        
        # Verificar token
        token_result = self.auth_service.verify_token(token)
        if token_result.is_failure():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido"
            )

        token_data = token_result.data
        user_id = token_data.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido"
            )

        # Buscar usuario
        user_result = await self.user_repository.find_by_id(user_id)
        if user_result.is_failure() or not user_result.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Usuario no encontrado"
            )

        user = user_result.data

        return CurrentUser(
            id=str(user.id.value),
            email=user.email.value,
            username=user.username.value,
            full_name=user.full_name,
            organization_id=str(user.organization_id.value) if user.organization_id else None,
            roles=user.roles,
            permissions=user.permissions
        )

    async def logout(
        self,
        credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
    ) -> dict:
        """Cerrar sesión del usuario."""
        # En JWT stateless, el logout se maneja en el cliente
        # invalidando el token localmente
        return {"message": "Sesión cerrada exitosamente"}


# Definir las rutas
def create_auth_routes(auth_controller: AuthController) -> APIRouter:
    """Crear las rutas de autenticación."""
    
    @router.post(
        "/register",
        response_model=AuthResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Registrar nuevo usuario",
        description="Registra un nuevo usuario en el sistema"
    )
    async def register(request: RegisterUserRequest) -> AuthResponse:
        return await auth_controller.register(request)

    @router.post(
        "/login",
        response_model=AuthResponse,
        summary="Iniciar sesión",
        description="Autentica un usuario y devuelve tokens de acceso"
    )
    async def login(request: LoginUserRequest) -> AuthResponse:
        return await auth_controller.login(request)

    @router.post(
        "/refresh",
        response_model=TokenResponse,
        summary="Refrescar token",
        description="Refresca un token de acceso usando el refresh token"
    )
    async def refresh_token(request: RefreshTokenRequest) -> TokenResponse:
        return await auth_controller.refresh_token(request)

    @router.get(
        "/me",
        response_model=CurrentUser,
        summary="Información del usuario actual",
        description="Obtiene la información del usuario autenticado"
    )
    async def get_current_user(
        credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
    ) -> CurrentUser:
        return await auth_controller.get_current_user(credentials)

    @router.post(
        "/logout",
        summary="Cerrar sesión",
        description="Cierra la sesión del usuario"
    )
    async def logout(
        credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
    ) -> dict:
        return await auth_controller.logout(credentials)

    return router
