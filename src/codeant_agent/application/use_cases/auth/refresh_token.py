"""Caso de uso para refrescar tokens de acceso."""

from dataclasses import dataclass

from codeant_agent.domain.repositories.user_repository import UserRepository
from codeant_agent.domain.services.auth_service import AuthDomainService
from codeant_agent.utils.result import Result
from codeant_agent.utils.error import ValidationError, AuthenticationError, NotFoundError


@dataclass
class RefreshTokenCommand:
    """Comando para refrescar un token."""
    refresh_token: str


@dataclass
class RefreshTokenResponse:
    """Respuesta del refresh token."""
    user_id: str
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int


class RefreshTokenUseCase:
    """Caso de uso para refrescar tokens de acceso."""

    def __init__(
        self,
        user_repository: UserRepository,
        auth_service: AuthDomainService
    ):
        """Inicializar el caso de uso."""
        self.user_repository = user_repository
        self.auth_service = auth_service

    async def execute(self, command: RefreshTokenCommand) -> Result[RefreshTokenResponse, Exception]:
        """Ejecutar el refresh del token."""
        try:
            # Validar entrada
            validation_result = self._validate_command(command)
            if validation_result.is_failure():
                return Result.failure(validation_result.error)

            # Verificar refresh token
            token_verification = self.auth_service.verify_refresh_token(command.refresh_token)
            if token_verification.is_failure():
                return Result.failure(AuthenticationError(token_verification.error))

            user_id = token_verification.data

            # Buscar usuario
            user_result = await self.user_repository.find_by_id(user_id)
            if user_result.is_failure():
                return Result.failure(user_result.error)

            user = user_result.data
            if not user:
                return Result.failure(NotFoundError("Usuario no encontrado"))

            # Verificar que el usuario esté activo
            if not user.is_active:
                return Result.failure(AuthenticationError("La cuenta está desactivada"))

            # Generar nuevos tokens
            tokens = self.auth_service.create_tokens(user)

            # Crear respuesta
            response = RefreshTokenResponse(
                user_id=str(user.id.value),
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                token_type=tokens["token_type"],
                expires_in=tokens["expires_in"]
            )

            return Result.success(response)

        except Exception as e:
            return Result.failure(e)

    def _validate_command(self, command: RefreshTokenCommand) -> Result[bool, ValidationError]:
        """Validar el comando de entrada."""
        if not command.refresh_token or not command.refresh_token.strip():
            return Result.failure(ValidationError("El refresh token es requerido"))

        return Result.success(True)
