"""Caso de uso para autenticar un usuario."""

from dataclasses import dataclass
from typing import Union

from codeant_agent.domain.repositories.user_repository import UserRepository
from codeant_agent.domain.services.auth_service import AuthDomainService
from codeant_agent.domain.value_objects import Email, Username
from codeant_agent.utils.result import Result
from codeant_agent.utils.error import ValidationError, AuthenticationError, NotFoundError


@dataclass
class LoginUserCommand:
    """Comando para autenticar un usuario."""
    identifier: str  # Email o username
    password: str


@dataclass
class LoginUserResponse:
    """Respuesta del login de usuario."""
    user_id: str
    email: str
    username: str
    full_name: str
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    roles: list
    permissions: list


class LoginUserUseCase:
    """Caso de uso para autenticar un usuario."""

    def __init__(
        self,
        user_repository: UserRepository,
        auth_service: AuthDomainService
    ):
        """Inicializar el caso de uso."""
        self.user_repository = user_repository
        self.auth_service = auth_service

    async def execute(self, command: LoginUserCommand) -> Result[LoginUserResponse, Exception]:
        """Ejecutar el login de usuario."""
        try:
            # Validar entrada
            validation_result = self._validate_command(command)
            if validation_result.is_failure():
                return Result.failure(validation_result.error)

            # Buscar usuario por email o username
            user_result = await self._find_user_by_identifier(command.identifier)
            if user_result.is_failure():
                return Result.failure(user_result.error)

            user = user_result.data
            if not user:
                return Result.failure(AuthenticationError("Credenciales inválidas"))

            # Verificar que el usuario esté activo
            if not user.is_active:
                return Result.failure(AuthenticationError("La cuenta está desactivada"))

            # Verificar contraseña
            if not self.auth_service.verify_password(command.password, user.password_hash):
                return Result.failure(AuthenticationError("Credenciales inválidas"))

            # Actualizar último login
            await self.user_repository.update_last_login(user.id)

            # Generar tokens
            tokens = self.auth_service.create_tokens(user)

            # Crear respuesta
            response = LoginUserResponse(
                user_id=str(user.id.value),
                email=user.email.value,
                username=user.username.value,
                full_name=user.full_name,
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                token_type=tokens["token_type"],
                expires_in=tokens["expires_in"],
                roles=user.roles,
                permissions=user.permissions
            )

            return Result.success(response)

        except Exception as e:
            return Result.failure(e)

    def _validate_command(self, command: LoginUserCommand) -> Result[bool, ValidationError]:
        """Validar el comando de entrada."""
        errors = []

        # Validar identifier
        if not command.identifier or not command.identifier.strip():
            errors.append("El email o username es requerido")

        # Validar contraseña
        if not command.password or not command.password.strip():
            errors.append("La contraseña es requerida")

        if errors:
            return Result.failure(ValidationError("; ".join(errors)))

        return Result.success(True)

    async def _find_user_by_identifier(self, identifier: str):
        """Buscar usuario por email o username."""
        # Intentar buscar por email primero
        if "@" in identifier:
            try:
                email = Email(identifier)
                return await self.user_repository.find_by_email(email)
            except ValueError:
                # Si no es un email válido, intentar como username
                pass
        
        # Buscar por username
        try:
            username = Username(identifier)
            return await self.user_repository.find_by_username(username)
        except ValueError:
            return Result.failure(ValidationError("Formato de identificador inválido"))
