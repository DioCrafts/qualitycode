"""Caso de uso para registrar un nuevo usuario."""

from dataclasses import dataclass
from typing import Optional

from codeant_agent.domain.entities.user import User
from codeant_agent.domain.repositories.user_repository import UserRepository
from codeant_agent.domain.services.auth_service import AuthDomainService
from codeant_agent.domain.value_objects import Email, Username, OrganizationId
from codeant_agent.utils.result import Result
from codeant_agent.utils.error import ValidationError, ConflictError


@dataclass
class RegisterUserCommand:
    """Comando para registrar un usuario."""
    email: str
    username: str
    full_name: str
    password: str
    organization_id: Optional[str] = None


@dataclass
class RegisterUserResponse:
    """Respuesta del registro de usuario."""
    user_id: str
    email: str
    username: str
    full_name: str
    is_active: bool
    roles: list


class RegisterUserUseCase:
    """Caso de uso para registrar un nuevo usuario."""

    def __init__(
        self,
        user_repository: UserRepository,
        auth_service: AuthDomainService
    ):
        """Inicializar el caso de uso."""
        self.user_repository = user_repository
        self.auth_service = auth_service

    async def execute(self, command: RegisterUserCommand) -> Result[RegisterUserResponse, Exception]:
        """Ejecutar el registro de usuario."""
        try:
            # Validar entrada
            validation_result = await self._validate_command(command)
            if validation_result.is_failure():
                return Result.failure(validation_result.error)

            # Validar fortaleza de contraseña
            password_validation = self.auth_service.validate_password_strength(command.password)
            if password_validation.is_failure():
                return Result.failure(ValidationError(password_validation.error))

            # Crear value objects
            email = Email(command.email)
            username = Username(command.username)
            organization_id = OrganizationId.from_str(command.organization_id) if command.organization_id else None

            # Verificar que el email no exista
            email_exists_result = await self.user_repository.exists_by_email(email)
            if email_exists_result.is_failure():
                return Result.failure(email_exists_result.error)
            
            if email_exists_result.data:
                return Result.failure(ConflictError(f"El email {command.email} ya está registrado"))

            # Verificar que el username no exista
            username_exists_result = await self.user_repository.exists_by_username(username)
            if username_exists_result.is_failure():
                return Result.failure(username_exists_result.error)
            
            if username_exists_result.data:
                return Result.failure(ConflictError(f"El username {command.username} ya está en uso"))

            # Hashear contraseña
            password_hash = self.auth_service.hash_password(command.password)

            # Crear usuario
            user = User.create(
                email=command.email,
                username=command.username,
                full_name=command.full_name,
                password_hash=password_hash,
                organization_id=command.organization_id,
                roles=["viewer"]  # Rol por defecto
            )

            # Guardar usuario
            save_result = await self.user_repository.save(user)
            if save_result.is_failure():
                return Result.failure(save_result.error)

            # Crear respuesta
            response = RegisterUserResponse(
                user_id=str(user.id.value),
                email=user.email.value,
                username=user.username.value,
                full_name=user.full_name,
                is_active=user.is_active,
                roles=user.roles
            )

            return Result.success(response)

        except Exception as e:
            return Result.failure(e)

    async def _validate_command(self, command: RegisterUserCommand) -> Result[bool, ValidationError]:
        """Validar el comando de entrada."""
        errors = []

        # Validar email
        if not command.email or not command.email.strip():
            errors.append("El email es requerido")

        # Validar username
        if not command.username or not command.username.strip():
            errors.append("El username es requerido")

        # Validar nombre completo
        if not command.full_name or len(command.full_name.strip()) < 2:
            errors.append("El nombre completo debe tener al menos 2 caracteres")

        # Validar contraseña
        if not command.password or len(command.password) < 8:
            errors.append("La contraseña debe tener al menos 8 caracteres")

        # Validar organization_id si se proporciona
        if command.organization_id:
            try:
                OrganizationId.from_str(command.organization_id)
            except Exception:
                errors.append("El ID de organización no es válido")

        if errors:
            return Result.failure(ValidationError("; ".join(errors)))

        return Result.success(True)
