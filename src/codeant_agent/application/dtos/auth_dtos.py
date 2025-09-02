"""DTOs para operaciones de autenticación."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, validator


class RegisterUserRequest(BaseModel):
    """DTO para solicitud de registro de usuario."""
    email: EmailStr = Field(..., description="Email del usuario")
    username: str = Field(..., min_length=3, max_length=50, description="Nombre de usuario")
    full_name: str = Field(..., min_length=2, max_length=255, description="Nombre completo")
    password: str = Field(..., min_length=8, max_length=128, description="Contraseña")
    organization_id: Optional[str] = Field(None, description="ID de la organización")

    @validator('username')
    def validate_username(cls, v):
        """Validar formato del username."""
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username solo puede contener letras, números, guiones y guiones bajos')
        return v.lower()

    @validator('full_name')
    def validate_full_name(cls, v):
        """Validar nombre completo."""
        return v.strip()


class LoginUserRequest(BaseModel):
    """DTO para solicitud de login."""
    identifier: str = Field(..., description="Email o username")
    password: str = Field(..., description="Contraseña")


class RefreshTokenRequest(BaseModel):
    """DTO para solicitud de refresh token."""
    refresh_token: str = Field(..., description="Refresh token")


class UserResponse(BaseModel):
    """DTO para respuesta de usuario."""
    id: str = Field(..., description="ID del usuario")
    email: str = Field(..., description="Email del usuario")
    username: str = Field(..., description="Username del usuario")
    full_name: str = Field(..., description="Nombre completo")
    is_active: bool = Field(..., description="Si el usuario está activo")
    is_verified: bool = Field(False, description="Si el usuario está verificado")
    roles: List[str] = Field(default_factory=list, description="Roles del usuario")
    permissions: List[str] = Field(default_factory=list, description="Permisos del usuario")
    last_login_at: Optional[datetime] = Field(None, description="Último login")
    created_at: datetime = Field(..., description="Fecha de creación")
    updated_at: datetime = Field(..., description="Fecha de actualización")


class AuthResponse(BaseModel):
    """DTO para respuesta de autenticación."""
    user: UserResponse = Field(..., description="Datos del usuario")
    access_token: str = Field(..., description="Token de acceso")
    refresh_token: str = Field(..., description="Token de refresco")
    token_type: str = Field(default="bearer", description="Tipo de token")
    expires_in: int = Field(..., description="Tiempo de expiración en segundos")


class TokenResponse(BaseModel):
    """DTO para respuesta de token."""
    access_token: str = Field(..., description="Token de acceso")
    refresh_token: str = Field(..., description="Token de refresco")
    token_type: str = Field(default="bearer", description="Tipo de token")
    expires_in: int = Field(..., description="Tiempo de expiración en segundos")


class CurrentUser(BaseModel):
    """DTO para usuario actual."""
    id: str = Field(..., description="ID del usuario")
    email: str = Field(..., description="Email del usuario")
    username: str = Field(..., description="Username del usuario")
    full_name: str = Field(..., description="Nombre completo")
    organization_id: Optional[str] = Field(None, description="ID de la organización")
    roles: List[str] = Field(default_factory=list, description="Roles del usuario")
    permissions: List[str] = Field(default_factory=list, description="Permisos del usuario")


class ChangePasswordRequest(BaseModel):
    """DTO para cambio de contraseña."""
    current_password: str = Field(..., description="Contraseña actual")
    new_password: str = Field(..., min_length=8, max_length=128, description="Nueva contraseña")


class UpdateProfileRequest(BaseModel):
    """DTO para actualización de perfil."""
    full_name: Optional[str] = Field(None, min_length=2, max_length=255, description="Nombre completo")
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="Username")

    @validator('username')
    def validate_username(cls, v):
        """Validar formato del username."""
        if v is not None:
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError('Username solo puede contener letras, números, guiones y guiones bajos')
            return v.lower()
        return v

    @validator('full_name')
    def validate_full_name(cls, v):
        """Validar nombre completo."""
        if v is not None:
            return v.strip()
        return v
