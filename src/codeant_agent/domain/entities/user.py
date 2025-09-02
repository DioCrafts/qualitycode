"""Entidad de dominio User."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import uuid

from codeant_agent.domain.value_objects import (
    UserId,
    OrganizationId,
    Email,
    Username
)


@dataclass
class User:
    """Entidad que representa un usuario del sistema."""
    
    id: UserId
    organization_id: Optional[OrganizationId]
    email: Email
    username: Username
    full_name: str
    password_hash: str
    is_active: bool = True
    is_verified: bool = False
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    last_login_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validaciones post-inicialización."""
        if not self.full_name or len(self.full_name.strip()) < 2:
            raise ValueError("El nombre completo debe tener al menos 2 caracteres")
        
        if not self.password_hash:
            raise ValueError("El hash de la contraseña es requerido")
    
    @classmethod
    def create(
        cls,
        email: str,
        username: str,
        full_name: str,
        password_hash: str,
        organization_id: Optional[str] = None,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None
    ) -> "User":
        """Crear una nueva instancia de User."""
        return cls(
            id=UserId.generate(),
            organization_id=OrganizationId(uuid.UUID(organization_id)) if organization_id else None,
            email=Email(email),
            username=Username(username),
            full_name=full_name.strip(),
            password_hash=password_hash,
            roles=roles or [],
            permissions=permissions or []
        )
    
    def add_role(self, role: str) -> None:
        """Agregar un rol al usuario."""
        if role not in self.roles:
            self.roles.append(role)
            self.updated_at = datetime.utcnow()
    
    def remove_role(self, role: str) -> None:
        """Remover un rol del usuario."""
        if role in self.roles:
            self.roles.remove(role)
            self.updated_at = datetime.utcnow()
    
    def add_permission(self, permission: str) -> None:
        """Agregar un permiso al usuario."""
        if permission not in self.permissions:
            self.permissions.append(permission)
            self.updated_at = datetime.utcnow()
    
    def remove_permission(self, permission: str) -> None:
        """Remover un permiso del usuario."""
        if permission in self.permissions:
            self.permissions.remove(permission)
            self.updated_at = datetime.utcnow()
    
    def has_role(self, role: str) -> bool:
        """Verificar si el usuario tiene un rol específico."""
        return role in self.roles
    
    def has_permission(self, permission: str) -> bool:
        """Verificar si el usuario tiene un permiso específico."""
        return permission in self.permissions
    
    def mark_as_verified(self) -> None:
        """Marcar el usuario como verificado."""
        self.is_verified = True
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Desactivar el usuario."""
        self.is_active = False
        self.updated_at = datetime.utcnow()
    
    def activate(self) -> None:
        """Activar el usuario."""
        self.is_active = True
        self.updated_at = datetime.utcnow()
    
    def update_last_login(self) -> None:
        """Actualizar el timestamp del último login."""
        self.last_login_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def can_access_organization(self, organization_id: OrganizationId) -> bool:
        """Verificar si el usuario puede acceder a una organización."""
        return (
            self.organization_id == organization_id or
            self.has_role("super_admin") or
            self.has_permission("organization:access_all")
        )
    
    def __str__(self) -> str:
        """Representación en string del User."""
        status = "activo" if self.is_active else "inactivo"
        return f"Usuario(email={self.email.value}, username={self.username.value}, status={status})"
    
    def __repr__(self) -> str:
        """Representación detallada del User."""
        return (
            f"User(id={self.id.value}, email={self.email.value}, "
            f"username={self.username.value}, active={self.is_active}, "
            f"roles={self.roles})"
        )
