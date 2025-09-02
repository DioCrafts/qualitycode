"""
Value Object para el ID de organización.
"""
from dataclasses import dataclass
from typing import Optional
import uuid


@dataclass(frozen=True)
class OrganizationId:
    """ID único de una organización."""
    
    value: uuid.UUID
    
    def __post_init__(self):
        """Validar que el ID sea un UUID válido."""
        if not isinstance(self.value, uuid.UUID):
            raise ValueError("OrganizationId debe ser un UUID válido")
    
    @classmethod
    def generate(cls) -> "OrganizationId":
        """Generar un nuevo OrganizationId."""
        return cls(uuid.uuid4())
    
    @classmethod
    def from_string(cls, value: str) -> "OrganizationId":
        """Crear OrganizationId desde un string."""
        try:
            return cls(uuid.UUID(value))
        except ValueError as e:
            raise ValueError(f"String inválido para OrganizationId: {value}") from e
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __repr__(self) -> str:
        return f"OrganizationId({self.value})"
