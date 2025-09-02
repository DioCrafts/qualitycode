"""
Value Object para el ID de usuario.
"""
from dataclasses import dataclass
import uuid


@dataclass(frozen=True)
class UserId:
    """ID único de un usuario."""
    
    value: uuid.UUID
    
    def __post_init__(self):
        """Validar que el ID sea un UUID válido."""
        if not isinstance(self.value, uuid.UUID):
            raise ValueError("UserId debe ser un UUID válido")
    
    @classmethod
    def generate(cls) -> "UserId":
        """Generar un nuevo UserId."""
        return cls(uuid.uuid4())
    
    @classmethod
    def from_string(cls, value: str) -> "UserId":
        """Crear UserId desde un string."""
        try:
            return cls(uuid.UUID(value))
        except ValueError as e:
            raise ValueError(f"String inválido para UserId: {value}") from e
    
    @classmethod
    def from_str(cls, value: str) -> "UserId":
        """Crear UserId desde un string (alias para from_string)."""
        return cls.from_string(value)
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __repr__(self) -> str:
        return f"UserId({self.value})"

