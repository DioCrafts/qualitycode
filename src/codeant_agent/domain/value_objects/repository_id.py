"""
Value Object para identificar repositorios de forma única.
"""
from dataclasses import dataclass
import uuid
from codeant_agent.utils.error import ValidationError


@dataclass(frozen=True)
class RepositoryId:
    """
    Value Object para identificar repositorios de forma única.
    
    Attributes:
        value: UUID string que identifica el repositorio
    """
    value: str
    
    def __post_init__(self) -> None:
        """Validar que el RepositoryId sea un UUID válido."""
        if not self.value:
            raise ValidationError("RepositoryId no puede estar vacío")
        
        try:
            uuid.UUID(self.value)
        except ValueError:
            raise ValidationError(f"RepositoryId debe ser un UUID válido: {self.value}")
    
    @classmethod
    def generate(cls) -> "RepositoryId":
        """Generar un nuevo RepositoryId único."""
        return cls(str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"RepositoryId('{self.value}')"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RepositoryId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash(self.value)
