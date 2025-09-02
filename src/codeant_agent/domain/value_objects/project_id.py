"""
Value Object para identificar proyectos de forma única.
"""
from dataclasses import dataclass
from typing import Optional
import uuid
from codeant_agent.utils.error import ValidationError


@dataclass(frozen=True)
class ProjectId:
    """
    Value Object para identificar proyectos de forma única.
    
    Attributes:
        value: UUID string que identifica el proyecto
    """
    value: str
    
    def __post_init__(self) -> None:
        """Validar que el ProjectId sea un UUID válido."""
        if not self.value:
            raise ValidationError("ProjectId no puede estar vacío")
        
        try:
            uuid.UUID(self.value)
        except ValueError:
            raise ValidationError(f"ProjectId debe ser un UUID válido: {self.value}")
    
    @classmethod
    def generate(cls) -> "ProjectId":
        """Generar un nuevo ProjectId único."""
        return cls(str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"ProjectId('{self.value}')"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProjectId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash(self.value)
