"""
Value Object para identificar archivos de forma única.
"""
from dataclasses import dataclass
import uuid
from codeant_agent.utils.error import ValidationError


@dataclass(frozen=True)
class FileId:
    """
    Value Object para identificar archivos de forma única.
    
    Attributes:
        value: UUID string que identifica el archivo
    """
    value: str
    
    def __post_init__(self) -> None:
        """Validar que el FileId sea un UUID válido."""
        if not self.value:
            raise ValidationError("FileId no puede estar vacío")
        
        try:
            uuid.UUID(self.value)
        except ValueError:
            raise ValidationError(f"FileId debe ser un UUID válido: {self.value}")
    
    @classmethod
    def generate(cls) -> "FileId":
        """Generar un nuevo FileId único."""
        return cls(str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"FileId('{self.value}')"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FileId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash(self.value)
