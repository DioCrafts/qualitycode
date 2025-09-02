"""
Value Object para representar hashes de commits Git.
"""
from dataclasses import dataclass
import re
from codeant_agent.utils.error import ValidationError


@dataclass(frozen=True)
class CommitHash:
    """
    Value Object para representar hashes de commits Git.
    
    Attributes:
        value: String que representa el hash del commit (SHA-1)
    """
    value: str
    
    def __post_init__(self) -> None:
        """Validar que el CommitHash sea un SHA-1 válido."""
        if not self.value:
            raise ValidationError("CommitHash no puede estar vacío")
        
        # Validar formato SHA-1 (40 caracteres hexadecimales)
        if not re.match(r'^[a-fA-F0-9]{40}$', self.value):
            raise ValidationError(f"CommitHash debe ser un SHA-1 válido (40 caracteres hex): {self.value}")
    
    def short_hash(self, length: int = 7) -> str:
        """Obtener versión corta del hash."""
        if length > 40:
            length = 40
        return self.value[:length]
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"CommitHash('{self.value}')"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommitHash):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash(self.value)
