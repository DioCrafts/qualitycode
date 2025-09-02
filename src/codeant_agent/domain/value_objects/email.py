"""Value object para Email."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Email:
    """Value object que representa un email válido."""
    
    value: str
    
    def __post_init__(self) -> None:
        """Validar el formato del email."""
        if not self.value or not isinstance(self.value, str):
            raise ValueError("El email no puede estar vacío")
        
        # Patrón básico de validación de email
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, self.value.lower()):
            raise ValueError(f"Email inválido: {self.value}")
        
        if len(self.value) > 254:  # RFC 5321 límite
            raise ValueError("El email no puede exceder 254 caracteres")
        
        # Normalizar a minúsculas
        object.__setattr__(self, 'value', self.value.lower())
    
    @property
    def domain(self) -> str:
        """Obtener el dominio del email."""
        return self.value.split('@')[1]
    
    @property
    def local_part(self) -> str:
        """Obtener la parte local del email."""
        return self.value.split('@')[0]
    
    def is_same_domain(self, other: "Email") -> bool:
        """Verificar si dos emails tienen el mismo dominio."""
        return self.domain == other.domain
    
    def __str__(self) -> str:
        """Representación en string del Email."""
        return self.value
