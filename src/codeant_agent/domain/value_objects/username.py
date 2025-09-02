"""Value object para Username."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Username:
    """Value object que representa un nombre de usuario válido."""
    
    value: str
    
    def __post_init__(self) -> None:
        """Validar el formato del username."""
        if not self.value or not isinstance(self.value, str):
            raise ValueError("El username no puede estar vacío")
        
        # Normalizar: sin espacios extra
        normalized = self.value.strip()
        if not normalized:
            raise ValueError("El username no puede estar vacío")
        
        # Validaciones de formato
        if len(normalized) < 3:
            raise ValueError("El username debe tener al menos 3 caracteres")
        
        if len(normalized) > 50:
            raise ValueError("El username no puede exceder 50 caracteres")
        
        # Solo letras, números, guiones y guiones bajos
        if not re.match(r'^[a-zA-Z0-9_-]+$', normalized):
            raise ValueError(
                "El username solo puede contener letras, números, "
                "guiones (-) y guiones bajos (_)"
            )
        
        # No puede empezar o terminar con guiones
        if normalized.startswith('-') or normalized.endswith('-'):
            raise ValueError("El username no puede empezar o terminar con guión")
        
        # No puede empezar o terminar con guiones bajos
        if normalized.startswith('_') or normalized.endswith('_'):
            raise ValueError("El username no puede empezar o terminar con guión bajo")
        
        # Normalizar a minúsculas
        object.__setattr__(self, 'value', normalized.lower())
    
    def is_similar_to(self, other: "Username") -> bool:
        """Verificar si dos usernames son similares."""
        # Simple similarity check - mismas letras sin considerar guiones/underscores
        self_clean = re.sub(r'[_-]', '', self.value)
        other_clean = re.sub(r'[_-]', '', other.value)
        return self_clean == other_clean
    
    def __str__(self) -> str:
        """Representación en string del Username."""
        return self.value
