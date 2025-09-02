"""
Módulo que define los tipos de lenguaje soportados por el sistema de reglas naturales.
"""
from enum import Enum, auto


class Language(Enum):
    """Enumeración de lenguajes soportados por el sistema."""
    SPANISH = auto()
    ENGLISH = auto()
    
    @classmethod
    def from_string(cls, language_str: str) -> 'Language':
        """Convierte una cadena de texto en un valor de Language.
        
        Args:
            language_str: Cadena de texto que representa el lenguaje
            
        Returns:
            El valor de Language correspondiente
            
        Raises:
            ValueError: Si el lenguaje no es soportado
        """
        normalized = language_str.lower().strip()
        if normalized in ('es', 'esp', 'spanish', 'español', 'castellano'):
            return cls.SPANISH
        elif normalized in ('en', 'eng', 'english', 'inglés', 'ingles'):
            return cls.ENGLISH
        else:
            raise ValueError(f"Lenguaje no soportado: {language_str}")
    
    def __str__(self) -> str:
        """Devuelve una representación en string del lenguaje."""
        if self == Language.SPANISH:
            return "español"
        elif self == Language.ENGLISH:
            return "english"
        return super().__str__()
