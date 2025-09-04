"""
Value objects relacionados con lenguajes de programación.
"""
from enum import Enum, auto

class ProgrammingLanguage(Enum):
    """Lenguajes de programación soportados."""
    PYTHON = auto()
    JAVASCRIPT = auto()
    TYPESCRIPT = auto()
    RUST = auto()
    
    def get_name(self) -> str:
        """Obtener el nombre del lenguaje en formato legible."""
        return self.name.lower()
    
    def get_file_extensions(self) -> list[str]:
        """Obtener las extensiones de archivo asociadas al lenguaje."""
        if self == ProgrammingLanguage.PYTHON:
            return [".py"]
        elif self == ProgrammingLanguage.JAVASCRIPT:
            return [".js", ".jsx"]
        elif self == ProgrammingLanguage.TYPESCRIPT:
            return [".ts", ".tsx"]
        elif self == ProgrammingLanguage.RUST:
            return [".rs"]
        else:
            return []
    
    @classmethod
    def from_extension(cls, extension: str) -> 'ProgrammingLanguage':
        """
        Determinar el lenguaje a partir de una extensión de archivo.
        
        Args:
            extension: Extensión de archivo (con o sin punto)
            
        Returns:
            El lenguaje correspondiente o None si no se reconoce
        """
        if not extension.startswith('.'):
            extension = f".{extension}"
        
        extension = extension.lower()
        
        if extension in [".py"]:
            return cls.PYTHON
        elif extension in [".js", ".jsx"]:
            return cls.JAVASCRIPT
        elif extension in [".ts", ".tsx"]:
            return cls.TYPESCRIPT
        elif extension in [".rs"]:
            return cls.RUST
        else:
            # Default a Python
            return cls.PYTHON