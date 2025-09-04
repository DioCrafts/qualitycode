"""
Interfaz para el repositorio de parsers.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..entities.parse_result import ParseResult
from ..value_objects.programming_language import ProgrammingLanguage
from ...utils.error import Result

class ParserRepository(ABC):
    """
    Repositorio para parsing de archivos.
    """
    
    @abstractmethod
    async def parse_file(self, file_path: Path) -> Result[ParseResult, Exception]:
        """
        Parsear un archivo de código fuente.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Result con el resultado del parsing o error
        """
        pass
    
    @abstractmethod
    async def parse_content(
        self, 
        content: str, 
        language: ProgrammingLanguage,
        file_path: Optional[Path] = None
    ) -> Result[ParseResult, Exception]:
        """
        Parsear contenido de código fuente.
        
        Args:
            content: Contenido a parsear
            language: Lenguaje del contenido
            file_path: Ruta opcional al archivo
            
        Returns:
            Result con el resultado del parsing o error
        """
        pass
    
    @abstractmethod
    async def detect_language(
        self, 
        file_path: Optional[Path] = None, 
        content: Optional[str] = None
    ) -> Result[ProgrammingLanguage, Exception]:
        """
        Detectar lenguaje de programación.
        
        Args:
            file_path: Ruta opcional al archivo
            content: Contenido opcional
            
        Returns:
            Result con el lenguaje detectado o error
        """
        pass