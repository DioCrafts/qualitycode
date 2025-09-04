"""
Implementación del repositorio de parsers.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from ...domain.entities.parse_result import ParseResult
from ...domain.repositories.parser_repository import ParserRepository
from ...domain.value_objects.programming_language import ProgrammingLanguage
from ...utils.error import Result

logger = logging.getLogger(__name__)

class ParserRepositoryImpl(ParserRepository):
    """
    Implementación del repositorio para parsing de archivos.
    """
    
    async def parse_file(self, file_path: Path) -> Result[ParseResult, Exception]:
        """
        Parsear un archivo de código fuente.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Result con el resultado del parsing o error
        """
        try:
            logger.info(f"Parseando archivo: {file_path}")
            
            # Detectar lenguaje
            language = self._detect_language_from_extension(file_path)
            
            # Leer contenido
            content = await self._read_file_content(file_path)
            
            # Parsear contenido
            return await self.parse_content(content, language, file_path)
            
        except Exception as e:
            logger.error(f"Error parseando archivo {file_path}: {str(e)}")
            return Result.failure(e)
    
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
        try:
            logger.info(f"Parseando contenido de lenguaje {language}")
            
            # En una implementación real, se usaría un parser adecuado
            # para el lenguaje específico (tree-sitter, AST, etc.)
            
            # Por ahora, devolvemos un ParseResult simulado
            parse_result = ParseResult(
                file_path=file_path or Path("unknown"),
                language=language,
                ast=None,  # Aquí iría el AST real
                symbols=[],
                imports=[],
                exports=[],
                errors=[],
                warnings=[]
            )
            
            logger.info(f"Parsing completado para {file_path or 'contenido'}")
            
            return Result.success(parse_result)
            
        except Exception as e:
            logger.error(f"Error parseando contenido: {str(e)}")
            return Result.failure(e)
    
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
        try:
            # Si tenemos file_path, usar la extensión
            if file_path:
                return Result.success(self._detect_language_from_extension(file_path))
            
            # Si tenemos contenido, intentar detectar por contenido
            if content:
                return Result.success(self._detect_language_from_content(content))
            
            return Result.failure(ValueError("Se requiere file_path o content"))
            
        except Exception as e:
            logger.error(f"Error detectando lenguaje: {str(e)}")
            return Result.failure(e)
    
    def _detect_language_from_extension(self, file_path: Path) -> ProgrammingLanguage:
        """
        Detectar lenguaje por extensión de archivo.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Lenguaje detectado
        """
        suffix = file_path.suffix.lower()
        
        if suffix == ".py":
            return ProgrammingLanguage.PYTHON
        elif suffix == ".ts" or suffix == ".tsx":
            return ProgrammingLanguage.TYPESCRIPT
        elif suffix == ".js" or suffix == ".jsx":
            return ProgrammingLanguage.JAVASCRIPT
        elif suffix == ".rs":
            return ProgrammingLanguage.RUST
        else:
            # Default a Python si no se reconoce
            return ProgrammingLanguage.PYTHON
    
    def _detect_language_from_content(self, content: str) -> ProgrammingLanguage:
        """
        Detectar lenguaje por contenido.
        
        Args:
            content: Contenido a analizar
            
        Returns:
            Lenguaje detectado
        """
        # Implementación básica
        if "fn " in content and "pub " in content:
            return ProgrammingLanguage.RUST
        elif "import " in content and "from " in content and "def " in content:
            return ProgrammingLanguage.PYTHON
        elif "interface " in content or "type " in content or "class " in content and ":" in content:
            return ProgrammingLanguage.TYPESCRIPT
        elif "const " in content or "function " in content or "let " in content:
            return ProgrammingLanguage.JAVASCRIPT
        else:
            # Default a Python
            return ProgrammingLanguage.PYTHON
    
    async def _read_file_content(self, file_path: Path) -> str:
        """
        Leer contenido de un archivo.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Contenido del archivo
        """
        try:
            async def read_file():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # Ejecutar en un thread aparte para no bloquear
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, lambda: open(file_path, 'r', encoding='utf-8').read())
            
            return content
            
        except Exception as e:
            logger.error(f"Error leyendo archivo {file_path}: {str(e)}")
            raise
