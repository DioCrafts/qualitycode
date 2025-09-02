"""
Parser Universal - Sistema de parsing multi-lenguaje basado en Tree-sitter.

Este módulo proporciona una interfaz unificada para el análisis de código
en múltiples lenguajes de programación usando Tree-sitter como base.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class ProgrammingLanguage(Enum):
    """Lenguajes de programación soportados."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    JSX = "jsx"
    TSX = "tsx"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    UNKNOWN = "unknown"


@dataclass
class ParseResult:
    """Resultado del parsing de un archivo."""
    file_path: Path
    language: ProgrammingLanguage
    ast: Optional[Any] = None
    tokens: List[Dict[str, Any]] = None
    errors: List[str] = None
    parse_time_ms: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tokens is None:
            self.tokens = []
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ParserConfig:
    """Configuración del parser universal."""
    enable_cache: bool = True
    cache_size: int = 1000
    enable_parallel_processing: bool = True
    max_workers: int = 4
    timeout_seconds: int = 30
    enable_error_recovery: bool = True
    strict_mode: bool = False
    debug_mode: bool = False


class Parser(Protocol):
    """Protocolo para parsers específicos de lenguaje."""
    
    @abstractmethod
    async def parse_file(self, file_path: Path) -> ParseResult:
        """Parsea un archivo y retorna el resultado."""
        pass
    
    @abstractmethod
    async def parse_content(self, content: str, language: ProgrammingLanguage) -> ParseResult:
        """Parsea contenido de texto y retorna el resultado."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Retorna los lenguajes soportados por este parser."""
        pass


class UniversalParser:
    """
    Parser universal que coordina parsers específicos por lenguaje.
    
    Implementa el patrón Strategy para manejar diferentes lenguajes
    de programación de manera unificada.
    """
    
    def __init__(self, config: ParserConfig):
        self.config = config
        self._parsers: Dict[ProgrammingLanguage, Parser] = {}
        self._cache: Dict[str, ParseResult] = {}
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
        
    async def register_parser(self, language: ProgrammingLanguage, parser: Parser):
        """Registra un parser para un lenguaje específico."""
        self._parsers[language] = parser
        self._logger.info(f"Parser registrado para {language.value}")
    
    async def parse_file(self, file_path: Path) -> ParseResult:
        """Parsea un archivo detectando automáticamente el lenguaje."""
        start_time = asyncio.get_event_loop().time()
        
        # Verificar cache
        if self.config.enable_cache:
            cache_key = f"{file_path}:{file_path.stat().st_mtime}"
            if cache_key in self._cache:
                self._logger.debug(f"Cache hit para {file_path}")
                return self._cache[cache_key]
        
        # Detectar lenguaje
        language = self._detect_language(file_path)
        
        # Obtener parser apropiado
        parser = self._parsers.get(language)
        if not parser:
            error_msg = f"No parser disponible para {language.value}"
            self._logger.error(error_msg)
            return ParseResult(
                file_path=file_path,
                language=language,
                errors=[error_msg]
            )
        
        try:
            # Parsear archivo
            result = await parser.parse_file(file_path)
            result.parse_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Guardar en cache
            if self.config.enable_cache:
                await self._add_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error parsing {file_path}: {str(e)}"
            self._logger.error(error_msg, exc_info=True)
            return ParseResult(
                file_path=file_path,
                language=language,
                errors=[error_msg],
                parse_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    async def parse_content(self, content: str, language: ProgrammingLanguage) -> ParseResult:
        """Parsea contenido de texto."""
        start_time = asyncio.get_event_loop().time()
        
        # Obtener parser apropiado
        parser = self._parsers.get(language)
        if not parser:
            error_msg = f"No parser disponible para {language.value}"
            self._logger.error(error_msg)
            return ParseResult(
                file_path=Path("<string>"),
                language=language,
                errors=[error_msg]
            )
        
        try:
            # Parsear contenido
            result = await parser.parse_content(content, language)
            result.parse_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return result
            
        except Exception as e:
            error_msg = f"Error parsing content: {str(e)}"
            self._logger.error(error_msg, exc_info=True)
            return ParseResult(
                file_path=Path("<string>"),
                language=language,
                errors=[error_msg],
                parse_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    def _detect_language(self, file_path: Path) -> ProgrammingLanguage:
        """Detecta el lenguaje basado en la extensión del archivo."""
        extension_map = {
            '.py': ProgrammingLanguage.PYTHON,
            '.pyw': ProgrammingLanguage.PYTHON,
            '.ts': ProgrammingLanguage.TYPESCRIPT,
            '.tsx': ProgrammingLanguage.TSX,
            '.js': ProgrammingLanguage.JAVASCRIPT,
            '.mjs': ProgrammingLanguage.JAVASCRIPT,
            '.cjs': ProgrammingLanguage.JAVASCRIPT,
            '.jsx': ProgrammingLanguage.JSX,
            '.rs': ProgrammingLanguage.RUST,
            '.go': ProgrammingLanguage.GO,
            '.java': ProgrammingLanguage.JAVA,
            '.cpp': ProgrammingLanguage.CPP,
            '.cc': ProgrammingLanguage.CPP,
            '.cxx': ProgrammingLanguage.CPP,
            '.hpp': ProgrammingLanguage.CPP,
            '.h': ProgrammingLanguage.CPP,
            '.cs': ProgrammingLanguage.CSHARP,
            '.php': ProgrammingLanguage.PHP,
            '.rb': ProgrammingLanguage.RUBY,
            '.swift': ProgrammingLanguage.SWIFT,
            '.kt': ProgrammingLanguage.KOTLIN,
            '.scala': ProgrammingLanguage.SCALA,
        }
        
        return extension_map.get(file_path.suffix.lower(), ProgrammingLanguage.UNKNOWN)
    
    async def _add_to_cache(self, key: str, result: ParseResult):
        """Agrega un resultado al cache con gestión de tamaño."""
        async with self._lock:
            # Limpiar cache si está lleno
            if len(self._cache) >= self.config.cache_size:
                # Eliminar el 20% más antiguo
                items_to_remove = int(self.config.cache_size * 0.2)
                keys_to_remove = list(self._cache.keys())[:items_to_remove]
                for k in keys_to_remove:
                    del self._cache[k]
            
            self._cache[key] = result
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Retorna los lenguajes soportados."""
        return list(self._parsers.keys())
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del cache."""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.config.cache_size,
            "cache_hit_ratio": 0.0,  # TODO: Implementar tracking
        }


# Singleton para el parser universal
_global_parser: Optional[UniversalParser] = None


async def get_universal_parser(config: Optional[ParserConfig] = None) -> UniversalParser:
    """Obtiene la instancia global del parser universal."""
    global _global_parser
    
    if _global_parser is None:
        if config is None:
            config = ParserConfig()
        _global_parser = UniversalParser(config)
    elif config is not None:
        # Si se proporciona una nueva configuración, crear una nueva instancia
        _global_parser = UniversalParser(config)
    
    return _global_parser


async def initialize_parsers():
    """Inicializa todos los parsers disponibles."""
    try:
        from .python_parser import PythonSpecializedParser
        from .typescript_parser import TypeScriptSpecializedParser
        
        parser = await get_universal_parser()
        
        # Registrar parser de Python
        python_parser = PythonSpecializedParser()
        await parser.register_parser(ProgrammingLanguage.PYTHON, python_parser)
        
        # Registrar parser de TypeScript/JavaScript
        typescript_parser = TypeScriptSpecializedParser()
        await parser.register_parser(ProgrammingLanguage.TYPESCRIPT, typescript_parser)
        await parser.register_parser(ProgrammingLanguage.JAVASCRIPT, typescript_parser)
        await parser.register_parser(ProgrammingLanguage.JSX, typescript_parser)
        await parser.register_parser(ProgrammingLanguage.TSX, typescript_parser)
        
        # Registrar parser de Rust
        from .rust_parser import RustSpecializedParser
        rust_parser = RustSpecializedParser()
        await parser.register_parser(ProgrammingLanguage.RUST, rust_parser)
        
        logger.info("Parsers inicializados correctamente")
    except ImportError as e:
        logger.warning(f"No se pudo importar algún parser: {e}")
        logger.info("Parsers inicializados con parsers disponibles")


__all__ = [
    'ProgrammingLanguage',
    'ParseResult', 
    'ParserConfig',
    'Parser',
    'UniversalParser',
    'get_universal_parser',
    'initialize_parsers'
]
