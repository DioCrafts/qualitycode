"""
Interfaz del repositorio de parser universal.

Este módulo define el contrato para el repositorio de parser que
proporciona capacidades de parsing universal usando Tree-sitter.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..value_objects.programming_language import ProgrammingLanguage
from ..entities.parse_result import ParseResult, ParseRequest


class ParserRepository(ABC):
    """
    Interfaz del repositorio de parser universal.
    
    Esta interfaz define el contrato para el repositorio que maneja
    el parsing de código fuente en múltiples lenguajes de programación
    usando Tree-sitter como motor de parsing.
    """
    
    @abstractmethod
    async def parse_file(self, file_path: Path) -> ParseResult:
        """
        Parsea un archivo de código fuente.
        
        Args:
            file_path: Ruta al archivo a parsear
            
        Returns:
            ParseResult con el AST y metadatos del parsing
            
        Raises:
            ParserError: Si hay un error durante el parsing
        """
        pass
    
    @abstractmethod
    async def parse_content(self, content: str, language: ProgrammingLanguage) -> ParseResult:
        """
        Parsea contenido de código fuente directamente.
        
        Args:
            content: Contenido del código fuente
            language: Lenguaje de programación del contenido
            
        Returns:
            ParseResult con el AST y metadatos del parsing
            
        Raises:
            ParserError: Si hay un error durante el parsing
        """
        pass
    
    @abstractmethod
    async def parse_incremental(
        self, 
        old_tree: Any, 
        content: str, 
        edits: List[Dict[str, Any]]
    ) -> ParseResult:
        """
        Parsea incrementalmente un archivo con cambios.
        
        Args:
            old_tree: Árbol AST anterior
            content: Nuevo contenido del archivo
            edits: Lista de ediciones aplicadas
            
        Returns:
            ParseResult con el nuevo AST
            
        Raises:
            ParserError: Si hay un error durante el parsing incremental
        """
        pass
    
    @abstractmethod
    async def detect_language(self, file_path: Path, content: str) -> ProgrammingLanguage:
        """
        Detecta automáticamente el lenguaje de programación.
        
        Args:
            file_path: Ruta al archivo
            content: Contenido del archivo
            
        Returns:
            ProgrammingLanguage detectado
            
        Raises:
            LanguageDetectionError: Si no se puede detectar el lenguaje
        """
        pass
    
    @abstractmethod
    async def get_ast_json(self, tree: Any) -> str:
        """
        Convierte un AST a formato JSON.
        
        Args:
            tree: Árbol AST a convertir
            
        Returns:
            String JSON del AST
            
        Raises:
            SerializationError: Si hay un error en la serialización
        """
        pass
    
    @abstractmethod
    async def query_ast(
        self, 
        tree: Any, 
        query: str, 
        language: ProgrammingLanguage
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta una consulta sobre un AST.
        
        Args:
            tree: Árbol AST a consultar
            query: Consulta en formato Tree-sitter
            language: Lenguaje de programación del AST
            
        Returns:
            Lista de coincidencias de la consulta
            
        Raises:
            QueryError: Si hay un error en la consulta
        """
        pass
    
    @abstractmethod
    async def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """
        Obtiene la lista de lenguajes soportados.
        
        Returns:
            Lista de lenguajes de programación soportados
        """
        pass
    
    @abstractmethod
    async def is_language_supported(self, language: ProgrammingLanguage) -> bool:
        """
        Verifica si un lenguaje está soportado.
        
        Args:
            language: Lenguaje a verificar
            
        Returns:
            True si el lenguaje está soportado, False en caso contrario
        """
        pass
    
    @abstractmethod
    async def get_parser_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del parser.
        
        Returns:
            Diccionario con estadísticas del parser
        """
        pass
    
    @abstractmethod
    async def clear_cache(self) -> None:
        """
        Limpia la caché del parser.
        """
        pass
    
    @abstractmethod
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la caché.
        
        Returns:
            Diccionario con estadísticas de la caché
        """
        pass
