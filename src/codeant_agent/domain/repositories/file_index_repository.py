"""
Interfaz del repositorio de índices de archivos.
"""
from typing import List, Optional, Protocol
from codeant_agent.domain.entities.file_index import FileIndex
from codeant_agent.domain.value_objects.file_id import FileId
from codeant_agent.domain.value_objects.repository_id import RepositoryId
from codeant_agent.domain.value_objects.programming_language import ProgrammingLanguage
from codeant_agent.domain.value_objects.repository_type import AnalysisStatus
from codeant_agent.utils.error import Result


class FileIndexRepository(Protocol):
    """
    Protocolo para el repositorio de índices de archivos.
    
    Define las operaciones que debe implementar cualquier repositorio
    de índices de archivos, siguiendo el patrón Repository del Domain-Driven Design.
    """
    
    async def save(self, file_index: FileIndex) -> Result[FileIndex, Exception]:
        """
        Guardar un índice de archivo.
        
        Args:
            file_index: Índice de archivo a guardar
            
        Returns:
            Result con el índice guardado o error
        """
        ...
    
    async def save_many(self, file_indices: List[FileIndex]) -> Result[List[FileIndex], Exception]:
        """
        Guardar múltiples índices de archivo.
        
        Args:
            file_indices: Lista de índices de archivo a guardar
            
        Returns:
            Result con la lista de índices guardados o error
        """
        ...
    
    async def find_by_id(self, file_id: FileId) -> Result[Optional[FileIndex], Exception]:
        """
        Buscar un índice de archivo por su ID.
        
        Args:
            file_id: ID del archivo a buscar
            
        Returns:
            Result con el índice encontrado o None si no existe
        """
        ...
    
    async def find_by_repository_id(self, repository_id: RepositoryId) -> Result[List[FileIndex], Exception]:
        """
        Buscar todos los índices de archivo de un repositorio.
        
        Args:
            repository_id: ID del repositorio
            
        Returns:
            Result con la lista de índices de archivo
        """
        ...
    
    async def find_by_path(self, repository_id: RepositoryId, relative_path: str) -> Result[Optional[FileIndex], Exception]:
        """
        Buscar un índice de archivo por su ruta relativa en un repositorio.
        
        Args:
            repository_id: ID del repositorio
            relative_path: Ruta relativa del archivo
            
        Returns:
            Result con el índice encontrado o None si no existe
        """
        ...
    
    async def find_by_language(self, repository_id: RepositoryId, language: ProgrammingLanguage) -> Result[List[FileIndex], Exception]:
        """
        Buscar índices de archivo por lenguaje de programación en un repositorio.
        
        Args:
            repository_id: ID del repositorio
            language: Lenguaje de programación
            
        Returns:
            Result con la lista de índices de archivo
        """
        ...
    
    async def find_by_analysis_status(self, repository_id: RepositoryId, status: AnalysisStatus) -> Result[List[FileIndex], Exception]:
        """
        Buscar índices de archivo por estado de análisis en un repositorio.
        
        Args:
            repository_id: ID del repositorio
            status: Estado de análisis
            
        Returns:
            Result con la lista de índices de archivo
        """
        ...
    
    async def find_source_code_files(self, repository_id: RepositoryId) -> Result[List[FileIndex], Exception]:
        """
        Buscar todos los archivos de código fuente en un repositorio.
        
        Args:
            repository_id: ID del repositorio
            
        Returns:
            Result con la lista de archivos de código fuente
        """
        ...
    
    async def find_by_extension(self, repository_id: RepositoryId, extension: str) -> Result[List[FileIndex], Exception]:
        """
        Buscar índices de archivo por extensión en un repositorio.
        
        Args:
            repository_id: ID del repositorio
            extension: Extensión de archivo (con o sin punto)
            
        Returns:
            Result con la lista de índices de archivo
        """
        ...
    
    async def find_large_files(self, repository_id: RepositoryId, min_size_mb: float) -> Result[List[FileIndex], Exception]:
        """
        Buscar archivos grandes en un repositorio.
        
        Args:
            repository_id: ID del repositorio
            min_size_mb: Tamaño mínimo en MB
            
        Returns:
            Result con la lista de archivos grandes
        """
        ...
    
    async def delete(self, file_id: FileId) -> Result[bool, Exception]:
        """
        Eliminar un índice de archivo.
        
        Args:
            file_id: ID del archivo a eliminar
            
        Returns:
            Result con True si se eliminó correctamente
        """
        ...
    
    async def delete_by_repository_id(self, repository_id: RepositoryId) -> Result[bool, Exception]:
        """
        Eliminar todos los índices de archivo de un repositorio.
        
        Args:
            repository_id: ID del repositorio
            
        Returns:
            Result con True si se eliminaron correctamente
        """
        ...
    
    async def exists(self, file_id: FileId) -> Result[bool, Exception]:
        """
        Verificar si existe un índice de archivo con el ID dado.
        
        Args:
            file_id: ID del archivo a verificar
            
        Returns:
            Result con True si existe, False en caso contrario
        """
        ...
    
    async def count(self, repository_id: Optional[RepositoryId] = None) -> Result[int, Exception]:
        """
        Contar el número de índices de archivo, opcionalmente filtrados por repositorio.
        
        Args:
            repository_id: ID del repositorio opcional para filtrar
            
        Returns:
            Result con el número de índices de archivo
        """
        ...
    
    async def get_language_stats(self, repository_id: RepositoryId) -> Result[dict, Exception]:
        """
        Obtener estadísticas de lenguajes de programación en un repositorio.
        
        Args:
            repository_id: ID del repositorio
            
        Returns:
            Result con las estadísticas de lenguajes
        """
        ...
