"""
Interfaz del repositorio de repositorios.
"""
from typing import List, Optional, Protocol
from codeant_agent.domain.entities.repository import Repository
from codeant_agent.domain.value_objects.repository_id import RepositoryId
from codeant_agent.domain.value_objects.project_id import ProjectId
from codeant_agent.domain.value_objects.repository_type import SyncStatus
from codeant_agent.utils.error import Result


class RepositoryRepository(Protocol):
    """
    Protocolo para el repositorio de repositorios.
    
    Define las operaciones que debe implementar cualquier repositorio
    de repositorios, siguiendo el patrón Repository del Domain-Driven Design.
    """
    
    async def save(self, repository: Repository) -> Result[Repository, Exception]:
        """
        Guardar un repositorio.
        
        Args:
            repository: Repositorio a guardar
            
        Returns:
            Result con el repositorio guardado o error
        """
        ...
    
    async def find_by_id(self, repository_id: RepositoryId) -> Result[Optional[Repository], Exception]:
        """
        Buscar un repositorio por su ID.
        
        Args:
            repository_id: ID del repositorio a buscar
            
        Returns:
            Result con el repositorio encontrado o None si no existe
        """
        ...
    
    async def find_by_project_id(self, project_id: ProjectId) -> Result[Optional[Repository], Exception]:
        """
        Buscar un repositorio por el ID de su proyecto.
        
        Args:
            project_id: ID del proyecto
            
        Returns:
            Result con el repositorio encontrado o None si no existe
        """
        ...
    
    async def find_by_local_path(self, local_path: str) -> Result[Optional[Repository], Exception]:
        """
        Buscar un repositorio por su ruta local.
        
        Args:
            local_path: Ruta local del repositorio
            
        Returns:
            Result con el repositorio encontrado o None si no existe
        """
        ...
    
    async def find_by_remote_url(self, remote_url: str) -> Result[Optional[Repository], Exception]:
        """
        Buscar un repositorio por su URL remota.
        
        Args:
            remote_url: URL remota del repositorio
            
        Returns:
            Result con el repositorio encontrado o None si no existe
        """
        ...
    
    async def find_all(self, sync_status: Optional[SyncStatus] = None) -> Result[List[Repository], Exception]:
        """
        Obtener todos los repositorios, opcionalmente filtrados por estado de sincronización.
        
        Args:
            sync_status: Estado de sincronización opcional para filtrar
            
        Returns:
            Result con la lista de repositorios
        """
        ...
    
    async def find_synced_repositories(self) -> Result[List[Repository], Exception]:
        """
        Obtener todos los repositorios sincronizados.
        
        Returns:
            Result con la lista de repositorios sincronizados
        """
        ...
    
    async def find_failed_sync_repositories(self) -> Result[List[Repository], Exception]:
        """
        Obtener todos los repositorios con sincronización fallida.
        
        Returns:
            Result con la lista de repositorios con sincronización fallida
        """
        ...
    
    async def delete(self, repository_id: RepositoryId) -> Result[bool, Exception]:
        """
        Eliminar un repositorio.
        
        Args:
            repository_id: ID del repositorio a eliminar
            
        Returns:
            Result con True si se eliminó correctamente
        """
        ...
    
    async def delete_by_project_id(self, project_id: ProjectId) -> Result[bool, Exception]:
        """
        Eliminar un repositorio por el ID de su proyecto.
        
        Args:
            project_id: ID del proyecto
            
        Returns:
            Result con True si se eliminó correctamente
        """
        ...
    
    async def exists(self, repository_id: RepositoryId) -> Result[bool, Exception]:
        """
        Verificar si existe un repositorio con el ID dado.
        
        Args:
            repository_id: ID del repositorio a verificar
            
        Returns:
            Result con True si existe, False en caso contrario
        """
        ...
    
    async def count(self, sync_status: Optional[SyncStatus] = None) -> Result[int, Exception]:
        """
        Contar el número de repositorios, opcionalmente filtrados por estado de sincronización.
        
        Args:
            sync_status: Estado de sincronización opcional para filtrar
            
        Returns:
            Result con el número de repositorios
        """
        ...
