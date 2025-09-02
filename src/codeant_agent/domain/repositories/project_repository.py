"""
Interfaz del repositorio de proyectos.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol
from codeant_agent.domain.entities.project import Project
from codeant_agent.domain.value_objects.project_id import ProjectId
from codeant_agent.domain.value_objects.repository_type import ProjectStatus
from codeant_agent.utils.error import Result


class ProjectRepository(Protocol):
    """
    Protocolo para el repositorio de proyectos.
    
    Define las operaciones que debe implementar cualquier repositorio
    de proyectos, siguiendo el patrón Repository del Domain-Driven Design.
    """
    
    async def save(self, project: Project) -> Result[Project, Exception]:
        """
        Guardar un proyecto.
        
        Args:
            project: Proyecto a guardar
            
        Returns:
            Result con el proyecto guardado o error
        """
        ...
    
    async def find_by_id(self, project_id: ProjectId) -> Result[Optional[Project], Exception]:
        """
        Buscar un proyecto por su ID.
        
        Args:
            project_id: ID del proyecto a buscar
            
        Returns:
            Result con el proyecto encontrado o None si no existe
        """
        ...
    
    async def find_by_name(self, name: str) -> Result[Optional[Project], Exception]:
        """
        Buscar un proyecto por su nombre.
        
        Args:
            name: Nombre del proyecto a buscar
            
        Returns:
            Result con el proyecto encontrado o None si no existe
        """
        ...
    
    async def find_by_repository_url(self, repository_url: str) -> Result[Optional[Project], Exception]:
        """
        Buscar un proyecto por su URL de repositorio.
        
        Args:
            repository_url: URL del repositorio a buscar
            
        Returns:
            Result con el proyecto encontrado o None si no existe
        """
        ...
    
    async def find_all(self, status: Optional[ProjectStatus] = None) -> Result[List[Project], Exception]:
        """
        Obtener todos los proyectos, opcionalmente filtrados por estado.
        
        Args:
            status: Estado opcional para filtrar los proyectos
            
        Returns:
            Result con la lista de proyectos
        """
        ...
    
    async def find_active_projects(self) -> Result[List[Project], Exception]:
        """
        Obtener todos los proyectos activos.
        
        Returns:
            Result con la lista de proyectos activos
        """
        ...
    
    async def delete(self, project_id: ProjectId) -> Result[bool, Exception]:
        """
        Eliminar un proyecto.
        
        Args:
            project_id: ID del proyecto a eliminar
            
        Returns:
            Result con True si se eliminó correctamente
        """
        ...
    
    async def exists(self, project_id: ProjectId) -> Result[bool, Exception]:
        """
        Verificar si existe un proyecto con el ID dado.
        
        Args:
            project_id: ID del proyecto a verificar
            
        Returns:
            Result con True si existe, False en caso contrario
        """
        ...
    
    async def count(self, status: Optional[ProjectStatus] = None) -> Result[int, Exception]:
        """
        Contar el número de proyectos, opcionalmente filtrados por estado.
        
        Args:
            status: Estado opcional para filtrar los proyectos
            
        Returns:
            Result con el número de proyectos
        """
        ...
