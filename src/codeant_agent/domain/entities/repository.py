"""
Entidad Repository que representa un repositorio local.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from codeant_agent.domain.value_objects.repository_id import RepositoryId
from codeant_agent.domain.value_objects.project_id import ProjectId
from codeant_agent.domain.value_objects.commit_hash import CommitHash
from codeant_agent.domain.value_objects.repository_type import SyncStatus, RepositoryType


@dataclass
class Branch:
    """Información de una rama del repositorio."""
    name: str
    commit_hash: CommitHash
    is_default: bool = False
    is_protected: bool = False
    last_commit_date: Optional[datetime] = None


@dataclass
class Tag:
    """Información de un tag del repositorio."""
    name: str
    commit_hash: CommitHash
    message: Optional[str] = None
    author: Optional[str] = None
    date: Optional[datetime] = None


@dataclass
class Repository:
    """
    Entidad que representa un repositorio local.
    
    Attributes:
        id: Identificador único del repositorio
        project_id: ID del proyecto asociado
        name: Nombre del repositorio
        url: URL del repositorio remoto
        type: Tipo de repositorio (git, svn, etc.)
        default_branch: Rama por defecto
        sync_status: Estado de la sincronización
        size_bytes: Tamaño del repositorio en bytes
        file_count: Número de archivos en el repositorio
        language_stats: Estadísticas de lenguajes
        settings: Configuración del repositorio
        branches: Lista de ramas del repositorio
        tags: Lista de tags del repositorio
        created_at: Fecha de creación
        updated_at: Fecha de última actualización
    """
    id: RepositoryId
    project_id: ProjectId
    name: str
    url: str
    type: RepositoryType
    default_branch: str
    sync_status: SyncStatus
    size_bytes: int
    file_count: int
    language_stats: dict
    settings: dict
    branches: List[Branch]
    tags: List[Tag]
    created_at: datetime
    updated_at: datetime
    
    def __post_init__(self) -> None:
        """Validar que el Repository sea válido."""
        if not self.name:
            raise ValueError("Repository name no puede estar vacío")
        
        if not self.url:
            raise ValueError("Repository url no puede estar vacío")
        
        if self.size_bytes < 0:
            raise ValueError("Repository size_bytes no puede ser negativo")
        
        if self.file_count < 0:
            raise ValueError("Repository file_count no puede ser negativo")
    
    @classmethod
    def create(
        cls,
        project_id: ProjectId,
        name: str,
        url: str,
        type: RepositoryType = RepositoryType.GIT,
        default_branch: str = "main"
    ) -> "Repository":
        """Crear un nuevo repositorio."""
        now = datetime.utcnow()
        
        return cls(
            id=RepositoryId.generate(),
            project_id=project_id,
            name=name,
            url=url,
            type=type,
            default_branch=default_branch,
            sync_status=SyncStatus.PENDING,
            size_bytes=0,
            file_count=0,
            language_stats={},
            settings={},
            branches=[],
            tags=[],
            created_at=now,
            updated_at=now
        )
    
    def update_branches(self, branches: List[Branch]) -> None:
        """Actualizar la lista de ramas del repositorio."""
        self.branches = branches
        self.updated_at = datetime.utcnow()
    
    def update_tags(self, tags: List[Tag]) -> None:
        """Actualizar la lista de tags del repositorio."""
        self.tags = tags
        self.updated_at = datetime.utcnow()
    
    def update_stats(self, size_bytes: int, file_count: int) -> None:
        """Actualizar las estadísticas del repositorio."""
        self.size_bytes = size_bytes
        self.file_count = file_count
        self.updated_at = datetime.utcnow()
    
    def update_sync_status(self, status: SyncStatus) -> None:
        """Actualizar el estado de sincronización."""
        self.sync_status = status
        self.updated_at = datetime.utcnow()
    
    def get_default_branch(self) -> Optional[Branch]:
        """Obtener la rama por defecto del repositorio."""
        for branch in self.branches:
            if branch.is_default:
                return branch
        return None
    
    def get_branch_by_name(self, name: str) -> Optional[Branch]:
        """Obtener una rama por su nombre."""
        for branch in self.branches:
            if branch.name == name:
                return branch
        return None
    
    def get_tag_by_name(self, name: str) -> Optional[Tag]:
        """Obtener un tag por su nombre."""
        for tag in self.tags:
            if tag.name == name:
                return tag
        return None
    
    def is_synced(self) -> bool:
        """Verificar si el repositorio está sincronizado."""
        return self.sync_status == SyncStatus.COMPLETED
    
    def is_sync_failed(self) -> bool:
        """Verificar si la sincronización falló."""
        return self.sync_status == SyncStatus.FAILED
    
    def size_mb(self) -> float:
        """Obtener el tamaño del repositorio en MB."""
        return self.size_bytes / (1024 * 1024)
    
    def size_gb(self) -> float:
        """Obtener el tamaño del repositorio en GB."""
        return self.size_bytes / (1024 * 1024 * 1024)
    
    def __str__(self) -> str:
        return f"Repository({self.name}, {self.id})"
    
    def __repr__(self) -> str:
        return f"Repository(id={self.id}, name='{self.name}', status={self.sync_status.value})"