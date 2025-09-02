"""
Entidad Project que representa un proyecto de código.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from ..value_objects.project_id import ProjectId
from ..value_objects.programming_language import ProgrammingLanguage
from ..value_objects.repository_type import ProjectStatus, RepositoryType


@dataclass
class ProjectSettings:
    """Configuración de un proyecto."""
    analysis_config: Dict[str, Any] = field(default_factory=dict)
    ignore_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    max_file_size_mb: int = 10
    enable_incremental_analysis: bool = True
    webhook_url: Optional[str] = None
    notification_settings: Dict[str, Any] = field(default_factory=dict)
    custom_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProjectMetadata:
    """Metadatos adicionales del proyecto."""
    stars: int = 0
    forks: int = 0
    issues: int = 0
    pull_requests: int = 0
    contributors: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    license: Optional[str] = None
    homepage: Optional[str] = None
    description: Optional[str] = None
    language_stats: Dict[str, int] = field(default_factory=dict)


@dataclass
class Project:
    """
    Entidad que representa un proyecto de código.
    
    Attributes:
        id: Identificador único del proyecto
        organization_id: ID de la organización a la que pertenece
        name: Nombre del proyecto
        description: Descripción opcional del proyecto
        slug: Slug único del proyecto
        status: Estado actual del proyecto
        settings: Configuración del proyecto
        metadata: Metadatos adicionales del proyecto
        created_by: ID del usuario que creó el proyecto
        created_at: Fecha de creación
        updated_at: Fecha de última actualización
    """
    id: ProjectId
    organization_id: "OrganizationId"
    name: str
    description: Optional[str]
    slug: str
    status: ProjectStatus
    settings: ProjectSettings
    metadata: ProjectMetadata
    created_by: "UserId"
    created_at: datetime
    updated_at: datetime
    
    def __post_init__(self) -> None:
        """Validar que el Project sea válido."""
        if not self.name:
            raise ValueError("Project name no puede estar vacío")
        
        if not self.slug:
            raise ValueError("Project slug no puede estar vacío")
    
    @classmethod
    def create(
        cls,
        organization_id: "OrganizationId",
        name: str,
        slug: str,
        created_by: "UserId",
        description: Optional[str] = None
    ) -> "Project":
        """Crear un nuevo proyecto."""
        now = datetime.utcnow()
        
        return cls(
            id=ProjectId.generate(),
            organization_id=organization_id,
            name=name,
            description=description,
            slug=slug,
            status=ProjectStatus.ACTIVE,
            settings=ProjectSettings(),
            metadata=ProjectMetadata(),
            created_by=created_by,
            created_at=now,
            updated_at=now
        )
    
    def update_languages(self, languages: List[ProgrammingLanguage]) -> None:
        """Actualizar la lista de lenguajes del proyecto."""
        self.languages = languages
        self.updated_at = datetime.utcnow()
    
    def mark_as_analyzed(self) -> None:
        """Marcar el proyecto como analizado."""
        self.last_analyzed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_status(self, status: ProjectStatus) -> None:
        """Actualizar el estado del proyecto."""
        self.status = status
        self.updated_at = datetime.utcnow()
    
    def update_settings(self, settings: ProjectSettings) -> None:
        """Actualizar la configuración del proyecto."""
        self.settings = settings
        self.updated_at = datetime.utcnow()
    
    def update_metadata(self, metadata: ProjectMetadata) -> None:
        """Actualizar los metadatos del proyecto."""
        self.metadata = metadata
        self.updated_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Verificar si el proyecto está activo."""
        return self.status == ProjectStatus.ACTIVE
    
    def can_be_analyzed(self) -> bool:
        """Verificar si el proyecto puede ser analizado."""
        return self.is_active() and self.repository_url
    
    def get_primary_language(self) -> Optional[ProgrammingLanguage]:
        """Obtener el lenguaje principal del proyecto."""
        if not self.languages:
            return None
        return self.languages[0]
    
    def __str__(self) -> str:
        return f"Project({self.name}, {self.id})"
    
    def __repr__(self) -> str:
        return f"Project(id={self.id}, name='{self.name}', status={self.status.value})"
