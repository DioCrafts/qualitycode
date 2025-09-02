"""
Entidad FileIndex que representa un archivo indexado.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from codeant_agent.domain.value_objects.file_id import FileId
from codeant_agent.domain.value_objects.repository_id import RepositoryId
from codeant_agent.domain.value_objects.programming_language import ProgrammingLanguage
from codeant_agent.domain.value_objects.repository_type import AnalysisStatus


@dataclass
class FileMetadata:
    """Metadatos adicionales del archivo."""
    encoding: str = "utf-8"
    line_ending: str = "\n"
    has_bom: bool = False
    is_binary: bool = False
    is_executable: bool = False
    is_symlink: bool = False
    permissions: Optional[str] = None
    owner: Optional[str] = None
    group: Optional[str] = None
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileIndex:
    """
    Entidad que representa un archivo indexado.
    
    Attributes:
        id: Identificador único del archivo
        repository_id: ID del repositorio al que pertenece
        file_path: Ruta del archivo en el repositorio
        file_name: Nombre del archivo
        file_extension: Extensión del archivo
        language: Lenguaje de programación detectado
        mime_type: Tipo MIME del archivo
        size_bytes: Tamaño del archivo en bytes
        line_count: Número de líneas del archivo
        commit_hash: Hash del commit donde se indexó
        branch_name: Nombre de la rama
        is_binary: Si el archivo es binario
        is_ignored: Si el archivo fue ignorado
        metadata: Metadatos adicionales del archivo
        created_at: Fecha de creación
        updated_at: Fecha de última actualización
    """
    id: FileId
    repository_id: RepositoryId
    file_path: str
    file_name: str
    file_extension: Optional[str]
    language: Optional[ProgrammingLanguage]
    mime_type: Optional[str]
    size_bytes: int
    line_count: int
    commit_hash: str
    branch_name: str
    is_binary: bool
    is_ignored: bool
    metadata: FileMetadata
    created_at: datetime
    updated_at: datetime
    
    def __post_init__(self) -> None:
        """Validar que el FileIndex sea válido."""
        if not self.file_path:
            raise ValueError("FileIndex file_path no puede estar vacío")
        
        if not self.file_name:
            raise ValueError("FileIndex file_name no puede estar vacío")
        
        if self.size_bytes < 0:
            raise ValueError("FileIndex size_bytes no puede ser negativo")
        
        if self.line_count < 0:
            raise ValueError("FileIndex line_count no puede ser negativo")
        
        if not self.commit_hash:
            raise ValueError("FileIndex commit_hash no puede estar vacío")
        
        if not self.branch_name:
            raise ValueError("FileIndex branch_name no puede estar vacío")
    
    @classmethod
    def create(
        cls,
        repository_id: RepositoryId,
        file_path: str,
        file_name: str,
        file_extension: Optional[str],
        size_bytes: int,
        line_count: int,
        commit_hash: str,
        branch_name: str,
        language: Optional[ProgrammingLanguage] = None,
        mime_type: Optional[str] = None,
        is_binary: bool = False,
        is_ignored: bool = False
    ) -> "FileIndex":
        """Crear un nuevo FileIndex."""
        return cls(
            id=FileId.generate(),
            repository_id=repository_id,
            file_path=file_path,
            file_name=file_name,
            file_extension=file_extension,
            language=language,
            mime_type=mime_type,
            size_bytes=size_bytes,
            line_count=line_count,
            commit_hash=commit_hash,
            branch_name=branch_name,
            is_binary=is_binary,
            is_ignored=is_ignored,
            metadata=FileMetadata(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    def update_language(self, language: ProgrammingLanguage) -> None:
        """Actualizar el lenguaje detectado del archivo."""
        self.language = language
        self.updated_at = datetime.utcnow()
    
    def update_metadata(self, metadata: FileMetadata) -> None:
        """Actualizar los metadatos del archivo."""
        self.metadata = metadata
        self.updated_at = datetime.utcnow()
    
    def update_content_info(self, size_bytes: int, line_count: int, commit_hash: str) -> None:
        """Actualizar la información del contenido del archivo."""
        self.size_bytes = size_bytes
        self.line_count = line_count
        self.commit_hash = commit_hash
        self.updated_at = datetime.utcnow()
    
    def get_extension(self) -> str:
        """Obtener la extensión del archivo."""
        if "." not in self.file_path:
            return ""
        return self.file_path.split(".")[-1]
    
    def get_filename(self) -> str:
        """Obtener el nombre del archivo sin la ruta."""
        return self.file_path.split("/")[-1]
    
    def get_directory(self) -> str:
        """Obtener el directorio del archivo."""
        parts = self.file_path.split("/")
        if len(parts) <= 1:
            return ""
        return "/".join(parts[:-1])
    
    def is_analyzed(self) -> bool:
        """Verificar si el archivo ha sido analizado."""
        # Por ahora, consideramos que un archivo está analizado si tiene lenguaje detectado
        return self.language is not None and self.language.type.value != "unknown"
    
    def is_analysis_failed(self) -> bool:
        """Verificar si el análisis del archivo falló."""
        # Por ahora, consideramos que falló si está ignorado
        return self.is_ignored
    
    def is_skipped(self) -> bool:
        """Verificar si el archivo fue omitido del análisis."""
        return self.is_ignored
    
    def size_kb(self) -> float:
        """Obtener el tamaño del archivo en KB."""
        return self.size_bytes / 1024
    
    def size_mb(self) -> float:
        """Obtener el tamaño del archivo en MB."""
        return self.size_bytes / (1024 * 1024)
    
    def is_text_file(self) -> bool:
        """Verificar si el archivo es de texto."""
        return not self.is_binary
    
    def is_source_code(self) -> bool:
        """Verificar si el archivo es código fuente."""
        return self.language is not None and self.language.type.value != "unknown"
    
    def __str__(self) -> str:
        return f"FileIndex({self.file_path}, {self.id})"
    
    def __repr__(self) -> str:
        return f"FileIndex(id={self.id}, path='{self.file_path}', language={self.language})"
