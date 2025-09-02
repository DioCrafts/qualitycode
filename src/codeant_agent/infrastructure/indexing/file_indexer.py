"""
Sistema de indexación de archivos.
"""
import os
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import fnmatch

from codeant_agent.domain.entities.file_index import FileIndex, FileMetadata
from codeant_agent.domain.entities.repository import Repository
from codeant_agent.domain.value_objects.programming_language import ProgrammingLanguage
from codeant_agent.domain.value_objects.repository_type import AnalysisStatus
from codeant_agent.utils.error import Result, BaseError, ValidationError
from codeant_agent.utils.logging import get_logger

logger = get_logger(__name__)


class IndexingError(BaseError):
    """Error en operaciones de indexación."""
    pass


class FileIndexingError(IndexingError):
    """Error al indexar un archivo específico."""
    pass


@dataclass
class IndexResult:
    """Resultado de una operación de indexación."""
    total_files: int
    indexed_files: int
    skipped_files: int
    failed_files: int
    total_size_bytes: int
    languages_detected: Set[ProgrammingLanguage]
    processing_time_seconds: float


class FileIndexer:
    """
    Sistema de indexación de archivos.
    
    Responsable de escanear repositorios, detectar archivos,
    calcular metadatos y crear índices de archivos.
    """
    
    def __init__(self):
        """Inicializar el indexador de archivos."""
        # Patrones de archivos a ignorar por defecto
        self.default_ignore_patterns = {
            # Archivos del sistema
            '.DS_Store', 'Thumbs.db', 'desktop.ini',
            
            # Archivos temporales
            '*.tmp', '*.temp', '*.swp', '*.swo', '*~',
            
            # Archivos de build
            '*.o', '*.obj', '*.exe', '*.dll', '*.so', '*.dylib',
            '*.class', '*.jar', '*.war', '*.ear',
            '*.pyc', '*.pyo', '__pycache__',
            '*.min.js', '*.min.css',
            
            # Directorios de build
            'build/', 'dist/', 'target/', 'out/', 'bin/', 'obj/',
            'node_modules/', 'vendor/', '.venv/', 'venv/',
            
            # Archivos de configuración del sistema
            '.git/', '.svn/', '.hg/', '.bzr/',
            '.idea/', '.vscode/', '.vs/',
            
            # Archivos de logs
            '*.log', 'logs/',
            
            # Archivos de datos
            '*.db', '*.sqlite', '*.sqlite3',
            '*.csv', '*.tsv', '*.xlsx', '*.xls',
            
            # Archivos de imágenes y multimedia
            '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.svg',
            '*.mp3', '*.mp4', '*.avi', '*.mov', '*.wmv',
            '*.pdf', '*.doc', '*.docx', '*.ppt', '*.pptx',
            
            # Archivos comprimidos
            '*.zip', '*.tar', '*.gz', '*.rar', '*.7z',
        }
        
        # Configurar mimetypes
        mimetypes.init()
    
    async def index_repository(
        self, 
        repository: Repository,
        ignore_patterns: Optional[Set[str]] = None,
        include_patterns: Optional[Set[str]] = None,
        max_file_size_mb: int = 10
    ) -> Result[List[FileIndex], Exception]:
        """
        Indexar todos los archivos de un repositorio.
        
        Args:
            repository: Repositorio a indexar
            ignore_patterns: Patrones de archivos a ignorar
            include_patterns: Patrones de archivos a incluir
            max_file_size_mb: Tamaño máximo de archivo en MB
            
        Returns:
            Result con la lista de índices de archivo creados
        """
        try:
            logger.info(f"Iniciando indexación del repositorio: {repository.local_path}")
            start_time = datetime.utcnow()
            
            # Combinar patrones de ignorado
            all_ignore_patterns = self.default_ignore_patterns.copy()
            if ignore_patterns:
                all_ignore_patterns.update(ignore_patterns)
            
            # Encontrar todos los archivos
            files = await self._find_files(
                repository.local_path,
                all_ignore_patterns,
                include_patterns,
                max_file_size_mb
            )
            
            # Indexar cada archivo
            file_indices = []
            for file_path in files:
                try:
                    file_index = await self._index_single_file(repository, file_path)
                    if file_index:
                        file_indices.append(file_index)
                except Exception as e:
                    logger.warning(f"Error al indexar archivo {file_path}: {str(e)}")
                    continue
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Indexación completada: {len(file_indices)} archivos en {processing_time:.2f}s")
            
            return Result.success(file_indices)
            
        except Exception as e:
            logger.error(f"Error en indexación del repositorio: {str(e)}")
            return Result.failure(IndexingError(f"Error en indexación: {str(e)}"))
    
    async def update_file_index(self, file_path: str, repository: Repository) -> Result[FileIndex, Exception]:
        """
        Actualizar el índice de un archivo específico.
        
        Args:
            file_path: Ruta del archivo a actualizar
            repository: Repositorio al que pertenece el archivo
            
        Returns:
            Result con el índice de archivo actualizado
        """
        try:
            full_path = os.path.join(repository.local_path, file_path)
            
            if not os.path.exists(full_path):
                return Result.failure(ValidationError(f"Archivo no encontrado: {file_path}"))
            
            file_index = await self._index_single_file(repository, file_path)
            if not file_index:
                return Result.failure(FileIndexingError(f"No se pudo indexar el archivo: {file_path}"))
            
            return Result.success(file_index)
            
        except Exception as e:
            logger.error(f"Error al actualizar índice de archivo {file_path}: {str(e)}")
            return Result.failure(FileIndexingError(f"Error al actualizar índice: {str(e)}"))
    
    async def detect_changes(self, repository: Repository) -> Result[List[str], Exception]:
        """
        Detectar archivos que han cambiado en el repositorio.
        
        Args:
            repository: Repositorio a analizar
            
        Returns:
            Result con la lista de archivos que han cambiado
        """
        try:
            # Por ahora, una implementación simple que escanea todos los archivos
            # En el futuro, esto podría usar Git para detectar cambios más eficientemente
            files = await self._find_files(
                repository.local_path,
                self.default_ignore_patterns,
                None,
                10
            )
            
            return Result.success(files)
            
        except Exception as e:
            logger.error(f"Error al detectar cambios: {str(e)}")
            return Result.failure(IndexingError(f"Error al detectar cambios: {str(e)}"))
    
    async def cleanup_deleted_files(
        self, 
        repository: Repository, 
        existing_files: List[FileIndex]
    ) -> Result[List[str], Exception]:
        """
        Identificar archivos que han sido eliminados del repositorio.
        
        Args:
            repository: Repositorio a analizar
            existing_files: Lista de archivos que existen en el índice
            
        Returns:
            Result con la lista de archivos eliminados
        """
        try:
            deleted_files = []
            
            for file_index in existing_files:
                full_path = os.path.join(repository.local_path, file_index.relative_path)
                if not os.path.exists(full_path):
                    deleted_files.append(file_index.relative_path)
            
            return Result.success(deleted_files)
            
        except Exception as e:
            logger.error(f"Error al limpiar archivos eliminados: {str(e)}")
            return Result.failure(IndexingError(f"Error al limpiar archivos: {str(e)}"))
    
    async def _find_files(
        self,
        base_path: str,
        ignore_patterns: Set[str],
        include_patterns: Optional[Set[str]],
        max_file_size_mb: int
    ) -> List[str]:
        """Encontrar todos los archivos en un directorio."""
        files = []
        max_size_bytes = max_file_size_mb * 1024 * 1024
        
        for root, dirs, filenames in os.walk(base_path):
            # Filtrar directorios
            dirs[:] = [d for d in dirs if not self._should_ignore(d, ignore_patterns)]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, base_path)
                
                # Verificar si el archivo debe ser ignorado
                if self._should_ignore(relative_path, ignore_patterns):
                    continue
                
                # Verificar si el archivo debe ser incluido
                if include_patterns and not self._should_include(relative_path, include_patterns):
                    continue
                
                # Verificar tamaño del archivo
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > max_size_bytes:
                        logger.debug(f"Archivo demasiado grande ignorado: {relative_path} ({file_size} bytes)")
                        continue
                except OSError:
                    logger.warning(f"No se pudo obtener tamaño del archivo: {relative_path}")
                    continue
                
                files.append(relative_path)
        
        return files
    
    async def _index_single_file(self, repository: Repository, relative_path: str) -> Optional[FileIndex]:
        """Indexar un archivo individual."""
        try:
            full_path = os.path.join(repository.local_path, relative_path)
            
            # Obtener información básica del archivo
            stat = os.stat(full_path)
            file_size = stat.st_size
            last_modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Calcular hash del archivo
            file_hash = await self._calculate_file_hash(full_path)
            
            # Contar líneas
            line_count = await self._count_lines(full_path)
            
            # Detectar lenguaje de programación
            language = self._detect_language(relative_path)
            
            # Obtener metadatos del archivo
            metadata = await self._extract_file_metadata(full_path)
            
            # Crear el índice de archivo
            file_index = FileIndex.create(
                repository_id=repository.id,
                relative_path=relative_path,
                size_bytes=file_size,
                line_count=line_count,
                file_hash=file_hash,
                last_modified=last_modified,
                language=language
            )
            
            # Asignar metadatos
            file_index.metadata = metadata
            
            return file_index
            
        except Exception as e:
            logger.warning(f"Error al indexar archivo {relative_path}: {str(e)}")
            return None
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calcular hash SHA-256 de un archivo."""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        except Exception as e:
            logger.warning(f"Error al calcular hash de {file_path}: {str(e)}")
            return ""
        
        return hash_sha256.hexdigest()
    
    async def _count_lines(self, file_path: str) -> int:
        """Contar líneas en un archivo."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.warning(f"Error al contar líneas de {file_path}: {str(e)}")
            return 0
    
    def _detect_language(self, file_path: str) -> Optional[ProgrammingLanguage]:
        """Detectar el lenguaje de programación de un archivo."""
        try:
            # Obtener la extensión del archivo
            extension = Path(file_path).suffix.lower()
            
            # Detectar lenguaje basado en la extensión
            return ProgrammingLanguage.from_extension(extension)
            
        except Exception as e:
            logger.warning(f"Error al detectar lenguaje de {file_path}: {str(e)}")
            return None
    
    async def _extract_file_metadata(self, file_path: str) -> FileMetadata:
        """Extraer metadatos de un archivo."""
        try:
            metadata = FileMetadata()
            
            # Detectar tipo MIME
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                metadata.custom_attributes['mime_type'] = mime_type
            
            # Detectar si es binario
            metadata.is_binary = self._is_binary_file(file_path)
            
            # Detectar encoding
            metadata.encoding = await self._detect_encoding(file_path)
            
            # Detectar line endings
            metadata.line_ending = await self._detect_line_endings(file_path)
            
            # Detectar BOM
            metadata.has_bom = await self._detect_bom(file_path)
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error al extraer metadatos de {file_path}: {str(e)}")
            return FileMetadata()
    
    def _is_binary_file(self, file_path: str) -> bool:
        """Detectar si un archivo es binario."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\x00' in chunk
        except Exception:
            return True
    
    async def _detect_encoding(self, file_path: str) -> str:
        """Detectar el encoding de un archivo."""
        # Por simplicidad, asumimos UTF-8
        # En una implementación real, usaríamos chardet o similar
        return "utf-8"
    
    async def _detect_line_endings(self, file_path: str) -> str:
        """Detectar los line endings de un archivo."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read(1024)
                if b'\r\n' in content:
                    return '\r\n'
                elif b'\r' in content:
                    return '\r'
                else:
                    return '\n'
        except Exception:
            return '\n'
    
    async def _detect_bom(self, file_path: str) -> bool:
        """Detectar si un archivo tiene BOM."""
        try:
            with open(file_path, 'rb') as f:
                bom = f.read(3)
                return bom.startswith(b'\xef\xbb\xbf')  # UTF-8 BOM
        except Exception:
            return False
    
    def _should_ignore(self, path: str, ignore_patterns: Set[str]) -> bool:
        """Verificar si un archivo debe ser ignorado."""
        path_lower = path.lower()
        
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(path_lower, pattern.lower()):
                return True
        
        return False
    
    def _should_include(self, path: str, include_patterns: Set[str]) -> bool:
        """Verificar si un archivo debe ser incluido."""
        path_lower = path.lower()
        
        for pattern in include_patterns:
            if fnmatch.fnmatch(path_lower, pattern.lower()):
                return True
        
        return False
