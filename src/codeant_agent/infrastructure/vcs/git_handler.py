"""
Handler de Git para operaciones de control de versiones.
"""
import asyncio
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import git
from git import Repo, GitCommandError
from git.objects.commit import Commit as GitCommit
from git.objects.blob import Blob
from git.refs.remote import RemoteReference

from codeant_agent.domain.entities.repository import Repository, Branch, Tag
from codeant_agent.domain.value_objects.commit_hash import CommitHash
from codeant_agent.domain.value_objects.repository_type import RepositoryType
from codeant_agent.utils.error import Result, BaseError, ValidationError
from codeant_agent.utils.logging import get_logger

logger = get_logger(__name__)


class GitOperationError(BaseError):
    """Error en operaciones de Git."""
    pass


class GitCloneError(GitOperationError):
    """Error al clonar un repositorio."""
    pass


class GitFetchError(GitOperationError):
    """Error al hacer fetch de un repositorio."""
    pass


@dataclass
class CommitInfo:
    """Información de un commit."""
    hash: CommitHash
    author: str
    author_email: str
    message: str
    date: datetime
    parents: List[CommitHash]


@dataclass
class FileChange:
    """Información de un cambio en un archivo."""
    file_path: str
    change_type: str  # 'A' (added), 'M' (modified), 'D' (deleted), 'R' (renamed)
    old_path: Optional[str] = None
    additions: int = 0
    deletions: int = 0


@dataclass
class Diff:
    """Información de diferencias entre commits."""
    from_commit: CommitHash
    to_commit: CommitHash
    changes: List[FileChange]
    total_additions: int
    total_deletions: int


@dataclass
class BlameInfo:
    """Información de blame de un archivo."""
    file_path: str
    lines: List[Dict[str, Any]]  # Cada línea con commit, autor, fecha, etc.


class GitHandler:
    """
    Handler para operaciones de Git.
    
    Proporciona una interfaz de alto nivel para operaciones comunes
    de Git como clonar, fetch, obtener historial, etc.
    """
    
    def __init__(self, base_path: str = "/tmp/repositories"):
        """
        Inicializar el handler de Git.
        
        Args:
            base_path: Ruta base donde se clonarán los repositorios
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar Git para evitar prompts interactivos
        os.environ['GIT_TERMINAL_PROGRESS'] = '0'
    
    async def clone_repository(
        self, 
        url: str, 
        target_path: str,
        credentials: Optional[Dict[str, str]] = None
    ) -> Result[Repository, Exception]:
        """
        Clonar un repositorio Git.
        
        Args:
            url: URL del repositorio a clonar
            target_path: Ruta donde clonar el repositorio
            credentials: Credenciales opcionales (username, password/token)
            
        Returns:
            Result con el repositorio clonado o error
        """
        try:
            logger.info(f"Clonando repositorio: {url} -> {target_path}")
            
            # Preparar la URL con credenciales si se proporcionan
            clone_url = self._prepare_url_with_credentials(url, credentials)
            
            # Clonar el repositorio
            repo = await self._clone_repo_async(clone_url, target_path)
            
            # Obtener información del repositorio
            current_commit = CommitHash(repo.head.commit.hexsha)
            
            # Crear la entidad Repository
            repository = Repository.create(
                project_id=None,  # Se asignará después
                local_path=target_path,
                remote_url=url,
                current_commit=current_commit
            )
            
            logger.info(f"Repositorio clonado exitosamente: {target_path}")
            return Result.success(repository)
            
        except Exception as e:
            logger.error(f"Error al clonar repositorio {url}: {str(e)}")
            return Result.failure(GitCloneError(f"Error al clonar repositorio: {str(e)}"))
    
    async def fetch_updates(self, repo: Repository) -> Result[List[CommitInfo], Exception]:
        """
        Obtener actualizaciones de un repositorio.
        
        Args:
            repo: Repositorio del cual obtener actualizaciones
            
        Returns:
            Result con la lista de commits nuevos o error
        """
        try:
            logger.info(f"Obteniendo actualizaciones para: {repo.local_path}")
            
            git_repo = Repo(repo.local_path)
            
            # Obtener el commit actual antes del fetch
            old_head = git_repo.head.commit.hexsha
            
            # Hacer fetch de las actualizaciones
            await self._fetch_repo_async(git_repo)
            
            # Obtener commits nuevos
            new_commits = []
            for commit in git_repo.iter_commits(f"{old_head}..HEAD"):
                commit_info = CommitInfo(
                    hash=CommitHash(commit.hexsha),
                    author=commit.author.name,
                    author_email=commit.author.email,
                    message=commit.message.strip(),
                    date=datetime.fromtimestamp(commit.committed_date),
                    parents=[CommitHash(parent.hexsha) for parent in commit.parents]
                )
                new_commits.append(commit_info)
            
            logger.info(f"Obtenidos {len(new_commits)} commits nuevos")
            return Result.success(new_commits)
            
        except Exception as e:
            logger.error(f"Error al obtener actualizaciones: {str(e)}")
            return Result.failure(GitFetchError(f"Error al obtener actualizaciones: {str(e)}"))
    
    async def get_file_history(
        self, 
        repo: Repository, 
        file_path: str,
        max_commits: int = 100
    ) -> Result[List[CommitInfo], Exception]:
        """
        Obtener el historial de cambios de un archivo.
        
        Args:
            repo: Repositorio
            file_path: Ruta del archivo
            max_commits: Número máximo de commits a obtener
            
        Returns:
            Result con la lista de commits que modificaron el archivo
        """
        try:
            git_repo = Repo(repo.local_path)
            
            # Verificar que el archivo existe
            if not os.path.exists(os.path.join(repo.local_path, file_path)):
                return Result.failure(ValidationError(f"Archivo no encontrado: {file_path}"))
            
            commits = []
            for commit in git_repo.iter_commits(paths=file_path, max_count=max_commits):
                commit_info = CommitInfo(
                    hash=CommitHash(commit.hexsha),
                    author=commit.author.name,
                    author_email=commit.author.email,
                    message=commit.message.strip(),
                    date=datetime.fromtimestamp(commit.committed_date),
                    parents=[CommitHash(parent.hexsha) for parent in commit.parents]
                )
                commits.append(commit_info)
            
            return Result.success(commits)
            
        except Exception as e:
            logger.error(f"Error al obtener historial de archivo {file_path}: {str(e)}")
            return Result.failure(GitOperationError(f"Error al obtener historial: {str(e)}"))
    
    async def get_diff(
        self, 
        repo: Repository, 
        from_commit: CommitHash, 
        to_commit: CommitHash
    ) -> Result[Diff, Exception]:
        """
        Obtener las diferencias entre dos commits.
        
        Args:
            repo: Repositorio
            from_commit: Commit de origen
            to_commit: Commit de destino
            
        Returns:
            Result con las diferencias entre los commits
        """
        try:
            git_repo = Repo(repo.local_path)
            
            # Obtener los commits
            from_git_commit = git_repo.commit(from_commit.value)
            to_git_commit = git_repo.commit(to_commit.value)
            
            # Obtener el diff
            diff = from_git_commit.diff(to_git_commit)
            
            changes = []
            total_additions = 0
            total_deletions = 0
            
            for change in diff:
                file_change = FileChange(
                    file_path=change.b_path or change.a_path,
                    change_type=change.change_type,
                    old_path=change.a_path if change.change_type == 'R' else None,
                    additions=change.stats.get('insertions', 0),
                    deletions=change.stats.get('deletions', 0)
                )
                changes.append(file_change)
                total_additions += file_change.additions
                total_deletions += file_change.deletions
            
            diff_info = Diff(
                from_commit=from_commit,
                to_commit=to_commit,
                changes=changes,
                total_additions=total_additions,
                total_deletions=total_deletions
            )
            
            return Result.success(diff_info)
            
        except Exception as e:
            logger.error(f"Error al obtener diff: {str(e)}")
            return Result.failure(GitOperationError(f"Error al obtener diff: {str(e)}"))
    
    async def get_blame(self, repo: Repository, file_path: str) -> Result[BlameInfo, Exception]:
        """
        Obtener información de blame de un archivo.
        
        Args:
            repo: Repositorio
            file_path: Ruta del archivo
            
        Returns:
            Result con la información de blame
        """
        try:
            git_repo = Repo(repo.local_path)
            
            # Verificar que el archivo existe
            full_path = os.path.join(repo.local_path, file_path)
            if not os.path.exists(full_path):
                return Result.failure(ValidationError(f"Archivo no encontrado: {file_path}"))
            
            # Obtener blame
            blame = git_repo.blame('HEAD', file_path)
            
            lines = []
            for commit, lines_in_commit in blame:
                for line in lines_in_commit:
                    line_info = {
                        'line_number': line.lineno,
                        'commit_hash': CommitHash(commit.hexsha),
                        'author': commit.author.name,
                        'author_email': commit.author.email,
                        'date': datetime.fromtimestamp(commit.committed_date),
                        'message': commit.message.strip(),
                        'content': line.content
                    }
                    lines.append(line_info)
            
            blame_info = BlameInfo(
                file_path=file_path,
                lines=lines
            )
            
            return Result.success(blame_info)
            
        except Exception as e:
            logger.error(f"Error al obtener blame de {file_path}: {str(e)}")
            return Result.failure(GitOperationError(f"Error al obtener blame: {str(e)}"))
    
    async def list_branches(self, repo: Repository) -> Result[List[Branch], Exception]:
        """
        Listar todas las ramas del repositorio.
        
        Args:
            repo: Repositorio
            
        Returns:
            Result con la lista de ramas
        """
        try:
            git_repo = Repo(repo.local_path)
            
            branches = []
            for ref in git_repo.references:
                if isinstance(ref, git.refs.Head):
                    branch = Branch(
                        name=ref.name,
                        commit_hash=CommitHash(ref.commit.hexsha),
                        is_default=ref.name == git_repo.active_branch.name,
                        is_protected=ref.name in ['main', 'master'],
                        last_commit_date=datetime.fromtimestamp(ref.commit.committed_date)
                    )
                    branches.append(branch)
            
            return Result.success(branches)
            
        except Exception as e:
            logger.error(f"Error al listar ramas: {str(e)}")
            return Result.failure(GitOperationError(f"Error al listar ramas: {str(e)}"))
    
    async def checkout_commit(self, repo: Repository, commit: CommitHash) -> Result[None, Exception]:
        """
        Hacer checkout a un commit específico.
        
        Args:
            repo: Repositorio
            commit: Hash del commit al cual hacer checkout
            
        Returns:
            Result con None si es exitoso o error
        """
        try:
            git_repo = Repo(repo.local_path)
            git_repo.git.checkout(commit.value)
            return Result.success(None)
            
        except Exception as e:
            logger.error(f"Error al hacer checkout a {commit.value}: {str(e)}")
            return Result.failure(GitOperationError(f"Error al hacer checkout: {str(e)}"))
    
    def _prepare_url_with_credentials(self, url: str, credentials: Optional[Dict[str, str]]) -> str:
        """Preparar URL con credenciales si se proporcionan."""
        if not credentials:
            return url
        
        username = credentials.get('username')
        password = credentials.get('password') or credentials.get('token')
        
        if username and password:
            # Insertar credenciales en la URL
            if url.startswith('https://'):
                return f"https://{username}:{password}@{url[8:]}"
            elif url.startswith('http://'):
                return f"http://{username}:{password}@{url[7:]}"
        
        return url
    
    async def _clone_repo_async(self, url: str, target_path: str) -> Repo:
        """Clonar repositorio de forma asíncrona."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, Repo.clone_from, url, target_path)
    
    async def _fetch_repo_async(self, repo: Repo) -> None:
        """Hacer fetch de un repositorio de forma asíncrona."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, repo.remotes.origin.fetch)
    
    def cleanup_repository(self, local_path: str) -> Result[bool, Exception]:
        """
        Limpiar un repositorio local.
        
        Args:
            local_path: Ruta del repositorio a limpiar
            
        Returns:
            Result con True si se limpió correctamente
        """
        try:
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
                logger.info(f"Repositorio limpiado: {local_path}")
            return Result.success(True)
            
        except Exception as e:
            logger.error(f"Error al limpiar repositorio {local_path}: {str(e)}")
            return Result.failure(GitOperationError(f"Error al limpiar repositorio: {str(e)}"))
