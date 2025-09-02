"""
Enums para tipos de repositorios y estados.
"""
from enum import Enum


class RepositoryType(Enum):
    """Tipos de sistemas de control de versiones."""
    GIT = "git"
    MERCURIAL = "mercurial"
    SUBVERSION = "subversion"
    PERFORCE = "perforce"
    UNKNOWN = "unknown"


class ProjectStatus(Enum):
    """Estados de un proyecto."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"


class SyncStatus(Enum):
    """Estados de sincronización de repositorio."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SYNCED = "synced"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisStatus(Enum):
    """Estados de análisis de archivo."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WebhookProvider(Enum):
    """Proveedores de webhooks."""
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    AZURE_DEVOPS = "azure_devops"
    GENERIC = "generic"


class WebhookEventType(Enum):
    """Tipos de eventos de webhook."""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    MERGE_REQUEST = "merge_request"
    ISSUE = "issue"
    COMMIT = "commit"
    TAG = "tag"
    BRANCH = "branch"
    DELETE = "delete"
