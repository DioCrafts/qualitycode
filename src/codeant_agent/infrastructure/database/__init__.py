"""
Módulo de base de datos de CodeAnt Agent.
"""

from .migration_manager import (
    MigrationManager,
    Migration,
    MigrationResult,
    MigrationError,
    MigrationNotFoundError,
    MigrationExecutionError
)

from .connection_pool import (
    DatabaseConnectionPool,
    DatabaseConfig,
    DatabaseConnectionError,
    ConnectionPoolError,
    get_connection_pool,
    close_connection_pool
)

from .models import (
    Base,
    Organization,
    User,
    OrganizationMember,
    Project,
    Repository,
    RepositoryBranch,
    FileIndex,
    AnalysisJob,
    AnalysisResult,
    CodeMetrics,
    Rule,
    ProjectRuleConfig
)

__all__ = [
    # Migration Manager
    "MigrationManager",
    "Migration",
    "MigrationResult",
    "MigrationError",
    "MigrationNotFoundError",
    "MigrationExecutionError",
    
    # Connection Pool
    "DatabaseConnectionPool",
    "DatabaseConfig",
    "DatabaseConnectionError",
    "ConnectionPoolError",
    "get_connection_pool",
    "close_connection_pool",
    
    # Models
    "Base",
    "Organization",
    "User",
    "OrganizationMember",
    "Project",
    "Repository",
    "RepositoryBranch",
    "FileIndex",
    "AnalysisJob",
    "AnalysisResult",
    "CodeMetrics",
    "Rule",
    "ProjectRuleConfig"
]
