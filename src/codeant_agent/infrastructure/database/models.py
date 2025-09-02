"""
Modelos SQLAlchemy para la base de datos.
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    Float, JSON, Enum as SQLEnum, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY

from codeant_agent.domain.value_objects import (
    RepositoryType, ProjectStatus, SyncStatus, 
    AnalysisStatus, WebhookProvider, WebhookEventType
)

Base = declarative_base()


class Organization(Base):
    """Modelo para organizaciones."""
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()")
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    website_url = Column(String(500))
    logo_url = Column(String(500))
    settings = Column(JSON, default={})
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    
    # Relaciones
    members = relationship("OrganizationMember", back_populates="organization", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="organization", cascade="all, delete-orphan")
    
    # Índices
    __table_args__ = (
        Index("idx_organizations_slug", "slug"),
        Index("idx_organizations_active", "is_active"),
    )


class User(Base):
    """Modelo para usuarios."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()")
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    avatar_url = Column(String(500))
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    roles = Column(ARRAY(String), default=[], nullable=False)
    permissions = Column(ARRAY(String), default=[], nullable=False)
    preferences = Column(JSON, default={})
    last_login_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    
    # Relaciones
    organization_memberships = relationship("OrganizationMember", back_populates="user", cascade="all, delete-orphan")
    
    # Índices
    __table_args__ = (
        Index("idx_users_email", "email"),
        Index("idx_users_username", "username"),
        Index("idx_users_active", "is_active"),
        Index("idx_users_verified", "is_verified"),
        Index("idx_users_roles", "roles"),
    )


class OrganizationMember(Base):
    """Modelo para miembros de organizaciones."""
    __tablename__ = "organization_members"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()")
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(String(50), nullable=False, default="member")  # owner, admin, member
    permissions = Column(JSON, default={})
    joined_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    
    # Relaciones
    organization = relationship("Organization", back_populates="members")
    user = relationship("User", back_populates="organization_memberships")
    
    # Índices y constraints
    __table_args__ = (
        UniqueConstraint("organization_id", "user_id", name="uq_org_member"),
        Index("idx_org_members_org_id", "organization_id"),
        Index("idx_org_members_user_id", "user_id"),
        Index("idx_org_members_role", "role"),
    )


class Project(Base):
    """Modelo para proyectos."""
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()")
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    slug = Column(String(100), nullable=False)
    status = Column(SQLEnum(ProjectStatus), nullable=False, default=ProjectStatus.ACTIVE)
    settings = Column(JSON, default={})
    metadata_json = Column(JSON, default={})
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    
    # Relaciones
    organization = relationship("Organization", back_populates="projects")
    repositories = relationship("Repository", back_populates="project", cascade="all, delete-orphan")
    analysis_jobs = relationship("AnalysisJob", back_populates="project", cascade="all, delete-orphan")
    rule_configs = relationship("ProjectRuleConfig", back_populates="project", cascade="all, delete-orphan")
    
    # Índices y constraints
    __table_args__ = (
        UniqueConstraint("organization_id", "slug", name="uq_project_org_slug"),
        Index("idx_projects_org_id", "organization_id"),
        Index("idx_projects_slug", "slug"),
        Index("idx_projects_status", "status"),
        Index("idx_projects_created_by", "created_by"),
    )


class Repository(Base):
    """Modelo para repositorios."""
    __tablename__ = "repositories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()")
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    url = Column(String(500), nullable=False)
    type = Column(SQLEnum(RepositoryType), nullable=False, default=RepositoryType.GIT)
    default_branch = Column(String(100), nullable=False, default="main")
    sync_status = Column(SQLEnum(SyncStatus), nullable=False, default=SyncStatus.PENDING)
    last_sync_at = Column(DateTime(timezone=True))
    last_commit_hash = Column(String(40))
    last_commit_message = Column(Text)
    last_commit_author = Column(String(255))
    last_commit_date = Column(DateTime(timezone=True))
    size_bytes = Column(BigInteger, default=0)
    file_count = Column(Integer, default=0)
    language_stats = Column(JSON, default={})
    webhook_url = Column(String(500))
    webhook_secret = Column(String(255))
    settings = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    
    # Relaciones
    project = relationship("Project", back_populates="repositories")
    branches = relationship("RepositoryBranch", back_populates="repository", cascade="all, delete-orphan")
    file_index = relationship("FileIndex", back_populates="repository", cascade="all, delete-orphan")
    
    # Índices y constraints
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_repo_project_name"),
        Index("idx_repositories_project_id", "project_id"),
        Index("idx_repositories_url", "url"),
        Index("idx_repositories_sync_status", "sync_status"),
        Index("idx_repositories_last_sync", "last_sync_at"),
    )


class RepositoryBranch(Base):
    """Modelo para ramas de repositorio."""
    __tablename__ = "repository_branches"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()")
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id"), nullable=False)
    name = Column(String(255), nullable=False)
    commit_hash = Column(String(40), nullable=False)
    commit_message = Column(Text)
    commit_author = Column(String(255))
    commit_date = Column(DateTime(timezone=True))
    is_default = Column(Boolean, default=False, nullable=False)
    is_protected = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    
    # Relaciones
    repository = relationship("Repository", back_populates="branches")
    
    # Índices y constraints
    __table_args__ = (
        UniqueConstraint("repository_id", "name", name="uq_branch_repo_name"),
        Index("idx_branches_repo_id", "repository_id"),
        Index("idx_branches_name", "name"),
        Index("idx_branches_default", "is_default"),
    )


class FileIndex(Base):
    """Modelo para índice de archivos."""
    __tablename__ = "file_index"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()")
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id"), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_extension = Column(String(50))
    language = Column(String(100))
    mime_type = Column(String(255))
    size_bytes = Column(BigInteger, nullable=False)
    line_count = Column(Integer, default=0)
    commit_hash = Column(String(40), nullable=False)
    branch_name = Column(String(255), nullable=False)
    is_binary = Column(Boolean, default=False, nullable=False)
    is_ignored = Column(Boolean, default=False, nullable=False)
    metadata_json = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    
    # Relaciones
    repository = relationship("Repository", back_populates="file_index")
    
    # Índices y constraints
    __table_args__ = (
        UniqueConstraint("repository_id", "file_path", "commit_hash", name="uq_file_repo_path_commit"),
        Index("idx_file_index_repo_id", "repository_id"),
        Index("idx_file_index_path", "file_path"),
        Index("idx_file_index_language", "language"),
        Index("idx_file_index_extension", "file_extension"),
        Index("idx_file_index_commit", "commit_hash"),
        Index("idx_file_index_branch", "branch_name"),
        Index("idx_file_index_ignored", "is_ignored"),
    )


class AnalysisJob(Base):
    """Modelo para trabajos de análisis."""
    __tablename__ = "analysis_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()")
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id"), nullable=False)
    commit_hash = Column(String(40), nullable=False)
    branch_name = Column(String(255), nullable=False)
    status = Column(SQLEnum(AnalysisStatus), nullable=False, default=AnalysisStatus.PENDING)
    analysis_type = Column(String(100), nullable=False)  # complexity, quality, security, etc.
    priority = Column(Integer, default=5, nullable=False)  # 1-10, 1 más alta
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    progress_percentage = Column(Float, default=0.0)
    metadata_json = Column(JSON, default={})
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    
    # Relaciones
    project = relationship("Project", back_populates="analysis_jobs")
    results = relationship("AnalysisResult", back_populates="job", cascade="all, delete-orphan")
    
    # Índices y constraints
    __table_args__ = (
        Index("idx_analysis_jobs_project_id", "project_id"),
        Index("idx_analysis_jobs_repo_id", "repository_id"),
        Index("idx_analysis_jobs_status", "status"),
        Index("idx_analysis_jobs_commit", "commit_hash"),
        Index("idx_analysis_jobs_type", "analysis_type"),
        Index("idx_analysis_jobs_priority", "priority"),
        Index("idx_analysis_jobs_created_by", "created_by"),
    )


class AnalysisResult(Base):
    """Modelo para resultados de análisis."""
    __tablename__ = "analysis_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()")
    job_id = Column(UUID(as_uuid=True), ForeignKey("analysis_jobs.id"), nullable=False)
    file_path = Column(String(1000), nullable=False)
    rule_id = Column(UUID(as_uuid=True), ForeignKey("rules.id"))
    severity = Column(String(20), nullable=False)  # info, warning, error, critical
    message = Column(Text, nullable=False)
    line_number = Column(Integer)
    column_number = Column(Integer)
    end_line_number = Column(Integer)
    end_column_number = Column(Integer)
    code_snippet = Column(Text)
    suggestions = Column(JSON, default=[])
    metadata_json = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    
    # Relaciones
    job = relationship("AnalysisJob", back_populates="results")
    rule = relationship("Rule")
    
    # Índices y constraints
    __table_args__ = (
        Index("idx_analysis_results_job_id", "job_id"),
        Index("idx_analysis_results_file_path", "file_path"),
        Index("idx_analysis_results_severity", "severity"),
        Index("idx_analysis_results_rule_id", "rule_id"),
        Index("idx_analysis_results_line", "line_number"),
    )


class CodeMetrics(Base):
    """Modelo para métricas de código."""
    __tablename__ = "code_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()")
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id"), nullable=False)
    commit_hash = Column(String(40), nullable=False)
    file_path = Column(String(1000), nullable=False)
    language = Column(String(100))
    cyclomatic_complexity = Column(Integer, default=0)
    cognitive_complexity = Column(Integer, default=0)
    lines_of_code = Column(Integer, default=0)
    lines_of_comments = Column(Integer, default=0)
    lines_of_blank = Column(Integer, default=0)
    function_count = Column(Integer, default=0)
    class_count = Column(Integer, default=0)
    average_function_length = Column(Float, default=0.0)
    average_class_length = Column(Float, default=0.0)
    maintainability_index = Column(Float, default=0.0)
    technical_debt_ratio = Column(Float, default=0.0)
    code_smells = Column(Integer, default=0)
    bugs = Column(Integer, default=0)
    vulnerabilities = Column(Integer, default=0)
    security_hotspots = Column(Integer, default=0)
    coverage_percentage = Column(Float, default=0.0)
    duplication_percentage = Column(Float, default=0.0)
    metadata_json = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    
    # Índices y constraints
    __table_args__ = (
        UniqueConstraint("repository_id", "commit_hash", "file_path", name="uq_metrics_repo_commit_file"),
        Index("idx_code_metrics_repo_id", "repository_id"),
        Index("idx_code_metrics_commit", "commit_hash"),
        Index("idx_code_metrics_file_path", "file_path"),
        Index("idx_code_metrics_language", "language"),
        Index("idx_code_metrics_complexity", "cyclomatic_complexity"),
    )


class Rule(Base):
    """Modelo para reglas de análisis."""
    __tablename__ = "rules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()")
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100), nullable=False)  # complexity, quality, security, style
    severity = Column(String(20), nullable=False)  # info, warning, error, critical
    language = Column(String(100))  # null para reglas multi-lenguaje
    rule_type = Column(String(50), nullable=False)  # static, dynamic, custom
    is_active = Column(Boolean, default=True, nullable=False)
    configuration = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    
    # Relaciones
    project_configs = relationship("ProjectRuleConfig", back_populates="rule", cascade="all, delete-orphan")
    
    # Índices y constraints
    __table_args__ = (
        Index("idx_rules_category", "category"),
        Index("idx_rules_severity", "severity"),
        Index("idx_rules_language", "language"),
        Index("idx_rules_active", "is_active"),
    )


class ProjectRuleConfig(Base):
    """Modelo para configuración de reglas por proyecto."""
    __tablename__ = "project_rule_configs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()")
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    rule_id = Column(UUID(as_uuid=True), ForeignKey("rules.id"), nullable=False)
    is_enabled = Column(Boolean, default=True, nullable=False)
    severity_override = Column(String(20))  # null para usar severidad por defecto
    configuration = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default="NOW()")
    
    # Relaciones
    project = relationship("Project", back_populates="rule_configs")
    rule = relationship("Rule", back_populates="project_configs")
    
    # Índices y constraints
    __table_args__ = (
        UniqueConstraint("project_id", "rule_id", name="uq_project_rule"),
        Index("idx_project_rule_configs_project_id", "project_id"),
        Index("idx_project_rule_configs_rule_id", "rule_id"),
        Index("idx_project_rule_configs_enabled", "is_enabled"),
    )
