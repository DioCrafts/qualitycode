-- Migración inicial: Esquema base del sistema CodeAnt
-- Fecha: 2024-01-01
-- Descripción: Creación de las tablas principales del sistema

-- Extensión para UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Extensión para full-text search
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Tabla: organizations
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    settings JSONB NOT NULL DEFAULT '{}',
    subscription_plan VARCHAR(50) NOT NULL DEFAULT 'free',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Tabla: users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(320) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    avatar_url TEXT,
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    preferences JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE
);

-- Tabla: organization_members
CREATE TABLE organization_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'member',
    permissions JSONB NOT NULL DEFAULT '{}',
    joined_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    invited_by UUID REFERENCES users(id),
    
    UNIQUE(organization_id, user_id)
);

-- Tabla: projects
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL,
    description TEXT,
    repository_url TEXT NOT NULL,
    repository_type VARCHAR(20) NOT NULL DEFAULT 'git',
    default_branch VARCHAR(100) NOT NULL DEFAULT 'main',
    visibility VARCHAR(20) NOT NULL DEFAULT 'private',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    settings JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_analyzed_at TIMESTAMP WITH TIME ZONE,
    
    UNIQUE(organization_id, slug)
);

-- Tabla: repositories
CREATE TABLE repositories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    local_path TEXT NOT NULL,
    remote_url TEXT NOT NULL,
    current_commit VARCHAR(40),
    default_branch VARCHAR(100) NOT NULL,
    size_bytes BIGINT NOT NULL DEFAULT 0,
    file_count INTEGER NOT NULL DEFAULT 0,
    last_sync_at TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    sync_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Tabla: repository_branches
CREATE TABLE repository_branches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    commit_hash VARCHAR(40) NOT NULL,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    is_protected BOOLEAN NOT NULL DEFAULT FALSE,
    last_commit_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE(repository_id, name)
);

-- Tabla: file_index
CREATE TABLE file_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    absolute_path TEXT NOT NULL,
    language VARCHAR(50),
    size_bytes BIGINT NOT NULL,
    line_count INTEGER NOT NULL DEFAULT 0,
    file_hash VARCHAR(64) NOT NULL,
    content_hash VARCHAR(64), -- Hash of normalized content
    last_modified TIMESTAMP WITH TIME ZONE NOT NULL,
    analysis_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE(repository_id, relative_path)
);

-- Tabla: analysis_jobs
CREATE TABLE analysis_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 5,
    config JSONB NOT NULL DEFAULT '{}',
    progress INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Tabla: analysis_results
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    file_id UUID REFERENCES file_index(id) ON DELETE CASCADE,
    rule_id VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    category VARCHAR(50) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    column_start INTEGER,
    column_end INTEGER,
    suggestion TEXT,
    confidence DECIMAL(3,2) NOT NULL DEFAULT 1.0,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Tabla: code_metrics
CREATE TABLE code_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL REFERENCES file_index(id) ON DELETE CASCADE,
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value DECIMAL(10,4) NOT NULL,
    threshold_min DECIMAL(10,4),
    threshold_max DECIMAL(10,4),
    is_violation BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE(file_id, job_id, metric_type, metric_name)
);

-- Tabla: rules
CREATE TABLE rules (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    languages TEXT[] NOT NULL,
    rule_type VARCHAR(50) NOT NULL, -- 'static', 'ai', 'custom'
    implementation JSONB NOT NULL,
    examples JSONB,
    documentation_url TEXT,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    is_builtin BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0'
);

-- Tabla: project_rule_configs
CREATE TABLE project_rule_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    rule_id VARCHAR(100) NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    severity_override VARCHAR(20),
    config_override JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE(project_id, rule_id)
);

-- Crear índices para optimizar consultas
CREATE INDEX idx_organizations_slug ON organizations(slug);
CREATE INDEX idx_organizations_deleted_at ON organizations(deleted_at) WHERE deleted_at IS NULL;

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = TRUE;

CREATE INDEX idx_org_members_org_id ON organization_members(organization_id);
CREATE INDEX idx_org_members_user_id ON organization_members(user_id);

CREATE INDEX idx_projects_org_id ON projects(organization_id);
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_last_analyzed ON projects(last_analyzed_at);
CREATE INDEX idx_projects_repo_url ON projects USING HASH(repository_url);

CREATE INDEX idx_repositories_project_id ON repositories(project_id);
CREATE INDEX idx_repositories_sync_status ON repositories(sync_status);
CREATE INDEX idx_repositories_last_sync ON repositories(last_sync_at);

CREATE INDEX idx_repo_branches_repo_id ON repository_branches(repository_id);
CREATE INDEX idx_repo_branches_default ON repository_branches(is_default) WHERE is_default = TRUE;

CREATE INDEX idx_file_index_repo_id ON file_index(repository_id);
CREATE INDEX idx_file_index_language ON file_index(language);
CREATE INDEX idx_file_index_analysis_status ON file_index(analysis_status);
CREATE INDEX idx_file_index_hash ON file_index(file_hash);
CREATE INDEX idx_file_index_path_gin ON file_index USING GIN(to_tsvector('english', relative_path));

CREATE INDEX idx_analysis_jobs_project_id ON analysis_jobs(project_id);
CREATE INDEX idx_analysis_jobs_status ON analysis_jobs(status);
CREATE INDEX idx_analysis_jobs_type ON analysis_jobs(job_type);
CREATE INDEX idx_analysis_jobs_created ON analysis_jobs(created_at);

CREATE INDEX idx_analysis_results_job_id ON analysis_results(job_id);
CREATE INDEX idx_analysis_results_file_id ON analysis_results(file_id);
CREATE INDEX idx_analysis_results_severity ON analysis_results(severity);
CREATE INDEX idx_analysis_results_category ON analysis_results(category);
CREATE INDEX idx_analysis_results_rule_id ON analysis_results(rule_id);

CREATE INDEX idx_code_metrics_file_id ON code_metrics(file_id);
CREATE INDEX idx_code_metrics_job_id ON code_metrics(job_id);
CREATE INDEX idx_code_metrics_type ON code_metrics(metric_type);
CREATE INDEX idx_code_metrics_violations ON code_metrics(is_violation) WHERE is_violation = TRUE;

CREATE INDEX idx_rules_category ON rules(category);
CREATE INDEX idx_rules_severity ON rules(severity);
CREATE INDEX idx_rules_languages ON rules USING GIN(languages);
CREATE INDEX idx_rules_enabled ON rules(is_enabled) WHERE is_enabled = TRUE;
CREATE INDEX idx_rules_type ON rules(rule_type);

CREATE INDEX idx_project_rules_project_id ON project_rule_configs(project_id);
CREATE INDEX idx_project_rules_enabled ON project_rule_configs(is_enabled) WHERE is_enabled = TRUE;

-- Crear triggers para updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Aplicar triggers a las tablas que necesitan updated_at
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_repositories_updated_at BEFORE UPDATE ON repositories FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_file_index_updated_at BEFORE UPDATE ON file_index FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_analysis_jobs_updated_at BEFORE UPDATE ON analysis_jobs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_rules_updated_at BEFORE UPDATE ON rules FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_project_rule_configs_updated_at BEFORE UPDATE ON project_rule_configs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
