# Fase 3: Base de Datos y Modelos de Datos Fundamentales

## Objetivo General
Diseñar e implementar un sistema de persistencia robusto y escalable que soporte todas las necesidades de almacenamiento del agente CodeAnt, incluyendo metadatos de proyectos, resultados de análisis, métricas históricas, y datos de machine learning.

## Descripción Técnica Detallada

### 3.1 Arquitectura de Datos

#### 3.1.1 Estrategia Multi-Store
```
┌─────────────────────────────────────────┐
│           Data Architecture             │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ PostgreSQL  │ │      Redis          │ │
│  │ (Primary)   │ │    (Cache/Queue)    │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Qdrant    │ │    ClickHouse       │ │
│  │ (Vectors)   │ │   (Analytics)       │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│         Repository Pattern Layer        │
└─────────────────────────────────────────┘
```

#### 3.1.2 Justificación de Tecnologías
- **PostgreSQL**: ACID compliance, JSONB, full-text search, extensibilidad
- **Redis**: Cache L1, session storage, job queues, real-time data
- **Qdrant**: Vector similarity search para code embeddings
- **ClickHouse**: Analytics y métricas históricas (fase posterior)

### 3.2 Esquema de Base de Datos Principal (PostgreSQL)

#### 3.2.1 Tablas Core del Sistema

##### Tabla: organizations
```sql
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

CREATE INDEX idx_organizations_slug ON organizations(slug);
CREATE INDEX idx_organizations_deleted_at ON organizations(deleted_at) WHERE deleted_at IS NULL;
```

##### Tabla: users
```sql
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

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = TRUE;
```

##### Tabla: organization_members
```sql
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

CREATE INDEX idx_org_members_org_id ON organization_members(organization_id);
CREATE INDEX idx_org_members_user_id ON organization_members(user_id);
```

##### Tabla: projects
```sql
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

CREATE INDEX idx_projects_org_id ON projects(organization_id);
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_last_analyzed ON projects(last_analyzed_at);
CREATE INDEX idx_projects_repo_url ON projects USING HASH(repository_url);
```

#### 3.2.2 Tablas de Repositorio y Archivos

##### Tabla: repositories
```sql
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

CREATE INDEX idx_repositories_project_id ON repositories(project_id);
CREATE INDEX idx_repositories_sync_status ON repositories(sync_status);
CREATE INDEX idx_repositories_last_sync ON repositories(last_sync_at);
```

##### Tabla: repository_branches
```sql
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

CREATE INDEX idx_repo_branches_repo_id ON repository_branches(repository_id);
CREATE INDEX idx_repo_branches_default ON repository_branches(is_default) WHERE is_default = TRUE;
```

##### Tabla: file_index
```sql
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

CREATE INDEX idx_file_index_repo_id ON file_index(repository_id);
CREATE INDEX idx_file_index_language ON file_index(language);
CREATE INDEX idx_file_index_analysis_status ON file_index(analysis_status);
CREATE INDEX idx_file_index_hash ON file_index(file_hash);
CREATE INDEX idx_file_index_path_gin ON file_index USING GIN(to_tsvector('english', relative_path));
```

#### 3.2.3 Tablas de Análisis y Resultados

##### Tabla: analysis_jobs
```sql
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

CREATE INDEX idx_analysis_jobs_project_id ON analysis_jobs(project_id);
CREATE INDEX idx_analysis_jobs_status ON analysis_jobs(status);
CREATE INDEX idx_analysis_jobs_type ON analysis_jobs(job_type);
CREATE INDEX idx_analysis_jobs_created ON analysis_jobs(created_at);
```

##### Tabla: analysis_results
```sql
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

CREATE INDEX idx_analysis_results_job_id ON analysis_results(job_id);
CREATE INDEX idx_analysis_results_file_id ON analysis_results(file_id);
CREATE INDEX idx_analysis_results_severity ON analysis_results(severity);
CREATE INDEX idx_analysis_results_category ON analysis_results(category);
CREATE INDEX idx_analysis_results_rule_id ON analysis_results(rule_id);
```

##### Tabla: code_metrics
```sql
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

CREATE INDEX idx_code_metrics_file_id ON code_metrics(file_id);
CREATE INDEX idx_code_metrics_job_id ON code_metrics(job_id);
CREATE INDEX idx_code_metrics_type ON code_metrics(metric_type);
CREATE INDEX idx_code_metrics_violations ON code_metrics(is_violation) WHERE is_violation = TRUE;
```

#### 3.2.4 Tablas de Reglas y Configuración

##### Tabla: rules
```sql
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

CREATE INDEX idx_rules_category ON rules(category);
CREATE INDEX idx_rules_severity ON rules(severity);
CREATE INDEX idx_rules_languages ON rules USING GIN(languages);
CREATE INDEX idx_rules_enabled ON rules(is_enabled) WHERE is_enabled = TRUE;
CREATE INDEX idx_rules_type ON rules(rule_type);
```

##### Tabla: project_rule_configs
```sql
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

CREATE INDEX idx_project_rules_project_id ON project_rule_configs(project_id);
CREATE INDEX idx_project_rules_enabled ON project_rule_configs(is_enabled) WHERE is_enabled = TRUE;
```

### 3.3 Modelos de Datos en Rust

#### 3.3.1 Domain Entities

##### Organization Entity
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Organization {
    pub id: OrganizationId,
    pub name: String,
    pub slug: String,
    pub description: Option<String>,
    pub settings: OrganizationSettings,
    pub subscription_plan: SubscriptionPlan,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub deleted_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganizationSettings {
    pub default_analysis_config: AnalysisConfig,
    pub webhook_settings: WebhookSettings,
    pub notification_preferences: NotificationPreferences,
    pub security_settings: SecuritySettings,
}
```

##### Project Entity (Extended)
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub id: ProjectId,
    pub organization_id: OrganizationId,
    pub name: String,
    pub slug: String,
    pub description: Option<String>,
    pub repository_url: Url,
    pub repository_type: RepositoryType,
    pub default_branch: String,
    pub visibility: ProjectVisibility,
    pub status: ProjectStatus,
    pub settings: ProjectSettings,
    pub metadata: ProjectMetadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_analyzed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectMetadata {
    pub languages: HashMap<ProgrammingLanguage, LanguageStats>,
    pub total_files: u32,
    pub total_lines: u64,
    pub repository_size: u64,
    pub last_commit: Option<CommitInfo>,
    pub quality_score: Option<f64>,
    pub technical_debt_hours: Option<f64>,
}
```

##### File Index Entity (Extended)
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileIndex {
    pub id: FileId,
    pub repository_id: RepositoryId,
    pub relative_path: PathBuf,
    pub absolute_path: PathBuf,
    pub language: Option<ProgrammingLanguage>,
    pub size_bytes: u64,
    pub line_count: u32,
    pub file_hash: String,
    pub content_hash: Option<String>,
    pub last_modified: DateTime<Utc>,
    pub analysis_status: AnalysisStatus,
    pub metadata: FileMetadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileMetadata {
    pub encoding: String,
    pub has_bom: bool,
    pub line_endings: LineEndingType,
    pub complexity_metrics: Option<ComplexityMetrics>,
    pub imports: Vec<ImportDeclaration>,
    pub exports: Vec<ExportDeclaration>,
    pub functions: Vec<FunctionSignature>,
    pub classes: Vec<ClassSignature>,
}
```

#### 3.3.2 Value Objects

##### Analysis Result
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub id: AnalysisResultId,
    pub job_id: JobId,
    pub file_id: Option<FileId>,
    pub rule_id: String,
    pub severity: Severity,
    pub category: Category,
    pub title: String,
    pub description: String,
    pub location: Option<SourceLocation>,
    pub suggestion: Option<String>,
    pub confidence: f64,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub line_start: u32,
    pub line_end: u32,
    pub column_start: u32,
    pub column_end: u32,
    pub file_path: PathBuf,
}
```

##### Code Metrics
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeMetric {
    pub id: MetricId,
    pub file_id: FileId,
    pub job_id: JobId,
    pub metric_type: MetricType,
    pub metric_name: String,
    pub value: f64,
    pub threshold_min: Option<f64>,
    pub threshold_max: Option<f64>,
    pub is_violation: bool,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetricType {
    Complexity,
    Quality,
    Security,
    Performance,
    Maintainability,
    Coverage,
    Duplication,
}
```

### 3.4 Repository Pattern Implementation

#### 3.4.1 Repository Traits

##### Project Repository
```rust
#[async_trait]
pub trait ProjectRepository: Send + Sync {
    async fn create(&self, project: CreateProjectRequest) -> Result<Project, RepositoryError>;
    async fn find_by_id(&self, id: ProjectId) -> Result<Option<Project>, RepositoryError>;
    async fn find_by_slug(&self, org_id: OrganizationId, slug: &str) -> Result<Option<Project>, RepositoryError>;
    async fn update(&self, project: Project) -> Result<Project, RepositoryError>;
    async fn delete(&self, id: ProjectId) -> Result<(), RepositoryError>;
    async fn list_by_organization(&self, org_id: OrganizationId, params: ListParams) -> Result<Vec<Project>, RepositoryError>;
    async fn search(&self, query: &str, filters: SearchFilters) -> Result<Vec<Project>, RepositoryError>;
}
```

##### File Index Repository
```rust
#[async_trait]
pub trait FileIndexRepository: Send + Sync {
    async fn create(&self, file: CreateFileIndexRequest) -> Result<FileIndex, RepositoryError>;
    async fn find_by_id(&self, id: FileId) -> Result<Option<FileIndex>, RepositoryError>;
    async fn find_by_path(&self, repo_id: RepositoryId, path: &Path) -> Result<Option<FileIndex>, RepositoryError>;
    async fn update(&self, file: FileIndex) -> Result<FileIndex, RepositoryError>;
    async fn delete(&self, id: FileId) -> Result<(), RepositoryError>;
    async fn list_by_repository(&self, repo_id: RepositoryId, params: FileListParams) -> Result<Vec<FileIndex>, RepositoryError>;
    async fn find_by_language(&self, repo_id: RepositoryId, language: ProgrammingLanguage) -> Result<Vec<FileIndex>, RepositoryError>;
    async fn find_modified_since(&self, repo_id: RepositoryId, since: DateTime<Utc>) -> Result<Vec<FileIndex>, RepositoryError>;
    async fn bulk_update_status(&self, file_ids: Vec<FileId>, status: AnalysisStatus) -> Result<u64, RepositoryError>;
}
```

##### Analysis Repository
```rust
#[async_trait]
pub trait AnalysisRepository: Send + Sync {
    async fn create_job(&self, job: CreateAnalysisJobRequest) -> Result<AnalysisJob, RepositoryError>;
    async fn find_job_by_id(&self, id: JobId) -> Result<Option<AnalysisJob>, RepositoryError>;
    async fn update_job(&self, job: AnalysisJob) -> Result<AnalysisJob, RepositoryError>;
    async fn list_jobs(&self, project_id: ProjectId, params: JobListParams) -> Result<Vec<AnalysisJob>, RepositoryError>;
    
    async fn create_results(&self, results: Vec<CreateAnalysisResultRequest>) -> Result<Vec<AnalysisResult>, RepositoryError>;
    async fn find_results_by_job(&self, job_id: JobId) -> Result<Vec<AnalysisResult>, RepositoryError>;
    async fn find_results_by_file(&self, file_id: FileId) -> Result<Vec<AnalysisResult>, RepositoryError>;
    async fn delete_results_by_job(&self, job_id: JobId) -> Result<u64, RepositoryError>;
    
    async fn create_metrics(&self, metrics: Vec<CreateCodeMetricRequest>) -> Result<Vec<CodeMetric>, RepositoryError>;
    async fn find_metrics_by_file(&self, file_id: FileId) -> Result<Vec<CodeMetric>, RepositoryError>;
    async fn aggregate_metrics(&self, project_id: ProjectId, metric_type: MetricType) -> Result<MetricAggregation, RepositoryError>;
}
```

#### 3.4.2 PostgreSQL Implementation

##### Database Connection Pool
```rust
pub struct DatabasePool {
    pool: PgPool,
    config: DatabaseConfig,
}

impl DatabasePool {
    pub async fn new(config: DatabaseConfig) -> Result<Self, DatabaseError> {
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(config.min_connections)
            .acquire_timeout(Duration::from_secs(config.acquire_timeout_secs))
            .idle_timeout(Duration::from_secs(config.idle_timeout_secs))
            .max_lifetime(Duration::from_secs(config.max_lifetime_secs))
            .connect(&config.database_url)
            .await?;
        
        Ok(Self { pool, config })
    }
    
    pub fn get_pool(&self) -> &PgPool {
        &self.pool
    }
}
```

##### Project Repository Implementation
```rust
pub struct PostgresProjectRepository {
    pool: Arc<DatabasePool>,
}

#[async_trait]
impl ProjectRepository for PostgresProjectRepository {
    async fn create(&self, request: CreateProjectRequest) -> Result<Project, RepositoryError> {
        let mut tx = self.pool.get_pool().begin().await?;
        
        let project = sqlx::query_as!(
            ProjectRow,
            r#"
            INSERT INTO projects (
                organization_id, name, slug, description, repository_url,
                repository_type, default_branch, visibility, settings
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING *
            "#,
            request.organization_id,
            request.name,
            request.slug,
            request.description,
            request.repository_url,
            request.repository_type as _,
            request.default_branch,
            request.visibility as _,
            serde_json::to_value(&request.settings)?
        )
        .fetch_one(&mut *tx)
        .await?;
        
        tx.commit().await?;
        Ok(project.into())
    }
    
    async fn find_by_id(&self, id: ProjectId) -> Result<Option<Project>, RepositoryError> {
        let row = sqlx::query_as!(
            ProjectRow,
            "SELECT * FROM projects WHERE id = $1 AND deleted_at IS NULL",
            id
        )
        .fetch_optional(self.pool.get_pool())
        .await?;
        
        Ok(row.map(Into::into))
    }
}
```

### 3.5 Migraciones de Base de Datos

#### 3.5.1 Sistema de Migraciones
```rust
pub struct MigrationManager {
    pool: Arc<DatabasePool>,
    migrations_path: PathBuf,
}

impl MigrationManager {
    pub async fn run_migrations(&self) -> Result<(), MigrationError> {
        let migrator = Migrator::new(Path::new(&self.migrations_path)).await?;
        migrator.run(self.pool.get_pool()).await?;
        Ok(())
    }
    
    pub async fn rollback_migration(&self, version: i64) -> Result<(), MigrationError> {
        let migrator = Migrator::new(Path::new(&self.migrations_path)).await?;
        migrator.rollback(self.pool.get_pool(), version).await?;
        Ok(())
    }
}
```

#### 3.5.2 Estructura de Migraciones
```
migrations/
├── 001_initial_schema.up.sql
├── 001_initial_schema.down.sql
├── 002_add_organizations.up.sql
├── 002_add_organizations.down.sql
├── 003_add_analysis_tables.up.sql
├── 003_add_analysis_tables.down.sql
├── 004_add_indexes.up.sql
├── 004_add_indexes.down.sql
└── 005_add_full_text_search.up.sql
└── 005_add_full_text_search.down.sql
```

### 3.6 Caching Strategy (Redis)

#### 3.6.1 Cache Layers
```rust
pub struct CacheManager {
    redis: Arc<RedisPool>,
    config: CacheConfig,
}

impl CacheManager {
    // L1 Cache: Hot data (5 minutes TTL)
    pub async fn get_project(&self, id: ProjectId) -> Result<Option<Project>, CacheError> {
        let key = format!("project:{}", id);
        self.get_json(&key).await
    }
    
    // L2 Cache: Analysis results (1 hour TTL)
    pub async fn get_analysis_results(&self, job_id: JobId) -> Result<Option<Vec<AnalysisResult>>, CacheError> {
        let key = format!("analysis:results:{}", job_id);
        self.get_json(&key).await
    }
    
    // L3 Cache: Metrics aggregations (24 hours TTL)
    pub async fn get_project_metrics(&self, project_id: ProjectId) -> Result<Option<ProjectMetrics>, CacheError> {
        let key = format!("metrics:project:{}", project_id);
        self.get_json(&key).await
    }
}
```

#### 3.6.2 Cache Invalidation Strategy
```rust
pub enum CacheInvalidationEvent {
    ProjectUpdated(ProjectId),
    AnalysisCompleted(JobId),
    FileChanged(FileId),
    ConfigurationChanged(ProjectId),
}

pub struct CacheInvalidationHandler {
    cache: Arc<CacheManager>,
    event_bus: Arc<EventBus>,
}

impl CacheInvalidationHandler {
    pub async fn handle_event(&self, event: CacheInvalidationEvent) -> Result<(), CacheError> {
        match event {
            CacheInvalidationEvent::ProjectUpdated(project_id) => {
                self.invalidate_project_cache(project_id).await?;
            }
            CacheInvalidationEvent::AnalysisCompleted(job_id) => {
                self.invalidate_analysis_cache(job_id).await?;
            }
            // ... other events
        }
        Ok(())
    }
}
```

### 3.7 Vector Database (Qdrant)

#### 3.7.1 Code Embeddings Storage
```rust
pub struct VectorStore {
    client: Arc<QdrantClient>,
    collection_name: String,
}

impl VectorStore {
    pub async fn store_code_embedding(&self, embedding: CodeEmbedding) -> Result<(), VectorError> {
        let point = PointStruct::new(
            embedding.id.to_string(),
            embedding.vector,
            Payload::from([
                ("file_id".to_string(), embedding.file_id.into()),
                ("language".to_string(), embedding.language.into()),
                ("function_name".to_string(), embedding.function_name.into()),
                ("code_snippet".to_string(), embedding.code_snippet.into()),
            ])
        );
        
        self.client
            .upsert_points_blocking(&self.collection_name, vec![point], None)
            .await?;
            
        Ok(())
    }
    
    pub async fn search_similar_code(&self, query_vector: Vec<f32>, limit: u64) -> Result<Vec<SimilarCodeResult>, VectorError> {
        let search_result = self.client
            .search_points(&SearchPoints {
                collection_name: self.collection_name.clone(),
                vector: query_vector,
                limit,
                with_payload: Some(WithPayloadSelector::from(true)),
                ..Default::default()
            })
            .await?;
            
        Ok(search_result.result.into_iter().map(Into::into).collect())
    }
}
```

### 3.8 Data Access Layer (DAL)

#### 3.8.1 Unit of Work Pattern
```rust
pub struct UnitOfWork {
    transaction: Option<Transaction<'static, Postgres>>,
    repositories: RepositoryContainer,
    is_committed: bool,
}

impl UnitOfWork {
    pub async fn begin(pool: &PgPool) -> Result<Self, DatabaseError> {
        let transaction = pool.begin().await?;
        let repositories = RepositoryContainer::new(&transaction);
        
        Ok(Self {
            transaction: Some(transaction),
            repositories,
            is_committed: false,
        })
    }
    
    pub async fn commit(mut self) -> Result<(), DatabaseError> {
        if let Some(tx) = self.transaction.take() {
            tx.commit().await?;
            self.is_committed = true;
        }
        Ok(())
    }
    
    pub async fn rollback(mut self) -> Result<(), DatabaseError> {
        if let Some(tx) = self.transaction.take() {
            tx.rollback().await?;
        }
        Ok(())
    }
    
    pub fn projects(&self) -> &dyn ProjectRepository {
        &self.repositories.projects
    }
    
    pub fn files(&self) -> &dyn FileIndexRepository {
        &self.repositories.files
    }
    
    pub fn analysis(&self) -> &dyn AnalysisRepository {
        &self.repositories.analysis
    }
}
```

### 3.9 Query Optimization

#### 3.9.1 Índices Especializados
```sql
-- Full-text search para archivos
CREATE INDEX idx_file_index_fulltext 
ON file_index USING GIN(to_tsvector('english', relative_path || ' ' || COALESCE(metadata->>'description', '')));

-- Índice parcial para archivos activos
CREATE INDEX idx_file_index_active 
ON file_index(repository_id, language) 
WHERE analysis_status != 'deleted';

-- Índice compuesto para consultas de análisis
CREATE INDEX idx_analysis_results_composite 
ON analysis_results(job_id, severity, category) 
INCLUDE (title, description);

-- Índice para métricas con threshold violations
CREATE INDEX idx_metrics_violations 
ON code_metrics(file_id, metric_type, is_violation, value) 
WHERE is_violation = true;
```

#### 3.9.2 Consultas Optimizadas
```rust
impl PostgresAnalysisRepository {
    pub async fn get_project_issues_summary(&self, project_id: ProjectId) -> Result<IssuesSummary, RepositoryError> {
        let summary = sqlx::query_as!(
            IssuesSummaryRow,
            r#"
            SELECT 
                ar.severity,
                ar.category,
                COUNT(*) as count,
                AVG(ar.confidence) as avg_confidence
            FROM analysis_results ar
            JOIN analysis_jobs aj ON ar.job_id = aj.id
            WHERE aj.project_id = $1 
              AND aj.status = 'completed'
              AND ar.created_at > NOW() - INTERVAL '30 days'
            GROUP BY ar.severity, ar.category
            ORDER BY 
                CASE ar.severity 
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END,
                COUNT(*) DESC
            "#,
            project_id
        )
        .fetch_all(self.pool.get_pool())
        .await?;
        
        Ok(summary.into())
    }
}
```

### 3.10 Backup y Recovery

#### 3.10.1 Estrategia de Backup
```rust
pub struct BackupManager {
    database_url: String,
    s3_client: Option<S3Client>,
    backup_config: BackupConfig,
}

impl BackupManager {
    pub async fn create_backup(&self) -> Result<BackupInfo, BackupError> {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let backup_name = format!("codeant_backup_{}.sql", timestamp);
        
        // Create database dump
        let output = Command::new("pg_dump")
            .args(&[
                &self.database_url,
                "--format=custom",
                "--no-owner",
                "--no-privileges",
                "--verbose"
            ])
            .output()
            .await?;
            
        if !output.status.success() {
            return Err(BackupError::DumpFailed(String::from_utf8_lossy(&output.stderr).to_string()));
        }
        
        // Upload to S3 if configured
        if let Some(s3) = &self.s3_client {
            self.upload_to_s3(s3, &backup_name, &output.stdout).await?;
        }
        
        Ok(BackupInfo {
            name: backup_name,
            size: output.stdout.len(),
            created_at: Utc::now(),
            location: BackupLocation::S3,
        })
    }
}
```

### 3.11 Monitoring y Health Checks

#### 3.11.1 Database Health Monitoring
```rust
pub struct DatabaseHealthChecker {
    pool: Arc<DatabasePool>,
}

impl DatabaseHealthChecker {
    pub async fn check_health(&self) -> Result<HealthStatus, HealthCheckError> {
        let start = Instant::now();
        
        // Test connection
        let _row = sqlx::query!("SELECT 1 as test")
            .fetch_one(self.pool.get_pool())
            .await?;
            
        let connection_time = start.elapsed();
        
        // Check pool stats
        let pool_stats = self.pool.get_pool().size();
        let active_connections = self.pool.get_pool().num_idle();
        
        // Check disk space (if available)
        let disk_usage = self.check_disk_usage().await?;
        
        Ok(HealthStatus {
            status: if connection_time < Duration::from_millis(100) { "healthy" } else { "degraded" },
            connection_time_ms: connection_time.as_millis() as u32,
            pool_size: pool_stats,
            active_connections,
            disk_usage_percent: disk_usage,
            last_check: Utc::now(),
        })
    }
}
```

### 3.12 Criterios de Completitud

#### 3.12.1 Entregables de la Fase
- [ ] Esquema completo de base de datos PostgreSQL
- [ ] Sistema de migraciones implementado
- [ ] Repository pattern con implementaciones
- [ ] Caching layer con Redis
- [ ] Vector database setup (Qdrant)
- [ ] Unit of Work pattern
- [ ] Health checks y monitoring
- [ ] Backup/recovery procedures
- [ ] Performance optimization (índices)
- [ ] Tests de integración con base de datos

#### 3.12.2 Criterios de Aceptación
- [ ] Migraciones ejecutan sin errores
- [ ] CRUD operations funcionan correctamente
- [ ] Queries optimizadas con performance acceptable
- [ ] Cache invalidation funciona correctamente
- [ ] Vector search operativo
- [ ] Health checks reportan estado correcto
- [ ] Backup/restore procedures validados
- [ ] Tests de carga pasan (1000+ concurrent operations)
- [ ] Transacciones ACID funcionando correctamente

### 3.13 Performance Benchmarks

#### 3.13.1 Targets de Performance
- **Insert operations**: < 10ms per record
- **Query operations**: < 50ms for simple queries, < 200ms for complex
- **Full-text search**: < 100ms for typical queries
- **Vector similarity search**: < 500ms for 1M+ vectors
- **Cache hit ratio**: > 85% for hot data
- **Connection pool utilization**: < 80% under normal load

### 3.14 Estimación de Tiempo

#### 3.14.1 Breakdown de Tareas
- Diseño del esquema de base de datos: 4 días
- Implementación de migraciones: 2 días
- Repository pattern y implementaciones: 5 días
- Sistema de caching con Redis: 3 días
- Vector database integration: 3 días
- Unit of Work y transacciones: 2 días
- Health checks y monitoring: 2 días
- Optimización de queries e índices: 3 días
- Testing e integración: 4 días
- Documentación: 2 días

**Total estimado: 30 días de desarrollo**

### 3.15 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Base de datos robusta y escalable
- Patrones de acceso a datos bien definidos
- Sistema de caching eficiente
- Capacidades de búsqueda avanzada
- Monitoring y health checks

La Fase 4 construirá sobre esta base implementando la API REST y sistema de autenticación.
