# Fase 2: Sistema de Gestión de Proyectos y Repositorios

## Objetivo General
Implementar un sistema robusto para la gestión, clonado, indexación y mantenimiento de repositorios de código, con soporte para múltiples sistemas de control de versiones y capacidades de análisis incremental.

## Descripción Técnica Detallada

### 2.1 Arquitectura del Sistema de Repositorios

#### 2.1.1 Componentes Principales
```
┌─────────────────────────────────────────┐
│         Repository Manager             │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Git VCS   │ │    Project Index    │ │
│  │   Handler   │ │     Manager         │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   File      │ │    Change           │ │
│  │   Watcher   │ │    Detector         │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│         Storage Abstraction Layer       │
└─────────────────────────────────────────┘
```

#### 2.1.2 Entidades de Dominio

##### Project Entity
```python
@dataclass
class Project:
    id: ProjectId
    name: str
    description: Optional[str]
    repository_url: str
    repository_type: RepositoryType
    default_branch: str
    languages: List[ProgrammingLanguage]
    status: ProjectStatus
    settings: ProjectSettings
    created_at: datetime
    updated_at: datetime
    last_analyzed_at: Optional[datetime]
    metadata: ProjectMetadata
```

##### Repository Entity
```python
@dataclass
class Repository:
    id: RepositoryId
    project_id: ProjectId
    local_path: str
    remote_url: str
    current_commit: CommitHash
    branches: List[Branch]
    tags: List[Tag]
    size_bytes: int
    file_count: int
    last_sync: datetime
    sync_status: SyncStatus
```

##### FileIndex Entity
```python
@dataclass
class FileIndex:
    id: FileId
    repository_id: RepositoryId
    relative_path: str
    language: Optional[ProgrammingLanguage]
    size_bytes: int
    line_count: int
    hash: FileHash
    last_modified: datetime
    analysis_status: AnalysisStatus
    metadata: FileMetadata
```

### 2.2 Sistema de Control de Versiones

#### 2.2.1 Abstracción VCS
```python
from abc import ABC, abstractmethod
from typing import Protocol

class VcsHandler(Protocol):
    async def clone_repository(self, url: str, target_path: str) -> Repository:
        ...
    
    async def fetch_updates(self, repo: Repository) -> List[Commit]:
        ...
    
    async def get_file_history(self, repo: Repository, file_path: str) -> List[FileChange]:
        ...
    
    async def get_diff(self, repo: Repository, from_commit: CommitHash, to_commit: CommitHash) -> Diff:
        ...
    
    async def get_blame(self, repo: Repository, file_path: str) -> BlameInfo:
        ...
    
    async def list_branches(self, repo: Repository) -> List[Branch]:
        ...
    
    async def checkout_commit(self, repo: Repository, commit: CommitHash) -> None:
        ...
```

#### 2.2.2 Implementación Git
- **Librería principal**: `GitPython` (Python Git library)
- **Funcionalidades**:
  - Clonado con soporte para credenciales
  - Fetch incremental con detección de cambios
  - Análisis de historial y blame
  - Manejo de branches y tags
  - Soporte para Git LFS
  - Autenticación SSH/HTTPS

#### 2.2.3 Soporte Futuro para Otros VCS
- **Mercurial**: Via `hg` command line
- **Subversion**: Via `svn` command line
- **Perforce**: Via P4 API
- **Azure DevOps**: Via REST API

### 2.3 Gestión de Proyectos

#### 2.3.1 Project Manager Service
```python
@dataclass
class ProjectManager:
    repository: ProjectRepository
    vcs_factory: VcsHandlerFactory
    file_watcher: FileWatcher
    event_bus: EventBus
    
    async def create_project(self, request: CreateProjectRequest) -> Project:
        ...
    
    async def import_repository(self, url: str) -> Project:
        ...
    
    async def sync_project(self, project_id: ProjectId) -> SyncResult:
        ...
    
    async def analyze_project(self, project_id: ProjectId) -> AnalysisJob:
        ...
    
    async def get_project_stats(self, project_id: ProjectId) -> ProjectStats:
        ...
```

#### 2.3.2 Configuración de Proyecto
```python
@dataclass
class ProjectSettings:
    analysis_config: AnalysisConfig
    ignore_patterns: List[GlobPattern]
    include_patterns: List[GlobPattern]
    max_file_size_mb: int
    enable_incremental_analysis: bool
    webhook_url: Optional[str]
    notification_settings: NotificationSettings
    custom_rules: List[CustomRule]

@dataclass
class AnalysisConfig:
    enable_static_analysis: bool
    enable_ai_analysis: bool
    enable_security_scan: bool
    enable_performance_analysis: bool
    languages: List[ProgrammingLanguage]
    complexity_thresholds: ComplexityThresholds
    quality_gates: QualityGates
```

### 2.4 Indexación de Archivos

#### 2.4.1 File Indexer Service
```python
@dataclass
class FileIndexer:
    repository: FileIndexRepository
    language_detector: LanguageDetector
    hash_calculator: HashCalculator
    metadata_extractor: MetadataExtractor
    
    async def index_repository(self, repo: Repository) -> IndexResult:
        ...
    
    async def update_file_index(self, file_path: str) -> FileIndex:
        ...
    
    async def detect_changes(self, repo: Repository) -> List[FileChange]:
        ...
    
    async def cleanup_deleted_files(self, repo: Repository) -> int:
        ...
```

#### 2.4.2 Detección de Lenguajes
- **Librería principal**: Custom implementation + `tree-sitter`
- **Estrategias de detección**:
  1. Extensión de archivo
  2. Shebang line analysis
  3. Content-based detection
  4. Filename patterns
  5. Tree-sitter parsing validation

#### 2.4.3 Cálculo de Hashes
- **Algoritmo**: BLAKE3 para performance
- **Niveles de hash**:
  - Contenido completo del archivo
  - Hash de estructura (sin comentarios/whitespace)
  - Hash semántico (AST-based)

### 2.5 Monitoreo de Cambios

#### 2.5.1 File Watcher Implementation
```python
@dataclass
class FileWatcher:
    watcher: Any  # watchdog.Watcher
    event_sender: asyncio.Queue
    watched_paths: Dict[str, ProjectId]
    
    class FileEvent(Enum):
        CREATED = "created"
        MODIFIED = "modified"
        DELETED = "deleted"
        RENAMED = "renamed"
```

#### 2.5.2 Change Detection Strategy
- **Inotify** (Linux) / **FSEvents** (macOS) / **ReadDirectoryChangesW** (Windows)
- **Debouncing**: 500ms para evitar eventos duplicados
- **Filtering**: Ignorar archivos temporales y build artifacts
- **Batch processing**: Agrupar cambios relacionados

### 2.6 Sistema de Storage

#### 2.6.1 Repository Storage Layout
```
/data/repositories/
├── projects/
│   └── {project_id}/
│       ├── repository/          # Git repository
│       ├── analysis/            # Analysis results cache
│       │   ├── static/          # Static analysis results
│       │   ├── ai/              # AI analysis results
│       │   └── metrics/         # Computed metrics
│       ├── index/               # File index cache
│       └── metadata/            # Project metadata
└── global/
    ├── models/                  # AI models cache
    ├── rules/                   # Global rules cache
    └── templates/               # Project templates
```

#### 2.6.2 Storage Abstraction
```python
from abc import ABC, abstractmethod

class StorageProvider(ABC):
    @abstractmethod
    async def store_file(self, key: str, content: bytes) -> None:
        ...
    
    @abstractmethod
    async def retrieve_file(self, key: str) -> bytes:
        ...
    
    @abstractmethod
    async def delete_file(self, key: str) -> None:
        ...
    
    @abstractmethod
    async def list_files(self, prefix: str) -> List[str]:
        ...
    
    @abstractmethod
    async def get_metadata(self, key: str) -> FileMetadata:
        ...
```

#### 2.6.3 Implementaciones de Storage
- **Local FileSystem**: Para desarrollo y deployments simples
- **AWS S3**: Para producción cloud
- **MinIO**: Para on-premise object storage
- **Google Cloud Storage**: Soporte futuro

### 2.7 Sincronización y Actualizaciones

#### 2.7.1 Sync Manager
```python
@dataclass
class SyncManager:
    project_manager: ProjectManager
    scheduler: JobScheduler
    webhook_handler: WebhookHandler
    
    async def schedule_sync(self, project_id: ProjectId, interval: timedelta) -> None:
        ...
    
    async def trigger_immediate_sync(self, project_id: ProjectId) -> SyncJob:
        ...
    
    async def handle_webhook(self, payload: WebhookPayload) -> None:
        ...
```

#### 2.7.2 Estrategias de Sincronización
- **Polling**: Intervalo configurable (default: 5 minutos)
- **Webhooks**: Push notifications de Git providers
- **Manual**: Trigger manual via API/UI
- **Event-driven**: Basado en file system events

#### 2.7.3 Webhook Support
```python
class WebhookProvider(Enum):
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    AZURE_DEVOPS = "azure_devops"
    GENERIC = "generic"

@dataclass
class WebhookPayload:
    provider: WebhookProvider
    repository_url: str
    branch: str
    commits: List[CommitInfo]
    event_type: WebhookEventType
```

### 2.8 Análisis Incremental

#### 2.8.1 Change Detection Engine
```python
@dataclass
class ChangeDetector:
    file_indexer: FileIndexer
    diff_analyzer: DiffAnalyzer
    dependency_tracker: DependencyTracker
    
    async def detect_changes(self, from_commit: CommitHash, to_commit: CommitHash) -> ChangeSet:
        ...
    
    async def calculate_affected_files(self, changes: ChangeSet) -> List[str]:
        ...
    
    async def determine_analysis_scope(self, affected_files: List[str]) -> AnalysisScope:
        ...
```

#### 2.8.2 Dependency Tracking
- **Import/Export analysis**: Rastrear dependencias entre archivos
- **Call graph construction**: Para lenguajes compilados
- **Module dependency mapping**: Para sistemas modulares
- **Transitive dependency calculation**: Impacto de cambios

### 2.9 Métricas y Estadísticas

#### 2.9.1 Project Statistics
```python
@dataclass
class ProjectStats:
    total_files: int
    total_lines: int
    languages: Dict[ProgrammingLanguage, LanguageStats]
    repository_size: int
    last_commit: Optional[CommitInfo]
    commit_frequency: CommitFrequency
    contributor_count: int
    technical_debt: TechnicalDebtMetrics

@dataclass
class LanguageStats:
    file_count: int
    line_count: int
    percentage: float
    complexity_average: float
```

#### 2.9.2 Health Metrics
- **Repository freshness**: Tiempo desde último commit
- **Sync status**: Estado de sincronización
- **Analysis coverage**: Porcentaje de archivos analizados
- **Error rates**: Frecuencia de errores de sync/análisis

### 2.10 API Endpoints

#### 2.10.1 Project Management API
```python
# POST /api/v1/projects
@router.post("/projects")
async def create_project(request: CreateProjectRequest) -> Project:
    ...

# GET /api/v1/projects/{id}
@router.get("/projects/{id}")
async def get_project(id: ProjectId) -> Project:
    ...

# PUT /api/v1/projects/{id}
@router.put("/projects/{id}")
async def update_project(id: ProjectId, request: UpdateProjectRequest) -> Project:
    ...

# DELETE /api/v1/projects/{id}
@router.delete("/projects/{id}")
async def delete_project(id: ProjectId) -> None:
    ...

# POST /api/v1/projects/{id}/sync
@router.post("/projects/{id}/sync")
async def sync_project(id: ProjectId) -> SyncJob:
    ...

# GET /api/v1/projects/{id}/stats
@router.get("/projects/{id}/stats")
async def get_project_stats(id: ProjectId) -> ProjectStats:
    ...
```

#### 2.10.2 Repository Management API
```python
# GET /api/v1/projects/{id}/files
@router.get("/projects/{id}/files")
async def list_project_files(id: ProjectId, params: FileListParams) -> List[FileIndex]:
    ...

# GET /api/v1/projects/{id}/files/{file_id}
@router.get("/projects/{id}/files/{file_id}")
async def get_file_details(id: ProjectId, file_id: FileId) -> FileDetails:
    ...

# POST /api/v1/projects/{id}/reindex
@router.post("/projects/{id}/reindex")
async def reindex_project(id: ProjectId) -> IndexJob:
    ...
```

### 2.11 Sistema de Eventos

#### 2.11.1 Event Bus Implementation
```python
class ProjectEvent(Enum):
    PROJECT_CREATED = "project_created"
    PROJECT_UPDATED = "project_updated"
    PROJECT_DELETED = "project_deleted"
    SYNC_STARTED = "sync_started"
    SYNC_COMPLETED = "sync_completed"
    SYNC_FAILED = "sync_failed"
    FILE_CHANGED = "file_changed"
    ANALYSIS_REQUESTED = "analysis_requested"

@dataclass
class ProjectEventData:
    event_type: ProjectEvent
    project_id: ProjectId
    data: Dict[str, Any]
```

#### 2.11.2 Event Handlers
- **Notification Service**: Envío de notificaciones
- **Analytics Service**: Recolección de métricas
- **Webhook Service**: Notificaciones externas
- **Cache Invalidation**: Limpieza de caches

### 2.12 Configuración y Seguridad

#### 2.12.1 Authentication & Authorization
```python
@dataclass
class ProjectPermissions:
    can_read: bool
    can_write: bool
    can_delete: bool
    can_analyze: bool
    can_configure: bool

class ProjectRole(Enum):
    OWNER = "owner"
    MAINTAINER = "maintainer"
    DEVELOPER = "developer"
    VIEWER = "viewer"
```

#### 2.12.2 Rate Limiting
- **Per-user limits**: API calls por usuario
- **Per-project limits**: Operaciones por proyecto
- **Global limits**: Límites del sistema
- **Adaptive throttling**: Basado en carga del sistema

### 2.13 Testing Strategy

#### 2.13.1 Unit Tests
- Repository abstractions con mocks
- VCS handler implementations
- File indexing logic
- Change detection algorithms

#### 2.13.2 Integration Tests
- End-to-end project creation flow
- Git repository cloning and syncing
- Webhook handling
- File watching functionality

#### 2.13.3 Performance Tests
- Large repository handling (>100k files)
- Concurrent sync operations
- Memory usage during indexing
- API response times

### 2.14 Criterios de Completitud

#### 2.14.1 Entregables de la Fase
- [ ] Sistema completo de gestión de proyectos
- [ ] Implementación Git VCS handler
- [ ] File indexing system funcionando
- [ ] Change detection implementado
- [ ] API REST completa para gestión
- [ ] Sistema de eventos funcionando
- [ ] Tests de integración pasando
- [ ] Documentación de API

#### 2.14.2 Criterios de Aceptación
- [ ] Crear proyecto desde repository URL
- [ ] Clonar y sincronizar repositorios Git
- [ ] Indexar archivos automáticamente
- [ ] Detectar cambios incrementales
- [ ] API endpoints responden correctamente
- [ ] File watcher detecta cambios en tiempo real
- [ ] Webhooks procesan correctamente
- [ ] Performance acceptable para repos de 10k+ archivos

### 2.15 Estimación de Tiempo

#### 2.15.1 Breakdown de Tareas
- Diseño de entidades y repositorios: 3 días
- Implementación VCS Git handler: 4 días
- Sistema de indexación de archivos: 4 días
- File watcher y change detection: 3 días
- API REST endpoints: 3 días
- Sistema de eventos: 2 días
- Testing e integración: 3 días
- Documentación: 2 días

**Total estimado: 24 días de desarrollo**

### 2.16 Próximos Pasos

Al completar esta fase, el sistema podrá:
- Gestionar múltiples proyectos de código
- Sincronizar repositorios automáticamente
- Indexar y rastrear cambios en archivos
- Proporcionar APIs para gestión de proyectos
- Detectar cambios incrementales eficientemente

La Fase 3 construirá sobre esta base implementando la persistencia y modelos de datos fundamentales.
