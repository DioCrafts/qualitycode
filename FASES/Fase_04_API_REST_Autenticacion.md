# Fase 4: API REST Básica y Sistema de Autenticación

## Objetivo General
Implementar una API REST completa y segura con sistema de autenticación robusto, autorización basada en roles, rate limiting, y documentación automática que sirva como interfaz principal para todas las operaciones del sistema CodeAnt.

## Descripción Técnica Detallada

### 4.1 Arquitectura de la API

#### 4.1.1 Diseño RESTful
```
┌─────────────────────────────────────────┐
│              API Gateway                │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │    Auth     │ │    Rate Limiting    │ │
│  │ Middleware  │ │    Middleware       │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Router    │ │    Validation       │ │
│  │  Handler    │ │    Middleware       │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│         Application Services            │
└─────────────────────────────────────────┘
```

#### 4.1.2 Principios de Diseño API
- **RESTful**: Recursos bien definidos con operaciones HTTP estándar
- **Stateless**: Sin estado en el servidor, JWT para sesiones
- **Versionado**: API versionada desde el inicio (/api/v1/)
- **Consistent**: Estructura consistente de respuestas
- **Secure**: HTTPS obligatorio, autenticación en todos los endpoints
- **Documented**: OpenAPI 3.0 con documentación automática

### 4.2 Framework y Stack Tecnológico

#### 4.2.1 Axum Web Framework
```rust
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Json},
    routing::{delete, get, post, put},
    Router,
};

pub struct ApiServer {
    app: Router,
    config: ServerConfig,
    state: AppState,
}

impl ApiServer {
    pub fn new(config: ServerConfig, state: AppState) -> Self {
        let app = create_router(state.clone());
        Self { app, config, state }
    }
    
    pub async fn serve(self) -> Result<(), ServerError> {
        let listener = tokio::net::TcpListener::bind(
            format!("{}:{}", self.config.host, self.config.port)
        ).await?;
        
        axum::serve(listener, self.app).await?;
        Ok(())
    }
}
```

#### 4.2.2 Application State
```rust
#[derive(Clone)]
pub struct AppState {
    pub database: Arc<DatabasePool>,
    pub cache: Arc<CacheManager>,
    pub auth_service: Arc<AuthService>,
    pub project_service: Arc<ProjectService>,
    pub analysis_service: Arc<AnalysisService>,
    pub config: Arc<AppConfig>,
    pub event_bus: Arc<EventBus>,
    pub rate_limiter: Arc<RateLimiter>,
}
```

### 4.3 Sistema de Autenticación

#### 4.3.1 JWT Implementation
```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,              // Subject (user ID)
    pub email: String,            // User email
    pub org_id: Option<String>,   // Organization ID
    pub roles: Vec<String>,       // User roles
    pub permissions: Vec<String>, // Specific permissions
    pub exp: usize,               // Expiration timestamp
    pub iat: usize,               // Issued at timestamp
    pub iss: String,              // Issuer
    pub aud: String,              // Audience
}

pub struct JwtService {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    validation: Validation,
}

impl JwtService {
    pub fn new(secret: &str) -> Result<Self, JwtError> {
        let encoding_key = EncodingKey::from_secret(secret.as_bytes());
        let decoding_key = DecodingKey::from_secret(secret.as_bytes());
        
        let mut validation = Validation::new(Algorithm::HS256);
        validation.set_audience(&["codeant-api"]);
        validation.set_issuer(&["codeant-server"]);
        
        Ok(Self {
            encoding_key,
            decoding_key,
            validation,
        })
    }
    
    pub fn create_token(&self, claims: Claims) -> Result<String, JwtError> {
        encode(&Header::default(), &claims, &self.encoding_key)
            .map_err(JwtError::from)
    }
    
    pub fn validate_token(&self, token: &str) -> Result<Claims, JwtError> {
        decode::<Claims>(token, &self.decoding_key, &self.validation)
            .map(|data| data.claims)
            .map_err(JwtError::from)
    }
}
```

#### 4.3.2 Authentication Middleware
```rust
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Result<Response, AuthError> {
    let auth_header = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|header| header.to_str().ok())
        .ok_or(AuthError::MissingToken)?;
    
    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or(AuthError::InvalidTokenFormat)?;
    
    let claims = state.auth_service.validate_token(token)?;
    
    // Add user info to request extensions
    request.extensions_mut().insert(CurrentUser {
        id: UserId::from_str(&claims.sub)?,
        email: claims.email,
        organization_id: claims.org_id.map(|id| OrganizationId::from_str(&id)).transpose()?,
        roles: claims.roles,
        permissions: claims.permissions,
    });
    
    Ok(next.run(request).await)
}
```

#### 4.3.3 Authorization System
```rust
pub struct AuthorizationService {
    role_permissions: HashMap<Role, Vec<Permission>>,
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Role {
    SuperAdmin,
    OrganizationAdmin,
    ProjectMaintainer,
    Developer,
    Viewer,
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Permission {
    // Organization permissions
    OrganizationRead,
    OrganizationWrite,
    OrganizationDelete,
    OrganizationManageUsers,
    
    // Project permissions
    ProjectCreate,
    ProjectRead,
    ProjectWrite,
    ProjectDelete,
    ProjectAnalyze,
    ProjectConfigure,
    
    // Analysis permissions
    AnalysisRead,
    AnalysisCreate,
    AnalysisDelete,
    
    // System permissions
    SystemAdmin,
    SystemMetrics,
}

impl AuthorizationService {
    pub fn has_permission(&self, user: &CurrentUser, permission: Permission) -> bool {
        user.permissions.contains(&permission.to_string()) ||
        user.roles.iter().any(|role| {
            if let Ok(role) = Role::from_str(role) {
                self.role_permissions
                    .get(&role)
                    .map(|perms| perms.contains(&permission))
                    .unwrap_or(false)
            } else {
                false
            }
        })
    }
    
    pub fn check_project_access(&self, user: &CurrentUser, project_id: ProjectId, permission: Permission) -> Result<(), AuthorizationError> {
        // Check if user has global permission
        if self.has_permission(user, permission.clone()) {
            return Ok(());
        }
        
        // Check project-specific permissions
        // This would involve checking project membership tables
        todo!("Implement project-specific permission checking")
    }
}
```

### 4.4 Estructura de Endpoints

#### 4.4.1 Authentication Endpoints
```rust
pub fn auth_routes() -> Router<AppState> {
    Router::new()
        .route("/auth/register", post(register_user))
        .route("/auth/login", post(login_user))
        .route("/auth/logout", post(logout_user))
        .route("/auth/refresh", post(refresh_token))
        .route("/auth/forgot-password", post(forgot_password))
        .route("/auth/reset-password", post(reset_password))
        .route("/auth/verify-email", post(verify_email))
        .route("/auth/me", get(get_current_user))
}

#[derive(Debug, Deserialize, Validate)]
pub struct RegisterRequest {
    #[validate(email)]
    pub email: String,
    #[validate(length(min = 8, max = 128))]
    pub password: String,
    #[validate(length(min = 2, max = 100))]
    pub full_name: String,
    #[validate(length(min = 3, max = 50))]
    pub username: String,
    pub organization_name: Option<String>,
}

pub async fn register_user(
    State(state): State<AppState>,
    Json(request): Json<RegisterRequest>,
) -> Result<Json<AuthResponse>, ApiError> {
    request.validate()?;
    
    let user = state.auth_service.register_user(request).await?;
    let token = state.auth_service.create_session(&user).await?;
    
    Ok(Json(AuthResponse {
        user: user.into(),
        token,
        expires_in: 3600,
    }))
}
```

#### 4.4.2 Project Management Endpoints
```rust
pub fn project_routes() -> Router<AppState> {
    Router::new()
        .route("/projects", get(list_projects).post(create_project))
        .route("/projects/:id", get(get_project).put(update_project).delete(delete_project))
        .route("/projects/:id/sync", post(sync_project))
        .route("/projects/:id/analyze", post(analyze_project))
        .route("/projects/:id/stats", get(get_project_stats))
        .route("/projects/:id/files", get(list_project_files))
        .route("/projects/:id/files/:file_id", get(get_file_details))
        .route("/projects/:id/issues", get(list_project_issues))
        .route("/projects/:id/metrics", get(get_project_metrics))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
}

#[derive(Debug, Deserialize, Validate)]
pub struct CreateProjectRequest {
    #[validate(length(min = 1, max = 255))]
    pub name: String,
    #[validate(length(min = 3, max = 100))]
    pub slug: String,
    pub description: Option<String>,
    #[validate(url)]
    pub repository_url: String,
    pub repository_type: Option<RepositoryType>,
    pub default_branch: Option<String>,
    pub visibility: Option<ProjectVisibility>,
    pub settings: Option<ProjectSettings>,
}

pub async fn create_project(
    State(state): State<AppState>,
    Extension(user): Extension<CurrentUser>,
    Json(request): Json<CreateProjectRequest>,
) -> Result<Json<ProjectResponse>, ApiError> {
    request.validate()?;
    
    // Check permissions
    state.auth_service.check_permission(&user, Permission::ProjectCreate)?;
    
    let project = state.project_service.create_project(user.organization_id, request).await?;
    
    Ok(Json(project.into()))
}
```

#### 4.4.3 Analysis Endpoints
```rust
pub fn analysis_routes() -> Router<AppState> {
    Router::new()
        .route("/analysis/jobs", get(list_analysis_jobs).post(create_analysis_job))
        .route("/analysis/jobs/:id", get(get_analysis_job).delete(cancel_analysis_job))
        .route("/analysis/jobs/:id/results", get(get_analysis_results))
        .route("/analysis/jobs/:id/download", get(download_analysis_report))
        .route("/analysis/rules", get(list_rules))
        .route("/analysis/rules/:id", get(get_rule_details))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
}

#[derive(Debug, Deserialize, Validate)]
pub struct CreateAnalysisJobRequest {
    pub project_id: ProjectId,
    pub job_type: AnalysisType,
    pub config: Option<AnalysisConfig>,
    pub priority: Option<i32>,
    pub scope: Option<AnalysisScope>,
}

pub async fn create_analysis_job(
    State(state): State<AppState>,
    Extension(user): Extension<CurrentUser>,
    Json(request): Json<CreateAnalysisJobRequest>,
) -> Result<Json<AnalysisJobResponse>, ApiError> {
    request.validate()?;
    
    // Check project access
    state.auth_service.check_project_access(&user, request.project_id, Permission::ProjectAnalyze)?;
    
    let job = state.analysis_service.create_job(request).await?;
    
    Ok(Json(job.into()))
}
```

### 4.5 Response Format Standardization

#### 4.5.1 Standard Response Types
```rust
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<ApiErrorDetails>,
    pub meta: Option<ResponseMeta>,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorDetails {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub trace_id: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseMeta {
    pub timestamp: DateTime<Utc>,
    pub request_id: String,
    pub version: String,
    pub pagination: Option<PaginationMeta>,
}

#[derive(Debug, Serialize)]
pub struct PaginationMeta {
    pub page: u32,
    pub per_page: u32,
    pub total: u64,
    pub total_pages: u32,
    pub has_next: bool,
    pub has_prev: bool,
}
```

#### 4.5.2 Error Handling
```rust
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Authentication required")]
    Unauthorized,
    
    #[error("Access denied")]
    Forbidden,
    
    #[error("Resource not found")]
    NotFound,
    
    #[error("Validation failed: {0}")]
    ValidationError(#[from] validator::ValidationErrors),
    
    #[error("Database error: {0}")]
    DatabaseError(#[from] DatabaseError),
    
    #[error("Service error: {0}")]
    ServiceError(String),
    
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    
    #[error("Internal server error")]
    InternalError,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_code, message) = match self {
            ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, "UNAUTHORIZED", "Authentication required"),
            ApiError::Forbidden => (StatusCode::FORBIDDEN, "FORBIDDEN", "Access denied"),
            ApiError::NotFound => (StatusCode::NOT_FOUND, "NOT_FOUND", "Resource not found"),
            ApiError::ValidationError(_) => (StatusCode::BAD_REQUEST, "VALIDATION_ERROR", "Validation failed"),
            ApiError::RateLimitExceeded => (StatusCode::TOO_MANY_REQUESTS, "RATE_LIMIT_EXCEEDED", "Rate limit exceeded"),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", "Internal server error"),
        };
        
        let response = ApiResponse::<()> {
            success: false,
            data: None,
            error: Some(ApiErrorDetails {
                code: error_code.to_string(),
                message: message.to_string(),
                details: None,
                trace_id: generate_trace_id(),
            }),
            meta: Some(ResponseMeta {
                timestamp: Utc::now(),
                request_id: generate_request_id(),
                version: "v1".to_string(),
                pagination: None,
            }),
        };
        
        (status, Json(response)).into_response()
    }
}
```

### 4.6 Request Validation

#### 4.6.1 Validation Middleware
```rust
use validator::{Validate, ValidationErrors};

pub async fn validation_middleware<T>(
    request: Json<T>,
) -> Result<Json<T>, ApiError>
where
    T: Validate,
{
    request.validate()?;
    Ok(request)
}

// Custom validators
pub fn validate_project_slug(slug: &str) -> Result<(), validator::ValidationError> {
    let regex = Regex::new(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$").unwrap();
    if regex.is_match(slug) && slug.len() >= 3 && slug.len() <= 100 {
        Ok(())
    } else {
        Err(validator::ValidationError::new("invalid_slug"))
    }
}

pub fn validate_repository_url(url: &str) -> Result<(), validator::ValidationError> {
    let parsed_url = Url::parse(url).map_err(|_| validator::ValidationError::new("invalid_url"))?;
    
    match parsed_url.scheme() {
        "https" | "http" | "ssh" | "git" => Ok(()),
        _ => Err(validator::ValidationError::new("unsupported_scheme"))
    }
}
```

### 4.7 Rate Limiting

#### 4.7.1 Rate Limiter Implementation
```rust
use governor::{Quota, RateLimiter as GovRateLimiter, state::{InMemoryState, keyed::DashMapStateStore}};

pub struct RateLimiter {
    limiter: Arc<GovRateLimiter<String, DashMapStateStore<String>, clock::DefaultClock>>,
    config: RateLimitConfig,
}

#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub burst_size: u32,
    pub whitelist: Vec<String>,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        let quota = Quota::per_minute(NonZeroU32::new(config.requests_per_minute).unwrap())
            .allow_burst(NonZeroU32::new(config.burst_size).unwrap());
            
        let limiter = Arc::new(GovRateLimiter::keyed(quota));
        
        Self { limiter, config }
    }
    
    pub fn check_rate_limit(&self, key: &str) -> Result<(), RateLimitError> {
        if self.config.whitelist.contains(&key.to_string()) {
            return Ok(());
        }
        
        match self.limiter.check_key(key) {
            Ok(_) => Ok(()),
            Err(_) => Err(RateLimitError::LimitExceeded),
        }
    }
}

pub async fn rate_limit_middleware(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let client_ip = addr.ip().to_string();
    
    state.rate_limiter.check_rate_limit(&client_ip)?;
    
    Ok(next.run(request).await)
}
```

### 4.8 OpenAPI Documentation

#### 4.8.1 Utoipa Integration
```rust
use utoipa::{OpenApi, ToSchema};

#[derive(OpenApi)]
#[openapi(
    paths(
        register_user,
        login_user,
        create_project,
        get_project,
        list_projects,
        create_analysis_job,
        get_analysis_results,
    ),
    components(
        schemas(
            RegisterRequest,
            LoginRequest,
            AuthResponse,
            CreateProjectRequest,
            ProjectResponse,
            AnalysisJobResponse,
            ApiError
        )
    ),
    tags(
        (name = "auth", description = "Authentication endpoints"),
        (name = "projects", description = "Project management endpoints"),
        (name = "analysis", description = "Code analysis endpoints")
    ),
    info(
        title = "CodeAnt API",
        version = "1.0.0",
        description = "Professional AI-powered code quality analysis API",
        contact(
            name = "CodeAnt Support",
            email = "support@codeant.com"
        ),
        license(
            name = "MIT",
            url = "https://opensource.org/licenses/MIT"
        )
    ),
    servers(
        (url = "https://api.codeant.com/v1", description = "Production server"),
        (url = "https://staging-api.codeant.com/v1", description = "Staging server"),
        (url = "http://localhost:8080/api/v1", description = "Development server")
    )
)]
pub struct ApiDoc;

pub fn create_docs_router() -> Router<AppState> {
    Router::new()
        .route("/docs", get(serve_swagger_ui))
        .route("/docs/openapi.json", get(serve_openapi_spec))
}

async fn serve_openapi_spec() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}
```

### 4.9 Middleware Stack

#### 4.9.1 Complete Middleware Chain
```rust
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .merge(auth_routes())
        .merge(project_routes())
        .merge(analysis_routes())
        .merge(create_docs_router())
        .layer(
            ServiceBuilder::new()
                .layer(HandleErrorLayer::new(|error: BoxError| async move {
                    if error.is::<tower::timeout::error::Elapsed>() {
                        Ok(StatusCode::REQUEST_TIMEOUT)
                    } else {
                        Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Unhandled internal error: {}", error),
                        ))
                    }
                }))
                .timeout(Duration::from_secs(30))
                .layer(TraceLayer::new_for_http())
                .layer(middleware::from_fn_with_state(state.clone(), request_id_middleware))
                .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware))
                .layer(middleware::from_fn_with_state(state.clone(), cors_middleware))
                .layer(middleware::from_fn_with_state(state.clone(), security_headers_middleware))
                .into_inner(),
        )
        .with_state(state)
}
```

#### 4.9.2 Security Headers Middleware
```rust
pub async fn security_headers_middleware(
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    
    let headers = response.headers_mut();
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert("Strict-Transport-Security", "max-age=31536000; includeSubDomains".parse().unwrap());
    headers.insert("Content-Security-Policy", "default-src 'self'".parse().unwrap());
    headers.insert("Referrer-Policy", "strict-origin-when-cross-origin".parse().unwrap());
    
    response
}
```

### 4.10 Testing Strategy

#### 4.10.1 Integration Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;
    use serde_json::json;
    
    async fn setup_test_server() -> TestServer {
        let config = TestConfig::default();
        let state = AppState::new_test(config).await;
        let app = create_router(state);
        TestServer::new(app).unwrap()
    }
    
    #[tokio::test]
    async fn test_user_registration() {
        let server = setup_test_server().await;
        
        let response = server
            .post("/auth/register")
            .json(&json!({
                "email": "test@example.com",
                "password": "password123",
                "full_name": "Test User",
                "username": "testuser"
            }))
            .await;
            
        response.assert_status_ok();
        
        let auth_response: AuthResponse = response.json();
        assert!(!auth_response.token.is_empty());
        assert_eq!(auth_response.user.email, "test@example.com");
    }
    
    #[tokio::test]
    async fn test_project_creation_requires_auth() {
        let server = setup_test_server().await;
        
        let response = server
            .post("/projects")
            .json(&json!({
                "name": "Test Project",
                "slug": "test-project",
                "repository_url": "https://github.com/user/repo.git"
            }))
            .await;
            
        response.assert_status(StatusCode::UNAUTHORIZED);
    }
    
    #[tokio::test]
    async fn test_rate_limiting() {
        let server = setup_test_server().await;
        
        // Make multiple requests rapidly
        for i in 0..100 {
            let response = server.get("/auth/me").await;
            if i < 60 {
                // Should be within rate limit
                assert!(response.status_code() == StatusCode::UNAUTHORIZED); // No auth header
            } else {
                // Should hit rate limit
                if response.status_code() == StatusCode::TOO_MANY_REQUESTS {
                    break;
                }
            }
        }
    }
}
```

### 4.11 Performance Optimization

#### 4.11.1 Connection Pooling
```rust
pub struct ConnectionManager {
    database_pool: Arc<DatabasePool>,
    redis_pool: Arc<RedisPool>,
    http_client: Arc<HttpClient>,
}

impl ConnectionManager {
    pub async fn new(config: &AppConfig) -> Result<Self, ConnectionError> {
        let database_pool = Arc::new(
            DatabasePool::new(config.database.clone()).await?
        );
        
        let redis_pool = Arc::new(
            RedisPool::new(config.redis.clone()).await?
        );
        
        let http_client = Arc::new(
            HttpClient::builder()
                .pool_max_idle_per_host(20)
                .pool_idle_timeout(Duration::from_secs(90))
                .timeout(Duration::from_secs(30))
                .build()?
        );
        
        Ok(Self {
            database_pool,
            redis_pool,
            http_client,
        })
    }
}
```

#### 4.11.2 Response Caching
```rust
pub async fn cache_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    // Only cache GET requests
    if request.method() != Method::GET {
        return next.run(request).await;
    }
    
    let cache_key = generate_cache_key(&request);
    
    // Try to get from cache
    if let Ok(Some(cached_response)) = state.cache.get(&cache_key).await {
        return cached_response;
    }
    
    let response = next.run(request).await;
    
    // Cache successful responses
    if response.status().is_success() {
        let _ = state.cache.set(&cache_key, &response, Duration::from_secs(300)).await;
    }
    
    response
}
```

### 4.12 Health Checks y Monitoring

#### 4.12.1 Health Check Endpoints
```rust
pub fn health_routes() -> Router<AppState> {
    Router::new()
        .route("/health", get(health_check))
        .route("/health/ready", get(readiness_check))
        .route("/health/live", get(liveness_check))
        .route("/metrics", get(metrics_endpoint))
}

pub async fn health_check(State(state): State<AppState>) -> Json<HealthStatus> {
    let database_health = state.database.health_check().await;
    let cache_health = state.cache.health_check().await;
    
    let overall_status = if database_health.is_healthy && cache_health.is_healthy {
        "healthy"
    } else {
        "unhealthy"
    };
    
    Json(HealthStatus {
        status: overall_status.to_string(),
        timestamp: Utc::now(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        services: vec![
            ServiceHealth {
                name: "database".to_string(),
                status: database_health.status,
                response_time_ms: database_health.response_time_ms,
            },
            ServiceHealth {
                name: "cache".to_string(),
                status: cache_health.status,
                response_time_ms: cache_health.response_time_ms,
            },
        ],
    })
}
```

### 4.13 Configuration Management

#### 4.13.1 Environment-based Configuration
```rust
#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub redis: RedisConfig,
    pub auth: AuthConfig,
    pub rate_limiting: RateLimitConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
    pub max_request_size: usize,
    pub request_timeout_secs: u64,
    pub keep_alive_timeout_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    pub jwt_secret: String,
    pub jwt_expiration_hours: u64,
    pub refresh_token_expiration_days: u64,
    pub bcrypt_cost: u32,
    pub password_min_length: usize,
    pub session_timeout_minutes: u64,
}

impl AppConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        let mut config = config::Config::builder()
            .add_source(config::File::with_name("config/default"))
            .add_source(config::File::with_name(&format!(
                "config/{}",
                std::env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string())
            )).required(false))
            .add_source(config::Environment::with_prefix("CODEANT"))
            .build()?;
            
        config.try_deserialize()
    }
}
```

### 4.14 Criterios de Completitud

#### 4.14.1 Entregables de la Fase
- [ ] API REST completa con todos los endpoints
- [ ] Sistema de autenticación JWT implementado
- [ ] Autorización basada en roles funcionando
- [ ] Rate limiting configurado
- [ ] Documentación OpenAPI generada
- [ ] Middleware stack completo
- [ ] Tests de integración pasando
- [ ] Health checks implementados
- [ ] Error handling estandarizado
- [ ] Security headers configurados

#### 4.14.2 Criterios de Aceptación
- [ ] Registro y login de usuarios funciona
- [ ] JWT tokens se generan y validan correctamente
- [ ] Endpoints protegidos requieren autenticación
- [ ] Rate limiting previene abuso
- [ ] Documentación API accesible en /docs
- [ ] Responses tienen formato consistente
- [ ] Health checks reportan estado correcto
- [ ] Performance acceptable (< 100ms para endpoints simples)
- [ ] Security headers presentes en todas las responses
- [ ] Tests de integración cubren casos principales

### 4.15 Security Checklist

#### 4.15.1 OWASP Top 10 Compliance
- [ ] **A01 - Broken Access Control**: Autorización implementada
- [ ] **A02 - Cryptographic Failures**: HTTPS obligatorio, JWT seguro
- [ ] **A03 - Injection**: Queries parametrizadas, validación de input
- [ ] **A04 - Insecure Design**: Arquitectura segura por diseño
- [ ] **A05 - Security Misconfiguration**: Headers de seguridad
- [ ] **A06 - Vulnerable Components**: Dependencias actualizadas
- [ ] **A07 - Identity Failures**: Autenticación robusta
- [ ] **A08 - Software Integrity**: Checksums y validación
- [ ] **A09 - Logging Failures**: Logging de seguridad
- [ ] **A10 - SSRF**: Validación de URLs externas

### 4.16 Estimación de Tiempo

#### 4.16.1 Breakdown de Tareas
- Configuración de Axum y estructura básica: 3 días
- Sistema de autenticación JWT: 4 días
- Autorización y roles: 3 días
- Endpoints de autenticación: 2 días
- Endpoints de proyectos: 4 días
- Endpoints de análisis: 3 días
- Rate limiting y middleware: 3 días
- Documentación OpenAPI: 2 días
- Error handling y validación: 3 días
- Health checks y monitoring: 2 días
- Security headers y hardening: 2 días
- Testing de integración: 4 días
- Documentación: 2 días

**Total estimado: 37 días de desarrollo**

### 4.17 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- API REST completa y documentada
- Sistema de autenticación y autorización robusto
- Endpoints para todas las operaciones principales
- Rate limiting y protecciones de seguridad
- Monitoring y health checks
- Testing comprehensivo

La Fase 5 construirá sobre esta base implementando el sistema de logging, métricas y monitoreo avanzado.
