# Fase 5: Sistema de Logging, Métricas y Monitoreo Básico

## Objetivo General
Implementar un sistema completo de observabilidad que incluya logging estructurado, métricas de performance, monitoreo de salud del sistema, alertas automáticas y dashboards para garantizar la operación confiable y el debugging eficiente del agente CodeAnt.

## Descripción Técnica Detallada

### 5.1 Arquitectura de Observabilidad

#### 5.1.1 Stack de Observabilidad
```
┌─────────────────────────────────────────┐
│            Observability Stack         │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Tracing   │ │      Metrics        │ │
│  │  (Jaeger)   │ │   (Prometheus)      │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Logging   │ │    Dashboards       │ │
│  │(OpenSearch) │ │     (Grafana)       │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│           Application Layer             │
└─────────────────────────────────────────┘
```

#### 5.1.2 Principios de Observabilidad
- **Three Pillars**: Logs, Metrics, Traces
- **High Cardinality**: Soporte para dimensiones múltiples
- **Real-time**: Monitoreo en tiempo real
- **Structured**: Datos estructurados y searchables
- **Correlation**: Correlación entre logs, metrics y traces
- **Retention**: Políticas de retención configurables

### 5.2 Sistema de Logging Estructurado

#### 5.2.1 Tracing Framework Implementation
```rust
use tracing::{info, warn, error, debug, trace, instrument, Span};
use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    fmt,
    EnvFilter,
};
use tracing_opentelemetry::OpenTelemetryLayer;

pub struct LoggingSystem {
    config: LoggingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: LogFormat,
    pub output: LogOutput,
    pub structured: bool,
    pub include_location: bool,
    pub include_spans: bool,
    pub max_level: String,
    pub retention_days: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub enum LogFormat {
    Json,
    Pretty,
    Compact,
}

#[derive(Debug, Clone, Deserialize)]
pub enum LogOutput {
    Stdout,
    File(String),
    OpenSearch(OpenSearchConfig),
    Multiple(Vec<LogOutput>),
}

impl LoggingSystem {
    pub fn init(config: LoggingConfig) -> Result<(), LoggingError> {
        let env_filter = EnvFilter::try_from_default_env()
            .or_else(|_| EnvFilter::try_new(&config.level))?;
        
        let fmt_layer = match config.format {
            LogFormat::Json => fmt::layer()
                .json()
                .with_current_span(config.include_spans)
                .with_file(config.include_location)
                .with_line_number(config.include_location)
                .boxed(),
            LogFormat::Pretty => fmt::layer()
                .pretty()
                .with_current_span(config.include_spans)
                .with_file(config.include_location)
                .with_line_number(config.include_location)
                .boxed(),
            LogFormat::Compact => fmt::layer()
                .compact()
                .with_current_span(config.include_spans)
                .boxed(),
        };
        
        // OpenTelemetry layer for distributed tracing
        let tracer = opentelemetry_jaeger::new_agent_pipeline()
            .with_service_name("codeant-agent")
            .install_simple()?;
        let telemetry_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .with(telemetry_layer)
            .init();
            
        Ok(())
    }
}
```

#### 5.2.2 Structured Logging Macros
```rust
use serde::Serialize;

#[derive(Serialize)]
pub struct LogContext {
    pub request_id: String,
    pub user_id: Option<String>,
    pub organization_id: Option<String>,
    pub project_id: Option<String>,
    pub operation: String,
    pub duration_ms: Option<u64>,
    pub error_code: Option<String>,
}

// Custom logging macros with context
macro_rules! log_info {
    ($ctx:expr, $msg:expr, $($key:expr => $value:expr),*) => {
        tracing::info!(
            request_id = %$ctx.request_id,
            user_id = ?$ctx.user_id,
            organization_id = ?$ctx.organization_id,
            project_id = ?$ctx.project_id,
            operation = %$ctx.operation,
            $($key = %$value,)*
            $msg
        );
    };
}

macro_rules! log_error {
    ($ctx:expr, $error:expr, $msg:expr, $($key:expr => $value:expr),*) => {
        tracing::error!(
            request_id = %$ctx.request_id,
            user_id = ?$ctx.user_id,
            organization_id = ?$ctx.organization_id,
            project_id = ?$ctx.project_id,
            operation = %$ctx.operation,
            error = %$error,
            error_type = std::any::type_name_of_val(&$error),
            $($key = %$value,)*
            $msg
        );
    };
}

// Usage example
#[instrument(skip(ctx, service))]
pub async fn create_project(
    ctx: LogContext,
    service: &ProjectService,
    request: CreateProjectRequest,
) -> Result<Project, ProjectError> {
    let start = Instant::now();
    
    log_info!(ctx, "Creating new project", 
        "project_name" => request.name,
        "repository_url" => request.repository_url
    );
    
    match service.create(request).await {
        Ok(project) => {
            log_info!(ctx, "Project created successfully",
                "project_id" => project.id,
                "duration_ms" => start.elapsed().as_millis()
            );
            Ok(project)
        }
        Err(error) => {
            log_error!(ctx, error, "Failed to create project",
                "duration_ms" => start.elapsed().as_millis()
            );
            Err(error)
        }
    }
}
```

#### 5.2.3 Request Correlation
```rust
use uuid::Uuid;

pub struct CorrelationMiddleware;

impl CorrelationMiddleware {
    pub async fn layer(
        request: Request,
        next: Next,
    ) -> Response {
        let request_id = request
            .headers()
            .get("X-Request-ID")
            .and_then(|v| v.to_str().ok())
            .unwrap_or(&Uuid::new_v4().to_string())
            .to_string();
        
        let trace_id = request
            .headers()
            .get("X-Trace-ID")
            .and_then(|v| v.to_str().ok())
            .unwrap_or(&Uuid::new_v4().to_string())
            .to_string();
        
        // Create span for this request
        let span = tracing::info_span!(
            "http_request",
            request_id = %request_id,
            trace_id = %trace_id,
            method = %request.method(),
            uri = %request.uri(),
            version = ?request.version()
        );
        
        let response = async move {
            // Add correlation IDs to request extensions
            request.extensions_mut().insert(RequestId(request_id.clone()));
            request.extensions_mut().insert(TraceId(trace_id.clone()));
            
            let start = Instant::now();
            let response = next.run(request).await;
            let duration = start.elapsed();
            
            tracing::info!(
                status = %response.status(),
                duration_ms = %duration.as_millis(),
                "Request completed"
            );
            
            response
        }
        .instrument(span)
        .await;
        
        // Add correlation headers to response
        let mut response = response;
        response.headers_mut().insert(
            "X-Request-ID",
            HeaderValue::from_str(&request_id).unwrap(),
        );
        response.headers_mut().insert(
            "X-Trace-ID",
            HeaderValue::from_str(&trace_id).unwrap(),
        );
        
        response
    }
}
```

### 5.3 Sistema de Métricas

#### 5.3.1 Prometheus Metrics Implementation
```rust
use prometheus::{
    Counter, Histogram, Gauge, IntCounter, IntGauge,
    register_counter, register_histogram, register_gauge,
    register_int_counter, register_int_gauge,
    HistogramOpts, Opts,
};
use lazy_static::lazy_static;

lazy_static! {
    // HTTP Metrics
    pub static ref HTTP_REQUESTS_TOTAL: Counter = register_counter!(
        "http_requests_total",
        "Total number of HTTP requests"
    ).unwrap();
    
    pub static ref HTTP_REQUEST_DURATION: Histogram = register_histogram!(
        HistogramOpts::new(
            "http_request_duration_seconds",
            "HTTP request duration in seconds"
        ).buckets(vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    ).unwrap();
    
    pub static ref HTTP_REQUESTS_IN_FLIGHT: IntGauge = register_int_gauge!(
        "http_requests_in_flight",
        "Number of HTTP requests currently being processed"
    ).unwrap();
    
    // Database Metrics
    pub static ref DATABASE_CONNECTIONS_ACTIVE: IntGauge = register_int_gauge!(
        "database_connections_active",
        "Number of active database connections"
    ).unwrap();
    
    pub static ref DATABASE_QUERIES_TOTAL: Counter = register_counter!(
        "database_queries_total",
        "Total number of database queries"
    ).unwrap();
    
    pub static ref DATABASE_QUERY_DURATION: Histogram = register_histogram!(
        HistogramOpts::new(
            "database_query_duration_seconds",
            "Database query duration in seconds"
        ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
    ).unwrap();
    
    // Analysis Metrics
    pub static ref ANALYSIS_JOBS_TOTAL: Counter = register_counter!(
        "analysis_jobs_total",
        "Total number of analysis jobs"
    ).unwrap();
    
    pub static ref ANALYSIS_JOB_DURATION: Histogram = register_histogram!(
        HistogramOpts::new(
            "analysis_job_duration_seconds",
            "Analysis job duration in seconds"
        ).buckets(vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0])
    ).unwrap();
    
    pub static ref ANALYSIS_JOBS_QUEUED: IntGauge = register_int_gauge!(
        "analysis_jobs_queued",
        "Number of analysis jobs in queue"
    ).unwrap();
    
    // Business Metrics
    pub static ref PROJECTS_TOTAL: IntGauge = register_int_gauge!(
        "projects_total",
        "Total number of projects"
    ).unwrap();
    
    pub static ref USERS_ACTIVE: IntGauge = register_int_gauge!(
        "users_active_total",
        "Number of active users"
    ).unwrap();
    
    pub static ref CODE_ISSUES_DETECTED: Counter = register_counter!(
        "code_issues_detected_total",
        "Total number of code issues detected"
    ).unwrap();
}
```

#### 5.3.2 Custom Metrics Collector
```rust
pub struct MetricsCollector {
    registry: Arc<prometheus::Registry>,
    custom_metrics: HashMap<String, Box<dyn CustomMetric>>,
}

pub trait CustomMetric: Send + Sync {
    fn collect(&self) -> Result<Vec<prometheus::proto::MetricFamily>, MetricsError>;
    fn name(&self) -> &str;
    fn help(&self) -> &str;
}

pub struct SystemMetricsCollector {
    system: sysinfo::System,
}

impl CustomMetric for SystemMetricsCollector {
    fn collect(&self) -> Result<Vec<prometheus::proto::MetricFamily>, MetricsError> {
        let mut system = self.system.clone();
        system.refresh_all();
        
        let mut metrics = Vec::new();
        
        // CPU Usage
        let cpu_usage = system.global_cpu_info().cpu_usage() as f64 / 100.0;
        let cpu_metric = create_gauge_metric(
            "system_cpu_usage_ratio",
            "System CPU usage ratio",
            cpu_usage,
        );
        metrics.push(cpu_metric);
        
        // Memory Usage
        let memory_used = system.used_memory() as f64;
        let memory_total = system.total_memory() as f64;
        let memory_metric = create_gauge_metric(
            "system_memory_usage_bytes",
            "System memory usage in bytes",
            memory_used,
        );
        metrics.push(memory_metric);
        
        // Disk Usage
        for disk in system.disks() {
            let disk_usage = create_gauge_metric_with_labels(
                "system_disk_usage_bytes",
                "System disk usage in bytes",
                disk.total_space() - disk.available_space(),
                vec![("mount_point", disk.mount_point().to_string_lossy().to_string())],
            );
            metrics.push(disk_usage);
        }
        
        Ok(metrics)
    }
    
    fn name(&self) -> &str {
        "system_metrics"
    }
    
    fn help(&self) -> &str {
        "System resource usage metrics"
    }
}
```

#### 5.3.3 Application Metrics Middleware
```rust
pub struct MetricsMiddleware;

impl MetricsMiddleware {
    pub async fn layer(
        request: Request,
        next: Next,
    ) -> Response {
        let start = Instant::now();
        let method = request.method().clone();
        let path = request.uri().path().to_string();
        
        // Increment in-flight requests
        HTTP_REQUESTS_IN_FLIGHT.inc();
        
        let response = next.run(request).await;
        let duration = start.elapsed();
        let status = response.status().as_u16();
        
        // Record metrics
        HTTP_REQUESTS_TOTAL
            .with_label_values(&[&method.to_string(), &path, &status.to_string()])
            .inc();
            
        HTTP_REQUEST_DURATION
            .with_label_values(&[&method.to_string(), &path])
            .observe(duration.as_secs_f64());
        
        // Decrement in-flight requests
        HTTP_REQUESTS_IN_FLIGHT.dec();
        
        response
    }
}
```

### 5.4 Distributed Tracing

#### 5.4.1 OpenTelemetry Integration
```rust
use opentelemetry::{
    global,
    sdk::{trace as sdktrace, Resource},
    trace::{TraceError, Tracer},
    KeyValue,
};
use opentelemetry_jaeger as jaeger;

pub struct TracingSystem {
    tracer: Box<dyn Tracer + Send + Sync>,
}

impl TracingSystem {
    pub fn init(config: TracingConfig) -> Result<Self, TraceError> {
        let tracer = jaeger::new_agent_pipeline()
            .with_service_name("codeant-agent")
            .with_service_version(env!("CARGO_PKG_VERSION"))
            .with_trace_config(
                sdktrace::config().with_resource(Resource::new(vec![
                    KeyValue::new("service.name", "codeant-agent"),
                    KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                    KeyValue::new("deployment.environment", config.environment),
                ]))
            )
            .install_batch(opentelemetry::runtime::Tokio)?;
            
        global::set_text_map_propagator(jaeger::Propagator::new());
        
        Ok(Self {
            tracer: Box::new(tracer),
        })
    }
}
```

#### 5.4.2 Span Creation and Context Propagation
```rust
use opentelemetry::{
    trace::{Span, SpanKind, Status, TraceContextExt, Tracer},
    Context, KeyValue,
};

pub struct SpanBuilder {
    tracer: Arc<dyn Tracer + Send + Sync>,
}

impl SpanBuilder {
    pub fn create_span(&self, name: &str, operation_type: &str) -> SpanGuard {
        let mut span = self.tracer
            .span_builder(name)
            .with_kind(SpanKind::Server)
            .with_attributes(vec![
                KeyValue::new("operation.type", operation_type.to_string()),
                KeyValue::new("service.name", "codeant-agent"),
            ])
            .start(&self.tracer);
            
        SpanGuard::new(span)
    }
    
    pub fn create_child_span(&self, parent_ctx: &Context, name: &str) -> SpanGuard {
        let span = self.tracer
            .span_builder(name)
            .with_kind(SpanKind::Internal)
            .start_with_context(&self.tracer, parent_ctx);
            
        SpanGuard::new(span)
    }
}

pub struct SpanGuard {
    span: Box<dyn Span + Send + Sync>,
    _guard: tracing::span::Entered,
}

impl SpanGuard {
    fn new(span: Box<dyn Span + Send + Sync>) -> Self {
        let tracing_span = tracing::info_span!(
            "operation",
            trace_id = %span.span_context().trace_id(),
            span_id = %span.span_context().span_id()
        );
        let guard = tracing_span.entered();
        
        Self {
            span,
            _guard: guard,
        }
    }
    
    pub fn add_event(&mut self, name: &str, attributes: Vec<KeyValue>) {
        self.span.add_event(name, attributes);
    }
    
    pub fn set_status(&mut self, status: Status) {
        self.span.set_status(status);
    }
    
    pub fn record_error(&mut self, error: &dyn std::error::Error) {
        self.span.record_error(error);
        self.span.set_status(Status::error(error.to_string()));
    }
}
```

### 5.5 Health Monitoring

#### 5.5.1 Comprehensive Health Checks
```rust
use async_trait::async_trait;

#[async_trait]
pub trait HealthCheck: Send + Sync {
    async fn check(&self) -> HealthCheckResult;
    fn name(&self) -> &str;
    fn timeout(&self) -> Duration;
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthCheckResult {
    pub status: HealthStatus,
    pub message: String,
    pub duration_ms: u64,
    pub details: Option<serde_json::Value>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

pub struct DatabaseHealthCheck {
    pool: Arc<DatabasePool>,
}

#[async_trait]
impl HealthCheck for DatabaseHealthCheck {
    async fn check(&self) -> HealthCheckResult {
        let start = Instant::now();
        
        match timeout(self.timeout(), self.perform_check()).await {
            Ok(Ok(_)) => HealthCheckResult {
                status: HealthStatus::Healthy,
                message: "Database connection successful".to_string(),
                duration_ms: start.elapsed().as_millis() as u64,
                details: Some(serde_json::json!({
                    "pool_size": self.pool.size(),
                    "active_connections": self.pool.num_idle()
                })),
                timestamp: Utc::now(),
            },
            Ok(Err(e)) => HealthCheckResult {
                status: HealthStatus::Unhealthy,
                message: format!("Database check failed: {}", e),
                duration_ms: start.elapsed().as_millis() as u64,
                details: None,
                timestamp: Utc::now(),
            },
            Err(_) => HealthCheckResult {
                status: HealthStatus::Unhealthy,
                message: "Database check timed out".to_string(),
                duration_ms: self.timeout().as_millis() as u64,
                details: None,
                timestamp: Utc::now(),
            },
        }
    }
    
    fn name(&self) -> &str {
        "database"
    }
    
    fn timeout(&self) -> Duration {
        Duration::from_secs(5)
    }
}

impl DatabaseHealthCheck {
    async fn perform_check(&self) -> Result<(), DatabaseError> {
        sqlx::query!("SELECT 1 as health_check")
            .fetch_one(self.pool.get())
            .await?;
        Ok(())
    }
}
```

#### 5.5.2 Health Monitoring Service
```rust
pub struct HealthMonitor {
    checks: Vec<Box<dyn HealthCheck>>,
    cache: Arc<RwLock<HashMap<String, HealthCheckResult>>>,
    config: HealthMonitorConfig,
}

#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    pub check_interval: Duration,
    pub cache_ttl: Duration,
    pub parallel_execution: bool,
    pub fail_fast: bool,
}

impl HealthMonitor {
    pub fn new(config: HealthMonitorConfig) -> Self {
        Self {
            checks: Vec::new(),
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }
    
    pub fn add_check(&mut self, check: Box<dyn HealthCheck>) {
        self.checks.push(check);
    }
    
    pub async fn check_all(&self) -> HealthSummary {
        let start = Instant::now();
        let results = if self.config.parallel_execution {
            self.run_checks_parallel().await
        } else {
            self.run_checks_sequential().await
        };
        
        let overall_status = self.determine_overall_status(&results);
        
        HealthSummary {
            status: overall_status,
            checks: results,
            duration_ms: start.elapsed().as_millis() as u64,
            timestamp: Utc::now(),
        }
    }
    
    async fn run_checks_parallel(&self) -> HashMap<String, HealthCheckResult> {
        let futures: Vec<_> = self.checks
            .iter()
            .map(|check| async move {
                let name = check.name().to_string();
                let result = check.check().await;
                (name, result)
            })
            .collect();
            
        let results = futures::future::join_all(futures).await;
        results.into_iter().collect()
    }
    
    fn determine_overall_status(&self, results: &HashMap<String, HealthCheckResult>) -> HealthStatus {
        let statuses: Vec<_> = results.values().map(|r| &r.status).collect();
        
        if statuses.iter().all(|s| **s == HealthStatus::Healthy) {
            HealthStatus::Healthy
        } else if statuses.iter().any(|s| **s == HealthStatus::Unhealthy) {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        }
    }
}

#[derive(Debug, Serialize)]
pub struct HealthSummary {
    pub status: HealthStatus,
    pub checks: HashMap<String, HealthCheckResult>,
    pub duration_ms: u64,
    pub timestamp: DateTime<Utc>,
}
```

### 5.6 Alerting System

#### 5.6.1 Alert Manager
```rust
use tokio::sync::broadcast;

pub struct AlertManager {
    rules: Vec<AlertRule>,
    channels: Vec<Box<dyn AlertChannel>>,
    sender: broadcast::Sender<Alert>,
    config: AlertConfig,
}

#[derive(Debug, Clone)]
pub struct AlertRule {
    pub name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub cooldown: Duration,
    pub message_template: String,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub enum AlertCondition {
    MetricThreshold {
        metric: String,
        operator: ComparisonOperator,
        threshold: f64,
        duration: Duration,
    },
    HealthCheckFailed {
        check_name: String,
        consecutive_failures: u32,
    },
    ErrorRate {
        threshold_percent: f64,
        window: Duration,
    },
    Custom(Box<dyn Fn(&MetricsSnapshot) -> bool + Send + Sync>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[async_trait]
pub trait AlertChannel: Send + Sync {
    async fn send_alert(&self, alert: &Alert) -> Result<(), AlertError>;
    fn name(&self) -> &str;
    fn supports_severity(&self, severity: &AlertSeverity) -> bool;
}

pub struct SlackAlertChannel {
    webhook_url: String,
    channel: String,
    username: String,
}

#[async_trait]
impl AlertChannel for SlackAlertChannel {
    async fn send_alert(&self, alert: &Alert) -> Result<(), AlertError> {
        let payload = serde_json::json!({
            "channel": self.channel,
            "username": self.username,
            "attachments": [{
                "color": match alert.severity {
                    AlertSeverity::Critical => "danger",
                    AlertSeverity::High => "warning",
                    AlertSeverity::Medium => "good",
                    AlertSeverity::Low => "#439FE0",
                    AlertSeverity::Info => "#439FE0",
                },
                "title": format!("[{}] {}", alert.severity, alert.title),
                "text": alert.message,
                "fields": [
                    {
                        "title": "Service",
                        "value": "CodeAnt Agent",
                        "short": true
                    },
                    {
                        "title": "Timestamp",
                        "value": alert.timestamp.to_rfc3339(),
                        "short": true
                    }
                ]
            }]
        });
        
        let client = reqwest::Client::new();
        let response = client
            .post(&self.webhook_url)
            .json(&payload)
            .send()
            .await?;
            
        if response.status().is_success() {
            Ok(())
        } else {
            Err(AlertError::SendFailed(format!(
                "Slack API returned: {}",
                response.status()
            )))
        }
    }
    
    fn name(&self) -> &str {
        "slack"
    }
    
    fn supports_severity(&self, severity: &AlertSeverity) -> bool {
        matches!(severity, AlertSeverity::Critical | AlertSeverity::High)
    }
}
```

#### 5.6.2 Alert Evaluation Engine
```rust
pub struct AlertEvaluator {
    rules: Vec<AlertRule>,
    state: Arc<RwLock<HashMap<String, AlertState>>>,
    metrics_provider: Arc<dyn MetricsProvider>,
}

#[derive(Debug, Clone)]
struct AlertState {
    last_triggered: Option<DateTime<Utc>>,
    consecutive_failures: u32,
    is_firing: bool,
    cooldown_until: Option<DateTime<Utc>>,
}

impl AlertEvaluator {
    pub async fn evaluate_all(&self) -> Vec<Alert> {
        let mut alerts = Vec::new();
        let metrics = self.metrics_provider.get_current_snapshot().await;
        
        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }
            
            let should_fire = self.evaluate_rule(rule, &metrics).await;
            let mut state = self.state.write().await;
            let alert_state = state.entry(rule.name.clone())
                .or_insert_with(|| AlertState {
                    last_triggered: None,
                    consecutive_failures: 0,
                    is_firing: false,
                    cooldown_until: None,
                });
            
            if should_fire && !alert_state.is_firing {
                if let Some(cooldown_until) = alert_state.cooldown_until {
                    if Utc::now() < cooldown_until {
                        continue; // Still in cooldown
                    }
                }
                
                // Fire alert
                let alert = Alert {
                    id: Uuid::new_v4(),
                    rule_name: rule.name.clone(),
                    severity: rule.severity.clone(),
                    title: self.render_title(rule, &metrics),
                    message: self.render_message(rule, &metrics),
                    timestamp: Utc::now(),
                    metadata: self.collect_metadata(rule, &metrics),
                };
                
                alert_state.is_firing = true;
                alert_state.last_triggered = Some(Utc::now());
                alert_state.cooldown_until = Some(Utc::now() + rule.cooldown);
                
                alerts.push(alert);
            } else if !should_fire && alert_state.is_firing {
                // Resolve alert
                alert_state.is_firing = false;
                alert_state.consecutive_failures = 0;
            }
        }
        
        alerts
    }
    
    async fn evaluate_rule(&self, rule: &AlertRule, metrics: &MetricsSnapshot) -> bool {
        match &rule.condition {
            AlertCondition::MetricThreshold { metric, operator, threshold, .. } => {
                if let Some(value) = metrics.get_metric(metric) {
                    operator.compare(value, *threshold)
                } else {
                    false
                }
            }
            AlertCondition::HealthCheckFailed { check_name, consecutive_failures } => {
                if let Some(health_result) = metrics.get_health_check(check_name) {
                    health_result.status == HealthStatus::Unhealthy &&
                    health_result.consecutive_failures >= *consecutive_failures
                } else {
                    false
                }
            }
            AlertCondition::ErrorRate { threshold_percent, window } => {
                let error_rate = metrics.get_error_rate(*window);
                error_rate > *threshold_percent
            }
            AlertCondition::Custom(evaluator) => {
                evaluator(metrics)
            }
        }
    }
}
```

### 5.7 Performance Monitoring

#### 5.7.1 Performance Profiler
```rust
use std::collections::VecDeque;

pub struct PerformanceProfiler {
    samples: Arc<RwLock<VecDeque<PerformanceSample>>>,
    config: ProfilerConfig,
}

#[derive(Debug, Clone)]
pub struct PerformanceSample {
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: f64,
    pub memory_usage: u64,
    pub heap_size: u64,
    pub gc_time_ms: u64,
    pub active_threads: u32,
    pub database_connections: u32,
    pub request_latency_p95: f64,
    pub request_latency_p99: f64,
}

#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    pub sample_interval: Duration,
    pub max_samples: usize,
    pub enable_gc_profiling: bool,
    pub enable_memory_profiling: bool,
    pub enable_cpu_profiling: bool,
}

impl PerformanceProfiler {
    pub async fn start_profiling(&self) -> Result<(), ProfilerError> {
        let mut interval = tokio::time::interval(self.config.sample_interval);
        
        loop {
            interval.tick().await;
            
            let sample = self.collect_sample().await?;
            
            let mut samples = self.samples.write().await;
            if samples.len() >= self.config.max_samples {
                samples.pop_front();
            }
            samples.push_back(sample);
        }
    }
    
    async fn collect_sample(&self) -> Result<PerformanceSample, ProfilerError> {
        let mut system = sysinfo::System::new_all();
        system.refresh_all();
        
        Ok(PerformanceSample {
            timestamp: Utc::now(),
            cpu_usage: system.global_cpu_info().cpu_usage() as f64,
            memory_usage: system.used_memory(),
            heap_size: self.get_heap_size()?,
            gc_time_ms: self.get_gc_time()?,
            active_threads: system.processes().len() as u32,
            database_connections: self.get_db_connections().await?,
            request_latency_p95: self.get_request_latency_percentile(0.95).await?,
            request_latency_p99: self.get_request_latency_percentile(0.99).await?,
        })
    }
    
    pub async fn get_performance_report(&self, duration: Duration) -> PerformanceReport {
        let samples = self.samples.read().await;
        let cutoff = Utc::now() - duration;
        
        let recent_samples: Vec<_> = samples
            .iter()
            .filter(|s| s.timestamp > cutoff)
            .cloned()
            .collect();
            
        PerformanceReport::from_samples(recent_samples)
    }
}
```

### 5.8 Log Aggregation y Analysis

#### 5.8.1 OpenSearch Integration
```rust
use opensearch::{OpenSearch, SearchParts, IndexParts};

pub struct LogAggregator {
    client: OpenSearch,
    index_pattern: String,
    config: LogAggregatorConfig,
}

#[derive(Debug, Clone)]
pub struct LogAggregatorConfig {
    pub batch_size: usize,
    pub flush_interval: Duration,
    pub retention_days: u32,
    pub index_template: String,
    pub mapping: serde_json::Value,
}

impl LogAggregator {
    pub async fn index_log_entry(&self, entry: LogEntry) -> Result<(), LogError> {
        let doc = serde_json::to_value(entry)?;
        
        self.client
            .index(IndexParts::IndexId(&self.index_pattern, &entry.id))
            .body(doc)
            .send()
            .await?;
            
        Ok(())
    }
    
    pub async fn search_logs(&self, query: LogSearchQuery) -> Result<LogSearchResult, LogError> {
        let search_body = serde_json::json!({
            "query": {
                "bool": {
                    "must": self.build_must_clauses(&query),
                    "filter": self.build_filter_clauses(&query)
                }
            },
            "sort": [
                { "@timestamp": { "order": "desc" } }
            ],
            "size": query.size.unwrap_or(100),
            "from": query.from.unwrap_or(0)
        });
        
        let response = self.client
            .search(SearchParts::Index(&[&self.index_pattern]))
            .body(search_body)
            .send()
            .await?;
            
        let search_result: opensearch::SearchResponse<LogEntry> = response.json().await?;
        
        Ok(LogSearchResult {
            total_hits: search_result.hits.total.value,
            logs: search_result.hits.hits.into_iter()
                .map(|hit| hit.source)
                .collect(),
        })
    }
    
    pub async fn create_log_dashboard(&self, dashboard_config: DashboardConfig) -> Result<Dashboard, LogError> {
        let visualizations = vec![
            self.create_log_level_chart().await?,
            self.create_error_trend_chart().await?,
            self.create_response_time_histogram().await?,
            self.create_top_errors_table().await?,
        ];
        
        Ok(Dashboard {
            id: Uuid::new_v4(),
            name: dashboard_config.name,
            visualizations,
            refresh_interval: dashboard_config.refresh_interval,
            time_range: dashboard_config.time_range,
        })
    }
}
```

### 5.9 Dashboards y Visualización

#### 5.9.1 Grafana Dashboard Configuration
```rust
use serde_json::json;

pub struct DashboardBuilder {
    panels: Vec<Panel>,
    config: DashboardConfig,
}

impl DashboardBuilder {
    pub fn create_main_dashboard() -> serde_json::Value {
        json!({
            "dashboard": {
                "id": null,
                "title": "CodeAnt Agent - Main Dashboard",
                "tags": ["codeant", "monitoring"],
                "timezone": "browser",
                "refresh": "30s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "HTTP Request Rate",
                        "type": "stat",
                        "targets": [{
                            "expr": "rate(http_requests_total[5m])",
                            "legendFormat": "{{method}} {{path}}"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "reqps",
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": null},
                                        {"color": "yellow", "value": 100},
                                        {"color": "red", "value": 500}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Response Time",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "95th percentile"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "s",
                                "custom": {
                                    "drawStyle": "line",
                                    "fillOpacity": 10
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Database Connections",
                        "type": "gauge",
                        "targets": [{
                            "expr": "database_connections_active"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "min": 0,
                                "max": 100,
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": null},
                                        {"color": "yellow", "value": 70},
                                        {"color": "red", "value": 90}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "Analysis Jobs Queue",
                        "type": "stat",
                        "targets": [{
                            "expr": "analysis_jobs_queued"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "short",
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": null},
                                        {"color": "yellow", "value": 10},
                                        {"color": "red", "value": 50}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 8}
                    },
                    {
                        "id": 5,
                        "title": "System Resources",
                        "type": "timeseries",
                        "targets": [
                            {
                                "expr": "system_cpu_usage_ratio * 100",
                                "legendFormat": "CPU Usage %"
                            },
                            {
                                "expr": "system_memory_usage_bytes / system_memory_total_bytes * 100",
                                "legendFormat": "Memory Usage %"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percent",
                                "max": 100,
                                "min": 0
                            }
                        },
                        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 8}
                    }
                ]
            }
        })
    }
    
    pub fn create_analysis_dashboard() -> serde_json::Value {
        json!({
            "dashboard": {
                "title": "CodeAnt Agent - Analysis Dashboard",
                "panels": [
                    {
                        "title": "Analysis Jobs by Status",
                        "type": "piechart",
                        "targets": [{
                            "expr": "sum by (status) (analysis_jobs_total)"
                        }]
                    },
                    {
                        "title": "Analysis Duration Distribution",
                        "type": "histogram",
                        "targets": [{
                            "expr": "histogram_quantile(0.50, rate(analysis_job_duration_seconds_bucket[5m]))",
                            "legendFormat": "50th percentile"
                        }]
                    },
                    {
                        "title": "Issues Detected by Severity",
                        "type": "bargraph",
                        "targets": [{
                            "expr": "sum by (severity) (code_issues_detected_total)"
                        }]
                    }
                ]
            }
        })
    }
}
```

### 5.10 Configuration Management

#### 5.10.1 Observability Configuration
```rust
#[derive(Debug, Clone, Deserialize)]
pub struct ObservabilityConfig {
    pub logging: LoggingConfig,
    pub metrics: MetricsConfig,
    pub tracing: TracingConfig,
    pub health_checks: HealthCheckConfig,
    pub alerting: AlertingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub push_interval: Duration,
    pub custom_metrics: Vec<CustomMetricConfig>,
    pub retention: Duration,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AlertingConfig {
    pub enabled: bool,
    pub rules_file: PathBuf,
    pub channels: Vec<AlertChannelConfig>,
    pub evaluation_interval: Duration,
    pub cooldown_period: Duration,
}

impl ObservabilityConfig {
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
    
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.logging.enabled && self.logging.level.is_empty() {
            return Err(ConfigError::InvalidConfig("Logging level cannot be empty".to_string()));
        }
        
        if self.metrics.enabled && self.metrics.endpoint.is_empty() {
            return Err(ConfigError::InvalidConfig("Metrics endpoint cannot be empty".to_string()));
        }
        
        if self.alerting.enabled && self.alerting.channels.is_empty() {
            return Err(ConfigError::InvalidConfig("At least one alert channel must be configured".to_string()));
        }
        
        Ok(())
    }
}
```

### 5.11 Criterios de Completitud

#### 5.11.1 Entregables de la Fase
- [ ] Sistema de logging estructurado implementado
- [ ] Métricas Prometheus configuradas y funcionando
- [ ] Distributed tracing con OpenTelemetry/Jaeger
- [ ] Health checks comprehensivos
- [ ] Sistema de alertas automatizado
- [ ] Dashboards Grafana configurados
- [ ] Log aggregation con OpenSearch
- [ ] Performance profiling
- [ ] Configuration management
- [ ] Tests de monitoreo

#### 5.11.2 Criterios de Aceptación
- [ ] Logs estructurados se generan correctamente
- [ ] Métricas se exponen en /metrics endpoint
- [ ] Traces se envían a Jaeger correctamente
- [ ] Health checks reportan estado accurate
- [ ] Alertas se disparan según condiciones
- [ ] Dashboards muestran métricas en tiempo real
- [ ] Logs son searchables en OpenSearch
- [ ] Performance profiling detecta bottlenecks
- [ ] Correlation entre logs, metrics y traces
- [ ] Zero downtime durante monitoring

### 5.12 Performance Targets

#### 5.12.1 Latency Targets
- **Logging overhead**: < 1ms per log entry
- **Metrics collection**: < 0.1ms per metric
- **Health check response**: < 100ms
- **Alert evaluation**: < 5 seconds
- **Dashboard refresh**: < 2 seconds
- **Log search**: < 500ms for typical queries

#### 5.12.2 Throughput Targets
- **Log ingestion**: 10,000+ entries/second
- **Metrics ingestion**: 1,000+ metrics/second
- **Concurrent health checks**: 100+ simultaneous
- **Alert processing**: 100+ alerts/minute

### 5.13 Estimación de Tiempo

#### 5.13.1 Breakdown de Tareas
- Sistema de logging estructurado: 4 días
- Métricas Prometheus: 3 días
- Distributed tracing: 4 días
- Health checks: 3 días
- Sistema de alertas: 5 días
- Dashboards Grafana: 4 días
- Log aggregation: 4 días
- Performance profiling: 3 días
- Configuration management: 2 días
- Testing e integración: 4 días
- Documentación: 2 días

**Total estimado: 38 días de desarrollo**

### 5.14 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Observabilidad completa (logs, metrics, traces)
- Monitoring proactivo con alertas
- Dashboards para visualización en tiempo real
- Herramientas de debugging y troubleshooting
- Performance monitoring y profiling
- Foundation sólida para operaciones

Las siguientes fases (6-10) se enfocarán en implementar los parsers especializados y el análisis sintáctico, construyendo sobre esta base de observabilidad robusta.
