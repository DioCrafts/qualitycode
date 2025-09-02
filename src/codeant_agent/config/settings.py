"""
Sistema de configuración centralizado para CodeAnt Agent.

Este módulo implementa la configuración por capas:
1. Valores por defecto
2. Archivos de configuración (TOML/YAML)
3. Variables de entorno
4. Argumentos de línea de comandos
"""

from typing import Optional, List
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Configuración de la base de datos."""
    
    url: str = Field(
        default="postgresql://codeant:dev_password@localhost:5432/codeant_dev",
        description="URL de conexión a la base de datos PostgreSQL"
    )
    max_connections: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Número máximo de conexiones en el pool"
    )
    min_connections: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Número mínimo de conexiones en el pool"
    )
    echo: bool = Field(
        default=False,
        description="Habilitar logging de SQL queries"
    )
    
    @validator('url')
    def validate_database_url(cls, v: str) -> str:
        """Valida que la URL de la base de datos sea válida."""
        if not v.startswith(('postgresql://', 'postgres://')):
            raise ValueError('La URL debe ser una conexión PostgreSQL válida')
        return v


class RedisSettings(BaseSettings):
    """Configuración de Redis."""
    
    url: str = Field(
        default="redis://localhost:6379",
        description="URL de conexión a Redis"
    )
    max_connections: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Número máximo de conexiones a Redis"
    )
    password: Optional[str] = Field(
        default=None,
        description="Contraseña de Redis (opcional)"
    )
    db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Número de base de datos Redis"
    )


class ServerSettings(BaseSettings):
    """Configuración del servidor web."""
    
    host: str = Field(
        default="0.0.0.0",
        description="Host del servidor"
    )
    port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="Puerto del servidor"
    )
    workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Número de workers del servidor"
    )
    reload: bool = Field(
        default=False,
        description="Habilitar auto-reload en desarrollo"
    )
    log_level: str = Field(
        default="info",
        description="Nivel de logging del servidor"
    )


class LoggingSettings(BaseSettings):
    """Configuración del sistema de logging."""
    
    level: str = Field(
        default="INFO",
        description="Nivel de logging global"
    )
    format: str = Field(
        default="json",
        description="Formato de logging (json, text)"
    )
    file_path: Optional[Path] = Field(
        default=None,
        description="Ruta al archivo de log (opcional)"
    )
    max_size: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        ge=1024 * 1024,  # 1MB mínimo
        description="Tamaño máximo del archivo de log"
    )
    backup_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de archivos de backup"
    )


class AISettings(BaseSettings):
    """Configuración de IA y Machine Learning."""
    
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key de OpenAI"
    )
    huggingface_token: Optional[str] = Field(
        default=None,
        description="Token de Hugging Face"
    )
    model_cache_dir: Path = Field(
        default=Path("./models"),
        description="Directorio para cachear modelos de IA"
    )
    enable_ai_analysis: bool = Field(
        default=True,
        description="Habilitar análisis con IA"
    )
    max_tokens: int = Field(
        default=4096,
        ge=512,
        le=32768,
        description="Número máximo de tokens para análisis"
    )


class AuthSettings(BaseSettings):
    """Configuración de autenticación."""
    
    jwt_secret: str = Field(
        default="your-super-secret-key-change-in-production",
        description="Clave secreta para JWT"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="Algoritmo para JWT"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Tiempo de expiración del access token en minutos"
    )
    refresh_token_expire_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Tiempo de expiración del refresh token en días"
    )
    
    
class SecuritySettings(BaseSettings):
    """Configuración de seguridad."""
    
    encryption_key: Optional[str] = Field(
        default=None,
        description="Clave de encriptación de 32 bytes"
    )
    enable_rate_limiting: bool = Field(
        default=True,
        description="Habilitar rate limiting"
    )
    max_requests_per_minute: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Máximo de requests por minuto por IP"
    )


class TelemetrySettings(BaseSettings):
    """Configuración de telemetría y observabilidad."""
    
    # Prometheus
    prometheus_enabled: bool = Field(
        default=True,
        description="Habilitar métricas Prometheus"
    )
    prometheus_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Puerto para métricas Prometheus"
    )
    
    # Jaeger Tracing
    jaeger_enabled: bool = Field(
        default=True,
        description="Habilitar distributed tracing con Jaeger"
    )
    jaeger_endpoint: str = Field(
        default="http://localhost:14268/api/traces",
        description="Endpoint de Jaeger para envío de traces"
    )
    jaeger_service_name: str = Field(
        default="codeant-agent",
        description="Nombre del servicio para Jaeger"
    )
    
    # OpenTelemetry
    otel_enabled: bool = Field(
        default=True,
        description="Habilitar OpenTelemetry"
    )
    otel_endpoint: str = Field(
        default="http://localhost:4317",
        description="Endpoint de OpenTelemetry Collector"
    )
    
    # Log Aggregation
    log_aggregation_enabled: bool = Field(
        default=False,
        description="Habilitar agregación de logs"
    )
    elasticsearch_url: str = Field(
        default="http://localhost:9200",
        description="URL de Elasticsearch para logs"
    )
    opensearch_url: str = Field(
        default="http://localhost:9200",
        description="URL de OpenSearch para logs"
    )
    
    # Alerting
    alerting_enabled: bool = Field(
        default=False,
        description="Habilitar sistema de alertas"
    )
    alertmanager_url: str = Field(
        default="http://localhost:9093",
        description="URL de Alertmanager"
    )
    
    # Sampling
    trace_sampling_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Tasa de sampling para traces (0.0 = none, 1.0 = all)"
    )
    metric_sampling_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Tasa de sampling para métricas"
    )


class CORSSettings(BaseSettings):
    """Configuración de CORS."""
    
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8080", 
            "http://localhost:4200"
        ],
        description="Orígenes permitidos para CORS"
    )
    allowed_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        description="Métodos HTTP permitidos"
    )
    allowed_headers: List[str] = Field(
        default=["*"],
        description="Headers permitidos"
    )
    allow_credentials: bool = Field(
        default=True,
        description="Permitir credenciales en requests CORS"
    )


class FeatureFlags(BaseSettings):
    """Feature flags para habilitar/deshabilitar funcionalidades."""
    
    enable_telemetry: bool = Field(
        default=True,
        description="Habilitar telemetría y métricas"
    )
    enable_experimental: bool = Field(
        default=False,
        description="Habilitar funcionalidades experimentales"
    )
    enable_debug_endpoints: bool = Field(
        default=False,
        description="Habilitar endpoints de debug"
    )
    enable_health_checks: bool = Field(
        default=True,
        description="Habilitar health checks"
    )


class Settings(BaseSettings):
    """Configuración principal de la aplicación."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Configuraciones específicas
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    ai: AISettings = Field(default_factory=AISettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    
    # Configuración general
    environment: str = Field(
        default="development",
        description="Entorno de ejecución"
    )
    debug: bool = Field(
        default=False,
        description="Modo debug"
    )
    version: str = Field(
        default="0.1.0",
        description="Versión de la aplicación"
    )
    
    @validator('environment')
    def validate_environment(cls, v: str) -> str:
        """Valida que el entorno sea válido."""
        valid_environments = ['development', 'testing', 'staging', 'production']
        if v not in valid_environments:
            raise ValueError(f'Entorno debe ser uno de: {valid_environments}')
        return v
    
    @validator('debug')
    def validate_debug(cls, v: bool, values: dict) -> bool:
        """Valida que debug solo esté habilitado en desarrollo."""
        if v and values.get('environment') == 'production':
            raise ValueError('Debug no puede estar habilitado en producción')
        return v
    
    def is_development(self) -> bool:
        """Retorna True si estamos en entorno de desarrollo."""
        return self.environment == 'development'
    
    def is_production(self) -> bool:
        """Retorna True si estamos en entorno de producción."""
        return self.environment == 'production'
    
    def is_testing(self) -> bool:
        """Retorna True si estamos en entorno de testing."""
        return self.environment == 'testing'


# Instancia global de configuración
settings = Settings()


def get_settings() -> Settings:
    """Retorna la instancia de configuración."""
    return settings


def reload_settings() -> None:
    """Recarga la configuración desde las fuentes."""
    global settings
    settings = Settings()
