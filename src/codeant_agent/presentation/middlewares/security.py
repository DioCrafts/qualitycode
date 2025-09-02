"""Middleware de seguridad y CORS."""

from typing import List, Optional
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware para agregar headers de seguridad."""
    
    def __init__(self, app, enforce_https: bool = True):
        """Inicializar el middleware."""
        super().__init__(app)
        self.enforce_https = enforce_https
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Procesar request y agregar headers de seguridad."""
        # Verificar HTTPS en producción
        if self.enforce_https and request.url.scheme != "https":
            # Permitir HTTP solo en desarrollo local
            if not (request.client.host in ["127.0.0.1", "localhost", "::1"]):
                # Redirigir a HTTPS
                https_url = request.url.replace(scheme="https")
                return Response(
                    status_code=301,
                    headers={"Location": str(https_url)}
                )
        
        # Procesar request
        response = await call_next(request)
        
        # Agregar headers de seguridad
        self._add_security_headers(response)
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Agregar headers de seguridad a la respuesta."""
        # Prevenir MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Prevenir clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # XSS protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # HSTS - HTTP Strict Transport Security
        if self.enforce_https:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline'",  # Permitir scripts inline para Swagger UI
            "style-src 'self' 'unsafe-inline'",   # Permitir estilos inline para Swagger UI
            "img-src 'self' data: https:",
            "font-src 'self' https:",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Feature Policy / Permissions Policy
        permissions_directives = [
            "camera=()",
            "microphone=()",
            "geolocation=()",
            "payment=()",
            "usb=()",
            "magnetometer=()",
            "accelerometer=()",
            "gyroscope=()"
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions_directives)
        
        # Prevenir información del servidor
        response.headers.pop("server", None)
        
        # API versioning
        response.headers["API-Version"] = "v1"
        
        # Cache control para endpoints de API
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"


class CORSConfig:
    """Configuración de CORS."""
    
    def __init__(
        self,
        allowed_origins: List[str] = None,
        allowed_methods: List[str] = None,
        allowed_headers: List[str] = None,
        allow_credentials: bool = True,
        expose_headers: List[str] = None,
        max_age: int = 600
    ):
        """Inicializar configuración de CORS."""
        self.allowed_origins = allowed_origins or [
            "http://localhost:3000",  # React dev server
            "http://localhost:8080",  # Vue dev server
            "http://localhost:4200",  # Angular dev server
            "https://app.codeant.com",  # Producción
            "https://staging.codeant.com"  # Staging
        ]
        
        self.allowed_methods = allowed_methods or [
            "GET",
            "POST", 
            "PUT",
            "DELETE",
            "PATCH",
            "OPTIONS"
        ]
        
        self.allowed_headers = allowed_headers or [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Request-ID",
            "X-API-Version",
            "X-Client-Version"
        ]
        
        self.expose_headers = expose_headers or [
            "X-RateLimit-Limit-Minute",
            "X-RateLimit-Limit-Hour",
            "X-RateLimit-Remaining-Minute",
            "X-RateLimit-Remaining-Hour",
            "X-RateLimit-Reset-Minute",
            "X-RateLimit-Reset-Hour",
            "X-Request-ID",
            "API-Version"
        ]
        
        self.allow_credentials = allow_credentials
        self.max_age = max_age


def setup_cors(app, cors_config: Optional[CORSConfig] = None):
    """Configurar CORS para la aplicación."""
    if cors_config is None:
        cors_config = CORSConfig()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.allowed_origins,
        allow_credentials=cors_config.allow_credentials,
        allow_methods=cors_config.allowed_methods,
        allow_headers=cors_config.allowed_headers,
        expose_headers=cors_config.expose_headers,
        max_age=cors_config.max_age,
    )


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware para rastreo de requests."""
    
    def __init__(self, app):
        """Inicializar el middleware."""
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Procesar request con rastreo."""
        import uuid
        import time
        
        # Generar ID único para la request
        request_id = str(uuid.uuid4())
        
        # Agregar a request state
        request.state.request_id = request_id
        
        # Medir tiempo de procesamiento
        start_time = time.time()
        
        # Procesar request
        response = await call_next(request)
        
        # Calcular tiempo de procesamiento
        process_time = time.time() - start_time
        
        # Agregar headers de rastreo
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        return response


class ContentValidationMiddleware(BaseHTTPMiddleware):
    """Middleware para validación de contenido."""
    
    def __init__(self, app, max_content_length: int = 10 * 1024 * 1024):  # 10MB por defecto
        """Inicializar el middleware."""
        super().__init__(app)
        self.max_content_length = max_content_length
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Validar contenido de la request."""
        # Verificar Content-Length
        content_length = request.headers.get("content-length")
        if content_length:
            content_length = int(content_length)
            if content_length > self.max_content_length:
                return Response(
                    status_code=413,
                    content=f"Request entity too large. Maximum size: {self.max_content_length} bytes"
                )
        
        # Verificar Content-Type para requests con body
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            
            # Lista de content types permitidos
            allowed_types = [
                "application/json",
                "application/x-www-form-urlencoded",
                "multipart/form-data",
                "text/plain"
            ]
            
            if not any(allowed_type in content_type for allowed_type in allowed_types):
                return Response(
                    status_code=415,
                    content=f"Unsupported Media Type. Allowed types: {', '.join(allowed_types)}"
                )
        
        return await call_next(request)


def setup_security_middleware(
    app,
    enforce_https: bool = True,
    cors_config: Optional[CORSConfig] = None,
    max_content_length: int = 10 * 1024 * 1024,
    enable_request_tracing: bool = True
):
    """Configurar todos los middlewares de seguridad."""
    # Request tracing (primero para capturar todo)
    if enable_request_tracing:
        app.add_middleware(RequestTracingMiddleware)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware, enforce_https=enforce_https)
    
    # Content validation
    app.add_middleware(ContentValidationMiddleware, max_content_length=max_content_length)
    
    # CORS (último para que se aplique a todas las responses)
    setup_cors(app, cors_config)
