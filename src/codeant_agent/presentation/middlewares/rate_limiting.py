"""Middleware de rate limiting."""

import time
from typing import Dict, Optional
from collections import defaultdict, deque
from fastapi import HTTPException, status, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


class RateLimitConfig:
    """Configuración de rate limiting."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_allowance: int = 10,
        whitelist_ips: Optional[list] = None,
        custom_limits: Optional[Dict[str, str]] = None
    ):
        """Inicializar configuración de rate limiting."""
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_allowance = burst_allowance
        self.whitelist_ips = whitelist_ips or []
        self.custom_limits = custom_limits or {}


class AdvancedRateLimiter:
    """Rate limiter avanzado con múltiples ventanas de tiempo."""
    
    def __init__(self, config: RateLimitConfig):
        """Inicializar el rate limiter."""
        self.config = config
        self.requests_minute = defaultdict(deque)  # IP -> deque de timestamps
        self.requests_hour = defaultdict(deque)
        self.burst_tracker = defaultdict(deque)
        
    def is_whitelisted(self, ip: str) -> bool:
        """Verificar si una IP está en la whitelist."""
        return ip in self.config.whitelist_ips
    
    def clean_old_requests(self, ip: str, current_time: float):
        """Limpiar requests antiguos para una IP."""
        # Limpiar requests de más de 1 minuto
        minute_ago = current_time - 60
        while (self.requests_minute[ip] and 
               self.requests_minute[ip][0] < minute_ago):
            self.requests_minute[ip].popleft()
        
        # Limpiar requests de más de 1 hora
        hour_ago = current_time - 3600
        while (self.requests_hour[ip] and 
               self.requests_hour[ip][0] < hour_ago):
            self.requests_hour[ip].popleft()
            
        # Limpiar burst tracker (últimos 10 segundos)
        ten_seconds_ago = current_time - 10
        while (self.burst_tracker[ip] and 
               self.burst_tracker[ip][0] < ten_seconds_ago):
            self.burst_tracker[ip].popleft()
    
    def check_rate_limit(self, ip: str) -> bool:
        """Verificar si una IP puede hacer una request."""
        if self.is_whitelisted(ip):
            return True
            
        current_time = time.time()
        self.clean_old_requests(ip, current_time)
        
        # Verificar límite por minuto
        if len(self.requests_minute[ip]) >= self.config.requests_per_minute:
            return False
            
        # Verificar límite por hora
        if len(self.requests_hour[ip]) >= self.config.requests_per_hour:
            return False
            
        # Verificar burst protection
        if len(self.burst_tracker[ip]) >= self.config.burst_allowance:
            return False
        
        return True
    
    def record_request(self, ip: str):
        """Registrar una request."""
        if self.is_whitelisted(ip):
            return
            
        current_time = time.time()
        self.requests_minute[ip].append(current_time)
        self.requests_hour[ip].append(current_time)
        self.burst_tracker[ip].append(current_time)
    
    def get_rate_limit_info(self, ip: str) -> Dict[str, int]:
        """Obtener información sobre el rate limit actual."""
        if self.is_whitelisted(ip):
            return {
                "requests_per_minute": 0,
                "requests_per_hour": 0,
                "burst_requests": 0,
                "remaining_minute": self.config.requests_per_minute,
                "remaining_hour": self.config.requests_per_hour,
                "remaining_burst": self.config.burst_allowance
            }
        
        current_time = time.time()
        self.clean_old_requests(ip, current_time)
        
        return {
            "requests_per_minute": len(self.requests_minute[ip]),
            "requests_per_hour": len(self.requests_hour[ip]),
            "burst_requests": len(self.burst_tracker[ip]),
            "remaining_minute": max(0, self.config.requests_per_minute - len(self.requests_minute[ip])),
            "remaining_hour": max(0, self.config.requests_per_hour - len(self.requests_hour[ip])),
            "remaining_burst": max(0, self.config.burst_allowance - len(self.burst_tracker[ip]))
        }


class RateLimitMiddleware:
    """Middleware de rate limiting para FastAPI."""
    
    def __init__(self, config: RateLimitConfig):
        """Inicializar el middleware."""
        self.config = config
        self.limiter = AdvancedRateLimiter(config)
        
        # Configurar slowapi limiter para casos simples
        self.slowapi_limiter = Limiter(key_func=get_remote_address)
    
    async def __call__(self, request: Request, call_next):
        """Procesar request con rate limiting."""
        client_ip = get_remote_address(request)
        
        # Verificar rate limit
        if not self.limiter.check_rate_limit(client_ip):
            rate_info = self.limiter.get_rate_limit_info(client_ip)
            
            # Determinar qué límite se excedió
            if rate_info["remaining_burst"] == 0:
                retry_after = 10  # 10 segundos para burst
                detail = f"Rate limit exceeded: too many requests in short time. Try again in {retry_after} seconds."
            elif rate_info["remaining_minute"] == 0:
                retry_after = 60  # 1 minuto
                detail = f"Rate limit exceeded: {self.config.requests_per_minute} requests per minute. Try again in {retry_after} seconds."
            else:
                retry_after = 3600  # 1 hora
                detail = f"Rate limit exceeded: {self.config.requests_per_hour} requests per hour. Try again in {retry_after} seconds."
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=detail,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit-Minute": str(self.config.requests_per_minute),
                    "X-RateLimit-Limit-Hour": str(self.config.requests_per_hour),
                    "X-RateLimit-Remaining-Minute": str(rate_info["remaining_minute"]),
                    "X-RateLimit-Remaining-Hour": str(rate_info["remaining_hour"]),
                    "X-RateLimit-Reset-Minute": str(int(time.time()) + 60),
                    "X-RateLimit-Reset-Hour": str(int(time.time()) + 3600),
                }
            )
        
        # Registrar la request
        self.limiter.record_request(client_ip)
        
        # Continuar con el procesamiento
        response = await call_next(request)
        
        # Agregar headers de rate limit info
        rate_info = self.limiter.get_rate_limit_info(client_ip)
        response.headers["X-RateLimit-Limit-Minute"] = str(self.config.requests_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.config.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Minute"] = str(rate_info["remaining_minute"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(rate_info["remaining_hour"])
        response.headers["X-RateLimit-Reset-Minute"] = str(int(time.time()) + 60)
        response.headers["X-RateLimit-Reset-Hour"] = str(int(time.time()) + 3600)
        
        return response


def create_rate_limiter(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    burst_allowance: int = 10,
    whitelist_ips: Optional[list] = None
) -> RateLimitMiddleware:
    """Crear un rate limiter configurado."""
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        burst_allowance=burst_allowance,
        whitelist_ips=whitelist_ips
    )
    return RateLimitMiddleware(config)


# Rate limiters específicos para diferentes endpoints
def get_auth_rate_limiter() -> RateLimitMiddleware:
    """Rate limiter más estricto para endpoints de autenticación."""
    return create_rate_limiter(
        requests_per_minute=10,
        requests_per_hour=100,
        burst_allowance=3
    )


def get_api_rate_limiter() -> RateLimitMiddleware:
    """Rate limiter estándar para endpoints de API."""
    return create_rate_limiter(
        requests_per_minute=60,
        requests_per_hour=1000,
        burst_allowance=10
    )


def get_analysis_rate_limiter() -> RateLimitMiddleware:
    """Rate limiter para endpoints de análisis (más permisivo)."""
    return create_rate_limiter(
        requests_per_minute=30,
        requests_per_hour=500,
        burst_allowance=5
    )
