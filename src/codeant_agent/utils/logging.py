"""
Utilidades para logging.
"""
import logging
import json
import uuid
import sys
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
import asyncio

# Intenta importar FastAPI y Starlette si están disponibles
try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.types import ASGIApp
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Clases dummy para evitar errores de tipo
    class BaseHTTPMiddleware:
        pass
    class Request:
        pass
    class ASGIApp:
        pass

# Contexto de request para tracking
_request_context = {}

def setup_logging(level=logging.INFO, json_format=True):
    """
    Configurar logging para la aplicación.
    
    Args:
        level: Nivel de logging
        json_format: Si se debe usar formato JSON
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Eliminar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Crear handler para consola
    handler = logging.StreamHandler(sys.stdout)
    
    if json_format:
        formatter = logging.Formatter('{"timestamp": null, "level": null, "name": "%(name)s", "message": %(message)s}')
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def get_logger(name):
    """
    Obtener un logger con el nombre especificado.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    return logger

def set_request_context(request_id: str, **kwargs):
    """
    Establecer contexto de request para logging.
    
    Args:
        request_id: ID único de request
        **kwargs: Otros datos de contexto
    """
    global _request_context
    _request_context = {
        'request_id': request_id,
        **kwargs
    }

def get_request_context():
    """
    Obtener contexto de request actual.
    
    Returns:
        Diccionario con datos de contexto
    """
    return _request_context.copy()

def clear_request_context():
    """Limpiar contexto de request."""
    global _request_context
    _request_context = {}

@contextmanager
def request_context(request_id=None, **kwargs):
    """
    Context manager para establecer contexto de request temporalmente.
    
    Args:
        request_id: ID único de request
        **kwargs: Otros datos de contexto
        
    Yields:
        ID de request
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    old_context = _request_context.copy()
    set_request_context(request_id, **kwargs)
    
    try:
        yield request_id
    finally:
        global _request_context
        _request_context = old_context

# Middleware para FastAPI si está disponible
if FASTAPI_AVAILABLE:
    class RequestContextMiddleware(BaseHTTPMiddleware):
        """
        Middleware para establecer contexto de request en FastAPI.
        """
        
        def __init__(
            self,
            app: ASGIApp,
        ) -> None:
            super().__init__(app)
        
        async def dispatch(
            self, request: Request, call_next
        ):
            request_id = str(uuid.uuid4())
            
            # Establecer contexto
            with request_context(request_id=request_id, method=request.method, path=request.url.path, client=request.client.host if request.client else None):
                # Log inicio de request
                logger = get_logger("request")
                logger.info(json.dumps({
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client": request.client.host if request.client else None,
                    "event": "Request started",
                    "level": "info",
                    "timestamp": str(asyncio.get_event_loop().time())
                }))
                
                # Ejecutar request
                response = await call_next(request)
                
                # Log fin de request
                logger.info(json.dumps({
                    "request_id": request_id,
                    "event": "Request finished",
                    "level": "info",
                    "timestamp": str(asyncio.get_event_loop().time())
                }))
                
                return response
else:
    # Clase dummy si FastAPI no está disponible
    class RequestContextMiddleware:
        def __init__(self, app):
            self.app = app