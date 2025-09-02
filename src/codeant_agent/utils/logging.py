"""
Sistema de logging estructurado para CodeAnt Agent.

Este módulo implementa logging estructurado usando structlog con:
- Formato JSON para producción
- Formato legible para desarrollo
- Contexto automático (request ID, user ID, etc.)
- Integración con FastAPI
- Rotación de archivos
"""

import sys
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import structlog
from structlog.stdlib import LoggerFactory
from structlog.processors import (
    TimeStamper,
    JSONRenderer,
    format_exc_info,
    add_log_level,
    StackInfoRenderer,
)
from pythonjsonlogger import jsonlogger

from ..config.settings import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> None:
    """
    Configura el sistema de logging estructurado.
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Formato de logging (json, text)
        log_file: Ruta al archivo de log (opcional)
    """
    settings = get_settings()
    
    # Usar configuración por defecto si no se especifica
    log_level = log_level or settings.logging.level
    log_format = log_format or settings.logging.format
    log_file = log_file or settings.logging.file_path
    
    # Configurar nivel de logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configurar handlers
    handlers = []
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    handlers.append(console_handler)
    
    # Handler para archivo si se especifica
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=settings.logging.max_size,
            backupCount=settings.logging.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)
    
    # Configurar formato según el tipo
    if log_format.lower() == 'json':
        setup_json_logging(handlers, numeric_level)
    else:
        setup_text_logging(handlers, numeric_level)
    
    # Configurar structlog
    setup_structlog(log_format.lower() == 'json')


def setup_json_logging(handlers: list, level: int) -> None:
    """Configura logging en formato JSON."""
    formatter = jsonlogger.JsonFormatter(
        fmt='%(timestamp)s %(level)s %(name)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    for handler in handlers:
        handler.setFormatter(formatter)
    
    # Configurar logging básico
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(message)s'
    )


def setup_text_logging(handlers: list, level: int) -> None:
    """Configura logging en formato de texto legible."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    for handler in handlers:
        handler.setFormatter(formatter)
    
    # Configurar logging básico
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(message)s'
    )


def setup_structlog(is_json: bool = False) -> None:
    """
    Configura structlog con procesadores apropiados.
    
    Args:
        is_json: Si es True, usa formato JSON; si es False, formato legible
    """
    processors = [
        StackInfoRenderer(),
        add_log_level,
        TimeStamper(fmt="iso"),
        format_exc_info,
    ]
    
    if is_json:
        processors.append(JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Obtiene un logger estructurado.
    
    Args:
        name: Nombre del logger (generalmente __name__)
        
    Returns:
        Logger estructurado configurado
    """
    return structlog.get_logger(name)


class RequestContextMiddleware:
    """Middleware para agregar contexto de request al logging."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        # Generar ID único para el request
        import uuid
        request_id = str(uuid.uuid4())
        
        # Agregar contexto al logger
        logger = get_logger("request")
        logger = logger.bind(request_id=request_id)
        
        # Agregar request_id al scope para que esté disponible
        scope['request_id'] = request_id
        
        # Log del inicio del request
        logger.info(
            "Request started",
            method=scope.get('method', 'UNKNOWN'),
            path=scope.get('path', 'UNKNOWN'),
            client=scope.get('client', ('UNKNOWN', 0))[0]
        )
        
        # Continuar con el request
        await self.app(scope, receive, send)
        
        # Log del fin del request
        logger.info("Request finished")


class ContextualLogger:
    """
    Logger que mantiene contexto entre llamadas.
    
    Permite agregar contexto que se mantiene en todas las llamadas
    posteriores del logger.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._logger = get_logger(name)
        self._context: Dict[str, Any] = {}
    
    def bind(self, **kwargs) -> 'ContextualLogger':
        """
        Agrega contexto al logger.
        
        Args:
            **kwargs: Pares clave-valor para el contexto
            
        Returns:
            Nueva instancia del logger con contexto
        """
        new_logger = ContextualLogger(self.name)
        new_logger._context = {**self._context, **kwargs}
        new_logger._logger = self._logger.bind(**new_logger._context)
        return new_logger
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a nivel DEBUG."""
        self._logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log a nivel INFO."""
        self._logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a nivel WARNING."""
        self._logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log a nivel ERROR."""
        self._logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log a nivel CRITICAL."""
        self._logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log de excepción con traceback."""
        self._logger.exception(message, **kwargs)


def log_function_call(func_name: str, **kwargs):
    """
    Decorator para logging automático de llamadas a funciones.
    
    Args:
        func_name: Nombre de la función
        **kwargs: Argumentos adicionales para el log
    """
    def decorator(func):
        async def async_wrapper(*args, **func_kwargs):
            logger = get_logger(f"function.{func_name}")
            logger.info(
                "Function called",
                function=func_name,
                args_count=len(args),
                kwargs_count=len(func_kwargs),
                **kwargs
            )
            
            try:
                result = await func(*args, **func_kwargs)
                logger.info(
                    "Function completed successfully",
                    function=func_name
                )
                return result
            except Exception as e:
                logger.exception(
                    "Function failed",
                    function=func_name,
                    error=str(e)
                )
                raise
        
        def sync_wrapper(*args, **func_kwargs):
            logger = get_logger(f"function.{func_name}")
            logger.info(
                "Function called",
                function=func_name,
                args_count=len(args),
                kwargs_count=len(func_kwargs),
                **kwargs
            )
            
            try:
                result = func(*args, **func_kwargs)
                logger.info(
                    "Function completed successfully",
                    function=func_name
                )
                return result
            except Exception as e:
                logger.exception(
                    "Function failed",
                    function=func_name,
                    error=str(e)
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Logger global para la aplicación
app_logger = get_logger("codeant_agent")


def get_app_logger() -> structlog.stdlib.BoundLogger:
    """Retorna el logger principal de la aplicación."""
    return app_logger
