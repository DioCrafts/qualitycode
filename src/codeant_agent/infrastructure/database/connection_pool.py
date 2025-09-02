"""
Pool de conexiones de base de datos.
"""
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from codeant_agent.utils.error import Result, BaseError
from codeant_agent.utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseConnectionError(BaseError):
    """Error en operaciones de conexión a base de datos."""
    pass


class ConnectionPoolError(DatabaseConnectionError):
    """Error en el pool de conexiones."""
    pass


@dataclass
class DatabaseConfig:
    """Configuración de la base de datos."""
    host: str
    port: int
    database: str
    username: str
    password: str
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: int = 30
    command_timeout: int = 60
    ssl_mode: str = "prefer"
    
    @property
    def url(self) -> str:
        """Generar URL de conexión."""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def asyncpg_url(self) -> str:
        """Generar URL para asyncpg."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class DatabaseConnectionPool:
    """
    Pool de conexiones de base de datos.
    
    Maneja conexiones asíncronas a PostgreSQL usando SQLAlchemy
    y asyncpg, con configuración de pool y health checks.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Inicializar el pool de conexiones.
        
        Args:
            config: Configuración de la base de datos
        """
        self.config = config
        self.engine: Optional[AsyncEngine] = None
        self.session_factory = None
        self._pool: Optional[asyncpg.Pool] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_initialized = False
    
    async def initialize(self) -> Result[None, Exception]:
        """Inicializar el pool de conexiones."""
        try:
            logger.info("Inicializando pool de conexiones de base de datos")
            
            # Crear engine de SQLAlchemy
            self.engine = create_async_engine(
                self.config.url,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=self.config.min_connections,
                max_overflow=self.config.max_connections - self.config.min_connections,
                pool_timeout=self.config.connection_timeout,
                poolclass=None  # Usar pool por defecto
            )
            
            # Crear session factory
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Crear pool de asyncpg para operaciones directas
            self._pool = await asyncpg.create_pool(
                self.config.asyncpg_url,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout,
                server_settings={
                    'application_name': 'codeant_agent',
                    'timezone': 'UTC'
                }
            )
            
            # Verificar conexión
            await self._test_connection()
            
            # Iniciar health check
            self._start_health_check()
            
            self._is_initialized = True
            logger.info("Pool de conexiones inicializado correctamente")
            
            return Result.success(None)
            
        except Exception as e:
            logger.error(f"Error al inicializar pool de conexiones: {str(e)}")
            return Result.failure(ConnectionPoolError(f"Error de inicialización: {str(e)}"))
    
    async def _test_connection(self) -> None:
        """Probar la conexión a la base de datos."""
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                result.fetchone()
            
            logger.info("Conexión a base de datos verificada correctamente")
            
        except Exception as e:
            logger.error(f"Error al verificar conexión: {str(e)}")
            raise ConnectionPoolError(f"No se pudo conectar a la base de datos: {str(e)}")
    
    def _start_health_check(self) -> None:
        """Iniciar tarea de health check."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self) -> None:
        """Loop de health check."""
        while self._is_initialized:
            try:
                await asyncio.sleep(30)  # Health check cada 30 segundos
                await self._perform_health_check()
                
            except asyncio.CancelledError:
                logger.info("Health check loop cancelado")
                break
            except Exception as e:
                logger.error(f"Error en health check: {str(e)}")
    
    async def _perform_health_check(self) -> None:
        """Realizar health check."""
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                result.fetchone()
            
            # Log solo si hay problemas
            logger.debug("Health check completado exitosamente")
            
        except Exception as e:
            logger.warning(f"Health check falló: {str(e)}")
            # Aquí podríamos implementar reconexión automática si es necesario
    
    @asynccontextmanager
    async def get_session(self):
        """
        Obtener una sesión de base de datos.
        
        Yields:
            AsyncSession: Sesión de SQLAlchemy
            
        Raises:
            ConnectionPoolError: Si el pool no está inicializado
        """
        if not self._is_initialized:
            raise ConnectionPoolError("Pool de conexiones no inicializado")
        
        session = self.session_factory()
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def get_connection(self):
        """
        Obtener una conexión directa de asyncpg.
        
        Returns:
            asyncpg.Connection: Conexión directa
            
        Raises:
            ConnectionPoolError: Si el pool no está inicializado
        """
        if not self._is_initialized or self._pool is None:
            raise ConnectionPoolError("Pool de conexiones no inicializado")
        
        return await self._pool.acquire()
    
    async def release_connection(self, connection) -> None:
        """
        Liberar una conexión de asyncpg.
        
        Args:
            connection: Conexión a liberar
        """
        if self._pool is not None:
            await self._pool.release(connection)
    
    async def execute_query(self, query: str, *args, **kwargs) -> Any:
        """
        Ejecutar una consulta SQL.
        
        Args:
            query: Consulta SQL
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Resultado de la consulta
        """
        async with self.get_session() as session:
            result = await session.execute(text(query), *args, **kwargs)
            return result
    
    async def execute_transaction(self, queries: list) -> Result[None, Exception]:
        """
        Ejecutar múltiples consultas en una transacción.
        
        Args:
            queries: Lista de tuplas (query, params)
            
        Returns:
            Result con el resultado de la operación
        """
        try:
            async with self.get_session() as session:
                for query, params in queries:
                    await session.execute(text(query), params)
                await session.commit()
            
            return Result.success(None)
            
        except Exception as e:
            logger.error(f"Error en transacción: {str(e)}")
            return Result.failure(DatabaseConnectionError(f"Error en transacción: {str(e)}"))
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del pool de conexiones.
        
        Returns:
            Diccionario con estadísticas del pool
        """
        if not self._is_initialized or self._pool is None:
            return {
                "initialized": False,
                "min_connections": self.config.min_connections,
                "max_connections": self.config.max_connections,
                "current_connections": 0,
                "available_connections": 0
            }
        
        stats = {
            "initialized": True,
            "min_connections": self.config.min_connections,
            "max_connections": self.config.max_connections,
            "current_connections": self._pool.get_size(),
            "available_connections": self._pool.get_free_size(),
            "connection_timeout": self.config.connection_timeout,
            "command_timeout": self.config.command_timeout
        }
        
        return stats
    
    async def close(self) -> None:
        """Cerrar el pool de conexiones."""
        logger.info("Cerrando pool de conexiones")
        
        # Cancelar health check
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cerrar pool de asyncpg
        if self._pool:
            await self._pool.close()
        
        # Cerrar engine de SQLAlchemy
        if self.engine:
            await self.engine.dispose()
        
        self._is_initialized = False
        logger.info("Pool de conexiones cerrado")


# Singleton para el pool de conexiones
_connection_pool: Optional[DatabaseConnectionPool] = None


async def get_connection_pool(config: DatabaseConfig) -> DatabaseConnectionPool:
    """
    Obtener instancia singleton del pool de conexiones.
    
    Args:
        config: Configuración de la base de datos
        
    Returns:
        Instancia del pool de conexiones
    """
    global _connection_pool
    
    if _connection_pool is None:
        _connection_pool = DatabaseConnectionPool(config)
        await _connection_pool.initialize()
    
    return _connection_pool


async def close_connection_pool() -> None:
    """Cerrar el pool de conexiones singleton."""
    global _connection_pool
    
    if _connection_pool is not None:
        await _connection_pool.close()
        _connection_pool = None
