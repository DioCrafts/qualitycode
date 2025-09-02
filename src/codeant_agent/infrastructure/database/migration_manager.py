"""
Gestor de migraciones de base de datos.
"""
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
from dataclasses import dataclass

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from codeant_agent.utils.error import Result, BaseError
from codeant_agent.utils.logging import get_logger

logger = get_logger(__name__)


class MigrationError(BaseError):
    """Error en operaciones de migración."""
    pass


class MigrationNotFoundError(MigrationError):
    """Error cuando no se encuentra una migración."""
    pass


class MigrationExecutionError(MigrationError):
    """Error al ejecutar una migración."""
    pass


@dataclass
class Migration:
    """Información de una migración."""
    version: int
    name: str
    up_file: Path
    down_file: Path
    created_at: datetime
    executed_at: Optional[datetime] = None


@dataclass
class MigrationResult:
    """Resultado de una operación de migración."""
    migration: Migration
    success: bool
    duration_ms: int
    error_message: Optional[str] = None


class MigrationManager:
    """
    Gestor de migraciones de base de datos.
    
    Maneja la ejecución, rollback y seguimiento de migraciones
    de base de datos de forma asíncrona.
    """
    
    def __init__(self, database_url: str, migrations_path: str = "migrations"):
        """
        Inicializar el gestor de migraciones.
        
        Args:
            database_url: URL de conexión a la base de datos
            migrations_path: Ruta al directorio de migraciones
        """
        self.database_url = database_url
        self.migrations_path = Path(migrations_path)
        self.engine = None
        self.session_factory = None
        
        # Crear directorio de migraciones si no existe
        self.migrations_path.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> Result[None, Exception]:
        """Inicializar la conexión a la base de datos."""
        try:
            # Crear engine de SQLAlchemy
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Crear session factory
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Crear tabla de migraciones si no existe
            await self._create_migrations_table()
            
            logger.info("MigrationManager inicializado correctamente")
            return Result.success(None)
            
        except Exception as e:
            logger.error(f"Error al inicializar MigrationManager: {str(e)}")
            return Result.failure(MigrationError(f"Error de inicialización: {str(e)}"))
    
    async def discover_migrations(self) -> Result[List[Migration], Exception]:
        """Descubrir todas las migraciones disponibles."""
        try:
            migrations = []
            
            # Buscar archivos de migración
            for file_path in self.migrations_path.glob("*.up.sql"):
                # Extraer información del nombre del archivo
                match = re.match(r"(\d+)_(.+)\.up\.sql", file_path.name)
                if not match:
                    logger.warning(f"Archivo de migración con formato inválido: {file_path.name}")
                    continue
                
                version = int(match.group(1))
                name = match.group(2).replace("_", " ")
                
                # Buscar archivo de rollback correspondiente
                down_file = file_path.parent / f"{match.group(1)}_{match.group(2)}.down.sql"
                if not down_file.exists():
                    logger.warning(f"Archivo de rollback no encontrado para: {file_path.name}")
                    continue
                
                migration = Migration(
                    version=version,
                    name=name,
                    up_file=file_path,
                    down_file=down_file,
                    created_at=datetime.fromtimestamp(file_path.stat().st_mtime)
                )
                migrations.append(migration)
            
            # Ordenar por versión
            migrations.sort(key=lambda m: m.version)
            
            logger.info(f"Descubiertas {len(migrations)} migraciones")
            return Result.success(migrations)
            
        except Exception as e:
            logger.error(f"Error al descubrir migraciones: {str(e)}")
            return Result.failure(MigrationError(f"Error al descubrir migraciones: {str(e)}"))
    
    async def get_executed_migrations(self) -> Result[List[Migration], Exception]:
        """Obtener lista de migraciones ya ejecutadas."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    text("SELECT version, name, up_file, down_file, created_at, executed_at FROM migrations ORDER BY version")
                )
                rows = result.fetchall()
                
                migrations = []
                for row in rows:
                    migration = Migration(
                        version=row[0],
                        name=row[1],
                        up_file=Path(row[2]) if row[2] else Path(),
                        down_file=Path(row[3]) if row[3] else Path(),
                        created_at=row[4],
                        executed_at=row[5]
                    )
                    migrations.append(migration)
                
                return Result.success(migrations)
                
        except Exception as e:
            logger.error(f"Error al obtener migraciones ejecutadas: {str(e)}")
            return Result.failure(MigrationError(f"Error al obtener migraciones ejecutadas: {str(e)}"))
    
    async def run_migrations(self, target_version: Optional[int] = None) -> Result[List[MigrationResult], Exception]:
        """
        Ejecutar migraciones pendientes.
        
        Args:
            target_version: Versión objetivo (None para ejecutar todas las pendientes)
            
        Returns:
            Result con la lista de resultados de migración
        """
        try:
            # Descubrir migraciones disponibles
            available_result = await self.discover_migrations()
            if not available_result.success:
                return available_result
            
            available_migrations = available_result.data
            
            # Obtener migraciones ejecutadas
            executed_result = await self.get_executed_migrations()
            if not executed_result.success:
                return executed_result
            
            executed_migrations = executed_result.data
            executed_versions = {m.version for m in executed_migrations}
            
            # Filtrar migraciones pendientes
            pending_migrations = [
                m for m in available_migrations 
                if m.version not in executed_versions
            ]
            
            if target_version is not None:
                pending_migrations = [
                    m for m in pending_migrations 
                    if m.version <= target_version
                ]
            
            if not pending_migrations:
                logger.info("No hay migraciones pendientes")
                return Result.success([])
            
            # Ejecutar migraciones
            results = []
            for migration in pending_migrations:
                result = await self._execute_migration(migration)
                results.append(result)
                
                if not result.success:
                    logger.error(f"Migración {migration.version} falló: {result.error_message}")
                    break
            
            logger.info(f"Ejecutadas {len([r for r in results if r.success])} migraciones")
            return Result.success(results)
            
        except Exception as e:
            logger.error(f"Error al ejecutar migraciones: {str(e)}")
            return Result.failure(MigrationError(f"Error al ejecutar migraciones: {str(e)}"))
    
    async def rollback_migration(self, target_version: int) -> Result[MigrationResult, Exception]:
        """
        Hacer rollback a una versión específica.
        
        Args:
            target_version: Versión objetivo para el rollback
            
        Returns:
            Result con el resultado del rollback
        """
        try:
            # Obtener migraciones ejecutadas
            executed_result = await self.get_executed_migrations()
            if not executed_result.success:
                return executed_result
            
            executed_migrations = executed_result.data
            
            # Encontrar migraciones a revertir
            migrations_to_rollback = [
                m for m in executed_migrations 
                if m.version > target_version
            ]
            
            if not migrations_to_rollback:
                logger.info(f"No hay migraciones para revertir hasta la versión {target_version}")
                return Result.success(MigrationResult(
                    migration=Migration(0, "none", Path(), Path(), datetime.utcnow()),
                    success=True,
                    duration_ms=0
                ))
            
            # Ordenar en orden inverso para revertir
            migrations_to_rollback.sort(key=lambda m: m.version, reverse=True)
            
            # Ejecutar rollback de la primera migración
            migration = migrations_to_rollback[0]
            result = await self._rollback_migration(migration)
            
            return result
            
        except Exception as e:
            logger.error(f"Error al hacer rollback: {str(e)}")
            return Result.failure(MigrationError(f"Error al hacer rollback: {str(e)}"))
    
    async def get_migration_status(self) -> Result[Dict[str, Any], Exception]:
        """Obtener el estado actual de las migraciones."""
        try:
            # Descubrir migraciones disponibles
            available_result = await self.discover_migrations()
            if not available_result.success:
                return available_result
            
            available_migrations = available_result.data
            
            # Obtener migraciones ejecutadas
            executed_result = await self.get_executed_migrations()
            if not executed_result.success:
                return executed_result
            
            executed_migrations = executed_result.data
            executed_versions = {m.version for m in executed_migrations}
            
            # Calcular estadísticas
            total_migrations = len(available_migrations)
            executed_count = len(executed_migrations)
            pending_count = total_migrations - executed_count
            
            # Obtener versión actual
            current_version = max(executed_versions) if executed_versions else 0
            
            # Obtener última migración ejecutada
            last_executed = max(executed_migrations, key=lambda m: m.version) if executed_migrations else None
            
            status = {
                "total_migrations": total_migrations,
                "executed_count": executed_count,
                "pending_count": pending_count,
                "current_version": current_version,
                "last_executed": last_executed.name if last_executed else None,
                "last_executed_at": last_executed.executed_at if last_executed else None,
                "is_up_to_date": pending_count == 0
            }
            
            return Result.success(status)
            
        except Exception as e:
            logger.error(f"Error al obtener estado de migraciones: {str(e)}")
            return Result.failure(MigrationError(f"Error al obtener estado: {str(e)}"))
    
    async def _create_migrations_table(self) -> None:
        """Crear tabla de migraciones si no existe."""
        async with self.session_factory() as session:
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS migrations (
                    version INTEGER PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    executed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                )
            """))
            await session.commit()
    
    async def _execute_migration(self, migration: Migration) -> MigrationResult:
        """Ejecutar una migración específica."""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Ejecutando migración {migration.version}: {migration.name}")
            
            # Leer contenido del archivo
            with open(migration.up_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Ejecutar SQL
            async with self.session_factory() as session:
                # Dividir en statements individuales
                statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement:
                        await session.execute(text(statement))
                
                # Registrar migración como ejecutada
                await session.execute(
                    text("INSERT INTO migrations (version, name) VALUES (:version, :name)"),
                    {"version": migration.version, "name": migration.name}
                )
                
                await session.commit()
            
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"Migración {migration.version} ejecutada exitosamente en {duration:.2f}ms")
            
            return MigrationResult(
                migration=migration,
                success=True,
                duration_ms=int(duration)
            )
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"Error al ejecutar migración {migration.version}: {str(e)}")
            
            return MigrationResult(
                migration=migration,
                success=False,
                duration_ms=int(duration),
                error_message=str(e)
            )
    
    async def _rollback_migration(self, migration: Migration) -> MigrationResult:
        """Hacer rollback de una migración específica."""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Haciendo rollback de migración {migration.version}: {migration.name}")
            
            # Buscar archivo de rollback
            down_file = self.migrations_path / f"{migration.version:03d}_{migration.name.replace(' ', '_')}.down.sql"
            
            if not down_file.exists():
                raise MigrationNotFoundError(f"Archivo de rollback no encontrado: {down_file}")
            
            # Leer contenido del archivo
            with open(down_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Ejecutar SQL
            async with self.session_factory() as session:
                # Dividir en statements individuales
                statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement:
                        await session.execute(text(statement))
                
                # Eliminar registro de migración
                await session.execute(
                    text("DELETE FROM migrations WHERE version = :version"),
                    {"version": migration.version}
                )
                
                await session.commit()
            
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"Rollback de migración {migration.version} completado en {duration:.2f}ms")
            
            return MigrationResult(
                migration=migration,
                success=True,
                duration_ms=int(duration)
            )
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"Error al hacer rollback de migración {migration.version}: {str(e)}")
            
            return MigrationResult(
                migration=migration,
                success=False,
                duration_ms=int(duration),
                error_message=str(e)
            )
    
    async def close(self) -> None:
        """Cerrar conexiones de base de datos."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Conexiones de base de datos cerradas")
