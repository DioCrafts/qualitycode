"""
Servicio de integración para el sistema incremental.

Este módulo integra el sistema incremental con el sistema principal de CodeAnt,
proporcionando una interfaz unificada para el análisis de código.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from ...domain.entities.incremental import (
    AnalysisSession,
    ChangeSet,
    CacheEntry,
    DependencyGraph
)
from ...domain.value_objects.incremental_metrics import (
    IncrementalMetrics,
    CacheMetrics,
    PerformanceMetrics
)
from ...application.use_cases.incremental_use_cases import (
    IncrementalAnalysisUseCase,
    CacheManagementUseCase,
    ChangeDetectionUseCase,
    PredictiveCacheUseCase
)
from ...application.dtos.incremental_dtos import (
    IncrementalAnalysisRequest,
    IncrementalAnalysisResponse,
    ChangeDetectionRequest,
    ChangeDetectionResponse,
    PredictiveCacheRequest,
    PredictiveCacheResponse
)
from ...utils.logging import get_logger
from ...utils.error import BaseError, ValidationError

logger = get_logger(__name__)


class IncrementalIntegrationService:
    """
    Servicio de integración para el sistema incremental.
    
    Coordina la interacción entre el sistema incremental y el sistema principal
    de CodeAnt, proporcionando una interfaz unificada.
    """
    
    def __init__(
        self,
        incremental_analysis_use_case: IncrementalAnalysisUseCase,
        cache_management_use_case: CacheManagementUseCase,
        change_detection_use_case: ChangeDetectionUseCase,
        predictive_cache_use_case: PredictiveCacheUseCase
    ):
        """
        Inicializar el servicio de integración.
        
        Args:
            incremental_analysis_use_case: Caso de uso de análisis incremental
            cache_management_use_case: Caso de uso de gestión de caché
            change_detection_use_case: Caso de uso de detección de cambios
            predictive_cache_use_case: Caso de uso de caché predictivo
        """
        self.incremental_analysis_use_case = incremental_analysis_use_case
        self.cache_management_use_case = cache_management_use_case
        self.change_detection_use_case = change_detection_use_case
        self.predictive_cache_use_case = predictive_cache_use_case
        
        # Estado de sesiones activas
        self.active_sessions: Dict[str, AnalysisSession] = {}
        
        # Métricas de rendimiento
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        
        logger.info("Servicio de integración incremental inicializado")
    
    async def start_analysis_session(
        self,
        project_path: str,
        session_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Inicia una nueva sesión de análisis incremental.
        
        Args:
            project_path: Ruta del proyecto a analizar
            session_config: Configuración de la sesión
            
        Returns:
            ID de la sesión iniciada
            
        Raises:
            ValidationError: Si los parámetros no son válidos
            BaseError: Si hay un error en el dominio
        """
        try:
            logger.info(
                "Iniciando sesión de análisis incremental",
                project_path=project_path
            )
            
            # Validar ruta del proyecto
            if not Path(project_path).exists():
                raise ValidationError(f"La ruta del proyecto no existe: {project_path}")
            
            # Crear sesión
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{project_path.replace('/', '_')}"
            session = AnalysisSession(
                session_id=session_id,
                project_path=project_path,
                start_time=datetime.now(),
                config=session_config or {}
            )
            
            # Registrar sesión activa
            self.active_sessions[session_id] = session
            
            # Inicializar métricas de rendimiento
            self.performance_metrics[session_id] = PerformanceMetrics(
                session_id=session_id,
                start_time=datetime.now()
            )
            
            logger.info(
                "Sesión de análisis incremental iniciada",
                session_id=session_id,
                project_path=project_path
            )
            
            return session_id
            
        except Exception as e:
            logger.exception(
                "Error iniciando sesión de análisis incremental",
                project_path=project_path,
                error=str(e)
            )
            raise
    
    async def perform_incremental_analysis(
        self,
        session_id: str,
        force_full_analysis: bool = False,
        include_metrics: bool = True
    ) -> IncrementalAnalysisResponse:
        """
        Realiza análisis incremental en una sesión activa.
        
        Args:
            session_id: ID de la sesión
            force_full_analysis: Si forzar análisis completo
            include_metrics: Si incluir métricas detalladas
            
        Returns:
            Resultado del análisis incremental
            
        Raises:
            ValidationError: Si la sesión no existe
            BaseError: Si hay un error en el análisis
        """
        try:
            # Verificar sesión activa
            if session_id not in self.active_sessions:
                raise ValidationError(f"Sesión no encontrada: {session_id}")
            
            session = self.active_sessions[session_id]
            
            logger.info(
                "Realizando análisis incremental",
                session_id=session_id,
                project_path=session.project_path,
                force_full=force_full_analysis
            )
            
            # Crear request de análisis
            analysis_request = IncrementalAnalysisRequest(
                project_path=session.project_path,
                previous_analysis_id=session.last_analysis_id,
                force_full_analysis=force_full_analysis,
                include_metrics=include_metrics,
                cache_strategy=session.config.get("cache_strategy", "smart")
            )
            
            # Ejecutar análisis
            start_time = datetime.now()
            result = await self.incremental_analysis_use_case.execute(analysis_request)
            end_time = datetime.now()
            
            # Actualizar sesión
            session.last_analysis_id = result.analysis_id
            session.last_analysis_time = end_time
            session.analysis_count += 1
            
            # Actualizar métricas de rendimiento
            if session_id in self.performance_metrics:
                self.performance_metrics[session_id].add_analysis_time(
                    end_time - start_time
                )
                self.performance_metrics[session_id].increment_analysis_count()
            
            logger.info(
                "Análisis incremental completado",
                session_id=session_id,
                analysis_id=result.analysis_id,
                files_analyzed=result.files_analyzed,
                cache_hit_rate=result.cache_hit_rate
            )
            
            return result
            
        except Exception as e:
            logger.exception(
                "Error en análisis incremental",
                session_id=session_id,
                error=str(e)
            )
            raise
    
    async def detect_changes(
        self,
        session_id: str,
        previous_commit: Optional[str] = None,
        current_commit: Optional[str] = None
    ) -> ChangeDetectionResponse:
        """
        Detecta cambios en el código de una sesión.
        
        Args:
            session_id: ID de la sesión
            previous_commit: Commit anterior (opcional)
            current_commit: Commit actual (opcional)
            
        Returns:
            Resultado de la detección de cambios
            
        Raises:
            ValidationError: Si la sesión no existe
            BaseError: Si hay un error en la detección
        """
        try:
            # Verificar sesión activa
            if session_id not in self.active_sessions:
                raise ValidationError(f"Sesión no encontrada: {session_id}")
            
            session = self.active_sessions[session_id]
            
            logger.info(
                "Detectando cambios",
                session_id=session_id,
                project_path=session.project_path,
                previous_commit=previous_commit
            )
            
            # Crear request de detección
            detection_request = ChangeDetectionRequest(
                project_path=session.project_path,
                previous_commit=previous_commit,
                current_commit=current_commit,
                file_patterns=session.config.get("file_patterns")
            )
            
            # Ejecutar detección
            result = await self.change_detection_use_case.execute(detection_request)
            
            # Actualizar sesión con información de cambios
            session.last_change_detection = result
            session.change_count += len(result.changed_files)
            
            logger.info(
                "Detección de cambios completada",
                session_id=session_id,
                files_changed=len(result.changed_files),
                functions_changed=len(result.changed_functions)
            )
            
            return result
            
        except Exception as e:
            logger.exception(
                "Error en detección de cambios",
                session_id=session_id,
                error=str(e)
            )
            raise
    
    async def get_cache_status(self, session_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado del caché para una sesión.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Estado del caché
            
        Raises:
            ValidationError: Si la sesión no existe
        """
        try:
            # Verificar sesión activa
            if session_id not in self.active_sessions:
                raise ValidationError(f"Sesión no encontrada: {session_id}")
            
            session = self.active_sessions[session_id]
            
            logger.info(
                "Obteniendo estado del caché",
                session_id=session_id,
                project_path=session.project_path
            )
            
            # Obtener estado del caché
            cache_status = await self.cache_management_use_case.get_cache_status(
                session.project_path
            )
            
            return cache_status.to_dict()
            
        except Exception as e:
            logger.exception(
                "Error obteniendo estado del caché",
                session_id=session_id,
                error=str(e)
            )
            raise
    
    async def predict_cache_usage(
        self,
        session_id: str,
        analysis_history_days: int = 7,
        prediction_horizon_hours: int = 24
    ) -> PredictiveCacheResponse:
        """
        Predice el uso del caché para una sesión.
        
        Args:
            session_id: ID de la sesión
            analysis_history_days: Días de historial a considerar
            prediction_horizon_hours: Horizonte de predicción en horas
            
        Returns:
            Predicción del uso del caché
            
        Raises:
            ValidationError: Si la sesión no existe
            BaseError: Si hay un error en la predicción
        """
        try:
            # Verificar sesión activa
            if session_id not in self.active_sessions:
                raise ValidationError(f"Sesión no encontrada: {session_id}")
            
            session = self.active_sessions[session_id]
            
            logger.info(
                "Prediciendo uso del caché",
                session_id=session_id,
                project_path=session.project_path,
                history_days=analysis_history_days
            )
            
            # Crear request de predicción
            prediction_request = PredictiveCacheRequest(
                project_path=session.project_path,
                analysis_history_days=analysis_history_days,
                prediction_horizon_hours=prediction_horizon_hours
            )
            
            # Ejecutar predicción
            result = await self.predictive_cache_use_case.execute(prediction_request)
            
            logger.info(
                "Predicción de caché completada",
                session_id=session_id,
                predicted_files=len(result.predicted_files),
                confidence_score=result.confidence_score
            )
            
            return result
            
        except Exception as e:
            logger.exception(
                "Error en predicción de caché",
                session_id=session_id,
                error=str(e)
            )
            raise
    
    async def end_analysis_session(self, session_id: str) -> Dict[str, Any]:
        """
        Finaliza una sesión de análisis incremental.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Resumen de la sesión
            
        Raises:
            ValidationError: Si la sesión no existe
        """
        try:
            # Verificar sesión activa
            if session_id not in self.active_sessions:
                raise ValidationError(f"Sesión no encontrada: {session_id}")
            
            session = self.active_sessions[session_id]
            
            logger.info(
                "Finalizando sesión de análisis incremental",
                session_id=session_id,
                project_path=session.project_path
            )
            
            # Finalizar sesión
            session.end_time = datetime.now()
            session.duration = session.end_time - session.start_time
            
            # Obtener métricas finales
            final_metrics = self.performance_metrics.get(session_id)
            
            # Crear resumen de la sesión
            session_summary = {
                "session_id": session_id,
                "project_path": session.project_path,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat(),
                "duration_seconds": session.duration.total_seconds(),
                "analysis_count": session.analysis_count,
                "change_count": session.change_count,
                "last_analysis_id": session.last_analysis_id,
                "performance_metrics": final_metrics.to_dict() if final_metrics else None
            }
            
            # Limpiar sesión activa
            del self.active_sessions[session_id]
            if session_id in self.performance_metrics:
                del self.performance_metrics[session_id]
            
            logger.info(
                "Sesión de análisis incremental finalizada",
                session_id=session_id,
                duration_seconds=session_summary["duration_seconds"],
                analysis_count=session.analysis_count
            )
            
            return session_summary
            
        except Exception as e:
            logger.exception(
                "Error finalizando sesión de análisis incremental",
                session_id=session_id,
                error=str(e)
            )
            raise
    
    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Obtiene la lista de sesiones activas.
        
        Returns:
            Lista de sesiones activas
        """
        try:
            sessions_info = []
            
            for session_id, session in self.active_sessions.items():
                session_info = {
                    "session_id": session_id,
                    "project_path": session.project_path,
                    "start_time": session.start_time.isoformat(),
                    "analysis_count": session.analysis_count,
                    "change_count": session.change_count,
                    "last_analysis_id": session.last_analysis_id,
                    "is_active": True
                }
                
                # Agregar métricas de rendimiento si están disponibles
                if session_id in self.performance_metrics:
                    session_info["performance_metrics"] = self.performance_metrics[session_id].to_dict()
                
                sessions_info.append(session_info)
            
            return sessions_info
            
        except Exception as e:
            logger.exception("Error obteniendo sesiones activas", error=str(e))
            raise
    
    async def cleanup_expired_sessions(self, max_duration_hours: int = 24) -> int:
        """
        Limpia sesiones expiradas.
        
        Args:
            max_duration_hours: Duración máxima en horas
            
        Returns:
            Número de sesiones limpiadas
        """
        try:
            current_time = datetime.now()
            max_duration = timedelta(hours=max_duration_hours)
            expired_sessions = []
            
            # Identificar sesiones expiradas
            for session_id, session in self.active_sessions.items():
                if current_time - session.start_time > max_duration:
                    expired_sessions.append(session_id)
            
            # Limpiar sesiones expiradas
            for session_id in expired_sessions:
                await self.end_analysis_session(session_id)
            
            if expired_sessions:
                logger.info(
                    "Sesiones expiradas limpiadas",
                    count=len(expired_sessions),
                    session_ids=expired_sessions
                )
            
            return len(expired_sessions)
            
        except Exception as e:
            logger.exception("Error limpiando sesiones expiradas", error=str(e))
            raise
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas del sistema incremental.
        
        Returns:
            Métricas del sistema
        """
        try:
            total_sessions = len(self.active_sessions)
            total_analyses = sum(
                session.analysis_count for session in self.active_sessions.values()
            )
            total_changes = sum(
                session.change_count for session in self.active_sessions.values()
            )
            
            # Calcular métricas de rendimiento promedio
            avg_analysis_time = 0.0
            if self.performance_metrics:
                total_analysis_time = sum(
                    metrics.total_analysis_time.total_seconds()
                    for metrics in self.performance_metrics.values()
                )
                total_analysis_count = sum(
                    metrics.analysis_count
                    for metrics in self.performance_metrics.values()
                )
                if total_analysis_count > 0:
                    avg_analysis_time = total_analysis_time / total_analysis_count
            
            return {
                "active_sessions": total_sessions,
                "total_analyses": total_analyses,
                "total_changes": total_changes,
                "average_analysis_time_seconds": avg_analysis_time,
                "system_uptime": "N/A",  # TODO: Implementar tracking de uptime
                "cache_status": "N/A"    # TODO: Implementar estado global del caché
            }
            
        except Exception as e:
            logger.exception("Error obteniendo métricas del sistema", error=str(e))
            raise
