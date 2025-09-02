"""
Sistema de alertas y notificaciones.

Este módulo implementa:
- Alertas automáticas basadas en métricas
- Integración con Alertmanager
- Notificaciones por email, Slack, etc.
- Alertas de negocio
- Alertas de sistema
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import httpx

from ...config.settings import get_settings
from ...utils.logging import get_logger


class AlertSeverity(Enum):
    """Severidad de las alertas."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Estado de las alertas."""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Alert:
    """Definición de una alerta."""
    name: str
    severity: AlertSeverity
    message: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    starts_at: datetime = field(default_factory=datetime.utcnow)
    ends_at: Optional[datetime] = None
    status: AlertStatus = AlertStatus.FIRING
    fingerprint: Optional[str] = None


@dataclass
class AlertRule:
    """Regla de alerta."""
    name: str
    condition: Callable
    severity: AlertSeverity
    message: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    for_duration: timedelta = timedelta(minutes=5)
    check_interval: timedelta = timedelta(minutes=1)
    enabled: bool = True


class AlertingService:
    """Servicio de alertas y notificaciones."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def register_rule(self, rule: AlertRule) -> None:
        """Registra una regla de alerta."""
        self.rules[rule.name] = rule
        self.logger.info(f"Regla de alerta registrada: {rule.name}")
    
    def start_monitoring(self) -> None:
        """Inicia el monitoreo de alertas."""
        if self._running:
            return
        
        if not self.settings.telemetry.alerting_enabled:
            self.logger.info("Sistema de alertas deshabilitado")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_alerts())
        self.logger.info("Monitoreo de alertas iniciado")
    
    def stop_monitoring(self) -> None:
        """Detiene el monitoreo de alertas."""
        self._running = False
        if self._task:
            self._task.cancel()
        self.logger.info("Monitoreo de alertas detenido")
    
    async def _monitor_alerts(self) -> None:
        """Loop principal de monitoreo de alertas."""
        while self._running:
            try:
                await self._check_all_rules()
                await asyncio.sleep(60)  # Verificar cada minuto
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error en monitoreo de alertas: {e}")
                await asyncio.sleep(60)
    
    async def _check_all_rules(self) -> None:
        """Verifica todas las reglas de alerta."""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Verificar condición
                if asyncio.iscoroutinefunction(rule.condition):
                    should_fire = await rule.condition()
                else:
                    should_fire = rule.condition()
                
                if should_fire:
                    await self._fire_alert(rule)
                else:
                    await self._resolve_alert(rule_name)
                    
            except Exception as e:
                self.logger.error(f"Error verificando regla {rule_name}: {e}")
    
    async def _fire_alert(self, rule: AlertRule) -> None:
        """Dispara una alerta."""
        alert_name = rule.name
        
        # Verificar si ya existe una alerta activa
        if alert_name in self.active_alerts:
            return
        
        # Crear nueva alerta
        alert = Alert(
            name=alert_name,
            severity=rule.severity,
            message=rule.message,
            description=rule.description,
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy(),
            fingerprint=f"{alert_name}_{int(time.time())}"
        )
        
        # Agregar a alertas activas
        self.active_alerts[alert_name] = alert
        
        # Agregar al historial
        self.alert_history.append(alert)
        
        # Limpiar historial antiguo (mantener últimos 1000)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Enviar notificación
        await self._send_notification(alert)
        
        self.logger.warning(
            f"Alerta disparada: {alert_name}",
            severity=rule.severity.value,
            message=rule.message
        )
    
    async def _resolve_alert(self, alert_name: str) -> None:
        """Resuelve una alerta."""
        if alert_name not in self.active_alerts:
            return
        
        alert = self.active_alerts[alert_name]
        alert.status = AlertStatus.RESOLVED
        alert.ends_at = datetime.utcnow()
        
        # Remover de alertas activas
        del self.active_alerts[alert_name]
        
        # Enviar notificación de resolución
        await self._send_resolution_notification(alert)
        
        self.logger.info(f"Alerta resuelta: {alert_name}")
    
    async def _send_notification(self, alert: Alert) -> None:
        """Envía una notificación de alerta."""
        try:
            # Enviar a Alertmanager si está configurado
            if self.settings.telemetry.alertmanager_url:
                await self._send_to_alertmanager(alert)
            
            # Enviar notificaciones adicionales según severidad
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                await self._send_urgent_notification(alert)
                
        except Exception as e:
            self.logger.error(f"Error enviando notificación: {e}")
    
    async def _send_resolution_notification(self, alert: Alert) -> None:
        """Envía una notificación de resolución."""
        try:
            # Enviar a Alertmanager
            if self.settings.telemetry.alertmanager_url:
                await self._send_resolution_to_alertmanager(alert)
                
        except Exception as e:
            self.logger.error(f"Error enviando notificación de resolución: {e}")
    
    async def _send_to_alertmanager(self, alert: Alert) -> None:
        """Envía alerta a Alertmanager."""
        try:
            payload = {
                "alerts": [{
                    "labels": {
                        "alertname": alert.name,
                        "severity": alert.severity.value,
                        **alert.labels
                    },
                    "annotations": {
                        "message": alert.message,
                        "description": alert.description,
                        **alert.annotations
                    },
                    "startsAt": alert.starts_at.isoformat() + "Z",
                    "endsAt": alert.ends_at.isoformat() + "Z" if alert.ends_at else None,
                    "fingerprint": alert.fingerprint
                }]
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.settings.telemetry.alertmanager_url}/api/v1/alerts",
                    json=payload
                )
                response.raise_for_status()
                
        except Exception as e:
            self.logger.error(f"Error enviando a Alertmanager: {e}")
    
    async def _send_resolution_to_alertmanager(self, alert: Alert) -> None:
        """Envía resolución a Alertmanager."""
        try:
            payload = {
                "alerts": [{
                    "labels": {
                        "alertname": alert.name,
                        "severity": alert.severity.value,
                        **alert.labels
                    },
                    "annotations": {
                        "message": f"Resolved: {alert.message}",
                        "description": alert.description,
                        **alert.annotations
                    },
                    "startsAt": alert.starts_at.isoformat() + "Z",
                    "endsAt": alert.ends_at.isoformat() + "Z",
                    "fingerprint": alert.fingerprint
                }]
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.settings.telemetry.alertmanager_url}/api/v1/alerts",
                    json=payload
                )
                response.raise_for_status()
                
        except Exception as e:
            self.logger.error(f"Error enviando resolución a Alertmanager: {e}")
    
    async def _send_urgent_notification(self, alert: Alert) -> None:
        """Envía notificación urgente (email, Slack, etc.)."""
        # TODO: Implementar notificaciones urgentes
        self.logger.warning(f"Notificación urgente requerida: {alert.name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Retorna las alertas activas."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Retorna el historial de alertas."""
        return self.alert_history[-limit:]
    
    def acknowledge_alert(self, alert_name: str) -> bool:
        """Reconoce una alerta."""
        if alert_name not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_name]
        alert.status = AlertStatus.ACKNOWLEDGED
        return True
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Retorna un resumen de las alertas."""
        active_count = len(self.active_alerts)
        total_count = len(self.alert_history)
        
        severity_counts = {}
        for alert in self.active_alerts.values():
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "active_alerts": active_count,
            "total_alerts": total_count,
            "severity_breakdown": severity_counts,
            "last_check": datetime.utcnow().isoformat()
        }


# Reglas de alerta predefinidas
async def high_cpu_alert() -> bool:
    """Alerta por alto uso de CPU."""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent > 90
    except Exception:
        return False


async def high_memory_alert() -> bool:
    """Alerta por alto uso de memoria."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent > 90
    except Exception:
        return False


async def high_disk_alert() -> bool:
    """Alerta por alto uso de disco."""
    try:
        import psutil
        disk_usage = psutil.disk_usage('/')
        return disk_usage.percent > 90
    except Exception:
        return False


async def database_connection_alert() -> bool:
    """Alerta por problemas de conexión a la base de datos."""
    try:
        from ...infrastructure.database.connection_pool import DatabaseConnectionPool
        pool = DatabaseConnectionPool()
        
        # Verificar si hay conexiones disponibles
        if pool.pool.checkedout() >= pool.pool.size * 0.9:
            return True
        
        # Verificar conexión
        async with pool.get_connection() as conn:
            result = await conn.execute("SELECT 1")
            await result.fetchone()
        
        return False
    except Exception:
        return True


async def high_error_rate_alert() -> bool:
    """Alerta por alta tasa de errores."""
    try:
        from .metrics import get_metrics_service
        metrics_service = get_metrics_service()
        
        # Obtener métricas de errores (esto requeriría implementación específica)
        # Por ahora, retornar False
        return False
    except Exception:
        return False


# Instancia global
_alerting_service: Optional[AlertingService] = None


def get_alerting_service() -> AlertingService:
    """Retorna la instancia global del servicio de alertas."""
    global _alerting_service
    if _alerting_service is None:
        _alerting_service = AlertingService()
        
        # Registrar reglas de alerta básicas
        _alerting_service.register_rule(AlertRule(
            name="high_cpu_usage",
            condition=high_cpu_alert,
            severity=AlertSeverity.WARNING,
            message="High CPU usage detected",
            description="CPU usage is above 90%",
            labels={"component": "system"},
            for_duration=timedelta(minutes=2)
        ))
        
        _alerting_service.register_rule(AlertRule(
            name="high_memory_usage",
            condition=high_memory_alert,
            severity=AlertSeverity.WARNING,
            message="High memory usage detected",
            description="Memory usage is above 90%",
            labels={"component": "system"},
            for_duration=timedelta(minutes=2)
        ))
        
        _alerting_service.register_rule(AlertRule(
            name="high_disk_usage",
            condition=high_disk_alert,
            severity=AlertSeverity.WARNING,
            message="High disk usage detected",
            description="Disk usage is above 90%",
            labels={"component": "system"},
            for_duration=timedelta(minutes=2)
        ))
        
        _alerting_service.register_rule(AlertRule(
            name="database_connection_issues",
            condition=database_connection_alert,
            severity=AlertSeverity.CRITICAL,
            message="Database connection issues detected",
            description="Unable to connect to database or high connection usage",
            labels={"component": "database"},
            for_duration=timedelta(minutes=1)
        ))
        
        _alerting_service.register_rule(AlertRule(
            name="high_error_rate",
            condition=high_error_rate_alert,
            severity=AlertSeverity.CRITICAL,
            message="High error rate detected",
            description="Error rate is above threshold",
            labels={"component": "application"},
            for_duration=timedelta(minutes=5)
        ))
    
    return _alerting_service


def initialize_alerting() -> None:
    """Inicializa el sistema de alertas."""
    service = get_alerting_service()
    service.start_monitoring()
