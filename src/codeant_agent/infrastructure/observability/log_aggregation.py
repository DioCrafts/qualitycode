"""
Sistema de agregación de logs.

Este módulo implementa:
- Envío de logs a Elasticsearch/OpenSearch
- Log parsing y indexing
- Log retention policies
- Log search capabilities
- Log-based alerting
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import httpx

from ...config.settings import get_settings
from ...utils.logging import get_logger


class LogLevel(Enum):
    """Niveles de log."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Entrada de log estructurada."""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class LogAggregationService:
    """Servicio de agregación de logs."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._log_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._batch_size = 100
        self._batch_timeout = 5.0  # segundos
        self._retention_days = 30
        self._index_pattern = "codeant-logs-{date}"
    
    def start_service(self) -> None:
        """Inicia el servicio de agregación de logs."""
        if self._running:
            return
        
        if not self.settings.telemetry.log_aggregation_enabled:
            self.logger.info("Log aggregation deshabilitado")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._process_logs())
        self.logger.info("Servicio de agregación de logs iniciado")
    
    def stop_service(self) -> None:
        """Detiene el servicio de agregación de logs."""
        self._running = False
        if self._task:
            self._task.cancel()
        self.logger.info("Servicio de agregación de logs detenido")
    
    async def send_log(self, log_entry: LogEntry) -> None:
        """Envía un log para agregación."""
        if not self._running:
            return
        
        try:
            await self._log_queue.put(log_entry)
        except asyncio.QueueFull:
            self.logger.warning("Cola de logs llena, descartando log")
    
    async def _process_logs(self) -> None:
        """Procesa logs en lotes."""
        while self._running:
            try:
                batch = []
                start_time = time.time()
                
                # Recolectar logs hasta llenar el lote o alcanzar timeout
                while len(batch) < self._batch_size and time.time() - start_time < self._batch_timeout:
                    try:
                        log_entry = await asyncio.wait_for(
                            self._log_queue.get(), 
                            timeout=1.0
                        )
                        batch.append(log_entry)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    await self._send_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error procesando logs: {e}")
                await asyncio.sleep(5)
    
    async def _send_batch(self, batch: List[LogEntry]) -> None:
        """Envía un lote de logs."""
        try:
            # Preparar documentos para Elasticsearch/OpenSearch
            documents = []
            for log_entry in batch:
                doc = self._prepare_document(log_entry)
                documents.append(doc)
            
            # Enviar a Elasticsearch/OpenSearch
            await self._send_to_elasticsearch(documents)
            
            self.logger.debug(f"Lote de {len(batch)} logs enviado")
            
        except Exception as e:
            self.logger.error(f"Error enviando lote de logs: {e}")
    
    def _prepare_document(self, log_entry: LogEntry) -> Dict[str, Any]:
        """Prepara un documento para Elasticsearch/OpenSearch."""
        # Crear índice basado en fecha
        index_name = self._get_index_name(log_entry.timestamp)
        
        # Preparar documento
        doc = {
            "@timestamp": log_entry.timestamp.isoformat(),
            "level": log_entry.level.value,
            "message": log_entry.message,
            "logger_name": log_entry.logger_name,
            "module": log_entry.module,
            "function": log_entry.function,
            "line_number": log_entry.line_number,
            "tags": log_entry.tags,
            "service": {
                "name": "codeant-agent",
                "version": self.settings.version,
                "environment": self.settings.environment
            }
        }
        
        # Agregar campos opcionales
        if log_entry.request_id:
            doc["request_id"] = log_entry.request_id
        
        if log_entry.user_id:
            doc["user_id"] = log_entry.user_id
        
        if log_entry.session_id:
            doc["session_id"] = log_entry.session_id
        
        # Agregar campos extra
        if log_entry.extra_fields:
            doc["extra"] = log_entry.extra_fields
        
        return {
            "index": index_name,
            "document": doc
        }
    
    def _get_index_name(self, timestamp: datetime) -> str:
        """Genera el nombre del índice basado en la fecha."""
        date_str = timestamp.strftime("%Y.%m.%d")
        return self._index_pattern.format(date=date_str)
    
    async def _send_to_elasticsearch(self, documents: List[Dict[str, Any]]) -> None:
        """Envía documentos a Elasticsearch/OpenSearch."""
        if not documents:
            return
        
        # Agrupar por índice
        index_groups = {}
        for doc in documents:
            index_name = doc["index"]
            if index_name not in index_groups:
                index_groups[index_name] = []
            index_groups[index_name].append(doc["document"])
        
        # Enviar cada grupo
        for index_name, docs in index_groups.items():
            await self._send_to_index(index_name, docs)
    
    async def _send_to_index(self, index_name: str, documents: List[Dict[str, Any]]) -> None:
        """Envía documentos a un índice específico."""
        try:
            # Usar OpenSearch si está configurado, sino Elasticsearch
            base_url = self.settings.telemetry.opensearch_url or self.settings.telemetry.elasticsearch_url
            
            if not base_url:
                self.logger.warning("No hay URL configurada para log aggregation")
                return
            
            # Preparar bulk request
            bulk_data = []
            for doc in documents:
                # Action
                bulk_data.append(json.dumps({"index": {"_index": index_name}}))
                # Document
                bulk_data.append(json.dumps(doc))
            
            bulk_body = "\n".join(bulk_data) + "\n"
            
            # Enviar request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{base_url}/_bulk",
                    content=bulk_body,
                    headers={"Content-Type": "application/x-ndjson"}
                )
                response.raise_for_status()
                
                # Verificar respuesta
                result = response.json()
                if result.get("errors"):
                    self.logger.error(f"Errores en bulk request: {result}")
                
        except Exception as e:
            self.logger.error(f"Error enviando a índice {index_name}: {e}")
    
    async def create_index_template(self) -> None:
        """Crea template de índice para logs."""
        try:
            base_url = self.settings.telemetry.opensearch_url or self.settings.telemetry.elasticsearch_url
            
            if not base_url:
                return
            
            template = {
                "index_patterns": ["codeant-logs-*"],
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "index": {
                        "refresh_interval": "5s",
                        "max_result_window": 10000
                    }
                },
                "mappings": {
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "level": {"type": "keyword"},
                        "message": {"type": "text"},
                        "logger_name": {"type": "keyword"},
                        "module": {"type": "keyword"},
                        "function": {"type": "keyword"},
                        "line_number": {"type": "integer"},
                        "request_id": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "session_id": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "service": {
                            "properties": {
                                "name": {"type": "keyword"},
                                "version": {"type": "keyword"},
                                "environment": {"type": "keyword"}
                            }
                        },
                        "extra": {"type": "object", "dynamic": True}
                    }
                }
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.put(
                    f"{base_url}/_template/codeant-logs-template",
                    json=template
                )
                response.raise_for_status()
                
            self.logger.info("Template de índice creado")
            
        except Exception as e:
            self.logger.error(f"Error creando template de índice: {e}")
    
    async def cleanup_old_logs(self) -> None:
        """Limpia logs antiguos según la política de retención."""
        try:
            base_url = self.settings.telemetry.opensearch_url or self.settings.telemetry.elasticsearch_url
            
            if not base_url:
                return
            
            # Calcular fecha de corte
            cutoff_date = datetime.utcnow() - timedelta(days=self._retention_days)
            cutoff_index = self._get_index_name(cutoff_date)
            
            # Obtener índices antiguos
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{base_url}/_cat/indices/codeant-logs-*?format=json")
                response.raise_for_status()
                
                indices = response.json()
                old_indices = []
                
                for index_info in indices:
                    index_name = index_info["index"]
                    if index_name < cutoff_index:
                        old_indices.append(index_name)
                
                # Eliminar índices antiguos
                if old_indices:
                    for index_name in old_indices:
                        try:
                            delete_response = await client.delete(f"{base_url}/{index_name}")
                            delete_response.raise_for_status()
                            self.logger.info(f"Índice eliminado: {index_name}")
                        except Exception as e:
                            self.logger.error(f"Error eliminando índice {index_name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error limpiando logs antiguos: {e}")
    
    async def search_logs(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[LogLevel] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Busca logs en Elasticsearch/OpenSearch."""
        try:
            base_url = self.settings.telemetry.opensearch_url or self.settings.telemetry.elasticsearch_url
            
            if not base_url:
                return []
            
            # Construir query
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"query_string": {"query": query}}
                        ],
                        "filter": []
                    }
                },
                "sort": [{"@timestamp": {"order": "desc"}}],
                "size": limit
            }
            
            # Agregar filtros de tiempo
            if start_time or end_time:
                time_filter = {}
                if start_time:
                    time_filter["gte"] = start_time.isoformat()
                if end_time:
                    time_filter["lte"] = end_time.isoformat()
                
                search_query["query"]["bool"]["filter"].append({
                    "range": {"@timestamp": time_filter}
                })
            
            # Agregar filtro de nivel
            if level:
                search_query["query"]["bool"]["filter"].append({
                    "term": {"level": level.value}
                })
            
            # Ejecutar búsqueda
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{base_url}/codeant-logs-*/_search",
                    json=search_query
                )
                response.raise_for_status()
                
                result = response.json()
                hits = result.get("hits", {}).get("hits", [])
                
                return [hit["_source"] for hit in hits]
                
        except Exception as e:
            self.logger.error(f"Error buscando logs: {e}")
            return []
    
    def get_service_status(self) -> Dict[str, Any]:
        """Retorna el estado del servicio."""
        return {
            "running": self._running,
            "queue_size": self._log_queue.qsize(),
            "batch_size": self._batch_size,
            "retention_days": self._retention_days,
            "index_pattern": self._index_pattern
        }


# Instancia global
_log_aggregation_service: Optional[LogAggregationService] = None


def get_log_aggregation_service() -> LogAggregationService:
    """Retorna la instancia global del servicio de agregación de logs."""
    global _log_aggregation_service
    if _log_aggregation_service is None:
        _log_aggregation_service = LogAggregationService()
    return _log_aggregation_service


def initialize_log_aggregation() -> None:
    """Inicializa el servicio de agregación de logs."""
    service = get_log_aggregation_service()
    service.start_service()
    
    # Crear template de índice
    asyncio.create_task(service.create_index_template())
    
    # Programar limpieza de logs antiguos (diariamente)
    async def cleanup_scheduler():
        while True:
            try:
                await service.cleanup_old_logs()
                await asyncio.sleep(24 * 60 * 60)  # 24 horas
            except Exception as e:
                service.logger.error(f"Error en cleanup scheduler: {e}")
                await asyncio.sleep(60 * 60)  # 1 hora
    
    asyncio.create_task(cleanup_scheduler())
