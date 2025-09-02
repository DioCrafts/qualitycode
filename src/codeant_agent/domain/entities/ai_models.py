"""
Entidades del dominio para modelos de IA y análisis semántico.

Este módulo contiene todas las entidades que representan el sistema
de integración de IA, embeddings de código y análisis semántico.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from enum import Enum
import uuid

from ..value_objects.programming_language import ProgrammingLanguage


class DeviceType(Enum):
    """Tipos de dispositivo para ejecución de IA."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


class ModelType(Enum):
    """Tipos de modelos de IA."""
    CODE_BERT = "CodeBERT"
    CODE_T5 = "CodeT5"
    GRAPH_CODE_BERT = "GraphCodeBERT"
    UNIX_CODER = "UnixCoder"
    CODE_SEARCH_NET = "CodeSearchNet"
    PLBART = "PLBART"
    CUSTOM = "Custom"


class PoolingStrategy(Enum):
    """Estrategias de pooling para embeddings."""
    CLS = "cls"
    MEAN = "mean"
    MAX = "max"
    ATTENTION_WEIGHTED = "attention_weighted"


class AnalysisType(Enum):
    """Tipos de análisis de IA."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    PATTERN_DETECTION = "pattern_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    CODE_QUALITY_PREDICTION = "code_quality_prediction"
    INTENT_CLASSIFICATION = "intent_classification"
    CODE_SEARCH = "code_search"


class ModelStatus(Enum):
    """Estados de modelo de IA."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class AIConfig:
    """Configuración general del sistema de IA."""
    device_type: DeviceType = DeviceType.AUTO
    max_models_in_memory: int = 3
    model_cache_size_gb: float = 8.0
    inference_batch_size: int = 32
    max_sequence_length: int = 512
    enable_quantization: bool = False
    enable_model_parallelism: bool = False
    fallback_to_cpu: bool = True
    model_download_timeout_secs: int = 300
    enable_gpu_acceleration: bool = True
    memory_optimization: bool = True
    
    def get_effective_device(self) -> str:
        """Obtiene dispositivo efectivo a usar."""
        if self.device_type == DeviceType.AUTO:
            # En implementación real verificaría CUDA/MPS availability
            return "cpu"  # Fallback seguro
        return self.device_type.value


@dataclass
class ModelConfig:
    """Configuración de modelo específico."""
    model_id: str
    model_name: str
    model_type: ModelType
    huggingface_repo: str
    local_path: Optional[Path] = None
    supported_languages: List[ProgrammingLanguage] = field(default_factory=list)
    max_input_length: int = 512
    embedding_dimension: int = 768
    requires_preprocessing: bool = True
    quantization_config: Optional[Dict[str, Any]] = None
    memory_estimate_mb: float = 500.0
    
    def supports_language(self, language: ProgrammingLanguage) -> bool:
        """Verifica si el modelo soporta un lenguaje."""
        return language in self.supported_languages or not self.supported_languages
    
    def get_cache_key(self) -> str:
        """Obtiene clave de cache para el modelo."""
        return f"{self.model_id}_{self.model_type.value}"


@dataclass
class LoadedModel:
    """Modelo cargado en memoria."""
    model_id: str
    model_type: ModelType
    model: Any  # El modelo actual (torch, transformers, etc.)
    tokenizer: Any  # Tokenizer correspondiente
    config: ModelConfig
    status: ModelStatus = ModelStatus.LOADED
    loaded_at: datetime = field(default_factory=datetime.now)
    memory_usage_mb: float = 0.0
    inference_count: int = 0
    last_used_at: datetime = field(default_factory=datetime.now)
    
    def update_usage(self) -> None:
        """Actualiza estadísticas de uso."""
        self.inference_count += 1
        self.last_used_at = datetime.now()
    
    def get_age_minutes(self) -> float:
        """Obtiene edad del modelo en memoria."""
        return (datetime.now() - self.loaded_at).total_seconds() / 60.0
    
    def is_recently_used(self, threshold_minutes: float = 30.0) -> bool:
        """Verifica si se usó recientemente."""
        return (datetime.now() - self.last_used_at).total_seconds() / 60.0 < threshold_minutes


@dataclass
class EmbeddingConfig:
    """Configuración para generación de embeddings."""
    default_model: str = "microsoft/codebert-base"
    batch_size: int = 32
    max_code_length: int = 512
    enable_caching: bool = True
    normalize_embeddings: bool = True
    pooling_strategy: PoolingStrategy = PoolingStrategy.CLS
    language_specific_models: Dict[ProgrammingLanguage, str] = field(default_factory=dict)
    preprocessing_intensity: str = "normal"  # "light", "normal", "aggressive"
    
    def get_model_for_language(self, language: ProgrammingLanguage) -> str:
        """Obtiene modelo específico para lenguaje."""
        return self.language_specific_models.get(language, self.default_model)


@dataclass
class EmbeddingMetadata:
    """Metadatos de embedding generado."""
    code_length: int
    token_count: int
    preprocessing_applied: bool
    generation_time_ms: int
    model_version: str
    confidence_score: float = 1.0
    original_index: int = 0
    file_path: Optional[Path] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    
    def get_quality_score(self) -> float:
        """Obtiene score de calidad del embedding."""
        # Score basado en varios factores
        base_score = self.confidence_score
        
        # Penalizar si es muy corto o muy largo
        if self.code_length < 20:
            base_score *= 0.8
        elif self.code_length > 2000:
            base_score *= 0.9
        
        # Boost si tiene buen preprocessing
        if self.preprocessing_applied:
            base_score *= 1.1
        
        return min(1.0, base_score)


@dataclass
class CodeEmbedding:
    """Embedding de código generado por IA."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    code_snippet: str = ""
    language: ProgrammingLanguage = ProgrammingLanguage.UNKNOWN
    model_id: str = ""
    embedding_vector: List[float] = field(default_factory=list)
    metadata: EmbeddingMetadata = field(default_factory=lambda: EmbeddingMetadata(0, 0, False, 0, ""))
    created_at: datetime = field(default_factory=datetime.now)
    similarity_hash: Optional[str] = None
    
    def get_dimension(self) -> int:
        """Obtiene dimensión del embedding."""
        return len(self.embedding_vector)
    
    def is_valid(self) -> bool:
        """Verifica si el embedding es válido."""
        return (len(self.embedding_vector) > 0 and 
                self.metadata.code_length > 0 and
                self.model_id != "")
    
    def calculate_similarity_hash(self) -> str:
        """Calcula hash para detección rápida de similaridad."""
        import hashlib
        content = f"{self.code_snippet}_{self.language.value}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class SimilarityMatch:
    """Resultado de búsqueda de similaridad."""
    embedding_id: str
    similarity_score: float  # 0-1
    code_snippet: str
    language: ProgrammingLanguage
    metadata: Dict[str, Any] = field(default_factory=dict)
    match_type: str = "semantic"  # "semantic", "syntactic", "structural"
    
    def is_high_similarity(self, threshold: float = 0.8) -> bool:
        """Verifica si es alta similaridad."""
        return self.similarity_score >= threshold


@dataclass
class AIAnalysisResult:
    """Resultado del análisis con IA."""
    code_snippet: str
    language: ProgrammingLanguage
    analysis_type: AnalysisType
    model_used: str
    embeddings: List[CodeEmbedding] = field(default_factory=list)
    similarity_matches: List[SimilarityMatch] = field(default_factory=list)
    ai_detected_patterns: List['AIPattern'] = field(default_factory=list)
    anomalies: List['CodeAnomaly'] = field(default_factory=list)
    semantic_insights: List['SemanticInsight'] = field(default_factory=list)
    quality_predictions: List['QualityPrediction'] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    execution_time_ms: int = 0
    analyzed_at: datetime = field(default_factory=datetime.now)
    
    def get_overall_confidence(self) -> float:
        """Obtiene confidence general del análisis."""
        if not self.confidence_scores:
            return 0.5
        
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)
    
    def get_top_insights(self, limit: int = 5) -> List['SemanticInsight']:
        """Obtiene top insights por confidence."""
        return sorted(self.semantic_insights, key=lambda i: i.confidence, reverse=True)[:limit]


@dataclass
class AIPattern:
    """Patrón detectado por IA."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_name: str = ""
    pattern_type: str = "design_pattern"  # "design_pattern", "anti_pattern", "idiom"
    confidence: float = 0.0
    description: str = ""
    location_info: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    def is_anti_pattern(self) -> bool:
        """Verifica si es un anti-pattern."""
        return self.pattern_type == "anti_pattern"
    
    def is_confident(self, threshold: float = 0.7) -> bool:
        """Verifica si la detección es confiable."""
        return self.confidence >= threshold


@dataclass
class CodeAnomaly:
    """Anomalía detectada en código."""
    anomaly_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    anomaly_type: str = "unusual_pattern"  # "unusual_pattern", "complexity_spike", "style_deviation"
    severity: str = "medium"  # "low", "medium", "high"
    confidence: float = 0.0
    description: str = ""
    affected_code: str = ""
    potential_issues: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)
    
    def is_critical(self) -> bool:
        """Verifica si es anomalía crítica."""
        return self.severity == "high" and self.confidence > 0.8


@dataclass
class SemanticInsight:
    """Insight semántico del código."""
    insight_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    insight_type: str = "function_purpose"  # "function_purpose", "code_intent", "business_logic"
    description: str = ""
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    abstraction_level: str = "function"  # "statement", "function", "class", "module"
    
    def get_importance_score(self) -> float:
        """Obtiene score de importancia."""
        base_score = self.confidence
        
        # Boost para insights de nivel más alto
        level_multipliers = {
            "statement": 0.5,
            "function": 1.0,
            "class": 1.3,
            "module": 1.5
        }
        
        return base_score * level_multipliers.get(self.abstraction_level, 1.0)


@dataclass
class QualityPrediction:
    """Predicción de calidad usando IA."""
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_name: str = "maintainability"  # "maintainability", "readability", "complexity"
    predicted_score: float = 0.0  # 0-100
    confidence: float = 0.0
    model_reasoning: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    def is_reliable_prediction(self, threshold: float = 0.75) -> bool:
        """Verifica si la predicción es confiable."""
        return self.confidence >= threshold


@dataclass
class VectorStoreConfig:
    """Configuración para vector database."""
    qdrant_url: str = "http://localhost:6333"
    default_collection: str = "code_embeddings"
    vector_dimension: int = 768
    distance_metric: str = "cosine"  # "cosine", "euclidean", "dot"
    replication_factor: int = 1
    shard_number: int = 1
    enable_compression: bool = False
    timeout_seconds: int = 30
    
    def get_collection_name(self, language: ProgrammingLanguage) -> str:
        """Obtiene nombre de colección para lenguaje."""
        language_collections = {
            ProgrammingLanguage.PYTHON: "code_embeddings_python",
            ProgrammingLanguage.JAVASCRIPT: "code_embeddings_javascript", 
            ProgrammingLanguage.TYPESCRIPT: "code_embeddings_typescript",
            ProgrammingLanguage.RUST: "code_embeddings_rust",
            ProgrammingLanguage.JAVA: "code_embeddings_java",
            ProgrammingLanguage.GO: "code_embeddings_go"
        }
        
        return language_collections.get(language, self.default_collection)


@dataclass
class SearchResult:
    """Resultado de búsqueda en vector store."""
    embedding_id: str
    similarity_score: float
    code_snippet: str
    language: ProgrammingLanguage
    metadata: Dict[str, Any]
    ranking_position: int = 0
    
    def get_display_snippet(self, max_length: int = 100) -> str:
        """Obtiene snippet para display."""
        if len(self.code_snippet) <= max_length:
            return self.code_snippet
        return self.code_snippet[:max_length] + "..."


@dataclass
class BatchEmbeddingJob:
    """Job de generación de embeddings en batch."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    code_snippets: List[Tuple[str, ProgrammingLanguage]] = field(default_factory=list)
    model_id: str = ""
    status: str = "pending"  # "pending", "processing", "completed", "failed"
    progress_percentage: float = 0.0
    generated_embeddings: List[CodeEmbedding] = field(default_factory=list)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def get_total_snippets(self) -> int:
        """Obtiene total de snippets a procesar."""
        return len(self.code_snippets)
    
    def get_completed_count(self) -> int:
        """Obtiene cantidad completada."""
        return len(self.generated_embeddings)
    
    def update_progress(self) -> None:
        """Actualiza progreso del job."""
        total = self.get_total_snippets()
        if total > 0:
            self.progress_percentage = (self.get_completed_count() / total) * 100.0


@dataclass
class AIAnalysisConfig:
    """Configuración para análisis de IA."""
    enable_pattern_detection: bool = True
    enable_anomaly_detection: bool = True
    enable_semantic_analysis: bool = True
    enable_code_quality_prediction: bool = True
    confidence_threshold: float = 0.7
    batch_analysis_size: int = 32
    enable_cross_language_analysis: bool = False
    model_ensemble: bool = False
    cache_analysis_results: bool = True
    max_analysis_time_seconds: int = 30
    
    def should_use_ensemble(self) -> bool:
        """Verifica si debe usar ensemble de modelos."""
        return self.model_ensemble and self.confidence_threshold > 0.8


@dataclass
class ModelPerformanceMetrics:
    """Métricas de performance de modelos."""
    model_id: str
    total_inferences: int = 0
    average_inference_time_ms: float = 0.0
    total_inference_time_ms: int = 0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    error_count: int = 0
    successful_inferences: int = 0
    
    def get_success_rate(self) -> float:
        """Obtiene tasa de éxito."""
        if self.total_inferences == 0:
            return 1.0
        return self.successful_inferences / self.total_inferences
    
    def get_average_throughput(self) -> float:
        """Obtiene throughput promedio (inferences/segundo)."""
        if self.total_inference_time_ms == 0:
            return 0.0
        return (self.total_inferences * 1000.0) / self.total_inference_time_ms
    
    def update_inference_metrics(self, inference_time_ms: int, success: bool) -> None:
        """Actualiza métricas de inferencia."""
        self.total_inferences += 1
        self.total_inference_time_ms += inference_time_ms
        
        if success:
            self.successful_inferences += 1
        else:
            self.error_count += 1
        
        # Recalcular promedio
        self.average_inference_time_ms = self.total_inference_time_ms / self.total_inferences


@dataclass
class CodeAnalysisJob:
    """Job de análisis de código con IA."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    code_files: List[Path] = field(default_factory=list)
    analysis_types: List[AnalysisType] = field(default_factory=list)
    models_to_use: List[str] = field(default_factory=list)
    status: str = "queued"  # "queued", "running", "completed", "failed"
    progress: float = 0.0
    results: List[AIAnalysisResult] = field(default_factory=list)
    error_logs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def get_duration_seconds(self) -> float:
        """Obtiene duración del job."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return 0.0
    
    def add_error(self, error: str) -> None:
        """Añade error al job."""
        self.error_logs.append(f"{datetime.now().isoformat()}: {error}")
    
    def update_progress(self, completed_files: int) -> None:
        """Actualiza progreso del job."""
        total_files = len(self.code_files)
        if total_files > 0:
            self.progress = (completed_files / total_files) * 100.0


@dataclass
class SemanticSearchQuery:
    """Query para búsqueda semántica."""
    query_text: str
    query_embedding: Optional[List[float]] = None
    target_language: Optional[ProgrammingLanguage] = None
    similarity_threshold: float = 0.7
    max_results: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    
    def add_filter(self, key: str, value: Any) -> None:
        """Añade filtro a la búsqueda."""
        self.filters[key] = value


@dataclass
class SemanticSearchResult:
    """Resultado de búsqueda semántica."""
    query: SemanticSearchQuery
    matches: List[SimilarityMatch] = field(default_factory=list)
    total_candidates: int = 0
    search_time_ms: int = 0
    model_used: str = ""
    
    def get_best_match(self) -> Optional[SimilarityMatch]:
        """Obtiene mejor match."""
        return self.matches[0] if self.matches else None
    
    def get_high_confidence_matches(self, threshold: float = 0.8) -> List[SimilarityMatch]:
        """Obtiene matches de alta confianza."""
        return [match for match in self.matches if match.similarity_score >= threshold]


@dataclass
class AIModelCache:
    """Cache para modelos de IA."""
    cache_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_size_gb: float = 8.0
    current_size_gb: float = 0.0
    cached_models: Dict[str, LoadedModel] = field(default_factory=dict)
    access_times: Dict[str, datetime] = field(default_factory=dict)
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    
    def get_hit_rate(self) -> float:
        """Obtiene tasa de hit del cache."""
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return 0.0
        return self.hit_count / total_requests
    
    def get_cache_utilization(self) -> float:
        """Obtiene utilización del cache."""
        return (self.current_size_gb / self.max_size_gb) * 100.0
    
    def needs_eviction(self) -> bool:
        """Verifica si necesita eviction."""
        return self.current_size_gb >= self.max_size_gb * 0.9  # 90% threshold


@dataclass
class EmbeddingGenerationStats:
    """Estadísticas de generación de embeddings."""
    total_embeddings_generated: int = 0
    total_generation_time_ms: int = 0
    average_generation_time_ms: float = 0.0
    embeddings_per_second: float = 0.0
    batch_jobs_completed: int = 0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    
    def update_stats(self, generation_time_ms: int, batch_size: int = 1) -> None:
        """Actualiza estadísticas."""
        self.total_embeddings_generated += batch_size
        self.total_generation_time_ms += generation_time_ms
        
        # Recalcular promedios
        self.average_generation_time_ms = self.total_generation_time_ms / self.total_embeddings_generated
        self.embeddings_per_second = (self.total_embeddings_generated * 1000.0) / self.total_generation_time_ms
        
        if batch_size > 1:
            self.batch_jobs_completed += 1


@dataclass
class AISystemStatus:
    """Estado del sistema de IA."""
    system_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    models_loaded: Dict[str, ModelStatus] = field(default_factory=dict)
    vector_store_status: str = "unknown"  # "connected", "disconnected", "error"
    embedding_engine_status: str = "unknown"
    inference_engine_status: str = "unknown"
    total_memory_usage_mb: float = 0.0
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    system_health_score: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)
    
    def calculate_health_score(self) -> float:
        """Calcula score de salud del sistema."""
        health_factors = []
        
        # Factor 1: Modelos cargados exitosamente
        loaded_models = sum(1 for status in self.models_loaded.values() if status == ModelStatus.LOADED)
        total_models = len(self.models_loaded) if self.models_loaded else 1
        model_health = loaded_models / total_models
        health_factors.append(model_health)
        
        # Factor 2: Vector store conectado
        vs_health = 1.0 if self.vector_store_status == "connected" else 0.0
        health_factors.append(vs_health)
        
        # Factor 3: Engines funcionando
        engine_health = 0.0
        if self.embedding_engine_status == "ready":
            engine_health += 0.5
        if self.inference_engine_status == "ready":
            engine_health += 0.5
        health_factors.append(engine_health)
        
        # Factor 4: Tasa de éxito de jobs
        total_jobs = self.completed_jobs + self.failed_jobs
        if total_jobs > 0:
            job_success_rate = self.completed_jobs / total_jobs
        else:
            job_success_rate = 1.0
        health_factors.append(job_success_rate)
        
        # Score promedio
        self.system_health_score = sum(health_factors) / len(health_factors)
        self.last_health_check = datetime.now()
        
        return self.system_health_score
    
    def is_healthy(self, threshold: float = 0.8) -> bool:
        """Verifica si el sistema está saludable."""
        return self.calculate_health_score() >= threshold


@dataclass
class AIIntegrationConfig:
    """Configuración de integración de IA."""
    ai_config: AIConfig = field(default_factory=AIConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store_config: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    analysis_config: AIAnalysisConfig = field(default_factory=AIAnalysisConfig)
    enable_fallback_mode: bool = True
    fallback_similarity_threshold: float = 0.6
    max_concurrent_jobs: int = 5
    enable_performance_monitoring: bool = True
    
    def validate_config(self) -> List[str]:
        """Valida la configuración."""
        issues = []
        
        if self.ai_config.model_cache_size_gb <= 0:
            issues.append("Model cache size must be positive")
        
        if self.embedding_config.batch_size <= 0:
            issues.append("Batch size must be positive")
        
        if self.vector_store_config.vector_dimension <= 0:
            issues.append("Vector dimension must be positive")
        
        if self.analysis_config.confidence_threshold < 0 or self.analysis_config.confidence_threshold > 1:
            issues.append("Confidence threshold must be between 0 and 1")
        
        return issues


@dataclass
class CodeFragment:
    """Fragmento de código para análisis."""
    fragment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    code: str = ""
    language: ProgrammingLanguage = ProgrammingLanguage.UNKNOWN
    file_path: Optional[Path] = None
    start_line: int = 0
    end_line: int = 0
    fragment_type: str = "function"  # "function", "class", "method", "block"
    context: Dict[str, Any] = field(default_factory=dict)
    
    def get_lines_count(self) -> int:
        """Obtiene número de líneas."""
        return max(1, self.end_line - self.start_line + 1)
    
    def get_complexity_estimate(self) -> float:
        """Estima complejidad del fragmento."""
        # Estimación simple basada en longitud y keywords
        lines_factor = min(2.0, self.get_lines_count() / 20.0)
        
        # Contar palabras clave de complejidad
        complexity_keywords = ['if', 'for', 'while', 'try', 'catch', 'switch', 'case']
        keyword_count = sum(self.code.lower().count(keyword) for keyword in complexity_keywords)
        keyword_factor = min(2.0, keyword_count / 5.0)
        
        return (lines_factor + keyword_factor) / 2.0


@dataclass
class AIProcessingSession:
    """Sesión de procesamiento con IA."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_name: str = ""
    models_used: List[str] = field(default_factory=list)
    total_code_analyzed: int = 0
    total_embeddings_generated: int = 0
    total_processing_time_ms: int = 0
    cache_performance: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def complete_session(self) -> None:
        """Completa la sesión."""
        self.completed_at = datetime.now()
    
    def get_session_duration(self) -> float:
        """Obtiene duración de la sesión en segundos."""
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def get_processing_rate(self) -> float:
        """Obtiene tasa de procesamiento (archivos/segundo)."""
        duration = self.get_session_duration()
        if duration == 0:
            return 0.0
        return self.total_code_analyzed / duration


@dataclass
class CrossLanguageSimilarity:
    """Similaridad entre códigos de diferentes lenguajes."""
    source_code: str
    source_language: ProgrammingLanguage
    target_code: str
    target_language: ProgrammingLanguage
    similarity_score: float
    semantic_similarity: float
    structural_similarity: float
    functional_similarity: float
    model_used: str
    analysis_confidence: float
    
    def is_significant_similarity(self, threshold: float = 0.7) -> bool:
        """Verifica si hay similaridad significativa."""
        return self.similarity_score >= threshold and self.analysis_confidence >= 0.6
    
    def get_similarity_explanation(self) -> str:
        """Obtiene explicación de la similaridad."""
        explanations = []
        
        if self.semantic_similarity > 0.8:
            explanations.append("high semantic similarity")
        if self.structural_similarity > 0.8:
            explanations.append("similar code structure")
        if self.functional_similarity > 0.8:
            explanations.append("equivalent functionality")
        
        if not explanations:
            explanations.append("limited similarity")
        
        return f"Cross-language similarity: {', '.join(explanations)}"
