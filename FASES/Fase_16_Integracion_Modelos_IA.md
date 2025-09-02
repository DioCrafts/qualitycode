# Fase 16: Integración de Modelos de IA Pre-entrenados (CodeBERT)

## Objetivo General
Integrar modelos de inteligencia artificial pre-entrenados especializados en código (CodeBERT, CodeT5, GraphCodeBERT) para proporcionar capacidades avanzadas de comprensión semántica de código, generación de embeddings, análisis de similitud basado en IA, y detección de patrones complejos que van más allá de las reglas estáticas tradicionales.

## Descripción Técnica Detallada

### 16.1 Arquitectura del Sistema de IA

#### 16.1.1 Diseño del AI Integration System
```
┌─────────────────────────────────────────┐
│           AI Integration Layer          │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │  CodeBERT   │ │     CodeT5          │ │
│  │  Models     │ │    Models           │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ Embedding   │ │   Inference         │ │
│  │  Engine     │ │    Engine           │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Model     │ │    Vector           │ │
│  │  Manager    │ │   Database          │ │
│  └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
```

#### 16.1.2 Modelos de IA Integrados
- **CodeBERT**: Embeddings y comprensión semántica de código
- **CodeT5**: Generación de código y traducciones
- **GraphCodeBERT**: Análisis de grafos de código
- **UnixCoder**: Análisis cross-modal (código + comentarios)
- **CodeSearchNet**: Búsqueda semántica de código
- **PLBART**: Análisis multi-lenguaje avanzado

### 16.2 Model Management System

#### 16.2.1 AI Model Manager
# Fase 16: Integración de Modelos de IA Pre-entrenados (CodeBERT)
# Implementación en Python

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import hashlib
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, BertModel, T5Model, 
    RobertaModel, AutoConfig
)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue, SearchRequest
)
from pydantic import BaseModel
import logging
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 16.1 Model Management System
# ============================================================================

class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"

class ModelType(Enum):
    CODE_BERT = "CodeBERT"
    CODE_T5 = "CodeT5"
    GRAPH_CODE_BERT = "GraphCodeBERT"
    UNIX_CODER = "UnixCoder"
    CODE_SEARCH_NET = "CodeSearchNet"
    PLBART = "PLBART"
    CUSTOM = "Custom"

class ProgrammingLanguage(Enum):
    PYTHON = "Python"
    JAVASCRIPT = "JavaScript"
    TYPESCRIPT = "TypeScript"
    JAVA = "Java"
    GO = "Go"
    RUST = "Rust"
    CPP = "C++"
    CSHARP = "C#"
    UNKNOWN = "Unknown"

@dataclass
class AIConfig:
    device_type: DeviceType = DeviceType.AUTO
    max_models_in_memory: int = 3
    model_cache_size_gb: float = 8.0
    inference_batch_size: int = 32
    max_sequence_length: int = 512
    enable_quantization: bool = False
    enable_model_parallelism: bool = False
    fallback_to_cpu: bool = True
    model_download_timeout_secs: int = 300

@dataclass
class ModelConfig:
    model_id: str
    model_name: str
    model_type: ModelType
    huggingface_repo: str
    local_path: Optional[Path] = None
    supported_languages: List[ProgrammingLanguage] = field(default_factory=list)
    max_input_length: int = 512
    embedding_dimension: int = 768
    requires_preprocessing: bool = True
    quantization_config: Optional[Dict] = None

@dataclass
class LoadedModel:
    model_id: str
    model_type: ModelType
    model: Any  # The actual model object
    tokenizer: Any
    config: ModelConfig
    loaded_at: datetime
    memory_usage_mb: float

class ModelCache:
    def __init__(self, cache_size_gb: float):
        self.cache_size_gb = cache_size_gb
        self.cache: Dict[str, LoadedModel] = {}
        self.access_times: Dict[str, datetime] = {}
        
    async def get(self, model_id: str) -> Optional[LoadedModel]:
        if model_id in self.cache:
            self.access_times[model_id] = datetime.now(timezone.utc)
            return self.cache[model_id]
        return None
    
    async def insert(self, model_id: str, model: LoadedModel):
        # Check cache size and evict if necessary
        await self._evict_if_needed()
        self.cache[model_id] = model
        self.access_times[model_id] = datetime.now(timezone.utc)
    
    async def _evict_if_needed(self):
        # Calculate current cache size
        current_size_mb = sum(m.memory_usage_mb for m in self.cache.values())
        max_size_mb = self.cache_size_gb * 1024
        
        if current_size_mb >= max_size_mb:
            # Evict least recently used
            lru_model_id = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_model_id]
            del self.access_times[lru_model_id]

class AIModelManager:
    def __init__(self, config: AIConfig):
        self.config = config
        self.models: Dict[str, LoadedModel] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.device = self._initialize_device()
        self.model_cache = ModelCache(config.model_cache_size_gb)
        self.inference_engine = None  # Will be initialized later
        
    def _initialize_device(self) -> torch.device:
        if self.config.device_type == DeviceType.AUTO:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device_type.value)
    
    async def initialize(self):
        """Initialize the model manager and load default configurations"""
        await self.load_default_model_configs()
        await self.preload_essential_models()
        self.inference_engine = InferenceEngine(self.device)
    
    async def load_default_model_configs(self):
        """Load default model configurations"""
        # CodeBERT configuration
        self.model_configs["microsoft/codebert-base"] = ModelConfig(
            model_id="microsoft/codebert-base",
            model_name="CodeBERT Base",
            model_type=ModelType.CODE_BERT,
            huggingface_repo="microsoft/codebert-base",
            supported_languages=[
                ProgrammingLanguage.PYTHON,
                ProgrammingLanguage.JAVASCRIPT,
                ProgrammingLanguage.TYPESCRIPT,
                ProgrammingLanguage.JAVA,
                ProgrammingLanguage.GO,
                ProgrammingLanguage.RUST,
            ],
            max_input_length=512,
            embedding_dimension=768,
            requires_preprocessing=True
        )
        
        # CodeT5 configuration
        self.model_configs["Salesforce/codet5-small"] = ModelConfig(
            model_id="Salesforce/codet5-small",
            model_name="CodeT5 Small",
            model_type=ModelType.CODE_T5,
            huggingface_repo="Salesforce/codet5-small",
            supported_languages=[
                ProgrammingLanguage.PYTHON,
                ProgrammingLanguage.JAVASCRIPT,
                ProgrammingLanguage.JAVA,
                ProgrammingLanguage.GO,
            ],
            max_input_length=512,
            embedding_dimension=512,
            requires_preprocessing=True
        )
        
        # GraphCodeBERT configuration
        self.model_configs["microsoft/graphcodebert-base"] = ModelConfig(
            model_id="microsoft/graphcodebert-base",
            model_name="GraphCodeBERT Base",
            model_type=ModelType.GRAPH_CODE_BERT,
            huggingface_repo="microsoft/graphcodebert-base",
            supported_languages=[
                ProgrammingLanguage.PYTHON,
                ProgrammingLanguage.JAVASCRIPT,
                ProgrammingLanguage.JAVA,
                ProgrammingLanguage.GO,
            ],
            max_input_length=512,
            embedding_dimension=768,
            requires_preprocessing=True
        )
    
    async def load_model(self, model_id: str) -> LoadedModel:
        """Load a model by its ID"""
        # Check if already loaded
        if model_id in self.models:
            return self.models[model_id]
        
        # Check cache
        cached_model = await self.model_cache.get(model_id)
        if cached_model:
            self.models[model_id] = cached_model
            return cached_model
        
        # Get model configuration
        if model_id not in self.model_configs:
            raise ValueError(f"Model not configured: {model_id}")
        
        model_config = self.model_configs[model_id]
        
        # Download and load model
        loaded_model = await self.download_and_load_model(model_config)
        
        # Cache the model
        await self.model_cache.insert(model_id, loaded_model)
        
        # Store in memory
        self.models[model_id] = loaded_model
        
        return loaded_model
    
    async def download_and_load_model(self, config: ModelConfig) -> LoadedModel:
        """Download and load a model from HuggingFace or local path"""
        logger.info(f"Loading model: {config.model_name}")
        
        if config.model_type == ModelType.CODE_BERT:
            return await self.load_codebert_model(config)
        elif config.model_type == ModelType.CODE_T5:
            return await self.load_codet5_model(config)
        elif config.model_type == ModelType.GRAPH_CODE_BERT:
            return await self.load_graphcodebert_model(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    async def load_codebert_model(self, config: ModelConfig) -> LoadedModel:
        """Load a CodeBERT model"""
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.huggingface_repo)
        
        # Load model
        model = RobertaModel.from_pretrained(config.huggingface_repo)
        model = model.to(self.device)
        model.eval()
        
        # Calculate memory usage
        memory_usage_mb = self.calculate_model_memory_usage(model)
        
        return LoadedModel(
            model_id=config.model_id,
            model_type=config.model_type,
            model=model,
            tokenizer=tokenizer,
            config=config,
            loaded_at=datetime.now(timezone.utc),
            memory_usage_mb=memory_usage_mb
        )
    
    async def load_codet5_model(self, config: ModelConfig) -> LoadedModel:
        """Load a CodeT5 model"""
        tokenizer = AutoTokenizer.from_pretrained(config.huggingface_repo)
        model = T5Model.from_pretrained(config.huggingface_repo)
        model = model.to(self.device)
        model.eval()
        
        memory_usage_mb = self.calculate_model_memory_usage(model)
        
        return LoadedModel(
            model_id=config.model_id,
            model_type=config.model_type,
            model=model,
            tokenizer=tokenizer,
            config=config,
            loaded_at=datetime.now(timezone.utc),
            memory_usage_mb=memory_usage_mb
        )
    
    async def load_graphcodebert_model(self, config: ModelConfig) -> LoadedModel:
        """Load a GraphCodeBERT model"""
        tokenizer = AutoTokenizer.from_pretrained(config.huggingface_repo)
        model = RobertaModel.from_pretrained(config.huggingface_repo)
        model = model.to(self.device)
        model.eval()
        
        memory_usage_mb = self.calculate_model_memory_usage(model)
        
        return LoadedModel(
            model_id=config.model_id,
            model_type=config.model_type,
            model=model,
            tokenizer=tokenizer,
            config=config,
            loaded_at=datetime.now(timezone.utc),
            memory_usage_mb=memory_usage_mb
        )
    
    async def preload_essential_models(self):
        """Preload essential models for basic operations"""
        # Load CodeBERT for basic code understanding
        await self.load_model("microsoft/codebert-base")
        
        # Load CodeT5 for code generation
        await self.load_model("Salesforce/codet5-small")
        
        logger.info("Essential AI models preloaded successfully")
    
    def calculate_model_memory_usage(self, model: nn.Module) -> float:
        """Calculate approximate memory usage of a model in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

# ============================================================================
# 16.2 Code Embedding System
# ============================================================================

class PoolingStrategy(Enum):
    CLS = "cls"
    MEAN = "mean"
    MAX = "max"
    ATTENTION_WEIGHTED = "attention_weighted"

@dataclass
class EmbeddingConfig:
    default_model: str = "microsoft/codebert-base"
    batch_size: int = 32
    max_code_length: int = 512
    enable_caching: bool = True
    normalize_embeddings: bool = True
    pooling_strategy: PoolingStrategy = PoolingStrategy.CLS
    language_specific_models: Dict[ProgrammingLanguage, str] = field(default_factory=dict)

@dataclass
class EmbeddingMetadata:
    code_length: int
    token_count: int
    preprocessing_applied: bool
    generation_time_ms: int
    model_version: str
    original_index: int = 0
    file_path: Optional[Path] = None
    function_name: Optional[str] = None

@dataclass
class CodeEmbedding:
    id: str
    code_snippet: str
    language: ProgrammingLanguage
    model_id: str
    embedding_vector: List[float]
    metadata: EmbeddingMetadata
    created_at: datetime

class CodeEmbeddingEngine:
    def __init__(self, model_manager: AIModelManager, config: EmbeddingConfig):
        self.model_manager = model_manager
        self.config = config
        self.preprocessor = CodePreprocessor()
        self.postprocessor = EmbeddingPostprocessor()
        self.vector_store = None  # Will be initialized separately
        
    async def initialize(self):
        """Initialize the embedding engine"""
        self.vector_store = await VectorStore.create(VectorStoreConfig())
    
    async def generate_code_embedding(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> CodeEmbedding:
        """Generate embedding for a code snippet"""
        start_time = time.time()
        
        # Select appropriate model for language
        model_id = self.select_model_for_language(language)
        model = await self.model_manager.load_model(model_id)
        
        # Preprocess code
        preprocessed_code = await self.preprocessor.preprocess(code, language)
        
        # Tokenize
        tokens = model.tokenizer(
            preprocessed_code,
            padding=True,
            truncation=True,
            max_length=self.config.max_code_length,
            return_tensors="pt"
        )
        
        # Move to device
        tokens = {k: v.to(self.model_manager.device) for k, v in tokens.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = model.model(**tokens)
            
            # Apply pooling strategy
            if self.config.pooling_strategy == PoolingStrategy.CLS:
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif self.config.pooling_strategy == PoolingStrategy.MEAN:
                attention_mask = tokens['attention_mask'].unsqueeze(-1)
                embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
            elif self.config.pooling_strategy == PoolingStrategy.MAX:
                embeddings = outputs.last_hidden_state.max(1)[0]
            else:  # ATTENTION_WEIGHTED
                embeddings = await self.attention_weighted_pooling(
                    outputs.last_hidden_state, 
                    tokens['attention_mask']
                )
        
        # Convert to list
        embedding_vector = embeddings.squeeze().cpu().numpy().tolist()
        
        # Normalize if configured
        if self.config.normalize_embeddings:
            embedding_vector = self.normalize_vector(embedding_vector)
        
        # Create embedding object
        embedding_id = hashlib.md5(code.encode()).hexdigest()
        generation_time = int((time.time() - start_time) * 1000)
        
        code_embedding = CodeEmbedding(
            id=embedding_id,
            code_snippet=code,
            language=language,
            model_id=model_id,
            embedding_vector=embedding_vector,
            metadata=EmbeddingMetadata(
                code_length=len(code),
                token_count=tokens['input_ids'].shape[1],
                preprocessing_applied=True,
                generation_time_ms=generation_time,
                model_version=model.config.model_name
            ),
            created_at=datetime.now(timezone.utc)
        )
        
        # Store in vector database if caching is enabled
        if self.config.enable_caching and self.vector_store:
            await self.vector_store.store_embedding(code_embedding)
        
        return code_embedding
    
    async def batch_generate_embeddings(
        self, 
        code_snippets: List[Tuple[str, ProgrammingLanguage]]
    ) -> List[CodeEmbedding]:
        """Generate embeddings for multiple code snippets"""
        embeddings = []
        
        # Group by language for efficient batching
        language_groups: Dict[ProgrammingLanguage, List[Tuple[int, str]]] = {}
        for idx, (code, lang) in enumerate(code_snippets):
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append((idx, code))
        
        # Process each language group
        for language, group in language_groups.items():
            model_id = self.select_model_for_language(language)
            model = await self.model_manager.load_model(model_id)
            
            # Process in batches
            for i in range(0, len(group), self.config.batch_size):
                batch = group[i:i + self.config.batch_size]
                batch_embeddings = await self.generate_batch_embeddings_internal(
                    batch, language, model
                )
                embeddings.extend(batch_embeddings)
        
        # Sort back to original order
        embeddings.sort(key=lambda e: e.metadata.original_index)
        
        return embeddings
    
    async def generate_batch_embeddings_internal(
        self,
        batch: List[Tuple[int, str]],
        language: ProgrammingLanguage,
        model: LoadedModel
    ) -> List[CodeEmbedding]:
        """Generate embeddings for a batch of code snippets"""
        embeddings = []
        
        # Preprocess all codes in batch
        preprocessed_codes = []
        for _, code in batch:
            preprocessed = await self.preprocessor.preprocess(code, language)
            preprocessed_codes.append(preprocessed)
        
        # Tokenize batch
        tokens = model.tokenizer(
            preprocessed_codes,
            padding=True,
            truncation=True,
            max_length=self.config.max_code_length,
            return_tensors="pt"
        )
        
        # Move to device
        tokens = {k: v.to(self.model_manager.device) for k, v in tokens.items()}
        
        # Generate embeddings for batch
        with torch.no_grad():
            outputs = model.model(**tokens)
            
            # Apply pooling
            if self.config.pooling_strategy == PoolingStrategy.CLS:
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
            elif self.config.pooling_strategy == PoolingStrategy.MEAN:
                attention_mask = tokens['attention_mask'].unsqueeze(-1)
                batch_embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
            else:
                batch_embeddings = outputs.last_hidden_state.max(1)[0]
        
        # Create CodeEmbedding objects
        for i, ((original_idx, original_code), embedding) in enumerate(
            zip(batch, batch_embeddings)
        ):
            embedding_vector = embedding.cpu().numpy().tolist()
            
            if self.config.normalize_embeddings:
                embedding_vector = self.normalize_vector(embedding_vector)
            
            code_embedding = CodeEmbedding(
                id=hashlib.md5(original_code.encode()).hexdigest(),
                code_snippet=original_code,
                language=language,
                model_id=model.model_id,
                embedding_vector=embedding_vector,
                metadata=EmbeddingMetadata(
                    code_length=len(original_code),
                    token_count=tokens['input_ids'].shape[1],
                    preprocessing_applied=True,
                    generation_time_ms=0,
                    model_version=model.config.model_name,
                    original_index=original_idx
                ),
                created_at=datetime.now(timezone.utc)
            )
            
            embeddings.append(code_embedding)
        
        return embeddings
    
    async def attention_weighted_pooling(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply attention-weighted pooling to hidden states"""
        # Simple attention mechanism
        attention_scores = torch.softmax(hidden_states.mean(dim=-1), dim=-1)
        attention_scores = attention_scores * attention_mask
        attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)
        
        weighted_embeddings = (hidden_states * attention_scores.unsqueeze(-1)).sum(dim=1)
        return weighted_embeddings
    
    def select_model_for_language(self, language: ProgrammingLanguage) -> str:
        """Select the best model for a given programming language"""
        if language in self.config.language_specific_models:
            return self.config.language_specific_models[language]
        return self.config.default_model
    
    def normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize a vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return (np.array(vector) / norm).tolist()

class EmbeddingPostprocessor:
    """Post-process embeddings after generation"""
    
    async def process(self, embedding: List[float], config: EmbeddingConfig) -> List[float]:
        """Post-process an embedding vector"""
        # Placeholder for additional post-processing
        return embedding

# ============================================================================
# 16.3 Inference Engine
# ============================================================================

class InferenceEngine:
    def __init__(self, device: torch.device):
        self.device = device
        self.batch_processor = BatchProcessor()
        self.memory_manager = MemoryManager()
    
    async def generate_embedding(
        self, 
        model: Any, 
        tokens: Dict[str, torch.Tensor]
    ) -> List[float]:
        """Generate embedding using a model"""
        with torch.no_grad():
            outputs = model(**tokens)
            # Use CLS token by default
            embeddings = outputs.last_hidden_state[:, 0, :]
            return embeddings.squeeze().cpu().numpy().tolist()
    
    async def generate_batch_embeddings(
        self,
        model: Any,
        token_batches: List[Dict[str, torch.Tensor]]
    ) -> List[List[float]]:
        """Generate embeddings for multiple token batches"""
        all_embeddings = []
        
        for tokens in token_batches:
            embedding = await self.generate_embedding(model, tokens)
            all_embeddings.append(embedding)
        
        return all_embeddings
    
    def calculate_cosine_similarity(
        self, 
        vec1: List[float], 
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class BatchProcessor:
    """Process data in batches for efficiency"""
    
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size

class MemoryManager:
    """Manage memory usage during inference"""
    
    def __init__(self):
        self.memory_threshold_mb = 1024  # 1GB threshold

# ============================================================================
# 16.4 Code Preprocessing System
# ============================================================================

class LanguagePreprocessor(ABC):
    """Abstract base class for language-specific preprocessors"""
    
    @abstractmethod
    async def preprocess(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_language(self) -> ProgrammingLanguage:
        pass
    
    @abstractmethod
    def normalize_identifiers(self, code: str) -> str:
        pass
    
    @abstractmethod
    def remove_language_specific_noise(self, code: str) -> str:
        pass
    
    @abstractmethod
    def extract_semantic_structure(self, code: str) -> str:
        pass

class PythonPreprocessor(LanguagePreprocessor):
    """Preprocessor for Python code"""
    
    def __init__(self):
        self.python_keywords = {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
            'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 
            'except', 'finally', 'for', 'from', 'global', 'if', 'import',
            'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise',
            'return', 'try', 'while', 'with', 'yield'
        }
    
    async def preprocess(self, code: str) -> str:
        """Preprocess Python code"""
        processed = code
        
        # Remove docstrings
        processed = self.remove_docstrings(processed)
        
        # Normalize string literals
        processed = self.normalize_string_literals(processed)
        
        # Normalize numeric literals
        processed = self.normalize_numeric_literals(processed)
        
        # Remove type hints
        processed = self.remove_type_hints(processed)
        
        # Normalize indentation
        processed = self.normalize_indentation(processed)
        
        # Remove comments
        processed = self.remove_comments(processed)
        
        return processed
    
    def get_language(self) -> ProgrammingLanguage:
        return ProgrammingLanguage.PYTHON
    
    def normalize_identifiers(self, code: str) -> str:
        """Replace identifiers with generic placeholders"""
        identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        identifier_map = {}
        counter = 0
        
        def replace_identifier(match):
            nonlocal counter
            identifier = match.group(0)
            
            if identifier in self.python_keywords:
                return identifier
            
            if identifier not in identifier_map:
                counter += 1
                identifier_map[identifier] = f'VAR_{counter}'
            
            return identifier_map[identifier]
        
        return re.sub(identifier_pattern, replace_identifier, code)
    
    def remove_language_specific_noise(self, code: str) -> str:
        """Remove Python-specific noise"""
        # Remove decorators
        code = re.sub(r'@\w+.*\n', '', code)
        
        # Remove import statements
        code = re.sub(r'^(import|from)\s+.*\n', '', code, flags=re.MULTILINE)
        
        return code
    
    def extract_semantic_structure(self, code: str) -> str:
        """Extract semantic structure from Python code"""
        structure = []
        
        for line in code.split('\n'):
            trimmed = line.strip()
            
            if trimmed.startswith(('if ', 'elif ', 'else:')):
                structure.append('CONDITIONAL')
            elif trimmed.startswith(('for ', 'while ')):
                structure.append('LOOP')
            elif trimmed.startswith('def '):
                structure.append('FUNCTION_DEF')
            elif trimmed.startswith('class '):
                structure.append('CLASS_DEF')
            elif trimmed.startswith('return '):
                structure.append('RETURN')
            elif trimmed.startswith(('raise ', 'except ')):
                structure.append('EXCEPTION')
            elif '=' in trimmed and '==' not in trimmed:
                structure.append('ASSIGNMENT')
            elif trimmed and not trimmed.startswith('#'):
                structure.append('STATEMENT')
        
        return '\n'.join(structure)
    
    def remove_docstrings(self, code: str) -> str:
        """Remove docstrings from Python code"""
        # Remove triple-quoted strings
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        return code
    
    def normalize_string_literals(self, code: str) -> str:
        """Normalize string literals to a placeholder"""
        code = re.sub(r'"[^"]*"', '"STRING"', code)
        code = re.sub(r"'[^']*'", '"STRING"', code)
        return code
    
    def normalize_numeric_literals(self, code: str) -> str:
        """Normalize numeric literals to a placeholder"""
        return re.sub(r'\b\d+\.?\d*\b', 'NUM', code)
    
    def remove_type_hints(self, code: str) -> str:
        """Remove type hints from Python code"""
        # Remove function argument type hints
        code = re.sub(r':\s*[\w\[\],\s]+(?=[,\)])', '', code)
        # Remove return type hints
        code = re.sub(r'->\s*[\w\[\],\s]+:', ':', code)
        return code
    
    def normalize_indentation(self, code: str) -> str:
        """Normalize indentation to 4 spaces"""
        lines = []
        for line in code.split('\n'):
            if line.strip():
                indent_level = (len(line) - len(line.lstrip())) // 4
                normalized_line = '    ' * indent_level + line.lstrip()
                lines.append(normalized_line)
            else:
                lines.append('')
        return '\n'.join(lines)
    
    def remove_comments(self, code: str) -> str:
        """Remove comments from Python code"""
        return re.sub(r'#.*$', '', code, flags=re.MULTILINE)

class TypeScriptPreprocessor(LanguagePreprocessor):
    """Preprocessor for TypeScript code"""
    
    def __init__(self):
        self.js_keywords = {
            'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger',
            'default', 'delete', 'do', 'else', 'enum', 'export', 'extends',
            'false', 'finally', 'for', 'function', 'if', 'import', 'in',
            'instanceof', 'new', 'null', 'return', 'super', 'switch', 'this',
            'throw', 'true', 'try', 'typeof', 'var', 'void', 'while', 'with'
        }
    
    async def preprocess(self, code: str) -> str:
        """Preprocess TypeScript code"""
        processed = code
        
        # Remove type annotations
        processed = self.remove_type_annotations(processed)
        
        # Normalize arrow functions
        processed = self.normalize_arrow_functions(processed)
        
        # Remove JSDoc comments
        processed = self.remove_jsdoc_comments(processed)
        
        # Normalize template literals
        processed = self.normalize_template_literals(processed)
        
        # Remove decorators
        processed = self.remove_decorators(processed)
        
        return processed
    
    def get_language(self) -> ProgrammingLanguage:
        return ProgrammingLanguage.TYPESCRIPT
    
    def normalize_identifiers(self, code: str) -> str:
        """Replace identifiers with generic placeholders"""
        identifier_pattern = r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b'
        identifier_map = {}
        counter = 0
        
        def replace_identifier(match):
            nonlocal counter
            identifier = match.group(0)
            
            if identifier in self.js_keywords:
                return identifier
            
            if identifier not in identifier_map:
                counter += 1
                identifier_map[identifier] = f'VAR_{counter}'
            
            return identifier_map[identifier]
        
        return re.sub(identifier_pattern, replace_identifier, code)
    
    def remove_language_specific_noise(self, code: str) -> str:
        """Remove TypeScript-specific noise"""
        # Remove import/export statements
        code = re.sub(r'^(import|export)\s+.*$', '', code, flags=re.MULTILINE)
        return code
    
    def extract_semantic_structure(self, code: str) -> str:
        """Extract semantic structure from TypeScript code"""
        structure = []
        
        for line in code.split('\n'):
            trimmed = line.strip()
            
            if any(trimmed.startswith(x) for x in ['if ', '} else if ', '} else {']):
                structure.append('CONDITIONAL')
            elif any(trimmed.startswith(x) for x in ['for ', 'while ']):
                structure.append('LOOP')
            elif 'function ' in trimmed or '=>' in trimmed:
                structure.append('FUNCTION_DEF')
            elif trimmed.startswith('class '):
                structure.append('CLASS_DEF')
            elif trimmed.startswith('return '):
                structure.append('RETURN')
            elif any(trimmed.startswith(x) for x in ['throw ', 'catch ']):
                structure.append('EXCEPTION')
            elif '=' in trimmed and '==' not in trimmed and '===' not in trimmed:
                structure.append('ASSIGNMENT')
            elif trimmed and not trimmed.startswith('//'):
                structure.append('STATEMENT')
        
        return '\n'.join(structure)
    
    def remove_type_annotations(self, code: str) -> str:
        """Remove TypeScript type annotations"""
        # Remove type declarations after colons
        code = re.sub(r':\s*[\w\[\]<>,\s|&{}]+(?=[,;)\]}])', '', code)
        # Remove generic type parameters
        code = re.sub(r'<[\w\[\],\s]+>', '', code)
        # Remove type assertions
        code = re.sub(r'as\s+[\w\[\]<>,\s]+', '', code)
        return code
    
    def normalize_arrow_functions(self, code: str) -> str:
        """Normalize arrow functions to regular function syntax"""
        return re.sub(r'(\w+)\s*=>\s*', r'function(\1) ', code)
    
    def remove_jsdoc_comments(self, code: str) -> str:
        """Remove JSDoc comments"""
        return re.sub(r'/\*\*.*?\*/', '', code, flags=re.DOTALL)
    
    def normalize_template_literals(self, code: str) -> str:
        """Normalize template literals"""
        return re.sub(r'`[^`]*`', '"STRING"', code)
    
    def remove_decorators(self, code: str) -> str:
        """Remove TypeScript decorators"""
        return re.sub(r'@\w+(\([^)]*\))?\s*', '', code)

class CodePreprocessor:
    """Main code preprocessor that delegates to language-specific preprocessors"""
    
    def __init__(self):
        self.language_processors: Dict[ProgrammingLanguage, LanguagePreprocessor] = {
            ProgrammingLanguage.PYTHON: PythonPreprocessor(),
            ProgrammingLanguage.TYPESCRIPT: TypeScriptPreprocessor(),
        }
        self.common_processor = CommonPreprocessor()
    
    async def preprocess(self, code: str, language: ProgrammingLanguage) -> str:
        """Preprocess code based on language"""
        # Apply common preprocessing
        processed = await self.common_processor.preprocess(code)
        
        # Apply language-specific preprocessing
        if language in self.language_processors:
            processor = self.language_processors[language]
            processed = await processor.preprocess(processed)
        
        return processed
    
    async def preprocess_for_similarity(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> str:
        """More aggressive preprocessing for similarity analysis"""
        if language not in self.language_processors:
            raise ValueError(f"Unsupported language: {language}")
        
        processor = self.language_processors[language]
        
        # Normalize identifiers
        processed = processor.normalize_identifiers(code)
        
        # Remove language-specific noise
        processed = processor.remove_language_specific_noise(processed)
        
        # Extract semantic structure
        processed = processor.extract_semantic_structure(processed)
        
        return processed

class CommonPreprocessor:
    """Common preprocessing steps for all languages"""
    
    async def preprocess(self, code: str) -> str:
        """Apply common preprocessing steps"""
        # Normalize line endings
        processed = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Normalize whitespace
        processed = self.normalize_whitespace(processed)
        
        # Remove excessive empty lines
        processed = self.remove_excessive_empty_lines(processed)
        
        return processed
    
    def normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace in code"""
        lines = []
        for line in code.split('\n'):
            # Remove trailing whitespace
            line = line.rstrip()
            # Normalize tabs to spaces
            line = line.replace('\t', '    ')
            lines.append(line)
        return '\n'.join(lines)
    
    def remove_excessive_empty_lines(self, code: str) -> str:
        """Remove excessive empty lines"""
        lines = code.split('\n')
        result = []
        consecutive_empty = 0
        
        for line in lines:
            if not line.strip():
                consecutive_empty += 1
                if consecutive_empty <= 1:
                    result.append(line)
            else:
                consecutive_empty = 0
                result.append(line)
        
        return '\n'.join(result)

# ============================================================================
# 16.5 Vector Database Integration
# ============================================================================

@dataclass
class VectorStoreConfig:
    qdrant_url: str = "http://localhost:6333"
    default_collection: str = "code_embeddings"
    vector_dimension: int = 768
    distance_metric: Distance = Distance.COSINE
    replication_factor: int = 1
    shard_number: int = 1
    enable_compression: bool = False

class VectorStore:
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.collections: Dict[str, Any] = {}
    
    @classmethod
    async def create(cls, config: VectorStoreConfig) -> 'VectorStore':
        """Create and initialize a VectorStore"""
        store = cls(config)
        await store.initialize()
        return store
    
    async def initialize(self):
        """Initialize the vector store and create collections"""
        self.client = QdrantClient(url=self.config.qdrant_url)
        await self.initialize_collections()
    
    async def initialize_collections(self):
        """Initialize default collections"""
        collections_to_create = [
            self.config.default_collection,
            "code_embeddings_python",
            "code_embeddings_javascript",
            "code_embeddings_typescript",
            "code_embeddings_rust",
        ]
        
        for collection_name in collections_to_create:
            try:
                await self.create_collection(collection_name)
            except Exception as e:
                logger.warning(f"Collection {collection_name} might already exist: {e}")
    
    async def create_collection(self, collection_name: str):
        """Create a new collection in Qdrant"""
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.config.vector_dimension,
                distance=self.config.distance_metric
            )
        )
    
    async def store_embedding(self, embedding: CodeEmbedding):
        """Store an embedding in the vector database"""
        collection_name = self.get_collection_for_language(embedding.language)
        
        payload = {
            "code_snippet": embedding.code_snippet,
            "language": embedding.language.value,
            "model_id": embedding.model_id,
            "code_length": embedding.metadata.code_length,
            "token_count": embedding.metadata.token_count,
            "created_at": embedding.created_at.isoformat(),
        }
        
        if embedding.metadata.file_path:
            payload["file_path"] = str(embedding.metadata.file_path)
        if embedding.metadata.function_name:
            payload["function_name"] = embedding.metadata.function_name
        
        point = PointStruct(
            id=embedding.id,
            vector=embedding.embedding_vector,
            payload=payload
        )
        
        self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )
    
    async def search_similar_embeddings(
        self,
        query_vector: List[float],
        limit: int = 10,
        language: Optional[ProgrammingLanguage] = None
    ) -> List[Dict]:
        """Search for similar embeddings"""
        collection_name = (
            self.get_collection_for_language(language) 
            if language 
            else self.config.default_collection
        )
        
        # Build filter if language is specified
        query_filter = None
        if language:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="language",
                        match=MatchValue(value=language.value)
                    )
                ]
            )
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True
        )
        
        search_results = []
        for point in results:
            result = {
                "id": point.id,
                "score": point.score,
                "payload": point.payload
            }
            search_results.append(result)
        
        return search_results
    
    def get_collection_for_language(self, language: ProgrammingLanguage) -> str:
        """Get the collection name for a specific language"""
        language_collections = {
            ProgrammingLanguage.PYTHON: "code_embeddings_python",
            ProgrammingLanguage.JAVASCRIPT: "code_embeddings_javascript",
            ProgrammingLanguage.TYPESCRIPT: "code_embeddings_typescript",
            ProgrammingLanguage.RUST: "code_embeddings_rust",
        }
        
        return language_collections.get(language, self.config.default_collection)

# ============================================================================
# 16.6 AI-Powered Code Analysis
# ============================================================================

@dataclass
class AIAnalysisConfig:
    enable_pattern_detection: bool = True
    enable_anomaly_detection: bool = True
    enable_semantic_analysis: bool = True
    enable_code_quality_prediction: bool = True
    confidence_threshold: float = 0.7
    batch_analysis_size: int = 32
    enable_cross_language_analysis: bool = False
    model_ensemble: bool = False

class AICodeAnalyzer:
    def __init__(
        self,
        model_manager: AIModelManager,
        embedding_engine: CodeEmbeddingEngine,
        config: AIAnalysisConfig
    ):
        self.model_manager = model_manager
        self.embedding_engine = embedding_engine
        self.config = config
        self.pattern_detector = AIPatternDetector()
        self.anomaly_detector = AnomalyDetector()
        self.semantic_analyzer = SemanticAnalyzer()
    
    async def analyze_code_with_ai(self, code: str, language: ProgrammingLanguage) -> Dict:
        """Analyze code using AI models"""
        start_time = time.time()
        
        analysis_result = {
            "language": language.value,
            "embeddings": [],
            "ai_detected_patterns": [],
            "anomalies": [],
            "semantic_insights": [],
            "quality_predictions": [],
            "confidence_scores": {},
            "model_explanations": [],
            "execution_time_ms": 0
        }
        
        # Generate embeddings
        if self.config.enable_pattern_detection or self.config.enable_semantic_analysis:
            embedding = await self.embedding_engine.generate_code_embedding(code, language)
            analysis_result["embeddings"].append({
                "id": embedding.id,
                "model_id": embedding.model_id,
                "dimension": len(embedding.embedding_vector)
            })
        
        # Pattern detection
        if self.config.enable_pattern_detection:
            patterns = await self.pattern_detector.detect_patterns(code, language)
            analysis_result["ai_detected_patterns"] = patterns
        
        # Anomaly detection
        if self.config.enable_anomaly_detection:
            anomalies = await self.anomaly_detector.detect_anomalies(code, language)
            analysis_result["anomalies"] = anomalies
        
        # Semantic analysis
        if self.config.enable_semantic_analysis:
            insights = await self.semantic_analyzer.analyze(code, language)
            analysis_result["semantic_insights"] = insights
        
        # Code quality prediction
        if self.config.enable_code_quality_prediction:
            quality = await self.predict_code_quality(code, language)
            analysis_result["quality_predictions"] = quality
        
        analysis_result["execution_time_ms"] = int((time.time() - start_time) * 1000)
        
        return analysis_result
    
    async def predict_code_quality(self, code: str, language: ProgrammingLanguage) -> List[Dict]:
        """Predict code quality metrics"""
        # Simplified quality prediction
        return [{
            "metric": "maintainability",
            "score": 0.75,
            "confidence": 0.8,
            "recommendations": [
                "Consider adding more comments",
                "Reduce function complexity"
            ]
        }]

class AIPatternDetector:
    """Detect patterns in code using AI"""
    
    async def detect_patterns(self, code: str, language: ProgrammingLanguage) -> List[Dict]:
        """Detect design patterns and anti-patterns"""
        # Simplified pattern detection
        return [{
            "pattern_name": "Singleton Pattern",
            "pattern_type": "DesignPattern",
            "confidence": 0.85,
            "description": "Detected singleton pattern implementation"
        }]

class AnomalyDetector:
    """Detect anomalies in code"""
    
    async def detect_anomalies(self, code: str, language: ProgrammingLanguage) -> List[Dict]:
        """Detect code anomalies"""
        # Simplified anomaly detection
        return []

class SemanticAnalyzer:
    """Analyze semantic meaning of code"""
    
    async def analyze(self, code: str, language: ProgrammingLanguage) -> List[Dict]:
        """Perform semantic analysis"""
        # Simplified semantic analysis
        return [{
            "insight_type": "function_purpose",
            "description": "This function appears to handle data validation",
            "confidence": 0.7
        }]

# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main function to demonstrate the AI integration system"""
    
    # Initialize AI configuration
    ai_config = AIConfig(
        device_type=DeviceType.AUTO,
        max_models_in_memory=3,
        model_cache_size_gb=8.0,
        inference_batch_size=32
    )
    
    # Create model manager
    model_manager = AIModelManager(ai_config)
    await model_manager.initialize()
    
    # Create embedding engine
    embedding_config = EmbeddingConfig(
        default_model="microsoft/codebert-base",
        batch_size=32,
        enable_caching=True,
        normalize_embeddings=True
    )
    
    embedding_engine = CodeEmbeddingEngine(model_manager, embedding_config)
    await embedding_engine.initialize()
    
    # Create AI analyzer
    analysis_config = AIAnalysisConfig(
        enable_pattern_detection=True,
        enable_anomaly_detection=True,
        enable_semantic_analysis=True,
        enable_code_quality_prediction=True
    )
    
    analyzer = AICodeAnalyzer(model_manager, embedding_engine, analysis_config)
    
    # Example code to analyze
    example_code = '''
def calculate_factorial(n):
    """Calculate factorial of a number"""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * calculate_factorial(n - 1)
'''
    
    # Generate embedding
    logger.info("Generating code embedding...")
    embedding = await embedding_engine.generate_code_embedding(
        example_code, 
        ProgrammingLanguage.PYTHON
    )
    logger.info(f"Generated embedding with dimension: {len(embedding.embedding_vector)}")
    
    # Analyze code
    logger.info("Analyzing code with AI...")
    analysis = await analyzer.analyze_code_with_ai(
        example_code,
        ProgrammingLanguage.PYTHON
    )
    logger.info(f"Analysis complete in {analysis['execution_time_ms']}ms")
    
    # Print results
    print(json.dumps(analysis, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())

### 16.8 Criterios de Completitud

#### 16.8.1 Entregables de la Fase
- [ ] Sistema de gestión de modelos de IA implementado
- [ ] Integración completa con CodeBERT y CodeT5
- [ ] Motor de generación de embeddings de código
- [ ] Vector database con Qdrant configurado
- [ ] Sistema de preprocessing específico por lenguaje
- [ ] Motor de inferencia optimizado
- [ ] Cache inteligente de modelos y embeddings
- [ ] API para análisis con IA
- [ ] Sistema de métricas de IA
- [ ] Tests comprehensivos de integración

#### 16.8.2 Criterios de Aceptación
- [ ] Modelos de IA se cargan y ejecutan correctamente
- [ ] Embeddings de código son consistentes y útiles
- [ ] Vector database almacena y busca eficientemente
- [ ] Preprocessing mejora calidad de embeddings
- [ ] Performance acceptable para uso en producción
- [ ] Búsqueda semántica funciona correctamente
- [ ] Cache reduce latencia significativamente
- [ ] Integration seamless con sistema principal
- [ ] Manejo robusto de errores de IA
- [ ] Documentación completa de APIs de IA

### 16.9 Performance Targets

#### 16.9.1 Benchmarks de IA
- **Model loading**: <30 segundos para modelos base
- **Embedding generation**: <100ms para snippets típicos
- **Vector search**: <200ms para búsquedas similares
- **Batch processing**: >10x speedup vs individual processing
- **Memory usage**: <4GB para modelos base cargados

### 16.10 Estimación de Tiempo

#### 16.10.1 Breakdown de Tareas
- Setup de infraestructura de IA: 5 días
- Model manager y carga de modelos: 8 días
- Code embedding engine: 10 días
- Vector database integration: 8 días
- Code preprocessor por lenguaje: 12 días
- Inference engine optimization: 8 días
- AI code analyzer: 10 días
- Cache y performance optimization: 6 días
- Integration con sistema principal: 6 días
- Testing comprehensivo: 10 días
- Documentación: 4 días

**Total estimado: 87 días de desarrollo**

### 16.11 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Capacidades de IA integradas para análisis de código
- Comprensión semántica profunda del código
- Base sólida para análisis avanzados con ML
- Vector database para búsqueda semántica
- Foundation para las siguientes fases de IA

La Fase 17 construirá sobre esta base implementando el sistema de embeddings de código y análisis semántico avanzado.

