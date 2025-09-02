"""
Motor de generación de embeddings multi-nivel.

Este módulo implementa la generación de embeddings jerárquicos
desde tokens hasta proyectos completos con agregación inteligente.
"""

import logging
import asyncio
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from ...domain.entities.semantic_analysis import (
    MultiLevelEmbeddings, MultiLevelConfig, TokenEmbedding, ExpressionEmbedding,
    StatementEmbedding, FunctionEmbedding, ClassEmbedding, FileEmbedding,
    HierarchicalStructure, LevelInfo, HierarchicalRelationship,
    EmbeddingLevel, AggregationStrategy, SemanticRelationship
)
from ...domain.value_objects.programming_language import ProgrammingLanguage
from ..ai_models.model_manager import AIModelManager
from ..ai_models.embedding_engine import CodeEmbeddingEngine

logger = logging.getLogger(__name__)

# Fallback para numpy si no está disponible
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    class MockNumPy:
        def array(self, data): 
            return MockArray(data)
        def mean(self, data, axis=None): 
            if isinstance(data, list):
                return sum(data) / len(data) if data else 0.0
            return 0.0
        def dot(self, a, b): 
            return 0.5
        def linalg(self): 
            return MockLinalg()
    
    class MockArray:
        def __init__(self, data):
            self.data = data if isinstance(data, list) else [data]
        def tolist(self): 
            return self.data
        def mean(self): 
            return sum(self.data) / len(self.data) if self.data else 0.0
    
    class MockLinalg:
        def norm(self, vector): 
            return sum(x**2 for x in vector)**0.5 if vector else 0.0
    
    np = MockNumPy()
    NUMPY_AVAILABLE = False


@dataclass
class EmbeddingGenerationContext:
    """Contexto para generación de embeddings."""
    source_code: str
    language: ProgrammingLanguage
    file_path: Path
    ast_nodes: Optional[Any] = None  # AST nodes si están disponibles
    parent_context: Optional[str] = None
    generation_options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.generation_options is None:
            self.generation_options = {}


class TokenEmbedder:
    """Generador de embeddings a nivel de token."""
    
    def __init__(self, base_embedding_engine: CodeEmbeddingEngine):
        self.base_embedding_engine = base_embedding_engine
        self.token_cache: Dict[str, List[float]] = {}
    
    async def generate_token_embeddings(
        self, 
        tokens: List[str], 
        context: EmbeddingGenerationContext
    ) -> Dict[str, TokenEmbedding]:
        """Genera embeddings para tokens individuales."""
        embeddings = {}
        
        for i, token in enumerate(tokens):
            if not token.strip():
                continue
            
            # Generar embedding para el token con contexto
            context_window = tokens[max(0, i-3):i] + tokens[i+1:min(len(tokens), i+4)]
            token_with_context = f"{' '.join(context_window[:3])} {token} {' '.join(context_window[3:])}"
            
            # Usar cache si disponible
            cache_key = f"{token}_{hash(str(context_window))}"
            
            if cache_key in self.token_cache:
                embedding_vector = self.token_cache[cache_key]
            else:
                # Generar embedding usando el motor base
                embedding_result = await self.base_embedding_engine.generate_embedding(
                    token_with_context, context.language
                )
                embedding_vector = embedding_result.embedding_vector[:256]  # Truncar a 256D
                self.token_cache[cache_key] = embedding_vector
            
            # Detectar tipo y rol semántico del token
            token_type = self._detect_token_type(token)
            semantic_role = self._detect_semantic_role(token, tokens, i)
            
            token_embedding = TokenEmbedding(
                token=token,
                token_type=token_type,
                embedding_vector=embedding_vector,
                position=i,
                context_tokens=context_window,
                semantic_role=semantic_role
            )
            
            embeddings[token_embedding.id] = token_embedding
        
        return embeddings
    
    def _detect_token_type(self, token: str) -> str:
        """Detecta el tipo de token."""
        # Keywords comunes
        keywords = {
            'python': {'def', 'class', 'if', 'for', 'while', 'import', 'from', 'return'},
            'javascript': {'function', 'class', 'if', 'for', 'while', 'import', 'export', 'return'},
            'rust': {'fn', 'struct', 'impl', 'if', 'for', 'while', 'use', 'return'},
        }
        
        # Verificar si es keyword
        for lang_keywords in keywords.values():
            if token.lower() in lang_keywords:
                return "keyword"
        
        # Verificar si es operador
        if token in {'+', '-', '*', '/', '=', '==', '!=', '<', '>', '&&', '||'}:
            return "operator"
        
        # Verificar si es literal
        if token.startswith('"') and token.endswith('"'):
            return "string_literal"
        if token.startswith("'") and token.endswith("'"):
            return "string_literal"
        if token.isdigit() or (token.count('.') == 1 and token.replace('.', '').isdigit()):
            return "numeric_literal"
        
        # Verificar si es identificador
        if token.replace('_', '').replace('$', '').isalnum():
            return "identifier"
        
        return "punctuation"
    
    def _detect_semantic_role(self, token: str, tokens: List[str], position: int) -> str:
        """Detecta el rol semántico del token."""
        # Verificar contexto anterior y posterior
        prev_token = tokens[position - 1] if position > 0 else ""
        next_token = tokens[position + 1] if position < len(tokens) - 1 else ""
        
        # Detectar declaraciones de función
        if prev_token.lower() in {'def', 'function', 'fn'}:
            return "function_name"
        
        # Detectar declaraciones de clase
        if prev_token.lower() == 'class':
            return "class_name"
        
        # Detectar nombres de variables
        if prev_token in {'=', ':', 'let', 'var', 'const'}:
            return "variable_name"
        
        # Detectar llamadas a función
        if next_token == '(':
            return "function_call"
        
        # Detectar acceso a propiedades
        if prev_token == '.':
            return "property_access"
        
        return "unknown"


class ExpressionEmbedder:
    """Generador de embeddings a nivel de expresión."""
    
    def __init__(self, base_embedding_engine: CodeEmbeddingEngine):
        self.base_embedding_engine = base_embedding_engine
    
    async def generate_expression_embeddings(
        self,
        expressions: List[str],
        token_embeddings: Dict[str, TokenEmbedding],
        context: EmbeddingGenerationContext
    ) -> Dict[str, ExpressionEmbedding]:
        """Genera embeddings para expresiones."""
        embeddings = {}
        
        for i, expression in enumerate(expressions):
            if not expression.strip():
                continue
            
            # Generar embedding para la expresión
            expr_embedding_result = await self.base_embedding_engine.generate_embedding(
                expression, context.language
            )
            embedding_vector = expr_embedding_result.embedding_vector[:384]  # 384D para expresiones
            
            # Detectar tipo de expresión
            expr_type = self._detect_expression_type(expression)
            
            # Calcular score de complejidad
            complexity_score = self._calculate_expression_complexity(expression)
            
            # Extraer características semánticas
            semantic_features = self._extract_semantic_features(expression)
            
            # Encontrar tokens relacionados
            related_token_ids = self._find_related_tokens(expression, token_embeddings)
            
            expr_embedding = ExpressionEmbedding(
                expression=expression,
                expression_type=expr_type,
                embedding_vector=embedding_vector,
                token_embeddings=related_token_ids,
                start_position=i * 10,  # Estimación
                end_position=(i + 1) * 10,
                complexity_score=complexity_score,
                semantic_features=semantic_features
            )
            
            embeddings[expr_embedding.id] = expr_embedding
        
        return embeddings
    
    def _detect_expression_type(self, expression: str) -> str:
        """Detecta el tipo de expresión."""
        expr = expression.strip()
        
        if '(' in expr and ')' in expr and not any(op in expr for op in ['=', '==', '!=']):
            return "function_call"
        
        if '=' in expr and '==' not in expr and '!=' not in expr:
            return "assignment"
        
        if any(op in expr for op in ['+', '-', '*', '/', '%']):
            return "binary_operation"
        
        if any(op in expr for op in ['==', '!=', '<', '>', '<=', '>=']):
            return "comparison"
        
        if any(op in expr for op in ['&&', '||', 'and', 'or']):
            return "logical_operation"
        
        if '[' in expr and ']' in expr:
            return "array_access"
        
        if '.' in expr:
            return "property_access"
        
        return "simple_expression"
    
    def _calculate_expression_complexity(self, expression: str) -> float:
        """Calcula complejidad de la expresión."""
        complexity = 0.0
        
        # Contar operadores
        operators = ['+', '-', '*', '/', '%', '==', '!=', '<', '>', '<=', '>=', '&&', '||']
        for op in operators:
            complexity += expression.count(op) * 0.1
        
        # Contar parentesis (anidamiento)
        complexity += expression.count('(') * 0.2
        
        # Contar llamadas a función
        if '(' in expression and ')' in expression:
            complexity += 0.3
        
        # Longitud de la expresión
        complexity += len(expression) * 0.01
        
        return min(complexity, 5.0)  # Cap a 5.0
    
    def _extract_semantic_features(self, expression: str) -> Dict[str, Any]:
        """Extrae características semánticas."""
        features = {}
        
        # Tipo de operación predominante
        if any(op in expression for op in ['==', '!=', '<', '>']):
            features['operation_type'] = 'comparison'
        elif any(op in expression for op in ['+', '-', '*', '/']):
            features['operation_type'] = 'arithmetic'
        elif '(' in expression:
            features['operation_type'] = 'function_call'
        else:
            features['operation_type'] = 'simple'
        
        # Nivel de anidamiento
        features['nesting_level'] = expression.count('(')
        
        # Tiene literales
        features['has_string_literal'] = '"' in expression or "'" in expression
        features['has_numeric_literal'] = any(char.isdigit() for char in expression)
        
        return features
    
    def _find_related_tokens(
        self, 
        expression: str, 
        token_embeddings: Dict[str, TokenEmbedding]
    ) -> List[str]:
        """Encuentra tokens relacionados con la expresión."""
        related_ids = []
        expr_tokens = set(expression.split())
        
        for token_id, token_embedding in token_embeddings.items():
            if token_embedding.token in expr_tokens:
                related_ids.append(token_id)
        
        return related_ids


class FunctionEmbedder:
    """Generador de embeddings a nivel de función."""
    
    def __init__(self, base_embedding_engine: CodeEmbeddingEngine):
        self.base_embedding_engine = base_embedding_engine
    
    async def generate_function_embeddings(
        self,
        functions: List[Dict[str, Any]],
        context: EmbeddingGenerationContext
    ) -> Dict[str, FunctionEmbedding]:
        """Genera embeddings para funciones."""
        embeddings = {}
        
        for func_info in functions:
            func_code = func_info.get('code', '')
            func_name = func_info.get('name', 'unknown')
            
            if not func_code.strip():
                continue
            
            # Generar embedding para la función completa
            func_embedding_result = await self.base_embedding_engine.generate_embedding(
                func_code, context.language
            )
            embedding_vector = func_embedding_result.embedding_vector  # 768D completo
            
            # Extraer información de la función
            signature = self._extract_function_signature(func_code)
            parameters = self._extract_parameters(func_code)
            return_type = self._extract_return_type(func_code)
            docstring = self._extract_docstring(func_code)
            
            # Calcular métricas de complejidad
            complexity_metrics = self._calculate_complexity_metrics(func_code)
            
            # Detectar propósito semántico
            semantic_purpose = self._detect_semantic_purpose(func_name, func_code)
            
            func_embedding = FunctionEmbedding(
                function_name=func_name,
                signature=signature,
                embedding_vector=embedding_vector,
                parameters=parameters,
                return_type=return_type,
                docstring=docstring,
                complexity_metrics=complexity_metrics,
                semantic_purpose=semantic_purpose
            )
            
            embeddings[func_embedding.id] = func_embedding
        
        return embeddings
    
    def _extract_function_signature(self, func_code: str) -> str:
        """Extrae la signatura de la función."""
        lines = func_code.split('\n')
        for line in lines:
            line = line.strip()
            if any(line.startswith(keyword) for keyword in ['def ', 'function ', 'fn ']):
                # Encontrar el final de la declaración
                if ':' in line:
                    return line.split(':')[0]
                elif '{' in line:
                    return line.split('{')[0]
                return line
        
        return "unknown_signature"
    
    def _extract_parameters(self, func_code: str) -> List[str]:
        """Extrae parámetros de la función."""
        signature = self._extract_function_signature(func_code)
        
        # Buscar paréntesis
        if '(' in signature and ')' in signature:
            params_str = signature.split('(')[1].split(')')[0]
            if params_str.strip():
                return [p.strip() for p in params_str.split(',')]
        
        return []
    
    def _extract_return_type(self, func_code: str) -> Optional[str]:
        """Extrae tipo de retorno si está disponible."""
        signature = self._extract_function_signature(func_code)
        
        # TypeScript/Rust style: -> Type
        if '->' in signature:
            return signature.split('->')[1].strip()
        
        # Python type hints: : Type
        if ':' in signature and ')' in signature:
            after_params = signature.split(')')
            if len(after_params) > 1 and ':' in after_params[1]:
                return after_params[1].split(':')[1].strip()
        
        return None
    
    def _extract_docstring(self, func_code: str) -> Optional[str]:
        """Extrae docstring de la función."""
        lines = func_code.split('\n')
        in_docstring = False
        docstring_lines = []
        quote_type = None
        
        for line in lines[1:]:  # Skip first line (signature)
            stripped = line.strip()
            
            if not in_docstring:
                if stripped.startswith('"""'):
                    in_docstring = True
                    quote_type = '"""'
                    if stripped.endswith('"""') and len(stripped) > 3:
                        return stripped[3:-3].strip()
                    docstring_lines.append(stripped[3:])
                elif stripped.startswith("'''"):
                    in_docstring = True
                    quote_type = "'''"
                    if stripped.endswith("'''") and len(stripped) > 3:
                        return stripped[3:-3].strip()
                    docstring_lines.append(stripped[3:])
            else:
                if stripped.endswith(quote_type):
                    docstring_lines.append(stripped[:-3])
                    break
                else:
                    docstring_lines.append(stripped)
        
        if docstring_lines:
            return '\n'.join(docstring_lines).strip()
        
        return None
    
    def _calculate_complexity_metrics(self, func_code: str) -> Dict[str, float]:
        """Calcula métricas de complejidad."""
        metrics = {}
        
        # Complejidad ciclomática básica
        decision_points = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'case', 'switch']
        cyclomatic = 1  # Base complexity
        
        for keyword in decision_points:
            cyclomatic += func_code.lower().count(keyword)
        
        metrics['cyclomatic_complexity'] = float(cyclomatic)
        
        # Líneas de código
        lines = [line for line in func_code.split('\n') if line.strip()]
        metrics['lines_of_code'] = float(len(lines))
        
        # Profundidad de anidamiento
        max_nesting = 0
        current_nesting = 0
        for line in func_code.split('\n'):
            stripped = line.strip()
            if any(stripped.startswith(kw) for kw in ['if', 'for', 'while', 'try']):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif stripped in ['end', '}'] or not stripped:
                current_nesting = max(0, current_nesting - 1)
        
        metrics['nesting_depth'] = float(max_nesting)
        
        return metrics
    
    def _detect_semantic_purpose(self, func_name: str, func_code: str) -> str:
        """Detecta el propósito semántico de la función."""
        name_lower = func_name.lower()
        code_lower = func_code.lower()
        
        # Análisis por nombre
        if any(prefix in name_lower for prefix in ['get', 'fetch', 'read', 'retrieve']):
            return "data_retrieval"
        elif any(prefix in name_lower for prefix in ['set', 'update', 'write', 'save']):
            return "data_modification"
        elif any(prefix in name_lower for prefix in ['create', 'make', 'build', 'generate']):
            return "object_creation"
        elif any(prefix in name_lower for prefix in ['delete', 'remove', 'destroy', 'clear']):
            return "object_destruction"
        elif any(prefix in name_lower for prefix in ['validate', 'check', 'verify', 'test']):
            return "validation"
        elif any(prefix in name_lower for prefix in ['calculate', 'compute', 'sum', 'count']):
            return "calculation"
        elif any(prefix in name_lower for prefix in ['send', 'receive', 'request', 'respond']):
            return "communication"
        
        # Análisis por contenido
        if any(keyword in code_lower for keyword in ['return', 'yield']):
            if any(keyword in code_lower for keyword in ['calculate', 'compute', '*', '/', '+']):
                return "calculation"
            elif any(keyword in code_lower for keyword in ['fetch', 'get', 'find']):
                return "data_retrieval"
        
        if any(keyword in code_lower for keyword in ['try', 'except', 'catch', 'error']):
            return "error_handling"
        
        if any(keyword in code_lower for keyword in ['log', 'print', 'debug']):
            return "logging"
        
        return "general"


class HierarchicalAggregator:
    """Agregador jerárquico de embeddings."""
    
    def __init__(self, config: MultiLevelConfig):
        self.config = config
    
    async def build_hierarchy(self, embeddings: MultiLevelEmbeddings) -> HierarchicalStructure:
        """Construye estructura jerárquica de embeddings."""
        structure = HierarchicalStructure()
        
        # Construir información de niveles
        if embeddings.token_embeddings:
            level_info = LevelInfo(
                level=EmbeddingLevel.TOKEN,
                embedding_count=len(embeddings.token_embeddings),
                average_dimension=self._calculate_average_dimension(
                    [emb.embedding_vector for emb in embeddings.token_embeddings.values()]
                ),
                quality_score=0.8
            )
            structure.levels[EmbeddingLevel.TOKEN] = level_info
        
        if embeddings.expression_embeddings:
            level_info = LevelInfo(
                level=EmbeddingLevel.EXPRESSION,
                embedding_count=len(embeddings.expression_embeddings),
                average_dimension=self._calculate_average_dimension(
                    [emb.embedding_vector for emb in embeddings.expression_embeddings.values()]
                ),
                quality_score=0.85
            )
            structure.levels[EmbeddingLevel.EXPRESSION] = level_info
        
        if embeddings.function_embeddings:
            level_info = LevelInfo(
                level=EmbeddingLevel.FUNCTION,
                embedding_count=len(embeddings.function_embeddings),
                average_dimension=self._calculate_average_dimension(
                    [emb.embedding_vector for emb in embeddings.function_embeddings.values()]
                ),
                quality_score=0.9
            )
            structure.levels[EmbeddingLevel.FUNCTION] = level_info
        
        # Construir relaciones jerárquicas
        structure.parent_child_relationships = await self._build_relationships(embeddings)
        
        # Calcular pesos de agregación
        structure.aggregation_weights = self._calculate_aggregation_weights(structure.levels)
        
        structure.hierarchy_depth = len(structure.levels)
        
        return structure
    
    def _calculate_average_dimension(self, embedding_vectors: List[List[float]]) -> int:
        """Calcula dimensión promedio."""
        if not embedding_vectors:
            return 0
        
        total_dims = sum(len(vec) for vec in embedding_vectors)
        return total_dims // len(embedding_vectors)
    
    async def _build_relationships(self, embeddings: MultiLevelEmbeddings) -> List[HierarchicalRelationship]:
        """Construye relaciones jerárquicas."""
        relationships = []
        
        # Relaciones token -> expression
        for expr_id, expr_emb in embeddings.expression_embeddings.items():
            for token_id in expr_emb.token_embeddings:
                if token_id in embeddings.token_embeddings:
                    rel = HierarchicalRelationship(
                        parent_id=expr_id,
                        child_id=token_id,
                        parent_level=EmbeddingLevel.EXPRESSION,
                        child_level=EmbeddingLevel.TOKEN,
                        relationship_strength=0.8,
                        relationship_type="contains"
                    )
                    relationships.append(rel)
        
        # Relaciones statement -> expression (simplificado)
        for stmt_id, stmt_emb in embeddings.statement_embeddings.items():
            for expr_id in stmt_emb.expression_embeddings:
                if expr_id in embeddings.expression_embeddings:
                    rel = HierarchicalRelationship(
                        parent_id=stmt_id,
                        child_id=expr_id,
                        parent_level=EmbeddingLevel.STATEMENT,
                        child_level=EmbeddingLevel.EXPRESSION,
                        relationship_strength=0.85,
                        relationship_type="contains"
                    )
                    relationships.append(rel)
        
        # Relaciones function -> statement (simplificado)
        for func_id, func_emb in embeddings.function_embeddings.items():
            for stmt_id in func_emb.statement_embeddings:
                if stmt_id in embeddings.statement_embeddings:
                    rel = HierarchicalRelationship(
                        parent_id=func_id,
                        child_id=stmt_id,
                        parent_level=EmbeddingLevel.FUNCTION,
                        child_level=EmbeddingLevel.STATEMENT,
                        relationship_strength=0.9,
                        relationship_type="contains"
                    )
                    relationships.append(rel)
        
        return relationships
    
    def _calculate_aggregation_weights(self, levels: Dict[EmbeddingLevel, LevelInfo]) -> Dict[EmbeddingLevel, float]:
        """Calcula pesos para agregación."""
        weights = {}
        total_quality = sum(level.quality_score for level in levels.values())
        
        for level, level_info in levels.items():
            if total_quality > 0:
                weights[level] = level_info.quality_score / total_quality
            else:
                weights[level] = 1.0 / len(levels)
        
        return weights
    
    async def aggregate_embeddings(
        self, 
        embeddings: Dict[str, List[float]], 
        strategy: AggregationStrategy
    ) -> List[float]:
        """Agrega embeddings usando la estrategia especificada."""
        if not embeddings:
            return []
        
        vectors = list(embeddings.values())
        
        if strategy == AggregationStrategy.MEAN:
            return self._mean_aggregation(vectors)
        elif strategy == AggregationStrategy.WEIGHTED_MEAN:
            # Pesos uniformes por defecto
            weights = [1.0] * len(vectors)
            return self._weighted_mean_aggregation(vectors, weights)
        elif strategy == AggregationStrategy.ATTENTION:
            return await self._attention_aggregation(vectors)
        else:
            # Fallback a MEAN
            return self._mean_aggregation(vectors)
    
    def _mean_aggregation(self, vectors: List[List[float]]) -> List[float]:
        """Agregación por promedio simple."""
        if not vectors:
            return []
        
        # Asegurar que todos los vectores tengan la misma dimensión
        min_dim = min(len(vec) for vec in vectors)
        normalized_vectors = [vec[:min_dim] for vec in vectors]
        
        if NUMPY_AVAILABLE:
            return np.mean(normalized_vectors, axis=0).tolist()
        else:
            # Implementación manual
            aggregated = [0.0] * min_dim
            for vec in normalized_vectors:
                for i, val in enumerate(vec):
                    aggregated[i] += val
            
            return [val / len(normalized_vectors) for val in aggregated]
    
    def _weighted_mean_aggregation(self, vectors: List[List[float]], weights: List[float]) -> List[float]:
        """Agregación por promedio ponderado."""
        if not vectors or len(vectors) != len(weights):
            return []
        
        min_dim = min(len(vec) for vec in vectors)
        normalized_vectors = [vec[:min_dim] for vec in vectors]
        
        # Normalizar pesos
        total_weight = sum(weights)
        if total_weight == 0:
            return self._mean_aggregation(vectors)
        
        normalized_weights = [w / total_weight for w in weights]
        
        # Agregar con pesos
        aggregated = [0.0] * min_dim
        for vec, weight in zip(normalized_vectors, normalized_weights):
            for i, val in enumerate(vec):
                aggregated[i] += val * weight
        
        return aggregated
    
    async def _attention_aggregation(self, vectors: List[List[float]]) -> List[float]:
        """Agregación con mecanismo de atención simplificado."""
        if not vectors:
            return []
        
        # Calcular scores de atención basados en magnitud
        attention_scores = []
        for vec in vectors:
            if NUMPY_AVAILABLE:
                magnitude = np.linalg.norm(vec)
            else:
                magnitude = sum(x**2 for x in vec)**0.5
            attention_scores.append(magnitude)
        
        # Softmax sobre los scores
        if NUMPY_AVAILABLE:
            attention_weights = np.exp(attention_scores)
            attention_weights = attention_weights / np.sum(attention_weights)
            attention_weights = attention_weights.tolist()
        else:
            # Softmax manual simplificado
            max_score = max(attention_scores)
            exp_scores = [np.exp(score - max_score) for score in attention_scores]
            sum_exp = sum(exp_scores)
            attention_weights = [score / sum_exp for score in exp_scores]
        
        return self._weighted_mean_aggregation(vectors, attention_weights)


class MultiLevelEmbeddingEngine:
    """Motor principal de generación de embeddings multi-nivel."""
    
    def __init__(
        self, 
        model_manager: AIModelManager,
        base_embedding_engine: CodeEmbeddingEngine,
        config: Optional[MultiLevelConfig] = None
    ):
        """
        Inicializa el motor de embeddings multi-nivel.
        
        Args:
            model_manager: Gestor de modelos de IA
            base_embedding_engine: Motor base de embeddings
            config: Configuración multi-nivel
        """
        self.model_manager = model_manager
        self.base_embedding_engine = base_embedding_engine
        self.config = config or MultiLevelConfig()
        
        # Inicializar componentes especializados
        self.token_embedder = TokenEmbedder(base_embedding_engine)
        self.expression_embedder = ExpressionEmbedder(base_embedding_engine)
        self.function_embedder = FunctionEmbedder(base_embedding_engine)
        self.hierarchical_aggregator = HierarchicalAggregator(self.config)
        
        # Estadísticas
        self.generation_stats = {
            'total_generated': 0,
            'generation_time_ms': 0,
            'levels_generated': set()
        }
    
    async def generate_multilevel_embeddings(
        self, 
        code: str, 
        language: ProgrammingLanguage,
        file_path: Optional[Path] = None
    ) -> MultiLevelEmbeddings:
        """
        Genera embeddings multi-nivel para código.
        
        Args:
            code: Código fuente
            language: Lenguaje de programación
            file_path: Ruta del archivo
            
        Returns:
            Conjunto completo de embeddings multi-nivel
        """
        start_time = time.time()
        
        context = EmbeddingGenerationContext(
            source_code=code,
            language=language,
            file_path=file_path or Path("unknown.py")
        )
        
        # Inicializar resultado
        embeddings = MultiLevelEmbeddings(
            file_path=context.file_path,
            language=language
        )
        
        try:
            # Generar embeddings por nivel
            if self.config.enable_token_embeddings:
                await self._generate_token_level(code, context, embeddings)
            
            if self.config.enable_expression_embeddings:
                await self._generate_expression_level(code, context, embeddings)
            
            if self.config.enable_function_embeddings:
                await self._generate_function_level(code, context, embeddings)
            
            if self.config.enable_file_embeddings:
                await self._generate_file_level(code, context, embeddings)
            
            # Construir estructura jerárquica
            embeddings.hierarchical_structure = await self.hierarchical_aggregator.build_hierarchy(embeddings)
            
            # Analizar relaciones semánticas
            embeddings.semantic_relationships = await self._analyze_semantic_relationships(embeddings)
            
            # Actualizar estadísticas
            embeddings.generation_time_ms = int((time.time() - start_time) * 1000)
            self._update_generation_stats(embeddings)
            
            logger.info(f"Generados {embeddings.get_total_embeddings()} embeddings en {embeddings.generation_time_ms}ms")
            
        except Exception as e:
            logger.error(f"Error generando embeddings multi-nivel: {e}")
            # Retornar resultado parcial
            embeddings.generation_time_ms = int((time.time() - start_time) * 1000)
        
        return embeddings
    
    async def _generate_token_level(
        self, 
        code: str, 
        context: EmbeddingGenerationContext, 
        embeddings: MultiLevelEmbeddings
    ) -> None:
        """Genera embeddings a nivel de token."""
        # Tokenización básica
        tokens = self._simple_tokenize(code)
        
        # Limitar tokens si es necesario
        if len(tokens) > self.config.max_tokens_per_level:
            tokens = tokens[:self.config.max_tokens_per_level]
        
        # Generar embeddings de tokens
        embeddings.token_embeddings = await self.token_embedder.generate_token_embeddings(
            tokens, context
        )
        
        self.generation_stats['levels_generated'].add(EmbeddingLevel.TOKEN)
    
    async def _generate_expression_level(
        self, 
        code: str, 
        context: EmbeddingGenerationContext, 
        embeddings: MultiLevelEmbeddings
    ) -> None:
        """Genera embeddings a nivel de expresión."""
        # Extraer expresiones básicas
        expressions = self._extract_expressions(code)
        
        # Generar embeddings de expresiones
        embeddings.expression_embeddings = await self.expression_embedder.generate_expression_embeddings(
            expressions, embeddings.token_embeddings, context
        )
        
        self.generation_stats['levels_generated'].add(EmbeddingLevel.EXPRESSION)
    
    async def _generate_function_level(
        self, 
        code: str, 
        context: EmbeddingGenerationContext, 
        embeddings: MultiLevelEmbeddings
    ) -> None:
        """Genera embeddings a nivel de función."""
        # Extraer funciones
        functions = self._extract_functions(code)
        
        # Generar embeddings de funciones
        embeddings.function_embeddings = await self.function_embedder.generate_function_embeddings(
            functions, context
        )
        
        self.generation_stats['levels_generated'].add(EmbeddingLevel.FUNCTION)
    
    async def _generate_file_level(
        self, 
        code: str, 
        context: EmbeddingGenerationContext, 
        embeddings: MultiLevelEmbeddings
    ) -> None:
        """Genera embedding a nivel de archivo."""
        # Generar embedding del archivo completo
        file_embedding_result = await self.base_embedding_engine.generate_embedding(
            code, context.language
        )
        
        # Extraer información del archivo
        imports = self._extract_imports(code)
        exports = self._extract_exports(code)
        file_purpose = self._detect_file_purpose(code, context.file_path)
        
        # Crear embedding de archivo
        embeddings.file_embedding = FileEmbedding(
            file_path=context.file_path,
            language=context.language,
            embedding_vector=file_embedding_result.embedding_vector,
            function_embeddings=list(embeddings.function_embeddings.keys()),
            imports=imports,
            exports=exports,
            file_purpose=file_purpose
        )
        
        self.generation_stats['levels_generated'].add(EmbeddingLevel.FILE)
    
    def _simple_tokenize(self, code: str) -> List[str]:
        """Tokenización simple del código."""
        import re
        
        # Patrones para diferentes tipos de tokens
        token_pattern = r'''
            \b\w+\b|           # Palabras
            \d+\.?\d*|         # Números
            "[^"]*"|           # Strings con comillas dobles
            '[^']*'|           # Strings con comillas simples
            [+\-*/=<>!&|]+|    # Operadores
            [(){}\[\];,.]      # Puntuación
        '''
        
        tokens = re.findall(token_pattern, code, re.VERBOSE)
        return [token for token in tokens if token.strip()]
    
    def _extract_expressions(self, code: str) -> List[str]:
        """Extrae expresiones del código."""
        expressions = []
        
        # Dividir por líneas y buscar expresiones
        lines = code.split('\n')
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                continue
            
            # Buscar asignaciones
            if '=' in stripped and not any(op in stripped for op in ['==', '!=', '<=', '>=']):
                expressions.append(stripped)
            
            # Buscar llamadas a función
            elif '(' in stripped and ')' in stripped:
                expressions.append(stripped)
            
            # Buscar operaciones
            elif any(op in stripped for op in ['+', '-', '*', '/', '<', '>']):
                expressions.append(stripped)
        
        return expressions[:100]  # Limitar cantidad
    
    def _extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extrae funciones del código."""
        functions = []
        lines = code.split('\n')
        current_function = None
        function_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Detectar inicio de función
            if any(stripped.startswith(keyword) for keyword in ['def ', 'function ', 'fn ']):
                if current_function:
                    # Guardar función anterior
                    functions.append({
                        'name': current_function,
                        'code': '\n'.join(function_lines)
                    })
                
                # Extraer nombre de función
                if stripped.startswith('def '):
                    current_function = stripped.split('def ')[1].split('(')[0].strip()
                elif stripped.startswith('function '):
                    current_function = stripped.split('function ')[1].split('(')[0].strip()
                elif stripped.startswith('fn '):
                    current_function = stripped.split('fn ')[1].split('(')[0].strip()
                
                function_lines = [line]
            
            elif current_function:
                function_lines.append(line)
                
                # Detectar fin de función (heurística simple)
                if not line.strip() and len(function_lines) > 10:
                    functions.append({
                        'name': current_function,
                        'code': '\n'.join(function_lines)
                    })
                    current_function = None
                    function_lines = []
        
        # Guardar última función si existe
        if current_function:
            functions.append({
                'name': current_function,
                'code': '\n'.join(function_lines)
            })
        
        return functions
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extrae imports del código."""
        imports = []
        lines = code.split('\n')
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                imports.append(stripped)
            elif stripped.startswith('const ') and ' = require(' in stripped:
                imports.append(stripped)
            elif stripped.startswith('use ') and '::' in stripped:
                imports.append(stripped)
        
        return imports
    
    def _extract_exports(self, code: str) -> List[str]:
        """Extrae exports del código."""
        exports = []
        lines = code.split('\n')
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('export ') or stripped.startswith('module.exports'):
                exports.append(stripped)
            elif stripped.startswith('pub '):
                exports.append(stripped)
        
        return exports
    
    def _detect_file_purpose(self, code: str, file_path: Path) -> str:
        """Detecta el propósito del archivo."""
        filename = file_path.name.lower()
        
        # Análisis por nombre de archivo
        if 'test' in filename:
            return "testing"
        elif 'config' in filename:
            return "configuration"
        elif 'util' in filename or 'helper' in filename:
            return "utility"
        elif 'model' in filename or 'entity' in filename:
            return "data_model"
        elif 'service' in filename:
            return "business_service"
        elif 'controller' in filename:
            return "request_handler"
        elif 'view' in filename or 'component' in filename:
            return "user_interface"
        
        # Análisis por contenido
        code_lower = code.lower()
        if 'class ' in code_lower and 'def __init__' in code_lower:
            return "class_definition"
        elif 'def ' in code_lower and code.count('def ') > 3:
            return "function_library"
        elif 'import ' in code_lower and code.count('\n') < 50:
            return "module_interface"
        
        return "general"
    
    async def _analyze_semantic_relationships(
        self, 
        embeddings: MultiLevelEmbeddings
    ) -> List[SemanticRelationship]:
        """Analiza relaciones semánticas entre elementos."""
        relationships = []
        
        # Analizar relaciones entre funciones
        func_embeddings = list(embeddings.function_embeddings.values())
        for i, func1 in enumerate(func_embeddings):
            for func2 in func_embeddings[i+1:]:
                similarity = self._calculate_semantic_similarity(
                    func1.embedding_vector, 
                    func2.embedding_vector
                )
                
                if similarity > 0.7:  # Umbral de similitud
                    relationship = SemanticRelationship(
                        source_id=func1.id,
                        target_id=func2.id,
                        relationship_type="similar",
                        strength=similarity,
                        confidence=similarity,
                        evidence=[f"Similarity score: {similarity:.3f}"],
                        semantic_distance=1.0 - similarity
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _calculate_semantic_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud semántica entre vectores."""
        if not vec1 or not vec2:
            return 0.0
        
        # Asegurar misma dimensión
        min_dim = min(len(vec1), len(vec2))
        v1 = vec1[:min_dim]
        v2 = vec2[:min_dim]
        
        if NUMPY_AVAILABLE:
            # Similitud coseno con numpy
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        else:
            # Similitud coseno manual
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm1 = sum(a * a for a in v1) ** 0.5
            norm2 = sum(b * b for b in v2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
    
    def _update_generation_stats(self, embeddings: MultiLevelEmbeddings) -> None:
        """Actualiza estadísticas de generación."""
        self.generation_stats['total_generated'] += embeddings.get_total_embeddings()
        self.generation_stats['generation_time_ms'] += embeddings.generation_time_ms
    
    async def get_generation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de generación."""
        return {
            "total_embeddings_generated": self.generation_stats['total_generated'],
            "total_generation_time_ms": self.generation_stats['generation_time_ms'],
            "levels_generated": [level.value for level in self.generation_stats['levels_generated']],
            "average_generation_time_ms": (
                self.generation_stats['generation_time_ms'] / 
                max(1, self.generation_stats['total_generated'])
            ),
            "config": {
                "enable_token_embeddings": self.config.enable_token_embeddings,
                "enable_expression_embeddings": self.config.enable_expression_embeddings,
                "enable_function_embeddings": self.config.enable_function_embeddings,
                "enable_file_embeddings": self.config.enable_file_embeddings,
                "aggregation_strategy": self.config.aggregation_strategy.value,
                "context_window_size": self.config.context_window_size
            }
        }
