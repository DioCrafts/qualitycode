"""
Extractor de features para detección de antipatrones usando IA.
"""

import ast
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from ...domain.entities.antipattern_analysis import (
    AntipatternFeatures, ClassFeatures, SecurityFeatures, PerformanceFeatures,
    ResponsibilityType, AlgorithmicComplexity
)
from ...domain.entities.ai_models import AIAnalysisResult
from ...domain.value_objects.programming_language import ProgrammingLanguage
from ...domain.value_objects.source_position import SourcePosition
from ..ast_analysis.unified_ast import UnifiedAST, UnifiedNode, UnifiedNodeType


logger = logging.getLogger(__name__)


class AntipatternFeatureExtractor:
    """Extractor de features para detección de antipatrones."""
    
    def __init__(self):
        self.structural_extractor = StructuralFeatureExtractor()
        self.semantic_extractor = SemanticFeatureExtractor()
        self.security_extractor = SecurityFeatureExtractor()
        self.performance_extractor = PerformanceFeatureExtractor()
        self.responsibility_extractor = ResponsibilityFeatureExtractor()
    
    async def extract_features(
        self, 
        ai_analysis: AIAnalysisResult, 
        unified_ast: UnifiedAST
    ) -> AntipatternFeatures:
        """Extraer todas las features para detección de antipatrones."""
        
        try:
            # Features estructurales básicas
            structural_features = await self.structural_extractor.extract(unified_ast)
            
            # Features semánticas del análisis de IA
            semantic_features = await self.semantic_extractor.extract(ai_analysis)
            
            # Features de seguridad
            security_features = await self.security_extractor.extract(unified_ast)
            
            # Features de performance
            performance_features = await self.performance_extractor.extract(unified_ast)
            
            # Features de responsabilidades
            responsibility_features = await self.responsibility_extractor.extract(unified_ast)
            
            # Combinar todas las features
            features = AntipatternFeatures(
                file_path=unified_ast.file_path,
                language=unified_ast.language,
                
                # Features estructurales
                lines_of_code=structural_features['lines_of_code'],
                methods_count=structural_features['methods_count'],
                classes_count=structural_features['classes_count'],
                functions_count=structural_features['functions_count'],
                max_method_length=structural_features['max_method_length'],
                max_class_size=structural_features['max_class_size'],
                
                # Features de complejidad
                cyclomatic_complexity=structural_features.get('cyclomatic_complexity', 0.0),
                cognitive_complexity=structural_features.get('cognitive_complexity', 0.0),
                nesting_depth=structural_features.get('nesting_depth', 0),
                
                # Features de acoplamiento
                import_count=structural_features.get('import_count', 0),
                external_dependencies=structural_features.get('external_dependencies', 0),
                class_coupling=structural_features.get('class_coupling', 0.0),
                
                # Features de responsabilidad
                distinct_responsibilities=responsibility_features['distinct_responsibilities'],
                responsibility_types=responsibility_features['responsibility_types'],
                
                # Features de seguridad
                has_sql_operations=security_features['has_sql_operations'],
                has_user_input=security_features['has_user_input'],
                has_file_operations=security_features['has_file_operations'],
                has_network_operations=security_features['has_network_operations'],
                has_crypto_operations=security_features['has_crypto_operations'],
                
                # Features de performance
                has_loops=performance_features['has_loops'],
                has_nested_loops=performance_features['has_nested_loops'],
                has_recursive_calls=performance_features['has_recursive_calls'],
                algorithmic_complexity=performance_features['algorithmic_complexity'],
                
                # Features adicionales del análisis IA
                custom_features={
                    'semantic_complexity': semantic_features.get('complexity_score', 0.0),
                    'ai_quality_score': semantic_features.get('quality_score', 0.0),
                    'pattern_density': semantic_features.get('pattern_density', 0.0)
                }
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting antipattern features: {e}")
            # Retornar features mínimas en caso de error
            return AntipatternFeatures(
                file_path=unified_ast.file_path,
                language=unified_ast.language
            )


class StructuralFeatureExtractor:
    """Extractor de features estructurales."""
    
    async def extract(self, unified_ast: UnifiedAST) -> Dict[str, Any]:
        """Extraer features estructurales del AST."""
        
        features = {
            'lines_of_code': 0,
            'methods_count': 0,
            'classes_count': 0,
            'functions_count': 0,
            'max_method_length': 0,
            'max_class_size': 0,
            'cyclomatic_complexity': 0.0,
            'cognitive_complexity': 0.0,
            'nesting_depth': 0,
            'import_count': 0,
            'external_dependencies': 0,
            'class_coupling': 0.0
        }
        
        try:
            # Contar líneas de código
            if unified_ast.source_code:
                features['lines_of_code'] = len([
                    line for line in unified_ast.source_code.split('\n') 
                    if line.strip() and not line.strip().startswith('#')
                ])
            
            # Analizar nodos del AST
            await self._analyze_ast_nodes(unified_ast.root_node, features)
            
        except Exception as e:
            logger.error(f"Error in structural feature extraction: {e}")
        
        return features
    
    async def _analyze_ast_nodes(self, node: UnifiedNode, features: Dict[str, Any]):
        """Analizar nodos del AST recursivamente."""
        
        # Contar elementos estructurales
        if node.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
            features['functions_count'] += 1
            method_length = self._calculate_node_length(node)
            features['max_method_length'] = max(features['max_method_length'], method_length)
        
        elif node.node_type == UnifiedNodeType.CLASS_DECLARATION:
            features['classes_count'] += 1
            class_size = self._calculate_node_length(node)
            features['max_class_size'] = max(features['max_class_size'], class_size)
        
        elif node.node_type == UnifiedNodeType.METHOD_DECLARATION:
            features['methods_count'] += 1
            method_length = self._calculate_node_length(node)
            features['max_method_length'] = max(features['max_method_length'], method_length)
        
        elif node.node_type == UnifiedNodeType.IMPORT:
            features['import_count'] += 1
        
        # Calcular profundidad de anidamiento
        current_depth = self._calculate_nesting_depth(node)
        features['nesting_depth'] = max(features['nesting_depth'], current_depth)
        
        # Calcular complejidad ciclomática aproximada
        if node.node_type in [UnifiedNodeType.IF_STATEMENT, UnifiedNodeType.WHILE_LOOP, 
                             UnifiedNodeType.FOR_LOOP, UnifiedNodeType.TRY_STATEMENT]:
            features['cyclomatic_complexity'] += 1
        
        # Procesar hijos
        for child in node.children:
            await self._analyze_ast_nodes(child, features)
    
    def _calculate_node_length(self, node: UnifiedNode) -> int:
        """Calcular la longitud aproximada de un nodo."""
        if node.position and node.position.end_line:
            return node.position.end_line - node.position.line + 1
        return 1
    
    def _calculate_nesting_depth(self, node: UnifiedNode, depth: int = 0) -> int:
        """Calcular profundidad máxima de anidamiento."""
        
        current_depth = depth
        
        # Incrementar profundidad para estructuras de control
        if node.node_type in [UnifiedNodeType.IF_STATEMENT, UnifiedNodeType.WHILE_LOOP,
                             UnifiedNodeType.FOR_LOOP, UnifiedNodeType.TRY_STATEMENT,
                             UnifiedNodeType.FUNCTION_DECLARATION, UnifiedNodeType.CLASS_DECLARATION]:
            current_depth += 1
        
        max_child_depth = current_depth
        for child in node.children:
            child_depth = self._calculate_nesting_depth(child, current_depth)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth


class SemanticFeatureExtractor:
    """Extractor de features semánticas del análisis de IA."""
    
    async def extract(self, ai_analysis: AIAnalysisResult) -> Dict[str, Any]:
        """Extraer features semánticas."""
        
        features = {
            'complexity_score': 0.0,
            'quality_score': 0.0,
            'pattern_density': 0.0,
            'semantic_coherence': 0.0
        }
        
        try:
            # Obtener embeddings si están disponibles
            if hasattr(ai_analysis, 'code_embeddings') and ai_analysis.code_embeddings:
                # Calcular métricas basadas en embeddings
                embedding_vector = ai_analysis.code_embeddings.embedding_vector
                features['complexity_score'] = self._calculate_embedding_complexity(embedding_vector)
                features['semantic_coherence'] = self._calculate_semantic_coherence(embedding_vector)
            
            # Obtener patterns detectados
            if hasattr(ai_analysis, 'patterns_detected') and ai_analysis.patterns_detected:
                features['pattern_density'] = len(ai_analysis.patterns_detected) / max(1, ai_analysis.lines_analyzed or 1)
            
            # Obtener quality score general
            if hasattr(ai_analysis, 'quality_metrics') and ai_analysis.quality_metrics:
                features['quality_score'] = ai_analysis.quality_metrics.get('overall_quality', 0.0)
                
        except Exception as e:
            logger.error(f"Error in semantic feature extraction: {e}")
        
        return features
    
    def _calculate_embedding_complexity(self, embedding_vector: List[float]) -> float:
        """Calcular complejidad basada en el vector de embedding."""
        if not embedding_vector:
            return 0.0
        
        # Usar la varianza del vector como medida de complejidad
        mean = sum(embedding_vector) / len(embedding_vector)
        variance = sum((x - mean) ** 2 for x in embedding_vector) / len(embedding_vector)
        
        # Normalizar entre 0 y 1
        return min(1.0, variance * 10)
    
    def _calculate_semantic_coherence(self, embedding_vector: List[float]) -> float:
        """Calcular coherencia semántica."""
        if not embedding_vector:
            return 0.0
        
        # Usar la magnitud del vector normalizada
        magnitude = sum(x ** 2 for x in embedding_vector) ** 0.5
        return min(1.0, magnitude / len(embedding_vector))


class SecurityFeatureExtractor:
    """Extractor de features de seguridad."""
    
    def __init__(self):
        # Patrones de riesgo de seguridad
        self.sql_patterns = [
            r'SELECT\s+.*\s+FROM',
            r'INSERT\s+INTO',
            r'UPDATE\s+.*\s+SET',
            r'DELETE\s+FROM',
            r'DROP\s+TABLE',
            r'ALTER\s+TABLE'
        ]
        
        self.secret_patterns = [
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'api[_-]?key\s*=\s*["\'][^"\']{20,}["\']',
            r'secret\s*=\s*["\'][^"\']{10,}["\']',
            r'token\s*=\s*["\'][^"\']{20,}["\']',
            r'["\'][A-Za-z0-9]{32,}["\']',  # Posibles hashes/tokens
        ]
        
        self.crypto_weak_patterns = [
            r'md5\s*\(',
            r'sha1\s*\(',
            r'des\s*\(',
            r'rc4\s*\(',
        ]
    
    async def extract(self, unified_ast: UnifiedAST) -> Dict[str, Any]:
        """Extraer features de seguridad."""
        
        features = {
            'has_sql_operations': False,
            'has_user_input': False,
            'has_file_operations': False,
            'has_network_operations': False,
            'has_crypto_operations': False,
            'has_hardcoded_secrets': False,
            'uses_weak_cryptography': False,
            'has_input_validation': False,
            'secret_patterns_found': [],
            'crypto_algorithms_found': []
        }
        
        try:
            source_code = unified_ast.source_code or ""
            source_lower = source_code.lower()
            
            # Detectar operaciones SQL
            for pattern in self.sql_patterns:
                if re.search(pattern, source_code, re.IGNORECASE):
                    features['has_sql_operations'] = True
                    break
            
            # Detectar secrets hardcodeados
            for pattern in self.secret_patterns:
                matches = re.findall(pattern, source_code, re.IGNORECASE)
                if matches:
                    features['has_hardcoded_secrets'] = True
                    features['secret_patterns_found'].extend(matches[:5])  # Limitar a 5
            
            # Detectar criptografía débil
            for pattern in self.crypto_weak_patterns:
                if re.search(pattern, source_code, re.IGNORECASE):
                    features['uses_weak_cryptography'] = True
                    features['crypto_algorithms_found'].append(pattern)
            
            # Detectar entrada de usuario
            user_input_indicators = [
                'input(', 'raw_input(', 'gets(', 'scanf(',
                'request.', 'params', 'query', 'form',
                'argv', 'stdin'
            ]
            
            for indicator in user_input_indicators:
                if indicator in source_lower:
                    features['has_user_input'] = True
                    break
            
            # Detectar operaciones de archivo
            file_operations = [
                'open(', 'file(', 'fopen(', 'read(', 'write(',
                'os.path', 'pathlib', 'glob', 'listdir'
            ]
            
            for operation in file_operations:
                if operation in source_lower:
                    features['has_file_operations'] = True
                    break
            
            # Detectar operaciones de red
            network_operations = [
                'socket', 'urllib', 'requests', 'http', 'https',
                'fetch(', 'ajax', 'xhr'
            ]
            
            for operation in network_operations:
                if operation in source_lower:
                    features['has_network_operations'] = True
                    break
            
            # Detectar operaciones criptográficas
            crypto_operations = [
                'encrypt', 'decrypt', 'hash', 'crypto', 'ssl',
                'aes', 'rsa', 'sha256', 'hmac'
            ]
            
            for operation in crypto_operations:
                if operation in source_lower:
                    features['has_crypto_operations'] = True
                    break
            
            # Detectar validación de entrada
            validation_indicators = [
                'validate', 'sanitize', 'escape', 'filter',
                'isinstance(', 'type(', 'assert', 'raise'
            ]
            
            for indicator in validation_indicators:
                if indicator in source_lower:
                    features['has_input_validation'] = True
                    break
                    
        except Exception as e:
            logger.error(f"Error in security feature extraction: {e}")
        
        return features


class PerformanceFeatureExtractor:
    """Extractor de features de performance."""
    
    async def extract(self, unified_ast: UnifiedAST) -> Dict[str, Any]:
        """Extraer features de performance."""
        
        features = {
            'has_loops': False,
            'has_nested_loops': False,
            'has_recursive_calls': False,
            'algorithmic_complexity': AlgorithmicComplexity.LINEAR,
            'loop_nesting_depth': 0,
            'recursive_function_count': 0,
            'database_calls_in_loops': False,
            'string_concatenation_in_loops': False
        }
        
        try:
            # Analizar AST para patterns de performance
            await self._analyze_performance_patterns(unified_ast.root_node, features, depth=0)
            
            # Estimar complejidad algorítmica
            features['algorithmic_complexity'] = self._estimate_algorithmic_complexity(features)
            
        except Exception as e:
            logger.error(f"Error in performance feature extraction: {e}")
        
        return features
    
    async def _analyze_performance_patterns(
        self, 
        node: UnifiedNode, 
        features: Dict[str, Any], 
        depth: int = 0,
        in_loop: bool = False
    ):
        """Analizar patrones de performance recursivamente."""
        
        # Detectar loops
        if node.node_type in [UnifiedNodeType.FOR_LOOP, UnifiedNodeType.WHILE_LOOP]:
            features['has_loops'] = True
            
            if in_loop:
                features['has_nested_loops'] = True
                features['loop_nesting_depth'] = max(features['loop_nesting_depth'], depth + 1)
            
            # Analizar contenido del loop
            for child in node.children:
                await self._analyze_performance_patterns(child, features, depth + 1, True)
            
            return
        
        # Detectar llamadas recursivas
        if node.node_type == UnifiedNodeType.FUNCTION_CALL:
            if node.name and hasattr(node, 'parent_function') and node.name == node.parent_function:
                features['has_recursive_calls'] = True
                features['recursive_function_count'] += 1
        
        # Detectar problemas específicos en loops
        if in_loop:
            if node.node_type == UnifiedNodeType.FUNCTION_CALL:
                # Detectar llamadas a base de datos en loops
                if node.name and any(db_indicator in node.name.lower() for db_indicator in 
                                   ['query', 'select', 'insert', 'update', 'delete', 'execute']):
                    features['database_calls_in_loops'] = True
            
            # Detectar concatenación de strings en loops
            if node.node_type == UnifiedNodeType.BINARY_OPERATION and node.value == '+':
                # Simplificado: detectar operaciones binarias que podrían ser concatenación
                features['string_concatenation_in_loops'] = True
        
        # Procesar hijos
        for child in node.children:
            await self._analyze_performance_patterns(child, features, depth, in_loop)
    
    def _estimate_algorithmic_complexity(self, features: Dict[str, Any]) -> AlgorithmicComplexity:
        """Estimar complejidad algorítmica basada en features."""
        
        if features['has_nested_loops']:
            if features['loop_nesting_depth'] >= 3:
                return AlgorithmicComplexity.CUBIC
            elif features['loop_nesting_depth'] >= 2:
                return AlgorithmicComplexity.QUADRATIC
            else:
                return AlgorithmicComplexity.LINEAR_LOGARITHMIC
        
        elif features['has_loops']:
            return AlgorithmicComplexity.LINEAR
        
        elif features['has_recursive_calls']:
            # Simplificado: asumir exponencial para recursión
            return AlgorithmicComplexity.EXPONENTIAL
        
        else:
            return AlgorithmicComplexity.CONSTANT


class ResponsibilityFeatureExtractor:
    """Extractor de features de responsabilidades."""
    
    def __init__(self):
        # Mapeo de keywords a tipos de responsabilidad
        self.responsibility_keywords = {
            ResponsibilityType.DATA_ACCESS: [
                'query', 'select', 'insert', 'update', 'delete', 'database',
                'collection', 'repository', 'dao', 'entity'
            ],
            ResponsibilityType.BUSINESS_LOGIC: [
                'calculate', 'compute', 'process', 'validate', 'business',
                'rule', 'policy', 'decision', 'algorithm'
            ],
            ResponsibilityType.PRESENTATION: [
                'render', 'display', 'view', 'template', 'ui', 'html',
                'css', 'style', 'format', 'print'
            ],
            ResponsibilityType.VALIDATION: [
                'validate', 'check', 'verify', 'assert', 'ensure',
                'sanitize', 'clean', 'filter'
            ],
            ResponsibilityType.COMMUNICATION: [
                'send', 'receive', 'request', 'response', 'http',
                'api', 'rest', 'soap', 'message', 'notification'
            ],
            ResponsibilityType.CONFIGURATION: [
                'config', 'settings', 'properties', 'parameter',
                'option', 'preference', 'environment'
            ],
            ResponsibilityType.LOGGING: [
                'log', 'debug', 'info', 'warn', 'error', 'trace',
                'audit', 'monitor', 'record'
            ],
            ResponsibilityType.ERROR_HANDLING: [
                'error', 'exception', 'try', 'catch', 'handle',
                'recover', 'fallback', 'retry'
            ],
            ResponsibilityType.SECURITY: [
                'auth', 'login', 'password', 'token', 'encrypt',
                'decrypt', 'secure', 'permission', 'role'
            ],
            ResponsibilityType.PERFORMANCE: [
                'cache', 'optimize', 'performance', 'benchmark',
                'profile', 'memory', 'speed', 'efficient'
            ]
        }
    
    async def extract(self, unified_ast: UnifiedAST) -> Dict[str, Any]:
        """Extraer features de responsabilidades."""
        
        features = {
            'distinct_responsibilities': 0,
            'responsibility_types': [],
            'responsibility_distribution': {},
            'single_responsibility_score': 0.0
        }
        
        try:
            if not unified_ast.source_code:
                return features
            
            source_lower = unified_ast.source_code.lower()
            detected_responsibilities = set()
            responsibility_counts = {}
            
            # Detectar responsabilidades por keywords
            for resp_type, keywords in self.responsibility_keywords.items():
                count = 0
                for keyword in keywords:
                    count += source_lower.count(keyword)
                
                if count > 0:
                    detected_responsibilities.add(resp_type)
                    responsibility_counts[resp_type] = count
            
            features['distinct_responsibilities'] = len(detected_responsibilities)
            features['responsibility_types'] = list(detected_responsibilities)
            features['responsibility_distribution'] = responsibility_counts
            
            # Calcular score de Single Responsibility Principle
            if features['distinct_responsibilities'] <= 1:
                features['single_responsibility_score'] = 1.0
            elif features['distinct_responsibilities'] <= 2:
                features['single_responsibility_score'] = 0.8
            elif features['distinct_responsibilities'] <= 3:
                features['single_responsibility_score'] = 0.6
            elif features['distinct_responsibilities'] <= 4:
                features['single_responsibility_score'] = 0.4
            else:
                features['single_responsibility_score'] = 0.2
                
        except Exception as e:
            logger.error(f"Error in responsibility feature extraction: {e}")
        
        return features
