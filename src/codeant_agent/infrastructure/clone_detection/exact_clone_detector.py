"""
Implementación del detector de clones exactos.

Este módulo implementa la detección de duplicación exacta usando técnicas
de hashing y normalización de código.
"""

import hashlib
import logging
import re
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time

from ...domain.entities.clone_analysis import (
    ExactClone, CodeBlock, CodeLocation, CloneId, CloneType,
    HashAlgorithm, DuplicationMetrics
)
from ...domain.entities.parse_result import ParseResult
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class NormalizationOptions:
    """Opciones para normalización de código."""
    remove_whitespace: bool = True
    remove_comments: bool = True
    remove_empty_lines: bool = True
    normalize_string_literals: bool = True
    normalize_numeric_literals: bool = True
    normalize_variable_names: bool = False
    normalize_case: bool = False


@dataclass
class ExactCloneDetectionResult:
    """Resultado de la detección de clones exactos."""
    exact_clones: List[ExactClone]
    code_blocks: List[CodeBlock]
    hash_collisions: Dict[str, List[CodeBlock]]
    analysis_time_ms: int
    normalization_time_ms: int
    hashing_time_ms: int


class CodeNormalizer:
    """Normalizador de código para comparación."""
    
    # Patrones de comentarios por lenguaje
    COMMENT_PATTERNS = {
        ProgrammingLanguage.PYTHON: [
            r'#.*$',  # Comentarios de línea
            r'"""[\s\S]*?"""',  # Docstrings triple comillas
            r"'''[\s\S]*?'''",  # Docstrings triple comillas simples
        ],
        ProgrammingLanguage.JAVASCRIPT: [
            r'//.*$',  # Comentarios de línea
            r'/\*[\s\S]*?\*/',  # Comentarios de bloque
        ],
        ProgrammingLanguage.TYPESCRIPT: [
            r'//.*$',  # Comentarios de línea
            r'/\*[\s\S]*?\*/',  # Comentarios de bloque
        ],
        ProgrammingLanguage.RUST: [
            r'//.*$',  # Comentarios de línea
            r'/\*[\s\S]*?\*/',  # Comentarios de bloque
        ],
    }
    
    def normalize_code(
        self, 
        code: str, 
        language: ProgrammingLanguage,
        options: NormalizationOptions
    ) -> str:
        """
        Normaliza código según las opciones especificadas.
        
        Args:
            code: Código a normalizar
            language: Lenguaje del código
            options: Opciones de normalización
            
        Returns:
            Código normalizado
        """
        normalized = code
        
        # Remover comentarios
        if options.remove_comments:
            normalized = self._remove_comments(normalized, language)
        
        # Remover espacios en blanco extra
        if options.remove_whitespace:
            normalized = self._normalize_whitespace(normalized)
        
        # Remover líneas vacías
        if options.remove_empty_lines:
            normalized = self._remove_empty_lines(normalized)
        
        # Normalizar literales de string
        if options.normalize_string_literals:
            normalized = self._normalize_string_literals(normalized)
        
        # Normalizar literales numéricos
        if options.normalize_numeric_literals:
            normalized = self._normalize_numeric_literals(normalized)
        
        # Normalizar nombres de variables
        if options.normalize_variable_names:
            normalized = self._normalize_variable_names(normalized, language)
        
        # Normalizar case
        if options.normalize_case:
            normalized = normalized.lower()
        
        return normalized.strip()
    
    def _remove_comments(self, code: str, language: ProgrammingLanguage) -> str:
        """Remueve comentarios del código."""
        patterns = self.COMMENT_PATTERNS.get(language, [])
        
        for pattern in patterns:
            if pattern.endswith('$'):
                # Comentarios de línea - procesar línea por línea
                lines = code.split('\n')
                cleaned_lines = []
                for line in lines:
                    cleaned_line = re.sub(pattern, '', line, flags=re.MULTILINE)
                    cleaned_lines.append(cleaned_line)
                code = '\n'.join(cleaned_lines)
            else:
                # Comentarios de bloque
                code = re.sub(pattern, '', code, flags=re.DOTALL | re.MULTILINE)
        
        return code
    
    def _normalize_whitespace(self, code: str) -> str:
        """Normaliza espacios en blanco."""
        # Reemplazar múltiples espacios con uno solo
        code = re.sub(r' +', ' ', code)
        
        # Normalizar tabs a espacios
        code = code.replace('\t', '    ')
        
        # Remover espacios al final de líneas
        lines = code.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        return '\n'.join(cleaned_lines)
    
    def _remove_empty_lines(self, code: str) -> str:
        """Remueve líneas vacías."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        return '\n'.join(non_empty_lines)
    
    def _normalize_string_literals(self, code: str) -> str:
        """Normaliza literales de string."""
        # Reemplazar strings con placeholder
        string_patterns = [
            r'"[^"]*"',      # Comillas dobles
            r"'[^']*'",      # Comillas simples
            r'`[^`]*`',      # Template strings (JS/TS)
            r'f"[^"]*"',     # f-strings Python
            r"f'[^']*'",     # f-strings Python
        ]
        
        for pattern in string_patterns:
            code = re.sub(pattern, '"STRING_LITERAL"', code)
        
        return code
    
    def _normalize_numeric_literals(self, code: str) -> str:
        """Normaliza literales numéricos."""
        # Reemplazar números con placeholder
        numeric_patterns = [
            r'\b\d+\.\d+\b',  # Decimales
            r'\b\d+\b',       # Enteros
            r'\b0x[0-9a-fA-F]+\b',  # Hexadecimales
            r'\b0b[01]+\b',   # Binarios
        ]
        
        for pattern in numeric_patterns:
            code = re.sub(pattern, 'NUMERIC_LITERAL', code)
        
        return code
    
    def _normalize_variable_names(self, code: str, language: ProgrammingLanguage) -> str:
        """Normaliza nombres de variables."""
        # Implementación básica - reemplazar identificadores con placeholders
        # En una implementación real, se usaría el AST para mayor precisión
        
        if language == ProgrammingLanguage.PYTHON:
            # Reemplazar identificadores Python
            code = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', 'VAR', code)
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            # Reemplazar identificadores JS/TS
            code = re.sub(r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b', 'VAR', code)
        
        return code


class CodeHasher:
    """Calculador de hashes de código."""
    
    def __init__(self, algorithm: HashAlgorithm = HashAlgorithm.SHA256):
        self.algorithm = algorithm
    
    def hash_code(self, code: str) -> str:
        """
        Calcula hash del código usando el algoritmo especificado.
        
        Args:
            code: Código a hashear
            
        Returns:
            Hash del código como string
        """
        if self.algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(code.encode('utf-8')).hexdigest()
        elif self.algorithm == HashAlgorithm.MD5:
            return hashlib.md5(code.encode('utf-8')).hexdigest()
        elif self.algorithm == HashAlgorithm.SIMHASH:
            return self._calculate_simhash(code)
        else:
            # Default to SHA256
            return hashlib.sha256(code.encode('utf-8')).hexdigest()
    
    def _calculate_simhash(self, code: str) -> str:
        """Calcula SimHash para detección de similitud aproximada."""
        # Tokenizar el código
        tokens = self._tokenize_code(code)
        
        # Vector de características
        feature_vector = [0] * 64
        
        # Procesar cada token
        for token in tokens:
            token_hash = int(hashlib.sha256(token.encode()).hexdigest()[:16], 16)
            
            for i in range(64):
                if (token_hash >> i) & 1:
                    feature_vector[i] += 1
                else:
                    feature_vector[i] -= 1
        
        # Generar SimHash
        simhash = 0
        for i in range(64):
            if feature_vector[i] > 0:
                simhash |= (1 << i)
        
        return format(simhash, '016x')
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Tokeniza código en tokens significativos."""
        # Tokenización básica - separar por espacios y símbolos
        tokens = re.findall(r'\w+', code)
        return [token.lower() for token in tokens if len(token) > 1]
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Calcula distancia de Hamming entre dos hashes."""
        if len(hash1) != len(hash2):
            raise ValueError("Los hashes deben tener la misma longitud")
        
        distance = 0
        for i in range(len(hash1)):
            if hash1[i] != hash2[i]:
                distance += 1
        
        return distance


class CodeBlockExtractor:
    """Extractor de bloques de código para análisis."""
    
    def extract_code_blocks(
        self, 
        parse_result: ParseResult,
        min_size_lines: int = 6,
        min_size_tokens: int = 50
    ) -> List[CodeBlock]:
        """
        Extrae bloques de código del resultado de parsing.
        
        Args:
            parse_result: Resultado del parsing
            min_size_lines: Tamaño mínimo en líneas
            min_size_tokens: Tamaño mínimo en tokens
            
        Returns:
            Lista de bloques de código extraídos
        """
        blocks = []
        
        # Obtener contenido completo del archivo
        if hasattr(parse_result.tree.root_node, 'text'):
            full_content = parse_result.tree.root_node.text.decode('utf-8')
        else:
            return blocks  # No hay contenido para analizar
        
        lines = full_content.split('\n')
        total_lines = len(lines)
        
        # Extraer bloques de diferentes tamaños
        for start_line in range(total_lines):
            for size in range(min_size_lines, min(total_lines - start_line + 1, 50)):
                end_line = start_line + size - 1
                
                if end_line >= total_lines:
                    break
                
                # Extraer contenido del bloque
                block_lines = lines[start_line:end_line + 1]
                block_content = '\n'.join(block_lines)
                
                # Verificar tamaño mínimo
                if len(block_lines) < min_size_lines:
                    continue
                
                # Contar tokens aproximadamente
                token_count = len(re.findall(r'\w+', block_content))
                if token_count < min_size_tokens:
                    continue
                
                # Crear bloque
                location = CodeLocation(
                    file_path=parse_result.file_path,
                    start_line=start_line + 1,  # 1-indexed
                    end_line=end_line + 1,
                    start_column=0,
                    end_column=len(block_lines[-1]) if block_lines else 0
                )
                
                block = CodeBlock(
                    content=block_content,
                    location=location,
                    size_lines=len(block_lines),
                    size_tokens=token_count,
                    language=parse_result.language
                )
                
                blocks.append(block)
        
        return blocks
    
    def extract_function_blocks(self, parse_result: ParseResult) -> List[CodeBlock]:
        """Extrae bloques a nivel de función."""
        blocks = []
        
        # Usar el AST para encontrar funciones
        self._extract_functions_recursive(
            parse_result.tree.root_node, 
            blocks, 
            parse_result
        )
        
        return blocks
    
    def _extract_functions_recursive(
        self, 
        ast_node: Any, 
        blocks: List[CodeBlock],
        parse_result: ParseResult
    ) -> None:
        """Extrae funciones recursivamente del AST."""
        if hasattr(ast_node, 'type'):
            node_type = ast_node.type
            
            if node_type in ['function_definition', 'method_definition', 'function_declaration']:
                # Extraer función completa
                if hasattr(ast_node, 'text') and ast_node.text:
                    function_content = ast_node.text.decode('utf-8')
                    
                    location = CodeLocation(
                        file_path=parse_result.file_path,
                        start_line=ast_node.start_point[0] + 1,
                        end_line=ast_node.end_point[0] + 1,
                        start_column=ast_node.start_point[1],
                        end_column=ast_node.end_point[1]
                    )
                    
                    # Contar líneas y tokens
                    lines = function_content.split('\n')
                    tokens = len(re.findall(r'\w+', function_content))
                    
                    block = CodeBlock(
                        content=function_content,
                        location=location,
                        size_lines=len(lines),
                        size_tokens=tokens,
                        language=parse_result.language
                    )
                    
                    blocks.append(block)
        
        # Procesar hijos
        if hasattr(ast_node, 'children'):
            for child in ast_node.children:
                self._extract_functions_recursive(child, blocks, parse_result)


class ExactCloneDetector:
    """Detector de clones exactos."""
    
    def __init__(
        self, 
        hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        normalization_options: Optional[NormalizationOptions] = None
    ):
        """
        Inicializa el detector de clones exactos.
        
        Args:
            hash_algorithm: Algoritmo de hashing a usar
            normalization_options: Opciones de normalización
        """
        self.hasher = CodeHasher(hash_algorithm)
        self.normalizer = CodeNormalizer()
        self.block_extractor = CodeBlockExtractor()
        self.normalization_options = normalization_options or NormalizationOptions()
    
    async def detect_exact_clones(
        self, 
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> ExactCloneDetectionResult:
        """
        Detecta clones exactos en un archivo.
        
        Args:
            parse_result: Resultado del parsing
            config: Configuración opcional
            
        Returns:
            ExactCloneDetectionResult con los clones encontrados
        """
        start_time = time.time()
        
        try:
            # Configurar parámetros
            min_size_lines = config.get('min_clone_size_lines', 6) if config else 6
            min_size_tokens = config.get('min_clone_size_tokens', 50) if config else 50
            
            # 1. Extraer bloques de código
            logger.debug(f"Extrayendo bloques de código de {parse_result.file_path}")
            code_blocks = self.block_extractor.extract_code_blocks(
                parse_result, min_size_lines, min_size_tokens
            )
            
            # 2. Normalizar y hashear bloques
            normalization_start = time.time()
            normalized_blocks = []
            for block in code_blocks:
                normalized_content = self.normalizer.normalize_code(
                    block.content,
                    block.language,
                    self.normalization_options
                )
                
                block.normalized_content = normalized_content
                block.hash_value = self.hasher.hash_code(normalized_content)
                normalized_blocks.append(block)
            
            normalization_time = int((time.time() - normalization_start) * 1000)
            
            # 3. Agrupar por hash y detectar duplicados
            hashing_start = time.time()
            hash_to_blocks: Dict[str, List[CodeBlock]] = defaultdict(list)
            
            for block in normalized_blocks:
                hash_to_blocks[block.hash_value].append(block)
            
            hashing_time = int((time.time() - hashing_start) * 1000)
            
            # 4. Crear clones exactos
            exact_clones = []
            hash_collisions = {}
            
            for hash_value, blocks in hash_to_blocks.items():
                if len(blocks) > 1:
                    hash_collisions[hash_value] = blocks
                    
                    # Crear clones para cada par de bloques duplicados
                    for i in range(len(blocks)):
                        for j in range(i + 1, len(blocks)):
                            original_block = blocks[i]
                            duplicate_block = blocks[j]
                            
                            # Verificar que no sean el mismo bloque
                            if (original_block.location.file_path == duplicate_block.location.file_path and
                                original_block.location.start_line == duplicate_block.location.start_line):
                                continue
                            
                            exact_clone = ExactClone(
                                id=CloneId(),
                                clone_type=CloneType.EXACT,
                                original_location=original_block.location,
                                duplicate_location=duplicate_block.location,
                                similarity_score=1.0,
                                confidence=1.0,
                                size_lines=original_block.size_lines,
                                size_tokens=original_block.size_tokens,
                                content=original_block.content,
                                hash_value=hash_value,
                                normalization_applied=self._get_applied_normalizations()
                            )
                            
                            exact_clones.append(exact_clone)
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Detección de clones exactos completada para {parse_result.file_path}: "
                f"{len(exact_clones)} clones encontrados en {total_time}ms"
            )
            
            return ExactCloneDetectionResult(
                exact_clones=exact_clones,
                code_blocks=code_blocks,
                hash_collisions=hash_collisions,
                analysis_time_ms=total_time,
                normalization_time_ms=normalization_time,
                hashing_time_ms=hashing_time
            )
            
        except Exception as e:
            logger.error(f"Error detectando clones exactos: {e}")
            raise
    
    async def detect_exact_clones_between_files(
        self, 
        parse_results: List[ParseResult],
        config: Optional[Dict[str, Any]] = None
    ) -> List[ExactClone]:
        """
        Detecta clones exactos entre múltiples archivos.
        
        Args:
            parse_results: Lista de resultados de parsing
            config: Configuración opcional
            
        Returns:
            Lista de clones exactos entre archivos
        """
        try:
            all_blocks = []
            
            # Extraer bloques de todos los archivos
            for parse_result in parse_results:
                file_blocks = self.block_extractor.extract_code_blocks(parse_result)
                all_blocks.extend(file_blocks)
            
            # Normalizar y hashear todos los bloques
            hash_to_blocks: Dict[str, List[CodeBlock]] = defaultdict(list)
            
            for block in all_blocks:
                normalized_content = self.normalizer.normalize_code(
                    block.content,
                    block.language,
                    self.normalization_options
                )
                
                hash_value = self.hasher.hash_code(normalized_content)
                hash_to_blocks[hash_value].append(block)
            
            # Crear clones inter-archivo
            inter_file_clones = []
            
            for hash_value, blocks in hash_to_blocks.items():
                if len(blocks) > 1:
                    # Verificar que hay al menos un par de bloques en archivos diferentes
                    for i in range(len(blocks)):
                        for j in range(i + 1, len(blocks)):
                            block1 = blocks[i]
                            block2 = blocks[j]
                            
                            # Solo considerar si están en archivos diferentes
                            if block1.location.file_path != block2.location.file_path:
                                exact_clone = ExactClone(
                                    id=CloneId(),
                                    clone_type=CloneType.EXACT,
                                    original_location=block1.location,
                                    duplicate_location=block2.location,
                                    similarity_score=1.0,
                                    confidence=1.0,
                                    size_lines=block1.size_lines,
                                    size_tokens=block1.size_tokens,
                                    content=block1.content,
                                    hash_value=hash_value,
                                    normalization_applied=self._get_applied_normalizations()
                                )
                                
                                inter_file_clones.append(exact_clone)
            
            logger.info(f"Detección inter-archivo completada: {len(inter_file_clones)} clones encontrados")
            
            return inter_file_clones
            
        except Exception as e:
            logger.error(f"Error detectando clones inter-archivo: {e}")
            raise
    
    def _get_applied_normalizations(self) -> List[str]:
        """Obtiene lista de normalizaciones aplicadas."""
        applied = []
        
        if self.normalization_options.remove_whitespace:
            applied.append("whitespace_removal")
        if self.normalization_options.remove_comments:
            applied.append("comment_removal")
        if self.normalization_options.normalize_string_literals:
            applied.append("string_literal_normalization")
        if self.normalization_options.normalize_numeric_literals:
            applied.append("numeric_literal_normalization")
        if self.normalization_options.normalize_variable_names:
            applied.append("variable_name_normalization")
        
        return applied
    
    def calculate_exact_clone_metrics(
        self, 
        exact_clones: List[ExactClone],
        total_lines: int
    ) -> DuplicationMetrics:
        """
        Calcula métricas para clones exactos.
        
        Args:
            exact_clones: Lista de clones exactos
            total_lines: Líneas totales del código
            
        Returns:
            DuplicationMetrics calculadas
        """
        if not exact_clones:
            return DuplicationMetrics(total_lines=total_lines)
        
        # Calcular líneas duplicadas (evitando doble conteo)
        duplicated_lines = 0
        processed_locations = set()
        
        for clone in exact_clones:
            # Contar cada ubicación solo una vez
            locations = [clone.original_location, clone.duplicate_location]
            
            for location in locations:
                location_key = (str(location.file_path), location.start_line, location.end_line)
                if location_key not in processed_locations:
                    duplicated_lines += clone.size_lines
                    processed_locations.add(location_key)
        
        # Calcular métricas
        metrics = DuplicationMetrics(
            total_lines=total_lines,
            duplicated_lines=duplicated_lines,
            total_clones=len(exact_clones),
            exact_clones=len(exact_clones),
            largest_clone_size=max(clone.size_lines for clone in exact_clones) if exact_clones else 0
        )
        
        metrics.calculate_derived_metrics()
        
        return metrics
