"""
Implementación del calculador de métricas de Halstead.

Este módulo implementa el cálculo de métricas de Halstead incluyendo
operadores, operandos, volumen, dificultad, esfuerzo y bugs predichos.
"""

import logging
import asyncio
import math
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from collections import Counter, defaultdict

from ...domain.entities.code_metrics import HalsteadMetrics, ComplexityMetrics
from ...domain.entities.parse_result import ParseResult
from ...domain.entities.ast_normalization import NormalizedNode as UnifiedNode, NodeType as UnifiedNodeType
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class OperatorCollection:
    """Colección de operadores encontrados."""
    operators: Counter = field(default_factory=Counter)
    
    def add_operator(self, operator: str) -> None:
        """Añade un operador."""
        self.operators[operator] += 1
    
    def distinct_count(self) -> int:
        """Número de operadores distintos (n1)."""
        return len(self.operators)
    
    def total_count(self) -> int:
        """Número total de operadores (N1)."""
        return sum(self.operators.values())
    
    def get_most_common(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Obtiene operadores más comunes."""
        return self.operators.most_common(limit)


@dataclass
class OperandCollection:
    """Colección de operandos encontrados."""
    operands: Counter = field(default_factory=Counter)
    
    def add_operand(self, operand: str) -> None:
        """Añade un operando."""
        if operand and operand.strip():  # Solo operandos no vacíos
            self.operands[operand.strip()] += 1
    
    def distinct_count(self) -> int:
        """Número de operandos distintos (n2)."""
        return len(self.operands)
    
    def total_count(self) -> int:
        """Número total de operandos (N2)."""
        return sum(self.operands.values())
    
    def get_most_common(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Obtiene operandos más comunes."""
        return self.operands.most_common(limit)


@dataclass
class HalsteadAnalysisResult:
    """Resultado del análisis de Halstead."""
    metrics: HalsteadMetrics
    operators: OperatorCollection
    operands: OperandCollection
    analysis_time_ms: int
    language_specific_info: Dict[str, Any]


class OperatorExtractor:
    """Extractor de operadores del código."""
    
    def __init__(self):
        # Operadores por lenguaje
        self.language_operators = {
            ProgrammingLanguage.PYTHON: {
                'arithmetic': ['+', '-', '*', '/', '//', '%', '**'],
                'comparison': ['==', '!=', '<', '>', '<=', '>=', 'is', 'is not', 'in', 'not in'],
                'logical': ['and', 'or', 'not'],
                'bitwise': ['&', '|', '^', '~', '<<', '>>'],
                'assignment': ['=', '+=', '-=', '*=', '/=', '//=', '%=', '**=', '&=', '|=', '^=', '<<=', '>>='],
                'control': ['if', 'elif', 'else', 'for', 'while', 'break', 'continue', 'return', 'yield', 'raise', 'try', 'except', 'finally'],
                'structural': ['.', '[', ']', '(', ')', '{', '}', ':', ',', ';']
            },
            ProgrammingLanguage.JAVASCRIPT: {
                'arithmetic': ['+', '-', '*', '/', '%', '++', '--'],
                'comparison': ['==', '!=', '===', '!==', '<', '>', '<=', '>='],
                'logical': ['&&', '||', '!'],
                'bitwise': ['&', '|', '^', '~', '<<', '>>', '>>>'],
                'assignment': ['=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '>>>='],
                'control': ['if', 'else', 'for', 'while', 'do', 'break', 'continue', 'return', 'throw', 'try', 'catch', 'finally', 'switch', 'case', 'default'],
                'structural': ['.', '[', ']', '(', ')', '{', '}', ':', ',', ';', '=>']
            },
            ProgrammingLanguage.RUST: {
                'arithmetic': ['+', '-', '*', '/', '%'],
                'comparison': ['==', '!=', '<', '>', '<=', '>='],
                'logical': ['&&', '||', '!'],
                'bitwise': ['&', '|', '^', '!', '<<', '>>'],
                'assignment': ['=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>='],
                'control': ['if', 'else', 'for', 'while', 'loop', 'break', 'continue', 'return', 'match', 'fn', 'let', 'mut'],
                'structural': ['.', '[', ']', '(', ')', '{', '}', ':', ',', ';', '->', '=>', '::']
            }
        }
    
    def extract_operators(self, node: UnifiedNode, language: ProgrammingLanguage) -> OperatorCollection:
        """
        Extrae operadores del AST.
        
        Args:
            node: Nodo raíz del AST
            language: Lenguaje del código
            
        Returns:
            OperatorCollection con operadores encontrados
        """
        operators = OperatorCollection()
        language_ops = self.language_operators.get(language, self.language_operators[ProgrammingLanguage.PYTHON])
        
        self._extract_recursive(node, operators, language, language_ops)
        return operators
    
    def _extract_recursive(self, node: UnifiedNode, operators: OperatorCollection, 
                          language: ProgrammingLanguage, language_ops: Dict[str, List[str]]) -> None:
        """Extrae operadores recursivamente."""
        
        # Extraer operadores según tipo de nodo
        if node.node_type == UnifiedNodeType.BINARY_EXPRESSION:
            op = self._extract_binary_operator(node, language_ops)
            if op:
                operators.add_operator(op)
        
        elif node.node_type == UnifiedNodeType.UNARY_EXPRESSION:
            op = self._extract_unary_operator(node, language_ops)
            if op:
                operators.add_operator(op)
        
        elif node.node_type == UnifiedNodeType.ASSIGNMENT_EXPRESSION:
            operators.add_operator("=")
        
        elif node.node_type == UnifiedNodeType.CALL_EXPRESSION:
            operators.add_operator("()")
        
        elif node.node_type == UnifiedNodeType.MEMBER_EXPRESSION:
            operators.add_operator(".")
        
        elif node.node_type == UnifiedNodeType.ARRAY_EXPRESSION:
            operators.add_operator("[]")
        
        elif node.node_type == UnifiedNodeType.IF_STATEMENT:
            operators.add_operator("if")
        
        elif node.node_type == UnifiedNodeType.FOR_STATEMENT:
            operators.add_operator("for")
        
        elif node.node_type == UnifiedNodeType.WHILE_STATEMENT:
            operators.add_operator("while")
        
        elif node.node_type == UnifiedNodeType.RETURN_STATEMENT:
            operators.add_operator("return")
        
        elif node.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
            if language == ProgrammingLanguage.PYTHON:
                operators.add_operator("def")
            elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
                operators.add_operator("function")
            elif language == ProgrammingLanguage.RUST:
                operators.add_operator("fn")
        
        elif node.node_type == UnifiedNodeType.CLASS_DECLARATION:
            if language == ProgrammingLanguage.PYTHON:
                operators.add_operator("class")
            elif language == ProgrammingLanguage.RUST:
                operators.add_operator("struct")
        
        # Procesar hijos
        for child in node.children:
            self._extract_recursive(child, operators, language, language_ops)
    
    def _extract_binary_operator(self, node: UnifiedNode, language_ops: Dict[str, List[str]]) -> Optional[str]:
        """Extrae operador binario del nodo."""
        if hasattr(node, 'value') and node.value:
            # Buscar operadores conocidos en el valor
            for category, ops in language_ops.items():
                for op in ops:
                    if op in node.value:
                        return op
        return None
    
    def _extract_unary_operator(self, node: UnifiedNode, language_ops: Dict[str, List[str]]) -> Optional[str]:
        """Extrae operador unario del nodo."""
        if hasattr(node, 'value') and node.value:
            unary_ops = ['!', 'not', '~', '-', '+', '++', '--']
            for op in unary_ops:
                if node.value.strip().startswith(op):
                    return op
        return None


class OperandExtractor:
    """Extractor de operandos del código."""
    
    def extract_operands(self, node: UnifiedNode, language: ProgrammingLanguage) -> OperandCollection:
        """
        Extrae operandos del AST.
        
        Args:
            node: Nodo raíz del AST
            language: Lenguaje del código
            
        Returns:
            OperandCollection con operandos encontrados
        """
        operands = OperandCollection()
        self._extract_recursive(node, operands, language)
        return operands
    
    def _extract_recursive(self, node: UnifiedNode, operands: OperandCollection, 
                          language: ProgrammingLanguage) -> None:
        """Extrae operandos recursivamente."""
        
        # Extraer operandos según tipo de nodo
        if node.node_type == UnifiedNodeType.IDENTIFIER:
            if hasattr(node, 'value') and node.value:
                operands.add_operand(node.value)
        
        elif node.node_type == UnifiedNodeType.STRING_LITERAL:
            if hasattr(node, 'value') and node.value:
                # Usar el literal como operando único (no el contenido)
                operands.add_operand("STRING_LITERAL")
        
        elif node.node_type == UnifiedNodeType.NUMBER_LITERAL:
            if hasattr(node, 'value') and node.value:
                operands.add_operand(node.value)
        
        elif node.node_type == UnifiedNodeType.BOOLEAN_LITERAL:
            if hasattr(node, 'value') and node.value:
                operands.add_operand(node.value)
        
        elif node.node_type == UnifiedNodeType.NULL_LITERAL:
            operands.add_operand("null")
        
        # Extraer nombres de funciones como operandos
        elif node.node_type == UnifiedNodeType.CALL_EXPRESSION:
            function_name = self._extract_function_name_from_call(node)
            if function_name:
                operands.add_operand(function_name)
        
        # Procesar hijos
        for child in node.children:
            self._extract_recursive(child, operands, language)
    
    def _extract_function_name_from_call(self, node: UnifiedNode) -> Optional[str]:
        """Extrae nombre de función de una llamada."""
        # Buscar primer hijo que sea identificador
        for child in node.children:
            if child.node_type == UnifiedNodeType.IDENTIFIER and child.value:
                return child.value
        return None


class HalsteadCalculator:
    """Calculadora principal de métricas de Halstead."""
    
    def __init__(self):
        self.operator_extractor = OperatorExtractor()
        self.operand_extractor = OperandExtractor()
    
    async def calculate_halstead_metrics(
        self, 
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> HalsteadAnalysisResult:
        """
        Calcula métricas de Halstead completas.
        
        Args:
            parse_result: Resultado del parsing
            config: Configuración opcional
            
        Returns:
            HalsteadAnalysisResult completo
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Calculando métricas de Halstead para {parse_result.file_path}")
            
            # Convertir a unified node
            unified_root = self._convert_to_unified_node(parse_result.tree.root_node)
            
            # Extraer operadores y operandos
            operators = self.operator_extractor.extract_operators(unified_root, parse_result.language)
            operands = self.operand_extractor.extract_operands(unified_root, parse_result.language)
            
            # Calcular métricas de Halstead
            metrics = self._calculate_metrics(operators, operands)
            
            # Información específica del lenguaje
            language_info = self._collect_language_specific_info(unified_root, parse_result.language, operators, operands)
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Métricas de Halstead calculadas para {parse_result.file_path}: "
                f"Volumen={metrics.volume:.1f}, Dificultad={metrics.difficulty:.1f}, "
                f"Esfuerzo={metrics.effort:.1f} en {total_time}ms"
            )
            
            return HalsteadAnalysisResult(
                metrics=metrics,
                operators=operators,
                operands=operands,
                analysis_time_ms=total_time,
                language_specific_info=language_info
            )
            
        except Exception as e:
            logger.error(f"Error calculando métricas de Halstead: {e}")
            raise
    
    async def calculate_function_halstead(self, function_node: UnifiedNode, language: ProgrammingLanguage) -> HalsteadMetrics:
        """Calcula métricas de Halstead para una función específica."""
        operators = self.operator_extractor.extract_operators(function_node, language)
        operands = self.operand_extractor.extract_operands(function_node, language)
        
        return self._calculate_metrics(operators, operands)
    
    def _calculate_metrics(self, operators: OperatorCollection, operands: OperandCollection) -> HalsteadMetrics:
        """Calcula métricas de Halstead desde las colecciones."""
        # Valores básicos
        n1 = operators.distinct_count()
        n2 = operands.distinct_count()
        N1 = operators.total_count()
        N2 = operands.total_count()
        
        # Crear métricas
        metrics = HalsteadMetrics(
            distinct_operators=n1,
            distinct_operands=n2,
            total_operators=N1,
            total_operands=N2
        )
        
        # Calcular métricas derivadas
        metrics.calculate_derived_metrics()
        
        return metrics
    
    def _convert_to_unified_node(self, tree_sitter_node) -> UnifiedNode:
        """Convierte tree-sitter node a UnifiedNode."""
        from ...domain.entities.ast_normalization import SourcePosition
        
        # Crear SourcePosition segura
        try:
            start_point = getattr(tree_sitter_node, 'start_point', (0, 0))
            end_point = getattr(tree_sitter_node, 'end_point', (0, 0))
            start_byte = getattr(tree_sitter_node, 'start_byte', 0)
            end_byte = getattr(tree_sitter_node, 'end_byte', 0)
            
            # Asegurar que son enteros
            if isinstance(start_point, tuple) and len(start_point) >= 2:
                start_line, start_col = start_point[0], start_point[1]
            else:
                start_line, start_col = 0, 0
            
            if isinstance(end_point, tuple) and len(end_point) >= 2:
                end_line, end_col = end_point[0], end_point[1]
            else:
                end_line, end_col = 0, 0
            
            # Asegurar que start_byte y end_byte son enteros
            if not isinstance(start_byte, int):
                start_byte = 0
            if not isinstance(end_byte, int):
                end_byte = 0
                
            position = SourcePosition(
                start_line=int(start_line),
                start_column=int(start_col),
                end_line=int(end_line),
                end_column=int(end_col),
                start_byte=int(start_byte),
                end_byte=int(end_byte)
            )
        except Exception as e:
            logger.warning(f"Error creando posición: {e}, usando posición por defecto")
            position = SourcePosition(start_line=0, start_column=0, end_line=0, end_column=0, start_byte=0, end_byte=0)
        
        unified_node = UnifiedNode(
            node_type=UnifiedNodeType.LANGUAGE_SPECIFIC,
            position=position,
            children=[],
            value=tree_sitter_node.text.decode('utf-8') if hasattr(tree_sitter_node, 'text') and tree_sitter_node.text else ""
        )
        
        # Mapear tipo de nodo
        if hasattr(tree_sitter_node, 'type'):
            type_mapping = {
                'function_definition': UnifiedNodeType.FUNCTION_DECLARATION,
                'function_def': UnifiedNodeType.FUNCTION_DECLARATION,
                'class_definition': UnifiedNodeType.CLASS_DECLARATION,
                'class_def': UnifiedNodeType.CLASS_DECLARATION,
                'if_statement': UnifiedNodeType.IF_STATEMENT,
                'for_statement': UnifiedNodeType.FOR_STATEMENT,
                'while_statement': UnifiedNodeType.WHILE_STATEMENT,
                'binary_expression': UnifiedNodeType.BINARY_EXPRESSION,
                'unary_expression': UnifiedNodeType.UNARY_EXPRESSION,
                'assignment': UnifiedNodeType.ASSIGNMENT_EXPRESSION,
                'call_expression': UnifiedNodeType.CALL_EXPRESSION,
                'member_expression': UnifiedNodeType.MEMBER_EXPRESSION,
                'identifier': UnifiedNodeType.IDENTIFIER,
                'string': UnifiedNodeType.STRING_LITERAL,
                'string_literal': UnifiedNodeType.STRING_LITERAL,
                'number': UnifiedNodeType.NUMBER_LITERAL,
                'integer': UnifiedNodeType.NUMBER_LITERAL,
                'float': UnifiedNodeType.NUMBER_LITERAL,
                'boolean': UnifiedNodeType.BOOLEAN_LITERAL,
                'true': UnifiedNodeType.BOOLEAN_LITERAL,
                'false': UnifiedNodeType.BOOLEAN_LITERAL,
                'return_statement': UnifiedNodeType.RETURN_STATEMENT,
            }
            
            unified_node.node_type = type_mapping.get(tree_sitter_node.type, UnifiedNodeType.LANGUAGE_SPECIFIC)
        
        # Convertir hijos recursivamente
        if hasattr(tree_sitter_node, 'children'):
            for child in tree_sitter_node.children:
                try:
                    unified_child = self._convert_to_unified_node(child)
                    unified_node.children.append(unified_child)
                except Exception as e:
                    logger.debug(f"Error convirtiendo nodo hijo: {e}")
        
        return unified_node
    
    def _collect_language_specific_info(self, node: UnifiedNode, language: ProgrammingLanguage,
                                      operators: OperatorCollection, operands: OperandCollection) -> Dict[str, Any]:
        """Recopila información específica del lenguaje."""
        info = {
            "language": language.value,
            "most_common_operators": operators.get_most_common(5),
            "most_common_operands": operands.get_most_common(5),
            "operator_diversity": operators.distinct_count() / max(1, operators.total_count()),
            "operand_diversity": operands.distinct_count() / max(1, operands.total_count())
        }
        
        # Información específica por lenguaje
        if language == ProgrammingLanguage.PYTHON:
            info.update(self._analyze_python_specific(node))
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            info.update(self._analyze_javascript_specific(node))
        elif language == ProgrammingLanguage.RUST:
            info.update(self._analyze_rust_specific(node))
        
        return info
    
    def _analyze_python_specific(self, node: UnifiedNode) -> Dict[str, Any]:
        """Análisis específico de Python."""
        return {
            "list_comprehensions": self._count_comprehensions(node),
            "decorators": self._count_decorators(node),
            "context_managers": self._count_context_managers(node),
            "lambda_functions": self._count_lambdas(node)
        }
    
    def _analyze_javascript_specific(self, node: UnifiedNode) -> Dict[str, Any]:
        """Análisis específico de JavaScript."""
        return {
            "arrow_functions": self._count_arrow_functions(node),
            "promises": self._count_promises(node),
            "async_functions": self._count_async_functions(node),
            "closures": self._count_closures(node)
        }
    
    def _analyze_rust_specific(self, node: UnifiedNode) -> Dict[str, Any]:
        """Análisis específico de Rust."""
        return {
            "pattern_matches": self._count_pattern_matches(node),
            "lifetimes": self._count_lifetimes(node),
            "unsafe_blocks": self._count_unsafe_blocks(node),
            "trait_implementations": self._count_trait_impls(node)
        }
    
    # Métodos auxiliares para análisis específico por lenguaje
    
    def _count_comprehensions(self, node: UnifiedNode) -> int:
        """Cuenta list/dict/set comprehensions en Python."""
        count = 0
        if hasattr(node, 'value') and node.value and 'comprehension' in node.value.lower():
            count += 1
        
        for child in node.children:
            count += self._count_comprehensions(child)
        
        return count
    
    def _count_decorators(self, node: UnifiedNode) -> int:
        """Cuenta decoradores en Python."""
        count = 0
        if hasattr(node, 'value') and node.value and node.value.strip().startswith('@'):
            count += 1
        
        for child in node.children:
            count += self._count_decorators(child)
        
        return count
    
    def _count_context_managers(self, node: UnifiedNode) -> int:
        """Cuenta context managers (with statements)."""
        count = 0
        if node.node_type == UnifiedNodeType.WITH_STATEMENT:
            count += 1
        
        for child in node.children:
            count += self._count_context_managers(child)
        
        return count
    
    def _count_lambdas(self, node: UnifiedNode) -> int:
        """Cuenta funciones lambda."""
        count = 0
        if hasattr(node, 'value') and node.value and 'lambda' in node.value:
            count += 1
        
        for child in node.children:
            count += self._count_lambdas(child)
        
        return count
    
    def _count_arrow_functions(self, node: UnifiedNode) -> int:
        """Cuenta arrow functions en JavaScript."""
        count = 0
        if hasattr(node, 'value') and node.value and '=>' in node.value:
            count += 1
        
        for child in node.children:
            count += self._count_arrow_functions(child)
        
        return count
    
    def _count_promises(self, node: UnifiedNode) -> int:
        """Cuenta promises en JavaScript."""
        count = 0
        if hasattr(node, 'value') and node.value and any(keyword in node.value.lower() for keyword in ['promise', '.then', '.catch']):
            count += 1
        
        for child in node.children:
            count += self._count_promises(child)
        
        return count
    
    def _count_async_functions(self, node: UnifiedNode) -> int:
        """Cuenta funciones async."""
        count = 0
        if hasattr(node, 'value') and node.value and 'async' in node.value:
            count += 1
        
        for child in node.children:
            count += self._count_async_functions(child)
        
        return count
    
    def _count_closures(self, node: UnifiedNode) -> int:
        """Cuenta closures en JavaScript."""
        # Simplificación: contar funciones anidadas
        count = 0
        function_depth = self._calculate_function_nesting_depth(node, 0)
        return max(0, function_depth - 1)  # Closures = nested functions
    
    def _count_pattern_matches(self, node: UnifiedNode) -> int:
        """Cuenta pattern matches en Rust."""
        count = 0
        if node.node_type == UnifiedNodeType.MATCH_STATEMENT:
            count += 1
        
        for child in node.children:
            count += self._count_pattern_matches(child)
        
        return count
    
    def _count_lifetimes(self, node: UnifiedNode) -> int:
        """Cuenta lifetimes en Rust."""
        count = 0
        if hasattr(node, 'value') and node.value and "'" in node.value:
            # Buscar patrones de lifetime
            import re
            lifetime_matches = re.findall(r"'[a-zA-Z_][a-zA-Z0-9_]*", node.value)
            count += len(lifetime_matches)
        
        for child in node.children:
            count += self._count_lifetimes(child)
        
        return count
    
    def _count_unsafe_blocks(self, node: UnifiedNode) -> int:
        """Cuenta bloques unsafe en Rust."""
        count = 0
        if hasattr(node, 'value') and node.value and 'unsafe' in node.value:
            count += 1
        
        for child in node.children:
            count += self._count_unsafe_blocks(child)
        
        return count
    
    def _count_trait_impls(self, node: UnifiedNode) -> int:
        """Cuenta implementaciones de trait en Rust."""
        count = 0
        if node.node_type == UnifiedNodeType.IMPL_DECLARATION:
            count += 1
        
        for child in node.children:
            count += self._count_trait_impls(child)
        
        return count
    
    def _calculate_function_nesting_depth(self, node: UnifiedNode, current_depth: int) -> int:
        """Calcula profundidad de anidamiento de funciones."""
        max_depth = current_depth
        
        if node.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        
        for child in node.children:
            child_depth = self._calculate_function_nesting_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)
        
        return max_depth


class VocabularyAnalyzer:
    """Analizador de vocabulario del código."""
    
    def analyze_vocabulary_complexity(self, operators: OperatorCollection, operands: OperandCollection) -> Dict[str, Any]:
        """
        Analiza la complejidad del vocabulario usado.
        
        Returns:
            Diccionario con análisis de vocabulario
        """
        analysis = {
            "vocabulary_size": operators.distinct_count() + operands.distinct_count(),
            "operator_operand_ratio": operators.distinct_count() / max(1, operands.distinct_count()),
            "vocabulary_diversity": self._calculate_vocabulary_diversity(operators, operands),
            "complexity_indicators": self._identify_complexity_indicators(operators, operands)
        }
        
        return analysis
    
    def _calculate_vocabulary_diversity(self, operators: OperatorCollection, operands: OperandCollection) -> float:
        """Calcula diversidad del vocabulario."""
        total_symbols = operators.total_count() + operands.total_count()
        unique_symbols = operators.distinct_count() + operands.distinct_count()
        
        if total_symbols == 0:
            return 0.0
        
        return unique_symbols / total_symbols
    
    def _identify_complexity_indicators(self, operators: OperatorCollection, operands: OperandCollection) -> List[str]:
        """Identifica indicadores de complejidad en el vocabulario."""
        indicators = []
        
        # Analizar operadores complejos
        complex_operators = ['**', '//', '<<', '>>', '&=', '|=', '^=']
        for op, count in operators.get_most_common():
            if op in complex_operators and count > 3:
                indicators.append(f"High usage of complex operator '{op}': {count} times")
        
        # Analizar diversidad de operandos
        if operands.distinct_count() / max(1, operands.total_count()) < 0.1:
            indicators.append("Low operand diversity - possible repetitive code")
        
        # Analizar vocabulario muy grande
        vocab_size = operators.distinct_count() + operands.distinct_count()
        if vocab_size > 50:
            indicators.append(f"Large vocabulary size ({vocab_size}) - possible complexity")
        
        return indicators


class ComplexityBenchmarker:
    """Benchmarker para comparar complejidad con estándares."""
    
    def __init__(self):
        # Benchmarks de industria para complejidad
        self.industry_benchmarks = {
            "cyclomatic_complexity": {
                "excellent": (1, 5),
                "good": (6, 10),
                "fair": (11, 20),
                "poor": (21, 50),
                "very_poor": (51, float('inf'))
            },
            "cognitive_complexity": {
                "excellent": (0, 5),
                "good": (6, 15),
                "fair": (16, 25),
                "poor": (26, 50),
                "very_poor": (51, float('inf'))
            },
            "halstead_volume": {
                "excellent": (0, 100),
                "good": (101, 500),
                "fair": (501, 1000),
                "poor": (1001, 2000),
                "very_poor": (2001, float('inf'))
            }
        }
    
    def benchmark_complexity(self, metrics: HalsteadMetrics, complexity: ComplexityMetrics) -> Dict[str, str]:
        """
        Compara métricas con benchmarks de industria.
        
        Returns:
            Diccionario con ratings por métrica
        """
        ratings = {}
        
        # Rating de complejidad ciclomática
        ratings["cyclomatic_complexity"] = self._get_rating(
            complexity.cyclomatic_complexity, 
            self.industry_benchmarks["cyclomatic_complexity"]
        )
        
        # Rating de complejidad cognitiva
        ratings["cognitive_complexity"] = self._get_rating(
            complexity.cognitive_complexity,
            self.industry_benchmarks["cognitive_complexity"]
        )
        
        # Rating de volumen de Halstead
        ratings["halstead_volume"] = self._get_rating(
            metrics.volume,
            self.industry_benchmarks["halstead_volume"]
        )
        
        return ratings
    
    def _get_rating(self, value: float, benchmark_ranges: Dict[str, Tuple[float, float]]) -> str:
        """Obtiene rating basado en rangos de benchmark."""
        for rating, (min_val, max_val) in benchmark_ranges.items():
            if min_val <= value <= max_val:
                return rating
        
        return "unknown"
    
    def generate_complexity_recommendations(self, ratings: Dict[str, str]) -> List[str]:
        """Genera recomendaciones basadas en ratings."""
        recommendations = []
        
        if ratings.get("cyclomatic_complexity") in ["poor", "very_poor"]:
            recommendations.extend([
                "Reduce cyclomatic complexity by extracting methods",
                "Simplify conditional logic",
                "Use polymorphism instead of complex conditionals"
            ])
        
        if ratings.get("cognitive_complexity") in ["poor", "very_poor"]:
            recommendations.extend([
                "Reduce cognitive load by decreasing nesting levels",
                "Use early returns to flatten code structure",
                "Extract complex logic into well-named methods"
            ])
        
        if ratings.get("halstead_volume") in ["poor", "very_poor"]:
            recommendations.extend([
                "Reduce code volume by eliminating duplication",
                "Simplify algorithms and data structures",
                "Use more concise language constructs where appropriate"
            ])
        
        return recommendations
