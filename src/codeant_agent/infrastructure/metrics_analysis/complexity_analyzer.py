"""
Implementación del analizador de complejidad.

Este módulo implementa el análisis de complejidad ciclomática, cognitiva,
y otras métricas de complejidad usando análisis AST.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from pathlib import Path
from collections import defaultdict, Counter

from ...domain.entities.code_metrics import (
    ComplexityMetrics, ComplexityThresholds, ComplexityLevel,
    ComplexityDistribution, FunctionMetrics, ComplexityHotspot,
    CodeSmell, CodeSmellType, CodeSmellSeverity
)
from ...domain.entities.parse_result import ParseResult
from ...domain.entities.ast_normalization import NormalizedNode as UnifiedNode, NodeType as UnifiedNodeType, SourcePosition
from ...domain.value_objects.programming_language import ProgrammingLanguage
from ...domain.entities.dead_code_analysis import SourceRange

logger = logging.getLogger(__name__)


@dataclass
class FunctionNode:
    """Representación de función para análisis de métricas."""
    name: str
    location: SourceRange
    body: UnifiedNode
    parameters: List[str]
    return_type: Optional[str] = None
    visibility: str = "public"
    is_static: bool = False
    is_async: bool = False
    
    def get_signature(self) -> str:
        """Obtiene firma de la función."""
        params = ", ".join(self.parameters)
        return f"{self.name}({params})"


@dataclass
class ClassNode:
    """Representación de clase para análisis de métricas."""
    name: str
    location: SourceRange
    methods: List['MethodNode']
    attributes: List['AttributeNode']
    parent_classes: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    visibility: str = "public"
    is_abstract: bool = False
    
    def get_public_methods(self) -> List['MethodNode']:
        """Obtiene métodos públicos."""
        return [method for method in self.methods if method.visibility == "public"]
    
    def get_private_methods(self) -> List['MethodNode']:
        """Obtiene métodos privados."""
        return [method for method in self.methods if method.visibility == "private"]


@dataclass
class MethodNode:
    """Representación de método para análisis."""
    name: str
    location: SourceRange
    body: UnifiedNode
    parameters: List[str]
    visibility: str = "public"
    is_static: bool = False
    is_abstract: bool = False
    accessed_attributes: Set[str] = field(default_factory=set)
    called_methods: Set[str] = field(default_factory=set)


@dataclass
class AttributeNode:
    """Representación de atributo para análisis."""
    name: str
    location: SourceRange
    attribute_type: Optional[str] = None
    visibility: str = "public"
    is_static: bool = False
    default_value: Optional[str] = None


@dataclass
class ComplexityAnalysisResult:
    """Resultado del análisis de complejidad."""
    complexity_metrics: ComplexityMetrics
    function_complexities: List[Tuple[str, int]]  # (function_name, complexity)
    class_complexities: List[Tuple[str, int]]     # (class_name, wmc)
    complexity_distribution: ComplexityDistribution
    hotspots: List[ComplexityHotspot]
    analysis_time_ms: int


class CyclomaticComplexityCalculator:
    """Calculadora de complejidad ciclomática (McCabe)."""
    
    def __init__(self):
        self.decision_node_types = {
            UnifiedNodeType.IF_STATEMENT,
            UnifiedNodeType.FOR_STATEMENT,
            UnifiedNodeType.WHILE_STATEMENT,
            UnifiedNodeType.DO_WHILE_STATEMENT,
            UnifiedNodeType.SWITCH_STATEMENT,
            UnifiedNodeType.MATCH_STATEMENT,
            UnifiedNodeType.TRY_STATEMENT,
            UnifiedNodeType.CONDITIONAL_EXPRESSION
        }
    
    async def calculate(self, node: UnifiedNode, language: ProgrammingLanguage) -> int:
        """
        Calcula complejidad ciclomática para un nodo.
        
        Args:
            node: Nodo AST a analizar
            language: Lenguaje del código
            
        Returns:
            Complejidad ciclomática
        """
        complexity = 1  # Complejidad base
        complexity += self._count_decision_points(node)
        complexity += self._count_logical_operators(node)
        complexity += self._count_exception_handlers(node)
        
        return complexity
    
    async def calculate_function_complexity(self, function: FunctionNode, language: ProgrammingLanguage) -> int:
        """Calcula complejidad ciclomática de una función específica."""
        return await self.calculate(function.body, language)
    
    def _count_decision_points(self, node: UnifiedNode) -> int:
        """Cuenta puntos de decisión en el AST."""
        count = 0
        
        if node.node_type in self.decision_node_types:
            if node.node_type == UnifiedNodeType.SWITCH_STATEMENT:
                # Cada case en switch añade complejidad
                case_count = self._count_switch_cases(node)
                count += max(1, case_count)
            elif node.node_type == UnifiedNodeType.TRY_STATEMENT:
                # Cada catch block añade complejidad
                catch_count = self._count_catch_blocks(node)
                count += max(1, catch_count)
            else:
                count += 1
        
        # Procesar hijos recursivamente
        for child in node.children:
            count += self._count_decision_points(child)
        
        return count
    
    def _count_logical_operators(self, node: UnifiedNode) -> int:
        """Cuenta operadores lógicos (&&, ||)."""
        count = 0
        
        if node.node_type == UnifiedNodeType.BINARY_EXPRESSION:
            # Buscar operadores lógicos en el valor o atributos del nodo
            if hasattr(node, 'value') and node.value:
                if any(op in node.value for op in ['&&', '||', 'and', 'or']):
                    count += 1
        
        # Procesar hijos
        for child in node.children:
            count += self._count_logical_operators(child)
        
        return count
    
    def _count_exception_handlers(self, node: UnifiedNode) -> int:
        """Cuenta manejadores de excepciones."""
        return self._count_catch_blocks(node)
    
    def _count_switch_cases(self, node: UnifiedNode) -> int:
        """Cuenta casos en statement switch."""
        case_count = 0
        for child in node.children:
            if (hasattr(child, 'node_type') and 
                str(child.node_type).lower() in ['case_clause', 'match_case', 'case']):
                case_count += 1
        return case_count
    
    def _count_catch_blocks(self, node: UnifiedNode) -> int:
        """Cuenta bloques catch/except."""
        catch_count = 0
        for child in node.children:
            if (hasattr(child, 'node_type') and 
                str(child.node_type).lower() in ['catch_block', 'except_clause', 'catch']):
                catch_count += 1
        return catch_count


class CognitiveComplexityCalculator:
    """Calculadora de complejidad cognitiva."""
    
    def __init__(self):
        self.nesting_structures = {
            UnifiedNodeType.IF_STATEMENT,
            UnifiedNodeType.FOR_STATEMENT,
            UnifiedNodeType.WHILE_STATEMENT,
            UnifiedNodeType.DO_WHILE_STATEMENT,
            UnifiedNodeType.SWITCH_STATEMENT,
            UnifiedNodeType.TRY_STATEMENT,
            UnifiedNodeType.FUNCTION_DECLARATION,  # Funciones anidadas
        }
    
    async def calculate(self, node: UnifiedNode, language: ProgrammingLanguage) -> int:
        """
        Calcula complejidad cognitiva según la definición de SonarSource.
        
        Args:
            node: Nodo AST a analizar
            language: Lenguaje del código
            
        Returns:
            Complejidad cognitiva
        """
        complexity = 0
        self._calculate_recursive(node, 0, complexity)
        return complexity
    
    def _calculate_recursive(self, node: UnifiedNode, nesting_level: int, complexity: int) -> int:
        """Calcula complejidad cognitiva recursivamente."""
        increment, new_nesting = self._get_complexity_increment(node, nesting_level)
        complexity += increment
        
        # Procesar hijos con nuevo nivel de anidamiento
        for child in node.children:
            complexity = self._calculate_recursive(child, new_nesting, complexity)
        
        return complexity
    
    def _get_complexity_increment(self, node: UnifiedNode, nesting_level: int) -> Tuple[int, int]:
        """
        Obtiene incremento de complejidad y nuevo nivel de anidamiento.
        
        Returns:
            Tupla (incremento_complejidad, nuevo_nivel_anidamiento)
        """
        if node.node_type == UnifiedNodeType.IF_STATEMENT:
            return (1 + nesting_level, nesting_level + 1)
        elif node.node_type in [UnifiedNodeType.FOR_STATEMENT, UnifiedNodeType.WHILE_STATEMENT]:
            return (1 + nesting_level, nesting_level + 1)
        elif node.node_type == UnifiedNodeType.SWITCH_STATEMENT:
            return (1 + nesting_level, nesting_level + 1)
        elif node.node_type == UnifiedNodeType.TRY_STATEMENT:
            # Try en sí no añade complejidad, pero aumenta anidamiento
            return (0, nesting_level + 1)
        elif self._is_catch_block(node):
            return (1 + nesting_level, nesting_level)
        elif self._is_logical_operator(node):
            # Solo el primero en una secuencia cuenta
            if self._is_first_in_logical_sequence(node):
                return (1, nesting_level)
            else:
                return (0, nesting_level)
        elif self._is_recursive_call(node):
            return (1 + nesting_level, nesting_level)
        else:
            return (0, nesting_level)
    
    def _is_catch_block(self, node: UnifiedNode) -> bool:
        """Verifica si es un bloque catch/except."""
        return str(node.node_type).lower() in ['catch_block', 'except_clause']
    
    def _is_logical_operator(self, node: UnifiedNode) -> bool:
        """Verifica si es operador lógico."""
        if node.node_type == UnifiedNodeType.BINARY_EXPRESSION:
            if hasattr(node, 'value') and node.value:
                return any(op in node.value for op in ['&&', '||', 'and', 'or'])
        return False
    
    def _is_first_in_logical_sequence(self, node: UnifiedNode) -> bool:
        """Verifica si es el primer operador en secuencia lógica."""
        # Simplificación: considerar todos como primeros
        # En implementación completa, requiere análisis del AST padre
        return True
    
    def _is_recursive_call(self, node: UnifiedNode) -> bool:
        """Verifica si es llamada recursiva."""
        # Requiere contexto de función actual
        # Simplificación para esta implementación
        return False


class NestingDepthAnalyzer:
    """Analizador de profundidad de anidamiento."""
    
    def calculate_max_nesting_depth(self, node: UnifiedNode) -> int:
        """Calcula profundidad máxima de anidamiento."""
        return self._calculate_depth_recursive(node, 0, 0)
    
    def calculate_average_nesting_depth(self, node: UnifiedNode) -> float:
        """Calcula profundidad promedio de anidamiento."""
        depths = []
        self._collect_depths(node, 0, depths)
        
        if not depths:
            return 0.0
        
        return sum(depths) / len(depths)
    
    def _calculate_depth_recursive(self, node: UnifiedNode, current_depth: int, max_depth: int) -> int:
        """Calcula profundidad recursivamente."""
        if self._increases_nesting(node):
            current_depth += 1
        
        max_depth = max(max_depth, current_depth)
        
        for child in node.children:
            max_depth = max(max_depth, self._calculate_depth_recursive(child, current_depth, max_depth))
        
        return max_depth
    
    def _collect_depths(self, node: UnifiedNode, current_depth: int, depths: List[int]) -> None:
        """Recopila todas las profundidades encontradas."""
        if self._increases_nesting(node):
            current_depth += 1
        
        # Solo contar profundidades de nodos que realmente ejecutan lógica
        if self._is_executable_node(node):
            depths.append(current_depth)
        
        for child in node.children:
            self._collect_depths(child, current_depth, depths)
    
    def _increases_nesting(self, node: UnifiedNode) -> bool:
        """Verifica si el nodo aumenta el nivel de anidamiento."""
        nesting_types = {
            UnifiedNodeType.IF_STATEMENT,
            UnifiedNodeType.FOR_STATEMENT,
            UnifiedNodeType.WHILE_STATEMENT,
            UnifiedNodeType.DO_WHILE_STATEMENT,
            UnifiedNodeType.SWITCH_STATEMENT,
            UnifiedNodeType.TRY_STATEMENT,
            UnifiedNodeType.FUNCTION_DECLARATION,
            UnifiedNodeType.CLASS_DECLARATION
        }
        return node.node_type in nesting_types
    
    def _is_executable_node(self, node: UnifiedNode) -> bool:
        """Verifica si es un nodo ejecutable."""
        executable_types = {
            UnifiedNodeType.EXPRESSION_STATEMENT,
            UnifiedNodeType.ASSIGNMENT_EXPRESSION,
            UnifiedNodeType.CALL_EXPRESSION,
            UnifiedNodeType.RETURN_STATEMENT,
            UnifiedNodeType.THROW_STATEMENT
        }
        return node.node_type in executable_types


class DecisionPointAnalyzer:
    """Analizador de puntos de decisión."""
    
    def analyze_decision_points(self, node: UnifiedNode) -> Dict[str, int]:
        """
        Analiza puntos de decisión en detalle.
        
        Returns:
            Diccionario con conteos por tipo de decisión
        """
        decision_counts = {
            "if_statements": 0,
            "loops": 0,
            "switch_statements": 0,
            "logical_operators": 0,
            "exception_handlers": 0,
            "conditional_expressions": 0,
            "recursive_calls": 0
        }
        
        self._analyze_recursive(node, decision_counts)
        return decision_counts
    
    def _analyze_recursive(self, node: UnifiedNode, counts: Dict[str, int]) -> None:
        """Analiza recursivamente contando tipos de decisiones."""
        if node.node_type == UnifiedNodeType.IF_STATEMENT:
            counts["if_statements"] += 1
        elif node.node_type in [UnifiedNodeType.FOR_STATEMENT, UnifiedNodeType.WHILE_STATEMENT, UnifiedNodeType.DO_WHILE_STATEMENT]:
            counts["loops"] += 1
        elif node.node_type in [UnifiedNodeType.SWITCH_STATEMENT, UnifiedNodeType.MATCH_STATEMENT]:
            counts["switch_statements"] += 1
        elif node.node_type == UnifiedNodeType.CONDITIONAL_EXPRESSION:
            counts["conditional_expressions"] += 1
        elif node.node_type == UnifiedNodeType.BINARY_EXPRESSION:
            if self._is_logical_operator(node):
                counts["logical_operators"] += 1
        elif self._is_exception_handler(node):
            counts["exception_handlers"] += 1
        elif self._is_recursive_call(node):
            counts["recursive_calls"] += 1
        
        # Procesar hijos
        for child in node.children:
            self._analyze_recursive(child, counts)
    
    def _is_logical_operator(self, node: UnifiedNode) -> bool:
        """Verifica si es operador lógico."""
        if hasattr(node, 'value') and node.value:
            return any(op in node.value for op in ['&&', '||', 'and', 'or'])
        return False
    
    def _is_exception_handler(self, node: UnifiedNode) -> bool:
        """Verifica si es manejador de excepción."""
        return str(node.node_type).lower() in ['catch_block', 'except_clause', 'finally_block']
    
    def _is_recursive_call(self, node: UnifiedNode) -> bool:
        """Verifica si es llamada recursiva (simplificado)."""
        # En implementación completa requiere análisis de contexto
        return False


class ComplexityAnalyzer:
    """Analizador principal de complejidad."""
    
    def __init__(self):
        self.cyclomatic_calculator = CyclomaticComplexityCalculator()
        self.cognitive_calculator = CognitiveComplexityCalculator()
        self.nesting_analyzer = NestingDepthAnalyzer()
        self.decision_analyzer = DecisionPointAnalyzer()
    
    async def calculate_complexity(
        self, 
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> ComplexityAnalysisResult:
        """
        Calcula métricas de complejidad completas.
        
        Args:
            parse_result: Resultado del parsing
            config: Configuración opcional
            
        Returns:
            ComplexityAnalysisResult completo
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Calculando complejidad para {parse_result.file_path}")
            
            # Convertir a unified node
            unified_root = self._convert_to_unified_node(parse_result.tree.root_node)
            
            # Calcular métricas base
            cyclomatic = await self.cyclomatic_calculator.calculate(unified_root, parse_result.language)
            cognitive = await self.cognitive_calculator.calculate(unified_root, parse_result.language)
            
            # Calcular métricas de anidamiento
            max_nesting = self.nesting_analyzer.calculate_max_nesting_depth(unified_root)
            avg_nesting = self.nesting_analyzer.calculate_average_nesting_depth(unified_root)
            
            # Analizar puntos de decisión
            decision_points = self.decision_analyzer.analyze_decision_points(unified_root)
            
            # Extraer funciones y calcular métricas por función
            functions = self._extract_functions(unified_root, parse_result.file_path)
            function_complexities = []
            
            for function in functions:
                func_complexity = await self.cyclomatic_calculator.calculate_function_complexity(function, parse_result.language)
                function_complexities.append((function.name, func_complexity))
            
            # Extraer clases y calcular WMC (Weighted Methods per Class)
            classes = self._extract_classes(unified_root, parse_result.file_path)
            class_complexities = []
            
            for class_node in classes:
                wmc = await self._calculate_wmc(class_node, parse_result.language)
                class_complexities.append((class_node.name, wmc))
            
            # Calcular distribución de complejidad
            thresholds = ComplexityThresholds()
            distribution = self._calculate_complexity_distribution(function_complexities, thresholds)
            
            # Identificar hotspots de complejidad
            hotspots = self._identify_complexity_hotspots(functions, classes, thresholds)
            
            # Crear métricas principales
            complexity_metrics = ComplexityMetrics(
                cyclomatic_complexity=cyclomatic,
                cognitive_complexity=cognitive,
                max_nesting_depth=max_nesting,
                average_nesting_depth=avg_nesting,
                decision_points=sum(decision_points.values()),
                loop_count=decision_points.get("loops", 0),
                condition_count=decision_points.get("if_statements", 0) + decision_points.get("conditional_expressions", 0)
            )
            
            # Calcular densidad de complejidad
            if parse_result.metadata.line_count > 0:
                complexity_metrics.complexity_density = cyclomatic / parse_result.metadata.line_count
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Análisis de complejidad completado para {parse_result.file_path}: "
                f"CC={cyclomatic}, CogC={cognitive}, max_nesting={max_nesting} en {total_time}ms"
            )
            
            return ComplexityAnalysisResult(
                complexity_metrics=complexity_metrics,
                function_complexities=function_complexities,
                class_complexities=class_complexities,
                complexity_distribution=distribution,
                hotspots=hotspots,
                analysis_time_ms=total_time
            )
            
        except Exception as e:
            logger.error(f"Error calculando complejidad: {e}")
            raise
    
    def _convert_to_unified_node(self, tree_sitter_node) -> UnifiedNode:
        """Convierte tree-sitter node a UnifiedNode."""
        # Crear SourcePosition correcta
        position = SourcePosition(
            start_line=getattr(tree_sitter_node, 'start_point', (0, 0))[0],
            start_column=getattr(tree_sitter_node, 'start_point', (0, 0))[1],
            end_line=getattr(tree_sitter_node, 'end_point', (0, 0))[0],
            end_column=getattr(tree_sitter_node, 'end_point', (0, 0))[1],
            start_byte=getattr(tree_sitter_node, 'start_byte', 0),
            end_byte=getattr(tree_sitter_node, 'end_byte', 0)
        )
        
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
                'method_definition': UnifiedNodeType.FUNCTION_DECLARATION,
                'class_definition': UnifiedNodeType.CLASS_DECLARATION,
                'class_def': UnifiedNodeType.CLASS_DECLARATION,
                'if_statement': UnifiedNodeType.IF_STATEMENT,
                'for_statement': UnifiedNodeType.FOR_STATEMENT,
                'while_statement': UnifiedNodeType.WHILE_STATEMENT,
                'switch_statement': UnifiedNodeType.SWITCH_STATEMENT,
                'try_statement': UnifiedNodeType.TRY_STATEMENT,
                'return_statement': UnifiedNodeType.RETURN_STATEMENT,
                'assignment': UnifiedNodeType.ASSIGNMENT_EXPRESSION,
                'binary_expression': UnifiedNodeType.BINARY_EXPRESSION,
                'call_expression': UnifiedNodeType.CALL_EXPRESSION,
            }
            
            unified_node.node_type = type_mapping.get(tree_sitter_node.type, UnifiedNodeType.LANGUAGE_SPECIFIC)
        
        # Convertir hijos recursivamente
        if hasattr(tree_sitter_node, 'children'):
            for child in tree_sitter_node.children:
                unified_child = self._convert_to_unified_node(child)
                unified_node.children.append(unified_child)
        
        return unified_node
    
    def _extract_functions(self, node: UnifiedNode, file_path: Path) -> List[FunctionNode]:
        """Extrae funciones del AST."""
        functions = []
        self._extract_functions_recursive(node, functions, file_path)
        return functions
    
    def _extract_functions_recursive(self, node: UnifiedNode, functions: List[FunctionNode], file_path: Path) -> None:
        """Extrae funciones recursivamente."""
        if node.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
            function_name = self._extract_function_name(node)
            parameters = self._extract_function_parameters(node)
            
            location = SourceRange(
                start=SourcePosition(
                    line=node.position.start_line + 1,
                    column=node.position.start_column
                ),
                end=SourcePosition(
                    line=node.position.end_line + 1,
                    column=node.position.end_column
                )
            )
            
            function = FunctionNode(
                name=function_name,
                location=location,
                body=node,
                parameters=parameters
            )
            
            functions.append(function)
        
        # Continuar con hijos
        for child in node.children:
            self._extract_functions_recursive(child, functions, file_path)
    
    def _extract_classes(self, node: UnifiedNode, file_path: Path) -> List[ClassNode]:
        """Extrae clases del AST."""
        classes = []
        self._extract_classes_recursive(node, classes, file_path)
        return classes
    
    def _extract_classes_recursive(self, node: UnifiedNode, classes: List[ClassNode], file_path: Path) -> None:
        """Extrae clases recursivamente."""
        if node.node_type == UnifiedNodeType.CLASS_DECLARATION:
            class_name = self._extract_class_name(node)
            methods = self._extract_class_methods(node, file_path)
            attributes = self._extract_class_attributes(node, file_path)
            
            location = SourceRange(
                start=SourcePosition(
                    line=node.position.start_line + 1,
                    column=node.position.start_column
                ),
                end=SourcePosition(
                    line=node.position.end_line + 1,
                    column=node.position.end_column
                )
            )
            
            class_node = ClassNode(
                name=class_name,
                location=location,
                methods=methods,
                attributes=attributes
            )
            
            classes.append(class_node)
        
        # Continuar con hijos
        for child in node.children:
            self._extract_classes_recursive(child, classes, file_path)
    
    async def _calculate_wmc(self, class_node: ClassNode, language: ProgrammingLanguage) -> int:
        """Calcula Weighted Methods per Class."""
        total_complexity = 0
        
        for method in class_node.methods:
            method_complexity = await self.cyclomatic_calculator.calculate(method.body, language)
            total_complexity += method_complexity
        
        return total_complexity
    
    def _calculate_complexity_distribution(self, function_complexities: List[Tuple[str, int]], 
                                         thresholds: ComplexityThresholds) -> ComplexityDistribution:
        """Calcula distribución de complejidad."""
        distribution = ComplexityDistribution()
        
        for _, complexity in function_complexities:
            level = thresholds.get_complexity_level(complexity)
            
            if level == ComplexityLevel.LOW:
                distribution.low_complexity_functions += 1
            elif level == ComplexityLevel.MEDIUM:
                distribution.medium_complexity_functions += 1
            elif level == ComplexityLevel.HIGH:
                distribution.high_complexity_functions += 1
            else:
                distribution.very_high_complexity_functions += 1
        
        # Crear histograma
        complexity_values = [comp for _, comp in function_complexities]
        if complexity_values:
            max_complexity = max(complexity_values)
            histogram = [0] * (max_complexity + 1)
            
            for complexity in complexity_values:
                histogram[complexity] += 1
            
            distribution.complexity_histogram = histogram
        
        return distribution
    
    def _identify_complexity_hotspots(self, functions: List[FunctionNode], classes: List[ClassNode], 
                                    thresholds: ComplexityThresholds) -> List[ComplexityHotspot]:
        """Identifica hotspots de complejidad."""
        hotspots = []
        
        # Hotspots de funciones
        for function in functions:
            # Calcular complejidad simplificada para hotspot
            estimated_complexity = self._estimate_function_complexity(function)
            
            if estimated_complexity > thresholds.cyclomatic_medium:
                severity = (CodeSmellSeverity.HIGH if estimated_complexity > thresholds.cyclomatic_high 
                           else CodeSmellSeverity.MEDIUM)
                
                impact_score = min(100.0, estimated_complexity * 2.0)
                
                hotspot = ComplexityHotspot(
                    location=function.location,
                    hotspot_type="function",
                    name=function.name,
                    cyclomatic_complexity=estimated_complexity,
                    cognitive_complexity=0,  # Calculado después
                    lines_of_code=self._estimate_function_lines(function),
                    severity=severity,
                    impact_score=impact_score,
                    suggested_actions=self._generate_function_suggestions(function, estimated_complexity)
                )
                
                hotspots.append(hotspot)
        
        # Hotspots de clases
        for class_node in classes:
            estimated_wmc = len(class_node.methods) * 3  # Estimación
            
            if estimated_wmc > 30:  # Threshold para WMC alto
                hotspot = ComplexityHotspot(
                    location=class_node.location,
                    hotspot_type="class",
                    name=class_node.name,
                    cyclomatic_complexity=estimated_wmc,
                    cognitive_complexity=0,
                    lines_of_code=self._estimate_class_lines(class_node),
                    severity=CodeSmellSeverity.MEDIUM,
                    impact_score=min(100.0, estimated_wmc * 1.5),
                    suggested_actions=self._generate_class_suggestions(class_node)
                )
                
                hotspots.append(hotspot)
        
        # Ordenar por impact score
        hotspots.sort(key=lambda h: h.impact_score, reverse=True)
        
        return hotspots[:20]  # Top 20 hotspots
    
    def _estimate_function_complexity(self, function: FunctionNode) -> int:
        """Estima complejidad de función (análisis rápido)."""
        # Análisis simplificado contando estructuras de control
        complexity = 1
        complexity += self._quick_count_decision_structures(function.body)
        return complexity
    
    def _quick_count_decision_structures(self, node: UnifiedNode) -> int:
        """Cuenta rápidamente estructuras de decisión."""
        count = 0
        
        decision_types = {
            UnifiedNodeType.IF_STATEMENT,
            UnifiedNodeType.FOR_STATEMENT,
            UnifiedNodeType.WHILE_STATEMENT,
            UnifiedNodeType.SWITCH_STATEMENT,
            UnifiedNodeType.TRY_STATEMENT
        }
        
        if node.node_type in decision_types:
            count += 1
        
        for child in node.children:
            count += self._quick_count_decision_structures(child)
        
        return count
    
    def _estimate_function_lines(self, function: FunctionNode) -> int:
        """Estima líneas de código de función."""
        return (function.location.end.line - function.location.start.line + 1)
    
    def _estimate_class_lines(self, class_node: ClassNode) -> int:
        """Estima líneas de código de clase."""
        return (class_node.location.end.line - class_node.location.start.line + 1)
    
    def _generate_function_suggestions(self, function: FunctionNode, complexity: int) -> List[str]:
        """Genera sugerencias para función compleja."""
        suggestions = []
        
        if complexity > 20:
            suggestions.extend([
                "Consider breaking down this function into smaller functions",
                "Extract complex conditional logic into separate methods",
                "Use early returns to reduce nesting"
            ])
        elif complexity > 10:
            suggestions.extend([
                "Consider extracting some logic into helper methods",
                "Simplify conditional expressions"
            ])
        
        if len(function.parameters) > 7:
            suggestions.append("Reduce number of parameters using parameter objects")
        
        return suggestions
    
    def _generate_class_suggestions(self, class_node: ClassNode) -> List[str]:
        """Genera sugerencias para clase compleja."""
        suggestions = []
        
        if len(class_node.methods) > 20:
            suggestions.extend([
                "Consider splitting this class into smaller, more focused classes",
                "Apply Single Responsibility Principle",
                "Extract related methods into separate classes"
            ])
        
        if len(class_node.attributes) > 15:
            suggestions.append("Reduce number of instance variables")
        
        return suggestions
    
    # Métodos auxiliares de extracción
    
    def _extract_function_name(self, node: UnifiedNode) -> str:
        """Extrae nombre de función."""
        # Buscar en hijos un nodo identificador
        for child in node.children:
            if (child.node_type == UnifiedNodeType.IDENTIFIER and 
                child.value and child.value.strip()):
                return child.value.strip()
        
        # Fallback: extraer del valor del nodo
        if hasattr(node, 'value') and node.value:
            # Buscar patrón de función en el texto
            import re
            func_match = re.search(r'def\s+(\w+)|function\s+(\w+)|fn\s+(\w+)', node.value)
            if func_match:
                return next(group for group in func_match.groups() if group)
        
        return "anonymous_function"
    
    def _extract_function_parameters(self, node: UnifiedNode) -> List[str]:
        """Extrae parámetros de función."""
        parameters = []
        
        # Buscar lista de parámetros en hijos
        for child in node.children:
            if str(child.node_type).lower() in ['parameter_list', 'parameters', 'formal_parameters']:
                parameters.extend(self._extract_parameter_names(child))
        
        return parameters
    
    def _extract_parameter_names(self, param_list_node: UnifiedNode) -> List[str]:
        """Extrae nombres de parámetros."""
        params = []
        
        for child in param_list_node.children:
            if child.node_type == UnifiedNodeType.IDENTIFIER and child.value:
                params.append(child.value)
        
        return params
    
    def _extract_class_name(self, node: UnifiedNode) -> str:
        """Extrae nombre de clase."""
        for child in node.children:
            if (child.node_type == UnifiedNodeType.IDENTIFIER and 
                child.value and child.value.strip()):
                return child.value.strip()
        
        # Fallback
        if hasattr(node, 'value') and node.value:
            import re
            class_match = re.search(r'class\s+(\w+)|struct\s+(\w+)|interface\s+(\w+)', node.value)
            if class_match:
                return next(group for group in class_match.groups() if group)
        
        return "AnonymousClass"
    
    def _extract_class_methods(self, class_node: UnifiedNode, file_path: Path) -> List[MethodNode]:
        """Extrae métodos de clase."""
        methods = []
        
        # Buscar funciones dentro de la clase
        for child in class_node.children:
            if child.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
                method_name = self._extract_function_name(child)
                parameters = self._extract_function_parameters(child)
                
                location = SourceRange(
                    start=SourcePosition(line=child.position.start_line + 1, column=child.position.start_column),
                    end=SourcePosition(line=child.position.end_line + 1, column=child.position.end_column)
                )
                
                method = MethodNode(
                    name=method_name,
                    location=location,
                    body=child,
                    parameters=parameters,
                    visibility="public"  # Simplificación
                )
                
                methods.append(method)
        
        return methods
    
    def _extract_class_attributes(self, class_node: UnifiedNode, file_path: Path) -> List[AttributeNode]:
        """Extrae atributos de clase."""
        attributes = []
        
        # Buscar declaraciones de variables/atributos
        for child in class_node.children:
            if child.node_type == UnifiedNodeType.VARIABLE_DECLARATION:
                attr_name = self._extract_variable_name(child)
                
                location = SourceRange(
                    start=SourcePosition(line=child.position.start_line + 1, column=child.position.start_column),
                    end=SourcePosition(line=child.position.end_line + 1, column=child.position.end_column)
                )
                
                attribute = AttributeNode(
                    name=attr_name,
                    location=location,
                    visibility="public"  # Simplificación
                )
                
                attributes.append(attribute)
        
        return attributes
    
    def _extract_variable_name(self, node: UnifiedNode) -> str:
        """Extrae nombre de variable."""
        for child in node.children:
            if child.node_type == UnifiedNodeType.IDENTIFIER and child.value:
                return child.value
        
        return "unknown_variable"
