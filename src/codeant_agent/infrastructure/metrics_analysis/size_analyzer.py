"""
Implementación del analizador de métricas de tamaño.

Este módulo implementa el cálculo de métricas de tamaño incluyendo
LOC, SLOC, comentarios, líneas en blanco y métricas de función/clase.
"""

import logging
import asyncio
import re
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import time

from ...domain.entities.code_metrics import SizeMetrics
from ...domain.entities.parse_result import ParseResult
from ...domain.entities.ast_normalization import NormalizedNode as UnifiedNode, NodeType as UnifiedNodeType
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class LineClassification:
    """Clasificación de líneas de código."""
    total_lines: int = 0
    source_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    mixed_lines: int = 0  # Líneas con código y comentario
    
    def get_comment_ratio(self) -> float:
        """Obtiene ratio de comentarios."""
        return self.comment_lines / self.total_lines if self.total_lines > 0 else 0.0
    
    def get_blank_ratio(self) -> float:
        """Obtiene ratio de líneas en blanco."""
        return self.blank_lines / self.total_lines if self.total_lines > 0 else 0.0


@dataclass
class FunctionSizeMetrics:
    """Métricas de tamaño específicas de función."""
    name: str
    lines_of_code: int
    logical_lines: int
    parameters: int
    local_variables: int
    return_statements: int
    complexity_ratio: float  # LOC / Complexity


@dataclass
class ClassSizeMetrics:
    """Métricas de tamaño específicas de clase."""
    name: str
    lines_of_code: int
    method_count: int
    attribute_count: int
    public_methods: int
    private_methods: int
    average_method_size: float


@dataclass
class SizeAnalysisResult:
    """Resultado del análisis de tamaño."""
    size_metrics: SizeMetrics
    line_classification: LineClassification
    function_sizes: List[FunctionSizeMetrics]
    class_sizes: List[ClassSizeMetrics]
    analysis_time_ms: int


class LineAnalyzer:
    """Analizador de líneas de código."""
    
    def __init__(self):
        # Patrones de comentarios por lenguaje
        self.comment_patterns = {
            ProgrammingLanguage.PYTHON: [
                r'#.*$',                    # Comentarios de línea
                r'"""[\s\S]*?"""',          # Docstrings triple comillas
                r"'''[\s\S]*?'''",          # Docstrings triple comillas simples
            ],
            ProgrammingLanguage.JAVASCRIPT: [
                r'//.*$',                   # Comentarios de línea
                r'/\*[\s\S]*?\*/',          # Comentarios de bloque
            ],
            ProgrammingLanguage.TYPESCRIPT: [
                r'//.*$',                   # Comentarios de línea
                r'/\*[\s\S]*?\*/',          # Comentarios de bloque
            ],
            ProgrammingLanguage.RUST: [
                r'//.*$',                   # Comentarios de línea
                r'/\*[\s\S]*?\*/',          # Comentarios de bloque
                r'///.*$',                  # Documentation comments
                r'//!.*$',                  # Module documentation
            ],
        }
    
    def classify_lines(self, content: str, language: ProgrammingLanguage) -> LineClassification:
        """
        Clasifica líneas de código por tipo.
        
        Args:
            content: Contenido del archivo
            language: Lenguaje del código
            
        Returns:
            LineClassification con conteos por tipo
        """
        lines = content.split('\n')
        classification = LineClassification(total_lines=len(lines))
        
        # Obtener patrones de comentarios para el lenguaje
        patterns = self.comment_patterns.get(language, [])
        
        for line in lines:
            line_type = self._classify_single_line(line, patterns)
            
            if line_type == "blank":
                classification.blank_lines += 1
            elif line_type == "comment":
                classification.comment_lines += 1
            elif line_type == "source":
                classification.source_lines += 1
            elif line_type == "mixed":
                classification.mixed_lines += 1
                classification.source_lines += 1  # Contar también como source
                classification.comment_lines += 1  # Contar también como comment
        
        return classification
    
    def _classify_single_line(self, line: str, comment_patterns: List[str]) -> str:
        """Clasifica una línea individual."""
        stripped = line.strip()
        
        # Línea en blanco
        if not stripped:
            return "blank"
        
        # Verificar si es solo comentario
        is_comment = False
        for pattern in comment_patterns:
            if re.match(pattern.replace('$', ''), stripped):
                is_comment = True
                break
        
        if is_comment:
            return "comment"
        
        # Verificar si tiene tanto código como comentario
        has_code = False
        has_comment = False
        
        for pattern in comment_patterns:
            if re.search(pattern, line):
                has_comment = True
                # Verificar si hay código antes del comentario
                comment_start = re.search(pattern, line).start()
                if line[:comment_start].strip():
                    has_code = True
                break
        
        if has_code and has_comment:
            return "mixed"
        elif has_comment:
            return "comment"
        else:
            return "source"


class CodeStructureAnalyzer:
    """Analizador de estructura del código."""
    
    def analyze_code_structure(self, node: UnifiedNode) -> Dict[str, int]:
        """
        Analiza estructura del código contando elementos.
        
        Returns:
            Diccionario con conteos de elementos estructurales
        """
        structure_counts = {
            "functions": 0,
            "classes": 0,
            "methods": 0,
            "variables": 0,
            "constants": 0,
            "imports": 0,
            "interfaces": 0,
            "enums": 0,
            "structs": 0
        }
        
        self._count_structures_recursive(node, structure_counts, in_class=False)
        return structure_counts
    
    def _count_structures_recursive(self, node: UnifiedNode, counts: Dict[str, int], in_class: bool = False) -> None:
        """Cuenta estructuras recursivamente."""
        if node.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
            if in_class:
                counts["methods"] += 1
            else:
                counts["functions"] += 1
        
        elif node.node_type == UnifiedNodeType.CLASS_DECLARATION:
            counts["classes"] += 1
            # Procesar contenido de clase
            for child in node.children:
                self._count_structures_recursive(child, counts, in_class=True)
            return  # No procesar hijos nuevamente
        
        elif node.node_type == UnifiedNodeType.VARIABLE_DECLARATION:
            counts["variables"] += 1
        
        elif node.node_type == UnifiedNodeType.CONSTANT_DECLARATION:
            counts["constants"] += 1
        
        elif node.node_type == UnifiedNodeType.IMPORT_DECLARATION:
            counts["imports"] += 1
        
        elif node.node_type == UnifiedNodeType.INTERFACE_DECLARATION:
            counts["interfaces"] += 1
        
        elif node.node_type == UnifiedNodeType.ENUM_DECLARATION:
            counts["enums"] += 1
        
        elif node.node_type == UnifiedNodeType.STRUCT_DECLARATION:
            counts["structs"] += 1
        
        # Procesar hijos
        for child in node.children:
            self._count_structures_recursive(child, counts, in_class)


class SizeAnalyzer:
    """Analizador principal de métricas de tamaño."""
    
    def __init__(self):
        self.line_analyzer = LineAnalyzer()
        self.structure_analyzer = CodeStructureAnalyzer()
    
    async def calculate_size_metrics(
        self, 
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> SizeAnalysisResult:
        """
        Calcula métricas de tamaño completas.
        
        Args:
            parse_result: Resultado del parsing
            config: Configuración opcional
            
        Returns:
            SizeAnalysisResult completo
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Calculando métricas de tamaño para {parse_result.file_path}")
            
            # Obtener contenido del archivo
            content = self._get_file_content(parse_result)
            
            # Clasificar líneas
            line_classification = self.line_analyzer.classify_lines(content, parse_result.language)
            
            # Convertir AST y analizar estructura
            unified_root = self._convert_to_unified_node(parse_result.tree.root_node)
            structure_counts = self.structure_analyzer.analyze_code_structure(unified_root)
            
            # Analizar funciones individuales
            function_sizes = self._analyze_function_sizes(unified_root)
            
            # Analizar clases individuales
            class_sizes = self._analyze_class_sizes(unified_root)
            
            # Crear métricas de tamaño
            size_metrics = SizeMetrics(
                total_lines=line_classification.total_lines,
                logical_lines_of_code=line_classification.source_lines,
                comment_lines=line_classification.comment_lines,
                blank_lines=line_classification.blank_lines,
                source_lines=line_classification.source_lines,
                function_count=structure_counts["functions"] + structure_counts["methods"],
                class_count=structure_counts["classes"],
                method_count=structure_counts["methods"]
            )
            
            # Calcular métricas derivadas
            size_metrics.calculate_derived_metrics()
            
            # Calcular métricas adicionales
            if function_sizes:
                size_metrics.max_function_length = max(f.lines_of_code for f in function_sizes)
            
            if class_sizes:
                size_metrics.max_class_length = max(c.lines_of_code for c in class_sizes)
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Métricas de tamaño calculadas para {parse_result.file_path}: "
                f"LOC={size_metrics.total_lines}, SLOC={size_metrics.logical_lines_of_code}, "
                f"Functions={size_metrics.function_count}, Classes={size_metrics.class_count} en {total_time}ms"
            )
            
            return SizeAnalysisResult(
                size_metrics=size_metrics,
                line_classification=line_classification,
                function_sizes=function_sizes,
                class_sizes=class_sizes,
                analysis_time_ms=total_time
            )
            
        except Exception as e:
            logger.error(f"Error calculando métricas de tamaño: {e}")
            raise
    
    def _get_file_content(self, parse_result: ParseResult) -> str:
        """Obtiene contenido del archivo."""
        if hasattr(parse_result.tree, 'root_node') and hasattr(parse_result.tree.root_node, 'text'):
            return parse_result.tree.root_node.text.decode('utf-8')
        else:
            # Fallback: leer archivo si existe
            try:
                if parse_result.file_path.exists():
                    return parse_result.file_path.read_text(encoding='utf-8')
            except Exception:
                pass
            
            return ""
    
    def _convert_to_unified_node(self, tree_sitter_node) -> UnifiedNode:
        """Convierte tree-sitter node a UnifiedNode."""
        from ...domain.entities.ast_normalization import SourcePosition
        
        # Crear SourcePosition segura (similar a otros analizadores)
        try:
            start_point = getattr(tree_sitter_node, 'start_point', (0, 0))
            end_point = getattr(tree_sitter_node, 'end_point', (0, 0))
            start_byte = getattr(tree_sitter_node, 'start_byte', 0)
            end_byte = getattr(tree_sitter_node, 'end_byte', 0)
            
            if isinstance(start_point, tuple) and len(start_point) >= 2:
                start_line, start_col = start_point[0], start_point[1]
            else:
                start_line, start_col = 0, 0
            
            if isinstance(end_point, tuple) and len(end_point) >= 2:
                end_line, end_col = end_point[0], end_point[1]
            else:
                end_line, end_col = 0, 0
            
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
        except Exception:
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
            }
            
            unified_node.node_type = type_mapping.get(tree_sitter_node.type, UnifiedNodeType.LANGUAGE_SPECIFIC)
        
        # Convertir hijos
        if hasattr(tree_sitter_node, 'children'):
            for child in tree_sitter_node.children:
                try:
                    unified_child = self._convert_to_unified_node(child)
                    unified_node.children.append(unified_child)
                except Exception:
                    continue  # Skip problematic children
        
        return unified_node
    
    def _analyze_function_sizes(self, node: UnifiedNode) -> List[FunctionSizeMetrics]:
        """Analiza tamaños de funciones."""
        function_sizes = []
        self._analyze_functions_recursive(node, function_sizes)
        return function_sizes
    
    def _analyze_functions_recursive(self, node: UnifiedNode, function_sizes: List[FunctionSizeMetrics]) -> None:
        """Analiza funciones recursivamente."""
        if node.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
            function_name = self._extract_function_name(node)
            
            # Calcular métricas de la función
            lines_of_code = node.position.end_line - node.position.start_line + 1
            logical_lines = self._count_logical_lines_in_function(node)
            parameters = len(self._extract_function_parameters(node))
            local_vars = self._count_local_variables(node)
            return_statements = self._count_return_statements(node)
            
            function_metric = FunctionSizeMetrics(
                name=function_name,
                lines_of_code=lines_of_code,
                logical_lines=logical_lines,
                parameters=parameters,
                local_variables=local_vars,
                return_statements=return_statements,
                complexity_ratio=logical_lines / max(1, self._estimate_complexity(node))
            )
            
            function_sizes.append(function_metric)
        
        # Procesar hijos
        for child in node.children:
            self._analyze_functions_recursive(child, function_sizes)
    
    def _analyze_class_sizes(self, node: UnifiedNode) -> List[ClassSizeMetrics]:
        """Analiza tamaños de clases."""
        class_sizes = []
        self._analyze_classes_recursive(node, class_sizes)
        return class_sizes
    
    def _analyze_classes_recursive(self, node: UnifiedNode, class_sizes: List[ClassSizeMetrics]) -> None:
        """Analiza clases recursivamente."""
        if node.node_type == UnifiedNodeType.CLASS_DECLARATION:
            class_name = self._extract_class_name(node)
            
            # Contar métodos en la clase
            methods = self._count_methods_in_class(node)
            attributes = self._count_attributes_in_class(node)
            public_methods, private_methods = self._count_method_visibility(node)
            
            lines_of_code = node.position.end_line - node.position.start_line + 1
            avg_method_size = self._calculate_average_method_size(node)
            
            class_metric = ClassSizeMetrics(
                name=class_name,
                lines_of_code=lines_of_code,
                method_count=methods,
                attribute_count=attributes,
                public_methods=public_methods,
                private_methods=private_methods,
                average_method_size=avg_method_size
            )
            
            class_sizes.append(class_metric)
        
        # Procesar hijos
        for child in node.children:
            self._analyze_classes_recursive(child, class_sizes)
    
    def _extract_function_name(self, node: UnifiedNode) -> str:
        """Extrae nombre de función."""
        # Buscar identificador en hijos
        for child in node.children:
            if child.node_type == UnifiedNodeType.IDENTIFIER and child.value:
                return child.value
        
        # Fallback: buscar en texto del nodo
        if hasattr(node, 'value') and node.value:
            import re
            func_match = re.search(r'def\s+(\w+)|function\s+(\w+)|fn\s+(\w+)', node.value)
            if func_match:
                return next(group for group in func_match.groups() if group)
        
        return "anonymous_function"
    
    def _extract_class_name(self, node: UnifiedNode) -> str:
        """Extrae nombre de clase."""
        for child in node.children:
            if child.node_type == UnifiedNodeType.IDENTIFIER and child.value:
                return child.value
        
        if hasattr(node, 'value') and node.value:
            import re
            class_match = re.search(r'class\s+(\w+)|struct\s+(\w+)', node.value)
            if class_match:
                return next(group for group in class_match.groups() if group)
        
        return "AnonymousClass"
    
    def _extract_function_parameters(self, node: UnifiedNode) -> List[str]:
        """Extrae parámetros de función."""
        parameters = []
        
        for child in node.children:
            if str(child.node_type).lower() in ['parameter_list', 'parameters']:
                for param_child in child.children:
                    if param_child.node_type == UnifiedNodeType.IDENTIFIER and param_child.value:
                        parameters.append(param_child.value)
        
        return parameters
    
    def _count_logical_lines_in_function(self, node: UnifiedNode) -> int:
        """Cuenta líneas lógicas en función."""
        # Contar statements ejecutables
        logical_lines = 0
        logical_lines += self._count_statements(node)
        return logical_lines
    
    def _count_statements(self, node: UnifiedNode) -> int:
        """Cuenta statements ejecutables."""
        statement_types = {
            UnifiedNodeType.EXPRESSION_STATEMENT,
            UnifiedNodeType.ASSIGNMENT_EXPRESSION,
            UnifiedNodeType.CALL_EXPRESSION,
            UnifiedNodeType.RETURN_STATEMENT,
            UnifiedNodeType.IF_STATEMENT,
            UnifiedNodeType.FOR_STATEMENT,
            UnifiedNodeType.WHILE_STATEMENT,
            UnifiedNodeType.THROW_STATEMENT,
            UnifiedNodeType.TRY_STATEMENT
        }
        
        count = 0
        if node.node_type in statement_types:
            count += 1
        
        for child in node.children:
            count += self._count_statements(child)
        
        return count
    
    def _count_local_variables(self, node: UnifiedNode) -> int:
        """Cuenta variables locales en función."""
        var_count = 0
        
        def count_vars_recursive(n: UnifiedNode):
            nonlocal var_count
            if n.node_type == UnifiedNodeType.VARIABLE_DECLARATION:
                var_count += 1
            
            for child in n.children:
                count_vars_recursive(child)
        
        count_vars_recursive(node)
        return var_count
    
    def _count_return_statements(self, node: UnifiedNode) -> int:
        """Cuenta statements return."""
        count = 0
        
        if node.node_type == UnifiedNodeType.RETURN_STATEMENT:
            count += 1
        
        for child in node.children:
            count += self._count_return_statements(child)
        
        return count
    
    def _estimate_complexity(self, node: UnifiedNode) -> int:
        """Estima complejidad básica para ratio."""
        complexity = 1
        
        decision_types = {
            UnifiedNodeType.IF_STATEMENT,
            UnifiedNodeType.FOR_STATEMENT,
            UnifiedNodeType.WHILE_STATEMENT,
            UnifiedNodeType.SWITCH_STATEMENT
        }
        
        if node.node_type in decision_types:
            complexity += 1
        
        for child in node.children:
            complexity += self._estimate_complexity(child)
        
        return complexity
    
    def _count_methods_in_class(self, node: UnifiedNode) -> int:
        """Cuenta métodos en clase."""
        method_count = 0
        
        for child in node.children:
            if child.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
                method_count += 1
        
        return method_count
    
    def _count_attributes_in_class(self, node: UnifiedNode) -> int:
        """Cuenta atributos en clase."""
        attr_count = 0
        
        for child in node.children:
            if child.node_type == UnifiedNodeType.VARIABLE_DECLARATION:
                attr_count += 1
        
        return attr_count
    
    def _count_method_visibility(self, node: UnifiedNode) -> Tuple[int, int]:
        """Cuenta métodos públicos y privados."""
        public_count = 0
        private_count = 0
        
        for child in node.children:
            if child.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
                # Simplificación: asumir público a menos que esté marcado como privado
                if self._is_private_method(child):
                    private_count += 1
                else:
                    public_count += 1
        
        return public_count, private_count
    
    def _is_private_method(self, node: UnifiedNode) -> bool:
        """Verifica si método es privado."""
        if hasattr(node, 'value') and node.value:
            # Convención Python: métodos que empiezan con _
            if re.search(r'def\s+_[^_]', node.value):
                return True
            
            # Convención JavaScript/TypeScript: palabra clave private
            if 'private' in node.value:
                return True
        
        return False
    
    def _calculate_average_method_size(self, node: UnifiedNode) -> float:
        """Calcula tamaño promedio de métodos en clase."""
        method_sizes = []
        
        for child in node.children:
            if child.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
                method_size = child.position.end_line - child.position.start_line + 1
                method_sizes.append(method_size)
        
        if not method_sizes:
            return 0.0
        
        return sum(method_sizes) / len(method_sizes)


class FileSizeClassifier:
    """Clasificador de archivos por tamaño."""
    
    SIZE_CATEGORIES = {
        "very_small": (0, 50),
        "small": (51, 200),
        "medium": (201, 500),
        "large": (501, 1000),
        "very_large": (1001, 2000),
        "huge": (2001, float('inf'))
    }
    
    def classify_file_size(self, lines_of_code: int) -> str:
        """Clasifica archivo por tamaño."""
        for category, (min_lines, max_lines) in self.SIZE_CATEGORIES.items():
            if min_lines <= lines_of_code <= max_lines:
                return category
        
        return "unknown"
    
    def get_size_recommendations(self, lines_of_code: int, language: ProgrammingLanguage) -> List[str]:
        """Genera recomendaciones basadas en tamaño."""
        category = self.classify_file_size(lines_of_code)
        recommendations = []
        
        if category in ["very_large", "huge"]:
            recommendations.extend([
                "Consider splitting this file into smaller modules",
                "Extract related functionality into separate files",
                "Apply Single Responsibility Principle at file level"
            ])
            
            if language == ProgrammingLanguage.PYTHON:
                recommendations.append("Consider creating a package with multiple modules")
            elif language == ProgrammingLanguage.RUST:
                recommendations.append("Split into multiple modules using mod.rs")
            elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
                recommendations.append("Split into multiple modules and use imports/exports")
        
        elif category == "large":
            recommendations.extend([
                "Monitor file growth",
                "Consider extracting utility functions",
                "Ensure good internal organization"
            ])
        
        return recommendations


class DocumentationAnalyzer:
    """Analizador de documentación."""
    
    def analyze_documentation_coverage(self, node: UnifiedNode, language: ProgrammingLanguage) -> Dict[str, Any]:
        """
        Analiza cobertura de documentación.
        
        Returns:
            Diccionario con métricas de documentación
        """
        analysis = {
            "total_functions": 0,
            "documented_functions": 0,
            "total_classes": 0,
            "documented_classes": 0,
            "documentation_ratio": 0.0,
            "documentation_quality": "unknown"
        }
        
        self._analyze_documentation_recursive(node, analysis, language)
        
        # Calcular ratios
        if analysis["total_functions"] > 0:
            func_doc_ratio = analysis["documented_functions"] / analysis["total_functions"]
        else:
            func_doc_ratio = 1.0
        
        if analysis["total_classes"] > 0:
            class_doc_ratio = analysis["documented_classes"] / analysis["total_classes"]
        else:
            class_doc_ratio = 1.0
        
        analysis["documentation_ratio"] = (func_doc_ratio + class_doc_ratio) / 2.0
        
        # Evaluar calidad de documentación
        if analysis["documentation_ratio"] >= 0.8:
            analysis["documentation_quality"] = "excellent"
        elif analysis["documentation_ratio"] >= 0.6:
            analysis["documentation_quality"] = "good"
        elif analysis["documentation_ratio"] >= 0.4:
            analysis["documentation_quality"] = "fair"
        else:
            analysis["documentation_quality"] = "poor"
        
        return analysis
    
    def _analyze_documentation_recursive(self, node: UnifiedNode, analysis: Dict[str, Any], 
                                       language: ProgrammingLanguage) -> None:
        """Analiza documentación recursivamente."""
        if node.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
            analysis["total_functions"] += 1
            if self._has_documentation(node, language):
                analysis["documented_functions"] += 1
        
        elif node.node_type == UnifiedNodeType.CLASS_DECLARATION:
            analysis["total_classes"] += 1
            if self._has_documentation(node, language):
                analysis["documented_classes"] += 1
        
        # Procesar hijos
        for child in node.children:
            self._analyze_documentation_recursive(child, analysis, language)
    
    def _has_documentation(self, node: UnifiedNode, language: ProgrammingLanguage) -> bool:
        """Verifica si el nodo tiene documentación."""
        if not hasattr(node, 'value') or not node.value:
            return False
        
        content = node.value.lower()
        
        # Patrones de documentación por lenguaje
        if language == ProgrammingLanguage.PYTHON:
            # Buscar docstrings
            return '"""' in content or "'''" in content
        
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            # Buscar JSDoc
            return '/**' in content or '@param' in content or '@returns' in content
        
        elif language == ProgrammingLanguage.RUST:
            # Buscar doc comments
            return '///' in content or '//!' in content
        
        return False


class ComplexityDistributionAnalyzer:
    """Analizador de distribución de complejidad."""
    
    def analyze_complexity_distribution(self, function_sizes: List[FunctionSizeMetrics],
                                      class_sizes: List[ClassSizeMetrics]) -> Dict[str, Any]:
        """
        Analiza distribución de complejidad en el archivo.
        
        Returns:
            Diccionario con análisis de distribución
        """
        analysis = {
            "function_size_distribution": self._analyze_function_distribution(function_sizes),
            "class_size_distribution": self._analyze_class_distribution(class_sizes),
            "outliers": self._identify_outliers(function_sizes, class_sizes),
            "recommendations": self._generate_distribution_recommendations(function_sizes, class_sizes)
        }
        
        return analysis
    
    def _analyze_function_distribution(self, function_sizes: List[FunctionSizeMetrics]) -> Dict[str, Any]:
        """Analiza distribución de tamaño de funciones."""
        if not function_sizes:
            return {"count": 0}
        
        sizes = [f.lines_of_code for f in function_sizes]
        
        return {
            "count": len(sizes),
            "min": min(sizes),
            "max": max(sizes),
            "average": sum(sizes) / len(sizes),
            "median": sorted(sizes)[len(sizes) // 2],
            "small_functions": len([s for s in sizes if s <= 10]),
            "medium_functions": len([s for s in sizes if 11 <= s <= 30]),
            "large_functions": len([s for s in sizes if 31 <= s <= 50]),
            "very_large_functions": len([s for s in sizes if s > 50])
        }
    
    def _analyze_class_distribution(self, class_sizes: List[ClassSizeMetrics]) -> Dict[str, Any]:
        """Analiza distribución de tamaño de clases."""
        if not class_sizes:
            return {"count": 0}
        
        sizes = [c.lines_of_code for c in class_sizes]
        
        return {
            "count": len(sizes),
            "min": min(sizes),
            "max": max(sizes),
            "average": sum(sizes) / len(sizes),
            "median": sorted(sizes)[len(sizes) // 2],
            "small_classes": len([s for s in sizes if s <= 100]),
            "medium_classes": len([s for s in sizes if 101 <= s <= 300]),
            "large_classes": len([s for s in sizes if 301 <= s <= 500]),
            "very_large_classes": len([s for s in sizes if s > 500])
        }
    
    def _identify_outliers(self, function_sizes: List[FunctionSizeMetrics], 
                          class_sizes: List[ClassSizeMetrics]) -> Dict[str, List[str]]:
        """Identifica outliers en tamaño."""
        outliers = {
            "oversized_functions": [],
            "oversized_classes": [],
            "functions_with_many_params": [],
            "functions_with_many_returns": []
        }
        
        # Funciones muy grandes
        for func in function_sizes:
            if func.lines_of_code > 100:
                outliers["oversized_functions"].append(f"{func.name} ({func.lines_of_code} LOC)")
            
            if func.parameters > 7:
                outliers["functions_with_many_params"].append(f"{func.name} ({func.parameters} params)")
            
            if func.return_statements > 5:
                outliers["functions_with_many_returns"].append(f"{func.name} ({func.return_statements} returns)")
        
        # Clases muy grandes
        for cls in class_sizes:
            if cls.lines_of_code > 1000:
                outliers["oversized_classes"].append(f"{cls.name} ({cls.lines_of_code} LOC)")
        
        return outliers
    
    def _generate_distribution_recommendations(self, function_sizes: List[FunctionSizeMetrics],
                                             class_sizes: List[ClassSizeMetrics]) -> List[str]:
        """Genera recomendaciones basadas en distribución."""
        recommendations = []
        
        if function_sizes:
            avg_func_size = sum(f.lines_of_code for f in function_sizes) / len(function_sizes)
            if avg_func_size > 30:
                recommendations.append(f"Average function size ({avg_func_size:.1f} LOC) is high - consider smaller functions")
            
            large_funcs = len([f for f in function_sizes if f.lines_of_code > 50])
            if large_funcs > len(function_sizes) * 0.2:  # Más del 20%
                recommendations.append(f"{large_funcs} functions are too large - break them down")
        
        if class_sizes:
            avg_class_size = sum(c.lines_of_code for c in class_sizes) / len(class_sizes)
            if avg_class_size > 500:
                recommendations.append(f"Average class size ({avg_class_size:.1f} LOC) is high - consider decomposition")
        
        return recommendations
