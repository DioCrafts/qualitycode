"""
Parser Especializado para Python con Análisis AST.

Este módulo implementa un parser especializado para Python que va más allá
del parsing básico, proporcionando análisis semántico profundo, detección
de patrones específicos, y análisis de flujo de datos.
"""

import ast
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from . import Parser, ParseResult, ProgrammingLanguage, ParserConfig

logger = logging.getLogger(__name__)


class PythonVersion(Enum):
    """Versiones de Python soportadas."""
    PYTHON_37 = "3.7"
    PYTHON_38 = "3.8"
    PYTHON_39 = "3.9"
    PYTHON_310 = "3.10"
    PYTHON_311 = "3.11"
    PYTHON_312 = "3.12"


@dataclass
class PythonParserConfig:
    """Configuración específica del parser de Python."""
    enable_semantic_analysis: bool = True
    enable_type_inference: bool = True
    enable_import_resolution: bool = True
    enable_data_flow_analysis: bool = True
    python_version: PythonVersion = PythonVersion.PYTHON_312
    strict_mode: bool = False
    enable_experimental_features: bool = False


@dataclass
class PythonScope:
    """Información de scope en Python."""
    scope_type: str  # 'module', 'function', 'class', 'comprehension'
    name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    parent_scope: Optional[str] = None
    symbols: Dict[str, Any] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    nonlocal_vars: List[str] = field(default_factory=list)
    global_vars: List[str] = field(default_factory=list)


@dataclass
class PythonType:
    """Representación de tipos en Python."""
    name: str
    is_builtin: bool = False
    is_generic: bool = False
    type_parameters: List['PythonType'] = field(default_factory=list)
    is_optional: bool = False
    is_union: bool = False
    union_types: List['PythonType'] = field(default_factory=list)


@dataclass
class FunctionDefinition:
    """Definición de función en Python."""
    name: str
    start_line: int
    end_line: int
    parameters: List[str]
    return_annotation: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    docstring: Optional[str] = None
    complexity: int = 1
    local_variables: List[str] = field(default_factory=list)
    calls_made: List[str] = field(default_factory=list)


@dataclass
class ClassDefinition:
    """Definición de clase en Python."""
    name: str
    start_line: int
    end_line: int
    bases: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    class_variables: List[str] = field(default_factory=list)
    instance_variables: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    metaclass: Optional[str] = None
    is_abstract: bool = False
    inheritance_depth: int = 0


@dataclass
class ImportStatement:
    """Declaración de import en Python."""
    module_name: str
    alias: Optional[str] = None
    start_line: int = 0
    is_relative: bool = False
    level: int = 0


@dataclass
class PythonPattern:
    """Patrón detectado en código Python."""
    pattern_name: str
    category: str  # 'antipattern', 'best_practice', 'performance', 'security'
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    message: str
    start_line: int
    end_line: int
    suggestion: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PythonMetrics:
    """Métricas específicas de Python."""
    lines_of_code: int = 0
    logical_lines_of_code: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    maintainability_index: float = 0.0
    
    # Python-specific metrics
    function_count: int = 0
    class_count: int = 0
    method_count: int = 0
    import_count: int = 0
    decorator_count: int = 0
    comprehension_count: int = 0
    lambda_count: int = 0
    generator_count: int = 0
    
    # Quality metrics
    docstring_coverage: float = 0.0
    type_annotation_coverage: float = 0.0
    duplication_percentage: float = 0.0
    
    # Complexity metrics
    max_function_complexity: int = 0
    average_function_complexity: float = 0.0
    max_class_complexity: int = 0
    nesting_depth: int = 0
    inheritance_depth: int = 0


@dataclass
class PythonAnalysisResult:
    """Resultado completo del análisis de Python."""
    file_path: Path
    ast: Optional[Any] = None  # ast.AST type
    scopes: List[PythonScope] = field(default_factory=list)
    functions: List[FunctionDefinition] = field(default_factory=list)
    classes: List[ClassDefinition] = field(default_factory=list)
    imports: List[ImportStatement] = field(default_factory=list)
    patterns: List[PythonPattern] = field(default_factory=list)
    metrics: PythonMetrics = field(default_factory=PythonMetrics)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PythonSemanticAnalyzer:
    """Analizador semántico para Python."""
    
    def __init__(self):
        self.scopes: List[PythonScope] = []
        self.current_scope: Optional[PythonScope] = None
        self.symbol_table: Dict[str, Any] = {}
        
    def analyze(self, tree: Any, content: str) -> PythonAnalysisResult:  # ast.AST type
        """Realiza análisis semántico completo."""
        result = PythonAnalysisResult(file_path=Path("<string>"))
        
        # Analizar scopes
        self.scopes = self._analyze_scopes(tree)
        result.scopes = self.scopes
        
        # Extraer funciones
        result.functions = self._extract_functions(tree)
        
        # Extraer clases
        result.classes = self._extract_classes(tree)
        
        # Extraer imports
        result.imports = self._extract_imports(tree)
        
        # Calcular métricas
        result.metrics = self._calculate_metrics(tree, content, result)
        
        return result
    
    def _analyze_scopes(self, tree: Any) -> List[PythonScope]:  # ast.AST type
        """Analiza los scopes en el AST."""
        scopes = []
        module_scope = PythonScope(
            scope_type="module",
            name="__main__",
            start_line=1,
            end_line=999999
        )
        scopes.append(module_scope)
        
        # TODO: Implementar análisis de scopes más detallado
        return scopes
    
    def _extract_functions(self, tree: Any) -> List[FunctionDefinition]:  # ast.AST type
        """Extrae definiciones de funciones."""
        functions = []
        
        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self, functions_list):
                self.functions = functions_list
                
            def visit_FunctionDef(self, node):
                func = FunctionDefinition(
                    name=node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    parameters=[arg.arg for arg in node.args.args],
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    docstring=ast.get_docstring(node),
                    complexity=self._calculate_complexity(node)
                )
                self.functions.append(func)
                self.generic_visit(node)
            
            def _calculate_complexity(self, node) -> int:
                """Calcula complejidad ciclomática básica."""
                complexity = 1  # Base complexity
                
                class ComplexityVisitor(ast.NodeVisitor):
                    def __init__(self):
                        self.complexity = 0
                    
                    def visit_If(self, node):
                        self.complexity += 1
                        self.generic_visit(node)
                    
                    def visit_For(self, node):
                        self.complexity += 1
                        self.generic_visit(node)
                    
                    def visit_While(self, node):
                        self.complexity += 1
                        self.generic_visit(node)
                    
                    def visit_ExceptHandler(self, node):
                        self.complexity += 1
                        self.generic_visit(node)
                    
                    def visit_With(self, node):
                        self.complexity += 1
                        self.generic_visit(node)
                
                visitor = ComplexityVisitor()
                visitor.visit(node)
                return complexity + visitor.complexity
        
        visitor = FunctionVisitor(functions)
        visitor.visit(tree)
        return functions
    
    def _extract_classes(self, tree: Any) -> List[ClassDefinition]:  # ast.AST type
        """Extrae definiciones de clases."""
        classes = []
        
        class ClassVisitor(ast.NodeVisitor):
            def __init__(self, classes_list):
                self.classes = classes_list
                
            def visit_ClassDef(self, node):
                cls = ClassDefinition(
                    name=node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    bases=[self._get_base_name(base) for base in node.bases],
                    docstring=ast.get_docstring(node)
                )
                self.classes.append(cls)
                self.generic_visit(node)
            
            def _get_base_name(self, base) -> str:
                """Obtiene el nombre de la clase base."""
                if isinstance(base, ast.Name):
                    return base.id
                elif isinstance(base, ast.Attribute):
                    return f"{self._get_base_name(base.value)}.{base.attr}"
                else:
                    return str(base)
        
        visitor = ClassVisitor(classes)
        visitor.visit(tree)
        return classes
    
    def _extract_imports(self, tree: Any) -> List[ImportStatement]:  # ast.AST type
        """Extrae declaraciones de import."""
        imports = []
        
        class ImportVisitor(ast.NodeVisitor):
            def __init__(self, imports_list):
                self.imports = imports_list
                
            def visit_Import(self, node):
                for alias in node.names:
                    import_stmt = ImportStatement(
                        module_name=alias.name,
                        alias=alias.asname,
                        start_line=node.lineno
                    )
                    self.imports.append(import_stmt)
            
            def visit_ImportFrom(self, node):
                module_name = node.module or ""
                for alias in node.names:
                    import_stmt = ImportStatement(
                        module_name=f"{module_name}.{alias.name}" if module_name else alias.name,
                        alias=alias.asname,
                        start_line=node.lineno,
                        is_relative=node.level > 0,
                        level=node.level
                    )
                    self.imports.append(import_stmt)
        
        visitor = ImportVisitor(imports)
        visitor.visit(tree)
        return imports
    
    def _calculate_metrics(self, tree: Any, content: str, result: PythonAnalysisResult) -> PythonMetrics:  # ast.AST type
        """Calcula métricas del código Python."""
        metrics = PythonMetrics()
        
        # Métricas básicas de líneas
        lines = content.split('\n')
        metrics.lines_of_code = len(lines)
        metrics.comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        metrics.blank_lines = sum(1 for line in lines if not line.strip())
        metrics.logical_lines_of_code = metrics.lines_of_code - metrics.comment_lines - metrics.blank_lines
        
        # Métricas de estructura
        metrics.function_count = len(result.functions)
        metrics.class_count = len(result.classes)
        metrics.import_count = len(result.imports)
        
        # Métricas de complejidad
        if result.functions:
            complexities = [f.complexity for f in result.functions]
            metrics.max_function_complexity = max(complexities)
            metrics.average_function_complexity = sum(complexities) / len(complexities)
            metrics.cyclomatic_complexity = sum(complexities)
        
        # Cobertura de docstrings
        total_definitions = metrics.function_count + metrics.class_count
        documented = sum(1 for f in result.functions if f.docstring) + sum(1 for c in result.classes if c.docstring)
        metrics.docstring_coverage = (documented / total_definitions * 100) if total_definitions > 0 else 100.0
        
        return metrics


class PythonPatternDetector:
    """Detector de patrones específicos de Python."""
    
    def __init__(self):
        self.patterns = [
            self._detect_mutable_default_arguments,
            self._detect_unused_imports,
            self._detect_too_many_arguments,
            self._detect_deep_nesting,
            self._detect_string_concatenation,
            self._detect_bare_except,
            self._detect_global_variables,
            self._detect_long_functions,
        ]
    
    def detect_patterns(self, result: PythonAnalysisResult) -> List[PythonPattern]:
        """Detecta patrones en el código Python."""
        patterns = []
        
        for pattern_func in self.patterns:
            try:
                detected = pattern_func(result)
                patterns.extend(detected)
            except Exception as e:
                logger.warning(f"Error detecting pattern {pattern_func.__name__}: {e}")
        
        return patterns
    
    def _detect_mutable_default_arguments(self, result: PythonAnalysisResult) -> List[PythonPattern]:
        """Detecta argumentos por defecto mutables."""
        patterns = []
        
        for func in result.functions:
            # TODO: Implementar detección de argumentos mutables
            pass
        
        return patterns
    
    def _detect_unused_imports(self, result: PythonAnalysisResult) -> List[PythonPattern]:
        """Detecta imports no utilizados."""
        patterns = []
        
        # TODO: Implementar detección de imports no utilizados
        return patterns
    
    def _detect_too_many_arguments(self, result: PythonAnalysisResult) -> List[PythonPattern]:
        """Detecta funciones con demasiados argumentos."""
        patterns = []
        
        for func in result.functions:
            if len(func.parameters) > 7:  # Threshold configurable
                patterns.append(PythonPattern(
                    pattern_name="too_many_arguments",
                    category="antipattern",
                    severity="medium",
                    message=f"Function '{func.name}' has {len(func.parameters)} arguments (max recommended: 7)",
                    start_line=func.start_line,
                    end_line=func.end_line,
                    suggestion="Consider using a data class or dictionary to group related parameters"
                ))
        
        return patterns
    
    def _detect_deep_nesting(self, result: PythonAnalysisResult) -> List[PythonPattern]:
        """Detecta anidamiento profundo."""
        patterns = []
        
        # TODO: Implementar detección de anidamiento profundo
        return patterns
    
    def _detect_string_concatenation(self, result: PythonAnalysisResult) -> List[PythonPattern]:
        """Detecta concatenación de strings ineficiente."""
        patterns = []
        
        # TODO: Implementar detección de concatenación de strings
        return patterns
    
    def _detect_bare_except(self, result: PythonAnalysisResult) -> List[PythonPattern]:
        """Detecta bloques except sin especificar excepción."""
        patterns = []
        
        # TODO: Implementar detección de bare except
        return patterns
    
    def _detect_global_variables(self, result: PythonAnalysisResult) -> List[PythonPattern]:
        """Detecta uso de variables globales."""
        patterns = []
        
        # TODO: Implementar detección de variables globales
        return patterns
    
    def _detect_long_functions(self, result: PythonAnalysisResult) -> List[PythonPattern]:
        """Detecta funciones muy largas."""
        patterns = []
        
        for func in result.functions:
            lines = func.end_line - func.start_line + 1
            if lines > 50:  # Threshold configurable
                patterns.append(PythonPattern(
                    pattern_name="long_function",
                    category="antipattern",
                    severity="medium",
                    message=f"Function '{func.name}' is {lines} lines long (max recommended: 50)",
                    start_line=func.start_line,
                    end_line=func.end_line,
                    suggestion="Consider breaking the function into smaller, more focused functions"
                ))
        
        return patterns


class PythonSpecializedParser(Parser):
    """
    Parser especializado para Python con análisis AST completo.
    
    Implementa análisis semántico profundo, detección de patrones,
    y cálculo de métricas específicas de Python.
    """
    
    def __init__(self, config: Optional[PythonParserConfig] = None):
        self.config = config or PythonParserConfig()
        self.semantic_analyzer = PythonSemanticAnalyzer()
        self.pattern_detector = PythonPatternDetector()
        self._logger = logging.getLogger(__name__)
    
    async def parse_file(self, file_path: Path) -> ParseResult:
        """Parsea un archivo Python y retorna el resultado."""
        start_time = time.time()
        
        try:
            # Leer contenido del archivo
            content = await asyncio.to_thread(file_path.read_text, encoding='utf-8')
            
            # Parsear con AST nativo de Python
            tree = ast.parse(content, filename=str(file_path))
            
            # Realizar análisis semántico
            analysis_result = self.semantic_analyzer.analyze(tree, content)
            analysis_result.file_path = file_path
            
            # Detectar patrones
            analysis_result.patterns = self.pattern_detector.detect_patterns(analysis_result)
            
            # Crear resultado del parser universal
            result = ParseResult(
                file_path=file_path,
                language=ProgrammingLanguage.PYTHON,
                ast=tree,
                metadata={
                    'python_analysis': analysis_result,
                    'scopes_count': len(analysis_result.scopes),
                    'functions_count': len(analysis_result.functions),
                    'classes_count': len(analysis_result.classes),
                    'imports_count': len(analysis_result.imports),
                    'patterns_count': len(analysis_result.patterns),
                    'metrics': analysis_result.metrics.__dict__
                }
            )
            
            result.parse_time_ms = (time.time() - start_time) * 1000
            return result
            
        except SyntaxError as e:
            error_msg = f"Syntax error in {file_path}: {e}"
            self._logger.error(error_msg)
            return ParseResult(
                file_path=file_path,
                language=ProgrammingLanguage.PYTHON,
                errors=[error_msg],
                parse_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            error_msg = f"Error parsing {file_path}: {e}"
            self._logger.error(error_msg, exc_info=True)
            return ParseResult(
                file_path=file_path,
                language=ProgrammingLanguage.PYTHON,
                errors=[error_msg],
                parse_time_ms=(time.time() - start_time) * 1000
            )
    
    async def parse_content(self, content: str, language: ProgrammingLanguage) -> ParseResult:
        """Parsea contenido de texto Python."""
        start_time = time.time()
        
        try:
            # Parsear con AST nativo de Python
            tree = ast.parse(content, filename="<string>")
            
            # Realizar análisis semántico
            analysis_result = self.semantic_analyzer.analyze(tree, content)
            
            # Detectar patrones
            analysis_result.patterns = self.pattern_detector.detect_patterns(analysis_result)
            
            # Crear resultado del parser universal
            result = ParseResult(
                file_path=Path("<string>"),
                language=ProgrammingLanguage.PYTHON,
                ast=tree,
                metadata={
                    'python_analysis': analysis_result,
                    'scopes_count': len(analysis_result.scopes),
                    'functions_count': len(analysis_result.functions),
                    'classes_count': len(analysis_result.classes),
                    'imports_count': len(analysis_result.imports),
                    'patterns_count': len(analysis_result.patterns),
                    'metrics': analysis_result.metrics.__dict__
                }
            )
            
            result.parse_time_ms = (time.time() - start_time) * 1000
            return result
            
        except SyntaxError as e:
            error_msg = f"Syntax error in content: {e}"
            self._logger.error(error_msg)
            return ParseResult(
                file_path=Path("<string>"),
                language=ProgrammingLanguage.PYTHON,
                errors=[error_msg],
                parse_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            error_msg = f"Error parsing content: {e}"
            self._logger.error(error_msg, exc_info=True)
            return ParseResult(
                file_path=Path("<string>"),
                language=ProgrammingLanguage.PYTHON,
                errors=[error_msg],
                parse_time_ms=(time.time() - start_time) * 1000
            )
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Retorna los lenguajes soportados por este parser."""
        return [ProgrammingLanguage.PYTHON]


# Funciones de utilidad para el parser de Python
async def analyze_python_file(file_path: Path, config: Optional[PythonParserConfig] = None) -> PythonAnalysisResult:
    """Función de conveniencia para analizar un archivo Python."""
    parser = PythonSpecializedParser(config)
    result = await parser.parse_file(file_path)
    
    if result.errors:
        raise ValueError(f"Error parsing {file_path}: {result.errors}")
    
    return result.metadata['python_analysis']


async def analyze_python_content(content: str, config: Optional[PythonParserConfig] = None) -> PythonAnalysisResult:
    """Función de conveniencia para analizar contenido Python."""
    parser = PythonSpecializedParser(config)
    result = await parser.parse_content(content, ProgrammingLanguage.PYTHON)
    
    if result.errors:
        raise ValueError(f"Error parsing content: {result.errors}")
    
    return result.metadata['python_analysis']


__all__ = [
    'PythonVersion',
    'PythonParserConfig',
    'PythonScope',
    'PythonType',
    'FunctionDefinition',
    'ClassDefinition',
    'ImportStatement',
    'PythonPattern',
    'PythonMetrics',
    'PythonAnalysisResult',
    'PythonSemanticAnalyzer',
    'PythonPatternDetector',
    'PythonSpecializedParser',
    'analyze_python_file',
    'analyze_python_content'
]
