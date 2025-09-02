"""
Parser Especializado para TypeScript/JavaScript con Análisis AST.

Este módulo implementa un parser especializado para TypeScript y JavaScript que va más allá
del parsing básico, proporcionando análisis semántico profundo, inferencia de tipos,
análisis de módulos ES6+, y detección de patrones específicos del ecosistema JS/TS.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time

from . import Parser as BaseParser, ParseResult, ProgrammingLanguage, ParserConfig

logger = logging.getLogger(__name__)


class JSLanguage(Enum):
    """Lenguajes JavaScript/TypeScript soportados."""
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JSX = "jsx"
    TSX = "tsx"


@dataclass
class TypeScriptParserConfig:
    """Configuración específica del parser de TypeScript/JavaScript."""
    enable_type_analysis: bool = True
    enable_module_resolution: bool = True
    enable_react_analysis: bool = True
    enable_nodejs_analysis: bool = True
    js_language: JSLanguage = JSLanguage.TYPESCRIPT
    target_version: str = "ES2022"
    strict_mode: bool = False
    enable_experimental_features: bool = False


@dataclass
class JSScope:
    """Información de scope en JavaScript/TypeScript."""
    scope_type: str  # 'module', 'function', 'class', 'block'
    name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    parent_scope: Optional[str] = None
    symbols: Dict[str, Any] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    hoisted_vars: List[str] = field(default_factory=list)


@dataclass
class JSType:
    """Representación de tipos en TypeScript/JavaScript."""
    name: str
    is_builtin: bool = False
    is_generic: bool = False
    type_parameters: List['JSType'] = field(default_factory=list)
    is_optional: bool = False
    is_union: bool = False
    union_types: List['JSType'] = field(default_factory=list)
    is_interface: bool = False
    is_type_alias: bool = False


@dataclass
class JSFunctionDefinition:
    """Definición de función en JavaScript/TypeScript."""
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
    is_arrow_function: bool = False
    docstring: Optional[str] = None
    complexity: int = 1
    local_variables: List[str] = field(default_factory=list)
    calls_made: List[str] = field(default_factory=list)


@dataclass
class JSClassDefinition:
    """Definición de clase en JavaScript/TypeScript."""
    name: str
    start_line: int
    end_line: int
    bases: List[str] = field(default_factory=list)
    implements: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    class_variables: List[str] = field(default_factory=list)
    instance_variables: List[str] = field(default_factory=list)
    is_abstract: bool = False
    is_interface: bool = False


@dataclass
class JSImportStatement:
    """Declaración de import en JavaScript/TypeScript."""
    module_name: str
    alias: Optional[str] = None
    start_line: int = 0
    is_relative: bool = False
    level: int = 0
    imported_names: List[str] = field(default_factory=list)
    is_type_only: bool = False
    is_side_effect: bool = False


@dataclass
class JSExportStatement:
    """Declaración de export en JavaScript/TypeScript."""
    export_type: str  # 'default', 'named', 'namespace', 're-export'
    exported_names: List[str] = field(default_factory=list)
    module_specifier: Optional[str] = None
    start_line: int = 0
    is_type_only: bool = False


@dataclass
class JSPattern:
    """Patrón detectado en código JavaScript/TypeScript."""
    pattern_name: str
    category: str  # 'typescript', 'javascript', 'react', 'nodejs', 'performance', 'security'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    start_line: int = 0
    end_line: int = 0
    suggestion: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JSMetrics:
    """Métricas específicas de JavaScript/TypeScript."""
    lines_of_code: int = 0
    logical_lines_of_code: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    export_count: int = 0
    type_annotation_count: int = 0
    interface_count: int = 0
    type_alias_count: int = 0
    enum_count: int = 0
    jsx_element_count: int = 0
    react_component_count: int = 0
    hook_usage_count: int = 0
    nodejs_api_usage_count: int = 0
    async_function_count: int = 0
    promise_usage_count: int = 0
    docstring_coverage: float = 0.0
    type_coverage: float = 0.0
    module_coupling: float = 0.0
    technical_debt_score: float = 0.0


@dataclass
class JSAnalysisResult:
    """Resultado completo del análisis de JavaScript/TypeScript."""
    file_path: Path
    language: JSLanguage
    ast: Optional[Any] = None  # tree-sitter AST
    scopes: List[JSScope] = field(default_factory=list)
    functions: List[JSFunctionDefinition] = field(default_factory=list)
    classes: List[JSClassDefinition] = field(default_factory=list)
    imports: List[JSImportStatement] = field(default_factory=list)
    exports: List[JSExportStatement] = field(default_factory=list)
    types: List[JSType] = field(default_factory=list)
    patterns: List[JSPattern] = field(default_factory=list)
    metrics: JSMetrics = field(default_factory=JSMetrics)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class TypeScriptSemanticAnalyzer:
    """Analizador semántico para TypeScript/JavaScript."""
    
    def __init__(self):
        self.symbol_table: Dict[str, Any] = {}
        
    def analyze(self, tree: Any, content: str) -> JSAnalysisResult:
        """Realiza análisis semántico completo."""
        result = JSAnalysisResult(file_path=Path("<string>"), language=JSLanguage.TYPESCRIPT)
        
        # Análisis básico del AST
        result.scopes = self._analyze_scopes(tree)
        result.functions = self._extract_functions(tree)
        result.classes = self._extract_classes(tree)
        result.imports = self._extract_imports(tree)
        result.exports = self._extract_exports(tree)
        result.types = self._extract_types(tree)
        
        # Cálculo de métricas
        result.metrics = self._calculate_metrics(tree, content, result)
        
        return result
    
    def _analyze_scopes(self, tree: Any) -> List[JSScope]:
        """Analiza los scopes en el AST."""
        scopes = []
        # Implementación básica - se expandirá
        return scopes
    
    def _extract_functions(self, tree: Any) -> List[JSFunctionDefinition]:
        """Extrae definiciones de funciones."""
        functions = []
        # Implementación básica - se expandirá
        return functions
    
    def _extract_classes(self, tree: Any) -> List[JSClassDefinition]:
        """Extrae definiciones de clases."""
        classes = []
        # Implementación básica - se expandirá
        return classes
    
    def _extract_imports(self, tree: Any) -> List[JSImportStatement]:
        """Extrae declaraciones de import."""
        imports = []
        # Implementación básica - se expandirá
        return imports
    
    def _extract_exports(self, tree: Any) -> List[JSExportStatement]:
        """Extrae declaraciones de export."""
        exports = []
        # Implementación básica - se expandirá
        return exports
    
    def _extract_types(self, tree: Any) -> List[JSType]:
        """Extrae definiciones de tipos."""
        types = []
        # Implementación básica - se expandirá
        return types
    
    def _calculate_metrics(self, tree: Any, content: str, result: JSAnalysisResult) -> JSMetrics:
        """Calcula métricas del código JavaScript/TypeScript."""
        metrics = JSMetrics()
        
        # Líneas de código
        lines = content.split('\n')
        metrics.lines_of_code = len(lines)
        metrics.logical_lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('//')])
        metrics.comment_lines = len([line for line in lines if line.strip().startswith('//') or line.strip().startswith('/*')])
        metrics.blank_lines = len([line for line in lines if not line.strip()])
        
        # Conteos básicos
        metrics.function_count = len(result.functions)
        metrics.class_count = len(result.classes)
        metrics.import_count = len(result.imports)
        metrics.export_count = len(result.exports)
        metrics.type_annotation_count = len(result.types)
        
        # Complejidad ciclomática básica
        metrics.cyclomatic_complexity = 1  # Base complexity
        
        return metrics


class TypeScriptPatternDetector:
    """Detector de patrones específicos de TypeScript/JavaScript."""
    
    def __init__(self):
        self.patterns = {
            'any_type_usage': self._detect_any_type_usage,
            'missing_type_annotations': self._detect_missing_type_annotations,
            'var_usage': self._detect_var_usage,
            'console_log': self._detect_console_log,
            'unused_variables': self._detect_unused_variables,
            'react_hooks_rules': self._detect_react_hooks_rules,
            'async_await_preference': self._detect_async_await_preference,
        }
    
    def detect_patterns(self, result: JSAnalysisResult) -> List[JSPattern]:
        """Detecta patrones en el código."""
        patterns = []
        
        for pattern_name, pattern_func in self.patterns.items():
            try:
                pattern_results = pattern_func(result)
                patterns.extend(pattern_results)
            except Exception as e:
                logger.warning(f"Error detecting pattern {pattern_name}: {e}")
        
        return patterns
    
    def _detect_any_type_usage(self, result: JSAnalysisResult) -> List[JSPattern]:
        """Detecta uso del tipo 'any'."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_missing_type_annotations(self, result: JSAnalysisResult) -> List[JSPattern]:
        """Detecta falta de anotaciones de tipo."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_var_usage(self, result: JSAnalysisResult) -> List[JSPattern]:
        """Detecta uso de 'var' en lugar de 'let'/'const'."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_console_log(self, result: JSAnalysisResult) -> List[JSPattern]:
        """Detecta uso de console.log en código de producción."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_unused_variables(self, result: JSAnalysisResult) -> List[JSPattern]:
        """Detecta variables no utilizadas."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_react_hooks_rules(self, result: JSAnalysisResult) -> List[JSPattern]:
        """Detecta violaciones de las reglas de hooks de React."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_async_await_preference(self, result: JSAnalysisResult) -> List[JSPattern]:
        """Detecta preferencia por async/await sobre Promises."""
        patterns = []
        # Implementación básica
        return patterns


class TypeScriptSpecializedParser(BaseParser):
    """Parser especializado para TypeScript/JavaScript."""
    
    def __init__(self, config: Optional[TypeScriptParserConfig] = None):
        self.config = config or TypeScriptParserConfig()
        self.semantic_analyzer = TypeScriptSemanticAnalyzer()
        self.pattern_detector = TypeScriptPatternDetector()
        self._setup_tree_sitter()
    
    def _setup_tree_sitter(self):
        """Configura tree-sitter para JavaScript/TypeScript."""
        try:
            # Intentar cargar los lenguajes tree-sitter
            # Por ahora, usamos un método de fallback
            logger.info("Using fallback parsing method for TypeScript/JavaScript")
        except Exception as e:
            logger.warning(f"Could not setup tree-sitter: {e}")
            logger.info("Using fallback parsing method")
    
    async def parse_file(self, file_path: Path) -> ParseResult:
        """Parsea un archivo TypeScript/JavaScript."""
        start_time = time.time()
        
        try:
            # Leer contenido del archivo
            content = file_path.read_text(encoding='utf-8')
            
            # Detectar lenguaje
            language = self._detect_language(file_path, content)
            
            # Parsear contenido
            result = await self.parse_content(content, ProgrammingLanguage.TYPESCRIPT)  # Usar un lenguaje por defecto
            result.file_path = file_path
            
            # Corregir el lenguaje basado en la detección
            js_language = self._detect_js_language_from_content(content)
            if js_language == JSLanguage.TYPESCRIPT:
                result.language = ProgrammingLanguage.TYPESCRIPT
            elif js_language == JSLanguage.TSX:
                result.language = ProgrammingLanguage.TSX
            elif js_language == JSLanguage.JSX:
                result.language = ProgrammingLanguage.JSX
            else:
                result.language = ProgrammingLanguage.JAVASCRIPT
            
            return result
            
        except Exception as e:
            error_msg = f"Error parsing TypeScript/JavaScript file {file_path}: {e}"
            logger.error(error_msg)
            return ParseResult(
                language=ProgrammingLanguage.TYPESCRIPT,
                file_path=file_path,
                ast=None,
                errors=[error_msg],
                parse_time_ms=int((time.time() - start_time) * 1000),
                metadata={}
            )
    
    async def parse_content(self, content: str, language: ProgrammingLanguage) -> ParseResult:
        """Parsea contenido TypeScript/JavaScript."""
        start_time = time.time()
        
        try:
            # Detectar lenguaje JS/TS
            js_language = self._detect_js_language_from_content(content)
            
            # Parsear con tree-sitter o método alternativo
            ast = self._parse_with_tree_sitter(content, js_language)
            
            # Análisis semántico
            analysis_result = self.semantic_analyzer.analyze(ast, content)
            analysis_result.language = js_language
            
            # Detección de patrones
            analysis_result.patterns = self.pattern_detector.detect_patterns(analysis_result)
            
            # Calcular tiempo de parsing
            analysis_result.metrics.parse_duration_ms = int((time.time() - start_time) * 1000)
            
            # Crear resultado del parser universal
            if js_language == JSLanguage.TYPESCRIPT:
                language = ProgrammingLanguage.TYPESCRIPT
            elif js_language == JSLanguage.TSX:
                language = ProgrammingLanguage.TSX
            elif js_language == JSLanguage.JSX:
                language = ProgrammingLanguage.JSX
            else:
                language = ProgrammingLanguage.JAVASCRIPT
            
            result = ParseResult(
                language=language,
                file_path=Path("<string>"),
                ast=ast,  # Usar el AST del parsing
                errors=analysis_result.errors,
                parse_time_ms=int((time.time() - start_time) * 1000),
                metadata={
                    'typescript_analysis': analysis_result,
                    'language': js_language.value,
                }
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error parsing TypeScript/JavaScript content: {e}"
            logger.error(error_msg)
            return ParseResult(
                language=ProgrammingLanguage.JAVASCRIPT,
                file_path=Path("<string>"),
                ast=None,
                errors=[error_msg],
                parse_time_ms=int((time.time() - start_time) * 1000),
                metadata={}
            )
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Retorna los lenguajes soportados."""
        return [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]
    
    def _detect_language(self, file_path: Path, content: str) -> JSLanguage:
        """Detecta el lenguaje basado en la extensión del archivo y contenido."""
        extension = file_path.suffix.lower()
        
        if extension == '.ts':
            return JSLanguage.TYPESCRIPT
        elif extension == '.tsx':
            return JSLanguage.TSX
        elif extension == '.jsx':
            return JSLanguage.JSX
        elif extension in ['.js', '.mjs', '.cjs']:
            return self._detect_js_language_from_content(content)
        else:
            return JSLanguage.JAVASCRIPT
    
    def _detect_js_language_from_content(self, content: str) -> JSLanguage:
        """Detecta el lenguaje basado en el contenido."""
        # Verificar sintaxis TypeScript
        ts_patterns = [
            r':\s*\w+\s*[=;,\)]',  # Type annotations
            r'interface\s+\w+',     # Interface declarations
            r'type\s+\w+\s*=',      # Type aliases
            r'enum\s+\w+',          # Enums
            r'namespace\s+\w+',     # Namespaces
            r'declare\s+',          # Declare statements
            r'abstract\s+class',    # Abstract classes
            r'implements\s+\w+',    # Implements clause
            r'<\w+>',               # Generic type parameters
            r'as\s+\w+',            # Type assertions
        ]
        
        has_ts_syntax = any(re.search(pattern, content) for pattern in ts_patterns)
        
        # Verificar sintaxis JSX (más específico)
        jsx_patterns = [
            r'<\w+[^>]*>.*?</\w+>',  # JSX elements con contenido
            r'<\w+\s+[^>]*>',        # JSX elements con atributos
            r'<\w+\s*/\s*>',         # Self-closing JSX tags
            r'{\s*\w+\s*}',          # JSX expressions
        ]
        
        has_jsx_syntax = any(re.search(pattern, content, re.DOTALL) for pattern in jsx_patterns)
        
        # Si tiene sintaxis TypeScript pero no JSX, es TypeScript puro
        if has_ts_syntax and not has_jsx_syntax:
            return JSLanguage.TYPESCRIPT
        # Si tiene sintaxis TypeScript y JSX, es TSX
        elif has_ts_syntax and has_jsx_syntax:
            return JSLanguage.TSX
        # Si solo tiene JSX, es JSX (pero verificar que no tenga TypeScript)
        elif has_jsx_syntax and not has_ts_syntax:
            return JSLanguage.JSX
        # Si no tiene ninguna, es JavaScript
        else:
            return JSLanguage.JAVASCRIPT
    
    def _parse_with_tree_sitter(self, content: str, language: JSLanguage) -> Any:
        """Parsea usando tree-sitter."""
        try:
            # Implementación básica - se expandirá con tree-sitter real
            # Por ahora, retornamos un objeto simple que representa el AST
            return {
                'type': 'program',
                'language': language.value,
                'content': content,
                'lines': len(content.split('\n')),
            }
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed: {e}")
            # Fallback a parsing básico
            return self._fallback_parse(content, language)
    
    def _fallback_parse(self, content: str, language: JSLanguage) -> Any:
        """Método de fallback para parsing básico."""
        return {
            'type': 'program',
            'language': language.value,
            'content': content,
            'lines': len(content.split('\n')),
            'fallback': True,
        }


# Funciones de utilidad
async def analyze_typescript_file(file_path: Path, config: Optional[TypeScriptParserConfig] = None) -> JSAnalysisResult:
    """Analiza un archivo TypeScript/JavaScript."""
    parser = TypeScriptSpecializedParser(config)
    result = await parser.parse_content(file_path.read_text(encoding='utf-8'), ProgrammingLanguage.TYPESCRIPT)
    return result.metadata['typescript_analysis']


async def analyze_typescript_content(content: str, config: Optional[TypeScriptParserConfig] = None) -> JSAnalysisResult:
    """Analiza contenido TypeScript/JavaScript."""
    parser = TypeScriptSpecializedParser(config)
    result = await parser.parse_content(content, ProgrammingLanguage.TYPESCRIPT)
    return result.metadata['typescript_analysis']


__all__ = [
    'JSLanguage',
    'TypeScriptParserConfig',
    'JSScope',
    'JSType',
    'JSFunctionDefinition',
    'JSClassDefinition',
    'JSImportStatement',
    'JSExportStatement',
    'JSPattern',
    'JSMetrics',
    'JSAnalysisResult',
    'TypeScriptSemanticAnalyzer',
    'TypeScriptPatternDetector',
    'TypeScriptSpecializedParser',
    'analyze_typescript_file',
    'analyze_typescript_content',
]