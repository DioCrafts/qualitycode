"""
Parser Especializado para Rust con Análisis AST.

Este módulo implementa un parser especializado para Rust que va más allá del parsing básico,
proporcionando análisis semántico profundo, análisis de ownership y borrowing, detección
de patrones idiomáticos de Rust, análisis de unsafe code, y capacidades avanzadas de
inspección específicas del ecosistema Rust.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import toml

from . import Parser as BaseParser, ParseResult, ProgrammingLanguage, ParserConfig

logger = logging.getLogger(__name__)


class RustEdition(Enum):
    """Ediciones de Rust soportadas."""
    EDITION_2015 = "2015"
    EDITION_2018 = "2018"
    EDITION_2021 = "2021"
    EDITION_2024 = "2024"


@dataclass
class RustParserConfig:
    """Configuración específica del parser de Rust."""
    enable_ownership_analysis: bool = True
    enable_lifetime_analysis: bool = True
    enable_trait_analysis: bool = True
    enable_unsafe_analysis: bool = True
    enable_cargo_analysis: bool = True
    enable_macro_analysis: bool = True
    analyze_dependencies: bool = True
    strict_mode: bool = False
    edition: RustEdition = RustEdition.EDITION_2021


@dataclass
class RustScope:
    """Información de scope en Rust."""
    scope_type: str  # 'module', 'function', 'struct', 'enum', 'trait', 'impl', 'block'
    name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    parent_scope: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    ownership_info: Dict[str, Any] = field(default_factory=dict)
    lifetimes: List[str] = field(default_factory=list)


@dataclass
class RustType:
    """Representación de tipos en Rust."""
    name: str
    is_builtin: bool = False
    is_generic: bool = False
    type_parameters: List['RustType'] = field(default_factory=list)
    lifetime_parameters: List[str] = field(default_factory=list)
    is_reference: bool = False
    is_mutable: bool = False
    reference_lifetime: Optional[str] = None
    is_unsafe: bool = False


@dataclass
class RustFunctionDefinition:
    """Definición de función en Rust."""
    name: str
    start_line: int
    end_line: int
    parameters: List[str]
    return_type: Optional[str] = None
    lifetime_parameters: List[str] = field(default_factory=list)
    type_parameters: List[str] = field(default_factory=list)
    is_unsafe: bool = False
    is_async: bool = False
    is_const: bool = False
    visibility: str = "private"
    docstring: Optional[str] = None
    complexity: int = 1
    ownership_transfers: List[str] = field(default_factory=list)
    borrows: List[str] = field(default_factory=list)


@dataclass
class RustStructDefinition:
    """Definición de struct en Rust."""
    name: str
    start_line: int
    end_line: int
    fields: List[str]
    lifetime_parameters: List[str] = field(default_factory=list)
    type_parameters: List[str] = field(default_factory=list)
    derives: List[str] = field(default_factory=list)
    visibility: str = "private"
    is_unsafe: bool = False


@dataclass
class RustEnumDefinition:
    """Definición de enum en Rust."""
    name: str
    start_line: int
    end_line: int
    variants: List[str]
    lifetime_parameters: List[str] = field(default_factory=list)
    type_parameters: List[str] = field(default_factory=list)
    derives: List[str] = field(default_factory=list)
    visibility: str = "private"


@dataclass
class RustTraitDefinition:
    """Definición de trait en Rust."""
    name: str
    start_line: int
    end_line: int
    associated_types: List[str] = field(default_factory=list)
    associated_functions: List[str] = field(default_factory=list)
    super_traits: List[str] = field(default_factory=list)
    lifetime_parameters: List[str] = field(default_factory=list)
    type_parameters: List[str] = field(default_factory=list)
    is_object_safe: bool = True
    visibility: str = "private"


@dataclass
class RustImplBlock:
    """Bloque de implementación en Rust."""
    type_name: str
    start_line: int
    end_line: int
    trait_name: Optional[str] = None
    methods: List[str] = field(default_factory=list)
    associated_types: List[str] = field(default_factory=list)
    lifetime_parameters: List[str] = field(default_factory=list)
    type_parameters: List[str] = field(default_factory=list)
    is_unsafe: bool = False


@dataclass
class RustImportStatement:
    """Declaración de import en Rust."""
    module_name: str
    alias: Optional[str] = None
    start_line: int = 0
    imported_items: List[str] = field(default_factory=list)
    is_glob: bool = False
    is_extern: bool = False


@dataclass
class RustUseStatement:
    """Declaración de use en Rust."""
    path: str
    alias: Optional[str] = None
    start_line: int = 0
    is_glob: bool = False


@dataclass
class RustOwnershipInfo:
    """Información de ownership y borrowing."""
    ownership_transfers: List[Dict[str, Any]] = field(default_factory=list)
    borrowing_patterns: List[Dict[str, Any]] = field(default_factory=list)
    move_semantics: List[Dict[str, Any]] = field(default_factory=list)
    ownership_violations: List[Dict[str, Any]] = field(default_factory=list)
    smart_pointer_usage: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RustLifetimeInfo:
    """Información de lifetimes."""
    explicit_lifetimes: List[str] = field(default_factory=list)
    elided_lifetimes: List[Dict[str, Any]] = field(default_factory=list)
    lifetime_bounds: List[Dict[str, Any]] = field(default_factory=list)
    lifetime_relationships: List[Dict[str, Any]] = field(default_factory=list)
    lifetime_errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RustTraitInfo:
    """Información del sistema de traits."""
    trait_definitions: List[RustTraitDefinition] = field(default_factory=list)
    trait_implementations: List[Dict[str, Any]] = field(default_factory=list)
    associated_types: List[Dict[str, Any]] = field(default_factory=list)
    trait_bounds: List[Dict[str, Any]] = field(default_factory=list)
    trait_objects: List[Dict[str, Any]] = field(default_factory=list)
    coherence_violations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RustUnsafeInfo:
    """Información de código unsafe."""
    unsafe_blocks: List[Dict[str, Any]] = field(default_factory=list)
    unsafe_functions: List[RustFunctionDefinition] = field(default_factory=list)
    unsafe_traits: List[RustTraitDefinition] = field(default_factory=list)
    raw_pointer_usage: List[Dict[str, Any]] = field(default_factory=list)
    ffi_declarations: List[Dict[str, Any]] = field(default_factory=list)
    transmute_usage: List[Dict[str, Any]] = field(default_factory=list)
    safety_violations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RustCargoInfo:
    """Información de Cargo y dependencias."""
    manifest: Optional[Dict[str, Any]] = None
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    dev_dependencies: List[Dict[str, Any]] = field(default_factory=list)
    build_dependencies: List[Dict[str, Any]] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    workspace_info: Optional[Dict[str, Any]] = None
    security_advisories: List[Dict[str, Any]] = field(default_factory=list)
    outdated_dependencies: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RustMacroInfo:
    """Información de macros."""
    macro_definitions: List[Dict[str, Any]] = field(default_factory=list)
    macro_invocations: List[Dict[str, Any]] = field(default_factory=list)
    procedural_macros: List[Dict[str, Any]] = field(default_factory=list)
    declarative_macros: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RustPattern:
    """Patrón detectado en código Rust."""
    pattern_name: str
    category: str  # 'ownership', 'lifetimes', 'safety', 'performance', 'idioms', 'error_handling'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    start_line: int = 0
    end_line: int = 0
    suggestion: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RustMetrics:
    """Métricas específicas de Rust."""
    lines_of_code: int = 0
    logical_lines_of_code: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    function_count: int = 0
    struct_count: int = 0
    enum_count: int = 0
    trait_count: int = 0
    impl_count: int = 0
    macro_count: int = 0
    unsafe_block_count: int = 0
    unsafe_function_count: int = 0
    lifetime_parameter_count: int = 0
    generic_parameter_count: int = 0
    ownership_transfer_count: int = 0
    borrow_count: int = 0
    raw_pointer_count: int = 0
    ffi_declaration_count: int = 0
    docstring_coverage: float = 0.0
    unsafe_code_percentage: float = 0.0
    memory_safety_score: float = 0.0
    performance_score: float = 0.0


@dataclass
class RustAnalysisResult:
    """Resultado completo del análisis de Rust."""
    file_path: Path
    ast: Optional[Any] = None  # tree-sitter AST
    scopes: List[RustScope] = field(default_factory=list)
    functions: List[RustFunctionDefinition] = field(default_factory=list)
    structs: List[RustStructDefinition] = field(default_factory=list)
    enums: List[RustEnumDefinition] = field(default_factory=list)
    traits: List[RustTraitDefinition] = field(default_factory=list)
    impls: List[RustImplBlock] = field(default_factory=list)
    imports: List[RustImportStatement] = field(default_factory=list)
    uses: List[RustUseStatement] = field(default_factory=list)
    ownership_info: Optional[RustOwnershipInfo] = None
    lifetime_info: Optional[RustLifetimeInfo] = None
    trait_info: Optional[RustTraitInfo] = None
    unsafe_info: Optional[RustUnsafeInfo] = None
    cargo_info: Optional[RustCargoInfo] = None
    macro_info: Optional[RustMacroInfo] = None
    patterns: List[RustPattern] = field(default_factory=list)
    metrics: RustMetrics = field(default_factory=RustMetrics)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class RustSemanticAnalyzer:
    """Analizador semántico para Rust."""
    
    def __init__(self):
        self.symbol_table: Dict[str, Any] = {}
        
    def analyze(self, tree: Any, content: str) -> RustAnalysisResult:
        """Realiza análisis semántico completo."""
        result = RustAnalysisResult(file_path=Path("<string>"))
        
        # Análisis básico del AST
        result.scopes = self._analyze_scopes(tree)
        result.functions = self._extract_functions(tree)
        result.structs = self._extract_structs(tree)
        result.enums = self._extract_enums(tree)
        result.traits = self._extract_traits(tree)
        result.impls = self._extract_impls(tree)
        result.imports = self._extract_imports(tree)
        result.uses = self._extract_uses(tree)
        
        # Cálculo de métricas
        result.metrics = self._calculate_metrics(tree, content, result)
        
        return result
    
    def _analyze_scopes(self, tree: Any) -> List[RustScope]:
        """Analiza los scopes en el AST."""
        scopes = []
        # Implementación básica - se expandirá
        return scopes
    
    def _extract_functions(self, tree: Any) -> List[RustFunctionDefinition]:
        """Extrae definiciones de funciones."""
        functions = []
        # Implementación básica - se expandirá
        return functions
    
    def _extract_structs(self, tree: Any) -> List[RustStructDefinition]:
        """Extrae definiciones de structs."""
        structs = []
        # Implementación básica - se expandirá
        return structs
    
    def _extract_enums(self, tree: Any) -> List[RustEnumDefinition]:
        """Extrae definiciones de enums."""
        enums = []
        # Implementación básica - se expandirá
        return enums
    
    def _extract_traits(self, tree: Any) -> List[RustTraitDefinition]:
        """Extrae definiciones de traits."""
        traits = []
        # Implementación básica - se expandirá
        return traits
    
    def _extract_impls(self, tree: Any) -> List[RustImplBlock]:
        """Extrae bloques de implementación."""
        impls = []
        # Implementación básica - se expandirá
        return impls
    
    def _extract_imports(self, tree: Any) -> List[RustImportStatement]:
        """Extrae declaraciones de import."""
        imports = []
        # Implementación básica - se expandirá
        return imports
    
    def _extract_uses(self, tree: Any) -> List[RustUseStatement]:
        """Extrae declaraciones de use."""
        uses = []
        # Implementación básica - se expandirá
        return uses
    
    def _calculate_metrics(self, tree: Any, content: str, result: RustAnalysisResult) -> RustMetrics:
        """Calcula métricas del código Rust."""
        metrics = RustMetrics()
        
        # Líneas de código
        lines = content.split('\n')
        metrics.lines_of_code = len(lines)
        metrics.logical_lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('//')])
        metrics.comment_lines = len([line for line in lines if line.strip().startswith('//') or line.strip().startswith('/*')])
        metrics.blank_lines = len([line for line in lines if not line.strip()])
        
        # Conteos básicos
        metrics.function_count = len(result.functions)
        metrics.struct_count = len(result.structs)
        metrics.enum_count = len(result.enums)
        metrics.trait_count = len(result.traits)
        metrics.impl_count = len(result.impls)
        metrics.macro_count = len(result.macro_info.macro_definitions) if result.macro_info else 0
        
        # Complejidad ciclomática básica
        metrics.cyclomatic_complexity = 1  # Base complexity
        
        return metrics


class RustPatternDetector:
    """Detector de patrones específicos de Rust."""
    
    def __init__(self):
        self.patterns = {
            'unnecessary_clone': self._detect_unnecessary_clone,
            'unwrap_usage': self._detect_unwrap_usage,
            'unused_variables': self._detect_unused_variables,
            'unsafe_usage': self._detect_unsafe_usage,
            'raw_pointer_usage': self._detect_raw_pointer_usage,
            'lifetime_elision': self._detect_lifetime_elision,
            'ownership_violations': self._detect_ownership_violations,
            'iterator_chain_optimization': self._detect_iterator_chain_optimization,
        }
    
    def detect_patterns(self, result: RustAnalysisResult) -> List[RustPattern]:
        """Detecta patrones en el código."""
        patterns = []
        
        for pattern_name, pattern_func in self.patterns.items():
            try:
                pattern_results = pattern_func(result)
                patterns.extend(pattern_results)
            except Exception as e:
                logger.warning(f"Error detecting pattern {pattern_name}: {e}")
        
        return patterns
    
    def _detect_unnecessary_clone(self, result: RustAnalysisResult) -> List[RustPattern]:
        """Detecta uso innecesario de .clone()."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_unwrap_usage(self, result: RustAnalysisResult) -> List[RustPattern]:
        """Detecta uso de .unwrap() que puede causar pánicos."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_unused_variables(self, result: RustAnalysisResult) -> List[RustPattern]:
        """Detecta variables no utilizadas."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_unsafe_usage(self, result: RustAnalysisResult) -> List[RustPattern]:
        """Detecta uso de código unsafe."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_raw_pointer_usage(self, result: RustAnalysisResult) -> List[RustPattern]:
        """Detecta uso de punteros raw."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_lifetime_elision(self, result: RustAnalysisResult) -> List[RustPattern]:
        """Detecta elisión de lifetimes."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_ownership_violations(self, result: RustAnalysisResult) -> List[RustPattern]:
        """Detecta violaciones de ownership."""
        patterns = []
        # Implementación básica
        return patterns
    
    def _detect_iterator_chain_optimization(self, result: RustAnalysisResult) -> List[RustPattern]:
        """Detecta optimizaciones en cadenas de iteradores."""
        patterns = []
        # Implementación básica
        return patterns


class RustSpecializedParser(BaseParser):
    """Parser especializado para Rust."""
    
    def __init__(self, config: Optional[RustParserConfig] = None):
        self.config = config or RustParserConfig()
        self.semantic_analyzer = RustSemanticAnalyzer()
        self.pattern_detector = RustPatternDetector()
        self._setup_tree_sitter()
    
    def _setup_tree_sitter(self):
        """Configura tree-sitter para Rust."""
        try:
            # Intentar cargar los lenguajes tree-sitter
            # Por ahora, usamos un método de fallback
            logger.info("Using fallback parsing method for Rust")
        except Exception as e:
            logger.warning(f"Could not setup tree-sitter: {e}")
            logger.info("Using fallback parsing method")
    
    async def parse_file(self, file_path: Path) -> ParseResult:
        """Parsea un archivo Rust."""
        start_time = time.time()
        
        try:
            # Leer contenido del archivo
            content = file_path.read_text(encoding='utf-8')
            
            # Parsear contenido
            result = await self.parse_content(content, ProgrammingLanguage.RUST)
            result.file_path = file_path
            
            return result
            
        except Exception as e:
            error_msg = f"Error parsing Rust file {file_path}: {e}"
            logger.error(error_msg)
            return ParseResult(
                language=ProgrammingLanguage.RUST,
                file_path=file_path,
                ast=None,
                errors=[error_msg],
                parse_time_ms=int((time.time() - start_time) * 1000),
                metadata={}
            )
    
    async def parse_content(self, content: str, language: ProgrammingLanguage) -> ParseResult:
        """Parsea contenido Rust."""
        start_time = time.time()
        
        try:
            # Parsear con tree-sitter o método alternativo
            ast = self._parse_with_tree_sitter(content)
            
            # Análisis semántico
            analysis_result = self.semantic_analyzer.analyze(ast, content)
            
            # Análisis de ownership
            if self.config.enable_ownership_analysis:
                analysis_result.ownership_info = self._analyze_ownership(ast, content)
            
            # Análisis de lifetimes
            if self.config.enable_lifetime_analysis:
                analysis_result.lifetime_info = self._analyze_lifetimes(ast, content)
            
            # Análisis de traits
            if self.config.enable_trait_analysis:
                analysis_result.trait_info = self._analyze_traits(ast, content)
            
            # Análisis de código unsafe
            if self.config.enable_unsafe_analysis:
                analysis_result.unsafe_info = self._analyze_unsafe(ast, content)
            
            # Análisis de Cargo
            if self.config.enable_cargo_analysis:
                analysis_result.cargo_info = self._analyze_cargo(content)
            
            # Análisis de macros
            if self.config.enable_macro_analysis:
                analysis_result.macro_info = self._analyze_macros(ast, content)
            
            # Detección de patrones
            analysis_result.patterns = self.pattern_detector.detect_patterns(analysis_result)
            
            # Crear resultado del parser universal
            result = ParseResult(
                language=ProgrammingLanguage.RUST,
                file_path=Path("<string>"),
                ast=ast,
                errors=analysis_result.errors,
                parse_time_ms=int((time.time() - start_time) * 1000),
                metadata={
                    'rust_analysis': analysis_result,
                }
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error parsing Rust content: {e}"
            logger.error(error_msg)
            return ParseResult(
                language=ProgrammingLanguage.RUST,
                file_path=Path("<string>"),
                ast=None,
                errors=[error_msg],
                parse_time_ms=int((time.time() - start_time) * 1000),
                metadata={}
            )
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Retorna los lenguajes soportados."""
        return [ProgrammingLanguage.RUST]
    
    def _parse_with_tree_sitter(self, content: str) -> Any:
        """Parsea usando tree-sitter."""
        try:
            # Implementación básica - se expandirá con tree-sitter real
            # Por ahora, retornamos un objeto simple que representa el AST
            return {
                'type': 'program',
                'language': 'rust',
                'content': content,
                'lines': len(content.split('\n')),
            }
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed: {e}")
            # Fallback a parsing básico
            return self._fallback_parse(content)
    
    def _fallback_parse(self, content: str) -> Any:
        """Método de fallback para parsing básico."""
        return {
            'type': 'program',
            'language': 'rust',
            'content': content,
            'lines': len(content.split('\n')),
            'fallback': True,
        }
    
    def _analyze_ownership(self, ast: Any, content: str) -> RustOwnershipInfo:
        """Analiza ownership y borrowing."""
        return RustOwnershipInfo()
    
    def _analyze_lifetimes(self, ast: Any, content: str) -> RustLifetimeInfo:
        """Analiza lifetimes."""
        return RustLifetimeInfo()
    
    def _analyze_traits(self, ast: Any, content: str) -> RustTraitInfo:
        """Analiza el sistema de traits."""
        return RustTraitInfo()
    
    def _analyze_unsafe(self, ast: Any, content: str) -> RustUnsafeInfo:
        """Analiza código unsafe."""
        return RustUnsafeInfo()
    
    def _analyze_cargo(self, content: str) -> RustCargoInfo:
        """Analiza información de Cargo."""
        return RustCargoInfo()
    
    def _analyze_macros(self, ast: Any, content: str) -> RustMacroInfo:
        """Analiza macros."""
        return RustMacroInfo()


# Funciones de utilidad
async def analyze_rust_file(file_path: Path, config: Optional[RustParserConfig] = None) -> RustAnalysisResult:
    """Analiza un archivo Rust."""
    parser = RustSpecializedParser(config)
    result = await parser.parse_content(file_path.read_text(encoding='utf-8'), ProgrammingLanguage.RUST)
    return result.metadata['rust_analysis']


async def analyze_rust_content(content: str, config: Optional[RustParserConfig] = None) -> RustAnalysisResult:
    """Analiza contenido Rust."""
    parser = RustSpecializedParser(config)
    result = await parser.parse_content(content, ProgrammingLanguage.RUST)
    return result.metadata['rust_analysis']


__all__ = [
    'RustEdition',
    'RustParserConfig',
    'RustScope',
    'RustType',
    'RustFunctionDefinition',
    'RustStructDefinition',
    'RustEnumDefinition',
    'RustTraitDefinition',
    'RustImplBlock',
    'RustImportStatement',
    'RustUseStatement',
    'RustOwnershipInfo',
    'RustLifetimeInfo',
    'RustTraitInfo',
    'RustUnsafeInfo',
    'RustCargoInfo',
    'RustMacroInfo',
    'RustPattern',
    'RustMetrics',
    'RustAnalysisResult',
    'RustSemanticAnalyzer',
    'RustPatternDetector',
    'RustSpecializedParser',
    'analyze_rust_file',
    'analyze_rust_content',
]
