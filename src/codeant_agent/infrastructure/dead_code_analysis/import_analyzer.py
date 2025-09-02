"""
Implementación del analizador de imports.

Este módulo implementa la detección de imports no utilizados
en diferentes lenguajes de programación.
"""

import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
import re

from ...domain.entities.dead_code_analysis import (
    UnusedImport, ImportStatement, ImportType, UnusedReason,
    SourceRange, SourcePosition
)
from ...domain.entities.parse_result import ParseResult
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class ImportUsage:
    """Información de uso de un import."""
    import_name: str
    symbol_name: str
    usage_locations: List[SourceRange]
    usage_count: int
    is_side_effect_only: bool = False


@dataclass
class ImportAnalysisResult:
    """Resultado del análisis de imports."""
    unused_imports: List[UnusedImport]
    partially_unused_imports: List[UnusedImport]
    side_effect_imports: List[ImportStatement]
    analysis_time_ms: int


class PythonImportAnalyzer:
    """Analizador de imports específico para Python."""
    
    # Módulos conocidos que tienen efectos secundarios
    SIDE_EFFECT_MODULES = {
        'matplotlib.pyplot',
        'seaborn',
        'django.setup',
        'logging.config',
        'warnings',
        'antigravity',
        'this',
        'turtle',
    }
    
    # Patrones de imports con efectos secundarios
    SIDE_EFFECT_PATTERNS = [
        r'.*__init__.*',
        r'.*setup.*',
        r'.*config.*',
        r'.*monkey.*patch.*',
    ]
    
    def analyze_python_imports(self, parse_result: ParseResult) -> ImportAnalysisResult:
        """
        Analiza imports en código Python.
        
        Args:
            parse_result: Resultado del parsing
            
        Returns:
            ImportAnalysisResult con imports no utilizados
        """
        import time
        start_time = time.time()
        
        try:
            # Extraer imports del AST
            imports = self._extract_imports_from_ast(parse_result.tree.root_node)
            
            # Analizar uso de cada import
            usage_analysis = self._analyze_import_usage(parse_result, imports)
            
            # Clasificar imports
            unused_imports = []
            partially_unused_imports = []
            side_effect_imports = []
            
            for import_stmt in imports:
                if self._is_side_effect_import(import_stmt):
                    side_effect_imports.append(import_stmt)
                else:
                    usage = usage_analysis.get(import_stmt.module_name, [])
                    if not usage:
                        # Import completamente no utilizado
                        unused_import = self._create_unused_import(
                            import_stmt, UnusedReason.NEVER_REFERENCED, 0.9
                        )
                        unused_imports.append(unused_import)
                    else:
                        # Verificar si algunos símbolos no son utilizados
                        unused_symbols = self._find_unused_symbols(import_stmt, usage)
                        if unused_symbols:
                            unused_import = self._create_partially_unused_import(
                                import_stmt, unused_symbols
                            )
                            partially_unused_imports.append(unused_import)
            
            analysis_time = int((time.time() - start_time) * 1000)
            
            return ImportAnalysisResult(
                unused_imports=unused_imports,
                partially_unused_imports=partially_unused_imports,
                side_effect_imports=side_effect_imports,
                analysis_time_ms=analysis_time
            )
            
        except Exception as e:
            logger.error(f"Error analizando imports de Python: {e}")
            raise
    
    def _extract_imports_from_ast(self, ast_node: Any) -> List[ImportStatement]:
        """Extrae imports del AST de Python."""
        imports = []
        self._find_imports_recursive(ast_node, imports)
        return imports
    
    def _find_imports_recursive(self, ast_node: Any, imports: List[ImportStatement]) -> None:
        """Busca imports recursivamente en el AST."""
        node_type = ast_node.type
        
        if node_type == 'import_statement':
            import_stmt = self._parse_import_statement(ast_node)
            if import_stmt:
                imports.append(import_stmt)
        elif node_type == 'import_from_statement':
            import_stmt = self._parse_from_import_statement(ast_node)
            if import_stmt:
                imports.append(import_stmt)
        
        # Procesar hijos
        for child in ast_node.children:
            self._find_imports_recursive(child, imports)
    
    def _parse_import_statement(self, ast_node: Any) -> Optional[ImportStatement]:
        """Parsea una declaración 'import'."""
        try:
            # Extraer el nombre del módulo
            module_name = ""
            alias = None
            
            for child in ast_node.children:
                if child.type == 'dotted_name':
                    module_name = child.text.decode('utf-8')
                elif child.type == 'aliased_import':
                    # Manejar 'import module as alias'
                    name_child = child.children[0] if child.children else None
                    if name_child:
                        module_name = name_child.text.decode('utf-8')
                    
                    # Buscar alias
                    for subchild in child.children:
                        if subchild.type == 'identifier' and subchild != name_child:
                            alias = subchild.text.decode('utf-8')
                            break
            
            if module_name:
                return ImportStatement(
                    module_name=module_name,
                    imported_symbols=[alias or module_name.split('.')[-1]],
                    import_type=ImportType.DEFAULT_IMPORT,
                    location=self._extract_source_range(ast_node),
                    language=ProgrammingLanguage.PYTHON,
                    alias=alias
                )
        except Exception as e:
            logger.warning(f"Error parseando import statement: {e}")
        
        return None
    
    def _parse_from_import_statement(self, ast_node: Any) -> Optional[ImportStatement]:
        """Parsea una declaración 'from ... import ...'."""
        try:
            module_name = ""
            imported_symbols = []
            
            for child in ast_node.children:
                if child.type == 'dotted_name':
                    module_name = child.text.decode('utf-8')
                elif child.type == 'import_list':
                    # Extraer símbolos importados
                    for symbol_child in child.children:
                        if symbol_child.type == 'identifier':
                            symbol_name = symbol_child.text.decode('utf-8')
                            imported_symbols.append(symbol_name)
                        elif symbol_child.type == 'aliased_import':
                            # Manejar 'from module import symbol as alias'
                            name_child = symbol_child.children[0] if symbol_child.children else None
                            if name_child:
                                symbol_name = name_child.text.decode('utf-8')
                                imported_symbols.append(symbol_name)
                elif child.type == 'wildcard_import':
                    # from module import *
                    imported_symbols = ['*']
            
            if module_name:
                import_type = ImportType.NAMED_IMPORTS
                if '*' in imported_symbols:
                    import_type = ImportType.NAMESPACE_IMPORT
                
                return ImportStatement(
                    module_name=module_name,
                    imported_symbols=imported_symbols,
                    import_type=import_type,
                    location=self._extract_source_range(ast_node),
                    language=ProgrammingLanguage.PYTHON
                )
        except Exception as e:
            logger.warning(f"Error parseando from import statement: {e}")
        
        return None
    
    def _analyze_import_usage(
        self, 
        parse_result: ParseResult, 
        imports: List[ImportStatement]
    ) -> Dict[str, List[ImportUsage]]:
        """Analiza el uso de cada import en el código."""
        usage_analysis = {}
        
        # Obtener el texto completo del archivo
        file_content = ""
        if hasattr(parse_result.tree.root_node, 'text'):
            file_content = parse_result.tree.root_node.text.decode('utf-8')
        
        for import_stmt in imports:
            usages = []
            
            # Para cada símbolo importado, buscar su uso
            for symbol in import_stmt.imported_symbols:
                if symbol == '*':
                    # Wildcard imports - difícil de analizar, asumir usado
                    usage = ImportUsage(
                        import_name=import_stmt.module_name,
                        symbol_name=symbol,
                        usage_locations=[],
                        usage_count=1,  # Asumir usado
                        is_side_effect_only=False
                    )
                    usages.append(usage)
                else:
                    usage_count, locations = self._find_symbol_usage(
                        symbol, file_content, parse_result.tree.root_node
                    )
                    
                    usage = ImportUsage(
                        import_name=import_stmt.module_name,
                        symbol_name=symbol,
                        usage_locations=locations,
                        usage_count=usage_count,
                        is_side_effect_only=usage_count == 0
                    )
                    usages.append(usage)
            
            usage_analysis[import_stmt.module_name] = usages
        
        return usage_analysis
    
    def _find_symbol_usage(
        self, 
        symbol: str, 
        file_content: str, 
        ast_root: Any
    ) -> Tuple[int, List[SourceRange]]:
        """Encuentra el uso de un símbolo en el código."""
        usage_count = 0
        usage_locations = []
        
        # Buscar usando regex como primera aproximación
        # Nota: esto es simplificado, una implementación real usaría el AST
        pattern = rf'\b{re.escape(symbol)}\b'
        matches = list(re.finditer(pattern, file_content))
        
        # Filtrar matches que están en declaraciones de import
        for match in matches:
            # Verificar que no esté en una línea de import
            start_pos = match.start()
            line_start = file_content.rfind('\n', 0, start_pos) + 1
            line_end = file_content.find('\n', start_pos)
            if line_end == -1:
                line_end = len(file_content)
            
            line_content = file_content[line_start:line_end].strip()
            
            # Si la línea no es un import, contar como uso
            if not (line_content.startswith('import ') or line_content.startswith('from ')):
                usage_count += 1
                
                # Calcular posición
                lines_before = file_content[:start_pos].count('\n')
                line_start_pos = file_content.rfind('\n', 0, start_pos) + 1
                column = start_pos - line_start_pos
                
                location = SourceRange(
                    start=SourcePosition(line=lines_before + 1, column=column),
                    end=SourcePosition(line=lines_before + 1, column=column + len(symbol))
                )
                usage_locations.append(location)
        
        return usage_count, usage_locations
    
    def _is_side_effect_import(self, import_stmt: ImportStatement) -> bool:
        """Verifica si un import tiene efectos secundarios."""
        module_name = import_stmt.module_name
        
        # Verificar módulos conocidos
        if module_name in self.SIDE_EFFECT_MODULES:
            return True
        
        # Verificar patrones
        for pattern in self.SIDE_EFFECT_PATTERNS:
            if re.match(pattern, module_name, re.IGNORECASE):
                return True
        
        return False
    
    def _create_unused_import(
        self, 
        import_stmt: ImportStatement, 
        reason: UnusedReason,
        confidence: float
    ) -> UnusedImport:
        """Crea un UnusedImport."""
        suggestion = f"Eliminar import no utilizado: {import_stmt.module_name}"
        
        return UnusedImport(
            import_statement=import_stmt,
            location=import_stmt.location,
            import_type=import_stmt.import_type,
            module_name=import_stmt.module_name,
            imported_symbols=import_stmt.imported_symbols,
            reason=reason,
            suggestion=suggestion,
            confidence=confidence,
            side_effects_possible=self._is_side_effect_import(import_stmt)
        )
    
    def _create_partially_unused_import(
        self, 
        import_stmt: ImportStatement,
        unused_symbols: List[str]
    ) -> UnusedImport:
        """Crea un UnusedImport para símbolos parcialmente no utilizados."""
        suggestion = f"Eliminar símbolos no utilizados: {', '.join(unused_symbols)}"
        
        return UnusedImport(
            import_statement=import_stmt,
            location=import_stmt.location,
            import_type=ImportType.PARTIALLY_UNUSED,
            module_name=import_stmt.module_name,
            imported_symbols=unused_symbols,
            reason=UnusedReason.PARTIALLY_UNUSED,
            suggestion=suggestion,
            confidence=0.95,
            side_effects_possible=False
        )
    
    def _find_unused_symbols(
        self, 
        import_stmt: ImportStatement, 
        usages: List[ImportUsage]
    ) -> List[str]:
        """Encuentra símbolos no utilizados en un import."""
        unused_symbols = []
        
        for usage in usages:
            if usage.usage_count == 0 and not usage.is_side_effect_only:
                unused_symbols.append(usage.symbol_name)
        
        return unused_symbols
    
    def _extract_source_range(self, ast_node: Any) -> SourceRange:
        """Extrae el rango de código fuente de un nodo AST."""
        start_pos = SourcePosition(
            line=ast_node.start_point[0] + 1,
            column=ast_node.start_point[1],
            offset=ast_node.start_byte
        )
        end_pos = SourcePosition(
            line=ast_node.end_point[0] + 1,
            column=ast_node.end_point[1],
            offset=ast_node.end_byte
        )
        
        return SourceRange(start=start_pos, end=end_pos)


class JavaScriptImportAnalyzer:
    """Analizador de imports específico para JavaScript/TypeScript."""
    
    # Extensiones que indican imports con efectos secundarios
    SIDE_EFFECT_EXTENSIONS = {'.css', '.scss', '.less', '.sass'}
    
    # Módulos conocidos con efectos secundarios
    SIDE_EFFECT_MODULES = {
        'babel-polyfill',
        'core-js',
        'regenerator-runtime',
        'zone.js',
        'reflect-metadata',
    }
    
    def analyze_js_imports(self, parse_result: ParseResult) -> ImportAnalysisResult:
        """
        Analiza imports en código JavaScript/TypeScript.
        
        Args:
            parse_result: Resultado del parsing
            
        Returns:
            ImportAnalysisResult con imports no utilizados
        """
        import time
        start_time = time.time()
        
        try:
            # Similar a Python pero adaptado para JS/TS
            imports = self._extract_js_imports_from_ast(parse_result.tree.root_node)
            usage_analysis = self._analyze_js_import_usage(parse_result, imports)
            
            unused_imports = []
            partially_unused_imports = []
            side_effect_imports = []
            
            for import_stmt in imports:
                if self._is_js_side_effect_import(import_stmt):
                    side_effect_imports.append(import_stmt)
                else:
                    # Análisis similar al de Python
                    usage = usage_analysis.get(import_stmt.module_name, [])
                    if not usage:
                        unused_import = self._create_unused_import(
                            import_stmt, UnusedReason.NEVER_REFERENCED, 0.9
                        )
                        unused_imports.append(unused_import)
            
            analysis_time = int((time.time() - start_time) * 1000)
            
            return ImportAnalysisResult(
                unused_imports=unused_imports,
                partially_unused_imports=partially_unused_imports,
                side_effect_imports=side_effect_imports,
                analysis_time_ms=analysis_time
            )
            
        except Exception as e:
            logger.error(f"Error analizando imports de JavaScript: {e}")
            raise
    
    def _extract_js_imports_from_ast(self, ast_node: Any) -> List[ImportStatement]:
        """Extrae imports del AST de JavaScript/TypeScript."""
        imports = []
        self._find_js_imports_recursive(ast_node, imports)
        return imports
    
    def _find_js_imports_recursive(self, ast_node: Any, imports: List[ImportStatement]) -> None:
        """Busca imports recursivamente en el AST de JS/TS."""
        node_type = ast_node.type
        
        if node_type == 'import_statement':
            import_stmt = self._parse_js_import_statement(ast_node)
            if import_stmt:
                imports.append(import_stmt)
        elif node_type in ['call_expression'] and self._is_require_call(ast_node):
            # Manejar require() en Node.js
            import_stmt = self._parse_require_statement(ast_node)
            if import_stmt:
                imports.append(import_stmt)
        
        # Procesar hijos
        for child in ast_node.children:
            self._find_js_imports_recursive(child, imports)
    
    def _parse_js_import_statement(self, ast_node: Any) -> Optional[ImportStatement]:
        """Parsea una declaración import de ES6."""
        # Implementación simplificada
        return None
    
    def _is_require_call(self, ast_node: Any) -> bool:
        """Verifica si es una llamada a require()."""
        # Implementación simplificada
        return False
    
    def _parse_require_statement(self, ast_node: Any) -> Optional[ImportStatement]:
        """Parsea una declaración require()."""
        # Implementación simplificada
        return None
    
    def _analyze_js_import_usage(
        self, 
        parse_result: ParseResult, 
        imports: List[ImportStatement]
    ) -> Dict[str, List[ImportUsage]]:
        """Analiza el uso de imports en código JS/TS."""
        # Implementación similar a Python pero adaptada
        return {}
    
    def _is_js_side_effect_import(self, import_stmt: ImportStatement) -> bool:
        """Verifica si un import JS tiene efectos secundarios."""
        module_name = import_stmt.module_name
        
        # Verificar extensiones
        for ext in self.SIDE_EFFECT_EXTENSIONS:
            if module_name.endswith(ext):
                return True
        
        # Verificar módulos conocidos
        return module_name in self.SIDE_EFFECT_MODULES
    
    def _create_unused_import(
        self, 
        import_stmt: ImportStatement, 
        reason: UnusedReason,
        confidence: float
    ) -> UnusedImport:
        """Crea un UnusedImport para JS/TS."""
        suggestion = f"Eliminar import no utilizado: {import_stmt.module_name}"
        
        return UnusedImport(
            import_statement=import_stmt,
            location=import_stmt.location,
            import_type=import_stmt.import_type,
            module_name=import_stmt.module_name,
            imported_symbols=import_stmt.imported_symbols,
            reason=reason,
            suggestion=suggestion,
            confidence=confidence,
            side_effects_possible=self._is_js_side_effect_import(import_stmt)
        )


class ImportAnalyzer:
    """Analizador principal de imports."""
    
    def __init__(self):
        self.python_analyzer = PythonImportAnalyzer()
        self.js_analyzer = JavaScriptImportAnalyzer()
    
    async def analyze_imports(
        self, 
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> ImportAnalysisResult:
        """
        Analiza imports según el lenguaje de programación.
        
        Args:
            parse_result: Resultado del parsing
            config: Configuración opcional
            
        Returns:
            ImportAnalysisResult con imports no utilizados
        """
        language = parse_result.language
        
        if language == ProgrammingLanguage.PYTHON:
            return self.python_analyzer.analyze_python_imports(parse_result)
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            return self.js_analyzer.analyze_js_imports(parse_result)
        else:
            # Para otros lenguajes, retornar resultado vacío por ahora
            return ImportAnalysisResult(
                unused_imports=[],
                partially_unused_imports=[],
                side_effect_imports=[],
                analysis_time_ms=0
            )
