"""
Sistema de preprocessamiento de código multi-lenguaje para IA.

Este módulo implementa el preprocessamiento específico por lenguaje
para optimizar la generación de embeddings y análisis de IA.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


class PreprocessingLevel(Enum):
    """Niveles de preprocessamiento."""
    LIGHT = "light"      # Normalización básica
    NORMAL = "normal"    # Preprocessamiento estándar  
    AGGRESSIVE = "aggressive"  # Preprocessamiento intensivo para similarity


@dataclass
class PreprocessingConfig:
    """Configuración de preprocessamiento."""
    level: PreprocessingLevel = PreprocessingLevel.NORMAL
    normalize_whitespace: bool = True
    normalize_identifiers: bool = False
    remove_comments: bool = True
    remove_docstrings: bool = False
    normalize_strings: bool = True
    normalize_numbers: bool = True
    preserve_structure: bool = True
    language_specific_rules: bool = True


@dataclass
class PreprocessingResult:
    """Resultado del preprocessamiento."""
    original_code: str
    processed_code: str
    language: ProgrammingLanguage
    transformations_applied: List[str] = field(default_factory=list)
    preserved_elements: Dict[str, str] = field(default_factory=dict)
    statistics: Dict[str, int] = field(default_factory=dict)
    
    def get_compression_ratio(self) -> float:
        """Obtiene ratio de compresión."""
        if len(self.original_code) == 0:
            return 0.0
        return len(self.processed_code) / len(self.original_code)


class LanguagePreprocessor(ABC):
    """Clase base para preprocessadores específicos de lenguaje."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.keywords = self.get_language_keywords()
        self.operators = self.get_language_operators()
        self.built_ins = self.get_language_built_ins()
    
    @abstractmethod
    async def preprocess(self, code: str) -> PreprocessingResult:
        """Preprocessa código del lenguaje específico."""
        pass
    
    @abstractmethod
    def get_language(self) -> ProgrammingLanguage:
        """Retorna el lenguaje que maneja este preprocessor."""
        pass
    
    @abstractmethod
    def get_language_keywords(self) -> Set[str]:
        """Retorna keywords del lenguaje."""
        pass
    
    @abstractmethod
    def get_language_operators(self) -> Set[str]:
        """Retorna operadores del lenguaje."""
        pass
    
    @abstractmethod
    def get_language_built_ins(self) -> Set[str]:
        """Retorna built-ins del lenguaje."""
        pass
    
    def normalize_whitespace(self, code: str) -> str:
        """Normaliza espacios en blanco."""
        # Normalizar saltos de línea
        code = re.sub(r'\r\n|\r', '\n', code)
        
        # Normalizar tabs a espacios
        code = re.sub(r'\t', '    ', code)
        
        # Remover espacios al final de líneas
        code = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)
        
        # Limitar líneas vacías consecutivas
        code = re.sub(r'\n{3,}', '\n\n', code)
        
        return code.strip()
    
    def normalize_string_literals(self, code: str) -> str:
        """Normaliza literales de string."""
        if not self.config.normalize_strings:
            return code
        
        # Normalizar strings simples (implementación básica)
        code = re.sub(r'"[^"]*"', '"STRING"', code)
        code = re.sub(r"'[^']*'", '"STRING"', code)
        
        return code
    
    def normalize_numeric_literals(self, code: str) -> str:
        """Normaliza literales numéricos."""
        if not self.config.normalize_numbers:
            return code
        
        # Normalizar números
        code = re.sub(r'\b\d+\.?\d*\b', 'NUM', code)
        code = re.sub(r'\b0x[0-9a-fA-F]+\b', 'HEX', code)
        code = re.sub(r'\b\d+e[+-]?\d+\b', 'SCI', code)
        
        return code
    
    def extract_semantic_structure(self, code: str) -> str:
        """Extrae estructura semántica básica."""
        if not self.config.preserve_structure:
            return code
        
        # Implementación básica - mantener estructura de control
        structure_patterns = [
            (r'\bif\b', 'IF'),
            (r'\belse\b', 'ELSE'),
            (r'\bfor\b', 'FOR'),
            (r'\bwhile\b', 'WHILE'),
            (r'\btry\b', 'TRY'),
            (r'\bcatch\b', 'CATCH'),
            (r'\bfinally\b', 'FINALLY'),
        ]
        
        structured_code = code
        for pattern, replacement in structure_patterns:
            structured_code = re.sub(pattern, replacement, structured_code)
        
        return structured_code


class PythonPreprocessor(LanguagePreprocessor):
    """Preprocessador específico para Python."""
    
    def get_language(self) -> ProgrammingLanguage:
        return ProgrammingLanguage.PYTHON
    
    def get_language_keywords(self) -> Set[str]:
        return {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
            'break', 'class', 'continue', 'def', 'del', 'elif', 'else',
            'except', 'finally', 'for', 'from', 'global', 'if', 'import',
            'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise',
            'return', 'try', 'while', 'with', 'yield'
        }
    
    def get_language_operators(self) -> Set[str]:
        return {'+', '-', '*', '/', '//', '%', '**', '=', '+=', '-=', '*=', '/=',
                '==', '!=', '<', '>', '<=', '>=', '&', '|', '^', '~', '<<', '>>'}
    
    def get_language_built_ins(self) -> Set[str]:
        return {
            'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes', 'callable',
            'chr', 'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir',
            'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format',
            'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex',
            'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len',
            'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object',
            'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr',
            'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod',
            'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'
        }
    
    async def preprocess(self, code: str) -> PreprocessingResult:
        """Preprocessa código Python."""
        result = PreprocessingResult(
            original_code=code,
            processed_code=code,
            language=self.get_language()
        )
        
        processed = code
        
        # Normalización básica
        if self.config.normalize_whitespace:
            processed = self.normalize_whitespace(processed)
            result.transformations_applied.append("normalize_whitespace")
        
        # Remover comentarios
        if self.config.remove_comments:
            processed = self.remove_python_comments(processed)
            result.transformations_applied.append("remove_comments")
        
        # Remover docstrings
        if self.config.remove_docstrings:
            processed = self.remove_python_docstrings(processed)
            result.transformations_applied.append("remove_docstrings")
        
        # Normalizar strings
        if self.config.normalize_strings:
            processed = self.normalize_python_strings(processed)
            result.transformations_applied.append("normalize_strings")
        
        # Normalizar números
        if self.config.normalize_numbers:
            processed = self.normalize_numeric_literals(processed)
            result.transformations_applied.append("normalize_numbers")
        
        # Normalizar identifiers (solo en modo agresivo)
        if (self.config.normalize_identifiers and 
            self.config.level == PreprocessingLevel.AGGRESSIVE):
            processed = self.normalize_python_identifiers(processed)
            result.transformations_applied.append("normalize_identifiers")
        
        # Normalizar indentación Python
        processed = self.normalize_python_indentation(processed)
        result.transformations_applied.append("normalize_indentation")
        
        # Remover imports (en modo agresivo)
        if self.config.level == PreprocessingLevel.AGGRESSIVE:
            processed = self.remove_python_imports(processed)
            result.transformations_applied.append("remove_imports")
        
        # Calcular estadísticas
        result.statistics = {
            "original_lines": len(code.splitlines()),
            "processed_lines": len(processed.splitlines()),
            "original_chars": len(code),
            "processed_chars": len(processed),
            "transformations_count": len(result.transformations_applied)
        }
        
        result.processed_code = processed
        return result
    
    def remove_python_comments(self, code: str) -> str:
        """Remover comentarios Python."""
        # Remover comentarios de línea
        lines = []
        for line in code.splitlines():
            # Buscar # que no esté dentro de string
            in_string = False
            quote_char = None
            comment_pos = -1
            
            for i, char in enumerate(line):
                if not in_string:
                    if char in ['"', "'"]:
                        in_string = True
                        quote_char = char
                    elif char == '#':
                        comment_pos = i
                        break
                else:
                    if char == quote_char and (i == 0 or line[i-1] != '\\'):
                        in_string = False
                        quote_char = None
            
            if comment_pos >= 0:
                line = line[:comment_pos].rstrip()
            
            lines.append(line)
        
        return '\n'.join(lines)
    
    def remove_python_docstrings(self, code: str) -> str:
        """Remover docstrings Python."""
        # Remover docstrings triple-quoted
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        
        return code
    
    def normalize_python_strings(self, code: str) -> str:
        """Normalizar strings Python."""
        # f-strings
        code = re.sub(r'f"[^"]*"', '"FSTRING"', code)
        code = re.sub(r"f'[^']*'", '"FSTRING"', code)
        
        # Raw strings
        code = re.sub(r'r"[^"]*"', '"RAWSTRING"', code)
        code = re.sub(r"r'[^']*'", '"RAWSTRING"', code)
        
        # Strings normales
        return self.normalize_string_literals(code)
    
    def normalize_python_identifiers(self, code: str) -> str:
        """Normalizar identificadores Python."""
        identifier_map = {}
        counter = 0
        
        def replace_identifier(match):
            nonlocal counter
            identifier = match.group(0)
            
            # No reemplazar keywords, built-ins, o nombres especiales
            if (identifier in self.keywords or 
                identifier in self.built_ins or
                identifier.startswith('__') or
                identifier.startswith('_')):
                return identifier
            
            if identifier not in identifier_map:
                counter += 1
                identifier_map[identifier] = f'var_{counter}'
            
            return identifier_map[identifier]
        
        # Buscar identificadores Python
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        return re.sub(pattern, replace_identifier, code)
    
    def normalize_python_indentation(self, code: str) -> str:
        """Normalizar indentación Python."""
        lines = []
        for line in code.splitlines():
            if line.strip():
                # Calcular nivel de indentación
                leading_spaces = len(line) - len(line.lstrip())
                indent_level = leading_spaces // 4
                
                # Normalizar a 4 espacios por nivel
                normalized_line = '    ' * indent_level + line.lstrip()
                lines.append(normalized_line)
            else:
                lines.append('')
        
        return '\n'.join(lines)
    
    def remove_python_imports(self, code: str) -> str:
        """Remover imports Python."""
        # Remover imports simples y from imports
        code = re.sub(r'^import\s+.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^from\s+.*import\s+.*$', '', code, flags=re.MULTILINE)
        
        return code


class JavaScriptPreprocessor(LanguagePreprocessor):
    """Preprocessador específico para JavaScript."""
    
    def get_language(self) -> ProgrammingLanguage:
        return ProgrammingLanguage.JAVASCRIPT
    
    def get_language_keywords(self) -> Set[str]:
        return {
            'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger',
            'default', 'delete', 'do', 'else', 'export', 'extends', 'finally',
            'for', 'function', 'if', 'import', 'in', 'instanceof', 'let', 'new',
            'return', 'super', 'switch', 'this', 'throw', 'try', 'typeof', 'var',
            'void', 'while', 'with', 'yield', 'async', 'await'
        }
    
    def get_language_operators(self) -> Set[str]:
        return {'+', '-', '*', '/', '%', '++', '--', '=', '+=', '-=', '*=', '/=',
                '==', '===', '!=', '!==', '<', '>', '<=', '>=', '&&', '||', '!',
                '&', '|', '^', '~', '<<', '>>', '>>>', '?', ':'}
    
    def get_language_built_ins(self) -> Set[str]:
        return {
            'Array', 'Object', 'String', 'Number', 'Boolean', 'Date', 'RegExp',
            'Math', 'JSON', 'console', 'window', 'document', 'parseInt',
            'parseFloat', 'isNaN', 'isFinite', 'encodeURI', 'decodeURI',
            'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval'
        }
    
    async def preprocess(self, code: str) -> PreprocessingResult:
        """Preprocessa código JavaScript."""
        result = PreprocessingResult(
            original_code=code,
            processed_code=code,
            language=self.get_language()
        )
        
        processed = code
        
        # Aplicar transformaciones similares a Python
        if self.config.normalize_whitespace:
            processed = self.normalize_whitespace(processed)
            result.transformations_applied.append("normalize_whitespace")
        
        if self.config.remove_comments:
            processed = self.remove_js_comments(processed)
            result.transformations_applied.append("remove_comments")
        
        if self.config.normalize_strings:
            processed = self.normalize_js_strings(processed)
            result.transformations_applied.append("normalize_strings")
        
        if self.config.normalize_numbers:
            processed = self.normalize_numeric_literals(processed)
            result.transformations_applied.append("normalize_numbers")
        
        if (self.config.normalize_identifiers and 
            self.config.level == PreprocessingLevel.AGGRESSIVE):
            processed = self.normalize_js_identifiers(processed)
            result.transformations_applied.append("normalize_identifiers")
        
        # Normalizar función arrows en modo agresivo
        if self.config.level == PreprocessingLevel.AGGRESSIVE:
            processed = self.normalize_arrow_functions(processed)
            result.transformations_applied.append("normalize_arrows")
        
        result.processed_code = processed
        return result
    
    def remove_js_comments(self, code: str) -> str:
        """Remover comentarios JavaScript."""
        # Remover comentarios de línea
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        
        # Remover comentarios de bloque
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        return code
    
    def normalize_js_strings(self, code: str) -> str:
        """Normalizar strings JavaScript."""
        # Template literals
        code = re.sub(r'`[^`]*`', '"TEMPLATE"', code)
        
        # Strings normales
        return self.normalize_string_literals(code)
    
    def normalize_js_identifiers(self, code: str) -> str:
        """Normalizar identificadores JavaScript."""
        identifier_map = {}
        counter = 0
        
        def replace_identifier(match):
            nonlocal counter
            identifier = match.group(0)
            
            if (identifier in self.keywords or 
                identifier in self.built_ins):
                return identifier
            
            if identifier not in identifier_map:
                counter += 1
                identifier_map[identifier] = f'var_{counter}'
            
            return identifier_map[identifier]
        
        pattern = r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b'
        return re.sub(pattern, replace_identifier, code)
    
    def normalize_arrow_functions(self, code: str) -> str:
        """Normalizar arrow functions a function declarations."""
        # Transformación simple de arrow functions
        code = re.sub(r'(\w+)\s*=>\s*', r'function(\1)', code)
        
        return code


class TypeScriptPreprocessor(JavaScriptPreprocessor):
    """Preprocessador específico para TypeScript."""
    
    def get_language(self) -> ProgrammingLanguage:
        return ProgrammingLanguage.TYPESCRIPT
    
    def get_language_keywords(self) -> Set[str]:
        # Extender keywords de JavaScript con TypeScript
        js_keywords = super().get_language_keywords()
        ts_keywords = {
            'abstract', 'any', 'as', 'boolean', 'constructor', 'declare',
            'enum', 'implements', 'interface', 'module', 'namespace', 'never',
            'number', 'private', 'protected', 'public', 'readonly', 'static',
            'string', 'type', 'undefined', 'unknown', 'void'
        }
        
        return js_keywords | ts_keywords
    
    async def preprocess(self, code: str) -> PreprocessingResult:
        """Preprocessa código TypeScript."""
        # Comenzar con preprocessado de JavaScript
        result = await super().preprocess(code)
        
        # Añadir transformaciones específicas de TypeScript
        processed = result.processed_code
        
        # Remover anotaciones de tipo
        processed = self.remove_type_annotations(processed)
        result.transformations_applied.append("remove_type_annotations")
        
        # Remover interfaces en modo agresivo
        if self.config.level == PreprocessingLevel.AGGRESSIVE:
            processed = self.remove_interfaces(processed)
            result.transformations_applied.append("remove_interfaces")
        
        result.processed_code = processed
        return result
    
    def remove_type_annotations(self, code: str) -> str:
        """Remover anotaciones de tipo TypeScript."""
        # Remover tipos después de dos puntos
        code = re.sub(r':\s*[\w\[\]<>,\s|&{}]+(?=[,;)\]}=])', '', code)
        
        # Remover parámetros genéricos
        code = re.sub(r'<[\w\[\],\s]+>', '', code)
        
        # Remover type assertions
        code = re.sub(r'as\s+[\w\[\]<>,\s]+', '', code)
        
        return code
    
    def remove_interfaces(self, code: str) -> str:
        """Remover declaraciones de interface."""
        code = re.sub(r'interface\s+\w+\s*\{[^{}]*\}', '', code, flags=re.DOTALL)
        
        return code


class RustPreprocessor(LanguagePreprocessor):
    """Preprocessador específico para Rust."""
    
    def get_language(self) -> ProgrammingLanguage:
        return ProgrammingLanguage.RUST
    
    def get_language_keywords(self) -> Set[str]:
        return {
            'as', 'break', 'const', 'continue', 'crate', 'else', 'enum', 'extern',
            'false', 'fn', 'for', 'if', 'impl', 'in', 'let', 'loop', 'match',
            'mod', 'move', 'mut', 'pub', 'ref', 'return', 'self', 'Self',
            'static', 'struct', 'super', 'trait', 'true', 'type', 'unsafe',
            'use', 'where', 'while', 'async', 'await', 'dyn'
        }
    
    def get_language_operators(self) -> Set[str]:
        return {'+', '-', '*', '/', '%', '=', '+=', '-=', '*=', '/=', '%=',
                '==', '!=', '<', '>', '<=', '>=', '&&', '||', '!', '&', '|',
                '^', '<<', '>>', '->', '=>', '::', '..', '...'}
    
    def get_language_built_ins(self) -> Set[str]:
        return {
            'Vec', 'String', 'Option', 'Result', 'Ok', 'Err', 'Some', 'None',
            'Box', 'Rc', 'Arc', 'Cell', 'RefCell', 'HashMap', 'BTreeMap',
            'println', 'print', 'panic', 'assert', 'debug_assert'
        }
    
    async def preprocess(self, code: str) -> PreprocessingResult:
        """Preprocessa código Rust."""
        result = PreprocessingResult(
            original_code=code,
            processed_code=code,
            language=self.get_language()
        )
        
        processed = code
        
        if self.config.normalize_whitespace:
            processed = self.normalize_whitespace(processed)
            result.transformations_applied.append("normalize_whitespace")
        
        if self.config.remove_comments:
            processed = self.remove_rust_comments(processed)
            result.transformations_applied.append("remove_comments")
        
        if self.config.normalize_strings:
            processed = self.normalize_rust_strings(processed)
            result.transformations_applied.append("normalize_strings")
        
        if self.config.normalize_numbers:
            processed = self.normalize_numeric_literals(processed)
            result.transformations_applied.append("normalize_numbers")
        
        # Remover atributos y macros en modo agresivo
        if self.config.level == PreprocessingLevel.AGGRESSIVE:
            processed = self.remove_rust_attributes(processed)
            processed = self.normalize_rust_macros(processed)
            result.transformations_applied.extend(["remove_attributes", "normalize_macros"])
        
        result.processed_code = processed
        return result
    
    def remove_rust_comments(self, code: str) -> str:
        """Remover comentarios Rust."""
        # Comentarios de línea
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        
        # Comentarios de bloque
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        return code
    
    def normalize_rust_strings(self, code: str) -> str:
        """Normalizar strings Rust."""
        # Raw strings
        code = re.sub(r'r#*"[^"]*"#*', '"RAWSTRING"', code)
        
        # Strings normales
        return self.normalize_string_literals(code)
    
    def remove_rust_attributes(self, code: str) -> str:
        """Remover atributos Rust (#[...])."""
        code = re.sub(r'#\[[^\]]*\]', '', code)
        
        return code
    
    def normalize_rust_macros(self, code: str) -> str:
        """Normalizar macros Rust."""
        code = re.sub(r'\w+!', 'MACRO!', code)
        
        return code


class CodePreprocessor:
    """Preprocessador principal de código multi-lenguaje."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Inicializa el preprocessador.
        
        Args:
            config: Configuración de preprocessamiento
        """
        self.config = config or PreprocessingConfig()
        self.language_processors: Dict[ProgrammingLanguage, LanguagePreprocessor] = {}
        self.common_processor = CommonPreprocessor(self.config)
        
    async def initialize(self) -> None:
        """Inicializa preprocessadores específicos."""
        logger.info("Inicializando CodePreprocessor...")
        
        # Crear preprocessadores para lenguajes soportados
        self.language_processors = {
            ProgrammingLanguage.PYTHON: PythonPreprocessor(self.config),
            ProgrammingLanguage.JAVASCRIPT: JavaScriptPreprocessor(self.config),
            ProgrammingLanguage.TYPESCRIPT: TypeScriptPreprocessor(self.config),
            ProgrammingLanguage.RUST: RustPreprocessor(self.config),
        }
        
        logger.info(f"CodePreprocessor inicializado con {len(self.language_processors)} lenguajes")
    
    async def preprocess(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> PreprocessingResult:
        """
        Preprocessa código para análisis de IA.
        
        Args:
            code: Código fuente a preprocessar
            language: Lenguaje del código
            
        Returns:
            Resultado del preprocessamiento
        """
        if not code.strip():
            return PreprocessingResult(
                original_code=code,
                processed_code=code,
                language=language,
                transformations_applied=["empty_code"]
            )
        
        # Aplicar preprocessamiento común
        common_result = await self.common_processor.preprocess(code)
        
        # Aplicar preprocessamiento específico del lenguaje
        if language in self.language_processors:
            processor = self.language_processors[language]
            result = await processor.preprocess(common_result)
            
            # Combinar transformaciones
            result.transformations_applied.insert(0, "common_preprocessing")
        else:
            # Lenguaje no soportado - solo common preprocessing
            result = PreprocessingResult(
                original_code=code,
                processed_code=common_result,
                language=language,
                transformations_applied=["common_preprocessing", "unsupported_language"]
            )
        
        logger.debug(f"Preprocessado {language.value}: {len(result.transformations_applied)} transformaciones")
        return result
    
    async def preprocess_for_similarity(
        self,
        code: str,
        language: ProgrammingLanguage
    ) -> PreprocessingResult:
        """
        Preprocessa código para análisis de similitud (más agresivo).
        
        Args:
            code: Código fuente
            language: Lenguaje del código
            
        Returns:
            Resultado del preprocessamiento agresivo
        """
        # Crear config agresiva temporal
        aggressive_config = PreprocessingConfig(
            level=PreprocessingLevel.AGGRESSIVE,
            normalize_identifiers=True,
            remove_comments=True,
            remove_docstrings=True,
            normalize_strings=True,
            normalize_numbers=True,
            preserve_structure=False
        )
        
        # Crear preprocessador temporal con config agresiva
        temp_preprocessor = CodePreprocessor(aggressive_config)
        await temp_preprocessor.initialize()
        
        return await temp_preprocessor.preprocess(code, language)
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Obtiene lista de lenguajes soportados."""
        return list(self.language_processors.keys())
    
    async def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del preprocessor."""
        return {
            "supported_languages": len(self.language_processors),
            "configuration": {
                "level": self.config.level.value,
                "normalize_whitespace": self.config.normalize_whitespace,
                "normalize_identifiers": self.config.normalize_identifiers,
                "remove_comments": self.config.remove_comments,
                "language_specific_rules": self.config.language_specific_rules
            },
            "languages": [lang.value for lang in self.get_supported_languages()]
        }


class CommonPreprocessor:
    """Preprocessador común para todos los lenguajes."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    async def preprocess(self, code: str) -> str:
        """Aplica transformaciones comunes."""
        processed = code
        
        if self.config.normalize_whitespace:
            processed = self.normalize_common_whitespace(processed)
        
        return processed
    
    def normalize_common_whitespace(self, code: str) -> str:
        """Normalización básica de espacios."""
        # Normalizar saltos de línea
        code = re.sub(r'\r\n|\r', '\n', code)
        
        # Remover espacios al final
        lines = []
        for line in code.splitlines():
            lines.append(line.rstrip())
        
        # Limitar líneas vacías
        result_lines = []
        consecutive_empty = 0
        
        for line in lines:
            if not line.strip():
                consecutive_empty += 1
                if consecutive_empty <= 1:
                    result_lines.append(line)
            else:
                consecutive_empty = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
