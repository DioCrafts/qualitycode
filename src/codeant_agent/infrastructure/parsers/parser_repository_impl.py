"""
Implementación del repositorio de parser universal.

Este módulo proporciona la implementación concreta del repositorio
que maneja el parsing de código fuente usando Tree-sitter.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field

from ...domain.repositories.parser_repository import ParserRepository
from ...domain.value_objects.programming_language import ProgrammingLanguage
from ...domain.entities.parse_result import ParseResult, ParseRequest, ParseStatus, ParseMetadata
from ...utils.error import ParsingError

@dataclass 
class FunctionInfo:
    """Información básica de una función."""
    name: str
    line: int
    params: List[str] = field(default_factory=list)
    
@dataclass
class ClassInfo:
    """Información básica de una clase."""
    name: str
    line: int
    methods: List[str] = field(default_factory=list)

logger = logging.getLogger(__name__)


class ParserRepositoryImpl(ParserRepository):
    """
    Implementación del repositorio de parser universal.
    
    Esta implementación proporciona parsing básico para múltiples lenguajes.
    En una implementación real, usaría Tree-sitter para parsing real del AST.
    """
    
    def __init__(self):
        """Inicializar el repositorio."""
        self.supported_languages = {
            ProgrammingLanguage.PYTHON,
            ProgrammingLanguage.JAVASCRIPT,
            ProgrammingLanguage.TYPESCRIPT,
            ProgrammingLanguage.RUST
        }
        self._cache = {}
        self._stats = {
            'files_parsed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parse_errors': 0
        }
    
    async def parse_file(self, file_path: Path) -> ParseResult:
        """
        Parsea un archivo de código fuente.
        """
        logger.debug(f"Parseando archivo: {file_path}")
        
        # Verificar caché
        cache_key = str(file_path)
        if cache_key in self._cache:
            self._stats['cache_hits'] += 1
            logger.debug(f"Cache hit para {file_path}")
            return self._cache[cache_key]
        
        self._stats['cache_misses'] += 1
        
        try:
            # Leer contenido del archivo
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detectar lenguaje
            language = await self.detect_language(file_path, content)
            
            # Parsear contenido
            result = await self.parse_content(content, language)
            result.file_path = file_path
            
            # Guardar en caché
            self._cache[cache_key] = result
            self._stats['files_parsed'] += 1
            
            return result
            
        except Exception as e:
            self._stats['parse_errors'] += 1
            logger.error(f"Error parseando {file_path}: {e}")
            raise ParsingError(
                message=f"Error parseando archivo: {str(e)}",
                language=str(language),
                file_path=str(file_path)
            )
    
    async def parse_content(self, content: str, language: ProgrammingLanguage) -> ParseResult:
        """
        Parsea contenido de código fuente directamente.
        """
        logger.debug(f"Parseando contenido para lenguaje {language}")
        
        # Análisis básico del contenido
        lines = content.split('\n')
        
        # Crear metadatos
        metadata = ParseMetadata(
            file_size_bytes=len(content.encode('utf-8')),
            line_count=len(lines),
            character_count=len(content),
            encoding="UTF-8",
            has_syntax_errors=False,
            complexity_estimate=0.0,
            parse_duration_ms=0,
            node_count=100,  # Valor simulado
            error_count=0,
            warning_count=0,
            cache_hit=False,
            incremental_parse=False
        )
        
        # Crear resultado básico
        result = ParseResult(
            tree={"type": "module", "children": []},  # AST simulado
            language=language,
            status=ParseStatus.SUCCESS,
            metadata=metadata,
            file_path=None,
            warnings=[],
            errors=[]
        )
        
        
        # Detectar funciones (implementación muy básica)
        result.functions = self._detect_functions(lines, language)
        
        # Detectar clases (implementación muy básica)
        result.classes = self._detect_classes(lines, language)
        
        # Detectar imports (implementación muy básica)
        result.imports = self._detect_imports(lines, language)
        
        # Calcular métricas básicas
        result.metrics = {
            'lines_of_code': len([line for line in lines if line.strip()]),
            'lines_total': len(lines),
            'function_count': len(result.functions),
            'class_count': len(result.classes),
            'import_count': len(result.imports),
            'cyclomatic_complexity': self._estimate_complexity(lines, language)
        }
        
        return result
    
    async def parse_incremental(
        self, 
        old_tree: Any, 
        content: str, 
        edits: List[Dict[str, Any]]
    ) -> ParseResult:
        """
        Parsea incrementalmente un archivo con cambios.
        """
        # Por ahora, simplemente re-parseamos todo el contenido
        # En una implementación real, Tree-sitter soporta parsing incremental
        language = ProgrammingLanguage.PYTHON  # Por defecto
        return await self.parse_content(content, language)
    
    async def detect_language(self, file_path: Path, content: str) -> ProgrammingLanguage:
        """
        Detecta automáticamente el lenguaje de programación.
        """
        # Detección por extensión de archivo
        extension_map = {
            '.py': ProgrammingLanguage.PYTHON,
            '.js': ProgrammingLanguage.JAVASCRIPT,
            '.jsx': ProgrammingLanguage.JAVASCRIPT,
            '.ts': ProgrammingLanguage.TYPESCRIPT,
            '.tsx': ProgrammingLanguage.TYPESCRIPT,
            '.rs': ProgrammingLanguage.RUST,
        }
        
        ext = file_path.suffix.lower()
        if ext in extension_map:
            return extension_map[ext]
        
        # Si no se puede detectar por extensión, intentar por contenido
        # (implementación muy básica)
        if 'def ' in content or 'import ' in content:
            return ProgrammingLanguage.PYTHON
        elif 'function ' in content or 'const ' in content or 'let ' in content:
            return ProgrammingLanguage.JAVASCRIPT
        elif 'fn ' in content or 'let mut' in content:
            return ProgrammingLanguage.RUST
        
        # Por defecto
        return ProgrammingLanguage.PYTHON
    
    async def get_ast_json(self, tree: Any) -> str:
        """
        Convierte un AST a formato JSON.
        """
        # En una implementación real, serializaría el árbol de Tree-sitter
        return json.dumps({"type": "placeholder_ast", "children": []})
    
    async def query_ast(
        self, 
        tree: Any, 
        query: str, 
        language: ProgrammingLanguage
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta una consulta sobre un AST.
        """
        # En una implementación real, ejecutaría queries de Tree-sitter
        return []
    
    async def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """
        Obtiene la lista de lenguajes soportados.
        """
        return list(self.supported_languages)
    
    async def is_language_supported(self, language: ProgrammingLanguage) -> bool:
        """
        Verifica si un lenguaje está soportado.
        """
        return language in self.supported_languages
    
    async def get_parser_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del parser.
        """
        return self._stats.copy()
    
    async def clear_cache(self) -> None:
        """
        Limpia la caché del parser.
        """
        self._cache.clear()
        logger.info("Caché del parser limpiada")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la caché.
        """
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        hit_rate = 0
        if total_requests > 0:
            hit_rate = self._stats['cache_hits'] / total_requests
        
        return {
            'cache_size': len(self._cache),
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'hit_rate': hit_rate
        }
    
    # Métodos auxiliares privados
    
    def _detect_functions(self, lines: List[str], language: ProgrammingLanguage) -> List[FunctionInfo]:
        """Detecta funciones en el código (implementación básica)."""
        functions = []
        
        for i, line in enumerate(lines):
            if language == ProgrammingLanguage.PYTHON:
                if line.strip().startswith('def '):
                    # Extraer nombre de función
                    parts = line.strip().split('(')
                    if parts:
                        name = parts[0].replace('def ', '').strip()
                        functions.append(FunctionInfo(
                            name=name,
                            start_line=i + 1,
                            end_line=i + 10,  # Estimación
                            parameters=[],
                            return_type=None,
                            is_async='async ' in line,
                            decorators=[],
                            complexity=5  # Estimación
                        ))
            
            elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
                if 'function ' in line or '=>' in line:
                    # Detección muy básica
                    if 'function ' in line:
                        parts = line.split('function ')
                        if len(parts) > 1:
                            name_part = parts[1].split('(')[0].strip()
                            if name_part:
                                functions.append(FunctionInfo(
                                    name=name_part,
                                    start_line=i + 1,
                                    end_line=i + 10,
                                    parameters=[],
                                    return_type=None,
                                    is_async='async ' in line,
                                    decorators=[],
                                    complexity=5
                                ))
            
            elif language == ProgrammingLanguage.RUST:
                if line.strip().startswith('fn '):
                    parts = line.strip().split('(')
                    if parts:
                        name = parts[0].replace('fn ', '').strip()
                        functions.append(FunctionInfo(
                            name=name,
                            start_line=i + 1,
                            end_line=i + 10,
                            parameters=[],
                            return_type=None,
                            is_async='async ' in line,
                            decorators=[],
                            complexity=5
                        ))
        
        return functions
    
    def _detect_classes(self, lines: List[str], language: ProgrammingLanguage) -> List[ClassInfo]:
        """Detecta clases en el código (implementación básica)."""
        classes = []
        
        for i, line in enumerate(lines):
            if language == ProgrammingLanguage.PYTHON:
                if line.strip().startswith('class '):
                    parts = line.strip().split(':')
                    if parts:
                        name_part = parts[0].replace('class ', '').strip()
                        name = name_part.split('(')[0].strip()
                        classes.append(ClassInfo(
                            name=name,
                            start_line=i + 1,
                            end_line=i + 20,  # Estimación
                            methods=[],
                            attributes=[],
                            base_classes=[],
                            decorators=[],
                            is_abstract=False
                        ))
            
            elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
                if line.strip().startswith('class '):
                    parts = line.strip().split(' ')
                    if len(parts) > 1:
                        name = parts[1].split('{')[0].strip()
                        classes.append(ClassInfo(
                            name=name,
                            start_line=i + 1,
                            end_line=i + 20,
                            methods=[],
                            attributes=[],
                            base_classes=[],
                            decorators=[],
                            is_abstract='abstract' in line
                        ))
            
            elif language == ProgrammingLanguage.RUST:
                if line.strip().startswith('struct ') or line.strip().startswith('impl '):
                    parts = line.strip().split(' ')
                    if len(parts) > 1:
                        name = parts[1].split('{')[0].strip()
                        classes.append(ClassInfo(
                            name=name,
                            start_line=i + 1,
                            end_line=i + 20,
                            methods=[],
                            attributes=[],
                            base_classes=[],
                            decorators=[],
                            is_abstract=False
                        ))
        
        return classes
    
    def _detect_imports(self, lines: List[str], language: ProgrammingLanguage) -> List[Dict[str, Any]]:
        """Detecta imports en el código (implementación básica)."""
        imports = []
        
        for i, line in enumerate(lines):
            if language == ProgrammingLanguage.PYTHON:
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    imports.append({
                        'module': line.strip(),
                        'line': i + 1,
                        'type': 'import'
                    })
            
            elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
                if 'import ' in line or 'require(' in line:
                    imports.append({
                        'module': line.strip(),
                        'line': i + 1,
                        'type': 'import'
                    })
            
            elif language == ProgrammingLanguage.RUST:
                if line.strip().startswith('use '):
                    imports.append({
                        'module': line.strip(),
                        'line': i + 1,
                        'type': 'use'
                    })
        
        return imports
    
    def _estimate_complexity(self, lines: List[str], language: ProgrammingLanguage) -> int:
        """Estima la complejidad ciclomática (implementación muy básica)."""
        complexity = 1  # Base complexity
        
        keywords = {
            ProgrammingLanguage.PYTHON: ['if ', 'elif ', 'else:', 'for ', 'while ', 'except:', 'with '],
            ProgrammingLanguage.JAVASCRIPT: ['if ', 'else ', 'for ', 'while ', 'switch ', 'case ', 'catch '],
            ProgrammingLanguage.TYPESCRIPT: ['if ', 'else ', 'for ', 'while ', 'switch ', 'case ', 'catch '],
            ProgrammingLanguage.RUST: ['if ', 'else ', 'for ', 'while ', 'match ', 'loop '],
        }
        
        lang_keywords = keywords.get(language, [])
        
        for line in lines:
            line_lower = line.lower().strip()
            for keyword in lang_keywords:
                if line_lower.startswith(keyword):
                    complexity += 1
        
        return min(complexity, 20)  # Cap at 20 for this basic implementation
