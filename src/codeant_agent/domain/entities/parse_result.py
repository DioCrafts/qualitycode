"""
Entidades relacionadas con resultados de parsing.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Dict

from ..value_objects.programming_language import ProgrammingLanguage

@dataclass
class Symbol:
    """Representa un símbolo encontrado durante el parsing."""
    name: str
    kind: str  # function, class, variable, etc.
    location: Dict[str, Any]  # start_line, start_column, end_line, end_column
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Import:
    """Representa una importación encontrada durante el parsing."""
    source: str  # Módulo importado
    symbols: List[str]  # Símbolos importados
    location: Dict[str, Any]  # start_line, start_column, end_line, end_column
    is_default: bool = False
    is_namespace: bool = False

@dataclass
class Export:
    """Representa una exportación encontrada durante el parsing."""
    name: str
    location: Dict[str, Any]  # start_line, start_column, end_line, end_column
    is_default: bool = False

@dataclass
class ParseResult:
    """Resultado del parsing de un archivo."""
    file_path: Path
    language: ProgrammingLanguage
    ast: Optional[Any] = None  # AST específico del parser
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[Import] = field(default_factory=list)
    exports: List[Export] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)