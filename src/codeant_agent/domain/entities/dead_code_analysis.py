"""
Entidades del dominio para análisis de código muerto.

Este módulo contiene todas las entidades que representan los resultados
del análisis de detección de código muerto.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum

from ..value_objects.programming_language import ProgrammingLanguage
from .parse_result import ParseResult


class UnusedReason(Enum):
    """Razones por las que el código se considera no utilizado."""
    NEVER_CALLED = "never_called"
    NEVER_REFERENCED = "never_referenced"
    ONLY_ASSIGNED_NEVER_READ = "only_assigned_never_read"
    SELF_ASSIGNMENT_ONLY = "self_assignment_only"
    DEAD_BRANCH = "dead_branch"
    UNREACHABLE_CODE = "unreachable_code"
    OBSOLETE_FUNCTION = "obsolete_function"
    DUPLICATE_IMPLEMENTATION = "duplicate_implementation"
    PARTIALLY_UNUSED = "partially_unused"


class UnreachabilityReason(Enum):
    """Razones por las que el código es inalcanzable."""
    AFTER_RETURN = "after_return"
    AFTER_THROW = "after_throw"
    AFTER_BREAK = "after_break"
    AFTER_CONTINUE = "after_continue"
    DEAD_BRANCH = "dead_branch"
    ALWAYS_FALSE_CONDITION = "always_false_condition"
    ALWAYS_TRUE_CONDITION = "always_true_condition"
    UNREACHABLE_FROM_ENTRY = "unreachable_from_entry"
    CIRCULAR_DEPENDENCY = "circular_dependency"


class ImportType(Enum):
    """Tipos de imports."""
    DEFAULT_IMPORT = "default_import"
    NAMED_IMPORTS = "named_imports"
    NAMESPACE_IMPORT = "namespace_import"
    SIDE_EFFECT_IMPORT = "side_effect_import"
    PARTIALLY_UNUSED = "partially_unused"


class Visibility(Enum):
    """Visibilidad de símbolos."""
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    INTERNAL = "internal"


class FunctionType(Enum):
    """Tipos de funciones."""
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    DESTRUCTOR = "destructor"
    STATIC_METHOD = "static_method"
    CLASS_METHOD = "class_method"
    PROPERTY = "property"
    GETTER = "getter"
    SETTER = "setter"


class ScopeType(Enum):
    """Tipos de scope."""
    GLOBAL = "global"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    BLOCK = "block"
    LOOP = "loop"
    CONDITION = "condition"


class AssignmentType(Enum):
    """Tipos de asignación."""
    SIMPLE_ASSIGNMENT = "simple_assignment"
    COMPOUND_ASSIGNMENT = "compound_assignment"
    DESTRUCTURING_ASSIGNMENT = "destructuring_assignment"
    INITIALIZATION_ASSIGNMENT = "initialization_assignment"


class RedundancyType(Enum):
    """Tipos de redundancia."""
    IMMEDIATE_REASSIGNMENT = "immediate_reassignment"
    UNUSED_BETWEEN_ASSIGNMENTS = "unused_between_assignments"
    SELF_ASSIGNMENT = "self_assignment"
    CONSTANT_REASSIGNMENT = "constant_reassignment"


class EntryPointType(Enum):
    """Tipos de puntos de entrada."""
    MAIN_FUNCTION = "main_function"
    PUBLIC_API = "public_api"
    TEST_FUNCTION = "test_function"
    EVENT_HANDLER = "event_handler"
    WEB_ENDPOINT = "web_endpoint"
    CLI_COMMAND = "cli_command"
    FRAMEWORK_HOOK = "framework_hook"


@dataclass
class SourcePosition:
    """Posición en el código fuente."""
    line: int
    column: int
    offset: int = 0

    def __post_init__(self) -> None:
        if self.line < 1:
            raise ValueError("El número de línea debe ser >= 1")
        if self.column < 0:
            raise ValueError("La columna no puede ser negativa")
        if self.offset < 0:
            raise ValueError("El offset no puede ser negativo")


@dataclass
class SourceRange:
    """Rango en el código fuente."""
    start: SourcePosition
    end: SourcePosition

    def __post_init__(self) -> None:
        if self.start.line > self.end.line:
            raise ValueError("La línea de inicio no puede ser mayor que la final")
        if (self.start.line == self.end.line and 
            self.start.column > self.end.column):
            raise ValueError("La columna de inicio no puede ser mayor que la final")


@dataclass
class ScopeInfo:
    """Información sobre el scope de un símbolo."""
    scope_type: ScopeType
    scope_name: Optional[str] = None
    parent_scope: Optional[str] = None
    nesting_level: int = 0


@dataclass
class Parameter:
    """Parámetro de función."""
    name: str
    type_hint: Optional[str] = None
    default_value: Optional[str] = None
    is_variadic: bool = False
    is_keyword_only: bool = False


@dataclass
class SideEffect:
    """Efecto secundario potencial."""
    effect_type: str
    description: str
    confidence: float = 1.0


@dataclass
class BlockingCondition:
    """Condición que bloquea la ejecución de código."""
    condition_location: SourceRange
    condition_expression: str
    reason: str


@dataclass
class UnusedVariable:
    """Variable no utilizada."""
    name: str
    declaration_location: SourceRange
    variable_type: Optional[str] = None
    scope: Optional[ScopeInfo] = None
    reason: UnusedReason = UnusedReason.NEVER_REFERENCED
    suggestion: str = ""
    confidence: float = 1.0


@dataclass
class UnusedFunction:
    """Función no utilizada."""
    name: str
    declaration_location: SourceRange
    function_type: FunctionType = FunctionType.FUNCTION
    visibility: Visibility = Visibility.PRIVATE
    parameters: List[Parameter] = field(default_factory=list)
    reason: UnusedReason = UnusedReason.NEVER_CALLED
    suggestion: str = ""
    confidence: float = 1.0
    potential_side_effects: List[SideEffect] = field(default_factory=list)


@dataclass
class UnusedClass:
    """Clase no utilizada."""
    name: str
    declaration_location: SourceRange
    visibility: Visibility = Visibility.PRIVATE
    is_abstract: bool = False
    parent_classes: List[str] = field(default_factory=list)
    reason: UnusedReason = UnusedReason.NEVER_REFERENCED
    suggestion: str = ""
    confidence: float = 1.0


@dataclass
class ImportStatement:
    """Declaración de import."""
    module_name: str
    imported_symbols: List[str] = field(default_factory=list)
    import_type: ImportType = ImportType.DEFAULT_IMPORT
    location: SourceRange = None
    language: ProgrammingLanguage = None
    alias: Optional[str] = None


@dataclass
class UnusedImport:
    """Import no utilizado."""
    import_statement: ImportStatement
    location: SourceRange
    import_type: ImportType
    module_name: str
    imported_symbols: List[str] = field(default_factory=list)
    reason: UnusedReason = UnusedReason.NEVER_REFERENCED
    suggestion: str = ""
    confidence: float = 1.0
    side_effects_possible: bool = False


@dataclass
class UnreachableCode:
    """Código inalcanzable."""
    location: SourceRange
    code_type: str
    reason: UnreachabilityReason = UnreachabilityReason.UNREACHABLE_FROM_ENTRY
    suggestion: str = ""
    confidence: float = 1.0
    blocking_condition: Optional[BlockingCondition] = None


@dataclass
class DeadBranch:
    """Rama muerta en condicionales."""
    location: SourceRange
    branch_type: str  # if, elif, else, case, etc.
    condition: Optional[str] = None
    reason: str = ""
    suggestion: str = ""
    confidence: float = 1.0


@dataclass
class UnusedParameter:
    """Parámetro no utilizado."""
    name: str
    function_name: str
    location: SourceRange
    parameter_type: Optional[str] = None
    is_self_parameter: bool = False
    suggestion: str = ""
    confidence: float = 1.0


@dataclass
class RedundantAssignment:
    """Asignación redundante."""
    location: SourceRange
    variable_name: str
    previous_assignment: SourceRange
    redundancy_type: RedundancyType = RedundancyType.IMMEDIATE_REASSIGNMENT
    suggestion: str = ""
    confidence: float = 1.0


@dataclass
class DeadCodeStatistics:
    """Estadísticas del análisis de código muerto."""
    total_unused_variables: int = 0
    total_unused_functions: int = 0
    total_unused_classes: int = 0
    total_unused_imports: int = 0
    total_unreachable_code_blocks: int = 0
    total_dead_branches: int = 0
    total_unused_parameters: int = 0
    total_redundant_assignments: int = 0
    percentage_dead_code: float = 0.0
    lines_of_dead_code: int = 0
    potential_savings_bytes: int = 0

    def get_total_issues(self) -> int:
        """Obtiene el total de issues encontrados."""
        return (
            self.total_unused_variables +
            self.total_unused_functions +
            self.total_unused_classes +
            self.total_unused_imports +
            self.total_unreachable_code_blocks +
            self.total_dead_branches +
            self.total_unused_parameters +
            self.total_redundant_assignments
        )


@dataclass
class EntryPoint:
    """Punto de entrada al código."""
    symbol_id: str
    entry_type: EntryPointType
    location: SourceRange
    confidence: float = 1.0


@dataclass
class DeadCodeAnalysis:
    """Resultado completo del análisis de código muerto."""
    file_path: Path
    language: ProgrammingLanguage
    unused_variables: List[UnusedVariable] = field(default_factory=list)
    unused_functions: List[UnusedFunction] = field(default_factory=list)
    unused_classes: List[UnusedClass] = field(default_factory=list)
    unused_imports: List[UnusedImport] = field(default_factory=list)
    unreachable_code: List[UnreachableCode] = field(default_factory=list)
    dead_branches: List[DeadBranch] = field(default_factory=list)
    unused_parameters: List[UnusedParameter] = field(default_factory=list)
    redundant_assignments: List[RedundantAssignment] = field(default_factory=list)
    statistics: DeadCodeStatistics = field(default_factory=DeadCodeStatistics)
    execution_time_ms: int = 0
    analyzed_at: datetime = field(default_factory=datetime.utcnow)

    def get_all_issues(self) -> List[Union[UnusedVariable, UnusedFunction, UnusedClass, 
                                         UnusedImport, UnreachableCode, DeadBranch,
                                         UnusedParameter, RedundantAssignment]]:
        """Obtiene todos los issues encontrados."""
        issues = []
        issues.extend(self.unused_variables)
        issues.extend(self.unused_functions)
        issues.extend(self.unused_classes)
        issues.extend(self.unused_imports)
        issues.extend(self.unreachable_code)
        issues.extend(self.dead_branches)
        issues.extend(self.unused_parameters)
        issues.extend(self.redundant_assignments)
        return issues

    def get_high_confidence_issues(self, threshold: float = 0.8) -> List[Any]:
        """Obtiene issues con alta confianza."""
        return [issue for issue in self.get_all_issues() if issue.confidence >= threshold]

    def get_issues_by_type(self, issue_type: str) -> List[Any]:
        """Obtiene issues por tipo específico."""
        type_mapping = {
            'variable': self.unused_variables,
            'function': self.unused_functions,
            'class': self.unused_classes,
            'import': self.unused_imports,
            'unreachable': self.unreachable_code,
            'branch': self.dead_branches,
            'parameter': self.unused_parameters,
            'assignment': self.redundant_assignments,
        }
        return type_mapping.get(issue_type, [])


@dataclass
class CrossModuleIssue:
    """Issue que afecta múltiples módulos."""
    issue_type: str
    description: str
    affected_modules: List[Path] = field(default_factory=list)
    suggestion: str = ""
    confidence: float = 1.0


@dataclass
class ProjectDeadCodeAnalysis:
    """Análisis de código muerto para un proyecto completo."""
    project_path: Optional[Path]
    file_analyses: List[DeadCodeAnalysis] = field(default_factory=list)
    global_statistics: DeadCodeStatistics = field(default_factory=DeadCodeStatistics)
    cross_module_issues: List[CrossModuleIssue] = field(default_factory=list)
    dependency_cycles: List[List[str]] = field(default_factory=list)
    execution_time_ms: int = 0
    analyzed_at: datetime = field(default_factory=datetime.utcnow)

    def get_total_files_analyzed(self) -> int:
        """Obtiene el total de archivos analizados."""
        return len(self.file_analyses)

    def get_files_with_issues(self) -> List[DeadCodeAnalysis]:
        """Obtiene archivos que tienen issues."""
        return [analysis for analysis in self.file_analyses 
                if analysis.statistics.get_total_issues() > 0]

    def get_worst_files(self, limit: int = 10) -> List[DeadCodeAnalysis]:
        """Obtiene los archivos con más issues."""
        return sorted(
            self.file_analyses,
            key=lambda x: x.statistics.get_total_issues(),
            reverse=True
        )[:limit]
