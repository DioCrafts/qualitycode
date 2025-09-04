"""
Entidades relacionadas con análisis de código muerto.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..value_objects.programming_language import ProgrammingLanguage

@dataclass
class UnifiedPosition:
    """Posición unificada en un archivo de código."""
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    file_path: Path

@dataclass
class UnusedImport:
    """Importación no utilizada."""
    name: str
    position: UnifiedPosition
    confidence: float
    reason: str

@dataclass
class UnusedVariable:
    """Variable no utilizada."""
    name: str
    position: UnifiedPosition
    confidence: float
    reason: str
    scope: Optional[str] = None
    type_hint: Optional[str] = None

@dataclass
class UnusedFunction:
    """Función no utilizada."""
    name: str
    position: UnifiedPosition
    confidence: float
    reason: str

@dataclass
class UnusedClass:
    """Clase no utilizada."""
    name: str
    position: UnifiedPosition
    confidence: float
    reason: str

@dataclass
class UnreachableCode:
    """Código inalcanzable."""
    position: UnifiedPosition
    confidence: float
    reason: str
    code: Optional[str] = None

@dataclass
class DeadBranch:
    """Rama muerta de código (if/else que nunca se ejecuta)."""
    position: UnifiedPosition
    confidence: float
    reason: str
    branch_type: str = "if"  # if, else, switch, etc.

@dataclass
class UnusedParameter:
    """Parámetro no utilizado en una función."""
    name: str
    position: UnifiedPosition
    function_name: str
    confidence: float
    reason: str

@dataclass
class RedundantAssignment:
    """Asignación redundante a una variable."""
    variable_name: str
    position: UnifiedPosition
    confidence: float
    reason: str

@dataclass
class DeadCodeStatistics:
    """Estadísticas de código muerto."""
    total_unused_variables: int = 0
    total_unused_functions: int = 0
    total_unused_classes: int = 0
    total_unused_imports: int = 0
    total_unreachable_code_blocks: int = 0
    total_dead_branches: int = 0
    total_unused_parameters: int = 0
    total_redundant_assignments: int = 0
    
    def get_total_issues(self) -> int:
        """Obtener el número total de issues."""
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
    
    def get_total_lines(self) -> int:
        """Estimar la cantidad de líneas de código muerto."""
        # Estimación simple: cada issue afecta a un número promedio de líneas
        return (
            self.total_unused_variables * 1 +
            self.total_unused_functions * 5 +
            self.total_unused_classes * 10 +
            self.total_unused_imports * 1 +
            self.total_unreachable_code_blocks * 3 +
            self.total_dead_branches * 2 +
            self.total_unused_parameters * 1 +
            self.total_redundant_assignments * 1
        )

@dataclass
class DeadCodeAnalysis:
    """Análisis de código muerto para un archivo."""
    file_path: Path
    language: ProgrammingLanguage
    unused_variables: List[UnusedVariable]
    unused_functions: List[UnusedFunction]
    unused_classes: List[UnusedClass]
    unused_imports: List[UnusedImport]
    unreachable_code: List[UnreachableCode]
    dead_branches: List[DeadBranch]
    unused_parameters: List[UnusedParameter]
    redundant_assignments: List[RedundantAssignment]
    statistics: DeadCodeStatistics
    execution_time_ms: int

@dataclass
class CrossModuleIssue:
    """Issue que involucra múltiples módulos."""
    issue_type: str  # unused_export, circular_dependency, etc.
    modules: List[Path]
    description: str
    confidence: float

@dataclass
class DependencyCycle:
    """Ciclo de dependencias entre módulos."""
    modules: List[Path]
    description: str
    cycle_path: List[str]

@dataclass
class ProjectDeadCodeAnalysis:
    """Análisis de código muerto para un proyecto completo."""
    project_path: Optional[Path]
    file_analyses: List[DeadCodeAnalysis]
    global_statistics: DeadCodeStatistics
    cross_module_issues: List[CrossModuleIssue]
    dependency_cycles: List[DependencyCycle]
    execution_time_ms: int