"""
Entidades del dominio para análisis de métricas de código.

Este módulo contiene todas las entidades que representan los resultados
del análisis de complejidad, métricas de Halstead, cohesión, acoplamiento y calidad.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum
import uuid

from ..value_objects.programming_language import ProgrammingLanguage
from .dead_code_analysis import SourceRange, SourcePosition


class ComplexityLevel(Enum):
    """Niveles de complejidad."""
    LOW = "low"           # 1-10
    MEDIUM = "medium"     # 11-20
    HIGH = "high"         # 21-50
    VERY_HIGH = "very_high"  # >50


class QualityGateStatus(Enum):
    """Estados de quality gate."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_EVALUATED = "not_evaluated"


class CodeSmellType(Enum):
    """Tipos de code smells."""
    HIGH_COMPLEXITY = "high_complexity"
    LONG_FUNCTION = "long_function"
    LARGE_CLASS = "large_class"
    LOW_COHESION = "low_cohesion"
    HIGH_COUPLING = "high_coupling"
    DEEP_INHERITANCE = "deep_inheritance"
    MANY_PARAMETERS = "many_parameters"
    DUPLICATED_CODE = "duplicated_code"
    DEAD_CODE = "dead_code"
    MAGIC_NUMBERS = "magic_numbers"
    LONG_PARAMETER_LIST = "long_parameter_list"
    FEATURE_ENVY = "feature_envy"
    DATA_CLUMPS = "data_clumps"
    GOD_CLASS = "god_class"
    SHOTGUN_SURGERY = "shotgun_surgery"


class CodeSmellSeverity(Enum):
    """Severidad de code smells."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Tipos de métricas."""
    COMPLEXITY = "complexity"
    HALSTEAD = "halstead"
    SIZE = "size"
    COHESION = "cohesion"
    COUPLING = "coupling"
    QUALITY = "quality"
    TECHNICAL_DEBT = "technical_debt"


@dataclass
class ComplexityThresholds:
    """Umbrales de complejidad configurables."""
    cyclomatic_low: int = 10
    cyclomatic_medium: int = 20
    cyclomatic_high: int = 50
    cognitive_low: int = 15
    cognitive_medium: int = 25
    cognitive_high: int = 50
    nesting_depth_max: int = 5
    function_length_max: int = 50
    class_length_max: int = 500
    
    def get_complexity_level(self, cyclomatic: int) -> ComplexityLevel:
        """Obtiene nivel de complejidad basado en valor ciclomático."""
        if cyclomatic <= self.cyclomatic_low:
            return ComplexityLevel.LOW
        elif cyclomatic <= self.cyclomatic_medium:
            return ComplexityLevel.MEDIUM
        elif cyclomatic <= self.cyclomatic_high:
            return ComplexityLevel.HIGH
        else:
            return ComplexityLevel.VERY_HIGH
    
    def get_cognitive_level(self, cognitive: int) -> ComplexityLevel:
        """Obtiene nivel de complejidad cognitiva."""
        if cognitive <= self.cognitive_low:
            return ComplexityLevel.LOW
        elif cognitive <= self.cognitive_medium:
            return ComplexityLevel.MEDIUM
        elif cognitive <= self.cognitive_high:
            return ComplexityLevel.HIGH
        else:
            return ComplexityLevel.VERY_HIGH


@dataclass
class ComplexityMetrics:
    """Métricas de complejidad."""
    cyclomatic_complexity: int = 1
    modified_cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    essential_complexity: int = 1
    max_nesting_depth: int = 0
    average_nesting_depth: float = 0.0
    complexity_density: float = 0.0  # Complejidad por LOC
    decision_points: int = 0
    loop_count: int = 0
    condition_count: int = 0
    
    def get_complexity_level(self, thresholds: ComplexityThresholds) -> ComplexityLevel:
        """Obtiene nivel de complejidad general."""
        return thresholds.get_complexity_level(self.cyclomatic_complexity)
    
    def get_cognitive_level(self, thresholds: ComplexityThresholds) -> ComplexityLevel:
        """Obtiene nivel de complejidad cognitiva."""
        return thresholds.get_cognitive_level(self.cognitive_complexity)


@dataclass
class HalsteadMetrics:
    """Métricas de Halstead."""
    distinct_operators: int = 0       # n1
    distinct_operands: int = 0        # n2
    total_operators: int = 0          # N1
    total_operands: int = 0           # N2
    vocabulary: int = 0               # n = n1 + n2
    length: int = 0                   # N = N1 + N2
    calculated_length: float = 0.0    # N̂ = n1*log2(n1) + n2*log2(n2)
    volume: float = 0.0               # V = N * log2(n)
    difficulty: float = 0.0           # D = (n1/2) * (N2/n2)
    effort: float = 0.0               # E = D * V
    time: float = 0.0                 # T = E / 18
    bugs: float = 0.0                 # B = V / 3000
    level: float = 0.0                # L = 1 / D
    intelligence: float = 0.0         # I = V / D
    
    def calculate_derived_metrics(self) -> None:
        """Calcula métricas derivadas de Halstead."""
        self.vocabulary = self.distinct_operators + self.distinct_operands
        self.length = self.total_operators + self.total_operands
        
        if self.distinct_operators > 0 and self.distinct_operands > 0:
            import math
            self.calculated_length = (
                self.distinct_operators * math.log2(self.distinct_operators) +
                self.distinct_operands * math.log2(self.distinct_operands)
            )
        
        if self.vocabulary > 0:
            import math
            self.volume = self.length * math.log2(self.vocabulary)
        
        if self.distinct_operands > 0:
            self.difficulty = (self.distinct_operators / 2.0) * (self.total_operands / self.distinct_operands)
        
        self.effort = self.difficulty * self.volume
        self.time = self.effort / 18.0  # Stroud number
        self.bugs = self.volume / 3000.0  # Empirical formula
        
        if self.difficulty > 0:
            self.level = 1.0 / self.difficulty
            self.intelligence = self.volume / self.difficulty


@dataclass
class SizeMetrics:
    """Métricas de tamaño."""
    total_lines: int = 0
    logical_lines_of_code: int = 0  # SLOC
    comment_lines: int = 0
    blank_lines: int = 0
    source_lines: int = 0  # Total - comments - blank
    function_count: int = 0
    class_count: int = 0
    method_count: int = 0
    average_function_length: float = 0.0
    average_class_length: float = 0.0
    max_function_length: int = 0
    max_class_length: int = 0
    
    def calculate_derived_metrics(self) -> None:
        """Calcula métricas derivadas de tamaño."""
        self.source_lines = self.total_lines - self.comment_lines - self.blank_lines
        
        if self.function_count > 0:
            self.average_function_length = self.logical_lines_of_code / self.function_count
        
        if self.class_count > 0:
            self.average_class_length = self.logical_lines_of_code / self.class_count
    
    def get_comment_ratio(self) -> float:
        """Obtiene ratio de comentarios."""
        return self.comment_lines / self.total_lines if self.total_lines > 0 else 0.0


@dataclass
class CohesionMetrics:
    """Métricas de cohesión."""
    average_lcom: float = 0.0  # Lack of Cohesion of Methods
    average_tcc: float = 0.0   # Tight Class Cohesion
    average_lcc: float = 0.0   # Loose Class Cohesion
    class_count: int = 0
    highly_cohesive_classes: int = 0
    poorly_cohesive_classes: int = 0
    
    def get_cohesion_level(self) -> str:
        """Obtiene nivel general de cohesión."""
        if self.average_lcom <= 0.3:
            return "high"
        elif self.average_lcom <= 0.6:
            return "medium"
        else:
            return "low"


@dataclass
class CouplingMetrics:
    """Métricas de acoplamiento."""
    average_cbo: float = 0.0   # Coupling Between Objects
    average_rfc: float = 0.0   # Response for Class
    average_dit: float = 0.0   # Depth of Inheritance Tree
    average_noc: float = 0.0   # Number of Children
    total_dependencies: int = 0
    circular_dependencies: int = 0
    max_inheritance_depth: int = 0
    interface_usage: int = 0
    
    def get_coupling_level(self) -> str:
        """Obtiene nivel general de acoplamiento."""
        if self.average_cbo <= 5.0:
            return "low"
        elif self.average_cbo <= 10.0:
            return "medium"
        else:
            return "high"


@dataclass
class CodeSmell:
    """Representación de un code smell."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    smell_type: CodeSmellType = CodeSmellType.HIGH_COMPLEXITY
    severity: CodeSmellSeverity = CodeSmellSeverity.MEDIUM
    description: str = ""
    location: Optional[SourceRange] = None
    affected_function: Optional[str] = None
    affected_class: Optional[str] = None
    metric_value: float = 0.0
    threshold_value: float = 0.0
    estimated_fix_time_minutes: float = 0.0
    impact_on_maintainability: float = 0.0
    suggestions: List[str] = field(default_factory=list)
    
    def get_priority_score(self) -> float:
        """Calcula score de prioridad del smell."""
        severity_weights = {
            CodeSmellSeverity.LOW: 1.0,
            CodeSmellSeverity.MEDIUM: 2.0,
            CodeSmellSeverity.HIGH: 4.0,
            CodeSmellSeverity.CRITICAL: 8.0
        }
        
        base_score = severity_weights.get(self.severity, 2.0)
        impact_factor = self.impact_on_maintainability
        
        return base_score * (1.0 + impact_factor)


@dataclass
class TechnicalDebtEstimate:
    """Estimación de deuda técnica."""
    total_minutes: float = 0.0
    total_hours: float = 0.0
    estimated_cost: float = 0.0  # En USD
    code_smells: List[CodeSmell] = field(default_factory=list)
    debt_ratio: float = 0.0  # Deuda por LOC
    sqale_rating: str = "A"  # SQALE rating
    maintainability_debt: float = 0.0
    reliability_debt: float = 0.0
    security_debt: float = 0.0
    
    def calculate_derived_metrics(self, hourly_rate: float = 75.0) -> None:
        """Calcula métricas derivadas de deuda técnica."""
        self.total_hours = self.total_minutes / 60.0
        self.estimated_cost = self.total_hours * hourly_rate
        
        # Calcular SQALE rating
        if self.debt_ratio <= 0.05:
            self.sqale_rating = "A"
        elif self.debt_ratio <= 0.1:
            self.sqale_rating = "B"
        elif self.debt_ratio <= 0.2:
            self.sqale_rating = "C"
        elif self.debt_ratio <= 0.5:
            self.sqale_rating = "D"
        else:
            self.sqale_rating = "E"
    
    def get_smells_by_severity(self, severity: CodeSmellSeverity) -> List[CodeSmell]:
        """Obtiene smells por severidad."""
        return [smell for smell in self.code_smells if smell.severity == severity]
    
    def get_highest_priority_smells(self, limit: int = 10) -> List[CodeSmell]:
        """Obtiene smells de mayor prioridad."""
        sorted_smells = sorted(self.code_smells, key=lambda s: s.get_priority_score(), reverse=True)
        return sorted_smells[:limit]


@dataclass
class QualityGateResult:
    """Resultado de evaluación de quality gate."""
    gate_name: str
    metric_type: MetricType
    threshold_value: float
    actual_value: float
    status: QualityGateStatus
    message: str = ""
    
    def is_passing(self) -> bool:
        """Verifica si el gate está pasando."""
        return self.status == QualityGateStatus.PASSED


@dataclass
class QualityMetrics:
    """Métricas de calidad general."""
    maintainability_index: float = 0.0  # 0-100
    technical_debt_hours: float = 0.0
    technical_debt_cost: float = 0.0
    code_smells_count: int = 0
    quality_gate_status: QualityGateStatus = QualityGateStatus.NOT_EVALUATED
    quality_gate_results: List[QualityGateResult] = field(default_factory=list)
    testability_score: float = 0.0     # 0-100
    readability_score: float = 0.0     # 0-100
    reliability_score: float = 0.0     # 0-100
    security_score: float = 0.0        # 0-100
    performance_score: float = 0.0     # 0-100
    
    def get_overall_quality_grade(self) -> str:
        """Obtiene grado general de calidad."""
        if self.maintainability_index >= 85:
            return "A"
        elif self.maintainability_index >= 70:
            return "B"
        elif self.maintainability_index >= 50:
            return "C"
        elif self.maintainability_index >= 25:
            return "D"
        else:
            return "E"
    
    def get_passed_gates_percentage(self) -> float:
        """Obtiene porcentaje de gates que pasaron."""
        if not self.quality_gate_results:
            return 0.0
        
        passed = sum(1 for gate in self.quality_gate_results if gate.is_passing())
        return (passed / len(self.quality_gate_results)) * 100.0


@dataclass
class FunctionMetrics:
    """Métricas específicas de función."""
    name: str
    location: SourceRange
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    lines_of_code: int = 0
    parameter_count: int = 0
    local_variable_count: int = 0
    return_points: int = 1
    nested_depth: int = 0
    halstead_metrics: Optional['HalsteadMetrics'] = None
    maintainability_index: float = 0.0
    complexity_level: ComplexityLevel = ComplexityLevel.LOW
    smells: List[CodeSmell] = field(default_factory=list)
    
    def calculate_derived_metrics(self, thresholds: ComplexityThresholds) -> None:
        """Calcula métricas derivadas."""
        self.complexity_level = thresholds.get_complexity_level(self.cyclomatic_complexity)
        
        # Detectar smells básicos
        if self.cyclomatic_complexity > thresholds.cyclomatic_medium:
            self.smells.append(CodeSmell(
                smell_type=CodeSmellType.HIGH_COMPLEXITY,
                severity=CodeSmellSeverity.HIGH if self.cyclomatic_complexity > thresholds.cyclomatic_high else CodeSmellSeverity.MEDIUM,
                description=f"Function has high cyclomatic complexity: {self.cyclomatic_complexity}",
                affected_function=self.name,
                metric_value=self.cyclomatic_complexity,
                threshold_value=thresholds.cyclomatic_medium,
                estimated_fix_time_minutes=self.cyclomatic_complexity * 3.0
            ))
        
        if self.lines_of_code > thresholds.function_length_max:
            self.smells.append(CodeSmell(
                smell_type=CodeSmellType.LONG_FUNCTION,
                severity=CodeSmellSeverity.MEDIUM,
                description=f"Function is too long: {self.lines_of_code} lines",
                affected_function=self.name,
                metric_value=self.lines_of_code,
                threshold_value=thresholds.function_length_max,
                estimated_fix_time_minutes=(self.lines_of_code - thresholds.function_length_max) * 0.5
            ))
        
        if self.parameter_count > 7:
            self.smells.append(CodeSmell(
                smell_type=CodeSmellType.MANY_PARAMETERS,
                severity=CodeSmellSeverity.MEDIUM,
                description=f"Function has too many parameters: {self.parameter_count}",
                affected_function=self.name,
                metric_value=self.parameter_count,
                threshold_value=7,
                estimated_fix_time_minutes=15.0
            ))


@dataclass
class ClassMetrics:
    """Métricas específicas de clase."""
    name: str
    location: SourceRange
    lines_of_code: int = 0
    method_count: int = 0
    attribute_count: int = 0
    public_method_count: int = 0
    private_method_count: int = 0
    inheritance_depth: int = 0
    number_of_children: int = 0
    coupling_between_objects: int = 0
    response_for_class: int = 0
    lack_of_cohesion: float = 0.0
    weighted_methods_per_class: int = 0
    complexity_level: ComplexityLevel = ComplexityLevel.LOW
    smells: List[CodeSmell] = field(default_factory=list)
    
    def calculate_derived_metrics(self, thresholds: ComplexityThresholds) -> None:
        """Calcula métricas derivadas de clase."""
        # Determinar nivel de complejidad basado en WMC
        if self.weighted_methods_per_class <= 10:
            self.complexity_level = ComplexityLevel.LOW
        elif self.weighted_methods_per_class <= 20:
            self.complexity_level = ComplexityLevel.MEDIUM
        elif self.weighted_methods_per_class <= 50:
            self.complexity_level = ComplexityLevel.HIGH
        else:
            self.complexity_level = ComplexityLevel.VERY_HIGH
        
        # Detectar smells de clase
        if self.lines_of_code > thresholds.class_length_max:
            self.smells.append(CodeSmell(
                smell_type=CodeSmellType.LARGE_CLASS,
                severity=CodeSmellSeverity.HIGH if self.lines_of_code > thresholds.class_length_max * 2 else CodeSmellSeverity.MEDIUM,
                description=f"Class is too large: {self.lines_of_code} lines",
                affected_class=self.name,
                metric_value=self.lines_of_code,
                threshold_value=thresholds.class_length_max,
                estimated_fix_time_minutes=(self.lines_of_code - thresholds.class_length_max) * 0.3
            ))
        
        if self.lack_of_cohesion > 0.8:
            self.smells.append(CodeSmell(
                smell_type=CodeSmellType.LOW_COHESION,
                severity=CodeSmellSeverity.HIGH,
                description=f"Class has low cohesion: LCOM={self.lack_of_cohesion:.2f}",
                affected_class=self.name,
                metric_value=self.lack_of_cohesion,
                threshold_value=0.8,
                estimated_fix_time_minutes=45.0
            ))
        
        if self.coupling_between_objects > 10:
            self.smells.append(CodeSmell(
                smell_type=CodeSmellType.HIGH_COUPLING,
                severity=CodeSmellSeverity.MEDIUM,
                description=f"Class has high coupling: CBO={self.coupling_between_objects}",
                affected_class=self.name,
                metric_value=self.coupling_between_objects,
                threshold_value=10,
                estimated_fix_time_minutes=30.0
            ))


@dataclass
class ComplexityDistribution:
    """Distribución de complejidad."""
    low_complexity_functions: int = 0
    medium_complexity_functions: int = 0
    high_complexity_functions: int = 0
    very_high_complexity_functions: int = 0
    complexity_histogram: List[int] = field(default_factory=list)
    
    def get_total_functions(self) -> int:
        """Obtiene total de funciones."""
        return (self.low_complexity_functions + self.medium_complexity_functions + 
                self.high_complexity_functions + self.very_high_complexity_functions)
    
    def get_high_complexity_percentage(self) -> float:
        """Obtiene porcentaje de funciones de alta complejidad."""
        total = self.get_total_functions()
        if total == 0:
            return 0.0
        
        high_complexity = self.high_complexity_functions + self.very_high_complexity_functions
        return (high_complexity / total) * 100.0


@dataclass
class MetricsConfig:
    """Configuración para el sistema de métricas."""
    enable_cyclomatic_complexity: bool = True
    enable_cognitive_complexity: bool = True
    enable_halstead_metrics: bool = True
    enable_cohesion_metrics: bool = True
    enable_coupling_metrics: bool = True
    enable_size_metrics: bool = True
    enable_quality_metrics: bool = True
    complexity_thresholds: ComplexityThresholds = field(default_factory=ComplexityThresholds)
    calculate_per_function: bool = True
    calculate_per_class: bool = True
    calculate_technical_debt: bool = True
    enable_quality_gates: bool = True
    language_specific_configs: Dict[ProgrammingLanguage, Dict[str, Any]] = field(default_factory=dict)
    
    def get_language_config(self, language: ProgrammingLanguage) -> Dict[str, Any]:
        """Obtiene configuración específica del lenguaje."""
        return self.language_specific_configs.get(language, {})


@dataclass
class QualityGateDefinition:
    """Definición de quality gate."""
    name: str
    metric_type: MetricType
    threshold_value: float
    comparison_operator: str  # "<=", ">=", "<", ">", "==", "!="
    severity: CodeSmellSeverity = CodeSmellSeverity.MEDIUM
    enabled: bool = True
    description: str = ""
    
    def evaluate(self, actual_value: float) -> QualityGateResult:
        """Evalúa el quality gate."""
        passed = self._compare_values(actual_value, self.threshold_value)
        
        status = QualityGateStatus.PASSED if passed else QualityGateStatus.FAILED
        
        message = (f"{self.name}: {actual_value:.2f} {self.comparison_operator} {self.threshold_value:.2f}" +
                  f" - {'PASSED' if passed else 'FAILED'}")
        
        return QualityGateResult(
            gate_name=self.name,
            metric_type=self.metric_type,
            threshold_value=self.threshold_value,
            actual_value=actual_value,
            status=status,
            message=message
        )
    
    def _compare_values(self, actual: float, threshold: float) -> bool:
        """Compara valores según el operador."""
        if self.comparison_operator == "<=":
            return actual <= threshold
        elif self.comparison_operator == ">=":
            return actual >= threshold
        elif self.comparison_operator == "<":
            return actual < threshold
        elif self.comparison_operator == ">":
            return actual > threshold
        elif self.comparison_operator == "==":
            return abs(actual - threshold) < 0.001
        elif self.comparison_operator == "!=":
            return abs(actual - threshold) >= 0.001
        else:
            return False


@dataclass
class CodeMetrics:
    """Métricas completas de código para un archivo."""
    file_path: Path = Path(".")
    language: ProgrammingLanguage = ProgrammingLanguage.UNKNOWN
    complexity_metrics: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    halstead_metrics: HalsteadMetrics = field(default_factory=HalsteadMetrics)
    size_metrics: SizeMetrics = field(default_factory=SizeMetrics)
    cohesion_metrics: CohesionMetrics = field(default_factory=CohesionMetrics)
    coupling_metrics: CouplingMetrics = field(default_factory=CouplingMetrics)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    function_metrics: List[FunctionMetrics] = field(default_factory=list)
    class_metrics: List[ClassMetrics] = field(default_factory=list)
    complexity_distribution: ComplexityDistribution = field(default_factory=ComplexityDistribution)
    technical_debt: TechnicalDebtEstimate = field(default_factory=TechnicalDebtEstimate)
    overall_quality_score: float = 0.0
    calculation_time_ms: int = 0
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_all_smells(self) -> List[CodeSmell]:
        """Obtiene todos los code smells detectados."""
        all_smells = []
        all_smells.extend(self.technical_debt.code_smells)
        
        for func_metric in self.function_metrics:
            all_smells.extend(func_metric.smells)
        
        for class_metric in self.class_metrics:
            all_smells.extend(class_metric.smells)
        
        return all_smells
    
    def get_hotspot_functions(self, limit: int = 10) -> List[FunctionMetrics]:
        """Obtiene funciones más problemáticas."""
        return sorted(
            self.function_metrics,
            key=lambda f: f.cyclomatic_complexity * f.lines_of_code,
            reverse=True
        )[:limit]
    
    def get_hotspot_classes(self, limit: int = 10) -> List[ClassMetrics]:
        """Obtiene clases más problemáticas."""
        return sorted(
            self.class_metrics,
            key=lambda c: c.weighted_methods_per_class * c.lines_of_code,
            reverse=True
        )[:limit]
    
    def calculate_overall_score(self) -> float:
        """Calcula score general de calidad."""
        weights = {
            'maintainability': 0.3,
            'testability': 0.2,
            'readability': 0.2,
            'reliability': 0.15,
            'security': 0.1,
            'performance': 0.05
        }
        
        score = (
            self.quality_metrics.maintainability_index * weights['maintainability'] +
            self.quality_metrics.testability_score * weights['testability'] +
            self.quality_metrics.readability_score * weights['readability'] +
            self.quality_metrics.reliability_score * weights['reliability'] +
            self.quality_metrics.security_score * weights['security'] +
            self.quality_metrics.performance_score * weights['performance']
        )
        
        return min(100.0, max(0.0, score))


@dataclass
class ProjectMetrics:
    """Métricas agregadas de proyecto completo."""
    project_path: Optional[Path] = None
    file_metrics: List[CodeMetrics] = field(default_factory=list)
    aggregated_complexity: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    aggregated_halstead: HalsteadMetrics = field(default_factory=HalsteadMetrics)
    aggregated_size: SizeMetrics = field(default_factory=SizeMetrics)
    project_cohesion: CohesionMetrics = field(default_factory=CohesionMetrics)
    project_coupling: CouplingMetrics = field(default_factory=CouplingMetrics)
    project_quality: QualityMetrics = field(default_factory=QualityMetrics)
    hotspots: List['ComplexityHotspot'] = field(default_factory=list)
    quality_distribution: 'QualityDistribution' = field(default_factory=lambda: QualityDistribution())
    technical_debt_estimate: TechnicalDebtEstimate = field(default_factory=TechnicalDebtEstimate)
    maintainability_index: float = 0.0
    calculation_time_ms: int = 0
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_total_files(self) -> int:
        """Obtiene total de archivos analizados."""
        return len(self.file_metrics)
    
    def get_files_with_high_complexity(self) -> List[CodeMetrics]:
        """Obtiene archivos con alta complejidad."""
        return [
            metrics for metrics in self.file_metrics
            if metrics.complexity_metrics.cyclomatic_complexity > 20
        ]
    
    def get_worst_quality_files(self, limit: int = 10) -> List[CodeMetrics]:
        """Obtiene archivos con peor calidad."""
        return sorted(
            self.file_metrics,
            key=lambda m: m.quality_metrics.maintainability_index
        )[:limit]
    
    def get_project_health_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de salud del proyecto."""
        return {
            "total_files": self.get_total_files(),
            "high_complexity_files": len(self.get_files_with_high_complexity()),
            "overall_maintainability": self.maintainability_index,
            "total_technical_debt_hours": self.technical_debt_estimate.total_hours,
            "total_code_smells": len(self.technical_debt_estimate.code_smells),
            "quality_gate_status": self.project_quality.quality_gate_status.value,
            "sqale_rating": self.technical_debt_estimate.sqale_rating
        }


@dataclass
class ComplexityHotspot:
    """Hotspot de complejidad en el proyecto."""
    location: SourceRange
    hotspot_type: str  # "function", "class", "module"
    name: str
    cyclomatic_complexity: int
    cognitive_complexity: int
    lines_of_code: int
    severity: CodeSmellSeverity
    impact_score: float
    suggested_actions: List[str] = field(default_factory=list)
    
    def get_risk_level(self) -> str:
        """Obtiene nivel de riesgo del hotspot."""
        if self.impact_score >= 80:
            return "critical"
        elif self.impact_score >= 60:
            return "high"
        elif self.impact_score >= 40:
            return "medium"
        else:
            return "low"


@dataclass
class QualityDistribution:
    """Distribución de calidad en el proyecto."""
    excellent_files: int = 0    # MI >= 85
    good_files: int = 0         # MI >= 70
    average_files: int = 0      # MI >= 50
    poor_files: int = 0         # MI >= 25
    very_poor_files: int = 0    # MI < 25
    
    def get_total_files(self) -> int:
        """Obtiene total de archivos."""
        return (self.excellent_files + self.good_files + self.average_files + 
                self.poor_files + self.very_poor_files)
    
    def get_quality_percentage(self, level: str) -> float:
        """Obtiene porcentaje de archivos por nivel de calidad."""
        total = self.get_total_files()
        if total == 0:
            return 0.0
        
        counts = {
            "excellent": self.excellent_files,
            "good": self.good_files,
            "average": self.average_files,
            "poor": self.poor_files,
            "very_poor": self.very_poor_files
        }
        
        return (counts.get(level, 0) / total) * 100.0


@dataclass
class MetricsAnalysisResult:
    """Resultado completo del análisis de métricas."""
    file_metrics: CodeMetrics
    calculation_success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    performance_info: Dict[str, Any] = field(default_factory=dict)
    
    def add_warning(self, message: str) -> None:
        """Añade warning al resultado."""
        self.warnings.append(message)
    
    def is_successful(self) -> bool:
        """Verifica si el análisis fue exitoso."""
        return self.calculation_success and self.error_message is None


@dataclass
class MetricsTrend:
    """Tendencia de métricas a lo largo del tiempo."""
    metric_name: str
    time_points: List[datetime] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    trend_direction: str = "stable"  # "improving", "degrading", "stable"
    trend_strength: float = 0.0  # 0.0 - 1.0
    
    def add_data_point(self, timestamp: datetime, value: float) -> None:
        """Añade punto de datos a la tendencia."""
        self.time_points.append(timestamp)
        self.values.append(value)
        self._calculate_trend()
    
    def _calculate_trend(self) -> None:
        """Calcula dirección y fuerza de la tendencia."""
        if len(self.values) < 2:
            return
        
        # Cálculo simple de tendencia
        recent_avg = sum(self.values[-3:]) / len(self.values[-3:])
        older_avg = sum(self.values[:3]) / len(self.values[:3]) if len(self.values) >= 6 else self.values[0]
        
        if recent_avg > older_avg * 1.1:
            self.trend_direction = "improving"
            self.trend_strength = min(1.0, (recent_avg - older_avg) / older_avg)
        elif recent_avg < older_avg * 0.9:
            self.trend_direction = "degrading"
            self.trend_strength = min(1.0, (older_avg - recent_avg) / older_avg)
        else:
            self.trend_direction = "stable"
            self.trend_strength = 0.1
