"""
Entidades de dominio para análisis de antipatrones usando IA.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from uuid import uuid4

from ..value_objects.programming_language import ProgrammingLanguage
from ..value_objects.source_position import SourcePosition


class AntipatternId:
    """ID único para antipatrón detectado."""
    
    def __init__(self, value: Optional[str] = None):
        self.value = value or str(uuid4())
    
    def __str__(self) -> str:
        return self.value
    
    def __eq__(self, other) -> bool:
        return isinstance(other, AntipatternId) and self.value == other.value
    
    def __hash__(self) -> int:
        return hash(self.value)


class AntipatternCategory(Enum):
    """Categorías de antipatrones."""
    ARCHITECTURAL = "architectural"
    DESIGN = "design"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    CONCURRENCY = "concurrency"
    LANGUAGE_SPECIFIC = "language_specific"
    CROSS_CUTTING = "cross_cutting"


class AntipatternType(Enum):
    """Tipos específicos de antipatrones."""
    
    # Architectural Antipatterns
    GOD_OBJECT = "god_object"
    SPAGHETTI_CODE = "spaghetti_code"
    LAVA_FLOW = "lava_flow"
    DEAD_CODE = "dead_code"
    GOLDEN_HAMMER = "golden_hammer"
    BIG_BALL_OF_MUD = "big_ball_of_mud"
    
    # Design Antipatterns
    SINGLETON_ABUSE = "singleton_abuse"
    FEATURE_ENVY = "feature_envy"
    DATA_CLUMPS = "data_clumps"
    PRIMITIVE_OBSESSION = "primitive_obsession"
    LONG_PARAMETER_LIST = "long_parameter_list"
    LARGE_CLASS = "large_class"
    LONG_METHOD = "long_method"
    
    # Performance Antipatterns
    N_PLUS_ONE_QUERY = "n_plus_one_query"
    MEMORY_LEAK = "memory_leak"
    INEFFICIENT_ALGORITHM = "inefficient_algorithm"
    STRING_CONCATENATION_IN_LOOP = "string_concatenation_in_loop"
    UNOPTIMIZED_LOOP = "unoptimized_loop"
    PREMATURE_OPTIMIZATION = "premature_optimization"
    
    # Security Antipatterns
    HARDCODED_SECRETS = "hardcoded_secrets"
    SQL_INJECTION = "sql_injection"
    XSS_VULNERABILITY = "xss_vulnerability"
    INSECURE_RANDOMNESS = "insecure_randomness"
    WEAK_CRYPTOGRAPHY = "weak_cryptography"
    PATH_TRAVERSAL = "path_traversal"
    
    # Concurrency Antipatterns
    RACE_CONDITION = "race_condition"
    DEADLOCK = "deadlock"
    THREAD_LEAK = "thread_leak"
    UNSYNCHRONIZED_ACCESS = "unsynchronized_access"
    
    # Language-Specific
    PYTHON_GIL_ISSUE = "python_gil_issue"
    JAVASCRIPT_CALLBACK_HELL = "javascript_callback_hell"
    RUST_BORROW_CHECKER_VIOLATION = "rust_borrow_checker_violation"
    TYPESCRIPT_ANY_ABUSE = "typescript_any_abuse"
    
    # Custom
    CUSTOM = "custom"


class AntipatternSeverity(Enum):
    """Severidad del antipatrón."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SecurityRisk(Enum):
    """Niveles de riesgo de seguridad."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PerformanceImpact(Enum):
    """Impacto en performance."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComplexityType(Enum):
    """Tipos de complejidad."""
    TIME_COMPLEXITY = "time_complexity"
    SPACE_COMPLEXITY = "space_complexity"
    CYCLOMATIC_COMPLEXITY = "cyclomatic_complexity"
    COGNITIVE_COMPLEXITY = "cognitive_complexity"


class AlgorithmicComplexity(Enum):
    """Complejidad algorítmica."""
    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    LINEAR = "O(n)"
    LINEAR_LOGARITHMIC = "O(n log n)"
    QUADRATIC = "O(n²)"
    CUBIC = "O(n³)"
    EXPONENTIAL = "O(2^n)"
    FACTORIAL = "O(n!)"
    UNKNOWN = "O(?)"


class ResponsibilityType(Enum):
    """Tipos de responsabilidades."""
    DATA_ACCESS = "data_access"
    BUSINESS_LOGIC = "business_logic"
    PRESENTATION = "presentation"
    VALIDATION = "validation"
    COMMUNICATION = "communication"
    CONFIGURATION = "configuration"
    LOGGING = "logging"
    ERROR_HANDLING = "error_handling"
    SECURITY = "security"
    PERFORMANCE = "performance"


class HighlightType(Enum):
    """Tipos de resaltado de código."""
    PROBLEM = "problem"
    SOLUTION = "solution"
    IMPORTANT = "important"
    EXAMPLE = "example"


class ExplanationStyle(Enum):
    """Estilos de explicación."""
    TECHNICAL = "technical"
    EDUCATIONAL = "educational"
    BUSINESS_FOCUSED = "business_focused"
    CONCISE = "concise"
    DETAILED = "detailed"


class TargetAudience(Enum):
    """Audiencia objetivo."""
    DEVELOPER = "developer"
    TECHNICAL_LEAD = "technical_lead"
    MANAGER = "manager"
    SECURITY_TEAM = "security_team"
    QUALITY_ASSURANCE = "quality_assurance"


class VerbosityLevel(Enum):
    """Nivel de verbosidad."""
    MINIMAL = "minimal"
    NORMAL = "normal"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


@dataclass
class SeverityIndicator:
    """Indicador de severidad."""
    indicator_type: str
    value: Any
    description: str = ""


@dataclass
class CodeHighlight:
    """Resaltado de código."""
    start_line: int
    end_line: int
    highlight_type: HighlightType
    message: str


@dataclass
class CodeExample:
    """Ejemplo de código."""
    language: ProgrammingLanguage
    code: str
    explanation: str
    highlights: List[CodeHighlight] = field(default_factory=list)


@dataclass
class ImpactAnalysis:
    """Análisis de impacto de un antipatrón."""
    performance_impact: PerformanceImpact
    security_impact: SecurityRisk
    maintainability_impact: str
    business_impact: str
    technical_debt_hours: float = 0.0
    affected_components: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class FixSuggestion:
    """Sugerencia de solución."""
    title: str
    description: str
    difficulty: str  # "easy", "medium", "hard"
    estimated_time_hours: float
    steps: List[str] = field(default_factory=list)
    code_examples: List[CodeExample] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class AntipatternExplanation:
    """Explicación detallada de un antipatrón."""
    pattern_type: AntipatternType
    summary: str
    detailed_explanation: str
    why_its_problematic: str
    potential_consequences: List[str]
    how_to_fix: str
    good_example: Optional[CodeExample] = None
    bad_example: Optional[CodeExample] = None
    references: List[str] = field(default_factory=list)
    confidence_explanation: str = ""


@dataclass
class DetectedAntipattern:
    """Antipatrón detectado."""
    id: AntipatternId = field(default_factory=AntipatternId)
    pattern_type: AntipatternType = AntipatternType.CUSTOM
    category: AntipatternCategory = AntipatternCategory.DESIGN
    severity: AntipatternSeverity = AntipatternSeverity.MEDIUM
    confidence: float = 0.0
    locations: List[SourcePosition] = field(default_factory=list)
    description: str = ""
    explanation: Optional[AntipatternExplanation] = None
    fix_suggestions: List[FixSuggestion] = field(default_factory=list)
    impact_analysis: Optional[ImpactAnalysis] = None
    detected_at: datetime = field(default_factory=datetime.now)
    evidence: List[str] = field(default_factory=list)
    severity_indicators: List[SeverityIndicator] = field(default_factory=list)


@dataclass
class AntipatternFeatures:
    """Features extraídas para detección de antipatrones."""
    file_path: Path
    language: ProgrammingLanguage
    
    # Features estructurales
    lines_of_code: int = 0
    methods_count: int = 0
    classes_count: int = 0
    functions_count: int = 0
    max_method_length: int = 0
    max_class_size: int = 0
    
    # Features de complejidad
    cyclomatic_complexity: float = 0.0
    cognitive_complexity: float = 0.0
    nesting_depth: int = 0
    
    # Features de acoplamiento
    import_count: int = 0
    external_dependencies: int = 0
    class_coupling: float = 0.0
    
    # Features de responsabilidad
    distinct_responsibilities: int = 0
    responsibility_types: List[ResponsibilityType] = field(default_factory=list)
    
    # Features de seguridad
    has_sql_operations: bool = False
    has_user_input: bool = False
    has_file_operations: bool = False
    has_network_operations: bool = False
    has_crypto_operations: bool = False
    
    # Features de performance
    has_loops: bool = False
    has_nested_loops: bool = False
    has_recursive_calls: bool = False
    algorithmic_complexity: AlgorithmicComplexity = AlgorithmicComplexity.LINEAR
    
    # Features adicionales
    custom_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassFeatures:
    """Features específicas de clases."""
    name: str
    declaration_location: SourcePosition
    lines_of_code: int
    method_count: int
    attribute_count: int
    total_complexity: float
    public_method_ratio: float
    inheritance_depth: int
    cohesion_score: float
    coupling_score: float
    methods: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SecurityFeatures:
    """Features de seguridad."""
    has_sql_string_concatenation: bool = False
    uses_user_input_in_queries: bool = False
    uses_parameterized_queries: bool = False
    has_dynamic_sql_construction: bool = False
    has_hardcoded_secrets: bool = False
    uses_weak_cryptography: bool = False
    has_input_validation: bool = False
    uses_secure_random: bool = False
    has_xss_vulnerabilities: bool = False
    validates_file_uploads: bool = False
    secret_patterns: List[str] = field(default_factory=list)
    crypto_algorithms: List[str] = field(default_factory=list)


@dataclass
class PerformanceFeatures:
    """Features de performance."""
    has_nested_loops: bool = False
    has_database_calls_in_loops: bool = False
    has_string_concatenation_in_loops: bool = False
    has_inefficient_data_structures: bool = False
    has_recursive_calls: bool = False
    has_memory_allocations_in_loops: bool = False
    algorithmic_complexity_estimate: AlgorithmicComplexity = AlgorithmicComplexity.LINEAR
    has_caching: bool = False
    has_lazy_loading: bool = False
    loop_complexity_scores: List[float] = field(default_factory=list)
    memory_patterns: List[str] = field(default_factory=list)


@dataclass
class AntipatternDetectionResult:
    """Resultado de detección de antipatrones."""
    file_path: Path
    language: ProgrammingLanguage
    detected_antipatterns: List[DetectedAntipattern] = field(default_factory=list)
    architectural_issues: List[DetectedAntipattern] = field(default_factory=list)
    design_issues: List[DetectedAntipattern] = field(default_factory=list)
    performance_issues: List[DetectedAntipattern] = field(default_factory=list)
    security_issues: List[DetectedAntipattern] = field(default_factory=list)
    confidence_scores: Dict[AntipatternType, float] = field(default_factory=dict)
    explanations: List[AntipatternExplanation] = field(default_factory=list)
    detection_time_ms: int = 0
    quality_score: float = 0.0


@dataclass
class AntipatternHotspot:
    """Hotspot de antipatrones."""
    location: SourcePosition
    antipattern_count: int
    severity_score: float
    pattern_types: List[AntipatternType]
    description: str


@dataclass
class AntipatternTrend:
    """Tendencia de antipatrones."""
    pattern_type: AntipatternType
    frequency: int
    trend_direction: str  # "increasing", "decreasing", "stable"
    locations: List[SourcePosition]


@dataclass
class RemediationPriority:
    """Prioridad de remediación."""
    antipattern: DetectedAntipattern
    priority_score: float
    business_impact: str
    technical_impact: str
    effort_estimate: float
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ProjectAntipatternAnalysis:
    """Análisis de antipatrones a nivel de proyecto."""
    project_path: Optional[Path] = None
    file_antipatterns: List[AntipatternDetectionResult] = field(default_factory=list)
    project_level_antipatterns: List[DetectedAntipattern] = field(default_factory=list)
    architectural_analysis: Dict[str, Any] = field(default_factory=dict)
    cross_file_antipatterns: List[DetectedAntipattern] = field(default_factory=list)
    hotspots: List[AntipatternHotspot] = field(default_factory=list)
    trends: List[AntipatternTrend] = field(default_factory=list)
    remediation_priorities: List[RemediationPriority] = field(default_factory=list)
    detection_time_ms: int = 0
    overall_quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AntipatternDetectionConfig:
    """Configuración para detección de antipatrones."""
    enable_ensemble_detection: bool = True
    enable_contextual_analysis: bool = True
    enable_explanation_generation: bool = True
    confidence_threshold: float = 0.7
    max_patterns_per_analysis: int = 50
    enable_cross_language_detection: bool = True
    
    # Pesos del ensemble
    model_ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "structural": 0.3,
        "semantic": 0.25,
        "behavioral": 0.25,
        "contextual": 0.2
    })
    
    # Thresholds por categoría
    category_specific_thresholds: Dict[AntipatternCategory, float] = field(default_factory=lambda: {
        AntipatternCategory.SECURITY: 0.6,
        AntipatternCategory.PERFORMANCE: 0.7,
        AntipatternCategory.ARCHITECTURAL: 0.8,
        AntipatternCategory.DESIGN: 0.7,
        AntipatternCategory.MAINTAINABILITY: 0.75
    })
    
    # Configuración de explicaciones
    explanation_style: ExplanationStyle = ExplanationStyle.TECHNICAL
    target_audience: TargetAudience = TargetAudience.DEVELOPER
    verbosity_level: VerbosityLevel = VerbosityLevel.NORMAL
    include_examples: bool = True
    include_fix_suggestions: bool = True


@dataclass
class ExplanationConfig:
    """Configuración para generación de explicaciones."""
    explanation_style: ExplanationStyle = ExplanationStyle.TECHNICAL
    include_examples: bool = True
    include_fix_suggestions: bool = True
    include_impact_analysis: bool = True
    target_audience: TargetAudience = TargetAudience.DEVELOPER
    verbosity_level: VerbosityLevel = VerbosityLevel.NORMAL
    max_explanation_length: int = 1000
    include_code_snippets: bool = True


@dataclass
class ModelPrediction:
    """Predicción de un modelo."""
    pattern_type: AntipatternType
    confidence: float
    evidence: List[str]
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Predicción del ensemble."""
    final_prediction: AntipatternType
    confidence: float
    model_predictions: List[ModelPrediction]
    consensus_score: float
    uncertainty: float
