"""
Modelos principales para el motor de reglas estáticas.

Este módulo define las estructuras de datos fundamentales para el sistema
de reglas, incluyendo reglas, resultados, violaciones y configuraciones.
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Protocol
from dataclasses import dataclass, field
from uuid import uuid4

from pydantic import BaseModel, Field, validator, ConfigDict

from ...parsers.unified.unified_ast import UnifiedAST, UnifiedNode
from ...parsers.universal import ProgrammingLanguage

logger = logging.getLogger(__name__)


class RuleId(str):
    """Identificador único para una regla."""
    
    def __new__(cls, value: str):
        if not value or not value.strip():
            value = f"rule-{uuid4().hex[:8]}"
        return super().__new__(cls, value)


class RuleCategory(str, Enum):
    """Categorías de reglas."""
    
    # Code Quality
    BEST_PRACTICES = "best_practices"
    CODE_SMELL = "code_smell"
    MAINTAINABILITY = "maintainability"
    READABILITY = "readability"
    
    # Security
    SECURITY = "security"
    VULNERABILITY = "vulnerability"
    CRYPTOGRAPHIC_ISSUES = "cryptographic_issues"
    INPUT_VALIDATION = "input_validation"
    
    # Performance
    PERFORMANCE = "performance"
    MEMORY_USAGE = "memory_usage"
    ALGORITHMIC_COMPLEXITY = "algorithmic_complexity"
    RESOURCE_LEAKS = "resource_leaks"
    
    # Reliability
    BUG_PRONE = "bug_prone"
    ERROR_HANDLING = "error_handling"
    NULL_POINTER = "null_pointer"
    CONCURRENCY_ISSUES = "concurrency_issues"
    
    # Design
    DESIGN_PATTERNS = "design_patterns"
    ARCHITECTURE = "architecture"
    SOLID = "solid"
    DRY = "dry"
    
    # Language Specific
    PYTHON_SPECIFIC = "python_specific"
    JAVASCRIPT_SPECIFIC = "javascript_specific"
    TYPESCRIPT_SPECIFIC = "typescript_specific"
    RUST_SPECIFIC = "rust_specific"
    
    # Cross-Language
    CROSS_LANGUAGE = "cross_language"
    MIGRATION = "migration"
    CONSISTENCY = "consistency"


class RuleSeverity(str, Enum):
    """Niveles de severidad de las reglas."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ParameterType(str, Enum):
    """Tipos de parámetros de configuración."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"


@dataclass
class ConfigParameter:
    """Parámetro de configuración para una regla."""
    name: str
    parameter_type: ParameterType
    default_value: Any
    description: str
    validation: Optional[Dict[str, Any]] = None
    enum_values: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.parameter_type == ParameterType.ENUM and not self.enum_values:
            raise ValueError("Enum parameters must specify enum_values")


@dataclass
class ThresholdConfig:
    """Configuración de umbrales para una regla."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    error_threshold: Optional[float] = None


@dataclass
class ExclusionPattern:
    """Patrón de exclusión para una regla."""
    pattern: str
    description: str
    pattern_type: str = "glob"  # glob, regex, path


@dataclass
class RuleConfiguration:
    """Configuración de una regla."""
    parameters: Dict[str, ConfigParameter] = field(default_factory=dict)
    thresholds: Dict[str, ThresholdConfig] = field(default_factory=dict)
    exclusions: List[ExclusionPattern] = field(default_factory=list)
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleMetadata:
    """Metadatos de una regla."""
    author: str = "CodeAnt"
    created_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cwe_ids: List[str] = field(default_factory=list)
    owasp_categories: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    difficulty_to_fix: str = "medium"  # trivial, easy, medium, hard, expert
    false_positive_rate: float = 0.05


class PatternImplementation(BaseModel):
    """Implementación basada en patrones AST."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    ast_pattern: Dict[str, Any]  # Será definido en pattern_models.py
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    actions: List[Dict[str, Any]] = Field(default_factory=list)


class QueryImplementation(BaseModel):
    """Implementación basada en queries unificadas."""
    unified_query: str
    post_processors: List[Dict[str, Any]] = Field(default_factory=list)


class ProceduralImplementation(BaseModel):
    """Implementación procedural."""
    analyzer_function: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class CompositeImplementation(BaseModel):
    """Implementación compuesta de múltiples reglas."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    sub_rules: List[RuleId]
    combination_logic: str = "all"  # all, any, majority


class MachineLearningImplementation(BaseModel):
    """Implementación basada en machine learning."""
    model_id: str
    confidence_threshold: float = 0.8
    feature_extractors: List[Dict[str, Any]] = Field(default_factory=list)


class RuleImplementation(BaseModel):
    """Implementación de una regla."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    pattern: Optional[PatternImplementation] = None
    query: Optional[QueryImplementation] = None
    procedural: Optional[ProceduralImplementation] = None
    composite: Optional[CompositeImplementation] = None
    machine_learning: Optional[MachineLearningImplementation] = None
    
    @validator('*', pre=True)
    def validate_single_implementation(cls, v, values):
        """Validar que solo una implementación esté presente."""
        if v is not None:
            non_none_count = sum(1 for val in values.values() if val is not None)
            if non_none_count > 1:
                raise ValueError("Only one implementation type can be specified")
        return v


@dataclass
class Rule:
    """Definición de una regla estática."""
    id: RuleId
    name: str
    description: str
    category: RuleCategory
    severity: RuleSeverity
    languages: List[ProgrammingLanguage]
    implementation: RuleImplementation
    tags: List[str] = field(default_factory=list)
    configuration: RuleConfiguration = field(default_factory=RuleConfiguration)
    metadata: RuleMetadata = field(default_factory=RuleMetadata)
    version: str = "1.0.0"
    enabled: bool = True
    
    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Rule name cannot be empty")
        if not self.description or not self.description.strip():
            raise ValueError("Rule description cannot be empty")
        if not self.languages:
            raise ValueError("Rule must support at least one language")


@dataclass
class ViolationLocation:
    """Ubicación de una violación en el código."""
    file_path: Path
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    context_lines: List[str] = field(default_factory=list)
    node_id: Optional[str] = None


@dataclass
class FixSuggestion:
    """Sugerencia de corrección para una violación."""
    description: str
    code_snippet: str
    confidence: float = 0.8
    automatic: bool = False
    complexity: str = "medium"  # trivial, easy, medium, hard


@dataclass
class Violation:
    """Violación detectada por una regla."""
    rule_id: RuleId
    severity: RuleSeverity
    message: str
    location: ViolationLocation
    rule_category: RuleCategory
    confidence: float = 1.0
    fix_suggestions: List[FixSuggestion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Suggestion:
    """Sugerencia de mejora."""
    rule_id: RuleId
    message: str
    category: str
    priority: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleExecutionMetrics:
    """Métricas de ejecución de una regla."""
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit: bool = False
    violations_found: int = 0
    suggestions_generated: int = 0
    error_count: int = 0


@dataclass
class RuleResult:
    """Resultado de la ejecución de una regla."""
    rule_id: RuleId
    violations: List[Violation] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    metrics: RuleExecutionMetrics = field(default_factory=RuleExecutionMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class AnalysisMetrics:
    """Métricas de análisis de código."""
    total_violations: int = 0
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0
    low_violations: int = 0
    info_violations: int = 0
    quality_score: float = 100.0
    maintainability_index: float = 100.0
    technical_debt_hours: float = 0.0
    code_smells: int = 0
    bugs: int = 0
    vulnerabilities: int = 0
    security_hotspots: int = 0
    duplications: int = 0
    test_coverage: float = 0.0


@dataclass
class AnalysisResult:
    """Resultado del análisis de un archivo."""
    file_path: Path
    language: str
    violations: List[Violation] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    metrics: AnalysisMetrics = field(default_factory=AnalysisMetrics)
    execution_time_ms: float = 0.0
    rules_executed: int = 0
    cache_hits: int = 0
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ProjectMetrics:
    """Métricas agregadas del proyecto."""
    total_files: int = 0
    total_lines: int = 0
    total_violations: int = 0
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0
    low_violations: int = 0
    info_violations: int = 0
    quality_score: float = 100.0
    maintainability_index: float = 100.0
    technical_debt_hours: float = 0.0
    reliability_rating: str = "A"
    security_rating: str = "A"
    maintainability_rating: str = "A"
    security_review_rating: str = "A"
    
    def aggregate(self, other: AnalysisMetrics):
        """Agregar métricas de otro análisis."""
        self.total_violations += other.total_violations
        self.critical_violations += other.critical_violations
        self.high_violations += other.high_violations
        self.medium_violations += other.medium_violations
        self.low_violations += other.low_violations
        self.info_violations += other.info_violations
        self.technical_debt_hours += other.technical_debt_hours
        self.code_smells += other.code_smells
        self.bugs += other.bugs
        self.vulnerabilities += other.vulnerabilities
        self.security_hotspots += other.security_hotspots
        self.duplications += other.duplications


@dataclass
class QualityGates:
    """Puertas de calidad para el proyecto."""
    max_critical_violations: int = 0
    max_high_violations: int = 10
    max_medium_violations: int = 50
    min_quality_score: float = 80.0
    min_maintainability_rating: str = "B"
    min_security_rating: str = "B"
    min_reliability_rating: str = "B"
    max_technical_debt_hours: float = 100.0


@dataclass
class ProjectConfig:
    """Configuración del proyecto para análisis."""
    enabled_categories: List[RuleCategory] = field(default_factory=list)
    severity_threshold: RuleSeverity = RuleSeverity.INFO
    custom_rule_configs: Dict[RuleId, RuleConfiguration] = field(default_factory=dict)
    exclusion_patterns: List[str] = field(default_factory=list)
    parallel_analysis_batch_size: Optional[int] = None
    quality_gates: QualityGates = field(default_factory=QualityGates)
    language_specific_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ProjectAnalysisResult:
    """Resultado del análisis de un proyecto completo."""
    project_path: Path
    file_results: List[AnalysisResult] = field(default_factory=list)
    summary_metrics: ProjectMetrics = field(default_factory=ProjectMetrics)
    total_violations: int = 0
    critical_violations: int = 0
    high_violations: int = 0
    quality_score: float = 100.0
    recommendations: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
