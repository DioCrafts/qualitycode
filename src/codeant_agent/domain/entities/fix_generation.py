"""
Domain entities for automatic fix generation and suggestions system.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from enum import Enum
from datetime import datetime
from uuid import UUID, uuid4

from .base import ValueObject, Entity
from .language import ProgrammingLanguage
from .antipattern_analysis import AntipatternType, AntipatternSeverity, UnifiedPosition


class GenerationType(str, Enum):
    """Types of code generation."""
    FIX = "fix"
    REFACTOR = "refactor"
    OPTIMIZE = "optimize"
    MODERNIZE = "modernize"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TEST = "test"


class FixType(str, Enum):
    """Types of fixes that can be applied."""
    DIRECT_REPLACEMENT = "direct_replacement"
    REFACTORING = "refactoring"
    ADD_CODE = "add_code"
    REMOVE_CODE = "remove_code"
    RESTRUCTURE = "restructure"
    CONFIGURATION_CHANGE = "configuration_change"
    MULTI_FILE = "multi_file"
    DEPENDENCY_UPDATE = "dependency_update"


class RefactoringType(str, Enum):
    """Types of refactoring operations."""
    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    MOVE_METHOD = "move_method"
    INTRODUCE_PARAMETER_OBJECT = "introduce_parameter_object"
    REPLACE_CONDITIONAL_WITH_POLYMORPHISM = "replace_conditional_with_polymorphism"
    INLINE_METHOD = "inline_method"
    RENAME_VARIABLE = "rename_variable"
    RENAME_METHOD = "rename_method"
    RENAME_CLASS = "rename_class"
    DECOMPOSE_CONDITIONAL = "decompose_conditional"
    REPLACE_TEMP_WITH_QUERY = "replace_temp_with_query"
    INTRODUCE_EXPLAINING_VARIABLE = "introduce_explaining_variable"


class ChangeType(str, Enum):
    """Types of code changes."""
    VARIABLE_RENAME = "variable_rename"
    FUNCTION_RENAME = "function_rename"
    TYPE_CHANGE = "type_change"
    STRUCTURAL_CHANGE = "structural_change"
    LOGIC_CHANGE = "logic_change"
    METHOD_EXTRACTED = "method_extracted"
    METHOD_ADDED = "method_added"
    METHOD_REMOVED = "method_removed"
    CLASS_ADDED = "class_added"
    CLASS_REMOVED = "class_removed"
    IMPORT_ADDED = "import_added"
    IMPORT_REMOVED = "import_removed"


class ValidationStatus(str, Enum):
    """Validation status for generated code."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    NOT_VALIDATED = "not_validated"


class FixApplicationStatus(str, Enum):
    """Status of fix application."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"
    ROLLED_BACK = "rolled_back"
    PENDING = "pending"


class ConfidenceLevel(str, Enum):
    """Confidence levels for generated fixes."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class ComplexityLevel(str, Enum):
    """Complexity levels for fixes and refactorings."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class RiskLevel(str, Enum):
    """Risk levels for code changes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class CodeGenerationConfig(ValueObject):
    """Configuration for code generation."""
    max_generation_length: int = 2000
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    num_return_sequences: int = 3
    enable_beam_search: bool = True
    beam_size: int = 5
    enable_sampling: bool = True
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0
    early_stopping: bool = True
    use_cache: bool = True
    timeout_seconds: int = 30


@dataclass(frozen=True)
class RefactoringConfig(ValueObject):
    """Configuration for refactoring operations."""
    enable_aggressive_refactoring: bool = False
    preserve_behavior: bool = True
    maintain_api_compatibility: bool = True
    max_refactoring_scope: str = "module"  # function, class, module, package, project
    enable_multi_file_refactoring: bool = True
    safety_threshold: float = 0.8
    enable_backup_creation: bool = True
    max_changes_per_refactoring: int = 50


@dataclass(frozen=True)
class FixApplicationConfig(ValueObject):
    """Configuration for fix application."""
    create_backup_before_fix: bool = True
    verify_after_fix: bool = True
    enable_atomic_operations: bool = True
    max_concurrent_fixes: int = 5
    rollback_on_failure: bool = True
    run_tests_after_fix: bool = False
    backup_retention_days: int = 7
    enable_dry_run: bool = False


@dataclass
class DiffLine(ValueObject):
    """Represents a line in a diff."""
    line_number: int
    content: str
    line_type: str  # added, removed, modified, context
    
    
@dataclass
class DiffModification(ValueObject):
    """Represents a modification in a diff."""
    line_number: int
    original_content: str
    new_content: str
    change_type: ChangeType


@dataclass
class DiffStats(ValueObject):
    """Statistics about a diff."""
    lines_added: int
    lines_removed: int
    lines_modified: int
    files_changed: int
    total_changes: int


@dataclass
class CodeDiff(ValueObject):
    """Represents a code diff between original and fixed code."""
    additions: List[DiffLine] = field(default_factory=list)
    deletions: List[DiffLine] = field(default_factory=list)
    modifications: List[DiffModification] = field(default_factory=list)
    context_lines: List[DiffLine] = field(default_factory=list)
    unified_diff: str = ""
    stats: Optional[DiffStats] = None


@dataclass
class CodeChange(ValueObject):
    """Represents a single code change."""
    change_type: ChangeType
    location: UnifiedPosition
    description: str
    original_content: str
    new_content: str
    justification: str = ""


@dataclass
class AlternativeApproach(ValueObject):
    """Alternative approach for fixing an issue."""
    approach_name: str
    description: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    complexity: ComplexityLevel = ComplexityLevel.MODERATE
    estimated_effort: str = "medium"


@dataclass
class FixExplanation(ValueObject):
    """Detailed explanation of a fix."""
    summary: str
    detailed_explanation: str
    changes_made: List[CodeChange] = field(default_factory=list)
    why_this_fix: str = ""
    potential_impacts: List[str] = field(default_factory=list)
    testing_recommendations: List[str] = field(default_factory=list)
    alternative_approaches: List[AlternativeApproach] = field(default_factory=list)
    educational_content: Optional[str] = None
    references: List[str] = field(default_factory=list)


@dataclass
class SideEffect(ValueObject):
    """Potential side effect of a fix."""
    description: str
    severity: str  # low, medium, high
    likelihood: str  # unlikely, possible, likely
    mitigation: Optional[str] = None


@dataclass
class TestingSuggestion(ValueObject):
    """Suggestion for testing a fix."""
    test_type: str  # unit, integration, e2e
    description: str
    test_cases: List[str] = field(default_factory=list)
    coverage_areas: List[str] = field(default_factory=list)


@dataclass
class ValidationResult(ValueObject):
    """Result of code validation."""
    validation_type: str  # syntax, semantic, functional, style, security
    status: ValidationStatus
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0
    details: Optional[Dict[str, Any]] = None


@dataclass
class GeneratedFix(Entity):
    """Represents a generated fix for an issue."""
    id: UUID = field(default_factory=uuid4)
    issue_id: str = ""
    fix_type: FixType = FixType.DIRECT_REPLACEMENT
    generation_type: GenerationType = GenerationType.FIX
    original_code: str = ""
    fixed_code: str = ""
    diff: Optional[CodeDiff] = None
    confidence_score: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    validation_results: List[ValidationResult] = field(default_factory=list)
    explanation: Optional[FixExplanation] = None
    side_effects: List[SideEffect] = field(default_factory=list)
    testing_suggestions: List[TestingSuggestion] = field(default_factory=list)
    generation_time_ms: int = 0
    model_used: str = ""
    language: ProgrammingLanguage = ProgrammingLanguage.UNKNOWN
    file_path: str = ""
    complexity: ComplexityLevel = ComplexityLevel.MODERATE
    risk_level: RiskLevel = RiskLevel.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_valid(self) -> bool:
        """Check if the fix is valid based on validation results."""
        if not self.validation_results:
            return False
        return all(
            result.status in [ValidationStatus.VALID, ValidationStatus.WARNING]
            for result in self.validation_results
        )
    
    @property
    def has_high_confidence(self) -> bool:
        """Check if the fix has high confidence."""
        return self.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
    
    @property
    def is_safe(self) -> bool:
        """Check if the fix is safe to apply."""
        return self.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM] and self.is_valid


@dataclass
class RefactoringPlan(Entity):
    """Plan for a refactoring operation."""
    id: UUID = field(default_factory=uuid4)
    refactoring_type: RefactoringType = RefactoringType.EXTRACT_METHOD
    description: str = ""
    target_code: str = ""
    estimated_effort: str = "medium"
    confidence: float = 0.0
    complexity: ComplexityLevel = ComplexityLevel.MODERATE
    affected_files: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    risks: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RefactoringImpact(ValueObject):
    """Impact analysis of a refactoring."""
    complexity_reduction: float = 0.0
    maintainability_improvement: float = 0.0
    testability_improvement: float = 0.0
    reusability_increase: float = 0.0
    performance_impact: str = "neutral"  # positive, neutral, negative
    breaking_changes: bool = False
    affected_files: int = 0
    affected_functions: int = 0
    affected_classes: int = 0
    risk_level: RiskLevel = RiskLevel.MEDIUM


@dataclass
class RefactoringResult(Entity):
    """Result of a refactoring operation."""
    id: UUID = field(default_factory=uuid4)
    plan_id: UUID = field(default_factory=uuid4)
    refactoring_type: RefactoringType = RefactoringType.EXTRACT_METHOD
    original_code: str = ""
    refactored_code: str = ""
    changes: List[CodeChange] = field(default_factory=list)
    impact_analysis: Optional[RefactoringImpact] = None
    validation_results: List[ValidationResult] = field(default_factory=list)
    explanation: Optional[FixExplanation] = None
    execution_time_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class FixApplicationResult(Entity):
    """Result of applying a fix."""
    id: UUID = field(default_factory=uuid4)
    fix_id: UUID = field(default_factory=uuid4)
    status: FixApplicationStatus = FixApplicationStatus.PENDING
    applied_changes: List[CodeChange] = field(default_factory=list)
    backup_created: bool = False
    backup_path: Optional[str] = None
    verification_passed: bool = False
    test_results: Optional[Dict[str, Any]] = None
    application_time_ms: int = 0
    rollback_available: bool = False
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_successful(self) -> bool:
        """Check if the fix was successfully applied."""
        return self.status == FixApplicationStatus.SUCCESS


@dataclass
class BatchFixResult(ValueObject):
    """Result of applying multiple fixes."""
    total_fixes: int = 0
    successful_fixes: int = 0
    failed_fixes: int = 0
    partial_fixes: int = 0
    individual_results: List[FixApplicationResult] = field(default_factory=list)
    overall_success_rate: float = 0.0
    total_time_ms: int = 0
    
    @property
    def has_failures(self) -> bool:
        """Check if any fixes failed."""
        return self.failed_fixes > 0 or self.partial_fixes > 0


@dataclass
class FixSuggestion(Entity):
    """A suggested fix for an issue."""
    id: UUID = field(default_factory=uuid4)
    issue_id: str = ""
    suggestion_type: GenerationType = GenerationType.FIX
    title: str = ""
    description: str = ""
    generated_fixes: List[GeneratedFix] = field(default_factory=list)
    refactoring_plans: List[RefactoringPlan] = field(default_factory=list)
    priority: int = 0  # Higher number = higher priority
    estimated_impact: str = "medium"
    confidence: float = 0.0
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def best_fix(self) -> Optional[GeneratedFix]:
        """Get the best fix based on confidence and validity."""
        valid_fixes = [fix for fix in self.generated_fixes if fix.is_valid]
        if not valid_fixes:
            return None
        return max(valid_fixes, key=lambda f: f.confidence_score)
    
    @property
    def has_safe_fixes(self) -> bool:
        """Check if there are any safe fixes available."""
        return any(fix.is_safe for fix in self.generated_fixes)
