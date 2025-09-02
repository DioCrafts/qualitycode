"""
Motor de Reglas Estáticas Configurable.

Este módulo implementa un motor de reglas estáticas robusto y altamente configurable
que puede ejecutar más de 30,000 reglas de análisis de código, aprovechando el
sistema AST unificado para proporcionar análisis cross-language.
"""

from .core.rule_engine import RulesEngine, RulesEngineConfig
from .core.rule_executor import RuleExecutor, ExecutorConfig
from .core.rule_registry import RuleRegistry
from .core.rule_cache import RuleCache, CacheConfig
from .core.performance_optimizer import PerformanceOptimizer
from .core.result_aggregator import ResultAggregator
from .core.configuration_manager import ConfigurationManager

from .models.rule_models import (
    Rule,
    RuleId,
    RuleCategory,
    RuleSeverity,
    ParameterType,
    RuleImplementation,
    PatternImplementation,
    RuleConfiguration,
    RuleMetadata,
    RuleResult,
    Violation,
    ViolationLocation,
    Suggestion,
    FixSuggestion,
    AnalysisResult,
    ProjectAnalysisResult,
    ProjectConfig,
    QualityGates,
)

from .models.pattern_models import (
    ASTPattern,
    PatternType,
    NodeSelector,
    PatternConstraint,
    PatternMatch,
    CaptureGroup,
    Quantifier,
    SiblingDirection,
    ScopeType,
    EquivalenceType,
)

from .models.condition_models import (
    RuleCondition,
    ConditionType,
    NodeCountCondition,
    AttributeValueCondition,
    CustomPredicateCondition,
    CrossLanguageCondition,
)

from .models.action_models import (
    RuleAction,
    ActionType,
    ReportViolationAction,
    GenerateSuggestionAction,
    ApplyFixAction,
    LogAction,
)

from .models.config_models import (
    ConfigParameter,
    ParameterValidation,
    ThresholdConfig,
    ExclusionPattern,
    GlobalRuleConfig,
    ProjectRuleConfig,
    RuleOverride,
    EffectiveRuleConfig,
    PerformanceSettings,
    OutputFormat,
    LanguageConfig,
)

from .builtin.builtin_rules_library import BuiltinRulesLibrary



__all__ = [
    # Core Engine
    "RulesEngine",
    "RulesEngineConfig",
    "RuleExecutor", 
    "ExecutorConfig",
    "RuleRegistry",
    "RuleCache",
    "CacheConfig",
    "PerformanceOptimizer",
    "ResultAggregator",
    "ConfigurationManager",
    
    # Models
    "Rule",
    "RuleId",
    "RuleCategory",
    "RuleSeverity",
    "RuleImplementation",
    "PatternImplementation",
    "RuleConfiguration",
    "RuleMetadata",
    "RuleResult",
    "Violation",
    "ViolationLocation",
    "Suggestion",
    "FixSuggestion",
    "AnalysisResult",
    "ProjectAnalysisResult",
    "ProjectConfig",
    "QualityGates",
    
    # Pattern Models
    "ASTPattern",
    "PatternType",
    "NodeSelector",
    "PatternConstraint",
    "PatternMatch",
    "CaptureGroup",
    "Quantifier",
    "SiblingDirection",
    "ScopeType",
    "EquivalenceType",
    
    # Condition Models
    "RuleCondition",
    "ConditionType",
    "NodeCountCondition",
    "AttributeValueCondition",
    "CustomPredicateCondition",
    "CrossLanguageCondition",
    
    # Action Models
    "RuleAction",
    "ActionType",
    "ReportViolationAction",
    "GenerateSuggestionAction",
    "ApplyFixAction",
    "LogAction",
    
    # Config Models
    "ConfigParameter",
    "ParameterType",
    "ParameterValidation",
    "ThresholdConfig",
    "ExclusionPattern",
    "GlobalRuleConfig",
    "ProjectRuleConfig",
    "RuleOverride",
    "EffectiveRuleConfig",
    "PerformanceSettings",
    "OutputFormat",
    "LanguageConfig",
    
    # Builtin Rules
    "BuiltinRulesLibrary",
]
