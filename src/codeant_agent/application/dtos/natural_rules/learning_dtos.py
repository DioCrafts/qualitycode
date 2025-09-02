"""
Módulo que define los DTOs para el aprendizaje y mejora continua.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import UUID


@dataclass
class RuleFeedbackDTO:
    """DTO para feedback de reglas."""
    rule_id: UUID
    user_id: str
    is_positive: bool
    comments: Optional[str] = None
    false_positive: bool = False
    false_negative: bool = False
    suggested_improvements: List[str] = field(default_factory=list)


@dataclass
class RulePerformanceMetricsDTO:
    """DTO para métricas de rendimiento de reglas."""
    rule_id: UUID
    execution_time_ms: float
    memory_usage_kb: float
    cpu_usage_percent: float
    timestamp: str = ""


@dataclass
class RuleAccuracyMetricsDTO:
    """DTO para métricas de precisión de reglas."""
    rule_id: UUID
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    timestamp: str = ""


@dataclass
class LearningResultDTO:
    """DTO para resultados de aprendizaje."""
    rule_id: UUID
    feedback_analyzed: bool
    improvements_suggested: int
    improvements_applied: int
    new_accuracy: float
    learning_confidence: float


@dataclass
class OptimizationResultDTO:
    """DTO para resultados de optimización."""
    rule_id: UUID
    optimizations_applied: int
    performance_improvement: float
    accuracy_impact: float


@dataclass
class RuleImprovementDTO:
    """DTO para mejoras de reglas."""
    rule_id: UUID
    improvement_type: str
    description: str
    confidence: float
    before_code: Optional[str] = None
    after_code: Optional[str] = None


@dataclass
class FeedbackAnalysisDTO:
    """DTO para análisis de feedback."""
    rule_id: UUID
    feedback_count: int
    positive_ratio: float
    common_issues: List[str] = field(default_factory=list)
    suggested_improvements: List[str] = field(default_factory=list)
    estimated_new_accuracy: float = 0.0
    confidence: float = 0.0
