"""
Módulo que define métricas y estadísticas para las reglas en lenguaje natural.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID


@dataclass(frozen=True)
class RulePerformanceMetrics:
    """Métricas de rendimiento para una regla."""
    rule_id: UUID
    execution_time_ms: float
    memory_usage_kb: float
    cpu_usage_percent: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class RuleAccuracyMetrics:
    """Métricas de precisión para una regla."""
    rule_id: UUID
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def precision(self) -> float:
        """Calcula la precisión de la regla."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Calcula el recall de la regla."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        """Calcula el F1-score de la regla."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass(frozen=True)
class RuleUsageMetrics:
    """Métricas de uso para una regla."""
    rule_id: UUID
    execution_count: int
    user_feedback_count: int
    positive_feedback_count: int
    negative_feedback_count: int
    last_used: datetime
    created_at: datetime
    
    @property
    def feedback_ratio(self) -> float:
        """Calcula el ratio de feedback positivo vs negativo."""
        if self.negative_feedback_count == 0:
            return float(self.positive_feedback_count) if self.positive_feedback_count > 0 else 0.0
        return self.positive_feedback_count / self.negative_feedback_count


@dataclass(frozen=True)
class RuleFeedback:
    """Feedback proporcionado por un usuario sobre una regla."""
    rule_id: UUID
    user_id: str
    is_positive: bool
    comments: Optional[str] = None
    false_positive: bool = False
    false_negative: bool = False
    suggested_improvements: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class RuleGenerationStats:
    """Estadísticas sobre la generación de reglas."""
    total_rules_generated: int
    successful_generations: int
    failed_generations: int
    average_generation_time_ms: float
    average_confidence_score: float
    language_distribution: Dict[str, int]
    intent_distribution: Dict[str, int]
    domain_distribution: Dict[str, int]
    period_start: datetime
    period_end: datetime
