"""
Entidades del dominio para gestión de issues.

Este módulo contiene todas las entidades que representan el sistema
de categorización, priorización y gestión inteligente de issues.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from enum import Enum
import uuid

from ..value_objects.programming_language import ProgrammingLanguage
from .dead_code_analysis import SourceRange, SourcePosition
from .code_metrics import CodeSmell, CodeSmellSeverity


class IssueCategory(Enum):
    """Categorías principales de issues."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    COMPATIBILITY = "compatibility"
    USABILITY = "usability"
    TESTABILITY = "testability"
    DOCUMENTATION = "documentation"
    BEST_PRACTICES = "best_practices"
    CODE_STYLE = "code_style"
    ARCHITECTURE = "architecture"
    CUSTOM = "custom"


class IssueSeverity(Enum):
    """Severidad de issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PriorityLevel(Enum):
    """Niveles de prioridad."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    LOWEST = "lowest"


class IssueStatus(Enum):
    """Estados de issue."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    FIXED = "fixed"
    DISMISSED = "dismissed"
    DEFERRED = "deferred"


class FixType(Enum):
    """Tipos de fix."""
    CODE_CHANGE = "code_change"
    REFACTORING = "refactoring"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    ARCHITECTURAL = "architectural"
    PROCESS_CHANGE = "process_change"
    TOOL_UPDATE = "tool_update"


class ClusterType(Enum):
    """Tipos de clusters de issues."""
    SIMILAR = "similar"
    RELATED = "related"
    SINGLETON = "singleton"
    OUTLIER = "outlier"
    ROOT_CAUSE = "root_cause"


class BusinessImpactLevel(Enum):
    """Nivel de impacto en el negocio."""
    BLOCKING = "blocking"
    SIGNIFICANT = "significant"
    MODERATE = "moderate"
    MINOR = "minor"
    NEGLIGIBLE = "negligible"


@dataclass
class IssueId:
    """Identificador único de issue."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        return isinstance(other, IssueId) and self.value == other.value


@dataclass
class ClusterId:
    """Identificador único de cluster."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RawIssue:
    """Issue sin procesar del motor de reglas."""
    rule_id: str
    message: str
    severity: IssueSeverity
    file_path: Path
    location: SourceRange
    language: ProgrammingLanguage
    category: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    rule_type: str = "unknown"
    confidence: float = 1.0
    complexity_metrics: Optional['ComplexityContext'] = None
    related_code: Optional[str] = None
    
    def get_unique_key(self) -> str:
        """Obtiene clave única para el issue."""
        return f"{self.rule_id}:{self.file_path}:{self.location.start.line}"


@dataclass
class ComplexityContext:
    """Contexto de complejidad para un issue."""
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    nesting_depth: int = 0
    function_length: int = 0
    class_length: int = 0


@dataclass
class IssueMetadata:
    """Metadatos enriquecidos de issue."""
    estimated_fix_time_hours: Optional[float] = None
    business_impact_level: BusinessImpactLevel = BusinessImpactLevel.MINOR
    affected_users_count: int = 0
    performance_impact_percentage: float = 0.0
    security_risk_score: float = 0.0
    technical_debt_contribution: float = 0.0
    fix_complexity_score: float = 0.0
    regression_risk_score: float = 0.0
    priority: Optional['IssuePriority'] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ContextInfo:
    """Información contextual del issue."""
    surrounding_issues: List[str] = field(default_factory=list)
    file_change_frequency: float = 0.0
    author_experience_level: str = "unknown"
    code_age_days: int = 0
    test_coverage_percentage: float = 0.0
    dependency_count: int = 0
    module_criticality: str = "normal"
    deployment_frequency: str = "unknown"


@dataclass
class CategorizedIssue:
    """Issue categorizado y enriquecido."""
    id: IssueId = field(default_factory=IssueId)
    original_issue: Optional[RawIssue] = None
    primary_category: IssueCategory = IssueCategory.MAINTAINABILITY
    secondary_categories: List[IssueCategory] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    metadata: IssueMetadata = field(default_factory=IssueMetadata)
    context_info: ContextInfo = field(default_factory=ContextInfo)
    categorization_timestamp: datetime = field(default_factory=datetime.now)
    status: IssueStatus = IssueStatus.OPEN
    
    def get_display_title(self) -> str:
        """Obtiene título para display."""
        if self.original_issue:
            return f"{self.primary_category.value}: {self.original_issue.message[:60]}..."
        return f"{self.primary_category.value}: Issue {self.id.value[:8]}"
    
    def get_priority_score(self) -> float:
        """Obtiene score de prioridad."""
        return self.metadata.priority.score if self.metadata.priority else 0.0
    
    def is_high_priority(self) -> bool:
        """Verifica si es alta prioridad."""
        if self.metadata.priority:
            return self.metadata.priority.level in [PriorityLevel.CRITICAL, PriorityLevel.HIGH]
        return False


@dataclass
class IssueFeatureVector:
    """Vector de características para ML y clustering."""
    issue_id: IssueId
    category_vector: List[float] = field(default_factory=list)
    severity_score: float = 0.0
    complexity_features: List[float] = field(default_factory=list)
    location_features: List[float] = field(default_factory=list)
    context_features: List[float] = field(default_factory=list)
    textual_features: List[float] = field(default_factory=list)  # TF-IDF del mensaje
    
    def get_combined_vector(self) -> List[float]:
        """Obtiene vector combinado para clustering."""
        combined = []
        combined.extend(self.category_vector)
        combined.append(self.severity_score)
        combined.extend(self.complexity_features)
        combined.extend(self.location_features)
        combined.extend(self.context_features)
        combined.extend(self.textual_features)
        return combined


@dataclass
class ImpactScore:
    """Score de impacto de un issue."""
    score: float  # 0-100
    code_impact: float = 0.0  # Impacto en calidad del código
    user_impact: float = 0.0  # Impacto en usuarios
    system_impact: float = 0.0  # Impacto en sistema
    business_impact: float = 0.0  # Impacto en negocio
    affected_components: List[str] = field(default_factory=list)
    blast_radius: str = "file"  # "file", "module", "component", "system"
    impact_reasoning: str = ""


@dataclass
class UrgencyScore:
    """Score de urgencia de un issue."""
    score: float  # 0-100
    change_frequency: float = 0.0  # Frecuencia de cambios en archivo
    recency_score: float = 0.0  # Qué tan reciente es el código
    trend_score: float = 0.0  # Tendencia de empeoramiento
    is_critical_path: bool = False
    temporal_factors: List[str] = field(default_factory=list)
    urgency_reasoning: str = ""


@dataclass
class BusinessValueScore:
    """Score de valor de negocio."""
    score: float  # 0-100
    revenue_impact: float = 0.0
    cost_savings: float = 0.0
    risk_mitigation_value: float = 0.0
    productivity_improvement: float = 0.0
    customer_satisfaction_impact: float = 0.0
    competitive_advantage: float = 0.0
    value_reasoning: str = ""


@dataclass
class RiskScore:
    """Score de riesgo asociado."""
    score: float  # 0-100
    security_risk: float = 0.0
    stability_risk: float = 0.0
    performance_risk: float = 0.0
    maintenance_risk: float = 0.0
    compliance_risk: float = 0.0
    reputation_risk: float = 0.0
    risk_reasoning: str = ""


@dataclass
class IssuePriority:
    """Prioridad calculada de un issue."""
    score: float  # 0-100
    level: PriorityLevel
    impact: ImpactScore
    urgency: UrgencyScore
    business_value: BusinessValueScore
    risk: RiskScore
    reasoning: str = ""
    confidence: float = 1.0
    calculated_at: datetime = field(default_factory=datetime.now)
    factors_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def get_priority_explanation(self) -> str:
        """Obtiene explicación de la prioridad."""
        return (f"{self.level.value.title()} priority (score: {self.score:.1f}). "
                f"Impact: {self.impact.score:.1f}, Urgency: {self.urgency.score:.1f}, "
                f"Business Value: {self.business_value.score:.1f}, Risk: {self.risk.score:.1f}")


@dataclass
class IssueCluster:
    """Cluster de issues relacionados."""
    id: ClusterId = field(default_factory=ClusterId)
    cluster_type: ClusterType = ClusterType.SIMILAR
    issues: List[CategorizedIssue] = field(default_factory=list)
    centroid: 'ClusterCentroid' = None
    cohesion_score: float = 0.0  # 0-1, donde 1 es más cohesivo
    common_characteristics: 'CommonCharacteristics' = None
    suggested_fix_strategy: 'FixStrategy' = None
    priority_distribution: 'PriorityDistribution' = None
    estimated_batch_fix_time: float = 0.0
    roi_score: float = 0.0
    
    def get_cluster_size(self) -> int:
        """Obtiene tamaño del cluster."""
        return len(self.issues)
    
    def get_average_priority(self) -> float:
        """Obtiene prioridad promedio del cluster."""
        if not self.issues:
            return 0.0
        
        priorities = [issue.get_priority_score() for issue in self.issues]
        return sum(priorities) / len(priorities)
    
    def get_dominant_category(self) -> IssueCategory:
        """Obtiene categoría dominante."""
        if self.common_characteristics and self.common_characteristics.dominant_category:
            return self.common_characteristics.dominant_category
        elif self.issues:
            return self.issues[0].primary_category
        return IssueCategory.MAINTAINABILITY


@dataclass
class ClusterCentroid:
    """Centroide de cluster para representar características centrales."""
    category_weights: Dict[IssueCategory, float] = field(default_factory=dict)
    average_severity: float = 0.0
    average_complexity: float = 0.0
    common_location_pattern: str = ""
    representative_message: str = ""


@dataclass
class CommonCharacteristics:
    """Características comunes de un cluster."""
    dominant_category: Optional[IssueCategory] = None
    common_tags: List[str] = field(default_factory=list)
    file_patterns: List[str] = field(default_factory=list)
    rule_patterns: List[str] = field(default_factory=list)
    severity_distribution: Dict[IssueSeverity, int] = field(default_factory=dict)
    language_distribution: Dict[ProgrammingLanguage, int] = field(default_factory=dict)
    common_root_causes: List[str] = field(default_factory=list)
    
    def get_similarity_summary(self) -> str:
        """Obtiene resumen de similitudes."""
        summary_parts = []
        
        if self.dominant_category:
            summary_parts.append(f"Category: {self.dominant_category.value}")
        
        if self.common_tags:
            summary_parts.append(f"Tags: {', '.join(self.common_tags[:3])}")
        
        if self.common_root_causes:
            summary_parts.append(f"Root causes: {', '.join(self.common_root_causes[:2])}")
        
        return "; ".join(summary_parts)


@dataclass
class FixStrategy:
    """Estrategia de fix para cluster."""
    strategy_type: FixType
    description: str
    batch_applicable: bool = False
    estimated_effort_multiplier: float = 1.0
    risk_level: str = "medium"
    prerequisites: List[str] = field(default_factory=list)
    automation_potential: float = 0.0  # 0-1
    recommended_approach: str = ""


@dataclass
class PriorityDistribution:
    """Distribución de prioridades en cluster."""
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    lowest_count: int = 0
    
    def get_total_issues(self) -> int:
        """Obtiene total de issues."""
        return self.critical_count + self.high_count + self.medium_count + self.low_count + self.lowest_count
    
    def get_high_priority_percentage(self) -> float:
        """Obtiene porcentaje de alta prioridad."""
        total = self.get_total_issues()
        if total == 0:
            return 0.0
        return ((self.critical_count + self.high_count) / total) * 100.0
    
    @classmethod
    def from_issues(cls, issues: List[CategorizedIssue]) -> 'PriorityDistribution':
        """Crea distribución desde lista de issues."""
        distribution = cls()
        
        for issue in issues:
            if issue.metadata.priority:
                level = issue.metadata.priority.level
                if level == PriorityLevel.CRITICAL:
                    distribution.critical_count += 1
                elif level == PriorityLevel.HIGH:
                    distribution.high_count += 1
                elif level == PriorityLevel.MEDIUM:
                    distribution.medium_count += 1
                elif level == PriorityLevel.LOW:
                    distribution.low_count += 1
                else:
                    distribution.lowest_count += 1
        
        return distribution


@dataclass
class FixPlan:
    """Plan de fix para cluster o issue individual."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cluster_id: Optional[ClusterId] = None
    fix_type: FixType = FixType.CODE_CHANGE
    title: str = ""
    description: str = ""
    affected_issues: List[IssueId] = field(default_factory=list)
    estimated_effort_hours: float = 1.0
    confidence_level: float = 0.7
    priority_score: float = 50.0
    implementation_steps: List['ImplementationStep'] = field(default_factory=list)
    testing_strategy: Optional['TestingStrategy'] = None
    rollback_plan: Optional['RollbackPlan'] = None
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    expected_benefits: List[str] = field(default_factory=list)
    
    def get_roi_estimate(self) -> float:
        """Estima ROI del fix."""
        if self.estimated_effort_hours == 0:
            return 0.0
        
        # ROI simplificado: benefit score / effort
        benefit_score = self.priority_score * len(self.affected_issues)
        return benefit_score / self.estimated_effort_hours


@dataclass
class ImplementationStep:
    """Paso de implementación de fix."""
    step_number: int
    description: str
    estimated_time_minutes: int = 30
    skill_requirements: List[str] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


@dataclass
class TestingStrategy:
    """Estrategia de testing para fix."""
    unit_tests_required: bool = True
    integration_tests_required: bool = False
    regression_tests_required: bool = True
    performance_tests_required: bool = False
    security_tests_required: bool = False
    estimated_test_effort_hours: float = 0.5
    test_coverage_target: float = 80.0
    specific_test_cases: List[str] = field(default_factory=list)


@dataclass
class RollbackPlan:
    """Plan de rollback para fix."""
    rollback_complexity: str = "simple"  # "simple", "moderate", "complex"
    rollback_time_minutes: int = 10
    rollback_steps: List[str] = field(default_factory=list)
    data_backup_required: bool = False
    service_downtime_required: bool = False
    rollback_risks: List[str] = field(default_factory=list)


@dataclass
class ROIAnalysis:
    """Análisis de ROI para fix."""
    fix_plan_id: str
    investment_hours: float
    investment_cost: float  # Costo en USD
    benefits: 'FixBenefits'
    payback_period_weeks: float
    roi_score: float  # ROI percentage
    net_present_value: float
    risk_adjusted_roi: float
    confidence_interval: Tuple[float, float]  # (min_roi, max_roi)
    
    def is_profitable(self) -> bool:
        """Verifica si el fix es rentable."""
        return self.roi_score > 0.0 and self.payback_period_weeks < 52.0  # 1 año


@dataclass
class FixBenefits:
    """Beneficios esperados de un fix."""
    quality_improvement_score: float = 0.0
    maintenance_cost_reduction: float = 0.0  # USD por año
    developer_productivity_gain: float = 0.0  # Horas ahorradas por año
    bug_reduction_estimate: int = 0
    security_improvement_score: float = 0.0
    performance_improvement_percentage: float = 0.0
    user_satisfaction_improvement: float = 0.0
    
    def get_total_annual_value(self) -> float:
        """Obtiene valor total anual esperado."""
        return (self.maintenance_cost_reduction + 
                self.developer_productivity_gain * 75.0 +  # $75/hora
                self.bug_reduction_estimate * 500.0)  # $500 por bug evitado


@dataclass
class RemediationPlan:
    """Plan completo de remediación."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    fix_plans: List[FixPlan] = field(default_factory=list)
    dependencies: List['FixDependency'] = field(default_factory=list)
    resource_estimates: Optional['ResourceEstimate'] = None
    roi_analyses: List[ROIAnalysis] = field(default_factory=list)
    execution_order: List['FixExecutionStep'] = field(default_factory=list)
    sprint_plans: List['SprintPlan'] = field(default_factory=list)
    total_estimated_effort_hours: float = 0.0
    expected_quality_improvement: float = 0.0
    total_investment: float = 0.0
    expected_annual_savings: float = 0.0
    payback_period_weeks: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_quick_wins(self) -> List[FixPlan]:
        """Obtiene fixes de quick win (bajo esfuerzo, alto impacto)."""
        return [
            fix_plan for fix_plan in self.fix_plans
            if fix_plan.estimated_effort_hours <= 4.0 and fix_plan.priority_score >= 70.0
        ]
    
    def get_high_roi_fixes(self) -> List[FixPlan]:
        """Obtiene fixes con mayor ROI."""
        return sorted(self.fix_plans, key=lambda f: f.get_roi_estimate(), reverse=True)[:10]


@dataclass
class FixDependency:
    """Dependencia entre fixes."""
    prerequisite_fix_id: str
    dependent_fix_id: str
    dependency_type: str = "sequential"  # "sequential", "blocking", "optional"
    dependency_reason: str = ""
    
    def is_blocking(self) -> bool:
        """Verifica si la dependencia es bloqueante."""
        return self.dependency_type == "blocking"


@dataclass
class FixExecutionStep:
    """Paso de ejecución en plan de remediación."""
    step_number: int
    fix_ids: List[str]
    estimated_duration_hours: float
    parallel_execution: bool = False
    prerequisites: List[str] = field(default_factory=list)
    risk_level: str = "medium"
    validation_checkpoints: List[str] = field(default_factory=list)
    
    def can_execute_in_parallel(self) -> bool:
        """Verifica si se puede ejecutar en paralelo."""
        return self.parallel_execution and len(self.fix_ids) > 1


@dataclass
class SprintPlan:
    """Plan de sprint para remediación."""
    sprint_number: int
    title: str = ""
    execution_steps: List[FixExecutionStep] = field(default_factory=list)
    total_effort_hours: float = 0.0
    team_capacity_hours: float = 160.0  # 40h/week * 4 weeks
    expected_completion_date: Optional[datetime] = None
    quality_goals: List['QualityGoal'] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    risk_mitigation_strategies: List[str] = field(default_factory=list)
    
    def get_capacity_utilization(self) -> float:
        """Obtiene utilización de capacidad del sprint."""
        return (self.total_effort_hours / self.team_capacity_hours) * 100.0 if self.team_capacity_hours > 0 else 0.0
    
    def is_overallocated(self) -> bool:
        """Verifica si el sprint está sobrecargado."""
        return self.get_capacity_utilization() > 90.0


@dataclass
class QualityGoal:
    """Objetivo de calidad para sprint."""
    goal_name: str
    metric_type: str
    target_value: float
    current_value: float = 0.0
    improvement_percentage: float = 0.0
    measurement_method: str = ""
    
    def is_achievable(self) -> bool:
        """Verifica si el objetivo es alcanzable."""
        return abs(self.target_value - self.current_value) / max(1.0, self.current_value) <= 0.5  # 50% improvement max


@dataclass
class ResourceEstimate:
    """Estimación de recursos para remediación."""
    total_hours: float = 0.0
    total_cost: float = 0.0
    skill_requirements: Dict[str, float] = field(default_factory=dict)  # skill -> hours
    tool_requirements: List[str] = field(default_factory=list)
    timeline_weeks: int = 1
    team_size_required: int = 1
    external_dependencies: List[str] = field(default_factory=list)
    
    def get_weekly_effort(self) -> float:
        """Obtiene esfuerzo semanal requerido."""
        return self.total_hours / max(1, self.timeline_weeks)


@dataclass
class IssueAnalysisResult:
    """Resultado completo del análisis de issues."""
    categorized_issues: List[CategorizedIssue] = field(default_factory=list)
    issue_clusters: List[IssueCluster] = field(default_factory=list)
    remediation_plan: Optional[RemediationPlan] = None
    priority_distribution: PriorityDistribution = field(default_factory=PriorityDistribution)
    category_distribution: Dict[IssueCategory, int] = field(default_factory=dict)
    analysis_summary: 'AnalysisSummary' = None
    analysis_time_ms: int = 0
    analyzed_at: datetime = field(default_factory=datetime.now)
    
    def get_total_issues(self) -> int:
        """Obtiene total de issues analizados."""
        return len(self.categorized_issues)
    
    def get_high_priority_issues(self) -> List[CategorizedIssue]:
        """Obtiene issues de alta prioridad."""
        return [issue for issue in self.categorized_issues if issue.is_high_priority()]
    
    def get_issues_by_category(self, category: IssueCategory) -> List[CategorizedIssue]:
        """Obtiene issues por categoría."""
        return [issue for issue in self.categorized_issues if issue.primary_category == category]


@dataclass
class AnalysisSummary:
    """Resumen del análisis de issues."""
    total_issues_analyzed: int = 0
    categorization_accuracy: float = 0.0
    clustering_efficiency: float = 0.0
    priority_distribution_balance: float = 0.0
    estimated_remediation_time_weeks: int = 0
    estimated_quality_improvement: float = 0.0
    top_categories: List[Tuple[IssueCategory, int]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def get_analysis_quality_score(self) -> float:
        """Obtiene score de calidad del análisis."""
        return (self.categorization_accuracy + self.clustering_efficiency + 
                self.priority_distribution_balance) / 3.0


@dataclass
class CategorizationConfig:
    """Configuración para categorización."""
    enable_ml_classification: bool = True
    enable_similarity_grouping: bool = True
    enable_context_analysis: bool = True
    similarity_threshold: float = 0.7
    confidence_threshold: float = 0.6
    max_categories_per_issue: int = 3
    enable_auto_tagging: bool = True
    custom_classification_rules: List[str] = field(default_factory=list)
    language_specific_rules: Dict[ProgrammingLanguage, List[str]] = field(default_factory=dict)


@dataclass
class PriorityConfig:
    """Configuración para cálculo de prioridades."""
    impact_weight: float = 0.4
    urgency_weight: float = 0.3
    business_value_weight: float = 0.2
    risk_weight: float = 0.1
    complexity_penalty_factor: float = 0.1
    fix_time_factor: float = 0.05
    enable_dynamic_priorities: bool = True
    priority_decay_factor: float = 0.95  # Prioridad decrece 5% por semana
    enable_machine_learning: bool = True


@dataclass
class ClusteringConfig:
    """Configuración para clustering."""
    similarity_threshold: float = 0.8
    min_cluster_size: int = 2
    max_cluster_size: int = 20
    clustering_method: str = "hierarchical"  # "hierarchical", "kmeans", "dbscan"
    enable_hierarchical_clustering: bool = True
    distance_metric: str = "cosine"  # "cosine", "euclidean", "manhattan"
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "category": 0.3,
        "severity": 0.2,
        "complexity": 0.2,
        "location": 0.1,
        "context": 0.1,
        "textual": 0.1
    })


@dataclass
class RemediationConfig:
    """Configuración para planificación de remediación."""
    available_developer_hours_per_week: float = 40.0
    sprint_duration_weeks: int = 2
    team_velocity_points: int = 20
    risk_tolerance: str = "medium"  # "low", "medium", "high"
    prefer_quick_wins: bool = True
    max_parallel_fixes: int = 3
    quality_improvement_target: float = 20.0  # Porcentaje
    budget_limit_usd: Optional[float] = None
    deadline: Optional[datetime] = None


@dataclass
class IssueTrend:
    """Tendencia de issues a lo largo del tiempo."""
    issue_category: IssueCategory
    time_points: List[datetime] = field(default_factory=list)
    issue_counts: List[int] = field(default_factory=list)
    severity_trends: Dict[IssueSeverity, List[int]] = field(default_factory=dict)
    trend_direction: str = "stable"  # "increasing", "decreasing", "stable", "volatile"
    trend_strength: float = 0.0  # 0-1
    prediction_next_period: int = 0
    
    def add_data_point(self, timestamp: datetime, count: int, severity_breakdown: Dict[IssueSeverity, int]) -> None:
        """Añade punto de datos a la tendencia."""
        self.time_points.append(timestamp)
        self.issue_counts.append(count)
        
        for severity, count in severity_breakdown.items():
            if severity not in self.severity_trends:
                self.severity_trends[severity] = []
            self.severity_trends[severity].append(count)
        
        self._calculate_trend()
    
    def _calculate_trend(self) -> None:
        """Calcula dirección y fuerza de tendencia."""
        if len(self.issue_counts) < 3:
            return
        
        # Análisis de tendencia simple
        recent_avg = sum(self.issue_counts[-3:]) / 3
        older_avg = sum(self.issue_counts[:3]) / 3 if len(self.issue_counts) >= 6 else self.issue_counts[0]
        
        if recent_avg > older_avg * 1.2:
            self.trend_direction = "increasing"
            self.trend_strength = min(1.0, (recent_avg - older_avg) / older_avg)
        elif recent_avg < older_avg * 0.8:
            self.trend_direction = "decreasing"
            self.trend_strength = min(1.0, (older_avg - recent_avg) / older_avg)
        else:
            self.trend_direction = "stable"
            self.trend_strength = 0.1


@dataclass
class IssueManagerConfig:
    """Configuración principal del gestor de issues."""
    categorization_config: CategorizationConfig = field(default_factory=CategorizationConfig)
    priority_config: PriorityConfig = field(default_factory=PriorityConfig)
    clustering_config: ClusteringConfig = field(default_factory=ClusteringConfig)
    remediation_config: RemediationConfig = field(default_factory=RemediationConfig)
    enable_machine_learning: bool = True
    enable_adaptive_learning: bool = True
    enable_trend_analysis: bool = True
    performance_mode: str = "balanced"  # "fast", "balanced", "accurate"
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Obtiene configuraciones de performance."""
        settings = {
            "fast": {
                "batch_size": 1000,
                "ml_enabled": False,
                "clustering_enabled": False
            },
            "balanced": {
                "batch_size": 500,
                "ml_enabled": True,
                "clustering_enabled": True
            },
            "accurate": {
                "batch_size": 100,
                "ml_enabled": True,
                "clustering_enabled": True,
                "deep_analysis": True
            }
        }
        
        return settings.get(self.performance_mode, settings["balanced"])
