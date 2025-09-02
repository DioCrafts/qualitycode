"""
Entidades del dominio para análisis de duplicación de código.

Este módulo contiene todas las entidades que representan los resultados
del análisis de detección de código duplicado y similitud.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum
import uuid

from ..value_objects.programming_language import ProgrammingLanguage
from .dead_code_analysis import SourceRange, SourcePosition


class CloneType(Enum):
    """Tipos de clones detectados."""
    EXACT = "exact"                    # Type 1: Clones exactos
    RENAMED = "renamed"                # Type 2: Clones con variables renombradas
    NEAR_MISS = "near_miss"           # Type 3: Clones con pequeñas diferencias
    SEMANTIC = "semantic"              # Type 4: Clones funcionalmente equivalentes
    CROSS_LANGUAGE = "cross_language" # Clones entre lenguajes diferentes


class SimilarityAlgorithm(Enum):
    """Algoritmos de cálculo de similitud."""
    # String-based algorithms
    LEVENSHTEIN = "levenshtein"
    JACCARD = "jaccard"
    COSINE = "cosine"
    LONGEST_COMMON_SUBSEQUENCE = "lcs"
    
    # Tree-based algorithms  
    AST_TREE_MATCHING = "ast_tree_matching"
    TREE_EDIT_DISTANCE = "tree_edit_distance"
    SUBTREE_ISOMORPHISM = "subtree_isomorphism"
    
    # Hash-based algorithms
    SIMHASH = "simhash"
    MINHASH = "minhash"
    
    # Semantic algorithms
    SEMANTIC_EMBEDDING = "semantic_embedding"
    BEHAVIORAL_SIMILARITY = "behavioral_similarity"


class HashAlgorithm(Enum):
    """Algoritmos de hashing para detección exacta."""
    SHA256 = "sha256"
    MD5 = "md5"
    XXHASH = "xxhash"
    SIMHASH = "simhash"


class RefactoringType(Enum):
    """Tipos de refactoring sugeridos."""
    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    PARAMETERIZE_METHOD = "parameterize_method"
    TEMPLATE_METHOD = "template_method"
    STRATEGY_PATTERN = "strategy_pattern"
    EXTRACT_LIBRARY = "extract_library"
    MERGE_CLASSES = "merge_classes"
    REPLACE_CONDITIONAL_WITH_POLYMORPHISM = "replace_conditional_with_polymorphism"


class EstimatedEffort(Enum):
    """Esfuerzo estimado para refactoring."""
    LOW = "low"           # < 1 hora
    MEDIUM = "medium"     # 1-4 horas
    HIGH = "high"         # 4-16 horas
    VERY_HIGH = "very_high"  # > 16 horas


class RefactoringBenefit(Enum):
    """Beneficios potenciales del refactoring."""
    REDUCED_DUPLICATION = "reduced_duplication"
    IMPROVED_MAINTAINABILITY = "improved_maintainability"
    IMPROVED_DESIGN = "improved_design"
    BETTER_TESTABILITY = "better_testability"
    REDUCED_COMPLEXITY = "reduced_complexity"
    BETTER_EXTENSIBILITY = "better_extensibility"
    CONSISTENT_BEHAVIOR = "consistent_behavior"
    CENTRALIZED_MAINTENANCE = "centralized_maintenance"


class SemanticUnitType(Enum):
    """Tipos de unidades semánticas."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    CODE_BLOCK = "code_block"
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"


class SimilarityEvidence(Enum):
    """Evidencia de similitud semántica."""
    SAME_INPUT_OUTPUT_TYPES = "same_input_output_types"
    SIMILAR_CONTROL_FLOW = "similar_control_flow"
    SIMILAR_DATA_FLOW = "similar_data_flow"
    SAME_ALGORITHMIC_COMPLEXITY = "same_algorithmic_complexity"
    SIMILAR_VARIABLE_USAGE = "similar_variable_usage"
    SAME_EXTERNAL_DEPENDENCIES = "same_external_dependencies"
    SIMILAR_ERROR_HANDLING = "similar_error_handling"
    EQUIVALENT_MATH_OPERATIONS = "equivalent_math_operations"


@dataclass
class CloneId:
    """Identificador único de clone."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        return isinstance(other, CloneId) and self.value == other.value


@dataclass
class CloneClassId:
    """Identificador único de clase de clones."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass
class CodeLocation:
    """Ubicación de código en un archivo."""
    file_path: Path
    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int = 0
    function_context: Optional[str] = None
    class_context: Optional[str] = None
    module_context: Optional[str] = None

    def to_source_range(self) -> SourceRange:
        """Convierte a SourceRange."""
        return SourceRange(
            start=SourcePosition(line=self.start_line, column=self.start_column),
            end=SourcePosition(line=self.end_line, column=self.end_column)
        )


@dataclass
class CodeBlock:
    """Bloque de código extraído para análisis."""
    content: str
    location: CodeLocation
    size_lines: int
    size_tokens: int
    language: ProgrammingLanguage
    hash_value: Optional[str] = None
    normalized_content: Optional[str] = None


@dataclass
class Clone:
    """Clase base para todos los tipos de clones."""
    id: CloneId
    clone_type: CloneType
    original_location: CodeLocation
    duplicate_location: CodeLocation
    similarity_score: float
    confidence: float
    size_lines: int = 0
    size_tokens: int = 0


@dataclass
class ExactClone(Clone):
    """Clone exacto (Type 1)."""
    content: str = ""
    hash_value: str = ""
    normalization_applied: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.clone_type = CloneType.EXACT
        if self.similarity_score == 0.0:
            self.similarity_score = 1.0  # Exact clones have perfect similarity


@dataclass
class StructuralClone(Clone):
    """Clone estructural (Type 2/3)."""
    structural_similarity: float = 0.0
    differences: List['StructuralDifference'] = field(default_factory=list)
    node_mapping: Optional['NodeMapping'] = None
    
    def __post_init__(self):
        if self.clone_type not in [CloneType.RENAMED, CloneType.NEAR_MISS]:
            self.clone_type = CloneType.RENAMED if self.structural_similarity > 0.9 else CloneType.NEAR_MISS


@dataclass
class SemanticClone(Clone):
    """Clone semántico (Type 4)."""
    semantic_similarity: float = 0.0
    behavioral_similarity: float = 0.0
    data_flow_similarity: float = 0.0
    evidence: List[SimilarityEvidence] = field(default_factory=list)
    
    def __post_init__(self):
        self.clone_type = CloneType.SEMANTIC


@dataclass
class CrossLanguageClone(Clone):
    """Clone cross-language."""
    language1: ProgrammingLanguage = ProgrammingLanguage.UNKNOWN
    language2: ProgrammingLanguage = ProgrammingLanguage.UNKNOWN
    concept_mapping: Optional['ConceptMapping'] = None
    translation_evidence: List['TranslationEvidence'] = field(default_factory=list)
    
    def __post_init__(self):
        self.clone_type = CloneType.CROSS_LANGUAGE


@dataclass
class StructuralDifference:
    """Diferencia estructural entre clones."""
    difference_type: str
    location: CodeLocation
    original_content: Optional[str] = None
    duplicate_content: Optional[str] = None
    description: str = ""
    impact_score: float = 0.0


@dataclass
class NodeMapping:
    """Mapeo entre nodos de diferentes ASTs."""
    mappings: Dict[str, str] = field(default_factory=dict)
    unmapped_original: Set[str] = field(default_factory=set)
    unmapped_duplicate: Set[str] = field(default_factory=set)
    
    def add_mapping(self, original_node_id: str, duplicate_node_id: str) -> None:
        """Añade un mapeo entre nodos."""
        self.mappings[original_node_id] = duplicate_node_id
    
    def get_mapping(self, original_node_id: str) -> Optional[str]:
        """Obtiene el mapeo para un nodo original."""
        return self.mappings.get(original_node_id)


@dataclass
class ConceptMapping:
    """Mapeo de conceptos entre lenguajes."""
    concept_mappings: Dict[str, str] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def add_concept_mapping(self, concept1: str, concept2: str, confidence: float = 1.0) -> None:
        """Añade un mapeo de conceptos."""
        self.concept_mappings[concept1] = concept2
        self.confidence_scores[concept1] = confidence


@dataclass
class TranslationEvidence:
    """Evidencia de traducción entre lenguajes."""
    evidence_type: str
    description: str
    confidence: float
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CloneClassMetrics:
    """Métricas de una clase de clones."""
    total_instances: int
    total_lines: int
    total_tokens: int
    average_similarity: float
    size_variance: float
    complexity_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RefactoringPotential:
    """Potencial de refactoring de una clase de clones."""
    overall_score: float
    maintainability_impact: float
    complexity_reduction: float
    reusability_improvement: float
    factors: List[str] = field(default_factory=list)


@dataclass
class CloneClass:
    """Clase o familia de clones relacionados."""
    id: CloneClassId
    clone_type: CloneType
    instances: List[Clone]
    similarity_score: float
    size_metrics: CloneClassMetrics
    refactoring_potential: RefactoringPotential
    representative_clone: Optional[Clone] = None
    
    def __post_init__(self):
        if self.representative_clone is None and self.instances:
            # Usar el primer clone como representativo por defecto
            self.representative_clone = self.instances[0]
    
    def get_total_instances(self) -> int:
        """Obtiene el total de instancias."""
        return len(self.instances)
    
    def get_affected_files(self) -> Set[Path]:
        """Obtiene archivos afectados por esta clase de clones."""
        files = set()
        for clone in self.instances:
            files.add(clone.original_location.file_path)
            files.add(clone.duplicate_location.file_path)
        return files


@dataclass
class DuplicationMetrics:
    """Métricas de duplicación para un archivo o proyecto."""
    total_lines: int = 0
    duplicated_lines: int = 0
    duplication_percentage: float = 0.0
    total_clones: int = 0
    clone_classes: int = 0
    exact_clones: int = 0
    structural_clones: int = 0
    semantic_clones: int = 0
    cross_language_clones: int = 0
    average_clone_size: float = 0.0
    largest_clone_size: int = 0
    duplication_ratio: float = 0.0  # duplicated_lines / total_lines
    
    def calculate_derived_metrics(self) -> None:
        """Calcula métricas derivadas."""
        if self.total_lines > 0:
            self.duplication_percentage = (self.duplicated_lines / self.total_lines) * 100
            self.duplication_ratio = self.duplicated_lines / self.total_lines
        
        if self.total_clones > 0 and self.duplicated_lines > 0:
            self.average_clone_size = self.duplicated_lines / self.total_clones


@dataclass
class RefactoringStep:
    """Paso de implementación de refactoring."""
    step_number: int
    description: str
    code_changes: List['CodeChange'] = field(default_factory=list)
    validation_steps: List[str] = field(default_factory=list)
    estimated_time_minutes: int = 0


@dataclass
class CodeChange:
    """Cambio de código específico."""
    file_path: Path
    location: CodeLocation
    change_type: str  # 'add', 'remove', 'modify'
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    description: str = ""


@dataclass
class RefactoringOpportunity:
    """Oportunidad de refactoring identificada."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    refactoring_type: RefactoringType = RefactoringType.EXTRACT_METHOD
    description: str = ""
    affected_clones: List[Clone] = field(default_factory=list)
    estimated_effort: EstimatedEffort = EstimatedEffort.MEDIUM
    potential_benefits: List[RefactoringBenefit] = field(default_factory=list)
    implementation_steps: List[RefactoringStep] = field(default_factory=list)
    confidence: float = 0.5
    impact_analysis: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    
    def get_affected_files(self) -> Set[Path]:
        """Obtiene archivos afectados por el refactoring."""
        files = set()
        for clone in self.affected_clones:
            files.add(clone.original_location.file_path)
            files.add(clone.duplicate_location.file_path)
        return files
    
    def get_estimated_time_hours(self) -> float:
        """Obtiene tiempo estimado en horas."""
        effort_hours = {
            EstimatedEffort.LOW: 0.5,
            EstimatedEffort.MEDIUM: 2.5,
            EstimatedEffort.HIGH: 10.0,
            EstimatedEffort.VERY_HIGH: 24.0
        }
        return effort_hours.get(self.estimated_effort, 1.0)


@dataclass
class SimilarityMetrics:
    """Métricas de similitud entre bloques de código."""
    overall_similarity: float = 0.0
    lexical_similarity: float = 0.0
    structural_similarity: float = 0.0
    semantic_similarity: float = 0.0
    algorithm_used: SimilarityAlgorithm = SimilarityAlgorithm.JACCARD
    calculation_time_ms: int = 0
    confidence: float = 1.0
    
    def is_significant_similarity(self, threshold: float = 0.7) -> bool:
        """Verifica si la similitud es significativa."""
        return self.overall_similarity >= threshold


@dataclass
class CloneAnalysis:
    """Resultado completo del análisis de clones para un archivo."""
    file_path: Path = Path(".")
    language: ProgrammingLanguage = ProgrammingLanguage.UNKNOWN
    exact_clones: List[ExactClone] = field(default_factory=list)
    structural_clones: List[StructuralClone] = field(default_factory=list)
    semantic_clones: List[SemanticClone] = field(default_factory=list)
    cross_language_clones: List[CrossLanguageClone] = field(default_factory=list)
    clone_classes: List[CloneClass] = field(default_factory=list)
    duplication_metrics: 'DuplicationMetrics' = field(default_factory=lambda: DuplicationMetrics())
    refactoring_opportunities: List[RefactoringOpportunity] = field(default_factory=list)
    similarity_metrics: Dict[str, SimilarityMetrics] = field(default_factory=dict)
    execution_time_ms: int = 0
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_all_clones(self) -> List[Clone]:
        """Obtiene todos los clones encontrados."""
        clones = []
        clones.extend(self.exact_clones)
        clones.extend(self.structural_clones) 
        clones.extend(self.semantic_clones)
        clones.extend(self.cross_language_clones)
        return clones
    
    def get_total_clones(self) -> int:
        """Obtiene el total de clones."""
        return len(self.get_all_clones())
    
    def get_clones_by_type(self, clone_type: CloneType) -> List[Clone]:
        """Obtiene clones por tipo."""
        return [clone for clone in self.get_all_clones() if clone.clone_type == clone_type]
    
    def get_high_confidence_clones(self, threshold: float = 0.8) -> List[Clone]:
        """Obtiene clones con alta confianza."""
        return [clone for clone in self.get_all_clones() if clone.confidence >= threshold]


@dataclass
class InterFileClone:
    """Clone que se extiende entre múltiples archivos."""
    id: CloneId = field(default_factory=CloneId)
    clone_type: CloneType = CloneType.EXACT
    file_locations: List[CodeLocation] = field(default_factory=list)
    similarity_score: float = 0.0
    confidence: float = 0.0
    cross_file_dependencies: List[str] = field(default_factory=list)


@dataclass
class ProjectCloneAnalysis:
    """Análisis de clones para un proyecto completo."""
    project_path: Optional[Path] = None
    file_analyses: List[CloneAnalysis] = field(default_factory=list)
    inter_file_clones: List[InterFileClone] = field(default_factory=list)
    cross_language_clones: List[CrossLanguageClone] = field(default_factory=list)
    global_clone_classes: List[CloneClass] = field(default_factory=list)
    project_metrics: 'DuplicationMetrics' = field(default_factory=lambda: DuplicationMetrics())
    project_refactoring_opportunities: List[RefactoringOpportunity] = field(default_factory=list)
    execution_time_ms: int = 0
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_total_files_analyzed(self) -> int:
        """Obtiene el total de archivos analizados."""
        return len(self.file_analyses)
    
    def get_files_with_clones(self) -> List[CloneAnalysis]:
        """Obtiene archivos que tienen clones."""
        return [analysis for analysis in self.file_analyses if analysis.get_total_clones() > 0]
    
    def get_worst_duplication_files(self, limit: int = 10) -> List[CloneAnalysis]:
        """Obtiene archivos con más duplicación."""
        return sorted(
            self.file_analyses,
            key=lambda x: x.duplication_metrics.duplication_percentage,
            reverse=True
        )[:limit]
    
    def get_total_clone_classes(self) -> int:
        """Obtiene total de clases de clones globales."""
        return len(self.global_clone_classes)
    
    def get_languages_analyzed(self) -> Set[ProgrammingLanguage]:
        """Obtiene lenguajes analizados."""
        return {analysis.language for analysis in self.file_analyses}


@dataclass
class CloneDetectionConfig:
    """Configuración para detección de clones."""
    min_clone_size_lines: int = 6
    min_clone_size_tokens: int = 50
    min_similarity_threshold: float = 0.7
    enable_exact_detection: bool = True
    enable_structural_detection: bool = True
    enable_semantic_detection: bool = True
    enable_cross_language_detection: bool = False
    ignore_whitespace: bool = True
    ignore_comments: bool = True
    ignore_variable_names: bool = False
    max_gap_size: int = 2
    similarity_algorithms: List[SimilarityAlgorithm] = field(default_factory=lambda: [
        SimilarityAlgorithm.LEVENSHTEIN,
        SimilarityAlgorithm.AST_TREE_MATCHING
    ])
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    language_specific_configs: Dict[ProgrammingLanguage, Dict[str, Any]] = field(default_factory=dict)
    excluded_patterns: List[str] = field(default_factory=list)
    
    def get_language_config(self, language: ProgrammingLanguage) -> Dict[str, Any]:
        """Obtiene configuración específica del lenguaje."""
        return self.language_specific_configs.get(language, {})
