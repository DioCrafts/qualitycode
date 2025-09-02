"""
Implementación del detector de clones cross-language.

Este módulo implementa la detección de clones entre diferentes lenguajes
usando mapeo de conceptos, traducción semántica y análisis de equivalencias.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from pathlib import Path
from enum import Enum
from collections import defaultdict

from ...domain.entities.clone_analysis import (
    CrossLanguageClone, CodeLocation, CloneId, CloneType,
    ConceptMapping, TranslationEvidence
)
from ...domain.entities.parse_result import ParseResult
from ...domain.entities.ast_normalization import NormalizedNode as UnifiedNode, NodeType as UnifiedNodeType, SourcePosition
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


class CrossLanguageUnitType(Enum):
    """Tipos de unidades cross-language."""
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    DESIGN_PATTERN = "design_pattern"
    BUSINESS_LOGIC = "business_logic"
    UTILITY_FUNCTION = "utility_function"
    API_ENDPOINT = "api_endpoint"
    CONFIGURATION = "configuration"


@dataclass
class ConceptMappingRule:
    """Regla de mapeo de conceptos entre lenguajes."""
    source_concept: str
    target_concept: str
    source_language: ProgrammingLanguage
    target_language: ProgrammingLanguage
    confidence: float = 1.0
    context_required: Optional[str] = None


@dataclass
class SemanticRepresentation:
    """Representación semántica independiente del lenguaje."""
    concept_type: str
    intent: str
    data_flow: Dict[str, List[str]] = field(default_factory=dict)
    control_patterns: List[str] = field(default_factory=list)
    algorithms: List[str] = field(default_factory=list)
    external_interactions: List[str] = field(default_factory=list)
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_canonical_hash(self) -> str:
        """Obtiene hash canónico de la representación."""
        import hashlib
        
        content = ":".join([
            self.concept_type,
            self.intent,
            "|".join(sorted(f"{k}={v}" for k, v in self.data_flow.items())),
            "|".join(sorted(self.control_patterns)),
            "|".join(sorted(self.algorithms)),
            "|".join(sorted(self.external_interactions))
        ])
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class CrossLanguageUnit:
    """Unidad de código para análisis cross-language."""
    id: str
    unit_type: CrossLanguageUnitType
    language: ProgrammingLanguage
    ast_node: UnifiedNode
    location: CodeLocation
    semantic_representation: SemanticRepresentation
    language_specific_info: Dict[str, Any] = field(default_factory=dict)
    
    def is_comparable_to(self, other: 'CrossLanguageUnit') -> bool:
        """Verifica si esta unidad es comparable con otra."""
        return (
            self.unit_type == other.unit_type and
            self.language != other.language and  # Diferentes lenguajes
            self._complexity_compatible(other)
        )
    
    def _complexity_compatible(self, other: 'CrossLanguageUnit') -> bool:
        """Verifica compatibilidad de complejidad."""
        my_complexity = self.semantic_representation.complexity_metrics.get('cyclomatic', 1.0)
        other_complexity = other.semantic_representation.complexity_metrics.get('cyclomatic', 1.0)
        
        # Permitir hasta 50% de diferencia en complejidad
        ratio = max(my_complexity, other_complexity) / min(my_complexity, other_complexity)
        return ratio <= 1.5


@dataclass
class CrossLanguageSimilarity:
    """Resultado de comparación cross-language."""
    overall_similarity: float
    semantic_similarity: float
    pattern_similarity: float
    concept_mapping: ConceptMapping
    translation_evidence: List[TranslationEvidence]
    confidence: float
    
    def is_significant(self, threshold: float = 0.6) -> bool:
        """Verifica si la similitud es significativa."""
        return self.overall_similarity >= threshold and self.confidence >= 0.5


@dataclass
class CrossLanguageCloneDetectionResult:
    """Resultado de detección cross-language."""
    cross_language_clones: List[CrossLanguageClone]
    units_analyzed_by_language: Dict[ProgrammingLanguage, int]
    comparison_count: int
    analysis_time_ms: int
    language_pairs_processed: List[Tuple[ProgrammingLanguage, ProgrammingLanguage]]


class ConceptMapper:
    """Mapea conceptos entre diferentes lenguajes."""
    
    def __init__(self):
        self.mapping_rules = self._initialize_mapping_rules()
        self.framework_mappings = self._initialize_framework_mappings()
    
    def _initialize_mapping_rules(self) -> List[ConceptMappingRule]:
        """Inicializa reglas de mapeo básicas."""
        rules = []
        
        # Mapeos de estructuras de control
        control_mappings = [
            # Python -> JavaScript
            ConceptMappingRule("for ... in ...", "for (let ... of ...)", 
                             ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVASCRIPT, 0.9),
            ConceptMappingRule("if __name__ == '__main__':", "if (require.main === module)", 
                             ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVASCRIPT, 0.8),
            
            # Python -> Rust
            ConceptMappingRule("def", "fn", 
                             ProgrammingLanguage.PYTHON, ProgrammingLanguage.RUST, 0.9),
            ConceptMappingRule("class", "struct/impl", 
                             ProgrammingLanguage.PYTHON, ProgrammingLanguage.RUST, 0.7),
            
            # JavaScript -> TypeScript
            ConceptMappingRule("function", "function", 
                             ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT, 1.0),
            ConceptMappingRule("const", "const", 
                             ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT, 1.0),
            
            # Tipos de datos comunes
            ConceptMappingRule("list", "Array", 
                             ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVASCRIPT, 0.9),
            ConceptMappingRule("dict", "Object/Map", 
                             ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVASCRIPT, 0.8),
            ConceptMappingRule("str", "string", 
                             ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVASCRIPT, 0.9),
        ]
        
        rules.extend(control_mappings)
        
        # Añadir mapeos inversos automáticamente
        reverse_rules = []
        for rule in control_mappings:
            if rule.confidence >= 0.8:  # Solo mapeos de alta confianza
                reverse_rules.append(ConceptMappingRule(
                    rule.target_concept, rule.source_concept,
                    rule.target_language, rule.source_language,
                    rule.confidence * 0.9  # Ligeramente menor confianza en reverso
                ))
        
        rules.extend(reverse_rules)
        return rules
    
    def _initialize_framework_mappings(self) -> Dict[str, Dict[ProgrammingLanguage, str]]:
        """Inicializa mapeos específicos de frameworks."""
        return {
            # Frameworks web
            'http_server': {
                ProgrammingLanguage.PYTHON: 'flask.Flask / fastapi.FastAPI',
                ProgrammingLanguage.JAVASCRIPT: 'express.js',
                ProgrammingLanguage.RUST: 'axum / warp',
                ProgrammingLanguage.TYPESCRIPT: 'express.js / nest.js'
            },
            
            # Testing frameworks
            'unit_testing': {
                ProgrammingLanguage.PYTHON: 'unittest / pytest',
                ProgrammingLanguage.JAVASCRIPT: 'jest / mocha',
                ProgrammingLanguage.RUST: 'cargo test',
                ProgrammingLanguage.TYPESCRIPT: 'jest / vitest'
            },
            
            # Async patterns
            'async_operations': {
                ProgrammingLanguage.PYTHON: 'async/await',
                ProgrammingLanguage.JAVASCRIPT: 'async/await / Promise',
                ProgrammingLanguage.RUST: 'async/await / tokio',
                ProgrammingLanguage.TYPESCRIPT: 'async/await / Promise'
            },
            
            # Database ORM
            'database_orm': {
                ProgrammingLanguage.PYTHON: 'SQLAlchemy / Django ORM',
                ProgrammingLanguage.JAVASCRIPT: 'Sequelize / Prisma',
                ProgrammingLanguage.RUST: 'Diesel / SeaORM',
                ProgrammingLanguage.TYPESCRIPT: 'TypeORM / Prisma'
            }
        }
    
    async def map_concepts(self, unit1: CrossLanguageUnit, unit2: CrossLanguageUnit) -> ConceptMapping:
        """
        Mapea conceptos entre dos unidades de diferentes lenguajes.
        
        Args:
            unit1: Primera unidad
            unit2: Segunda unidad
            
        Returns:
            ConceptMapping con los mapeos encontrados
        """
        concept_mapping = ConceptMapping()
        
        # Mapeos directos usando reglas predefinidas
        direct_mappings = self._find_direct_mappings(unit1, unit2)
        concept_mapping.direct_mappings.update(direct_mappings)
        
        # Mapeos inferenciales basados en contexto
        inferred_mappings = await self._infer_mappings(unit1, unit2)
        concept_mapping.inferred_mappings.update(inferred_mappings)
        
        # Mapeos de frameworks
        framework_mappings = self._find_framework_mappings(unit1, unit2)
        concept_mapping.framework_mappings.update(framework_mappings)
        
        # Calcular confianza global del mapeo
        concept_mapping.overall_confidence = self._calculate_mapping_confidence(
            direct_mappings, inferred_mappings, framework_mappings
        )
        
        return concept_mapping
    
    def _find_direct_mappings(self, unit1: CrossLanguageUnit, unit2: CrossLanguageUnit) -> Dict[str, str]:
        """Encuentra mapeos directos usando reglas predefinidas."""
        mappings = {}
        
        # Buscar en reglas de mapeo
        for rule in self.mapping_rules:
            if (rule.source_language == unit1.language and 
                rule.target_language == unit2.language):
                
                # Verificar si el concepto fuente existe en unit1
                if self._concept_exists_in_unit(rule.source_concept, unit1):
                    mappings[rule.source_concept] = rule.target_concept
        
        return mappings
    
    async def _infer_mappings(self, unit1: CrossLanguageUnit, unit2: CrossLanguageUnit) -> Dict[str, str]:
        """Infiere mapeos basado en análisis contextual."""
        mappings = {}
        
        # Inferencia basada en patrones de algoritmos
        algo_mappings = self._infer_algorithm_mappings(unit1, unit2)
        mappings.update(algo_mappings)
        
        # Inferencia basada en flujo de datos similar
        data_flow_mappings = self._infer_data_flow_mappings(unit1, unit2)
        mappings.update(data_flow_mappings)
        
        return mappings
    
    def _find_framework_mappings(self, unit1: CrossLanguageUnit, unit2: CrossLanguageUnit) -> Dict[str, str]:
        """Encuentra mapeos de frameworks equivalentes."""
        mappings = {}
        
        for framework_category, language_frameworks in self.framework_mappings.items():
            framework1 = language_frameworks.get(unit1.language)
            framework2 = language_frameworks.get(unit2.language)
            
            if framework1 and framework2:
                # Verificar si las unidades usan estos frameworks
                if (self._unit_uses_framework(unit1, framework1) and 
                    self._unit_uses_framework(unit2, framework2)):
                    mappings[framework1] = framework2
        
        return mappings
    
    def _concept_exists_in_unit(self, concept: str, unit: CrossLanguageUnit) -> bool:
        """Verifica si un concepto existe en la unidad."""
        # Buscar en patrones de control
        if concept in unit.semantic_representation.control_patterns:
            return True
        
        # Buscar en algoritmos
        if concept in unit.semantic_representation.algorithms:
            return True
        
        # Buscar en información específica del lenguaje
        return any(concept.lower() in str(value).lower() 
                  for value in unit.language_specific_info.values())
    
    def _infer_algorithm_mappings(self, unit1: CrossLanguageUnit, unit2: CrossLanguageUnit) -> Dict[str, str]:
        """Infiere mapeos basado en algoritmos similares."""
        mappings = {}
        
        algo1 = set(unit1.semantic_representation.algorithms)
        algo2 = set(unit2.semantic_representation.algorithms)
        
        # Algoritmos comunes sugieren implementaciones equivalentes
        common_algorithms = algo1.intersection(algo2)
        for algo in common_algorithms:
            # Mapeo implícito de mismo algoritmo en diferentes lenguajes
            mappings[f"{algo}_{unit1.language.name}"] = f"{algo}_{unit2.language.name}"
        
        return mappings
    
    def _infer_data_flow_mappings(self, unit1: CrossLanguageUnit, unit2: CrossLanguageUnit) -> Dict[str, str]:
        """Infiere mapeos basado en flujo de datos similar."""
        mappings = {}
        
        flow1 = unit1.semantic_representation.data_flow
        flow2 = unit2.semantic_representation.data_flow
        
        # Patrones de flujo similares sugieren funcionalidad equivalente
        for pattern1, vars1 in flow1.items():
            for pattern2, vars2 in flow2.items():
                # Si tienen patrones de variables similares, mapear patrones
                if len(set(vars1).intersection(set(vars2))) > 0:
                    mappings[pattern1] = pattern2
        
        return mappings
    
    def _unit_uses_framework(self, unit: CrossLanguageUnit, framework: str) -> bool:
        """Verifica si una unidad usa un framework específico."""
        framework_lower = framework.lower()
        
        # Buscar en interacciones externas
        for interaction in unit.semantic_representation.external_interactions:
            if any(fw_part in interaction.lower() for fw_part in framework_lower.split('/')):
                return True
        
        return False
    
    def _calculate_mapping_confidence(self, direct: Dict[str, str], 
                                    inferred: Dict[str, str], 
                                    framework: Dict[str, str]) -> float:
        """Calcula confianza global del mapeo."""
        total_mappings = len(direct) + len(inferred) + len(framework)
        
        if total_mappings == 0:
            return 0.0
        
        # Pesos por tipo de mapeo
        confidence = (
            len(direct) * 0.9 +      # Mapeos directos tienen alta confianza
            len(inferred) * 0.6 +    # Mapeos inferenciales tienen confianza media
            len(framework) * 0.8     # Mapeos de framework tienen buena confianza
        ) / total_mappings
        
        return min(1.0, confidence)


class SemanticTranslator:
    """Traduce unidades de código a representación semántica común."""
    
    def __init__(self):
        self.language_analyzers = {
            ProgrammingLanguage.PYTHON: self._analyze_python,
            ProgrammingLanguage.JAVASCRIPT: self._analyze_javascript,
            ProgrammingLanguage.TYPESCRIPT: self._analyze_typescript,
            ProgrammingLanguage.RUST: self._analyze_rust,
        }
    
    async def translate_to_common_representation(self, unit: CrossLanguageUnit) -> SemanticRepresentation:
        """
        Traduce una unidad a representación semántica común.
        
        Args:
            unit: Unidad a traducir
            
        Returns:
            SemanticRepresentation común
        """
        if unit.language in self.language_analyzers:
            return await self.language_analyzers[unit.language](unit)
        else:
            return await self._analyze_generic(unit)
    
    async def _analyze_python(self, unit: CrossLanguageUnit) -> SemanticRepresentation:
        """Análisis específico para Python."""
        representation = SemanticRepresentation(
            concept_type=unit.unit_type.value,
            intent=self._extract_intent(unit)
        )
        
        # Analizar patrones de control específicos de Python
        representation.control_patterns = self._extract_python_control_patterns(unit.ast_node)
        
        # Analizar algoritmos comunes en Python
        representation.algorithms = self._detect_python_algorithms(unit.ast_node)
        
        # Analizar interacciones externas (imports, packages)
        representation.external_interactions = self._extract_python_externals(unit.ast_node)
        
        # Métricas de complejidad
        representation.complexity_metrics = self._calculate_python_complexity(unit.ast_node)
        
        return representation
    
    async def _analyze_javascript(self, unit: CrossLanguageUnit) -> SemanticRepresentation:
        """Análisis específico para JavaScript."""
        representation = SemanticRepresentation(
            concept_type=unit.unit_type.value,
            intent=self._extract_intent(unit)
        )
        
        # Patrones específicos de JavaScript
        representation.control_patterns = self._extract_javascript_control_patterns(unit.ast_node)
        representation.algorithms = self._detect_javascript_algorithms(unit.ast_node)
        representation.external_interactions = self._extract_javascript_externals(unit.ast_node)
        representation.complexity_metrics = self._calculate_javascript_complexity(unit.ast_node)
        
        return representation
    
    async def _analyze_typescript(self, unit: CrossLanguageUnit) -> SemanticRepresentation:
        """Análisis específico para TypeScript."""
        # Similar a JavaScript pero con información de tipos
        representation = await self._analyze_javascript(unit)
        
        # Añadir información específica de TypeScript
        representation.control_patterns.extend(self._extract_typescript_type_patterns(unit.ast_node))
        
        return representation
    
    async def _analyze_rust(self, unit: CrossLanguageUnit) -> SemanticRepresentation:
        """Análisis específico para Rust."""
        representation = SemanticRepresentation(
            concept_type=unit.unit_type.value,
            intent=self._extract_intent(unit)
        )
        
        # Patrones específicos de Rust
        representation.control_patterns = self._extract_rust_control_patterns(unit.ast_node)
        representation.algorithms = self._detect_rust_algorithms(unit.ast_node)
        representation.external_interactions = self._extract_rust_externals(unit.ast_node)
        representation.complexity_metrics = self._calculate_rust_complexity(unit.ast_node)
        
        return representation
    
    async def _analyze_generic(self, unit: CrossLanguageUnit) -> SemanticRepresentation:
        """Análisis genérico para lenguajes no específicos."""
        return SemanticRepresentation(
            concept_type=unit.unit_type.value,
            intent="generic_functionality",
            control_patterns=["unknown"],
            algorithms=["generic"],
            complexity_metrics={"cyclomatic": 1.0}
        )
    
    def _extract_intent(self, unit: CrossLanguageUnit) -> str:
        """Extrae la intención/propósito de la unidad."""
        # Análisis basado en nombre de función, comentarios, etc.
        if hasattr(unit.ast_node, 'value') and unit.ast_node.value:
            value = unit.ast_node.value.lower()
            
            # Patrones comunes de intención
            if any(word in value for word in ['calculate', 'compute', 'process']):
                return "calculation"
            elif any(word in value for word in ['validate', 'check', 'verify']):
                return "validation"
            elif any(word in value for word in ['parse', 'convert', 'transform']):
                return "transformation"
            elif any(word in value for word in ['save', 'store', 'persist']):
                return "persistence"
            elif any(word in value for word in ['get', 'fetch', 'retrieve']):
                return "retrieval"
            elif any(word in value for word in ['handle', 'manage', 'control']):
                return "management"
        
        return "general_purpose"
    
    def _extract_python_control_patterns(self, node: UnifiedNode) -> List[str]:
        """Extrae patrones de control específicos de Python."""
        patterns = []
        
        def traverse(n: UnifiedNode):
            # List comprehensions
            if 'comprehension' in n.value.lower():
                patterns.append("list_comprehension")
            
            # Context managers (with statements)
            if n.node_type == UnifiedNodeType.WITH_STATEMENT:
                patterns.append("context_manager")
            
            # Decorators
            if '@' in n.value:
                patterns.append("decorator_pattern")
            
            # Exception handling
            if n.node_type == UnifiedNodeType.TRY_STATEMENT:
                patterns.append("exception_handling")
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return patterns
    
    def _detect_python_algorithms(self, node: UnifiedNode) -> List[str]:
        """Detecta algoritmos específicos de Python."""
        algorithms = []
        
        def traverse(n: UnifiedNode):
            value = n.value.lower()
            
            # Algoritmos de sorting
            if any(word in value for word in ['sort', 'sorted']):
                algorithms.append("sorting")
            
            # Algoritmos de búsqueda
            if any(word in value for word in ['search', 'find', 'index']):
                algorithms.append("searching")
            
            # Algoritmos de filtrado
            if 'filter' in value or 'lambda' in value:
                algorithms.append("filtering")
            
            # Map-reduce patterns
            if any(word in value for word in ['map', 'reduce']):
                algorithms.append("map_reduce")
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return list(set(algorithms))
    
    def _extract_python_externals(self, node: UnifiedNode) -> List[str]:
        """Extrae interacciones externas de Python."""
        externals = []
        
        def traverse(n: UnifiedNode):
            if n.node_type == UnifiedNodeType.IMPORT_STATEMENT:
                externals.append(f"import_{n.value}")
            
            # API calls comunes
            if any(pattern in n.value.lower() for pattern in ['requests.', 'urllib', 'http']):
                externals.append("http_client")
            
            # Database interactions
            if any(pattern in n.value.lower() for pattern in ['sqlite', 'mysql', 'postgres', 'mongodb']):
                externals.append("database_interaction")
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return externals
    
    def _calculate_python_complexity(self, node: UnifiedNode) -> Dict[str, float]:
        """Calcula métricas de complejidad para Python."""
        metrics = {"cyclomatic": 1.0, "nesting_depth": 0.0, "function_calls": 0.0}
        
        def traverse(n: UnifiedNode, depth: int = 0):
            # Complejidad ciclomática
            if n.node_type in [UnifiedNodeType.IF_STATEMENT, UnifiedNodeType.FOR_STATEMENT,
                              UnifiedNodeType.WHILE_STATEMENT, UnifiedNodeType.TRY_STATEMENT]:
                metrics["cyclomatic"] += 1.0
            
            # Profundidad de anidamiento
            metrics["nesting_depth"] = max(metrics["nesting_depth"], depth)
            
            # Llamadas a funciones
            if n.node_type == UnifiedNodeType.FUNCTION_CALL:
                metrics["function_calls"] += 1.0
            
            for child in n.children:
                traverse(child, depth + 1)
        
        traverse(node)
        return metrics
    
    # Métodos similares para otros lenguajes (JavaScript, TypeScript, Rust)
    # Implementación simplificada para mantener el ejemplo conciso
    
    def _extract_javascript_control_patterns(self, node: UnifiedNode) -> List[str]:
        """Extrae patrones específicos de JavaScript."""
        return ["callback_pattern", "promise_pattern", "async_await"]
    
    def _detect_javascript_algorithms(self, node: UnifiedNode) -> List[str]:
        """Detecta algoritmos de JavaScript."""
        return ["array_methods", "functional_programming"]
    
    def _extract_javascript_externals(self, node: UnifiedNode) -> List[str]:
        """Extrae interacciones externas de JavaScript."""
        return ["npm_modules", "browser_apis", "node_apis"]
    
    def _calculate_javascript_complexity(self, node: UnifiedNode) -> Dict[str, float]:
        """Calcula complejidad de JavaScript."""
        return {"cyclomatic": 1.0, "callback_depth": 0.0}
    
    def _extract_typescript_type_patterns(self, node: UnifiedNode) -> List[str]:
        """Extrae patrones de tipos de TypeScript."""
        return ["interface_usage", "generic_types", "type_guards"]
    
    def _extract_rust_control_patterns(self, node: UnifiedNode) -> List[str]:
        """Extrae patrones de Rust."""
        return ["ownership_patterns", "borrowing", "error_handling"]
    
    def _detect_rust_algorithms(self, node: UnifiedNode) -> List[str]:
        """Detecta algoritmos de Rust."""
        return ["iterator_patterns", "zero_cost_abstractions"]
    
    def _extract_rust_externals(self, node: UnifiedNode) -> List[str]:
        """Extrae interacciones externas de Rust."""
        return ["crate_usage", "system_calls"]
    
    def _calculate_rust_complexity(self, node: UnifiedNode) -> Dict[str, float]:
        """Calcula complejidad de Rust."""
        return {"cyclomatic": 1.0, "ownership_complexity": 0.0}


class CrossLanguagePatternMatcher:
    """Encuentra patrones comunes entre lenguajes."""
    
    async def find_common_patterns(self, unit1: CrossLanguageUnit, unit2: CrossLanguageUnit) -> float:
        """
        Encuentra patrones comunes entre dos unidades.
        
        Args:
            unit1: Primera unidad
            unit2: Segunda unidad
            
        Returns:
            Similitud de patrones (0.0 - 1.0)
        """
        pattern_similarities = []
        
        # Similitud de algoritmos
        algo_sim = self._calculate_algorithm_similarity(unit1, unit2)
        pattern_similarities.append(algo_sim)
        
        # Similitud de patrones de control
        control_sim = self._calculate_control_pattern_similarity(unit1, unit2)
        pattern_similarities.append(control_sim)
        
        # Similitud de interacciones externas
        external_sim = self._calculate_external_interaction_similarity(unit1, unit2)
        pattern_similarities.append(external_sim)
        
        # Promedio ponderado
        weights = [0.4, 0.4, 0.2]  # Algoritmos y control más importantes
        return sum(sim * weight for sim, weight in zip(pattern_similarities, weights))
    
    def _calculate_algorithm_similarity(self, unit1: CrossLanguageUnit, unit2: CrossLanguageUnit) -> float:
        """Calcula similitud de algoritmos."""
        algos1 = set(unit1.semantic_representation.algorithms)
        algos2 = set(unit2.semantic_representation.algorithms)
        
        if not algos1 and not algos2:
            return 1.0
        
        intersection = len(algos1.intersection(algos2))
        union = len(algos1.union(algos2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_control_pattern_similarity(self, unit1: CrossLanguageUnit, unit2: CrossLanguageUnit) -> float:
        """Calcula similitud de patrones de control."""
        patterns1 = set(unit1.semantic_representation.control_patterns)
        patterns2 = set(unit2.semantic_representation.control_patterns)
        
        if not patterns1 and not patterns2:
            return 1.0
        
        # Mapeo de patrones equivalentes entre lenguajes
        equivalent_patterns = {
            'for_loop': {'for_statement', 'iteration', 'loop_for'},
            'conditional': {'if_statement', 'conditional', 'branching'},
            'exception_handling': {'try_catch', 'error_handling', 'exception_handling'},
        }
        
        # Normalizar patrones a equivalentes
        normalized1 = self._normalize_patterns(patterns1, equivalent_patterns)
        normalized2 = self._normalize_patterns(patterns2, equivalent_patterns)
        
        intersection = len(normalized1.intersection(normalized2))
        union = len(normalized1.union(normalized2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_external_interaction_similarity(self, unit1: CrossLanguageUnit, unit2: CrossLanguageUnit) -> float:
        """Calcula similitud de interacciones externas."""
        externals1 = set(unit1.semantic_representation.external_interactions)
        externals2 = set(unit2.semantic_representation.external_interactions)
        
        if not externals1 and not externals2:
            return 1.0
        
        # Categorizar interacciones externas
        categories1 = self._categorize_external_interactions(externals1)
        categories2 = self._categorize_external_interactions(externals2)
        
        intersection = len(categories1.intersection(categories2))
        union = len(categories1.union(categories2))
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_patterns(self, patterns: Set[str], equivalents: Dict[str, Set[str]]) -> Set[str]:
        """Normaliza patrones a equivalentes canónicos."""
        normalized = set()
        
        for pattern in patterns:
            found_equivalent = False
            for canonical, equivalents_set in equivalents.items():
                if pattern in equivalents_set:
                    normalized.add(canonical)
                    found_equivalent = True
                    break
            
            if not found_equivalent:
                normalized.add(pattern)
        
        return normalized
    
    def _categorize_external_interactions(self, interactions: Set[str]) -> Set[str]:
        """Categoriza interacciones externas."""
        categories = set()
        
        for interaction in interactions:
            interaction_lower = interaction.lower()
            
            if any(keyword in interaction_lower for keyword in ['http', 'api', 'request']):
                categories.add('http_client')
            elif any(keyword in interaction_lower for keyword in ['database', 'sql', 'db']):
                categories.add('database')
            elif any(keyword in interaction_lower for keyword in ['file', 'io', 'read', 'write']):
                categories.add('file_io')
            elif any(keyword in interaction_lower for keyword in ['import', 'require', 'use']):
                categories.add('module_import')
            else:
                categories.add('other')
        
        return categories


class CrossLanguageCloneDetector:
    """Detector principal de clones cross-language."""
    
    def __init__(self, min_similarity: float = 0.6):
        self.concept_mapper = ConceptMapper()
        self.semantic_translator = SemanticTranslator()
        self.pattern_matcher = CrossLanguagePatternMatcher()
        self.min_similarity = min_similarity
    
    async def detect_cross_language_clones(
        self, 
        parse_results: List[ParseResult],
        config: Optional[Dict[str, Any]] = None
    ) -> CrossLanguageCloneDetectionResult:
        """
        Detecta clones entre diferentes lenguajes.
        
        Args:
            parse_results: Lista de resultados de parsing
            config: Configuración opcional
            
        Returns:
            CrossLanguageCloneDetectionResult con los clones encontrados
        """
        start_time = time.time()
        
        try:
            # Configurar parámetros
            min_similarity = config.get('min_similarity', self.min_similarity) if config else self.min_similarity
            
            logger.debug(f"Analizando {len(parse_results)} archivos para clones cross-language")
            
            # 1. Extraer unidades cross-language de todos los archivos
            all_units = []
            units_by_language = defaultdict(list)
            
            for parse_result in parse_results:
                units = await self._extract_cross_language_units(parse_result)
                all_units.extend(units)
                units_by_language[parse_result.language].extend(units)
            
            logger.debug(f"Extraídas {len(all_units)} unidades cross-language")
            
            # 2. Agrupar por pares de lenguajes
            language_pairs = []
            languages = list(units_by_language.keys())
            
            for i in range(len(languages)):
                for j in range(i + 1, len(languages)):
                    lang1, lang2 = languages[i], languages[j]
                    language_pairs.append((lang1, lang2))
            
            # 3. Comparar unidades entre diferentes lenguajes
            cross_language_clones = []
            comparison_count = 0
            
            for lang1, lang2 in language_pairs:
                units1 = units_by_language[lang1]
                units2 = units_by_language[lang2]
                
                for unit1 in units1:
                    for unit2 in units2:
                        # Solo comparar unidades compatibles
                        if unit1.is_comparable_to(unit2):
                            comparison_count += 1
                            
                            # Comparar cross-language
                            similarity = await self._compare_cross_language_units(unit1, unit2)
                            
                            if similarity.is_significant(min_similarity):
                                cross_language_clone = CrossLanguageClone(
                                    id=CloneId(),
                                    clone_type=CloneType.CROSS_LANGUAGE,
                                    original_location=unit1.location,
                                    duplicate_location=unit2.location,
                                    similarity_score=similarity.overall_similarity,
                                    confidence=similarity.confidence,
                                    size_lines=unit2.location.end_line - unit2.location.start_line + 1,
                                    size_tokens=len(unit2.ast_node.children),
                                    language1=unit1.language,
                                    language2=unit2.language,
                                    concept_mapping=similarity.concept_mapping,
                                    translation_evidence=similarity.translation_evidence
                                )
                                
                                cross_language_clones.append(cross_language_clone)
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Detección cross-language completada: "
                f"{len(cross_language_clones)} clones encontrados en {total_time}ms "
                f"({comparison_count} comparaciones entre {len(language_pairs)} pares de lenguajes)"
            )
            
            return CrossLanguageCloneDetectionResult(
                cross_language_clones=cross_language_clones,
                units_analyzed_by_language={lang: len(units) for lang, units in units_by_language.items()},
                comparison_count=comparison_count,
                analysis_time_ms=total_time,
                language_pairs_processed=language_pairs
            )
            
        except Exception as e:
            logger.error(f"Error detectando clones cross-language: {e}")
            raise
    
    async def _extract_cross_language_units(self, parse_result: ParseResult) -> List[CrossLanguageUnit]:
        """Extrae unidades cross-language de un archivo."""
        units = []
        unit_counter = 0
        
        async def traverse_and_extract(node: UnifiedNode):
            nonlocal unit_counter
            
            # Extraer funciones como unidades cross-language
            if node.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
                location = self._get_node_location(node, parse_result.file_path)
                
                # Determinar tipo de unidad
                unit_type = self._determine_unit_type(node)
                
                # Crear representación semántica
                semantic_repr = await self.semantic_translator.translate_to_common_representation(
                    CrossLanguageUnit(  # Crear temporal para traducción
                        id=f"temp_{unit_counter}",
                        unit_type=unit_type,
                        language=parse_result.language,
                        ast_node=node,
                        location=location,
                        semantic_representation=SemanticRepresentation(
                            concept_type=unit_type.value,
                            intent="unknown"
                        )
                    )
                )
                
                # Crear unidad final
                unit = CrossLanguageUnit(
                    id=f"cl_unit_{unit_counter}",
                    unit_type=unit_type,
                    language=parse_result.language,
                    ast_node=node,
                    location=location,
                    semantic_representation=semantic_repr,
                    language_specific_info=self._extract_language_specific_info(node, parse_result.language)
                )
                
                units.append(unit)
                unit_counter += 1
            
            # Continuar con hijos
            for child in node.children:
                await traverse_and_extract(child)
        
        # Convertir tree-sitter a unified y procesar
        if hasattr(parse_result.tree, 'root_node'):
            root_unified = self._convert_tree_sitter_to_unified(parse_result.tree.root_node)
            await traverse_and_extract(root_unified)
        
        return units
    
    async def _compare_cross_language_units(self, unit1: CrossLanguageUnit, unit2: CrossLanguageUnit) -> CrossLanguageSimilarity:
        """Compara dos unidades cross-language."""
        # Mapeo de conceptos
        concept_mapping = await self.concept_mapper.map_concepts(unit1, unit2)
        
        # Similitud semántica basada en representaciones traducidas
        semantic_similarity = self._calculate_semantic_similarity(
            unit1.semantic_representation, 
            unit2.semantic_representation
        )
        
        # Similitud de patrones
        pattern_similarity = await self.pattern_matcher.find_common_patterns(unit1, unit2)
        
        # Similitud general
        overall_similarity = (semantic_similarity + pattern_similarity) / 2.0
        
        # Recopilar evidencia de traducción
        translation_evidence = self._collect_translation_evidence(unit1, unit2, concept_mapping)
        
        # Calcular confianza
        confidence = self._calculate_cross_language_confidence(
            overall_similarity, concept_mapping, translation_evidence
        )
        
        return CrossLanguageSimilarity(
            overall_similarity=overall_similarity,
            semantic_similarity=semantic_similarity,
            pattern_similarity=pattern_similarity,
            concept_mapping=concept_mapping,
            translation_evidence=translation_evidence,
            confidence=confidence
        )
    
    def _determine_unit_type(self, node: UnifiedNode) -> CrossLanguageUnitType:
        """Determina el tipo de unidad cross-language."""
        if hasattr(node, 'value') and node.value:
            value_lower = node.value.lower()
            
            # Patrones de algoritmos
            if any(word in value_lower for word in ['sort', 'search', 'calculate', 'compute']):
                return CrossLanguageUnitType.ALGORITHM
            
            # Patrones de lógica de negocio
            elif any(word in value_lower for word in ['process', 'handle', 'manage', 'validate']):
                return CrossLanguageUnitType.BUSINESS_LOGIC
            
            # Patrones de utilidades
            elif any(word in value_lower for word in ['util', 'helper', 'convert', 'transform']):
                return CrossLanguageUnitType.UTILITY_FUNCTION
            
            # Patrones de API
            elif any(word in value_lower for word in ['api', 'endpoint', 'route', 'handler']):
                return CrossLanguageUnitType.API_ENDPOINT
        
        return CrossLanguageUnitType.BUSINESS_LOGIC  # Default
    
    def _calculate_semantic_similarity(self, repr1: SemanticRepresentation, repr2: SemanticRepresentation) -> float:
        """Calcula similitud semántica entre representaciones."""
        similarities = []
        
        # Similitud de intención
        intent_sim = 1.0 if repr1.intent == repr2.intent else 0.5
        similarities.append(intent_sim)
        
        # Similitud de algoritmos
        algos1, algos2 = set(repr1.algorithms), set(repr2.algorithms)
        algo_sim = len(algos1.intersection(algos2)) / len(algos1.union(algos2)) if algos1.union(algos2) else 1.0
        similarities.append(algo_sim)
        
        # Similitud de patrones de control
        patterns1, patterns2 = set(repr1.control_patterns), set(repr2.control_patterns)
        pattern_sim = len(patterns1.intersection(patterns2)) / len(patterns1.union(patterns2)) if patterns1.union(patterns2) else 1.0
        similarities.append(pattern_sim)
        
        # Similitud de complejidad
        complexity1 = repr1.complexity_metrics.get('cyclomatic', 1.0)
        complexity2 = repr2.complexity_metrics.get('cyclomatic', 1.0)
        complexity_sim = 1.0 - abs(complexity1 - complexity2) / max(complexity1, complexity2)
        similarities.append(complexity_sim)
        
        return sum(similarities) / len(similarities)
    
    def _collect_translation_evidence(self, unit1: CrossLanguageUnit, unit2: CrossLanguageUnit, 
                                    mapping: ConceptMapping) -> List[TranslationEvidence]:
        """Recopila evidencia de traducción."""
        evidence = []
        
        # Evidencia de mapeos directos
        if mapping.direct_mappings:
            evidence.append(TranslationEvidence(
                evidence_type="direct_concept_mapping",
                description=f"Found {len(mapping.direct_mappings)} direct concept mappings",
                confidence=0.9
            ))
        
        # Evidencia de representación semántica similar
        if unit1.semantic_representation.get_canonical_hash() == unit2.semantic_representation.get_canonical_hash():
            evidence.append(TranslationEvidence(
                evidence_type="identical_semantic_representation",
                description="Identical semantic representations after translation",
                confidence=1.0
            ))
        
        # Evidencia de complejidad similar
        complexity1 = unit1.semantic_representation.complexity_metrics.get('cyclomatic', 1.0)
        complexity2 = unit2.semantic_representation.complexity_metrics.get('cyclomatic', 1.0)
        if abs(complexity1 - complexity2) < 2.0:
            evidence.append(TranslationEvidence(
                evidence_type="similar_complexity",
                description=f"Similar complexity: {complexity1:.1f} vs {complexity2:.1f}",
                confidence=0.8
            ))
        
        return evidence
    
    def _calculate_cross_language_confidence(self, similarity: float, mapping: ConceptMapping, 
                                           evidence: List[TranslationEvidence]) -> float:
        """Calcula confianza en la comparación cross-language."""
        # Confianza base de la similitud
        base_confidence = similarity
        
        # Boost por calidad del mapeo
        mapping_boost = mapping.overall_confidence * 0.2
        
        # Boost por evidencia fuerte
        evidence_boost = sum(ev.confidence for ev in evidence) * 0.1
        
        confidence = base_confidence + mapping_boost + evidence_boost
        return min(1.0, confidence)
    
    def _extract_language_specific_info(self, node: UnifiedNode, language: ProgrammingLanguage) -> Dict[str, Any]:
        """Extrae información específica del lenguaje."""
        info = {
            "language": language.value,
            "node_count": len(node.children),
            "has_docstring": False,
            "async_patterns": [],
        }
        
        # Información específica por lenguaje
        if language == ProgrammingLanguage.PYTHON:
            info.update(self._extract_python_specific_info(node))
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            info.update(self._extract_js_specific_info(node))
        elif language == ProgrammingLanguage.RUST:
            info.update(self._extract_rust_specific_info(node))
        
        return info
    
    def _extract_python_specific_info(self, node: UnifiedNode) -> Dict[str, Any]:
        """Extrae información específica de Python."""
        return {
            "decorators": [],
            "context_managers": False,
            "type_hints": False,
            "comprehensions": False
        }
    
    def _extract_js_specific_info(self, node: UnifiedNode) -> Dict[str, Any]:
        """Extrae información específica de JavaScript/TypeScript."""
        return {
            "arrow_functions": False,
            "promises": False,
            "closures": False,
            "prototype_usage": False
        }
    
    def _extract_rust_specific_info(self, node: UnifiedNode) -> Dict[str, Any]:
        """Extrae información específica de Rust."""
        return {
            "ownership_patterns": [],
            "lifetimes": False,
            "traits": [],
            "unsafe_blocks": False
        }
    
    def _convert_tree_sitter_to_unified(self, tree_sitter_node) -> UnifiedNode:
        """Convierte tree-sitter node a UnifiedNode."""
        # Implementación similar a la de otros detectores
        # Crear SourcePosition correcta
        position = SourcePosition(
            start_line=tree_sitter_node.start_point[0],
            start_column=tree_sitter_node.start_point[1],
            end_line=tree_sitter_node.end_point[0],
            end_column=tree_sitter_node.end_point[1],
            start_byte=tree_sitter_node.start_byte if hasattr(tree_sitter_node, 'start_byte') else 0,
            end_byte=tree_sitter_node.end_byte if hasattr(tree_sitter_node, 'end_byte') else 0
        )
        
        unified_node = UnifiedNode(
            node_type=UnifiedNodeType.LANGUAGE_SPECIFIC,
            position=position,
            children=[],
            value=tree_sitter_node.text.decode('utf-8') if tree_sitter_node.text else ""
        )
        
        # Mapeo de tipos
        type_mapping = {
            'function_definition': UnifiedNodeType.FUNCTION_DECLARATION,
            'class_definition': UnifiedNodeType.CLASS_DECLARATION,
            'if_statement': UnifiedNodeType.IF_STATEMENT,
            'for_statement': UnifiedNodeType.FOR_STATEMENT,
            'while_statement': UnifiedNodeType.WHILE_STATEMENT,
            'return_statement': UnifiedNodeType.RETURN_STATEMENT,
        }
        
        if hasattr(tree_sitter_node, 'type'):
            unified_node.node_type = type_mapping.get(tree_sitter_node.type, UnifiedNodeType.LANGUAGE_SPECIFIC)
        
        # Convertir hijos
        if hasattr(tree_sitter_node, 'children'):
            for child in tree_sitter_node.children:
                unified_child = self._convert_tree_sitter_to_unified(child)
                unified_node.children.append(unified_child)
        
        return unified_node
    
    def _get_node_location(self, node: UnifiedNode, file_path: Path) -> CodeLocation:
        """Obtiene ubicación del nodo."""
        return CodeLocation(
            file_path=file_path,
            start_line=node.position.start_line + 1,
            end_line=node.position.end_line + 1,
            start_column=node.position.start_column,
            end_column=node.position.end_column
        )
