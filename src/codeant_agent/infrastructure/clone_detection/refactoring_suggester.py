"""
Implementación del generador de sugerencias de refactoring.

Este módulo analiza clones detectados y genera sugerencias inteligentes
de refactoring para eliminar duplicación y mejorar la calidad del código.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
import time
from pathlib import Path
from collections import defaultdict, Counter

from ...domain.entities.clone_analysis import (
    RefactoringOpportunity, RefactoringType, EstimatedEffort, RefactoringBenefit,
    RefactoringStep, CodeChange, CloneClass, Clone, CloneType,
    ExactClone, StructuralClone, SemanticClone, CrossLanguageClone
)
from ...domain.entities.ast_normalization import NormalizedNode as UnifiedNode, NodeType as UnifiedNodeType
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class RefactoringAnalysis:
    """Análisis de oportunidades de refactoring."""
    clone_class: CloneClass
    refactoring_complexity: str  # "simple", "moderate", "complex"
    code_pattern: str  # Patrón de código identificado
    abstraction_level: str  # "method", "class", "module", "library"
    impact_scope: List[str]  # Archivos o módulos afectados
    prerequisite_checks: List[str]  # Verificaciones necesarias


@dataclass
class RefactoringImpact:
    """Análisis de impacto de un refactoring."""
    files_affected: Set[Path]
    functions_affected: List[str]
    classes_affected: List[str]
    modules_affected: List[str]
    potential_breaking_changes: List[str]
    test_files_to_update: List[Path]
    documentation_updates: List[str]
    
    def get_risk_level(self) -> str:
        """Obtiene nivel de riesgo del refactoring."""
        risk_factors = (
            len(self.files_affected) +
            len(self.functions_affected) +
            len(self.potential_breaking_changes) * 2
        )
        
        if risk_factors <= 3:
            return "low"
        elif risk_factors <= 8:
            return "medium"
        else:
            return "high"


@dataclass
class RefactoringTemplate:
    """Template de refactoring específico."""
    refactoring_type: RefactoringType
    applicable_clone_types: List[CloneType]
    min_clone_count: int
    max_clone_count: int
    language_support: List[ProgrammingLanguage]
    complexity_factors: List[str]
    
    def is_applicable(self, clone_class: CloneClass) -> bool:
        """Verifica si el template es aplicable."""
        return (
            clone_class.clone_type in self.applicable_clone_types and
            self.min_clone_count <= len(clone_class.instances) <= self.max_clone_count
        )


class PatternAnalyzer:
    """Analizador de patrones de código en clones."""
    
    def analyze_clone_patterns(self, clone_class: CloneClass) -> Dict[str, Any]:
        """
        Analiza patrones en una clase de clones.
        
        Args:
            clone_class: Clase de clones a analizar
            
        Returns:
            Diccionario con patrones identificados
        """
        patterns = {
            'common_parameters': self._find_common_parameters(clone_class),
            'variable_differences': self._analyze_variable_differences(clone_class),
            'control_flow_patterns': self._analyze_control_flow_patterns(clone_class),
            'data_patterns': self._analyze_data_patterns(clone_class),
            'algorithmic_patterns': self._analyze_algorithmic_patterns(clone_class),
            'abstraction_candidates': self._identify_abstraction_candidates(clone_class)
        }
        
        return patterns
    
    def _find_common_parameters(self, clone_class: CloneClass) -> List[str]:
        """Encuentra parámetros comunes entre clones."""
        if not clone_class.instances:
            return []
        
        # Simplificación: buscar patrones en nombres de variables
        common_params = []
        
        for clone in clone_class.instances:
            # Extraer variables/parámetros del clone
            if hasattr(clone, 'original_location'):
                # Análisis simplificado basado en patrones de nombres
                clone_content = getattr(clone, 'content', '')
                if 'def ' in clone_content or 'function' in clone_content:
                    # Buscar parámetros de función
                    params = self._extract_function_parameters(clone_content)
                    common_params.extend(params)
        
        # Retornar parámetros que aparecen en múltiples clones
        param_counts = Counter(common_params)
        return [param for param, count in param_counts.items() if count >= 2]
    
    def _analyze_variable_differences(self, clone_class: CloneClass) -> Dict[str, Any]:
        """Analiza diferencias en variables entre clones."""
        differences = {
            'variable_name_patterns': [],
            'type_differences': [],
            'scope_differences': [],
            'initialization_patterns': []
        }
        
        if len(clone_class.instances) >= 2:
            # Comparar variables entre pares de clones
            for i in range(len(clone_class.instances) - 1):
                clone1 = clone_class.instances[i]
                clone2 = clone_class.instances[i + 1]
                
                vars1 = self._extract_variables(clone1)
                vars2 = self._extract_variables(clone2)
                
                # Analizar patrones de diferencias
                name_diffs = self._compare_variable_names(vars1, vars2)
                differences['variable_name_patterns'].extend(name_diffs)
        
        return differences
    
    def _analyze_control_flow_patterns(self, clone_class: CloneClass) -> List[str]:
        """Analiza patrones de control de flujo."""
        patterns = []
        
        for clone in clone_class.instances:
            if isinstance(clone, StructuralClone):
                # Buscar patrones estructurales
                if clone.differences:
                    for diff in clone.differences:
                        if 'control_flow' in diff.difference_type:
                            patterns.append(diff.difference_type)
        
        return list(set(patterns))  # Únicos
    
    def _analyze_data_patterns(self, clone_class: CloneClass) -> Dict[str, Any]:
        """Analiza patrones de datos."""
        return {
            'data_structures_used': self._identify_data_structures(clone_class),
            'data_transformation_patterns': self._identify_transformations(clone_class),
            'input_output_patterns': self._analyze_io_patterns(clone_class)
        }
    
    def _analyze_algorithmic_patterns(self, clone_class: CloneClass) -> List[str]:
        """Analiza patrones algorítmicos."""
        patterns = []
        
        for clone in clone_class.instances:
            content = getattr(clone, 'content', '')
            
            # Detectar algoritmos comunes
            if any(keyword in content.lower() for keyword in ['sort', 'sorted']):
                patterns.append('sorting_algorithm')
            
            if any(keyword in content.lower() for keyword in ['search', 'find']):
                patterns.append('search_algorithm')
            
            if any(keyword in content.lower() for keyword in ['filter', 'map', 'reduce']):
                patterns.append('functional_programming')
            
            if 'for' in content and 'in' in content:
                patterns.append('iteration_pattern')
        
        return list(set(patterns))
    
    def _identify_abstraction_candidates(self, clone_class: CloneClass) -> Dict[str, Any]:
        """Identifica candidatos para abstracción."""
        candidates = {
            'extract_method': self._can_extract_method(clone_class),
            'extract_class': self._can_extract_class(clone_class),
            'parameterize': self._can_parameterize(clone_class),
            'template_method': self._can_use_template_method(clone_class),
            'strategy_pattern': self._can_use_strategy_pattern(clone_class)
        }
        
        return candidates
    
    # Métodos auxiliares
    
    def _extract_function_parameters(self, content: str) -> List[str]:
        """Extrae parámetros de función (implementación simplificada)."""
        params = []
        # Implementación básica - en producción sería más sofisticada
        if '(' in content and ')' in content:
            param_section = content[content.find('(')+1:content.find(')')]
            if param_section.strip():
                params = [p.strip() for p in param_section.split(',')]
        return params
    
    def _extract_variables(self, clone: Clone) -> List[str]:
        """Extrae variables de un clone."""
        variables = []
        content = getattr(clone, 'content', '')
        
        # Implementación simplificada
        import re
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        matches = re.findall(var_pattern, content)
        variables.extend(matches)
        
        return list(set(variables))
    
    def _compare_variable_names(self, vars1: List[str], vars2: List[str]) -> List[str]:
        """Compara nombres de variables entre dos listas."""
        differences = []
        
        # Buscar patrones de diferencias (ej: user1 vs user2, dataA vs dataB)
        for v1 in vars1:
            for v2 in vars2:
                if self._are_similar_variable_names(v1, v2):
                    differences.append(f"{v1} <-> {v2}")
        
        return differences
    
    def _are_similar_variable_names(self, name1: str, name2: str) -> bool:
        """Verifica si dos nombres de variables son similares."""
        # Implementación simplificada
        return (
            abs(len(name1) - len(name2)) <= 2 and  # Similar length
            name1[:3] == name2[:3]  # Same prefix
        )
    
    def _identify_data_structures(self, clone_class: CloneClass) -> List[str]:
        """Identifica estructuras de datos usadas."""
        structures = []
        
        for clone in clone_class.instances:
            content = getattr(clone, 'content', '')
            content_lower = content.lower()
            
            if any(word in content_lower for word in ['list', 'array', '[]']):
                structures.append('list')
            
            if any(word in content_lower for word in ['dict', 'map', 'object', '{}']):
                structures.append('dictionary')
            
            if any(word in content_lower for word in ['set', 'hash']):
                structures.append('set')
        
        return list(set(structures))
    
    def _identify_transformations(self, clone_class: CloneClass) -> List[str]:
        """Identifica patrones de transformación de datos."""
        transformations = []
        
        for clone in clone_class.instances:
            content = getattr(clone, 'content', '')
            
            if 'json' in content.lower():
                transformations.append('json_serialization')
            
            if any(word in content.lower() for word in ['parse', 'convert']):
                transformations.append('data_parsing')
            
            if any(word in content.lower() for word in ['validate', 'sanitize']):
                transformations.append('data_validation')
        
        return list(set(transformations))
    
    def _analyze_io_patterns(self, clone_class: CloneClass) -> List[str]:
        """Analiza patrones de entrada/salida."""
        patterns = []
        
        for clone in clone_class.instances:
            content = getattr(clone, 'content', '')
            content_lower = content.lower()
            
            if any(word in content_lower for word in ['read', 'load', 'get']):
                patterns.append('input_pattern')
            
            if any(word in content_lower for word in ['write', 'save', 'put']):
                patterns.append('output_pattern')
            
            if any(word in content_lower for word in ['print', 'log', 'console']):
                patterns.append('logging_pattern')
        
        return list(set(patterns))
    
    def _can_extract_method(self, clone_class: CloneClass) -> bool:
        """Verifica si se puede extraer método."""
        avg_lines = clone_class.size_metrics.complexity_metrics.get('average_lines', 0)
        return (
            clone_class.clone_type == CloneType.EXACT and
            len(clone_class.instances) >= 2 and
            avg_lines >= 5
        )
    
    def _can_extract_class(self, clone_class: CloneClass) -> bool:
        """Verifica si se puede extraer clase."""
        avg_lines = clone_class.size_metrics.complexity_metrics.get('average_lines', 0)
        return (
            len(clone_class.instances) >= 3 and
            avg_lines >= 20 and
            any('class' in getattr(clone, 'content', '').lower() for clone in clone_class.instances)
        )
    
    def _can_parameterize(self, clone_class: CloneClass) -> bool:
        """Verifica si se puede parametrizar."""
        return (
            clone_class.clone_type in [CloneType.RENAMED, CloneType.NEAR_MISS] and
            len(self._find_common_parameters(clone_class)) > 0
        )
    
    def _can_use_template_method(self, clone_class: CloneClass) -> bool:
        """Verifica si se puede usar template method pattern."""
        return (
            clone_class.clone_type == CloneType.NEAR_MISS and
            len(clone_class.instances) >= 2 and
            clone_class.similarity_score >= 0.7
        )
    
    def _can_use_strategy_pattern(self, clone_class: CloneClass) -> bool:
        """Verifica si se puede usar strategy pattern."""
        return (
            clone_class.clone_type == CloneType.SEMANTIC and
            len(clone_class.instances) >= 3 and
            len(self._analyze_algorithmic_patterns(clone_class)) > 0
        )


class ComplexityAnalyzer:
    """Analizador de complejidad de refactoring."""
    
    def analyze_refactoring_complexity(self, clone_class: CloneClass, 
                                     refactoring_type: RefactoringType) -> RefactoringAnalysis:
        """
        Analiza la complejidad de un refactoring propuesto.
        
        Args:
            clone_class: Clase de clones
            refactoring_type: Tipo de refactoring propuesto
            
        Returns:
            RefactoringAnalysis con análisis de complejidad
        """
        # Determinar complejidad base
        complexity = self._determine_base_complexity(refactoring_type)
        
        # Ajustar por factores del clone class
        complexity = self._adjust_for_clone_factors(complexity, clone_class)
        
        # Identificar patrón de código
        code_pattern = self._identify_code_pattern(clone_class)
        
        # Determinar nivel de abstracción apropiado
        abstraction_level = self._determine_abstraction_level(refactoring_type, clone_class)
        
        # Calcular scope de impacto
        impact_scope = self._calculate_impact_scope(clone_class)
        
        # Identificar prerequisitos
        prerequisites = self._identify_prerequisites(refactoring_type, clone_class)
        
        return RefactoringAnalysis(
            clone_class=clone_class,
            refactoring_complexity=complexity,
            code_pattern=code_pattern,
            abstraction_level=abstraction_level,
            impact_scope=impact_scope,
            prerequisite_checks=prerequisites
        )
    
    def _determine_base_complexity(self, refactoring_type: RefactoringType) -> str:
        """Determina complejidad base por tipo de refactoring."""
        complexity_map = {
            RefactoringType.EXTRACT_METHOD: "simple",
            RefactoringType.PARAMETERIZE_METHOD: "simple",
            RefactoringType.EXTRACT_CLASS: "moderate",
            RefactoringType.TEMPLATE_METHOD: "moderate",
            RefactoringType.STRATEGY_PATTERN: "complex",
            RefactoringType.EXTRACT_LIBRARY: "complex",
            RefactoringType.MERGE_CLASSES: "moderate",
            RefactoringType.REPLACE_CONDITIONAL_WITH_POLYMORPHISM: "complex"
        }
        
        return complexity_map.get(refactoring_type, "moderate")
    
    def _adjust_for_clone_factors(self, base_complexity: str, clone_class: CloneClass) -> str:
        """Ajusta complejidad basado en factores del clone class."""
        complexity_levels = ["simple", "moderate", "complex"]
        current_level = complexity_levels.index(base_complexity)
        
        # Factores que aumentan complejidad
        if len(clone_class.instances) > 5:
            current_level = min(current_level + 1, 2)
        
        if clone_class.size_metrics.average_lines > 50:
            current_level = min(current_level + 1, 2)
        
        if clone_class.similarity_score < 0.8:
            current_level = min(current_level + 1, 2)
        
        return complexity_levels[current_level]
    
    def _identify_code_pattern(self, clone_class: CloneClass) -> str:
        """Identifica el patrón de código predominante."""
        if clone_class.clone_type == CloneType.EXACT:
            return "exact_duplication"
        elif clone_class.clone_type == CloneType.RENAMED:
            return "parameter_variation"
        elif clone_class.clone_type == CloneType.NEAR_MISS:
            return "structural_similarity"
        elif clone_class.clone_type == CloneType.SEMANTIC:
            return "algorithmic_equivalence"
        else:
            return "unknown_pattern"
    
    def _determine_abstraction_level(self, refactoring_type: RefactoringType, clone_class: CloneClass) -> str:
        """Determina el nivel de abstracción apropiado."""
        if refactoring_type in [RefactoringType.EXTRACT_METHOD, RefactoringType.PARAMETERIZE_METHOD]:
            return "method"
        elif refactoring_type in [RefactoringType.EXTRACT_CLASS, RefactoringType.TEMPLATE_METHOD]:
            return "class"
        elif refactoring_type == RefactoringType.EXTRACT_LIBRARY:
            return "library"
        else:
            return "module"
    
    def _calculate_impact_scope(self, clone_class: CloneClass) -> List[str]:
        """Calcula scope de impacto del refactoring."""
        affected_files = set()
        
        for clone in clone_class.instances:
            if hasattr(clone, 'original_location'):
                affected_files.add(str(clone.original_location.file_path))
            if hasattr(clone, 'duplicate_location'):
                affected_files.add(str(clone.duplicate_location.file_path))
        
        return list(affected_files)
    
    def _identify_prerequisites(self, refactoring_type: RefactoringType, clone_class: CloneClass) -> List[str]:
        """Identifica prerequisitos para el refactoring."""
        prerequisites = []
        
        # Prerequisites generales
        prerequisites.extend([
            "backup_code",
            "run_existing_tests",
            "analyze_dependencies"
        ])
        
        # Prerequisites específicos por tipo
        if refactoring_type == RefactoringType.EXTRACT_METHOD:
            prerequisites.extend([
                "identify_shared_variables",
                "analyze_return_values",
                "check_side_effects"
            ])
        elif refactoring_type == RefactoringType.EXTRACT_CLASS:
            prerequisites.extend([
                "identify_cohesive_methods",
                "analyze_data_usage",
                "check_inheritance_hierarchy"
            ])
        elif refactoring_type == RefactoringType.STRATEGY_PATTERN:
            prerequisites.extend([
                "identify_algorithm_variations",
                "analyze_context_usage",
                "design_strategy_interface"
            ])
        
        return prerequisites


class DependencyAnalyzer:
    """Analizador de dependencias para refactoring."""
    
    def analyze_refactoring_impact(self, clone_class: CloneClass, 
                                 refactoring_type: RefactoringType) -> RefactoringImpact:
        """
        Analiza el impacto de un refactoring propuesto.
        
        Args:
            clone_class: Clase de clones a refactorizar
            refactoring_type: Tipo de refactoring
            
        Returns:
            RefactoringImpact con análisis completo
        """
        # Identificar archivos afectados
        files_affected = self._identify_affected_files(clone_class)
        
        # Identificar funciones afectadas
        functions_affected = self._identify_affected_functions(clone_class)
        
        # Identificar clases afectadas
        classes_affected = self._identify_affected_classes(clone_class)
        
        # Identificar módulos afectados
        modules_affected = self._identify_affected_modules(clone_class)
        
        # Identificar cambios potencialmente breaking
        breaking_changes = self._identify_breaking_changes(refactoring_type, clone_class)
        
        # Identificar archivos de test a actualizar
        test_files = self._identify_test_files_to_update(files_affected)
        
        # Identificar documentación a actualizar
        doc_updates = self._identify_documentation_updates(refactoring_type, clone_class)
        
        return RefactoringImpact(
            files_affected=files_affected,
            functions_affected=functions_affected,
            classes_affected=classes_affected,
            modules_affected=modules_affected,
            potential_breaking_changes=breaking_changes,
            test_files_to_update=test_files,
            documentation_updates=doc_updates
        )
    
    def _identify_affected_files(self, clone_class: CloneClass) -> Set[Path]:
        """Identifica archivos afectados por el refactoring."""
        files = set()
        
        for clone in clone_class.instances:
            if hasattr(clone, 'original_location'):
                files.add(clone.original_location.file_path)
            if hasattr(clone, 'duplicate_location'):
                files.add(clone.duplicate_location.file_path)
        
        return files
    
    def _identify_affected_functions(self, clone_class: CloneClass) -> List[str]:
        """Identifica funciones afectadas."""
        functions = []
        
        for clone in clone_class.instances:
            if hasattr(clone, 'original_location') and clone.original_location.function_context:
                functions.append(clone.original_location.function_context)
            if hasattr(clone, 'duplicate_location') and clone.duplicate_location.function_context:
                functions.append(clone.duplicate_location.function_context)
        
        return list(set(functions))
    
    def _identify_affected_classes(self, clone_class: CloneClass) -> List[str]:
        """Identifica clases afectadas."""
        classes = []
        
        for clone in clone_class.instances:
            if hasattr(clone, 'original_location') and clone.original_location.class_context:
                classes.append(clone.original_location.class_context)
            if hasattr(clone, 'duplicate_location') and clone.duplicate_location.class_context:
                classes.append(clone.duplicate_location.class_context)
        
        return list(set(classes))
    
    def _identify_affected_modules(self, clone_class: CloneClass) -> List[str]:
        """Identifica módulos afectados."""
        modules = []
        
        for clone in clone_class.instances:
            if hasattr(clone, 'original_location'):
                module = clone.original_location.file_path.stem
                modules.append(module)
            if hasattr(clone, 'duplicate_location'):
                module = clone.duplicate_location.file_path.stem
                modules.append(module)
        
        return list(set(modules))
    
    def _identify_breaking_changes(self, refactoring_type: RefactoringType, 
                                 clone_class: CloneClass) -> List[str]:
        """Identifica cambios potencialmente breaking."""
        breaking_changes = []
        
        if refactoring_type == RefactoringType.EXTRACT_METHOD:
            breaking_changes.append("Method signature changes")
            
        elif refactoring_type == RefactoringType.EXTRACT_CLASS:
            breaking_changes.extend([
                "New class creation",
                "Method relocation",
                "Import statement changes"
            ])
            
        elif refactoring_type == RefactoringType.STRATEGY_PATTERN:
            breaking_changes.extend([
                "Interface introduction",
                "Client code adaptation",
                "Dependency injection changes"
            ])
        
        # Factores adicionales basados en el clone class
        if len(clone_class.instances) > 3:
            breaking_changes.append("Multiple file modifications")
        
        return breaking_changes
    
    def _identify_test_files_to_update(self, affected_files: Set[Path]) -> List[Path]:
        """Identifica archivos de test que necesitan actualización."""
        test_files = []
        
        for file_path in affected_files:
            # Buscar archivos de test asociados
            possible_test_files = [
                file_path.parent / f"test_{file_path.stem}.py",
                file_path.parent / f"{file_path.stem}_test.py",
                file_path.parent / "tests" / f"test_{file_path.stem}.py",
                file_path.parent.parent / "tests" / f"test_{file_path.stem}.py"
            ]
            
            for test_file in possible_test_files:
                if test_file.exists():
                    test_files.append(test_file)
        
        return test_files
    
    def _identify_documentation_updates(self, refactoring_type: RefactoringType, 
                                      clone_class: CloneClass) -> List[str]:
        """Identifica documentación que necesita actualización."""
        doc_updates = []
        
        if refactoring_type in [RefactoringType.EXTRACT_CLASS, RefactoringType.EXTRACT_LIBRARY]:
            doc_updates.extend([
                "API documentation",
                "Architecture diagrams",
                "Usage examples"
            ])
        
        if len(clone_class.instances) >= 3:
            doc_updates.append("Refactoring changelog")
        
        return doc_updates


class RefactoringSuggester:
    """Generador principal de sugerencias de refactoring."""
    
    def __init__(self):
        self.pattern_analyzer = PatternAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.refactoring_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> List[RefactoringTemplate]:
        """Inicializa templates de refactoring."""
        templates = [
            # Extract Method Template
            RefactoringTemplate(
                refactoring_type=RefactoringType.EXTRACT_METHOD,
                applicable_clone_types=[CloneType.EXACT],
                min_clone_count=2,
                max_clone_count=10,
                language_support=[
                    ProgrammingLanguage.PYTHON,
                    ProgrammingLanguage.JAVASCRIPT,
                    ProgrammingLanguage.TYPESCRIPT,
                    ProgrammingLanguage.RUST
                ],
                complexity_factors=['parameter_analysis', 'return_value_analysis']
            ),
            
            # Parameterize Method Template
            RefactoringTemplate(
                refactoring_type=RefactoringType.PARAMETERIZE_METHOD,
                applicable_clone_types=[CloneType.RENAMED, CloneType.NEAR_MISS],
                min_clone_count=2,
                max_clone_count=8,
                language_support=[ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVASCRIPT],
                complexity_factors=['variable_analysis', 'type_inference']
            ),
            
            # Template Method Template
            RefactoringTemplate(
                refactoring_type=RefactoringType.TEMPLATE_METHOD,
                applicable_clone_types=[CloneType.NEAR_MISS, CloneType.SEMANTIC],
                min_clone_count=2,
                max_clone_count=6,
                language_support=[ProgrammingLanguage.PYTHON, ProgrammingLanguage.RUST],
                complexity_factors=['inheritance_design', 'abstract_method_identification']
            ),
            
            # Strategy Pattern Template
            RefactoringTemplate(
                refactoring_type=RefactoringType.STRATEGY_PATTERN,
                applicable_clone_types=[CloneType.SEMANTIC],
                min_clone_count=3,
                max_clone_count=10,
                language_support=[
                    ProgrammingLanguage.PYTHON,
                    ProgrammingLanguage.RUST,
                    ProgrammingLanguage.TYPESCRIPT
                ],
                complexity_factors=['interface_design', 'context_analysis']
            ),
            
            # Extract Class Template
            RefactoringTemplate(
                refactoring_type=RefactoringType.EXTRACT_CLASS,
                applicable_clone_types=[CloneType.EXACT, CloneType.NEAR_MISS],
                min_clone_count=3,
                max_clone_count=15,
                language_support=[
                    ProgrammingLanguage.PYTHON,
                    ProgrammingLanguage.JAVASCRIPT,
                    ProgrammingLanguage.RUST
                ],
                complexity_factors=['cohesion_analysis', 'coupling_analysis']
            )
        ]
        
        return templates
    
    async def suggest_refactorings(self, clone_classes: List[CloneClass],
                                 config: Optional[Dict[str, Any]] = None) -> List[RefactoringOpportunity]:
        """
        Genera sugerencias de refactoring para clases de clones.
        
        Args:
            clone_classes: Lista de clases de clones
            config: Configuración opcional
            
        Returns:
            Lista de oportunidades de refactoring ordenadas por prioridad
        """
        opportunities = []
        
        for clone_class in clone_classes:
            # Analizar patrones en la clase de clones
            patterns = self.pattern_analyzer.analyze_clone_patterns(clone_class)
            
            # Encontrar templates aplicables
            applicable_templates = [
                template for template in self.refactoring_templates
                if template.is_applicable(clone_class)
            ]
            
            # Generar oportunidades para cada template aplicable
            for template in applicable_templates:
                opportunity = await self._generate_refactoring_opportunity(
                    clone_class, template, patterns, config
                )
                
                if opportunity:
                    opportunities.append(opportunity)
        
        # Ordenar por prioridad (confianza * impacto)
        opportunities.sort(
            key=lambda opp: opp.confidence * self._calculate_impact_score(opp),
            reverse=True
        )
        
        return opportunities
    
    async def _generate_refactoring_opportunity(
        self,
        clone_class: CloneClass,
        template: RefactoringTemplate,
        patterns: Dict[str, Any],
        config: Optional[Dict[str, Any]]
    ) -> Optional[RefactoringOpportunity]:
        """Genera una oportunidad de refactoring específica."""
        
        # Analizar complejidad
        complexity_analysis = self.complexity_analyzer.analyze_refactoring_complexity(
            clone_class, template.refactoring_type
        )
        
        # Analizar impacto
        impact_analysis = self.dependency_analyzer.analyze_refactoring_impact(
            clone_class, template.refactoring_type
        )
        
        # Determinar esfuerzo estimado
        estimated_effort = self._estimate_effort(template.refactoring_type, complexity_analysis, impact_analysis)
        
        # Calcular beneficios potenciales
        potential_benefits = self._calculate_benefits(template.refactoring_type, clone_class, patterns)
        
        # Generar pasos de implementación
        implementation_steps = await self._generate_implementation_steps(
            template.refactoring_type, clone_class, patterns, complexity_analysis
        )
        
        # Calcular confianza
        confidence = self._calculate_confidence(clone_class, template, patterns, impact_analysis)
        
        # Generar descripción
        description = self._generate_description(template.refactoring_type, clone_class, patterns)
        
        return RefactoringOpportunity(
            refactoring_type=template.refactoring_type,
            description=description,
            affected_clones=clone_class.instances,
            estimated_effort=estimated_effort,
            potential_benefits=potential_benefits,
            implementation_steps=implementation_steps,
            confidence=confidence,
            impact_analysis={
                "files_affected": len(impact_analysis.files_affected),
                "risk_level": impact_analysis.get_risk_level(),
                "breaking_changes": len(impact_analysis.potential_breaking_changes)
            },
            prerequisites=complexity_analysis.prerequisite_checks
        )
    
    def _estimate_effort(self, refactoring_type: RefactoringType, 
                        complexity: RefactoringAnalysis, 
                        impact: RefactoringImpact) -> EstimatedEffort:
        """Estima el esfuerzo requerido."""
        # Base effort por tipo
        base_efforts = {
            RefactoringType.EXTRACT_METHOD: EstimatedEffort.LOW,
            RefactoringType.PARAMETERIZE_METHOD: EstimatedEffort.LOW,
            RefactoringType.EXTRACT_CLASS: EstimatedEffort.MEDIUM,
            RefactoringType.TEMPLATE_METHOD: EstimatedEffort.MEDIUM,
            RefactoringType.STRATEGY_PATTERN: EstimatedEffort.HIGH,
            RefactoringType.EXTRACT_LIBRARY: EstimatedEffort.VERY_HIGH,
        }
        
        base_effort = base_efforts.get(refactoring_type, EstimatedEffort.MEDIUM)
        effort_levels = [EstimatedEffort.LOW, EstimatedEffort.MEDIUM, EstimatedEffort.HIGH, EstimatedEffort.VERY_HIGH]
        current_level = effort_levels.index(base_effort)
        
        # Ajustar por complejidad
        if complexity.refactoring_complexity == "complex":
            current_level = min(current_level + 1, 3)
        
        # Ajustar por impacto
        if len(impact.files_affected) > 5:
            current_level = min(current_level + 1, 3)
        
        if len(impact.potential_breaking_changes) > 2:
            current_level = min(current_level + 1, 3)
        
        return effort_levels[current_level]
    
    def _calculate_benefits(self, refactoring_type: RefactoringType, 
                          clone_class: CloneClass, patterns: Dict[str, Any]) -> List[RefactoringBenefit]:
        """Calcula beneficios potenciales."""
        benefits = []
        
        # Beneficios generales
        benefits.append(RefactoringBenefit.REDUCED_DUPLICATION)
        benefits.append(RefactoringBenefit.IMPROVED_MAINTAINABILITY)
        
        # Beneficios específicos por tipo
        if refactoring_type in [RefactoringType.TEMPLATE_METHOD, RefactoringType.STRATEGY_PATTERN]:
            benefits.extend([
                RefactoringBenefit.IMPROVED_DESIGN,
                RefactoringBenefit.BETTER_EXTENSIBILITY
            ])
        
        if refactoring_type == RefactoringType.EXTRACT_CLASS:
            benefits.extend([
                RefactoringBenefit.BETTER_TESTABILITY,
                RefactoringBenefit.REDUCED_COMPLEXITY
            ])
        
        # Beneficios basados en patrones
        if patterns.get('algorithmic_patterns'):
            benefits.append(RefactoringBenefit.IMPROVED_DESIGN)
        
        if len(clone_class.instances) >= 5:
            benefits.append(RefactoringBenefit.CENTRALIZED_MAINTENANCE)
        
        return list(set(benefits))  # Únicos
    
    async def _generate_implementation_steps(
        self,
        refactoring_type: RefactoringType,
        clone_class: CloneClass,
        patterns: Dict[str, Any],
        complexity: RefactoringAnalysis
    ) -> List[RefactoringStep]:
        """Genera pasos de implementación."""
        steps = []
        
        if refactoring_type == RefactoringType.EXTRACT_METHOD:
            steps.extend(await self._generate_extract_method_steps(clone_class, patterns))
        elif refactoring_type == RefactoringType.PARAMETERIZE_METHOD:
            steps.extend(await self._generate_parameterize_method_steps(clone_class, patterns))
        elif refactoring_type == RefactoringType.TEMPLATE_METHOD:
            steps.extend(await self._generate_template_method_steps(clone_class, patterns))
        elif refactoring_type == RefactoringType.STRATEGY_PATTERN:
            steps.extend(await self._generate_strategy_pattern_steps(clone_class, patterns))
        elif refactoring_type == RefactoringType.EXTRACT_CLASS:
            steps.extend(await self._generate_extract_class_steps(clone_class, patterns))
        
        return steps
    
    async def _generate_extract_method_steps(self, clone_class: CloneClass, 
                                           patterns: Dict[str, Any]) -> List[RefactoringStep]:
        """Genera pasos para Extract Method."""
        steps = [
            RefactoringStep(
                step_number=1,
                description="Identify common code in clones and extract into new method",
                code_changes=[
                    CodeChange(
                        file_path="",  # To be determined
                        change_type="create_method",
                        description="Create new extracted method",
                    )
                ],
                validation_steps=[
                    "Run existing tests to ensure no regression",
                    "Verify method signature is appropriate"
                ]
            ),
            RefactoringStep(
                step_number=2,
                description="Replace duplicated code with method calls",
                code_changes=[
                    CodeChange(
                        change_type="replace_code",
                        description="Replace clone instances with method calls",
                    )
                ],
                validation_steps=[
                    "Test each replacement individually",
                    "Verify parameter passing works correctly"
                ]
            ),
            RefactoringStep(
                step_number=3,
                description="Clean up and optimize",
                code_changes=[
                    CodeChange(
                        change_type="cleanup",
                        description="Remove unused variables and optimize",
                    )
                ],
                validation_steps=[
                    "Run full test suite",
                    "Check code coverage maintained"
                ]
            )
        ]
        
        return steps
    
    async def _generate_parameterize_method_steps(self, clone_class: CloneClass, 
                                                patterns: Dict[str, Any]) -> List[RefactoringStep]:
        """Genera pasos para Parameterize Method."""
        common_params = patterns.get('common_parameters', [])
        
        steps = [
            RefactoringStep(
                step_number=1,
                description=f"Add parameters for varying values: {', '.join(common_params[:3])}",
                code_changes=[
                    CodeChange(
                        change_type="modify_signature",
                        description="Update method signature with new parameters",
                    )
                ],
                validation_steps=[
                    "Verify parameter types are correct",
                    "Check default values if needed"
                ]
            ),
            RefactoringStep(
                step_number=2,
                description="Update method body to use parameters",
                code_changes=[
                    CodeChange(
                        change_type="update_body",
                        description="Replace hardcoded values with parameters",
                    )
                ],
                validation_steps=[
                    "Test with different parameter combinations",
                    "Verify logic still correct"
                ]
            ),
            RefactoringStep(
                step_number=3,
                description="Update all call sites",
                code_changes=[
                    CodeChange(
                        change_type="update_calls",
                        description="Pass appropriate arguments at call sites",
                    )
                ],
                validation_steps=[
                    "Test each call site individually",
                    "Run integration tests"
                ]
            )
        ]
        
        return steps
    
    async def _generate_template_method_steps(self, clone_class: CloneClass, 
                                            patterns: Dict[str, Any]) -> List[RefactoringStep]:
        """Genera pasos para Template Method."""
        steps = [
            RefactoringStep(
                step_number=1,
                description="Create base class with template method",
                code_changes=[
                    CodeChange(
                        change_type="create_base_class",
                        description="Define abstract base class with template method",
                    )
                ],
                validation_steps=[
                    "Verify class hierarchy design",
                    "Check abstract method definitions"
                ]
            ),
            RefactoringStep(
                step_number=2,
                description="Identify varying parts and create abstract methods",
                code_changes=[
                    CodeChange(
                        change_type="create_abstract_methods",
                        description="Define abstract methods for varying behavior",
                    )
                ],
                validation_steps=[
                    "Ensure method signatures are consistent",
                    "Verify abstraction level is appropriate"
                ]
            ),
            RefactoringStep(
                step_number=3,
                description="Create concrete subclasses",
                code_changes=[
                    CodeChange(
                        change_type="create_subclasses",
                        description="Implement concrete classes for each variant",
                    )
                ],
                validation_steps=[
                    "Test each subclass implementation",
                    "Verify polymorphic behavior works"
                ]
            )
        ]
        
        return steps
    
    async def _generate_strategy_pattern_steps(self, clone_class: CloneClass, 
                                             patterns: Dict[str, Any]) -> List[RefactoringStep]:
        """Genera pasos para Strategy Pattern."""
        algorithmic_patterns = patterns.get('algorithmic_patterns', [])
        
        steps = [
            RefactoringStep(
                step_number=1,
                description=f"Define strategy interface for {', '.join(algorithmic_patterns[:2])}",
                code_changes=[
                    CodeChange(
                        change_type="create_interface",
                        description="Create strategy interface/trait",
                    )
                ],
                validation_steps=[
                    "Review interface design with team",
                    "Ensure interface is cohesive"
                ]
            ),
            RefactoringStep(
                step_number=2,
                description="Implement concrete strategy classes",
                code_changes=[
                    CodeChange(
                        change_type="implement_strategies",
                        description="Create strategy implementations",
                    )
                ],
                validation_steps=[
                    "Test each strategy independently",
                    "Verify interface compliance"
                ]
            ),
            RefactoringStep(
                step_number=3,
                description="Update context class to use strategies",
                code_changes=[
                    CodeChange(
                        change_type="update_context",
                        description="Modify context to use strategy pattern",
                    )
                ],
                validation_steps=[
                    "Test strategy switching",
                    "Verify dependency injection works"
                ]
            )
        ]
        
        return steps
    
    async def _generate_extract_class_steps(self, clone_class: CloneClass, 
                                          patterns: Dict[str, Any]) -> List[RefactoringStep]:
        """Genera pasos para Extract Class."""
        steps = [
            RefactoringStep(
                step_number=1,
                description="Identify cohesive methods and data",
                code_changes=[
                    CodeChange(
                        change_type="analyze_cohesion",
                        description="Group related methods and data",
                    )
                ],
                validation_steps=[
                    "Review grouping with team",
                    "Ensure high cohesion within groups"
                ]
            ),
            RefactoringStep(
                step_number=2,
                description="Create new class with extracted elements",
                code_changes=[
                    CodeChange(
                        change_type="create_new_class",
                        description="Define new class with extracted methods/data",
                    )
                ],
                validation_steps=[
                    "Test new class independently",
                    "Verify encapsulation is proper"
                ]
            ),
            RefactoringStep(
                step_number=3,
                description="Update original class to use new class",
                code_changes=[
                    CodeChange(
                        change_type="update_original_class",
                        description="Delegate to new class or compose with it",
                    )
                ],
                validation_steps=[
                    "Test collaboration between classes",
                    "Run full integration tests"
                ]
            )
        ]
        
        return steps
    
    def _calculate_confidence(self, clone_class: CloneClass, template: RefactoringTemplate,
                            patterns: Dict[str, Any], impact: RefactoringImpact) -> float:
        """Calcula confianza en la sugerencia."""
        base_confidence = 0.7
        
        # Factores que aumentan confianza
        confidence_factors = []
        
        # Similitud alta
        if clone_class.similarity_score >= 0.9:
            confidence_factors.append(0.1)
        
        # Template muy aplicable
        if template.refactoring_type in [RefactoringType.EXTRACT_METHOD, RefactoringType.PARAMETERIZE_METHOD]:
            confidence_factors.append(0.1)
        
        # Patrones claros identificados
        if patterns.get('common_parameters') or patterns.get('algorithmic_patterns'):
            confidence_factors.append(0.1)
        
        # Bajo riesgo
        if impact.get_risk_level() == "low":
            confidence_factors.append(0.1)
        
        final_confidence = base_confidence + sum(confidence_factors)
        return min(1.0, final_confidence)
    
    def _generate_description(self, refactoring_type: RefactoringType, 
                            clone_class: CloneClass, patterns: Dict[str, Any]) -> str:
        """Genera descripción de la oportunidad."""
        clone_count = len(clone_class.instances)
        avg_size = clone_class.size_metrics.average_lines
        
        base_descriptions = {
            RefactoringType.EXTRACT_METHOD: f"Extract duplicated code from {clone_count} locations into a reusable method (avg {avg_size} lines per clone)",
            RefactoringType.PARAMETERIZE_METHOD: f"Parameterize method to handle {clone_count} variations with common parameters",
            RefactoringType.TEMPLATE_METHOD: f"Apply Template Method pattern to {clone_count} similar algorithms with {avg_size} lines average",
            RefactoringType.STRATEGY_PATTERN: f"Implement Strategy pattern for {clone_count} algorithmic variants",
            RefactoringType.EXTRACT_CLASS: f"Extract cohesive functionality from {clone_count} locations into a separate class"
        }
        
        description = base_descriptions.get(
            refactoring_type, 
            f"Apply {refactoring_type.value} refactoring to {clone_count} similar code instances"
        )
        
        # Añadir información de patrones si está disponible
        if patterns.get('algorithmic_patterns'):
            algo_info = ", ".join(patterns['algorithmic_patterns'][:2])
            description += f" (algorithms: {algo_info})"
        
        return description
    
    def _calculate_impact_score(self, opportunity: RefactoringOpportunity) -> float:
        """Calcula score de impacto de la oportunidad."""
        # Factor base por número de clones afectados
        clone_factor = min(len(opportunity.affected_clones) / 10.0, 1.0)
        
        # Factor por beneficios potenciales
        benefit_factor = len(opportunity.potential_benefits) / 8.0  # Max 8 beneficios
        
        # Factor por esfuerzo (menos esfuerzo = más impacto)
        effort_weights = {
            EstimatedEffort.LOW: 1.0,
            EstimatedEffort.MEDIUM: 0.8,
            EstimatedEffort.HIGH: 0.6,
            EstimatedEffort.VERY_HIGH: 0.4
        }
        effort_factor = effort_weights.get(opportunity.estimated_effort, 0.5)
        
        return (clone_factor + benefit_factor + effort_factor) / 3.0
