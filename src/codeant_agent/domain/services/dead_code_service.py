"""
Servicios del dominio para análisis de código muerto.

Este módulo contiene los servicios del dominio que encapsulan la lógica
de negocio para la detección y análisis de código muerto.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

from ..entities.dead_code_analysis import (
    DeadCodeAnalysis, ProjectDeadCodeAnalysis, UnusedVariable, UnusedFunction,
    UnusedClass, UnusedImport, UnreachableCode, EntryPoint, DeadCodeStatistics
)
from ..entities.dependency_analysis import (
    ControlFlowGraph, DependencyGraph, GlobalDependencyGraph, SymbolId,
    SymbolInfo, LivenessInfo, DefUseChain, UsageAnalysis
)
from ..entities.parse_result import ParseResult
from ..value_objects.programming_language import ProgrammingLanguage


class ReachabilityDomainService(ABC):
    """
    Servicio de dominio para análisis de alcanzabilidad.
    
    Encapsula la lógica de negocio para determinar qué código
    es alcanzable desde los puntos de entrada del programa.
    """
    
    @abstractmethod
    def calculate_reachability(
        self,
        global_graph: GlobalDependencyGraph,
        entry_points: List[EntryPoint]
    ) -> Set[SymbolId]:
        """
        Calcula todos los símbolos alcanzables desde los puntos de entrada.
        
        Args:
            global_graph: Grafo global de dependencias
            entry_points: Lista de puntos de entrada
            
        Returns:
            Set de símbolos alcanzables
        """
        pass
    
    @abstractmethod
    def find_unreachable_symbols(
        self,
        global_graph: GlobalDependencyGraph,
        reachable_symbols: Set[SymbolId]
    ) -> Set[SymbolId]:
        """
        Encuentra símbolos que no son alcanzables.
        
        Args:
            global_graph: Grafo global de dependencias
            reachable_symbols: Set de símbolos alcanzables
            
        Returns:
            Set de símbolos no alcanzables
        """
        pass
    
    @abstractmethod
    def analyze_code_paths(
        self,
        control_flow_graph: ControlFlowGraph
    ) -> Dict[str, Any]:
        """
        Analiza todos los caminos posibles en el código.
        
        Args:
            control_flow_graph: Grafo de control de flujo
            
        Returns:
            Diccionario con información de los caminos
        """
        pass
    
    @abstractmethod
    def detect_dead_paths(
        self,
        control_flow_graph: ControlFlowGraph
    ) -> List[List[str]]:
        """
        Detecta caminos de código muerto.
        
        Args:
            control_flow_graph: Grafo de control de flujo
            
        Returns:
            Lista de caminos muertos
        """
        pass


class DataFlowDomainService(ABC):
    """
    Servicio de dominio para análisis de flujo de datos.
    
    Encapsula la lógica de negocio para el análisis del flujo
    de datos y variables en el código.
    """
    
    @abstractmethod
    def analyze_variable_liveness(
        self,
        control_flow_graph: ControlFlowGraph
    ) -> LivenessInfo:
        """
        Analiza la liveness de variables en el código.
        
        Args:
            control_flow_graph: Grafo de control de flujo
            
        Returns:
            LivenessInfo con información de liveness
        """
        pass
    
    @abstractmethod
    def build_def_use_chains(
        self,
        control_flow_graph: ControlFlowGraph
    ) -> List[DefUseChain]:
        """
        Construye cadenas de definición-uso para variables.
        
        Args:
            control_flow_graph: Grafo de control de flujo
            
        Returns:
            Lista de cadenas def-use
        """
        pass
    
    @abstractmethod
    def find_unused_definitions(
        self,
        def_use_chains: List[DefUseChain]
    ) -> List[str]:
        """
        Encuentra definiciones de variables no utilizadas.
        
        Args:
            def_use_chains: Lista de cadenas def-use
            
        Returns:
            Lista de definiciones no utilizadas
        """
        pass
    
    @abstractmethod
    def analyze_assignment_patterns(
        self,
        control_flow_graph: ControlFlowGraph
    ) -> Dict[str, Any]:
        """
        Analiza patrones de asignación de variables.
        
        Args:
            control_flow_graph: Grafo de control de flujo
            
        Returns:
            Diccionario con patrones de asignación
        """
        pass


class ConfidenceScoringService:
    """
    Servicio de dominio para calcular scores de confianza.
    
    Encapsula la lógica para determinar qué tan confiados estamos
    de que cierto código está realmente muerto.
    """
    
    # Weights para diferentes factores de confianza
    CONFIDENCE_WEIGHTS = {
        'static_analysis': 0.4,
        'cross_module_usage': 0.3,
        'framework_patterns': 0.2,
        'dynamic_indicators': 0.1,
    }
    
    # Patrones que reducen la confianza
    LOW_CONFIDENCE_PATTERNS = [
        'reflection_usage',
        'dynamic_imports',
        'eval_usage',
        'getattr_usage',
        'metaclass_usage',
        'decorator_magic',
        'framework_hooks',
    ]
    
    def calculate_unused_variable_confidence(
        self,
        variable: UnusedVariable,
        context: Dict[str, Any]
    ) -> float:
        """
        Calcula la confianza para una variable no utilizada.
        
        Args:
            variable: Variable no utilizada
            context: Contexto adicional del análisis
            
        Returns:
            Score de confianza entre 0.0 y 1.0
        """
        base_confidence = 0.9
        
        # Reducir confianza si es una variable especial
        if variable.name.startswith('_'):
            base_confidence -= 0.1
        
        # Reducir confianza en ciertos scopes
        if variable.scope and variable.scope.scope_type.value in ['global', 'module']:
            base_confidence -= 0.2
        
        # Considerar patrones del lenguaje
        if 'language_patterns' in context:
            for pattern in context['language_patterns']:
                if pattern in self.LOW_CONFIDENCE_PATTERNS:
                    base_confidence -= 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def calculate_unused_function_confidence(
        self,
        function: UnusedFunction,
        context: Dict[str, Any]
    ) -> float:
        """
        Calcula la confianza para una función no utilizada.
        
        Args:
            function: Función no utilizada
            context: Contexto adicional del análisis
            
        Returns:
            Score de confianza entre 0.0 y 1.0
        """
        base_confidence = 0.8
        
        # Funciones públicas tienen menor confianza
        if function.visibility.value == 'public':
            base_confidence -= 0.3
        
        # Funciones con efectos secundarios tienen menor confianza
        if function.potential_side_effects:
            base_confidence -= 0.2
        
        # Funciones especiales (magic methods, etc.)
        if function.name.startswith('__') and function.name.endswith('__'):
            base_confidence -= 0.4
        
        # Considerar framework patterns
        if self._is_framework_function(function, context):
            base_confidence -= 0.5
        
        return max(0.1, min(1.0, base_confidence))
    
    def calculate_unreachable_code_confidence(
        self,
        unreachable: UnreachableCode,
        context: Dict[str, Any]
    ) -> float:
        """
        Calcula la confianza para código inalcanzable.
        
        Args:
            unreachable: Código inalcanzable
            context: Contexto adicional del análisis
            
        Returns:
            Score de confianza entre 0.0 y 1.0
        """
        # Código después de return/throw tiene alta confianza
        high_confidence_reasons = [
            'after_return',
            'after_throw',
            'always_false_condition'
        ]
        
        if unreachable.reason.value in high_confidence_reasons:
            return 0.95
        
        # Otras situaciones tienen confianza media-alta
        return 0.75
    
    def calculate_import_confidence(
        self,
        import_item: UnusedImport,
        context: Dict[str, Any]
    ) -> float:
        """
        Calcula la confianza para un import no utilizado.
        
        Args:
            import_item: Import no utilizado
            context: Contexto adicional del análisis
            
        Returns:
            Score de confianza entre 0.0 y 1.0
        """
        base_confidence = 0.9
        
        # Imports con efectos secundarios tienen menor confianza
        if import_item.side_effects_possible:
            base_confidence -= 0.7
        
        # Imports de sistema/framework tienen menor confianza
        if self._is_system_import(import_item):
            base_confidence -= 0.3
        
        return max(0.1, min(1.0, base_confidence))
    
    def _is_framework_function(
        self,
        function: UnusedFunction,
        context: Dict[str, Any]
    ) -> bool:
        """Verifica si una función sigue patrones de framework."""
        framework_patterns = [
            'test_',
            'setUp',
            'tearDown',
            'handle_',
            'on_',
            'callback_',
        ]
        
        return any(function.name.startswith(pattern) for pattern in framework_patterns)
    
    def _is_system_import(self, import_item: UnusedImport) -> bool:
        """Verifica si es un import de sistema."""
        system_modules = [
            'os',
            'sys',
            'logging',
            'warnings',
            '__future__',
        ]
        
        return import_item.module_name in system_modules


class DeadCodeClassificationService:
    """
    Servicio de dominio para clasificar y categorizar código muerto.
    
    Encapsula la lógica para categorizar diferentes tipos de código muerto
    y proporcionar sugerencias contextuales.
    """
    
    # Categorías de código muerto
    CATEGORIES = {
        'SAFE_TO_REMOVE': {
            'description': 'Código seguro de eliminar',
            'min_confidence': 0.9,
            'types': ['unused_variables', 'unreachable_code', 'dead_branches']
        },
        'LIKELY_DEAD': {
            'description': 'Probablemente muerto, revisar antes de eliminar',
            'min_confidence': 0.7,
            'types': ['unused_functions', 'unused_classes']
        },
        'POTENTIAL_DEAD': {
            'description': 'Potencialmente muerto, investigar más',
            'min_confidence': 0.5,
            'types': ['unused_imports', 'unused_parameters']
        },
        'REVIEW_REQUIRED': {
            'description': 'Requiere revisión manual',
            'min_confidence': 0.0,
            'types': ['public_api', 'framework_hooks', 'dynamic_usage']
        }
    }
    
    def classify_dead_code_analysis(
        self,
        analysis: DeadCodeAnalysis
    ) -> Dict[str, List[Any]]:
        """
        Clasifica todos los issues encontrados en el análisis.
        
        Args:
            analysis: Análisis de código muerto
            
        Returns:
            Diccionario con issues clasificados por categoría
        """
        classification = {category: [] for category in self.CATEGORIES}
        
        # Clasificar cada tipo de issue
        for issue in analysis.get_all_issues():
            category = self._classify_issue(issue)
            classification[category].append(issue)
        
        return classification
    
    def generate_removal_suggestions(
        self,
        classified_issues: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Genera sugerencias de eliminación basadas en la clasificación.
        
        Args:
            classified_issues: Issues clasificados por categoría
            
        Returns:
            Lista de sugerencias de eliminación
        """
        suggestions = []
        
        # Sugerencias para código seguro de eliminar
        safe_to_remove = classified_issues['SAFE_TO_REMOVE']
        if safe_to_remove:
            suggestions.append({
                'action': 'remove_immediately',
                'items': safe_to_remove,
                'description': 'Eliminar inmediatamente - alta confianza',
                'risk_level': 'low'
            })
        
        # Sugerencias para código probablemente muerto
        likely_dead = classified_issues['LIKELY_DEAD']
        if likely_dead:
            suggestions.append({
                'action': 'review_and_remove',
                'items': likely_dead,
                'description': 'Revisar y eliminar - confianza media-alta',
                'risk_level': 'medium'
            })
        
        # Sugerencias para código potencialmente muerto
        potential_dead = classified_issues['POTENTIAL_DEAD']
        if potential_dead:
            suggestions.append({
                'action': 'investigate_further',
                'items': potential_dead,
                'description': 'Investigar más antes de eliminar',
                'risk_level': 'medium-high'
            })
        
        return suggestions
    
    def calculate_removal_impact(
        self,
        issues: List[Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calcula el impacto de eliminar ciertos issues.
        
        Args:
            issues: Lista de issues a eliminar
            context: Contexto del proyecto
            
        Returns:
            Diccionario con métricas de impacto
        """
        impact = {
            'lines_saved': 0,
            'bytes_saved': 0,
            'complexity_reduction': 0,
            'maintenance_reduction': 0,
            'risk_assessment': 'low'
        }
        
        for issue in issues:
            # Calcular líneas que se eliminarían
            if hasattr(issue, 'location'):
                lines = issue.location.end.line - issue.location.start.line + 1
                impact['lines_saved'] += lines
                impact['bytes_saved'] += lines * 50  # Estimación
        
        # Evaluar riesgo general
        high_confidence_issues = [i for i in issues if i.confidence > 0.8]
        if len(high_confidence_issues) / len(issues) < 0.5:
            impact['risk_assessment'] = 'high'
        elif len(high_confidence_issues) / len(issues) < 0.7:
            impact['risk_assessment'] = 'medium'
        
        return impact
    
    def _classify_issue(self, issue: Any) -> str:
        """Clasifica un issue individual."""
        confidence = getattr(issue, 'confidence', 0.0)
        
        # Clasificar por confianza y tipo
        for category, config in self.CATEGORIES.items():
            if confidence >= config['min_confidence']:
                return category
        
        return 'REVIEW_REQUIRED'
