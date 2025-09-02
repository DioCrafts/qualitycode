"""
Interfaz del repositorio para análisis de código muerto.

Este módulo define el contrato para el repositorio que proporciona
capacidades de detección y análisis de código muerto.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from ..entities.dead_code_analysis import (
    DeadCodeAnalysis, ProjectDeadCodeAnalysis, DeadCodeStatistics,
    UnusedVariable, UnusedFunction, UnusedClass, UnusedImport,
    UnreachableCode, DeadBranch, UnusedParameter, RedundantAssignment,
    EntryPoint
)
from ..entities.dependency_analysis import (
    ControlFlowGraph, DependencyGraph, GlobalDependencyGraph,
    SymbolId, ModuleId, UsageAnalysis
)
from ..entities.parse_result import ParseResult
from ..value_objects.programming_language import ProgrammingLanguage


class DeadCodeRepository(ABC):
    """
    Interfaz del repositorio de análisis de código muerto.
    
    Esta interfaz define el contrato para el repositorio que maneja
    la detección y análisis de código muerto en proyectos de software.
    """
    
    @abstractmethod
    async def analyze_file_dead_code(
        self, 
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> DeadCodeAnalysis:
        """
        Analiza código muerto en un archivo individual.
        
        Args:
            parse_result: Resultado del parsing del archivo
            config: Configuración opcional para el análisis
            
        Returns:
            DeadCodeAnalysis con los resultados del análisis
            
        Raises:
            DeadCodeAnalysisError: Si hay un error durante el análisis
        """
        pass
    
    @abstractmethod
    async def analyze_project_dead_code(
        self, 
        parse_results: List[ParseResult],
        config: Optional[Dict[str, Any]] = None
    ) -> ProjectDeadCodeAnalysis:
        """
        Analiza código muerto en todo un proyecto.
        
        Args:
            parse_results: Lista de resultados de parsing de todos los archivos
            config: Configuración opcional para el análisis
            
        Returns:
            ProjectDeadCodeAnalysis con los resultados del análisis completo
            
        Raises:
            DeadCodeAnalysisError: Si hay un error durante el análisis
        """
        pass
    
    @abstractmethod
    async def build_control_flow_graph(self, parse_result: ParseResult) -> ControlFlowGraph:
        """
        Construye el grafo de control de flujo para un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            
        Returns:
            ControlFlowGraph del archivo
            
        Raises:
            ControlFlowGraphError: Si hay un error construyendo el grafo
        """
        pass
    
    @abstractmethod
    async def build_dependency_graph(self, parse_result: ParseResult) -> DependencyGraph:
        """
        Construye el grafo de dependencias para un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            
        Returns:
            DependencyGraph del archivo
            
        Raises:
            DependencyGraphError: Si hay un error construyendo el grafo
        """
        pass
    
    @abstractmethod
    async def build_global_dependency_graph(
        self, 
        parse_results: List[ParseResult]
    ) -> GlobalDependencyGraph:
        """
        Construye el grafo global de dependencias para todo el proyecto.
        
        Args:
            parse_results: Lista de resultados de parsing de todos los archivos
            
        Returns:
            GlobalDependencyGraph del proyecto completo
            
        Raises:
            DependencyGraphError: Si hay un error construyendo el grafo
        """
        pass
    
    @abstractmethod
    async def find_entry_points(
        self, 
        parse_results: List[ParseResult],
        config: Optional[Dict[str, Any]] = None
    ) -> List[EntryPoint]:
        """
        Encuentra los puntos de entrada en el proyecto.
        
        Args:
            parse_results: Lista de resultados de parsing
            config: Configuración opcional con entry points adicionales
            
        Returns:
            Lista de puntos de entrada encontrados
            
        Raises:
            EntryPointDetectionError: Si hay un error detectando entry points
        """
        pass
    
    @abstractmethod
    async def find_reachable_symbols(
        self, 
        global_graph: GlobalDependencyGraph,
        entry_points: List[EntryPoint]
    ) -> Set[SymbolId]:
        """
        Encuentra todos los símbolos alcanzables desde los entry points.
        
        Args:
            global_graph: Grafo global de dependencias
            entry_points: Lista de puntos de entrada
            
        Returns:
            Set de símbolos alcanzables
            
        Raises:
            ReachabilityAnalysisError: Si hay un error en el análisis
        """
        pass
    
    @abstractmethod
    async def detect_unused_variables(
        self, 
        parse_result: ParseResult,
        control_flow_graph: ControlFlowGraph
    ) -> List[UnusedVariable]:
        """
        Detecta variables no utilizadas en un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            control_flow_graph: Grafo de control de flujo
            
        Returns:
            Lista de variables no utilizadas
            
        Raises:
            VariableAnalysisError: Si hay un error en el análisis
        """
        pass
    
    @abstractmethod
    async def detect_unused_functions(
        self, 
        parse_result: ParseResult,
        dependency_graph: DependencyGraph,
        reachable_symbols: Optional[Set[SymbolId]] = None
    ) -> List[UnusedFunction]:
        """
        Detecta funciones no utilizadas en un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            dependency_graph: Grafo de dependencias
            reachable_symbols: Símbolos alcanzables (para análisis cross-module)
            
        Returns:
            Lista de funciones no utilizadas
            
        Raises:
            FunctionAnalysisError: Si hay un error en el análisis
        """
        pass
    
    @abstractmethod
    async def detect_unused_classes(
        self, 
        parse_result: ParseResult,
        dependency_graph: DependencyGraph,
        reachable_symbols: Optional[Set[SymbolId]] = None
    ) -> List[UnusedClass]:
        """
        Detecta clases no utilizadas en un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            dependency_graph: Grafo de dependencias
            reachable_symbols: Símbolos alcanzables (para análisis cross-module)
            
        Returns:
            Lista de clases no utilizadas
            
        Raises:
            ClassAnalysisError: Si hay un error en el análisis
        """
        pass
    
    @abstractmethod
    async def detect_unused_imports(self, parse_result: ParseResult) -> List[UnusedImport]:
        """
        Detecta imports no utilizados en un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            
        Returns:
            Lista de imports no utilizados
            
        Raises:
            ImportAnalysisError: Si hay un error en el análisis
        """
        pass
    
    @abstractmethod
    async def detect_unreachable_code(
        self, 
        parse_result: ParseResult,
        control_flow_graph: ControlFlowGraph
    ) -> List[UnreachableCode]:
        """
        Detecta código inalcanzable en un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            control_flow_graph: Grafo de control de flujo
            
        Returns:
            Lista de bloques de código inalcanzable
            
        Raises:
            ReachabilityAnalysisError: Si hay un error en el análisis
        """
        pass
    
    @abstractmethod
    async def detect_dead_branches(
        self, 
        parse_result: ParseResult,
        control_flow_graph: ControlFlowGraph
    ) -> List[DeadBranch]:
        """
        Detecta ramas muertas en condicionales.
        
        Args:
            parse_result: Resultado del parsing del archivo
            control_flow_graph: Grafo de control de flujo
            
        Returns:
            Lista de ramas muertas
            
        Raises:
            BranchAnalysisError: Si hay un error en el análisis
        """
        pass
    
    @abstractmethod
    async def detect_unused_parameters(
        self, 
        parse_result: ParseResult
    ) -> List[UnusedParameter]:
        """
        Detecta parámetros no utilizados en funciones.
        
        Args:
            parse_result: Resultado del parsing del archivo
            
        Returns:
            Lista de parámetros no utilizados
            
        Raises:
            ParameterAnalysisError: Si hay un error en el análisis
        """
        pass
    
    @abstractmethod
    async def detect_redundant_assignments(
        self, 
        parse_result: ParseResult,
        control_flow_graph: ControlFlowGraph
    ) -> List[RedundantAssignment]:
        """
        Detecta asignaciones redundantes de variables.
        
        Args:
            parse_result: Resultado del parsing del archivo
            control_flow_graph: Grafo de control de flujo
            
        Returns:
            Lista de asignaciones redundantes
            
        Raises:
            AssignmentAnalysisError: Si hay un error en el análisis
        """
        pass
    
    @abstractmethod
    async def analyze_symbol_usage(
        self, 
        symbol_id: SymbolId,
        global_graph: GlobalDependencyGraph
    ) -> UsageAnalysis:
        """
        Analiza el uso de un símbolo específico en todo el proyecto.
        
        Args:
            symbol_id: Identificador del símbolo
            global_graph: Grafo global de dependencias
            
        Returns:
            UsageAnalysis con información detallada del uso
            
        Raises:
            SymbolAnalysisError: Si hay un error en el análisis
        """
        pass
    
    @abstractmethod
    async def calculate_statistics(
        self, 
        analysis: DeadCodeAnalysis
    ) -> DeadCodeStatistics:
        """
        Calcula estadísticas detalladas del análisis de código muerto.
        
        Args:
            analysis: Análisis de código muerto
            
        Returns:
            DeadCodeStatistics con métricas calculadas
            
        Raises:
            StatisticsCalculationError: Si hay un error calculando estadísticas
        """
        pass
    
    @abstractmethod
    async def calculate_project_statistics(
        self, 
        project_analysis: ProjectDeadCodeAnalysis
    ) -> DeadCodeStatistics:
        """
        Calcula estadísticas del proyecto completo.
        
        Args:
            project_analysis: Análisis de código muerto del proyecto
            
        Returns:
            DeadCodeStatistics agregadas del proyecto
            
        Raises:
            StatisticsCalculationError: Si hay un error calculando estadísticas
        """
        pass
    
    @abstractmethod
    async def get_language_specific_config(
        self, 
        language: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Obtiene configuración específica para un lenguaje.
        
        Args:
            language: Lenguaje de programación
            
        Returns:
            Diccionario con configuración específica del lenguaje
        """
        pass
    
    @abstractmethod
    async def is_analysis_supported(self, language: ProgrammingLanguage) -> bool:
        """
        Verifica si el análisis está soportado para un lenguaje.
        
        Args:
            language: Lenguaje de programación
            
        Returns:
            True si está soportado, False en caso contrario
        """
        pass
    
    @abstractmethod
    async def get_analysis_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento del análisis.
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        pass
