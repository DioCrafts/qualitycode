"""
Implementación del repositorio de análisis de código muerto.

Este módulo implementa la interfaz DeadCodeRepository definida en el dominio.
"""

import logging
from typing import Dict, List, Set, Optional, Any
from pathlib import Path

from ...domain.repositories.dead_code_repository import DeadCodeRepository
from ...domain.entities.dead_code_analysis import (
    DeadCodeAnalysis, ProjectDeadCodeAnalysis, DeadCodeStatistics,
    UnusedVariable, UnusedFunction, UnusedClass, UnusedImport,
    UnreachableCode, DeadBranch, UnusedParameter, RedundantAssignment,
    EntryPoint
)
from ...domain.entities.dependency_analysis import (
    ControlFlowGraph, DependencyGraph, GlobalDependencyGraph,
    SymbolId, UsageAnalysis
)
from ...domain.entities.parse_result import ParseResult
from ...domain.value_objects.programming_language import ProgrammingLanguage

from ..dead_code_analysis.dead_code_detector import DeadCodeDetector, DeadCodeConfig
from ..dead_code_analysis.reachability_analyzer import ReachabilityAnalyzer
from ..dead_code_analysis.data_flow_analyzer import DataFlowAnalyzer
from ..dead_code_analysis.import_analyzer import ImportAnalyzer
from ..dead_code_analysis.cross_module_analyzer import CrossModuleAnalyzer

logger = logging.getLogger(__name__)


class DeadCodeRepositoryImpl(DeadCodeRepository):
    """Implementación del repositorio de análisis de código muerto."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el repositorio.
        
        Args:
            config: Configuración opcional para el análisis
        """
        self.config = config or {}
        self.dead_code_detector = DeadCodeDetector(self._create_detector_config())
        self.reachability_analyzer = ReachabilityAnalyzer()
        self.data_flow_analyzer = DataFlowAnalyzer()
        self.import_analyzer = ImportAnalyzer()
        self.cross_module_analyzer = CrossModuleAnalyzer()
        
        # Cache para resultados
        self._analysis_cache: Dict[str, DeadCodeAnalysis] = {}
        self._cfg_cache: Dict[str, ControlFlowGraph] = {}
    
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
        """
        try:
            # Verificar cache
            cache_key = self._get_cache_key(parse_result)
            if cache_key in self._analysis_cache:
                logger.debug(f"Usando análisis cached para {parse_result.file_path}")
                return self._analysis_cache[cache_key]
            
            # Realizar análisis
            logger.info(f"Analizando código muerto en {parse_result.file_path}")
            analysis = await self.dead_code_detector.detect_dead_code(parse_result)
            
            # Guardar en cache
            self._analysis_cache[cache_key] = analysis
            
            logger.info(
                f"Análisis completado para {parse_result.file_path}: "
                f"{analysis.statistics.get_total_issues()} issues encontrados"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando archivo {parse_result.file_path}: {e}")
            raise
    
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
        """
        try:
            logger.info(f"Iniciando análisis de proyecto con {len(parse_results)} archivos")
            
            # Actualizar configuración si se proporciona
            if config:
                self.dead_code_detector.config = self._merge_configs(
                    self.dead_code_detector.config, config
                )
            
            # Realizar análisis del proyecto
            project_analysis = await self.dead_code_detector.detect_dead_code_project(
                parse_results
            )
            
            logger.info(
                f"Análisis de proyecto completado: "
                f"{project_analysis.global_statistics.get_total_issues()} issues totales, "
                f"{len(project_analysis.cross_module_issues)} issues cross-module"
            )
            
            return project_analysis
            
        except Exception as e:
            logger.error(f"Error analizando proyecto: {e}")
            raise
    
    async def build_control_flow_graph(self, parse_result: ParseResult) -> ControlFlowGraph:
        """
        Construye el grafo de control de flujo para un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            
        Returns:
            ControlFlowGraph del archivo
        """
        try:
            # Verificar cache
            cache_key = self._get_cache_key(parse_result)
            if cache_key in self._cfg_cache:
                return self._cfg_cache[cache_key]
            
            # Construir CFG
            cfg = self.reachability_analyzer.cfg_builder.build_cfg(parse_result)
            
            # Guardar en cache
            self._cfg_cache[cache_key] = cfg
            
            logger.debug(
                f"CFG construido para {parse_result.file_path}: "
                f"{len(cfg.nodes)} nodos, {len(cfg.edges)} aristas"
            )
            
            return cfg
            
        except Exception as e:
            logger.error(f"Error construyendo CFG para {parse_result.file_path}: {e}")
            raise
    
    async def build_dependency_graph(self, parse_result: ParseResult) -> DependencyGraph:
        """
        Construye el grafo de dependencias para un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            
        Returns:
            DependencyGraph del archivo
        """
        try:
            # Crear grafo de dependencias básico
            # En una implementación real, esto sería más sofisticado
            dependency_graph = DependencyGraph()
            
            logger.debug(f"Grafo de dependencias construido para {parse_result.file_path}")
            
            return dependency_graph
            
        except Exception as e:
            logger.error(f"Error construyendo grafo de dependencias: {e}")
            raise
    
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
        """
        try:
            logger.info("Construyendo grafo global de dependencias")
            
            global_graph = self.cross_module_analyzer.graph_builder.build_global_dependency_graph(
                parse_results
            )
            
            logger.info(
                f"Grafo global construido: "
                f"{len(global_graph.modules)} módulos, "
                f"{len(global_graph.symbols)} símbolos"
            )
            
            return global_graph
            
        except Exception as e:
            logger.error(f"Error construyendo grafo global: {e}")
            raise
    
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
        """
        try:
            logger.info("Detectando puntos de entrada del proyecto")
            
            # Construir grafo global primero
            global_graph = await self.build_global_dependency_graph(parse_results)
            
            # Analizar estructura del proyecto
            project_structure = self.cross_module_analyzer.structure_analyzer.analyze_project_structure(
                parse_results
            )
            
            # Encontrar entry points
            entry_points = self.cross_module_analyzer.entry_point_detector.find_entry_points(
                parse_results, global_graph, project_structure
            )
            
            logger.info(f"Encontrados {len(entry_points)} puntos de entrada")
            
            return entry_points
            
        except Exception as e:
            logger.error(f"Error detectando puntos de entrada: {e}")
            raise
    
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
        """
        try:
            logger.info("Calculando símbolos alcanzables")
            
            # Usar el cross module analyzer para calcular alcanzabilidad
            cross_result = await self.cross_module_analyzer.analyze_cross_module_dependencies(
                [], {}  # Los parse_results ya están en el global_graph
            )
            
            reachable_symbols = cross_result.reachable_symbols
            
            logger.info(f"Encontrados {len(reachable_symbols)} símbolos alcanzables")
            
            return reachable_symbols
            
        except Exception as e:
            logger.error(f"Error calculando símbolos alcanzables: {e}")
            raise
    
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
        """
        try:
            # Usar el data flow analyzer
            data_flow_result = await self.data_flow_analyzer.analyze_data_flow(
                parse_result, control_flow_graph
            )
            
            return data_flow_result.unused_variables
            
        except Exception as e:
            logger.error(f"Error detectando variables no utilizadas: {e}")
            raise
    
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
        """
        try:
            # Análisis básico sin cross-module
            cfg = await self.build_control_flow_graph(parse_result)
            unused_functions = await self.dead_code_detector._detect_unused_functions_basic(
                parse_result, cfg
            )
            
            # Si hay símbolos alcanzables, filtrar usando esa información
            if reachable_symbols:
                filtered_functions = []
                for func in unused_functions:
                    # Verificar si la función está en los símbolos alcanzables
                    func_symbol = SymbolId(
                        str(parse_result.file_path), 
                        func.name, 
                        'function'
                    )
                    
                    if func_symbol not in reachable_symbols:
                        filtered_functions.append(func)
                
                return filtered_functions
            
            return unused_functions
            
        except Exception as e:
            logger.error(f"Error detectando funciones no utilizadas: {e}")
            raise
    
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
        """
        try:
            # Análisis básico sin cross-module
            cfg = await self.build_control_flow_graph(parse_result)
            unused_classes = await self.dead_code_detector._detect_unused_classes_basic(
                parse_result, cfg
            )
            
            return unused_classes
            
        except Exception as e:
            logger.error(f"Error detectando clases no utilizadas: {e}")
            raise
    
    async def detect_unused_imports(self, parse_result: ParseResult) -> List[UnusedImport]:
        """
        Detecta imports no utilizados en un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            
        Returns:
            Lista de imports no utilizados
        """
        try:
            import_result = await self.import_analyzer.analyze_imports(parse_result)
            return import_result.unused_imports
            
        except Exception as e:
            logger.error(f"Error detectando imports no utilizados: {e}")
            raise
    
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
        """
        try:
            reachability_result = await self.reachability_analyzer.analyze_reachability(
                parse_result
            )
            
            return reachability_result.unreachable_code_blocks
            
        except Exception as e:
            logger.error(f"Error detectando código inalcanzable: {e}")
            raise
    
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
        """
        try:
            reachability_result = await self.reachability_analyzer.analyze_reachability(
                parse_result
            )
            
            return reachability_result.dead_branches
            
        except Exception as e:
            logger.error(f"Error detectando ramas muertas: {e}")
            raise
    
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
        """
        try:
            cfg = await self.build_control_flow_graph(parse_result)
            data_flow_result = await self.data_flow_analyzer.analyze_data_flow(
                parse_result, cfg
            )
            
            return data_flow_result.unused_parameters
            
        except Exception as e:
            logger.error(f"Error detectando parámetros no utilizados: {e}")
            raise
    
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
        """
        try:
            data_flow_result = await self.data_flow_analyzer.analyze_data_flow(
                parse_result, control_flow_graph
            )
            
            return data_flow_result.redundant_assignments
            
        except Exception as e:
            logger.error(f"Error detectando asignaciones redundantes: {e}")
            raise
    
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
        """
        try:
            # Contar usos del símbolo
            dependents = global_graph.get_dependents(symbol_id)
            usage_count = len(dependents) if dependents else 0
            
            # Verificar si es exportado
            symbol_info = global_graph.symbols.get(symbol_id)
            is_exported = symbol_info.is_exported if symbol_info else False
            
            # Contar usos cross-module
            cross_module_usages = 0
            if dependents:
                for dependent in dependents:
                    if dependent.module_path != symbol_id.module_path:
                        cross_module_usages += 1
            
            usage_analysis = UsageAnalysis(
                symbol_id=symbol_id,
                total_usages=usage_count,
                usage_locations=[],  # Por implementar detalladamente
                is_exported=is_exported,
                is_public_api=is_exported and symbol_info.visibility == 'public',
                cross_module_usages=cross_module_usages
            )
            
            return usage_analysis
            
        except Exception as e:
            logger.error(f"Error analizando uso del símbolo {symbol_id}: {e}")
            raise
    
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
        """
        try:
            return self.dead_code_detector._calculate_file_statistics(analysis)
            
        except Exception as e:
            logger.error(f"Error calculando estadísticas: {e}")
            raise
    
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
        """
        try:
            return self.dead_code_detector._calculate_project_statistics(
                project_analysis.file_analyses
            )
            
        except Exception as e:
            logger.error(f"Error calculando estadísticas del proyecto: {e}")
            raise
    
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
        return self.dead_code_detector._get_language_config(language)
    
    async def is_analysis_supported(self, language: ProgrammingLanguage) -> bool:
        """
        Verifica si el análisis está soportado para un lenguaje.
        
        Args:
            language: Lenguaje de programación
            
        Returns:
            True si está soportado, False en caso contrario
        """
        supported_languages = {
            ProgrammingLanguage.PYTHON,
            ProgrammingLanguage.JAVASCRIPT,
            ProgrammingLanguage.TYPESCRIPT,
            ProgrammingLanguage.RUST,
        }
        
        return language in supported_languages
    
    async def get_analysis_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento del análisis.
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        metrics = self.dead_code_detector.get_performance_metrics()
        
        return {
            'total_time_ms': metrics.total_time_ms,
            'files_analyzed': metrics.files_analyzed,
            'lines_analyzed': metrics.lines_analyzed,
            'symbols_analyzed': metrics.symbols_analyzed,
            'analysis_rate_lines_per_second': metrics.get_analysis_rate(),
            'cache_hits': len(self._analysis_cache),
            'cfg_cache_hits': len(self._cfg_cache)
        }
    
    # Métodos auxiliares privados
    
    def _create_detector_config(self) -> DeadCodeConfig:
        """Crea la configuración del detector."""
        return DeadCodeConfig(
            analyze_unused_variables=self.config.get('analyze_unused_variables', True),
            analyze_unused_functions=self.config.get('analyze_unused_functions', True),
            analyze_unused_classes=self.config.get('analyze_unused_classes', True),
            analyze_unused_imports=self.config.get('analyze_unused_imports', True),
            analyze_unreachable_code=self.config.get('analyze_unreachable_code', True),
            analyze_dead_branches=self.config.get('analyze_dead_branches', True),
            cross_module_analysis=self.config.get('cross_module_analysis', True),
            confidence_threshold=self.config.get('confidence_threshold', 0.7),
            aggressive_mode=self.config.get('aggressive_mode', False)
        )
    
    def _get_cache_key(self, parse_result: ParseResult) -> str:
        """Genera una clave de cache para un resultado de parsing."""
        return f"{parse_result.file_path}:{parse_result.content_hash}"
    
    def _merge_configs(self, base_config: DeadCodeConfig, override_config: Dict[str, Any]) -> DeadCodeConfig:
        """Mezcla configuraciones."""
        # Implementación simplificada - en una implementación real sería más sofisticado
        return base_config
    
    def clear_cache(self) -> None:
        """Limpia todos los caches."""
        self._analysis_cache.clear()
        self._cfg_cache.clear()
        logger.info("Cache de análisis limpiado")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Obtiene estadísticas de cache."""
        return {
            'analysis_cache_size': len(self._analysis_cache),
            'cfg_cache_size': len(self._cfg_cache)
        }
