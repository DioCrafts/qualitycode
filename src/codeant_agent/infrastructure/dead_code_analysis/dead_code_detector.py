"""
Detector principal de código muerto.

Este módulo implementa el detector principal que integra todos los
analizadores de código muerto para proporcionar análisis completo.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
import time

from ...domain.entities.dead_code_analysis import (
    DeadCodeAnalysis, ProjectDeadCodeAnalysis, DeadCodeStatistics,
    UnusedVariable, UnusedFunction, UnusedClass, UnusedImport,
    UnreachableCode, DeadBranch, UnusedParameter, RedundantAssignment,
    EntryPoint, CrossModuleIssue
)
from ...domain.entities.dependency_analysis import (
    ControlFlowGraph, DependencyGraph, GlobalDependencyGraph,
    SymbolId, LivenessInfo
)
from ...domain.entities.parse_result import ParseResult
from ...domain.services.dead_code_service import (
    ConfidenceScoringService, DeadCodeClassificationService
)
from ...domain.value_objects.programming_language import ProgrammingLanguage

from .reachability_analyzer import ReachabilityAnalyzer, ReachabilityResult
from .data_flow_analyzer import DataFlowAnalyzer, DataFlowAnalysisResult
from .import_analyzer import ImportAnalyzer, ImportAnalysisResult
from .cross_module_analyzer import CrossModuleAnalyzer, CrossModuleAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class DeadCodeConfig:
    """Configuración para el detector de código muerto."""
    analyze_unused_variables: bool = True
    analyze_unused_functions: bool = True
    analyze_unused_classes: bool = True
    analyze_unused_imports: bool = True
    analyze_unreachable_code: bool = True
    analyze_dead_branches: bool = True
    analyze_unused_parameters: bool = True
    analyze_redundant_assignments: bool = True
    cross_module_analysis: bool = True
    entry_points: List[str] = None
    keep_patterns: List[str] = None
    aggressive_mode: bool = False
    confidence_threshold: float = 0.7
    language_specific_configs: Dict[ProgrammingLanguage, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.entry_points is None:
            self.entry_points = []
        if self.keep_patterns is None:
            self.keep_patterns = []
        if self.language_specific_configs is None:
            self.language_specific_configs = {}


@dataclass
class AnalysisPerformanceMetrics:
    """Métricas de rendimiento del análisis."""
    total_time_ms: int
    reachability_time_ms: int
    data_flow_time_ms: int
    import_analysis_time_ms: int
    cross_module_time_ms: int
    files_analyzed: int
    lines_analyzed: int
    symbols_analyzed: int
    
    def get_analysis_rate(self) -> float:
        """Obtiene la tasa de análisis en líneas por segundo."""
        if self.total_time_ms == 0:
            return 0.0
        return (self.lines_analyzed * 1000) / self.total_time_ms


class LanguageSpecificDetector:
    """Detector específico por lenguaje."""
    
    # Patrones específicos por lenguaje que se consideran "seguros"
    SAFE_PATTERNS = {
        ProgrammingLanguage.PYTHON: [
            '__init__',
            '__str__',
            '__repr__',
            'test_*',
            'setUp',
            'tearDown',
        ],
        ProgrammingLanguage.JAVASCRIPT: [
            'constructor',
            'render',
            'componentDidMount',
            'componentWillUnmount',
        ],
        ProgrammingLanguage.RUST: [
            'main',
            'new',
            'default',
        ]
    }
    
    def is_safe_to_remove(
        self, 
        symbol_name: str, 
        language: ProgrammingLanguage,
        context: Dict[str, Any]
    ) -> bool:
        """
        Verifica si un símbolo es seguro de eliminar.
        
        Args:
            symbol_name: Nombre del símbolo
            language: Lenguaje de programación
            context: Contexto adicional
            
        Returns:
            True si es seguro eliminar, False en caso contrario
        """
        safe_patterns = self.SAFE_PATTERNS.get(language, [])
        
        # Verificar patrones seguros
        for pattern in safe_patterns:
            if pattern.endswith('*'):
                if symbol_name.startswith(pattern[:-1]):
                    return False
            elif pattern == symbol_name:
                return False
        
        # Verificar si es parte de la API pública
        if context.get('is_public_api', False):
            return False
        
        # Verificar si tiene decoradores/atributos especiales
        if context.get('has_special_decorators', False):
            return False
        
        return True
    
    def adjust_confidence_for_language(
        self, 
        base_confidence: float,
        language: ProgrammingLanguage,
        symbol_type: str,
        context: Dict[str, Any]
    ) -> float:
        """
        Ajusta la confianza basada en características específicas del lenguaje.
        
        Args:
            base_confidence: Confianza base
            language: Lenguaje de programación
            symbol_type: Tipo de símbolo
            context: Contexto adicional
            
        Returns:
            Confianza ajustada
        """
        adjusted_confidence = base_confidence
        
        # Ajustes específicos por lenguaje
        if language == ProgrammingLanguage.PYTHON:
            # Python permite mucha reflexión y uso dinámico
            if symbol_type == 'function' and context.get('is_method', False):
                adjusted_confidence -= 0.1
            
            # Funciones que empiezan con _ son "privadas"
            if context.get('name', '').startswith('_'):
                adjusted_confidence += 0.1
        
        elif language == ProgrammingLanguage.JAVASCRIPT:
            # JavaScript tiene mucho uso dinámico
            adjusted_confidence -= 0.2
            
            # Funciones exportadas tienen menor confianza
            if context.get('is_exported', False):
                adjusted_confidence -= 0.3
        
        elif language == ProgrammingLanguage.RUST:
            # Rust es más estático, mayor confianza
            adjusted_confidence += 0.1
            
            # Funciones públicas en Rust tienen propósito claro
            if context.get('visibility') == 'pub':
                adjusted_confidence -= 0.2
        
        return max(0.1, min(1.0, adjusted_confidence))


class DeadCodeDetector:
    """Detector principal de código muerto."""
    
    def __init__(self, config: Optional[DeadCodeConfig] = None):
        self.config = config or DeadCodeConfig()
        self.reachability_analyzer = ReachabilityAnalyzer()
        self.data_flow_analyzer = DataFlowAnalyzer()
        self.import_analyzer = ImportAnalyzer()
        self.cross_module_analyzer = CrossModuleAnalyzer()
        self.confidence_service = ConfidenceScoringService()
        self.classification_service = DeadCodeClassificationService()
        self.language_detector = LanguageSpecificDetector()
    
    async def detect_dead_code(self, parse_result: ParseResult) -> DeadCodeAnalysis:
        """
        Detecta código muerto en un archivo individual.
        
        Args:
            parse_result: Resultado del parsing del archivo
            
        Returns:
            DeadCodeAnalysis con los resultados
        """
        start_time = time.time()
        
        try:
            analysis = DeadCodeAnalysis(
                file_path=parse_result.file_path,
                language=parse_result.language
            )
            
            # Construir grafo de control de flujo
            cfg = self.reachability_analyzer.cfg_builder.build_cfg(parse_result)
            
            # Análisis de alcanzabilidad
            if self.config.analyze_unreachable_code or self.config.analyze_dead_branches:
                reachability_result = await self.reachability_analyzer.analyze_reachability(
                    parse_result, self._get_language_config(parse_result.language)
                )
                
                if self.config.analyze_unreachable_code:
                    analysis.unreachable_code = reachability_result.unreachable_code_blocks
                
                if self.config.analyze_dead_branches:
                    analysis.dead_branches = reachability_result.dead_branches
            
            # Análisis de flujo de datos
            if (self.config.analyze_unused_variables or 
                self.config.analyze_redundant_assignments or
                self.config.analyze_unused_parameters):
                
                data_flow_result = await self.data_flow_analyzer.analyze_data_flow(
                    parse_result, cfg, self._get_language_config(parse_result.language)
                )
                
                if self.config.analyze_unused_variables:
                    # Filtrar y ajustar confianza
                    analysis.unused_variables = self._filter_and_adjust_unused_variables(
                        data_flow_result.unused_variables, parse_result
                    )
                
                if self.config.analyze_redundant_assignments:
                    analysis.redundant_assignments = data_flow_result.redundant_assignments
                
                if self.config.analyze_unused_parameters:
                    analysis.unused_parameters = data_flow_result.unused_parameters
            
            # Análisis de imports
            if self.config.analyze_unused_imports:
                import_result = await self.import_analyzer.analyze_imports(
                    parse_result, self._get_language_config(parse_result.language)
                )
                analysis.unused_imports = import_result.unused_imports
            
            # Detectar funciones y clases no utilizadas (análisis básico)
            if self.config.analyze_unused_functions:
                analysis.unused_functions = await self._detect_unused_functions_basic(
                    parse_result, cfg
                )
            
            if self.config.analyze_unused_classes:
                analysis.unused_classes = await self._detect_unused_classes_basic(
                    parse_result, cfg
                )
            
            # Calcular estadísticas
            analysis.statistics = self._calculate_file_statistics(analysis)
            
            analysis.execution_time_ms = int((time.time() - start_time) * 1000)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error detectando código muerto en {parse_result.file_path}: {e}")
            raise
    
    async def detect_dead_code_project(
        self, 
        parse_results: List[ParseResult]
    ) -> ProjectDeadCodeAnalysis:
        """
        Detecta código muerto en todo un proyecto.
        
        Args:
            parse_results: Lista de resultados de parsing de todos los archivos
            
        Returns:
            ProjectDeadCodeAnalysis con los resultados del proyecto completo
        """
        start_time = time.time()
        
        try:
            # Análizar archivos individuales
            file_analyses = []
            individual_analysis_tasks = []
            
            # Crear tareas para análisis paralelo
            for parse_result in parse_results:
                task = self.detect_dead_code(parse_result)
                individual_analysis_tasks.append(task)
            
            # Ejecutar análisis de archivos en paralelo
            file_analyses = await asyncio.gather(*individual_analysis_tasks)
            
            # Análisis cross-module si está habilitado
            cross_module_result = None
            if self.config.cross_module_analysis:
                cross_module_result = await self.cross_module_analyzer.analyze_cross_module_dependencies(
                    parse_results, self._get_cross_module_config()
                )
                
                # Refinar análisis de archivos individuales con información cross-module
                file_analyses = self._refine_with_cross_module_analysis(
                    file_analyses, cross_module_result
                )
            
            # Crear análisis del proyecto
            project_analysis = ProjectDeadCodeAnalysis(
                project_path=parse_results[0].file_path.parent if parse_results else None,
                file_analyses=file_analyses
            )
            
            # Agregar información cross-module si está disponible
            if cross_module_result:
                project_analysis.cross_module_issues = cross_module_result.cross_module_issues
                project_analysis.dependency_cycles = [
                    [str(symbol) for symbol in cycle]
                    for cycle in cross_module_result.circular_dependencies
                ]
            
            # Calcular estadísticas globales
            project_analysis.global_statistics = self._calculate_project_statistics(
                file_analyses
            )
            
            project_analysis.execution_time_ms = int((time.time() - start_time) * 1000)
            
            return project_analysis
            
        except Exception as e:
            logger.error(f"Error detectando código muerto en el proyecto: {e}")
            raise
    
    def _filter_and_adjust_unused_variables(
        self, 
        unused_variables: List[UnusedVariable],
        parse_result: ParseResult
    ) -> List[UnusedVariable]:
        """Filtra y ajusta la confianza de variables no utilizadas."""
        filtered_variables = []
        
        for variable in unused_variables:
            # Verificar patrones de exclusión
            if self._should_keep_variable(variable, parse_result):
                continue
            
            # Ajustar confianza basada en el lenguaje
            context = {
                'name': variable.name,
                'scope': variable.scope.scope_type.value if variable.scope else 'unknown',
                'language': parse_result.language.value
            }
            
            adjusted_confidence = self.language_detector.adjust_confidence_for_language(
                variable.confidence,
                parse_result.language,
                'variable',
                context
            )
            
            variable.confidence = adjusted_confidence
            
            # Solo incluir si supera el umbral de confianza
            if variable.confidence >= self.config.confidence_threshold:
                filtered_variables.append(variable)
        
        return filtered_variables
    
    def _should_keep_variable(self, variable: UnusedVariable, parse_result: ParseResult) -> bool:
        """Verifica si una variable debe mantenerse."""
        # Verificar patrones de exclusión configurados
        for pattern in self.config.keep_patterns:
            if pattern in variable.name:
                return True
        
        # Variables especiales por lenguaje
        if parse_result.language == ProgrammingLanguage.PYTHON:
            if variable.name in ['__all__', '__version__', '__author__']:
                return True
        
        return False
    
    async def _detect_unused_functions_basic(
        self, 
        parse_result: ParseResult,
        cfg: ControlFlowGraph
    ) -> List[UnusedFunction]:
        """Detecta funciones no utilizadas (análisis básico para archivo individual)."""
        # Implementación básica - en análisis cross-module se refinará
        unused_functions = []
        
        # Extraer funciones del AST
        functions = self._extract_functions_from_ast(parse_result.tree.root_node)
        
        for func in functions:
            # Verificar si es usada localmente (análisis básico)
            if not self._is_function_used_locally(func, parse_result):
                # Crear UnusedFunction
                unused_func = UnusedFunction(
                    name=func['name'],
                    declaration_location=func['location'],
                    function_type=func.get('type', 'function'),
                    confidence=0.6,  # Baja confianza sin análisis cross-module
                    suggestion=f"Verificar si la función '{func['name']}' es utilizada"
                )
                unused_functions.append(unused_func)
        
        return unused_functions
    
    async def _detect_unused_classes_basic(
        self, 
        parse_result: ParseResult,
        cfg: ControlFlowGraph
    ) -> List[UnusedClass]:
        """Detecta clases no utilizadas (análisis básico para archivo individual)."""
        # Similar a funciones pero para clases
        return []
    
    def _refine_with_cross_module_analysis(
        self, 
        file_analyses: List[DeadCodeAnalysis],
        cross_module_result: CrossModuleAnalysisResult
    ) -> List[DeadCodeAnalysis]:
        """Refina análisis de archivos con información cross-module."""
        refined_analyses = []
        
        for analysis in file_analyses:
            refined_analysis = analysis  # Copiar
            
            # Filtrar funciones que son alcanzables desde otros módulos
            refined_analysis.unused_functions = [
                func for func in analysis.unused_functions
                if not self._is_symbol_reachable_cross_module(
                    func.name, str(analysis.file_path), cross_module_result
                )
            ]
            
            # Ajustar confianza de funciones restantes
            for func in refined_analysis.unused_functions:
                func.confidence = min(func.confidence + 0.3, 1.0)  # Mayor confianza con análisis cross-module
            
            refined_analyses.append(refined_analysis)
        
        return refined_analyses
    
    def _is_symbol_reachable_cross_module(
        self, 
        symbol_name: str, 
        module_path: str,
        cross_module_result: CrossModuleAnalysisResult
    ) -> bool:
        """Verifica si un símbolo es alcanzable desde otros módulos."""
        # Buscar el símbolo en los alcanzables
        for reachable_symbol in cross_module_result.reachable_symbols:
            if (reachable_symbol.symbol_name == symbol_name and 
                reachable_symbol.module_path == module_path):
                return True
        return False
    
    def _calculate_file_statistics(self, analysis: DeadCodeAnalysis) -> DeadCodeStatistics:
        """Calcula estadísticas para un archivo."""
        stats = DeadCodeStatistics()
        
        stats.total_unused_variables = len(analysis.unused_variables)
        stats.total_unused_functions = len(analysis.unused_functions)
        stats.total_unused_classes = len(analysis.unused_classes)
        stats.total_unused_imports = len(analysis.unused_imports)
        stats.total_unreachable_code_blocks = len(analysis.unreachable_code)
        stats.total_dead_branches = len(analysis.dead_branches)
        stats.total_unused_parameters = len(analysis.unused_parameters)
        stats.total_redundant_assignments = len(analysis.redundant_assignments)
        
        # Calcular líneas de código muerto estimadas
        stats.lines_of_dead_code = self._estimate_dead_code_lines(analysis)
        
        # Calcular porcentaje (necesitaría líneas totales del archivo)
        # stats.percentage_dead_code = ...
        
        return stats
    
    def _calculate_project_statistics(
        self, 
        file_analyses: List[DeadCodeAnalysis]
    ) -> DeadCodeStatistics:
        """Calcula estadísticas agregadas del proyecto."""
        total_stats = DeadCodeStatistics()
        
        for analysis in file_analyses:
            stats = analysis.statistics
            total_stats.total_unused_variables += stats.total_unused_variables
            total_stats.total_unused_functions += stats.total_unused_functions
            total_stats.total_unused_classes += stats.total_unused_classes
            total_stats.total_unused_imports += stats.total_unused_imports
            total_stats.total_unreachable_code_blocks += stats.total_unreachable_code_blocks
            total_stats.total_dead_branches += stats.total_dead_branches
            total_stats.total_unused_parameters += stats.total_unused_parameters
            total_stats.total_redundant_assignments += stats.total_redundant_assignments
            total_stats.lines_of_dead_code += stats.lines_of_dead_code
        
        return total_stats
    
    def _estimate_dead_code_lines(self, analysis: DeadCodeAnalysis) -> int:
        """Estima el número de líneas de código muerto."""
        total_lines = 0
        
        # Sumar líneas de cada tipo de issue
        for item in analysis.unreachable_code:
            total_lines += item.location.end.line - item.location.start.line + 1
        
        for item in analysis.unused_functions:
            # Estimar líneas promedio de una función
            total_lines += 10  # Estimación conservadora
        
        # Agregar otros tipos...
        
        return total_lines
    
    def _get_language_config(self, language: ProgrammingLanguage) -> Dict[str, Any]:
        """Obtiene configuración específica del lenguaje."""
        return self.config.language_specific_configs.get(language, {})
    
    def _get_cross_module_config(self) -> Dict[str, Any]:
        """Obtiene configuración para análisis cross-module."""
        return {
            'entry_points': self.config.entry_points,
            'aggressive_mode': self.config.aggressive_mode
        }
    
    # Métodos auxiliares (implementaciones simplificadas)
    
    def _extract_functions_from_ast(self, ast_node: Any) -> List[Dict[str, Any]]:
        """Extrae funciones del AST."""
        functions = []
        # Implementación simplificada
        return functions
    
    def _is_function_used_locally(self, func: Dict[str, Any], parse_result: ParseResult) -> bool:
        """Verifica si una función es usada localmente."""
        # Implementación simplificada
        return False
    
    def get_performance_metrics(self) -> AnalysisPerformanceMetrics:
        """Obtiene métricas de rendimiento del último análisis."""
        # Implementación de seguimiento de métricas
        return AnalysisPerformanceMetrics(
            total_time_ms=0,
            reachability_time_ms=0,
            data_flow_time_ms=0,
            import_analysis_time_ms=0,
            cross_module_time_ms=0,
            files_analyzed=0,
            lines_analyzed=0,
            symbols_analyzed=0
        )
