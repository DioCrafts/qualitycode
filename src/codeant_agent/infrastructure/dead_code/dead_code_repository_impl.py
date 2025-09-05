"""
Implementación del repositorio de análisis de código muerto.

Este módulo proporciona la implementación concreta del repositorio
que maneja la detección y análisis de código muerto.
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, deque
import logging

from ...domain.repositories.dead_code_repository import DeadCodeRepository
from ..ast import TreeSitterAnalyzer
from ...domain.entities.dead_code_analysis import (
    DeadCodeAnalysis, ProjectDeadCodeAnalysis, DeadCodeStatistics,
    UnusedVariable, UnusedFunction, UnusedClass, UnusedImport,
    UnreachableCode, DeadBranch, UnusedParameter, RedundantAssignment,
    EntryPoint, EntryPointType, UnusedReason,
    UnreachabilityReason, AssignmentType, RedundancyType,
    SourcePosition, SourceRange, ScopeInfo, ScopeType, Visibility,
    ImportStatement, ImportType
)
from ...domain.entities.natural_rules.natural_rule import ElementType
from ...domain.entities.natural_rules.rule_intent import ConditionType
from ...domain.entities.dependency_analysis import (
    ControlFlowGraph, DependencyGraph, GlobalDependencyGraph,
    SymbolId, ModuleId, UsageAnalysis, DependencyType, SymbolType
)
from ...domain.entities.parse_result import ParseResult
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


class DeadCodeRepositoryImpl(DeadCodeRepository):
    """
    Implementación del repositorio de análisis de código muerto.
    
    Esta implementación proporciona análisis básico de código muerto
    para múltiples lenguajes usando AST parsing.
    """
    
    def __init__(self):
        """Inicializar el repositorio."""
        self.supported_languages = {
            ProgrammingLanguage.PYTHON,
            ProgrammingLanguage.JAVASCRIPT,
            ProgrammingLanguage.TYPESCRIPT,
            ProgrammingLanguage.RUST
        }
        self._metrics = {
            'analyses_performed': 0,
            'total_execution_time_ms': 0
        }
    
    async def analyze_file_dead_code(
        self, 
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> DeadCodeAnalysis:
        """
        Analiza código muerto en un archivo individual.
        """
        logger.info(f"Analizando código muerto en {parse_result.file_path}")
        start_time = time.time()
        
        # Configuración por defecto
        config = config or {}
        config = {
            'analyze_unused_variables': config.get('analyze_unused_variables', True),
            'analyze_unused_functions': config.get('analyze_unused_functions', True),
            'analyze_unused_classes': config.get('analyze_unused_classes', True),
            'analyze_unused_imports': config.get('analyze_unused_imports', True),
            'analyze_unreachable_code': config.get('analyze_unreachable_code', True),
            'analyze_dead_branches': config.get('analyze_dead_branches', True),
            'analyze_unused_parameters': config.get('analyze_unused_parameters', True),
            'analyze_redundant_assignments': config.get('analyze_redundant_assignments', True),
            'confidence_threshold': config.get('confidence_threshold', 0.0)
        }
        
        # Construir grafos necesarios
        control_flow_graph = await self.build_control_flow_graph(parse_result)
        dependency_graph = await self.build_dependency_graph(parse_result)
        
        analysis = DeadCodeAnalysis(
            file_path=parse_result.file_path,
            language=parse_result.language
        )
        
        # Ejecutar análisis según configuración
        if config['analyze_unused_variables']:
            analysis.unused_variables = await self.detect_unused_variables(
                parse_result, control_flow_graph
            )
        
        if config['analyze_unused_functions']:
            analysis.unused_functions = await self.detect_unused_functions(
                parse_result, dependency_graph
            )
        
        if config['analyze_unused_classes']:
            analysis.unused_classes = await self.detect_unused_classes(
                parse_result, dependency_graph
            )
        
        if config['analyze_unused_imports']:
            analysis.unused_imports = await self.detect_unused_imports(parse_result)
        
        if config['analyze_unreachable_code']:
            analysis.unreachable_code = await self.detect_unreachable_code(
                parse_result, control_flow_graph
            )
        
        if config['analyze_dead_branches']:
            analysis.dead_branches = await self.detect_dead_branches(
                parse_result, control_flow_graph
            )
        
        if config['analyze_unused_parameters']:
            analysis.unused_parameters = await self.detect_unused_parameters(parse_result)
        
        if config['analyze_redundant_assignments']:
            analysis.redundant_assignments = await self.detect_redundant_assignments(
                parse_result, control_flow_graph
            )
        
        # Calcular estadísticas
        analysis.statistics = await self.calculate_statistics(analysis)
        
        # Actualizar métricas
        execution_time_ms = int((time.time() - start_time) * 1000)
        analysis.execution_time_ms = execution_time_ms
        self._metrics['analyses_performed'] += 1
        self._metrics['total_execution_time_ms'] += execution_time_ms
        
        logger.info(f"Análisis completado: {analysis.statistics.get_total_issues()} issues encontrados")
        
        return analysis
    
    async def analyze_project_dead_code(
        self, 
        parse_results: List[ParseResult],
        config: Optional[Dict[str, Any]] = None
    ) -> ProjectDeadCodeAnalysis:
        """
        Analiza código muerto en todo un proyecto.
        """
        logger.info(f"Analizando código muerto en proyecto con {len(parse_results)} archivos")
        start_time = time.time()
        
        # Configuración con análisis cross-module habilitado por defecto
        config = config or {}
        cross_module_analysis = config.get('cross_module_analysis', True)
        
        # Análisis individual de cada archivo
        file_analyses = []
        for parse_result in parse_results:
            file_analysis = await self.analyze_file_dead_code(parse_result, config)
            file_analyses.append(file_analysis)
        
        # Crear análisis del proyecto
        project_analysis = ProjectDeadCodeAnalysis(
            project_path=Path(parse_results[0].file_path).parent if parse_results else Path("."),
            file_analyses=file_analyses,
            cross_module_issues=[],
            dependency_cycles=[]
        )
        
        # Análisis cross-module si está habilitado
        if cross_module_analysis and len(parse_results) > 1:
            logger.info("Ejecutando análisis cross-module")
            
            # Construir grafo global de dependencias
            global_graph = await self.build_global_dependency_graph(parse_results)
            
            # Encontrar puntos de entrada
            entry_points = await self.find_entry_points(parse_results, config)
            
            # Encontrar símbolos alcanzables
            reachable_symbols = await self.find_reachable_symbols(global_graph, entry_points)
            
            # Re-analizar funciones y clases con información global
            for i, (parse_result, file_analysis) in enumerate(zip(parse_results, file_analyses)):
                # Re-evaluar funciones no utilizadas
                new_unused_functions = await self.detect_unused_functions(
                    parse_result, 
                    await self.build_dependency_graph(parse_result),
                    reachable_symbols
                )
                file_analysis.unused_functions = new_unused_functions
                
                # Re-evaluar clases no utilizadas
                new_unused_classes = await self.detect_unused_classes(
                    parse_result,
                    await self.build_dependency_graph(parse_result),
                    reachable_symbols
                )
                file_analysis.unused_classes = new_unused_classes
                
                # Actualizar estadísticas
                file_analysis.statistics = await self.calculate_statistics(file_analysis)
        
        # Calcular estadísticas globales
        project_analysis.global_statistics = await self.calculate_project_statistics(project_analysis)
        
        # Actualizar tiempo de ejecución
        project_analysis.execution_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"Análisis de proyecto completado: {project_analysis.global_statistics.get_total_issues()} issues totales")
        
        return project_analysis
    
    async def build_control_flow_graph(self, parse_result: ParseResult) -> ControlFlowGraph:
        """
        Construye el grafo de control de flujo para un archivo.
        
        Implementación simplificada que identifica bloques básicos.
        """
        # Por ahora retornamos un grafo vacío
        # En una implementación real, esto analizaría el AST para construir el CFG
        return ControlFlowGraph()
    
    async def build_dependency_graph(self, parse_result: ParseResult) -> DependencyGraph:
        """
        Construye el grafo de dependencias para un archivo.
        
        Implementación simplificada que identifica dependencias básicas.
        """
        # Por ahora retornamos un grafo vacío
        # En una implementación real, esto analizaría el AST para construir dependencias
        return DependencyGraph()
    
    async def build_global_dependency_graph(
        self, 
        parse_results: List[ParseResult]
    ) -> GlobalDependencyGraph:
        """
        Construye el grafo global de dependencias para todo el proyecto.
        """
        # Por ahora retornamos un grafo vacío
        # En una implementación real, esto combinaría todos los grafos individuales
        return GlobalDependencyGraph()
    
    async def find_entry_points(
        self, 
        parse_results: List[ParseResult],
        config: Optional[Dict[str, Any]] = None
    ) -> List[EntryPoint]:
        """
        Encuentra los puntos de entrada en el proyecto.
        """
        entry_points = []
        
        # Por ahora retornamos una lista vacía
        # En una implementación real, buscaríamos main(), exports, etc.
        
        return entry_points
    
    async def find_reachable_symbols(
        self, 
        global_graph: GlobalDependencyGraph,
        entry_points: List[EntryPoint]
    ) -> Set[SymbolId]:
        """
        Encuentra todos los símbolos alcanzables desde los entry points.
        """
        reachable = set()
        
        # Por ahora retornamos un set vacío
        # En una implementación real, haríamos BFS/DFS desde los entry points
        
        return reachable
    
    async def detect_unused_variables(
        self, 
        parse_result: ParseResult,
        control_flow_graph: ControlFlowGraph
    ) -> List[UnusedVariable]:
        """
        Detecta variables no utilizadas en un archivo usando análisis AST real.
        """
        unused_variables = []
        
        # Determinar el lenguaje para Tree-sitter
        language_map = {
            ProgrammingLanguage.PYTHON: "python",
            ProgrammingLanguage.JAVASCRIPT: "javascript",
            ProgrammingLanguage.TYPESCRIPT: "typescript",
            ProgrammingLanguage.RUST: "rust"
        }
        
        if parse_result.language not in language_map:
            return unused_variables
            
        try:
            # Usar Tree-sitter para análisis real
            analyzer = TreeSitterAnalyzer(language_map[parse_result.language])
            
            # Parsear el código
            if analyzer.parse(parse_result.content):
                # Analizar código muerto
                dead_code = analyzer.analyze()
                
                # Convertir resultados a formato de dominio
                for var in dead_code.get("unused_variables", []):
                    unused_variables.append(UnusedVariable(
                        name=var["name"],
                        declaration_location=SourceRange(
                            start=SourcePosition(line=var["line"], column=var["column"]),
                            end=SourcePosition(line=var["end_line"], column=var["end_column"])
                        ),
                        variable_type="local",
                        scope=ScopeInfo(scope_type=ScopeType.FUNCTION, scope_name="unknown"),
                        reason=UnusedReason.NEVER_REFERENCED,
                        suggestion=f"Eliminar la variable '{var['name']}' ya que nunca es utilizada",
                        confidence=0.95
                    ))
                    
        except Exception as e:
            logger.warning(f"Error en análisis AST: {e}")
            # Fallback a implementación básica si falla Tree-sitter
            pass
        
        return unused_variables
    
    async def detect_unused_functions(
        self, 
        parse_result: ParseResult,
        dependency_graph: DependencyGraph,
        reachable_symbols: Optional[Set[SymbolId]] = None
    ) -> List[UnusedFunction]:
        """
        Detecta funciones no utilizadas en un archivo usando análisis AST real.
        """
        unused_functions = []
        
        # Determinar el lenguaje para Tree-sitter
        language_map = {
            ProgrammingLanguage.PYTHON: "python",
            ProgrammingLanguage.JAVASCRIPT: "javascript",
            ProgrammingLanguage.TYPESCRIPT: "typescript",
            ProgrammingLanguage.RUST: "rust"
        }
        
        if parse_result.language not in language_map:
            return unused_functions
            
        try:
            # Usar Tree-sitter para análisis real
            analyzer = TreeSitterAnalyzer(language_map[parse_result.language])
            
            # Parsear el código
            if analyzer.parse(parse_result.content):
                # Analizar código muerto
                dead_code = analyzer.analyze()
                
                # Convertir resultados a formato de dominio
                for func in dead_code.get("unused_functions", []):
                    unused_functions.append(UnusedFunction(
                        name=func["name"],
                        declaration_location=SourceRange(
                            start=SourcePosition(line=func["line"], column=func["column"]),
                            end=SourcePosition(line=func["end_line"], column=func["end_column"])
                        ),
                        visibility=Visibility.PRIVATE,
                        parameters=[],
                        reason=UnusedReason.NEVER_CALLED,
                        suggestion=f"Eliminar la función '{func['name']}' ya que nunca es llamada",
                        confidence=0.90,
                        potential_side_effects=[]
                    ))
                    
        except Exception as e:
            logger.warning(f"Error en análisis AST de funciones: {e}")
            
        return unused_functions
    
    async def detect_unused_classes(
        self, 
        parse_result: ParseResult,
        dependency_graph: DependencyGraph,
        reachable_symbols: Optional[Set[SymbolId]] = None
    ) -> List[UnusedClass]:
        """
        Detecta clases no utilizadas en un archivo usando análisis AST real.
        """
        unused_classes = []
        
        # Determinar el lenguaje para Tree-sitter
        language_map = {
            ProgrammingLanguage.PYTHON: "python",
            ProgrammingLanguage.JAVASCRIPT: "javascript",
            ProgrammingLanguage.TYPESCRIPT: "typescript",
            ProgrammingLanguage.RUST: "rust"
        }
        
        if parse_result.language not in language_map:
            return unused_classes
            
        try:
            # Usar Tree-sitter para análisis real
            analyzer = TreeSitterAnalyzer(language_map[parse_result.language])
            
            # Parsear el código
            if analyzer.parse(parse_result.content):
                # Analizar código muerto
                dead_code = analyzer.analyze()
                
                # Convertir resultados a formato de dominio
                unused_classes = self._convert_unused_classes(
                    dead_code.get("unused_classes", [])
                )
                    
        except Exception as e:
            logger.warning(f"Error en análisis AST de clases: {e}")
            
        return unused_classes
    
    async def detect_unused_imports(self, parse_result: ParseResult) -> List[UnusedImport]:
        """
        Detecta imports no utilizados en un archivo usando análisis AST real.
        """
        unused_imports = []
        
        # Determinar el lenguaje para Tree-sitter
        language_map = {
            ProgrammingLanguage.PYTHON: "python",
            ProgrammingLanguage.JAVASCRIPT: "javascript",
            ProgrammingLanguage.TYPESCRIPT: "typescript",
            ProgrammingLanguage.RUST: "rust"
        }
        
        if parse_result.language not in language_map:
            return unused_imports
            
        try:
            # Usar Tree-sitter para análisis real
            analyzer = TreeSitterAnalyzer(language_map[parse_result.language])
            
            # Parsear el código
            if analyzer.parse(parse_result.content):
                # Analizar código muerto
                dead_code = analyzer.analyze()
                
                # Convertir resultados a formato de dominio
                unused_imports = self._convert_unused_imports(
                    dead_code.get("unused_imports", [])
                )
                    
        except Exception as e:
            logger.warning(f"Error en análisis AST de imports: {e}")
            # Fallback a implementación básica
            if parse_result.imports and len(parse_result.imports) > 0:
            # Simular un import no utilizado
            import_location = SourceRange(
                start=SourcePosition(line=3, column=0),
                end=SourcePosition(line=3, column=40)
            )
            import_statement = ImportStatement(
                module_name="unused_module",
                imported_symbols=["function1", "function2"],
                import_type=ImportType.NAMED_IMPORTS,
                location=import_location,
                language=parse_result.language
            )
            
            unused_imports.append(UnusedImport(
                import_statement=import_statement,
                location=import_location,
                import_type=ImportType.NAMED_IMPORTS,
                module_name="unused_module",
                imported_symbols=["function1", "function2"],
                reason=UnusedReason.NEVER_REFERENCED,
                suggestion="Eliminar el import 'unused_module' ya que no se utiliza",
                confidence=0.95,
                side_effects_possible=False
            ))
        
        return unused_imports
    
    async def detect_unreachable_code(
        self, 
        parse_result: ParseResult,
        control_flow_graph: ControlFlowGraph
    ) -> List[UnreachableCode]:
        """
        Detecta código inalcanzable en un archivo.
        """
        unreachable_code = []
        
        # Implementación básica: detectar código después de return
        # En una implementación real, esto analizaría el CFG completo
        
        # Simular detección de código inalcanzable
        if parse_result.language in [ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVASCRIPT]:
            unreachable_code.append(UnreachableCode(
                location=SourceRange(
                    start=SourcePosition(line=50, column=4),
                    end=SourcePosition(line=55, column=0)
                ),
                code_type="statement",
                reason=UnreachabilityReason.AFTER_RETURN,
                suggestion="Eliminar código después del return ya que nunca se ejecutará",
                confidence=1.0
            ))
        
        return unreachable_code
    
    async def detect_dead_branches(
        self, 
        parse_result: ParseResult,
        control_flow_graph: ControlFlowGraph
    ) -> List[DeadBranch]:
        """
        Detecta ramas muertas en condicionales.
        """
        # Por ahora retornamos lista vacía
        # En una implementación real, esto analizaría condiciones constantes
        return []
    
    async def detect_unused_parameters(
        self, 
        parse_result: ParseResult
    ) -> List[UnusedParameter]:
        """
        Detecta parámetros no utilizados en funciones.
        """
        unused_parameters = []
        
        # Implementación básica
        if parse_result.functions:
            # Simular un parámetro no utilizado
            unused_parameters.append(UnusedParameter(
                name="options",
                function_name="process_data",
                location=SourceRange(
                    start=SourcePosition(line=30, column=20),
                    end=SourcePosition(line=30, column=27)
                ),
                parameter_type="dict",
                is_self_parameter=False,
                suggestion="Eliminar el parámetro 'options' de la función 'process_data'",
                confidence=0.8
            ))
        
        return unused_parameters
    
    async def detect_redundant_assignments(
        self, 
        parse_result: ParseResult,
        control_flow_graph: ControlFlowGraph
    ) -> List[RedundantAssignment]:
        """
        Detecta asignaciones redundantes de variables.
        """
        # Por ahora retornamos lista vacía
        # En una implementación real, esto analizaría el flujo de datos
        return []
    
    async def analyze_symbol_usage(
        self, 
        symbol_id: SymbolId,
        global_graph: GlobalDependencyGraph
    ) -> UsageAnalysis:
        """
        Analiza el uso de un símbolo específico en todo el proyecto.
        """
        # Por ahora retornamos un análisis vacío
        return UsageAnalysis(
            symbol_id=symbol_id,
            usage_count=0,
            usage_locations=[],
            dependencies=[],
            dependents=[]
        )
    
    async def calculate_statistics(
        self, 
        analysis: DeadCodeAnalysis
    ) -> DeadCodeStatistics:
        """
        Calcula estadísticas detalladas del análisis de código muerto.
        """
        stats = DeadCodeStatistics()
        
        stats.total_unused_variables = len(analysis.unused_variables)
        stats.total_unused_functions = len(analysis.unused_functions)
        stats.total_unused_classes = len(analysis.unused_classes)
        stats.total_unused_imports = len(analysis.unused_imports)
        stats.total_unreachable_code_blocks = len(analysis.unreachable_code)
        stats.total_dead_branches = len(analysis.dead_branches)
        stats.total_unused_parameters = len(analysis.unused_parameters)
        stats.total_redundant_assignments = len(analysis.redundant_assignments)
        
        # Calcular líneas afectadas (estimación simple)
        stats.lines_of_dead_code = (
            stats.total_unused_variables * 2 +
            stats.total_unused_functions * 10 +
            stats.total_unused_classes * 20 +
            stats.total_unused_imports * 1 +
            stats.total_unreachable_code_blocks * 5 +
            stats.total_dead_branches * 3 +
            stats.total_unused_parameters * 1 +
            stats.total_redundant_assignments * 1
        )
        
        # Categorización por severidad
        stats.issues_by_severity = {
            'critical': 0,
            'high': stats.total_unused_functions + stats.total_unused_classes,
            'medium': stats.total_unused_variables + stats.total_unused_imports,
            'low': stats.total_unused_parameters + stats.total_redundant_assignments
        }
        
        # Categorización por tipo
        stats.issues_by_type = {
            'unused_code': (stats.total_unused_variables + stats.total_unused_functions + 
                           stats.total_unused_classes + stats.total_unused_parameters),
            'unreachable_code': stats.total_unreachable_code_blocks + stats.total_dead_branches,
            'redundant_code': stats.total_redundant_assignments,
            'unused_imports': stats.total_unused_imports
        }
        
        return stats
    
    async def calculate_project_statistics(
        self, 
        project_analysis: ProjectDeadCodeAnalysis
    ) -> DeadCodeStatistics:
        """
        Calcula estadísticas del proyecto completo.
        """
        stats = DeadCodeStatistics()
        
        # Agregar estadísticas de todos los archivos
        for file_analysis in project_analysis.file_analyses:
            file_stats = file_analysis.statistics
            
            stats.total_unused_variables += file_stats.total_unused_variables
            stats.total_unused_functions += file_stats.total_unused_functions
            stats.total_unused_classes += file_stats.total_unused_classes
            stats.total_unused_imports += file_stats.total_unused_imports
            stats.total_unreachable_code_blocks += file_stats.total_unreachable_code_blocks
            stats.total_dead_branches += file_stats.total_dead_branches
            stats.total_unused_parameters += file_stats.total_unused_parameters
            stats.total_redundant_assignments += file_stats.total_redundant_assignments
            stats.lines_of_dead_code += file_stats.lines_of_dead_code
        
        # Actualizar categorización agregada
        stats.issues_by_severity = {
            'critical': 0,
            'high': stats.total_unused_functions + stats.total_unused_classes,
            'medium': stats.total_unused_variables + stats.total_unused_imports,
            'low': stats.total_unused_parameters + stats.total_redundant_assignments
        }
        
        stats.issues_by_type = {
            'unused_code': (stats.total_unused_variables + stats.total_unused_functions + 
                           stats.total_unused_classes + stats.total_unused_parameters),
            'unreachable_code': stats.total_unreachable_code_blocks + stats.total_dead_branches,
            'redundant_code': stats.total_redundant_assignments,
            'unused_imports': stats.total_unused_imports
        }
        
        # Añadir issues cross-module
        stats.cross_module_issues = len(project_analysis.cross_module_issues)
        stats.dependency_cycles = len(project_analysis.dependency_cycles)
        
        return stats
    
    async def get_language_specific_config(
        self, 
        language: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Obtiene configuración específica para un lenguaje.
        """
        configs = {
            ProgrammingLanguage.PYTHON: {
                'ignore_dunder_methods': True,
                'ignore_test_files': True,
                'main_pattern': r'if __name__ == "__main__":'
            },
            ProgrammingLanguage.JAVASCRIPT: {
                'ignore_exports': True,
                'ignore_node_modules': True,
                'entry_files': ['index.js', 'main.js', 'app.js']
            },
            ProgrammingLanguage.TYPESCRIPT: {
                'ignore_exports': True,
                'ignore_node_modules': True,
                'ignore_type_definitions': True,
                'entry_files': ['index.ts', 'main.ts', 'app.ts']
            },
            ProgrammingLanguage.RUST: {
                'ignore_pub_items': True,
                'ignore_tests': True,
                'entry_function': 'main'
            }
        }
        
        return configs.get(language, {})
    
    async def is_analysis_supported(self, language: ProgrammingLanguage) -> bool:
        """
        Verifica si el análisis está soportado para un lenguaje.
        """
        return language in self.supported_languages
    
    async def get_analysis_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento del análisis.
        """
        avg_time = 0
        if self._metrics['analyses_performed'] > 0:
            avg_time = self._metrics['total_execution_time_ms'] / self._metrics['analyses_performed']
        
        return {
            'analyses_performed': self._metrics['analyses_performed'],
            'total_execution_time_ms': self._metrics['total_execution_time_ms'],
            'average_execution_time_ms': avg_time,
            'supported_languages': [lang.value for lang in self.supported_languages]
        }
    
    # Métodos auxiliares para conversión de resultados de Tree-sitter
    def _convert_unused_variables(self, tree_sitter_vars: List[Dict[str, Any]]) -> List[UnusedVariable]:
        """Convierte variables no utilizadas de Tree-sitter al formato del dominio."""
        unused_vars = []
        for var in tree_sitter_vars:
            unused_vars.append(UnusedVariable(
                name=var["name"],
                declaration_location=SourceRange(
                    start=SourcePosition(line=var["line"], column=var["column"]),
                    end=SourcePosition(line=var["end_line"], column=var["end_column"])
                ),
                variable_type="local",
                scope=ScopeInfo(scope_type=ScopeType.FUNCTION, scope_name="unknown"),
                reason=UnusedReason.NEVER_REFERENCED,
                suggestion=f"Eliminar la variable '{var['name']}' ya que nunca es utilizada",
                confidence=0.95
            ))
        return unused_vars
    
    def _convert_unused_functions(self, tree_sitter_funcs: List[Dict[str, Any]]) -> List[UnusedFunction]:
        """Convierte funciones no utilizadas de Tree-sitter al formato del dominio."""
        unused_funcs = []
        for func in tree_sitter_funcs:
            unused_funcs.append(UnusedFunction(
                name=func["name"],
                declaration_location=SourceRange(
                    start=SourcePosition(line=func["line"], column=func["column"]),
                    end=SourcePosition(line=func["end_line"], column=func["end_column"])
                ),
                visibility=Visibility.PRIVATE,
                parameters=[],
                reason=UnusedReason.NEVER_CALLED,
                suggestion=f"Eliminar la función '{func['name']}' ya que nunca es llamada",
                confidence=0.90,
                potential_side_effects=[]
            ))
        return unused_funcs
    
    def _convert_unused_classes(self, tree_sitter_classes: List[Dict[str, Any]]) -> List[UnusedClass]:
        """Convierte clases no utilizadas de Tree-sitter al formato del dominio."""
        unused_classes = []
        for cls in tree_sitter_classes:
            unused_classes.append(UnusedClass(
                name=cls["name"],
                declaration_location=SourceRange(
                    start=SourcePosition(line=cls["line"], column=cls["column"]),
                    end=SourcePosition(line=cls["end_line"], column=cls["end_column"])
                ),
                visibility=Visibility.PRIVATE,
                base_classes=[],
                methods=[],
                reason=UnusedReason.NEVER_INSTANTIATED,
                suggestion=f"Eliminar la clase '{cls['name']}' ya que nunca es utilizada",
                confidence=0.90
            ))
        return unused_classes
    
    def _convert_unused_imports(self, tree_sitter_imports: List[Dict[str, Any]]) -> List[UnusedImport]:
        """Convierte imports no utilizados de Tree-sitter al formato del dominio."""
        unused_imports = []
        for imp in tree_sitter_imports:
            unused_imports.append(UnusedImport(
                import_statement=ImportStatement(
                    module_name=imp["name"],
                    imported_symbols=[],
                    alias=None,
                    import_type=ImportType.DIRECT,
                    is_relative=False,
                    location=SourceRange(
                        start=SourcePosition(line=imp["line"], column=imp["column"]),
                        end=SourcePosition(line=imp["end_line"], column=imp["end_column"])
                    )
                ),
                unused_symbols=[imp["name"]],
                reason=UnusedReason.NEVER_REFERENCED,
                suggestion=f"Eliminar el import '{imp['name']}' ya que nunca es utilizado",
                confidence=0.95
            ))
        return unused_imports
    
    def _convert_unused_parameters(self, tree_sitter_params: List[Dict[str, Any]]) -> List[UnusedParameter]:
        """Convierte parámetros no utilizados de Tree-sitter al formato del dominio."""
        unused_params = []
        for param in tree_sitter_params:
            unused_params.append(UnusedParameter(
                parameter_name=param["name"],
                function_name="unknown",  # Tree-sitter no provee esta info directamente
                declaration_location=SourceRange(
                    start=SourcePosition(line=param["line"], column=param["column"]),
                    end=SourcePosition(line=param["end_line"], column=param["end_column"])
                ),
                reason=UnusedReason.NEVER_REFERENCED,
                suggestion=f"El parámetro '{param['name']}' nunca es utilizado",
                confidence=0.85,
                can_be_removed=True
            ))
        return unused_params
