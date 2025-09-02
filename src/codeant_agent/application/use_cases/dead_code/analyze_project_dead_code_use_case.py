"""
Caso de uso para analizar código muerto en un proyecto completo.
"""

import logging
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

from ....domain.entities.dead_code_analysis import (
    ProjectDeadCodeAnalysis, DeadCodeAnalysis
)
from ....domain.entities.parse_result import ParseResult
from ....domain.entities.project import Project
from ....domain.repositories.dead_code_repository import DeadCodeRepository
from ....domain.repositories.parser_repository import ParserRepository
from ....domain.repositories.project_repository import ProjectRepository
from ....domain.services.dead_code_service import DeadCodeClassificationService
from ....domain.value_objects.programming_language import ProgrammingLanguage
from ....utils.result import Result, BaseError

logger = logging.getLogger(__name__)


class AnalyzeProjectDeadCodeError(BaseError):
    """Error al analizar código muerto en proyecto."""
    pass


class ProjectNotFoundError(AnalyzeProjectDeadCodeError):
    """Error cuando el proyecto no existe."""
    pass


class NoFilesToAnalyzeError(AnalyzeProjectDeadCodeError):
    """Error cuando no hay archivos para analizar."""
    pass


@dataclass
class AnalyzeProjectDeadCodeRequest:
    """Request para analizar código muerto en un proyecto."""
    project_id: Optional[str] = None
    project_path: Optional[Path] = None
    file_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    languages: Optional[List[ProgrammingLanguage]] = None
    config: Optional[Dict[str, Any]] = None
    include_cross_module_analysis: bool = True
    include_suggestions: bool = True
    include_classification: bool = True
    confidence_threshold: float = 0.5
    max_files: Optional[int] = None
    parallel_analysis: bool = True


@dataclass
class AnalysisProgress:
    """Progreso del análisis."""
    total_files: int
    analyzed_files: int
    current_file: Optional[Path] = None
    elapsed_time_ms: int = 0
    estimated_remaining_ms: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def percentage(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.analyzed_files / self.total_files) * 100


@dataclass
class AnalyzeProjectDeadCodeResponse:
    """Response del análisis de código muerto de un proyecto."""
    project_analysis: ProjectDeadCodeAnalysis
    classified_issues: Optional[Dict[str, List[Any]]] = None
    suggestions: Optional[List[Dict[str, Any]]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    analysis_summary: Optional[Dict[str, Any]] = None


class AnalyzeProjectDeadCodeUseCase:
    """
    Caso de uso para analizar código muerto en un proyecto completo.
    
    Este caso de uso se encarga de:
    1. Validar y preparar el análisis
    2. Descubrir archivos a analizar
    3. Parsear todos los archivos
    4. Realizar análisis de código muerto (con cross-module)
    5. Clasificar y generar sugerencias
    6. Generar reporte completo
    """
    
    def __init__(
        self,
        dead_code_repository: DeadCodeRepository,
        parser_repository: ParserRepository,
        project_repository: Optional[ProjectRepository] = None,
        classification_service: Optional[DeadCodeClassificationService] = None
    ):
        """
        Inicializar el caso de uso.
        
        Args:
            dead_code_repository: Repositorio para análisis de código muerto
            parser_repository: Repositorio para parsing de archivos
            project_repository: Repositorio de proyectos (opcional)
            classification_service: Servicio para clasificar issues
        """
        self.dead_code_repository = dead_code_repository
        self.parser_repository = parser_repository
        self.project_repository = project_repository
        self.classification_service = classification_service or DeadCodeClassificationService()
    
    async def execute(
        self, 
        request: AnalyzeProjectDeadCodeRequest,
        progress_callback: Optional[callable] = None
    ) -> Result[AnalyzeProjectDeadCodeResponse, Exception]:
        """
        Ejecutar el análisis de código muerto en un proyecto.
        
        Args:
            request: Datos para el análisis
            progress_callback: Callback opcional para reportar progreso
            
        Returns:
            Result con el análisis o error
        """
        try:
            logger.info("Iniciando análisis de código muerto para proyecto")
            
            # 1. Validar y preparar
            preparation_result = await self._prepare_analysis(request)
            if not preparation_result.success:
                return preparation_result
            
            project_path = preparation_result.data
            
            # 2. Descubrir archivos
            files_discovery_result = await self._discover_files(project_path, request)
            if not files_discovery_result.success:
                return files_discovery_result
            
            files_to_analyze = files_discovery_result.data
            logger.info(f"Encontrados {len(files_to_analyze)} archivos para analizar")
            
            # 3. Parsear archivos
            parsing_result = await self._parse_files(
                files_to_analyze, request, progress_callback
            )
            if not parsing_result.success:
                return parsing_result
            
            parse_results = parsing_result.data
            logger.info(f"Parseados exitosamente {len(parse_results)} archivos")
            
            # 4. Realizar análisis de código muerto
            analysis_result = await self._analyze_project_dead_code(
                parse_results, request, progress_callback
            )
            if not analysis_result.success:
                return analysis_result
            
            project_analysis = analysis_result.data
            
            # 5. Crear respuesta base
            response = AnalyzeProjectDeadCodeResponse(
                project_analysis=project_analysis
            )
            
            # 6. Clasificar issues si se solicita
            if request.include_classification:
                response.classified_issues = self._classify_project_issues(project_analysis)
                
                # 7. Generar sugerencias si se solicita
                if request.include_suggestions:
                    response.suggestions = self._generate_project_suggestions(
                        project_analysis, response.classified_issues
                    )
            
            # 8. Agregar métricas y resumen
            response.performance_metrics = await self.dead_code_repository.get_analysis_metrics()
            response.analysis_summary = self._generate_analysis_summary(project_analysis)
            
            logger.info(
                f"Análisis de proyecto completado: "
                f"{project_analysis.global_statistics.get_total_issues()} issues totales"
            )
            
            return Result.success(response)
            
        except Exception as e:
            logger.error(f"Error analizando código muerto en proyecto: {e}")
            return Result.failure(
                AnalyzeProjectDeadCodeError(f"Error inesperado: {str(e)}")
            )
    
    async def _prepare_analysis(
        self, 
        request: AnalyzeProjectDeadCodeRequest
    ) -> Result[Path, Exception]:
        """Preparar el análisis validando y obteniendo la ruta del proyecto."""
        try:
            # Validar que se proporcione project_id o project_path
            if not request.project_id and not request.project_path:
                return Result.failure(
                    AnalyzeProjectDeadCodeError(
                        "Se debe proporcionar project_id o project_path"
                    )
                )
            
            # Si se proporciona project_id, obtener el proyecto
            if request.project_id:
                if not self.project_repository:
                    return Result.failure(
                        AnalyzeProjectDeadCodeError(
                            "project_repository es requerido cuando se usa project_id"
                        )
                    )
                
                project_result = await self.project_repository.find_by_id(request.project_id)
                if not project_result.success or not project_result.data:
                    return Result.failure(
                        ProjectNotFoundError(f"Proyecto no encontrado: {request.project_id}")
                    )
                
                project = project_result.data
                project_path = Path(project.repository_path) if hasattr(project, 'repository_path') else None
                
                if not project_path or not project_path.exists():
                    return Result.failure(
                        AnalyzeProjectDeadCodeError(
                            f"Ruta del proyecto no existe: {project_path}"
                        )
                    )
            else:
                # Usar project_path directamente
                project_path = request.project_path
                
                if not project_path.exists():
                    return Result.failure(
                        AnalyzeProjectDeadCodeError(
                            f"Ruta del proyecto no existe: {project_path}"
                        )
                    )
            
            return Result.success(project_path)
            
        except Exception as e:
            return Result.failure(
                AnalyzeProjectDeadCodeError(f"Error preparando análisis: {e}")
            )
    
    async def _discover_files(
        self, 
        project_path: Path,
        request: AnalyzeProjectDeadCodeRequest
    ) -> Result[List[Path], Exception]:
        """Descubrir archivos para analizar."""
        try:
            # Patrones por defecto basados en lenguajes soportados
            default_patterns = [
                '**/*.py',    # Python
                '**/*.js',    # JavaScript
                '**/*.ts',    # TypeScript
                '**/*.tsx',   # TypeScript React
                '**/*.rs',    # Rust
            ]
            
            patterns = request.file_patterns or default_patterns
            exclude_patterns = request.exclude_patterns or [
                '**/node_modules/**',
                '**/venv/**',
                '**/env/**',
                '**/__pycache__/**',
                '**/target/**',
                '**/build/**',
                '**/dist/**',
                '**/.git/**',
            ]
            
            # Descubrir archivos
            files = []
            for pattern in patterns:
                found_files = list(project_path.glob(pattern))
                files.extend(found_files)
            
            # Filtrar archivos excluidos
            filtered_files = []
            for file_path in files:
                exclude = False
                for exclude_pattern in exclude_patterns:
                    if file_path.match(exclude_pattern):
                        exclude = True
                        break
                
                if not exclude and file_path.is_file():
                    filtered_files.append(file_path)
            
            # Limitar número de archivos si se especifica
            if request.max_files and len(filtered_files) > request.max_files:
                filtered_files = filtered_files[:request.max_files]
                logger.info(f"Limitando análisis a {request.max_files} archivos")
            
            if not filtered_files:
                return Result.failure(
                    NoFilesToAnalyzeError("No se encontraron archivos para analizar")
                )
            
            return Result.success(filtered_files)
            
        except Exception as e:
            return Result.failure(
                AnalyzeProjectDeadCodeError(f"Error descubriendo archivos: {e}")
            )
    
    async def _parse_files(
        self, 
        files: List[Path],
        request: AnalyzeProjectDeadCodeRequest,
        progress_callback: Optional[callable] = None
    ) -> Result[List[ParseResult], Exception]:
        """Parsear todos los archivos."""
        try:
            parse_results = []
            errors = []
            
            total_files = len(files)
            
            if request.parallel_analysis:
                # Parsing en paralelo
                semaphore = asyncio.Semaphore(10)  # Limitar concurrencia
                
                async def parse_file_with_semaphore(file_path: Path) -> Optional[ParseResult]:
                    async with semaphore:
                        try:
                            return await self.parser_repository.parse_file(file_path)
                        except Exception as e:
                            errors.append(f"Error parsing {file_path}: {e}")
                            return None
                
                tasks = [parse_file_with_semaphore(file_path) for file_path in files]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, ParseResult):
                        # Filtrar por lenguajes si se especifica
                        if not request.languages or result.language in request.languages:
                            parse_results.append(result)
                    elif isinstance(result, Exception):
                        errors.append(str(result))
            else:
                # Parsing secuencial
                for i, file_path in enumerate(files):
                    try:
                        if progress_callback:
                            progress = AnalysisProgress(
                                total_files=total_files,
                                analyzed_files=i,
                                current_file=file_path
                            )
                            progress_callback(progress)
                        
                        parse_result = await self.parser_repository.parse_file(file_path)
                        
                        # Filtrar por lenguajes si se especifica
                        if not request.languages or parse_result.language in request.languages:
                            parse_results.append(parse_result)
                        
                    except Exception as e:
                        error_msg = f"Error parsing {file_path}: {e}"
                        errors.append(error_msg)
                        logger.warning(error_msg)
                        continue
            
            if errors:
                logger.warning(f"Se produjeron {len(errors)} errores durante el parsing")
            
            if not parse_results:
                return Result.failure(
                    AnalyzeProjectDeadCodeError("No se pudo parsear ningún archivo")
                )
            
            return Result.success(parse_results)
            
        except Exception as e:
            return Result.failure(
                AnalyzeProjectDeadCodeError(f"Error parseando archivos: {e}")
            )
    
    async def _analyze_project_dead_code(
        self, 
        parse_results: List[ParseResult],
        request: AnalyzeProjectDeadCodeRequest,
        progress_callback: Optional[callable] = None
    ) -> Result[ProjectDeadCodeAnalysis, Exception]:
        """Realizar análisis de código muerto del proyecto."""
        try:
            if progress_callback:
                progress = AnalysisProgress(
                    total_files=len(parse_results),
                    analyzed_files=0,
                    current_file=None
                )
                progress_callback(progress)
            
            # Configurar análisis cross-module
            analysis_config = request.config or {}
            analysis_config['cross_module_analysis'] = request.include_cross_module_analysis
            analysis_config['confidence_threshold'] = request.confidence_threshold
            
            # Realizar análisis
            project_analysis = await self.dead_code_repository.analyze_project_dead_code(
                parse_results, analysis_config
            )
            
            return Result.success(project_analysis)
            
        except Exception as e:
            return Result.failure(
                AnalyzeProjectDeadCodeError(f"Error en análisis de código muerto: {e}")
            )
    
    def _classify_project_issues(
        self, 
        project_analysis: ProjectDeadCodeAnalysis
    ) -> Dict[str, List[Any]]:
        """Clasificar issues a nivel de proyecto."""
        all_classifications = {}
        
        # Clasificar issues de cada archivo
        for file_analysis in project_analysis.file_analyses:
            file_classifications = self.classification_service.classify_dead_code_analysis(
                file_analysis
            )
            
            # Agregar al resultado global
            for category, issues in file_classifications.items():
                if category not in all_classifications:
                    all_classifications[category] = []
                all_classifications[category].extend(issues)
        
        return all_classifications
    
    def _generate_project_suggestions(
        self, 
        project_analysis: ProjectDeadCodeAnalysis,
        classified_issues: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generar sugerencias a nivel de proyecto."""
        suggestions = []
        
        # Generar sugerencias basadas en la clasificación
        base_suggestions = self.classification_service.generate_removal_suggestions(
            classified_issues
        )
        suggestions.extend(base_suggestions)
        
        # Agregar sugerencias específicas del proyecto
        if project_analysis.cross_module_issues:
            suggestions.append({
                'action': 'review_cross_module',
                'items': project_analysis.cross_module_issues,
                'description': 'Revisar issues que afectan múltiples módulos',
                'risk_level': 'high',
                'priority': 'low'
            })
        
        if project_analysis.dependency_cycles:
            suggestions.append({
                'action': 'resolve_cycles',
                'items': project_analysis.dependency_cycles,
                'description': 'Resolver dependencias circulares',
                'risk_level': 'medium',
                'priority': 'medium'
            })
        
        return suggestions
    
    def _generate_analysis_summary(
        self, 
        project_analysis: ProjectDeadCodeAnalysis
    ) -> Dict[str, Any]:
        """Generar resumen del análisis."""
        stats = project_analysis.global_statistics
        
        # Archivos con más issues
        worst_files = sorted(
            project_analysis.file_analyses,
            key=lambda x: x.statistics.get_total_issues(),
            reverse=True
        )[:10]
        
        return {
            'total_files_analyzed': len(project_analysis.file_analyses),
            'total_issues_found': stats.get_total_issues(),
            'issues_by_type': {
                'unused_variables': stats.total_unused_variables,
                'unused_functions': stats.total_unused_functions,
                'unused_classes': stats.total_unused_classes,
                'unused_imports': stats.total_unused_imports,
                'unreachable_code': stats.total_unreachable_code_blocks,
                'dead_branches': stats.total_dead_branches,
                'unused_parameters': stats.total_unused_parameters,
                'redundant_assignments': stats.total_redundant_assignments,
            },
            'worst_files': [
                {
                    'file_path': str(file_analysis.file_path),
                    'issues_count': file_analysis.statistics.get_total_issues(),
                    'language': file_analysis.language.get_name()
                }
                for file_analysis in worst_files
            ],
            'cross_module_issues_count': len(project_analysis.cross_module_issues),
            'dependency_cycles_count': len(project_analysis.dependency_cycles),
            'execution_time_ms': project_analysis.execution_time_ms,
        }
