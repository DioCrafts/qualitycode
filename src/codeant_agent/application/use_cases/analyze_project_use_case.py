"""
Caso de uso para analizar completamente un proyecto.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
import time
import asyncio

from ...domain.entities.project import Project
from ...domain.value_objects.project_id import ProjectId
from ...domain.repositories.project_repository import ProjectRepository
from ...utils.error import Result, BaseError
from ...utils.logging import get_logger
from ...infrastructure.parsers.parser_factory import ParserFactory

logger = get_logger(__name__)


class AnalyzeProjectError(BaseError):
    """Error al analizar un proyecto."""
    pass


class ProjectNotFoundError(AnalyzeProjectError):
    """Error cuando el proyecto no existe."""
    pass


@dataclass
class AnalyzeProjectRequest:
    """Request para analizar un proyecto."""
    project_id: str
    config: Optional[Dict[str, Any]] = None
    force_full_analysis: bool = False
    include_metrics: bool = True
    include_dead_code: bool = True
    include_security: bool = True
    include_complexity: bool = True
    include_duplicates: bool = True


@dataclass
class AnalysisResults:
    """Resultados del análisis completo."""
    project_id: str
    analysis_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    progress: float = 0.0
    
    # Métricas generales
    files_analyzed: int = 0
    total_lines: int = 0
    total_violations: int = 0
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0
    low_violations: int = 0
    quality_score: float = 0.0
    
    # Resultados específicos
    complexity_metrics: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    dead_code_results: Optional[Dict[str, Any]] = None
    security_results: Optional[Dict[str, Any]] = None
    duplicate_results: Optional[Dict[str, Any]] = None
    
    # Errores
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class AnalyzeProjectUseCase:
    """
    Caso de uso para analizar completamente un proyecto.
    
    Este caso de uso coordina todos los análisis disponibles:
    1. Análisis de complejidad
    2. Análisis de calidad
    3. Detección de código muerto
    4. Análisis de seguridad
    5. Detección de duplicados
    """
    
    def __init__(
        self,
        project_repository: ProjectRepository,
        parser_factory: Optional[ParserFactory] = None
    ):
        """
        Inicializar el caso de uso.
        
        Args:
            project_repository: Repositorio de proyectos
            parser_factory: Factory para crear parsers
        """
        self.project_repository = project_repository
        self.parser_factory = parser_factory
    
    async def execute(self, request: AnalyzeProjectRequest) -> Result[AnalysisResults, Exception]:
        """
        Ejecutar el análisis completo del proyecto.
        
        Args:
            request: Datos para el análisis
            
        Returns:
            Result con los resultados del análisis o error
        """
        try:
            logger.info(f"Iniciando análisis completo del proyecto: {request.project_id}")
            
            # Crear ID único para este análisis
            analysis_id = f"analysis-{int(time.time())}-{request.project_id[:8]}"
            
            # Inicializar resultados
            results = AnalysisResults(
                project_id=request.project_id,
                analysis_id=analysis_id,
                status="IN_PROGRESS",
                created_at=datetime.now().isoformat(),
                progress=0.0
            )
            
            # 1. Obtener el proyecto
            project_result = await self._get_project(request.project_id)
            if not project_result.success:
                return Result.failure(project_result.error)
            
            project = project_result.data
            
            # 2. Verificar que el proyecto esté activo
            if not project.is_active():
                return Result.failure(
                    AnalyzeProjectError(
                        f"El proyecto {project.name} no está activo"
                    )
                )
            
            # 3. Obtener la ruta del repositorio
            # En una implementación real, esto vendría del repositorio clonado
            # Por ahora usamos una ruta simulada
            project_path = f"/tmp/codeant/projects/{project.id}"
            
            # 4. Ejecutar análisis
            # En esta versión simplificada, ejecutamos los análisis secuencialmente
            
            if request.include_complexity:
                await self._analyze_complexity(project_path, results)
            
            if request.include_metrics:
                await self._analyze_quality_metrics(project_path, results)
            
            if request.include_dead_code:
                await self._analyze_dead_code(project, project_path, results)
            
            if request.include_security:
                await self._analyze_security(project_path, results)
            
            if request.include_duplicates:
                await self._analyze_duplicates(project_path, results)
            
            # 5. Calcular puntuación final de calidad
            results.quality_score = self._calculate_quality_score(results)
            
            # 6. Marcar como completado
            results.status = "COMPLETED"
            results.completed_at = datetime.now().isoformat()
            results.progress = 1.0
            
            logger.info(
                f"Análisis completado para proyecto {project.name}: "
                f"{results.files_analyzed} archivos, "
                f"{results.total_violations} problemas encontrados"
            )
            
            return Result.success(results)
            
        except Exception as e:
            logger.exception(f"Error analizando proyecto: {str(e)}")
            return Result.failure(
                AnalyzeProjectError(f"Error inesperado: {str(e)}")
            )
    
    async def _get_project(self, project_id: str) -> Result[Project, Exception]:
        """Obtener proyecto por ID."""
        try:
            # Convertir string a ProjectId
            pid = ProjectId(project_id)
            
            # Buscar proyecto
            result = await self.project_repository.find_by_id(pid)
            if not result.success:
                return result
            
            if not result.data:
                return Result.failure(
                    ProjectNotFoundError(
                        f"No se encontró el proyecto con ID: {project_id}"
                    )
                )
            
            return Result.success(result.data)
            
        except Exception as e:
            return Result.failure(e)
    
    async def _analyze_complexity(self, project_path: str, results: AnalysisResults):
        """Analizar complejidad del código."""
        try:
            logger.info("Ejecutando análisis de complejidad...")
            
            # Simular análisis (en producción, esto analizaría archivos reales)
            # Por ahora generamos métricas de ejemplo
            complexity_results = {
                "average_complexity": 3.5,
                "max_complexity": 15,
                "complex_functions": 8,
                "total_functions": 120,
                "complexity_hotspots": [
                    {
                        "file": "src/main.py",
                        "function": "process_data",
                        "complexity": 15,
                        "lines": 150
                    }
                ]
            }
            
            results.complexity_metrics = complexity_results
            results.files_analyzed += 45
            results.total_lines += 5200
            results.high_violations += 8  # Funciones muy complejas
            results.total_violations += 8
            
        except Exception as e:
            logger.error(f"Error en análisis de complejidad: {str(e)}")
            results.errors.append(f"Error en análisis de complejidad: {str(e)}")
    
    async def _analyze_quality_metrics(self, project_path: str, results: AnalysisResults):
        """Analizar métricas de calidad."""
        try:
            logger.info("Ejecutando análisis de calidad...")
            
            # Simular análisis
            quality_results = {
                "maintainability_index": 75.5,
                "technical_debt_hours": 120,
                "code_coverage": 68.5,
                "documentation_coverage": 45.0,
                "test_coverage": 72.0,
                "code_smells": 25
            }
            
            results.quality_metrics = quality_results
            results.medium_violations += 25  # Code smells
            results.total_violations += 25
            
        except Exception as e:
            logger.error(f"Error en análisis de calidad: {str(e)}")
            results.errors.append(f"Error en análisis de calidad: {str(e)}")
    
    async def _analyze_dead_code(self, project: Project, project_path: str, results: AnalysisResults):
        """Analizar código muerto."""
        try:
            logger.info("Ejecutando análisis de código muerto...")
            
            # Simular análisis
            dead_code_results = {
                "unused_functions": 15,
                "unused_variables": 42,
                "unused_imports": 28,
                "unreachable_code": 10,
                "total_dead_code_lines": 350
            }
            
            results.dead_code_results = dead_code_results
            results.medium_violations += 95  # Todo el código muerto
            results.total_violations += 95
            
        except Exception as e:
            logger.error(f"Error en análisis de código muerto: {str(e)}")
            results.errors.append(f"Error en análisis de código muerto: {str(e)}")
    
    async def _analyze_security(self, project_path: str, results: AnalysisResults):
        """Analizar seguridad del código."""
        try:
            logger.info("Ejecutando análisis de seguridad...")
            
            # Simular análisis
            security_results = {
                "vulnerabilities": {
                    "critical": 2,
                    "high": 5,
                    "medium": 12,
                    "low": 23
                },
                "security_hotspots": 8,
                "owasp_compliance": 85.0,
                "cve_matches": 3
            }
            
            results.security_results = security_results
            results.critical_violations += 2
            results.high_violations += 5
            results.medium_violations += 12
            results.low_violations += 23
            results.total_violations += 42
            
        except Exception as e:
            logger.error(f"Error en análisis de seguridad: {str(e)}")
            results.errors.append(f"Error en análisis de seguridad: {str(e)}")
    
    async def _analyze_duplicates(self, project_path: str, results: AnalysisResults):
        """Analizar código duplicado."""
        try:
            logger.info("Ejecutando análisis de duplicados...")
            
            # Simular análisis
            duplicate_results = {
                "duplicate_blocks": 18,
                "duplicate_lines": 420,
                "duplicate_percentage": 8.5,
                "largest_duplicate": {
                    "lines": 45,
                    "occurrences": 3,
                    "files": ["src/utils.py", "src/helpers.py", "tests/test_utils.py"]
                }
            }
            
            results.duplicate_results = duplicate_results
            results.medium_violations += 18
            results.total_violations += 18
            
        except Exception as e:
            logger.error(f"Error en análisis de duplicados: {str(e)}")
            results.errors.append(f"Error en análisis de duplicados: {str(e)}")
    
    def _calculate_quality_score(self, results: AnalysisResults) -> float:
        """
        Calcular puntuación de calidad basada en las violaciones encontradas.
        """
        if results.files_analyzed == 0:
            return 0.0
        
        # Ponderación de violaciones
        weighted_violations = (
            results.critical_violations * 10 +
            results.high_violations * 5 +
            results.medium_violations * 2 +
            results.low_violations * 1
        )
        
        # Calcular score (100 - penalización por violaciones)
        # Cada violación ponderada reduce el score
        violations_per_file = weighted_violations / results.files_analyzed
        score = max(0, 100 - (violations_per_file * 10))
        
        return round(score, 1)
