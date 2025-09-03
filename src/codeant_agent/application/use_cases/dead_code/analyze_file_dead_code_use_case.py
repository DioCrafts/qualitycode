"""
Caso de uso para analizar código muerto en un archivo.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

from ....domain.entities.dead_code_analysis import DeadCodeAnalysis
from ....domain.entities.parse_result import ParseResult
from ....domain.repositories.dead_code_repository import DeadCodeRepository
from ....domain.repositories.parser_repository import ParserRepository
from ....domain.services.dead_code_service import DeadCodeClassificationService
from ....utils.error import Result, BaseError

logger = logging.getLogger(__name__)


class AnalyzeFileDeadCodeError(BaseError):
    """Error al analizar código muerto en archivo."""
    pass


class FileNotFoundError(AnalyzeFileDeadCodeError):
    """Error cuando el archivo no existe."""
    pass


class UnsupportedLanguageError(AnalyzeFileDeadCodeError):
    """Error cuando el lenguaje no está soportado."""
    pass


@dataclass
class AnalyzeFileDeadCodeRequest:
    """Request para analizar código muerto en un archivo."""
    file_path: Path
    content: Optional[str] = None  # Si no se proporciona, se lee del archivo
    config: Optional[Dict[str, Any]] = None
    include_suggestions: bool = True
    include_classification: bool = True
    confidence_threshold: float = 0.5


@dataclass
class AnalyzeFileDeadCodeResponse:
    """Response del análisis de código muerto de un archivo."""
    analysis: DeadCodeAnalysis
    classified_issues: Optional[Dict[str, List[Any]]] = None
    suggestions: Optional[List[Dict[str, Any]]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class AnalyzeFileDeadCodeUseCase:
    """
    Caso de uso para analizar código muerto en un archivo individual.
    
    Este caso de uso se encarga de:
    1. Validar la entrada
    2. Parsear el archivo
    3. Verificar soporte del lenguaje
    4. Realizar análisis de código muerto
    5. Clasificar los issues encontrados
    6. Generar sugerencias de fixes
    """
    
    def __init__(
        self,
        dead_code_repository: DeadCodeRepository,
        parser_repository: ParserRepository,
        classification_service: Optional[DeadCodeClassificationService] = None
    ):
        """
        Inicializar el caso de uso.
        
        Args:
            dead_code_repository: Repositorio para análisis de código muerto
            parser_repository: Repositorio para parsing de archivos
            classification_service: Servicio para clasificar issues
        """
        self.dead_code_repository = dead_code_repository
        self.parser_repository = parser_repository
        self.classification_service = classification_service or DeadCodeClassificationService()
    
    async def execute(
        self, 
        request: AnalyzeFileDeadCodeRequest
    ) -> Result[AnalyzeFileDeadCodeResponse, Exception]:
        """
        Ejecutar el análisis de código muerto en un archivo.
        
        Args:
            request: Datos para el análisis
            
        Returns:
            Result con el análisis o error
        """
        try:
            logger.info(f"Iniciando análisis de código muerto para: {request.file_path}")
            
            # 1. Validar request
            validation_result = self._validate_request(request)
            if not validation_result.success:
                return validation_result
            
            # 2. Parsear el archivo
            parse_result = await self._parse_file(request)
            if not parse_result.success:
                return Result.failure(
                    AnalyzeFileDeadCodeError(f"Error parsing archivo: {parse_result.error}")
                )
            
            # 3. Verificar soporte del lenguaje
            language_support_result = await self._check_language_support(parse_result.data)
            if not language_support_result.success:
                return language_support_result
            
            # 4. Realizar análisis de código muerto
            dead_code_analysis = await self._analyze_dead_code(parse_result.data, request.config)
            if not dead_code_analysis.success:
                return Result.failure(
                    AnalyzeFileDeadCodeError(f"Error en análisis: {dead_code_analysis.error}")
                )
            
            # 5. Filtrar por confianza
            filtered_analysis = self._filter_by_confidence(
                dead_code_analysis.data, request.confidence_threshold
            )
            
            # 6. Crear respuesta base
            response = AnalyzeFileDeadCodeResponse(analysis=filtered_analysis)
            
            # 7. Clasificar issues si se solicita
            if request.include_classification:
                classified_issues = self.classification_service.classify_dead_code_analysis(
                    filtered_analysis
                )
                response.classified_issues = classified_issues
                
                # 8. Generar sugerencias si se solicita
                if request.include_suggestions:
                    suggestions = self.classification_service.generate_removal_suggestions(
                        classified_issues
                    )
                    response.suggestions = suggestions
            
            # 9. Agregar métricas de rendimiento
            performance_metrics = await self.dead_code_repository.get_analysis_metrics()
            response.performance_metrics = performance_metrics
            
            logger.info(
                f"Análisis completado para {request.file_path}: "
                f"{filtered_analysis.statistics.get_total_issues()} issues encontrados"
            )
            
            return Result.success(response)
            
        except Exception as e:
            logger.error(f"Error analizando código muerto en {request.file_path}: {e}")
            return Result.failure(
                AnalyzeFileDeadCodeError(f"Error inesperado: {str(e)}")
            )
    
    def _validate_request(
        self, 
        request: AnalyzeFileDeadCodeRequest
    ) -> Result[None, Exception]:
        """Validar la request."""
        if not request.file_path:
            return Result.failure(
                AnalyzeFileDeadCodeError("La ruta del archivo es requerida")
            )
        
        # Si no se proporciona contenido, verificar que el archivo existe
        if not request.content and not request.file_path.exists():
            return Result.failure(
                FileNotFoundError(f"El archivo {request.file_path} no existe")
            )
        
        if request.confidence_threshold < 0 or request.confidence_threshold > 1:
            return Result.failure(
                AnalyzeFileDeadCodeError("El umbral de confianza debe estar entre 0 y 1")
            )
        
        return Result.success(None)
    
    async def _parse_file(
        self, 
        request: AnalyzeFileDeadCodeRequest
    ) -> Result[ParseResult, Exception]:
        """Parsear el archivo."""
        try:
            if request.content:
                # Detectar lenguaje y parsear contenido
                language = await self.parser_repository.detect_language(
                    request.file_path, request.content
                )
                parse_result = await self.parser_repository.parse_content(
                    request.content, language
                )
            else:
                # Parsear archivo directamente
                parse_result = await self.parser_repository.parse_file(request.file_path)
            
            return Result.success(parse_result)
            
        except Exception as e:
            logger.error(f"Error parsing archivo {request.file_path}: {e}")
            return Result.failure(e)
    
    async def _check_language_support(
        self, 
        parse_result: ParseResult
    ) -> Result[None, Exception]:
        """Verificar soporte del lenguaje."""
        try:
            is_supported = await self.dead_code_repository.is_analysis_supported(
                parse_result.language
            )
            
            if not is_supported:
                return Result.failure(
                    UnsupportedLanguageError(
                        f"Análisis de código muerto no soportado para {parse_result.language.get_name()}"
                    )
                )
            
            return Result.success(None)
            
        except Exception as e:
            return Result.failure(
                AnalyzeFileDeadCodeError(f"Error verificando soporte de lenguaje: {e}")
            )
    
    async def _analyze_dead_code(
        self, 
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]]
    ) -> Result[DeadCodeAnalysis, Exception]:
        """Realizar el análisis de código muerto."""
        try:
            analysis = await self.dead_code_repository.analyze_file_dead_code(
                parse_result, config
            )
            
            return Result.success(analysis)
            
        except Exception as e:
            logger.error(f"Error en análisis de código muerto: {e}")
            return Result.failure(e)
    
    def _filter_by_confidence(
        self, 
        analysis: DeadCodeAnalysis,
        threshold: float
    ) -> DeadCodeAnalysis:
        """Filtrar issues por umbral de confianza."""
        # Filtrar variables no utilizadas
        analysis.unused_variables = [
            var for var in analysis.unused_variables 
            if var.confidence >= threshold
        ]
        
        # Filtrar funciones no utilizadas
        analysis.unused_functions = [
            func for func in analysis.unused_functions 
            if func.confidence >= threshold
        ]
        
        # Filtrar clases no utilizadas
        analysis.unused_classes = [
            cls for cls in analysis.unused_classes 
            if cls.confidence >= threshold
        ]
        
        # Filtrar imports no utilizados
        analysis.unused_imports = [
            imp for imp in analysis.unused_imports 
            if imp.confidence >= threshold
        ]
        
        # Filtrar código inalcanzable
        analysis.unreachable_code = [
            code for code in analysis.unreachable_code 
            if code.confidence >= threshold
        ]
        
        # Filtrar ramas muertas
        analysis.dead_branches = [
            branch for branch in analysis.dead_branches 
            if branch.confidence >= threshold
        ]
        
        # Filtrar parámetros no utilizados
        analysis.unused_parameters = [
            param for param in analysis.unused_parameters 
            if param.confidence >= threshold
        ]
        
        # Filtrar asignaciones redundantes
        analysis.redundant_assignments = [
            assign for assign in analysis.redundant_assignments 
            if assign.confidence >= threshold
        ]
        
        # Recalcular estadísticas después del filtrado
        analysis.statistics = self._recalculate_statistics(analysis)
        
        return analysis
    
    def _recalculate_statistics(self, analysis: DeadCodeAnalysis) -> Any:
        """Recalcular estadísticas después del filtrado."""
        from ....domain.entities.dead_code_analysis import DeadCodeStatistics
        
        stats = DeadCodeStatistics()
        stats.total_unused_variables = len(analysis.unused_variables)
        stats.total_unused_functions = len(analysis.unused_functions)
        stats.total_unused_classes = len(analysis.unused_classes)
        stats.total_unused_imports = len(analysis.unused_imports)
        stats.total_unreachable_code_blocks = len(analysis.unreachable_code)
        stats.total_dead_branches = len(analysis.dead_branches)
        stats.total_unused_parameters = len(analysis.unused_parameters)
        stats.total_redundant_assignments = len(analysis.redundant_assignments)
        
        return stats


class QuickAnalyzeFileDeadCodeUseCase:
    """
    Caso de uso simplificado para análisis rápido de código muerto.
    
    Versión optimizada que realiza un análisis menos exhaustivo 
    pero más rápido, útil para feedback en tiempo real.
    """
    
    def __init__(
        self,
        dead_code_repository: DeadCodeRepository,
        parser_repository: ParserRepository
    ):
        self.dead_code_repository = dead_code_repository
        self.parser_repository = parser_repository
    
    async def execute(
        self, 
        file_path: Path, 
        content: Optional[str] = None
    ) -> Result[DeadCodeAnalysis, Exception]:
        """
        Ejecutar análisis rápido.
        
        Args:
            file_path: Ruta del archivo
            content: Contenido opcional del archivo
            
        Returns:
            Result con análisis básico o error
        """
        try:
            # Configuración para análisis rápido
            quick_config = {
                'analyze_unused_variables': True,
                'analyze_unused_imports': True,
                'analyze_unreachable_code': True,
                'analyze_unused_functions': False,  # Más lento
                'analyze_unused_classes': False,   # Más lento
                'cross_module_analysis': False,    # Mucho más lento
                'confidence_threshold': 0.8        # Solo alta confianza
            }
            
            request = AnalyzeFileDeadCodeRequest(
                file_path=file_path,
                content=content,
                config=quick_config,
                include_suggestions=False,
                include_classification=False,
                confidence_threshold=0.8
            )
            
            full_use_case = AnalyzeFileDeadCodeUseCase(
                self.dead_code_repository,
                self.parser_repository
            )
            
            result = await full_use_case.execute(request)
            
            if result.success:
                return Result.success(result.data.analysis)
            else:
                return Result.failure(result.error)
                
        except Exception as e:
            logger.error(f"Error en análisis rápido: {e}")
            return Result.failure(
                AnalyzeFileDeadCodeError(f"Error en análisis rápido: {str(e)}")
            )
