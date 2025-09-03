"""
Dependencias compartidas para los endpoints de la API.
"""
from typing import Optional, Any

from ...domain.repositories.project_repository import ProjectRepository
from ...infrastructure.parsers.parser_factory import ParserFactory

# Para desarrollo, usaremos implementaciones mock o en memoria
_project_repository = None
_parser_factory = None
_dead_code_engine = None


def get_project_repository() -> ProjectRepository:
    """Obtener instancia del repositorio de proyectos."""
    global _project_repository
    
    if _project_repository is None:
        # Por ahora retornamos el mismo mock que se usa en projects.py
        from .routers.projects import get_project_repository as get_mock_repo
        return get_mock_repo()
    
    return _project_repository


def get_parser_factory() -> ParserFactory:
    """Obtener instancia del parser factory."""
    global _parser_factory
    
    if _parser_factory is None:
        _parser_factory = ParserFactory()
    
    return _parser_factory


def get_dead_code_engine() -> Optional[Any]:
    """Obtener instancia del motor de análisis de código muerto."""
    global _dead_code_engine
    
    if _dead_code_engine is None:
        # Por ahora retornamos None ya que requiere configuración compleja
        return None
    
    return _dead_code_engine


def get_analyze_project_use_case() -> AnalyzeProjectUseCase:
    """Obtener instancia del caso de uso de análisis."""
    from ...application.use_cases.analyze_project_use_case import AnalyzeProjectUseCase
    
    return AnalyzeProjectUseCase(
        project_repository=get_project_repository(),
        parser_factory=get_parser_factory(),
        dead_code_engine=get_dead_code_engine(),
        security_analyzer=None  # Por ahora sin analyzer de seguridad
    )
