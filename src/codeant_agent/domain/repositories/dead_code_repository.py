"""
Interfaz para el repositorio de código muerto.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..entities.dead_code_analysis import DeadCodeAnalysis, ProjectDeadCodeAnalysis
from ..entities.parse_result import ParseResult
from ..value_objects.programming_language import ProgrammingLanguage
from ...utils.error import Result

class DeadCodeRepository(ABC):
    """
    Repositorio para análisis de código muerto.
    """
    
    @abstractmethod
    async def analyze_file_dead_code(
        self, 
        parse_result: ParseResult, 
        config: Optional[Dict[str, Any]] = None
    ) -> Result[DeadCodeAnalysis, Exception]:
        """
        Analizar código muerto en un archivo.
        
        Args:
            parse_result: Resultado del parsing
            config: Configuración opcional
            
        Returns:
            Result con el análisis de código muerto o error
        """
        pass
    
    @abstractmethod
    async def analyze_project_dead_code(
        self,
        parse_results: List[ParseResult],
        config: Optional[Dict[str, Any]] = None
    ) -> Result[ProjectDeadCodeAnalysis, Exception]:
        """
        Analizar código muerto en un proyecto completo.
        
        Args:
            parse_results: Lista de resultados de parsing
            config: Configuración opcional
            
        Returns:
            Result con el análisis de proyecto o error
        """
        pass
    
    @abstractmethod
    async def is_analysis_supported(self, language: ProgrammingLanguage) -> Result[bool, Exception]:
        """
        Verificar si el análisis está soportado para un lenguaje.
        
        Args:
            language: Lenguaje a verificar
            
        Returns:
            Result con True si está soportado o error
        """
        pass
    
    @abstractmethod
    async def get_analysis_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del análisis.
        
        Returns:
            Diccionario con métricas
        """
        pass