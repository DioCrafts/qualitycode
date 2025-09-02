"""
Interfaz del repositorio para detección de código duplicado.

Este módulo define el contrato para el repositorio que proporciona
capacidades de detección de duplicación y análisis de similitud.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

from ..entities.clone_analysis import (
    CloneAnalysis, ProjectCloneAnalysis, Clone, ExactClone, StructuralClone,
    SemanticClone, CrossLanguageClone, InterFileClone, CloneClass,
    DuplicationMetrics, RefactoringOpportunity, SimilarityMetrics,
    CloneDetectionConfig, CodeBlock, CodeLocation, CloneType, SimilarityAlgorithm
)
from ..entities.parse_result import ParseResult
from ..value_objects.programming_language import ProgrammingLanguage


class CloneRepository(ABC):
    """
    Interfaz del repositorio de detección de código duplicado.
    
    Esta interfaz define el contrato para el repositorio que maneja
    la detección de duplicación de código y análisis de similitud.
    """
    
    @abstractmethod
    async def analyze_file_clones(
        self,
        parse_result: ParseResult,
        config: Optional[CloneDetectionConfig] = None
    ) -> CloneAnalysis:
        """
        Analiza duplicación de código en un archivo individual.
        
        Args:
            parse_result: Resultado del parsing del archivo
            config: Configuración opcional para la detección
            
        Returns:
            CloneAnalysis con los resultados del análisis
            
        Raises:
            CloneDetectionError: Si hay un error durante el análisis
        """
        pass
    
    @abstractmethod
    async def analyze_project_clones(
        self,
        parse_results: List[ParseResult],
        config: Optional[CloneDetectionConfig] = None
    ) -> ProjectCloneAnalysis:
        """
        Analiza duplicación de código en todo un proyecto.
        
        Args:
            parse_results: Lista de resultados de parsing de todos los archivos
            config: Configuración opcional para la detección
            
        Returns:
            ProjectCloneAnalysis con los resultados del análisis completo
            
        Raises:
            CloneDetectionError: Si hay un error durante el análisis
        """
        pass
    
    @abstractmethod
    async def detect_exact_clones(
        self,
        parse_result: ParseResult,
        min_size: int = 6
    ) -> List[ExactClone]:
        """
        Detecta clones exactos en un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            min_size: Tamaño mínimo en líneas para considerar un clone
            
        Returns:
            Lista de clones exactos encontrados
            
        Raises:
            ExactCloneDetectionError: Si hay un error en la detección
        """
        pass
    
    @abstractmethod
    async def detect_structural_clones(
        self,
        parse_result: ParseResult,
        similarity_threshold: float = 0.8
    ) -> List[StructuralClone]:
        """
        Detecta clones estructurales en un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            similarity_threshold: Umbral mínimo de similitud estructural
            
        Returns:
            Lista de clones estructurales encontrados
            
        Raises:
            StructuralCloneDetectionError: Si hay un error en la detección
        """
        pass
    
    @abstractmethod
    async def detect_semantic_clones(
        self,
        parse_result: ParseResult,
        similarity_threshold: float = 0.7
    ) -> List[SemanticClone]:
        """
        Detecta clones semánticos en un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            similarity_threshold: Umbral mínimo de similitud semántica
            
        Returns:
            Lista de clones semánticos encontrados
            
        Raises:
            SemanticCloneDetectionError: Si hay un error en la detección
        """
        pass
    
    @abstractmethod
    async def detect_cross_language_clones(
        self,
        parse_results: List[ParseResult],
        similarity_threshold: float = 0.6
    ) -> List[CrossLanguageClone]:
        """
        Detecta clones entre diferentes lenguajes.
        
        Args:
            parse_results: Lista de resultados de parsing en diferentes lenguajes
            similarity_threshold: Umbral mínimo de similitud cross-language
            
        Returns:
            Lista de clones cross-language encontrados
            
        Raises:
            CrossLanguageCloneDetectionError: Si hay un error en la detección
        """
        pass
    
    @abstractmethod
    async def detect_inter_file_clones(
        self,
        parse_results: List[ParseResult],
        config: Optional[CloneDetectionConfig] = None
    ) -> List[InterFileClone]:
        """
        Detecta clones que se extienden entre múltiples archivos.
        
        Args:
            parse_results: Lista de resultados de parsing
            config: Configuración opcional para la detección
            
        Returns:
            Lista de clones inter-archivo encontrados
            
        Raises:
            InterFileCloneDetectionError: Si hay un error en la detección
        """
        pass
    
    @abstractmethod
    async def calculate_similarity(
        self,
        code_block1: CodeBlock,
        code_block2: CodeBlock,
        algorithm: SimilarityAlgorithm
    ) -> SimilarityMetrics:
        """
        Calcula la similitud entre dos bloques de código.
        
        Args:
            code_block1: Primer bloque de código
            code_block2: Segundo bloque de código
            algorithm: Algoritmo de similitud a usar
            
        Returns:
            SimilarityMetrics con las métricas de similitud
            
        Raises:
            SimilarityCalculationError: Si hay un error en el cálculo
        """
        pass
    
    @abstractmethod
    async def extract_code_blocks(
        self,
        parse_result: ParseResult,
        min_size: int = 6
    ) -> List[CodeBlock]:
        """
        Extrae bloques de código de un archivo para análisis.
        
        Args:
            parse_result: Resultado del parsing del archivo
            min_size: Tamaño mínimo en líneas para extraer un bloque
            
        Returns:
            Lista de bloques de código extraídos
            
        Raises:
            CodeBlockExtractionError: Si hay un error en la extracción
        """
        pass
    
    @abstractmethod
    async def normalize_code(
        self,
        code: str,
        language: ProgrammingLanguage,
        normalization_options: Optional[Dict[str, bool]] = None
    ) -> str:
        """
        Normaliza código para comparación.
        
        Args:
            code: Código a normalizar
            language: Lenguaje del código
            normalization_options: Opciones de normalización
            
        Returns:
            Código normalizado
            
        Raises:
            CodeNormalizationError: Si hay un error en la normalización
        """
        pass
    
    @abstractmethod
    async def hash_code_block(
        self,
        code_block: CodeBlock,
        algorithm: str = "sha256"
    ) -> str:
        """
        Calcula hash de un bloque de código.
        
        Args:
            code_block: Bloque de código
            algorithm: Algoritmo de hashing a usar
            
        Returns:
            Hash del bloque de código
            
        Raises:
            HashCalculationError: Si hay un error calculando el hash
        """
        pass
    
    @abstractmethod
    async def classify_clones(
        self,
        clones: List[Clone]
    ) -> List[CloneClass]:
        """
        Clasifica clones en clases de clones.
        
        Args:
            clones: Lista de clones a clasificar
            
        Returns:
            Lista de clases de clones
            
        Raises:
            CloneClassificationError: Si hay un error en la clasificación
        """
        pass
    
    @abstractmethod
    async def suggest_refactorings(
        self,
        clone_analysis: CloneAnalysis
    ) -> List[RefactoringOpportunity]:
        """
        Sugiere oportunidades de refactoring basadas en clones.
        
        Args:
            clone_analysis: Análisis de clones
            
        Returns:
            Lista de oportunidades de refactoring
            
        Raises:
            RefactoringSuggestionError: Si hay un error generando sugerencias
        """
        pass
    
    @abstractmethod
    async def calculate_duplication_metrics(
        self,
        analysis: CloneAnalysis
    ) -> DuplicationMetrics:
        """
        Calcula métricas de duplicación detalladas.
        
        Args:
            analysis: Análisis de clones
            
        Returns:
            DuplicationMetrics con métricas calculadas
            
        Raises:
            MetricsCalculationError: Si hay un error calculando métricas
        """
        pass
    
    @abstractmethod
    async def calculate_project_duplication_metrics(
        self,
        project_analysis: ProjectCloneAnalysis
    ) -> DuplicationMetrics:
        """
        Calcula métricas de duplicación para todo el proyecto.
        
        Args:
            project_analysis: Análisis de clones del proyecto
            
        Returns:
            DuplicationMetrics agregadas del proyecto
            
        Raises:
            MetricsCalculationError: Si hay un error calculando métricas
        """
        pass
    
    @abstractmethod
    async def find_similar_code_blocks(
        self,
        target_block: CodeBlock,
        candidate_blocks: List[CodeBlock],
        similarity_threshold: float = 0.7,
        algorithm: SimilarityAlgorithm = SimilarityAlgorithm.LEVENSHTEIN
    ) -> List[Tuple[CodeBlock, SimilarityMetrics]]:
        """
        Encuentra bloques de código similares a un bloque objetivo.
        
        Args:
            target_block: Bloque objetivo
            candidate_blocks: Bloques candidatos
            similarity_threshold: Umbral de similitud mínimo
            algorithm: Algoritmo de similitud a usar
            
        Returns:
            Lista de bloques similares con sus métricas de similitud
            
        Raises:
            SimilaritySearchError: Si hay un error en la búsqueda
        """
        pass
    
    @abstractmethod
    async def is_detection_supported(
        self,
        language: ProgrammingLanguage
    ) -> bool:
        """
        Verifica si la detección está soportada para un lenguaje.
        
        Args:
            language: Lenguaje de programación
            
        Returns:
            True si está soportado, False en caso contrario
        """
        pass
    
    @abstractmethod
    async def get_supported_similarity_algorithms(
        self,
        language: ProgrammingLanguage
    ) -> List[SimilarityAlgorithm]:
        """
        Obtiene algoritmos de similitud soportados para un lenguaje.
        
        Args:
            language: Lenguaje de programación
            
        Returns:
            Lista de algoritmos soportados
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
    async def get_detection_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento de la detección.
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        pass
    
    @abstractmethod
    async def clear_cache(self) -> None:
        """
        Limpia la caché del detector de clones.
        """
        pass
    
    @abstractmethod
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la caché.
        
        Returns:
            Diccionario con estadísticas de la caché
        """
        pass
