"""
Detector principal de código duplicado (Orquestador).

Este módulo implementa el detector principal que coordina todos los
detectores especializados para proporcionar análisis completo de duplicación.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
import time
from pathlib import Path

from ...domain.entities.clone_analysis import (
    CloneAnalysis, ProjectCloneAnalysis, CloneDetectionConfig,
    CloneClass, CloneType, DuplicationMetrics, RefactoringOpportunity,
    ExactClone, StructuralClone, SemanticClone, CrossLanguageClone,
    InterFileClone, CloneId, Clone, CloneClassId, CloneClassMetrics, RefactoringPotential
)
from ...domain.entities.parse_result import ParseResult
from ...domain.value_objects.programming_language import ProgrammingLanguage

from .exact_clone_detector import ExactCloneDetector, ExactCloneDetectionResult
from .structural_clone_detector import StructuralCloneDetector, StructuralCloneDetectionResult
from .semantic_clone_detector import SemanticCloneDetector, SemanticCloneDetectionResult
from .cross_language_detector import CrossLanguageCloneDetector, CrossLanguageCloneDetectionResult
from .similarity_calculator import SimilarityCalculator
from .refactoring_suggester import RefactoringSuggester

logger = logging.getLogger(__name__)


@dataclass
class CloneDetectionMetrics:
    """Métricas de performance de detección."""
    total_analysis_time_ms: int
    exact_detection_time_ms: int
    structural_detection_time_ms: int
    semantic_detection_time_ms: int
    cross_language_detection_time_ms: int
    refactoring_suggestion_time_ms: int
    files_analyzed: int
    total_clones_found: int
    clones_by_type: Dict[CloneType, int]


@dataclass
class CloneClassificationResult:
    """Resultado de clasificación de clones."""
    clone_classes: List[CloneClass]
    classification_time_ms: int
    total_clone_instances: int
    largest_clone_class_size: int


class CloneClassifier:
    """Clasificador de clones en familias/clases."""
    
    def __init__(self, similarity_calculator: SimilarityCalculator):
        self.similarity_calculator = similarity_calculator
        self.similarity_threshold = 0.8
    
    async def classify_clones(self, all_clones: List[Clone]) -> CloneClassificationResult:
        """
        Clasifica clones en familias/clases basado en similitud.
        
        Args:
            all_clones: Lista de todos los clones detectados
            
        Returns:
            CloneClassificationResult con clases organizadas
        """
        start_time = time.time()
        
        clone_classes = []
        processed_clones = set()
        
        for clone in all_clones:
            if clone.id.value in processed_clones:
                continue
            
            # Encontrar clones similares para formar una clase
            similar_clones = await self._find_similar_clones(clone, all_clones, processed_clones)
            
            if len(similar_clones) >= 2:  # Al menos 2 clones para formar una clase
                clone_class = await self._create_clone_class(similar_clones)
                clone_classes.append(clone_class)
                
                # Marcar como procesados
                for similar_clone in similar_clones:
                    processed_clones.add(similar_clone.id.value)
        
        analysis_time = int((time.time() - start_time) * 1000)
        largest_class_size = max((len(cc.instances) for cc in clone_classes), default=0)
        
        return CloneClassificationResult(
            clone_classes=clone_classes,
            classification_time_ms=analysis_time,
            total_clone_instances=len(all_clones),
            largest_clone_class_size=largest_class_size
        )
    
    async def _find_similar_clones(self, target_clone: Clone, all_clones: List[Clone], 
                                 processed: Set[str]) -> List[Clone]:
        """Encuentra clones similares al clone objetivo."""
        similar_clones = [target_clone]
        
        for candidate_clone in all_clones:
            if (candidate_clone.id.value != target_clone.id.value and
                candidate_clone.id.value not in processed):
                
                # Verificar si son del mismo tipo
                if candidate_clone.clone_type == target_clone.clone_type:
                    # Calcular similitud adicional si es necesario
                    additional_similarity = await self._calculate_clone_similarity(
                        target_clone, candidate_clone
                    )
                    
                    if additional_similarity >= self.similarity_threshold:
                        similar_clones.append(candidate_clone)
        
        return similar_clones
    
    async def _calculate_clone_similarity(self, clone1: Clone, clone2: Clone) -> float:
        """Calcula similitud adicional entre dos clones."""
        # Para clones exactos, similitud perfecta
        if clone1.clone_type == CloneType.EXACT and clone2.clone_type == CloneType.EXACT:
            if (hasattr(clone1, 'hash_value') and hasattr(clone2, 'hash_value') and
                clone1.hash_value == clone2.hash_value):
                return 1.0
        
        # Para clones estructurales, usar similitud existente
        if isinstance(clone1, StructuralClone) and isinstance(clone2, StructuralClone):
            return min(clone1.structural_similarity, clone2.structural_similarity)
        
        # Para clones semánticos, usar similitud semántica
        if isinstance(clone1, SemanticClone) and isinstance(clone2, SemanticClone):
            return min(clone1.semantic_similarity, clone2.semantic_similarity)
        
        # Default: usar similarity score promedio
        return (clone1.similarity_score + clone2.similarity_score) / 2.0
    
    async def _create_clone_class(self, similar_clones: List[Clone]) -> CloneClass:
        """Crea una clase de clones."""
        
        # Calcular métricas de la clase
        total_lines = sum(clone.size_lines for clone in similar_clones)
        avg_lines = total_lines / len(similar_clones)
        
        total_tokens = sum(clone.size_tokens for clone in similar_clones)
        avg_tokens = total_tokens / len(similar_clones)
        
        avg_similarity = sum(clone.similarity_score for clone in similar_clones) / len(similar_clones)
        
        # Determinar potencial de refactoring
        refactoring_potential = RefactoringPotential(
            overall_score=avg_similarity,
            maintainability_impact=0.8 if len(similar_clones) >= 3 else 0.6,
            complexity_reduction=0.7 if similar_clones[0].clone_type == CloneType.EXACT else 0.5,
            reusability_improvement=0.9 if len(similar_clones) >= 2 else 0.4,
            factors=[
                f"clone_count_{len(similar_clones)}",
                f"clone_type_{similar_clones[0].clone_type.value}",
                f"similarity_{avg_similarity:.2f}"
            ]
        )
        
        clone_class = CloneClass(
            id=CloneClassId(),
            clone_type=similar_clones[0].clone_type,
            instances=similar_clones,
            similarity_score=avg_similarity,
            size_metrics=CloneClassMetrics(
                total_instances=len(similar_clones),
                total_lines=total_lines,
                total_tokens=total_tokens,
                average_similarity=avg_similarity,
                size_variance=self._calculate_size_variance(similar_clones),
                complexity_metrics={
                    "average_lines": avg_lines,
                    "average_tokens": avg_tokens,
                    "files_involved": len(set(clone.original_location.file_path for clone in similar_clones))
                }
            ),
            refactoring_potential=refactoring_potential
        )
        
        return clone_class
    
    def _calculate_size_variance(self, clones: List[Clone]) -> float:
        """Calcula varianza en tamaños de clones."""
        if len(clones) <= 1:
            return 0.0
        
        sizes = [clone.size_lines for clone in clones]
        mean_size = sum(sizes) / len(sizes)
        variance = sum((size - mean_size) ** 2 for size in sizes) / len(sizes)
        
        return variance


class CloneDetector:
    """Detector principal de código duplicado que orquesta todos los detectores."""
    
    def __init__(self, config: Optional[CloneDetectionConfig] = None):
        """
        Inicializa el detector principal.
        
        Args:
            config: Configuración del detector
        """
        self.config = config or CloneDetectionConfig()
        
        # Inicializar detectores especializados
        self.exact_detector = ExactCloneDetector()
        self.structural_detector = StructuralCloneDetector()
        self.semantic_detector = SemanticCloneDetector()
        self.cross_language_detector = CrossLanguageCloneDetector()
        
        # Inicializar componentes auxiliares
        self.similarity_calculator = SimilarityCalculator()
        self.refactoring_suggester = RefactoringSuggester()
        self.clone_classifier = CloneClassifier(self.similarity_calculator)
    
    async def detect_clones(self, parse_result: ParseResult) -> CloneAnalysis:
        """
        Detecta todos los tipos de clones en un archivo.
        
        Args:
            parse_result: Resultado del parsing del archivo
            
        Returns:
            CloneAnalysis completo del archivo
        """
        start_time = time.time()
        
        try:
            logger.info(f"Iniciando detección de clones para {parse_result.file_path}")
            
            # Inicializar análisis
            analysis = CloneAnalysis(
                file_path=parse_result.file_path,
                language=parse_result.language
            )
            
            detection_times = {}
            
            # 1. Detección de clones exactos
            if self.config.enable_exact_detection:
                exact_start = time.time()
                exact_result = await self.exact_detector.detect_exact_clones(
                    parse_result, self._get_config_dict()
                )
                analysis.exact_clones = exact_result.exact_clones
                detection_times['exact'] = int((time.time() - exact_start) * 1000)
                logger.debug(f"Clones exactos detectados: {len(analysis.exact_clones)}")
            
            # 2. Detección de clones estructurales
            if self.config.enable_structural_detection:
                structural_start = time.time()
                structural_result = await self.structural_detector.detect_structural_clones(
                    parse_result, self._get_config_dict()
                )
                analysis.structural_clones = structural_result.structural_clones
                detection_times['structural'] = int((time.time() - structural_start) * 1000)
                logger.debug(f"Clones estructurales detectados: {len(analysis.structural_clones)}")
            
            # 3. Detección de clones semánticos
            if self.config.enable_semantic_detection:
                semantic_start = time.time()
                semantic_result = await self.semantic_detector.detect_semantic_clones(
                    parse_result, self._get_config_dict()
                )
                analysis.semantic_clones = semantic_result.semantic_clones
                detection_times['semantic'] = int((time.time() - semantic_start) * 1000)
                logger.debug(f"Clones semánticos detectados: {len(analysis.semantic_clones)}")
            
            # 4. Clasificar clones en familias/clases
            all_clones = analysis.get_all_clones()
            if all_clones:
                classification_result = await self.clone_classifier.classify_clones(all_clones)
                analysis.clone_classes = classification_result.clone_classes
                logger.debug(f"Clases de clones creadas: {len(analysis.clone_classes)}")
            
            # 5. Generar sugerencias de refactoring
            if analysis.clone_classes:
                refactoring_start = time.time()
                analysis.refactoring_opportunities = await self.refactoring_suggester.suggest_refactorings(
                    analysis.clone_classes, self._get_config_dict()
                )
                detection_times['refactoring'] = int((time.time() - refactoring_start) * 1000)
                logger.debug(f"Oportunidades de refactoring: {len(analysis.refactoring_opportunities)}")
            
            # 6. Calcular métricas de duplicación
            analysis.duplication_metrics = self._calculate_duplication_metrics(analysis, parse_result)
            
            # 7. Registrar tiempo total
            analysis.execution_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Detección completada para {parse_result.file_path}: "
                f"{analysis.get_total_clones()} clones en {analysis.execution_time_ms}ms"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error en detección de clones para {parse_result.file_path}: {e}")
            raise
    
    async def detect_clones_project(self, parse_results: List[ParseResult]) -> ProjectCloneAnalysis:
        """
        Detecta clones en un proyecto completo (múltiples archivos).
        
        Args:
            parse_results: Lista de resultados de parsing de todos los archivos
            
        Returns:
            ProjectCloneAnalysis completo del proyecto
        """
        start_time = time.time()
        
        try:
            logger.info(f"Iniciando análisis de proyecto con {len(parse_results)} archivos")
            
            # 1. Análisis individual por archivo
            file_analyses = []
            for parse_result in parse_results:
                try:
                    file_analysis = await self.detect_clones(parse_result)
                    file_analyses.append(file_analysis)
                except Exception as e:
                    logger.warning(f"Error analizando {parse_result.file_path}: {e}")
                    # Crear análisis vacío para mantener consistencia
                    empty_analysis = CloneAnalysis(
                        file_path=parse_result.file_path,
                        language=parse_result.language,
                        execution_time_ms=0
                    )
                    file_analyses.append(empty_analysis)
            
            # 2. Detección de clones inter-archivo
            inter_file_clones = await self._detect_inter_file_clones(parse_results)
            logger.info(f"Clones inter-archivo detectados: {len(inter_file_clones)}")
            
            # 3. Detección cross-language (si está habilitado)
            cross_language_clones = []
            if self.config.enable_cross_language_detection:
                cross_lang_result = await self.cross_language_detector.detect_cross_language_clones(
                    parse_results, self._get_config_dict()
                )
                cross_language_clones = cross_lang_result.cross_language_clones
                logger.info(f"Clones cross-language detectados: {len(cross_language_clones)}")
            
            # 4. Construir clases globales de clones
            global_clone_classes = await self._build_global_clone_classes(
                file_analyses, inter_file_clones, cross_language_clones
            )
            
            # 5. Calcular métricas del proyecto
            project_metrics = self._calculate_project_metrics(
                file_analyses, inter_file_clones, cross_language_clones
            )
            
            # 6. Generar oportunidades de refactoring a nivel de proyecto
            project_refactoring_opportunities = await self.refactoring_suggester.suggest_refactorings(
                global_clone_classes, self._get_config_dict()
            )
            
            # 7. Construir análisis final
            project_analysis = ProjectCloneAnalysis(
                project_path=parse_results[0].file_path.parent if parse_results else None,
                file_analyses=file_analyses,
                inter_file_clones=inter_file_clones,
                cross_language_clones=cross_language_clones,
                global_clone_classes=global_clone_classes,
                project_metrics=project_metrics,
                project_refactoring_opportunities=project_refactoring_opportunities,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
            
            logger.info(
                f"Análisis de proyecto completado: "
                f"{len(global_clone_classes)} clases de clones, "
                f"{len(project_refactoring_opportunities)} oportunidades de refactoring "
                f"en {project_analysis.execution_time_ms}ms"
            )
            
            return project_analysis
            
        except Exception as e:
            logger.error(f"Error en análisis de proyecto: {e}")
            raise
    
    async def _detect_inter_file_clones(self, parse_results: List[ParseResult]) -> List[InterFileClone]:
        """Detecta clones entre diferentes archivos."""
        inter_file_clones = []
        
        try:
            # Agrupar por lenguaje para optimización
            by_language = {}
            for pr in parse_results:
                if pr.language not in by_language:
                    by_language[pr.language] = []
                by_language[pr.language].append(pr)
            
            # Detectar clones exactos inter-archivo
            if self.config.enable_exact_detection:
                exact_inter_file = await self.exact_detector.detect_exact_clones_between_files(
                    parse_results, self._get_config_dict()
                )
                
                # Convertir a InterFileClone
                for exact_clone in exact_inter_file:
                    inter_file_clone = InterFileClone(
                        clone_type=CloneType.EXACT,
                        file_locations=[exact_clone.original_location, exact_clone.duplicate_location],
                        similarity_score=exact_clone.similarity_score,
                        confidence=exact_clone.confidence
                    )
                    inter_file_clones.append(inter_file_clone)
            
            # Detectar clones estructurales inter-archivo
            if self.config.enable_structural_detection:
                structural_inter_file = await self.structural_detector.detect_structural_clones_between_files(
                    parse_results, self._get_config_dict()
                )
                
                # Convertir a InterFileClone
                for structural_clone in structural_inter_file:
                    inter_file_clone = InterFileClone(
                        clone_type=structural_clone.clone_type,
                        file_locations=[structural_clone.original_location, structural_clone.duplicate_location],
                        similarity_score=structural_clone.similarity_score,
                        confidence=structural_clone.confidence
                    )
                    inter_file_clones.append(inter_file_clone)
            
        except Exception as e:
            logger.error(f"Error detectando clones inter-archivo: {e}")
        
        return inter_file_clones
    
    async def _build_global_clone_classes(self, file_analyses: List[CloneAnalysis],
                                        inter_file_clones: List[InterFileClone],
                                        cross_language_clones: List[CrossLanguageClone]) -> List[CloneClass]:
        """Construye clases globales de clones a nivel de proyecto."""
        # Recopilar todos los clones
        all_clones = []
        
        # Clones de archivos individuales
        for analysis in file_analyses:
            all_clones.extend(analysis.get_all_clones())
        
        # Clones inter-archivo (convertir a formato estándar)
        for inter_file_clone in inter_file_clones:
            # Crear clone estándar basado en inter_file_clone
            if len(inter_file_clone.file_locations) >= 2:
                standard_clone = ExactClone(  # Usar ExactClone como base
                    clone_type=inter_file_clone.clone_type,
                    original_location=inter_file_clone.file_locations[0],
                    duplicate_location=inter_file_clone.file_locations[1],
                    similarity_score=inter_file_clone.similarity_score,
                    confidence=inter_file_clone.confidence
                )
                all_clones.append(standard_clone)
        
        # Clones cross-language
        all_clones.extend(cross_language_clones)
        
        # Clasificar todos los clones
        if all_clones:
            classification_result = await self.clone_classifier.classify_clones(all_clones)
            return classification_result.clone_classes
        
        return []
    
    def _calculate_duplication_metrics(self, analysis: CloneAnalysis, parse_result: ParseResult) -> DuplicationMetrics:
        """Calcula métricas de duplicación para un archivo."""
        total_clones = analysis.get_total_clones()
        
        # Calcular líneas duplicadas
        duplicated_lines = sum(clone.size_lines for clone in analysis.get_all_clones())
        
        # Obtener líneas totales del archivo
        total_lines = parse_result.metadata.line_count
        
        # Calcular porcentajes
        duplication_percentage = (duplicated_lines / total_lines * 100) if total_lines > 0 else 0.0
        
        metrics = DuplicationMetrics(
            total_lines=total_lines,
            duplicated_lines=duplicated_lines,
            total_clones=total_clones,
            clone_classes=len(analysis.clone_classes),
            exact_clones=len(analysis.exact_clones),
            structural_clones=len(analysis.structural_clones),
            semantic_clones=len(analysis.semantic_clones),
            cross_language_clones=len(analysis.cross_language_clones),
            largest_clone_size=max((clone.size_lines for clone in analysis.get_all_clones()), default=0)
        )
        
        # Calcular métricas derivadas
        metrics.calculate_derived_metrics()
        
        return metrics
    
    def _calculate_project_metrics(self, file_analyses: List[CloneAnalysis],
                                 inter_file_clones: List[InterFileClone],
                                 cross_language_clones: List[CrossLanguageClone]) -> DuplicationMetrics:
        """Calcula métricas de duplicación para el proyecto completo."""
        # Agregar métricas de todos los archivos
        total_clones = sum(analysis.get_total_clones() for analysis in file_analyses)
        total_clones += len(inter_file_clones) + len(cross_language_clones)
        
        total_duplicated_lines = sum(analysis.duplication_metrics.duplicated_lines for analysis in file_analyses)
        total_project_lines = sum(
            analysis.duplication_metrics.duplicated_lines + 
            (1000 - analysis.duplication_metrics.duplicated_lines)  # Estimación simplificada
            for analysis in file_analyses
        )
        
        # Agregar conteos por tipo
        total_exact_clones = sum(analysis.duplication_metrics.exact_clones for analysis in file_analyses)
        total_structural_clones = sum(analysis.duplication_metrics.structural_clones for analysis in file_analyses)
        total_semantic_clones = sum(analysis.duplication_metrics.semantic_clones for analysis in file_analyses)
        total_cross_language_clones = sum(analysis.duplication_metrics.cross_language_clones for analysis in file_analyses) + len(cross_language_clones)
        
        duplication_percentage = (total_duplicated_lines / total_project_lines * 100) if total_project_lines > 0 else 0.0
        
        # Crear métricas del proyecto
        project_metrics = DuplicationMetrics(
            total_lines=total_project_lines,
            duplicated_lines=total_duplicated_lines,
            total_clones=total_clones,
            clone_classes=sum(len(analysis.clone_classes) for analysis in file_analyses),
            exact_clones=total_exact_clones,
            structural_clones=total_structural_clones,
            semantic_clones=total_semantic_clones,
            cross_language_clones=total_cross_language_clones,
            largest_clone_size=max(
                (analysis.duplication_metrics.largest_clone_size for analysis in file_analyses),
                default=0
            )
        )
        
        # Calcular métricas derivadas
        project_metrics.calculate_derived_metrics()
        
        return project_metrics
    
    def _get_config_dict(self) -> Dict[str, Any]:
        """Convierte configuración a diccionario para paso a detectores."""
        return {
            'min_clone_size_lines': self.config.min_clone_size_lines,
            'min_clone_size_tokens': self.config.min_clone_size_tokens,
            'min_similarity': self.config.min_similarity_threshold,
            'ignore_whitespace': self.config.ignore_whitespace,
            'ignore_comments': self.config.ignore_comments,
            'aggressive_mode': self.config.language_specific_configs
        }
    
    async def get_detection_metrics(self, analysis: CloneAnalysis) -> CloneDetectionMetrics:
        """
        Obtiene métricas detalladas de la detección.
        
        Args:
            analysis: Análisis de clones completado
            
        Returns:
            CloneDetectionMetrics con información de performance
        """
        clones_by_type = {
            CloneType.EXACT: len(analysis.exact_clones),
            CloneType.RENAMED: len([c for c in analysis.structural_clones if c.clone_type == CloneType.RENAMED]),
            CloneType.NEAR_MISS: len([c for c in analysis.structural_clones if c.clone_type == CloneType.NEAR_MISS]),
            CloneType.SEMANTIC: len(analysis.semantic_clones),
            CloneType.CROSS_LANGUAGE: len(analysis.cross_language_clones)
        }
        
        return CloneDetectionMetrics(
            total_analysis_time_ms=analysis.execution_time_ms,
            exact_detection_time_ms=0,  # Sería calculado en implementación completa
            structural_detection_time_ms=0,
            semantic_detection_time_ms=0,
            cross_language_detection_time_ms=0,
            refactoring_suggestion_time_ms=0,
            files_analyzed=1,
            total_clones_found=analysis.get_total_clones(),
            clones_by_type=clones_by_type
        )
    
    async def analyze_clone_evolution(self, current_analysis: ProjectCloneAnalysis,
                                    previous_analysis: Optional[ProjectCloneAnalysis] = None) -> Dict[str, Any]:
        """
        Analiza la evolución de clones entre análisis.
        
        Args:
            current_analysis: Análisis actual
            previous_analysis: Análisis anterior (opcional)
            
        Returns:
            Diccionario con análisis de evolución
        """
        if not previous_analysis:
            return {
                "status": "baseline_analysis",
                "total_clones": current_analysis.project_metrics.total_clones,
                "duplication_percentage": current_analysis.project_metrics.duplication_percentage
            }
        
        # Comparar métricas
        current_metrics = current_analysis.project_metrics
        previous_metrics = previous_analysis.project_metrics
        
        clone_change = current_metrics.total_clones - previous_metrics.total_clones
        duplication_change = current_metrics.duplication_percentage - previous_metrics.duplication_percentage
        
        return {
            "status": "evolution_analysis",
            "clone_count_change": clone_change,
            "duplication_percentage_change": duplication_change,
            "trend": "improving" if clone_change < 0 else "degrading" if clone_change > 0 else "stable",
            "new_refactoring_opportunities": len(current_analysis.project_refactoring_opportunities),
            "analysis_time_comparison": {
                "current_ms": current_analysis.execution_time_ms,
                "previous_ms": previous_analysis.execution_time_ms,
                "performance_change": current_analysis.execution_time_ms - previous_analysis.execution_time_ms
            }
        }
    
    def get_high_priority_refactoring_opportunities(self, analysis: ProjectCloneAnalysis, 
                                                   limit: int = 10) -> List[RefactoringOpportunity]:
        """
        Obtiene oportunidades de refactoring de alta prioridad.
        
        Args:
            analysis: Análisis del proyecto
            limit: Número máximo de oportunidades a retornar
            
        Returns:
            Lista de oportunidades ordenadas por prioridad
        """
        # Ordenar por impacto potencial
        sorted_opportunities = sorted(
            analysis.project_refactoring_opportunities,
            key=lambda opp: (
                opp.confidence * len(opp.affected_clones) * len(opp.potential_benefits)
            ),
            reverse=True
        )
        
        return sorted_opportunities[:limit]
    
    def generate_duplication_report(self, analysis: ProjectCloneAnalysis) -> Dict[str, Any]:
        """
        Genera reporte comprehensivo de duplicación.
        
        Args:
            analysis: Análisis del proyecto
            
        Returns:
            Diccionario con reporte detallado
        """
        return {
            "summary": {
                "total_files_analyzed": analysis.get_total_files_analyzed(),
                "files_with_clones": len(analysis.get_files_with_clones()),
                "total_clone_classes": len(analysis.global_clone_classes),
                "duplication_percentage": analysis.project_metrics.duplication_percentage,
                "refactoring_opportunities": len(analysis.project_refactoring_opportunities)
            },
            "breakdown_by_type": {
                CloneType.EXACT: analysis.project_metrics.exact_clones,
                CloneType.RENAMED: 0,  # Calculado desde structural_clones
                CloneType.NEAR_MISS: 0,  # Calculado desde structural_clones  
                CloneType.SEMANTIC: analysis.project_metrics.semantic_clones,
                CloneType.CROSS_LANGUAGE: analysis.project_metrics.cross_language_clones
            },
            "top_duplicated_files": [
                {
                    "file": str(fa.file_path),
                    "duplication_percentage": fa.duplication_metrics.duplication_percentage,
                    "clone_count": fa.get_total_clones()
                }
                for fa in analysis.get_worst_duplication_files(5)
            ],
            "high_priority_refactorings": [
                {
                    "type": opp.refactoring_type.value,
                    "description": opp.description,
                    "confidence": opp.confidence,
                    "effort": opp.estimated_effort.value,
                    "affected_files": len(opp.get_affected_files())
                }
                for opp in self.get_high_priority_refactoring_opportunities(analysis, 5)
            ],
            "performance": {
                "analysis_time_ms": analysis.execution_time_ms,
                "average_time_per_file": (
                    analysis.execution_time_ms / analysis.get_total_files_analyzed()
                    if analysis.get_total_files_analyzed() > 0 else 0
                )
            }
        }
