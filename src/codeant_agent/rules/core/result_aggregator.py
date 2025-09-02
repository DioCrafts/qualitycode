"""
Agregador de resultados para el motor de reglas estáticas.

Este módulo implementa la funcionalidad para agregar y procesar resultados
de ejecución de reglas, generando reportes consolidados y métricas.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..models.rule_models import (
    RuleResult,
    Violation,
    Suggestion,
    ProjectConfig,
    AnalysisMetrics,
    RuleSeverity,
    RuleCategory
)

logger = logging.getLogger(__name__)


class AggregatorError(Exception):
    """Excepción base para errores del agregador."""
    pass


@dataclass
class AggregatedResults:
    """Resultados agregados de ejecución de reglas."""
    violations: List[Violation] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    metrics: AnalysisMetrics = field(default_factory=AnalysisMetrics)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    rule_execution_summary: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 100.0
    risk_level: str = "LOW"
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RuleExecutionSummary:
    """Resumen de ejecución de una regla."""
    rule_id: str
    rule_name: str
    category: RuleCategory
    severity: RuleSeverity
    execution_time_ms: float
    violations_found: int
    suggestions_generated: int
    success: bool
    error_message: Optional[str] = None
    cache_hit: bool = False


@dataclass
class PerformanceStats:
    """Estadísticas de performance."""
    total_execution_time_ms: float = 0.0
    average_execution_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    rules_executed: int = 0
    rules_successful: int = 0
    rules_failed: int = 0
    rules_timed_out: int = 0
    parallel_execution_efficiency: float = 0.0


class ResultAggregator:
    """
    Agregador de resultados de ejecución de reglas.
    
    Este agregador procesa los resultados de ejecución de reglas y genera
    reportes consolidados con métricas y recomendaciones.
    """
    
    def __init__(self):
        """Inicializar el agregador de resultados."""
        self.aggregation_rules: Dict[str, Any] = {}
        self.quality_thresholds: Dict[str, float] = {}
        self.risk_levels: Dict[str, str] = {}
        
        # Configuración por defecto
        self._setup_default_configuration()
        
        logger.info("ResultAggregator initialized")
    
    async def initialize(self) -> None:
        """Inicializar el agregador."""
        try:
            # Cargar configuración de agregación
            await self._load_aggregation_configuration()
            
            logger.info("ResultAggregator initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize ResultAggregator: {e}")
    
    async def aggregate_results(self, rule_results: List[RuleResult], 
                              project_config: ProjectConfig) -> AggregatedResults:
        """
        Agregar resultados de ejecución de reglas.
        
        Args:
            rule_results: Lista de resultados de reglas
            project_config: Configuración del proyecto
            
        Returns:
            Resultados agregados
        """
        try:
            # Extraer violaciones y sugerencias
            violations = self._extract_violations(rule_results)
            suggestions = self._extract_suggestions(rule_results)
            
            # Calcular métricas
            metrics = self._calculate_metrics(violations, suggestions, rule_results)
            
            # Generar estadísticas de performance
            performance_stats = self._calculate_performance_stats(rule_results)
            
            # Generar resumen de ejecución de reglas
            rule_summary = self._generate_rule_execution_summary(rule_results)
            
            # Calcular score de calidad
            quality_score = self._calculate_quality_score(metrics, project_config)
            
            # Determinar nivel de riesgo
            risk_level = self._determine_risk_level(metrics, quality_score)
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(metrics, violations, project_config)
            
            return AggregatedResults(
                violations=violations,
                suggestions=suggestions,
                metrics=metrics,
                performance_stats=performance_stats,
                rule_execution_summary=rule_summary,
                quality_score=quality_score,
                risk_level=risk_level,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            raise AggregatorError(f"Result aggregation failed: {e}")
    
    def _extract_violations(self, rule_results: List[RuleResult]) -> List[Violation]:
        """Extraer todas las violaciones de los resultados."""
        violations = []
        
        for result in rule_results:
            if result.success and result.violations:
                violations.extend(result.violations)
        
        # Ordenar por severidad y luego por ubicación
        violations.sort(key=lambda v: (
            self._severity_to_numeric(v.severity),
            v.location.file_path,
            v.location.start_line,
            v.location.start_column
        ), reverse=True)
        
        return violations
    
    def _extract_suggestions(self, rule_results: List[RuleResult]) -> List[Suggestion]:
        """Extraer todas las sugerencias de los resultados."""
        suggestions = []
        
        for result in rule_results:
            if result.success and result.suggestions:
                suggestions.extend(result.suggestions)
        
        # Ordenar por prioridad
        suggestions.sort(key=lambda s: self._priority_to_numeric(s.priority), reverse=True)
        
        return suggestions
    
    def _calculate_metrics(self, violations: List[Violation], suggestions: List[Suggestion],
                          rule_results: List[RuleResult]) -> AnalysisMetrics:
        """Calcular métricas de análisis."""
        metrics = AnalysisMetrics()
        
        # Contar violaciones por severidad
        for violation in violations:
            metrics.total_violations += 1
            
            if violation.severity == RuleSeverity.CRITICAL:
                metrics.critical_violations += 1
            elif violation.severity == RuleSeverity.HIGH:
                metrics.high_violations += 1
            elif violation.severity == RuleSeverity.MEDIUM:
                metrics.medium_violations += 1
            elif violation.severity == RuleSeverity.LOW:
                metrics.low_violations += 1
            elif violation.severity == RuleSeverity.INFO:
                metrics.info_violations += 1
        
        # Contar por categoría
        category_counts = defaultdict(int)
        for violation in violations:
            category_counts[violation.rule_category] += 1
        
        metrics.code_smells = category_counts.get(RuleCategory.CODE_SMELL, 0)
        metrics.bugs = category_counts.get(RuleCategory.BUG_PRONE, 0)
        metrics.vulnerabilities = (
            category_counts.get(RuleCategory.SECURITY, 0) +
            category_counts.get(RuleCategory.VULNERABILITY, 0)
        )
        metrics.security_hotspots = category_counts.get(RuleCategory.SECURITY, 0)
        
        # Calcular deuda técnica (estimación)
        metrics.technical_debt_hours = self._estimate_technical_debt(violations)
        
        # Calcular índices de calidad
        metrics.quality_score = self._calculate_quality_score_from_violations(violations)
        metrics.maintainability_index = self._calculate_maintainability_index(violations)
        
        return metrics
    
    def _calculate_performance_stats(self, rule_results: List[RuleResult]) -> Dict[str, Any]:
        """Calcular estadísticas de performance."""
        stats = PerformanceStats()
        
        if not rule_results:
            return stats.__dict__
        
        # Estadísticas básicas
        stats.rules_executed = len(rule_results)
        stats.rules_successful = sum(1 for r in rule_results if r.success)
        stats.rules_failed = sum(1 for r in rule_results if not r.success)
        
        # Tiempos de ejecución
        execution_times = []
        cache_hits = 0
        
        for result in rule_results:
            if result.metrics:
                execution_time = result.metrics.get('execution_time_ms', 0)
                execution_times.append(execution_time)
                
                if result.metrics.get('cache_hit', False):
                    cache_hits += 1
        
        if execution_times:
            stats.total_execution_time_ms = sum(execution_times)
            stats.average_execution_time_ms = stats.total_execution_time_ms / len(execution_times)
        
        # Tasa de cache hits
        if stats.rules_executed > 0:
            stats.cache_hit_rate = cache_hits / stats.rules_executed
        
        # Eficiencia de ejecución paralela (estimación)
        if stats.total_execution_time_ms > 0:
            # Asumimos que la ejecución secuencial tomaría más tiempo
            estimated_sequential_time = stats.total_execution_time_ms * 1.5
            stats.parallel_execution_efficiency = (
                estimated_sequential_time / stats.total_execution_time_ms
            )
        
        return stats.__dict__
    
    def _generate_rule_execution_summary(self, rule_results: List[RuleResult]) -> Dict[str, Any]:
        """Generar resumen de ejecución de reglas."""
        summary = {
            'total_rules': len(rule_results),
            'successful_rules': 0,
            'failed_rules': 0,
            'rules_by_category': defaultdict(int),
            'rules_by_severity': defaultdict(int),
            'slowest_rules': [],
            'most_violations': []
        }
        
        rule_details = []
        
        for result in rule_results:
            if result.success:
                summary['successful_rules'] += 1
            else:
                summary['failed_rules'] += 1
            
            # Detalles de la regla
            rule_detail = RuleExecutionSummary(
                rule_id=result.rule_id,
                rule_name="",  # Se llenaría con información de la regla
                category=RuleCategory.BEST_PRACTICES,  # Valor por defecto
                severity=RuleSeverity.MEDIUM,  # Valor por defecto
                execution_time_ms=result.metrics.get('execution_time_ms', 0),
                violations_found=len(result.violations),
                suggestions_generated=len(result.suggestions),
                success=result.success,
                error_message=result.error_message,
                cache_hit=result.metrics.get('cache_hit', False)
            )
            
            rule_details.append(rule_detail)
        
        # Ordenar por tiempo de ejecución
        summary['slowest_rules'] = sorted(
            rule_details,
            key=lambda r: r.execution_time_ms,
            reverse=True
        )[:10]
        
        # Ordenar por número de violaciones
        summary['most_violations'] = sorted(
            rule_details,
            key=lambda r: r.violations_found,
            reverse=True
        )[:10]
        
        return summary
    
    def _calculate_quality_score(self, metrics: AnalysisMetrics, 
                               project_config: ProjectConfig) -> float:
        """Calcular score de calidad."""
        base_score = 100.0
        
        # Penalizar por violaciones
        base_score -= metrics.critical_violations * 10
        base_score -= metrics.high_violations * 5
        base_score -= metrics.medium_violations * 2
        base_score -= metrics.low_violations * 1
        base_score -= metrics.info_violations * 0.5
        
        # Penalizar por deuda técnica
        base_score -= metrics.technical_debt_hours * 0.1
        
        # Aplicar umbrales de calidad del proyecto
        quality_gates = project_config.quality_gates
        if metrics.critical_violations > quality_gates.max_critical_violations:
            base_score -= 20
        
        if metrics.high_violations > quality_gates.max_high_violations:
            base_score -= 10
        
        return max(0.0, min(100.0, base_score))
    
    def _determine_risk_level(self, metrics: AnalysisMetrics, quality_score: float) -> str:
        """Determinar nivel de riesgo."""
        if quality_score >= 90 and metrics.critical_violations == 0:
            return "LOW"
        elif quality_score >= 70 and metrics.critical_violations == 0:
            return "MEDIUM"
        elif quality_score >= 50:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _generate_recommendations(self, metrics: AnalysisMetrics, violations: List[Violation],
                                project_config: ProjectConfig) -> List[str]:
        """Generar recomendaciones basadas en los resultados."""
        recommendations = []
        
        # Recomendaciones por severidad
        if metrics.critical_violations > 0:
            recommendations.append(
                f"🔴 CRÍTICO: Resolver {metrics.critical_violations} violación(es) crítica(s) inmediatamente"
            )
        
        if metrics.high_violations > 0:
            recommendations.append(
                f"🟠 ALTO: Revisar y corregir {metrics.high_violations} violación(es) de alta severidad"
            )
        
        # Recomendaciones por categoría
        if metrics.vulnerabilities > 0:
            recommendations.append(
                f"🛡️ SEGURIDAD: Revisar {metrics.vulnerabilities} vulnerabilidad(es) de seguridad"
            )
        
        if metrics.bugs > 0:
            recommendations.append(
                f"🐛 BUGS: Corregir {metrics.bugs} problema(s) propenso(s) a errores"
            )
        
        # Recomendaciones de calidad
        if metrics.technical_debt_hours > 50:
            recommendations.append(
                f"⏰ DEUDA TÉCNICA: {metrics.technical_debt_hours:.1f} horas de deuda técnica acumulada"
            )
        
        if quality_score < 80:
            recommendations.append(
                f"📊 CALIDAD: Score de calidad bajo ({quality_score:.1f}/100). Revisar estándares de código"
            )
        
        # Recomendaciones específicas del proyecto
        if project_config.quality_gates.fail_on_quality_gate:
            if quality_score < project_config.quality_gates.min_quality_score:
                recommendations.append(
                    "🚫 PUERTA DE CALIDAD: El proyecto no cumple con las puertas de calidad configuradas"
                )
        
        return recommendations
    
    def _severity_to_numeric(self, severity: RuleSeverity) -> int:
        """Convertir severidad a valor numérico para ordenamiento."""
        severity_order = {
            RuleSeverity.CRITICAL: 5,
            RuleSeverity.HIGH: 4,
            RuleSeverity.MEDIUM: 3,
            RuleSeverity.LOW: 2,
            RuleSeverity.INFO: 1
        }
        return severity_order.get(severity, 0)
    
    def _priority_to_numeric(self, priority: str) -> int:
        """Convertir prioridad a valor numérico para ordenamiento."""
        priority_order = {
            'high': 3,
            'medium': 2,
            'low': 1
        }
        return priority_order.get(priority.lower(), 0)
    
    def _estimate_technical_debt(self, violations: List[Violation]) -> float:
        """Estimar deuda técnica en horas."""
        total_hours = 0.0
        
        for violation in violations:
            # Estimación basada en severidad y tipo
            if violation.severity == RuleSeverity.CRITICAL:
                total_hours += 4.0  # 4 horas para críticas
            elif violation.severity == RuleSeverity.HIGH:
                total_hours += 2.0  # 2 horas para altas
            elif violation.severity == RuleSeverity.MEDIUM:
                total_hours += 1.0  # 1 hora para medias
            elif violation.severity == RuleSeverity.LOW:
                total_hours += 0.5  # 30 minutos para bajas
            else:
                total_hours += 0.25  # 15 minutos para info
        
        return total_hours
    
    def _calculate_quality_score_from_violations(self, violations: List[Violation]) -> float:
        """Calcular score de calidad basado en violaciones."""
        if not violations:
            return 100.0
        
        # Penalizar por violaciones
        penalty = 0.0
        for violation in violations:
            if violation.severity == RuleSeverity.CRITICAL:
                penalty += 10.0
            elif violation.severity == RuleSeverity.HIGH:
                penalty += 5.0
            elif violation.severity == RuleSeverity.MEDIUM:
                penalty += 2.0
            elif violation.severity == RuleSeverity.LOW:
                penalty += 1.0
            else:
                penalty += 0.5
        
        return max(0.0, 100.0 - penalty)
    
    def _calculate_maintainability_index(self, violations: List[Violation]) -> float:
        """Calcular índice de mantenibilidad."""
        base_index = 100.0
        
        # Penalizar por problemas de mantenibilidad
        maintainability_violations = [
            v for v in violations
            if v.rule_category in [
                RuleCategory.MAINTAINABILITY,
                RuleCategory.READABILITY,
                RuleCategory.CODE_SMELL
            ]
        ]
        
        penalty = len(maintainability_violations) * 2.0
        return max(0.0, base_index - penalty)
    
    def _setup_default_configuration(self) -> None:
        """Configurar valores por defecto."""
        self.quality_thresholds = {
            'excellent': 90.0,
            'good': 80.0,
            'fair': 70.0,
            'poor': 60.0
        }
        
        self.risk_levels = {
            'LOW': 'Bajo riesgo',
            'MEDIUM': 'Riesgo medio',
            'HIGH': 'Alto riesgo',
            'CRITICAL': 'Riesgo crítico'
        }
    
    async def _load_aggregation_configuration(self) -> None:
        """Cargar configuración de agregación."""
        # En una implementación real, cargaría desde archivo o base de datos
        logger.info("Loading aggregation configuration...")
    
    async def shutdown(self) -> None:
        """Apagar el agregador."""
        try:
            # Guardar configuración si es necesario
            await self._save_aggregation_configuration()
            
            logger.info("ResultAggregator shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during aggregator shutdown: {e}")
    
    async def _save_aggregation_configuration(self) -> None:
        """Guardar configuración de agregación."""
        # En una implementación real, guardaría a archivo o base de datos
        logger.info("Saving aggregation configuration...")
