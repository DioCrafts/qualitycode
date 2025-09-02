"""
Módulo que implementa el monitor de precisión para el sistema de reglas en lenguaje natural.
"""
from typing import Dict, List, Optional
from uuid import UUID

from codeant_agent.application.ports.natural_rules.learning_ports import AccuracyMonitorPort
from codeant_agent.domain.value_objects.natural_rules.rule_metrics import RuleAccuracyMetrics


class AccuracyMonitor(AccuracyMonitorPort):
    """Implementación del monitor de precisión."""
    
    def __init__(self, feedback_repository):
        """Inicializa el monitor de precisión.
        
        Args:
            feedback_repository: Repositorio de feedback
        """
        self.feedback_repository = feedback_repository
        
        # Métricas de precisión por regla
        self.accuracy_metrics = {}
    
    async def track_accuracy(
        self, rule_id: UUID, true_positives: int, false_positives: int,
        true_negatives: int, false_negatives: int
    ) -> None:
        """Registra métricas de precisión para una regla.
        
        Args:
            rule_id: ID de la regla
            true_positives: Número de verdaderos positivos
            false_positives: Número de falsos positivos
            true_negatives: Número de verdaderos negativos
            false_negatives: Número de falsos negativos
        """
        # Crear métricas de precisión
        metrics = RuleAccuracyMetrics(
            rule_id=rule_id,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives
        )
        
        # Guardar métricas
        self.accuracy_metrics[str(rule_id)] = metrics
        
        # En un sistema real, se guardarían en el repositorio
        # await self.feedback_repository.save_accuracy_metrics(metrics)
    
    async def get_accuracy_metrics(
        self, rule_id: UUID
    ) -> Dict[str, float]:
        """Obtiene métricas de precisión para una regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Diccionario con métricas de precisión
        """
        # Obtener métricas de precisión
        metrics = self.accuracy_metrics.get(str(rule_id))
        
        if not metrics:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
            }
        
        # Calcular métricas adicionales
        accuracy = (metrics.true_positives + metrics.true_negatives) / (
            metrics.true_positives + metrics.true_negatives +
            metrics.false_positives + metrics.false_negatives
        ) if (metrics.true_positives + metrics.true_negatives +
              metrics.false_positives + metrics.false_negatives) > 0 else 0.0
        
        return {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "accuracy": accuracy,
        }
    
    async def analyze_accuracy_trends(
        self, rule_id: UUID, window_size: int = 10
    ) -> Dict[str, List[float]]:
        """Analiza tendencias de precisión para una regla.
        
        Args:
            rule_id: ID de la regla
            window_size: Tamaño de la ventana para el análisis
            
        Returns:
            Diccionario con tendencias de precisión
        """
        # En un sistema real, se obtendrían métricas históricas del repositorio
        # y se calcularían tendencias
        
        # Implementación simplificada para el ejemplo
        return {
            "precision_trend": [0.8, 0.82, 0.85],
            "recall_trend": [0.75, 0.78, 0.8],
            "f1_score_trend": [0.77, 0.8, 0.82],
        }
    
    async def detect_accuracy_issues(
        self, rule_id: UUID, threshold: float = 0.8
    ) -> List[str]:
        """Detecta problemas de precisión para una regla.
        
        Args:
            rule_id: ID de la regla
            threshold: Umbral para detectar problemas
            
        Returns:
            Lista de problemas detectados
        """
        issues = []
        
        # Obtener métricas de precisión
        metrics = await self.get_accuracy_metrics(rule_id)
        
        # Detectar problemas
        if metrics["precision"] < threshold:
            issues.append(f"Low precision: {metrics['precision']:.2f}")
        
        if metrics["recall"] < threshold:
            issues.append(f"Low recall: {metrics['recall']:.2f}")
        
        if metrics["f1_score"] < threshold:
            issues.append(f"Low F1 score: {metrics['f1_score']:.2f}")
        
        return issues
