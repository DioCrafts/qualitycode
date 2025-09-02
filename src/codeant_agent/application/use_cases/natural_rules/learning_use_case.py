"""
Módulo que define los casos de uso para el aprendizaje y mejora continua.
"""
from typing import Dict, List, Optional
from uuid import UUID

from codeant_agent.application.dtos.natural_rules.learning_dtos import (
    FeedbackAnalysisDTO, LearningResultDTO, OptimizationResultDTO, RuleFeedbackDTO,
    RuleImprovementDTO
)
from codeant_agent.application.ports.natural_rules.learning_ports import (
    AccuracyMonitorPort, FeedbackCollectorPort, PatternLearnerPort, RuleOptimizerPort
)
from codeant_agent.domain.entities.natural_rules.natural_rule import ExecutableRuleId
from codeant_agent.domain.repositories.natural_rule_repository import (
    ExecutableRuleRepository, RuleFeedbackRepository
)
from codeant_agent.domain.services.natural_rules.learning_service import (
    LearningService, PerformanceTrackingService
)
from codeant_agent.domain.value_objects.natural_rules.rule_metrics import RuleFeedback


class LearningUseCase:
    """Caso de uso para el aprendizaje y mejora continua."""
    
    def __init__(
        self,
        feedback_collector: FeedbackCollectorPort,
        rule_optimizer: RuleOptimizerPort,
        pattern_learner: PatternLearnerPort,
        accuracy_monitor: AccuracyMonitorPort,
        rule_feedback_repository: RuleFeedbackRepository,
        executable_rule_repository: ExecutableRuleRepository,
        learning_service: LearningService,
        performance_tracking_service: PerformanceTrackingService
    ):
        """Inicializa el caso de uso.
        
        Args:
            feedback_collector: Puerto para el recolector de feedback
            rule_optimizer: Puerto para el optimizador de reglas
            pattern_learner: Puerto para el aprendizaje de patrones
            accuracy_monitor: Puerto para el monitor de precisión
            rule_feedback_repository: Repositorio de feedback de reglas
            executable_rule_repository: Repositorio de reglas ejecutables
            learning_service: Servicio de dominio para aprendizaje
            performance_tracking_service: Servicio de dominio para seguimiento de rendimiento
        """
        self.feedback_collector = feedback_collector
        self.rule_optimizer = rule_optimizer
        self.pattern_learner = pattern_learner
        self.accuracy_monitor = accuracy_monitor
        self.rule_feedback_repository = rule_feedback_repository
        self.executable_rule_repository = executable_rule_repository
        self.learning_service = learning_service
        self.performance_tracking_service = performance_tracking_service
    
    async def process_feedback(
        self, feedback_dto: RuleFeedbackDTO
    ) -> LearningResultDTO:
        """Procesa feedback para una regla.
        
        Args:
            feedback_dto: Feedback a procesar
            
        Returns:
            Resultado del aprendizaje
        """
        # Mapear DTO a entidad de dominio
        domain_feedback = RuleFeedback(
            rule_id=feedback_dto.rule_id,
            user_id=feedback_dto.user_id,
            is_positive=feedback_dto.is_positive,
            comments=feedback_dto.comments,
            false_positive=feedback_dto.false_positive,
            false_negative=feedback_dto.false_negative,
            suggested_improvements=feedback_dto.suggested_improvements
        )
        
        # Guardar feedback
        await self.feedback_collector.collect_feedback(
            feedback_dto.rule_id, domain_feedback
        )
        
        # Aprender del feedback
        learning_result = await self.learning_service.learn_from_feedback(
            feedback_dto.rule_id, domain_feedback
        )
        
        # Mapear a DTO
        return LearningResultDTO(
            rule_id=feedback_dto.rule_id,
            feedback_analyzed=True,
            improvements_suggested=len(learning_result.get('improvements', [])),
            improvements_applied=len(learning_result.get('applied', [])),
            new_accuracy=learning_result.get('new_accuracy', 0.0),
            learning_confidence=learning_result.get('confidence', 0.0)
        )
    
    async def optimize_rule(
        self, rule_id: str
    ) -> OptimizationResultDTO:
        """Optimiza una regla.
        
        Args:
            rule_id: ID de la regla a optimizar
            
        Returns:
            Resultado de la optimización
        """
        # Convertir ID a dominio
        domain_id = ExecutableRuleId()
        domain_id.value = UUID(rule_id)
        
        # Optimizar regla
        optimized_rule = await self.rule_optimizer.optimize_rule(domain_id)
        
        if not optimized_rule:
            return OptimizationResultDTO(
                rule_id=UUID(rule_id),
                optimizations_applied=0,
                performance_improvement=0.0,
                accuracy_impact=0.0
            )
        
        # Obtener métricas de rendimiento
        performance_data = await self.performance_tracking_service.get_performance_data(domain_id)
        
        # Calcular mejora de rendimiento (simplificado)
        performance_improvement = 0.0
        if performance_data:
            # Simplificado: comparar último con primero
            if len(performance_data) >= 2:
                first = performance_data[0]
                last = performance_data[-1]
                if first.execution_time_ms > 0:
                    performance_improvement = (
                        (first.execution_time_ms - last.execution_time_ms) / 
                        first.execution_time_ms
                    )
        
        return OptimizationResultDTO(
            rule_id=UUID(rule_id),
            optimizations_applied=1,  # Simplificado
            performance_improvement=performance_improvement,
            accuracy_impact=0.0  # Simplificado
        )
    
    async def analyze_feedback(
        self, rule_id: UUID
    ) -> FeedbackAnalysisDTO:
        """Analiza el feedback para una regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Análisis del feedback
        """
        # Obtener feedback
        feedback_list = await self.rule_feedback_repository.find_feedback_by_rule_id(rule_id)
        
        if not feedback_list:
            return FeedbackAnalysisDTO(
                rule_id=rule_id,
                feedback_count=0,
                positive_ratio=0.0
            )
        
        # Contar feedback positivo y negativo
        positive_count = sum(1 for f in feedback_list if f.is_positive)
        
        # Calcular ratio positivo
        positive_ratio = positive_count / len(feedback_list) if feedback_list else 0.0
        
        # Analizar patrones de feedback (simplificado)
        feedback_analysis = await self.learning_service.analyze_feedback_patterns(rule_id)
        
        return FeedbackAnalysisDTO(
            rule_id=rule_id,
            feedback_count=len(feedback_list),
            positive_ratio=positive_ratio,
            common_issues=feedback_analysis.get('common_issues', []),
            suggested_improvements=feedback_analysis.get('suggested_improvements', []),
            estimated_new_accuracy=feedback_analysis.get('estimated_new_accuracy', 0.0),
            confidence=feedback_analysis.get('confidence', 0.0)
        )
    
    async def get_rule_improvements(
        self, rule_id: UUID
    ) -> List[RuleImprovementDTO]:
        """Obtiene mejoras sugeridas para una regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Lista de mejoras sugeridas
        """
        # Analizar feedback
        feedback_analysis = await self.learning_service.analyze_feedback_patterns(rule_id)
        
        # Generar mejoras
        improvements = await self.learning_service.generate_rule_improvements(
            rule_id, feedback_analysis
        )
        
        # Mapear a DTOs
        result = []
        for i, improvement in enumerate(improvements):
            result.append(RuleImprovementDTO(
                rule_id=rule_id,
                improvement_type=f"improvement_{i}",  # Simplificado
                description=improvement,
                confidence=0.8,  # Simplificado
            ))
        
        return result
