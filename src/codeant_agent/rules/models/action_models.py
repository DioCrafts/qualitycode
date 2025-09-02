"""
Modelos para acciones de reglas del motor de reglas estáticas.

Este módulo define las estructuras de datos para las acciones que se ejecutan
cuando una regla detecta una violación o genera una sugerencia.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from ...parsers.unified.unified_ast import UnifiedAST, UnifiedNode
from .pattern_models import PatternMatch
from .rule_models import Violation, Suggestion, FixSuggestion, ViolationLocation

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Tipos de acciones para reglas."""
    REPORT_VIOLATION = "report_violation"
    GENERATE_SUGGESTION = "generate_suggestion"
    APPLY_FIX = "apply_fix"
    LOG = "log"
    METRIC_UPDATE = "metric_update"
    CUSTOM_ACTION = "custom_action"


@dataclass
class ReportViolationAction:
    """Acción para reportar una violación."""
    message: str
    suggestion: Optional[str] = None
    severity_override: Optional[str] = None
    category_override: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, context: Dict[str, Any]) -> Optional[Violation]:
        """Ejecutar la acción de reportar violación."""
        rule = context.get('rule')
        pattern_match = context.get('pattern_match')
        ast = context.get('ast')
        
        if not rule or not pattern_match or not ast:
            logger.warning("Missing required context for ReportViolationAction")
            return None
        
        # Obtener el primer nodo coincidente para la ubicación
        first_node = pattern_match.get_first_node()
        if not first_node:
            return None
        
        # Crear ubicación de la violación
        location = self._create_violation_location(first_node, ast)
        
        # Crear sugerencias de corrección
        fix_suggestions = []
        if self.suggestion:
            fix_suggestions.append(FixSuggestion(
                description=self.suggestion,
                code_snippet="",  # Se generaría basado en el contexto
                confidence=self.confidence,
                automatic=False
            ))
        
        # Crear la violación
        violation = Violation(
            rule_id=rule.id,
            severity=rule.severity if not self.severity_override else self.severity_override,
            message=self.message,
            location=location,
            rule_category=rule.category,
            confidence=self.confidence,
            fix_suggestions=fix_suggestions,
            metadata=self.metadata
        )
        
        return violation
    
    def _create_violation_location(self, node: UnifiedNode, ast: UnifiedAST) -> ViolationLocation:
        """Crear la ubicación de la violación."""
        if node.position:
            return ViolationLocation(
                file_path=ast.file_path,
                start_line=node.position.start_line,
                start_column=node.position.start_column,
                end_line=node.position.end_line,
                end_column=node.position.end_column,
                context_lines=[],  # Se extraerían del archivo
                node_id=str(node.id)
            )
        else:
            # Ubicación por defecto si no hay posición
            return ViolationLocation(
                file_path=ast.file_path,
                start_line=1,
                start_column=1,
                end_line=1,
                end_column=1,
                context_lines=[],
                node_id=str(node.id)
            )


@dataclass
class GenerateSuggestionAction:
    """Acción para generar una sugerencia."""
    message: str
    category: str = "improvement"
    priority: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, context: Dict[str, Any]) -> Optional[Suggestion]:
        """Ejecutar la acción de generar sugerencia."""
        rule = context.get('rule')
        pattern_match = context.get('pattern_match')
        
        if not rule or not pattern_match:
            logger.warning("Missing required context for GenerateSuggestionAction")
            return None
        
        suggestion = Suggestion(
            rule_id=rule.id,
            message=self.message,
            category=self.category,
            priority=self.priority,
            metadata=self.metadata
        )
        
        return suggestion


@dataclass
class ApplyFixAction:
    """Acción para aplicar una corrección automática."""
    fix_type: str  # replace, insert, delete, refactor
    target: str  # node, parent, children, siblings
    new_code: str
    conditions: List[str] = field(default_factory=list)
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Ejecutar la acción de aplicar corrección."""
        pattern_match = context.get('pattern_match')
        ast = context.get('ast')
        
        if not pattern_match or not ast:
            logger.warning("Missing required context for ApplyFixAction")
            return None
        
        # Verificar condiciones antes de aplicar
        if not self._check_conditions(context):
            return None
        
        # Crear el fix
        fix = {
            'type': self.fix_type,
            'target': self.target,
            'new_code': self.new_code,
            'location': self._get_fix_location(pattern_match),
            'confidence': self.confidence,
            'metadata': self.metadata
        }
        
        return fix
    
    def _check_conditions(self, context: Dict[str, Any]) -> bool:
        """Verificar condiciones para aplicar el fix."""
        # Implementación simplificada
        return True  # En una implementación real, evaluaría las condiciones
    
    def _get_fix_location(self, pattern_match: PatternMatch) -> Dict[str, Any]:
        """Obtener la ubicación para aplicar el fix."""
        first_node = pattern_match.get_first_node()
        if not first_node or not first_node.position:
            return {'type': 'unknown'}
        
        return {
            'type': 'position',
            'start_line': first_node.position.start_line,
            'start_column': first_node.position.start_column,
            'end_line': first_node.position.end_line,
            'end_column': first_node.position.end_column
        }


@dataclass
class LogAction:
    """Acción para registrar información."""
    level: str = "info"  # debug, info, warning, error
    message: str = ""
    include_context: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, context: Dict[str, Any]) -> None:
        """Ejecutar la acción de logging."""
        rule = context.get('rule')
        pattern_match = context.get('pattern_match')
        
        log_message = self.message
        if self.include_context and rule:
            log_message = f"[{rule.id}] {log_message}"
        
        if pattern_match and self.include_context:
            log_message += f" (matched {pattern_match.get_node_count()} nodes)"
        
        # Log según el nivel
        if self.level == "debug":
            logger.debug(log_message, extra=self.metadata)
        elif self.level == "info":
            logger.info(log_message, extra=self.metadata)
        elif self.level == "warning":
            logger.warning(log_message, extra=self.metadata)
        elif self.level == "error":
            logger.error(log_message, extra=self.metadata)


@dataclass
class MetricUpdateAction:
    """Acción para actualizar métricas."""
    metric_name: str
    value: Union[int, float, str]
    operation: str = "set"  # set, increment, decrement, multiply, divide
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar la acción de actualizar métricas."""
        metrics = context.get('metrics', {})
        
        if self.operation == "set":
            metrics[self.metric_name] = self.value
        elif self.operation == "increment":
            current = metrics.get(self.metric_name, 0)
            metrics[self.metric_name] = current + self.value
        elif self.operation == "decrement":
            current = metrics.get(self.metric_name, 0)
            metrics[self.metric_name] = current - self.value
        elif self.operation == "multiply":
            current = metrics.get(self.metric_name, 1)
            metrics[self.metric_name] = current * self.value
        elif self.operation == "divide":
            current = metrics.get(self.metric_name, 1)
            if self.value != 0:
                metrics[self.metric_name] = current / self.value
        
        return {
            'metric_name': self.metric_name,
            'new_value': metrics[self.metric_name],
            'operation': self.operation,
            'metadata': self.metadata
        }


@dataclass
class CustomAction:
    """Acción personalizada."""
    action_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """Ejecutar la acción personalizada."""
        # En una implementación real, esto se conectaría con un sistema de plugins
        # o un registro de acciones personalizadas
        logger.info(f"Executing custom action: {self.action_name}")
        return {
            'action_name': self.action_name,
            'parameters': self.parameters,
            'metadata': self.metadata,
            'result': 'custom_action_executed'
        }


@dataclass
class RuleAction:
    """Acción para una regla."""
    action_type: ActionType
    report_violation: Optional[ReportViolationAction] = None
    generate_suggestion: Optional[GenerateSuggestionAction] = None
    apply_fix: Optional[ApplyFixAction] = None
    log: Optional[LogAction] = None
    metric_update: Optional[MetricUpdateAction] = None
    custom_action: Optional[CustomAction] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validar que solo una acción esté presente."""
        actions = [
            self.report_violation,
            self.generate_suggestion,
            self.apply_fix,
            self.log,
            self.metric_update,
            self.custom_action
        ]
        
        present_actions = [a for a in actions if a is not None]
        if len(present_actions) != 1:
            raise ValueError("Exactly one action type must be specified")
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """Ejecutar la acción."""
        if self.report_violation:
            return self.report_violation.execute(context)
        
        elif self.generate_suggestion:
            return self.generate_suggestion.execute(context)
        
        elif self.apply_fix:
            return self.apply_fix.execute(context)
        
        elif self.log:
            return self.log.execute(context)
        
        elif self.metric_update:
            return self.metric_update.execute(context)
        
        elif self.custom_action:
            return self.custom_action.execute(context)
        
        return None


@dataclass
class ActionGroup:
    """Grupo de acciones con lógica de ejecución."""
    actions: List[RuleAction]
    execution_order: str = "sequential"  # sequential, parallel, conditional
    stop_on_failure: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, context: Dict[str, Any]) -> List[Any]:
        """Ejecutar el grupo de acciones."""
        results = []
        
        if self.execution_order == "sequential":
            for action in self.actions:
                try:
                    result = action.execute(context)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Action execution failed: {e}")
                    if self.stop_on_failure:
                        break
        
        elif self.execution_order == "parallel":
            # En una implementación real, esto usaría asyncio o threading
            for action in self.actions:
                try:
                    result = action.execute(context)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Action execution failed: {e}")
        
        elif self.execution_order == "conditional":
            # Ejecutar solo si la primera acción es exitosa
            if self.actions:
                try:
                    first_result = self.actions[0].execute(context)
                    if first_result is not None:
                        results.append(first_result)
                        
                        # Ejecutar acciones restantes
                        for action in self.actions[1:]:
                            try:
                                result = action.execute(context)
                                if result is not None:
                                    results.append(result)
                            except Exception as e:
                                logger.error(f"Conditional action execution failed: {e}")
                except Exception as e:
                    logger.error(f"First action execution failed: {e}")
        
        return results
