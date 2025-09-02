"""
Módulo que define los controladores para la API REST de aprendizaje y mejora continua.
"""
from typing import Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from codeant_agent.application.dtos.natural_rules.learning_dtos import (
    FeedbackAnalysisDTO, LearningResultDTO, OptimizationResultDTO,
    RuleFeedbackDTO, RuleImprovementDTO
)
from codeant_agent.application.use_cases.natural_rules.learning_use_case import (
    LearningUseCase
)


router = APIRouter(
    prefix="/api/v1/natural-rules/learning",
    tags=["natural-rules-learning"],
    responses={404: {"description": "Not found"}},
)


def get_learning_use_case() -> LearningUseCase:
    """Obtiene el caso de uso para aprendizaje y mejora continua.
    
    Returns:
        Caso de uso para aprendizaje y mejora continua
    """
    # En un sistema real, esto se obtendría mediante inyección de dependencias
    # Esta es una implementación simplificada para el ejemplo
    return None


@router.post("/feedback", response_model=LearningResultDTO)
async def process_feedback(
    feedback: RuleFeedbackDTO,
    use_case: LearningUseCase = Depends(get_learning_use_case)
):
    """Procesa feedback para una regla.
    
    Args:
        feedback: Feedback a procesar
        use_case: Caso de uso para aprendizaje y mejora continua
        
    Returns:
        Resultado del aprendizaje
    """
    try:
        return await use_case.process_feedback(feedback)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/optimize/{rule_id}", response_model=OptimizationResultDTO)
async def optimize_rule(
    rule_id: str,
    use_case: LearningUseCase = Depends(get_learning_use_case)
):
    """Optimiza una regla.
    
    Args:
        rule_id: ID de la regla a optimizar
        use_case: Caso de uso para aprendizaje y mejora continua
        
    Returns:
        Resultado de la optimización
    """
    try:
        return await use_case.optimize_rule(rule_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/analyze-feedback/{rule_id}", response_model=FeedbackAnalysisDTO)
async def analyze_feedback(
    rule_id: UUID,
    use_case: LearningUseCase = Depends(get_learning_use_case)
):
    """Analiza el feedback para una regla.
    
    Args:
        rule_id: ID de la regla
        use_case: Caso de uso para aprendizaje y mejora continua
        
    Returns:
        Análisis del feedback
    """
    try:
        return await use_case.analyze_feedback(rule_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/improvements/{rule_id}", response_model=List[RuleImprovementDTO])
async def get_rule_improvements(
    rule_id: UUID,
    use_case: LearningUseCase = Depends(get_learning_use_case)
):
    """Obtiene mejoras sugeridas para una regla.
    
    Args:
        rule_id: ID de la regla
        use_case: Caso de uso para aprendizaje y mejora continua
        
    Returns:
        Lista de mejoras sugeridas
    """
    try:
        return await use_case.get_rule_improvements(rule_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
