"""
Módulo que define los controladores para la API REST de reglas en lenguaje natural.
"""
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from codeant_agent.application.dtos.natural_rules.nlp_dtos import (
    ProcessTextRequestDTO, ProcessTextResponseDTO
)
from codeant_agent.application.dtos.natural_rules.rule_dtos import (
    CreateNaturalRuleRequestDTO, ExecutableRuleDTO, NaturalRuleResponseDTO,
    RuleStructureDTO, RuleValidationResultDTO
)
from codeant_agent.application.use_cases.natural_rules.process_natural_rule_use_case import (
    ProcessNaturalRuleUseCase
)
from codeant_agent.application.use_cases.natural_rules.rule_generation_use_case import (
    GenerateRuleUseCase
)


router = APIRouter(
    prefix="/api/v1/natural-rules",
    tags=["natural-rules"],
    responses={404: {"description": "Not found"}},
)


def get_process_natural_rule_use_case() -> ProcessNaturalRuleUseCase:
    """Obtiene el caso de uso para procesar reglas en lenguaje natural.
    
    Returns:
        Caso de uso para procesar reglas en lenguaje natural
    """
    # En un sistema real, esto se obtendría mediante inyección de dependencias
    # Esta es una implementación simplificada para el ejemplo
    return None


def get_generate_rule_use_case() -> GenerateRuleUseCase:
    """Obtiene el caso de uso para generar reglas.
    
    Returns:
        Caso de uso para generar reglas
    """
    # En un sistema real, esto se obtendría mediante inyección de dependencias
    # Esta es una implementación simplificada para el ejemplo
    return None


@router.post("/process-text", response_model=ProcessTextResponseDTO)
async def process_text(
    request: ProcessTextRequestDTO,
    use_case: ProcessNaturalRuleUseCase = Depends(get_process_natural_rule_use_case)
):
    """Procesa un texto en lenguaje natural.
    
    Args:
        request: Solicitud de procesamiento
        use_case: Caso de uso para procesar reglas en lenguaje natural
        
    Returns:
        Resultado del procesamiento
    """
    try:
        return await use_case.process_text(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/", response_model=NaturalRuleResponseDTO)
async def create_natural_rule(
    request: CreateNaturalRuleRequestDTO,
    use_case: ProcessNaturalRuleUseCase = Depends(get_process_natural_rule_use_case)
):
    """Crea una regla en lenguaje natural.
    
    Args:
        request: Solicitud de creación
        use_case: Caso de uso para procesar reglas en lenguaje natural
        
    Returns:
        Regla creada
    """
    try:
        return await use_case.create_natural_rule(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{rule_id}", response_model=NaturalRuleResponseDTO)
async def get_natural_rule(
    rule_id: UUID,
    use_case: ProcessNaturalRuleUseCase = Depends(get_process_natural_rule_use_case)
):
    """Obtiene una regla en lenguaje natural por su ID.
    
    Args:
        rule_id: ID de la regla
        use_case: Caso de uso para procesar reglas en lenguaje natural
        
    Returns:
        Regla encontrada
    """
    try:
        rule = await use_case.get_rule_by_id(rule_id)
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rule with ID {rule_id} not found"
            )
        return rule
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/generate", response_model=ExecutableRuleDTO)
async def generate_rule(
    structure: RuleStructureDTO,
    use_case: GenerateRuleUseCase = Depends(get_generate_rule_use_case)
):
    """Genera una regla ejecutable a partir de una estructura.
    
    Args:
        structure: Estructura de la regla
        use_case: Caso de uso para generar reglas
        
    Returns:
        Regla ejecutable generada
    """
    try:
        return await use_case.generate_rule_from_structure(structure)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/validate", response_model=RuleValidationResultDTO)
async def validate_rule(
    rule: ExecutableRuleDTO,
    use_case: GenerateRuleUseCase = Depends(get_generate_rule_use_case)
):
    """Valida una regla ejecutable.
    
    Args:
        rule: Regla a validar
        use_case: Caso de uso para generar reglas
        
    Returns:
        Resultado de la validación
    """
    try:
        return await use_case.validate_rule(rule)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/generate-code")
async def generate_code(
    structure: RuleStructureDTO,
    language: str = Query("python", description="Programming language for the code"),
    use_case: GenerateRuleUseCase = Depends(get_generate_rule_use_case)
):
    """Genera código para una regla.
    
    Args:
        structure: Estructura de la regla
        language: Lenguaje de programación para el código
        use_case: Caso de uso para generar reglas
        
    Returns:
        Código generado
    """
    try:
        code = await use_case.generate_code(structure, language)
        return {"code": code}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
