"""
Módulo que define los casos de uso para la generación de reglas ejecutables.
"""
from typing import Dict, List, Optional
from uuid import UUID

from codeant_agent.application.dtos.natural_rules.rule_dtos import (
    ExecutableRuleDTO, RuleStructureDTO, RuleValidationResultDTO
)
from codeant_agent.application.ports.natural_rules.rule_generation_ports import (
    CodeGeneratorPort, RuleGeneratorPort, RuleValidatorPort
)
from codeant_agent.domain.entities.natural_rules.natural_rule import (
    ExecutableRule, ExecutableRuleId, RuleStructure, RuleValidationResult
)
from codeant_agent.domain.repositories.natural_rule_repository import ExecutableRuleRepository
from codeant_agent.domain.services.natural_rules.rule_generation_service import (
    CodeGenerationService, RuleGenerationService, RuleTemplateService
)


class GenerateRuleUseCase:
    """Caso de uso para generar reglas ejecutables."""
    
    def __init__(
        self,
        rule_generator: RuleGeneratorPort,
        code_generator: CodeGeneratorPort,
        rule_validator: RuleValidatorPort,
        executable_rule_repository: ExecutableRuleRepository,
        rule_generation_service: RuleGenerationService,
        code_generation_service: CodeGenerationService,
        rule_template_service: RuleTemplateService
    ):
        """Inicializa el caso de uso.
        
        Args:
            rule_generator: Puerto para el generador de reglas
            code_generator: Puerto para el generador de código
            rule_validator: Puerto para el validador de reglas
            executable_rule_repository: Repositorio de reglas ejecutables
            rule_generation_service: Servicio de dominio para generación de reglas
            code_generation_service: Servicio de dominio para generación de código
            rule_template_service: Servicio de dominio para plantillas de reglas
        """
        self.rule_generator = rule_generator
        self.code_generator = code_generator
        self.rule_validator = rule_validator
        self.executable_rule_repository = executable_rule_repository
        self.rule_generation_service = rule_generation_service
        self.code_generation_service = code_generation_service
        self.rule_template_service = rule_template_service
    
    async def generate_rule_from_structure(
        self, structure: RuleStructureDTO, context: Dict[str, str] = None
    ) -> ExecutableRuleDTO:
        """Genera una regla ejecutable a partir de una estructura.
        
        Args:
            structure: Estructura de la regla
            context: Contexto adicional para la generación
            
        Returns:
            Regla ejecutable generada
        """
        # Mapear DTO a entidad de dominio
        domain_structure = self._map_structure_from_dto(structure)
        
        # Generar regla ejecutable
        executable_rule = await self.rule_generator.generate_rule(
            domain_structure, context
        )
        
        # Guardar regla ejecutable
        saved_rule = await self.executable_rule_repository.save(executable_rule)
        
        # Mapear a DTO
        return self._map_executable_rule_to_dto(saved_rule)
    
    async def validate_rule(
        self, rule_dto: ExecutableRuleDTO
    ) -> RuleValidationResultDTO:
        """Valida una regla ejecutable.
        
        Args:
            rule_dto: Regla a validar
            
        Returns:
            Resultado de la validación
        """
        # Mapear DTO a entidad de dominio
        domain_rule = self._map_executable_rule_from_dto(rule_dto)
        
        # Validar regla
        validation_result = await self.rule_validator.validate_rule(domain_rule)
        
        # Mapear a DTO
        return RuleValidationResultDTO(
            is_valid=validation_result.is_valid,
            errors=validation_result.errors,
            warnings=validation_result.warnings
        )
    
    async def get_rule_by_id(
        self, rule_id: str
    ) -> Optional[ExecutableRuleDTO]:
        """Obtiene una regla por su ID.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Regla encontrada o None si no existe
        """
        domain_id = ExecutableRuleId()
        domain_id.value = UUID(rule_id)
        
        rule = await self.executable_rule_repository.find_by_id(domain_id)
        if not rule:
            return None
        
        return self._map_executable_rule_to_dto(rule)
    
    async def generate_code(
        self, structure: RuleStructureDTO, language: str = "python"
    ) -> str:
        """Genera código para una regla.
        
        Args:
            structure: Estructura de la regla
            language: Lenguaje de programación para el código
            
        Returns:
            Código generado
        """
        # Mapear DTO a entidad de dominio
        domain_structure = self._map_structure_from_dto(structure)
        
        # Generar código
        code = await self.code_generator.generate_code(domain_structure, language)
        
        return code
    
    async def validate_code(
        self, code: str, language: str
    ) -> List[str]:
        """Valida el código de una regla.
        
        Args:
            code: Código a validar
            language: Lenguaje de programación del código
            
        Returns:
            Lista de errores de validación (vacía si es válido)
        """
        return await self.rule_validator.validate_code(code, language)
    
    def _map_structure_from_dto(self, dto: RuleStructureDTO) -> RuleStructure:
        """Mapea un DTO de estructura a una entidad de dominio.
        
        Args:
            dto: DTO a mapear
            
        Returns:
            Entidad de dominio
        """
        # Implementación simplificada para el ejemplo
        return RuleStructure(
            intent_analysis=IntentAnalysis(primary_intent="UNKNOWN"),
            description=dto.description
        )
    
    def _map_executable_rule_to_dto(self, rule: ExecutableRule) -> ExecutableRuleDTO:
        """Mapea una entidad de regla ejecutable a su DTO.
        
        Args:
            rule: Entidad a mapear
            
        Returns:
            DTO de la regla
        """
        return ExecutableRuleDTO(
            id=str(rule.id),
            rule_name=rule.rule_name,
            description=rule.description,
            implementation={
                'code': rule.implementation.code,
                'language': rule.implementation.language,
                'parameters': rule.implementation.parameters
            },
            languages=rule.languages,
            category=str(rule.category),
            severity=str(rule.severity),
            configuration=rule.configuration,
            metadata=rule.metadata
        )
    
    def _map_executable_rule_from_dto(self, dto: ExecutableRuleDTO) -> ExecutableRule:
        """Mapea un DTO de regla ejecutable a una entidad de dominio.
        
        Args:
            dto: DTO a mapear
            
        Returns:
            Entidad de dominio
        """
        # Implementación simplificada para el ejemplo
        rule_id = ExecutableRuleId()
        rule_id.value = UUID(dto.id) if dto.id else None
        
        from codeant_agent.domain.entities.natural_rules.rule_intent import (
            ActionSeverity, RuleDomain
        )
        
        return ExecutableRule(
            id=rule_id,
            rule_name=dto.rule_name,
            description=dto.description,
            implementation=None,  # Simplificado para el ejemplo
            languages=dto.languages,
            category=RuleDomain.BEST_PRACTICES,  # Simplificado
            severity=ActionSeverity.WARNING,  # Simplificado
            configuration=dto.configuration,
            metadata=dto.metadata
        )
