"""
Módulo que define los casos de uso para el procesamiento de reglas en lenguaje natural.
"""
from dataclasses import asdict
from typing import Dict, Optional
from uuid import UUID

from codeant_agent.application.dtos.natural_rules.nlp_dtos import (
    ProcessTextRequestDTO, ProcessTextResponseDTO
)
from codeant_agent.application.dtos.natural_rules.rule_dtos import (
    CreateNaturalRuleRequestDTO, NaturalRuleResponseDTO
)
from codeant_agent.application.ports.natural_rules.nlp_ports import (
    EntityExtractorPort, IntentExtractorPort, NLPProcessorPort, PatternMatcherPort
)
from codeant_agent.application.ports.natural_rules.rule_generation_ports import (
    CodeGeneratorPort, RuleGeneratorPort, RuleValidatorPort
)
from codeant_agent.domain.entities.natural_rules.language import Language
from codeant_agent.domain.entities.natural_rules.natural_rule import (
    IntentAnalysis, NaturalRule, RuleStructure
)
from codeant_agent.domain.repositories.natural_rule_repository import NaturalRuleRepository
from codeant_agent.domain.services.natural_rules.nlp_service import NLPService
from codeant_agent.domain.services.natural_rules.rule_generation_service import RuleGenerationService
from codeant_agent.domain.value_objects.natural_rules.nlp_result import NLPConfig


class ProcessNaturalRuleUseCase:
    """Caso de uso para procesar reglas en lenguaje natural."""
    
    def __init__(
        self,
        nlp_processor: NLPProcessorPort,
        intent_extractor: IntentExtractorPort,
        entity_extractor: EntityExtractorPort,
        pattern_matcher: PatternMatcherPort,
        rule_generator: RuleGeneratorPort,
        code_generator: CodeGeneratorPort,
        rule_validator: RuleValidatorPort,
        natural_rule_repository: NaturalRuleRepository,
        nlp_service: NLPService,
        rule_generation_service: RuleGenerationService
    ):
        """Inicializa el caso de uso.
        
        Args:
            nlp_processor: Puerto para el procesador NLP
            intent_extractor: Puerto para el extractor de intenciones
            entity_extractor: Puerto para el extractor de entidades
            pattern_matcher: Puerto para el buscador de patrones
            rule_generator: Puerto para el generador de reglas
            code_generator: Puerto para el generador de código
            rule_validator: Puerto para el validador de reglas
            natural_rule_repository: Repositorio de reglas naturales
            nlp_service: Servicio de dominio para NLP
            rule_generation_service: Servicio de dominio para generación de reglas
        """
        self.nlp_processor = nlp_processor
        self.intent_extractor = intent_extractor
        self.entity_extractor = entity_extractor
        self.pattern_matcher = pattern_matcher
        self.rule_generator = rule_generator
        self.code_generator = code_generator
        self.rule_validator = rule_validator
        self.natural_rule_repository = natural_rule_repository
        self.nlp_service = nlp_service
        self.rule_generation_service = rule_generation_service
    
    async def process_text(
        self, request: ProcessTextRequestDTO
    ) -> ProcessTextResponseDTO:
        """Procesa un texto en lenguaje natural.
        
        Args:
            request: Solicitud de procesamiento
            
        Returns:
            Resultado del procesamiento
        """
        language = Language.from_string(request.language)
        
        config = NLPConfig(
            enable_entity_extraction=request.enable_entity_extraction,
            enable_intent_classification=request.enable_intent_classification,
            enable_pattern_matching=request.enable_pattern_matching,
            enable_ambiguity_detection=request.enable_ambiguity_detection
        )
        
        nlp_result = await self.nlp_processor.process_text(
            request.text, language, config
        )
        
        # Mapear el resultado a DTO
        response = ProcessTextResponseDTO(
            preprocessed_text=nlp_result.preprocessed_text,
            processing_time_ms=nlp_result.processing_time_ms,
            confidence_score=nlp_result.confidence_score
        )
        
        # Mapear entidades
        for entity in nlp_result.entities:
            response.entities.append({
                'text': entity.text,
                'entity_type': entity.entity_type,
                'start_pos': entity.start_pos,
                'end_pos': entity.end_pos,
                'confidence': entity.confidence,
                'metadata': entity.metadata
            })
        
        # Mapear coincidencias de patrones
        for match in nlp_result.pattern_matches:
            response.pattern_matches.append({
                'pattern_name': match.pattern_name,
                'matched_text': match.matched_text,
                'start_pos': match.start_pos,
                'end_pos': match.end_pos,
                'confidence': match.confidence,
                'captures': match.captures
            })
        
        # Mapear ambigüedades
        for ambiguity in nlp_result.ambiguities:
            response.ambiguities.append({
                'description': ambiguity.description,
                'ambiguous_text': ambiguity.ambiguous_text,
                'possible_interpretations': ambiguity.possible_interpretations,
                'start_pos': ambiguity.start_pos,
                'end_pos': ambiguity.end_pos,
                'severity': ambiguity.severity
            })
        
        return response
    
    async def create_natural_rule(
        self, request: CreateNaturalRuleRequestDTO
    ) -> NaturalRuleResponseDTO:
        """Crea una regla en lenguaje natural.
        
        Args:
            request: Solicitud de creación
            
        Returns:
            Regla creada
        """
        language = Language.from_string(request.language)
        
        # Crear regla natural inicial
        natural_rule = NaturalRule(
            original_text=request.text,
            language=language
        )
        
        # Procesar texto
        preprocessed_text = await self.nlp_service.preprocess_text(
            request.text, language
        )
        natural_rule.preprocessed_text = preprocessed_text
        
        # Extraer intención
        intent_analysis = await self.intent_extractor.extract_intent(
            preprocessed_text, language
        )
        natural_rule.intent_analysis = intent_analysis
        
        # Extraer entidades
        entities = await self.entity_extractor.extract_entities(
            preprocessed_text, language
        )
        
        # Buscar patrones
        pattern_matches = await self.pattern_matcher.find_patterns(
            preprocessed_text, language
        )
        
        # Generar estructura de regla
        rule_structure = await self.nlp_service.generate_rule_structure(
            intent_analysis, entities, pattern_matches
        )
        natural_rule.rule_structure = rule_structure
        
        # Generar regla ejecutable
        try:
            executable_rule = await self.rule_generator.generate_rule(
                rule_structure, request.context
            )
            natural_rule.executable_rule = executable_rule
            
            # Validar regla
            validation_result = await self.rule_validator.validate_rule(
                executable_rule
            )
            natural_rule.validation_result = validation_result
            
        except Exception as e:
            # En caso de error, guardar la regla sin ejecutable
            pass
        
        # Guardar regla
        saved_rule = await self.natural_rule_repository.save(natural_rule)
        
        # Mapear a DTO
        return self._map_to_dto(saved_rule)
    
    async def get_rule_by_id(self, rule_id: UUID) -> Optional[NaturalRuleResponseDTO]:
        """Obtiene una regla por su ID.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Regla encontrada o None si no existe
        """
        rule = await self.natural_rule_repository.find_by_id(rule_id)
        if not rule:
            return None
        
        return self._map_to_dto(rule)
    
    def _map_to_dto(self, rule: NaturalRule) -> NaturalRuleResponseDTO:
        """Mapea una regla a su DTO.
        
        Args:
            rule: Regla a mapear
            
        Returns:
            DTO de la regla
        """
        intent_analysis = {}
        if rule.intent_analysis:
            intent_analysis = {
                'primary_intent': str(rule.intent_analysis.primary_intent),
                'domain': str(rule.intent_analysis.domain),
                'rule_type': str(rule.intent_analysis.rule_type),
                'confidence': rule.intent_analysis.confidence
            }
        
        rule_structure = None
        if rule.rule_structure:
            rule_structure = {
                'description': rule.rule_structure.description
            }
        
        executable_rule = None
        if rule.executable_rule:
            executable_rule = {
                'id': str(rule.executable_rule.id),
                'rule_name': rule.executable_rule.rule_name,
                'description': rule.executable_rule.description
            }
        
        return NaturalRuleResponseDTO(
            id=rule.id,
            original_text=rule.original_text,
            language=str(rule.language),
            preprocessed_text=rule.preprocessed_text,
            intent_analysis=intent_analysis,
            rule_structure=rule_structure,
            executable_rule=executable_rule,
            is_valid=rule.is_valid,
            confidence_score=rule.confidence_score,
            generation_time_ms=rule.generation_time_ms,
            created_at=rule.created_at.isoformat()
        )
