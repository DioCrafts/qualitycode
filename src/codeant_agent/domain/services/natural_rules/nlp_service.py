"""
Módulo que define los servicios de dominio para el procesamiento de lenguaje natural.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from codeant_agent.domain.entities.natural_rules.language import Language
from codeant_agent.domain.entities.natural_rules.natural_rule import (
    IntentAnalysis, NaturalRule, RuleStructure
)
from codeant_agent.domain.value_objects.natural_rules.nlp_result import (
    ExtractedEntity, NLPConfig, NLPProcessingResult, PatternMatch
)


class NLPService(ABC):
    """Servicio de dominio para el procesamiento de lenguaje natural."""
    
    @abstractmethod
    async def preprocess_text(self, text: str, language: Language) -> str:
        """Preprocesa el texto para su análisis.
        
        Args:
            text: Texto a preprocesar
            language: Idioma del texto
            
        Returns:
            Texto preprocesado
        """
        pass
    
    @abstractmethod
    async def process_rule_text(
        self, text: str, language: Language, config: Optional[NLPConfig] = None
    ) -> NLPProcessingResult:
        """Procesa el texto de una regla.
        
        Args:
            text: Texto de la regla
            language: Idioma del texto
            config: Configuración opcional para el procesamiento
            
        Returns:
            Resultado del procesamiento NLP
        """
        pass
    
    @abstractmethod
    async def extract_entities(
        self, text: str, language: Language
    ) -> List[ExtractedEntity]:
        """Extrae entidades del texto.
        
        Args:
            text: Texto del que extraer entidades
            language: Idioma del texto
            
        Returns:
            Lista de entidades extraídas
        """
        pass
    
    @abstractmethod
    async def find_patterns(
        self, text: str, language: Language
    ) -> List[PatternMatch]:
        """Busca patrones en el texto.
        
        Args:
            text: Texto en el que buscar patrones
            language: Idioma del texto
            
        Returns:
            Lista de coincidencias de patrones
        """
        pass
    
    @abstractmethod
    async def detect_ambiguities(
        self, text: str, intent_analysis: IntentAnalysis, entities: List[ExtractedEntity]
    ) -> List[str]:
        """Detecta ambigüedades en el texto.
        
        Args:
            text: Texto a analizar
            intent_analysis: Análisis de intención
            entities: Entidades extraídas
            
        Returns:
            Lista de ambigüedades detectadas
        """
        pass


class IntentClassificationService(ABC):
    """Servicio de dominio para la clasificación de intenciones."""
    
    @abstractmethod
    async def classify_intent(self, text: str, language: Language) -> IntentAnalysis:
        """Clasifica la intención del texto.
        
        Args:
            text: Texto a clasificar
            language: Idioma del texto
            
        Returns:
            Análisis de intención
        """
        pass
    
    @abstractmethod
    async def extract_conditions(
        self, text: str, language: Language
    ) -> List[dict]:
        """Extrae condiciones del texto.
        
        Args:
            text: Texto del que extraer condiciones
            language: Idioma del texto
            
        Returns:
            Lista de condiciones extraídas
        """
        pass
    
    @abstractmethod
    async def extract_actions(
        self, text: str, language: Language
    ) -> List[dict]:
        """Extrae acciones del texto.
        
        Args:
            text: Texto del que extraer acciones
            language: Idioma del texto
            
        Returns:
            Lista de acciones extraídas
        """
        pass


class RuleStructureService(ABC):
    """Servicio de dominio para la estructuración de reglas."""
    
    @abstractmethod
    async def generate_rule_structure(
        self, 
        intent_analysis: IntentAnalysis,
        entities: List[ExtractedEntity],
        pattern_matches: List[PatternMatch]
    ) -> RuleStructure:
        """Genera la estructura de una regla.
        
        Args:
            intent_analysis: Análisis de intención
            entities: Entidades extraídas
            pattern_matches: Coincidencias de patrones
            
        Returns:
            Estructura de la regla
        """
        pass
    
    @abstractmethod
    async def validate_rule_structure(self, structure: RuleStructure) -> List[str]:
        """Valida la estructura de una regla.
        
        Args:
            structure: Estructura de la regla a validar
            
        Returns:
            Lista de errores de validación (vacía si es válida)
        """
        pass
