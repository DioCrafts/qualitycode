"""
Módulo que define los puertos de la capa de aplicación para el procesamiento NLP.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from codeant_agent.domain.entities.natural_rules.language import Language
from codeant_agent.domain.entities.natural_rules.natural_rule import IntentAnalysis
from codeant_agent.domain.value_objects.natural_rules.nlp_result import (
    ExtractedEntity, NLPConfig, NLPProcessingResult, PatternMatch
)


class NLPProcessorPort(ABC):
    """Puerto para el procesador de lenguaje natural."""
    
    @abstractmethod
    async def process_text(
        self, text: str, language: Language, config: Optional[NLPConfig] = None
    ) -> NLPProcessingResult:
        """Procesa un texto en lenguaje natural.
        
        Args:
            text: Texto a procesar
            language: Idioma del texto
            config: Configuración opcional para el procesamiento
            
        Returns:
            Resultado del procesamiento NLP
        """
        pass
    
    @abstractmethod
    async def preprocess_text(self, text: str, language: Language) -> str:
        """Preprocesa un texto para su análisis.
        
        Args:
            text: Texto a preprocesar
            language: Idioma del texto
            
        Returns:
            Texto preprocesado
        """
        pass


class IntentExtractorPort(ABC):
    """Puerto para el extractor de intenciones."""
    
    @abstractmethod
    async def extract_intent(self, text: str, language: Language) -> IntentAnalysis:
        """Extrae la intención de un texto.
        
        Args:
            text: Texto del que extraer la intención
            language: Idioma del texto
            
        Returns:
            Análisis de intención
        """
        pass
    
    @abstractmethod
    async def classify_domain(self, text: str, language: Language) -> str:
        """Clasifica el dominio de un texto.
        
        Args:
            text: Texto a clasificar
            language: Idioma del texto
            
        Returns:
            Dominio clasificado
        """
        pass


class EntityExtractorPort(ABC):
    """Puerto para el extractor de entidades."""
    
    @abstractmethod
    async def extract_entities(
        self, text: str, language: Language
    ) -> List[ExtractedEntity]:
        """Extrae entidades de un texto.
        
        Args:
            text: Texto del que extraer entidades
            language: Idioma del texto
            
        Returns:
            Lista de entidades extraídas
        """
        pass
    
    @abstractmethod
    async def extract_code_elements(
        self, text: str, language: Language
    ) -> Dict[str, List[str]]:
        """Extrae elementos de código de un texto.
        
        Args:
            text: Texto del que extraer elementos de código
            language: Idioma del texto
            
        Returns:
            Diccionario con elementos de código extraídos por tipo
        """
        pass


class PatternMatcherPort(ABC):
    """Puerto para el buscador de patrones."""
    
    @abstractmethod
    async def find_patterns(
        self, text: str, language: Language
    ) -> List[PatternMatch]:
        """Busca patrones en un texto.
        
        Args:
            text: Texto en el que buscar patrones
            language: Idioma del texto
            
        Returns:
            Lista de coincidencias de patrones
        """
        pass
    
    @abstractmethod
    async def register_pattern(self, pattern: str, name: str, language: Language) -> bool:
        """Registra un nuevo patrón.
        
        Args:
            pattern: Patrón a registrar
            name: Nombre del patrón
            language: Idioma del patrón
            
        Returns:
            True si el patrón se registró correctamente, False en caso contrario
        """
        pass
