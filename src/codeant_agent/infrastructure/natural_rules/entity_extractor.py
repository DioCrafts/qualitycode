"""
Módulo que implementa el extractor de entidades para el sistema de reglas en lenguaje natural.
"""
import re
from typing import Dict, List

from codeant_agent.application.ports.natural_rules.nlp_ports import EntityExtractorPort
from codeant_agent.domain.entities.natural_rules.language import Language
from codeant_agent.domain.value_objects.natural_rules.nlp_result import ExtractedEntity


class EntityExtractor(EntityExtractorPort):
    """Implementación del extractor de entidades."""
    
    def __init__(self):
        """Inicializa el extractor de entidades."""
        # Patrones para extraer entidades de código
        self.code_element_patterns = [
            (r"\bfunción\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", "FUNCTION_NAME"),
            (r"\bfunction\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", "FUNCTION_NAME"),
            (r"\bclase\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", "CLASS_NAME"),
            (r"\bclass\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", "CLASS_NAME"),
            (r"\bmétodo\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", "METHOD_NAME"),
            (r"\bmethod\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", "METHOD_NAME"),
            (r"\bvariable\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", "VARIABLE_NAME"),
            (r"\bparameter\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", "PARAMETER_NAME"),
            (r"\bparámetro\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", "PARAMETER_NAME"),
        ]
        
        # Patrones para extraer tipos de elementos
        self.element_type_patterns = [
            (r"\bfunción\b|\bfunction\b", "FUNCTION"),
            (r"\bclase\b|\bclass\b", "CLASS"),
            (r"\bmétodo\b|\bmethod\b", "METHOD"),
            (r"\bvariable\b", "VARIABLE"),
            (r"\bparámetro\b|\bparameter\b", "PARAMETER"),
            (r"\bbucle\b|\bloop\b", "LOOP"),
            (r"\bcondicional\b|\bconditional\b", "CONDITIONAL"),
            (r"\bexpresión\b|\bexpression\b", "EXPRESSION"),
            (r"\bdeclaración\b|\bstatement\b", "STATEMENT"),
            (r"\barchivo\b|\bfile\b", "FILE"),
            (r"\bmódulo\b|\bmodule\b", "MODULE"),
        ]
        
        # Patrones para extraer valores numéricos
        self.numeric_patterns = [
            (r"\b(\d+)\s+líneas\b", "LINE_COUNT"),
            (r"\b(\d+)\s+lines\b", "LINE_COUNT"),
            (r"\b(\d+)\s+caracteres\b", "CHAR_COUNT"),
            (r"\b(\d+)\s+characters\b", "CHAR_COUNT"),
            (r"\b(\d+)\s+parámetros\b", "PARAMETER_COUNT"),
            (r"\b(\d+)\s+parameters\b", "PARAMETER_COUNT"),
            (r"\b(\d+)\s+métodos\b", "METHOD_COUNT"),
            (r"\b(\d+)\s+methods\b", "METHOD_COUNT"),
            (r"\b(\d+)\s+funciones\b", "FUNCTION_COUNT"),
            (r"\b(\d+)\s+functions\b", "FUNCTION_COUNT"),
            (r"\b(\d+)\s+niveles\b", "NESTING_LEVEL"),
            (r"\b(\d+)\s+levels\b", "NESTING_LEVEL"),
            (r"\b(\d+)%\b", "PERCENTAGE"),
            (r"\bmás\s+de\s+(\d+)\b", "THRESHOLD_GT"),
            (r"\bmore\s+than\s+(\d+)\b", "THRESHOLD_GT"),
            (r"\bmenos\s+de\s+(\d+)\b", "THRESHOLD_LT"),
            (r"\bless\s+than\s+(\d+)\b", "THRESHOLD_LT"),
            (r"\bigual\s+a\s+(\d+)\b", "THRESHOLD_EQ"),
            (r"\bequal\s+to\s+(\d+)\b", "THRESHOLD_EQ"),
        ]
        
        # Patrones para extraer dominios
        self.domain_patterns = [
            (r"\bseguridad\b|\bsecurity\b", "DOMAIN_SECURITY"),
            (r"\brendimiento\b|\bperformance\b", "DOMAIN_PERFORMANCE"),
            (r"\bmantenibilidad\b|\bmaintainability\b", "DOMAIN_MAINTAINABILITY"),
            (r"\bmejores\s+prácticas\b|\bbest\s+practices\b", "DOMAIN_BEST_PRACTICES"),
            (r"\bnomenclatura\b|\bnaming\b", "DOMAIN_NAMING"),
            (r"\bestructura\b|\bstructure\b", "DOMAIN_STRUCTURE"),
            (r"\bcomplejidad\b|\bcomplexity\b", "DOMAIN_COMPLEXITY"),
            (r"\bdocumentación\b|\bdocumentation\b", "DOMAIN_DOCUMENTATION"),
            (r"\bpruebas\b|\btesting\b", "DOMAIN_TESTING"),
            (r"\barquitectura\b|\barchitecture\b", "DOMAIN_ARCHITECTURE"),
        ]
        
        # Patrones para extraer acciones
        self.action_patterns = [
            (r"\brefactorizar\b|\brefactor\b", "ACTION_REFACTOR"),
            (r"\bcorregir\b|\bfix\b", "ACTION_FIX"),
            (r"\breportar\b|\breport\b", "ACTION_REPORT"),
            (r"\bavisar\b|\bwarn\b", "ACTION_WARN"),
            (r"\bsugerir\b|\bsuggest\b", "ACTION_SUGGEST"),
            (r"\bfallar\b|\bfail\b", "ACTION_FAIL"),
        ]
    
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
        entities = []
        
        # Extraer nombres de elementos de código
        entities.extend(await self._extract_code_element_names(text))
        
        # Extraer tipos de elementos
        entities.extend(await self._extract_element_types(text))
        
        # Extraer valores numéricos
        entities.extend(await self._extract_numeric_values(text))
        
        # Extraer dominios
        entities.extend(await self._extract_domains(text))
        
        # Extraer acciones
        entities.extend(await self._extract_actions(text))
        
        return entities
    
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
        result = {}
        
        # Extraer nombres de elementos de código
        code_element_entities = await self._extract_code_element_names(text)
        
        # Agrupar por tipo
        for entity in code_element_entities:
            entity_type = entity.entity_type
            if entity_type not in result:
                result[entity_type] = []
            result[entity_type].append(entity.text)
        
        # Extraer tipos de elementos
        element_type_entities = await self._extract_element_types(text)
        
        # Agrupar por tipo
        for entity in element_type_entities:
            entity_type = entity.entity_type
            if entity_type not in result:
                result[entity_type] = []
            result[entity_type].append(entity.text)
        
        return result
    
    async def _extract_code_element_names(self, text: str) -> List[ExtractedEntity]:
        """Extrae nombres de elementos de código de un texto.
        
        Args:
            text: Texto del que extraer nombres de elementos de código
            
        Returns:
            Lista de entidades extraídas
        """
        entities = []
        
        # Buscar coincidencias con patrones de nombres de elementos de código
        for pattern, entity_type in self.code_element_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.groups():
                    element_name = match.group(1)
                    entities.append(ExtractedEntity(
                        text=element_name,
                        entity_type=entity_type,
                        start_pos=match.start(1),
                        end_pos=match.end(1),
                        confidence=0.9,
                        metadata={'full_match': match.group(0)}
                    ))
        
        return entities
    
    async def _extract_element_types(self, text: str) -> List[ExtractedEntity]:
        """Extrae tipos de elementos de un texto.
        
        Args:
            text: Texto del que extraer tipos de elementos
            
        Returns:
            Lista de entidades extraídas
        """
        entities = []
        
        # Buscar coincidencias con patrones de tipos de elementos
        for pattern, entity_type in self.element_type_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(0),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9,
                    metadata={}
                ))
        
        return entities
    
    async def _extract_numeric_values(self, text: str) -> List[ExtractedEntity]:
        """Extrae valores numéricos de un texto.
        
        Args:
            text: Texto del que extraer valores numéricos
            
        Returns:
            Lista de entidades extraídas
        """
        entities = []
        
        # Buscar coincidencias con patrones numéricos
        for pattern, entity_type in self.numeric_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.groups():
                    value = match.group(1)
                    entities.append(ExtractedEntity(
                        text=value,
                        entity_type=entity_type,
                        start_pos=match.start(1),
                        end_pos=match.end(1),
                        confidence=0.9,
                        metadata={'full_match': match.group(0)}
                    ))
        
        return entities
    
    async def _extract_domains(self, text: str) -> List[ExtractedEntity]:
        """Extrae dominios de un texto.
        
        Args:
            text: Texto del que extraer dominios
            
        Returns:
            Lista de entidades extraídas
        """
        entities = []
        
        # Buscar coincidencias con patrones de dominios
        for pattern, entity_type in self.domain_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(0),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8,
                    metadata={}
                ))
        
        return entities
    
    async def _extract_actions(self, text: str) -> List[ExtractedEntity]:
        """Extrae acciones de un texto.
        
        Args:
            text: Texto del que extraer acciones
            
        Returns:
            Lista de entidades extraídas
        """
        entities = []
        
        # Buscar coincidencias con patrones de acciones
        for pattern, entity_type in self.action_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(0),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8,
                    metadata={}
                ))
        
        return entities
