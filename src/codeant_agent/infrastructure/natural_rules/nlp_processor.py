"""
Módulo que implementa el procesador NLP para el sistema de reglas en lenguaje natural.
"""
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

from codeant_agent.application.ports.natural_rules.nlp_ports import NLPProcessorPort
from codeant_agent.domain.entities.natural_rules.language import Language
from codeant_agent.domain.value_objects.natural_rules.nlp_result import (
    Ambiguity, ExtractedEntity, NLPConfig, NLPProcessingResult, PatternMatch
)


class NLPProcessor(NLPProcessorPort):
    """Implementación del procesador NLP."""
    
    def __init__(self):
        """Inicializa el procesador NLP."""
        # Patrones para preprocesamiento
        self.spanish_patterns = [
            (r"\bque\s+no\s+", "que no "),
            (r"\bque\s+", "que "),
            (r"\bsi\s+", "si "),
            (r"\bcuando\s+", "cuando "),
            (r"\bdonde\s+", "donde "),
            (r"\bfunciones?\s+que\s+", "funciones que "),
            (r"\bclases?\s+que\s+", "clases que "),
            (r"\bvariables?\s+que\s+", "variables que "),
            (r"\bmétodos?\s+que\s+", "métodos que "),
            (r"\bno\s+debe\b", "no debe"),
            (r"\bdebe\b", "debe"),
            (r"\bpuede\b", "puede"),
            (r"\btiene\s+que\b", "tiene que"),
        ]
        
        # Mapeo de términos de programación
        self.spanish_term_mappings = {
            "función": "function",
            "funciones": "functions",
            "clase": "class",
            "clases": "classes",
            "método": "method",
            "métodos": "methods",
            "variable": "variable",
            "variables": "variables",
            "parámetro": "parameter",
            "parámetros": "parameters",
            "bucle": "loop",
            "bucles": "loops",
            "condicional": "conditional",
            "condición": "condition",
            "retorno": "return",
            "excepción": "exception",
            "error": "error",
            "seguridad": "security",
            "rendimiento": "performance",
            "complejidad": "complexity",
        }
        
        # Palabras vacías (stop words)
        self.spanish_stop_words = {
            "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero",
            "porque", "como", "cuando", "donde", "quien", "que", "cual", "cuales",
            "cuyo", "cuya", "cuyos", "cuyas", "este", "esta", "estos", "estas",
            "ese", "esa", "esos", "esas", "aquel", "aquella", "aquellos", "aquellas",
            "para", "por", "con", "sin", "sobre", "bajo", "ante", "contra", "entre",
        }
        
        self.english_stop_words = {
            "the", "a", "an", "and", "or", "but", "because", "as", "when", "where",
            "who", "which", "that", "this", "these", "those", "for", "by", "with",
            "without", "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "from", "up", "down", "in", "out",
            "on", "off", "over", "under", "again", "further", "then", "once", "here",
        }
    
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
        start_time = time.time()
        
        # Usar configuración por defecto si no se proporciona
        if config is None:
            config = NLPConfig()
        
        # Preprocesar texto
        preprocessed_text = await self.preprocess_text(text, language)
        
        # Extraer entidades
        entities = []
        if config.enable_entity_extraction:
            entities = await self._extract_entities(preprocessed_text, language)
        
        # Buscar patrones
        pattern_matches = []
        if config.enable_pattern_matching:
            pattern_matches = await self._find_patterns(preprocessed_text, language)
        
        # Detectar ambigüedades
        ambiguities = []
        if config.enable_ambiguity_detection:
            ambiguities = await self._detect_ambiguities(preprocessed_text)
        
        # Calcular tiempo de procesamiento
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Calcular puntuación de confianza
        confidence_score = self._calculate_confidence(
            preprocessed_text, entities, pattern_matches, ambiguities
        )
        
        return NLPProcessingResult(
            preprocessed_text=preprocessed_text,
            entities=entities,
            pattern_matches=pattern_matches,
            ambiguities=ambiguities,
            processing_time_ms=processing_time_ms,
            confidence_score=confidence_score
        )
    
    async def preprocess_text(self, text: str, language: Language) -> str:
        """Preprocesa un texto para su análisis.
        
        Args:
            text: Texto a preprocesar
            language: Idioma del texto
            
        Returns:
            Texto preprocesado
        """
        # Normalizar texto (minúsculas)
        processed = text.lower()
        
        # Aplicar preprocesamiento específico del idioma
        if language == Language.SPANISH:
            processed = await self._preprocess_spanish_text(processed)
        
        # Eliminar palabras vacías pero preservar términos de programación importantes
        processed = await self._remove_stop_words(processed, language)
        
        # Normalizar terminología de programación
        processed = await self._normalize_programming_terms(processed, language)
        
        return processed
    
    async def _preprocess_spanish_text(self, text: str) -> str:
        """Preprocesa texto en español.
        
        Args:
            text: Texto a preprocesar
            
        Returns:
            Texto preprocesado
        """
        processed = text
        
        # Aplicar patrones específicos del español
        for pattern, replacement in self.spanish_patterns:
            processed = re.sub(pattern, replacement, processed)
        
        return processed
    
    async def _remove_stop_words(self, text: str, language: Language) -> str:
        """Elimina palabras vacías del texto.
        
        Args:
            text: Texto del que eliminar palabras vacías
            language: Idioma del texto
            
        Returns:
            Texto sin palabras vacías
        """
        # Seleccionar conjunto de palabras vacías según el idioma
        stop_words = self.english_stop_words
        if language == Language.SPANISH:
            stop_words = self.spanish_stop_words
        
        # Dividir en palabras
        words = text.split()
        
        # Filtrar palabras vacías
        filtered_words = [word for word in words if word not in stop_words]
        
        # Reconstruir texto
        return ' '.join(filtered_words)
    
    async def _normalize_programming_terms(self, text: str, language: Language) -> str:
        """Normaliza términos de programación en el texto.
        
        Args:
            text: Texto a normalizar
            language: Idioma del texto
            
        Returns:
            Texto con términos normalizados
        """
        normalized = text
        
        # Aplicar mapeo de términos según el idioma
        if language == Language.SPANISH:
            for term, normalized_term in self.spanish_term_mappings.items():
                pattern = r"\b{}\b".format(re.escape(term))
                normalized = re.sub(pattern, normalized_term, normalized)
        
        return normalized
    
    async def _extract_entities(
        self, text: str, language: Language
    ) -> List[ExtractedEntity]:
        """Extrae entidades del texto.
        
        Args:
            text: Texto del que extraer entidades
            language: Idioma del texto
            
        Returns:
            Lista de entidades extraídas
        """
        entities = []
        
        # Patrones para extraer entidades comunes en reglas de código
        patterns = [
            (r"\bfunction\w*\b", "FUNCTION"),
            (r"\bclass\w*\b", "CLASS"),
            (r"\bmethod\w*\b", "METHOD"),
            (r"\bvariable\w*\b", "VARIABLE"),
            (r"\bparameter\w*\b", "PARAMETER"),
            (r"\bloop\w*\b", "LOOP"),
            (r"\bcondition\w*\b", "CONDITION"),
            (r"\b\d+\b", "NUMBER"),
            (r"\bsecurity\b", "DOMAIN"),
            (r"\bperformance\b", "DOMAIN"),
            (r"\bcomplexity\b", "DOMAIN"),
        ]
        
        # Buscar coincidencias de patrones
        for pattern, entity_type in patterns:
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8,
                    metadata={}
                ))
        
        return entities
    
    async def _find_patterns(
        self, text: str, language: Language
    ) -> List[PatternMatch]:
        """Busca patrones en el texto.
        
        Args:
            text: Texto en el que buscar patrones
            language: Idioma del texto
            
        Returns:
            Lista de coincidencias de patrones
        """
        pattern_matches = []
        
        # Patrones comunes en reglas de código
        patterns = [
            (r"(\w+)\s+must\s+not\s+(\w+)", "PROHIBITION"),
            (r"(\w+)\s+must\s+(\w+)", "REQUIREMENT"),
            (r"(\w+)\s+should\s+(\w+)", "RECOMMENDATION"),
            (r"more\s+than\s+(\d+)", "THRESHOLD_GT"),
            (r"less\s+than\s+(\d+)", "THRESHOLD_LT"),
            (r"equal\s+to\s+(\d+)", "THRESHOLD_EQ"),
            (r"(\w+)\s+no\s+debe\s+(\w+)", "PROHIBITION"),
            (r"(\w+)\s+debe\s+(\w+)", "REQUIREMENT"),
            (r"(\w+)\s+debería\s+(\w+)", "RECOMMENDATION"),
            (r"más\s+de\s+(\d+)", "THRESHOLD_GT"),
            (r"menos\s+de\s+(\d+)", "THRESHOLD_LT"),
            (r"igual\s+a\s+(\d+)", "THRESHOLD_EQ"),
        ]
        
        # Buscar coincidencias de patrones
        for pattern, pattern_name in patterns:
            for match in re.finditer(pattern, text):
                # Extraer capturas
                captures = {}
                for i, group in enumerate(match.groups()):
                    captures[f"group_{i+1}"] = group
                
                pattern_matches.append(PatternMatch(
                    pattern_name=pattern_name,
                    matched_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8,
                    captures=captures
                ))
        
        return pattern_matches
    
    async def _detect_ambiguities(self, text: str) -> List[Ambiguity]:
        """Detecta ambigüedades en el texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Lista de ambigüedades detectadas
        """
        ambiguities = []
        
        # Patrones de ambigüedad comunes
        ambiguity_patterns = [
            (r"\bit\b", "Referencia ambigua 'it'", ["podría referirse a varios elementos"]),
            (r"\bthey\b", "Referencia ambigua 'they'", ["podría referirse a varios elementos"]),
            (r"\bthis\b", "Referencia ambigua 'this'", ["podría referirse a varios elementos"]),
            (r"\besto\b", "Referencia ambigua 'esto'", ["podría referirse a varios elementos"]),
            (r"\bello\b", "Referencia ambigua 'ello'", ["podría referirse a varios elementos"]),
        ]
        
        # Buscar coincidencias de patrones de ambigüedad
        for pattern, description, interpretations in ambiguity_patterns:
            for match in re.finditer(pattern, text):
                ambiguities.append(Ambiguity(
                    description=description,
                    ambiguous_text=match.group(),
                    possible_interpretations=interpretations,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    severity=0.5
                ))
        
        return ambiguities
    
    def _calculate_confidence(
        self, text: str, entities: List[ExtractedEntity],
        pattern_matches: List[PatternMatch], ambiguities: List[Ambiguity]
    ) -> float:
        """Calcula la puntuación de confianza para el procesamiento.
        
        Args:
            text: Texto procesado
            entities: Entidades extraídas
            pattern_matches: Coincidencias de patrones
            ambiguities: Ambigüedades detectadas
            
        Returns:
            Puntuación de confianza (0.0-1.0)
        """
        # Base de confianza
        confidence = 0.7
        
        # Ajustar según entidades encontradas
        if entities:
            confidence += min(0.1, len(entities) * 0.02)
        
        # Ajustar según patrones encontrados
        if pattern_matches:
            confidence += min(0.1, len(pattern_matches) * 0.02)
        
        # Penalizar por ambigüedades
        if ambiguities:
            confidence -= min(0.3, len(ambiguities) * 0.05)
        
        # Asegurar rango válido
        return max(0.0, min(1.0, confidence))
