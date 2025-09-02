"""
Módulo que implementa el buscador de patrones para el sistema de reglas en lenguaje natural.
"""
import re
from typing import Dict, List, Optional

from codeant_agent.application.ports.natural_rules.nlp_ports import PatternMatcherPort
from codeant_agent.domain.entities.natural_rules.language import Language
from codeant_agent.domain.value_objects.natural_rules.nlp_result import PatternMatch


class PatternMatcher(PatternMatcherPort):
    """Implementación del buscador de patrones."""
    
    def __init__(self):
        """Inicializa el buscador de patrones."""
        # Patrones predefinidos para español
        self.spanish_patterns = {
            "prohibicion_funcion": r"las\s+funciones?\s+no\s+debe[n]?\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            "requisito_funcion": r"las\s+funciones?\s+debe[n]?\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            "recomendacion_funcion": r"las\s+funciones?\s+debería[n]?\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            "limite_lineas": r"no\s+más\s+de\s+(\d+)\s+líneas",
            "limite_parametros": r"no\s+más\s+de\s+(\d+)\s+parámetros",
            "limite_complejidad": r"complejidad\s+ciclomática\s+(?:de\s+)?no\s+más\s+de\s+(\d+)",
            "limite_anidamiento": r"no\s+más\s+de\s+(\d+)\s+niveles\s+de\s+anidamiento",
            "contiene_palabra": r"que\s+conteng[a|an]\s+la\s+palabra\s+['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]",
            "no_contiene_palabra": r"que\s+no\s+conteng[a|an]\s+la\s+palabra\s+['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]",
            "si_condicion": r"si\s+([^,]+),\s+entonces\s+([^\.]+)",
            "cuando_condicion": r"cuando\s+([^,]+),\s+([^\.]+)",
        }
        
        # Patrones predefinidos para inglés
        self.english_patterns = {
            "function_prohibition": r"functions?\s+must\s+not\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            "function_requirement": r"functions?\s+must\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            "function_recommendation": r"functions?\s+should\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            "line_limit": r"not\s+more\s+than\s+(\d+)\s+lines",
            "parameter_limit": r"not\s+more\s+than\s+(\d+)\s+parameters",
            "complexity_limit": r"cyclomatic\s+complexity\s+(?:of\s+)?not\s+more\s+than\s+(\d+)",
            "nesting_limit": r"not\s+more\s+than\s+(\d+)\s+levels\s+of\s+nesting",
            "contains_word": r"containing\s+the\s+word\s+['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]",
            "not_contains_word": r"not\s+containing\s+the\s+word\s+['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]",
            "if_condition": r"if\s+([^,]+),\s+then\s+([^\.]+)",
            "when_condition": r"when\s+([^,]+),\s+([^\.]+)",
        }
        
        # Patrones personalizados registrados dinámicamente
        self.custom_patterns = {
            Language.SPANISH: {},
            Language.ENGLISH: {},
        }
    
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
        pattern_matches = []
        
        # Seleccionar patrones predefinidos según el idioma
        predefined_patterns = self.english_patterns
        if language == Language.SPANISH:
            predefined_patterns = self.spanish_patterns
        
        # Buscar coincidencias con patrones predefinidos
        for pattern_name, pattern in predefined_patterns.items():
            pattern_matches.extend(await self._find_pattern_matches(
                text, pattern, pattern_name
            ))
        
        # Buscar coincidencias con patrones personalizados
        custom_patterns = self.custom_patterns[language]
        for pattern_name, pattern in custom_patterns.items():
            pattern_matches.extend(await self._find_pattern_matches(
                text, pattern, pattern_name
            ))
        
        return pattern_matches
    
    async def register_pattern(self, pattern: str, name: str, language: Language) -> bool:
        """Registra un nuevo patrón.
        
        Args:
            pattern: Patrón a registrar
            name: Nombre del patrón
            language: Idioma del patrón
            
        Returns:
            True si el patrón se registró correctamente, False en caso contrario
        """
        try:
            # Verificar que el patrón sea válido
            re.compile(pattern)
            
            # Registrar patrón
            self.custom_patterns[language][name] = pattern
            
            return True
        except re.error:
            # Patrón inválido
            return False
    
    async def _find_pattern_matches(
        self, text: str, pattern: str, pattern_name: str
    ) -> List[PatternMatch]:
        """Busca coincidencias de un patrón en un texto.
        
        Args:
            text: Texto en el que buscar coincidencias
            pattern: Patrón a buscar
            pattern_name: Nombre del patrón
            
        Returns:
            Lista de coincidencias del patrón
        """
        matches = []
        
        try:
            # Buscar coincidencias
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Extraer capturas
                captures = {}
                for i, group in enumerate(match.groups()):
                    if group is not None:
                        captures[f"group_{i+1}"] = group
                
                matches.append(PatternMatch(
                    pattern_name=pattern_name,
                    matched_text=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9,
                    captures=captures
                ))
        except re.error:
            # Ignorar errores en patrones
            pass
        
        return matches
