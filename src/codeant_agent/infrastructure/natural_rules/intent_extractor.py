"""
Módulo que implementa el extractor de intenciones para el sistema de reglas en lenguaje natural.
"""
import re
from typing import Dict, List, Optional, Tuple

from codeant_agent.application.ports.natural_rules.nlp_ports import IntentExtractorPort
from codeant_agent.domain.entities.natural_rules.language import Language
from codeant_agent.domain.entities.natural_rules.natural_rule import IntentAnalysis, RuleCondition, RuleAction
from codeant_agent.domain.entities.natural_rules.rule_intent import (
    ActionSeverity, ActionType, ConditionType, RuleDomain, RuleIntent, RuleType
)


class IntentExtractor(IntentExtractorPort):
    """Implementación del extractor de intenciones."""
    
    def __init__(self):
        """Inicializa el extractor de intenciones."""
        # Patrones de intención para español
        self.spanish_intent_patterns = [
            (r"no\s+debe[n]?\b", RuleIntent.PROHIBIT),
            (r"debe[n]?\b", RuleIntent.REQUIRE),
            (r"debería[n]?\b", RuleIntent.RECOMMEND),
            (r"no\s+más\s+de\b", RuleIntent.LIMIT),
            (r"asegurar\b", RuleIntent.ENSURE),
            (r"validar\b", RuleIntent.VALIDATE),
            (r"verificar\b", RuleIntent.CHECK),
            (r"contar\b", RuleIntent.COUNT),
            (r"medir\b", RuleIntent.MEASURE),
            (r"detectar\b", RuleIntent.DETECT),
        ]
        
        # Patrones de intención para inglés
        self.english_intent_patterns = [
            (r"must\s+not\b", RuleIntent.PROHIBIT),
            (r"must\b", RuleIntent.REQUIRE),
            (r"should\b", RuleIntent.RECOMMEND),
            (r"not\s+more\s+than\b", RuleIntent.LIMIT),
            (r"ensure\b", RuleIntent.ENSURE),
            (r"validate\b", RuleIntent.VALIDATE),
            (r"check\b", RuleIntent.CHECK),
            (r"count\b", RuleIntent.COUNT),
            (r"measure\b", RuleIntent.MEASURE),
            (r"detect\b", RuleIntent.DETECT),
        ]
        
        # Patrones de dominio
        self.domain_patterns = [
            (r"seguridad|security", RuleDomain.SECURITY),
            (r"rendimiento|performance", RuleDomain.PERFORMANCE),
            (r"manteni\w+|maintainab\w+", RuleDomain.MAINTAINABILITY),
            (r"práctica\w+|practice\w+", RuleDomain.BEST_PRACTICES),
            (r"nombr\w+|naming", RuleDomain.NAMING),
            (r"estructur\w+|structur\w+", RuleDomain.STRUCTURE),
            (r"complej\w+|complex\w+", RuleDomain.COMPLEXITY),
            (r"document\w+", RuleDomain.DOCUMENTATION),
            (r"prueba\w+|test\w+", RuleDomain.TESTING),
            (r"arquitect\w+|architect\w+", RuleDomain.ARCHITECTURE),
        ]
        
        # Patrones de tipo de regla
        self.rule_type_patterns = [
            (r"límit\w+|limit\w+|no\s+más\s+de|not\s+more\s+than", RuleType.CONSTRAINT),
            (r"calidad|quality", RuleType.QUALITY),
            (r"patrón\w*|pattern\w*", RuleType.PATTERN),
            (r"métric\w+|metric\w+", RuleType.METRIC),
            (r"valid\w+", RuleType.VALIDATION),
            (r"transform\w+", RuleType.TRANSFORMATION),
            (r"detect\w+", RuleType.DETECTION),
        ]
        
        # Patrones de condición para español
        self.spanish_condition_patterns = [
            (r"si\s+(.+?)\s+entonces", ConditionType.IF),
            (r"cuando\s+(.+)", ConditionType.WHEN),
            (r"que\s+(.+)", ConditionType.THAT),
            (r"con\s+más\s+de\s+(\d+)", ConditionType.GREATER_THAN),
            (r"con\s+menos\s+de\s+(\d+)", ConditionType.LESS_THAN),
            (r"que\s+contenga\s+(.+)", ConditionType.CONTAINS),
            (r"que\s+no\s+contenga\s+(.+)", ConditionType.NOT_CONTAINS),
            (r"en\s+(.+)", ConditionType.IN_LOCATION),
        ]
        
        # Patrones de condición para inglés
        self.english_condition_patterns = [
            (r"if\s+(.+?)\s+then", ConditionType.IF),
            (r"when\s+(.+)", ConditionType.WHEN),
            (r"that\s+(.+)", ConditionType.THAT),
            (r"with\s+more\s+than\s+(\d+)", ConditionType.GREATER_THAN),
            (r"with\s+less\s+than\s+(\d+)", ConditionType.LESS_THAN),
            (r"containing\s+(.+)", ConditionType.CONTAINS),
            (r"not\s+containing\s+(.+)", ConditionType.NOT_CONTAINS),
            (r"in\s+(.+)", ConditionType.IN_LOCATION),
        ]
        
        # Patrones de acción para español
        self.spanish_action_patterns = [
            (r"debe\s+ser\s+(.+)", ActionType.MUST_BE),
            (r"no\s+debe\s+(.+)", ActionType.MUST_NOT_BE),
            (r"debería\s+(.+)", ActionType.SHOULD),
            (r"reportar\s+(.+)", ActionType.REPORT),
            (r"sugerir\s+(.+)", ActionType.SUGGEST),
            (r"avisar\s+(.+)", ActionType.WARN),
            (r"fallar\s+(.+)", ActionType.FAIL),
            (r"refactorizar", ActionType.REFACTOR),
            (r"corregir", ActionType.FIX),
        ]
        
        # Patrones de acción para inglés
        self.english_action_patterns = [
            (r"must\s+be\s+(.+)", ActionType.MUST_BE),
            (r"must\s+not\s+(.+)", ActionType.MUST_NOT_BE),
            (r"should\s+(.+)", ActionType.SHOULD),
            (r"report\s+(.+)", ActionType.REPORT),
            (r"suggest\s+(.+)", ActionType.SUGGEST),
            (r"warn\s+(.+)", ActionType.WARN),
            (r"fail\s+(.+)", ActionType.FAIL),
            (r"refactor", ActionType.REFACTOR),
            (r"fix", ActionType.FIX),
        ]
    
    async def extract_intent(self, text: str, language: Language) -> IntentAnalysis:
        """Extrae la intención de un texto.
        
        Args:
            text: Texto del que extraer la intención
            language: Idioma del texto
            
        Returns:
            Análisis de intención
        """
        # Determinar intención primaria
        primary_intent, secondary_intents = await self._classify_intent(text, language)
        
        # Clasificar dominio
        domain = await self.classify_domain(text, language)
        
        # Determinar tipo de regla
        rule_type = await self._determine_rule_type(text, language)
        
        # Extraer condiciones
        conditions = await self._extract_conditions(text, language)
        
        # Extraer acciones
        actions = await self._extract_actions(text, language)
        
        # Calcular confianza
        confidence = self._calculate_intent_confidence(primary_intent, conditions, actions)
        
        return IntentAnalysis(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            domain=domain,
            rule_type=rule_type,
            conditions=conditions,
            actions=actions,
            confidence=confidence
        )
    
    async def classify_domain(self, text: str, language: Language) -> str:
        """Clasifica el dominio de un texto.
        
        Args:
            text: Texto a clasificar
            language: Idioma del texto
            
        Returns:
            Dominio clasificado
        """
        # Buscar coincidencias con patrones de dominio
        for pattern, domain in self.domain_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return domain
        
        # Dominio por defecto si no hay coincidencias
        return RuleDomain.BEST_PRACTICES
    
    async def _classify_intent(
        self, text: str, language: Language
    ) -> Tuple[RuleIntent, List[RuleIntent]]:
        """Clasifica la intención primaria y secundarias de un texto.
        
        Args:
            text: Texto a clasificar
            language: Idioma del texto
            
        Returns:
            Tupla con intención primaria y lista de intenciones secundarias
        """
        # Seleccionar patrones según el idioma
        patterns = self.english_intent_patterns
        if language == Language.SPANISH:
            patterns = self.spanish_intent_patterns
        
        # Buscar coincidencias con patrones de intención
        matches = []
        for pattern, intent in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append((intent, text.index(pattern)))
        
        # Ordenar por posición en el texto (las primeras tienen más prioridad)
        matches.sort(key=lambda x: x[1])
        
        # Extraer intenciones
        intents = [intent for intent, _ in matches]
        
        # Determinar intención primaria y secundarias
        primary_intent = RuleIntent.UNKNOWN
        secondary_intents = []
        
        if intents:
            primary_intent = intents[0]
            secondary_intents = intents[1:]
        
        return primary_intent, secondary_intents
    
    async def _determine_rule_type(self, text: str, language: Language) -> RuleType:
        """Determina el tipo de regla a partir del texto.
        
        Args:
            text: Texto a analizar
            language: Idioma del texto
            
        Returns:
            Tipo de regla
        """
        # Buscar coincidencias con patrones de tipo de regla
        for pattern, rule_type in self.rule_type_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return rule_type
        
        # Tipo por defecto si no hay coincidencias
        return RuleType.CONSTRAINT
    
    async def _extract_conditions(
        self, text: str, language: Language
    ) -> List[RuleCondition]:
        """Extrae condiciones del texto.
        
        Args:
            text: Texto del que extraer condiciones
            language: Idioma del texto
            
        Returns:
            Lista de condiciones extraídas
        """
        conditions = []
        
        # Seleccionar patrones según el idioma
        patterns = self.english_condition_patterns
        if language == Language.SPANISH:
            patterns = self.spanish_condition_patterns
        
        # Buscar coincidencias con patrones de condición
        for pattern_str, condition_type in patterns:
            for match in re.finditer(pattern_str, text, re.IGNORECASE):
                if match.groups():
                    condition_text = match.group(1)
                    parameters = await self._extract_condition_parameters(condition_text)
                    
                    conditions.append(RuleCondition(
                        condition_type=condition_type,
                        condition_text=condition_text,
                        parameters=parameters,
                        confidence=0.8
                    ))
        
        return conditions
    
    async def _extract_actions(
        self, text: str, language: Language
    ) -> List[RuleAction]:
        """Extrae acciones del texto.
        
        Args:
            text: Texto del que extraer acciones
            language: Idioma del texto
            
        Returns:
            Lista de acciones extraídas
        """
        actions = []
        
        # Seleccionar patrones según el idioma
        patterns = self.english_action_patterns
        if language == Language.SPANISH:
            patterns = self.spanish_action_patterns
        
        # Buscar coincidencias con patrones de acción
        for pattern_str, action_type in patterns:
            for match in re.finditer(pattern_str, text, re.IGNORECASE):
                action_text = ""
                if match.groups():
                    action_text = match.group(1)
                
                parameters = await self._extract_action_parameters(action_text)
                severity = self._infer_action_severity(action_type, action_text)
                
                actions.append(RuleAction(
                    action_type=action_type,
                    action_text=action_text,
                    parameters=parameters,
                    severity=severity
                ))
        
        return actions
    
    async def _extract_condition_parameters(self, condition_text: str) -> Dict[str, str]:
        """Extrae parámetros de una condición.
        
        Args:
            condition_text: Texto de la condición
            
        Returns:
            Diccionario con parámetros extraídos
        """
        parameters = {}
        
        # Extraer números
        number_match = re.search(r'\b(\d+)\b', condition_text)
        if number_match:
            parameters['value'] = number_match.group(1)
        
        # Extraer elementos de código
        code_element_match = re.search(r'\b(function|method|class|variable|parameter|loop)\b', condition_text)
        if code_element_match:
            parameters['element_type'] = code_element_match.group(1)
        
        return parameters
    
    async def _extract_action_parameters(self, action_text: str) -> Dict[str, str]:
        """Extrae parámetros de una acción.
        
        Args:
            action_text: Texto de la acción
            
        Returns:
            Diccionario con parámetros extraídos
        """
        parameters = {}
        
        # Extraer elementos de código
        code_element_match = re.search(r'\b(function|method|class|variable|parameter|loop)\b', action_text)
        if code_element_match:
            parameters['element_type'] = code_element_match.group(1)
        
        return parameters
    
    def _infer_action_severity(self, action_type: ActionType, action_text: str) -> ActionSeverity:
        """Infiere la severidad de una acción.
        
        Args:
            action_type: Tipo de acción
            action_text: Texto de la acción
            
        Returns:
            Severidad inferida
        """
        # Asignar severidad según el tipo de acción
        if action_type in [ActionType.MUST_BE, ActionType.MUST_NOT_BE, ActionType.FAIL]:
            return ActionSeverity.ERROR
        elif action_type in [ActionType.SHOULD, ActionType.WARN]:
            return ActionSeverity.WARNING
        elif action_type in [ActionType.SUGGEST, ActionType.REFACTOR, ActionType.FIX]:
            return ActionSeverity.INFO
        else:
            return ActionSeverity.WARNING
    
    def _calculate_intent_confidence(
        self, primary_intent: RuleIntent,
        conditions: List[RuleCondition], actions: List[RuleAction]
    ) -> float:
        """Calcula la confianza en el análisis de intención.
        
        Args:
            primary_intent: Intención primaria
            conditions: Condiciones extraídas
            actions: Acciones extraídas
            
        Returns:
            Puntuación de confianza (0.0-1.0)
        """
        # Base de confianza
        confidence = 0.5
        
        # Ajustar según intención
        if primary_intent != RuleIntent.UNKNOWN:
            confidence += 0.2
        
        # Ajustar según condiciones
        if conditions:
            confidence += min(0.15, len(conditions) * 0.05)
        
        # Ajustar según acciones
        if actions:
            confidence += min(0.15, len(actions) * 0.05)
        
        # Asegurar rango válido
        return max(0.0, min(1.0, confidence))
