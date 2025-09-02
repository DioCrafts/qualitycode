"""
Implementación del Adaptador de Idioma.

Este módulo implementa la adaptación de contenido entre diferentes idiomas,
incluyendo traducción y adaptación cultural.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from uuid import uuid4

from ...domain.entities.explanation import Language
from ...domain.value_objects.explanation_metrics import LanguageDetectionResult
from ..ports.explanation_ports import LanguageAdapterPort


logger = logging.getLogger(__name__)


class LanguageAdapter(LanguageAdapterPort):
    """Adaptador para manejo de múltiples idiomas."""
    
    def __init__(self):
        self.translation_cache = {}
        self.technical_terms_dictionary = self._initialize_technical_terms()
        self.cultural_adaptations = self._initialize_cultural_adaptations()
        self.language_patterns = self._initialize_language_patterns()
    
    def _initialize_technical_terms(self) -> Dict[str, Dict[str, str]]:
        """Inicializar diccionario de términos técnicos."""
        return {
            "es": {
                "cyclomatic_complexity": "complejidad ciclomática",
                "code_smell": "código con mal olor",
                "technical_debt": "deuda técnica",
                "refactoring": "refactorización",
                "unit_test": "prueba unitaria",
                "integration_test": "prueba de integración",
                "code_coverage": "cobertura de código",
                "static_analysis": "análisis estático",
                "dependency_injection": "inyección de dependencias",
                "design_pattern": "patrón de diseño",
                "antipattern": "antipatrón",
                "maintainability": "mantenibilidad",
                "readability": "legibilidad",
                "performance": "rendimiento",
                "security_vulnerability": "vulnerabilidad de seguridad",
                "bug": "error",
                "feature": "característica",
                "function": "función",
                "class": "clase",
                "method": "método",
                "variable": "variable",
                "algorithm": "algoritmo",
                "data_structure": "estructura de datos",
                "api": "API",
                "database": "base de datos",
                "framework": "framework",
                "library": "librería",
                "module": "módulo",
                "component": "componente",
                "interface": "interfaz",
                "implementation": "implementación"
            },
            "en": {
                "complejidad_ciclomática": "cyclomatic complexity",
                "código_con_mal_olor": "code smell",
                "deuda_técnica": "technical debt",
                "refactorización": "refactoring",
                "prueba_unitaria": "unit test",
                "prueba_de_integración": "integration test",
                "cobertura_de_código": "code coverage",
                "análisis_estático": "static analysis",
                "inyección_de_dependencias": "dependency injection",
                "patrón_de_diseño": "design pattern",
                "antipatrón": "antipattern",
                "mantenibilidad": "maintainability",
                "legibilidad": "readability",
                "rendimiento": "performance",
                "vulnerabilidad_de_seguridad": "security vulnerability",
                "error": "bug",
                "característica": "feature",
                "función": "function",
                "clase": "class",
                "método": "method",
                "variable": "variable",
                "algoritmo": "algorithm",
                "estructura_de_datos": "data structure",
                "API": "API",
                "base_de_datos": "database",
                "framework": "framework",
                "librería": "library",
                "módulo": "module",
                "componente": "component",
                "interfaz": "interface",
                "implementación": "implementation"
            }
        }
    
    def _initialize_cultural_adaptations(self) -> Dict[str, Dict[str, str]]:
        """Inicializar adaptaciones culturales."""
        return {
            "es": {
                "formal_greeting": "Estimado/a",
                "informal_greeting": "Hola",
                "formal_closing": "Saludos cordiales",
                "informal_closing": "¡Hasta luego!",
                "emphasis_high": "muy importante",
                "emphasis_medium": "importante",
                "emphasis_low": "recomendado",
                "time_reference": "tiempo",
                "cost_reference": "costo",
                "quality_reference": "calidad"
            },
            "en": {
                "formal_greeting": "Dear",
                "informal_greeting": "Hello",
                "formal_closing": "Best regards",
                "informal_closing": "See you later!",
                "emphasis_high": "very important",
                "emphasis_medium": "important",
                "emphasis_low": "recommended",
                "time_reference": "time",
                "cost_reference": "cost",
                "quality_reference": "quality"
            }
        }
    
    def _initialize_language_patterns(self) -> Dict[str, Dict[str, str]]:
        """Inicializar patrones de idioma para detección."""
        return {
            "es": {
                "common_words": ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo", "le", "da", "su", "por", "son", "con", "para", "al", "del", "los", "las", "una", "como", "más", "pero", "sus", "todo", "esta", "sobre", "entre", "cuando", "muy", "sin", "hasta", "desde", "está", "mi", "porque", "qué", "sólo", "han", "yo", "hay", "vez", "puede", "todos", "así", "nos", "ni", "parte", "tiene", "él", "uno", "donde", "bien", "tiempo", "mismo", "ese", "ahora", "cada", "e", "vida", "otro", "después", "te", "otros", "aunque", "esa", "esos", "estas", "estos", "otra", "otras", "otros", "otras", "mientras", "mejor", "nuevo", "nueva", "nuevos", "nuevas", "primer", "primera", "primeros", "primeras", "último", "última", "últimos", "últimas", "gran", "grande", "grandes", "pequeño", "pequeña", "pequeños", "pequeñas", "buen", "buena", "buenos", "buenas", "mal", "mala", "malos", "malas"],
                "patterns": [r"\b(el|la|de|que|y|a|en|un|es|se|no)\b", r"\b(con|para|por|sobre|entre|desde|hasta)\b", r"\b(muy|más|menos|tan|tanto)\b", r"\b(como|cuando|donde|porque|aunque)\b"]
            },
            "en": {
                "common_words": ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"],
                "patterns": [r"\b(the|be|to|of|and|a|in|that|have|i|it)\b", r"\b(for|not|on|with|he|as|you|do|at)\b", r"\b(this|but|his|by|from|they|we|say)\b", r"\b(her|she|or|an|will|my|one|all|would)\b"]
            }
        }
    
    async def translate_content(
        self, 
        content: str, 
        from_language: Language,
        to_language: Language
    ) -> str:
        """Traducir contenido entre idiomas."""
        try:
            if from_language == to_language:
                return content
            
            # Verificar caché de traducción
            cache_key = f"{from_language.value}_{to_language.value}_{hash(content)}"
            if cache_key in self.translation_cache:
                logger.info(f"Traducción encontrada en caché: {cache_key}")
                return self.translation_cache[cache_key]
            
            # Traducir términos técnicos primero
            translated_content = await self._translate_technical_terms(
                content, from_language, to_language
            )
            
            # Traducir contenido general
            translated_content = await self._translate_general_content(
                translated_content, from_language, to_language
            )
            
            # Aplicar adaptaciones culturales
            translated_content = await self._apply_cultural_adaptations(
                translated_content, to_language
            )
            
            # Guardar en caché
            self.translation_cache[cache_key] = translated_content
            
            logger.info(f"Contenido traducido de {from_language.value} a {to_language.value}")
            return translated_content
            
        except Exception as e:
            logger.error(f"Error traduciendo contenido: {str(e)}")
            return content
    
    async def detect_language(self, content: str) -> Language:
        """Detectar idioma del contenido."""
        try:
            # Limpiar contenido
            clean_content = re.sub(r'[^\w\s]', ' ', content.lower())
            words = clean_content.split()
            
            # Contar palabras comunes por idioma
            language_scores = {}
            
            for lang_code, patterns in self.language_patterns.items():
                score = 0
                common_words = patterns["common_words"]
                
                # Contar palabras comunes
                for word in words:
                    if word in common_words:
                        score += 1
                
                # Aplicar patrones regex
                for pattern in patterns["patterns"]:
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    score += matches * 2
                
                language_scores[lang_code] = score
            
            # Determinar idioma con mayor puntuación
            detected_lang = max(language_scores, key=language_scores.get)
            
            # Convertir a enum
            if detected_lang == "es":
                return Language.SPANISH
            elif detected_lang == "en":
                return Language.ENGLISH
            else:
                return Language.ENGLISH  # Default
                
        except Exception as e:
            logger.error(f"Error detectando idioma: {str(e)}")
            return Language.ENGLISH  # Default
    
    async def adapt_cultural_context(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Adaptar contexto cultural del contenido."""
        try:
            lang_code = language.value
            adaptations = self.cultural_adaptations.get(lang_code, {})
            
            adapted_content = content
            
            # Aplicar adaptaciones culturales
            for key, value in adaptations.items():
                if key == "formal_greeting":
                    adapted_content = self._adapt_greetings(adapted_content, value, "formal")
                elif key == "informal_greeting":
                    adapted_content = self._adapt_greetings(adapted_content, value, "informal")
                elif key == "emphasis_high":
                    adapted_content = self._adapt_emphasis(adapted_content, value, "high")
                elif key == "emphasis_medium":
                    adapted_content = self._adapt_emphasis(adapted_content, value, "medium")
                elif key == "emphasis_low":
                    adapted_content = self._adapt_emphasis(adapted_content, value, "low")
            
            logger.info(f"Contexto cultural adaptado para {lang_code}")
            return adapted_content
            
        except Exception as e:
            logger.error(f"Error adaptando contexto cultural: {str(e)}")
            return content
    
    async def _translate_technical_terms(
        self, 
        content: str, 
        from_language: Language,
        to_language: Language
    ) -> str:
        """Traducir términos técnicos."""
        from_terms = self.technical_terms_dictionary.get(from_language.value, {})
        to_terms = self.technical_terms_dictionary.get(to_language.value, {})
        
        translated_content = content
        
        # Crear mapeo inverso para términos técnicos
        reverse_mapping = {v: k for k, v in from_terms.items()}
        
        for term, translation in to_terms.items():
            if term in reverse_mapping:
                original_term = reverse_mapping[term]
                if original_term in from_terms:
                    # Reemplazar término técnico
                    pattern = rf'\b{re.escape(from_terms[original_term])}\b'
                    translated_content = re.sub(
                        pattern, 
                        translation, 
                        translated_content, 
                        flags=re.IGNORECASE
                    )
        
        return translated_content
    
    async def _translate_general_content(
        self, 
        content: str, 
        from_language: Language,
        to_language: Language
    ) -> str:
        """Traducir contenido general."""
        # Implementar lógica de traducción general
        # Por ahora, usar traducciones básicas predefinidas
        
        translations = {
            "es_en": {
                "El análisis de código": "Code analysis",
                "ha identificado": "has identified",
                "problemas": "issues",
                "críticos": "critical",
                "alta prioridad": "high priority",
                "puntuación de calidad": "quality score",
                "se recomienda": "it is recommended",
                "abordar": "address",
                "reducir riesgos": "reduce risks",
                "negocio": "business",
                "desarrollo": "development",
                "mantenimiento": "maintenance",
                "código": "code",
                "calidad": "quality",
                "problema": "issue",
                "solución": "solution",
                "mejora": "improvement",
                "refactorización": "refactoring",
                "prueba": "test",
                "error": "bug",
                "función": "function",
                "clase": "class",
                "método": "method",
                "variable": "variable"
            },
            "en_es": {
                "Code analysis": "El análisis de código",
                "has identified": "ha identificado",
                "issues": "problemas",
                "critical": "críticos",
                "high priority": "alta prioridad",
                "quality score": "puntuación de calidad",
                "it is recommended": "se recomienda",
                "address": "abordar",
                "reduce risks": "reducir riesgos",
                "business": "negocio",
                "development": "desarrollo",
                "maintenance": "mantenimiento",
                "code": "código",
                "quality": "calidad",
                "issue": "problema",
                "solution": "solución",
                "improvement": "mejora",
                "refactoring": "refactorización",
                "test": "prueba",
                "bug": "error",
                "function": "función",
                "class": "clase",
                "method": "método",
                "variable": "variable"
            }
        }
        
        translation_key = f"{from_language.value}_{to_language.value}"
        translation_dict = translations.get(translation_key, {})
        
        translated_content = content
        for original, translation in translation_dict.items():
            translated_content = translated_content.replace(original, translation)
        
        return translated_content
    
    async def _apply_cultural_adaptations(
        self, 
        content: str, 
        language: Language
    ) -> str:
        """Aplicar adaptaciones culturales."""
        lang_code = language.value
        adaptations = self.cultural_adaptations.get(lang_code, {})
        
        adapted_content = content
        
        # Aplicar adaptaciones específicas por idioma
        if lang_code == "es":
            # Adaptaciones para español
            adapted_content = self._apply_spanish_cultural_adaptations(adapted_content)
        elif lang_code == "en":
            # Adaptaciones para inglés
            adapted_content = self._apply_english_cultural_adaptations(adapted_content)
        
        return adapted_content
    
    def _apply_spanish_cultural_adaptations(self, content: str) -> str:
        """Aplicar adaptaciones culturales para español."""
        # Usar tono más formal y cortés
        adaptations = {
            "Hello": "Hola",
            "Hi": "Hola",
            "Dear": "Estimado/a",
            "Best regards": "Saludos cordiales",
            "Thank you": "Gracias",
            "Please": "Por favor",
            "Sorry": "Disculpe",
            "Excuse me": "Perdón",
            "very important": "muy importante",
            "important": "importante",
            "recommended": "recomendado"
        }
        
        adapted_content = content
        for english, spanish in adaptations.items():
            adapted_content = adapted_content.replace(english, spanish)
        
        return adapted_content
    
    def _apply_english_cultural_adaptations(self, content: str) -> str:
        """Aplicar adaptaciones culturales para inglés."""
        # Usar tono directo y profesional
        adaptations = {
            "Hola": "Hello",
            "Estimado/a": "Dear",
            "Saludos cordiales": "Best regards",
            "Gracias": "Thank you",
            "Por favor": "Please",
            "Disculpe": "Sorry",
            "Perdón": "Excuse me",
            "muy importante": "very important",
            "importante": "important",
            "recomendado": "recommended"
        }
        
        adapted_content = content
        for spanish, english in adaptations.items():
            adapted_content = adapted_content.replace(spanish, english)
        
        return adapted_content
    
    def _adapt_greetings(self, content: str, greeting: str, formality: str) -> str:
        """Adaptar saludos según formalidad."""
        if formality == "formal":
            # Reemplazar saludos informales con formales
            informal_greetings = ["Hola", "Hi", "Hello"]
            for informal in informal_greetings:
                if informal in content:
                    content = content.replace(informal, greeting)
        else:
            # Reemplazar saludos formales con informales
            formal_greetings = ["Estimado/a", "Dear"]
            for formal in formal_greetings:
                if formal in content:
                    content = content.replace(formal, greeting)
        
        return content
    
    def _adapt_emphasis(self, content: str, emphasis_word: str, level: str) -> str:
        """Adaptar nivel de énfasis."""
        emphasis_patterns = {
            "high": ["muy importante", "very important", "crítico", "critical"],
            "medium": ["importante", "important", "significativo", "significant"],
            "low": ["recomendado", "recommended", "sugerido", "suggested"]
        }
        
        patterns_to_replace = emphasis_patterns.get(level, [])
        
        for pattern in patterns_to_replace:
            if pattern in content:
                content = content.replace(pattern, emphasis_word)
        
        return content
    
    async def get_language_detection_result(self, content: str) -> LanguageDetectionResult:
        """Obtener resultado detallado de detección de idioma."""
        try:
            detected_language = await self.detect_language(content)
            
            # Calcular confianza basada en patrones encontrados
            confidence = await self._calculate_detection_confidence(content, detected_language)
            
            # Obtener idiomas alternativos
            alternative_languages = await self._get_alternative_languages(content, detected_language)
            
            return LanguageDetectionResult(
                detected_language=detected_language.value,
                confidence=confidence,
                alternative_languages=alternative_languages
            )
            
        except Exception as e:
            logger.error(f"Error obteniendo resultado de detección: {str(e)}")
            return LanguageDetectionResult(
                detected_language="en",
                confidence=0.5,
                alternative_languages=[]
            )
    
    async def _calculate_detection_confidence(
        self, 
        content: str, 
        detected_language: Language
    ) -> float:
        """Calcular confianza en la detección de idioma."""
        try:
            lang_code = detected_language.value
            patterns = self.language_patterns.get(lang_code, {})
            common_words = patterns.get("common_words", [])
            
            # Contar palabras comunes encontradas
            clean_content = re.sub(r'[^\w\s]', ' ', content.lower())
            words = clean_content.split()
            
            matches = sum(1 for word in words if word in common_words)
            total_words = len(words)
            
            if total_words == 0:
                return 0.0
            
            # Calcular confianza basada en porcentaje de palabras comunes
            confidence = min(matches / total_words * 10, 1.0)  # Normalizar a 0-1
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculando confianza: {str(e)}")
            return 0.5
    
    async def _get_alternative_languages(
        self, 
        content: str, 
        detected_language: Language
    ) -> List[Dict[str, Any]]:
        """Obtener idiomas alternativos con sus puntuaciones."""
        try:
            alternatives = []
            
            for lang_code, patterns in self.language_patterns.items():
                if lang_code != detected_language.value:
                    # Calcular puntuación para este idioma
                    score = await self._calculate_language_score(content, lang_code)
                    
                    if score > 0.1:  # Solo incluir si tiene puntuación significativa
                        alternatives.append({
                            "language": lang_code,
                            "confidence": score,
                            "name": self._get_language_name(lang_code)
                        })
            
            # Ordenar por confianza descendente
            alternatives.sort(key=lambda x: x["confidence"], reverse=True)
            
            return alternatives[:3]  # Retornar máximo 3 alternativas
            
        except Exception as e:
            logger.error(f"Error obteniendo idiomas alternativos: {str(e)}")
            return []
    
    async def _calculate_language_score(self, content: str, lang_code: str) -> float:
        """Calcular puntuación para un idioma específico."""
        try:
            patterns = self.language_patterns.get(lang_code, {})
            common_words = patterns.get("common_words", [])
            
            clean_content = re.sub(r'[^\w\s]', ' ', content.lower())
            words = clean_content.split()
            
            matches = sum(1 for word in words if word in common_words)
            total_words = len(words)
            
            if total_words == 0:
                return 0.0
            
            return matches / total_words
            
        except Exception as e:
            logger.error(f"Error calculando puntuación para {lang_code}: {str(e)}")
            return 0.0
    
    def _get_language_name(self, lang_code: str) -> str:
        """Obtener nombre del idioma."""
        language_names = {
            "es": "Español",
            "en": "English",
            "pt": "Português",
            "fr": "Français",
            "de": "Deutsch"
        }
        return language_names.get(lang_code, lang_code.upper())
