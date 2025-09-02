"""
Language Adapter for the Explanation Engine.

This module is responsible for adapting explanations to different languages,
currently supporting Spanish and English.
"""
from typing import Dict, List, Optional, Any, Callable
import logging
import re

from ...domain.entities.explanation import Language
from .exceptions import LanguageAdapterError, UnsupportedLanguageError


logger = logging.getLogger(__name__)


class LanguageAdapter:
    """
    Adapts explanations to different languages.
    
    This class provides functionality to translate and adapt explanations
    to different languages, ensuring that the content is not only translated
    but also culturally and contextually appropriate.
    """
    
    def __init__(self):
        """Initialize the language adapter."""
        self.translations = self._load_translations()
        self.language_patterns = self._load_language_patterns()
        
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translations for common terms and phrases."""
        return {
            # Technical terms
            "code_quality": {
                Language.SPANISH: "calidad del código",
                Language.ENGLISH: "code quality"
            },
            "complexity": {
                Language.SPANISH: "complejidad",
                Language.ENGLISH: "complexity"
            },
            "maintainability": {
                Language.SPANISH: "mantenibilidad",
                Language.ENGLISH: "maintainability"
            },
            "test_coverage": {
                Language.SPANISH: "cobertura de pruebas",
                Language.ENGLISH: "test coverage"
            },
            "documentation": {
                Language.SPANISH: "documentación",
                Language.ENGLISH: "documentation"
            },
            
            # Severity levels
            "critical": {
                Language.SPANISH: "crítico",
                Language.ENGLISH: "critical"
            },
            "high": {
                Language.SPANISH: "alto",
                Language.ENGLISH: "high"
            },
            "medium": {
                Language.SPANISH: "medio",
                Language.ENGLISH: "medium"
            },
            "low": {
                Language.SPANISH: "bajo",
                Language.ENGLISH: "low"
            },
            
            # Common phrases
            "detected_in": {
                Language.SPANISH: "detectado en",
                Language.ENGLISH: "detected in"
            },
            "line": {
                Language.SPANISH: "línea",
                Language.ENGLISH: "line"
            },
            "lines": {
                Language.SPANISH: "líneas",
                Language.ENGLISH: "lines"
            },
            "file": {
                Language.SPANISH: "archivo",
                Language.ENGLISH: "file"
            },
            "files": {
                Language.SPANISH: "archivos",
                Language.ENGLISH: "files"
            },
            "issue": {
                Language.SPANISH: "problema",
                Language.ENGLISH: "issue"
            },
            "issues": {
                Language.SPANISH: "problemas",
                Language.ENGLISH: "issues"
            },
            "recommendation": {
                Language.SPANISH: "recomendación",
                Language.ENGLISH: "recommendation"
            },
            "recommendations": {
                Language.SPANISH: "recomendaciones",
                Language.ENGLISH: "recommendations"
            },
            "priority": {
                Language.SPANISH: "prioridad",
                Language.ENGLISH: "priority"
            },
            "effort": {
                Language.SPANISH: "esfuerzo",
                Language.ENGLISH: "effort"
            },
            "impact": {
                Language.SPANISH: "impacto",
                Language.ENGLISH: "impact"
            },
            
            # Antipatterns
            "god_object": {
                Language.SPANISH: "Objeto Dios",
                Language.ENGLISH: "God Object"
            },
            "long_method": {
                Language.SPANISH: "Método Largo",
                Language.ENGLISH: "Long Method"
            },
            "duplicate_code": {
                Language.SPANISH: "Código Duplicado",
                Language.ENGLISH: "Duplicate Code"
            },
            "feature_envy": {
                Language.SPANISH: "Envidia de Características",
                Language.ENGLISH: "Feature Envy"
            },
            "shotgun_surgery": {
                Language.SPANISH: "Cirugía de Escopeta",
                Language.ENGLISH: "Shotgun Surgery"
            },
            "sql_injection": {
                Language.SPANISH: "Inyección SQL",
                Language.ENGLISH: "SQL Injection"
            },
            "n_plus_one_query": {
                Language.SPANISH: "Consulta N+1",
                Language.ENGLISH: "N+1 Query"
            },
            
            # Section titles
            "detected_issues": {
                Language.SPANISH: "Problemas Detectados",
                Language.ENGLISH: "Detected Issues"
            },
            "code_metrics": {
                Language.SPANISH: "Métricas de Código",
                Language.ENGLISH: "Code Metrics"
            },
            "detected_antipatterns": {
                Language.SPANISH: "Antipatrones Detectados",
                Language.ENGLISH: "Detected Antipatterns"
            },
            "recommendations": {
                Language.SPANISH: "Recomendaciones",
                Language.ENGLISH: "Recommendations"
            },
            "educational_resources": {
                Language.SPANISH: "Recursos Educativos",
                Language.ENGLISH: "Educational Resources"
            },
            "action_items": {
                Language.SPANISH: "Elementos de Acción",
                Language.ENGLISH: "Action Items"
            },
            
            # Quality ratings
            "excellent": {
                Language.SPANISH: "Excelente",
                Language.ENGLISH: "Excellent"
            },
            "good": {
                Language.SPANISH: "Bueno",
                Language.ENGLISH: "Good"
            },
            "acceptable": {
                Language.SPANISH: "Aceptable",
                Language.ENGLISH: "Acceptable"
            },
            "needs_improvement": {
                Language.SPANISH: "Necesita Mejoras",
                Language.ENGLISH: "Needs Improvement"
            },
            "poor": {
                Language.SPANISH: "Pobre",
                Language.ENGLISH: "Poor"
            },
            "very_poor": {
                Language.SPANISH: "Muy Pobre",
                Language.ENGLISH: "Very Poor"
            }
        }
    
    def _load_language_patterns(self) -> Dict[Language, Dict[str, Any]]:
        """Load language-specific patterns and rules."""
        return {
            Language.SPANISH: {
                "date_format": "%d/%m/%Y",
                "number_format": lambda n: f"{n:,}".replace(",", "."),
                "list_format": "• {item}",
                "emphasis_format": "**{text}**",
                "gender_rules": {
                    "high": {"m": "alto", "f": "alta"},
                    "medium": {"m": "medio", "f": "media"},
                    "low": {"m": "bajo", "f": "baja"}
                },
                "plural_rules": lambda n: "" if n == 1 else "s"
            },
            Language.ENGLISH: {
                "date_format": "%m/%d/%Y",
                "number_format": lambda n: f"{n:,}",
                "list_format": "• {item}",
                "emphasis_format": "**{text}**",
                "plural_rules": lambda n: "" if n == 1 else "s"
            }
        }
    
    async def adapt_content(
        self,
        content: str,
        source_language: Language,
        target_language: Language
    ) -> str:
        """
        Adapt content from source language to target language.
        
        Args:
            content: The content to adapt
            source_language: The source language
            target_language: The target language
            
        Returns:
            The adapted content in the target language
        """
        if source_language == target_language:
            return content
            
        try:
            # First, identify translatable segments
            segments = self._extract_translatable_segments(content)
            
            # Translate each segment
            translated_segments = {}
            for segment_id, segment_text in segments.items():
                translated_segments[segment_id] = await self._translate_segment(
                    segment_text, source_language, target_language
                )
            
            # Replace segments in the original content
            adapted_content = content
            for segment_id, translated_text in translated_segments.items():
                placeholder = f"{{{{SEGMENT_{segment_id}}}}}"
                adapted_content = adapted_content.replace(placeholder, translated_text)
            
            # Apply language-specific formatting
            adapted_content = self._apply_language_formatting(adapted_content, target_language)
            
            return adapted_content
            
        except Exception as e:
            logger.error(f"Error adapting content: {str(e)}")
            raise LanguageAdapterError(f"Failed to adapt content: {str(e)}")
    
    def _extract_translatable_segments(self, content: str) -> Dict[str, str]:
        """
        Extract translatable segments from content.
        
        This method identifies and extracts segments of text that should be
        translated, preserving special formatting and code blocks.
        """
        segments = {}
        segment_id = 0
        
        # Replace code blocks with placeholders
        code_blocks = []
        
        def replace_code_block(match):
            nonlocal code_blocks
            code_blocks.append(match.group(0))
            return f"{{{{CODE_BLOCK_{len(code_blocks) - 1}}}}}"
        
        content_without_code = re.sub(
            r'```(?:.*?)\n(.*?)```', replace_code_block, content, flags=re.DOTALL
        )
        
        # Extract paragraphs and other text segments
        lines = content_without_code.split('\n')
        current_segment = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                if current_segment:
                    segment_text = '\n'.join(current_segment)
                    segments[segment_id] = segment_text
                    current_segment = []
                    segment_id += 1
                continue
                
            # Skip markdown headings
            if re.match(r'^#+\s', line):
                if current_segment:
                    segment_text = '\n'.join(current_segment)
                    segments[segment_id] = segment_text
                    current_segment = []
                    segment_id += 1
                
                segments[segment_id] = line
                segment_id += 1
                continue
                
            # Add line to current segment
            current_segment.append(line)
        
        # Add the last segment if there is one
        if current_segment:
            segment_text = '\n'.join(current_segment)
            segments[segment_id] = segment_text
        
        # Replace code blocks back
        for segment_id, segment_text in segments.items():
            for i, code_block in enumerate(code_blocks):
                segment_text = segment_text.replace(f"{{{{CODE_BLOCK_{i}}}}}", code_block)
            segments[segment_id] = segment_text
        
        return segments
    
    async def _translate_segment(
        self,
        segment: str,
        source_language: Language,
        target_language: Language
    ) -> str:
        """
        Translate a segment of text.
        
        This method translates a segment of text from the source language
        to the target language, using the translations dictionary for
        known terms and phrases.
        """
        # Check if we have a direct translation for this segment
        for term, translations in self.translations.items():
            if segment.lower() == term.lower():
                if target_language in translations:
                    return translations[target_language]
        
        # For now, we'll use a simple word-by-word translation
        # In a real implementation, this would use a more sophisticated
        # translation service or machine learning model
        
        # Replace known terms and phrases
        translated = segment
        for term, translations in self.translations.items():
            if source_language in translations and target_language in translations:
                source_term = translations[source_language]
                target_term = translations[target_language]
                
                # Case-insensitive replacement
                pattern = re.compile(re.escape(source_term), re.IGNORECASE)
                translated = pattern.sub(target_term, translated)
        
        return translated
    
    def _apply_language_formatting(self, content: str, language: Language) -> str:
        """
        Apply language-specific formatting.
        
        This method applies language-specific formatting rules to the content,
        such as date formats, number formats, and list formats.
        """
        if language not in self.language_patterns:
            return content
            
        patterns = self.language_patterns[language]
        
        # Apply number formatting
        number_format = patterns.get("number_format")
        if number_format:
            def replace_number(match):
                try:
                    num = float(match.group(0).replace(',', ''))
                    return number_format(num)
                except ValueError:
                    return match.group(0)
                    
            content = re.sub(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', replace_number, content)
        
        return content
    
    def get_translation(
        self,
        term: str,
        language: Language,
        default: Optional[str] = None
    ) -> str:
        """
        Get the translation of a term in the specified language.
        
        Args:
            term: The term to translate
            language: The target language
            default: Default value if translation is not found
            
        Returns:
            The translated term
        """
        if term in self.translations and language in self.translations[term]:
            return self.translations[term][language]
        return default or term
    
    def format_number(self, number: float, language: Language) -> str:
        """
        Format a number according to language conventions.
        
        Args:
            number: The number to format
            language: The target language
            
        Returns:
            The formatted number as a string
        """
        if language in self.language_patterns:
            number_format = self.language_patterns[language].get("number_format")
            if number_format:
                return number_format(number)
        return str(number)
    
    def format_date(self, date: datetime, language: Language) -> str:
        """
        Format a date according to language conventions.
        
        Args:
            date: The date to format
            language: The target language
            
        Returns:
            The formatted date as a string
        """
        if language in self.language_patterns:
            date_format = self.language_patterns[language].get("date_format")
            if date_format:
                return date.strftime(date_format)
        return str(date)
    
    def get_plural_form(self, term: str, count: int, language: Language) -> str:
        """
        Get the plural form of a term based on count.
        
        Args:
            term: The term to pluralize
            count: The count to determine plurality
            language: The target language
            
        Returns:
            The pluralized term
        """
        if language in self.language_patterns:
            plural_rules = self.language_patterns[language].get("plural_rules")
            if plural_rules:
                suffix = plural_rules(count)
                return f"{term}{suffix}"
        return term
    
    def get_gender_form(
        self,
        term: str,
        gender: str,
        language: Language
    ) -> str:
        """
        Get the gender-specific form of a term.
        
        Args:
            term: The term to modify
            gender: The gender ('m' or 'f')
            language: The target language
            
        Returns:
            The gender-specific form of the term
        """
        if (language in self.language_patterns and
            "gender_rules" in self.language_patterns[language] and
            term in self.language_patterns[language]["gender_rules"] and
            gender in self.language_patterns[language]["gender_rules"][term]):
            return self.language_patterns[language]["gender_rules"][term][gender]
        return term
