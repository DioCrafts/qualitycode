"""
Módulo que implementa el aprendizaje de patrones para el sistema de reglas en lenguaje natural.
"""
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from codeant_agent.application.ports.natural_rules.learning_ports import PatternLearnerPort


class PatternLearner(PatternLearnerPort):
    """Implementación del aprendizaje de patrones."""
    
    def __init__(self):
        """Inicializa el aprendizaje de patrones."""
        # Patrones aprendidos por idioma
        self.learned_patterns = {
            "spanish": {},
            "english": {},
        }
        
        # Ejemplos positivos y negativos por patrón
        self.pattern_examples = {
            "spanish": {},
            "english": {},
        }
        
        # Umbrales para aprendizaje
        self.min_examples = 3
        self.min_confidence = 0.7
    
    async def learn_patterns(
        self, text_samples: List[str], language: str
    ) -> Dict[str, str]:
        """Aprende patrones a partir de muestras de texto.
        
        Args:
            text_samples: Muestras de texto
            language: Idioma de las muestras
            
        Returns:
            Diccionario con patrones aprendidos
        """
        if not text_samples or len(text_samples) < self.min_examples:
            return {}
        
        # Normalizar idioma
        norm_language = language.lower()
        if norm_language not in self.learned_patterns:
            norm_language = "english"  # Idioma por defecto
        
        # Extraer n-gramas comunes
        common_ngrams = await self._extract_common_ngrams(text_samples)
        
        # Generar patrones a partir de n-gramas
        patterns = await self._generate_patterns_from_ngrams(common_ngrams)
        
        # Evaluar patrones con las muestras
        evaluated_patterns = await self._evaluate_patterns(patterns, text_samples)
        
        # Filtrar patrones por confianza
        learned_patterns = {}
        for pattern_name, (pattern, confidence) in evaluated_patterns.items():
            if confidence >= self.min_confidence:
                learned_patterns[pattern_name] = pattern
                
                # Guardar patrón aprendido
                self.learned_patterns[norm_language][pattern_name] = pattern
        
        return learned_patterns
    
    async def improve_pattern(
        self, pattern_name: str, feedback: List[Dict[str, str]]
    ) -> Optional[str]:
        """Mejora un patrón basándose en feedback.
        
        Args:
            pattern_name: Nombre del patrón a mejorar
            feedback: Feedback para el patrón
            
        Returns:
            Patrón mejorado o None si no se pudo mejorar
        """
        # Buscar patrón en idiomas soportados
        pattern = None
        language = None
        
        for lang, patterns in self.learned_patterns.items():
            if pattern_name in patterns:
                pattern = patterns[pattern_name]
                language = lang
                break
        
        if not pattern:
            return None
        
        # Extraer ejemplos positivos y negativos del feedback
        positive_examples = []
        negative_examples = []
        
        for item in feedback:
            if "text" in item and "is_match" in item:
                if item["is_match"] == "true":
                    positive_examples.append(item["text"])
                else:
                    negative_examples.append(item["text"])
        
        if not positive_examples:
            return None
        
        # Actualizar ejemplos para el patrón
        if pattern_name not in self.pattern_examples[language]:
            self.pattern_examples[language][pattern_name] = {
                "positive": [],
                "negative": [],
            }
        
        self.pattern_examples[language][pattern_name]["positive"].extend(positive_examples)
        self.pattern_examples[language][pattern_name]["negative"].extend(negative_examples)
        
        # Generar patrón mejorado
        improved_pattern = await self._refine_pattern(
            pattern,
            self.pattern_examples[language][pattern_name]["positive"],
            self.pattern_examples[language][pattern_name]["negative"]
        )
        
        if improved_pattern:
            # Actualizar patrón aprendido
            self.learned_patterns[language][pattern_name] = improved_pattern
        
        return improved_pattern
    
    async def _extract_common_ngrams(
        self, text_samples: List[str], min_length: int = 2, max_length: int = 5
    ) -> List[str]:
        """Extrae n-gramas comunes de muestras de texto.
        
        Args:
            text_samples: Muestras de texto
            min_length: Longitud mínima de n-gramas
            max_length: Longitud máxima de n-gramas
            
        Returns:
            Lista de n-gramas comunes
        """
        # Extraer todos los n-gramas
        all_ngrams = []
        
        for text in text_samples:
            words = text.lower().split()
            
            for n in range(min_length, min(max_length + 1, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i:i+n])
                    all_ngrams.append(ngram)
        
        # Contar frecuencia de n-gramas
        ngram_counts = Counter(all_ngrams)
        
        # Filtrar n-gramas que aparecen en múltiples muestras
        common_ngrams = []
        sample_count = len(text_samples)
        
        for ngram, count in ngram_counts.items():
            if count >= max(2, sample_count // 3):  # Aparece en al menos un tercio de las muestras
                common_ngrams.append(ngram)
        
        return common_ngrams
    
    async def _generate_patterns_from_ngrams(
        self, ngrams: List[str]
    ) -> Dict[str, str]:
        """Genera patrones a partir de n-gramas.
        
        Args:
            ngrams: Lista de n-gramas
            
        Returns:
            Diccionario con patrones generados
        """
        patterns = {}
        
        for i, ngram in enumerate(ngrams):
            # Convertir n-grama a patrón regex
            pattern = re.escape(ngram)
            
            # Generalizar patrón
            pattern = pattern.replace(r"\ ", r"\s+")  # Espacios flexibles
            
            # Agregar captura para palabras variables
            pattern = re.sub(r"\\s\+", r"\\s+(\w+)\\s+", pattern, count=1)
            
            # Nombre del patrón
            pattern_name = f"learned_pattern_{i+1}"
            
            patterns[pattern_name] = pattern
        
        return patterns
    
    async def _evaluate_patterns(
        self, patterns: Dict[str, str], text_samples: List[str]
    ) -> Dict[str, Tuple[str, float]]:
        """Evalúa patrones con muestras de texto.
        
        Args:
            patterns: Diccionario con patrones a evaluar
            text_samples: Muestras de texto
            
        Returns:
            Diccionario con patrones evaluados y su confianza
        """
        evaluated_patterns = {}
        
        for pattern_name, pattern in patterns.items():
            # Contar coincidencias
            match_count = 0
            
            for text in text_samples:
                if re.search(pattern, text, re.IGNORECASE):
                    match_count += 1
            
            # Calcular confianza
            confidence = match_count / len(text_samples) if text_samples else 0
            
            evaluated_patterns[pattern_name] = (pattern, confidence)
        
        return evaluated_patterns
    
    async def _refine_pattern(
        self, pattern: str, positive_examples: List[str], negative_examples: List[str]
    ) -> Optional[str]:
        """Refina un patrón basándose en ejemplos positivos y negativos.
        
        Args:
            pattern: Patrón a refinar
            positive_examples: Ejemplos positivos
            negative_examples: Ejemplos negativos
            
        Returns:
            Patrón refinado o None si no se pudo refinar
        """
        if not positive_examples:
            return None
        
        # Evaluar patrón actual
        false_positives = 0
        false_negatives = 0
        
        for example in positive_examples:
            if not re.search(pattern, example, re.IGNORECASE):
                false_negatives += 1
        
        for example in negative_examples:
            if re.search(pattern, example, re.IGNORECASE):
                false_positives += 1
        
        # Si el patrón es perfecto, no refinarlo
        if false_positives == 0 and false_negatives == 0:
            return pattern
        
        # Intentar refinar el patrón
        refined_pattern = pattern
        
        if false_positives > 0:
            # Hacer el patrón más específico
            refined_pattern = self._make_pattern_more_specific(
                refined_pattern, negative_examples
            )
        
        if false_negatives > 0:
            # Hacer el patrón más general
            refined_pattern = self._make_pattern_more_general(
                refined_pattern, positive_examples
            )
        
        return refined_pattern
    
    def _make_pattern_more_specific(
        self, pattern: str, negative_examples: List[str]
    ) -> str:
        """Hace un patrón más específico para evitar falsos positivos.
        
        Args:
            pattern: Patrón a modificar
            negative_examples: Ejemplos negativos
            
        Returns:
            Patrón modificado
        """
        # Implementación simplificada para el ejemplo
        # En un sistema real, se harían modificaciones más sofisticadas
        return pattern
    
    def _make_pattern_more_general(
        self, pattern: str, positive_examples: List[str]
    ) -> str:
        """Hace un patrón más general para evitar falsos negativos.
        
        Args:
            pattern: Patrón a modificar
            positive_examples: Ejemplos positivos
            
        Returns:
            Patrón modificado
        """
        # Implementación simplificada para el ejemplo
        # En un sistema real, se harían modificaciones más sofisticadas
        return pattern
