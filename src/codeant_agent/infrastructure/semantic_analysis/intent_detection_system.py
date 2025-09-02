"""
Sistema de detección de intención de código.

Este módulo implementa la detección automática de la intención
y propósito de fragmentos de código usando análisis semántico.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict

from ...domain.entities.semantic_analysis import (
    IntentDetectionConfig, CodeIntentAnalysis, DetectedIntent, CodePurpose,
    IntentType, PurposeType, AbstractionLevel, CodeConcept, ConceptType
)
from ...domain.entities.ai_models import CodeEmbedding
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class IntentEvidence:
    """Evidencia para detección de intención."""
    evidence_type: str  # naming, structure, content, behavior
    evidence_text: str
    confidence: float
    supporting_details: List[str]


@dataclass
class BehavioralPattern:
    """Patrón de comportamiento detectado."""
    pattern_name: str
    pattern_indicators: List[str]
    confidence: float
    related_intents: List[IntentType]


class NamingAnalyzer:
    """Analizador de nomenclatura para detectar intención."""
    
    def __init__(self):
        self.intent_naming_patterns = self._initialize_naming_patterns()
        self.purpose_naming_patterns = self._initialize_purpose_patterns()
    
    def _initialize_naming_patterns(self) -> Dict[IntentType, List[str]]:
        """Inicializa patrones de nomenclatura por intención."""
        return {
            IntentType.DATA_RETRIEVAL: [
                "get", "fetch", "read", "load", "retrieve", "find", "query", 
                "select", "lookup", "search", "obtain", "acquire"
            ],
            IntentType.DATA_MODIFICATION: [
                "set", "update", "modify", "change", "edit", "write", "save",
                "store", "persist", "commit", "apply", "assign"
            ],
            IntentType.DATA_TRANSFORMATION: [
                "convert", "transform", "parse", "format", "encode", "decode",
                "serialize", "deserialize", "map", "filter", "reduce", "process"
            ],
            IntentType.OBJECT_CREATION: [
                "create", "new", "make", "build", "construct", "generate", 
                "initialize", "instantiate", "spawn", "allocate"
            ],
            IntentType.OBJECT_DESTRUCTION: [
                "delete", "remove", "destroy", "clear", "clean", "reset",
                "dispose", "free", "release", "terminate"
            ],
            IntentType.VALIDATION: [
                "validate", "check", "verify", "test", "assert", "ensure",
                "confirm", "authenticate", "authorize", "approve"
            ],
            IntentType.CALCULATION: [
                "calculate", "compute", "sum", "count", "measure", "evaluate",
                "determine", "analyze", "estimate", "derive"
            ],
            IntentType.COMMUNICATION: [
                "send", "receive", "request", "response", "notify", "publish",
                "subscribe", "broadcast", "emit", "listen", "call", "invoke"
            ],
            IntentType.ERROR_HANDLING: [
                "error", "exception", "handle", "catch", "throw", "raise",
                "fail", "recover", "retry", "fallback"
            ],
            IntentType.LOGGING: [
                "log", "print", "debug", "trace", "info", "warn", "error",
                "record", "report", "monitor", "track"
            ],
            IntentType.CONFIGURATION: [
                "config", "setting", "option", "parameter", "preference",
                "setup", "configure", "initialize", "bootstrap"
            ],
            IntentType.SECURITY: [
                "encrypt", "decrypt", "hash", "auth", "secure", "protect",
                "permission", "access", "role", "token", "credential"
            ],
            IntentType.TESTING: [
                "test", "mock", "stub", "fake", "verify", "assert",
                "should", "expect", "spec", "describe", "it"
            ]
        }
    
    def _initialize_purpose_patterns(self) -> Dict[PurposeType, List[str]]:
        """Inicializa patrones de propósito."""
        return {
            PurposeType.BUSINESS_LOGIC: [
                "business", "domain", "rule", "policy", "workflow", "process"
            ],
            PurposeType.DATA_ACCESS: [
                "repository", "dao", "database", "storage", "persistence", "query"
            ],
            PurposeType.USER_INTERFACE: [
                "view", "component", "widget", "form", "button", "display", "render"
            ],
            PurposeType.INFRASTRUCTURE: [
                "config", "setup", "bootstrap", "framework", "middleware", "pipeline"
            ],
            PurposeType.UTILITY: [
                "util", "helper", "common", "shared", "tool", "library"
            ],
            PurposeType.SERVICE: [
                "service", "provider", "manager", "handler", "controller", "processor"
            ]
        }
    
    def analyze_naming_intent(self, identifier: str) -> List[Tuple[IntentType, float]]:
        """
        Analiza intención basada en nomenclatura.
        
        Args:
            identifier: Nombre de función, clase o variable
            
        Returns:
            Lista de tuplas (IntentType, confidence)
        """
        results = []
        identifier_lower = identifier.lower()
        
        for intent_type, patterns in self.intent_naming_patterns.items():
            confidence = 0.0
            
            for pattern in patterns:
                if pattern in identifier_lower:
                    # Boost si está al inicio (prefijo)
                    if identifier_lower.startswith(pattern):
                        confidence = max(confidence, 0.9)
                    # Boost si está al final (sufijo)
                    elif identifier_lower.endswith(pattern):
                        confidence = max(confidence, 0.7)
                    # Boost menor si está en el medio
                    else:
                        confidence = max(confidence, 0.5)
            
            if confidence > 0.0:
                results.append((intent_type, confidence))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def analyze_purpose_from_naming(self, identifier: str, context: str = "") -> Optional[CodePurpose]:
        """Analiza propósito basado en nomenclatura."""
        identifier_lower = identifier.lower()
        context_lower = context.lower()
        combined_text = f"{identifier_lower} {context_lower}"
        
        best_purpose = None
        best_score = 0.0
        
        for purpose_type, patterns in self.purpose_naming_patterns.items():
            score = 0.0
            
            for pattern in patterns:
                if pattern in combined_text:
                    score += 0.3 if pattern in identifier_lower else 0.1
            
            if score > best_score:
                best_score = score
                best_purpose = purpose_type
        
        if best_purpose and best_score > 0.2:
            return CodePurpose(
                purpose_type=best_purpose,
                description=f"Based on naming patterns, appears to be {best_purpose.value}",
                abstraction_level=self._estimate_abstraction_from_naming(identifier_lower),
                domain_specificity=min(1.0, best_score),
                reusability_score=self._estimate_reusability_from_naming(identifier_lower)
            )
        
        return None
    
    def _estimate_abstraction_from_naming(self, identifier: str) -> AbstractionLevel:
        """Estima nivel de abstracción basado en nomenclatura."""
        # Indicadores de alta abstracción
        high_abstraction_indicators = [
            "abstract", "base", "interface", "contract", "template",
            "generic", "framework", "architecture", "system"
        ]
        
        # Indicadores de baja abstracción
        low_abstraction_indicators = [
            "impl", "concrete", "specific", "detail", "raw", "native",
            "low_level", "system", "hardware", "memory"
        ]
        
        if any(indicator in identifier for indicator in high_abstraction_indicators):
            return AbstractionLevel.HIGH
        elif any(indicator in identifier for indicator in low_abstraction_indicators):
            return AbstractionLevel.LOW
        else:
            return AbstractionLevel.MEDIUM
    
    def _estimate_reusability_from_naming(self, identifier: str) -> float:
        """Estima reusabilidad basada en nomenclatura."""
        base_score = 0.5
        
        # Indicadores de alta reusabilidad
        if any(word in identifier for word in ["util", "helper", "common", "shared", "generic"]):
            base_score += 0.3
        
        # Indicadores de baja reusabilidad
        if any(word in identifier for word in ["specific", "custom", "hardcode", "temp", "test"]):
            base_score -= 0.2
        
        return max(0.0, min(1.0, base_score))


class StructuralAnalyzer:
    """Analizador de estructura para detectar intención."""
    
    def __init__(self):
        self.structural_patterns = self._initialize_structural_patterns()
    
    def _initialize_structural_patterns(self) -> Dict[IntentType, Dict[str, float]]:
        """Inicializa patrones estructurales."""
        return {
            IntentType.VALIDATION: {
                "if_statements": 0.8,
                "assert_statements": 0.9,
                "exception_handling": 0.7,
                "return_boolean": 0.6
            },
            IntentType.CALCULATION: {
                "arithmetic_operations": 0.8,
                "math_functions": 0.9,
                "numeric_variables": 0.6,
                "return_numeric": 0.7
            },
            IntentType.ITERATION: {
                "for_loops": 0.9,
                "while_loops": 0.8,
                "list_comprehensions": 0.7,
                "iterator_patterns": 0.6
            },
            IntentType.ERROR_HANDLING: {
                "try_catch_blocks": 0.9,
                "exception_types": 0.8,
                "error_messages": 0.7,
                "logging_errors": 0.6
            }
        }
    
    def analyze_structural_intent(self, code: str, language: ProgrammingLanguage) -> List[Tuple[IntentType, float]]:
        """
        Analiza intención basada en estructura del código.
        
        Args:
            code: Código fuente
            language: Lenguaje de programación
            
        Returns:
            Lista de tuplas (IntentType, confidence)
        """
        structural_features = self._extract_structural_features(code, language)
        intent_scores = defaultdict(float)
        
        # Calcular scores para cada intención
        for intent_type, patterns in self.structural_patterns.items():
            for pattern, weight in patterns.items():
                if pattern in structural_features:
                    intent_scores[intent_type] += structural_features[pattern] * weight
        
        # Normalizar scores
        max_possible_score = max(sum(patterns.values()) for patterns in self.structural_patterns.values())
        normalized_scores = [
            (intent, score / max_possible_score) 
            for intent, score in intent_scores.items()
        ]
        
        return sorted(normalized_scores, key=lambda x: x[1], reverse=True)
    
    def _extract_structural_features(self, code: str, language: ProgrammingLanguage) -> Dict[str, float]:
        """Extrae características estructurales del código."""
        features = {}
        lines = code.splitlines()
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines == 0:
            return features
        
        # Contar statements condicionales
        if_count = sum(line.count('if ') + line.count('if(') for line in lines)
        features["if_statements"] = if_count / total_lines
        
        # Contar loops
        for_count = sum(line.count('for ') + line.count('for(') for line in lines)
        while_count = sum(line.count('while ') + line.count('while(') for line in lines)
        features["for_loops"] = for_count / total_lines
        features["while_loops"] = while_count / total_lines
        
        # Contar try-catch
        try_count = sum(line.count('try') + line.count('try:') for line in lines)
        catch_count = sum(line.count('catch') + line.count('except') for line in lines)
        features["try_catch_blocks"] = (try_count + catch_count) / total_lines
        
        # Contar operaciones aritméticas
        arithmetic_ops = ['+', '-', '*', '/', '%', '**', '//']
        arithmetic_count = sum(
            sum(line.count(op) for op in arithmetic_ops) 
            for line in lines
        )
        features["arithmetic_operations"] = arithmetic_count / total_lines
        
        # Detectar returns booleanos
        boolean_returns = sum(
            1 for line in lines 
            if re.search(r'return\s+(True|False|true|false|\w+\s*==|\w+\s*!=)', line.strip())
        )
        features["return_boolean"] = boolean_returns / total_lines
        
        # Detectar asserts
        assert_count = sum(line.count('assert') for line in lines)
        features["assert_statements"] = assert_count / total_lines
        
        # Detectar funciones matemáticas
        math_functions = ['math.', 'Math.', 'sqrt', 'pow', 'abs', 'sin', 'cos', 'log']
        math_count = sum(
            sum(line.count(func) for func in math_functions) 
            for line in lines
        )
        features["math_functions"] = math_count / total_lines
        
        return features


class BehavioralAnalyzer:
    """Analizador de comportamiento de código."""
    
    def __init__(self):
        self.behavioral_indicators = self._initialize_behavioral_indicators()
    
    def _initialize_behavioral_indicators(self) -> Dict[str, List[str]]:
        """Inicializa indicadores de comportamiento."""
        return {
            "data_processing": [
                "process", "transform", "convert", "parse", "format", "normalize"
            ],
            "state_management": [
                "state", "status", "flag", "mode", "current", "active", "enabled"
            ],
            "resource_management": [
                "open", "close", "acquire", "release", "lock", "unlock", "allocate", "free"
            ],
            "communication": [
                "send", "receive", "request", "response", "message", "signal", "event"
            ],
            "coordination": [
                "synchronize", "coordinate", "schedule", "queue", "dispatch", "route"
            ],
            "monitoring": [
                "monitor", "track", "measure", "collect", "report", "observe", "watch"
            ]
        }
    
    def analyze_behavioral_characteristics(self, code: str) -> List[str]:
        """
        Analiza características de comportamiento del código.
        
        Args:
            code: Código fuente
            
        Returns:
            Lista de características de comportamiento detectadas
        """
        characteristics = []
        code_lower = code.lower()
        
        for behavior_type, indicators in self.behavioral_indicators.items():
            indicator_count = sum(1 for indicator in indicators if indicator in code_lower)
            
            if indicator_count > 0:
                confidence = min(1.0, indicator_count / len(indicators))
                characteristics.append(f"{behavior_type} (confidence: {confidence:.2f})")
        
        # Análisis de patrones de control de flujo
        if any(pattern in code_lower for pattern in ['if', 'else', 'elif', 'switch', 'case']):
            characteristics.append("conditional_execution")
        
        if any(pattern in code_lower for pattern in ['for', 'while', 'loop', 'iterate']):
            characteristics.append("iterative_processing")
        
        if any(pattern in code_lower for pattern in ['try', 'except', 'catch', 'finally']):
            characteristics.append("exception_handling")
        
        if any(pattern in code_lower for pattern in ['async', 'await', 'promise', 'future']):
            characteristics.append("asynchronous_execution")
        
        return list(set(characteristics))  # Remover duplicados


class DomainAnalyzer:
    """Analizador de dominio de negocio."""
    
    def __init__(self):
        self.domain_vocabularies = self._initialize_domain_vocabularies()
    
    def _initialize_domain_vocabularies(self) -> Dict[str, List[str]]:
        """Inicializa vocabularios por dominio."""
        return {
            "authentication": [
                "user", "password", "login", "logout", "session", "token", 
                "auth", "credential", "authenticate", "authorize"
            ],
            "e_commerce": [
                "product", "order", "cart", "payment", "customer", "invoice",
                "purchase", "checkout", "shipping", "inventory"
            ],
            "financial": [
                "account", "balance", "transaction", "transfer", "payment",
                "currency", "money", "amount", "interest", "loan"
            ],
            "content_management": [
                "content", "article", "post", "page", "media", "document",
                "publish", "draft", "editor", "cms"
            ],
            "data_analytics": [
                "analytics", "metrics", "report", "dashboard", "chart",
                "statistics", "analysis", "insight", "trend"
            ],
            "social_media": [
                "user", "post", "comment", "like", "share", "follow",
                "friend", "profile", "feed", "notification"
            ],
            "healthcare": [
                "patient", "doctor", "appointment", "medical", "health",
                "diagnosis", "treatment", "prescription", "record"
            ],
            "education": [
                "student", "teacher", "course", "lesson", "grade", "assignment",
                "exam", "school", "education", "learning"
            ]
        }
    
    def analyze_domain_concepts(self, code: str, context: str = "") -> List[CodeConcept]:
        """
        Analiza conceptos de dominio en el código.
        
        Args:
            code: Código fuente
            context: Contexto adicional (nombre archivo, etc.)
            
        Returns:
            Lista de conceptos de dominio detectados
        """
        concepts = []
        combined_text = f"{code} {context}".lower()
        
        for domain, vocabulary in self.domain_vocabularies.items():
            domain_score = 0.0
            matching_terms = []
            
            for term in vocabulary:
                if term in combined_text:
                    domain_score += 1.0
                    matching_terms.append(term)
            
            if domain_score > 0:
                confidence = min(1.0, domain_score / len(vocabulary))
                
                concept = CodeConcept(
                    concept_type=ConceptType.DOMAIN,
                    name=domain,
                    confidence=confidence,
                    related_concepts=matching_terms,
                    context=f"Domain vocabulary matching: {len(matching_terms)}/{len(vocabulary)} terms"
                )
                concepts.append(concept)
        
        # Ordenar por confianza
        concepts.sort(key=lambda c: c.confidence, reverse=True)
        
        return concepts[:5]  # Top 5 dominios


class PatternBasedIntentDetector:
    """Detector de intención basado en patrones."""
    
    def __init__(self, config: IntentDetectionConfig):
        self.config = config
        self.naming_analyzer = NamingAnalyzer()
        self.structural_analyzer = StructuralAnalyzer()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.domain_analyzer = DomainAnalyzer()
    
    async def detect_intent(
        self,
        code: str,
        language: ProgrammingLanguage,
        function_name: Optional[str] = None,
        context: str = ""
    ) -> CodeIntentAnalysis:
        """
        Detecta intención del código usando patrones.
        
        Args:
            code: Código fuente
            language: Lenguaje de programación
            function_name: Nombre de función (opcional)
            context: Contexto adicional
            
        Returns:
            Análisis completo de intención
        """
        analysis = CodeIntentAnalysis(
            code_id=f"intent_{hash(code)}_{int(time.time())}"
        )
        
        detected_intents = []
        
        # Análisis por nomenclatura
        if function_name:
            naming_intents = self.naming_analyzer.analyze_naming_intent(function_name)
            for intent_type, confidence in naming_intents:
                if confidence >= self.config.confidence_threshold:
                    intent = DetectedIntent(
                        intent_type=intent_type,
                        description=f"Based on function name '{function_name}': {intent_type.value}",
                        evidence=[f"Naming pattern analysis: {confidence:.2f}"],
                        confidence=confidence,
                        context_clues=[function_name]
                    )
                    detected_intents.append(intent)
        
        # Análisis estructural
        structural_intents = self.structural_analyzer.analyze_structural_intent(code, language)
        for intent_type, confidence in structural_intents:
            if confidence >= self.config.confidence_threshold:
                intent = DetectedIntent(
                    intent_type=intent_type,
                    description=f"Based on code structure: {intent_type.value}",
                    evidence=[f"Structural analysis: {confidence:.2f}"],
                    confidence=confidence,
                    context_clues=["code_structure"]
                )
                detected_intents.append(intent)
        
        # Análisis de comportamiento
        behavioral_characteristics = self.behavioral_analyzer.analyze_behavioral_characteristics(code)
        analysis.behavioral_characteristics = behavioral_characteristics
        
        # Análisis de dominio
        if self.config.enable_domain_analysis:
            domain_concepts = self.domain_analyzer.analyze_domain_concepts(code, context)
            analysis.domain_concepts = domain_concepts
        
        # Análisis de propósito
        if function_name:
            purpose = self.naming_analyzer.analyze_purpose_from_naming(function_name, context)
            analysis.primary_purpose = purpose
        
        # Consolidar intenciones detectadas
        analysis.detected_intents = self._consolidate_intents(detected_intents)
        
        # Calcular scores de confianza
        for intent in analysis.detected_intents:
            analysis.confidence_scores[intent.intent_type] = intent.confidence
        
        return analysis
    
    def _consolidate_intents(self, intents: List[DetectedIntent]) -> List[DetectedIntent]:
        """Consolida intenciones detectadas eliminando duplicados."""
        # Agrupar por tipo de intención
        intent_groups = defaultdict(list)
        for intent in intents:
            intent_groups[intent.intent_type].append(intent)
        
        consolidated = []
        
        for intent_type, group_intents in intent_groups.items():
            if len(group_intents) == 1:
                consolidated.append(group_intents[0])
            else:
                # Combinar múltiples detecciones del mismo tipo
                avg_confidence = sum(intent.confidence for intent in group_intents) / len(group_intents)
                combined_evidence = []
                combined_context = []
                
                for intent in group_intents:
                    combined_evidence.extend(intent.evidence)
                    combined_context.extend(intent.context_clues)
                
                combined_intent = DetectedIntent(
                    intent_type=intent_type,
                    description=f"Multiple indicators suggest: {intent_type.value}",
                    evidence=list(set(combined_evidence)),
                    confidence=min(1.0, avg_confidence * 1.1),  # Boost por múltiple evidencia
                    context_clues=list(set(combined_context))
                )
                consolidated.append(combined_intent)
        
        # Limitar cantidad si está configurado
        if self.config.max_intents_per_code > 0:
            consolidated.sort(key=lambda x: x.confidence, reverse=True)
            consolidated = consolidated[:self.config.max_intents_per_code]
        
        return consolidated


class IntentDetectionSystem:
    """Sistema principal de detección de intención."""
    
    def __init__(self, config: Optional[IntentDetectionConfig] = None):
        """
        Inicializa el sistema de detección de intención.
        
        Args:
            config: Configuración del sistema
        """
        self.config = config or IntentDetectionConfig()
        self.pattern_detector = PatternBasedIntentDetector(self.config)
        
        # Estadísticas
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'average_confidence': 0.0,
            'intent_distribution': defaultdict(int)
        }
    
    async def analyze_code_intent(
        self,
        code: str,
        language: ProgrammingLanguage,
        embedding: Optional[CodeEmbedding] = None,
        function_name: Optional[str] = None,
        file_path: Optional[Path] = None
    ) -> CodeIntentAnalysis:
        """
        Analiza intención completa del código.
        
        Args:
            code: Código fuente
            language: Lenguaje de programación
            embedding: Embedding del código (opcional)
            function_name: Nombre de función (opcional)
            file_path: Ruta del archivo (opcional)
            
        Returns:
            Análisis completo de intención
        """
        # Crear contexto
        context = ""
        if file_path:
            context = f"File: {file_path.name}"
        if function_name:
            context += f" Function: {function_name}"
        
        # Ejecutar detección basada en patrones
        analysis = await self.pattern_detector.detect_intent(
            code, language, function_name, context
        )
        
        # Análisis adicional con embedding si está disponible
        if embedding and self.config.enable_ml_intent_detection:
            # En implementación futura: usar ML para detección
            ml_analysis = await self._analyze_with_embedding(embedding, code)
            self._merge_analyses(analysis, ml_analysis)
        
        # Enriquecer con análisis contextual
        analysis = await self._enrich_with_contextual_analysis(analysis, code, language)
        
        # Actualizar estadísticas
        self._update_detection_stats(analysis)
        
        return analysis
    
    async def _analyze_with_embedding(self, embedding: CodeEmbedding, code: str) -> CodeIntentAnalysis:
        """Analiza intención usando embeddings (placeholder para ML futuro)."""
        # Placeholder para análisis ML futuro
        # Por ahora, análisis heurístico basado en propiedades del embedding
        
        analysis = CodeIntentAnalysis(code_id=embedding.id)
        
        # Análisis simplificado basado en características del embedding
        embedding_magnitude = sum(x**2 for x in embedding.embedding_vector)**0.5
        
        if embedding_magnitude > 20.0:  # Alta magnitud sugiere complejidad
            complex_intent = DetectedIntent(
                intent_type=IntentType.GENERAL,
                description="Complex code detected via embedding analysis",
                confidence=0.6,
                evidence=["High embedding magnitude"]
            )
            analysis.detected_intents.append(complex_intent)
        
        return analysis
    
    def _merge_analyses(self, base_analysis: CodeIntentAnalysis, ml_analysis: CodeIntentAnalysis) -> None:
        """Combina análisis de diferentes fuentes."""
        # Añadir intenciones de ML que no estén ya detectadas
        existing_intent_types = {intent.intent_type for intent in base_analysis.detected_intents}
        
        for ml_intent in ml_analysis.detected_intents:
            if ml_intent.intent_type not in existing_intent_types:
                base_analysis.detected_intents.append(ml_intent)
            else:
                # Boost confidence si ML confirma detección existente
                for existing_intent in base_analysis.detected_intents:
                    if existing_intent.intent_type == ml_intent.intent_type:
                        existing_intent.confidence = min(1.0, existing_intent.confidence * 1.2)
                        existing_intent.evidence.extend(ml_intent.evidence)
    
    async def _enrich_with_contextual_analysis(
        self,
        analysis: CodeIntentAnalysis,
        code: str,
        language: ProgrammingLanguage
    ) -> CodeIntentAnalysis:
        """Enriquece análisis con información contextual."""
        # Análisis de complejidad contextual
        complexity_level = self._assess_complexity_level(code)
        
        # Ajustar confianza basada en complejidad
        for intent in analysis.detected_intents:
            if complexity_level == "high" and intent.intent_type == IntentType.GENERAL:
                intent.confidence *= 0.8  # Reducir confianza para código complejo sin intención clara
            elif complexity_level == "low" and intent.intent_type in [IntentType.VALIDATION, IntentType.CALCULATION]:
                intent.confidence *= 1.1  # Boost para intenciones simples en código simple
        
        # Añadir características de calidad
        if not analysis.behavioral_characteristics:
            analysis.behavioral_characteristics = []
        
        analysis.behavioral_characteristics.extend([
            f"complexity_level: {complexity_level}",
            f"language: {language.value}",
            f"code_length: {len(code)} chars"
        ])
        
        return analysis
    
    def _assess_complexity_level(self, code: str) -> str:
        """Evalúa nivel de complejidad del código."""
        # Contar indicadores de complejidad
        complexity_indicators = [
            code.count('if '),
            code.count('for '),
            code.count('while '),
            code.count('try'),
            code.count('nested'),
            len(re.findall(r'\{.*\{', code))  # Bloques anidados
        ]
        
        total_complexity = sum(complexity_indicators)
        lines_count = len(code.splitlines())
        
        complexity_ratio = total_complexity / max(1, lines_count / 10)
        
        if complexity_ratio > 2.0:
            return "high"
        elif complexity_ratio > 1.0:
            return "medium"
        else:
            return "low"
    
    def _update_detection_stats(self, analysis: CodeIntentAnalysis) -> None:
        """Actualiza estadísticas de detección."""
        self.detection_stats['total_detections'] += 1
        
        if analysis.detected_intents:
            self.detection_stats['successful_detections'] += 1
            
            # Actualizar confianza promedio
            avg_confidence = sum(intent.confidence for intent in analysis.detected_intents) / len(analysis.detected_intents)
            current_avg = self.detection_stats['average_confidence']
            total = self.detection_stats['total_detections']
            
            self.detection_stats['average_confidence'] = (
                (current_avg * (total - 1) + avg_confidence) / total
            )
            
            # Actualizar distribución de intenciones
            for intent in analysis.detected_intents:
                self.detection_stats['intent_distribution'][intent.intent_type] += 1
    
    async def get_detection_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de detección."""
        total = self.detection_stats['total_detections']
        success_rate = (
            self.detection_stats['successful_detections'] / max(1, total)
        )
        
        # Top intenciones detectadas
        top_intents = sorted(
            self.detection_stats['intent_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_detections": total,
            "successful_detections": self.detection_stats['successful_detections'],
            "success_rate": success_rate,
            "average_confidence": self.detection_stats['average_confidence'],
            "top_detected_intents": [
                {"intent": intent.value, "count": count} 
                for intent, count in top_intents
            ],
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "max_intents_per_code": self.config.max_intents_per_code,
                "domain_analysis_enabled": self.config.enable_domain_analysis
            }
        }
    
    async def batch_analyze_intents(
        self,
        code_samples: List[Tuple[str, ProgrammingLanguage, Optional[str]]]
    ) -> List[CodeIntentAnalysis]:
        """
        Analiza intenciones en batch para múltiples códigos.
        
        Args:
            code_samples: Lista de tuplas (código, lenguaje, nombre_función)
            
        Returns:
            Lista de análisis de intención
        """
        analyses = []
        
        # Procesar en paralelo (simulado)
        for code, language, func_name in code_samples:
            try:
                analysis = await self.analyze_code_intent(code, language, function_name=func_name)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analizando intención: {e}")
                # Crear análisis de error
                error_analysis = CodeIntentAnalysis(
                    code_id=f"error_{hash(code)}",
                    detected_intents=[
                        DetectedIntent(
                            intent_type=IntentType.GENERAL,
                            description="Error in intent analysis",
                            confidence=0.0,
                            evidence=[str(e)]
                        )
                    ]
                )
                analyses.append(error_analysis)
        
        return analyses
