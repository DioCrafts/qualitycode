"""
Analizador contextual para refinar detecciones de antipatrones basado en contexto.
"""

import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

from ...domain.entities.antipattern_analysis import (
    AntipatternDetectionResult, AntipatternType, AntipatternFeatures, 
    ResponsibilityType, AntipatternCategory
)
from ...domain.value_objects.programming_language import ProgrammingLanguage
from ..ast_analysis.unified_ast import UnifiedAST, UnifiedNode, UnifiedNodeType
from .classifiers.base_classifier import DetectedPattern

logger = logging.getLogger(__name__)


@dataclass
class ContextualRule:
    """Regla contextual para refinamiento."""
    rule_id: str
    pattern_types: List[AntipatternType]
    condition: str
    action: str  # "suppress", "enhance", "modify"
    confidence_adjustment: float = 0.0
    description: str = ""


@dataclass
class CodeContext:
    """Contexto del código analizado."""
    file_type: str
    project_type: str
    framework_indicators: List[str]
    architectural_patterns: List[str]
    testing_context: bool
    configuration_context: bool
    generated_code_context: bool
    legacy_code_indicators: List[str]


class ContextualAnalyzer:
    """Analizador contextual para refinar detecciones de antipatrones."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Reglas contextuales
        self.contextual_rules = self._initialize_contextual_rules()
        
        # Indicadores de framework/tecnología
        self.framework_indicators = self._initialize_framework_indicators()
        
        # Patrones arquitectónicos comunes
        self.architectural_patterns = self._initialize_architectural_patterns()
    
    def _initialize_contextual_rules(self) -> List[ContextualRule]:
        """Inicializar reglas contextuales."""
        
        return [
            # Reglas para código de testing
            ContextualRule(
                rule_id="test_code_large_class",
                pattern_types=[AntipatternType.LARGE_CLASS, AntipatternType.GOD_OBJECT],
                condition="testing_context",
                action="suppress",
                confidence_adjustment=-0.3,
                description="Las clases de test pueden ser legitimamente grandes"
            ),
            
            # Reglas para código generado
            ContextualRule(
                rule_id="generated_code_patterns",
                pattern_types=[
                    AntipatternType.LONG_METHOD, 
                    AntipatternType.LARGE_CLASS,
                    AntipatternType.PRIMITIVE_OBSESSION
                ],
                condition="generated_code_context",
                action="suppress",
                confidence_adjustment=-0.5,
                description="Código generado puede tener patrones que parecen antipatrones"
            ),
            
            # Reglas para configuración
            ContextualRule(
                rule_id="config_hardcoded_values",
                pattern_types=[AntipatternType.HARDCODED_SECRETS],
                condition="configuration_context",
                action="enhance",
                confidence_adjustment=0.2,
                description="Valores hardcodeados son más problemáticos en configuración"
            ),
            
            # Reglas para frameworks web
            ContextualRule(
                rule_id="web_framework_god_objects",
                pattern_types=[AntipatternType.GOD_OBJECT],
                condition="web_framework",
                action="modify",
                confidence_adjustment=-0.1,
                description="Los frameworks web pueden tener controladores legitimamente grandes"
            ),
            
            # Reglas para código legacy
            ContextualRule(
                rule_id="legacy_code_tolerance",
                pattern_types=[
                    AntipatternType.SPAGHETTI_CODE,
                    AntipatternType.BIG_BALL_OF_MUD,
                    AntipatternType.LAVA_FLOW
                ],
                condition="legacy_code",
                action="modify",
                confidence_adjustment=-0.15,
                description="Código legacy puede tener antipatrones por razones históricas"
            ),
            
            # Reglas para performance crítico
            ContextualRule(
                rule_id="performance_critical_optimization",
                pattern_types=[AntipatternType.PREMATURE_OPTIMIZATION],
                condition="performance_critical",
                action="suppress",
                confidence_adjustment=-0.4,
                description="Optimización puede ser justificada en contextos críticos"
            ),
        ]
    
    def _initialize_framework_indicators(self) -> Dict[str, List[str]]:
        """Inicializar indicadores de frameworks."""
        
        return {
            "web_frameworks": [
                "flask", "django", "express", "react", "vue", "angular",
                "spring", "struts", "rails", "laravel", "symfony"
            ],
            "testing_frameworks": [
                "pytest", "unittest", "jest", "mocha", "junit", "testng",
                "rspec", "phpunit", "nunit"
            ],
            "orm_frameworks": [
                "sqlalchemy", "hibernate", "sequelize", "mongoose",
                "activerecord", "eloquent"
            ],
            "ml_frameworks": [
                "tensorflow", "pytorch", "scikit-learn", "pandas",
                "numpy", "keras"
            ],
            "mobile_frameworks": [
                "react-native", "flutter", "ionic", "xamarin"
            ]
        }
    
    def _initialize_architectural_patterns(self) -> Dict[str, List[str]]:
        """Inicializar patrones arquitectónicos."""
        
        return {
            "mvc": ["controller", "model", "view", "mvc"],
            "microservices": ["service", "microservice", "api", "endpoint"],
            "repository": ["repository", "dao", "data_access"],
            "factory": ["factory", "builder", "creator"],
            "observer": ["observer", "listener", "event", "subscriber"],
            "singleton": ["singleton", "instance", "get_instance"],
            "adapter": ["adapter", "wrapper", "bridge"],
            "strategy": ["strategy", "policy", "algorithm"]
        }
    
    async def refine_detections(
        self, 
        detection_result: AntipatternDetectionResult,
        unified_ast: UnifiedAST
    ) -> AntipatternDetectionResult:
        """Refinar detecciones basándose en contexto."""
        
        try:
            # Extraer contexto del código
            code_context = await self._extract_code_context(detection_result, unified_ast)
            
            # Aplicar reglas contextuales
            refined_patterns = []
            
            for pattern in detection_result.detected_antipatterns:
                refined_pattern = await self._apply_contextual_rules(
                    pattern, code_context, unified_ast
                )
                
                if refined_pattern:  # Puede ser None si se suprime
                    refined_patterns.append(refined_pattern)
            
            # Aplicar análisis de coherencia global
            coherent_patterns = await self._apply_global_coherence_analysis(
                refined_patterns, code_context, unified_ast
            )
            
            # Crear resultado refinado
            refined_result = AntipatternDetectionResult(
                file_path=detection_result.file_path,
                language=detection_result.language,
                detected_antipatterns=coherent_patterns,
                architectural_issues=self._categorize_patterns(coherent_patterns, AntipatternCategory.ARCHITECTURAL),
                design_issues=self._categorize_patterns(coherent_patterns, AntipatternCategory.DESIGN),
                performance_issues=self._categorize_patterns(coherent_patterns, AntipatternCategory.PERFORMANCE),
                security_issues=self._categorize_patterns(coherent_patterns, AntipatternCategory.SECURITY),
                confidence_scores=self._recalculate_confidence_scores(coherent_patterns),
                explanations=detection_result.explanations,  # Mantener explicaciones originales
                detection_time_ms=detection_result.detection_time_ms
            )
            
            return refined_result
            
        except Exception as e:
            logger.error(f"Error in contextual analysis: {e}")
            # Fallback: retornar resultado original
            return detection_result
    
    async def _extract_code_context(
        self, 
        detection_result: AntipatternDetectionResult,
        unified_ast: UnifiedAST
    ) -> CodeContext:
        """Extraer contexto del código."""
        
        file_path = detection_result.file_path
        source_code = unified_ast.source_code or ""
        source_lower = source_code.lower()
        
        # Determinar tipo de archivo
        file_type = self._determine_file_type(file_path)
        
        # Detectar tipo de proyecto
        project_type = self._detect_project_type(file_path, source_code)
        
        # Detectar indicadores de framework
        framework_indicators = self._detect_framework_indicators(source_code)
        
        # Detectar patrones arquitectónicos
        architectural_patterns = self._detect_architectural_patterns(source_code)
        
        # Detectar contextos especiales
        testing_context = self._is_testing_context(file_path, source_code)
        configuration_context = self._is_configuration_context(file_path, source_code)
        generated_code_context = self._is_generated_code_context(source_code)
        
        # Detectar indicadores de código legacy
        legacy_indicators = self._detect_legacy_code_indicators(source_code)
        
        return CodeContext(
            file_type=file_type,
            project_type=project_type,
            framework_indicators=framework_indicators,
            architectural_patterns=architectural_patterns,
            testing_context=testing_context,
            configuration_context=configuration_context,
            generated_code_context=generated_code_context,
            legacy_code_indicators=legacy_indicators
        )
    
    async def _apply_contextual_rules(
        self, 
        pattern: DetectedPattern,
        context: CodeContext,
        unified_ast: UnifiedAST
    ) -> Optional[DetectedPattern]:
        """Aplicar reglas contextuales a un patrón."""
        
        applicable_rules = [
            rule for rule in self.contextual_rules
            if pattern.pattern_type in rule.pattern_types
        ]
        
        if not applicable_rules:
            return pattern  # Sin reglas aplicables, mantener original
        
        modified_pattern = DetectedPattern(
            pattern_type=pattern.pattern_type,
            confidence=pattern.confidence,
            locations=pattern.locations.copy(),
            description=pattern.description,
            evidence=pattern.evidence.copy(),
            severity_indicators=pattern.severity_indicators.copy(),
            feature_importance=pattern.feature_importance.copy() if pattern.feature_importance else {}
        )
        
        for rule in applicable_rules:
            if self._rule_condition_matches(rule, context):
                if rule.action == "suppress":
                    # Reducir confianza significativamente o suprimir
                    modified_pattern.confidence += rule.confidence_adjustment
                    if modified_pattern.confidence <= 0.2:
                        return None  # Suprimir patrón
                    
                    # Añadir nota contextual
                    modified_pattern.evidence.append(f"Contextual suppression: {rule.description}")
                
                elif rule.action == "enhance":
                    # Aumentar confianza
                    modified_pattern.confidence += rule.confidence_adjustment
                    modified_pattern.evidence.append(f"Contextual enhancement: {rule.description}")
                
                elif rule.action == "modify":
                    # Modificar confianza moderadamente
                    modified_pattern.confidence += rule.confidence_adjustment
                    modified_pattern.description += f" (Context: {rule.description})"
        
        # Asegurar que la confianza esté en rango válido
        modified_pattern.confidence = max(0.0, min(1.0, modified_pattern.confidence))
        
        return modified_pattern
    
    async def _apply_global_coherence_analysis(
        self, 
        patterns: List[DetectedPattern],
        context: CodeContext,
        unified_ast: UnifiedAST
    ) -> List[DetectedPattern]:
        """Aplicar análisis de coherencia global."""
        
        if len(patterns) <= 1:
            return patterns  # Sin análisis de coherencia para pocos patrones
        
        coherent_patterns = []
        
        # Agrupar patrones por tipo para análisis de coherencia
        patterns_by_type = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in patterns_by_type:
                patterns_by_type[pattern_type] = []
            patterns_by_type[pattern_type].append(pattern)
        
        # Aplicar coherencia tipo por tipo
        for pattern_type, type_patterns in patterns_by_type.items():
            if len(type_patterns) > 1:
                # Multiple detecciones del mismo tipo - verificar coherencia
                coherent_type_patterns = await self._resolve_multiple_same_type_patterns(
                    type_patterns, context
                )
                coherent_patterns.extend(coherent_type_patterns)
            else:
                coherent_patterns.extend(type_patterns)
        
        # Aplicar análisis de conflictos entre tipos diferentes
        final_patterns = await self._resolve_pattern_conflicts(coherent_patterns, context)
        
        return final_patterns
    
    async def _resolve_multiple_same_type_patterns(
        self, 
        patterns: List[DetectedPattern],
        context: CodeContext
    ) -> List[DetectedPattern]:
        """Resolver múltiples patrones del mismo tipo."""
        
        if len(patterns) <= 1:
            return patterns
        
        # Estrategia: mantener el patrón con mayor confianza
        # a menos que haya evidencia contradictoria
        
        # Ordenar por confianza descendente
        sorted_patterns = sorted(patterns, key=lambda p: p.confidence, reverse=True)
        
        # Verificar si los patrones son consistentes entre sí
        consistent_patterns = [sorted_patterns[0]]  # Empezar con el de mayor confianza
        
        for pattern in sorted_patterns[1:]:
            if await self._patterns_are_consistent(consistent_patterns[0], pattern):
                # Combinar evidencia si son consistentes
                combined_evidence = list(set(consistent_patterns[0].evidence + pattern.evidence))
                consistent_patterns[0].evidence = combined_evidence[:10]  # Limitar evidencia
                
                # Promediar confianza ponderadamente
                weight1 = 0.7  # Mayor peso al patrón de mayor confianza
                weight2 = 0.3
                new_confidence = (consistent_patterns[0].confidence * weight1 + 
                                pattern.confidence * weight2)
                consistent_patterns[0].confidence = new_confidence
            else:
                # Si no son consistentes, mantener ambos pero con menor confianza
                pattern.confidence *= 0.8
                consistent_patterns.append(pattern)
        
        return consistent_patterns
    
    async def _resolve_pattern_conflicts(
        self, 
        patterns: List[DetectedPattern],
        context: CodeContext
    ) -> List[DetectedPattern]:
        """Resolver conflictos entre patrones de diferentes tipos."""
        
        # Definir conflictos conocidos
        conflicting_pairs = [
            # Premature optimization vs Performance issues
            (AntipatternType.PREMATURE_OPTIMIZATION, AntipatternType.INEFFICIENT_ALGORITHM),
            
            # Singleton abuse vs God object (pueden ser el mismo problema)
            (AntipatternType.SINGLETON_ABUSE, AntipatternType.GOD_OBJECT),
            
            # Large class vs God object (similar problema, diferentes niveles)
            (AntipatternType.LARGE_CLASS, AntipatternType.GOD_OBJECT),
        ]
        
        # Detectar y resolver conflictos
        patterns_by_type = {p.pattern_type: p for p in patterns}
        resolved_patterns = patterns.copy()
        
        for type1, type2 in conflicting_pairs:
            if type1 in patterns_by_type and type2 in patterns_by_type:
                pattern1 = patterns_by_type[type1]
                pattern2 = patterns_by_type[type2]
                
                # Resolver conflicto: mantener el de mayor confianza
                if pattern1.confidence > pattern2.confidence:
                    # Combinar evidencia y remover el menor
                    pattern1.evidence.extend(pattern2.evidence[:3])  # Añadir algo de evidencia
                    resolved_patterns = [p for p in resolved_patterns if p != pattern2]
                else:
                    pattern2.evidence.extend(pattern1.evidence[:3])
                    resolved_patterns = [p for p in resolved_patterns if p != pattern1]
        
        return resolved_patterns
    
    async def _patterns_are_consistent(
        self, 
        pattern1: DetectedPattern,
        pattern2: DetectedPattern
    ) -> bool:
        """Verificar si dos patrones son consistentes entre sí."""
        
        # Verificar diferencia de confianza
        confidence_diff = abs(pattern1.confidence - pattern2.confidence)
        if confidence_diff > 0.4:
            return False  # Demasiada diferencia de confianza
        
        # Verificar solapamiento de evidencia
        evidence1_set = set(pattern1.evidence)
        evidence2_set = set(pattern2.evidence)
        
        overlap = len(evidence1_set.intersection(evidence2_set))
        total_unique = len(evidence1_set.union(evidence2_set))
        
        overlap_ratio = overlap / total_unique if total_unique > 0 else 0
        
        # Patrones consistentes deben tener algún solapamiento de evidencia
        return overlap_ratio >= 0.2
    
    def _rule_condition_matches(self, rule: ContextualRule, context: CodeContext) -> bool:
        """Verificar si la condición de una regla se cumple."""
        
        condition_map = {
            "testing_context": context.testing_context,
            "configuration_context": context.configuration_context,
            "generated_code_context": context.generated_code_context,
            "web_framework": any(fw in context.framework_indicators for fw in ["web_frameworks"]),
            "legacy_code": len(context.legacy_code_indicators) > 0,
            "performance_critical": "performance_critical" in context.framework_indicators
        }
        
        return condition_map.get(rule.condition, False)
    
    def _determine_file_type(self, file_path: Path) -> str:
        """Determinar tipo de archivo."""
        
        suffix = file_path.suffix.lower()
        
        type_mapping = {
            ".py": "python_source",
            ".js": "javascript_source", 
            ".ts": "typescript_source",
            ".java": "java_source",
            ".rs": "rust_source",
            ".test.py": "python_test",
            ".test.js": "javascript_test",
            ".spec.js": "javascript_spec",
            ".config.py": "python_config",
            ".settings.py": "python_settings"
        }
        
        # Verificar patrones específicos primero
        name_lower = file_path.name.lower()
        if "test" in name_lower or "spec" in name_lower:
            return f"{type_mapping.get(suffix, 'unknown')}_test"
        
        if "config" in name_lower or "settings" in name_lower:
            return f"{type_mapping.get(suffix, 'unknown')}_config"
        
        return type_mapping.get(suffix, "unknown")
    
    def _detect_project_type(self, file_path: Path, source_code: str) -> str:
        """Detectar tipo de proyecto."""
        
        # Verificar archivos de proyecto comunes
        project_indicators = {
            "web_application": ["app.py", "main.py", "server.py", "index.js", "app.js"],
            "api_service": ["api.py", "routes.py", "endpoints.py", "service.py"],
            "library": ["__init__.py", "lib.py", "utils.py", "helpers.py"],
            "test_suite": ["test_", "tests/", "spec/", "conftest.py"],
            "configuration": ["settings.py", "config.py", "configuration.py"]
        }
        
        file_name = file_path.name.lower()
        
        for project_type, indicators in project_indicators.items():
            if any(indicator in file_name for indicator in indicators):
                return project_type
        
        return "general"
    
    def _detect_framework_indicators(self, source_code: str) -> List[str]:
        """Detectar indicadores de frameworks."""
        
        detected_frameworks = []
        source_lower = source_code.lower()
        
        for category, frameworks in self.framework_indicators.items():
            for framework in frameworks:
                if framework in source_lower:
                    detected_frameworks.append(category)
                    break  # Solo añadir categoría una vez
        
        return detected_frameworks
    
    def _detect_architectural_patterns(self, source_code: str) -> List[str]:
        """Detectar patrones arquitectónicos."""
        
        detected_patterns = []
        source_lower = source_code.lower()
        
        for pattern, keywords in self.architectural_patterns.items():
            if any(keyword in source_lower for keyword in keywords):
                detected_patterns.append(pattern)
        
        return detected_patterns
    
    def _is_testing_context(self, file_path: Path, source_code: str) -> bool:
        """Verificar si es contexto de testing."""
        
        file_indicators = [
            "test" in file_path.name.lower(),
            "spec" in file_path.name.lower(),
            "/test/" in str(file_path).lower(),
            "/tests/" in str(file_path).lower()
        ]
        
        code_indicators = [
            "def test_" in source_code,
            "class Test" in source_code,
            "it(" in source_code,
            "describe(" in source_code,
            "assert" in source_code,
            "expect(" in source_code
        ]
        
        return any(file_indicators) or any(code_indicators)
    
    def _is_configuration_context(self, file_path: Path, source_code: str) -> bool:
        """Verificar si es contexto de configuración."""
        
        file_indicators = [
            "config" in file_path.name.lower(),
            "settings" in file_path.name.lower(),
            ".env" in file_path.name.lower()
        ]
        
        code_indicators = [
            "CONFIG" in source_code,
            "SETTINGS" in source_code,
            "os.environ" in source_code,
            "getenv(" in source_code
        ]
        
        return any(file_indicators) or any(code_indicators)
    
    def _is_generated_code_context(self, source_code: str) -> bool:
        """Verificar si es código generado."""
        
        generated_indicators = [
            "# Generated by",
            "// Generated by",
            "# Auto-generated",
            "// Auto-generated",
            "# This file was automatically generated",
            "DO NOT EDIT",
            "automatically created"
        ]
        
        return any(indicator in source_code for indicator in generated_indicators)
    
    def _detect_legacy_code_indicators(self, source_code: str) -> List[str]:
        """Detectar indicadores de código legacy."""
        
        legacy_indicators = []
        
        # Indicadores de versiones antiguas
        version_indicators = [
            "python2", "python 2", "# -*- coding:",
            "var ", "function()", "prototype",
            "TODO: migrate", "FIXME: legacy", "DEPRECATED"
        ]
        
        for indicator in version_indicators:
            if indicator in source_code:
                legacy_indicators.append(indicator)
        
        return legacy_indicators
    
    def _categorize_patterns(
        self, 
        patterns: List[DetectedPattern],
        category: AntipatternCategory
    ) -> List[DetectedPattern]:
        """Categorizar patrones por tipo."""
        
        category_mapping = {
            AntipatternCategory.ARCHITECTURAL: [
                AntipatternType.GOD_OBJECT, AntipatternType.BIG_BALL_OF_MUD,
                AntipatternType.SPAGHETTI_CODE, AntipatternType.LAVA_FLOW
            ],
            AntipatternCategory.DESIGN: [
                AntipatternType.LARGE_CLASS, AntipatternType.LONG_METHOD,
                AntipatternType.FEATURE_ENVY, AntipatternType.DATA_CLUMPS
            ],
            AntipatternCategory.PERFORMANCE: [
                AntipatternType.N_PLUS_ONE_QUERY, AntipatternType.MEMORY_LEAK,
                AntipatternType.INEFFICIENT_ALGORITHM
            ],
            AntipatternCategory.SECURITY: [
                AntipatternType.SQL_INJECTION, AntipatternType.HARDCODED_SECRETS,
                AntipatternType.XSS_VULNERABILITY
            ]
        }
        
        target_types = category_mapping.get(category, [])
        return [p for p in patterns if p.pattern_type in target_types]
    
    def _recalculate_confidence_scores(
        self, 
        patterns: List[DetectedPattern]
    ) -> Dict[AntipatternType, float]:
        """Recalcular scores de confianza."""
        
        confidence_scores = {}
        for pattern in patterns:
            confidence_scores[pattern.pattern_type] = pattern.confidence
        
        return confidence_scores
