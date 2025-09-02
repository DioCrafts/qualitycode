"""
Implementación del categorizador de issues.

Este módulo implementa la categorización automática de issues usando
reglas predefinidas, análisis ML y análisis contextual.
"""

import logging
import asyncio
import re
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np

from ...domain.entities.issue_management import (
    RawIssue, CategorizedIssue, IssueCategory, IssueSeverity, IssueId,
    IssueMetadata, ContextInfo, CategorizationConfig, BusinessImpactLevel,
    IssueFeatureVector
)
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class ClassificationRule:
    """Regla de clasificación para issues."""
    id: str
    category: IssueCategory
    rule_type: str  # "pattern", "metric", "context", "hybrid"
    pattern: Optional[str] = None
    conditions: List['ClassificationCondition'] = field(default_factory=list)
    confidence: float = 0.8
    auto_tags: List[str] = field(default_factory=list)
    language_specific: Optional[ProgrammingLanguage] = None
    
    def matches(self, issue: RawIssue) -> bool:
        """Verifica si la regla aplica al issue."""
        # Verificar patrón si existe
        if self.pattern:
            text_to_match = f"{issue.rule_id} {issue.message} {issue.category or ''}"
            try:
                if not re.search(self.pattern, text_to_match, re.IGNORECASE):
                    return False
            except re.error:
                logger.warning(f"Invalid regex pattern in rule {self.id}: {self.pattern}")
                return False
        
        # Verificar condiciones
        for condition in self.conditions:
            if not condition.check(issue):
                return False
        
        # Verificar lenguaje específico
        if self.language_specific and issue.language != self.language_specific:
            return False
        
        return True


@dataclass
class ClassificationCondition:
    """Condición para regla de clasificación."""
    condition_type: str
    value: Any
    comparison: str = "equals"  # "equals", "contains", "greater", "less", "in"
    
    def check(self, issue: RawIssue) -> bool:
        """Verifica si la condición se cumple."""
        if self.condition_type == "rule_id_contains":
            return str(self.value).lower() in issue.rule_id.lower()
        
        elif self.condition_type == "message_contains":
            return str(self.value).lower() in issue.message.lower()
        
        elif self.condition_type == "category_contains":
            category = issue.category or ""
            return str(self.value).lower() in category.lower()
        
        elif self.condition_type == "severity_at_least":
            severity_order = {
                IssueSeverity.INFO: 1,
                IssueSeverity.LOW: 2,
                IssueSeverity.MEDIUM: 3,
                IssueSeverity.HIGH: 4,
                IssueSeverity.CRITICAL: 5
            }
            current_level = severity_order.get(issue.severity, 1)
            required_level = severity_order.get(self.value, 1)
            return current_level >= required_level
        
        elif self.condition_type == "complexity_above":
            if issue.complexity_metrics:
                return issue.complexity_metrics.cyclomatic_complexity > self.value
            return False
        
        elif self.condition_type == "file_extension":
            return issue.file_path.suffix.lower() == str(self.value).lower()
        
        elif self.condition_type == "language_is":
            return issue.language == self.value
        
        else:
            logger.warning(f"Unknown condition type: {self.condition_type}")
            return False


@dataclass
class MLPrediction:
    """Predicción de clasificación ML."""
    category: IssueCategory
    confidence: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    def is_confident(self, threshold: float = 0.7) -> bool:
        """Verifica si la predicción es confiable."""
        return self.confidence >= threshold


@dataclass
class CategorizationResult:
    """Resultado de categorización de un issue."""
    issue_id: IssueId
    primary_category: IssueCategory
    secondary_categories: List[IssueCategory]
    confidence_scores: Dict[IssueCategory, float]
    applied_rules: List[str]
    ml_predictions: List[MLPrediction]
    auto_tags: List[str]
    categorization_time_ms: int


class DomainKnowledge:
    """Base de conocimiento del dominio para categorización."""
    
    def __init__(self):
        self.security_keywords = {
            'sql', 'injection', 'xss', 'csrf', 'authentication', 'authorization',
            'password', 'secret', 'token', 'encrypt', 'decrypt', 'vulnerability',
            'exploit', 'attack', 'malicious', 'sanitize', 'validate'
        }
        
        self.performance_keywords = {
            'slow', 'performance', 'memory', 'leak', 'cache', 'optimization',
            'bottleneck', 'latency', 'throughput', 'cpu', 'disk', 'network',
            'database', 'query', 'algorithm', 'complexity', 'timeout'
        }
        
        self.maintainability_keywords = {
            'duplicate', 'complexity', 'refactor', 'smell', 'design', 'pattern',
            'coupling', 'cohesion', 'modular', 'separation', 'responsibility',
            'maintainable', 'readable', 'clean', 'structure'
        }
        
        self.reliability_keywords = {
            'error', 'exception', 'crash', 'fail', 'bug', 'defect', 'fault',
            'handle', 'recovery', 'robust', 'stable', 'null', 'undefined',
            'boundary', 'edge case', 'validation'
        }
        
        self.documentation_keywords = {
            'comment', 'documentation', 'doc', 'readme', 'guide', 'example',
            'usage', 'api', 'specification', 'requirement', 'missing doc'
        }
    
    async def suggest_categories(self, issue: RawIssue) -> List[IssueCategory]:
        """Sugiere categorías basadas en conocimiento del dominio."""
        suggestions = []
        
        text = f"{issue.rule_id} {issue.message} {issue.category or ''}".lower()
        
        # Análisis por palabras clave
        if any(keyword in text for keyword in self.security_keywords):
            suggestions.append(IssueCategory.SECURITY)
        
        if any(keyword in text for keyword in self.performance_keywords):
            suggestions.append(IssueCategory.PERFORMANCE)
        
        if any(keyword in text for keyword in self.maintainability_keywords):
            suggestions.append(IssueCategory.MAINTAINABILITY)
        
        if any(keyword in text for keyword in self.reliability_keywords):
            suggestions.append(IssueCategory.RELIABILITY)
        
        if any(keyword in text for keyword in self.documentation_keywords):
            suggestions.append(IssueCategory.DOCUMENTATION)
        
        # Análisis específico por severidad
        if issue.severity == IssueSeverity.CRITICAL:
            if IssueCategory.SECURITY not in suggestions:
                suggestions.insert(0, IssueCategory.SECURITY)
        
        # Análisis por tipo de archivo
        if issue.file_path.suffix in ['.py', '.js', '.ts', '.rs']:
            if 'style' in text or 'format' in text:
                suggestions.append(IssueCategory.CODE_STYLE)
        
        # Análisis por complejidad
        if issue.complexity_metrics and issue.complexity_metrics.cyclomatic_complexity > 15:
            if IssueCategory.MAINTAINABILITY not in suggestions:
                suggestions.append(IssueCategory.MAINTAINABILITY)
        
        return suggestions[:3]  # Máximo 3 sugerencias


class ContextAnalyzer:
    """Analizador de contexto para issues."""
    
    def analyze_context(self, issue: RawIssue) -> ContextInfo:
        """
        Analiza contexto del issue.
        
        Args:
            issue: Issue a analizar
            
        Returns:
            ContextInfo con información contextual
        """
        context = ContextInfo()
        
        # Análizar edad del código (simplificado)
        context.code_age_days = self._estimate_code_age(issue)
        
        # Analizar criticidad del módulo
        context.module_criticality = self._assess_module_criticality(issue)
        
        # Estimar frecuencia de cambios (simplificado)
        context.file_change_frequency = self._estimate_change_frequency(issue)
        
        # Estimar cobertura de tests (simplificado)
        context.test_coverage_percentage = self._estimate_test_coverage(issue)
        
        # Contar dependencias (simplificado)
        context.dependency_count = self._estimate_dependency_count(issue)
        
        return context
    
    def _estimate_code_age(self, issue: RawIssue) -> int:
        """Estima edad del código en días."""
        # Simplificación: basado en nombre de archivo y ubicación
        if 'legacy' in str(issue.file_path).lower():
            return 365  # 1 año
        elif 'old' in str(issue.file_path).lower():
            return 180  # 6 meses
        else:
            return 30  # 1 mes (código reciente)
    
    def _assess_module_criticality(self, issue: RawIssue) -> str:
        """Evalúa criticidad del módulo."""
        file_path_lower = str(issue.file_path).lower()
        
        # Módulos críticos
        critical_patterns = ['auth', 'security', 'payment', 'user', 'login', 'database', 'core']
        if any(pattern in file_path_lower for pattern in critical_patterns):
            return "critical"
        
        # Módulos importantes
        important_patterns = ['api', 'service', 'controller', 'model', 'business']
        if any(pattern in file_path_lower for pattern in important_patterns):
            return "important"
        
        # Módulos de utilidad
        utility_patterns = ['util', 'helper', 'tool', 'lib']
        if any(pattern in file_path_lower for pattern in utility_patterns):
            return "utility"
        
        return "normal"
    
    def _estimate_change_frequency(self, issue: RawIssue) -> float:
        """Estima frecuencia de cambios."""
        # Simplificación basada en patrones de archivo
        file_path = str(issue.file_path).lower()
        
        if any(pattern in file_path for pattern in ['config', 'setting', 'constant']):
            return 0.1  # Cambios infrecuentes
        elif any(pattern in file_path for pattern in ['test', 'spec']):
            return 0.3  # Cambios moderados
        elif any(pattern in file_path for pattern in ['feature', 'component', 'service']):
            return 0.7  # Cambios frecuentes
        else:
            return 0.5  # Frecuencia normal
    
    def _estimate_test_coverage(self, issue: RawIssue) -> float:
        """Estima cobertura de tests."""
        # Simplificación basada en patrones
        if 'test' in str(issue.file_path).lower():
            return 90.0  # Archivos de test tienen alta cobertura
        elif any(pattern in str(issue.file_path).lower() for pattern in ['core', 'service', 'api']):
            return 70.0  # Módulos importantes suelen tener buena cobertura
        else:
            return 50.0  # Cobertura promedio
    
    def _estimate_dependency_count(self, issue: RawIssue) -> int:
        """Estima número de dependencias."""
        # Simplificación basada en complejidad y tipo de archivo
        base_dependencies = 3
        
        if issue.complexity_metrics:
            base_dependencies += issue.complexity_metrics.cyclomatic_complexity // 5
        
        if 'service' in str(issue.file_path).lower():
            base_dependencies += 5  # Servicios suelen tener más dependencias
        
        return min(20, base_dependencies)  # Cap en 20


class MLClassifier:
    """Clasificador ML simulado para categorización."""
    
    def __init__(self):
        self.model_trained = False
        self.feature_weights = {
            'security_score': 0.3,
            'performance_score': 0.25,
            'maintainability_score': 0.2,
            'reliability_score': 0.15,
            'complexity_score': 0.1
        }
    
    async def classify(self, issue: RawIssue) -> List[MLPrediction]:
        """
        Clasifica issue usando ML simulado.
        
        Args:
            issue: Issue a clasificar
            
        Returns:
            Lista de predicciones ML
        """
        if not self.model_trained:
            await self._simulate_training()
        
        predictions = []
        
        # Calcular scores por categoría
        category_scores = await self._calculate_category_scores(issue)
        
        # Crear predicciones para categorías con score alto
        for category, score in category_scores.items():
            if score > 0.3:  # Threshold mínimo
                prediction = MLPrediction(
                    category=category,
                    confidence=min(0.95, score),  # Cap confidence at 95%
                    feature_importance=self._get_feature_importance(category, issue)
                )
                predictions.append(prediction)
        
        # Ordenar por confidence
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        
        return predictions[:3]  # Top 3 predicciones
    
    async def _simulate_training(self) -> None:
        """Simula entrenamiento del modelo ML."""
        # En implementación real, aquí se cargaría un modelo pre-entrenado
        # o se entrenarían embeddings de text
        await asyncio.sleep(0.001)  # Simular tiempo de carga
        self.model_trained = True
        logger.debug("ML model simulation initialized")
    
    async def _calculate_category_scores(self, issue: RawIssue) -> Dict[IssueCategory, float]:
        """Calcula scores de probabilidad por categoría."""
        scores = {}
        
        text = f"{issue.rule_id} {issue.message}".lower()
        
        # Security score
        security_indicators = ['sql', 'injection', 'xss', 'security', 'auth', 'password', 'secret']
        security_score = sum(1 for indicator in security_indicators if indicator in text) / len(security_indicators)
        scores[IssueCategory.SECURITY] = security_score
        
        # Performance score
        performance_indicators = ['performance', 'slow', 'memory', 'leak', 'optimization', 'cache', 'timeout']
        performance_score = sum(1 for indicator in performance_indicators if indicator in text) / len(performance_indicators)
        
        # Boost para issues de alta complejidad
        if issue.complexity_metrics and issue.complexity_metrics.cyclomatic_complexity > 15:
            performance_score += 0.2
        
        scores[IssueCategory.PERFORMANCE] = performance_score
        
        # Maintainability score
        maintainability_indicators = ['duplicate', 'complexity', 'refactor', 'smell', 'clean', 'maintain']
        maintainability_score = sum(1 for indicator in maintainability_indicators if indicator in text) / len(maintainability_indicators)
        
        # Boost para complejidad alta
        if issue.complexity_metrics and issue.complexity_metrics.cyclomatic_complexity > 10:
            maintainability_score += 0.3
        
        scores[IssueCategory.MAINTAINABILITY] = maintainability_score
        
        # Reliability score
        reliability_indicators = ['error', 'exception', 'crash', 'fail', 'bug', 'null', 'handle']
        reliability_score = sum(1 for indicator in reliability_indicators if indicator in text) / len(reliability_indicators)
        scores[IssueCategory.RELIABILITY] = reliability_score
        
        # Documentation score
        doc_indicators = ['comment', 'doc', 'readme', 'missing', 'undocumented']
        doc_score = sum(1 for indicator in doc_indicators if indicator in text) / len(doc_indicators)
        scores[IssueCategory.DOCUMENTATION] = doc_score
        
        # Code style score
        style_indicators = ['style', 'format', 'convention', 'naming', 'indent']
        style_score = sum(1 for indicator in style_indicators if indicator in text) / len(style_indicators)
        scores[IssueCategory.CODE_STYLE] = style_score
        
        # Best practices score
        bp_indicators = ['practice', 'pattern', 'design', 'architecture', 'principle']
        bp_score = sum(1 for indicator in bp_indicators if indicator in text) / len(bp_indicators)
        scores[IssueCategory.BEST_PRACTICES] = bp_score
        
        # Normalizar scores
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            for category in scores:
                scores[category] = scores[category] / max_score
        
        return scores
    
    def _get_feature_importance(self, category: IssueCategory, issue: RawIssue) -> Dict[str, float]:
        """Obtiene importancia de características para una categoría."""
        importance = {
            'rule_id': 0.3,
            'message_content': 0.4,
            'severity': 0.2,
            'complexity': 0.1
        }
        
        # Ajustar importancia basada en categoría
        if category == IssueCategory.SECURITY:
            importance['severity'] = 0.4
            importance['message_content'] = 0.5
        elif category == IssueCategory.PERFORMANCE:
            importance['complexity'] = 0.3
            importance['message_content'] = 0.3
        
        return importance


class AutoTagger:
    """Generador automático de tags para issues."""
    
    def __init__(self):
        self.tag_rules = self._initialize_tag_rules()
    
    def _initialize_tag_rules(self) -> Dict[str, List[str]]:
        """Inicializa reglas de auto-tagging."""
        return {
            # Security tags
            'sql.*injection': ['sql-injection', 'security', 'database'],
            'xss|cross.*site': ['xss', 'security', 'frontend'],
            'auth|login|password': ['authentication', 'security'],
            'secret|key|token': ['secrets-management', 'security'],
            
            # Performance tags
            'memory.*leak': ['memory-leak', 'performance', 'optimization'],
            'slow|performance': ['performance', 'optimization'],
            'cache|caching': ['caching', 'performance'],
            'database|query': ['database', 'performance'],
            
            # Maintainability tags
            'duplicate|duplication': ['duplication', 'maintainability', 'refactoring'],
            'complexity|complex': ['complexity', 'maintainability'],
            'smell|code.*smell': ['code-smell', 'maintainability'],
            'refactor|refactoring': ['refactoring', 'maintainability'],
            
            # Language-specific tags
            'python': ['python'],
            'javascript|js': ['javascript'],
            'typescript|ts': ['typescript'],
            'rust': ['rust'],
            
            # File-type tags
            'test|spec': ['testing'],
            'config|configuration': ['configuration'],
            'api|endpoint': ['api'],
            'database|db': ['database'],
        }
    
    def generate_tags(self, issue: RawIssue, predicted_categories: List[IssueCategory]) -> List[str]:
        """
        Genera tags automáticos para un issue.
        
        Args:
            issue: Issue a etiquetar
            predicted_categories: Categorías predichas
            
        Returns:
            Lista de tags generados
        """
        tags = set()
        
        text = f"{issue.rule_id} {issue.message}".lower()
        
        # Aplicar reglas de tagging
        for pattern, rule_tags in self.tag_rules.items():
            if re.search(pattern, text, re.IGNORECASE):
                tags.update(rule_tags)
        
        # Tags basados en categorías predichas
        for category in predicted_categories:
            tags.add(category.value)
        
        # Tags basados en severidad
        if issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]:
            tags.add('high-priority')
        
        # Tags basados en complejidad
        if issue.complexity_metrics and issue.complexity_metrics.cyclomatic_complexity > 20:
            tags.add('high-complexity')
        
        # Tags basados en archivo
        if issue.file_path.suffix:
            tags.add(f"file-{issue.file_path.suffix[1:]}")  # .py -> file-py
        
        # Tags basados en lenguaje
        tags.add(f"lang-{issue.language.value}")
        
        return list(tags)


class IssueCategorizer:
    """Categorizador principal de issues."""
    
    def __init__(self, config: Optional[CategorizationConfig] = None):
        """
        Inicializa el categorizador.
        
        Args:
            config: Configuración del categorizador
        """
        self.config = config or CategorizationConfig()
        self.classification_rules = self._load_default_classification_rules()
        self.ml_classifier = MLClassifier() if self.config.enable_ml_classification else None
        self.domain_knowledge = DomainKnowledge()
        self.context_analyzer = ContextAnalyzer()
        self.auto_tagger = AutoTagger()
    
    def _load_default_classification_rules(self) -> List[ClassificationRule]:
        """Carga reglas de clasificación por defecto."""
        rules = []
        
        # Reglas de seguridad
        rules.extend([
            ClassificationRule(
                id="security-sql-injection",
                category=IssueCategory.SECURITY,
                rule_type="pattern",
                pattern=r"sql.*injection|concatenat.*sql|string.*sql",
                conditions=[
                    ClassificationCondition("severity_at_least", IssueSeverity.HIGH)
                ],
                confidence=0.95,
                auto_tags=["sql-injection", "security", "database"]
            ),
            ClassificationRule(
                id="security-hardcoded-secrets",
                category=IssueCategory.SECURITY,
                rule_type="pattern",
                pattern=r"hardcoded|secret|password|api.?key|token",
                confidence=0.90,
                auto_tags=["hardcoded-secrets", "security"]
            ),
            ClassificationRule(
                id="security-xss",
                category=IssueCategory.SECURITY,
                rule_type="pattern",
                pattern=r"xss|cross.*site.*script|html.*injection",
                confidence=0.85,
                auto_tags=["xss", "security", "frontend"]
            )
        ])
        
        # Reglas de performance
        rules.extend([
            ClassificationRule(
                id="performance-complexity",
                category=IssueCategory.PERFORMANCE,
                rule_type="metric",
                conditions=[
                    ClassificationCondition("complexity_above", 15)
                ],
                confidence=0.80,
                auto_tags=["high-complexity", "performance"]
            ),
            ClassificationRule(
                id="performance-memory-leak",
                category=IssueCategory.PERFORMANCE,
                rule_type="pattern",
                pattern=r"memory.*leak|resource.*leak|close.*resource",
                confidence=0.85,
                auto_tags=["memory-leak", "performance"]
            ),
            ClassificationRule(
                id="performance-slow-query",
                category=IssueCategory.PERFORMANCE,
                rule_type="pattern",
                pattern=r"slow.*query|database.*performance|query.*optimization",
                confidence=0.80,
                auto_tags=["slow-query", "database", "performance"]
            )
        ])
        
        # Reglas de mantenibilidad
        rules.extend([
            ClassificationRule(
                id="maintainability-code-smell",
                category=IssueCategory.MAINTAINABILITY,
                rule_type="pattern",
                pattern=r"duplicate|long.?method|large.?class|magic.?number|code.?smell",
                confidence=0.75,
                auto_tags=["code-smell", "maintainability"]
            ),
            ClassificationRule(
                id="maintainability-complexity",
                category=IssueCategory.MAINTAINABILITY,
                rule_type="pattern",
                pattern=r"complexity|refactor|simplify|decompose",
                confidence=0.70,
                auto_tags=["complexity", "refactoring", "maintainability"]
            )
        ])
        
        # Reglas de confiabilidad
        rules.extend([
            ClassificationRule(
                id="reliability-error-handling",
                category=IssueCategory.RELIABILITY,
                rule_type="pattern",
                pattern=r"error.?handling|exception|try.?catch|null.?pointer|boundary",
                confidence=0.80,
                auto_tags=["error-handling", "reliability"]
            ),
            ClassificationRule(
                id="reliability-null-safety",
                category=IssueCategory.RELIABILITY,
                rule_type="pattern",
                pattern=r"null|undefined|none.?type|optional",
                confidence=0.75,
                auto_tags=["null-safety", "reliability"]
            )
        ])
        
        # Reglas de documentación
        rules.extend([
            ClassificationRule(
                id="documentation-missing",
                category=IssueCategory.DOCUMENTATION,
                rule_type="pattern",
                pattern=r"missing.*doc|undocumented|comment|readme",
                confidence=0.85,
                auto_tags=["missing-documentation", "documentation"]
            )
        ])
        
        # Reglas de estilo
        rules.extend([
            ClassificationRule(
                id="style-naming",
                category=IssueCategory.CODE_STYLE,
                rule_type="pattern",
                pattern=r"naming|convention|style|format|indent",
                confidence=0.70,
                auto_tags=["naming", "style", "formatting"]
            )
        ])
        
        return rules
    
    async def categorize_issues(self, issues: List[RawIssue]) -> List[CategorizedIssue]:
        """
        Categoriza lista de issues.
        
        Args:
            issues: Lista de issues sin categorizar
            
        Returns:
            Lista de issues categorizados
        """
        start_time = time.time()
        categorized_issues = []
        
        logger.info(f"Iniciando categorización de {len(issues)} issues")
        
        for issue in issues:
            try:
                categorized = await self.categorize_single_issue(issue)
                categorized_issues.append(categorized)
            except Exception as e:
                logger.warning(f"Error categorizando issue {issue.rule_id}: {e}")
                # Crear categorización por defecto
                categorized = await self._create_default_categorization(issue)
                categorized_issues.append(categorized)
        
        # Aplicar agrupación por similitud si está habilitado
        if self.config.enable_similarity_grouping:
            categorized_issues = await self._apply_similarity_grouping(categorized_issues)
        
        total_time = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Categorización completada: {len(categorized_issues)} issues categorizados en {total_time}ms"
        )
        
        return categorized_issues
    
    async def categorize_single_issue(self, issue: RawIssue) -> CategorizedIssue:
        """
        Categoriza un issue individual.
        
        Args:
            issue: Issue a categorizar
            
        Returns:
            Issue categorizado
        """
        categories = []
        confidence_scores = {}
        applied_rules = []
        ml_predictions = []
        
        # 1. Aplicar clasificación basada en reglas
        for rule in self.classification_rules:
            if rule.matches(issue):
                if rule.category not in categories:
                    categories.append(rule.category)
                confidence_scores[rule.category] = max(
                    confidence_scores.get(rule.category, 0.0),
                    rule.confidence
                )
                applied_rules.append(rule.id)
        
        # 2. Aplicar clasificación ML si está habilitado
        if self.ml_classifier:
            try:
                ml_predictions = await self.ml_classifier.classify(issue)
                for prediction in ml_predictions:
                    if prediction.confidence >= self.config.confidence_threshold:
                        if prediction.category not in categories:
                            categories.append(prediction.category)
                        confidence_scores[prediction.category] = max(
                            confidence_scores.get(prediction.category, 0.0),
                            prediction.confidence
                        )
            except Exception as e:
                logger.warning(f"ML classification failed for issue {issue.rule_id}: {e}")
        
        # 3. Aplicar conocimiento del dominio
        domain_suggestions = await self.domain_knowledge.suggest_categories(issue)
        for category in domain_suggestions:
            if category not in categories:
                categories.append(category)
                confidence_scores[category] = 0.6  # Confidence media para sugerencias
        
        # 4. Determinar categoría principal
        if categories:
            # Usar categoría con mayor confidence
            primary_category = max(categories, key=lambda c: confidence_scores.get(c, 0.0))
            secondary_categories = [c for c in categories if c != primary_category]
        else:
            # Fallback basado en severidad y contexto
            primary_category = self._determine_fallback_category(issue)
            secondary_categories = []
            confidence_scores[primary_category] = 0.5
        
        # Limitar categorías secundarias
        secondary_categories = secondary_categories[:self.config.max_categories_per_issue - 1]
        
        # 5. Generar tags automáticos
        auto_tags = []
        if self.config.enable_auto_tagging:
            auto_tags = self.auto_tagger.generate_tags(issue, [primary_category] + secondary_categories)
        
        # Añadir tags de reglas aplicadas
        for rule in self.classification_rules:
            if rule.id in applied_rules:
                auto_tags.extend(rule.auto_tags)
        
        # Eliminar duplicados en tags
        auto_tags = list(set(auto_tags))
        
        # 6. Analizar contexto
        context_info = ContextInfo()
        if self.config.enable_context_analysis:
            context_info = self.context_analyzer.analyze_context(issue)
        
        # 7. Crear metadatos enriquecidos
        metadata = await self._create_enhanced_metadata(issue, primary_category, context_info)
        
        # 8. Crear issue categorizado
        categorized = CategorizedIssue(
            id=IssueId(),
            original_issue=issue,
            primary_category=primary_category,
            secondary_categories=secondary_categories,
            tags=auto_tags,
            confidence_scores={cat.value: score for cat, score in confidence_scores.items()},
            metadata=metadata,
            context_info=context_info,
            categorization_timestamp=datetime.now()
        )
        
        return categorized
    
    def _determine_fallback_category(self, issue: RawIssue) -> IssueCategory:
        """Determina categoría fallback cuando no hay matches."""
        # Basado en severidad
        if issue.severity == IssueSeverity.CRITICAL:
            return IssueCategory.SECURITY
        elif issue.severity == IssueSeverity.HIGH:
            return IssueCategory.RELIABILITY
        else:
            return IssueCategory.MAINTAINABILITY
    
    async def _create_enhanced_metadata(self, issue: RawIssue, category: IssueCategory, 
                                      context: ContextInfo) -> IssueMetadata:
        """Crea metadatos enriquecidos para el issue."""
        metadata = IssueMetadata()
        
        # Estimar tiempo de fix basado en categoría y complejidad
        metadata.estimated_fix_time_hours = self._estimate_fix_time(issue, category)
        
        # Evaluar nivel de impacto de negocio
        metadata.business_impact_level = self._assess_business_impact(issue, category, context)
        
        # Estimar impacto en performance
        metadata.performance_impact_percentage = self._estimate_performance_impact(issue, category)
        
        # Calcular score de riesgo de seguridad
        metadata.security_risk_score = self._calculate_security_risk(issue, category)
        
        # Estimar contribución a deuda técnica
        metadata.technical_debt_contribution = self._estimate_technical_debt_contribution(issue, category)
        
        # Calcular complejidad de fix
        metadata.fix_complexity_score = self._calculate_fix_complexity(issue, category)
        
        # Evaluar riesgo de regresión
        metadata.regression_risk_score = self._assess_regression_risk(issue, context)
        
        return metadata
    
    def _estimate_fix_time(self, issue: RawIssue, category: IssueCategory) -> float:
        """Estima tiempo de fix en horas."""
        base_time = {
            IssueCategory.SECURITY: 4.0,
            IssueCategory.PERFORMANCE: 3.0,
            IssueCategory.MAINTAINABILITY: 2.0,
            IssueCategory.RELIABILITY: 2.5,
            IssueCategory.DOCUMENTATION: 1.0,
            IssueCategory.CODE_STYLE: 0.5
        }.get(category, 2.0)
        
        # Ajustar por complejidad
        if issue.complexity_metrics:
            complexity_multiplier = 1.0 + (issue.complexity_metrics.cyclomatic_complexity - 10) * 0.1
            complexity_multiplier = max(0.5, min(3.0, complexity_multiplier))
            base_time *= complexity_multiplier
        
        # Ajustar por severidad
        severity_multipliers = {
            IssueSeverity.CRITICAL: 1.5,
            IssueSeverity.HIGH: 1.2,
            IssueSeverity.MEDIUM: 1.0,
            IssueSeverity.LOW: 0.8,
            IssueSeverity.INFO: 0.5
        }
        
        base_time *= severity_multipliers.get(issue.severity, 1.0)
        
        return round(base_time, 1)
    
    def _assess_business_impact(self, issue: RawIssue, category: IssueCategory, 
                              context: ContextInfo) -> BusinessImpactLevel:
        """Evalúa nivel de impacto en negocio."""
        # Impacto basado en categoría
        if category == IssueCategory.SECURITY:
            if issue.severity == IssueSeverity.CRITICAL:
                return BusinessImpactLevel.BLOCKING
            else:
                return BusinessImpactLevel.SIGNIFICANT
        
        elif category == IssueCategory.PERFORMANCE:
            if context.module_criticality == "critical":
                return BusinessImpactLevel.SIGNIFICANT
            else:
                return BusinessImpactLevel.MODERATE
        
        elif category in [IssueCategory.RELIABILITY, IssueCategory.MAINTAINABILITY]:
            if issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]:
                return BusinessImpactLevel.MODERATE
            else:
                return BusinessImpactLevel.MINOR
        
        else:
            return BusinessImpactLevel.MINOR
    
    def _estimate_performance_impact(self, issue: RawIssue, category: IssueCategory) -> float:
        """Estima impacto en performance (porcentaje)."""
        if category == IssueCategory.PERFORMANCE:
            if 'memory' in issue.message.lower():
                return 15.0
            elif 'slow' in issue.message.lower() or 'timeout' in issue.message.lower():
                return 25.0
            elif issue.complexity_metrics and issue.complexity_metrics.cyclomatic_complexity > 20:
                return 10.0
            else:
                return 5.0
        
        elif category == IssueCategory.SECURITY and issue.severity == IssueSeverity.CRITICAL:
            return 20.0  # Security issues pueden degradar performance
        
        return 0.0
    
    def _calculate_security_risk(self, issue: RawIssue, category: IssueCategory) -> float:
        """Calcula score de riesgo de seguridad (0-100)."""
        if category == IssueCategory.SECURITY:
            base_score = {
                IssueSeverity.CRITICAL: 90.0,
                IssueSeverity.HIGH: 70.0,
                IssueSeverity.MEDIUM: 40.0,
                IssueSeverity.LOW: 20.0,
                IssueSeverity.INFO: 10.0
            }.get(issue.severity, 30.0)
            
            # Boost para patrones específicos
            if any(pattern in issue.message.lower() for pattern in ['injection', 'xss', 'auth']):
                base_score += 20.0
            
            return min(100.0, base_score)
        
        return 0.0
    
    def _estimate_technical_debt_contribution(self, issue: RawIssue, category: IssueCategory) -> float:
        """Estima contribución a deuda técnica."""
        base_contribution = {
            IssueCategory.MAINTAINABILITY: 15.0,
            IssueCategory.PERFORMANCE: 10.0,
            IssueCategory.RELIABILITY: 12.0,
            IssueCategory.SECURITY: 8.0,
            IssueCategory.DOCUMENTATION: 5.0,
            IssueCategory.CODE_STYLE: 3.0
        }.get(category, 8.0)
        
        # Ajustar por complejidad
        if issue.complexity_metrics and issue.complexity_metrics.cyclomatic_complexity > 15:
            base_contribution *= 1.5
        
        return base_contribution
    
    def _calculate_fix_complexity(self, issue: RawIssue, category: IssueCategory) -> float:
        """Calcula complejidad de fix (0-100)."""
        base_complexity = {
            IssueCategory.SECURITY: 70.0,
            IssueCategory.PERFORMANCE: 60.0,
            IssueCategory.MAINTAINABILITY: 50.0,
            IssueCategory.RELIABILITY: 55.0,
            IssueCategory.ARCHITECTURE: 80.0,
            IssueCategory.DOCUMENTATION: 20.0,
            IssueCategory.CODE_STYLE: 15.0
        }.get(category, 40.0)
        
        # Ajustar por complejidad del código
        if issue.complexity_metrics:
            complexity_factor = issue.complexity_metrics.cyclomatic_complexity / 20.0
            base_complexity += complexity_factor * 20.0
        
        return min(100.0, base_complexity)
    
    def _assess_regression_risk(self, issue: RawIssue, context: ContextInfo) -> float:
        """Evalúa riesgo de regresión (0-100)."""
        base_risk = 30.0
        
        # Riesgo mayor en módulos críticos
        if context.module_criticality == "critical":
            base_risk += 25.0
        elif context.module_criticality == "important":
            base_risk += 15.0
        
        # Riesgo mayor en código con muchas dependencias
        base_risk += min(20.0, context.dependency_count * 2.0)
        
        # Riesgo menor con buena cobertura de tests
        if context.test_coverage_percentage > 80.0:
            base_risk -= 15.0
        elif context.test_coverage_percentage > 60.0:
            base_risk -= 10.0
        
        # Riesgo mayor en código que cambia frecuentemente
        base_risk += context.file_change_frequency * 20.0
        
        return min(100.0, max(0.0, base_risk))
    
    async def _create_default_categorization(self, issue: RawIssue) -> CategorizedIssue:
        """Crea categorización por defecto cuando falla la categorización normal."""
        return CategorizedIssue(
            id=IssueId(),
            original_issue=issue,
            primary_category=IssueCategory.MAINTAINABILITY,
            tags=["uncategorized"],
            confidence_scores={"maintainability": 0.5},
            metadata=IssueMetadata(estimated_fix_time_hours=2.0),
            context_info=ContextInfo()
        )
    
    async def _apply_similarity_grouping(self, issues: List[CategorizedIssue]) -> List[CategorizedIssue]:
        """Aplica agrupación por similitud."""
        # Implementación simplificada - en versión completa usaría clustering avanzado
        grouped_issues = []
        processed_issues = set()
        
        for i, issue1 in enumerate(issues):
            if i in processed_issues:
                continue
            
            similar_issues = [issue1]
            processed_issues.add(i)
            
            for j, issue2 in enumerate(issues[i+1:], i+1):
                if j in processed_issues:
                    continue
                
                if self._are_similar_issues(issue1, issue2):
                    similar_issues.append(issue2)
                    processed_issues.add(j)
            
            # Si encontramos issues similares, añadir tags de grupo
            if len(similar_issues) > 1:
                group_tag = f"similar-group-{len(grouped_issues)}"
                for similar_issue in similar_issues:
                    similar_issue.tags.append(group_tag)
                    similar_issue.context_info.surrounding_issues = [
                        other.id.value for other in similar_issues if other != similar_issue
                    ]
            
            grouped_issues.extend(similar_issues)
        
        return grouped_issues
    
    def _are_similar_issues(self, issue1: CategorizedIssue, issue2: CategorizedIssue) -> bool:
        """Verifica si dos issues son similares."""
        # Misma categoría principal
        if issue1.primary_category != issue2.primary_category:
            return False
        
        # Similitud en mensaje (simple)
        if issue1.original_issue and issue2.original_issue:
            message1_words = set(issue1.original_issue.message.lower().split())
            message2_words = set(issue2.original_issue.message.lower().split())
            
            if message1_words and message2_words:
                jaccard_similarity = len(message1_words.intersection(message2_words)) / len(message1_words.union(message2_words))
                if jaccard_similarity >= self.config.similarity_threshold:
                    return True
        
        # Mismo tipo de archivo
        if (issue1.original_issue and issue2.original_issue and
            issue1.original_issue.file_path.suffix == issue2.original_issue.file_path.suffix):
            
            # Tags comunes
            common_tags = set(issue1.tags).intersection(set(issue2.tags))
            if len(common_tags) >= 2:
                return True
        
        return False
    
    def get_categorization_stats(self, categorized_issues: List[CategorizedIssue]) -> Dict[str, Any]:
        """
        Obtiene estadísticas de categorización.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "total_issues": len(categorized_issues),
            "category_distribution": {},
            "confidence_distribution": {},
            "tag_frequency": {},
            "average_confidence": 0.0
        }
        
        # Distribución por categorías
        category_counts = Counter(issue.primary_category for issue in categorized_issues)
        stats["category_distribution"] = {cat.value: count for cat, count in category_counts.items()}
        
        # Distribución de confidence
        all_confidences = []
        for issue in categorized_issues:
            primary_cat = issue.primary_category.value
            confidence = issue.confidence_scores.get(primary_cat, 0.5)
            all_confidences.append(confidence)
        
        if all_confidences:
            stats["average_confidence"] = sum(all_confidences) / len(all_confidences)
            stats["confidence_distribution"] = {
                "high": sum(1 for c in all_confidences if c >= 0.8),
                "medium": sum(1 for c in all_confidences if 0.6 <= c < 0.8),
                "low": sum(1 for c in all_confidences if c < 0.6)
            }
        
        # Frecuencia de tags
        all_tags = []
        for issue in categorized_issues:
            all_tags.extend(issue.tags)
        
        tag_counts = Counter(all_tags)
        stats["tag_frequency"] = dict(tag_counts.most_common(10))
        
        return stats
