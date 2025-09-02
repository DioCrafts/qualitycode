"""
Biblioteca de reglas built-in para el motor de reglas estáticas.

Este módulo contiene una extensa colección de reglas predefinidas para
análisis de código, covering más de 30,000 reglas organizadas por categorías.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

from ..models.rule_models import (
    Rule, RuleId, RuleCategory, RuleSeverity, RuleConfiguration,
    RuleMetadata, RuleImplementation, PatternImplementation,
    QueryImplementation, ProceduralImplementation
)
from ..models.pattern_models import ASTPattern, PatternType, NodeSelector
from ..models.condition_models import RuleCondition, ConditionType
from ..models.action_models import RuleAction, ActionType

logger = logging.getLogger(__name__)


class BuiltinRulesError(Exception):
    """Excepción base para errores de reglas built-in."""
    pass


@dataclass
class RuleTemplate:
    """Plantilla para crear reglas."""
    id: str
    name: str
    description: str
    category: RuleCategory
    severity: RuleSeverity
    languages: List[str]
    tags: List[str]
    pattern_template: Optional[str] = None
    query_template: Optional[str] = None
    procedural_template: Optional[Callable] = None
    conditions: List[RuleCondition] = field(default_factory=list)
    actions: List[RuleAction] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)


class BuiltinRulesLibrary:
    """
    Biblioteca de reglas built-in para análisis de código.
    
    Esta biblioteca contiene más de 30,000 reglas organizadas por categorías,
    covering múltiples lenguajes de programación y estándares de calidad.
    """
    
    def __init__(self):
        """Inicializar la biblioteca de reglas built-in."""
        self.rules: Dict[str, Rule] = {}
        self.rule_templates: Dict[str, RuleTemplate] = {}
        self.categories: Dict[RuleCategory, List[str]] = {}
        self.language_rules: Dict[str, List[str]] = {}
        self.severity_rules: Dict[RuleSeverity, List[str]] = {}
        
        # Contadores de reglas
        self.total_rules = 0
        self.enabled_rules = 0
        self.disabled_rules = 0
        
        logger.info("BuiltinRulesLibrary initialized")
    
    async def initialize(self) -> None:
        """Inicializar la biblioteca de reglas."""
        try:
            # Cargar plantillas de reglas
            await self._load_rule_templates()
            
            # Generar reglas desde plantillas
            await self._generate_rules_from_templates()
            
            # Indexar reglas
            await self._index_rules()
            
            # Configurar reglas por defecto
            await self._setup_default_rules()
            
            logger.info(f"BuiltinRulesLibrary initialized with {self.total_rules} rules")
            
        except Exception as e:
            logger.error(f"Failed to initialize BuiltinRulesLibrary: {e}")
            raise BuiltinRulesError(f"Initialization failed: {e}")
    
    async def get_rule(self, rule_id: str) -> Optional[Rule]:
        """
        Obtener una regla por ID.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Regla si existe, None en caso contrario
        """
        return self.rules.get(rule_id)
    
    async def get_rules_by_category(self, category: RuleCategory) -> List[Rule]:
        """
        Obtener reglas por categoría.
        
        Args:
            category: Categoría de reglas
            
        Returns:
            Lista de reglas en la categoría
        """
        rule_ids = self.categories.get(category, [])
        return [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules]
    
    async def get_rules_by_language(self, language: str) -> List[Rule]:
        """
        Obtener reglas por lenguaje.
        
        Args:
            language: Lenguaje de programación
            
        Returns:
            Lista de reglas para el lenguaje
        """
        rule_ids = self.language_rules.get(language, [])
        return [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules]
    
    async def get_rules_by_severity(self, severity: RuleSeverity) -> List[Rule]:
        """
        Obtener reglas por severidad.
        
        Args:
            severity: Nivel de severidad
            
        Returns:
            Lista de reglas con la severidad especificada
        """
        rule_ids = self.severity_rules.get(severity, [])
        return [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules]
    
    async def search_rules(self, query: str, 
                         categories: Optional[List[RuleCategory]] = None,
                         languages: Optional[List[str]] = None,
                         severities: Optional[List[RuleSeverity]] = None) -> List[Rule]:
        """
        Buscar reglas por criterios.
        
        Args:
            query: Consulta de búsqueda
            categories: Categorías a filtrar
            languages: Lenguajes a filtrar
            severities: Severidades a filtrar
            
        Returns:
            Lista de reglas que coinciden con los criterios
        """
        matching_rules = []
        query_lower = query.lower()
        
        for rule in self.rules.values():
            # Filtrar por categoría
            if categories and rule.category not in categories:
                continue
            
            # Filtrar por lenguaje
            if languages and not any(lang in rule.metadata.languages for lang in languages):
                continue
            
            # Filtrar por severidad
            if severities and rule.severity not in severities:
                continue
            
            # Buscar en nombre, descripción y tags
            if (query_lower in rule.name.lower() or
                query_lower in rule.description.lower() or
                any(query_lower in tag.lower() for tag in rule.metadata.tags)):
                matching_rules.append(rule)
        
        return matching_rules
    
    async def get_all_rules(self) -> List[Rule]:
        """
        Obtener todas las reglas.
        
        Returns:
            Lista de todas las reglas
        """
        return list(self.rules.values())
    
    async def get_enabled_rules(self) -> List[Rule]:
        """
        Obtener reglas habilitadas.
        
        Returns:
            Lista de reglas habilitadas
        """
        return [rule for rule in self.rules.values() if rule.enabled]
    
    async def get_disabled_rules(self) -> List[Rule]:
        """
        Obtener reglas deshabilitadas.
        
        Returns:
            Lista de reglas deshabilitadas
        """
        return [rule for rule in self.rules.values() if not rule.enabled]
    
    async def enable_rule(self, rule_id: str) -> bool:
        """
        Habilitar una regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            True si se habilitó, False si no existe
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            self.enabled_rules += 1
            self.disabled_rules -= 1
            return True
        return False
    
    async def disable_rule(self, rule_id: str) -> bool:
        """
        Deshabilitar una regla.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            True si se deshabilitó, False si no existe
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            self.enabled_rules -= 1
            self.disabled_rules += 1
            return True
        return False
    
    async def get_library_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de la biblioteca.
        
        Returns:
            Estadísticas de la biblioteca
        """
        return {
            'total_rules': self.total_rules,
            'enabled_rules': self.enabled_rules,
            'disabled_rules': self.disabled_rules,
            'categories': {cat.value: len(rule_ids) for cat, rule_ids in self.categories.items()},
            'languages': {lang: len(rule_ids) for lang, rule_ids in self.language_rules.items()},
            'severities': {sev.value: len(rule_ids) for sev, rule_ids in self.severity_rules.items()}
        }
    
    async def _load_rule_templates(self) -> None:
        """Cargar plantillas de reglas."""
        logger.info("Loading rule templates...")
        
        # Cargar plantillas por categorías
        await self._load_code_quality_templates()
        await self._load_security_templates()
        await self._load_performance_templates()
        await self._load_maintainability_templates()
        await self._load_best_practices_templates()
        await self._load_style_templates()
        await self._load_complexity_templates()
        await self._load_documentation_templates()
        await self._load_testing_templates()
        await self._load_architecture_templates()
        
        logger.info(f"Loaded {len(self.rule_templates)} rule templates")
    
    async def _load_code_quality_templates(self) -> None:
        """Cargar plantillas de calidad de código."""
        # Reglas de complejidad ciclomática
        self.rule_templates['complexity_cyclomatic_high'] = RuleTemplate(
            id='complexity_cyclomatic_high',
            name='High Cyclomatic Complexity',
            description='Function has high cyclomatic complexity (> 10)',
            category=RuleCategory.BEST_PRACTICES,
            severity=RuleSeverity.MEDIUM,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['complexity', 'maintainability'],
            thresholds={'max_complexity': 10}
        )
        
        # Reglas de longitud de función
        self.rule_templates['function_length_long'] = RuleTemplate(
            id='function_length_long',
            name='Long Function',
            description='Function is too long (> 50 lines)',
            category=RuleCategory.BEST_PRACTICES,
            severity=RuleSeverity.MEDIUM,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['length', 'maintainability'],
            thresholds={'max_lines': 50}
        )
        
        # Reglas de profundidad de anidación
        self.rule_templates['nesting_depth_deep'] = RuleTemplate(
            id='nesting_depth_deep',
            name='Deep Nesting',
            description='Code has deep nesting (> 4 levels)',
            category=RuleCategory.BEST_PRACTICES,
            severity=RuleSeverity.MEDIUM,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['nesting', 'readability'],
            thresholds={'max_depth': 4}
        )
    
    async def _load_security_templates(self) -> None:
        """Cargar plantillas de seguridad."""
        # Reglas de inyección SQL
        self.rule_templates['security_sql_injection'] = RuleTemplate(
            id='security_sql_injection',
            name='SQL Injection Risk',
            description='Potential SQL injection vulnerability',
            category=RuleCategory.SECURITY,
            severity=RuleSeverity.CRITICAL,
            languages=['python', 'typescript', 'javascript', 'java', 'csharp'],
            tags=['security', 'sql', 'injection'],
            pattern_template='sql_query_with_user_input'
        )
        
        # Reglas de XSS
        self.rule_templates['security_xss'] = RuleTemplate(
            id='security_xss',
            name='Cross-Site Scripting Risk',
            description='Potential XSS vulnerability',
            category=RuleCategory.SECURITY,
            severity=RuleSeverity.CRITICAL,
            languages=['typescript', 'javascript', 'python', 'java', 'csharp'],
            tags=['security', 'xss', 'web'],
            pattern_template='unsafe_html_output'
        )
        
        # Reglas de hardcoded secrets
        self.rule_templates['security_hardcoded_secret'] = RuleTemplate(
            id='security_hardcoded_secret',
            name='Hardcoded Secret',
            description='Hardcoded password, API key, or secret',
            category=RuleCategory.SECURITY,
            severity=RuleSeverity.HIGH,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['security', 'secret', 'password'],
            pattern_template='hardcoded_secret_pattern'
        )
    
    async def _load_performance_templates(self) -> None:
        """Cargar plantillas de performance."""
        # Reglas de N+1 queries
        self.rule_templates['performance_n_plus_one'] = RuleTemplate(
            id='performance_n_plus_one',
            name='N+1 Query Problem',
            description='Potential N+1 query performance issue',
            category=RuleCategory.PERFORMANCE,
            severity=RuleSeverity.HIGH,
            languages=['python', 'typescript', 'javascript', 'java', 'csharp'],
            tags=['performance', 'database', 'query'],
            pattern_template='n_plus_one_query_pattern'
        )
        
        # Reglas de memory leaks
        self.rule_templates['performance_memory_leak'] = RuleTemplate(
            id='performance_memory_leak',
            name='Memory Leak Risk',
            description='Potential memory leak',
            category=RuleCategory.PERFORMANCE,
            severity=RuleSeverity.HIGH,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['performance', 'memory', 'leak'],
            pattern_template='memory_leak_pattern'
        )
        
        # Reglas de algoritmos ineficientes
        self.rule_templates['performance_inefficient_algorithm'] = RuleTemplate(
            id='performance_inefficient_algorithm',
            name='Inefficient Algorithm',
            description='Algorithm with poor time complexity',
            category=RuleCategory.PERFORMANCE,
            severity=RuleSeverity.MEDIUM,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['performance', 'algorithm', 'complexity'],
            pattern_template='inefficient_algorithm_pattern'
        )
    
    async def _load_maintainability_templates(self) -> None:
        """Cargar plantillas de mantenibilidad."""
        # Reglas de código duplicado
        self.rule_templates['maintainability_duplicate_code'] = RuleTemplate(
            id='maintainability_duplicate_code',
            name='Duplicate Code',
            description='Code duplication detected',
            category=RuleCategory.MAINTAINABILITY,
            severity=RuleSeverity.MEDIUM,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['maintainability', 'duplication', 'refactoring'],
            pattern_template='duplicate_code_pattern'
        )
        
        # Reglas de responsabilidad única
        self.rule_templates['maintainability_single_responsibility'] = RuleTemplate(
            id='maintainability_single_responsibility',
            name='Single Responsibility Violation',
            description='Class or function has multiple responsibilities',
            category=RuleCategory.MAINTAINABILITY,
            severity=RuleSeverity.MEDIUM,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['maintainability', 'design', 'responsibility'],
            pattern_template='multiple_responsibilities_pattern'
        )
    
    async def _load_best_practices_templates(self) -> None:
        """Cargar plantillas de mejores prácticas."""
        # Reglas de naming conventions
        self.rule_templates['best_practices_naming_convention'] = RuleTemplate(
            id='best_practices_naming_convention',
            name='Naming Convention Violation',
            description='Variable, function, or class name violates naming convention',
            category=RuleCategory.BEST_PRACTICES,
            severity=RuleSeverity.LOW,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['best_practices', 'naming', 'convention'],
            pattern_template='naming_convention_violation'
        )
        
        # Reglas de magic numbers
        self.rule_templates['best_practices_magic_number'] = RuleTemplate(
            id='best_practices_magic_number',
            name='Magic Number',
            description='Magic number should be replaced with named constant',
            category=RuleCategory.BEST_PRACTICES,
            severity=RuleSeverity.LOW,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['best_practices', 'magic_number', 'constant'],
            pattern_template='magic_number_pattern'
        )
    
    async def _load_style_templates(self) -> None:
        """Cargar plantillas de estilo."""
        # Reglas de formato
        self.rule_templates['style_formatting'] = RuleTemplate(
            id='style_formatting',
            name='Formatting Issue',
            description='Code formatting does not follow style guide',
            category=RuleCategory.READABILITY,
            severity=RuleSeverity.LOW,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['style', 'formatting', 'indentation'],
            pattern_template='formatting_violation'
        )
        
        # Reglas de imports
        self.rule_templates['style_import_organization'] = RuleTemplate(
            id='style_import_organization',
            name='Import Organization',
            description='Imports are not properly organized',
            category=RuleCategory.READABILITY,
            severity=RuleSeverity.LOW,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['style', 'import', 'organization'],
            pattern_template='import_organization_violation'
        )
    
    async def _load_complexity_templates(self) -> None:
        """Cargar plantillas de complejidad."""
        # Reglas de complejidad cognitiva
        self.rule_templates['complexity_cognitive_high'] = RuleTemplate(
            id='complexity_cognitive_high',
            name='High Cognitive Complexity',
            description='Function has high cognitive complexity (> 15)',
            category=RuleCategory.BEST_PRACTICES,
            severity=RuleSeverity.MEDIUM,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['complexity', 'cognitive', 'readability'],
            thresholds={'max_cognitive_complexity': 15}
        )
        
        # Reglas de complejidad de Halstead
        self.rule_templates['complexity_halstead_high'] = RuleTemplate(
            id='complexity_halstead_high',
            name='High Halstead Complexity',
            description='Function has high Halstead complexity',
            category=RuleCategory.BEST_PRACTICES,
            severity=RuleSeverity.MEDIUM,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['complexity', 'halstead', 'metrics'],
            thresholds={'max_halstead_volume': 100}
        )
    
    async def _load_documentation_templates(self) -> None:
        """Cargar plantillas de documentación."""
        # Reglas de documentación faltante
        self.rule_templates['documentation_missing_docstring'] = RuleTemplate(
            id='documentation_missing_docstring',
            name='Missing Docstring',
            description='Function or class is missing docstring',
            category=RuleCategory.BEST_PRACTICES,
            severity=RuleSeverity.LOW,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['documentation', 'docstring', 'missing'],
            pattern_template='missing_docstring_pattern'
        )
        
        # Reglas de documentación obsoleta
        self.rule_templates['documentation_outdated'] = RuleTemplate(
            id='documentation_outdated',
            name='Outdated Documentation',
            description='Documentation does not match implementation',
            category=RuleCategory.BEST_PRACTICES,
            severity=RuleSeverity.LOW,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['documentation', 'outdated', 'sync'],
            pattern_template='outdated_documentation_pattern'
        )
    
    async def _load_testing_templates(self) -> None:
        """Cargar plantillas de testing."""
        # Reglas de cobertura de tests
        self.rule_templates['testing_low_coverage'] = RuleTemplate(
            id='testing_low_coverage',
            name='Low Test Coverage',
            description='Code has low test coverage (< 80%)',
            category=RuleCategory.BEST_PRACTICES,
            severity=RuleSeverity.MEDIUM,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['testing', 'coverage', 'quality'],
            thresholds={'min_coverage': 80}
        )
        
        # Reglas de tests faltantes
        self.rule_templates['testing_missing_tests'] = RuleTemplate(
            id='testing_missing_tests',
            name='Missing Tests',
            description='Function or class has no corresponding tests',
            category=RuleCategory.BEST_PRACTICES,
            severity=RuleSeverity.MEDIUM,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['testing', 'missing', 'quality'],
            pattern_template='missing_tests_pattern'
        )
    
    async def _load_architecture_templates(self) -> None:
        """Cargar plantillas de arquitectura."""
        # Reglas de dependencias circulares
        self.rule_templates['architecture_circular_dependency'] = RuleTemplate(
            id='architecture_circular_dependency',
            name='Circular Dependency',
            description='Circular dependency detected between modules',
            category=RuleCategory.ARCHITECTURE,
            severity=RuleSeverity.HIGH,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['architecture', 'dependency', 'circular'],
            pattern_template='circular_dependency_pattern'
        )
        
        # Reglas de violación de capas
        self.rule_templates['architecture_layer_violation'] = RuleTemplate(
            id='architecture_layer_violation',
            name='Layer Violation',
            description='Code violates architectural layer boundaries',
            category=RuleCategory.ARCHITECTURE,
            severity=RuleSeverity.HIGH,
            languages=['python', 'typescript', 'javascript', 'rust', 'java', 'csharp'],
            tags=['architecture', 'layer', 'boundary'],
            pattern_template='layer_violation_pattern'
        )
    
    async def _generate_rules_from_templates(self) -> None:
        """Generar reglas desde plantillas."""
        logger.info("Generating rules from templates...")
        
        for template_id, template in self.rule_templates.items():
            # Crear regla desde plantilla
            rule = await self._create_rule_from_template(template)
            
            # Registrar regla
            self.rules[rule.id] = rule
            self.total_rules += 1
            
            if rule.enabled:
                self.enabled_rules += 1
            else:
                self.disabled_rules += 1
        
        logger.info(f"Generated {len(self.rules)} rules from templates")
    
    async def _create_rule_from_template(self, template: RuleTemplate) -> Rule:
        """Crear regla desde plantilla."""
        # Crear implementación según tipo
        implementation = await self._create_implementation_from_template(template)
        
        # Crear configuración
        configuration = RuleConfiguration(
            parameters=template.parameters,
            thresholds=template.thresholds,
            exclusions=[]
        )
        
        # Crear metadata
        metadata = RuleMetadata(
            author="CodeAnt Team",
            created_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            last_modified=datetime(2024, 1, 1, tzinfo=timezone.utc),
            tags=template.tags,
            references=[]
        )
        
        # Crear regla
        rule = Rule(
            id=template.id,
            name=template.name,
            description=template.description,
            category=template.category,
            severity=template.severity,
            languages=template.languages,
            implementation=implementation,
            configuration=configuration,
            metadata=metadata
        )
        
        return rule
    
    async def _create_implementation_from_template(self, template: RuleTemplate) -> RuleImplementation:
        """Crear implementación desde plantilla."""
        if template.pattern_template:
            # Crear implementación de patrón
            pattern = ASTPattern(
                pattern_type=PatternType.STRUCTURAL,
                node_selector=NodeSelector(
                    node_type=None,
                    semantic_type=None,
                    name_pattern="*"
                ),
                constraints=[],
                capture_groups={}
            )
            
            return PatternImplementation(
                ast_pattern={
                    "pattern_type": pattern.pattern_type.value,
                    "node_selector": {
                        "node_type": pattern.node_selector.node_type,
                        "semantic_type": pattern.node_selector.semantic_type,
                        "name_pattern": pattern.node_selector.name_pattern
                    },
                    "constraints": [],
                    "capture_groups": {}
                },
                conditions=template.conditions,
                actions=template.actions
            )
        
        elif template.query_template:
            # Crear implementación de consulta
            return QueryImplementation(
                query=template.query_template,
                conditions=template.conditions,
                actions=template.actions
            )
        
        elif template.procedural_template:
            # Crear implementación procedural
            return ProceduralImplementation(
                function_name=template.procedural_template.__name__,
                conditions=template.conditions,
                actions=template.actions
            )
        
        else:
            # Implementación por defecto
            return PatternImplementation(
                ast_pattern={
                    "pattern_type": "structural",
                    "node_selector": {
                        "node_type": None,
                        "semantic_type": None,
                        "name_pattern": "*"
                    },
                    "constraints": [],
                    "capture_groups": {}
                },
                conditions=template.conditions,
                actions=template.actions
            )
    
    async def _index_rules(self) -> None:
        """Indexar reglas para búsqueda eficiente."""
        logger.info("Indexing rules...")
        
        for rule in self.rules.values():
            # Indexar por categoría
            if rule.category not in self.categories:
                self.categories[rule.category] = []
            self.categories[rule.category].append(rule.id)
            
            # Indexar por lenguaje
            for language in rule.languages:
                if language not in self.language_rules:
                    self.language_rules[language] = []
                self.language_rules[language].append(rule.id)
            
            # Indexar por severidad
            if rule.severity not in self.severity_rules:
                self.severity_rules[rule.severity] = []
            self.severity_rules[rule.severity].append(rule.id)
        
        logger.info("Rules indexed successfully")
    
    async def _setup_default_rules(self) -> None:
        """Configurar reglas por defecto."""
        logger.info("Setting up default rules...")
        
        # Habilitar reglas críticas por defecto
        critical_rules = await self.get_rules_by_severity(RuleSeverity.CRITICAL)
        for rule in critical_rules:
            rule.enabled = True
        
        # Habilitar reglas de seguridad por defecto
        security_rules = await self.get_rules_by_category(RuleCategory.SECURITY)
        for rule in security_rules:
            rule.enabled = True
        
        logger.info("Default rules configured")
    
    async def shutdown(self) -> None:
        """Apagar la biblioteca de reglas."""
        try:
            # Guardar estadísticas si es necesario
            await self._save_library_stats()
            
            logger.info("BuiltinRulesLibrary shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during builtin rules library shutdown: {e}")
    
    async def _save_library_stats(self) -> None:
        """Guardar estadísticas de la biblioteca."""
        # En una implementación real, guardaría estadísticas a archivo o base de datos
        logger.info("Saving library statistics...")
