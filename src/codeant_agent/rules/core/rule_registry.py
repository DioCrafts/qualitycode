"""
Registro de reglas del motor de reglas estáticas.

Este módulo implementa el registro centralizado de reglas, proporcionando
funcionalidades para registrar, buscar y gestionar reglas de análisis.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field

from ..models.rule_models import Rule, RuleId, RuleCategory, RuleSeverity
from ...parsers.universal import ProgrammingLanguage

logger = logging.getLogger(__name__)


class RuleRegistryError(Exception):
    """Excepción base para errores del registro de reglas."""
    pass


class RuleNotFoundError(RuleRegistryError):
    """Error cuando no se encuentra una regla."""
    pass


class RuleAlreadyExistsError(RuleRegistryError):
    """Error cuando una regla ya existe."""
    pass


@dataclass
class RuleMetadata:
    """Metadatos de una regla en el registro."""
    rule: Rule
    registration_time: float
    usage_count: int = 0
    last_used: Optional[float] = None
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)


class RuleRegistry:
    """
    Registro centralizado de reglas.
    
    Este registro proporciona funcionalidades para gestionar reglas de análisis,
    incluyendo registro, búsqueda, categorización y estadísticas de uso.
    """
    
    def __init__(self):
        """Inicializar el registro de reglas."""
        # Almacenamiento principal de reglas
        self._rules: Dict[RuleId, RuleMetadata] = {}
        
        # Índices para búsqueda rápida
        self._rules_by_category: Dict[RuleCategory, Set[RuleId]] = defaultdict(set)
        self._rules_by_language: Dict[ProgrammingLanguage, Set[RuleId]] = defaultdict(set)
        self._rules_by_severity: Dict[RuleSeverity, Set[RuleId]] = defaultdict(set)
        self._rules_by_tag: Dict[str, Set[RuleId]] = defaultdict(set)
        
        # Estadísticas
        self._total_rules = 0
        self._enabled_rules = 0
        self._disabled_rules = 0
        
        # Lock para operaciones concurrentes
        self._lock = asyncio.Lock()
        
        logger.info("RuleRegistry initialized")
    
    async def register_rule(self, rule: Rule) -> None:
        """
        Registrar una nueva regla.
        
        Args:
            rule: Regla a registrar
            
        Raises:
            RuleAlreadyExistsError: Si la regla ya existe
        """
        async with self._lock:
            if rule.id in self._rules:
                raise RuleAlreadyExistsError(f"Rule {rule.id} already exists")
            
            # Crear metadatos de la regla
            metadata = RuleMetadata(
                rule=rule,
                registration_time=asyncio.get_event_loop().time(),
                enabled=rule.enabled,
                tags=set(rule.tags)
            )
            
            # Registrar en almacenamiento principal
            self._rules[rule.id] = metadata
            
            # Actualizar índices
            self._rules_by_category[rule.category].add(rule.id)
            for language in rule.languages:
                self._rules_by_language[language].add(rule.id)
            self._rules_by_severity[rule.severity].add(rule.id)
            
            for tag in rule.tags:
                self._rules_by_tag[tag].add(rule.id)
            
            # Actualizar estadísticas
            self._total_rules += 1
            if rule.enabled:
                self._enabled_rules += 1
            else:
                self._disabled_rules += 1
            
            logger.info(f"Registered rule: {rule.id} ({rule.name})")
    
    async def unregister_rule(self, rule_id: RuleId) -> None:
        """
        Desregistrar una regla.
        
        Args:
            rule_id: ID de la regla a desregistrar
            
        Raises:
            RuleNotFoundError: Si la regla no existe
        """
        async with self._lock:
            if rule_id not in self._rules:
                raise RuleNotFoundError(f"Rule {rule_id} not found")
            
            rule = self._rules[rule_id].rule
            
            # Remover de índices
            self._rules_by_category[rule.category].discard(rule_id)
            for language in rule.languages:
                self._rules_by_language[language].discard(rule_id)
            self._rules_by_severity[rule.severity].discard(rule_id)
            
            for tag in rule.tags:
                self._rules_by_tag[tag].discard(rule_id)
            
            # Remover del almacenamiento principal
            del self._rules[rule_id]
            
            # Actualizar estadísticas
            self._total_rules -= 1
            if rule.enabled:
                self._enabled_rules -= 1
            else:
                self._disabled_rules -= 1
            
            logger.info(f"Unregistered rule: {rule_id}")
    
    async def get_rule(self, rule_id: RuleId) -> Optional[Rule]:
        """
        Obtener una regla por su ID.
        
        Args:
            rule_id: ID de la regla
            
        Returns:
            Regla si existe, None en caso contrario
        """
        async with self._lock:
            if rule_id in self._rules:
                metadata = self._rules[rule_id]
                metadata.usage_count += 1
                metadata.last_used = asyncio.get_event_loop().time()
                return metadata.rule
            return None
    
    async def get_rules_by_category(self, category: RuleCategory) -> List[Rule]:
        """
        Obtener reglas por categoría.
        
        Args:
            category: Categoría de las reglas
            
        Returns:
            Lista de reglas en la categoría
        """
        async with self._lock:
            rule_ids = self._rules_by_category.get(category, set())
            rules = []
            
            for rule_id in rule_ids:
                if rule_id in self._rules:
                    metadata = self._rules[rule_id]
                    if metadata.enabled:
                        metadata.usage_count += 1
                        metadata.last_used = asyncio.get_event_loop().time()
                        rules.append(metadata.rule)
            
            return rules
    
    async def get_rules_by_language(self, language: ProgrammingLanguage) -> List[Rule]:
        """
        Obtener reglas por lenguaje de programación.
        
        Args:
            language: Lenguaje de programación
            
        Returns:
            Lista de reglas para el lenguaje
        """
        async with self._lock:
            rule_ids = self._rules_by_language.get(language, set())
            rules = []
            
            for rule_id in rule_ids:
                if rule_id in self._rules:
                    metadata = self._rules[rule_id]
                    if metadata.enabled:
                        metadata.usage_count += 1
                        metadata.last_used = asyncio.get_event_loop().time()
                        rules.append(metadata.rule)
            
            return rules
    
    async def get_rules_by_severity(self, severity: RuleSeverity) -> List[Rule]:
        """
        Obtener reglas por severidad.
        
        Args:
            severity: Nivel de severidad
            
        Returns:
            Lista de reglas con la severidad especificada
        """
        async with self._lock:
            rule_ids = self._rules_by_severity.get(severity, set())
            rules = []
            
            for rule_id in rule_ids:
                if rule_id in self._rules:
                    metadata = self._rules[rule_id]
                    if metadata.enabled:
                        metadata.usage_count += 1
                        metadata.last_used = asyncio.get_event_loop().time()
                        rules.append(metadata.rule)
            
            return rules
    
    async def get_rules_by_tag(self, tag: str) -> List[Rule]:
        """
        Obtener reglas por etiqueta.
        
        Args:
            tag: Etiqueta de las reglas
            
        Returns:
            Lista de reglas con la etiqueta especificada
        """
        async with self._lock:
            rule_ids = self._rules_by_tag.get(tag, set())
            rules = []
            
            for rule_id in rule_ids:
                if rule_id in self._rules:
                    metadata = self._rules[rule_id]
                    if metadata.enabled:
                        metadata.usage_count += 1
                        metadata.last_used = asyncio.get_event_loop().time()
                        rules.append(metadata.rule)
            
            return rules
    
    async def search_rules(self, query: str) -> List[Rule]:
        """
        Buscar reglas por texto.
        
        Args:
            query: Texto de búsqueda
            
        Returns:
            Lista de reglas que coinciden con la búsqueda
        """
        async with self._lock:
            query_lower = query.lower()
            matching_rules = []
            
            for metadata in self._rules.values():
                if not metadata.enabled:
                    continue
                
                rule = metadata.rule
                
                # Buscar en nombre, descripción y etiquetas
                if (query_lower in rule.name.lower() or
                    query_lower in rule.description.lower() or
                    any(query_lower in tag.lower() for tag in rule.tags)):
                    
                    metadata.usage_count += 1
                    metadata.last_used = asyncio.get_event_loop().time()
                    matching_rules.append(rule)
            
            return matching_rules
    
    async def get_all_rules(self) -> List[Rule]:
        """
        Obtener todas las reglas registradas.
        
        Returns:
            Lista de todas las reglas
        """
        async with self._lock:
            rules = []
            for metadata in self._rules.values():
                if metadata.enabled:
                    metadata.usage_count += 1
                    metadata.last_used = asyncio.get_event_loop().time()
                    rules.append(metadata.rule)
            return rules
    
    async def get_enabled_rules(self) -> List[Rule]:
        """
        Obtener todas las reglas habilitadas.
        
        Returns:
            Lista de reglas habilitadas
        """
        async with self._lock:
            rules = []
            for metadata in self._rules.values():
                if metadata.enabled:
                    metadata.usage_count += 1
                    metadata.last_used = asyncio.get_event_loop().time()
                    rules.append(metadata.rule)
            return rules
    
    async def get_disabled_rules(self) -> List[Rule]:
        """
        Obtener todas las reglas deshabilitadas.
        
        Returns:
            Lista de reglas deshabilitadas
        """
        async with self._lock:
            rules = []
            for metadata in self._rules.values():
                if not metadata.enabled:
                    metadata.usage_count += 1
                    metadata.last_used = asyncio.get_event_loop().time()
                    rules.append(metadata.rule)
            return rules
    
    async def enable_rule(self, rule_id: RuleId) -> None:
        """
        Habilitar una regla.
        
        Args:
            rule_id: ID de la regla a habilitar
            
        Raises:
            RuleNotFoundError: Si la regla no existe
        """
        async with self._lock:
            if rule_id not in self._rules:
                raise RuleNotFoundError(f"Rule {rule_id} not found")
            
            metadata = self._rules[rule_id]
            if not metadata.enabled:
                metadata.enabled = True
                metadata.rule.enabled = True
                self._enabled_rules += 1
                self._disabled_rules -= 1
                
                logger.info(f"Enabled rule: {rule_id}")
    
    async def disable_rule(self, rule_id: RuleId) -> None:
        """
        Deshabilitar una regla.
        
        Args:
            rule_id: ID de la regla a deshabilitar
            
        Raises:
            RuleNotFoundError: Si la regla no existe
        """
        async with self._lock:
            if rule_id not in self._rules:
                raise RuleNotFoundError(f"Rule {rule_id} not found")
            
            metadata = self._rules[rule_id]
            if metadata.enabled:
                metadata.enabled = False
                metadata.rule.enabled = False
                self._enabled_rules -= 1
                self._disabled_rules += 1
                
                logger.info(f"Disabled rule: {rule_id}")
    
    async def update_rule(self, rule_id: RuleId, updates: Dict[str, Any]) -> None:
        """
        Actualizar una regla existente.
        
        Args:
            rule_id: ID de la regla a actualizar
            updates: Diccionario con las actualizaciones
            
        Raises:
            RuleNotFoundError: Si la regla no existe
        """
        async with self._lock:
            if rule_id not in self._rules:
                raise RuleNotFoundError(f"Rule {rule_id} not found")
            
            metadata = self._rules[rule_id]
            rule = metadata.rule
            
            # Actualizar campos permitidos
            for field, value in updates.items():
                if hasattr(rule, field):
                    setattr(rule, field, value)
            
            # Reindexar si es necesario
            if 'category' in updates or 'languages' in updates or 'severity' in updates or 'tags' in updates:
                await self._reindex_rule(rule_id, rule)
            
            logger.info(f"Updated rule: {rule_id}")
    
    async def _reindex_rule(self, rule_id: RuleId, rule: Rule) -> None:
        """Reindexar una regla después de actualizaciones."""
        # Remover de índices existentes
        for category in self._rules_by_category:
            self._rules_by_category[category].discard(rule_id)
        for language in self._rules_by_language:
            self._rules_by_language[language].discard(rule_id)
        for severity in self._rules_by_severity:
            self._rules_by_severity[severity].discard(rule_id)
        for tag in self._rules_by_tag:
            self._rules_by_tag[tag].discard(rule_id)
        
        # Añadir a nuevos índices
        self._rules_by_category[rule.category].add(rule_id)
        for language in rule.languages:
            self._rules_by_language[language].add(rule_id)
        self._rules_by_severity[rule.severity].add(rule_id)
        for tag in rule.tags:
            self._rules_by_tag[tag].add(rule_id)
    
    async def get_rule_count(self) -> int:
        """
        Obtener el número total de reglas registradas.
        
        Returns:
            Número de reglas
        """
        async with self._lock:
            return self._total_rules
    
    async def get_enabled_rule_count(self) -> int:
        """
        Obtener el número de reglas habilitadas.
        
        Returns:
            Número de reglas habilitadas
        """
        async with self._lock:
            return self._enabled_rules
    
    async def get_disabled_rule_count(self) -> int:
        """
        Obtener el número de reglas deshabilitadas.
        
        Returns:
            Número de reglas deshabilitadas
        """
        async with self._lock:
            return self._disabled_rules
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del registro.
        
        Returns:
            Diccionario con estadísticas
        """
        async with self._lock:
            # Estadísticas por categoría
            category_stats = {}
            for category, rule_ids in self._rules_by_category.items():
                enabled_count = sum(1 for rid in rule_ids if rid in self._rules and self._rules[rid].enabled)
                category_stats[category.value] = {
                    'total': len(rule_ids),
                    'enabled': enabled_count,
                    'disabled': len(rule_ids) - enabled_count
                }
            
            # Estadísticas por lenguaje
            language_stats = {}
            for language, rule_ids in self._rules_by_language.items():
                enabled_count = sum(1 for rid in rule_ids if rid in self._rules and self._rules[rid].enabled)
                language_stats[language.value] = {
                    'total': len(rule_ids),
                    'enabled': enabled_count,
                    'disabled': len(rule_ids) - enabled_count
                }
            
            # Estadísticas por severidad
            severity_stats = {}
            for severity, rule_ids in self._rules_by_severity.items():
                enabled_count = sum(1 for rid in rule_ids if rid in self._rules and self._rules[rid].enabled)
                severity_stats[severity.value] = {
                    'total': len(rule_ids),
                    'enabled': enabled_count,
                    'disabled': len(rule_ids) - enabled_count
                }
            
            # Reglas más usadas
            most_used = sorted(
                self._rules.values(),
                key=lambda m: m.usage_count,
                reverse=True
            )[:10]
            
            return {
                'total_rules': self._total_rules,
                'enabled_rules': self._enabled_rules,
                'disabled_rules': self._disabled_rules,
                'category_stats': category_stats,
                'language_stats': language_stats,
                'severity_stats': severity_stats,
                'most_used_rules': [
                    {
                        'id': metadata.rule.id,
                        'name': metadata.rule.name,
                        'usage_count': metadata.usage_count
                    }
                    for metadata in most_used
                ]
            }
    
    async def clear(self) -> None:
        """Limpiar todas las reglas del registro."""
        async with self._lock:
            self._rules.clear()
            self._rules_by_category.clear()
            self._rules_by_language.clear()
            self._rules_by_severity.clear()
            self._rules_by_tag.clear()
            
            self._total_rules = 0
            self._enabled_rules = 0
            self._disabled_rules = 0
            
            logger.info("RuleRegistry cleared")
