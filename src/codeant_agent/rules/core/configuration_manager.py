"""
Gestor de configuración para el motor de reglas estáticas.

Este módulo implementa la gestión de configuraciones para el motor de reglas,
incluyendo configuraciones globales, de proyecto y de reglas individuales.
"""

import asyncio
import logging
import json
import yaml
import toml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..models.config_models import (
    ProjectRuleConfig,
    GlobalRuleConfig,
    RuleOverride,
    EffectiveRuleConfig,
    ConfigFileFormat,
    ConfigFile,
    ConfigWatcher
)
from ..models.rule_models import Rule, RuleId, RuleSeverity, RuleCategory

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Excepción base para errores de configuración."""
    pass


class ConfigFileError(ConfigurationError):
    """Error relacionado con archivos de configuración."""
    pass


class ConfigValidationError(ConfigurationError):
    """Error de validación de configuración."""
    pass


@dataclass
class ConfigurationContext:
    """Contexto de configuración."""
    project_path: Path
    global_config: GlobalRuleConfig
    project_config: ProjectRuleConfig
    effective_configs: Dict[str, EffectiveRuleConfig] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConfigurationManager:
    """
    Gestor de configuración para el motor de reglas.
    
    Este gestor maneja configuraciones globales, de proyecto y de reglas
    individuales, proporcionando un sistema flexible y extensible.
    """
    
    def __init__(self):
        """Inicializar el gestor de configuración."""
        # Configuraciones almacenadas
        self.global_config = GlobalRuleConfig()
        self.project_configs: Dict[Path, ProjectRuleConfig] = {}
        
        # Observadores de archivos de configuración
        self.config_watchers: List[ConfigWatcher] = []
        
        # Cache de configuraciones efectivas
        self.effective_config_cache: Dict[str, EffectiveRuleConfig] = {}
        
        # Archivos de configuración conocidos
        self.known_config_files: Dict[Path, ConfigFile] = {}
        
        # Lock para operaciones concurrentes
        self.lock = asyncio.Lock()
        
        logger.info("ConfigurationManager initialized")
    
    async def initialize(self) -> None:
        """Inicializar el gestor de configuración."""
        try:
            # Cargar configuración global
            await self._load_global_configuration()
            
            # Inicializar observadores de archivos
            await self._initialize_config_watchers()
            
            logger.info("ConfigurationManager initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize ConfigurationManager: {e}")
    
    async def load_project_config(self, project_path: Path) -> ProjectRuleConfig:
        """
        Cargar configuración de un proyecto.
        
        Args:
            project_path: Ruta del proyecto
            
        Returns:
            Configuración del proyecto
        """
        async with self.lock:
            # Verificar cache
            if project_path in self.project_configs:
                return self.project_configs[project_path]
            
            try:
                # Buscar archivos de configuración
                config_file = await self._find_config_file(project_path)
                
                if config_file:
                    # Parsear archivo de configuración
                    project_config = await self._parse_config_file(config_file)
                else:
                    # Usar configuración por defecto
                    project_config = ProjectRuleConfig(project_path=project_path)
                
                # Validar configuración
                await self._validate_project_config(project_config)
                
                # Cachear configuración
                self.project_configs[project_path] = project_config
                
                logger.info(f"Loaded project configuration for {project_path}")
                return project_config
                
            except Exception as e:
                logger.error(f"Failed to load project configuration for {project_path}: {e}")
                # Retornar configuración por defecto en caso de error
                return ProjectRuleConfig(project_path=project_path)
    
    async def get_effective_rule_config(self, rule: Rule, project_path: Path) -> EffectiveRuleConfig:
        """
        Obtener configuración efectiva para una regla.
        
        Args:
            rule: Regla para la cual obtener configuración
            project_path: Ruta del proyecto
            
        Returns:
            Configuración efectiva de la regla
        """
        cache_key = f"{rule.id}:{project_path}"
        
        async with self.lock:
            # Verificar cache
            if cache_key in self.effective_config_cache:
                return self.effective_config_cache[cache_key]
            
            # Cargar configuración del proyecto si no está en cache
            project_config = await self.load_project_config(project_path)
            
            # Crear configuración efectiva
            effective_config = EffectiveRuleConfig(
                rule_id=rule.id,
                enabled=rule.enabled,
                severity=rule.severity,
                parameters=rule.configuration.parameters.copy(),
                thresholds=rule.configuration.thresholds.copy()
            )
            
            # Aplicar overrides globales
            if rule.id in self.global_config.disabled_rules:
                effective_config.enabled = False
            
            # Aplicar overrides del proyecto
            if rule.id in project_config.rule_overrides:
                override = project_config.rule_overrides[rule.id]
                
                if override.enabled is not None:
                    effective_config.enabled = override.enabled
                
                if override.severity is not None:
                    effective_config.severity = override.severity
                
                # Aplicar parámetros personalizados
                for key, value in override.custom_parameters.items():
                    effective_config.parameters[key] = value
            
            # Cachear configuración efectiva
            self.effective_config_cache[cache_key] = effective_config
            
            return effective_config
    
    async def update_project_config(self, project_path: Path, 
                                  updates: Dict[str, Any]) -> ProjectRuleConfig:
        """
        Actualizar configuración de un proyecto.
        
        Args:
            project_path: Ruta del proyecto
            updates: Actualizaciones a aplicar
            
        Returns:
            Configuración actualizada del proyecto
        """
        async with self.lock:
            # Cargar configuración actual
            project_config = await self.load_project_config(project_path)
            
            # Aplicar actualizaciones
            for key, value in updates.items():
                if hasattr(project_config, key):
                    setattr(project_config, key, value)
            
            # Validar configuración actualizada
            await self._validate_project_config(project_config)
            
            # Actualizar cache
            self.project_configs[project_path] = project_config
            
            # Limpiar cache de configuraciones efectivas
            await self._clear_effective_config_cache(project_path)
            
            logger.info(f"Updated project configuration for {project_path}")
            return project_config
    
    async def add_rule_override(self, project_path: Path, rule_id: str, 
                              override: RuleOverride) -> None:
        """
        Añadir override de regla a un proyecto.
        
        Args:
            project_path: Ruta del proyecto
            rule_id: ID de la regla
            override: Override a aplicar
        """
        async with self.lock:
            project_config = await self.load_project_config(project_path)
            
            project_config.rule_overrides[rule_id] = override
            
            # Actualizar cache
            self.project_configs[project_path] = project_config
            
            # Limpiar cache de configuraciones efectivas
            await self._clear_effective_config_cache(project_path)
            
            logger.info(f"Added rule override {rule_id} for project {project_path}")
    
    async def remove_rule_override(self, project_path: Path, rule_id: str) -> bool:
        """
        Remover override de regla de un proyecto.
        
        Args:
            project_path: Ruta del proyecto
            rule_id: ID de la regla
            
        Returns:
            True si se removió, False si no existía
        """
        async with self.lock:
            project_config = await self.load_project_config(project_path)
            
            if rule_id in project_config.rule_overrides:
                del project_config.rule_overrides[rule_id]
                
                # Actualizar cache
                self.project_configs[project_path] = project_config
                
                # Limpiar cache de configuraciones efectivas
                await self._clear_effective_config_cache(project_path)
                
                logger.info(f"Removed rule override {rule_id} for project {project_path}")
                return True
            
            return False
    
    async def get_configuration_context(self, project_path: Path) -> ConfigurationContext:
        """
        Obtener contexto completo de configuración para un proyecto.
        
        Args:
            project_path: Ruta del proyecto
            
        Returns:
            Contexto de configuración
        """
        async with self.lock:
            project_config = await self.load_project_config(project_path)
            
            # Obtener configuraciones efectivas para todas las reglas conocidas
            effective_configs = {}
            # En una implementación real, iterarías sobre todas las reglas registradas
            
            return ConfigurationContext(
                project_path=project_path,
                global_config=self.global_config,
                project_config=project_config,
                effective_configs=effective_configs
            )
    
    async def validate_configuration(self, config: Union[GlobalRuleConfig, ProjectRuleConfig]) -> List[str]:
        """
        Validar una configuración.
        
        Args:
            config: Configuración a validar
            
        Returns:
            Lista de errores de validación
        """
        errors = []
        
        try:
            if isinstance(config, GlobalRuleConfig):
                errors.extend(await self._validate_global_config(config))
            elif isinstance(config, ProjectRuleConfig):
                errors.extend(await self._validate_project_config(config))
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return errors
    
    async def export_configuration(self, project_path: Path, format: ConfigFileFormat) -> str:
        """
        Exportar configuración de un proyecto.
        
        Args:
            project_path: Ruta del proyecto
            format: Formato de exportación
            
        Returns:
            Configuración serializada
        """
        async with self.lock:
            project_config = await self.load_project_config(project_path)
            
            # Convertir a diccionario
            config_dict = self._config_to_dict(project_config)
            
            # Serializar según formato
            if format == ConfigFileFormat.JSON:
                return json.dumps(config_dict, indent=2, default=str)
            elif format == ConfigFileFormat.YAML:
                return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
            elif format == ConfigFileFormat.TOML:
                return toml.dumps(config_dict)
            else:
                raise ConfigFileError(f"Unsupported export format: {format}")
    
    async def import_configuration(self, project_path: Path, config_data: str, 
                                 format: ConfigFileFormat) -> ProjectRuleConfig:
        """
        Importar configuración para un proyecto.
        
        Args:
            project_path: Ruta del proyecto
            config_data: Datos de configuración
            format: Formato de los datos
            
        Returns:
            Configuración importada
        """
        async with self.lock:
            try:
                # Deserializar según formato
                if format == ConfigFileFormat.JSON:
                    config_dict = json.loads(config_data)
                elif format == ConfigFileFormat.YAML:
                    config_dict = yaml.safe_load(config_data)
                elif format == ConfigFileFormat.TOML:
                    config_dict = toml.loads(config_data)
                else:
                    raise ConfigFileError(f"Unsupported import format: {format}")
                
                # Convertir a objeto de configuración
                project_config = self._dict_to_config(config_dict, project_path)
                
                # Validar configuración
                await self._validate_project_config(project_config)
                
                # Actualizar cache
                self.project_configs[project_path] = project_config
                
                # Limpiar cache de configuraciones efectivas
                await self._clear_effective_config_cache(project_path)
                
                logger.info(f"Imported configuration for project {project_path}")
                return project_config
                
            except Exception as e:
                raise ConfigFileError(f"Failed to import configuration: {e}")
    
    async def _find_config_file(self, project_path: Path) -> Optional[ConfigFile]:
        """Buscar archivo de configuración en el proyecto."""
        config_files = [
            project_path / ".codeant.toml",
            project_path / ".codeant.yaml",
            project_path / ".codeant.yml",
            project_path / ".codeant.json",
            project_path / "codeant.config.toml",
            project_path / "codeant.config.yaml",
            project_path / "codeant.config.yml",
            project_path / "codeant.config.json"
        ]
        
        for config_file_path in config_files:
            if config_file_path.exists():
                try:
                    content = config_file_path.read_text(encoding='utf-8')
                    format = self._detect_config_format(config_file_path)
                    
                    config_file = ConfigFile(
                        path=config_file_path,
                        format=format,
                        content=content,
                        last_modified=datetime.fromtimestamp(config_file_path.stat().st_mtime, tz=timezone.utc)
                    )
                    
                    self.known_config_files[config_file_path] = config_file
                    return config_file
                    
                except Exception as e:
                    logger.warning(f"Failed to read config file {config_file_path}: {e}")
        
        return None
    
    async def _parse_config_file(self, config_file: ConfigFile) -> ProjectRuleConfig:
        """Parsear archivo de configuración."""
        try:
            if config_file.format == ConfigFileFormat.TOML:
                config_dict = toml.loads(config_file.content)
            elif config_file.format in [ConfigFileFormat.YAML, ConfigFileFormat.YML]:
                config_dict = yaml.safe_load(config_file.content)
            elif config_file.format == ConfigFileFormat.JSON:
                config_dict = json.loads(config_file.content)
            else:
                raise ConfigFileError(f"Unsupported config file format: {config_file.format}")
            
            # Convertir a objeto de configuración
            project_config = self._dict_to_config(config_dict, config_file.path.parent)
            
            return project_config
            
        except Exception as e:
            raise ConfigFileError(f"Failed to parse config file {config_file.path}: {e}")
    
    def _detect_config_format(self, file_path: Path) -> ConfigFileFormat:
        """Detectar formato de archivo de configuración."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.toml':
            return ConfigFileFormat.TOML
        elif suffix in ['.yaml', '.yml']:
            return ConfigFileFormat.YAML
        elif suffix == '.json':
            return ConfigFileFormat.JSON
        else:
            return ConfigFileFormat.TOML  # Por defecto
    
    def _config_to_dict(self, config: Union[GlobalRuleConfig, ProjectRuleConfig]) -> Dict[str, Any]:
        """Convertir configuración a diccionario."""
        config_dict = {}
        
        if isinstance(config, ProjectRuleConfig):
            config_dict['project_path'] = str(config.project_path)
            
            if config.rule_overrides:
                config_dict['rule_overrides'] = {
                    rule_id: {
                        'enabled': override.enabled,
                        'severity': override.severity.value if override.severity else None,
                        'custom_parameters': override.custom_parameters,
                        'custom_message': override.custom_message,
                        'custom_suggestion': override.custom_suggestion
                    }
                    for rule_id, override in config.rule_overrides.items()
                }
            
            if config.custom_thresholds:
                config_dict['custom_thresholds'] = config.custom_thresholds
            
            if config.exclusion_patterns:
                config_dict['exclusion_patterns'] = config.exclusion_patterns
            
            if config.quality_gates:
                config_dict['quality_gates'] = {
                    'max_critical_violations': config.quality_gates.max_critical_violations,
                    'max_high_violations': config.quality_gates.max_high_violations,
                    'max_medium_violations': config.quality_gates.max_medium_violations,
                    'min_quality_score': config.quality_gates.min_quality_score,
                    'fail_on_quality_gate': config.quality_gates.fail_on_quality_gate
                }
        
        return config_dict
    
    def _dict_to_config(self, config_dict: Dict[str, Any], project_path: Path) -> ProjectRuleConfig:
        """Convertir diccionario a configuración de proyecto."""
        project_config = ProjectRuleConfig(project_path=project_path)
        
        # Aplicar configuraciones desde diccionario
        if 'rule_overrides' in config_dict:
            for rule_id, override_data in config_dict['rule_overrides'].items():
                override = RuleOverride(
                    rule_id=rule_id,
                    enabled=override_data.get('enabled'),
                    severity=RuleSeverity(override_data['severity']) if override_data.get('severity') else None,
                    custom_parameters=override_data.get('custom_parameters', {}),
                    custom_message=override_data.get('custom_message'),
                    custom_suggestion=override_data.get('custom_suggestion')
                )
                project_config.rule_overrides[rule_id] = override
        
        if 'custom_thresholds' in config_dict:
            project_config.custom_thresholds = config_dict['custom_thresholds']
        
        if 'exclusion_patterns' in config_dict:
            project_config.exclusion_patterns = config_dict['exclusion_patterns']
        
        if 'quality_gates' in config_dict:
            gates_data = config_dict['quality_gates']
            project_config.quality_gates.max_critical_violations = gates_data.get('max_critical_violations', 0)
            project_config.quality_gates.max_high_violations = gates_data.get('max_high_violations', 10)
            project_config.quality_gates.max_medium_violations = gates_data.get('max_medium_violations', 50)
            project_config.quality_gates.min_quality_score = gates_data.get('min_quality_score', 80.0)
            project_config.quality_gates.fail_on_quality_gate = gates_data.get('fail_on_quality_gate', True)
        
        return project_config
    
    async def _validate_global_config(self, config: GlobalRuleConfig) -> List[str]:
        """Validar configuración global."""
        errors = []
        
        # Validaciones básicas
        if not config.enabled_categories:
            errors.append("At least one category must be enabled")
        
        if config.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            errors.append("Invalid log level")
        
        return errors
    
    async def _validate_project_config(self, config: ProjectRuleConfig) -> List[str]:
        """Validar configuración de proyecto."""
        errors = []
        
        # Validaciones básicas
        if config.parallel_analysis_batch_size is not None and config.parallel_analysis_batch_size <= 0:
            errors.append("parallel_analysis_batch_size must be positive")
        
        # Validar quality gates
        if config.quality_gates.min_quality_score < 0 or config.quality_gates.min_quality_score > 100:
            errors.append("min_quality_score must be between 0 and 100")
        
        return errors
    
    async def _clear_effective_config_cache(self, project_path: Path) -> None:
        """Limpiar cache de configuraciones efectivas para un proyecto."""
        keys_to_remove = [
            key for key in self.effective_config_cache.keys()
            if str(project_path) in key
        ]
        
        for key in keys_to_remove:
            del self.effective_config_cache[key]
    
    async def _load_global_configuration(self) -> None:
        """Cargar configuración global."""
        # En una implementación real, cargaría desde archivo o base de datos
        logger.info("Loading global configuration...")
    
    async def _initialize_config_watchers(self) -> None:
        """Inicializar observadores de archivos de configuración."""
        # En una implementación real, configuraría observadores de archivos
        logger.info("Initializing config watchers...")
    
    async def shutdown(self) -> None:
        """Apagar el gestor de configuración."""
        try:
            # Detener observadores de archivos
            for watcher in self.config_watchers:
                watcher.enabled = False
            
            # Guardar configuraciones si es necesario
            await self._save_configurations()
            
            logger.info("ConfigurationManager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during configuration manager shutdown: {e}")
    
    async def _save_configurations(self) -> None:
        """Guardar configuraciones."""
        # En una implementación real, guardaría a archivos o base de datos
        logger.info("Saving configurations...")
