"""
AST Unifier - Sistema de Unificación de ASTs Cross-Language.

Este módulo implementa el sistema de unificación que convierte ASTs de diferentes
parsers especializados en una representación unificada y normalizada.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass, field

from .unified_ast import (
    UnifiedAST,
    UnifiedNode,
    UnifiedNodeType,
    SemanticNodeType,
    UnifiedType,
    UnifiedValue,
    UnifiedPosition,
    UnifiedASTMetadata,
    UnifiedSemanticInfo,
    CrossLanguageMapping,
    ASTId,
    NodeId,
    ASTVersion,
    TypedValue,
    Parameter,
)
from ..universal import ProgrammingLanguage, ParseResult

logger = logging.getLogger(__name__)


class UnificationError(Exception):
    """Error durante la unificación de ASTs."""
    
    def __init__(self, message: str, language: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.language = language
        self.details = details or {}


@dataclass
class UnificationConfig:
    """Configuración para la unificación de ASTs."""
    preserve_language_specific: bool = True
    enable_cross_language_mapping: bool = True
    normalize_semantics: bool = True
    unify_types: bool = True
    generate_cross_refs: bool = True
    include_metadata: bool = True
    strict_mode: bool = False
    debug_mode: bool = False


class SpecializedAnalysisResult(Protocol):
    """Protocolo para resultados de análisis especializados."""
    
    def get_language(self) -> ProgrammingLanguage:
        """Obtiene el lenguaje del análisis."""
        ...
    
    def get_file_path(self) -> Path:
        """Obtiene la ruta del archivo analizado."""
        ...
    
    def get_ast(self) -> Any:
        """Obtiene el AST del análisis."""
        ...


class LanguageUnifier(ABC):
    """Interfaz para unificadores específicos de lenguaje."""
    
    @abstractmethod
    def language(self) -> ProgrammingLanguage:
        """Retorna el lenguaje que este unificador maneja."""
        pass
    
    @abstractmethod
    async def unify_tree_sitter(self, tree: Any, content: str, file_path: Path) -> UnifiedAST:
        """Unifica un AST de tree-sitter."""
        pass
    
    @abstractmethod
    async def unify_specialized(self, analysis_result: SpecializedAnalysisResult) -> UnifiedAST:
        """Unifica un resultado de análisis especializado."""
        pass
    
    @abstractmethod
    def map_node_type(self, language_specific_type: str) -> UnifiedNodeType:
        """Mapea un tipo de nodo específico del lenguaje a un tipo unificado."""
        pass
    
    @abstractmethod
    def extract_semantic_type(self, node: Any) -> SemanticNodeType:
        """Extrae el tipo semántico de un nodo."""
        pass
    
    @abstractmethod
    def unify_type(self, language_type: Any) -> UnifiedType:
        """Unifica un tipo específico del lenguaje a un tipo unificado."""
        pass


class ASTUnifier:
    """Sistema principal de unificación de ASTs."""
    
    def __init__(self, config: Optional[UnificationConfig] = None):
        self.config = config or UnificationConfig()
        self.language_unifiers: Dict[ProgrammingLanguage, LanguageUnifier] = {}
        self._setup_default_unifiers()
    
    def _setup_default_unifiers(self):
        """Configura los unificadores por defecto."""
        # Los unificadores específicos se registrarán cuando se implementen
        pass
    
    def register_unifier(self, language: ProgrammingLanguage, unifier: LanguageUnifier):
        """Registra un unificador para un lenguaje específico."""
        self.language_unifiers[language] = unifier
        logger.info(f"Registrado unificador para {language.value}")
    
    async def unify_parse_result(self, parse_result: ParseResult) -> UnifiedAST:
        """Unifica un resultado de parsing."""
        try:
            language = parse_result.language
            unifier = self.language_unifiers.get(language)
            
            if not unifier:
                raise UnificationError(
                    f"No hay unificador registrado para {language.value}",
                    language=language.value
                )
            
            # Crear un AST unificado básico
            unified_ast = await self._create_basic_unified_ast(parse_result, unifier)
            
            # Aplicar normalización semántica si está habilitada
            if self.config.normalize_semantics:
                unified_ast = await self._normalize_semantics(unified_ast)
            
            # Generar mapeos cross-language si está habilitado
            if self.config.enable_cross_language_mapping:
                unified_ast.cross_language_mappings = await self._generate_cross_language_mappings(unified_ast)
            
            # Enriquecer con metadatos si está habilitado
            if self.config.include_metadata:
                unified_ast.metadata = await self._generate_metadata(unified_ast, parse_result)
            
            return unified_ast
            
        except Exception as e:
            if isinstance(e, UnificationError):
                raise
            raise UnificationError(
                f"Error durante la unificación: {str(e)}",
                language=parse_result.language.value if parse_result.language else None,
                details={"error_type": type(e).__name__}
            )
    
    async def unify_multiple_results(self, parse_results: List[ParseResult]) -> List[UnifiedAST]:
        """Unifica múltiples resultados de parsing."""
        unified_asts = []
        
        for parse_result in parse_results:
            try:
                unified_ast = await self.unify_parse_result(parse_result)
                unified_asts.append(unified_ast)
            except UnificationError as e:
                logger.warning(f"Error unificando {parse_result.file_path}: {e}")
                if self.config.strict_mode:
                    raise
        
        # Generar mapeos cross-file si está habilitado
        if self.config.enable_cross_language_mapping and len(unified_asts) > 1:
            await self._generate_cross_file_mappings(unified_asts)
        
        return unified_asts
    
    async def _create_basic_unified_ast(self, parse_result: ParseResult, unifier: LanguageUnifier) -> UnifiedAST:
        """Crea un AST unificado básico."""
        # Crear nodo raíz
        root_node = UnifiedNode(
            id=NodeId(),
            node_type=UnifiedNodeType.PROGRAM,
            semantic_type=SemanticNodeType.DEFINITION,
            name="root",
            position=UnifiedPosition(
                start_line=1,
                start_column=1,
                end_line=1,
                end_column=1,
                start_byte=0,
                end_byte=0,
                file_path=parse_result.file_path
            )
        )
        
        # Crear información semántica básica
        semantic_info = UnifiedSemanticInfo()
        
        # Extraer información del análisis especializado si está disponible
        if "rust_analysis" in parse_result.metadata:
            semantic_info = await self._extract_rust_semantic_info(parse_result.metadata["rust_analysis"])
        elif "python_analysis" in parse_result.metadata:
            semantic_info = await self._extract_python_semantic_info(parse_result.metadata["python_analysis"])
        elif "typescript_analysis" in parse_result.metadata:
            semantic_info = await self._extract_typescript_semantic_info(parse_result.metadata["typescript_analysis"])
        
        # Crear metadatos básicos
        metadata = UnifiedASTMetadata(
            original_language=parse_result.language.value,
            parser_used="universal",
            unification_version=ASTVersion.V1.value,
            node_count=1,  # Se actualizará después
            depth=1,  # Se actualizará después
            complexity_score=1.0,  # Se actualizará después
            created_at=datetime.now(timezone.utc)
        )
        
        return UnifiedAST(
            id=ASTId(),
            language=parse_result.language.value,
            file_path=parse_result.file_path,
            root_node=root_node,
            metadata=metadata,
            semantic_info=semantic_info,
            version=ASTVersion.V1,
            created_at=datetime.now(timezone.utc)
        )
    
    async def _extract_rust_semantic_info(self, rust_analysis: Any) -> UnifiedSemanticInfo:
        """Extrae información semántica de análisis de Rust."""
        semantic_info = UnifiedSemanticInfo()
        
        # Extraer funciones
        if hasattr(rust_analysis, 'functions'):
            for func in rust_analysis.functions:
                semantic_info.symbols[func.name] = {
                    'type': 'function',
                    'start_line': func.start_line,
                    'end_line': func.end_line,
                    'is_async': func.is_async,
                    'is_unsafe': func.is_unsafe,
                }
        
        # Extraer structs
        if hasattr(rust_analysis, 'structs'):
            for struct in rust_analysis.structs:
                semantic_info.symbols[struct.name] = {
                    'type': 'struct',
                    'start_line': struct.start_line,
                    'end_line': struct.end_line,
                    'fields': struct.fields,
                }
        
        # Extraer traits
        if hasattr(rust_analysis, 'traits'):
            for trait in rust_analysis.traits:
                semantic_info.symbols[trait.name] = {
                    'type': 'trait',
                    'start_line': trait.start_line,
                    'end_line': trait.end_line,
                }
        
        # Extraer imports
        if hasattr(rust_analysis, 'imports'):
            semantic_info.imports = [
                {
                    'module': imp.module_name,
                    'items': imp.imported_items,
                    'line': imp.start_line,
                }
                for imp in rust_analysis.imports
            ]
        
        return semantic_info
    
    async def _extract_python_semantic_info(self, python_analysis: Any) -> UnifiedSemanticInfo:
        """Extrae información semántica de análisis de Python."""
        semantic_info = UnifiedSemanticInfo()
        
        # Extraer funciones
        if hasattr(python_analysis, 'functions'):
            for func in python_analysis.functions:
                semantic_info.symbols[func.name] = {
                    'type': 'function',
                    'start_line': func.start_line,
                    'end_line': func.end_line,
                    'is_async': func.is_async,
                    'is_generator': func.is_generator,
                }
        
        # Extraer clases
        if hasattr(python_analysis, 'classes'):
            for cls in python_analysis.classes:
                semantic_info.symbols[cls.name] = {
                    'type': 'class',
                    'start_line': cls.start_line,
                    'end_line': cls.end_line,
                    'bases': cls.bases,
                }
        
        # Extraer imports
        if hasattr(python_analysis, 'imports'):
            semantic_info.imports = [
                {
                    'module': imp.module_name,
                    'items': imp.imported_items,
                    'line': imp.start_line,
                }
                for imp in python_analysis.imports
            ]
        
        return semantic_info
    
    async def _extract_typescript_semantic_info(self, typescript_analysis: Any) -> UnifiedSemanticInfo:
        """Extrae información semántica de análisis de TypeScript."""
        semantic_info = UnifiedSemanticInfo()
        
        # Extraer funciones
        if hasattr(typescript_analysis, 'functions'):
            for func in typescript_analysis.functions:
                semantic_info.symbols[func.name] = {
                    'type': 'function',
                    'start_line': func.start_line,
                    'end_line': func.end_line,
                    'is_async': func.is_async,
                    'return_type': func.return_type,
                }
        
        # Extraer clases
        if hasattr(typescript_analysis, 'classes'):
            for cls in typescript_analysis.classes:
                semantic_info.symbols[cls.name] = {
                    'type': 'class',
                    'start_line': cls.start_line,
                    'end_line': cls.end_line,
                    'extends': cls.extends,
                    'implements': cls.implements,
                }
        
        # Extraer interfaces
        if hasattr(typescript_analysis, 'interfaces'):
            for interface in typescript_analysis.interfaces:
                semantic_info.symbols[interface.name] = {
                    'type': 'interface',
                    'start_line': interface.start_line,
                    'end_line': interface.end_line,
                }
        
        # Extraer imports
        if hasattr(typescript_analysis, 'imports'):
            semantic_info.imports = [
                {
                    'module': imp.module_name,
                    'items': imp.imported_items,
                    'line': imp.start_line,
                }
                for imp in typescript_analysis.imports
            ]
        
        return semantic_info
    
    async def _normalize_semantics(self, unified_ast: UnifiedAST) -> UnifiedAST:
        """Normaliza la semántica del AST unificado."""
        # Implementación básica - se expandirá
        return unified_ast
    
    async def _generate_cross_language_mappings(self, unified_ast: UnifiedAST) -> List[CrossLanguageMapping]:
        """Genera mapeos cross-language."""
        # Implementación básica - se expandirá
        return []
    
    async def _generate_cross_file_mappings(self, unified_asts: List[UnifiedAST]) -> None:
        """Genera mapeos entre archivos."""
        # Implementación básica - se expandirá
        pass
    
    async def _generate_metadata(self, unified_ast: UnifiedAST, parse_result: ParseResult) -> UnifiedASTMetadata:
        """Genera metadatos del AST unificado."""
        node_count = unified_ast.get_node_count()
        depth = unified_ast.get_depth()
        complexity_score = unified_ast.get_complexity_score()
        
        # Extraer características semánticas
        semantic_features = []
        if unified_ast.semantic_info.symbols:
            semantic_features.append("symbols")
        if unified_ast.semantic_info.imports:
            semantic_features.append("imports")
        if unified_ast.semantic_info.ownership_info:
            semantic_features.append("ownership")
        if unified_ast.semantic_info.lifetime_info:
            semantic_features.append("lifetimes")
        if unified_ast.semantic_info.trait_info:
            semantic_features.append("traits")
        if unified_ast.semantic_info.unsafe_info:
            semantic_features.append("unsafe")
        
        # Evaluar compatibilidad cross-language
        cross_language_compatibility = {
            "python": 0.8,
            "typescript": 0.7,
            "rust": 0.9,
            "javascript": 0.6,
        }
        
        return UnifiedASTMetadata(
            original_language=unified_ast.language,
            parser_used=parse_result.language.value,
            unification_version=ASTVersion.V1.value,
            node_count=node_count,
            depth=depth,
            complexity_score=complexity_score,
            semantic_features=semantic_features,
            cross_language_compatibility=cross_language_compatibility,
            created_at=datetime.now(timezone.utc)
        )
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Retorna los lenguajes soportados por este unificador."""
        return list(self.language_unifiers.keys())
    
    def get_unification_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de unificación."""
        return {
            "supported_languages": len(self.language_unifiers),
            "languages": [lang.value for lang in self.language_unifiers.keys()],
            "config": {
                "preserve_language_specific": self.config.preserve_language_specific,
                "enable_cross_language_mapping": self.config.enable_cross_language_mapping,
                "normalize_semantics": self.config.normalize_semantics,
                "strict_mode": self.config.strict_mode,
            }
        }
