"""
Sistema de Normalización Semántica Cross-Language.

Este módulo implementa el sistema de normalización semántica que permite
mapear conceptos entre diferentes lenguajes de programación.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

from .unified_ast import (
    UnifiedAST,
    UnifiedNode,
    UnifiedNodeType,
    SemanticNodeType,
    UnifiedType,
    NodeId,
)

logger = logging.getLogger(__name__)


@dataclass
class ConceptMapping:
    """Mapeo de conceptos entre lenguajes."""
    source_concept: str
    target_concept: str
    source_language: str
    target_language: str
    confidence: float
    transformation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticNormalization:
    """Resultado de normalización semántica."""
    normalized_concepts: Dict[str, str]
    concept_mappings: List[ConceptMapping]
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticNormalizer:
    """Normalizador semántico principal."""
    
    def __init__(self):
        self.concept_mapper = ConceptMapper()
        self.type_unifier = TypeUnifier()
        self.cross_language_mapper = CrossLanguageMapper()
    
    async def normalize(self, unified_ast: UnifiedAST) -> UnifiedAST:
        """Normaliza la semántica de un AST unificado."""
        try:
            # Normalizar conceptos
            normalized_concepts = await self.concept_mapper.normalize_concepts(unified_ast)
            
            # Unificar tipos
            unified_ast = await self.type_unifier.unify_types(unified_ast)
            
            # Generar mapeos cross-language
            cross_language_mappings = await self.cross_language_mapper.generate_mappings(unified_ast)
            unified_ast.cross_language_mappings = cross_language_mappings
            
            # Aplicar normalizaciones al AST
            unified_ast = await self._apply_normalizations(unified_ast, normalized_concepts)
            
            return unified_ast
            
        except Exception as e:
            logger.error(f"Error normalizing AST: {e}")
            return unified_ast
    
    async def _apply_normalizations(self, unified_ast: UnifiedAST, normalized_concepts: Dict[str, str]) -> UnifiedAST:
        """Aplica normalizaciones al AST."""
        # Normalizar nombres de nodos
        await self._normalize_node_names(unified_ast.root_node, normalized_concepts)
        
        # Normalizar tipos semánticos
        await self._normalize_semantic_types(unified_ast.root_node)
        
        return unified_ast
    
    async def _normalize_node_names(self, node: UnifiedNode, normalized_concepts: Dict[str, str]):
        """Normaliza nombres de nodos."""
        if node.name and node.name in normalized_concepts:
            node.name = normalized_concepts[node.name]
        
        for child in node.children:
            await self._normalize_node_names(child, normalized_concepts)
    
    async def _normalize_semantic_types(self, node: UnifiedNode):
        """Normaliza tipos semánticos."""
        # Mapear tipos semánticos específicos del lenguaje a tipos unificados
        semantic_mapping = {
            "python_function": SemanticNodeType.DEFINITION,
            "typescript_function": SemanticNodeType.DEFINITION,
            "rust_function": SemanticNodeType.DEFINITION,
            "python_class": SemanticNodeType.DEFINITION,
            "typescript_class": SemanticNodeType.DEFINITION,
            "rust_struct": SemanticNodeType.DEFINITION,
        }
        
        if node.semantic_type.value in semantic_mapping:
            node.semantic_type = semantic_mapping[node.semantic_type.value]
        
        for child in node.children:
            await self._normalize_semantic_types(child)


class ConceptMapper:
    """Mapeador de conceptos entre lenguajes."""
    
    def __init__(self):
        self.concept_library = self._load_concept_library()
    
    def _load_concept_library(self) -> Dict[str, Dict[str, str]]:
        """Carga la biblioteca de conceptos."""
        return {
            "function": {
                "python": "def",
                "typescript": "function",
                "javascript": "function",
                "rust": "fn",
            },
            "class": {
                "python": "class",
                "typescript": "class",
                "javascript": "class",
                "rust": "struct",
            },
            "variable": {
                "python": "variable",
                "typescript": "let",
                "javascript": "var",
                "rust": "let",
            },
            "loop": {
                "python": "for",
                "typescript": "for",
                "javascript": "for",
                "rust": "for",
            },
            "condition": {
                "python": "if",
                "typescript": "if",
                "javascript": "if",
                "rust": "if",
            },
        }
    
    async def normalize_concepts(self, unified_ast: UnifiedAST) -> Dict[str, str]:
        """Normaliza conceptos en un AST."""
        normalized_concepts = {}
        
        # Extraer conceptos del AST
        concepts = await self._extract_concepts(unified_ast)
        
        # Normalizar cada concepto
        for concept in concepts:
            normalized = await self._normalize_concept(concept, unified_ast.language)
            if normalized:
                normalized_concepts[concept] = normalized
        
        return normalized_concepts
    
    async def _extract_concepts(self, unified_ast: UnifiedAST) -> Set[str]:
        """Extrae conceptos de un AST."""
        concepts = set()
        
        def traverse(node: UnifiedNode):
            if node.name:
                concepts.add(node.name)
            
            for child in node.children:
                traverse(child)
        
        traverse(unified_ast.root_node)
        return concepts
    
    async def _normalize_concept(self, concept: str, language: str) -> Optional[str]:
        """Normaliza un concepto específico."""
        # Buscar el concepto en la biblioteca
        for normalized_concept, language_mappings in self.concept_library.items():
            if language in language_mappings and language_mappings[language] == concept:
                return normalized_concept
        
        return None
    
    async def map_concepts_between_languages(self, source_language: str, target_language: str) -> List[ConceptMapping]:
        """Mapea conceptos entre dos lenguajes."""
        mappings = []
        
        for concept, language_mappings in self.concept_library.items():
            if source_language in language_mappings and target_language in language_mappings:
                mapping = ConceptMapping(
                    source_concept=language_mappings[source_language],
                    target_concept=language_mappings[target_language],
                    source_language=source_language,
                    target_language=target_language,
                    confidence=0.8,
                )
                mappings.append(mapping)
        
        return mappings


class TypeUnifier:
    """Unificador de tipos."""
    
    async def unify_types(self, unified_ast: UnifiedAST) -> UnifiedAST:
        """Unifica tipos en un AST."""
        # Normalizar tipos en el AST
        await self._normalize_types_in_ast(unified_ast.root_node)
        
        # Unificar tipos en la información semántica
        await self._unify_semantic_types(unified_ast)
        
        return unified_ast
    
    async def _normalize_types_in_ast(self, node: UnifiedNode):
        """Normaliza tipos en un nodo del AST."""
        # Normalizar tipos específicos del lenguaje
        if hasattr(node, 'value') and node.value:
            node.value = await self._normalize_value(node.value)
        
        # Recursivamente normalizar hijos
        for child in node.children:
            await self._normalize_types_in_ast(child)
    
    async def _normalize_value(self, value: Any) -> Any:
        """Normaliza un valor."""
        # Implementación básica - se expandirá
        return value
    
    async def _unify_semantic_types(self, unified_ast: UnifiedAST):
        """Unifica tipos en la información semántica."""
        # Normalizar tipos en symbols
        for symbol_name, symbol_info in unified_ast.semantic_info.symbols.items():
            if 'type' in symbol_info:
                symbol_info['type'] = await self._normalize_type_name(symbol_info['type'])
        
        # Normalizar tipos en la información de tipos
        for type_name, type_info in unified_ast.semantic_info.types.items():
            normalized_type = await self._normalize_type_name(type_name)
            if normalized_type != type_name:
                unified_ast.semantic_info.types[normalized_type] = type_info
                del unified_ast.semantic_info.types[type_name]
    
    async def _normalize_type_name(self, type_name: str) -> str:
        """Normaliza un nombre de tipo."""
        # Mapeo de tipos específicos del lenguaje a tipos unificados
        type_mapping = {
            # Python types
            "int": "integer",
            "float": "float",
            "str": "string",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "tuple": "tuple",
            
            # TypeScript types
            "number": "float",
            "string": "string",
            "boolean": "boolean",
            "Array": "array",
            "Object": "object",
            
            # JavaScript types
            "String": "string",
            "Number": "float",
            "Boolean": "boolean",
            "Array": "array",
            "Object": "object",
            
            # Rust types
            "i32": "integer",
            "i64": "integer",
            "u32": "integer",
            "u64": "integer",
            "f32": "float",
            "f64": "float",
            "String": "string",
            "bool": "boolean",
            "Vec": "array",
        }
        
        return type_mapping.get(type_name, type_name)


class CrossLanguageMapper:
    """Mapeador cross-language."""
    
    def __init__(self):
        self.mapping_rules = self._load_mapping_rules()
    
    def _load_mapping_rules(self) -> Dict[str, Dict[str, Any]]:
        """Carga reglas de mapeo cross-language."""
        return {
            "function_definition": {
                "python_to_typescript": {
                    "transformation": "def -> function",
                    "confidence": 0.9,
                },
                "python_to_rust": {
                    "transformation": "def -> fn",
                    "confidence": 0.8,
                },
                "typescript_to_rust": {
                    "transformation": "function -> fn",
                    "confidence": 0.8,
                },
            },
            "class_definition": {
                "python_to_typescript": {
                    "transformation": "class -> class",
                    "confidence": 0.9,
                },
                "python_to_rust": {
                    "transformation": "class -> struct",
                    "confidence": 0.7,
                },
                "typescript_to_rust": {
                    "transformation": "class -> struct",
                    "confidence": 0.7,
                },
            },
            "variable_declaration": {
                "python_to_typescript": {
                    "transformation": "variable -> let/const",
                    "confidence": 0.8,
                },
                "python_to_rust": {
                    "transformation": "variable -> let",
                    "confidence": 0.8,
                },
                "typescript_to_rust": {
                    "transformation": "let/const -> let",
                    "confidence": 0.9,
                },
            },
        }
    
    async def generate_mappings(self, unified_ast: UnifiedAST) -> List[Any]:
        """Genera mapeos cross-language para un AST."""
        mappings = []
        
        # Generar mapeos para cada tipo de nodo
        node_types = await self._extract_node_types(unified_ast)
        
        for node_type in node_types:
            type_mappings = await self._generate_type_mappings(node_type, unified_ast.language)
            mappings.extend(type_mappings)
        
        return mappings
    
    async def _extract_node_types(self, unified_ast: UnifiedAST) -> Set[str]:
        """Extrae tipos de nodos de un AST."""
        node_types = set()
        
        def traverse(node: UnifiedNode):
            node_types.add(node.node_type.value)
            
            for child in node.children:
                traverse(child)
        
        traverse(unified_ast.root_node)
        return node_types
    
    async def _generate_type_mappings(self, node_type: str, source_language: str) -> List[Any]:
        """Genera mapeos para un tipo de nodo específico."""
        mappings = []
        
        if node_type in self.mapping_rules:
            type_rules = self.mapping_rules[node_type]
            
            for mapping_key, mapping_info in type_rules.items():
                if source_language in mapping_key:
                    # Extraer lenguaje objetivo
                    target_language = mapping_key.replace(f"{source_language}_to_", "")
                    
                    mapping = {
                        "source_type": node_type,
                        "source_language": source_language,
                        "target_language": target_language,
                        "transformation": mapping_info["transformation"],
                        "confidence": mapping_info["confidence"],
                    }
                    mappings.append(mapping)
        
        return mappings
    
    async def map_concept_across_languages(self, concept: str, source_language: str, target_language: str) -> Optional[Dict[str, Any]]:
        """Mapea un concepto específico entre lenguajes."""
        mapping_key = f"{source_language}_to_{target_language}"
        
        if concept in self.mapping_rules:
            concept_rules = self.mapping_rules[concept]
            
            if mapping_key in concept_rules:
                return {
                    "concept": concept,
                    "source_language": source_language,
                    "target_language": target_language,
                    **concept_rules[mapping_key],
                }
        
        return None
    
    def get_supported_mappings(self) -> Dict[str, List[str]]:
        """Obtiene los mapeos soportados."""
        supported = {}
        
        for concept, mappings in self.mapping_rules.items():
            supported[concept] = list(mappings.keys())
        
        return supported
