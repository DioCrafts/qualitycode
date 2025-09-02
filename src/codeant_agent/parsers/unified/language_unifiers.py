"""
Unificadores Específicos por Lenguaje.

Este módulo implementa unificadores específicos para cada lenguaje de programación,
convirtiendo ASTs específicos del lenguaje a la representación unificada.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Protocol

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
    ASTId,
    NodeId,
    ASTVersion,
    TypedValue,
    Parameter,
    BinaryOperator,
    UnaryOperator,
    Visibility,
)
from .ast_unifier import LanguageUnifier, SpecializedAnalysisResult
from ..universal import ProgrammingLanguage

logger = logging.getLogger(__name__)


class PythonUnifier(LanguageUnifier):
    """Unificador específico para Python."""
    
    def __init__(self):
        self.type_mapper = PythonTypeMapper()
        self.semantic_mapper = PythonSemanticMapper()
    
    def language(self) -> ProgrammingLanguage:
        return ProgrammingLanguage.PYTHON
    
    async def unify_tree_sitter(self, tree: Any, content: str, file_path: Path) -> UnifiedAST:
        """Unifica un AST de tree-sitter de Python."""
        # Crear nodo raíz
        root_node = UnifiedNode(
            id=NodeId(),
            node_type=UnifiedNodeType.PROGRAM,
            semantic_type=SemanticNodeType.DEFINITION,
            name="python_module",
            position=UnifiedPosition(
                start_line=1,
                start_column=1,
                end_line=len(content.split('\n')),
                end_column=1,
                start_byte=0,
                end_byte=len(content),
                file_path=file_path
            )
        )
        
        # Crear información semántica básica
        semantic_info = UnifiedSemanticInfo()
        
        # Crear metadatos
        metadata = UnifiedASTMetadata(
            original_language="python",
            parser_used="tree-sitter",
            unification_version=ASTVersion.V1.value,
            node_count=1,
            depth=1,
            complexity_score=1.0,
            created_at=datetime.now(timezone.utc)
        )
        
        return UnifiedAST(
            id=ASTId(),
            language="python",
            file_path=file_path,
            root_node=root_node,
            metadata=metadata,
            semantic_info=semantic_info,
            version=ASTVersion.V1,
            created_at=datetime.now(timezone.utc)
        )
    
    async def unify_specialized(self, analysis_result: SpecializedAnalysisResult) -> UnifiedAST:
        """Unifica un resultado de análisis especializado de Python."""
        # Implementación básica - se expandirá
        root_node = UnifiedNode(
            id=NodeId(),
            node_type=UnifiedNodeType.PROGRAM,
            semantic_type=SemanticNodeType.DEFINITION,
            name="python_module",
            position=UnifiedPosition(
                start_line=1,
                start_column=1,
                end_line=1,
                end_column=1,
                start_byte=0,
                end_byte=0,
                file_path=analysis_result.get_file_path()
            )
        )
        
        semantic_info = UnifiedSemanticInfo()
        metadata = UnifiedASTMetadata(
            original_language="python",
            parser_used="specialized",
            unification_version=ASTVersion.V1.value,
            node_count=1,
            depth=1,
            complexity_score=1.0,
            created_at=datetime.now(timezone.utc)
        )
        
        return UnifiedAST(
            id=ASTId(),
            language="python",
            file_path=analysis_result.get_file_path(),
            root_node=root_node,
            metadata=metadata,
            semantic_info=semantic_info,
            version=ASTVersion.V1,
            created_at=datetime.now(timezone.utc)
        )
    
    def map_node_type(self, python_type: str) -> UnifiedNodeType:
        """Mapea tipos de nodos específicos de Python a tipos unificados."""
        mapping = {
            "module": UnifiedNodeType.MODULE,
            "function_definition": UnifiedNodeType.FUNCTION_DECLARATION,
            "class_definition": UnifiedNodeType.CLASS_DECLARATION,
            "if_statement": UnifiedNodeType.IF_STATEMENT,
            "for_statement": UnifiedNodeType.FOR_STATEMENT,
            "while_statement": UnifiedNodeType.WHILE_STATEMENT,
            "return_statement": UnifiedNodeType.RETURN_STATEMENT,
            "call": UnifiedNodeType.CALL_EXPRESSION,
            "binary_operator": UnifiedNodeType.BINARY_EXPRESSION,
            "identifier": UnifiedNodeType.IDENTIFIER,
            "string": UnifiedNodeType.STRING_LITERAL,
            "integer": UnifiedNodeType.NUMBER_LITERAL,
            "comment": UnifiedNodeType.COMMENT,
        }
        return mapping.get(python_type, UnifiedNodeType.LANGUAGE_SPECIFIC)
    
    def extract_semantic_type(self, node: Any) -> SemanticNodeType:
        """Extrae el tipo semántico de un nodo de Python."""
        # Implementación básica
        return SemanticNodeType.UNKNOWN
    
    def unify_type(self, python_type: Any) -> UnifiedType:
        """Unifica un tipo de Python a un tipo unificado."""
        return self.type_mapper.map_type(python_type)


class TypeScriptUnifier(LanguageUnifier):
    """Unificador específico para TypeScript."""
    
    def __init__(self):
        self.type_mapper = TypeScriptTypeMapper()
        self.semantic_mapper = TypeScriptSemanticMapper()
    
    def language(self) -> ProgrammingLanguage:
        return ProgrammingLanguage.TYPESCRIPT
    
    async def unify_tree_sitter(self, tree: Any, content: str, file_path: Path) -> UnifiedAST:
        """Unifica un AST de tree-sitter de TypeScript."""
        root_node = UnifiedNode(
            id=NodeId(),
            node_type=UnifiedNodeType.PROGRAM,
            semantic_type=SemanticNodeType.DEFINITION,
            name="typescript_module",
            position=UnifiedPosition(
                start_line=1,
                start_column=1,
                end_line=len(content.split('\n')),
                end_column=1,
                start_byte=0,
                end_byte=len(content),
                file_path=file_path
            )
        )
        
        semantic_info = UnifiedSemanticInfo()
        metadata = UnifiedASTMetadata(
            original_language="typescript",
            parser_used="tree-sitter",
            unification_version=ASTVersion.V1.value,
            node_count=1,
            depth=1,
            complexity_score=1.0,
            created_at=datetime.now(timezone.utc)
        )
        
        return UnifiedAST(
            id=ASTId(),
            language="typescript",
            file_path=file_path,
            root_node=root_node,
            metadata=metadata,
            semantic_info=semantic_info,
            version=ASTVersion.V1,
            created_at=datetime.now(timezone.utc)
        )
    
    async def unify_specialized(self, analysis_result: SpecializedAnalysisResult) -> UnifiedAST:
        """Unifica un resultado de análisis especializado de TypeScript."""
        root_node = UnifiedNode(
            id=NodeId(),
            node_type=UnifiedNodeType.PROGRAM,
            semantic_type=SemanticNodeType.DEFINITION,
            name="typescript_module",
            position=UnifiedPosition(
                start_line=1,
                start_column=1,
                end_line=1,
                end_column=1,
                start_byte=0,
                end_byte=0,
                file_path=analysis_result.get_file_path()
            )
        )
        
        semantic_info = UnifiedSemanticInfo()
        metadata = UnifiedASTMetadata(
            original_language="typescript",
            parser_used="specialized",
            unification_version=ASTVersion.V1.value,
            node_count=1,
            depth=1,
            complexity_score=1.0,
            created_at=datetime.now(timezone.utc)
        )
        
        return UnifiedAST(
            id=ASTId(),
            language="typescript",
            file_path=analysis_result.get_file_path(),
            root_node=root_node,
            metadata=metadata,
            semantic_info=semantic_info,
            version=ASTVersion.V1,
            created_at=datetime.now(timezone.utc)
        )
    
    def map_node_type(self, typescript_type: str) -> UnifiedNodeType:
        """Mapea tipos de nodos específicos de TypeScript a tipos unificados."""
        mapping = {
            "program": UnifiedNodeType.PROGRAM,
            "function_declaration": UnifiedNodeType.FUNCTION_DECLARATION,
            "class_declaration": UnifiedNodeType.CLASS_DECLARATION,
            "interface_declaration": UnifiedNodeType.INTERFACE_DECLARATION,
            "if_statement": UnifiedNodeType.IF_STATEMENT,
            "for_statement": UnifiedNodeType.FOR_STATEMENT,
            "while_statement": UnifiedNodeType.WHILE_STATEMENT,
            "return_statement": UnifiedNodeType.RETURN_STATEMENT,
            "call_expression": UnifiedNodeType.CALL_EXPRESSION,
            "binary_expression": UnifiedNodeType.BINARY_EXPRESSION,
            "identifier": UnifiedNodeType.IDENTIFIER,
            "string": UnifiedNodeType.STRING_LITERAL,
            "number": UnifiedNodeType.NUMBER_LITERAL,
            "comment": UnifiedNodeType.COMMENT,
        }
        return mapping.get(typescript_type, UnifiedNodeType.LANGUAGE_SPECIFIC)
    
    def extract_semantic_type(self, node: Any) -> SemanticNodeType:
        """Extrae el tipo semántico de un nodo de TypeScript."""
        return SemanticNodeType.UNKNOWN
    
    def unify_type(self, typescript_type: Any) -> UnifiedType:
        """Unifica un tipo de TypeScript a un tipo unificado."""
        return self.type_mapper.map_type(typescript_type)


class JavaScriptUnifier(LanguageUnifier):
    """Unificador específico para JavaScript."""
    
    def __init__(self):
        self.type_mapper = JavaScriptTypeMapper()
        self.semantic_mapper = JavaScriptSemanticMapper()
    
    def language(self) -> ProgrammingLanguage:
        return ProgrammingLanguage.JAVASCRIPT
    
    async def unify_tree_sitter(self, tree: Any, content: str, file_path: Path) -> UnifiedAST:
        """Unifica un AST de tree-sitter de JavaScript."""
        root_node = UnifiedNode(
            id=NodeId(),
            node_type=UnifiedNodeType.PROGRAM,
            semantic_type=SemanticNodeType.DEFINITION,
            name="javascript_module",
            position=UnifiedPosition(
                start_line=1,
                start_column=1,
                end_line=len(content.split('\n')),
                end_column=1,
                start_byte=0,
                end_byte=len(content),
                file_path=file_path
            )
        )
        
        semantic_info = UnifiedSemanticInfo()
        metadata = UnifiedASTMetadata(
            original_language="javascript",
            parser_used="tree-sitter",
            unification_version=ASTVersion.V1.value,
            node_count=1,
            depth=1,
            complexity_score=1.0,
            created_at=datetime.now(timezone.utc)
        )
        
        return UnifiedAST(
            id=ASTId(),
            language="javascript",
            file_path=file_path,
            root_node=root_node,
            metadata=metadata,
            semantic_info=semantic_info,
            version=ASTVersion.V1,
            created_at=datetime.now(timezone.utc)
        )
    
    async def unify_specialized(self, analysis_result: SpecializedAnalysisResult) -> UnifiedAST:
        """Unifica un resultado de análisis especializado de JavaScript."""
        root_node = UnifiedNode(
            id=NodeId(),
            node_type=UnifiedNodeType.PROGRAM,
            semantic_type=SemanticNodeType.DEFINITION,
            name="javascript_module",
            position=UnifiedPosition(
                start_line=1,
                start_column=1,
                end_line=1,
                end_column=1,
                start_byte=0,
                end_byte=0,
                file_path=analysis_result.get_file_path()
            )
        )
        
        semantic_info = UnifiedSemanticInfo()
        metadata = UnifiedASTMetadata(
            original_language="javascript",
            parser_used="specialized",
            unification_version=ASTVersion.V1.value,
            node_count=1,
            depth=1,
            complexity_score=1.0,
            created_at=datetime.now(timezone.utc)
        )
        
        return UnifiedAST(
            id=ASTId(),
            language="javascript",
            file_path=analysis_result.get_file_path(),
            root_node=root_node,
            metadata=metadata,
            semantic_info=semantic_info,
            version=ASTVersion.V1,
            created_at=datetime.now(timezone.utc)
        )
    
    def map_node_type(self, javascript_type: str) -> UnifiedNodeType:
        """Mapea tipos de nodos específicos de JavaScript a tipos unificados."""
        mapping = {
            "program": UnifiedNodeType.PROGRAM,
            "function_declaration": UnifiedNodeType.FUNCTION_DECLARATION,
            "class_declaration": UnifiedNodeType.CLASS_DECLARATION,
            "if_statement": UnifiedNodeType.IF_STATEMENT,
            "for_statement": UnifiedNodeType.FOR_STATEMENT,
            "while_statement": UnifiedNodeType.WHILE_STATEMENT,
            "return_statement": UnifiedNodeType.RETURN_STATEMENT,
            "call_expression": UnifiedNodeType.CALL_EXPRESSION,
            "binary_expression": UnifiedNodeType.BINARY_EXPRESSION,
            "identifier": UnifiedNodeType.IDENTIFIER,
            "string": UnifiedNodeType.STRING_LITERAL,
            "number": UnifiedNodeType.NUMBER_LITERAL,
            "comment": UnifiedNodeType.COMMENT,
        }
        return mapping.get(javascript_type, UnifiedNodeType.LANGUAGE_SPECIFIC)
    
    def extract_semantic_type(self, node: Any) -> SemanticNodeType:
        """Extrae el tipo semántico de un nodo de JavaScript."""
        return SemanticNodeType.UNKNOWN
    
    def unify_type(self, javascript_type: Any) -> UnifiedType:
        """Unifica un tipo de JavaScript a un tipo unificado."""
        return self.type_mapper.map_type(javascript_type)


class RustUnifier(LanguageUnifier):
    """Unificador específico para Rust."""
    
    def __init__(self):
        self.type_mapper = RustTypeMapper()
        self.semantic_mapper = RustSemanticMapper()
    
    def language(self) -> ProgrammingLanguage:
        return ProgrammingLanguage.RUST
    
    async def unify_tree_sitter(self, tree: Any, content: str, file_path: Path) -> UnifiedAST:
        """Unifica un AST de tree-sitter de Rust."""
        root_node = UnifiedNode(
            id=NodeId(),
            node_type=UnifiedNodeType.PROGRAM,
            semantic_type=SemanticNodeType.DEFINITION,
            name="rust_module",
            position=UnifiedPosition(
                start_line=1,
                start_column=1,
                end_line=len(content.split('\n')),
                end_column=1,
                start_byte=0,
                end_byte=len(content),
                file_path=file_path
            )
        )
        
        semantic_info = UnifiedSemanticInfo()
        metadata = UnifiedASTMetadata(
            original_language="rust",
            parser_used="tree-sitter",
            unification_version=ASTVersion.V1.value,
            node_count=1,
            depth=1,
            complexity_score=1.0,
            created_at=datetime.now(timezone.utc)
        )
        
        return UnifiedAST(
            id=ASTId(),
            language="rust",
            file_path=file_path,
            root_node=root_node,
            metadata=metadata,
            semantic_info=semantic_info,
            version=ASTVersion.V1,
            created_at=datetime.now(timezone.utc)
        )
    
    async def unify_specialized(self, analysis_result: SpecializedAnalysisResult) -> UnifiedAST:
        """Unifica un resultado de análisis especializado de Rust."""
        root_node = UnifiedNode(
            id=NodeId(),
            node_type=UnifiedNodeType.PROGRAM,
            semantic_type=SemanticNodeType.DEFINITION,
            name="rust_module",
            position=UnifiedPosition(
                start_line=1,
                start_column=1,
                end_line=1,
                end_column=1,
                start_byte=0,
                end_byte=0,
                file_path=analysis_result.get_file_path()
            )
        )
        
        semantic_info = UnifiedSemanticInfo()
        metadata = UnifiedASTMetadata(
            original_language="rust",
            parser_used="specialized",
            unification_version=ASTVersion.V1.value,
            node_count=1,
            depth=1,
            complexity_score=1.0,
            created_at=datetime.now(timezone.utc)
        )
        
        return UnifiedAST(
            id=ASTId(),
            language="rust",
            file_path=analysis_result.get_file_path(),
            root_node=root_node,
            metadata=metadata,
            semantic_info=semantic_info,
            version=ASTVersion.V1,
            created_at=datetime.now(timezone.utc)
        )
    
    def map_node_type(self, rust_type: str) -> UnifiedNodeType:
        """Mapea tipos de nodos específicos de Rust a tipos unificados."""
        mapping = {
            "source_file": UnifiedNodeType.PROGRAM,
            "function_item": UnifiedNodeType.FUNCTION_DECLARATION,
            "struct_item": UnifiedNodeType.STRUCT_DECLARATION,
            "enum_item": UnifiedNodeType.ENUM_DECLARATION,
            "trait_item": UnifiedNodeType.TRAIT_DECLARATION,
            "impl_item": UnifiedNodeType.CLASS_DECLARATION,
            "if_expression": UnifiedNodeType.IF_STATEMENT,
            "for_expression": UnifiedNodeType.FOR_STATEMENT,
            "while_expression": UnifiedNodeType.WHILE_STATEMENT,
            "loop_expression": UnifiedNodeType.LOOP_STATEMENT,
            "return_expression": UnifiedNodeType.RETURN_STATEMENT,
            "call_expression": UnifiedNodeType.CALL_EXPRESSION,
            "binary_expression": UnifiedNodeType.BINARY_EXPRESSION,
            "identifier": UnifiedNodeType.IDENTIFIER,
            "string_literal": UnifiedNodeType.STRING_LITERAL,
            "integer_literal": UnifiedNodeType.NUMBER_LITERAL,
            "comment": UnifiedNodeType.COMMENT,
        }
        return mapping.get(rust_type, UnifiedNodeType.LANGUAGE_SPECIFIC)
    
    def extract_semantic_type(self, node: Any) -> SemanticNodeType:
        """Extrae el tipo semántico de un nodo de Rust."""
        return SemanticNodeType.UNKNOWN
    
    def unify_type(self, rust_type: Any) -> UnifiedType:
        """Unifica un tipo de Rust a un tipo unificado."""
        return self.type_mapper.map_type(rust_type)


# Mappers de tipos específicos por lenguaje
class PythonTypeMapper:
    """Mapeador de tipos de Python."""
    
    def map_type(self, python_type: Any) -> UnifiedType:
        """Mapea un tipo de Python a un tipo unificado."""
        if isinstance(python_type, str):
            return UnifiedType(
                type_name=python_type,
                is_primitive=self._is_primitive_type(python_type)
            )
        return UnifiedType(type_name="unknown")
    
    def _is_primitive_type(self, type_name: str) -> bool:
        """Determina si un tipo es primitivo en Python."""
        primitive_types = {"int", "float", "str", "bool", "None", "list", "dict", "tuple", "set"}
        return type_name in primitive_types


class TypeScriptTypeMapper:
    """Mapeador de tipos de TypeScript."""
    
    def map_type(self, typescript_type: Any) -> UnifiedType:
        """Mapea un tipo de TypeScript a un tipo unificado."""
        if isinstance(typescript_type, str):
            return UnifiedType(
                type_name=typescript_type,
                is_primitive=self._is_primitive_type(typescript_type)
            )
        return UnifiedType(type_name="unknown")
    
    def _is_primitive_type(self, type_name: str) -> bool:
        """Determina si un tipo es primitivo en TypeScript."""
        primitive_types = {"string", "number", "boolean", "null", "undefined", "void", "any", "unknown"}
        return type_name in primitive_types


class JavaScriptTypeMapper:
    """Mapeador de tipos de JavaScript."""
    
    def map_type(self, javascript_type: Any) -> UnifiedType:
        """Mapea un tipo de JavaScript a un tipo unificado."""
        if isinstance(javascript_type, str):
            return UnifiedType(
                type_name=javascript_type,
                is_primitive=self._is_primitive_type(javascript_type)
            )
        return UnifiedType(type_name="unknown")
    
    def _is_primitive_type(self, type_name: str) -> bool:
        """Determina si un tipo es primitivo en JavaScript."""
        primitive_types = {"string", "number", "boolean", "null", "undefined", "object", "function"}
        return type_name in primitive_types


class RustTypeMapper:
    """Mapeador de tipos de Rust."""
    
    def map_type(self, rust_type: Any) -> UnifiedType:
        """Mapea un tipo de Rust a un tipo unificado."""
        if isinstance(rust_type, str):
            return UnifiedType(
                type_name=rust_type,
                is_primitive=self._is_primitive_type(rust_type)
            )
        return UnifiedType(type_name="unknown")
    
    def _is_primitive_type(self, type_name: str) -> bool:
        """Determina si un tipo es primitivo en Rust."""
        primitive_types = {
            "i8", "i16", "i32", "i64", "i128", "isize",
            "u8", "u16", "u32", "u64", "u128", "usize",
            "f32", "f64", "bool", "char", "str", "String"
        }
        return type_name in primitive_types


# Mappers semánticos específicos por lenguaje
class PythonSemanticMapper:
    """Mapeador semántico de Python."""
    
    def map_semantic_type(self, node_type: str) -> SemanticNodeType:
        """Mapea un tipo de nodo de Python a un tipo semántico."""
        mapping = {
            "function_definition": SemanticNodeType.DEFINITION,
            "class_definition": SemanticNodeType.DEFINITION,
            "call": SemanticNodeType.CALL,
            "identifier": SemanticNodeType.REFERENCE,
            "string": SemanticNodeType.LITERAL,
            "integer": SemanticNodeType.LITERAL,
        }
        return mapping.get(node_type, SemanticNodeType.UNKNOWN)


class TypeScriptSemanticMapper:
    """Mapeador semántico de TypeScript."""
    
    def map_semantic_type(self, node_type: str) -> SemanticNodeType:
        """Mapea un tipo de nodo de TypeScript a un tipo semántico."""
        mapping = {
            "function_declaration": SemanticNodeType.DEFINITION,
            "class_declaration": SemanticNodeType.DEFINITION,
            "interface_declaration": SemanticNodeType.DEFINITION,
            "call_expression": SemanticNodeType.CALL,
            "identifier": SemanticNodeType.REFERENCE,
            "string": SemanticNodeType.LITERAL,
            "number": SemanticNodeType.LITERAL,
        }
        return mapping.get(node_type, SemanticNodeType.UNKNOWN)


class JavaScriptSemanticMapper:
    """Mapeador semántico de JavaScript."""
    
    def map_semantic_type(self, node_type: str) -> SemanticNodeType:
        """Mapea un tipo de nodo de JavaScript a un tipo semántico."""
        mapping = {
            "function_declaration": SemanticNodeType.DEFINITION,
            "class_declaration": SemanticNodeType.DEFINITION,
            "call_expression": SemanticNodeType.CALL,
            "identifier": SemanticNodeType.REFERENCE,
            "string": SemanticNodeType.LITERAL,
            "number": SemanticNodeType.LITERAL,
        }
        return mapping.get(node_type, SemanticNodeType.UNKNOWN)


class RustSemanticMapper:
    """Mapeador semántico de Rust."""
    
    def map_semantic_type(self, node_type: str) -> SemanticNodeType:
        """Mapea un tipo de nodo de Rust a un tipo semántico."""
        mapping = {
            "function_item": SemanticNodeType.DEFINITION,
            "struct_item": SemanticNodeType.DEFINITION,
            "enum_item": SemanticNodeType.DEFINITION,
            "trait_item": SemanticNodeType.DEFINITION,
            "call_expression": SemanticNodeType.CALL,
            "identifier": SemanticNodeType.REFERENCE,
            "string_literal": SemanticNodeType.LITERAL,
            "integer_literal": SemanticNodeType.LITERAL,
        }
        return mapping.get(node_type, SemanticNodeType.UNKNOWN)
