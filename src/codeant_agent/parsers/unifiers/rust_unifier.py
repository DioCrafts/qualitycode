"""
Unificador de AST para Rust.
"""

from pathlib import Path
from typing import Any
import logging
import re

from ..unified_ast import (
    ASTUnifier, UnifiedAST, UnifiedNode, UnifiedNodeType,
    UnifiedType, SemanticConcept, create_unified_ast
)
from ..universal.rust_parser import RustAnalysisResult

logger = logging.getLogger(__name__)


class RustASTUnifier(ASTUnifier):
    """Unificador de AST Rust a UnifiedAST."""
    
    def unify(self, source_ast: Any, file_path: Path) -> UnifiedAST:
        """Convierte análisis de Rust a UnifiedAST."""
        # Si es un RustAnalysisResult, usar eso
        if hasattr(source_ast, 'functions') and hasattr(source_ast, 'structs'):
            return self._unify_from_analysis(source_ast, file_path)
        
        # Si es un dict del parser fallback
        if isinstance(source_ast, dict) and 'content' in source_ast:
            return self._unify_from_content(source_ast['content'], file_path)
        
        raise ValueError("source_ast debe ser RustAnalysisResult o dict con content")
    
    def supported_language(self) -> str:
        """Retorna el lenguaje soportado."""
        return "rust"
    
    def _unify_from_analysis(self, analysis: RustAnalysisResult, file_path: Path) -> UnifiedAST:
        """Convierte desde RustAnalysisResult."""
        # Crear nodo raíz
        root = UnifiedNode(
            node_type=UnifiedNodeType.MODULE,
            name=file_path.stem,
            source_language="rust"
        )
        
        # Añadir funciones
        for func in analysis.functions:
            unified_func = UnifiedNode(
                node_type=UnifiedNodeType.FUNCTION,
                name=func.name,
                start_line=func.start_line,
                end_line=func.end_line,
                source_language="rust"
            )
            
            # Conceptos semánticos
            if func.is_async:
                unified_func.semantic_concepts.add(SemanticConcept.ASYNC_OPERATION)
            
            if func.is_unsafe:
                unified_func.semantic_concepts.add(SemanticConcept.MEMORY_MANAGEMENT)
            
            # Ownership patterns
            if func.ownership_info and func.ownership_info.has_borrows:
                unified_func.semantic_concepts.add(SemanticConcept.MEMORY_MANAGEMENT)
            
            # Tipo de retorno
            if func.return_type and func.return_type != '()':
                unified_func.unified_type = UnifiedType(
                    name=func.return_type,
                    category='object' if func.return_type[0].isupper() else 'primitive'
                )
            
            # Parámetros
            for param in func.parameters:
                param_parts = param.split(':')
                param_node = UnifiedNode(
                    node_type=UnifiedNodeType.PARAMETER,
                    name=param_parts[0].strip() if param_parts else param,
                    source_language="rust"
                )
                
                if len(param_parts) > 1:
                    type_str = param_parts[1].strip()
                    # Detectar referencias/borrowing
                    is_reference = '&' in type_str
                    is_mutable = '&mut' in type_str
                    
                    param_node.unified_type = UnifiedType(
                        name=type_str.replace('&mut ', '').replace('&', ''),
                        category='reference' if is_reference else 'owned'
                    )
                    
                    if is_reference:
                        param_node.metadata['is_borrowed'] = True
                        param_node.metadata['is_mutable'] = is_mutable
                
                unified_func.add_child(param_node)
            
            root.add_child(unified_func)
        
        # Añadir structs
        for struct in analysis.structs:
            unified_struct = UnifiedNode(
                node_type=UnifiedNodeType.STRUCT,
                name=struct.name,
                start_line=struct.start_line,
                end_line=struct.end_line,
                source_language="rust"
            )
            
            unified_struct.semantic_concepts.add(SemanticConcept.OBJECT_ORIENTED)
            
            # Detectar patrones
            if struct.derives and 'Clone' in struct.derives:
                unified_struct.metadata['cloneable'] = True
            
            if struct.derives and 'Copy' in struct.derives:
                unified_struct.metadata['copyable'] = True
            
            # Añadir campos
            for field in struct.fields:
                field_node = UnifiedNode(
                    node_type=UnifiedNodeType.FIELD,
                    name=field.name,
                    start_line=field.start_line,
                    source_language="rust"
                )
                if field.field_type:
                    field_node.unified_type = UnifiedType(
                        name=field.field_type,
                        category='object' if field.field_type[0].isupper() else 'primitive'
                    )
                unified_struct.add_child(field_node)
            
            root.add_child(unified_struct)
        
        # Añadir enums
        for enum in analysis.enums:
            unified_enum = UnifiedNode(
                node_type=UnifiedNodeType.ENUM,
                name=enum.name,
                start_line=enum.start_line,
                end_line=enum.end_line,
                source_language="rust"
            )
            
            # Los enums en Rust son más poderosos que en otros lenguajes
            unified_enum.semantic_concepts.add(SemanticConcept.TYPE_SAFETY)
            if any(v.has_data for v in enum.variants):
                unified_enum.semantic_concepts.add(SemanticConcept.FUNCTIONAL)
            
            root.add_child(unified_enum)
        
        # Añadir traits
        for trait in analysis.traits:
            unified_trait = UnifiedNode(
                node_type=UnifiedNodeType.TRAIT,
                name=trait.name,
                start_line=trait.start_line,
                end_line=trait.end_line,
                source_language="rust"
            )
            
            unified_trait.semantic_concepts.add(SemanticConcept.OBJECT_ORIENTED)
            unified_trait.semantic_concepts.add(SemanticConcept.TYPE_SAFETY)
            
            root.add_child(unified_trait)
        
        # Añadir uses (imports)
        for use in analysis.uses:
            unified_import = UnifiedNode(
                node_type=UnifiedNodeType.IMPORT,
                name=use.path,
                start_line=use.start_line,
                source_language="rust"
            )
            root.add_child(unified_import)
        
        return create_unified_ast(root, file_path, "rust")
    
    def _unify_from_content(self, content: str, file_path: Path) -> UnifiedAST:
        """Convierte desde contenido raw usando regex."""
        root = UnifiedNode(
            node_type=UnifiedNodeType.MODULE,
            name=file_path.stem,
            source_language="rust"
        )
        
        lines = content.split('\n')
        
        # Extraer funciones
        func_pattern = r'(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)'
        
        for i, line in enumerate(lines, 1):
            # Funciones
            func_match = re.search(func_pattern, line)
            if func_match:
                func_node = UnifiedNode(
                    node_type=UnifiedNodeType.FUNCTION,
                    name=func_match.group(1),
                    start_line=i,
                    end_line=i,
                    source_language="rust"
                )
                
                if 'async' in line:
                    func_node.semantic_concepts.add(SemanticConcept.ASYNC_OPERATION)
                if 'unsafe' in line:
                    func_node.semantic_concepts.add(SemanticConcept.MEMORY_MANAGEMENT)
                
                root.add_child(func_node)
            
            # Structs
            struct_match = re.search(r'(?:pub\s+)?struct\s+(\w+)', line)
            if struct_match:
                struct_node = UnifiedNode(
                    node_type=UnifiedNodeType.STRUCT,
                    name=struct_match.group(1),
                    start_line=i,
                    end_line=i,
                    source_language="rust"
                )
                struct_node.semantic_concepts.add(SemanticConcept.OBJECT_ORIENTED)
                root.add_child(struct_node)
            
            # Enums
            enum_match = re.search(r'(?:pub\s+)?enum\s+(\w+)', line)
            if enum_match:
                enum_node = UnifiedNode(
                    node_type=UnifiedNodeType.ENUM,
                    name=enum_match.group(1),
                    start_line=i,
                    end_line=i,
                    source_language="rust"
                )
                enum_node.semantic_concepts.add(SemanticConcept.TYPE_SAFETY)
                root.add_child(enum_node)
            
            # Traits
            trait_match = re.search(r'(?:pub\s+)?trait\s+(\w+)', line)
            if trait_match:
                trait_node = UnifiedNode(
                    node_type=UnifiedNodeType.TRAIT,
                    name=trait_match.group(1),
                    start_line=i,
                    end_line=i,
                    source_language="rust"
                )
                trait_node.semantic_concepts.add(SemanticConcept.OBJECT_ORIENTED)
                trait_node.semantic_concepts.add(SemanticConcept.TYPE_SAFETY)
                root.add_child(trait_node)
            
            # Uses (imports)
            use_match = re.search(r'use\s+([^;]+);', line)
            if use_match:
                use_node = UnifiedNode(
                    node_type=UnifiedNodeType.IMPORT,
                    name=use_match.group(1).strip(),
                    start_line=i,
                    source_language="rust"
                )
                root.add_child(use_node)
            
            # Match expressions
            if re.search(r'\bmatch\s+', line):
                match_node = UnifiedNode(
                    node_type=UnifiedNodeType.MATCH_STATEMENT,
                    start_line=i,
                    source_language="rust"
                )
                match_node.semantic_concepts.add(SemanticConcept.FUNCTIONAL)
                match_node.semantic_concepts.add(SemanticConcept.TYPE_SAFETY)
                root.add_child(match_node)
            
            # Control de flujo
            if re.search(r'\bif\s+', line):
                if_node = UnifiedNode(
                    node_type=UnifiedNodeType.IF_STATEMENT,
                    start_line=i,
                    source_language="rust"
                )
                root.add_child(if_node)
            
            if re.search(r'\bfor\s+', line):
                for_node = UnifiedNode(
                    node_type=UnifiedNodeType.FOR_LOOP,
                    start_line=i,
                    source_language="rust"
                )
                for_node.semantic_concepts.add(SemanticConcept.ITERATION)
                
                # Detectar iteradores funcionales
                if '.iter()' in line or '.map(' in line or '.filter(' in line:
                    for_node.semantic_concepts.add(SemanticConcept.FUNCTIONAL)
                
                root.add_child(for_node)
            
            if re.search(r'\bwhile\s+', line):
                while_node = UnifiedNode(
                    node_type=UnifiedNodeType.WHILE_LOOP,
                    start_line=i,
                    source_language="rust"
                )
                while_node.semantic_concepts.add(SemanticConcept.ITERATION)
                root.add_child(while_node)
            
            # loop (específico de Rust)
            if re.search(r'\bloop\s*\{', line):
                loop_node = UnifiedNode(
                    node_type=UnifiedNodeType.WHILE_LOOP,
                    start_line=i,
                    source_language="rust"
                )
                loop_node.semantic_concepts.add(SemanticConcept.ITERATION)
                loop_node.metadata['infinite_loop'] = True
                root.add_child(loop_node)
        
        return create_unified_ast(root, file_path, "rust")
