"""
Unificador de AST para TypeScript/JavaScript.
"""

from pathlib import Path
from typing import Any, Dict, List
import logging
import re

from ..unified_ast import (
    ASTUnifier, UnifiedAST, UnifiedNode, UnifiedNodeType,
    UnifiedType, SemanticConcept, create_unified_ast
)
from ..universal.typescript_parser import JSAnalysisResult

logger = logging.getLogger(__name__)


class TypeScriptASTUnifier(ASTUnifier):
    """Unificador de AST TypeScript/JavaScript a UnifiedAST."""
    
    def unify(self, source_ast: Any, file_path: Path) -> UnifiedAST:
        """Convierte análisis de TypeScript/JavaScript a UnifiedAST."""
        # Si es un JSAnalysisResult, usar eso
        if hasattr(source_ast, 'functions') and hasattr(source_ast, 'classes'):
            return self._unify_from_analysis(source_ast, file_path)
        
        # Si es un dict del parser fallback
        if isinstance(source_ast, dict) and 'content' in source_ast:
            return self._unify_from_content(source_ast['content'], file_path)
        
        raise ValueError("source_ast debe ser JSAnalysisResult o dict con content")
    
    def supported_language(self) -> str:
        """Retorna el lenguaje soportado."""
        return "typescript"
    
    def _unify_from_analysis(self, analysis: JSAnalysisResult, file_path: Path) -> UnifiedAST:
        """Convierte desde JSAnalysisResult."""
        # Crear nodo raíz
        root = UnifiedNode(
            node_type=UnifiedNodeType.MODULE,
            name=file_path.stem,
            source_language=analysis.language.value if hasattr(analysis.language, 'value') else "typescript"
        )
        
        # Añadir funciones
        for func in analysis.functions:
            unified_func = UnifiedNode(
                node_type=UnifiedNodeType.FUNCTION,
                name=func.name,
                start_line=func.start_line,
                end_line=func.end_line,
                source_language="typescript"
            )
            
            # Conceptos semánticos
            if func.is_async:
                unified_func.semantic_concepts.add(SemanticConcept.ASYNC_OPERATION)
            
            if func.is_generator:
                unified_func.semantic_concepts.add(SemanticConcept.FUNCTIONAL)
            
            # Tipo de retorno
            if func.return_type:
                unified_func.unified_type = UnifiedType(
                    name=func.return_type,
                    category='object' if func.return_type[0].isupper() else 'primitive'
                )
            
            # Parámetros
            for param in func.parameters:
                param_node = UnifiedNode(
                    node_type=UnifiedNodeType.PARAMETER,
                    name=param.split(':')[0].strip() if ':' in param else param,
                    source_language="typescript"
                )
                if ':' in param:
                    type_str = param.split(':')[1].strip()
                    param_node.unified_type = UnifiedType(
                        name=type_str,
                        category='object' if type_str[0].isupper() else 'primitive'
                    )
                unified_func.add_child(param_node)
            
            root.add_child(unified_func)
        
        # Añadir clases
        for cls in analysis.classes:
            unified_class = UnifiedNode(
                node_type=UnifiedNodeType.CLASS,
                name=cls.name,
                start_line=cls.start_line,
                end_line=cls.end_line,
                source_language="typescript"
            )
            
            unified_class.semantic_concepts.add(SemanticConcept.OBJECT_ORIENTED)
            
            # Detectar patrones
            method_names = [m.name for m in cls.methods]
            if 'getInstance' in method_names or 'instance' in method_names:
                unified_class.semantic_concepts.add(SemanticConcept.SINGLETON)
            
            # Añadir métodos
            for method in cls.methods:
                method_node = UnifiedNode(
                    node_type=UnifiedNodeType.METHOD,
                    name=method.name,
                    start_line=method.start_line,
                    end_line=method.end_line,
                    source_language="typescript"
                )
                if method.is_async:
                    method_node.semantic_concepts.add(SemanticConcept.ASYNC_OPERATION)
                unified_class.add_child(method_node)
            
            # Añadir propiedades
            for prop in cls.properties:
                prop_node = UnifiedNode(
                    node_type=UnifiedNodeType.PROPERTY,
                    name=prop.name,
                    start_line=prop.start_line,
                    source_language="typescript"
                )
                if prop.type_annotation:
                    prop_node.unified_type = UnifiedType(
                        name=prop.type_annotation,
                        category='object' if prop.type_annotation[0].isupper() else 'primitive'
                    )
                unified_class.add_child(prop_node)
            
            root.add_child(unified_class)
        
        # Añadir imports
        for imp in analysis.imports:
            unified_import = UnifiedNode(
                node_type=UnifiedNodeType.IMPORT,
                name=imp.module_name,
                start_line=imp.start_line,
                source_language="typescript"
            )
            root.add_child(unified_import)
        
        # Añadir exports
        for exp in analysis.exports:
            unified_export = UnifiedNode(
                node_type=UnifiedNodeType.EXPORT,
                name=exp.export_name,
                start_line=exp.start_line,
                source_language="typescript"
            )
            root.add_child(unified_export)
        
        return create_unified_ast(root, file_path, "typescript")
    
    def _unify_from_content(self, content: str, file_path: Path) -> UnifiedAST:
        """Convierte desde contenido raw usando regex."""
        root = UnifiedNode(
            node_type=UnifiedNodeType.MODULE,
            name=file_path.stem,
            source_language="typescript"
        )
        
        lines = content.split('\n')
        
        # Extraer funciones
        func_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)'
        arrow_pattern = r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::[^=]+)?\s*=>'
        
        for i, line in enumerate(lines, 1):
            # Funciones tradicionales
            func_match = re.search(func_pattern, line)
            if func_match:
                func_node = UnifiedNode(
                    node_type=UnifiedNodeType.FUNCTION,
                    name=func_match.group(1),
                    start_line=i,
                    end_line=i,
                    source_language="typescript"
                )
                if 'async' in line:
                    func_node.semantic_concepts.add(SemanticConcept.ASYNC_OPERATION)
                root.add_child(func_node)
            
            # Arrow functions
            arrow_match = re.search(arrow_pattern, line)
            if arrow_match:
                func_node = UnifiedNode(
                    node_type=UnifiedNodeType.FUNCTION,
                    name=arrow_match.group(1),
                    start_line=i,
                    end_line=i,
                    source_language="typescript"
                )
                if 'async' in line:
                    func_node.semantic_concepts.add(SemanticConcept.ASYNC_OPERATION)
                func_node.semantic_concepts.add(SemanticConcept.FUNCTIONAL)
                root.add_child(func_node)
            
            # Clases
            class_match = re.search(r'(?:export\s+)?class\s+(\w+)', line)
            if class_match:
                class_node = UnifiedNode(
                    node_type=UnifiedNodeType.CLASS,
                    name=class_match.group(1),
                    start_line=i,
                    end_line=i,
                    source_language="typescript"
                )
                class_node.semantic_concepts.add(SemanticConcept.OBJECT_ORIENTED)
                root.add_child(class_node)
            
            # Interfaces
            interface_match = re.search(r'(?:export\s+)?interface\s+(\w+)', line)
            if interface_match:
                interface_node = UnifiedNode(
                    node_type=UnifiedNodeType.INTERFACE,
                    name=interface_match.group(1),
                    start_line=i,
                    end_line=i,
                    source_language="typescript"
                )
                interface_node.semantic_concepts.add(SemanticConcept.TYPE_SAFETY)
                root.add_child(interface_node)
            
            # Imports
            import_match = re.search(r'import\s+.*\s+from\s+[\'"]([^\'\"]+)[\'"]', line)
            if import_match:
                import_node = UnifiedNode(
                    node_type=UnifiedNodeType.IMPORT,
                    name=import_match.group(1),
                    start_line=i,
                    source_language="typescript"
                )
                root.add_child(import_node)
            
            # Control de flujo
            if re.search(r'\bif\s*\(', line):
                if_node = UnifiedNode(
                    node_type=UnifiedNodeType.IF_STATEMENT,
                    start_line=i,
                    source_language="typescript"
                )
                root.add_child(if_node)
            
            if re.search(r'\bfor\s*\(', line):
                for_node = UnifiedNode(
                    node_type=UnifiedNodeType.FOR_LOOP,
                    start_line=i,
                    source_language="typescript"
                )
                for_node.semantic_concepts.add(SemanticConcept.ITERATION)
                root.add_child(for_node)
            
            if re.search(r'\bwhile\s*\(', line):
                while_node = UnifiedNode(
                    node_type=UnifiedNodeType.WHILE_LOOP,
                    start_line=i,
                    source_language="typescript"
                )
                while_node.semantic_concepts.add(SemanticConcept.ITERATION)
                root.add_child(while_node)
            
            # Try-catch
            if re.search(r'\btry\s*\{', line):
                try_node = UnifiedNode(
                    node_type=UnifiedNodeType.TRY_CATCH,
                    start_line=i,
                    source_language="typescript"
                )
                try_node.semantic_concepts.add(SemanticConcept.ERROR_HANDLING)
                root.add_child(try_node)
        
        return create_unified_ast(root, file_path, "typescript")
