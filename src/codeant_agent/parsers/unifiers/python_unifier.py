"""
Unificador de AST para Python.
"""

import ast
from pathlib import Path
from typing import Any, Optional
import logging

from ..unified_ast import (
    ASTUnifier, UnifiedAST, UnifiedNode, UnifiedNodeType,
    UnifiedType, SemanticConcept, create_unified_ast
)

logger = logging.getLogger(__name__)


class PythonASTUnifier(ASTUnifier):
    """Unificador de AST Python a UnifiedAST."""
    
    def unify(self, source_ast: Any, file_path: Path) -> UnifiedAST:
        """Convierte un AST de Python a UnifiedAST."""
        if not isinstance(source_ast, ast.AST):
            raise ValueError("source_ast debe ser un ast.AST de Python")
        
        # Crear nodo raíz
        root = UnifiedNode(
            node_type=UnifiedNodeType.MODULE,
            name=file_path.stem,
            source_language="python"
        )
        
        # Convertir recursivamente
        self._convert_node(source_ast, root)
        
        return create_unified_ast(root, file_path, "python")
    
    def supported_language(self) -> str:
        """Retorna el lenguaje soportado."""
        return "python"
    
    def _convert_node(self, py_node: ast.AST, parent: UnifiedNode):
        """Convierte un nodo Python AST a UnifiedNode."""
        if isinstance(py_node, ast.FunctionDef) or isinstance(py_node, ast.AsyncFunctionDef):
            self._convert_function(py_node, parent)
        elif isinstance(py_node, ast.ClassDef):
            self._convert_class(py_node, parent)
        elif isinstance(py_node, ast.If):
            self._convert_if(py_node, parent)
        elif isinstance(py_node, ast.For) or isinstance(py_node, ast.AsyncFor):
            self._convert_for(py_node, parent)
        elif isinstance(py_node, ast.While):
            self._convert_while(py_node, parent)
        elif isinstance(py_node, ast.Import) or isinstance(py_node, ast.ImportFrom):
            self._convert_import(py_node, parent)
        elif isinstance(py_node, ast.Assign):
            self._convert_assignment(py_node, parent)
        elif isinstance(py_node, ast.Return):
            self._convert_return(py_node, parent)
        elif isinstance(py_node, ast.Try):
            self._convert_try(py_node, parent)
        else:
            # Para otros nodos, procesar hijos
            for child in ast.iter_child_nodes(py_node):
                self._convert_node(child, parent)
    
    def _convert_function(self, func_node: ast.FunctionDef, parent: UnifiedNode):
        """Convierte una función Python."""
        unified_func = UnifiedNode(
            node_type=UnifiedNodeType.FUNCTION,
            name=func_node.name,
            start_line=func_node.lineno,
            end_line=func_node.end_lineno or func_node.lineno,
            source_language="python",
            original_node=func_node
        )
        
        # Añadir conceptos semánticos
        if isinstance(func_node, ast.AsyncFunctionDef):
            unified_func.semantic_concepts.add(SemanticConcept.ASYNC_OPERATION)
        
        # Detectar patrones funcionales
        if any(isinstance(node, ast.Return) and isinstance(node.value, ast.Lambda) 
               for node in ast.walk(func_node)):
            unified_func.semantic_concepts.add(SemanticConcept.FUNCTIONAL)
        
        # Añadir tipo si está anotado
        if func_node.returns:
            unified_func.unified_type = self._extract_type(func_node.returns)
        
        # Añadir parámetros
        for arg in func_node.args.args:
            param = UnifiedNode(
                node_type=UnifiedNodeType.PARAMETER,
                name=arg.arg,
                source_language="python"
            )
            if arg.annotation:
                param.unified_type = self._extract_type(arg.annotation)
            unified_func.add_child(param)
        
        parent.add_child(unified_func)
        
        # Procesar cuerpo
        for stmt in func_node.body:
            self._convert_node(stmt, unified_func)
    
    def _convert_class(self, class_node: ast.ClassDef, parent: UnifiedNode):
        """Convierte una clase Python."""
        unified_class = UnifiedNode(
            node_type=UnifiedNodeType.CLASS,
            name=class_node.name,
            start_line=class_node.lineno,
            end_line=class_node.end_lineno or class_node.lineno,
            source_language="python",
            original_node=class_node
        )
        
        # Añadir conceptos semánticos
        unified_class.semantic_concepts.add(SemanticConcept.OBJECT_ORIENTED)
        
        # Detectar patrones de diseño
        method_names = {node.name for node in class_node.body 
                       if isinstance(node, ast.FunctionDef)}
        
        if '__new__' in method_names or ('instance' in method_names and 'get_instance' in method_names):
            unified_class.semantic_concepts.add(SemanticConcept.SINGLETON)
        
        parent.add_child(unified_class)
        
        # Procesar métodos y atributos
        for item in class_node.body:
            self._convert_node(item, unified_class)
    
    def _convert_if(self, if_node: ast.If, parent: UnifiedNode):
        """Convierte un if statement."""
        unified_if = UnifiedNode(
            node_type=UnifiedNodeType.IF_STATEMENT,
            start_line=if_node.lineno,
            end_line=if_node.end_lineno or if_node.lineno,
            source_language="python",
            original_node=if_node
        )
        
        parent.add_child(unified_if)
        
        # Procesar ramas
        for stmt in if_node.body:
            self._convert_node(stmt, unified_if)
        
        for stmt in if_node.orelse:
            self._convert_node(stmt, unified_if)
    
    def _convert_for(self, for_node: ast.For, parent: UnifiedNode):
        """Convierte un for loop."""
        unified_for = UnifiedNode(
            node_type=UnifiedNodeType.FOR_LOOP,
            start_line=for_node.lineno,
            end_line=for_node.end_lineno or for_node.lineno,
            source_language="python",
            original_node=for_node
        )
        
        # Detectar patrones funcionales (map, filter, etc.)
        if isinstance(for_node.iter, ast.Call):
            func_name = getattr(for_node.iter.func, 'id', '')
            if func_name in ['map', 'filter', 'reduce']:
                unified_for.semantic_concepts.add(SemanticConcept.FUNCTIONAL)
                if func_name == 'map':
                    unified_for.semantic_concepts.add(SemanticConcept.MAPPING)
                elif func_name == 'filter':
                    unified_for.semantic_concepts.add(SemanticConcept.FILTERING)
        
        unified_for.semantic_concepts.add(SemanticConcept.ITERATION)
        
        parent.add_child(unified_for)
        
        # Procesar cuerpo
        for stmt in for_node.body:
            self._convert_node(stmt, unified_for)
    
    def _convert_while(self, while_node: ast.While, parent: UnifiedNode):
        """Convierte un while loop."""
        unified_while = UnifiedNode(
            node_type=UnifiedNodeType.WHILE_LOOP,
            start_line=while_node.lineno,
            end_line=while_node.end_lineno or while_node.lineno,
            source_language="python",
            original_node=while_node
        )
        
        unified_while.semantic_concepts.add(SemanticConcept.ITERATION)
        parent.add_child(unified_while)
        
        # Procesar cuerpo
        for stmt in while_node.body:
            self._convert_node(stmt, unified_while)
    
    def _convert_import(self, import_node: ast.Import, parent: UnifiedNode):
        """Convierte un import."""
        for alias in import_node.names:
            unified_import = UnifiedNode(
                node_type=UnifiedNodeType.IMPORT,
                name=alias.name,
                start_line=import_node.lineno,
                source_language="python",
                original_node=import_node
            )
            parent.add_child(unified_import)
    
    def _convert_assignment(self, assign_node: ast.Assign, parent: UnifiedNode):
        """Convierte una asignación."""
        for target in assign_node.targets:
            if isinstance(target, ast.Name):
                unified_var = UnifiedNode(
                    node_type=UnifiedNodeType.VARIABLE,
                    name=target.id,
                    start_line=assign_node.lineno,
                    source_language="python",
                    original_node=assign_node
                )
                
                # Detectar constantes
                if target.id.isupper():
                    unified_var.node_type = UnifiedNodeType.CONSTANT
                
                parent.add_child(unified_var)
    
    def _convert_return(self, return_node: ast.Return, parent: UnifiedNode):
        """Convierte un return statement."""
        unified_return = UnifiedNode(
            node_type=UnifiedNodeType.RETURN,
            start_line=return_node.lineno,
            source_language="python",
            original_node=return_node
        )
        parent.add_child(unified_return)
    
    def _convert_try(self, try_node: ast.Try, parent: UnifiedNode):
        """Convierte un try-except."""
        unified_try = UnifiedNode(
            node_type=UnifiedNodeType.TRY_CATCH,
            start_line=try_node.lineno,
            end_line=try_node.end_lineno or try_node.lineno,
            source_language="python",
            original_node=try_node
        )
        
        unified_try.semantic_concepts.add(SemanticConcept.ERROR_HANDLING)
        parent.add_child(unified_try)
        
        # Procesar bloques
        for stmt in try_node.body:
            self._convert_node(stmt, unified_try)
        
        for handler in try_node.handlers:
            for stmt in handler.body:
                self._convert_node(stmt, unified_try)
    
    def _extract_type(self, annotation: ast.AST) -> UnifiedType:
        """Extrae información de tipo de una anotación."""
        if isinstance(annotation, ast.Name):
            return UnifiedType(
                name=annotation.id,
                category='object' if annotation.id[0].isupper() else 'primitive'
            )
        elif isinstance(annotation, ast.Constant):
            return UnifiedType(
                name=str(annotation.value),
                category='literal'
            )
        elif isinstance(annotation, ast.Subscript):
            # Tipo genérico como List[int]
            base_type = self._extract_type(annotation.value)
            if hasattr(annotation.slice, 'value'):
                generic_type = self._extract_type(annotation.slice.value)
                base_type.generics.append(generic_type)
            return base_type
        else:
            return UnifiedType(name='unknown', category='unknown')
