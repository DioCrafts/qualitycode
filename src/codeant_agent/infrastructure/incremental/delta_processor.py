"""
Procesador de deltas para el Sistema de Análisis Incremental.

Este módulo implementa el procesamiento eficiente de cambios diferenciales
para actualizar análisis existentes sin recalcular todo.
"""

import asyncio
import ast
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import difflib

from ...domain.entities.incremental import (
    GranularChange, ChangeType, DeltaAnalysisResult,
    ReusableComponents, ChangeLocation
)
from ...domain.services.incremental_service import DeltaProcessingService
from ...application.ports.incremental_ports import (
    AnalysisEngineOutputPort, MetricsCollectorOutputPort
)
from .incremental_config import IncrementalConfig


@dataclass
class DeltaOperation:
    """Operación de delta a aplicar."""
    operation_type: str  # add, remove, modify, replace
    path: List[str]     # Ruta en la estructura del análisis
    value: Any          # Valor a aplicar
    old_value: Optional[Any] = None  # Valor anterior (para modify)


@dataclass
class ASTDelta:
    """Delta entre dos ASTs."""
    added_nodes: List[ast.AST]
    removed_nodes: List[ast.AST]
    modified_nodes: List[Tuple[ast.AST, ast.AST]]  # (old, new)
    structural_changes: bool


class DeltaProcessor(DeltaProcessingService):
    """Procesador de deltas para análisis incremental."""
    
    def __init__(
        self,
        config: IncrementalConfig,
        analysis_engine: AnalysisEngineOutputPort,
        metrics_collector: MetricsCollectorOutputPort
    ):
        self.config = config
        self.analysis_engine = analysis_engine
        self.metrics_collector = metrics_collector
        
        # Cache de deltas computados
        self._delta_cache: Dict[str, Dict[str, Any]] = {}
        
        # Estrategias de merge por tipo de análisis
        self._merge_strategies = {
            'complexity': self._merge_complexity_analysis,
            'security': self._merge_security_analysis,
            'quality': self._merge_quality_analysis,
            'dependencies': self._merge_dependency_analysis,
            'metrics': self._merge_metrics_analysis
        }
    
    async def compute_ast_delta(
        self,
        old_ast: Any,
        new_ast: Any
    ) -> Dict[str, Any]:
        """
        Computar delta entre ASTs.
        
        Args:
            old_ast: AST anterior
            new_ast: AST nuevo
            
        Returns:
            Delta estructurado
        """
        start_time = datetime.now()
        
        # Generar clave de cache
        cache_key = self._generate_delta_cache_key(old_ast, new_ast)
        if cache_key in self._delta_cache:
            return self._delta_cache[cache_key]
        
        try:
            # Computar delta según tipo de AST
            if isinstance(old_ast, ast.AST) and isinstance(new_ast, ast.AST):
                ast_delta = await self._compute_python_ast_delta(old_ast, new_ast)
            else:
                # Para otros tipos de AST
                ast_delta = await self._compute_generic_delta(old_ast, new_ast)
            
            # Convertir a formato serializable
            delta = {
                'type': 'ast_delta',
                'timestamp': datetime.now().isoformat(),
                'added': self._serialize_nodes(ast_delta.added_nodes),
                'removed': self._serialize_nodes(ast_delta.removed_nodes),
                'modified': [
                    {
                        'old': self._serialize_node(old),
                        'new': self._serialize_node(new)
                    }
                    for old, new in ast_delta.modified_nodes
                ],
                'structural_changes': ast_delta.structural_changes,
                'computation_time_ms': int((datetime.now() - start_time).total_seconds() * 1000)
            }
            
            # Cachear resultado
            self._delta_cache[cache_key] = delta
            
            # Registrar métricas
            await self.metrics_collector.record_analysis_time(
                operation="compute_ast_delta",
                duration_ms=delta['computation_time_ms'],
                incremental=True
            )
            
            return delta
            
        except Exception as e:
            raise Exception(f"Error computing AST delta: {str(e)}")
    
    async def apply_delta_to_analysis(
        self,
        base_analysis: Any,
        delta: Dict[str, Any]
    ) -> Any:
        """
        Aplicar delta a un análisis base.
        
        Args:
            base_analysis: Análisis base
            delta: Delta a aplicar
            
        Returns:
            Análisis actualizado
        """
        start_time = datetime.now()
        
        try:
            # Clonar análisis base para no modificar el original
            updated_analysis = self._deep_copy(base_analysis)
            
            # Determinar tipo de análisis
            analysis_type = self._detect_analysis_type(base_analysis)
            
            # Aplicar operaciones del delta
            if 'operations' in delta:
                for operation in delta['operations']:
                    updated_analysis = await self._apply_delta_operation(
                        updated_analysis,
                        operation,
                        analysis_type
                    )
            
            # Si es un AST delta, aplicar cambios específicos
            elif delta.get('type') == 'ast_delta':
                updated_analysis = await self._apply_ast_delta(
                    updated_analysis,
                    delta,
                    analysis_type
                )
            
            # Recalcular métricas agregadas
            updated_analysis = await self._recalculate_aggregates(
                updated_analysis,
                analysis_type
            )
            
            # Validar resultado
            if not await self.validate_delta_consistency(
                base_analysis,
                delta,
                updated_analysis
            ):
                raise Exception("Delta application resulted in inconsistent state")
            
            # Registrar métricas
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            await self.metrics_collector.record_analysis_time(
                operation="apply_delta",
                duration_ms=duration_ms,
                incremental=True
            )
            
            return updated_analysis
            
        except Exception as e:
            raise Exception(f"Error applying delta: {str(e)}")
    
    async def compress_delta(self, delta: Dict[str, Any]) -> bytes:
        """
        Comprimir delta para almacenamiento eficiente.
        
        Args:
            delta: Delta a comprimir
            
        Returns:
            Delta comprimido
        """
        import zlib
        import pickle
        
        try:
            # Serializar delta
            serialized = pickle.dumps(delta)
            
            # Comprimir
            compressed = zlib.compress(serialized, level=9)
            
            # Verificar que la compresión vale la pena
            if len(compressed) < len(serialized) * 0.9:
                return compressed
            else:
                return serialized
                
        except Exception as e:
            raise Exception(f"Error compressing delta: {str(e)}")
    
    async def merge_deltas(
        self,
        deltas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fusionar múltiples deltas.
        
        Args:
            deltas: Lista de deltas a fusionar
            
        Returns:
            Delta fusionado
        """
        if not deltas:
            return {}
        
        if len(deltas) == 1:
            return deltas[0]
        
        try:
            # Inicializar delta fusionado
            merged = {
                'type': 'merged_delta',
                'timestamp': datetime.now().isoformat(),
                'source_deltas': len(deltas),
                'operations': []
            }
            
            # Fusionar operaciones
            all_operations = []
            for delta in deltas:
                if 'operations' in delta:
                    all_operations.extend(delta['operations'])
            
            # Optimizar operaciones (eliminar redundancias)
            optimized_ops = await self._optimize_operations(all_operations)
            merged['operations'] = optimized_ops
            
            # Si todos son AST deltas, fusionar cambios
            if all(d.get('type') == 'ast_delta' for d in deltas):
                merged['type'] = 'ast_delta'
                merged['added'] = []
                merged['removed'] = []
                merged['modified'] = []
                
                for delta in deltas:
                    merged['added'].extend(delta.get('added', []))
                    merged['removed'].extend(delta.get('removed', []))
                    merged['modified'].extend(delta.get('modified', []))
                
                # Detectar cambios estructurales
                merged['structural_changes'] = any(
                    d.get('structural_changes', False) for d in deltas
                )
            
            return merged
            
        except Exception as e:
            raise Exception(f"Error merging deltas: {str(e)}")
    
    async def validate_delta_consistency(
        self,
        base: Any,
        delta: Dict[str, Any],
        expected: Any
    ) -> bool:
        """
        Validar consistencia de aplicación de delta.
        
        Args:
            base: Estado base
            delta: Delta aplicado
            expected: Resultado esperado
            
        Returns:
            True si es consistente
        """
        try:
            # Aplicar delta al base
            result = await self.apply_delta_to_analysis(base, delta)
            
            # Comparar con esperado
            return self._compare_analysis_results(result, expected)
            
        except Exception:
            return False
    
    # Métodos auxiliares privados
    
    async def _compute_python_ast_delta(
        self,
        old_ast: ast.AST,
        new_ast: ast.AST
    ) -> ASTDelta:
        """Computar delta entre ASTs de Python."""
        # Extraer nodos de ambos ASTs
        old_nodes = self._extract_ast_nodes(old_ast)
        new_nodes = self._extract_ast_nodes(new_ast)
        
        added_nodes = []
        removed_nodes = []
        modified_nodes = []
        
        # Encontrar nodos añadidos
        for node_id, node in new_nodes.items():
            if node_id not in old_nodes:
                added_nodes.append(node)
        
        # Encontrar nodos eliminados
        for node_id, node in old_nodes.items():
            if node_id not in new_nodes:
                removed_nodes.append(node)
        
        # Encontrar nodos modificados
        for node_id in set(old_nodes) & set(new_nodes):
            old_node = old_nodes[node_id]
            new_node = new_nodes[node_id]
            
            if not self._ast_nodes_equal(old_node, new_node):
                modified_nodes.append((old_node, new_node))
        
        # Detectar cambios estructurales
        structural_changes = (
            len(added_nodes) > 5 or
            len(removed_nodes) > 5 or
            any(isinstance(n, (ast.ClassDef, ast.FunctionDef))
                for n in added_nodes + removed_nodes)
        )
        
        return ASTDelta(
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            modified_nodes=modified_nodes,
            structural_changes=structural_changes
        )
    
    async def _compute_generic_delta(
        self,
        old_data: Any,
        new_data: Any
    ) -> ASTDelta:
        """Computar delta genérico para estructuras no-AST."""
        # Convertir a JSON para comparación
        old_json = json.dumps(old_data, sort_keys=True, default=str)
        new_json = json.dumps(new_data, sort_keys=True, default=str)
        
        # Usar difflib para encontrar diferencias
        differ = difflib.unified_diff(
            old_json.splitlines(),
            new_json.splitlines(),
            lineterm=''
        )
        
        # Simplificado: contar líneas cambiadas
        changes = list(differ)
        
        return ASTDelta(
            added_nodes=[],
            removed_nodes=[],
            modified_nodes=[],
            structural_changes=len(changes) > 10
        )
    
    def _generate_delta_cache_key(self, old_ast: Any, new_ast: Any) -> str:
        """Generar clave de cache para un delta."""
        old_hash = hashlib.md5(str(old_ast).encode()).hexdigest()
        new_hash = hashlib.md5(str(new_ast).encode()).hexdigest()
        return f"delta_{old_hash}_{new_hash}"
    
    def _serialize_nodes(self, nodes: List[ast.AST]) -> List[Dict[str, Any]]:
        """Serializar lista de nodos AST."""
        return [self._serialize_node(node) for node in nodes]
    
    def _serialize_node(self, node: ast.AST) -> Dict[str, Any]:
        """Serializar un nodo AST individual."""
        if not isinstance(node, ast.AST):
            return {'type': 'unknown', 'value': str(node)}
        
        serialized = {
            'type': type(node).__name__,
            'lineno': getattr(node, 'lineno', None),
            'col_offset': getattr(node, 'col_offset', None)
        }
        
        # Añadir atributos específicos según tipo
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            serialized['name'] = node.name
        elif isinstance(node, ast.Name):
            serialized['id'] = node.id
        elif isinstance(node, ast.Constant):
            serialized['value'] = node.value
        
        return serialized
    
    def _deep_copy(self, obj: Any) -> Any:
        """Copia profunda de un objeto."""
        import copy
        return copy.deepcopy(obj)
    
    def _detect_analysis_type(self, analysis: Any) -> str:
        """Detectar tipo de análisis desde su estructura."""
        if isinstance(analysis, dict):
            # Buscar indicadores en las claves
            if 'complexity' in analysis or 'cyclomatic_complexity' in analysis:
                return 'complexity'
            elif 'vulnerabilities' in analysis or 'security_issues' in analysis:
                return 'security'
            elif 'code_smells' in analysis or 'quality_score' in analysis:
                return 'quality'
            elif 'dependencies' in analysis or 'imports' in analysis:
                return 'dependencies'
            elif 'loc' in analysis or 'metrics' in analysis:
                return 'metrics'
        
        return 'generic'
    
    async def _apply_delta_operation(
        self,
        analysis: Any,
        operation: DeltaOperation,
        analysis_type: str
    ) -> Any:
        """Aplicar una operación de delta."""
        if operation.operation_type == 'add':
            return self._apply_add_operation(analysis, operation.path, operation.value)
        elif operation.operation_type == 'remove':
            return self._apply_remove_operation(analysis, operation.path)
        elif operation.operation_type == 'modify':
            return self._apply_modify_operation(
                analysis, operation.path, operation.value, operation.old_value
            )
        elif operation.operation_type == 'replace':
            return self._apply_replace_operation(analysis, operation.path, operation.value)
        else:
            return analysis
    
    def _apply_add_operation(
        self,
        obj: Any,
        path: List[str],
        value: Any
    ) -> Any:
        """Aplicar operación de adición."""
        if not path:
            return value
        
        current = obj
        for i, key in enumerate(path[:-1]):
            if isinstance(current, dict):
                if key not in current:
                    current[key] = {}
                current = current[key]
            elif isinstance(current, list):
                idx = int(key)
                if idx >= len(current):
                    current.extend([None] * (idx - len(current) + 1))
                current = current[idx]
        
        # Aplicar valor final
        final_key = path[-1]
        if isinstance(current, dict):
            current[final_key] = value
        elif isinstance(current, list):
            idx = int(final_key)
            if idx >= len(current):
                current.extend([None] * (idx - len(current) + 1))
            current[idx] = value
        
        return obj
    
    def _apply_remove_operation(
        self,
        obj: Any,
        path: List[str]
    ) -> Any:
        """Aplicar operación de eliminación."""
        if not path:
            return None
        
        current = obj
        for i, key in enumerate(path[:-1]):
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list):
                idx = int(key)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return obj
            else:
                return obj
        
        # Eliminar elemento final
        final_key = path[-1]
        if isinstance(current, dict) and final_key in current:
            del current[final_key]
        elif isinstance(current, list):
            idx = int(final_key)
            if 0 <= idx < len(current):
                current.pop(idx)
        
        return obj
    
    def _apply_modify_operation(
        self,
        obj: Any,
        path: List[str],
        new_value: Any,
        old_value: Any
    ) -> Any:
        """Aplicar operación de modificación."""
        if not path:
            return new_value
        
        current = obj
        for key in path[:-1]:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list):
                idx = int(key)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return obj
            else:
                return obj
        
        # Modificar valor final
        final_key = path[-1]
        if isinstance(current, dict) and final_key in current:
            # Verificar que el valor actual coincide con old_value
            if current[final_key] == old_value:
                current[final_key] = new_value
        elif isinstance(current, list):
            idx = int(final_key)
            if 0 <= idx < len(current) and current[idx] == old_value:
                current[idx] = new_value
        
        return obj
    
    def _apply_replace_operation(
        self,
        obj: Any,
        path: List[str],
        value: Any
    ) -> Any:
        """Aplicar operación de reemplazo."""
        # Similar a modify pero sin verificar valor anterior
        return self._apply_add_operation(obj, path, value)
    
    async def _apply_ast_delta(
        self,
        analysis: Dict[str, Any],
        delta: Dict[str, Any],
        analysis_type: str
    ) -> Dict[str, Any]:
        """Aplicar delta de AST a un análisis."""
        # Obtener estrategia de merge específica
        merge_strategy = self._merge_strategies.get(
            analysis_type,
            self._merge_generic_analysis
        )
        
        # Aplicar cambios
        updated = await merge_strategy(analysis, delta)
        
        return updated
    
    async def _recalculate_aggregates(
        self,
        analysis: Dict[str, Any],
        analysis_type: str
    ) -> Dict[str, Any]:
        """Recalcular métricas agregadas después de aplicar delta."""
        if analysis_type == 'complexity':
            # Recalcular complejidad total
            if 'functions' in analysis:
                total_complexity = sum(
                    f.get('complexity', 0)
                    for f in analysis['functions'].values()
                )
                analysis['total_complexity'] = total_complexity
                
        elif analysis_type == 'quality':
            # Recalcular score de calidad
            if 'issues' in analysis:
                total_issues = len(analysis['issues'])
                analysis['quality_score'] = max(0, 100 - (total_issues * 5))
                
        elif analysis_type == 'metrics':
            # Recalcular métricas totales
            if 'files' in analysis:
                total_loc = sum(
                    f.get('loc', 0)
                    for f in analysis['files'].values()
                )
                analysis['total_loc'] = total_loc
        
        return analysis
    
    async def _optimize_operations(
        self,
        operations: List[DeltaOperation]
    ) -> List[DeltaOperation]:
        """Optimizar lista de operaciones eliminando redundancias."""
        if not operations:
            return []
        
        # Agrupar operaciones por path
        ops_by_path = {}
        for op in operations:
            path_key = tuple(op.path)
            if path_key not in ops_by_path:
                ops_by_path[path_key] = []
            ops_by_path[path_key].append(op)
        
        # Optimizar cada grupo
        optimized = []
        for path_key, ops in ops_by_path.items():
            if len(ops) == 1:
                optimized.append(ops[0])
            else:
                # Si hay múltiples operaciones en el mismo path,
                # mantener solo la última relevante
                last_op = ops[-1]
                
                # Si la última es 'remove', solo esa importa
                if last_op.operation_type == 'remove':
                    optimized.append(last_op)
                # Si hay add seguido de modify, convertir a add con valor final
                elif ops[0].operation_type == 'add' and last_op.operation_type == 'modify':
                    optimized.append(DeltaOperation(
                        operation_type='add',
                        path=list(path_key),
                        value=last_op.value
                    ))
                else:
                    optimized.append(last_op)
        
        return optimized
    
    def _compare_analysis_results(self, result1: Any, result2: Any) -> bool:
        """Comparar dos resultados de análisis."""
        # Comparación simplificada
        try:
            # Convertir a JSON para comparación normalizada
            json1 = json.dumps(result1, sort_keys=True, default=str)
            json2 = json.dumps(result2, sort_keys=True, default=str)
            return json1 == json2
        except Exception:
            # Fallback a comparación directa
            return result1 == result2
    
    def _extract_ast_nodes(self, tree: ast.AST) -> Dict[str, ast.AST]:
        """Extraer todos los nodos del AST con IDs únicos."""
        nodes = {}
        
        class NodeVisitor(ast.NodeVisitor):
            def visit(self, node):
                # Generar ID único para el nodo
                node_id = self._generate_node_id(node)
                nodes[node_id] = node
                self.generic_visit(node)
            
            def _generate_node_id(self, node):
                parts = [type(node).__name__]
                
                if hasattr(node, 'name'):
                    parts.append(f"name={node.name}")
                if hasattr(node, 'lineno'):
                    parts.append(f"line={node.lineno}")
                if hasattr(node, 'col_offset'):
                    parts.append(f"col={node.col_offset}")
                
                return ":".join(parts)
        
        visitor = NodeVisitor()
        visitor.visit(tree)
        return nodes
    
    def _ast_nodes_equal(self, node1: ast.AST, node2: ast.AST) -> bool:
        """Comparar si dos nodos AST son iguales."""
        # Comparar tipo
        if type(node1) != type(node2):
            return False
        
        # Comparar atributos relevantes
        if isinstance(node1, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node1.name != node2.name:
                return False
        
        # Comparar estructura completa
        return ast.dump(node1) == ast.dump(node2)
    
    # Estrategias de merge específicas por tipo
    
    async def _merge_complexity_analysis(
        self,
        analysis: Dict[str, Any],
        delta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge específico para análisis de complejidad."""
        # Aplicar cambios a funciones
        if 'functions' not in analysis:
            analysis['functions'] = {}
        
        # Procesar nodos añadidos
        for node_info in delta.get('added', []):
            if node_info['type'] in ['FunctionDef', 'AsyncFunctionDef']:
                # Analizar nueva función
                func_analysis = {
                    'name': node_info.get('name', 'unknown'),
                    'complexity': 1,  # Base
                    'line': node_info.get('lineno', 0)
                }
                analysis['functions'][func_analysis['name']] = func_analysis
        
        # Procesar nodos eliminados
        for node_info in delta.get('removed', []):
            if node_info['type'] in ['FunctionDef', 'AsyncFunctionDef']:
                func_name = node_info.get('name', 'unknown')
                if func_name in analysis['functions']:
                    del analysis['functions'][func_name]
        
        # Procesar nodos modificados
        for mod in delta.get('modified', []):
            old_node = mod['old']
            new_node = mod['new']
            
            if old_node['type'] in ['FunctionDef', 'AsyncFunctionDef']:
                func_name = old_node.get('name', 'unknown')
                if func_name in analysis['functions']:
                    # Re-analizar función modificada
                    analysis['functions'][func_name]['complexity'] += 1
        
        return analysis
    
    async def _merge_security_analysis(
        self,
        analysis: Dict[str, Any],
        delta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge específico para análisis de seguridad."""
        if 'vulnerabilities' not in analysis:
            analysis['vulnerabilities'] = []
        
        # Los cambios en el código pueden introducir o eliminar vulnerabilidades
        # Simplificado: marcar para re-análisis
        if delta.get('structural_changes', False):
            analysis['needs_full_rescan'] = True
        
        return analysis
    
    async def _merge_quality_analysis(
        self,
        analysis: Dict[str, Any],
        delta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge específico para análisis de calidad."""
        if 'code_smells' not in analysis:
            analysis['code_smells'] = []
        
        # Actualizar basado en cambios
        modified_lines = set()
        
        for node_info in delta.get('modified', []):
            if 'lineno' in node_info:
                modified_lines.add(node_info['lineno'])
        
        # Filtrar code smells en líneas modificadas
        analysis['code_smells'] = [
            smell for smell in analysis['code_smells']
            if smell.get('line', 0) not in modified_lines
        ]
        
        return analysis
    
    async def _merge_dependency_analysis(
        self,
        analysis: Dict[str, Any],
        delta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge específico para análisis de dependencias."""
        if 'imports' not in analysis:
            analysis['imports'] = []
        
        # Buscar cambios en imports
        for node_info in delta.get('added', []):
            if node_info['type'] in ['Import', 'ImportFrom']:
                analysis['imports'].append({
                    'type': node_info['type'],
                    'line': node_info.get('lineno', 0)
                })
        
        for node_info in delta.get('removed', []):
            if node_info['type'] in ['Import', 'ImportFrom']:
                # Eliminar import
                analysis['imports'] = [
                    imp for imp in analysis['imports']
                    if imp.get('line') != node_info.get('lineno', 0)
                ]
        
        return analysis
    
    async def _merge_metrics_analysis(
        self,
        analysis: Dict[str, Any],
        delta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge específico para análisis de métricas."""
        # Actualizar líneas de código basado en cambios
        added_lines = len(delta.get('added', []))
        removed_lines = len(delta.get('removed', []))
        
        if 'loc' in analysis:
            analysis['loc'] = max(0, analysis['loc'] + added_lines - removed_lines)
        
        # Actualizar contador de funciones
        added_functions = sum(
            1 for n in delta.get('added', [])
            if n['type'] in ['FunctionDef', 'AsyncFunctionDef']
        )
        removed_functions = sum(
            1 for n in delta.get('removed', [])
            if n['type'] in ['FunctionDef', 'AsyncFunctionDef']
        )
        
        if 'function_count' in analysis:
            analysis['function_count'] = max(
                0,
                analysis['function_count'] + added_functions - removed_functions
            )
        
        return analysis
    
    async def _merge_generic_analysis(
        self,
        analysis: Dict[str, Any],
        delta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge genérico para análisis no específicos."""
        # Simplemente marcar que hubo cambios
        analysis['last_delta_applied'] = datetime.now().isoformat()
        analysis['delta_changes'] = {
            'added': len(delta.get('added', [])),
            'removed': len(delta.get('removed', [])),
            'modified': len(delta.get('modified', []))
        }
        
        return analysis

