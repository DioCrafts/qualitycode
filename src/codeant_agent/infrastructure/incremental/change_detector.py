"""
Detector de cambios granular para el Sistema de Análisis Incremental.

Este módulo implementa la detección de cambios a nivel granular,
incluyendo cambios en archivos, funciones, clases y expresiones.
"""

import asyncio
import subprocess
from typing import List, Optional, Dict, Any, Set, Tuple
from pathlib import Path
from datetime import datetime
import difflib
import ast
import hashlib
import json

from ...domain.entities.incremental import (
    ChangeSet, FileChange, GranularChange, ChangeType,
    ChangeLocation, GranularityLevel, DependencyImpact
)
from ...domain.services.incremental_service import ChangeDetectionService
from ...application.ports.incremental_ports import (
    ChangeDetectionOutputPort, MetricsCollectorOutputPort
)
from .incremental_config import IncrementalConfig


class GranularChangeDetector(ChangeDetectionService, ChangeDetectionOutputPort):
    """Implementación del detector de cambios granular."""
    
    def __init__(
        self,
        config: IncrementalConfig,
        metrics_collector: MetricsCollectorOutputPort
    ):
        self.config = config
        self.metrics_collector = metrics_collector
        
        # Cache para ASTs parseados
        self._ast_cache: Dict[str, ast.AST] = {}
        # Cache para hashes de contenido
        self._content_hash_cache: Dict[str, str] = {}
    
    async def detect_changes(
        self,
        repository_path: Path,
        from_commit: Optional[str] = None,
        to_commit: Optional[str] = None
    ) -> ChangeSet:
        """
        Detectar cambios entre commits o estado actual.
        
        Args:
            repository_path: Ruta del repositorio
            from_commit: Commit inicial (None = HEAD~1)
            to_commit: Commit final (None = HEAD)
            
        Returns:
            Conjunto de cambios detectados
        """
        start_time = datetime.now()
        
        try:
            # Obtener cambios de Git
            git_changes = await self.get_git_changes(
                repository_path, from_commit, to_commit
            )
            
            # Procesar cambios en paralelo
            tasks = []
            for git_change in git_changes:
                task = self._process_file_change(
                    repository_path, git_change
                )
                tasks.append(task)
            
            file_changes = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrar errores
            valid_changes = []
            for change in file_changes:
                if isinstance(change, FileChange):
                    valid_changes.append(change)
                else:
                    # Log error but continue
                    pass
            
            # Crear ChangeSet
            change_set = ChangeSet(
                id=self._generate_change_set_id(from_commit, to_commit),
                from_commit=from_commit or "HEAD~1",
                to_commit=to_commit or "HEAD",
                timestamp=datetime.now(),
                file_changes=valid_changes,
                total_files=len(valid_changes),
                total_additions=sum(fc.additions for fc in valid_changes),
                total_deletions=sum(fc.deletions for fc in valid_changes),
                total_modifications=sum(fc.modifications for fc in valid_changes)
            )
            
            # Calcular impacto total
            if self.config.calculate_impact_score:
                change_set.total_impact = await self._calculate_total_impact(
                    change_set
                )
            
            # Registrar métricas
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            await self.metrics_collector.record_analysis_time(
                operation="change_detection",
                duration_ms=duration_ms,
                incremental=False
            )
            
            return change_set
            
        except Exception as e:
            raise Exception(f"Error detecting changes: {str(e)}")
    
    async def detect_file_changes(
        self,
        file_path: Path,
        old_content: Optional[str] = None,
        new_content: Optional[str] = None
    ) -> List[FileChange]:
        """
        Detectar cambios en un archivo específico.
        
        Args:
            file_path: Ruta del archivo
            old_content: Contenido anterior
            new_content: Contenido nuevo
            
        Returns:
            Lista de cambios en el archivo
        """
        if old_content is None and new_content is None:
            return []
        
        # Determinar tipo de cambio
        if old_content is None:
            change_type = ChangeType.FILE_ADDED
        elif new_content is None:
            change_type = ChangeType.FILE_REMOVED
        else:
            change_type = ChangeType.FILE_MODIFIED
        
        # Crear FileChange
        file_change = FileChange(
            file_path=str(file_path),
            change_type=change_type,
            additions=len(new_content.splitlines()) if new_content else 0,
            deletions=len(old_content.splitlines()) if old_content else 0,
            modifications=0,
            granular_changes=[]
        )
        
        # Detectar cambios granulares si el archivo fue modificado
        if change_type == ChangeType.FILE_MODIFIED and self.config.enable_granular_detection:
            file_change.granular_changes = await self.detect_granular_changes(
                file_change,
                GranularityLevel.FUNCTION
            )
        
        return [file_change]
    
    async def detect_granular_changes(
        self,
        file_change: FileChange,
        granularity: GranularityLevel
    ) -> List[GranularChange]:
        """
        Detectar cambios granulares dentro de un archivo.
        
        Args:
            file_change: Cambio de archivo
            granularity: Nivel de granularidad
            
        Returns:
            Lista de cambios granulares
        """
        granular_changes = []
        
        try:
            # Obtener contenido antiguo y nuevo
            old_content = await self._get_file_content_at_commit(
                Path(file_change.file_path),
                "old"
            )
            new_content = await self._get_file_content_at_commit(
                Path(file_change.file_path),
                "new"
            )
            
            if not old_content or not new_content:
                return []
            
            # Parsear ASTs
            old_ast = await self.parse_file_ast(
                Path(file_change.file_path), old_content
            )
            new_ast = await self.parse_file_ast(
                Path(file_change.file_path), new_content
            )
            
            if not old_ast or not new_ast:
                return []
            
            # Detectar cambios según granularidad
            if granularity == GranularityLevel.FILE:
                # Ya manejado en FileChange
                pass
            elif granularity == GranularityLevel.FUNCTION:
                granular_changes.extend(
                    await self._detect_function_level_changes(
                        old_ast, new_ast, file_change.file_path
                    )
                )
            elif granularity == GranularityLevel.STATEMENT:
                granular_changes.extend(
                    await self._detect_statement_level_changes(
                        old_ast, new_ast, file_change.file_path
                    )
                )
            elif granularity == GranularityLevel.EXPRESSION:
                granular_changes.extend(
                    await self._detect_expression_level_changes(
                        old_ast, new_ast, file_change.file_path
                    )
                )
            elif granularity == GranularityLevel.TOKEN:
                granular_changes.extend(
                    await self._detect_token_level_changes(
                        old_content, new_content, file_change.file_path
                    )
                )
            
            # Detectar cambios semánticos si está habilitado
            if self.config.detect_semantic_changes:
                semantic_changes = await self._detect_semantic_changes(
                    old_ast, new_ast, file_change.file_path
                )
                granular_changes.extend(semantic_changes)
            
        except Exception as e:
            # Log error but continue
            pass
        
        return granular_changes
    
    async def calculate_change_impact(
        self,
        changes: List[GranularChange]
    ) -> float:
        """
        Calcular el impacto de un conjunto de cambios.
        
        Args:
            changes: Lista de cambios granulares
            
        Returns:
            Score de impacto (0.0 - 1.0)
        """
        if not changes:
            return 0.0
        
        total_impact = 0.0
        
        for change in changes:
            # Peso base según tipo de cambio
            base_weight = self._get_change_type_weight(change.change_type)
            
            # Ajustar por complejidad
            complexity_factor = 1.0
            if hasattr(change, 'complexity_delta'):
                complexity_factor = 1.0 + (change.complexity_delta / 10)
            
            # Ajustar por alcance
            scope_factor = 1.0
            if change.scope == "public":
                scope_factor = 2.0
            elif change.scope == "protected":
                scope_factor = 1.5
            
            # Calcular impacto del cambio
            change_impact = base_weight * complexity_factor * scope_factor
            total_impact += change_impact
        
        # Normalizar a rango 0.0 - 1.0
        max_possible_impact = len(changes) * 2.0 * 2.0  # max weights
        normalized_impact = min(total_impact / max_possible_impact, 1.0)
        
        return normalized_impact
    
    async def aggregate_changes(
        self,
        changes: List[GranularChange],
        window_ms: int
    ) -> List[GranularChange]:
        """
        Agregar cambios dentro de una ventana temporal.
        
        Args:
            changes: Lista de cambios
            window_ms: Ventana temporal en milisegundos
            
        Returns:
            Lista de cambios agregados
        """
        if not changes:
            return []
        
        # Ordenar por timestamp
        sorted_changes = sorted(changes, key=lambda c: c.timestamp)
        
        aggregated = []
        current_group = [sorted_changes[0]]
        
        for change in sorted_changes[1:]:
            # Verificar si está dentro de la ventana
            time_diff = (change.timestamp - current_group[-1].timestamp).total_seconds() * 1000
            
            if time_diff <= window_ms and self._can_aggregate(current_group[-1], change):
                current_group.append(change)
            else:
                # Agregar grupo actual
                aggregated.append(self._aggregate_group(current_group))
                current_group = [change]
        
        # Agregar último grupo
        if current_group:
            aggregated.append(self._aggregate_group(current_group))
        
        return aggregated
    
    # Implementación de ChangeDetectionOutputPort
    
    async def get_git_changes(
        self,
        repository_path: Path,
        from_commit: Optional[str],
        to_commit: Optional[str]
    ) -> List[FileChange]:
        """Obtener cambios desde Git."""
        try:
            # Construir comando git diff
            cmd = ["git", "-C", str(repository_path), "diff", "--name-status"]
            
            if from_commit:
                cmd.append(from_commit)
            else:
                cmd.append("HEAD~1")
            
            if to_commit:
                cmd.append(to_commit)
            else:
                cmd.append("HEAD")
            
            # Ejecutar comando
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parsear salida
            file_changes = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                status = parts[0]
                file_path = parts[1]
                
                # Mapear status a ChangeType
                if status == 'A':
                    change_type = ChangeType.FILE_ADDED
                elif status == 'D':
                    change_type = ChangeType.FILE_REMOVED
                elif status == 'M':
                    change_type = ChangeType.FILE_MODIFIED
                elif status.startswith('R'):
                    change_type = ChangeType.FILE_RENAMED
                else:
                    continue
                
                file_change = FileChange(
                    file_path=file_path,
                    change_type=change_type,
                    additions=0,
                    deletions=0,
                    modifications=0,
                    granular_changes=[]
                )
                
                # Obtener estadísticas detalladas
                if change_type != ChangeType.FILE_REMOVED:
                    stats = await self._get_file_stats(
                        repository_path, file_path, from_commit, to_commit
                    )
                    file_change.additions = stats.get('additions', 0)
                    file_change.deletions = stats.get('deletions', 0)
                    file_change.modifications = stats.get('modifications', 0)
                
                file_changes.append(file_change)
            
            return file_changes
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"Git command failed: {e.stderr}")
    
    async def parse_file_ast(
        self,
        file_path: Path,
        content: str
    ) -> Any:
        """Parsear AST de un archivo."""
        # Check cache
        cache_key = f"{file_path}:{hashlib.md5(content.encode()).hexdigest()}"
        if cache_key in self._ast_cache:
            return self._ast_cache[cache_key]
        
        try:
            # Detectar lenguaje por extensión
            if file_path.suffix == '.py':
                parsed_ast = ast.parse(content)
            else:
                # Para otros lenguajes, retornar None por ahora
                parsed_ast = None
            
            # Cache result
            if parsed_ast:
                self._ast_cache[cache_key] = parsed_ast
            
            return parsed_ast
            
        except Exception as e:
            # Log error but continue
            return None
    
    async def compute_ast_diff(
        self,
        old_ast: Any,
        new_ast: Any
    ) -> List[Dict[str, Any]]:
        """Computar diferencias entre ASTs."""
        diffs = []
        
        if not old_ast or not new_ast:
            return diffs
        
        # Para Python AST
        if isinstance(old_ast, ast.AST) and isinstance(new_ast, ast.AST):
            # Comparar nodos del AST
            old_nodes = self._extract_ast_nodes(old_ast)
            new_nodes = self._extract_ast_nodes(new_ast)
            
            # Encontrar nodos añadidos
            for node_id, node in new_nodes.items():
                if node_id not in old_nodes:
                    diffs.append({
                        'type': 'added',
                        'node_type': type(node).__name__,
                        'node_id': node_id,
                        'node': node
                    })
            
            # Encontrar nodos eliminados
            for node_id, node in old_nodes.items():
                if node_id not in new_nodes:
                    diffs.append({
                        'type': 'removed',
                        'node_type': type(node).__name__,
                        'node_id': node_id,
                        'node': node
                    })
            
            # Encontrar nodos modificados
            for node_id in set(old_nodes) & set(new_nodes):
                if not self._ast_nodes_equal(old_nodes[node_id], new_nodes[node_id]):
                    diffs.append({
                        'type': 'modified',
                        'node_type': type(old_nodes[node_id]).__name__,
                        'node_id': node_id,
                        'old_node': old_nodes[node_id],
                        'new_node': new_nodes[node_id]
                    })
        
        return diffs
    
    # Métodos auxiliares privados
    
    async def _process_file_change(
        self,
        repository_path: Path,
        git_change: FileChange
    ) -> FileChange:
        """Procesar un cambio de archivo individual."""
        # Detectar cambios granulares si está habilitado
        if self.config.enable_granular_detection and git_change.change_type == ChangeType.FILE_MODIFIED:
            git_change.granular_changes = await self.detect_granular_changes(
                git_change,
                GranularityLevel[self.config.default_granularity_level]
            )
        
        return git_change
    
    def _generate_change_set_id(
        self,
        from_commit: Optional[str],
        to_commit: Optional[str]
    ) -> str:
        """Generar ID único para el ChangeSet."""
        timestamp = datetime.now().isoformat()
        from_str = from_commit or "HEAD~1"
        to_str = to_commit or "HEAD"
        return f"cs_{from_str}_{to_str}_{timestamp}"
    
    async def _calculate_total_impact(
        self,
        change_set: ChangeSet
    ) -> float:
        """Calcular impacto total del ChangeSet."""
        all_changes = []
        for file_change in change_set.file_changes:
            all_changes.extend(file_change.granular_changes)
        
        return await self.calculate_change_impact(all_changes)
    
    async def _get_file_content_at_commit(
        self,
        file_path: Path,
        version: str
    ) -> Optional[str]:
        """Obtener contenido de archivo en una versión específica."""
        # Implementación simplificada
        try:
            if version == "new" and file_path.exists():
                return file_path.read_text()
            # Para versiones antiguas, usar git show
            return None
        except Exception:
            return None
    
    async def _get_file_stats(
        self,
        repository_path: Path,
        file_path: str,
        from_commit: Optional[str],
        to_commit: Optional[str]
    ) -> Dict[str, int]:
        """Obtener estadísticas de cambios de un archivo."""
        try:
            cmd = [
                "git", "-C", str(repository_path),
                "diff", "--numstat",
                from_commit or "HEAD~1",
                to_commit or "HEAD",
                "--", file_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                parts = result.stdout.strip().split('\t')
                return {
                    'additions': int(parts[0]) if parts[0] != '-' else 0,
                    'deletions': int(parts[1]) if parts[1] != '-' else 0,
                    'modifications': 0  # Git doesn't provide this directly
                }
            
            return {'additions': 0, 'deletions': 0, 'modifications': 0}
            
        except Exception:
            return {'additions': 0, 'deletions': 0, 'modifications': 0}
    
    async def _detect_function_level_changes(
        self,
        old_ast: ast.AST,
        new_ast: ast.AST,
        file_path: str
    ) -> List[GranularChange]:
        """Detectar cambios a nivel de función."""
        changes = []
        
        # Extraer funciones de ambos ASTs
        old_functions = self._extract_functions(old_ast)
        new_functions = self._extract_functions(new_ast)
        
        # Funciones añadidas
        for func_name, func_node in new_functions.items():
            if func_name not in old_functions:
                changes.append(GranularChange(
                    change_type=ChangeType.FUNCTION_ADDED,
                    location=ChangeLocation(
                        file_path=file_path,
                        start_line=func_node.lineno,
                        end_line=func_node.end_lineno if hasattr(func_node, 'end_lineno') else func_node.lineno,
                        start_column=func_node.col_offset,
                        end_column=func_node.end_col_offset if hasattr(func_node, 'end_col_offset') else 0
                    ),
                    symbol_name=func_name,
                    scope="public" if not func_name.startswith('_') else "private",
                    timestamp=datetime.now()
                ))
        
        # Funciones eliminadas
        for func_name, func_node in old_functions.items():
            if func_name not in new_functions:
                changes.append(GranularChange(
                    change_type=ChangeType.FUNCTION_REMOVED,
                    location=ChangeLocation(
                        file_path=file_path,
                        start_line=func_node.lineno,
                        end_line=func_node.end_lineno if hasattr(func_node, 'end_lineno') else func_node.lineno,
                        start_column=func_node.col_offset,
                        end_column=func_node.end_col_offset if hasattr(func_node, 'end_col_offset') else 0
                    ),
                    symbol_name=func_name,
                    scope="public" if not func_name.startswith('_') else "private",
                    timestamp=datetime.now()
                ))
        
        # Funciones modificadas
        for func_name in set(old_functions) & set(new_functions):
            if not self._functions_equal(old_functions[func_name], new_functions[func_name]):
                changes.append(GranularChange(
                    change_type=ChangeType.FUNCTION_MODIFIED,
                    location=ChangeLocation(
                        file_path=file_path,
                        start_line=new_functions[func_name].lineno,
                        end_line=new_functions[func_name].end_lineno if hasattr(new_functions[func_name], 'end_lineno') else new_functions[func_name].lineno,
                        start_column=new_functions[func_name].col_offset,
                        end_column=new_functions[func_name].end_col_offset if hasattr(new_functions[func_name], 'end_col_offset') else 0
                    ),
                    symbol_name=func_name,
                    scope="public" if not func_name.startswith('_') else "private",
                    timestamp=datetime.now()
                ))
        
        return changes
    
    async def _detect_statement_level_changes(
        self,
        old_ast: ast.AST,
        new_ast: ast.AST,
        file_path: str
    ) -> List[GranularChange]:
        """Detectar cambios a nivel de sentencia."""
        # Implementación simplificada
        return []
    
    async def _detect_expression_level_changes(
        self,
        old_ast: ast.AST,
        new_ast: ast.AST,
        file_path: str
    ) -> List[GranularChange]:
        """Detectar cambios a nivel de expresión."""
        # Implementación simplificada
        return []
    
    async def _detect_token_level_changes(
        self,
        old_content: str,
        new_content: str,
        file_path: str
    ) -> List[GranularChange]:
        """Detectar cambios a nivel de token."""
        # Implementación simplificada usando diff línea por línea
        changes = []
        
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        
        differ = difflib.unified_diff(old_lines, new_lines, lineterm='')
        
        for line in differ:
            if line.startswith('+') and not line.startswith('+++'):
                # Línea añadida
                pass
            elif line.startswith('-') and not line.startswith('---'):
                # Línea eliminada
                pass
        
        return changes
    
    async def _detect_semantic_changes(
        self,
        old_ast: ast.AST,
        new_ast: ast.AST,
        file_path: str
    ) -> List[GranularChange]:
        """Detectar cambios semánticos."""
        changes = []
        
        # Detectar cambios en imports
        old_imports = self._extract_imports(old_ast)
        new_imports = self._extract_imports(new_ast)
        
        if old_imports != new_imports:
            changes.append(GranularChange(
                change_type=ChangeType.IMPORT_CHANGED,
                location=ChangeLocation(
                    file_path=file_path,
                    start_line=1,
                    end_line=1,
                    start_column=0,
                    end_column=0
                ),
                symbol_name="imports",
                scope="module",
                timestamp=datetime.now()
            ))
        
        # Detectar cambios en clases
        old_classes = self._extract_classes(old_ast)
        new_classes = self._extract_classes(new_ast)
        
        for class_name in set(new_classes) - set(old_classes):
            changes.append(GranularChange(
                change_type=ChangeType.CLASS_ADDED,
                location=ChangeLocation(
                    file_path=file_path,
                    start_line=new_classes[class_name].lineno,
                    end_line=new_classes[class_name].end_lineno if hasattr(new_classes[class_name], 'end_lineno') else new_classes[class_name].lineno,
                    start_column=new_classes[class_name].col_offset,
                    end_column=new_classes[class_name].end_col_offset if hasattr(new_classes[class_name], 'end_col_offset') else 0
                ),
                symbol_name=class_name,
                scope="public" if not class_name.startswith('_') else "private",
                timestamp=datetime.now()
            ))
        
        return changes
    
    def _get_change_type_weight(self, change_type: ChangeType) -> float:
        """Obtener peso base para un tipo de cambio."""
        weights = {
            ChangeType.FILE_ADDED: 1.0,
            ChangeType.FILE_REMOVED: 1.0,
            ChangeType.FILE_MODIFIED: 0.5,
            ChangeType.FILE_RENAMED: 0.3,
            ChangeType.FUNCTION_ADDED: 0.7,
            ChangeType.FUNCTION_REMOVED: 0.7,
            ChangeType.FUNCTION_MODIFIED: 0.5,
            ChangeType.CLASS_ADDED: 0.8,
            ChangeType.CLASS_REMOVED: 0.8,
            ChangeType.CLASS_MODIFIED: 0.6,
            ChangeType.IMPORT_CHANGED: 0.4,
            ChangeType.COMMENT_CHANGED: 0.1,
            ChangeType.WHITESPACE_CHANGED: 0.05,
            ChangeType.MAJOR_REFACTORING: 1.0
        }
        return weights.get(change_type, 0.5)
    
    def _can_aggregate(
        self,
        change1: GranularChange,
        change2: GranularChange
    ) -> bool:
        """Verificar si dos cambios se pueden agregar."""
        return (
            change1.location.file_path == change2.location.file_path and
            change1.change_type == change2.change_type and
            abs(change1.location.start_line - change2.location.start_line) < 10
        )
    
    def _aggregate_group(
        self,
        changes: List[GranularChange]
    ) -> GranularChange:
        """Agregar un grupo de cambios en uno solo."""
        if len(changes) == 1:
            return changes[0]
        
        # Tomar el primer cambio como base
        aggregated = changes[0]
        
        # Actualizar ubicación para cubrir todos los cambios
        min_line = min(c.location.start_line for c in changes)
        max_line = max(c.location.end_line for c in changes)
        
        aggregated.location.start_line = min_line
        aggregated.location.end_line = max_line
        
        # Agregar información adicional
        aggregated.aggregated_count = len(changes)
        
        return aggregated
    
    def _extract_ast_nodes(self, tree: ast.AST) -> Dict[str, ast.AST]:
        """Extraer todos los nodos del AST con IDs únicos."""
        nodes = {}
        
        class NodeVisitor(ast.NodeVisitor):
            def visit(self, node):
                # Generar ID único para el nodo
                node_id = f"{type(node).__name__}"
                if hasattr(node, 'name'):
                    node_id += f"_{node.name}"
                elif hasattr(node, 'lineno'):
                    node_id += f"_{node.lineno}"
                
                nodes[node_id] = node
                self.generic_visit(node)
        
        NodeVisitor().visit(tree)
        return nodes
    
    def _ast_nodes_equal(self, node1: ast.AST, node2: ast.AST) -> bool:
        """Comparar si dos nodos AST son iguales."""
        return ast.dump(node1) == ast.dump(node2)
    
    def _extract_functions(self, tree: ast.AST) -> Dict[str, ast.FunctionDef]:
        """Extraer todas las funciones del AST."""
        functions = {}
        
        class FunctionVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                functions[node.name] = node
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node):
                functions[node.name] = node
                self.generic_visit(node)
        
        FunctionVisitor().visit(tree)
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> Dict[str, ast.ClassDef]:
        """Extraer todas las clases del AST."""
        classes = {}
        
        class ClassVisitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                classes[node.name] = node
                self.generic_visit(node)
        
        ClassVisitor().visit(tree)
        return classes
    
    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extraer todos los imports del AST."""
        imports = set()
        
        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.add(alias.name)
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                module = node.module or ''
                for alias in node.names:
                    imports.add(f"{module}.{alias.name}")
                self.generic_visit(node)
        
        ImportVisitor().visit(tree)
        return imports
    
    def _functions_equal(self, func1: ast.FunctionDef, func2: ast.FunctionDef) -> bool:
        """Comparar si dos funciones son iguales."""
        # Comparar firma
        if func1.name != func2.name:
            return False
        
        # Comparar argumentos
        if ast.dump(func1.args) != ast.dump(func2.args):
            return False
        
        # Comparar cuerpo
        if ast.dump(func1.body) != ast.dump(func2.body):
            return False
        
        # Comparar decoradores
        if ast.dump(func1.decorator_list) != ast.dump(func2.decorator_list):
            return False
        
        return True
