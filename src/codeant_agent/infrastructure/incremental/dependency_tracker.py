"""
Tracker de dependencias para el Sistema de Análisis Incremental.

Este módulo implementa el seguimiento y análisis de dependencias entre archivos
para determinar el impacto de los cambios y optimizar el análisis incremental.
"""

import asyncio
import ast
import re
from typing import Set, Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
import networkx as nx

from ...domain.entities.incremental import (
    DependencyImpact, FileChange, GranularChange,
    ChangeType, ChangeLocation
)
from ...domain.services.incremental_service import DependencyAnalysisService
from ...domain.repositories.incremental_repository import DependencyGraphRepository
from ...application.ports.incremental_ports import (
    DependencyGraphOutputPort, MetricsCollectorOutputPort
)
from .incremental_config import IncrementalConfig


class DependencyNode:
    """Nodo en el grafo de dependencias."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.imports: Set[str] = set()
        self.exports: Set[str] = set()
        self.dependencies: Set[Path] = set()
        self.dependents: Set[Path] = set()
        self.last_analyzed: Optional[datetime] = None
        self.content_hash: Optional[str] = None
        self.symbols: Dict[str, Any] = {}  # symbol_name -> metadata


class DependencyTracker(
    DependencyAnalysisService,
    DependencyGraphRepository,
    DependencyGraphOutputPort
):
    """Implementación del tracker de dependencias."""
    
    def __init__(
        self,
        config: IncrementalConfig,
        metrics_collector: MetricsCollectorOutputPort
    ):
        self.config = config
        self.metrics_collector = metrics_collector
        
        # Grafo de dependencias usando NetworkX
        self.dependency_graph = nx.DiGraph()
        
        # Cache de análisis
        self._analysis_cache: Dict[Path, DependencyNode] = {}
        
        # Índice de símbolos
        self._symbol_index: Dict[str, Set[Path]] = defaultdict(set)
        
        # Estadísticas
        self._stats = {
            'total_files': 0,
            'total_dependencies': 0,
            'circular_dependencies': 0,
            'max_depth': 0
        }
    
    # Implementación de DependencyAnalysisService
    
    async def analyze_dependencies(self, file_path: Path) -> Set[Path]:
        """
        Analizar dependencias directas de un archivo.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Conjunto de archivos de los que depende
        """
        # Verificar cache
        if file_path in self._analysis_cache:
            node = self._analysis_cache[file_path]
            if self._is_cache_valid(node):
                return node.dependencies
        
        # Analizar archivo
        dependencies = await self._analyze_file_dependencies(file_path)
        
        # Actualizar grafo
        await self._update_dependency_graph(file_path, dependencies)
        
        # Registrar métricas
        await self.metrics_collector.record_analysis_time(
            operation="dependency_analysis",
            duration_ms=0,  # TODO: medir tiempo real
            incremental=False
        )
        
        return dependencies
    
    async def analyze_dependency_impact(
        self,
        changes: List[GranularChange]
    ) -> DependencyImpact:
        """
        Analizar impacto de cambios en las dependencias.
        
        Args:
            changes: Lista de cambios granulares
            
        Returns:
            Impacto en las dependencias
        """
        affected_files = set()
        affected_symbols = set()
        total_impact_score = 0.0
        
        # Analizar cada cambio
        for change in changes:
            file_path = Path(change.location.file_path)
            
            # Obtener archivos que dependen de este
            dependents = await self._get_transitive_dependents(
                file_path,
                max_depth=self.config.max_dependency_depth
            )
            affected_files.update(dependents)
            
            # Analizar impacto por símbolo si está disponible
            if change.symbol_name:
                symbol_dependents = await self._get_symbol_dependents(
                    file_path,
                    change.symbol_name
                )
                affected_symbols.update(symbol_dependents)
                
                # Calcular impacto basado en tipo de cambio
                impact = self._calculate_change_impact(change, len(symbol_dependents))
                total_impact_score += impact
        
        # Normalizar score de impacto
        max_possible_impact = len(changes) * len(affected_files)
        normalized_score = min(total_impact_score / max(max_possible_impact, 1), 1.0)
        
        return DependencyImpact(
            affected_files=list(affected_files),
            affected_symbols=list(affected_symbols),
            impact_score=normalized_score,
            dependency_depth=await self._calculate_max_dependency_depth(affected_files)
        )
    
    async def calculate_transitive_impact(
        self,
        file_path: Path,
        max_depth: Optional[int] = None
    ) -> Set[Path]:
        """
        Calcular impacto transitivo de cambios.
        
        Args:
            file_path: Archivo modificado
            max_depth: Profundidad máxima a explorar
            
        Returns:
            Conjunto de archivos afectados transitivamente
        """
        max_depth = max_depth or self.config.max_dependency_depth
        return await self._get_transitive_dependents(file_path, max_depth)
    
    async def find_circular_dependencies(
        self,
        project_path: Path
    ) -> List[List[Path]]:
        """
        Encontrar dependencias circulares en el proyecto.
        
        Args:
            project_path: Ruta del proyecto
            
        Returns:
            Lista de ciclos encontrados
        """
        # Construir grafo completo si no existe
        if not self.dependency_graph.nodes():
            await self._build_full_dependency_graph(project_path)
        
        # Buscar ciclos
        cycles = []
        try:
            # NetworkX encuentra todos los ciclos simples
            for cycle in nx.simple_cycles(self.dependency_graph):
                cycles.append([Path(node) for node in cycle])
                
                # Limitar número de ciclos reportados
                if len(cycles) >= self.config.max_circular_dependencies_to_report:
                    break
        except nx.NetworkXError:
            # No hay ciclos
            pass
        
        # Actualizar estadísticas
        self._stats['circular_dependencies'] = len(cycles)
        
        return cycles
    
    async def build_dependency_graph(
        self,
        project_path: Path
    ) -> Dict[Path, Set[Path]]:
        """
        Construir grafo de dependencias del proyecto.
        
        Args:
            project_path: Ruta del proyecto
            
        Returns:
            Diccionario de dependencias
        """
        await self._build_full_dependency_graph(project_path)
        
        # Convertir grafo a diccionario
        graph_dict = {}
        for node in self.dependency_graph.nodes():
            node_path = Path(node)
            dependencies = set()
            
            for successor in self.dependency_graph.successors(node):
                dependencies.add(Path(successor))
            
            graph_dict[node_path] = dependencies
        
        return graph_dict
    
    # Implementación de DependencyGraphRepository
    
    async def save_dependency_graph(
        self,
        file_path: Path,
        dependencies: Set[Path]
    ) -> None:
        """Guardar dependencias de un archivo."""
        await self._update_dependency_graph(file_path, dependencies)
    
    async def get_dependency_graph(
        self,
        file_path: Path
    ) -> Optional[Set[Path]]:
        """Obtener dependencias de un archivo."""
        if file_path in self._analysis_cache:
            return self._analysis_cache[file_path].dependencies
        
        # Analizar si no está en cache
        return await self.analyze_dependencies(file_path)
    
    async def update_dependency_graph(
        self,
        updates: Dict[Path, Set[Path]]
    ) -> None:
        """Actualizar múltiples nodos del grafo."""
        for file_path, dependencies in updates.items():
            await self._update_dependency_graph(file_path, dependencies)
    
    async def get_reverse_dependencies(
        self,
        file_path: Path
    ) -> Set[Path]:
        """Obtener archivos que dependen de uno dado."""
        if file_path in self._analysis_cache:
            return self._analysis_cache[file_path].dependents
        
        # Buscar en el grafo
        dependents = set()
        node_str = str(file_path)
        
        if node_str in self.dependency_graph:
            for predecessor in self.dependency_graph.predecessors(node_str):
                dependents.add(Path(predecessor))
        
        return dependents
    
    # Implementación de DependencyGraphOutputPort
    
    async def save_dependency_graph(
        self,
        graph: Dict[Path, Set[Path]]
    ) -> bool:
        """Guardar grafo de dependencias completo."""
        try:
            # Limpiar grafo existente
            self.dependency_graph.clear()
            self._analysis_cache.clear()
            
            # Reconstruir grafo
            for file_path, dependencies in graph.items():
                await self._update_dependency_graph(file_path, dependencies)
            
            return True
        except Exception:
            return False
    
    async def load_dependency_graph(self) -> Dict[Path, Set[Path]]:
        """Cargar grafo de dependencias."""
        return await self.build_dependency_graph(Path.cwd())
    
    async def query_dependencies(
        self,
        file_path: Path,
        direction: str = "both"
    ) -> Dict[str, Set[Path]]:
        """
        Consultar dependencias en una dirección.
        
        Args:
            file_path: Archivo a consultar
            direction: "dependencies", "dependents", o "both"
            
        Returns:
            Diccionario con las dependencias solicitadas
        """
        result = {}
        
        if direction in ["dependencies", "both"]:
            result["dependencies"] = await self.analyze_dependencies(file_path)
        
        if direction in ["dependents", "both"]:
            result["dependents"] = await self.get_reverse_dependencies(file_path)
        
        return result
    
    # Métodos auxiliares privados
    
    async def _analyze_file_dependencies(self, file_path: Path) -> Set[Path]:
        """Analizar dependencias de un archivo específico."""
        dependencies = set()
        
        if not file_path.exists():
            return dependencies
        
        try:
            content = file_path.read_text()
            
            # Detectar tipo de archivo y analizar
            if file_path.suffix == '.py':
                dependencies = await self._analyze_python_dependencies(
                    file_path, content
                )
            elif file_path.suffix in ['.js', '.jsx', '.ts', '.tsx']:
                dependencies = await self._analyze_javascript_dependencies(
                    file_path, content
                )
            # Agregar más lenguajes según necesidad
            
        except Exception as e:
            # Log error but continue
            pass
        
        return dependencies
    
    async def _analyze_python_dependencies(
        self,
        file_path: Path,
        content: str
    ) -> Set[Path]:
        """Analizar dependencias de un archivo Python."""
        dependencies = set()
        
        try:
            tree = ast.parse(content)
            
            # Visitor para extraer imports
            class ImportVisitor(ast.NodeVisitor):
                def __init__(self, base_path: Path):
                    self.base_path = base_path
                    self.imports = set()
                
                def visit_Import(self, node):
                    for alias in node.names:
                        self.imports.add(alias.name)
                    self.generic_visit(node)
                
                def visit_ImportFrom(self, node):
                    if node.module:
                        if node.level == 0:  # Absolute import
                            self.imports.add(node.module)
                        else:  # Relative import
                            # Resolver import relativo
                            parts = self.base_path.parts[:-1]  # Directorio padre
                            for _ in range(node.level - 1):
                                parts = parts[:-1]
                            
                            if node.module:
                                module_path = Path(*parts) / node.module.replace('.', '/')
                                self.imports.add(str(module_path))
                    
                    self.generic_visit(node)
            
            visitor = ImportVisitor(file_path)
            visitor.visit(tree)
            
            # Resolver imports a rutas de archivo
            for import_name in visitor.imports:
                resolved_paths = await self._resolve_import_to_file(
                    import_name, file_path
                )
                dependencies.update(resolved_paths)
            
        except Exception:
            pass
        
        return dependencies
    
    async def _analyze_javascript_dependencies(
        self,
        file_path: Path,
        content: str
    ) -> Set[Path]:
        """Analizar dependencias de un archivo JavaScript/TypeScript."""
        dependencies = set()
        
        # Patrones de import/require
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"](.+?)[\'"]',
            r'import\s*\([\'"](.+?)[\'"]\)',
            r'require\s*\([\'"](.+?)[\'"]\)',
            r'export\s+.*?\s+from\s+[\'"](.+?)[\'"]'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                resolved_paths = await self._resolve_import_to_file(
                    match, file_path
                )
                dependencies.update(resolved_paths)
        
        return dependencies
    
    async def _resolve_import_to_file(
        self,
        import_path: str,
        from_file: Path
    ) -> Set[Path]:
        """Resolver un import a rutas de archivo reales."""
        resolved = set()
        base_dir = from_file.parent
        
        # Casos especiales
        if import_path.startswith('.'):
            # Import relativo
            try:
                relative_path = base_dir / import_path
                
                # Buscar archivo con diferentes extensiones
                for ext in ['', '.py', '.js', '.jsx', '.ts', '.tsx']:
                    file_path = Path(str(relative_path) + ext)
                    if file_path.exists() and file_path.is_file():
                        resolved.add(file_path)
                        break
                
                # Buscar __init__.py en directorio
                init_path = relative_path / '__init__.py'
                if init_path.exists():
                    resolved.add(init_path)
                    
            except Exception:
                pass
        else:
            # Import absoluto - buscar en el proyecto
            # Simplificado: buscar archivo con ese nombre
            project_root = await self._find_project_root(from_file)
            if project_root:
                module_parts = import_path.split('.')
                possible_path = project_root
                
                for part in module_parts:
                    possible_path = possible_path / part
                
                # Buscar archivo
                for ext in ['', '.py', '.js', '.jsx', '.ts', '.tsx']:
                    file_path = Path(str(possible_path) + ext)
                    if file_path.exists() and file_path.is_file():
                        resolved.add(file_path)
                        break
        
        return resolved
    
    async def _find_project_root(self, file_path: Path) -> Optional[Path]:
        """Encontrar la raíz del proyecto."""
        current = file_path.parent
        
        # Buscar indicadores de raíz del proyecto
        root_indicators = [
            '.git', 'setup.py', 'pyproject.toml',
            'package.json', 'tsconfig.json'
        ]
        
        while current != current.parent:
            for indicator in root_indicators:
                if (current / indicator).exists():
                    return current
            current = current.parent
        
        return None
    
    async def _update_dependency_graph(
        self,
        file_path: Path,
        dependencies: Set[Path]
    ) -> None:
        """Actualizar el grafo de dependencias."""
        node_str = str(file_path)
        
        # Crear nodo si no existe
        if file_path not in self._analysis_cache:
            self._analysis_cache[file_path] = DependencyNode(file_path)
        
        node = self._analysis_cache[file_path]
        node.dependencies = dependencies
        node.last_analyzed = datetime.now()
        
        # Actualizar grafo NetworkX
        if node_str not in self.dependency_graph:
            self.dependency_graph.add_node(node_str)
        
        # Eliminar edges antiguos
        old_edges = list(self.dependency_graph.out_edges(node_str))
        self.dependency_graph.remove_edges_from(old_edges)
        
        # Añadir nuevos edges
        for dep in dependencies:
            dep_str = str(dep)
            if dep_str not in self.dependency_graph:
                self.dependency_graph.add_node(dep_str)
            self.dependency_graph.add_edge(node_str, dep_str)
            
            # Actualizar dependents
            if dep in self._analysis_cache:
                self._analysis_cache[dep].dependents.add(file_path)
            else:
                self._analysis_cache[dep] = DependencyNode(dep)
                self._analysis_cache[dep].dependents.add(file_path)
        
        # Actualizar estadísticas
        self._stats['total_files'] = len(self.dependency_graph.nodes())
        self._stats['total_dependencies'] = len(self.dependency_graph.edges())
    
    def _is_cache_valid(self, node: DependencyNode) -> bool:
        """Verificar si el cache de un nodo es válido."""
        if not node.last_analyzed:
            return False
        
        # Cache válido por tiempo configurado
        age = (datetime.now() - node.last_analyzed).total_seconds()
        return age < self.config.dependency_cache_ttl
    
    async def _get_transitive_dependents(
        self,
        file_path: Path,
        max_depth: int
    ) -> Set[Path]:
        """Obtener dependientes transitivos hasta cierta profundidad."""
        visited = set()
        to_visit = deque([(file_path, 0)])
        dependents = set()
        
        while to_visit:
            current_file, depth = to_visit.popleft()
            
            if current_file in visited or depth > max_depth:
                continue
            
            visited.add(current_file)
            
            # Obtener dependientes directos
            direct_dependents = await self.get_reverse_dependencies(current_file)
            dependents.update(direct_dependents)
            
            # Añadir a la cola para exploración
            for dep in direct_dependents:
                if dep not in visited:
                    to_visit.append((dep, depth + 1))
        
        return dependents
    
    async def _get_symbol_dependents(
        self,
        file_path: Path,
        symbol_name: str
    ) -> Set[Tuple[Path, str]]:
        """Obtener archivos y ubicaciones que usan un símbolo específico."""
        dependents = set()
        
        # Buscar en el índice de símbolos
        symbol_key = f"{file_path}:{symbol_name}"
        if symbol_key in self._symbol_index:
            for dependent_file in self._symbol_index[symbol_key]:
                # Aquí podríamos incluir la ubicación exacta del uso
                dependents.add((dependent_file, symbol_name))
        
        return dependents
    
    def _calculate_change_impact(
        self,
        change: GranularChange,
        num_dependents: int
    ) -> float:
        """Calcular impacto de un cambio específico."""
        # Peso base por tipo de cambio
        base_weights = {
            ChangeType.FUNCTION_REMOVED: 1.0,
            ChangeType.FUNCTION_MODIFIED: 0.7,
            ChangeType.FUNCTION_ADDED: 0.3,
            ChangeType.CLASS_REMOVED: 1.0,
            ChangeType.CLASS_MODIFIED: 0.8,
            ChangeType.CLASS_ADDED: 0.4,
        }
        
        base_weight = base_weights.get(change.change_type, 0.5)
        
        # Factor por número de dependientes
        dependent_factor = min(num_dependents / 10, 2.0)
        
        # Factor por alcance
        scope_factor = 1.0
        if hasattr(change, 'scope'):
            if change.scope == 'public':
                scope_factor = 1.5
            elif change.scope == 'private':
                scope_factor = 0.5
        
        return base_weight * dependent_factor * scope_factor
    
    async def _calculate_max_dependency_depth(
        self,
        affected_files: Set[Path]
    ) -> int:
        """Calcular la profundidad máxima de dependencias."""
        max_depth = 0
        
        for file_path in affected_files:
            # Usar BFS para encontrar profundidad
            depth = await self._find_dependency_depth(file_path)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    async def _find_dependency_depth(self, file_path: Path) -> int:
        """Encontrar profundidad de dependencias de un archivo."""
        visited = set()
        queue = deque([(file_path, 0)])
        max_depth = 0
        
        while queue:
            current_file, depth = queue.popleft()
            
            if current_file in visited:
                continue
            
            visited.add(current_file)
            max_depth = max(max_depth, depth)
            
            # Obtener dependencias
            dependencies = await self.analyze_dependencies(current_file)
            for dep in dependencies:
                if dep not in visited:
                    queue.append((dep, depth + 1))
        
        return max_depth
    
    async def _build_full_dependency_graph(self, project_path: Path) -> None:
        """Construir grafo completo de dependencias del proyecto."""
        # Encontrar todos los archivos del proyecto
        file_paths = []
        for pattern in self.config.file_patterns_to_analyze:
            file_paths.extend(project_path.rglob(pattern))
        
        # Analizar cada archivo
        tasks = []
        for file_path in file_paths:
            if self._should_analyze_file(file_path):
                task = self.analyze_dependencies(file_path)
                tasks.append(task)
        
        # Ejecutar análisis en paralelo
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Actualizar estadísticas
        self._stats['max_depth'] = await self._calculate_graph_max_depth()
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determinar si un archivo debe ser analizado."""
        # Excluir directorios
        exclude_dirs = {
            '__pycache__', '.git', 'node_modules',
            '.venv', 'venv', 'env', 'build', 'dist'
        }
        
        for part in file_path.parts:
            if part in exclude_dirs:
                return False
        
        # Incluir solo archivos de código
        return file_path.suffix in {
            '.py', '.js', '.jsx', '.ts', '.tsx',
            '.java', '.cpp', '.c', '.h', '.hpp'
        }
    
    async def _calculate_graph_max_depth(self) -> int:
        """Calcular profundidad máxima del grafo."""
        if not self.dependency_graph.nodes():
            return 0
        
        # Encontrar nodos sin dependencias (hojas)
        leaf_nodes = [
            node for node in self.dependency_graph.nodes()
            if self.dependency_graph.out_degree(node) == 0
        ]
        
        max_depth = 0
        for leaf in leaf_nodes:
            # BFS desde cada hoja
            try:
                paths = nx.single_source_shortest_path_length(
                    self.dependency_graph.reverse(),
                    leaf
                )
                node_max_depth = max(paths.values())
                max_depth = max(max_depth, node_max_depth)
            except Exception:
                continue
        
        return max_depth

