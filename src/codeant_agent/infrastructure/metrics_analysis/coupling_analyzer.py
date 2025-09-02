"""
Implementación del analizador de acoplamiento.

Este módulo implementa el análisis de acoplamiento entre clases usando
métricas CBO, RFC, DIT, NOC y análisis de dependencias.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from pathlib import Path
from collections import defaultdict

from ...domain.entities.code_metrics import CouplingMetrics
from ...domain.entities.parse_result import ParseResult
from ...domain.entities.ast_normalization import NormalizedNode as UnifiedNode, NodeType as UnifiedNodeType
from ...domain.value_objects.programming_language import ProgrammingLanguage
from .complexity_analyzer import ClassNode, MethodNode, AttributeNode

logger = logging.getLogger(__name__)


@dataclass
class ClassDependency:
    """Dependencia entre clases."""
    source_class: str
    target_class: str
    dependency_type: str  # "inheritance", "composition", "usage", "aggregation"
    strength: float = 1.0
    location: Optional[str] = None


@dataclass
class MethodCall:
    """Llamada a método."""
    caller_method: str
    called_method: str
    caller_class: str
    called_class: str
    call_count: int = 1


@dataclass
class InheritanceHierarchy:
    """Jerarquía de herencia."""
    class_name: str
    parent_classes: List[str]
    child_classes: List[str]
    depth: int = 0
    siblings: List[str] = field(default_factory=list)


@dataclass
class CouplingAnalysisResult:
    """Resultado del análisis de acoplamiento."""
    coupling_metrics: CouplingMetrics
    class_dependencies: List[ClassDependency]
    inheritance_hierarchies: List[InheritanceHierarchy]
    method_calls: List[MethodCall]
    circular_dependencies: List[List[str]]
    analysis_time_ms: int


class CBOCalculator:
    """Calculadora de CBO (Coupling Between Objects)."""
    
    def calculate_cbo(self, class_node: ClassNode, all_classes: List[ClassNode]) -> int:
        """
        Calcula CBO - número de clases a las que esta clase está acoplada.
        
        Args:
            class_node: Clase a analizar
            all_classes: Todas las clases del proyecto
            
        Returns:
            Valor CBO (número de clases acopladas)
        """
        coupled_classes = set()
        all_class_names = {cls.name for cls in all_classes}
        
        # Analizar dependencias por herencia
        coupled_classes.update(class_node.parent_classes)
        
        # Analizar dependencias en métodos
        for method in class_node.methods:
            method_dependencies = self._find_method_dependencies(method, all_class_names)
            coupled_classes.update(method_dependencies)
        
        # Analizar dependencias en atributos
        for attribute in class_node.attributes:
            attr_dependencies = self._find_attribute_dependencies(attribute, all_class_names)
            coupled_classes.update(attr_dependencies)
        
        # Excluir la clase misma
        coupled_classes.discard(class_node.name)
        
        return len(coupled_classes)
    
    def _find_method_dependencies(self, method: MethodNode, all_class_names: Set[str]) -> Set[str]:
        """Encuentra dependencias de clases en un método."""
        dependencies = set()
        
        # Buscar referencias a otras clases en el cuerpo del método
        self._find_class_references_recursive(method.body, all_class_names, dependencies)
        
        return dependencies
    
    def _find_attribute_dependencies(self, attribute: AttributeNode, all_class_names: Set[str]) -> Set[str]:
        """Encuentra dependencias de clases en un atributo."""
        dependencies = set()
        
        # Analizar tipo del atributo
        if attribute.attribute_type:
            for class_name in all_class_names:
                if class_name in attribute.attribute_type:
                    dependencies.add(class_name)
        
        return dependencies
    
    def _find_class_references_recursive(self, node: UnifiedNode, class_names: Set[str], dependencies: Set[str]) -> None:
        """Busca referencias a clases recursivamente."""
        # Buscar en identificadores
        if node.node_type == UnifiedNodeType.IDENTIFIER and hasattr(node, 'value'):
            if node.value in class_names:
                dependencies.add(node.value)
        
        # Buscar en llamadas (constructores)
        elif node.node_type == UnifiedNodeType.CALL_EXPRESSION:
            class_ref = self._extract_class_from_call(node, class_names)
            if class_ref:
                dependencies.add(class_ref)
        
        # Procesar hijos
        for child in node.children:
            self._find_class_references_recursive(child, class_names, dependencies)
    
    def _extract_class_from_call(self, node: UnifiedNode, class_names: Set[str]) -> Optional[str]:
        """Extrae referencia a clase de una llamada."""
        if hasattr(node, 'value') and node.value:
            for class_name in class_names:
                if class_name in node.value:
                    return class_name
        return None


class RFCCalculator:
    """Calculadora de RFC (Response For Class)."""
    
    def calculate_rfc(self, class_node: ClassNode, all_classes: List[ClassNode]) -> int:
        """
        Calcula RFC - número de métodos que pueden ejecutarse en respuesta a un mensaje.
        
        Incluye:
        - Todos los métodos de la clase
        - Todos los métodos llamados por la clase
        
        Returns:
            Valor RFC
        """
        response_set = set()
        
        # Añadir todos los métodos de la clase
        for method in class_node.methods:
            response_set.add(f"{class_node.name}.{method.name}")
        
        # Añadir métodos llamados por la clase
        all_method_names = self._build_all_methods_map(all_classes)
        
        for method in class_node.methods:
            called_methods = self._find_called_methods(method, all_method_names)
            response_set.update(called_methods)
        
        return len(response_set)
    
    def _build_all_methods_map(self, all_classes: List[ClassNode]) -> Dict[str, str]:
        """Construye mapa de todos los métodos por nombre."""
        method_map = {}
        
        for class_node in all_classes:
            for method in class_node.methods:
                # Mapear nombre del método a su clase
                method_map[method.name] = class_node.name
        
        return method_map
    
    def _find_called_methods(self, method: MethodNode, all_methods: Dict[str, str]) -> Set[str]:
        """Encuentra métodos llamados por un método."""
        called_methods = set()
        self._find_method_calls_recursive(method.body, all_methods, called_methods)
        return called_methods
    
    def _find_method_calls_recursive(self, node: UnifiedNode, all_methods: Dict[str, str], called: Set[str]) -> None:
        """Busca llamadas a métodos recursivamente."""
        if node.node_type == UnifiedNodeType.CALL_EXPRESSION:
            method_name = self._extract_method_name_from_call(node)
            if method_name and method_name in all_methods:
                class_name = all_methods[method_name]
                called.add(f"{class_name}.{method_name}")
        
        for child in node.children:
            self._find_method_calls_recursive(child, all_methods, called)
    
    def _extract_method_name_from_call(self, node: UnifiedNode) -> Optional[str]:
        """Extrae nombre de método de una llamada."""
        if hasattr(node, 'value') and node.value:
            import re
            # Buscar patrón de llamada a método
            call_match = re.search(r'(\w+)\s*\(', node.value)
            if call_match:
                return call_match.group(1)
        
        # Buscar en hijos
        for child in node.children:
            if child.node_type == UnifiedNodeType.IDENTIFIER and hasattr(child, 'value'):
                return child.value
        
        return None


class DITCalculator:
    """Calculadora de DIT (Depth of Inheritance Tree)."""
    
    def calculate_dit(self, class_node: ClassNode, all_classes: List[ClassNode]) -> int:
        """
        Calcula DIT - profundidad máxima en árbol de herencia.
        
        Args:
            class_node: Clase a analizar
            all_classes: Todas las clases para construcción de jerarquía
            
        Returns:
            Profundidad de herencia
        """
        # Construir mapa de jerarquía
        inheritance_map = self._build_inheritance_map(all_classes)
        
        # Calcular profundidad recursivamente
        return self._calculate_depth_recursive(class_node.name, inheritance_map, set())
    
    def _build_inheritance_map(self, all_classes: List[ClassNode]) -> Dict[str, List[str]]:
        """Construye mapa de herencia clase -> padres."""
        inheritance_map = {}
        
        for class_node in all_classes:
            inheritance_map[class_node.name] = class_node.parent_classes[:]
        
        return inheritance_map
    
    def _calculate_depth_recursive(self, class_name: str, inheritance_map: Dict[str, List[str]], 
                                 visited: Set[str]) -> int:
        """Calcula profundidad recursivamente evitando ciclos."""
        if class_name in visited:
            return 0  # Evitar ciclos infinitos
        
        visited.add(class_name)
        
        parents = inheritance_map.get(class_name, [])
        if not parents:
            return 0  # Clase base
        
        # Calcular profundidad máxima desde padres
        max_depth = 0
        for parent in parents:
            parent_depth = self._calculate_depth_recursive(parent, inheritance_map, visited.copy())
            max_depth = max(max_depth, parent_depth + 1)
        
        return max_depth


class NOCCalculator:
    """Calculadora de NOC (Number of Children)."""
    
    def calculate_noc(self, class_node: ClassNode, all_classes: List[ClassNode]) -> int:
        """
        Calcula NOC - número de subclases inmediatas.
        
        Returns:
            Número de hijos directos
        """
        children_count = 0
        
        for other_class in all_classes:
            if class_node.name in other_class.parent_classes:
                children_count += 1
        
        return children_count


class CircularDependencyDetector:
    """Detector de dependencias circulares."""
    
    def detect_circular_dependencies(self, class_dependencies: List[ClassDependency]) -> List[List[str]]:
        """
        Detecta dependencias circulares entre clases.
        
        Returns:
            Lista de ciclos encontrados (cada ciclo es lista de nombres de clase)
        """
        # Construir grafo dirigido
        graph = defaultdict(set)
        for dep in class_dependencies:
            graph[dep.source_class].add(dep.target_class)
        
        # Buscar ciclos usando DFS
        cycles = []
        visited = set()
        rec_stack = set()
        
        for node in graph:
            if node not in visited:
                cycle = self._dfs_find_cycle(node, graph, visited, rec_stack, [])
                if cycle:
                    cycles.append(cycle)
        
        return cycles
    
    def _dfs_find_cycle(self, node: str, graph: Dict[str, Set[str]], visited: Set[str], 
                       rec_stack: Set[str], path: List[str]) -> Optional[List[str]]:
        """DFS para encontrar ciclos."""
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                cycle = self._dfs_find_cycle(neighbor, graph, visited, rec_stack, path[:])
                if cycle:
                    return cycle
            elif neighbor in rec_stack:
                # Encontrado ciclo
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]
        
        rec_stack.remove(node)
        return None


class CouplingAnalyzer:
    """Analizador principal de acoplamiento."""
    
    def __init__(self):
        self.cbo_calculator = CBOCalculator()
        self.rfc_calculator = RFCCalculator()
        self.dit_calculator = DITCalculator()
        self.noc_calculator = NOCCalculator()
        self.circular_detector = CircularDependencyDetector()
    
    async def calculate_coupling_metrics(
        self, 
        parse_result: ParseResult,
        all_parse_results: Optional[List[ParseResult]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> CouplingAnalysisResult:
        """
        Calcula métricas de acoplamiento completas.
        
        Args:
            parse_result: Resultado del parsing del archivo actual
            all_parse_results: Todos los archivos del proyecto (para análisis completo)
            config: Configuración opcional
            
        Returns:
            CouplingAnalysisResult completo
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Calculando métricas de acoplamiento para {parse_result.file_path}")
            
            # Convertir a unified node y extraer clases
            unified_root = self._convert_to_unified_node(parse_result.tree.root_node)
            classes = self._extract_classes(unified_root, parse_result.file_path)
            
            if not classes:
                logger.debug("No se encontraron clases para análisis de acoplamiento")
                return CouplingAnalysisResult(
                    coupling_metrics=CouplingMetrics(),
                    class_dependencies=[],
                    inheritance_hierarchies=[],
                    method_calls=[],
                    circular_dependencies=[],
                    analysis_time_ms=int((time.time() - start_time) * 1000)
                )
            
            # Si tenemos información del proyecto completo, usarla
            all_classes = classes[:]
            if all_parse_results:
                all_classes = await self._extract_all_project_classes(all_parse_results)
            
            # Calcular métricas por clase
            cbo_values = []
            rfc_values = []
            dit_values = []
            noc_values = []
            class_dependencies = []
            inheritance_hierarchies = []
            method_calls = []
            
            for class_node in classes:
                # CBO - Coupling Between Objects
                cbo = self.cbo_calculator.calculate_cbo(class_node, all_classes)
                cbo_values.append(cbo)
                
                # RFC - Response For Class
                rfc = self.rfc_calculator.calculate_rfc(class_node, all_classes)
                rfc_values.append(rfc)
                
                # DIT - Depth of Inheritance Tree
                dit = self.dit_calculator.calculate_dit(class_node, all_classes)
                dit_values.append(dit)
                
                # NOC - Number of Children
                noc = self.noc_calculator.calculate_noc(class_node, all_classes)
                noc_values.append(noc)
                
                # Recopilar dependencias de esta clase
                class_deps = self._analyze_class_dependencies(class_node, all_classes)
                class_dependencies.extend(class_deps)
                
                # Construir jerarquía de herencia
                hierarchy = self._build_inheritance_hierarchy(class_node, all_classes)
                inheritance_hierarchies.append(hierarchy)
                
                # Recopilar llamadas a métodos
                class_method_calls = self._analyze_method_calls(class_node, all_classes)
                method_calls.extend(class_method_calls)
            
            # Detectar dependencias circulares
            circular_deps = self.circular_detector.detect_circular_dependencies(class_dependencies)
            
            # Calcular métricas promedio
            coupling_metrics = CouplingMetrics(
                average_cbo=sum(cbo_values) / len(cbo_values) if cbo_values else 0.0,
                average_rfc=sum(rfc_values) / len(rfc_values) if rfc_values else 0.0,
                average_dit=sum(dit_values) / len(dit_values) if dit_values else 0.0,
                average_noc=sum(noc_values) / len(noc_values) if noc_values else 0.0,
                total_dependencies=len(class_dependencies),
                circular_dependencies=len(circular_deps),
                max_inheritance_depth=max(dit_values) if dit_values else 0
            )
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Análisis de acoplamiento completado para {parse_result.file_path}: "
                f"{len(classes)} clases, CBO avg={coupling_metrics.average_cbo:.1f}, "
                f"RFC avg={coupling_metrics.average_rfc:.1f} en {total_time}ms"
            )
            
            return CouplingAnalysisResult(
                coupling_metrics=coupling_metrics,
                class_dependencies=class_dependencies,
                inheritance_hierarchies=inheritance_hierarchies,
                method_calls=method_calls,
                circular_dependencies=circular_deps,
                analysis_time_ms=total_time
            )
            
        except Exception as e:
            logger.error(f"Error calculando métricas de acoplamiento: {e}")
            raise
    
    async def _extract_all_project_classes(self, all_parse_results: List[ParseResult]) -> List[ClassNode]:
        """Extrae todas las clases del proyecto."""
        all_classes = []
        
        for parse_result in all_parse_results:
            try:
                unified_root = self._convert_to_unified_node(parse_result.tree.root_node)
                file_classes = self._extract_classes(unified_root, parse_result.file_path)
                all_classes.extend(file_classes)
            except Exception as e:
                logger.warning(f"Error extrayendo clases de {parse_result.file_path}: {e}")
        
        return all_classes
    
    def _convert_to_unified_node(self, tree_sitter_node) -> UnifiedNode:
        """Convierte tree-sitter node a UnifiedNode."""
        from ...domain.entities.ast_normalization import SourcePosition
        
        try:
            start_point = getattr(tree_sitter_node, 'start_point', (0, 0))
            end_point = getattr(tree_sitter_node, 'end_point', (0, 0))
            start_byte = getattr(tree_sitter_node, 'start_byte', 0)
            end_byte = getattr(tree_sitter_node, 'end_byte', 0)
            
            if isinstance(start_point, tuple) and len(start_point) >= 2:
                start_line, start_col = int(start_point[0]), int(start_point[1])
            else:
                start_line, start_col = 0, 0
            
            if isinstance(end_point, tuple) and len(end_point) >= 2:
                end_line, end_col = int(end_point[0]), int(end_point[1])
            else:
                end_line, end_col = 0, 0
            
            position = SourcePosition(
                start_line=start_line,
                start_column=start_col,
                end_line=end_line,
                end_column=end_col,
                start_byte=int(start_byte) if isinstance(start_byte, (int, str)) else 0,
                end_byte=int(end_byte) if isinstance(end_byte, (int, str)) else 0
            )
        except Exception:
            position = SourcePosition(start_line=0, start_column=0, end_line=0, end_column=0, start_byte=0, end_byte=0)
        
        unified_node = UnifiedNode(
            node_type=UnifiedNodeType.LANGUAGE_SPECIFIC,
            position=position,
            children=[],
            value=tree_sitter_node.text.decode('utf-8') if hasattr(tree_sitter_node, 'text') and tree_sitter_node.text else ""
        )
        
        # Mapear tipo de nodo
        if hasattr(tree_sitter_node, 'type'):
            type_mapping = {
                'class_definition': UnifiedNodeType.CLASS_DECLARATION,
                'class_def': UnifiedNodeType.CLASS_DECLARATION,
                'function_definition': UnifiedNodeType.FUNCTION_DECLARATION,
                'method_definition': UnifiedNodeType.FUNCTION_DECLARATION,
            }
            unified_node.node_type = type_mapping.get(tree_sitter_node.type, UnifiedNodeType.LANGUAGE_SPECIFIC)
        
        # Convertir hijos
        if hasattr(tree_sitter_node, 'children'):
            for child in tree_sitter_node.children:
                try:
                    unified_child = self._convert_to_unified_node(child)
                    unified_node.children.append(unified_child)
                except Exception:
                    continue
        
        return unified_node
    
    def _extract_classes(self, node: UnifiedNode, file_path: Path) -> List[ClassNode]:
        """Extrae clases del AST."""
        classes = []
        self._extract_classes_recursive(node, classes, file_path)
        return classes
    
    def _extract_classes_recursive(self, node: UnifiedNode, classes: List[ClassNode], file_path: Path) -> None:
        """Extrae clases recursivamente."""
        if node.node_type == UnifiedNodeType.CLASS_DECLARATION:
            class_name = self._extract_class_name(node)
            methods = self._extract_class_methods(node, file_path)
            attributes = self._extract_class_attributes(node, file_path)
            parent_classes = self._extract_parent_classes(node)
            
            from ...domain.entities.dead_code_analysis import SourceRange, SourcePosition
            location = SourceRange(
                start=SourcePosition(line=node.position.start_line + 1, column=node.position.start_column),
                end=SourcePosition(line=node.position.end_line + 1, column=node.position.end_column)
            )
            
            class_node = ClassNode(
                name=class_name,
                location=location,
                methods=methods,
                attributes=attributes,
                parent_classes=parent_classes
            )
            
            classes.append(class_node)
        
        for child in node.children:
            self._extract_classes_recursive(child, classes, file_path)
    
    def _analyze_class_dependencies(self, class_node: ClassNode, all_classes: List[ClassNode]) -> List[ClassDependency]:
        """Analiza dependencias de una clase."""
        dependencies = []
        all_class_names = {cls.name for cls in all_classes}
        
        # Dependencias de herencia
        for parent in class_node.parent_classes:
            if parent in all_class_names:
                dependencies.append(ClassDependency(
                    source_class=class_node.name,
                    target_class=parent,
                    dependency_type="inheritance",
                    strength=1.0
                ))
        
        # Dependencias de uso en métodos
        for method in class_node.methods:
            method_deps = self._find_method_class_dependencies(method, all_class_names)
            for target_class, strength in method_deps.items():
                dependencies.append(ClassDependency(
                    source_class=class_node.name,
                    target_class=target_class,
                    dependency_type="usage",
                    strength=strength,
                    location=f"{class_node.name}.{method.name}"
                ))
        
        return dependencies
    
    def _find_method_class_dependencies(self, method: MethodNode, all_class_names: Set[str]) -> Dict[str, float]:
        """Encuentra dependencias de clases en un método."""
        dependencies = defaultdict(float)
        
        # Buscar referencias a clases
        for class_name in all_class_names:
            reference_count = self._count_class_references(method.body, class_name)
            if reference_count > 0:
                dependencies[class_name] = reference_count / 10.0  # Normalizar
        
        return dependencies
    
    def _count_class_references(self, node: UnifiedNode, class_name: str) -> int:
        """Cuenta referencias a una clase específica."""
        count = 0
        
        if node.node_type == UnifiedNodeType.IDENTIFIER and hasattr(node, 'value'):
            if node.value == class_name:
                count += 1
        
        elif hasattr(node, 'value') and node.value and class_name in node.value:
            count += 1
        
        for child in node.children:
            count += self._count_class_references(child, class_name)
        
        return count
    
    def _build_inheritance_hierarchy(self, class_node: ClassNode, all_classes: List[ClassNode]) -> InheritanceHierarchy:
        """Construye jerarquía de herencia para una clase."""
        # Encontrar hijos
        children = []
        for other_class in all_classes:
            if class_node.name in other_class.parent_classes:
                children.append(other_class.name)
        
        # Calcular profundidad
        depth = self.dit_calculator.calculate_dit(class_node, all_classes)
        
        # Encontrar hermanos (clases con mismos padres)
        siblings = []
        for other_class in all_classes:
            if (other_class.name != class_node.name and 
                set(other_class.parent_classes).intersection(set(class_node.parent_classes))):
                siblings.append(other_class.name)
        
        return InheritanceHierarchy(
            class_name=class_node.name,
            parent_classes=class_node.parent_classes[:],
            child_classes=children,
            depth=depth,
            siblings=siblings
        )
    
    def _analyze_method_calls(self, class_node: ClassNode, all_classes: List[ClassNode]) -> List[MethodCall]:
        """Analiza llamadas a métodos."""
        method_calls = []
        all_methods = self._build_all_methods_map(all_classes)
        
        for method in class_node.methods:
            calls = self._find_method_calls_in_method(method, all_methods)
            for called_method, called_class in calls:
                method_call = MethodCall(
                    caller_method=method.name,
                    called_method=called_method,
                    caller_class=class_node.name,
                    called_class=called_class
                )
                method_calls.append(method_call)
        
        return method_calls
    
    def _build_all_methods_map(self, all_classes: List[ClassNode]) -> Dict[str, str]:
        """Construye mapa de métodos a clases."""
        method_to_class = {}
        
        for class_node in all_classes:
            for method in class_node.methods:
                method_to_class[method.name] = class_node.name
        
        return method_to_class
    
    def _find_method_calls_in_method(self, method: MethodNode, all_methods: Dict[str, str]) -> List[Tuple[str, str]]:
        """Encuentra llamadas a métodos en un método."""
        calls = []
        self._find_calls_recursive(method.body, all_methods, calls)
        return calls
    
    def _find_calls_recursive(self, node: UnifiedNode, all_methods: Dict[str, str], calls: List[Tuple[str, str]]) -> None:
        """Busca llamadas recursivamente."""
        if node.node_type == UnifiedNodeType.CALL_EXPRESSION:
            method_name = self._extract_called_method_name(node)
            if method_name and method_name in all_methods:
                calls.append((method_name, all_methods[method_name]))
        
        for child in node.children:
            self._find_calls_recursive(child, all_methods, calls)
    
    def _extract_called_method_name(self, node: UnifiedNode) -> Optional[str]:
        """Extrae nombre de método llamado."""
        if hasattr(node, 'value') and node.value:
            import re
            call_match = re.search(r'(\w+)\s*\(', node.value)
            if call_match:
                return call_match.group(1)
        
        # Buscar en primer hijo identificador
        for child in node.children:
            if child.node_type == UnifiedNodeType.IDENTIFIER and hasattr(child, 'value'):
                return child.value
        
        return None
    
    # Métodos auxiliares compartidos
    
    def _extract_class_name(self, node: UnifiedNode) -> str:
        """Extrae nombre de clase."""
        for child in node.children:
            if child.node_type == UnifiedNodeType.IDENTIFIER and hasattr(child, 'value') and child.value:
                return child.value
        
        if hasattr(node, 'value') and node.value:
            import re
            class_match = re.search(r'class\s+(\w+)', node.value)
            if class_match:
                return class_match.group(1)
        
        return "AnonymousClass"
    
    def _extract_class_methods(self, class_node: UnifiedNode, file_path: Path) -> List[MethodNode]:
        """Extrae métodos de clase."""
        methods = []
        
        for child in class_node.children:
            if child.node_type == UnifiedNodeType.FUNCTION_DECLARATION:
                method_name = self._extract_function_name(child)
                parameters = self._extract_function_parameters(child)
                
                from ...domain.entities.dead_code_analysis import SourceRange, SourcePosition
                location = SourceRange(
                    start=SourcePosition(line=child.position.start_line + 1, column=child.position.start_column),
                    end=SourcePosition(line=child.position.end_line + 1, column=child.position.end_column)
                )
                
                method = MethodNode(
                    name=method_name,
                    location=location,
                    body=child,
                    parameters=parameters
                )
                
                methods.append(method)
        
        return methods
    
    def _extract_class_attributes(self, class_node: UnifiedNode, file_path: Path) -> List[AttributeNode]:
        """Extrae atributos de clase."""
        attributes = []
        
        for child in class_node.children:
            if child.node_type == UnifiedNodeType.VARIABLE_DECLARATION:
                attr_name = self._extract_variable_name(child)
                
                from ...domain.entities.dead_code_analysis import SourceRange, SourcePosition
                location = SourceRange(
                    start=SourcePosition(line=child.position.start_line + 1, column=child.position.start_column),
                    end=SourcePosition(line=child.position.end_line + 1, column=child.position.end_column)
                )
                
                attribute = AttributeNode(
                    name=attr_name,
                    location=location
                )
                
                attributes.append(attribute)
        
        return attributes
    
    def _extract_parent_classes(self, node: UnifiedNode) -> List[str]:
        """Extrae clases padre de una clase."""
        parent_classes = []
        
        # Buscar en el valor del nodo patrones de herencia
        if hasattr(node, 'value') and node.value:
            import re
            
            # Python: class Child(Parent1, Parent2)
            python_match = re.search(r'class\s+\w+\s*\(\s*([^)]+)\s*\)', node.value)
            if python_match:
                parents_str = python_match.group(1)
                parent_classes = [p.strip() for p in parents_str.split(',') if p.strip()]
            
            # JavaScript/TypeScript: class Child extends Parent
            js_match = re.search(r'class\s+\w+\s+extends\s+(\w+)', node.value)
            if js_match:
                parent_classes = [js_match.group(1)]
            
            # Rust: impl Trait for Struct
            rust_match = re.search(r'impl\s+(\w+)\s+for\s+\w+', node.value)
            if rust_match:
                parent_classes = [rust_match.group(1)]
        
        return parent_classes
    
    def _extract_function_name(self, node: UnifiedNode) -> str:
        """Extrae nombre de función."""
        for child in node.children:
            if child.node_type == UnifiedNodeType.IDENTIFIER and hasattr(child, 'value') and child.value:
                return child.value
        
        if hasattr(node, 'value') and node.value:
            import re
            func_match = re.search(r'def\s+(\w+)|function\s+(\w+)|fn\s+(\w+)', node.value)
            if func_match:
                return next(group for group in func_match.groups() if group)
        
        return "anonymous_method"
    
    def _extract_function_parameters(self, node: UnifiedNode) -> List[str]:
        """Extrae parámetros de función."""
        parameters = []
        
        for child in node.children:
            if str(child.node_type).lower() in ['parameter_list', 'parameters']:
                for param_child in child.children:
                    if param_child.node_type == UnifiedNodeType.IDENTIFIER and hasattr(param_child, 'value'):
                        parameters.append(param_child.value)
        
        return parameters
    
    def _extract_variable_name(self, node: UnifiedNode) -> str:
        """Extrae nombre de variable."""
        for child in node.children:
            if child.node_type == UnifiedNodeType.IDENTIFIER and hasattr(child, 'value') and child.value:
                return child.value
        
        return "unknown_attribute"
