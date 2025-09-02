"""
Implementación del analizador de cohesión.

Este módulo implementa el análisis de cohesión de clases usando
métricas LCOM, TCC, LCC y análisis de interacciones método-atributo.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from pathlib import Path
from collections import defaultdict

from ...domain.entities.code_metrics import CohesionMetrics
from ...domain.entities.parse_result import ParseResult
from ...domain.entities.ast_normalization import NormalizedNode as UnifiedNode, NodeType as UnifiedNodeType
from ...domain.value_objects.programming_language import ProgrammingLanguage
from .complexity_analyzer import ClassNode, MethodNode, AttributeNode

logger = logging.getLogger(__name__)


@dataclass
class MethodAttributeInteraction:
    """Interacción entre método y atributo."""
    method_name: str
    attribute_name: str
    interaction_type: str  # "read", "write", "both"
    frequency: int = 1


@dataclass
class MethodInteraction:
    """Interacción entre métodos."""
    method1: str
    method2: str
    interaction_type: str  # "calls", "shared_attributes", "data_flow"
    strength: float = 1.0


@dataclass
class CohesionAnalysisResult:
    """Resultado del análisis de cohesión."""
    cohesion_metrics: CohesionMetrics
    class_cohesion_details: List['ClassCohesionDetail']
    method_interactions: List[MethodInteraction]
    analysis_time_ms: int


@dataclass
class ClassCohesionDetail:
    """Detalle de cohesión de una clase específica."""
    class_name: str
    lcom_score: float
    tcc_score: float
    lcc_score: float
    method_count: int
    attribute_count: int
    cohesion_level: str  # "high", "medium", "low"
    problematic_methods: List[str]
    suggestions: List[str]


class LCOMCalculator:
    """Calculadora de LCOM (Lack of Cohesion of Methods)."""
    
    def calculate_lcom(self, class_node: ClassNode) -> float:
        """
        Calcula LCOM usando el método Henderson-Sellers.
        
        LCOM = (|M| - |A|) / (1 - |A|)
        donde M = número de métodos que no acceden a atributos
        y A = número de atributos
        
        Args:
            class_node: Nodo de clase a analizar
            
        Returns:
            Valor LCOM (0.0 - 1.0, donde 0.0 es más cohesivo)
        """
        methods = class_node.methods
        attributes = class_node.attributes
        
        if not methods or not attributes:
            return 0.0  # Sin métodos o atributos = cohesión perfecta
        
        # Construir matriz de uso método-atributo
        method_attribute_usage = self._build_method_attribute_matrix(methods, attributes)
        
        # Calcular LCOM
        m = len(methods)
        a = len(attributes)
        
        # Contar métodos que no acceden a ningún atributo
        methods_without_attribute_access = 0
        for method_name in method_attribute_usage:
            if not method_attribute_usage[method_name]:
                methods_without_attribute_access += 1
        
        if a == 1:
            return 0.0  # Fórmula no aplicable con 1 atributo
        
        lcom = (methods_without_attribute_access - a) / (1 - a) if a != 1 else 0.0
        return max(0.0, min(1.0, lcom))  # Clamp entre 0 y 1
    
    def _build_method_attribute_matrix(self, methods: List[MethodNode], 
                                     attributes: List[AttributeNode]) -> Dict[str, Set[str]]:
        """Construye matriz de uso método-atributo."""
        matrix = defaultdict(set)
        attribute_names = {attr.name for attr in attributes}
        
        for method in methods:
            # Buscar accesos a atributos en el método
            accessed_attrs = self._find_attribute_accesses(method, attribute_names)
            matrix[method.name] = accessed_attrs
        
        return matrix
    
    def _find_attribute_accesses(self, method: MethodNode, attribute_names: Set[str]) -> Set[str]:
        """Encuentra accesos a atributos en un método."""
        accessed = set()
        
        # Buscar en el cuerpo del método
        self._find_accesses_recursive(method.body, attribute_names, accessed)
        
        return accessed
    
    def _find_accesses_recursive(self, node: UnifiedNode, attribute_names: Set[str], accessed: Set[str]) -> None:
        """Busca accesos recursivamente."""
        # Buscar identificadores que coincidan con nombres de atributos
        if node.node_type == UnifiedNodeType.IDENTIFIER and hasattr(node, 'value'):
            if node.value in attribute_names:
                accessed.add(node.value)
        
        # Buscar accesos con self/this
        elif node.node_type == UnifiedNodeType.MEMBER_EXPRESSION:
            member_name = self._extract_member_name(node)
            if member_name in attribute_names:
                accessed.add(member_name)
        
        # Procesar hijos
        for child in node.children:
            self._find_accesses_recursive(child, attribute_names, accessed)
    
    def _extract_member_name(self, node: UnifiedNode) -> str:
        """Extrae nombre de miembro de expresión member."""
        if hasattr(node, 'value') and node.value:
            # Buscar patrón self.attribute o this.attribute
            import re
            member_match = re.search(r'(?:self|this)\.(\w+)', node.value)
            if member_match:
                return member_match.group(1)
        
        return ""


class TCCCalculator:
    """Calculadora de TCC (Tight Class Cohesion)."""
    
    def calculate_tcc(self, class_node: ClassNode) -> float:
        """
        Calcula TCC - mide conexiones directas entre métodos.
        
        TCC = NDC / NP
        donde NDC = número de conexiones directas
        y NP = número máximo de conexiones posibles
        
        Returns:
            Valor TCC (0.0 - 1.0, donde 1.0 es más cohesivo)
        """
        methods = class_node.methods
        
        if len(methods) < 2:
            return 1.0  # Cohesión perfecta con 0-1 métodos
        
        # Calcular conexiones directas
        direct_connections = 0
        total_possible = len(methods) * (len(methods) - 1) // 2
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                if self._are_directly_connected(methods[i], methods[j], class_node.attributes):
                    direct_connections += 1
        
        return direct_connections / total_possible if total_possible > 0 else 1.0
    
    def _are_directly_connected(self, method1: MethodNode, method2: MethodNode, 
                              attributes: List[AttributeNode]) -> bool:
        """Verifica si dos métodos están directamente conectados."""
        attribute_names = {attr.name for attr in attributes}
        
        # Obtener atributos accedidos por cada método
        attrs1 = self._get_accessed_attributes(method1, attribute_names)
        attrs2 = self._get_accessed_attributes(method2, attribute_names)
        
        # Están conectados si comparten al menos un atributo
        return bool(attrs1.intersection(attrs2))
    
    def _get_accessed_attributes(self, method: MethodNode, attribute_names: Set[str]) -> Set[str]:
        """Obtiene atributos accedidos por un método."""
        accessed = set()
        self._find_attribute_accesses_recursive(method.body, attribute_names, accessed)
        return accessed
    
    def _find_attribute_accesses_recursive(self, node: UnifiedNode, attribute_names: Set[str], accessed: Set[str]) -> None:
        """Busca accesos a atributos recursivamente."""
        if node.node_type == UnifiedNodeType.IDENTIFIER and hasattr(node, 'value'):
            if node.value in attribute_names:
                accessed.add(node.value)
        
        elif node.node_type == UnifiedNodeType.MEMBER_EXPRESSION:
            member_name = self._extract_member_name(node)
            if member_name in attribute_names:
                accessed.add(member_name)
        
        for child in node.children:
            self._find_attribute_accesses_recursive(child, attribute_names, accessed)
    
    def _extract_member_name(self, node: UnifiedNode) -> str:
        """Extrae nombre de miembro."""
        if hasattr(node, 'value') and node.value:
            import re
            member_match = re.search(r'(?:self|this)\.(\w+)', node.value)
            if member_match:
                return member_match.group(1)
        return ""


class LCCCalculator:
    """Calculadora de LCC (Loose Class Cohesion)."""
    
    def calculate_lcc(self, class_node: ClassNode) -> float:
        """
        Calcula LCC - incluye conexiones directas e indirectas.
        
        Returns:
            Valor LCC (0.0 - 1.0, donde 1.0 es más cohesivo)
        """
        methods = class_node.methods
        
        if len(methods) < 2:
            return 1.0
        
        # Construir grafo de conexiones entre métodos
        connection_graph = self._build_method_connection_graph(methods, class_node.attributes)
        
        # Contar pares conectados (directa o indirectamente)
        connected_pairs = self._count_connected_pairs(connection_graph)
        total_possible_pairs = len(methods) * (len(methods) - 1) // 2
        
        return connected_pairs / total_possible_pairs if total_possible_pairs > 0 else 1.0
    
    def _build_method_connection_graph(self, methods: List[MethodNode], 
                                     attributes: List[AttributeNode]) -> Dict[str, Set[str]]:
        """Construye grafo de conexiones entre métodos."""
        graph = defaultdict(set)
        attribute_names = {attr.name for attr in attributes}
        
        # Crear conexiones directas (métodos que comparten atributos)
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    if self._methods_share_attributes(method1, method2, attribute_names):
                        graph[method1.name].add(method2.name)
                    
                    # También verificar si un método llama al otro
                    if self._method_calls_method(method1, method2):
                        graph[method1.name].add(method2.name)
        
        return graph
    
    def _methods_share_attributes(self, method1: MethodNode, method2: MethodNode, 
                                attribute_names: Set[str]) -> bool:
        """Verifica si métodos comparten atributos."""
        attrs1 = self._get_method_attributes(method1, attribute_names)
        attrs2 = self._get_method_attributes(method2, attribute_names)
        
        return bool(attrs1.intersection(attrs2))
    
    def _method_calls_method(self, caller: MethodNode, callee: MethodNode) -> bool:
        """Verifica si un método llama a otro."""
        # Buscar llamadas al método en el cuerpo del caller
        return self._contains_method_call(caller.body, callee.name)
    
    def _contains_method_call(self, node: UnifiedNode, method_name: str) -> bool:
        """Verifica si nodo contiene llamada a método específico."""
        if node.node_type == UnifiedNodeType.CALL_EXPRESSION:
            # Buscar nombre del método en la llamada
            if hasattr(node, 'value') and node.value and method_name in node.value:
                return True
        
        # Buscar en hijos
        for child in node.children:
            if self._contains_method_call(child, method_name):
                return True
        
        return False
    
    def _get_method_attributes(self, method: MethodNode, attribute_names: Set[str]) -> Set[str]:
        """Obtiene atributos accedidos por método."""
        accessed = set()
        self._find_attributes_recursive(method.body, attribute_names, accessed)
        return accessed
    
    def _find_attributes_recursive(self, node: UnifiedNode, attribute_names: Set[str], accessed: Set[str]) -> None:
        """Busca atributos recursivamente."""
        if node.node_type == UnifiedNodeType.IDENTIFIER and hasattr(node, 'value'):
            if node.value in attribute_names:
                accessed.add(node.value)
        
        for child in node.children:
            self._find_attributes_recursive(child, attribute_names, accessed)
    
    def _count_connected_pairs(self, graph: Dict[str, Set[str]]) -> int:
        """Cuenta pares conectados usando DFS."""
        visited = set()
        connected_pairs = 0
        
        for method in graph:
            if method not in visited:
                component = set()
                self._dfs(method, graph, visited, component)
                
                # Cada componente conectado contribuye con C(n,2) pares
                component_size = len(component)
                if component_size >= 2:
                    connected_pairs += component_size * (component_size - 1) // 2
        
        return connected_pairs
    
    def _dfs(self, method: str, graph: Dict[str, Set[str]], visited: Set[str], component: Set[str]) -> None:
        """Depth-first search para encontrar componentes conectados."""
        visited.add(method)
        component.add(method)
        
        for neighbor in graph.get(method, set()):
            if neighbor not in visited:
                self._dfs(neighbor, graph, visited, component)


class CohesionAnalyzer:
    """Analizador principal de cohesión."""
    
    def __init__(self):
        self.lcom_calculator = LCOMCalculator()
        self.tcc_calculator = TCCCalculator()
        self.lcc_calculator = LCCCalculator()
    
    async def calculate_cohesion_metrics(
        self, 
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> CohesionAnalysisResult:
        """
        Calcula métricas de cohesión completas.
        
        Args:
            parse_result: Resultado del parsing
            config: Configuración opcional
            
        Returns:
            CohesionAnalysisResult completo
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Calculando métricas de cohesión para {parse_result.file_path}")
            
            # Convertir a unified node y extraer clases
            unified_root = self._convert_to_unified_node(parse_result.tree.root_node)
            classes = self._extract_classes(unified_root, parse_result.file_path)
            
            if not classes:
                logger.debug("No se encontraron clases para análisis de cohesión")
                return CohesionAnalysisResult(
                    cohesion_metrics=CohesionMetrics(),
                    class_cohesion_details=[],
                    method_interactions=[],
                    analysis_time_ms=int((time.time() - start_time) * 1000)
                )
            
            # Analizar cohesión de cada clase
            class_details = []
            total_lcom = 0.0
            total_tcc = 0.0
            total_lcc = 0.0
            all_method_interactions = []
            
            for class_node in classes:
                # Calcular métricas de cohesión
                lcom_score = self.lcom_calculator.calculate_lcom(class_node)
                tcc_score = self.tcc_calculator.calculate_tcc(class_node)
                lcc_score = self.lcc_calculator.calculate_lcc(class_node)
                
                # Determinar nivel de cohesión
                cohesion_level = self._determine_cohesion_level(lcom_score, tcc_score, lcc_score)
                
                # Identificar métodos problemáticos
                problematic_methods = self._identify_problematic_methods(class_node)
                
                # Generar sugerencias
                suggestions = self._generate_cohesion_suggestions(class_node, lcom_score, tcc_score, lcc_score)
                
                class_detail = ClassCohesionDetail(
                    class_name=class_node.name,
                    lcom_score=lcom_score,
                    tcc_score=tcc_score,
                    lcc_score=lcc_score,
                    method_count=len(class_node.methods),
                    attribute_count=len(class_node.attributes),
                    cohesion_level=cohesion_level,
                    problematic_methods=problematic_methods,
                    suggestions=suggestions
                )
                
                class_details.append(class_detail)
                
                # Acumular para promedios
                total_lcom += lcom_score
                total_tcc += tcc_score
                total_lcc += lcc_score
                
                # Obtener interacciones de métodos
                class_interactions = self._analyze_method_interactions(class_node)
                all_method_interactions.extend(class_interactions)
            
            # Calcular métricas promedio
            num_classes = len(classes)
            cohesion_metrics = CohesionMetrics(
                average_lcom=total_lcom / num_classes,
                average_tcc=total_tcc / num_classes,
                average_lcc=total_lcc / num_classes,
                class_count=num_classes,
                highly_cohesive_classes=len([d for d in class_details if d.cohesion_level == "high"]),
                poorly_cohesive_classes=len([d for d in class_details if d.cohesion_level == "low"])
            )
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Análisis de cohesión completado para {parse_result.file_path}: "
                f"{num_classes} clases, LCOM avg={cohesion_metrics.average_lcom:.3f}, "
                f"TCC avg={cohesion_metrics.average_tcc:.3f} en {total_time}ms"
            )
            
            return CohesionAnalysisResult(
                cohesion_metrics=cohesion_metrics,
                class_cohesion_details=class_details,
                method_interactions=all_method_interactions,
                analysis_time_ms=total_time
            )
            
        except Exception as e:
            logger.error(f"Error calculando métricas de cohesión: {e}")
            raise
    
    def _convert_to_unified_node(self, tree_sitter_node) -> UnifiedNode:
        """Convierte tree-sitter node a UnifiedNode (implementación segura)."""
        from ...domain.entities.ast_normalization import SourcePosition
        
        try:
            start_point = getattr(tree_sitter_node, 'start_point', (0, 0))
            end_point = getattr(tree_sitter_node, 'end_point', (0, 0))
            start_byte = getattr(tree_sitter_node, 'start_byte', 0)
            end_byte = getattr(tree_sitter_node, 'end_byte', 0)
            
            # Asegurar tipos correctos
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
                'function_def': UnifiedNodeType.FUNCTION_DECLARATION,
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
            
            from ...domain.entities.dead_code_analysis import SourceRange, SourcePosition
            location = SourceRange(
                start=SourcePosition(line=node.position.start_line + 1, column=node.position.start_column),
                end=SourcePosition(line=node.position.end_line + 1, column=node.position.end_column)
            )
            
            class_node = ClassNode(
                name=class_name,
                location=location,
                methods=methods,
                attributes=attributes
            )
            
            classes.append(class_node)
        
        for child in node.children:
            self._extract_classes_recursive(child, classes, file_path)
    
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
                    parameters=parameters,
                    visibility=self._determine_method_visibility(child)
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
                    location=location,
                    visibility=self._determine_attribute_visibility(child)
                )
                
                attributes.append(attribute)
        
        return attributes
    
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
    
    def _determine_method_visibility(self, node: UnifiedNode) -> str:
        """Determina visibilidad del método."""
        if hasattr(node, 'value') and node.value:
            if '_' in node.value and not node.value.startswith('__'):
                return "private"  # Convención Python
            elif 'private' in node.value:
                return "private"
            elif 'protected' in node.value:
                return "protected"
        
        return "public"
    
    def _determine_attribute_visibility(self, node: UnifiedNode) -> str:
        """Determina visibilidad del atributo."""
        return self._determine_method_visibility(node)  # Misma lógica
    
    def _determine_cohesion_level(self, lcom: float, tcc: float, lcc: float) -> str:
        """Determina nivel general de cohesión."""
        # Combinar métricas para determinar nivel
        if lcom <= 0.3 and tcc >= 0.7:
            return "high"
        elif lcom <= 0.6 and tcc >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _identify_problematic_methods(self, class_node: ClassNode) -> List[str]:
        """Identifica métodos que pueden causar problemas de cohesión."""
        problematic = []
        attribute_names = {attr.name for attr in class_node.attributes}
        
        for method in class_node.methods:
            accessed_attrs = self._get_method_attributes_access(method, attribute_names)
            
            # Método que no accede a ningún atributo
            if not accessed_attrs and attribute_names:
                problematic.append(f"{method.name} (no accede a atributos de clase)")
            
            # Método que accede a muy pocos atributos comparado con otros
            access_ratio = len(accessed_attrs) / len(attribute_names) if attribute_names else 0
            if access_ratio < 0.2 and len(attribute_names) > 3:
                problematic.append(f"{method.name} (bajo uso de atributos: {access_ratio:.1%})")
        
        return problematic
    
    def _generate_cohesion_suggestions(self, class_node: ClassNode, lcom: float, tcc: float, lcc: float) -> List[str]:
        """Genera sugerencias para mejorar cohesión."""
        suggestions = []
        
        if lcom > 0.7:
            suggestions.extend([
                "Consider splitting this class into smaller, more focused classes",
                "Group related methods and attributes together",
                "Apply Single Responsibility Principle"
            ])
        
        if tcc < 0.3:
            suggestions.extend([
                "Increase method interactions by sharing more attributes",
                "Consider if some methods belong to different classes",
                "Look for methods that can call each other"
            ])
        
        if len(class_node.methods) > 20:
            suggestions.append("Class has too many methods - consider decomposition")
        
        if len(class_node.attributes) > 15:
            suggestions.append("Class has too many attributes - consider grouping into objects")
        
        return suggestions
    
    def _analyze_method_interactions(self, class_node: ClassNode) -> List[MethodInteraction]:
        """Analiza interacciones entre métodos."""
        interactions = []
        attribute_names = {attr.name for attr in class_node.attributes}
        
        for i, method1 in enumerate(class_node.methods):
            for j, method2 in enumerate(class_node.methods[i+1:], i+1):
                interaction = self._analyze_method_pair_interaction(method1, method2, attribute_names)
                if interaction:
                    interactions.append(interaction)
        
        return interactions
    
    def _analyze_method_pair_interaction(self, method1: MethodNode, method2: MethodNode, 
                                       attribute_names: Set[str]) -> Optional[MethodInteraction]:
        """Analiza interacción entre un par de métodos."""
        # Verificar si comparten atributos
        attrs1 = self._get_method_attributes_access(method1, attribute_names)
        attrs2 = self._get_method_attributes_access(method2, attribute_names)
        shared_attrs = attrs1.intersection(attrs2)
        
        if shared_attrs:
            # Calcular fuerza de interacción
            strength = len(shared_attrs) / len(attrs1.union(attrs2)) if attrs1.union(attrs2) else 0.0
            
            return MethodInteraction(
                method1=method1.name,
                method2=method2.name,
                interaction_type="shared_attributes",
                strength=strength
            )
        
        # Verificar si uno llama al otro
        if self._method_calls_method(method1, method2.name):
            return MethodInteraction(
                method1=method1.name,
                method2=method2.name,
                interaction_type="calls",
                strength=1.0
            )
        
        return None
    
    def _get_method_attributes_access(self, method: MethodNode, attribute_names: Set[str]) -> Set[str]:
        """Obtiene atributos accedidos por método."""
        accessed = set()
        self._find_attribute_accesses_recursive(method.body, attribute_names, accessed)
        return accessed
    
    def _find_attribute_accesses_recursive(self, node: UnifiedNode, attribute_names: Set[str], accessed: Set[str]) -> None:
        """Busca accesos a atributos recursivamente."""
        if node.node_type == UnifiedNodeType.IDENTIFIER and hasattr(node, 'value'):
            if node.value in attribute_names:
                accessed.add(node.value)
        
        elif node.node_type == UnifiedNodeType.MEMBER_EXPRESSION:
            member_name = self._extract_member_name(node)
            if member_name in attribute_names:
                accessed.add(member_name)
        
        for child in node.children:
            self._find_attribute_accesses_recursive(child, attribute_names, accessed)
    
    def _extract_member_name(self, node: UnifiedNode) -> str:
        """Extrae nombre de miembro de expresión."""
        if hasattr(node, 'value') and node.value:
            import re
            member_match = re.search(r'(?:self|this)\.(\w+)', node.value)
            if member_match:
                return member_match.group(1)
        return ""
    
    def _method_calls_method(self, caller: MethodNode, callee_name: str) -> bool:
        """Verifica si un método llama a otro."""
        return self._contains_method_call(caller.body, callee_name)
    
    def _contains_method_call(self, node: UnifiedNode, method_name: str) -> bool:
        """Verifica si nodo contiene llamada a método."""
        if node.node_type == UnifiedNodeType.CALL_EXPRESSION:
            if hasattr(node, 'value') and node.value and method_name in node.value:
                return True
        
        for child in node.children:
            if self._contains_method_call(child, method_name):
                return True
        
        return False
