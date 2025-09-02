"""
Implementación del analizador de flujo de datos.

Este módulo implementa el análisis de flujo de datos para detectar variables
no utilizadas, asignaciones redundantes y análisis def-use.
"""

import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ...domain.entities.dead_code_analysis import (
    UnusedVariable, RedundantAssignment, UnusedParameter,
    RedundancyType, AssignmentType, ScopeInfo, ScopeType,
    SourceRange, SourcePosition, UnusedReason
)
from ...domain.entities.dependency_analysis import (
    ControlFlowGraph, ControlFlowNode, LivenessInfo, DefUseChain,
    NodeId, NodeType
)
from ...domain.entities.parse_result import ParseResult
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class VariableDefinition:
    """Definición de una variable."""
    name: str
    node_id: NodeId
    location: SourceRange
    assignment_type: AssignmentType
    scope: ScopeInfo


@dataclass
class VariableUsage:
    """Uso de una variable."""
    name: str
    node_id: NodeId
    location: SourceRange
    usage_type: str  # read, write, call, etc.


@dataclass
class DataFlowAnalysisResult:
    """Resultado del análisis de flujo de datos."""
    unused_variables: List[UnusedVariable]
    redundant_assignments: List[RedundantAssignment]
    unused_parameters: List[UnusedParameter]
    liveness_info: LivenessInfo
    def_use_chains: List[DefUseChain]
    analysis_time_ms: int


class LiveVariableAnalyzer:
    """Analizador de variables vivas usando análisis backward."""
    
    def analyze_liveness(self, cfg: ControlFlowGraph, definitions: Dict[str, List[VariableDefinition]]) -> LivenessInfo:
        """
        Realiza análisis de liveness usando algoritmo iterativo backward.
        
        Args:
            cfg: Grafo de control de flujo
            definitions: Definiciones de variables por nombre
            
        Returns:
            LivenessInfo con información de liveness
        """
        liveness_info = LivenessInfo()
        
        # Inicializar conjuntos
        for node_id in cfg.get_all_nodes():
            liveness_info.live_in[node_id] = set()
            liveness_info.live_out[node_id] = set()
        
        # Análisis iterativo hasta convergencia
        changed = True
        iterations = 0
        max_iterations = 100
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            # Procesar nodos en orden reverso (backward)
            for node_id in reversed(cfg.get_all_nodes()):
                node = cfg.get_node(node_id)
                if not node:
                    continue
                
                old_live_in = liveness_info.live_in[node_id].copy()
                old_live_out = liveness_info.live_out[node_id].copy()
                
                # Calcular LIVE_OUT[n] = ∪ LIVE_IN[s] para todos los sucesores s
                new_live_out = set()
                for successor_id in cfg.get_successors(node_id):
                    new_live_out.update(liveness_info.live_in[successor_id])
                
                # Calcular USE[n] y DEF[n] para este nodo
                use_vars = self._get_used_variables(node)
                def_vars = self._get_defined_variables(node)
                
                # Calcular LIVE_IN[n] = USE[n] ∪ (LIVE_OUT[n] - DEF[n])
                new_live_in = use_vars.union(new_live_out - def_vars)
                
                liveness_info.live_out[node_id] = new_live_out
                liveness_info.live_in[node_id] = new_live_in
                
                if old_live_in != new_live_in or old_live_out != new_live_out:
                    changed = True
        
        if iterations >= max_iterations:
            logger.warning(f"Análisis de liveness no convergió después de {max_iterations} iteraciones")
        
        return liveness_info
    
    def _get_used_variables(self, node: ControlFlowNode) -> Set[str]:
        """Extrae variables usadas en un nodo."""
        used_vars = set()
        
        # Analizar el contenido del nodo para encontrar usos de variables
        if node.content:
            # Esta es una implementación simplificada
            # En una implementación real, se haría parsing del contenido
            import re
            
            # Buscar patrones de uso de variables
            var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            matches = re.findall(var_pattern, node.content)
            
            # Filtrar palabras clave y literales
            keywords = {'if', 'else', 'while', 'for', 'def', 'class', 'import', 
                       'return', 'break', 'continue', 'try', 'except', 'finally'}
            
            for match in matches:
                if match not in keywords and not match.isdigit():
                    # Solo agregamos si no es parte de una asignación
                    if not self._is_assignment_target(node.content, match):
                        used_vars.add(match)
        
        return used_vars
    
    def _get_defined_variables(self, node: ControlFlowNode) -> Set[str]:
        """Extrae variables definidas en un nodo."""
        defined_vars = set()
        
        if node.content:
            import re
            
            # Buscar patrones de asignación
            assignment_patterns = [
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=',  # x = ...
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\+=',  # x += ...
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*-=',  # x -= ...
                r'\bfor\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in',  # for x in ...
            ]
            
            for pattern in assignment_patterns:
                matches = re.findall(pattern, node.content)
                defined_vars.update(matches)
        
        return defined_vars
    
    def _is_assignment_target(self, content: str, var_name: str) -> bool:
        """Verifica si una variable es objetivo de asignación."""
        import re
        pattern = rf'\b{re.escape(var_name)}\s*='
        return bool(re.search(pattern, content))


class DefUseAnalyzer:
    """Analizador de cadenas definición-uso."""
    
    def build_def_use_chains(
        self, 
        cfg: ControlFlowGraph,
        definitions: Dict[str, List[VariableDefinition]],
        usages: Dict[str, List[VariableUsage]]
    ) -> List[DefUseChain]:
        """
        Construye cadenas de definición-uso.
        
        Args:
            cfg: Grafo de control de flujo
            definitions: Definiciones por variable
            usages: Usos por variable
            
        Returns:
            Lista de cadenas def-use
        """
        chains = []
        
        for var_name, var_definitions in definitions.items():
            var_usages = usages.get(var_name, [])
            
            for definition in var_definitions:
                # Encontrar todos los usos alcanzables desde esta definición
                reachable_uses = self._find_reachable_uses(
                    definition, var_usages, cfg
                )
                
                chain = DefUseChain(
                    definition=definition.node_id,
                    uses=[usage.node_id for usage in reachable_uses],
                    variable_name=var_name
                )
                chains.append(chain)
        
        return chains
    
    def _find_reachable_uses(
        self,
        definition: VariableDefinition,
        usages: List[VariableUsage],
        cfg: ControlFlowGraph
    ) -> List[VariableUsage]:
        """Encuentra usos alcanzables desde una definición."""
        reachable_uses = []
        
        for usage in usages:
            if self._is_reachable(definition.node_id, usage.node_id, cfg):
                # Verificar que no hay redefinición en el camino
                if not self._has_redefinition_between(
                    definition, usage, cfg
                ):
                    reachable_uses.append(usage)
        
        return reachable_uses
    
    def _is_reachable(self, from_node: NodeId, to_node: NodeId, cfg: ControlFlowGraph) -> bool:
        """Verifica si un nodo es alcanzable desde otro."""
        if from_node == to_node:
            return True
        
        visited = set()
        stack = [from_node]
        
        while stack:
            current = stack.pop()
            if current == to_node:
                return True
            
            if current in visited:
                continue
            
            visited.add(current)
            stack.extend(cfg.get_successors(current))
        
        return False
    
    def _has_redefinition_between(
        self,
        definition: VariableDefinition,
        usage: VariableUsage,
        cfg: ControlFlowGraph
    ) -> bool:
        """Verifica si hay redefinición entre definición y uso."""
        # Implementación simplificada
        # En una implementación real, se haría análisis más sofisticado
        return False


class DataFlowAnalyzer:
    """Analizador principal de flujo de datos."""
    
    def __init__(self):
        self.liveness_analyzer = LiveVariableAnalyzer()
        self.def_use_analyzer = DefUseAnalyzer()
    
    async def analyze_data_flow(
        self, 
        parse_result: ParseResult,
        cfg: ControlFlowGraph,
        config: Optional[Dict[str, Any]] = None
    ) -> DataFlowAnalysisResult:
        """
        Realiza análisis completo de flujo de datos.
        
        Args:
            parse_result: Resultado del parsing
            cfg: Grafo de control de flujo
            config: Configuración opcional
            
        Returns:
            DataFlowAnalysisResult con los resultados
        """
        import time
        start_time = time.time()
        
        try:
            # Extraer definiciones y usos de variables
            definitions = self._extract_variable_definitions(parse_result, cfg)
            usages = self._extract_variable_usages(parse_result, cfg)
            
            # Análisis de liveness
            liveness_info = self.liveness_analyzer.analyze_liveness(cfg, definitions)
            
            # Construcción de cadenas def-use
            def_use_chains = self.def_use_analyzer.build_def_use_chains(
                cfg, definitions, usages
            )
            
            # Detectar variables no utilizadas
            unused_variables = self._detect_unused_variables(
                definitions, def_use_chains, liveness_info
            )
            
            # Detectar asignaciones redundantes
            redundant_assignments = self._detect_redundant_assignments(
                definitions, liveness_info, cfg
            )
            
            # Detectar parámetros no utilizados
            unused_parameters = self._detect_unused_parameters(
                parse_result, usages
            )
            
            analysis_time = int((time.time() - start_time) * 1000)
            
            return DataFlowAnalysisResult(
                unused_variables=unused_variables,
                redundant_assignments=redundant_assignments,
                unused_parameters=unused_parameters,
                liveness_info=liveness_info,
                def_use_chains=def_use_chains,
                analysis_time_ms=analysis_time
            )
            
        except Exception as e:
            logger.error(f"Error en análisis de flujo de datos: {e}")
            raise
    
    def _extract_variable_definitions(
        self, 
        parse_result: ParseResult,
        cfg: ControlFlowGraph
    ) -> Dict[str, List[VariableDefinition]]:
        """Extrae definiciones de variables del código."""
        definitions = defaultdict(list)
        
        # Analizar el AST para encontrar definiciones
        self._analyze_ast_for_definitions(
            parse_result.tree.root_node, definitions, cfg
        )
        
        return dict(definitions)
    
    def _analyze_ast_for_definitions(
        self, 
        ast_node: Any, 
        definitions: Dict[str, List[VariableDefinition]],
        cfg: ControlFlowGraph,
        scope_stack: Optional[List[ScopeInfo]] = None
    ) -> None:
        """Analiza recursivamente el AST para encontrar definiciones."""
        if scope_stack is None:
            scope_stack = [ScopeInfo(ScopeType.MODULE, nesting_level=0)]
        
        node_type = ast_node.type
        
        # Manejar diferentes tipos de definiciones
        if node_type == 'assignment':
            self._handle_assignment(ast_node, definitions, cfg, scope_stack)
        elif node_type == 'function_definition':
            self._handle_function_definition(ast_node, definitions, cfg, scope_stack)
        elif node_type == 'for_statement':
            self._handle_for_loop(ast_node, definitions, cfg, scope_stack)
        
        # Procesar hijos recursivamente
        for child in ast_node.children:
            self._analyze_ast_for_definitions(child, definitions, cfg, scope_stack)
    
    def _handle_assignment(
        self, 
        ast_node: Any, 
        definitions: Dict[str, List[VariableDefinition]],
        cfg: ControlFlowGraph,
        scope_stack: List[ScopeInfo]
    ) -> None:
        """Maneja nodos de asignación."""
        # Extraer información de la asignación
        var_name = self._extract_assignment_target(ast_node)
        if var_name:
            # Encontrar el nodo correspondiente en el CFG
            node_id = self._find_cfg_node_for_ast(ast_node, cfg)
            if node_id:
                location = self._extract_source_range(ast_node)
                assignment_type = self._determine_assignment_type(ast_node)
                current_scope = scope_stack[-1] if scope_stack else ScopeInfo(ScopeType.MODULE)
                
                definition = VariableDefinition(
                    name=var_name,
                    node_id=node_id,
                    location=location,
                    assignment_type=assignment_type,
                    scope=current_scope
                )
                
                definitions[var_name].append(definition)
    
    def _extract_variable_usages(
        self, 
        parse_result: ParseResult,
        cfg: ControlFlowGraph
    ) -> Dict[str, List[VariableUsage]]:
        """Extrae usos de variables del código."""
        usages = defaultdict(list)
        
        # Analizar el AST para encontrar usos
        self._analyze_ast_for_usages(
            parse_result.tree.root_node, usages, cfg
        )
        
        return dict(usages)
    
    def _analyze_ast_for_usages(
        self, 
        ast_node: Any, 
        usages: Dict[str, List[VariableUsage]],
        cfg: ControlFlowGraph
    ) -> None:
        """Analiza recursivamente el AST para encontrar usos."""
        node_type = ast_node.type
        
        if node_type == 'identifier' and not self._is_assignment_target_context(ast_node):
            var_name = ast_node.text.decode('utf-8') if ast_node.text else ''
            if var_name:
                node_id = self._find_cfg_node_for_ast(ast_node, cfg)
                if node_id:
                    location = self._extract_source_range(ast_node)
                    
                    usage = VariableUsage(
                        name=var_name,
                        node_id=node_id,
                        location=location,
                        usage_type='read'
                    )
                    
                    usages[var_name].append(usage)
        
        # Procesar hijos recursivamente
        for child in ast_node.children:
            self._analyze_ast_for_usages(child, usages, cfg)
    
    def _detect_unused_variables(
        self, 
        definitions: Dict[str, List[VariableDefinition]],
        def_use_chains: List[DefUseChain],
        liveness_info: LivenessInfo
    ) -> List[UnusedVariable]:
        """Detecta variables no utilizadas."""
        unused_variables = []
        
        # Crear mapa de definiciones a cadenas
        def_to_chain = {}
        for chain in def_use_chains:
            def_to_chain[chain.definition] = chain
        
        for var_name, var_definitions in definitions.items():
            for definition in var_definitions:
                chain = def_to_chain.get(definition.node_id)
                
                # Si no hay cadena o no tiene usos
                if not chain or not chain.has_uses():
                    reason = UnusedReason.NEVER_REFERENCED
                    suggestion = f"Eliminar variable no utilizada '{var_name}'"
                    confidence = 0.9
                    
                    # Ajustar confianza basada en el scope
                    if definition.scope.scope_type == ScopeType.GLOBAL:
                        confidence = 0.7  # Menos confianza para variables globales
                    
                    unused_var = UnusedVariable(
                        name=var_name,
                        declaration_location=definition.location,
                        variable_type=None,  # Por implementar
                        scope=definition.scope,
                        reason=reason,
                        suggestion=suggestion,
                        confidence=confidence
                    )
                    
                    unused_variables.append(unused_var)
        
        return unused_variables
    
    def _detect_redundant_assignments(
        self, 
        definitions: Dict[str, List[VariableDefinition]],
        liveness_info: LivenessInfo,
        cfg: ControlFlowGraph
    ) -> List[RedundantAssignment]:
        """Detecta asignaciones redundantes."""
        redundant_assignments = []
        
        for var_name, var_definitions in definitions.items():
            # Buscar asignaciones consecutivas sin uso entre ellas
            for i, current_def in enumerate(var_definitions):
                # Buscar la siguiente definición
                next_def = None
                for j, other_def in enumerate(var_definitions):
                    if j > i and self._is_reachable(
                        current_def.node_id, other_def.node_id, cfg
                    ):
                        next_def = other_def
                        break
                
                if next_def:
                    # Verificar si la variable no es usada entre las definiciones
                    if not liveness_info.is_live(current_def.node_id, var_name):
                        redundant = RedundantAssignment(
                            location=current_def.location,
                            variable_name=var_name,
                            previous_assignment=next_def.location,
                            redundancy_type=RedundancyType.UNUSED_BETWEEN_ASSIGNMENTS,
                            suggestion=f"Eliminar asignación redundante de '{var_name}'",
                            confidence=0.8
                        )
                        
                        redundant_assignments.append(redundant)
        
        return redundant_assignments
    
    def _detect_unused_parameters(
        self, 
        parse_result: ParseResult,
        usages: Dict[str, List[VariableUsage]]
    ) -> List[UnusedParameter]:
        """Detecta parámetros no utilizados."""
        unused_parameters = []
        
        # Buscar definiciones de función en el AST
        self._find_function_parameters(
            parse_result.tree.root_node, unused_parameters, usages
        )
        
        return unused_parameters
    
    def _find_function_parameters(
        self, 
        ast_node: Any, 
        unused_parameters: List[UnusedParameter],
        usages: Dict[str, List[VariableUsage]]
    ) -> None:
        """Busca parámetros de función no utilizados."""
        if ast_node.type == 'function_definition':
            # Extraer parámetros de la función
            parameters = self._extract_function_parameters(ast_node)
            function_name = self._extract_function_name(ast_node)
            
            for param_name, param_location in parameters:
                # Verificar si el parámetro es utilizado
                if param_name not in usages or not usages[param_name]:
                    # Ignorar parámetros especiales como 'self', 'cls'
                    if param_name not in ['self', 'cls']:
                        unused_param = UnusedParameter(
                            name=param_name,
                            function_name=function_name or 'unknown',
                            location=param_location,
                            parameter_type=None,  # Por implementar
                            is_self_parameter=param_name == 'self',
                            suggestion=f"Eliminar parámetro no utilizado '{param_name}'",
                            confidence=0.85
                        )
                        
                        unused_parameters.append(unused_param)
        
        # Procesar hijos recursivamente
        for child in ast_node.children:
            self._find_function_parameters(child, unused_parameters, usages)
    
    # Métodos auxiliares (implementaciones simplificadas)
    
    def _extract_assignment_target(self, ast_node: Any) -> Optional[str]:
        """Extrae el objetivo de una asignación."""
        # Implementación simplificada
        for child in ast_node.children:
            if child.type == 'identifier':
                return child.text.decode('utf-8') if child.text else None
        return None
    
    def _find_cfg_node_for_ast(self, ast_node: Any, cfg: ControlFlowGraph) -> Optional[NodeId]:
        """Encuentra el nodo del CFG correspondiente a un nodo AST."""
        # Implementación simplificada - en una implementación real
        # se mantendría un mapeo entre nodos AST y CFG
        for node_id, cfg_node in cfg.nodes.items():
            if cfg_node.metadata and cfg_node.metadata.get('ast_id') == id(ast_node):
                return node_id
        return None
    
    def _extract_source_range(self, ast_node: Any) -> SourceRange:
        """Extrae el rango de código fuente de un nodo AST."""
        start_pos = SourcePosition(
            line=ast_node.start_point[0] + 1,
            column=ast_node.start_point[1],
            offset=ast_node.start_byte
        )
        end_pos = SourcePosition(
            line=ast_node.end_point[0] + 1,
            column=ast_node.end_point[1],
            offset=ast_node.end_byte
        )
        
        return SourceRange(start=start_pos, end=end_pos)
    
    def _determine_assignment_type(self, ast_node: Any) -> AssignmentType:
        """Determina el tipo de asignación."""
        # Implementación simplificada
        return AssignmentType.SIMPLE_ASSIGNMENT
    
    def _is_assignment_target_context(self, ast_node: Any) -> bool:
        """Verifica si un nodo está en contexto de asignación."""
        # Implementación simplificada
        parent = ast_node.parent if hasattr(ast_node, 'parent') else None
        return parent and parent.type == 'assignment'
    
    def _is_reachable(self, from_node: NodeId, to_node: NodeId, cfg: ControlFlowGraph) -> bool:
        """Verifica alcanzabilidad entre nodos."""
        return cfg.is_reachable(from_node, to_node)
    
    def _extract_function_parameters(self, ast_node: Any) -> List[Tuple[str, SourceRange]]:
        """Extrae parámetros de una función."""
        parameters = []
        # Implementación simplificada
        return parameters
    
    def _extract_function_name(self, ast_node: Any) -> Optional[str]:
        """Extrae el nombre de una función."""
        # Implementación simplificada
        for child in ast_node.children:
            if child.type == 'identifier':
                return child.text.decode('utf-8') if child.text else None
        return None
