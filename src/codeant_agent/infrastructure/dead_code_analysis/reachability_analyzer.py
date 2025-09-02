"""
Implementación del analizador de alcanzabilidad de código.

Este módulo implementa el análisis de reachability para detectar código
inalcanzable y determinar qué símbolos son alcanzables desde los entry points.
"""

import logging
from typing import Dict, List, Set, Optional, Any, Tuple, Deque
from collections import deque
from dataclasses import dataclass
import re

from ...domain.entities.dead_code_analysis import (
    UnreachableCode, UnreachabilityReason, DeadBranch, EntryPoint,
    EntryPointType, SourceRange, SourcePosition, BlockingCondition
)
from ...domain.entities.dependency_analysis import (
    ControlFlowGraph, ControlFlowNode, ControlFlowEdge, GlobalDependencyGraph,
    SymbolId, NodeId, NodeType, EdgeType
)
from ...domain.entities.parse_result import ParseResult
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class ReachabilityResult:
    """Resultado del análisis de alcanzabilidad."""
    reachable_nodes: Set[NodeId]
    unreachable_nodes: Set[NodeId]
    unreachable_code_blocks: List[UnreachableCode]
    dead_branches: List[DeadBranch]
    analysis_time_ms: int


class ControlFlowGraphBuilder:
    """Constructor de grafos de control de flujo."""
    
    def __init__(self):
        self.node_counter = 0
    
    def build_cfg(self, parse_result: ParseResult) -> ControlFlowGraph:
        """
        Construye el grafo de control de flujo para un archivo.
        
        Args:
            parse_result: Resultado del parsing
            
        Returns:
            ControlFlowGraph construido
        """
        cfg = ControlFlowGraph()
        
        try:
            # Crear nodo de entrada
            entry_node = self._create_entry_node()
            cfg.add_node(entry_node)
            
            # Procesar el AST
            exit_nodes = self._process_ast_node(
                parse_result.tree.root_node, 
                cfg, 
                entry_node.id
            )
            
            # Crear nodo de salida si es necesario
            if not exit_nodes:
                exit_node = self._create_exit_node()
                cfg.add_node(exit_node)
                exit_nodes = [exit_node.id]
            
            # Marcar nodos de salida
            for exit_id in exit_nodes:
                cfg.exit_nodes.add(exit_id)
            
            return cfg
            
        except Exception as e:
            logger.error(f"Error construyendo CFG: {e}")
            raise
    
    def _create_entry_node(self) -> ControlFlowNode:
        """Crea el nodo de entrada del CFG."""
        node_id = NodeId(self.node_counter)
        self.node_counter += 1
        
        return ControlFlowNode(
            id=node_id,
            node_type=NodeType.ENTRY,
            position=SourceRange(
                start=SourcePosition(line=1, column=0),
                end=SourcePosition(line=1, column=0)
            ),
            content="<ENTRY>"
        )
    
    def _create_exit_node(self) -> ControlFlowNode:
        """Crea el nodo de salida del CFG."""
        node_id = NodeId(self.node_counter)
        self.node_counter += 1
        
        return ControlFlowNode(
            id=node_id,
            node_type=NodeType.EXIT,
            position=SourceRange(
                start=SourcePosition(line=1, column=0),
                end=SourcePosition(line=1, column=0)
            ),
            content="<EXIT>"
        )
    
    def _process_ast_node(
        self, 
        ast_node: Any, 
        cfg: ControlFlowGraph, 
        predecessor_id: NodeId
    ) -> List[NodeId]:
        """
        Procesa un nodo del AST y actualiza el CFG.
        
        Args:
            ast_node: Nodo del AST a procesar
            cfg: Grafo de control de flujo
            predecessor_id: ID del nodo predecesor
            
        Returns:
            Lista de IDs de nodos finales
        """
        node_type_str = ast_node.type
        
        if node_type_str in ['function_definition', 'method_definition']:
            return self._process_function(ast_node, cfg, predecessor_id)
        elif node_type_str in ['if_statement']:
            return self._process_if_statement(ast_node, cfg, predecessor_id)
        elif node_type_str in ['while_statement', 'for_statement']:
            return self._process_loop(ast_node, cfg, predecessor_id)
        elif node_type_str in ['return_statement']:
            return self._process_return(ast_node, cfg, predecessor_id)
        elif node_type_str in ['break_statement']:
            return self._process_break(ast_node, cfg, predecessor_id)
        elif node_type_str in ['continue_statement']:
            return self._process_continue(ast_node, cfg, predecessor_id)
        else:
            return self._process_generic_statement(ast_node, cfg, predecessor_id)
    
    def _process_function(
        self, 
        ast_node: Any, 
        cfg: ControlFlowGraph, 
        predecessor_id: NodeId
    ) -> List[NodeId]:
        """Procesa una definición de función."""
        # Crear nodo para la función
        func_node = self._create_statement_node(ast_node, NodeType.STATEMENT)
        cfg.add_node(func_node)
        cfg.add_edge(ControlFlowEdge(
            source=predecessor_id,
            target=func_node.id,
            edge_type=EdgeType.SEQUENTIAL
        ))
        
        # Procesar el cuerpo de la función
        body_node = None
        for child in ast_node.children:
            if child.type == 'block':
                body_node = child
                break
        
        if body_node:
            return self._process_block(body_node, cfg, func_node.id)
        
        return [func_node.id]
    
    def _process_if_statement(
        self, 
        ast_node: Any, 
        cfg: ControlFlowGraph, 
        predecessor_id: NodeId
    ) -> List[NodeId]:
        """Procesa una declaración if."""
        # Crear nodo de condición
        condition_node = self._create_statement_node(ast_node, NodeType.CONDITION)
        cfg.add_node(condition_node)
        cfg.add_edge(ControlFlowEdge(
            source=predecessor_id,
            target=condition_node.id,
            edge_type=EdgeType.SEQUENTIAL
        ))
        
        end_nodes = []
        
        # Procesar rama verdadera
        if_body = self._find_child_by_type(ast_node, 'block')
        if if_body:
            true_nodes = self._process_block(if_body, cfg, condition_node.id)
            for node_id in true_nodes:
                cfg.add_edge(ControlFlowEdge(
                    source=condition_node.id,
                    target=node_id,
                    edge_type=EdgeType.CONDITIONAL_TRUE
                ))
            end_nodes.extend(true_nodes)
        
        # Procesar rama else si existe
        else_body = self._find_child_by_type(ast_node, 'else_clause')
        if else_body:
            false_nodes = self._process_ast_node(else_body, cfg, condition_node.id)
            for node_id in false_nodes:
                cfg.add_edge(ControlFlowEdge(
                    source=condition_node.id,
                    target=node_id,
                    edge_type=EdgeType.CONDITIONAL_FALSE
                ))
            end_nodes.extend(false_nodes)
        else:
            # Si no hay else, la condición puede pasar directamente
            end_nodes.append(condition_node.id)
        
        return end_nodes
    
    def _process_loop(
        self, 
        ast_node: Any, 
        cfg: ControlFlowGraph, 
        predecessor_id: NodeId
    ) -> List[NodeId]:
        """Procesa un bucle (while/for)."""
        # Crear nodo de condición del loop
        loop_node = self._create_statement_node(ast_node, NodeType.LOOP)
        cfg.add_node(loop_node)
        cfg.add_edge(ControlFlowEdge(
            source=predecessor_id,
            target=loop_node.id,
            edge_type=EdgeType.SEQUENTIAL
        ))
        
        # Procesar cuerpo del loop
        body = self._find_child_by_type(ast_node, 'block')
        if body:
            body_nodes = self._process_block(body, cfg, loop_node.id)
            
            # Conectar inicio del cuerpo
            if body_nodes:
                cfg.add_edge(ControlFlowEdge(
                    source=loop_node.id,
                    target=body_nodes[0],
                    edge_type=EdgeType.LOOP_ENTRY
                ))
                
                # Conectar final del cuerpo de vuelta al loop
                for end_node in body_nodes:
                    cfg.add_edge(ControlFlowEdge(
                        source=end_node,
                        target=loop_node.id,
                        edge_type=EdgeType.LOOP_BACK
                    ))
        
        # El loop puede salir
        return [loop_node.id]
    
    def _process_return(
        self, 
        ast_node: Any, 
        cfg: ControlFlowGraph, 
        predecessor_id: NodeId
    ) -> List[NodeId]:
        """Procesa una declaración return."""
        return_node = self._create_statement_node(ast_node, NodeType.RETURN)
        cfg.add_node(return_node)
        cfg.add_edge(ControlFlowEdge(
            source=predecessor_id,
            target=return_node.id,
            edge_type=EdgeType.SEQUENTIAL
        ))
        
        # Return es terminal
        cfg.exit_nodes.add(return_node.id)
        return []  # No hay nodos siguientes
    
    def _process_break(
        self, 
        ast_node: Any, 
        cfg: ControlFlowGraph, 
        predecessor_id: NodeId
    ) -> List[NodeId]:
        """Procesa una declaración break."""
        break_node = self._create_statement_node(ast_node, NodeType.BREAK)
        cfg.add_node(break_node)
        cfg.add_edge(ControlFlowEdge(
            source=predecessor_id,
            target=break_node.id,
            edge_type=EdgeType.SEQUENTIAL
        ))
        
        return []  # Break interrumpe el flujo
    
    def _process_continue(
        self, 
        ast_node: Any, 
        cfg: ControlFlowGraph, 
        predecessor_id: NodeId
    ) -> List[NodeId]:
        """Procesa una declaración continue."""
        continue_node = self._create_statement_node(ast_node, NodeType.CONTINUE)
        cfg.add_node(continue_node)
        cfg.add_edge(ControlFlowEdge(
            source=predecessor_id,
            target=continue_node.id,
            edge_type=EdgeType.SEQUENTIAL
        ))
        
        return []  # Continue interrumpe el flujo
    
    def _process_generic_statement(
        self, 
        ast_node: Any, 
        cfg: ControlFlowGraph, 
        predecessor_id: NodeId
    ) -> List[NodeId]:
        """Procesa una declaración genérica."""
        stmt_node = self._create_statement_node(ast_node, NodeType.STATEMENT)
        cfg.add_node(stmt_node)
        cfg.add_edge(ControlFlowEdge(
            source=predecessor_id,
            target=stmt_node.id,
            edge_type=EdgeType.SEQUENTIAL
        ))
        
        return [stmt_node.id]
    
    def _process_block(
        self, 
        block_node: Any, 
        cfg: ControlFlowGraph, 
        predecessor_id: NodeId
    ) -> List[NodeId]:
        """Procesa un bloque de código."""
        current_predecessors = [predecessor_id]
        
        for child in block_node.children:
            if child.type not in ['{', '}']:  # Ignorar delimitadores
                next_predecessors = []
                for pred_id in current_predecessors:
                    end_nodes = self._process_ast_node(child, cfg, pred_id)
                    next_predecessors.extend(end_nodes)
                current_predecessors = next_predecessors if next_predecessors else []
        
        return current_predecessors
    
    def _create_statement_node(self, ast_node: Any, node_type: NodeType) -> ControlFlowNode:
        """Crea un nodo de declaración."""
        node_id = NodeId(self.node_counter)
        self.node_counter += 1
        
        # Extraer posición del nodo AST
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
        
        return ControlFlowNode(
            id=node_id,
            node_type=node_type,
            position=SourceRange(start=start_pos, end=end_pos),
            content=ast_node.text.decode('utf-8') if ast_node.text else ast_node.type,
            metadata={
                'ast_type': ast_node.type,
                'ast_id': id(ast_node)
            }
        )
    
    def _find_child_by_type(self, ast_node: Any, child_type: str) -> Optional[Any]:
        """Encuentra un hijo del AST por tipo."""
        for child in ast_node.children:
            if child.type == child_type:
                return child
        return None


class ReachabilityAnalyzer:
    """Analizador de alcanzabilidad de código."""
    
    def __init__(self):
        self.cfg_builder = ControlFlowGraphBuilder()
    
    async def analyze_reachability(
        self, 
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> ReachabilityResult:
        """
        Analiza la alcanzabilidad del código en un archivo.
        
        Args:
            parse_result: Resultado del parsing
            config: Configuración opcional
            
        Returns:
            ReachabilityResult con los resultados del análisis
        """
        import time
        start_time = time.time()
        
        try:
            # Construir grafo de control de flujo
            cfg = self.cfg_builder.build_cfg(parse_result)
            
            # Encontrar nodos alcanzables
            reachable_nodes = self._find_reachable_nodes(cfg)
            
            # Encontrar nodos inalcanzables
            all_nodes = set(cfg.get_all_nodes())
            unreachable_nodes = all_nodes - reachable_nodes
            
            # Generar código inalcanzable
            unreachable_code_blocks = self._generate_unreachable_code_blocks(
                cfg, unreachable_nodes, parse_result
            )
            
            # Detectar ramas muertas
            dead_branches = self._detect_dead_branches(cfg, parse_result)
            
            analysis_time = int((time.time() - start_time) * 1000)
            
            return ReachabilityResult(
                reachable_nodes=reachable_nodes,
                unreachable_nodes=unreachable_nodes,
                unreachable_code_blocks=unreachable_code_blocks,
                dead_branches=dead_branches,
                analysis_time_ms=analysis_time
            )
            
        except Exception as e:
            logger.error(f"Error en análisis de alcanzabilidad: {e}")
            raise
    
    def _find_reachable_nodes(self, cfg: ControlFlowGraph) -> Set[NodeId]:
        """
        Encuentra todos los nodos alcanzables en el CFG.
        
        Args:
            cfg: Grafo de control de flujo
            
        Returns:
            Set de nodos alcanzables
        """
        reachable: Set[NodeId] = set()
        queue: Deque[NodeId] = deque()
        
        # Empezar desde el nodo de entrada
        if cfg.entry_node:
            queue.append(cfg.entry_node)
            reachable.add(cfg.entry_node)
        
        # BFS para encontrar nodos alcanzables
        while queue:
            current = queue.popleft()
            
            for successor in cfg.get_successors(current):
                if successor not in reachable:
                    reachable.add(successor)
                    queue.append(successor)
        
        return reachable
    
    def _generate_unreachable_code_blocks(
        self, 
        cfg: ControlFlowGraph, 
        unreachable_nodes: Set[NodeId],
        parse_result: ParseResult
    ) -> List[UnreachableCode]:
        """
        Genera bloques de código inalcanzable.
        
        Args:
            cfg: Grafo de control de flujo
            unreachable_nodes: Nodos inalcanzables
            parse_result: Resultado del parsing
            
        Returns:
            Lista de bloques de código inalcanzable
        """
        unreachable_blocks = []
        
        for node_id in unreachable_nodes:
            node = cfg.get_node(node_id)
            if node and node.node_type != NodeType.EXIT:
                reason = self._determine_unreachability_reason(node, cfg)
                blocking_condition = self._find_blocking_condition(node, cfg)
                
                unreachable_block = UnreachableCode(
                    location=node.position,
                    code_type=node.node_type.value,
                    reason=reason,
                    suggestion=self._generate_unreachability_suggestion(node, reason),
                    confidence=self._calculate_unreachability_confidence(node, reason),
                    blocking_condition=blocking_condition
                )
                
                unreachable_blocks.append(unreachable_block)
        
        return unreachable_blocks
    
    def _detect_dead_branches(
        self, 
        cfg: ControlFlowGraph, 
        parse_result: ParseResult
    ) -> List[DeadBranch]:
        """
        Detecta ramas muertas en condicionales.
        
        Args:
            cfg: Grafo de control de flujo
            parse_result: Resultado del parsing
            
        Returns:
            Lista de ramas muertas
        """
        dead_branches = []
        
        # Buscar nodos de condición
        for node_id, node in cfg.nodes.items():
            if node.node_type == NodeType.CONDITION:
                # Verificar si alguna rama nunca se ejecuta
                successors = cfg.get_successors(node_id)
                
                for edge in cfg.edges:
                    if edge.source == node_id and edge.is_conditional():
                        target_node = cfg.get_node(edge.target)
                        if target_node and self._is_dead_branch(edge, cfg):
                            branch_type = 'if' if edge.edge_type == EdgeType.CONDITIONAL_TRUE else 'else'
                            
                            dead_branch = DeadBranch(
                                location=target_node.position,
                                branch_type=branch_type,
                                condition=node.content,
                                reason=f"Rama {branch_type} nunca se ejecuta",
                                suggestion=f"Eliminar rama {branch_type} inalcanzable",
                                confidence=0.9
                            )
                            
                            dead_branches.append(dead_branch)
        
        return dead_branches
    
    def _determine_unreachability_reason(
        self, 
        node: ControlFlowNode, 
        cfg: ControlFlowGraph
    ) -> UnreachabilityReason:
        """Determina la razón de la inalcanzabilidad."""
        # Buscar predecessors para entender por qué es inalcanzable
        predecessors = cfg.get_predecessors(node.id)
        
        if not predecessors:
            return UnreachabilityReason.UNREACHABLE_FROM_ENTRY
        
        # Verificar si hay un return/throw antes
        for pred_id in predecessors:
            pred_node = cfg.get_node(pred_id)
            if pred_node:
                if pred_node.node_type == NodeType.RETURN:
                    return UnreachabilityReason.AFTER_RETURN
                elif pred_node.node_type == NodeType.THROW:
                    return UnreachabilityReason.AFTER_THROW
                elif pred_node.node_type == NodeType.BREAK:
                    return UnreachabilityReason.AFTER_BREAK
                elif pred_node.node_type == NodeType.CONTINUE:
                    return UnreachabilityReason.AFTER_CONTINUE
        
        return UnreachabilityReason.UNREACHABLE_FROM_ENTRY
    
    def _find_blocking_condition(
        self, 
        node: ControlFlowNode, 
        cfg: ControlFlowGraph
    ) -> Optional[BlockingCondition]:
        """Encuentra la condición que bloquea la ejecución."""
        # Buscar condiciones que podrían estar bloqueando
        for pred_id in cfg.get_predecessors(node.id):
            pred_node = cfg.get_node(pred_id)
            if pred_node and pred_node.node_type == NodeType.CONDITION:
                return BlockingCondition(
                    condition_location=pred_node.position,
                    condition_expression=pred_node.content,
                    reason="Condición siempre evaluada como falsa"
                )
        
        return None
    
    def _generate_unreachability_suggestion(
        self, 
        node: ControlFlowNode, 
        reason: UnreachabilityReason
    ) -> str:
        """Genera una sugerencia para el código inalcanzable."""
        if reason == UnreachabilityReason.AFTER_RETURN:
            return "Eliminar código después del return"
        elif reason == UnreachabilityReason.AFTER_THROW:
            return "Eliminar código después del throw"
        elif reason == UnreachabilityReason.ALWAYS_FALSE_CONDITION:
            return "Revisar condición - siempre evaluada como falsa"
        else:
            return "Eliminar código inalcanzable"
    
    def _calculate_unreachability_confidence(
        self, 
        node: ControlFlowNode, 
        reason: UnreachabilityReason
    ) -> float:
        """Calcula la confianza de la inalcanzabilidad."""
        high_confidence_reasons = [
            UnreachabilityReason.AFTER_RETURN,
            UnreachabilityReason.AFTER_THROW,
            UnreachabilityReason.AFTER_BREAK,
            UnreachabilityReason.AFTER_CONTINUE,
        ]
        
        if reason in high_confidence_reasons:
            return 0.95
        else:
            return 0.75
    
    def _is_dead_branch(self, edge: ControlFlowEdge, cfg: ControlFlowGraph) -> bool:
        """Verifica si una rama está muerta."""
        # Por ahora, una implementación simple
        # En una implementación real, se haría análisis más sofisticado
        return False  # Placeholder
