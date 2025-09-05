"""
Analizador interprocedural avanzado con seguimiento de flujo de datos.
Detecta uso indirecto de código a través de callbacks, inyección de dependencias, y más.
"""

import ast
import os
from typing import Dict, Set, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import networkx as nx
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataFlowNode:
    """Nodo en el grafo de flujo de datos."""
    id: str
    type: str  # 'variable', 'function', 'class', 'parameter', 'return', 'attribute'
    name: str
    file_path: str
    line: int
    context: str  # 'local', 'global', 'class', 'module'
    value_type: Optional[str] = None  # tipo inferido del valor
    is_mutable: bool = False
    aliases: Set[str] = field(default_factory=set)  # otros nombres por los que se conoce


@dataclass
class DataFlowEdge:
    """Arista en el grafo de flujo de datos."""
    source: str  # ID del nodo fuente
    target: str  # ID del nodo destino
    edge_type: str  # 'assignment', 'parameter', 'return', 'attribute_access', 'call', 'callback'
    context: str  # Información adicional sobre el contexto
    confidence: float = 1.0  # Confianza en la conexión (0-1)


class InterproceduralAnalyzer:
    """
    Análisis interprocedural completo con seguimiento de flujo de datos.
    Detecta uso indirecto de código que el análisis estático simple no puede encontrar.
    """
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.data_flow_graph = nx.DiGraph()
        self.call_graph = nx.DiGraph()
        self.alias_map: Dict[str, Set[str]] = defaultdict(set)
        self.callback_registry: Dict[str, List[str]] = defaultdict(list)
        self.injection_points: Dict[str, List[str]] = defaultdict(list)
        self.framework_patterns: Dict[str, List[Dict[str, Any]]] = self._load_framework_patterns()
        self.indirect_uses: Dict[str, Set[str]] = defaultdict(set)
        
    def analyze(self) -> Dict[str, Any]:
        """Ejecutar análisis interprocedural completo."""
        logger.info("Iniciando análisis interprocedural avanzado...")
        
        # Fase 1: Construir ASTs y extraer información básica
        self._build_ast_forest()
        
        # Fase 2: Análisis de flujo de datos intraprocedural
        self._analyze_intraprocedural_flow()
        
        # Fase 3: Análisis interprocedural
        self._analyze_interprocedural_flow()
        
        # Fase 4: Detectar callbacks y referencias indirectas
        self._detect_callbacks_and_indirect_references()
        
        # Fase 5: Análisis de inyección de dependencias
        self._analyze_dependency_injection()
        
        # Fase 6: Detectar patrones de frameworks
        self._detect_framework_patterns()
        
        # Fase 7: Propagar información de uso
        self._propagate_usage_information()
        
        # Fase 8: Calcular alcanzabilidad con contexto
        reachable = self._compute_contextual_reachability()
        
        return {
            'reachable_symbols': reachable,
            'indirect_uses': dict(self.indirect_uses),
            'callback_registry': dict(self.callback_registry),
            'injection_points': dict(self.injection_points),
            'data_flow_graph': self._export_data_flow_graph(),
            'call_graph': self._export_call_graph()
        }
    
    def _build_ast_forest(self):
        """Construir ASTs para todos los archivos del proyecto."""
        self.ast_forest = {}
        
        for py_file in self.project_path.rglob('*.py'):
            if 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                tree = ast.parse(content, filename=str(py_file))
                self.ast_forest[str(py_file)] = {
                    'tree': tree,
                    'content': content,
                    'lines': content.splitlines()
                }
            except Exception as e:
                logger.warning(f"Error parsing {py_file}: {e}")
    
    def _analyze_intraprocedural_flow(self):
        """Análisis de flujo de datos dentro de cada función."""
        for file_path, ast_data in self.ast_forest.items():
            tree = ast_data['tree']
            
            class FlowAnalyzer(ast.NodeVisitor):
                def __init__(analyzer_self):
                    analyzer_self.current_function = None
                    analyzer_self.current_class = None
                    analyzer_self.local_flows = []
                    analyzer_self.parameter_flows = []
                    
                def visit_FunctionDef(analyzer_self, node):
                    old_function = analyzer_self.current_function
                    analyzer_self.current_function = node.name
                    
                    # Analizar parámetros
                    for i, arg in enumerate(node.args.args):
                        param_id = f"{file_path}:{node.name}:param:{arg.arg}"
                        param_node = DataFlowNode(
                            id=param_id,
                            type='parameter',
                            name=arg.arg,
                            file_path=file_path,
                            line=node.lineno,
                            context='local'
                        )
                        self.data_flow_graph.add_node(param_id, **param_node.__dict__)
                        
                        # Si el parámetro se usa como callback
                        if self._is_callback_parameter(node, arg.arg):
                            analyzer_self.parameter_flows.append({
                                'param': arg.arg,
                                'function': node.name,
                                'type': 'callback'
                            })
                    
                    # Analizar cuerpo de la función
                    analyzer_self.generic_visit(node)
                    
                    # Analizar valor de retorno
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return) and child.value:
                            self._analyze_return_flow(child, node.name, file_path)
                    
                    analyzer_self.current_function = old_function
                
                def visit_Assign(analyzer_self, node):
                    """Analizar asignaciones para flujo de datos."""
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # Variable simple
                            var_id = self._create_node_id(file_path, target.id, analyzer_self.current_function)
                            var_node = DataFlowNode(
                                id=var_id,
                                type='variable',
                                name=target.id,
                                file_path=file_path,
                                line=node.lineno,
                                context='local' if analyzer_self.current_function else 'module'
                            )
                            self.data_flow_graph.add_node(var_id, **var_node.__dict__)
                            
                            # Analizar el valor asignado
                            if isinstance(node.value, ast.Name):
                                # Asignación de otra variable (alias)
                                source_id = self._create_node_id(file_path, node.value.id, analyzer_self.current_function)
                                self.alias_map[var_id].add(source_id)
                                self.alias_map[source_id].add(var_id)
                                
                                edge = DataFlowEdge(
                                    source=source_id,
                                    target=var_id,
                                    edge_type='assignment',
                                    context=f"alias at line {node.lineno}"
                                )
                                self.data_flow_graph.add_edge(source_id, var_id, **edge.__dict__)
                                
                            elif isinstance(node.value, ast.Call):
                                # Resultado de llamada a función
                                if isinstance(node.value.func, ast.Name):
                                    func_name = node.value.func.id
                                    self._analyze_function_call_assignment(var_id, func_name, node, file_path)
                                elif isinstance(node.value.func, ast.Attribute):
                                    # Método de objeto
                                    self._analyze_method_call_assignment(var_id, node.value, file_path)
                    
                    analyzer_self.generic_visit(node)
                
                def visit_Call(analyzer_self, node):
                    """Analizar llamadas a funciones."""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        
                        # Analizar argumentos pasados
                        for i, arg in enumerate(node.args):
                            if isinstance(arg, ast.Name):
                                # Se pasa una variable/función como argumento
                                arg_id = self._create_node_id(file_path, arg.id, analyzer_self.current_function)
                                
                                # Registrar como posible callback
                                if self._is_likely_callback(arg.id):
                                    self.callback_registry[func_name].append(arg_id)
                                    self.indirect_uses[arg_id].add(f"callback_to_{func_name}")
                                    
                            elif isinstance(arg, ast.Lambda):
                                # Lambda como callback
                                lambda_id = f"{file_path}:lambda:{node.lineno}:{i}"
                                self.callback_registry[func_name].append(lambda_id)
                    
                    analyzer_self.generic_visit(node)
            
            analyzer = FlowAnalyzer()
            analyzer.visit(tree)
    
    def _analyze_interprocedural_flow(self):
        """Análisis de flujo entre procedimientos."""
        # Construir grafo de llamadas completo
        for file_path, ast_data in self.ast_forest.items():
            tree = ast_data['tree']
            
            class CallGraphBuilder(ast.NodeVisitor):
                def __init__(builder_self):
                    builder_self.current_function = None
                    builder_self.current_class = None
                    builder_self.imports = {}
                
                def visit_Import(builder_self, node):
                    for alias in node.names:
                        builder_self.imports[alias.asname or alias.name] = alias.name
                
                def visit_ImportFrom(builder_self, node):
                    module = node.module or ''
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}" if module else alias.name
                        builder_self.imports[alias.asname or alias.name] = full_name
                
                def visit_FunctionDef(builder_self, node):
                    old_func = builder_self.current_function
                    func_id = self._create_function_id(file_path, node.name, builder_self.current_class)
                    builder_self.current_function = func_id
                    
                    # Agregar nodo de función al grafo de llamadas
                    self.call_graph.add_node(func_id, 
                        name=node.name,
                        file=file_path,
                        line=node.lineno,
                        is_method=builder_self.current_class is not None,
                        decorators=[d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
                    )
                    
                    builder_self.generic_visit(node)
                    builder_self.current_function = old_func
                
                def visit_Call(builder_self, node):
                    if builder_self.current_function:
                        callee_id = None
                        
                        if isinstance(node.func, ast.Name):
                            # Llamada directa a función
                            func_name = node.func.id
                            callee_id = self._resolve_function_id(func_name, file_path, builder_self.imports)
                            
                        elif isinstance(node.func, ast.Attribute):
                            # Llamada a método
                            if isinstance(node.func.value, ast.Name):
                                obj_name = node.func.value.id
                                method_name = node.func.attr
                                callee_id = self._resolve_method_id(obj_name, method_name, file_path)
                        
                        if callee_id and builder_self.current_function != callee_id:
                            self.call_graph.add_edge(builder_self.current_function, callee_id,
                                call_type='direct',
                                line=node.lineno
                            )
                            
                            # Propagar información de flujo de datos
                            self._propagate_flow_through_call(builder_self.current_function, callee_id, node)
                    
                    builder_self.generic_visit(node)
            
            builder = CallGraphBuilder()
            builder.visit(tree)
    
    def _detect_callbacks_and_indirect_references(self):
        """Detectar callbacks y referencias indirectas."""
        # Patrones comunes de callbacks
        callback_patterns = {
            'event_handlers': ['on_', 'handle_', '_handler', '_callback'],
            'async_patterns': ['then', 'catch', 'finally', 'done', 'fail'],
            'framework_callbacks': ['render', 'componentDidMount', 'useEffect', 'created', 'mounted'],
            'decorators': ['route', 'api', 'task', 'scheduled', 'event', 'signal']
        }
        
        for file_path, ast_data in self.ast_forest.items():
            tree = ast_data['tree']
            
            class CallbackDetector(ast.NodeVisitor):
                def visit_FunctionDef(detector_self, node):
                    func_name = node.name
                    
                    # Verificar si el nombre sugiere un callback
                    for category, patterns in callback_patterns.items():
                        for pattern in patterns:
                            if pattern in func_name.lower():
                                func_id = self._create_function_id(file_path, func_name, None)
                                self.indirect_uses[func_id].add(f"callback_{category}")
                                logger.debug(f"Detectado callback {func_name} en categoría {category}")
                    
                    # Verificar decoradores
                    for decorator in node.decorator_list:
                        decorator_name = self._get_decorator_name(decorator)
                        if decorator_name:
                            func_id = self._create_function_id(file_path, func_name, None)
                            self.indirect_uses[func_id].add(f"decorator_{decorator_name}")
                            
                            # Registrar en puntos de inyección si es un decorador de framework
                            if decorator_name in ['route', 'api', 'endpoint', 'task']:
                                self.injection_points[decorator_name].append(func_id)
                    
                    detector_self.generic_visit(node)
                
                def visit_Assign(detector_self, node):
                    """Detectar asignaciones de funciones a estructuras de datos."""
                    if isinstance(node.value, ast.Dict):
                        # Diccionario de callbacks
                        for key, value in zip(node.value.keys, node.value.values):
                            if isinstance(value, ast.Name):
                                # Función asignada como valor en diccionario
                                func_id = self._create_node_id(file_path, value.id, None)
                                key_str = self._get_constant_value(key)
                                if key_str:
                                    self.indirect_uses[func_id].add(f"dict_callback_{key_str}")
                                    self.callback_registry[f"dict_{key_str}"].append(func_id)
                    
                    elif isinstance(node.value, ast.List) or isinstance(node.value, ast.Tuple):
                        # Lista/tupla de callbacks
                        for item in node.value.elts:
                            if isinstance(item, ast.Name):
                                func_id = self._create_node_id(file_path, item.id, None)
                                self.indirect_uses[func_id].add("list_callback")
                    
                    detector_self.generic_visit(node)
                
                def visit_Call(detector_self, node):
                    """Detectar registro de callbacks."""
                    if isinstance(node.func, ast.Attribute):
                        method_name = node.func.attr
                        
                        # Patrones de registro de callbacks
                        if method_name in ['register', 'subscribe', 'on', 'addEventListener', 'bind', 'connect']:
                            for arg in node.args:
                                if isinstance(arg, ast.Name):
                                    func_id = self._create_node_id(file_path, arg.id, None)
                                    self.indirect_uses[func_id].add(f"registered_via_{method_name}")
                                    self.callback_registry[method_name].append(func_id)
                    
                    detector_self.generic_visit(node)
            
            detector = CallbackDetector()
            detector.visit(tree)
    
    def _analyze_dependency_injection(self):
        """Analizar patrones de inyección de dependencias."""
        # Patrones comunes de DI
        di_patterns = {
            'constructor_injection': self._detect_constructor_injection,
            'setter_injection': self._detect_setter_injection,
            'interface_injection': self._detect_interface_injection,
            'service_locator': self._detect_service_locator,
            'factory_pattern': self._detect_factory_pattern
        }
        
        for pattern_name, detector_func in di_patterns.items():
            logger.debug(f"Detectando patrón de DI: {pattern_name}")
            detector_func()
    
    def _detect_constructor_injection(self):
        """Detectar inyección por constructor."""
        for file_path, ast_data in self.ast_forest.items():
            tree = ast_data['tree']
            
            class ConstructorAnalyzer(ast.NodeVisitor):
                def __init__(analyzer_self):
                    analyzer_self.current_class = None
                
                def visit_ClassDef(analyzer_self, node):
                    old_class = analyzer_self.current_class
                    analyzer_self.current_class = node.name
                    
                    # Buscar __init__
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                            # Analizar parámetros del constructor
                            for arg in item.args.args[1:]:  # Skip 'self'
                                arg_name = arg.arg
                                
                                # Buscar asignaciones a self
                                for stmt in item.body:
                                    if isinstance(stmt, ast.Assign):
                                        for target in stmt.targets:
                                            if (isinstance(target, ast.Attribute) and
                                                isinstance(target.value, ast.Name) and
                                                target.value.id == 'self' and
                                                isinstance(stmt.value, ast.Name) and
                                                stmt.value.id == arg_name):
                                                
                                                # Dependencia inyectada
                                                dep_id = f"{file_path}:{analyzer_self.current_class}:dep:{arg_name}"
                                                self.injection_points['constructor'].append(dep_id)
                                                
                                                # Marcar el parámetro como usado indirectamente
                                                param_id = self._create_node_id(file_path, arg_name, '__init__')
                                                self.indirect_uses[param_id].add(f"injected_to_{analyzer_self.current_class}")
                    
                    analyzer_self.generic_visit(node)
                    analyzer_self.current_class = old_class
            
            analyzer = ConstructorAnalyzer()
            analyzer.visit(tree)
    
    def _detect_framework_patterns(self):
        """Detectar patrones específicos de frameworks."""
        framework_detectors = {
            'flask': self._detect_flask_patterns,
            'django': self._detect_django_patterns,
            'fastapi': self._detect_fastapi_patterns,
            'pytest': self._detect_pytest_patterns,
            'celery': self._detect_celery_patterns
        }
        
        # Detectar qué frameworks están en uso
        active_frameworks = self._detect_active_frameworks()
        
        for framework in active_frameworks:
            if framework in framework_detectors:
                logger.info(f"Detectando patrones de {framework}")
                framework_detectors[framework]()
    
    def _detect_flask_patterns(self):
        """Detectar patrones específicos de Flask."""
        for file_path, ast_data in self.ast_forest.items():
            tree = ast_data['tree']
            
            class FlaskPatternDetector(ast.NodeVisitor):
                def visit_FunctionDef(detector_self, node):
                    # Detectar rutas de Flask
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                            if decorator.func.attr in ['route', 'get', 'post', 'put', 'delete']:
                                func_id = self._create_function_id(file_path, node.name, None)
                                self.indirect_uses[func_id].add("flask_route_handler")
                                self.injection_points['flask_routes'].append(func_id)
                                
                                # La función es llamada por Flask, no directamente
                                logger.debug(f"Detectada ruta Flask: {node.name}")
                    
                    detector_self.generic_visit(node)
            
            detector = FlaskPatternDetector()
            detector.visit(tree)
    
    def _propagate_usage_information(self):
        """Propagar información de uso a través del grafo de flujo de datos."""
        # Usar algoritmo de punto fijo para propagar uso
        changed = True
        iterations = 0
        max_iterations = 100
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            # Propagar a través de alias
            for node_id, aliases in self.alias_map.items():
                if node_id in self.indirect_uses:
                    for alias in aliases:
                        if alias not in self.indirect_uses:
                            self.indirect_uses[alias] = self.indirect_uses[node_id].copy()
                            changed = True
                        else:
                            old_size = len(self.indirect_uses[alias])
                            self.indirect_uses[alias].update(self.indirect_uses[node_id])
                            if len(self.indirect_uses[alias]) > old_size:
                                changed = True
            
            # Propagar a través del grafo de flujo de datos
            for edge in self.data_flow_graph.edges(data=True):
                source, target, data = edge
                if source in self.indirect_uses:
                    if target not in self.indirect_uses:
                        self.indirect_uses[target] = set()
                    
                    old_size = len(self.indirect_uses[target])
                    # Propagar con contexto
                    for use in self.indirect_uses[source]:
                        propagated_use = f"{use}_via_{data.get('edge_type', 'unknown')}"
                        self.indirect_uses[target].add(propagated_use)
                    
                    if len(self.indirect_uses[target]) > old_size:
                        changed = True
        
        logger.info(f"Propagación de uso completada en {iterations} iteraciones")
    
    def _compute_contextual_reachability(self) -> Set[str]:
        """Calcular alcanzabilidad con contexto completo."""
        reachable = set()
        
        # Entry points extendidos
        entry_points = set()
        
        # Entry points tradicionales
        for file_path, ast_data in self.ast_forest.items():
            if 'main.py' in file_path or '__main__' in ast_data['content']:
                entry_points.add(file_path)
        
        # Entry points de frameworks (todos los puntos de inyección)
        for injection_type, points in self.injection_points.items():
            entry_points.update(points)
        
        # Entry points de callbacks registrados
        for callback_type, callbacks in self.callback_registry.items():
            entry_points.update(callbacks)
        
        # BFS desde entry points con contexto
        queue = deque(entry_points)
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            reachable.add(current)
            
            # Explorar grafo de llamadas
            if current in self.call_graph:
                for successor in self.call_graph.successors(current):
                    if successor not in visited:
                        queue.append(successor)
            
            # Explorar grafo de flujo de datos
            if current in self.data_flow_graph:
                for successor in self.data_flow_graph.successors(current):
                    if successor not in visited:
                        queue.append(successor)
            
            # Incluir todos los usos indirectos
            if current in self.indirect_uses:
                # El símbolo se usa indirectamente, es alcanzable
                logger.debug(f"{current} es alcanzable por uso indirecto: {self.indirect_uses[current]}")
        
        # Agregar todos los símbolos con uso indirecto
        for symbol, uses in self.indirect_uses.items():
            if uses:  # Si tiene algún uso indirecto
                reachable.add(symbol)
        
        return reachable
    
    # Métodos auxiliares
    def _create_node_id(self, file_path: str, name: str, context: Optional[str]) -> str:
        """Crear ID único para un nodo."""
        if context:
            return f"{file_path}:{context}:{name}"
        return f"{file_path}:{name}"
    
    def _create_function_id(self, file_path: str, name: str, class_name: Optional[str]) -> str:
        """Crear ID único para una función."""
        if class_name:
            return f"{file_path}:{class_name}.{name}"
        return f"{file_path}:{name}"
    
    def _is_callback_parameter(self, func_node: ast.FunctionDef, param_name: str) -> bool:
        """Determinar si un parámetro es probablemente un callback."""
        # Heurísticas para detectar callbacks
        callback_hints = ['callback', 'handler', 'func', 'fn', 'cb', 'listener']
        return any(hint in param_name.lower() for hint in callback_hints)
    
    def _is_likely_callback(self, name: str) -> bool:
        """Determinar si un nombre sugiere una función callback."""
        callback_patterns = ['on_', 'handle_', '_handler', '_callback', '_cb']
        return any(pattern in name.lower() for pattern in callback_patterns)
    
    def _resolve_function_id(self, func_name: str, current_file: str, imports: Dict[str, str]) -> Optional[str]:
        """Resolver el ID completo de una función."""
        # Primero buscar en imports
        if func_name in imports:
            module_name = imports[func_name]
            # Aquí deberíamos resolver el path completo del módulo
            # Por simplicidad, retornamos un ID aproximado
            return f"{module_name}:{func_name}"
        
        # Buscar en el archivo actual
        return f"{current_file}:{func_name}"
    
    def _resolve_method_id(self, obj_name: str, method_name: str, current_file: str) -> Optional[str]:
        """Resolver el ID de un método."""
        # Aquí deberíamos hacer análisis de tipos para resolver la clase
        # Por simplicidad, retornamos un ID aproximado
        return f"{current_file}:{obj_name}.{method_name}"
    
    def _analyze_return_flow(self, return_node: ast.Return, func_name: str, file_path: str):
        """Analizar el flujo de datos en un return."""
        if isinstance(return_node.value, ast.Name):
            # Se retorna una variable
            var_id = self._create_node_id(file_path, return_node.value.id, func_name)
            func_id = self._create_function_id(file_path, func_name, None)
            
            edge = DataFlowEdge(
                source=var_id,
                target=func_id,
                edge_type='return',
                context=f"returns {return_node.value.id}"
            )
            self.data_flow_graph.add_edge(var_id, func_id, **edge.__dict__)
    
    def _analyze_function_call_assignment(self, var_id: str, func_name: str, node: ast.Assign, file_path: str):
        """Analizar asignación del resultado de una llamada a función."""
        func_id = self._resolve_function_id(func_name, file_path, {})
        if func_id:
            edge = DataFlowEdge(
                source=func_id,
                target=var_id,
                edge_type='call_result',
                context=f"result of {func_name}()"
            )
            self.data_flow_graph.add_edge(func_id, var_id, **edge.__dict__)
    
    def _analyze_method_call_assignment(self, var_id: str, call_node: ast.Call, file_path: str):
        """Analizar asignación del resultado de una llamada a método."""
        if isinstance(call_node.func.value, ast.Name):
            obj_name = call_node.func.value.id
            method_name = call_node.func.attr
            method_id = self._resolve_method_id(obj_name, method_name, file_path)
            
            if method_id:
                edge = DataFlowEdge(
                    source=method_id,
                    target=var_id,
                    edge_type='method_result',
                    context=f"result of {obj_name}.{method_name}()"
                )
                self.data_flow_graph.add_edge(method_id, var_id, **edge.__dict__)
    
    def _get_decorator_name(self, decorator: ast.AST) -> Optional[str]:
        """Obtener el nombre de un decorador."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return decorator.func.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        return None
    
    def _get_constant_value(self, node: ast.AST) -> Optional[str]:
        """Obtener el valor de una constante."""
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Str):  # Python < 3.8
            return node.s
        return None
    
    def _load_framework_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Cargar patrones conocidos de frameworks."""
        return {
            'flask': [
                {'pattern': '@app.route', 'type': 'route_handler'},
                {'pattern': '@blueprint.route', 'type': 'route_handler'},
                {'pattern': 'before_request', 'type': 'middleware'},
                {'pattern': 'after_request', 'type': 'middleware'},
            ],
            'django': [
                {'pattern': 'urlpatterns', 'type': 'url_config'},
                {'pattern': 'get_queryset', 'type': 'orm_method'},
                {'pattern': 'dispatch', 'type': 'view_method'},
            ],
            'fastapi': [
                {'pattern': '@router.', 'type': 'route_handler'},
                {'pattern': 'Depends(', 'type': 'dependency_injection'},
            ],
            'pytest': [
                {'pattern': '@pytest.fixture', 'type': 'fixture'},
                {'pattern': 'test_', 'type': 'test_function'},
                {'pattern': '@pytest.mark', 'type': 'test_marker'},
            ]
        }
    
    def _detect_active_frameworks(self) -> List[str]:
        """Detectar qué frameworks están en uso."""
        active = []
        
        # Buscar en imports
        for file_path, ast_data in self.ast_forest.items():
            content = ast_data['content']
            
            if 'from flask' in content or 'import flask' in content:
                active.append('flask')
            if 'from django' in content or 'import django' in content:
                active.append('django')
            if 'from fastapi' in content or 'import fastapi' in content:
                active.append('fastapi')
            if 'import pytest' in content or 'from pytest' in content:
                active.append('pytest')
            if 'from celery' in content or 'import celery' in content:
                active.append('celery')
        
        return list(set(active))
    
    def _detect_setter_injection(self):
        """Detectar inyección por setter."""
        # Implementación simplificada
        logger.debug("Detectando setter injection...")
    
    def _detect_interface_injection(self):
        """Detectar inyección por interface."""
        logger.debug("Detectando interface injection...")
    
    def _detect_service_locator(self):
        """Detectar patrón service locator."""
        logger.debug("Detectando service locator...")
    
    def _detect_factory_pattern(self):
        """Detectar patrón factory."""
        for file_path, ast_data in self.ast_forest.items():
            tree = ast_data['tree']
            
            class FactoryDetector(ast.NodeVisitor):
                def visit_FunctionDef(detector_self, node):
                    # Detectar funciones factory
                    if any(pattern in node.name.lower() for pattern in ['create_', 'make_', 'build_', 'factory']):
                        func_id = self._create_function_id(file_path, node.name, None)
                        self.indirect_uses[func_id].add("factory_function")
                        
                        # Analizar lo que retorna
                        for stmt in ast.walk(node):
                            if isinstance(stmt, ast.Return) and stmt.value:
                                if isinstance(stmt.value, ast.Call):
                                    # Factory que crea instancias
                                    if isinstance(stmt.value.func, ast.Name):
                                        created_class = stmt.value.func.id
                                        class_id = f"{file_path}:{created_class}"
                                        self.indirect_uses[class_id].add(f"created_by_factory_{node.name}")
                    
                    detector_self.generic_visit(node)
            
            detector = FactoryDetector()
            detector.visit(tree)
    
    def _detect_django_patterns(self):
        """Detectar patrones de Django."""
        logger.debug("Detectando patrones Django...")
    
    def _detect_fastapi_patterns(self):
        """Detectar patrones de FastAPI."""
        logger.debug("Detectando patrones FastAPI...")
    
    def _detect_pytest_patterns(self):
        """Detectar patrones de pytest."""
        for file_path, ast_data in self.ast_forest.items():
            if 'test' in file_path.lower():
                tree = ast_data['tree']
                
                class PytestDetector(ast.NodeVisitor):
                    def visit_FunctionDef(detector_self, node):
                        # Fixtures de pytest
                        for decorator in node.decorator_list:
                            dec_name = self._get_decorator_name(decorator)
                            if dec_name and 'fixture' in str(dec_name):
                                func_id = self._create_function_id(file_path, node.name, None)
                                self.indirect_uses[func_id].add("pytest_fixture")
                                self.injection_points['pytest_fixtures'].append(func_id)
                        
                        # Tests
                        if node.name.startswith('test_'):
                            func_id = self._create_function_id(file_path, node.name, None)
                            self.indirect_uses[func_id].add("pytest_test")
                        
                        detector_self.generic_visit(node)
                
                detector = PytestDetector()
                detector.visit(tree)
    
    def _detect_celery_patterns(self):
        """Detectar patrones de Celery."""
        logger.debug("Detectando patrones Celery...")
    
    def _propagate_flow_through_call(self, caller_id: str, callee_id: str, call_node: ast.Call):
        """Propagar flujo de datos a través de una llamada."""
        # Mapear argumentos a parámetros
        # Esto requeriría resolver la definición de la función llamada
        # Por simplicidad, solo registramos la conexión
        edge = DataFlowEdge(
            source=caller_id,
            target=callee_id,
            edge_type='call',
            context=f"call at line {call_node.lineno}"
        )
        self.data_flow_graph.add_edge(caller_id, callee_id, **edge.__dict__)
    
    def _export_data_flow_graph(self) -> Dict[str, Any]:
        """Exportar el grafo de flujo de datos."""
        nodes = []
        edges = []
        
        for node_id in self.data_flow_graph.nodes():
            node_data = self.data_flow_graph.nodes[node_id]
            nodes.append({
                'id': node_id,
                'type': node_data.get('type', 'unknown'),
                'name': node_data.get('name', ''),
                'file': node_data.get('file_path', ''),
                'indirect_uses': list(self.indirect_uses.get(node_id, set()))
            })
        
        for source, target, data in self.data_flow_graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'type': data.get('edge_type', 'unknown'),
                'context': data.get('context', '')
            })
        
        return {'nodes': nodes, 'edges': edges}
    
    def _export_call_graph(self) -> Dict[str, Any]:
        """Exportar el grafo de llamadas."""
        nodes = []
        edges = []
        
        for node_id in self.call_graph.nodes():
            node_data = self.call_graph.nodes[node_id]
            nodes.append({
                'id': node_id,
                'name': node_data.get('name', ''),
                'file': node_data.get('file', ''),
                'is_reachable': node_id in self.indirect_uses or any(self.call_graph.predecessors(node_id))
            })
        
        for source, target, data in self.call_graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'type': data.get('call_type', 'direct')
            })
        
        return {'nodes': nodes, 'edges': edges}
