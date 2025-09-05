"""
Analizador interprocedural para JavaScript/TypeScript con seguimiento de flujo de datos.
Maneja las peculiaridades de JS como prototipos, closures, y async/await.
"""

import re
import json
from typing import Dict, Set, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import networkx as nx
import logging

logger = logging.getLogger(__name__)


@dataclass
class JSDataFlowNode:
    """Nodo en el grafo de flujo de datos para JavaScript."""
    id: str
    type: str  # 'function', 'class', 'variable', 'method', 'arrow_function', 'async_function'
    name: str
    file_path: str
    line: int
    is_exported: bool = False
    is_default_export: bool = False
    module_type: str = 'commonjs'  # 'commonjs' o 'esm'
    is_async: bool = False
    is_generator: bool = False
    is_constructor: bool = False
    closure_variables: Set[str] = field(default_factory=set)


class InterproceduralJSAnalyzer:
    """
    Análisis interprocedural para JavaScript/TypeScript.
    Maneja características específicas de JS como closures, prototipos, y módulos.
    """
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.data_flow_graph = nx.DiGraph()
        self.call_graph = nx.DiGraph()
        self.module_graph = nx.DiGraph()  # Grafo de dependencias entre módulos
        self.symbols: Dict[str, JSDataFlowNode] = {}
        self.exports: Dict[str, Dict[str, str]] = defaultdict(dict)  # file -> {export_name: symbol_id}
        self.imports: Dict[str, Dict[str, str]] = defaultdict(dict)  # file -> {import_name: source_module}
        self.indirect_uses: Dict[str, Set[str]] = defaultdict(set)
        self.promise_chains: Dict[str, List[str]] = defaultdict(list)
        self.event_listeners: Dict[str, List[str]] = defaultdict(list)
        self.react_components: Set[str] = set()
        self.vue_components: Set[str] = set()
        
    def analyze(self) -> Dict[str, Any]:
        """Ejecutar análisis interprocedural para JavaScript/TypeScript."""
        logger.info("Iniciando análisis interprocedural para JavaScript/TypeScript...")
        
        # Fase 1: Análisis léxico y construcción de símbolos
        self._analyze_all_files()
        
        # Fase 2: Resolución de módulos e imports/exports
        self._resolve_modules()
        
        # Fase 3: Análisis de closures y scope
        self._analyze_closures_and_scopes()
        
        # Fase 4: Detección de callbacks y promises
        self._detect_callbacks_and_promises()
        
        # Fase 5: Análisis de event listeners
        self._analyze_event_listeners()
        
        # Fase 6: Detección de componentes de frameworks
        self._detect_framework_components()
        
        # Fase 7: Análisis de flujo asíncrono
        self._analyze_async_flow()
        
        # Fase 8: Construcción del grafo de llamadas
        self._build_call_graph()
        
        # Fase 9: Propagación de uso
        self._propagate_usage()
        
        # Fase 10: Cálculo de alcanzabilidad
        reachable = self._compute_reachability()
        
        return {
            'reachable_symbols': reachable,
            'indirect_uses': dict(self.indirect_uses),
            'exports': dict(self.exports),
            'imports': dict(self.imports),
            'promise_chains': dict(self.promise_chains),
            'event_listeners': dict(self.event_listeners),
            'react_components': list(self.react_components),
            'vue_components': list(self.vue_components)
        }
    
    def _analyze_all_files(self):
        """Analizar todos los archivos JS/TS del proyecto."""
        extensions = ['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs']
        
        for ext in extensions:
            for file_path in self.project_path.rglob(f'*{ext}'):
                if self._should_skip_file(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    self._analyze_file(str(file_path), content)
                except Exception as e:
                    logger.warning(f"Error analizando {file_path}: {e}")
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Determinar si un archivo debe ser omitido."""
        skip_dirs = ['node_modules', 'dist', 'build', '.next', '.nuxt', 'coverage']
        return any(skip_dir in str(file_path) for skip_dir in skip_dirs)
    
    def _analyze_file(self, file_path: str, content: str):
        """Analizar un archivo JavaScript/TypeScript."""
        # Detectar tipo de módulo
        module_type = 'esm' if 'export ' in content or 'import ' in content else 'commonjs'
        
        # Extraer funciones
        self._extract_functions(file_path, content, module_type)
        
        # Extraer clases
        self._extract_classes(file_path, content, module_type)
        
        # Extraer exports
        self._extract_exports(file_path, content, module_type)
        
        # Extraer imports
        self._extract_imports(file_path, content)
        
        # Extraer variables globales/módulo
        self._extract_module_variables(file_path, content)
    
    def _extract_functions(self, file_path: str, content: str, module_type: str):
        """Extraer definiciones de funciones."""
        # Funciones tradicionales
        func_pattern = re.compile(
            r'(?:export\s+)?(?:async\s+)?function\s*(\*?)\s*(\w+)\s*\([^)]*\)',
            re.MULTILINE
        )
        
        for match in func_pattern.finditer(content):
            is_generator = match.group(1) == '*'
            func_name = match.group(2)
            is_exported = 'export' in content[max(0, match.start()-20):match.start()]
            is_async = 'async' in content[max(0, match.start()-20):match.start()]
            
            symbol_id = f"{file_path}:function:{func_name}"
            line = content[:match.start()].count('\n') + 1
            
            node = JSDataFlowNode(
                id=symbol_id,
                type='function',
                name=func_name,
                file_path=file_path,
                line=line,
                is_exported=is_exported,
                module_type=module_type,
                is_async=is_async,
                is_generator=is_generator
            )
            self.symbols[symbol_id] = node
            self.data_flow_graph.add_node(symbol_id, **node.__dict__)
            
            if is_exported:
                self.exports[file_path][func_name] = symbol_id
        
        # Arrow functions asignadas a variables
        arrow_pattern = re.compile(
            r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>'
        )
        
        for match in arrow_pattern.finditer(content):
            func_name = match.group(1)
            is_exported = 'export' in content[max(0, match.start()-20):match.start()]
            is_async = 'async' in content[match.start():match.end()]
            
            symbol_id = f"{file_path}:arrow:{func_name}"
            line = content[:match.start()].count('\n') + 1
            
            node = JSDataFlowNode(
                id=symbol_id,
                type='arrow_function',
                name=func_name,
                file_path=file_path,
                line=line,
                is_exported=is_exported,
                module_type=module_type,
                is_async=is_async
            )
            self.symbols[symbol_id] = node
            self.data_flow_graph.add_node(symbol_id, **node.__dict__)
            
            if is_exported:
                self.exports[file_path][func_name] = symbol_id
    
    def _extract_classes(self, file_path: str, content: str, module_type: str):
        """Extraer definiciones de clases."""
        class_pattern = re.compile(
            r'(?:export\s+)?(?:default\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?',
            re.MULTILINE
        )
        
        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            extends_class = match.group(2)
            is_exported = 'export' in content[max(0, match.start()-20):match.start()]
            is_default = 'default' in content[max(0, match.start()-20):match.start()]
            
            symbol_id = f"{file_path}:class:{class_name}"
            line = content[:match.start()].count('\n') + 1
            
            node = JSDataFlowNode(
                id=symbol_id,
                type='class',
                name=class_name,
                file_path=file_path,
                line=line,
                is_exported=is_exported,
                is_default_export=is_default,
                module_type=module_type
            )
            self.symbols[symbol_id] = node
            self.data_flow_graph.add_node(symbol_id, **node.__dict__)
            
            if is_exported:
                export_name = 'default' if is_default else class_name
                self.exports[file_path][export_name] = symbol_id
            
            # Analizar métodos de la clase
            self._extract_class_methods(file_path, content, class_name, match.end())
    
    def _extract_class_methods(self, file_path: str, content: str, class_name: str, start_pos: int):
        """Extraer métodos de una clase."""
        # Buscar el cierre de la clase
        brace_count = 0
        in_class = False
        class_content_start = content.find('{', start_pos)
        
        if class_content_start == -1:
            return
        
        i = class_content_start
        while i < len(content):
            if content[i] == '{':
                brace_count += 1
                in_class = True
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    break
            i += 1
        
        class_content = content[class_content_start:i]
        
        # Extraer métodos
        method_pattern = re.compile(
            r'(?:static\s+)?(?:async\s+)?(?:get\s+|set\s+)?(\w+)\s*\([^)]*\)\s*\{',
            re.MULTILINE
        )
        
        for match in method_pattern.finditer(class_content):
            method_name = match.group(1)
            is_constructor = method_name == 'constructor'
            is_static = 'static' in class_content[max(0, match.start()-10):match.start()]
            is_async = 'async' in class_content[max(0, match.start()-10):match.start()]
            
            symbol_id = f"{file_path}:method:{class_name}.{method_name}"
            line = content[:class_content_start + match.start()].count('\n') + 1
            
            node = JSDataFlowNode(
                id=symbol_id,
                type='method',
                name=f"{class_name}.{method_name}",
                file_path=file_path,
                line=line,
                is_async=is_async,
                is_constructor=is_constructor
            )
            self.symbols[symbol_id] = node
            self.data_flow_graph.add_node(symbol_id, **node.__dict__)
    
    def _extract_exports(self, file_path: str, content: str, module_type: str):
        """Extraer exports del módulo."""
        if module_type == 'esm':
            # Named exports
            named_export_pattern = re.compile(r'export\s*\{\s*([^}]+)\s*\}')
            for match in named_export_pattern.finditer(content):
                exports_str = match.group(1)
                for export in exports_str.split(','):
                    export = export.strip()
                    if ' as ' in export:
                        local, exported = export.split(' as ')
                        self.exports[file_path][exported.strip()] = f"{file_path}:{local.strip()}"
                    else:
                        self.exports[file_path][export] = f"{file_path}:{export}"
            
            # Default export
            default_pattern = re.compile(r'export\s+default\s+(\w+)')
            match = default_pattern.search(content)
            if match:
                exported_name = match.group(1)
                self.exports[file_path]['default'] = f"{file_path}:{exported_name}"
        
        else:  # CommonJS
            # module.exports
            module_exports_pattern = re.compile(r'module\.exports\s*=\s*(\w+)')
            match = module_exports_pattern.search(content)
            if match:
                exported_name = match.group(1)
                self.exports[file_path]['default'] = f"{file_path}:{exported_name}"
            
            # exports.name
            exports_pattern = re.compile(r'exports\.(\w+)\s*=\s*(\w+)')
            for match in exports_pattern.finditer(content):
                export_name = match.group(1)
                value_name = match.group(2)
                self.exports[file_path][export_name] = f"{file_path}:{value_name}"
    
    def _extract_imports(self, file_path: str, content: str):
        """Extraer imports del módulo."""
        # ES6 imports
        import_pattern = re.compile(
            r'import\s+(?:\*\s+as\s+(\w+)|(?:\{([^}]+)\}|(\w+)))\s+from\s+[\'"]([^\'"]+)[\'"]'
        )
        
        for match in import_pattern.finditer(content):
            if match.group(1):  # import * as name
                import_name = match.group(1)
                source = match.group(4)
                self.imports[file_path][import_name] = source
            elif match.group(2):  # import { ... }
                imports_str = match.group(2)
                source = match.group(4)
                for imp in imports_str.split(','):
                    imp = imp.strip()
                    if ' as ' in imp:
                        imported, local = imp.split(' as ')
                        self.imports[file_path][local.strip()] = source
                    else:
                        self.imports[file_path][imp] = source
            elif match.group(3):  # import name
                import_name = match.group(3)
                source = match.group(4)
                self.imports[file_path][import_name] = source
        
        # CommonJS require
        require_pattern = re.compile(r'(?:const|let|var)\s+(\w+)\s*=\s*require\s*\([\'"]([^\'"]+)[\'"]\)')
        for match in require_pattern.finditer(content):
            import_name = match.group(1)
            source = match.group(2)
            self.imports[file_path][import_name] = source
        
        # Destructured require
        destructure_pattern = re.compile(r'(?:const|let|var)\s+\{([^}]+)\}\s*=\s*require\s*\([\'"]([^\'"]+)[\'"]\)')
        for match in destructure_pattern.finditer(content):
            imports_str = match.group(1)
            source = match.group(2)
            for imp in imports_str.split(','):
                imp = imp.strip()
                self.imports[file_path][imp] = source
    
    def _extract_module_variables(self, file_path: str, content: str):
        """Extraer variables a nivel de módulo."""
        # Variables exportadas
        var_pattern = re.compile(
            r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?!function|async|\()',
            re.MULTILINE
        )
        
        for match in var_pattern.finditer(content):
            var_name = match.group(1)
            is_exported = 'export' in content[max(0, match.start()-10):match.start()]
            
            # Verificar que no sea una función
            next_chars = content[match.end():match.end()+50]
            if '=>' in next_chars or 'function' in next_chars:
                continue
            
            symbol_id = f"{file_path}:variable:{var_name}"
            line = content[:match.start()].count('\n') + 1
            
            node = JSDataFlowNode(
                id=symbol_id,
                type='variable',
                name=var_name,
                file_path=file_path,
                line=line,
                is_exported=is_exported
            )
            self.symbols[symbol_id] = node
            
            if is_exported:
                self.exports[file_path][var_name] = symbol_id
    
    def _resolve_modules(self):
        """Resolver dependencias entre módulos."""
        for file_path, imports in self.imports.items():
            for import_name, source in imports.items():
                # Resolver path del módulo
                resolved_path = self._resolve_module_path(file_path, source)
                if resolved_path and resolved_path in self.exports:
                    # Conectar import con export
                    if import_name in self.exports[resolved_path]:
                        exported_symbol = self.exports[resolved_path][import_name]
                        importing_symbol = f"{file_path}:import:{import_name}"
                        
                        self.data_flow_graph.add_edge(
                            exported_symbol,
                            importing_symbol,
                            edge_type='import'
                        )
                        
                        # Marcar el símbolo exportado como usado
                        self.indirect_uses[exported_symbol].add(f"imported_as_{import_name}_in_{Path(file_path).name}")
    
    def _resolve_module_path(self, from_file: str, module_specifier: str) -> Optional[str]:
        """Resolver el path real de un módulo."""
        from_path = Path(from_file).parent
        
        # Módulo relativo
        if module_specifier.startswith('.'):
            possible_paths = [
                from_path / module_specifier,
                from_path / f"{module_specifier}.js",
                from_path / f"{module_specifier}.ts",
                from_path / f"{module_specifier}.jsx",
                from_path / f"{module_specifier}.tsx",
                from_path / module_specifier / "index.js",
                from_path / module_specifier / "index.ts"
            ]
            
            for path in possible_paths:
                if path.exists():
                    return str(path.resolve())
        
        # Módulo de node_modules o alias
        # Simplificado para este ejemplo
        return None
    
    def _analyze_closures_and_scopes(self):
        """Analizar closures y variables capturadas."""
        for file_path, symbols in self.symbols.items():
            if not isinstance(symbols, dict):
                continue
                
            # Detectar funciones dentro de funciones (closures)
            for symbol_id, symbol in symbols.items():
                if symbol.type in ['function', 'arrow_function']:
                    # Buscar variables capturadas
                    # Esto requeriría un análisis más profundo del AST
                    # Por ahora, marcamos funciones internas como posibles closures
                    if symbol.name.count('.') > 1:  # Función anidada
                        parent_func = '.'.join(symbol.name.split('.')[:-1])
                        parent_id = f"{file_path}:function:{parent_func}"
                        if parent_id in self.symbols:
                            self.indirect_uses[symbol_id].add(f"closure_in_{parent_func}")
    
    def _detect_callbacks_and_promises(self):
        """Detectar callbacks y cadenas de promesas."""
        for file_path in self.imports.keys():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Detectar .then(), .catch(), .finally()
                promise_pattern = re.compile(r'\.then\s*\(\s*(\w+)\s*[),]')
                for match in promise_pattern.finditer(content):
                    callback_name = match.group(1)
                    callback_id = f"{file_path}:{callback_name}"
                    if callback_id in self.symbols:
                        self.indirect_uses[callback_id].add('promise_then_callback')
                        self.promise_chains[file_path].append(callback_id)
                
                # Detectar callbacks en funciones comunes
                callback_patterns = [
                    (r'setTimeout\s*\(\s*(\w+)', 'timeout_callback'),
                    (r'setInterval\s*\(\s*(\w+)', 'interval_callback'),
                    (r'\.forEach\s*\(\s*(\w+)', 'array_foreach_callback'),
                    (r'\.map\s*\(\s*(\w+)', 'array_map_callback'),
                    (r'\.filter\s*\(\s*(\w+)', 'array_filter_callback'),
                    (r'\.reduce\s*\(\s*(\w+)', 'array_reduce_callback'),
                    (r'\.addEventListener\s*\(\s*[\'"][^\'"]]+[\'"]\s*,\s*(\w+)', 'event_listener'),
                ]
                
                for pattern, callback_type in callback_patterns:
                    regex = re.compile(pattern)
                    for match in regex.finditer(content):
                        callback_name = match.group(1)
                        callback_id = f"{file_path}:{callback_name}"
                        if callback_id in self.symbols:
                            self.indirect_uses[callback_id].add(callback_type)
                
            except Exception as e:
                logger.warning(f"Error detectando callbacks en {file_path}: {e}")
    
    def _analyze_event_listeners(self):
        """Analizar event listeners del DOM y Node.js."""
        for file_path in self.imports.keys():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Event listeners del DOM
                dom_events = ['click', 'change', 'submit', 'load', 'resize', 'scroll', 'keydown', 'keyup']
                for event in dom_events:
                    pattern = re.compile(rf'on{event.capitalize()}\s*=\s*{{?\s*(\w+)')
                    for match in pattern.finditer(content):
                        handler_name = match.group(1)
                        handler_id = f"{file_path}:{handler_name}"
                        if handler_id in self.symbols:
                            self.indirect_uses[handler_id].add(f'dom_event_{event}')
                            self.event_listeners[event].append(handler_id)
                
                # Event emitters de Node.js
                emitter_pattern = re.compile(r'\.on\s*\(\s*[\'"](\w+)[\'"]\s*,\s*(\w+)')
                for match in emitter_pattern.finditer(content):
                    event_name = match.group(1)
                    handler_name = match.group(2)
                    handler_id = f"{file_path}:{handler_name}"
                    if handler_id in self.symbols:
                        self.indirect_uses[handler_id].add(f'node_event_{event_name}')
                        self.event_listeners[event_name].append(handler_id)
                
            except Exception as e:
                logger.warning(f"Error analizando event listeners en {file_path}: {e}")
    
    def _detect_framework_components(self):
        """Detectar componentes de frameworks populares."""
        for file_path in self.imports.keys():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # React components
                if 'react' in content.lower():
                    # Componentes funcionales
                    func_component_pattern = re.compile(r'(?:export\s+)?(?:const|function)\s+(\w+)\s*[=:]\s*(?:\([^)]*\)|function\s*\([^)]*\))\s*(?:=>|{)[\s\S]*?return\s*\(?[\s\S]*?<')
                    for match in func_component_pattern.finditer(content):
                        component_name = match.group(1)
                        if component_name[0].isupper():  # Componentes empiezan con mayúscula
                            component_id = f"{file_path}:{component_name}"
                            self.react_components.add(component_id)
                            self.indirect_uses[component_id].add('react_component')
                    
                    # Componentes de clase
                    class_component_pattern = re.compile(r'class\s+(\w+)\s+extends\s+(?:React\.)?Component')
                    for match in class_component_pattern.finditer(content):
                        component_name = match.group(1)
                        component_id = f"{file_path}:class:{component_name}"
                        self.react_components.add(component_id)
                        self.indirect_uses[component_id].add('react_class_component')
                    
                    # Hooks personalizados
                    hook_pattern = re.compile(r'(?:export\s+)?(?:const|function)\s+(use\w+)\s*[=:]')
                    for match in hook_pattern.finditer(content):
                        hook_name = match.group(1)
                        hook_id = f"{file_path}:{hook_name}"
                        self.indirect_uses[hook_id].add('react_custom_hook')
                
                # Vue components
                if 'vue' in content.lower():
                    # Componentes Vue
                    vue_component_pattern = re.compile(r'export\s+default\s*{[\s\S]*?name\s*:\s*[\'"](\w+)[\'"]')
                    for match in vue_component_pattern.finditer(content):
                        component_name = match.group(1)
                        component_id = f"{file_path}:{component_name}"
                        self.vue_components.add(component_id)
                        self.indirect_uses[component_id].add('vue_component')
                    
                    # Setup function (Vue 3)
                    if 'setup()' in content or 'setup:' in content:
                        file_component_id = f"{file_path}:vue_setup"
                        self.vue_components.add(file_component_id)
                        self.indirect_uses[file_component_id].add('vue3_composition_api')
                
                # Express routes
                if 'express' in content.lower() or 'router' in content:
                    route_methods = ['get', 'post', 'put', 'delete', 'patch', 'use', 'all']
                    for method in route_methods:
                        pattern = re.compile(rf'(?:app|router)\.{method}\s*\([^,]+,\s*(\w+)')
                        for match in pattern.finditer(content):
                            handler_name = match.group(1)
                            handler_id = f"{file_path}:{handler_name}"
                            if handler_id in self.symbols:
                                self.indirect_uses[handler_id].add(f'express_route_{method}')
                
            except Exception as e:
                logger.warning(f"Error detectando componentes en {file_path}: {e}")
    
    def _analyze_async_flow(self):
        """Analizar flujo asíncrono y await."""
        for file_path in self.imports.keys():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Detectar funciones awaited
                await_pattern = re.compile(r'await\s+(\w+)\s*\(')
                for match in await_pattern.finditer(content):
                    func_name = match.group(1)
                    # Buscar la función en símbolos locales e importados
                    if func_name in self.imports.get(file_path, {}):
                        # Es una función importada
                        module = self.imports[file_path][func_name]
                        resolved_path = self._resolve_module_path(file_path, module)
                        if resolved_path and resolved_path in self.exports:
                            if func_name in self.exports[resolved_path]:
                                func_id = self.exports[resolved_path][func_name]
                                self.indirect_uses[func_id].add('awaited_async_function')
                    else:
                        # Buscar localmente
                        func_id = f"{file_path}:{func_name}"
                        if func_id in self.symbols:
                            self.indirect_uses[func_id].add('awaited_async_function')
                
            except Exception as e:
                logger.warning(f"Error analizando flujo asíncrono en {file_path}: {e}")
    
    def _build_call_graph(self):
        """Construir el grafo de llamadas."""
        for file_path in self.imports.keys():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Detectar llamadas a funciones
                call_pattern = re.compile(r'(\w+)\s*\(')
                
                for match in call_pattern.finditer(content):
                    func_name = match.group(1)
                    line = content[:match.start()].count('\n') + 1
                    
                    # Ignorar palabras clave
                    if func_name in ['if', 'for', 'while', 'switch', 'catch', 'function', 'return']:
                        continue
                    
                    # Determinar el contexto de la llamada (qué función la hace)
                    caller = self._find_enclosing_function(file_path, line)
                    if caller:
                        # Buscar la función llamada
                        callee = self._resolve_function_call(file_path, func_name)
                        if callee and callee != caller:
                            self.call_graph.add_edge(caller, callee, line=line)
                
            except Exception as e:
                logger.warning(f"Error construyendo grafo de llamadas para {file_path}: {e}")
    
    def _find_enclosing_function(self, file_path: str, line: int) -> Optional[str]:
        """Encontrar la función que contiene una línea dada."""
        candidates = []
        
        for symbol_id, symbol in self.symbols.items():
            if symbol.file_path == file_path and symbol.line <= line:
                if symbol.type in ['function', 'arrow_function', 'method']:
                    candidates.append((symbol.line, symbol_id))
        
        if candidates:
            # Retornar la función más cercana
            candidates.sort(reverse=True)
            return candidates[0][1]
        
        return None
    
    def _resolve_function_call(self, file_path: str, func_name: str) -> Optional[str]:
        """Resolver a qué función se refiere una llamada."""
        # Primero buscar localmente
        local_id = f"{file_path}:function:{func_name}"
        if local_id in self.symbols:
            return local_id
        
        # Buscar en arrow functions
        arrow_id = f"{file_path}:arrow:{func_name}"
        if arrow_id in self.symbols:
            return arrow_id
        
        # Buscar en imports
        if func_name in self.imports.get(file_path, {}):
            module = self.imports[file_path][func_name]
            resolved_path = self._resolve_module_path(file_path, module)
            if resolved_path and resolved_path in self.exports:
                if func_name in self.exports[resolved_path]:
                    return self.exports[resolved_path][func_name]
        
        return None
    
    def _propagate_usage(self):
        """Propagar información de uso a través del grafo."""
        # Propagar a través del grafo de llamadas
        changed = True
        iterations = 0
        
        while changed and iterations < 50:
            changed = False
            iterations += 1
            
            for node in self.call_graph.nodes():
                if node in self.indirect_uses:
                    # Propagar a todos los sucesores
                    for successor in self.call_graph.successors(node):
                        old_size = len(self.indirect_uses[successor])
                        self.indirect_uses[successor].add(f"called_by_{Path(node).name}")
                        if len(self.indirect_uses[successor]) > old_size:
                            changed = True
        
        logger.info(f"Propagación de uso completada en {iterations} iteraciones")
    
    def _compute_reachability(self) -> Set[str]:
        """Calcular símbolos alcanzables."""
        reachable = set()
        
        # Entry points
        entry_points = set()
        
        # Archivos principales
        for file_path in self.symbols:
            if any(entry in file_path for entry in ['index.js', 'main.js', 'app.js', 'server.js']):
                # Agregar todas las funciones del archivo principal
                for symbol_id, symbol in self.symbols.items():
                    if symbol.file_path == file_path:
                        entry_points.add(symbol_id)
        
        # Exports (pueden ser usados externamente)
        for file_exports in self.exports.values():
            entry_points.update(file_exports.values())
        
        # Componentes de frameworks
        entry_points.update(self.react_components)
        entry_points.update(self.vue_components)
        
        # Event listeners
        for listeners in self.event_listeners.values():
            entry_points.update(listeners)
        
        # Símbolos con uso indirecto
        entry_points.update(self.indirect_uses.keys())
        
        # BFS desde entry points
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
        
        return reachable
