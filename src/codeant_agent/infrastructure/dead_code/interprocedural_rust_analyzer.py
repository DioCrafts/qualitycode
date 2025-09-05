"""
Analizador interprocedural para Rust con seguimiento de flujo de datos.
Maneja ownership, lifetimes, traits, y macros específicas de Rust.
"""

import re
from typing import Dict, Set, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import networkx as nx
import logging

logger = logging.getLogger(__name__)


@dataclass
class RustSymbol:
    """Símbolo en código Rust."""
    id: str
    name: str
    kind: str  # 'fn', 'struct', 'enum', 'trait', 'impl', 'mod', 'const', 'static'
    file_path: str
    line: int
    visibility: str  # 'pub', 'pub(crate)', 'pub(super)', 'private'
    is_async: bool = False
    is_unsafe: bool = False
    is_generic: bool = False
    traits_implemented: List[str] = field(default_factory=list)
    lifetime_params: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)  # #[derive], #[test], etc.


class InterproceduralRustAnalyzer:
    """
    Análisis interprocedural para Rust.
    Maneja características específicas como ownership, traits, y macros.
    """
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.symbols: Dict[str, RustSymbol] = {}
        self.call_graph = nx.DiGraph()
        self.trait_implementations: Dict[str, List[str]] = defaultdict(list)  # trait -> [implementors]
        self.macro_expansions: Dict[str, List[str]] = defaultdict(list)
        self.test_functions: Set[str] = set()
        self.bench_functions: Set[str] = set()
        self.main_functions: Set[str] = set()
        self.indirect_uses: Dict[str, Set[str]] = defaultdict(set)
        self.crate_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.unsafe_blocks: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.derive_macros: Dict[str, Set[str]] = defaultdict(set)
        
    def analyze(self) -> Dict[str, Any]:
        """Ejecutar análisis interprocedural para Rust."""
        logger.info("Iniciando análisis interprocedural para Rust...")
        
        # Fase 1: Analizar Cargo.toml para entender la estructura del proyecto
        self._analyze_cargo_structure()
        
        # Fase 2: Extraer todos los símbolos
        self._extract_all_symbols()
        
        # Fase 3: Analizar traits e implementaciones
        self._analyze_traits_and_impls()
        
        # Fase 4: Detectar macros y sus expansiones
        self._analyze_macros()
        
        # Fase 5: Analizar tests y benchmarks
        self._analyze_tests_and_benches()
        
        # Fase 6: Construir grafo de llamadas
        self._build_call_graph()
        
        # Fase 7: Analizar unsafe y FFI
        self._analyze_unsafe_and_ffi()
        
        # Fase 8: Detectar entry points
        self._detect_entry_points()
        
        # Fase 9: Propagar uso
        self._propagate_usage()
        
        # Fase 10: Calcular alcanzabilidad
        reachable = self._compute_reachability()
        
        return {
            'reachable_symbols': reachable,
            'indirect_uses': dict(self.indirect_uses),
            'trait_implementations': dict(self.trait_implementations),
            'test_functions': list(self.test_functions),
            'bench_functions': list(self.bench_functions),
            'main_functions': list(self.main_functions),
            'macro_expansions': dict(self.macro_expansions),
            'unsafe_usage': dict(self.unsafe_blocks)
        }
    
    def _analyze_cargo_structure(self):
        """Analizar la estructura del proyecto Rust desde Cargo.toml."""
        cargo_path = self.project_path / 'Cargo.toml'
        if not cargo_path.exists():
            logger.warning("No se encontró Cargo.toml")
            return
        
        try:
            with open(cargo_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detectar tipo de crate
            if '[[bin]]' in content or 'name = "main"' in content:
                self.crate_type = 'binary'
            elif '[lib]' in content:
                self.crate_type = 'library'
            else:
                self.crate_type = 'mixed'
            
            # Extraer dependencias
            deps_pattern = re.compile(r'\[dependencies\](.*?)\[', re.DOTALL)
            match = deps_pattern.search(content)
            if match:
                deps_section = match.group(1)
                dep_pattern = re.compile(r'^(\w+)\s*=', re.MULTILINE)
                for dep_match in dep_pattern.finditer(deps_section):
                    dep_name = dep_match.group(1)
                    self.crate_dependencies['external'].add(dep_name)
        
        except Exception as e:
            logger.warning(f"Error analizando Cargo.toml: {e}")
    
    def _extract_all_symbols(self):
        """Extraer todos los símbolos de archivos Rust."""
        for rs_file in self.project_path.rglob('*.rs'):
            if 'target' in str(rs_file) or '.git' in str(rs_file):
                continue
            
            try:
                with open(rs_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self._extract_symbols_from_file(str(rs_file), content)
            except Exception as e:
                logger.warning(f"Error analizando {rs_file}: {e}")
    
    def _extract_symbols_from_file(self, file_path: str, content: str):
        """Extraer símbolos de un archivo Rust."""
        # Funciones
        self._extract_functions(file_path, content)
        
        # Structs
        self._extract_structs(file_path, content)
        
        # Enums
        self._extract_enums(file_path, content)
        
        # Traits
        self._extract_traits(file_path, content)
        
        # Implementaciones
        self._extract_impls(file_path, content)
        
        # Módulos
        self._extract_modules(file_path, content)
        
        # Constantes y statics
        self._extract_constants(file_path, content)
    
    def _extract_functions(self, file_path: str, content: str):
        """Extraer definiciones de funciones."""
        # Patrón para funciones con atributos opcionales
        func_pattern = re.compile(
            r'(?:#\[([^\]]+)\]\s*)*'  # Atributos opcionales
            r'(?:(pub(?:\([^)]+\))?)\s+)?'  # Visibilidad opcional
            r'(?:(async)\s+)?'  # async opcional
            r'(?:(unsafe)\s+)?'  # unsafe opcional
            r'fn\s+(\w+)'  # nombre de función
            r'(<[^>]+>)?'  # genéricos opcionales
            r'\s*\([^)]*\)'  # parámetros
            r'(?:\s*->\s*[^{]+)?'  # tipo de retorno opcional
            r'\s*(?:where[^{]+)?'  # where clause opcional
            r'\s*\{',  # apertura del cuerpo
            re.MULTILINE | re.DOTALL
        )
        
        for match in func_pattern.finditer(content):
            attributes = match.group(1) or ''
            visibility = match.group(2) or 'private'
            is_async = match.group(3) is not None
            is_unsafe = match.group(4) is not None
            func_name = match.group(5)
            generics = match.group(6) is not None
            
            line = content[:match.start()].count('\n') + 1
            
            symbol_id = f"{file_path}:fn:{func_name}"
            
            # Extraer atributos
            attrs = []
            if attributes:
                attrs = [attr.strip() for attr in attributes.split(',')]
            
            symbol = RustSymbol(
                id=symbol_id,
                name=func_name,
                kind='fn',
                file_path=file_path,
                line=line,
                visibility=visibility,
                is_async=is_async,
                is_unsafe=is_unsafe,
                is_generic=generics,
                attributes=attrs
            )
            
            self.symbols[symbol_id] = symbol
            
            # Detectar funciones especiales
            if func_name == 'main':
                self.main_functions.add(symbol_id)
                self.indirect_uses[symbol_id].add('entry_point_main')
            
            if 'test' in attributes:
                self.test_functions.add(symbol_id)
                self.indirect_uses[symbol_id].add('test_function')
            
            if 'bench' in attributes:
                self.bench_functions.add(symbol_id)
                self.indirect_uses[symbol_id].add('benchmark_function')
            
            # Detectar handlers de frameworks web
            for attr in attrs:
                if any(framework in attr for framework in ['route', 'get', 'post', 'handler']):
                    self.indirect_uses[symbol_id].add(f'web_handler_{attr}')
                
                if 'tokio::main' in attr:
                    self.indirect_uses[symbol_id].add('tokio_entry_point')
                
                if 'actix' in attr:
                    self.indirect_uses[symbol_id].add('actix_handler')
    
    def _extract_structs(self, file_path: str, content: str):
        """Extraer definiciones de structs."""
        struct_pattern = re.compile(
            r'(?:#\[([^\]]+)\]\s*)*'  # Atributos opcionales
            r'(?:(pub(?:\([^)]+\))?)\s+)?'  # Visibilidad
            r'struct\s+(\w+)'  # nombre del struct
            r'(<[^>]+>)?'  # genéricos opcionales
            r'(?:\s*\([^)]*\))?'  # tuple struct opcional
            r'(?:\s*\{[^}]*\})?'  # struct fields opcional
            r'\s*;?',  # punto y coma opcional
            re.MULTILINE | re.DOTALL
        )
        
        for match in struct_pattern.finditer(content):
            attributes = match.group(1) or ''
            visibility = match.group(2) or 'private'
            struct_name = match.group(3)
            has_generics = match.group(4) is not None
            
            line = content[:match.start()].count('\n') + 1
            symbol_id = f"{file_path}:struct:{struct_name}"
            
            attrs = []
            if attributes:
                # Extraer derives
                derive_match = re.search(r'derive\(([^)]+)\)', attributes)
                if derive_match:
                    derives = derive_match.group(1).split(',')
                    for derive in derives:
                        derive = derive.strip()
                        self.derive_macros[derive].add(symbol_id)
                        attrs.append(f"derive({derive})")
            
            symbol = RustSymbol(
                id=symbol_id,
                name=struct_name,
                kind='struct',
                file_path=file_path,
                line=line,
                visibility=visibility,
                is_generic=has_generics,
                attributes=attrs
            )
            
            self.symbols[symbol_id] = symbol
    
    def _extract_enums(self, file_path: str, content: str):
        """Extraer definiciones de enums."""
        enum_pattern = re.compile(
            r'(?:#\[([^\]]+)\]\s*)*'
            r'(?:(pub(?:\([^)]+\))?)\s+)?'
            r'enum\s+(\w+)'
            r'(<[^>]+>)?'
            r'\s*\{',
            re.MULTILINE
        )
        
        for match in enum_pattern.finditer(content):
            attributes = match.group(1) or ''
            visibility = match.group(2) or 'private'
            enum_name = match.group(3)
            has_generics = match.group(4) is not None
            
            line = content[:match.start()].count('\n') + 1
            symbol_id = f"{file_path}:enum:{enum_name}"
            
            symbol = RustSymbol(
                id=symbol_id,
                name=enum_name,
                kind='enum',
                file_path=file_path,
                line=line,
                visibility=visibility,
                is_generic=has_generics
            )
            
            self.symbols[symbol_id] = symbol
            
            # Si es un error type, probablemente se use
            if 'Error' in enum_name or 'error' in attributes.lower():
                self.indirect_uses[symbol_id].add('error_type')
    
    def _extract_traits(self, file_path: str, content: str):
        """Extraer definiciones de traits."""
        trait_pattern = re.compile(
            r'(?:(pub(?:\([^)]+\))?)\s+)?'
            r'(?:(unsafe)\s+)?'
            r'trait\s+(\w+)'
            r'(<[^>]+>)?'
            r'(?:\s*:\s*([^{]+))?'  # supertraits
            r'\s*\{',
            re.MULTILINE
        )
        
        for match in trait_pattern.finditer(content):
            visibility = match.group(1) or 'private'
            is_unsafe = match.group(2) is not None
            trait_name = match.group(3)
            has_generics = match.group(4) is not None
            supertraits = match.group(5)
            
            line = content[:match.start()].count('\n') + 1
            symbol_id = f"{file_path}:trait:{trait_name}"
            
            symbol = RustSymbol(
                id=symbol_id,
                name=trait_name,
                kind='trait',
                file_path=file_path,
                line=line,
                visibility=visibility,
                is_unsafe=is_unsafe,
                is_generic=has_generics
            )
            
            self.symbols[symbol_id] = symbol
            
            # Traits estándar que indican uso
            if trait_name in ['Drop', 'Deref', 'From', 'Into', 'Iterator']:
                self.indirect_uses[symbol_id].add(f'standard_trait_{trait_name}')
    
    def _extract_impls(self, file_path: str, content: str):
        """Extraer bloques impl."""
        # impl Trait for Type
        impl_trait_pattern = re.compile(
            r'impl(?:<[^>]+>)?\s+(\w+)\s+for\s+(\w+)(?:<[^>]+>)?\s*(?:where[^{]+)?\s*\{',
            re.MULTILINE
        )
        
        for match in impl_trait_pattern.finditer(content):
            trait_name = match.group(1)
            type_name = match.group(2)
            
            line = content[:match.start()].count('\n') + 1
            
            # Registrar la implementación
            impl_id = f"{file_path}:impl:{trait_name}_for_{type_name}"
            type_id = f"{file_path}:struct:{type_name}"  # Asumimos struct, podría ser enum
            
            if type_id in self.symbols:
                self.symbols[type_id].traits_implemented.append(trait_name)
            
            self.trait_implementations[trait_name].append(type_id)
            
            # Marcar como usado si implementa traits importantes
            if trait_name in ['Default', 'Clone', 'Debug', 'Display', 'Error']:
                self.indirect_uses[type_id].add(f'implements_{trait_name}')
            
            # Serde traits
            if trait_name in ['Serialize', 'Deserialize']:
                self.indirect_uses[type_id].add('serde_serializable')
        
        # impl Type (métodos inherentes)
        impl_pattern = re.compile(
            r'impl(?:<[^>]+>)?\s+(\w+)(?:<[^>]+>)?\s*(?:where[^{]+)?\s*\{',
            re.MULTILINE
        )
        
        for match in impl_pattern.finditer(content):
            if ' for ' not in content[match.start():match.end()]:  # No es impl Trait for Type
                type_name = match.group(1)
                line = content[:match.start()].count('\n') + 1
                
                # Extraer métodos del bloque impl
                impl_block_start = match.end()
                impl_block_end = self._find_block_end(content, impl_block_start)
                impl_content = content[impl_block_start:impl_block_end]
                
                self._extract_impl_methods(file_path, type_name, impl_content, line)
    
    def _extract_impl_methods(self, file_path: str, type_name: str, impl_content: str, base_line: int):
        """Extraer métodos de un bloque impl."""
        method_pattern = re.compile(
            r'(?:(pub(?:\([^)]+\))?)\s+)?'
            r'(?:(async)\s+)?'
            r'(?:(unsafe)\s+)?'
            r'fn\s+(\w+)'
            r'(<[^>]+>)?'
            r'\s*\([^)]*\)',
            re.MULTILINE
        )
        
        for match in method_pattern.finditer(impl_content):
            visibility = match.group(1) or 'private'
            is_async = match.group(2) is not None
            is_unsafe = match.group(3) is not None
            method_name = match.group(4)
            
            line = base_line + impl_content[:match.start()].count('\n')
            
            symbol_id = f"{file_path}:method:{type_name}::{method_name}"
            
            symbol = RustSymbol(
                id=symbol_id,
                name=f"{type_name}::{method_name}",
                kind='method',
                file_path=file_path,
                line=line,
                visibility=visibility,
                is_async=is_async,
                is_unsafe=is_unsafe
            )
            
            self.symbols[symbol_id] = symbol
            
            # Métodos especiales
            if method_name in ['new', 'default', 'from']:
                self.indirect_uses[symbol_id].add(f'constructor_method_{method_name}')
            
            if method_name in ['fmt', 'to_string']:
                self.indirect_uses[symbol_id].add('formatting_method')
    
    def _extract_modules(self, file_path: str, content: str):
        """Extraer definiciones de módulos."""
        mod_pattern = re.compile(
            r'(?:(pub(?:\([^)]+\))?)\s+)?'
            r'mod\s+(\w+)\s*(?:\{|;)',
            re.MULTILINE
        )
        
        for match in mod_pattern.finditer(content):
            visibility = match.group(1) or 'private'
            mod_name = match.group(2)
            
            line = content[:match.start()].count('\n') + 1
            symbol_id = f"{file_path}:mod:{mod_name}"
            
            symbol = RustSymbol(
                id=symbol_id,
                name=mod_name,
                kind='mod',
                file_path=file_path,
                line=line,
                visibility=visibility
            )
            
            self.symbols[symbol_id] = symbol
            
            # Módulos especiales
            if mod_name in ['tests', 'test']:
                self.indirect_uses[symbol_id].add('test_module')
            elif mod_name == 'benches':
                self.indirect_uses[symbol_id].add('benchmark_module')
    
    def _extract_constants(self, file_path: str, content: str):
        """Extraer constantes y statics."""
        const_pattern = re.compile(
            r'(?:(pub(?:\([^)]+\))?)\s+)?'
            r'(const|static)\s+'
            r'(?:(mut)\s+)?'
            r'(\w+)\s*:\s*([^=]+)\s*=',
            re.MULTILINE
        )
        
        for match in const_pattern.finditer(content):
            visibility = match.group(1) or 'private'
            const_type = match.group(2)
            is_mut = match.group(3) is not None
            const_name = match.group(4)
            type_annotation = match.group(5).strip()
            
            line = content[:match.start()].count('\n') + 1
            symbol_id = f"{file_path}:{const_type}:{const_name}"
            
            symbol = RustSymbol(
                id=symbol_id,
                name=const_name,
                kind=const_type,
                file_path=file_path,
                line=line,
                visibility=visibility
            )
            
            self.symbols[symbol_id] = symbol
            
            # Statics mutables son peligrosos y probablemente se usan
            if const_type == 'static' and is_mut:
                self.indirect_uses[symbol_id].add('mutable_static')
    
    def _analyze_traits_and_impls(self):
        """Analizar relaciones entre traits e implementaciones."""
        # Para cada trait, verificar sus implementadores
        for trait_name, implementors in self.trait_implementations.items():
            trait_id = None
            
            # Buscar el trait en los símbolos
            for symbol_id, symbol in self.symbols.items():
                if symbol.type == 'class' and symbol.name == trait_name:  # 'class' se usa para traits en nuestro modelo
                    trait_id = symbol_id
                    break
            
            if trait_id:
                # El trait es usado por sus implementadores
                for impl_id in implementors:
                    self.indirect_uses[trait_id].add(f'implemented_by_{Path(impl_id).name}')
    
    def _analyze_macros(self):
        """Analizar macros y sus expansiones."""
        for file_path in self.symbols:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Macros declarativas
                macro_rules_pattern = re.compile(r'macro_rules!\s+(\w+)\s*\{')
                for match in macro_rules_pattern.finditer(content):
                    macro_name = match.group(1)
                    macro_id = f"{file_path}:macro:{macro_name}"
                    self.indirect_uses[macro_id].add('declarative_macro')
                
                # Uso de macros
                macro_use_pattern = re.compile(r'(\w+)!\s*\(')
                for match in macro_use_pattern.finditer(content):
                    macro_name = match.group(1)
                    
                    # Macros estándar que indican patrones de uso
                    if macro_name in ['println', 'eprintln', 'debug', 'info', 'error', 'warn']:
                        # Funciones de logging/output
                        pass
                    elif macro_name in ['vec', 'hashmap', 'btreemap']:
                        # Constructores de colecciones
                        pass
                    elif macro_name == 'include':
                        # Inclusión de archivos
                        self.macro_expansions['include'].append(file_path)
                    elif macro_name in ['lazy_static', 'once_cell']:
                        # Inicialización lazy
                        self.indirect_uses[file_path].add('lazy_initialization')
                
                # Proc macros y derives
                proc_macro_pattern = re.compile(r'#\[(\w+)(?:\([^)]*\))?\]')
                for match in proc_macro_pattern.finditer(content):
                    attr_name = match.group(1)
                    
                    if attr_name == 'tokio::main':
                        self.indirect_uses[file_path].add('tokio_runtime')
                    elif attr_name == 'actix_web::main':
                        self.indirect_uses[file_path].add('actix_runtime')
                    elif attr_name in ['get', 'post', 'put', 'delete']:
                        self.indirect_uses[file_path].add(f'http_handler_{attr_name}')
                
            except Exception as e:
                logger.warning(f"Error analizando macros en {file_path}: {e}")
    
    def _analyze_tests_and_benches(self):
        """Analizar funciones de test y benchmark."""
        # Los tests ya fueron detectados en _extract_functions
        # Aquí podemos hacer análisis adicional
        
        # Marcar módulos de test
        for symbol_id, symbol in self.symbols.items():
            if symbol.type == 'module' and symbol.name in ['tests', 'test']:  # Usar 'module' para módulos
                # Todas las funciones dentro de módulos de test son tests
                file_path = symbol.file_path
                for other_id, other_symbol in self.symbols.items():
                    if (other_symbol.file_path == file_path and 
                        other_symbol.type == 'function' and
                        other_symbol.line_number > symbol.line_number):
                        self.test_functions.add(other_id)
                        self.indirect_uses[other_id].add('test_function_in_test_module')
    
    def _build_call_graph(self):
        """Construir el grafo de llamadas."""
        for file_path in set(s.file_path for s in self.symbols.values()):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Llamadas a funciones
                call_pattern = re.compile(r'(\w+)(?:::\w+)*\s*\(')
                
                for match in call_pattern.finditer(content):
                    full_call = match.group(0)
                    line = content[:match.start()].count('\n') + 1
                    
                    # Descomponer la llamada
                    parts = match.group(1).split('::')
                    
                    # Encontrar el contexto (función que hace la llamada)
                    caller = self._find_enclosing_function(file_path, line)
                    
                    if caller:
                        # Resolver la función llamada
                        if len(parts) == 1:
                            # Llamada simple
                            callee = self._resolve_function_call(file_path, parts[0])
                        else:
                            # Llamada calificada (Type::method o module::function)
                            callee = self._resolve_qualified_call(file_path, parts)
                        
                        if callee and callee != caller:
                            self.call_graph.add_edge(caller, callee, line=line)
                
            except Exception as e:
                logger.warning(f"Error construyendo grafo de llamadas para {file_path}: {e}")
    
    def _analyze_unsafe_and_ffi(self):
        """Analizar bloques unsafe y FFI."""
        for file_path in set(s.file_path for s in self.symbols.values()):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Bloques unsafe
                unsafe_pattern = re.compile(r'unsafe\s*\{')
                for match in unsafe_pattern.finditer(content):
                    line = content[:match.start()].count('\n') + 1
                    enclosing_fn = self._find_enclosing_function(file_path, line)
                    if enclosing_fn:
                        self.unsafe_blocks[enclosing_fn].append((line, line))
                        self.indirect_uses[enclosing_fn].add('contains_unsafe_block')
                
                # extern functions (FFI)
                extern_pattern = re.compile(r'extern\s+"C"\s*\{([^}]*)\}', re.DOTALL)
                for match in extern_pattern.finditer(content):
                    ffi_block = match.group(1)
                    # Todas las funciones en extern blocks son entry points potenciales
                    fn_pattern = re.compile(r'fn\s+(\w+)')
                    for fn_match in fn_pattern.finditer(ffi_block):
                        fn_name = fn_match.group(1)
                        fn_id = f"{file_path}:fn:{fn_name}"
                        if fn_id in self.symbols:
                            self.indirect_uses[fn_id].add('ffi_function')
                
            except Exception as e:
                logger.warning(f"Error analizando unsafe/FFI en {file_path}: {e}")
    
    def _detect_entry_points(self):
        """Detectar puntos de entrada del programa."""
        entry_points = set()
        
        # main functions
        entry_points.update(self.main_functions)
        
        # Tests
        entry_points.update(self.test_functions)
        
        # Benchmarks
        entry_points.update(self.bench_functions)
        
        # Funciones públicas en libraries
        if hasattr(self, 'crate_type') and self.crate_type == 'library':
            for symbol_id, symbol in self.symbols.items():
                # Verificar si es público basándose en los decoradores o el nombre
                if hasattr(symbol, 'decorators'):
                    is_public = any('pub' in dec for dec in symbol.decorators)
                else:
                    # En Rust, asumimos que es público si el nombre empieza con mayúscula o tiene ciertos patrones
                    is_public = symbol.name[0].isupper() if symbol.name else False
                    
                # Verificar el tipo del símbolo
                if is_public and symbol.type in ['function', 'class']:
                    entry_points.add(symbol_id)
                    self.indirect_uses[symbol_id].add('public_api')
        
        # Funciones con atributos especiales (usando decorators en lugar de attributes)
        for symbol_id, symbol in self.symbols.items():
            if hasattr(symbol, 'decorators'):
                for decorator in symbol.decorators:
                    if any(entry_attr in decorator for entry_attr in ['export', 'no_mangle', 'extern']):
                        entry_points.add(symbol_id)
                        self.indirect_uses[symbol_id].add(f'exported_symbol_{decorator}')
        
        return entry_points
    
    def _propagate_usage(self):
        """Propagar información de uso."""
        # Propagar a través del grafo de llamadas
        changed = True
        iterations = 0
        
        while changed and iterations < 50:
            changed = False
            iterations += 1
            
            # Propagar uso a través de llamadas
            for node in self.call_graph.nodes():
                if node in self.indirect_uses:
                    for predecessor in self.call_graph.predecessors(node):
                        old_size = len(self.indirect_uses[predecessor])
                        self.indirect_uses[predecessor].add(f"calls_{Path(node).name}")
                        if len(self.indirect_uses[predecessor]) > old_size:
                            changed = True
            
            # Propagar a través de traits
            for trait_name, implementors in self.trait_implementations.items():
                # Si un trait es usado, sus implementadores también
                trait_used = any(impl_id in self.indirect_uses for impl_id in implementors)
                if trait_used:
                    for impl_id in implementors:
                        old_size = len(self.indirect_uses[impl_id])
                        self.indirect_uses[impl_id].add(f"implements_used_trait_{trait_name}")
                        if len(self.indirect_uses[impl_id]) > old_size:
                            changed = True
        
        logger.info(f"Propagación de uso completada en {iterations} iteraciones")
    
    def _compute_reachability(self) -> Set[str]:
        """Calcular símbolos alcanzables."""
        # Entry points
        entry_points = self._detect_entry_points()
        
        # BFS desde entry points
        reachable = set()
        queue = deque(entry_points)
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            reachable.add(current)
            
            # Seguir el grafo de llamadas
            if current in self.call_graph:
                for successor in self.call_graph.successors(current):
                    if successor not in visited:
                        queue.append(successor)
            
            # Incluir tipos usados por funciones alcanzables
            if current in self.symbols:
                symbol = self.symbols[current]
                # Si una función es alcanzable, los tipos que usa también lo son
                # Esto requeriría análisis de tipos más profundo
        
        # Agregar todos los símbolos con uso indirecto
        for symbol_id in self.indirect_uses:
            reachable.add(symbol_id)
        
        return reachable
    
    def _find_enclosing_function(self, file_path: str, line: int) -> Optional[str]:
        """Encontrar la función que contiene una línea."""
        candidates = []
        
        for symbol_id, symbol in self.symbols.items():
            if (symbol.file_path == file_path and 
                symbol.type == 'function' and 
                symbol.line_number <= line):
                candidates.append((symbol.line_number, symbol_id))
        
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        
        return None
    
    def _resolve_function_call(self, file_path: str, func_name: str) -> Optional[str]:
        """Resolver una llamada a función simple."""
        # Buscar en el archivo actual
        local_id = f"{file_path}:fn:{func_name}"
        if local_id in self.symbols:
            return local_id
        
        # Podría estar en otro módulo (requeriría análisis de imports)
        return None
    
    def _resolve_qualified_call(self, file_path: str, parts: List[str]) -> Optional[str]:
        """Resolver una llamada calificada como Type::method."""
        if len(parts) == 2:
            type_name, method_name = parts
            method_id = f"{file_path}:method:{type_name}::{method_name}"
            if method_id in self.symbols:
                return method_id
        
        return None
    
    def _find_block_end(self, content: str, start: int) -> int:
        """Encontrar el final de un bloque de código."""
        brace_count = 1
        i = start
        
        while i < len(content) and brace_count > 0:
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
            i += 1
        
        return i
