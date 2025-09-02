# Fase 7: Parser Especializado para Python con Análisis AST

## Objetivo General
Implementar un parser especializado para Python que vaya más allá del parsing básico de Tree-sitter, proporcionando análisis semántico profundo, detección de patrones específicos de Python, análisis de flujo de datos, y capacidades avanzadas de inspección de código necesarias para el agente CodeAnt.

## Descripción Técnica Detallada

### 7.1 Arquitectura del Parser Python Especializado

#### 7.1.1 Diseño del Sistema Python
```
┌─────────────────────────────────────────┐
│         Python Specialized Parser      │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ Tree-sitter │ │    Python AST       │ │
│  │   Python    │ │    Module           │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │  Semantic   │ │   Data Flow         │ │
│  │  Analysis   │ │   Analysis          │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Import    │ │   Type Inference    │ │
│  │  Resolver   │ │    Engine           │ │
│  └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
```

#### 7.1.2 Componentes Especializados
- **Python AST Module**: Análisis profundo del AST de Python
- **Semantic Analysis**: Análisis semántico y de scoping
- **Data Flow Analysis**: Seguimiento de flujo de datos
- **Import Resolver**: Resolución de imports y dependencias
- **Type Inference Engine**: Inferencia de tipos estática
- **Pattern Detector**: Detección de patrones Python-específicos

### 7.2 Python AST Analysis Engine

#### 7.2.1 Core Python Parser
```rust
use rustpython_parser::{ast, Parse};
use rustpython_common::source_code::SourceRange;

pub struct PythonSpecializedParser {
    tree_sitter_parser: Arc<UniversalParser>,
    semantic_analyzer: Arc<PythonSemanticAnalyzer>,
    type_inferencer: Arc<PythonTypeInferencer>,
    import_resolver: Arc<PythonImportResolver>,
    pattern_detector: Arc<PythonPatternDetector>,
    config: PythonParserConfig,
}

#[derive(Debug, Clone)]
pub struct PythonParserConfig {
    pub enable_semantic_analysis: bool,
    pub enable_type_inference: bool,
    pub enable_import_resolution: bool,
    pub enable_data_flow_analysis: bool,
    pub python_version: PythonVersion,
    pub strict_mode: bool,
    pub enable_experimental_features: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PythonVersion {
    Python37,
    Python38,
    Python39,
    Python310,
    Python311,
    Python312,
}

impl PythonSpecializedParser {
    pub async fn new(config: PythonParserConfig) -> Result<Self, PythonParserError> {
        let tree_sitter_parser = Arc::new(UniversalParser::new(ParserConfig::default()).await?);
        let semantic_analyzer = Arc::new(PythonSemanticAnalyzer::new());
        let type_inferencer = Arc::new(PythonTypeInferencer::new(config.python_version.clone()));
        let import_resolver = Arc::new(PythonImportResolver::new());
        let pattern_detector = Arc::new(PythonPatternDetector::new());
        
        Ok(Self {
            tree_sitter_parser,
            semantic_analyzer,
            type_inferencer,
            import_resolver,
            pattern_detector,
            config,
        })
    }
    
    pub async fn parse_python_file(&self, file_path: &Path) -> Result<PythonAnalysisResult, PythonParserError> {
        let start_time = Instant::now();
        
        // Read file content
        let content = tokio::fs::read_to_string(file_path).await
            .map_err(|e| PythonParserError::FileReadError(e.to_string()))?;
        
        // Parse with both Tree-sitter and RustPython
        let tree_sitter_result = self.tree_sitter_parser.parse_content(&content, ProgrammingLanguage::Python).await?;
        let rustpython_ast = self.parse_with_rustpython(&content)?;
        
        // Perform specialized analysis
        let mut analysis_result = PythonAnalysisResult {
            file_path: file_path.to_path_buf(),
            tree_sitter_result,
            rustpython_ast: Some(rustpython_ast.clone()),
            semantic_info: None,
            type_info: None,
            import_info: None,
            data_flow_info: None,
            patterns: Vec::new(),
            metrics: PythonMetrics::default(),
            parse_duration_ms: 0,
        };
        
        // Semantic analysis
        if self.config.enable_semantic_analysis {
            analysis_result.semantic_info = Some(
                self.semantic_analyzer.analyze(&rustpython_ast, &content).await?
            );
        }
        
        // Type inference
        if self.config.enable_type_inference {
            analysis_result.type_info = Some(
                self.type_inferencer.infer_types(&rustpython_ast, &content).await?
            );
        }
        
        // Import resolution
        if self.config.enable_import_resolution {
            analysis_result.import_info = Some(
                self.import_resolver.resolve_imports(&rustpython_ast, file_path).await?
            );
        }
        
        // Data flow analysis
        if self.config.enable_data_flow_analysis {
            analysis_result.data_flow_info = Some(
                self.analyze_data_flow(&rustpython_ast, &content).await?
            );
        }
        
        // Pattern detection
        analysis_result.patterns = self.pattern_detector.detect_patterns(&analysis_result).await?;
        
        // Calculate metrics
        analysis_result.metrics = self.calculate_python_metrics(&analysis_result);
        analysis_result.parse_duration_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(analysis_result)
    }
    
    fn parse_with_rustpython(&self, content: &str) -> Result<ast::Suite, PythonParserError> {
        let parse_result = ast::Suite::parse(content, "<string>")
            .map_err(|e| PythonParserError::RustPythonParseError(e.to_string()))?;
        
        Ok(parse_result)
    }
}
```

#### 7.2.2 Python Analysis Result Structure
```rust
#[derive(Debug, Clone)]
pub struct PythonAnalysisResult {
    pub file_path: PathBuf,
    pub tree_sitter_result: ParseResult,
    pub rustpython_ast: Option<ast::Suite>,
    pub semantic_info: Option<PythonSemanticInfo>,
    pub type_info: Option<PythonTypeInfo>,
    pub import_info: Option<PythonImportInfo>,
    pub data_flow_info: Option<PythonDataFlowInfo>,
    pub patterns: Vec<PythonPattern>,
    pub metrics: PythonMetrics,
    pub parse_duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct PythonSemanticInfo {
    pub scopes: Vec<PythonScope>,
    pub symbol_table: PythonSymbolTable,
    pub name_bindings: HashMap<String, Vec<NameBinding>>,
    pub function_definitions: Vec<FunctionDefinition>,
    pub class_definitions: Vec<ClassDefinition>,
    pub global_variables: Vec<GlobalVariable>,
    pub decorators: Vec<DecoratorUsage>,
}

#[derive(Debug, Clone)]
pub struct PythonTypeInfo {
    pub type_annotations: HashMap<SourceRange, PythonType>,
    pub inferred_types: HashMap<SourceRange, InferredType>,
    pub type_errors: Vec<TypeError>,
    pub generic_types: Vec<GenericType>,
    pub type_aliases: HashMap<String, PythonType>,
}

#[derive(Debug, Clone)]
pub struct PythonImportInfo {
    pub imports: Vec<ImportStatement>,
    pub from_imports: Vec<FromImportStatement>,
    pub resolved_modules: HashMap<String, ModuleInfo>,
    pub unresolved_imports: Vec<UnresolvedImport>,
    pub circular_imports: Vec<CircularImport>,
    pub dependency_graph: DependencyGraph,
}

#[derive(Debug, Clone)]
pub struct PythonDataFlowInfo {
    pub variable_flows: HashMap<String, Vec<DataFlowNode>>,
    pub function_calls: Vec<FunctionCall>,
    pub attribute_accesses: Vec<AttributeAccess>,
    pub control_flow_graph: ControlFlowGraph,
    pub def_use_chains: HashMap<String, DefUseChain>,
    pub dead_code_segments: Vec<DeadCodeSegment>,
}
```

### 7.3 Semantic Analysis Engine

#### 7.3.1 Python Semantic Analyzer
```rust
pub struct PythonSemanticAnalyzer {
    scope_analyzer: ScopeAnalyzer,
    symbol_resolver: SymbolResolver,
    decorator_analyzer: DecoratorAnalyzer,
}

impl PythonSemanticAnalyzer {
    pub fn new() -> Self {
        Self {
            scope_analyzer: ScopeAnalyzer::new(),
            symbol_resolver: SymbolResolver::new(),
            decorator_analyzer: DecoratorAnalyzer::new(),
        }
    }
    
    pub async fn analyze(&self, ast: &ast::Suite, content: &str) -> Result<PythonSemanticInfo, SemanticError> {
        let mut semantic_info = PythonSemanticInfo {
            scopes: Vec::new(),
            symbol_table: PythonSymbolTable::new(),
            name_bindings: HashMap::new(),
            function_definitions: Vec::new(),
            class_definitions: Vec::new(),
            global_variables: Vec::new(),
            decorators: Vec::new(),
        };
        
        // Analyze scopes
        semantic_info.scopes = self.scope_analyzer.analyze_scopes(ast)?;
        
        // Build symbol table
        semantic_info.symbol_table = self.symbol_resolver.build_symbol_table(ast, &semantic_info.scopes)?;
        
        // Analyze name bindings
        semantic_info.name_bindings = self.analyze_name_bindings(ast)?;
        
        // Extract definitions
        semantic_info.function_definitions = self.extract_function_definitions(ast)?;
        semantic_info.class_definitions = self.extract_class_definitions(ast)?;
        semantic_info.global_variables = self.extract_global_variables(ast)?;
        
        // Analyze decorators
        semantic_info.decorators = self.decorator_analyzer.analyze_decorators(ast)?;
        
        Ok(semantic_info)
    }
    
    fn analyze_name_bindings(&self, ast: &ast::Suite) -> Result<HashMap<String, Vec<NameBinding>>, SemanticError> {
        let mut bindings = HashMap::new();
        let mut visitor = NameBindingVisitor::new(&mut bindings);
        visitor.visit_suite(ast);
        Ok(bindings)
    }
    
    fn extract_function_definitions(&self, ast: &ast::Suite) -> Result<Vec<FunctionDefinition>, SemanticError> {
        let mut functions = Vec::new();
        let mut visitor = FunctionDefinitionVisitor::new(&mut functions);
        visitor.visit_suite(ast);
        Ok(functions)
    }
    
    fn extract_class_definitions(&self, ast: &ast::Suite) -> Result<Vec<ClassDefinition>, SemanticError> {
        let mut classes = Vec::new();
        let mut visitor = ClassDefinitionVisitor::new(&mut classes);
        visitor.visit_suite(ast);
        Ok(classes)
    }
}

#[derive(Debug, Clone)]
pub struct PythonScope {
    pub scope_type: ScopeType,
    pub name: Option<String>,
    pub range: SourceRange,
    pub parent_scope: Option<ScopeId>,
    pub child_scopes: Vec<ScopeId>,
    pub symbols: HashMap<String, Symbol>,
    pub imports: Vec<String>,
    pub nonlocal_vars: Vec<String>,
    pub global_vars: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScopeType {
    Module,
    Function,
    Class,
    Comprehension,
    AsyncFunction,
    Lambda,
}

#[derive(Debug, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub range: SourceRange,
    pub parameters: Vec<Parameter>,
    pub return_annotation: Option<ast::Expr>,
    pub decorators: Vec<ast::Expr>,
    pub is_async: bool,
    pub is_method: bool,
    pub is_classmethod: bool,
    pub is_staticmethod: bool,
    pub docstring: Option<String>,
    pub complexity: u32,
    pub local_variables: Vec<String>,
    pub calls_made: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ClassDefinition {
    pub name: String,
    pub range: SourceRange,
    pub bases: Vec<ast::Expr>,
    pub decorators: Vec<ast::Expr>,
    pub methods: Vec<String>,
    pub class_variables: Vec<String>,
    pub instance_variables: Vec<String>,
    pub docstring: Option<String>,
    pub metaclass: Option<String>,
    pub is_abstract: bool,
    pub inheritance_depth: u32,
}
```

### 7.4 Type Inference Engine

#### 7.4.1 Python Type Inferencer
```rust
pub struct PythonTypeInferencer {
    python_version: PythonVersion,
    builtin_types: HashMap<String, PythonType>,
    type_constraints: Vec<TypeConstraint>,
    generic_resolver: GenericTypeResolver,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PythonType {
    // Built-in types
    Int,
    Float,
    Str,
    Bool,
    Bytes,
    None,
    
    // Collections
    List(Box<PythonType>),
    Tuple(Vec<PythonType>),
    Dict(Box<PythonType>, Box<PythonType>),
    Set(Box<PythonType>),
    
    // Advanced types
    Union(Vec<PythonType>),
    Optional(Box<PythonType>),
    Callable(Vec<PythonType>, Box<PythonType>),
    Generic(String, Vec<PythonType>),
    
    // User-defined
    Class(String),
    Module(String),
    
    // Special
    Any,
    Unknown,
    TypeVar(String),
}

#[derive(Debug, Clone)]
pub struct InferredType {
    pub python_type: PythonType,
    pub confidence: f64,
    pub source: TypeSource,
    pub constraints: Vec<TypeConstraint>,
}

#[derive(Debug, Clone)]
pub enum TypeSource {
    Annotation,
    Literal,
    Assignment,
    FunctionReturn,
    BuiltinFunction,
    UserFunction,
    Attribute,
    Import,
    Inference,
}

impl PythonTypeInferencer {
    pub fn new(python_version: PythonVersion) -> Self {
        let mut inferencer = Self {
            python_version,
            builtin_types: HashMap::new(),
            type_constraints: Vec::new(),
            generic_resolver: GenericTypeResolver::new(),
        };
        
        inferencer.initialize_builtin_types();
        inferencer
    }
    
    pub async fn infer_types(&self, ast: &ast::Suite, content: &str) -> Result<PythonTypeInfo, TypeInferenceError> {
        let mut type_info = PythonTypeInfo {
            type_annotations: HashMap::new(),
            inferred_types: HashMap::new(),
            type_errors: Vec::new(),
            generic_types: Vec::new(),
            type_aliases: HashMap::new(),
        };
        
        // Extract explicit type annotations
        type_info.type_annotations = self.extract_type_annotations(ast)?;
        
        // Infer types for expressions
        type_info.inferred_types = self.infer_expression_types(ast, &type_info.type_annotations)?;
        
        // Detect type errors
        type_info.type_errors = self.detect_type_errors(ast, &type_info)?;
        
        // Resolve generic types
        type_info.generic_types = self.generic_resolver.resolve_generics(ast)?;
        
        // Extract type aliases
        type_info.type_aliases = self.extract_type_aliases(ast)?;
        
        Ok(type_info)
    }
    
    fn extract_type_annotations(&self, ast: &ast::Suite) -> Result<HashMap<SourceRange, PythonType>, TypeInferenceError> {
        let mut annotations = HashMap::new();
        let mut visitor = TypeAnnotationVisitor::new(&mut annotations);
        visitor.visit_suite(ast);
        Ok(annotations)
    }
    
    fn infer_expression_types(&self, ast: &ast::Suite, annotations: &HashMap<SourceRange, PythonType>) -> Result<HashMap<SourceRange, InferredType>, TypeInferenceError> {
        let mut inferred_types = HashMap::new();
        let mut visitor = TypeInferenceVisitor::new(&mut inferred_types, annotations, &self.builtin_types);
        visitor.visit_suite(ast);
        Ok(inferred_types)
    }
    
    fn detect_type_errors(&self, ast: &ast::Suite, type_info: &PythonTypeInfo) -> Result<Vec<TypeError>, TypeInferenceError> {
        let mut errors = Vec::new();
        let mut visitor = TypeErrorDetector::new(&mut errors, type_info);
        visitor.visit_suite(ast);
        Ok(errors)
    }
    
    fn initialize_builtin_types(&mut self) {
        self.builtin_types.insert("int".to_string(), PythonType::Int);
        self.builtin_types.insert("float".to_string(), PythonType::Float);
        self.builtin_types.insert("str".to_string(), PythonType::Str);
        self.builtin_types.insert("bool".to_string(), PythonType::Bool);
        self.builtin_types.insert("bytes".to_string(), PythonType::Bytes);
        self.builtin_types.insert("list".to_string(), PythonType::Generic("list".to_string(), vec![PythonType::Any]));
        self.builtin_types.insert("dict".to_string(), PythonType::Generic("dict".to_string(), vec![PythonType::Any, PythonType::Any]));
        self.builtin_types.insert("set".to_string(), PythonType::Generic("set".to_string(), vec![PythonType::Any]));
        self.builtin_types.insert("tuple".to_string(), PythonType::Generic("tuple".to_string(), vec![PythonType::Any]));
    }
}

#[derive(Debug, Clone)]
pub struct TypeError {
    pub error_type: TypeErrorType,
    pub message: String,
    pub range: SourceRange,
    pub expected_type: Option<PythonType>,
    pub actual_type: Option<PythonType>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum TypeErrorType {
    TypeMismatch,
    AttributeError,
    ArgumentError,
    ReturnTypeError,
    UnresolvedReference,
    IncompatibleTypes,
    InvalidOperation,
}
```

### 7.5 Import Resolution System

#### 7.5.1 Python Import Resolver
```rust
pub struct PythonImportResolver {
    module_cache: Arc<RwLock<HashMap<String, ModuleInfo>>>,
    python_path: Vec<PathBuf>,
    stdlib_modules: HashSet<String>,
    site_packages: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ImportStatement {
    pub module_name: String,
    pub alias: Option<String>,
    pub range: SourceRange,
    pub is_relative: bool,
    pub level: u32, // For relative imports
}

#[derive(Debug, Clone)]
pub struct FromImportStatement {
    pub module_name: String,
    pub imported_names: Vec<ImportedName>,
    pub range: SourceRange,
    pub is_relative: bool,
    pub level: u32,
}

#[derive(Debug, Clone)]
pub struct ImportedName {
    pub name: String,
    pub alias: Option<String>,
    pub range: SourceRange,
}

#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub name: String,
    pub file_path: Option<PathBuf>,
    pub is_package: bool,
    pub is_stdlib: bool,
    pub exports: Vec<String>,
    pub submodules: Vec<String>,
    pub dependencies: Vec<String>,
    pub docstring: Option<String>,
    pub version: Option<String>,
}

impl PythonImportResolver {
    pub fn new() -> Self {
        let mut resolver = Self {
            module_cache: Arc::new(RwLock::new(HashMap::new())),
            python_path: Vec::new(),
            stdlib_modules: HashSet::new(),
            site_packages: Vec::new(),
        };
        
        resolver.initialize_python_environment();
        resolver
    }
    
    pub async fn resolve_imports(&self, ast: &ast::Suite, file_path: &Path) -> Result<PythonImportInfo, ImportResolutionError> {
        let mut import_info = PythonImportInfo {
            imports: Vec::new(),
            from_imports: Vec::new(),
            resolved_modules: HashMap::new(),
            unresolved_imports: Vec::new(),
            circular_imports: Vec::new(),
            dependency_graph: DependencyGraph::new(),
        };
        
        // Extract import statements
        let (imports, from_imports) = self.extract_imports(ast)?;
        import_info.imports = imports;
        import_info.from_imports = from_imports;
        
        // Resolve each import
        for import in &import_info.imports {
            match self.resolve_module(&import.module_name, file_path, import.is_relative, import.level).await {
                Ok(module_info) => {
                    import_info.resolved_modules.insert(import.module_name.clone(), module_info);
                }
                Err(e) => {
                    import_info.unresolved_imports.push(UnresolvedImport {
                        module_name: import.module_name.clone(),
                        error: e.to_string(),
                        range: import.range.clone(),
                    });
                }
            }
        }
        
        for from_import in &import_info.from_imports {
            match self.resolve_module(&from_import.module_name, file_path, from_import.is_relative, from_import.level).await {
                Ok(module_info) => {
                    import_info.resolved_modules.insert(from_import.module_name.clone(), module_info);
                }
                Err(e) => {
                    import_info.unresolved_imports.push(UnresolvedImport {
                        module_name: from_import.module_name.clone(),
                        error: e.to_string(),
                        range: from_import.range.clone(),
                    });
                }
            }
        }
        
        // Build dependency graph
        import_info.dependency_graph = self.build_dependency_graph(&import_info).await?;
        
        // Detect circular imports
        import_info.circular_imports = self.detect_circular_imports(&import_info.dependency_graph)?;
        
        Ok(import_info)
    }
    
    async fn resolve_module(&self, module_name: &str, current_file: &Path, is_relative: bool, level: u32) -> Result<ModuleInfo, ImportResolutionError> {
        // Check cache first
        if let Some(cached_module) = self.module_cache.read().await.get(module_name) {
            return Ok(cached_module.clone());
        }
        
        let resolved_name = if is_relative {
            self.resolve_relative_import(module_name, current_file, level)?
        } else {
            module_name.to_string()
        };
        
        // Check if it's a stdlib module
        if self.stdlib_modules.contains(&resolved_name) {
            let module_info = ModuleInfo {
                name: resolved_name.clone(),
                file_path: None,
                is_package: false,
                is_stdlib: true,
                exports: self.get_stdlib_exports(&resolved_name),
                submodules: Vec::new(),
                dependencies: Vec::new(),
                docstring: None,
                version: None,
            };
            
            self.module_cache.write().await.insert(resolved_name, module_info.clone());
            return Ok(module_info);
        }
        
        // Try to find the module file
        if let Some(module_path) = self.find_module_file(&resolved_name, current_file).await? {
            let module_info = self.analyze_module_file(&module_path).await?;
            self.module_cache.write().await.insert(resolved_name, module_info.clone());
            Ok(module_info)
        } else {
            Err(ImportResolutionError::ModuleNotFound(resolved_name))
        }
    }
    
    fn resolve_relative_import(&self, module_name: &str, current_file: &Path, level: u32) -> Result<String, ImportResolutionError> {
        let current_dir = current_file.parent()
            .ok_or(ImportResolutionError::InvalidPath)?;
        
        let mut target_dir = current_dir;
        for _ in 0..level.saturating_sub(1) {
            target_dir = target_dir.parent()
                .ok_or(ImportResolutionError::RelativeImportError)?;
        }
        
        if module_name.is_empty() {
            // from . import ...
            Ok(self.path_to_module_name(target_dir)?)
        } else {
            // from .module import ...
            let module_path = target_dir.join(module_name);
            Ok(self.path_to_module_name(&module_path)?)
        }
    }
    
    async fn find_module_file(&self, module_name: &str, current_file: &Path) -> Result<Option<PathBuf>, ImportResolutionError> {
        let module_parts: Vec<&str> = module_name.split('.').collect();
        
        // Search in current directory first
        if let Some(current_dir) = current_file.parent() {
            if let Some(found) = self.search_for_module(&module_parts, current_dir).await? {
                return Ok(Some(found));
            }
        }
        
        // Search in Python path
        for path in &self.python_path {
            if let Some(found) = self.search_for_module(&module_parts, path).await? {
                return Ok(Some(found));
            }
        }
        
        // Search in site-packages
        for site_package in &self.site_packages {
            if let Some(found) = self.search_for_module(&module_parts, site_package).await? {
                return Ok(Some(found));
            }
        }
        
        Ok(None)
    }
    
    async fn search_for_module(&self, module_parts: &[&str], base_path: &Path) -> Result<Option<PathBuf>, ImportResolutionError> {
        let mut current_path = base_path.to_path_buf();
        
        for part in module_parts {
            current_path = current_path.join(part);
            
            // Try as a module file
            let module_file = current_path.with_extension("py");
            if module_file.exists() {
                return Ok(Some(module_file));
            }
            
            // Try as a package
            let init_file = current_path.join("__init__.py");
            if init_file.exists() {
                if module_parts.len() == 1 {
                    return Ok(Some(init_file));
                }
                // Continue searching in the package
                continue;
            }
            
            // Module part not found
            return Ok(None);
        }
        
        Ok(None)
    }
}
```

### 7.6 Data Flow Analysis

#### 7.6.1 Python Data Flow Analyzer
```rust
pub struct PythonDataFlowAnalyzer {
    cfg_builder: ControlFlowGraphBuilder,
    def_use_analyzer: DefUseAnalyzer,
    dead_code_detector: DeadCodeDetector,
}

#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    pub nodes: HashMap<NodeId, CFGNode>,
    pub edges: Vec<CFGEdge>,
    pub entry_node: NodeId,
    pub exit_nodes: Vec<NodeId>,
}

#[derive(Debug, Clone)]
pub struct CFGNode {
    pub id: NodeId,
    pub node_type: CFGNodeType,
    pub statement: Option<ast::Stmt>,
    pub expression: Option<ast::Expr>,
    pub range: SourceRange,
    pub predecessors: Vec<NodeId>,
    pub successors: Vec<NodeId>,
}

#[derive(Debug, Clone)]
pub enum CFGNodeType {
    Entry,
    Exit,
    Statement,
    Expression,
    Branch,
    Loop,
    FunctionCall,
    Return,
    Raise,
    Try,
    Except,
    Finally,
}

#[derive(Debug, Clone)]
pub struct DataFlowNode {
    pub variable_name: String,
    pub node_id: NodeId,
    pub flow_type: DataFlowType,
    pub range: SourceRange,
    pub value: Option<String>,
}

#[derive(Debug, Clone)]
pub enum DataFlowType {
    Definition,
    Use,
    Modification,
    Delete,
    Import,
}

#[derive(Debug, Clone)]
pub struct DefUseChain {
    pub variable_name: String,
    pub definitions: Vec<NodeId>,
    pub uses: Vec<NodeId>,
    pub live_ranges: Vec<LiveRange>,
}

#[derive(Debug, Clone)]
pub struct LiveRange {
    pub start_node: NodeId,
    pub end_node: NodeId,
    pub is_live: bool,
}

impl PythonDataFlowAnalyzer {
    pub fn new() -> Self {
        Self {
            cfg_builder: ControlFlowGraphBuilder::new(),
            def_use_analyzer: DefUseAnalyzer::new(),
            dead_code_detector: DeadCodeDetector::new(),
        }
    }
    
    pub async fn analyze_data_flow(&self, ast: &ast::Suite, content: &str) -> Result<PythonDataFlowInfo, DataFlowError> {
        // Build control flow graph
        let cfg = self.cfg_builder.build_cfg(ast)?;
        
        // Analyze variable flows
        let variable_flows = self.analyze_variable_flows(ast, &cfg)?;
        
        // Extract function calls
        let function_calls = self.extract_function_calls(ast)?;
        
        // Extract attribute accesses
        let attribute_accesses = self.extract_attribute_accesses(ast)?;
        
        // Build def-use chains
        let def_use_chains = self.def_use_analyzer.build_def_use_chains(ast, &cfg)?;
        
        // Detect dead code
        let dead_code_segments = self.dead_code_detector.detect_dead_code(&cfg, &def_use_chains)?;
        
        Ok(PythonDataFlowInfo {
            variable_flows,
            function_calls,
            attribute_accesses,
            control_flow_graph: cfg,
            def_use_chains,
            dead_code_segments,
        })
    }
    
    fn analyze_variable_flows(&self, ast: &ast::Suite, cfg: &ControlFlowGraph) -> Result<HashMap<String, Vec<DataFlowNode>>, DataFlowError> {
        let mut variable_flows = HashMap::new();
        let mut visitor = VariableFlowVisitor::new(&mut variable_flows, cfg);
        visitor.visit_suite(ast);
        Ok(variable_flows)
    }
    
    fn extract_function_calls(&self, ast: &ast::Suite) -> Result<Vec<FunctionCall>, DataFlowError> {
        let mut function_calls = Vec::new();
        let mut visitor = FunctionCallVisitor::new(&mut function_calls);
        visitor.visit_suite(ast);
        Ok(function_calls)
    }
    
    fn extract_attribute_accesses(&self, ast: &ast::Suite) -> Result<Vec<AttributeAccess>, DataFlowError> {
        let mut attribute_accesses = Vec::new();
        let mut visitor = AttributeAccessVisitor::new(&mut attribute_accesses);
        visitor.visit_suite(ast);
        Ok(attribute_accesses)
    }
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub function_name: String,
    pub range: SourceRange,
    pub arguments: Vec<Argument>,
    pub is_method_call: bool,
    pub receiver: Option<String>,
    pub return_type: Option<PythonType>,
}

#[derive(Debug, Clone)]
pub struct Argument {
    pub name: Option<String>, // For keyword arguments
    pub value: String,
    pub arg_type: Option<PythonType>,
    pub is_starred: bool,
    pub is_keyword: bool,
}

#[derive(Debug, Clone)]
pub struct AttributeAccess {
    pub object_name: String,
    pub attribute_name: String,
    pub range: SourceRange,
    pub access_type: AttributeAccessType,
    pub object_type: Option<PythonType>,
}

#[derive(Debug, Clone)]
pub enum AttributeAccessType {
    Read,
    Write,
    Delete,
    Call,
}
```

### 7.7 Python Pattern Detection

#### 7.7.1 Python-Specific Pattern Detector
```rust
pub struct PythonPatternDetector {
    patterns: Vec<Box<dyn PythonPattern>>,
    config: PatternDetectionConfig,
}

#[derive(Debug, Clone)]
pub struct PatternDetectionConfig {
    pub enable_antipatterns: bool,
    pub enable_best_practices: bool,
    pub enable_performance_patterns: bool,
    pub enable_security_patterns: bool,
    pub python_version: PythonVersion,
}

#[async_trait]
pub trait PythonPattern: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn category(&self) -> PatternCategory;
    fn severity(&self) -> PatternSeverity;
    
    async fn detect(&self, analysis_result: &PythonAnalysisResult) -> Result<Vec<PatternMatch>, PatternError>;
}

#[derive(Debug, Clone)]
pub enum PatternCategory {
    Antipattern,
    BestPractice,
    Performance,
    Security,
    Maintainability,
    Readability,
    Pythonic,
}

#[derive(Debug, Clone)]
pub enum PatternSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_name: String,
    pub category: PatternCategory,
    pub severity: PatternSeverity,
    pub message: String,
    pub range: SourceRange,
    pub suggestion: Option<String>,
    pub confidence: f64,
    pub metadata: HashMap<String, String>,
}

// Example: Mutable Default Argument Pattern
pub struct MutableDefaultArgumentPattern;

#[async_trait]
impl PythonPattern for MutableDefaultArgumentPattern {
    fn name(&self) -> &str {
        "mutable_default_argument"
    }
    
    fn description(&self) -> &str {
        "Detects mutable default arguments in function definitions"
    }
    
    fn category(&self) -> PatternCategory {
        PatternCategory::Antipattern
    }
    
    fn severity(&self) -> PatternSeverity {
        PatternSeverity::High
    }
    
    async fn detect(&self, analysis_result: &PythonAnalysisResult) -> Result<Vec<PatternMatch>, PatternError> {
        let mut matches = Vec::new();
        
        if let Some(semantic_info) = &analysis_result.semantic_info {
            for function_def in &semantic_info.function_definitions {
                for parameter in &function_def.parameters {
                    if let Some(default_value) = &parameter.default_value {
                        if self.is_mutable_type(default_value) {
                            matches.push(PatternMatch {
                                pattern_name: self.name().to_string(),
                                category: self.category(),
                                severity: self.severity(),
                                message: format!(
                                    "Function '{}' has mutable default argument '{}'. This can lead to unexpected behavior.",
                                    function_def.name, parameter.name
                                ),
                                range: parameter.range.clone(),
                                suggestion: Some(format!(
                                    "Use 'None' as default and check for None inside the function: 'if {} is None: {} = {}'",
                                    parameter.name, parameter.name, default_value
                                )),
                                confidence: 0.95,
                                metadata: HashMap::from([
                                    ("function_name".to_string(), function_def.name.clone()),
                                    ("parameter_name".to_string(), parameter.name.clone()),
                                ]),
                            });
                        }
                    }
                }
            }
        }
        
        Ok(matches)
    }
}

impl MutableDefaultArgumentPattern {
    fn is_mutable_type(&self, expr: &ast::Expr) -> bool {
        match expr {
            ast::Expr::List(_) | ast::Expr::Dict(_) | ast::Expr::Set(_) => true,
            ast::Expr::Call(call) => {
                if let ast::Expr::Name(name) = &*call.func {
                    matches!(name.id.as_str(), "list" | "dict" | "set")
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

// Example: List Comprehension Pattern
pub struct ListComprehensionPattern;

#[async_trait]
impl PythonPattern for ListComprehensionPattern {
    fn name(&self) -> &str {
        "list_comprehension_opportunity"
    }
    
    fn description(&self) -> &str {
        "Suggests using list comprehensions instead of explicit loops"
    }
    
    fn category(&self) -> PatternCategory {
        PatternCategory::Pythonic
    }
    
    fn severity(&self) -> PatternSeverity {
        PatternSeverity::Low
    }
    
    async fn detect(&self, analysis_result: &PythonAnalysisResult) -> Result<Vec<PatternMatch>, PatternError> {
        let mut matches = Vec::new();
        
        if let Some(ast) = &analysis_result.rustpython_ast {
            let mut visitor = ListComprehensionVisitor::new(&mut matches);
            visitor.visit_suite(ast);
        }
        
        Ok(matches)
    }
}

// Additional patterns to implement:
// - UnusedImportPattern
// - TooManyArgumentsPattern
// - DeepNestingPattern
// - StringConcatenationPattern
// - ExceptWithoutExceptionPattern
// - BareExceptPattern
// - GlobalVariablePattern
// - LongFunctionPattern
// - ComplexConditionPattern
// - DuplicateCodePattern
```

### 7.8 Python Metrics Calculator

#### 7.8.1 Python-Specific Metrics
```rust
#[derive(Debug, Clone, Default)]
pub struct PythonMetrics {
    pub lines_of_code: u32,
    pub logical_lines_of_code: u32,
    pub comment_lines: u32,
    pub blank_lines: u32,
    pub cyclomatic_complexity: u32,
    pub cognitive_complexity: u32,
    pub halstead_metrics: HalsteadMetrics,
    pub maintainability_index: f64,
    
    // Python-specific metrics
    pub function_count: u32,
    pub class_count: u32,
    pub method_count: u32,
    pub import_count: u32,
    pub decorator_count: u32,
    pub comprehension_count: u32,
    pub lambda_count: u32,
    pub generator_count: u32,
    
    // Quality metrics
    pub docstring_coverage: f64,
    pub type_annotation_coverage: f64,
    pub test_coverage: f64,
    pub duplication_percentage: f64,
    
    // Complexity metrics
    pub max_function_complexity: u32,
    pub average_function_complexity: f64,
    pub max_class_complexity: u32,
    pub nesting_depth: u32,
    pub inheritance_depth: u32,
}

#[derive(Debug, Clone, Default)]
pub struct HalsteadMetrics {
    pub distinct_operators: u32,
    pub distinct_operands: u32,
    pub total_operators: u32,
    pub total_operands: u32,
    pub vocabulary: u32,
    pub length: u32,
    pub calculated_length: f64,
    pub volume: f64,
    pub difficulty: f64,
    pub effort: f64,
    pub time: f64,
    pub bugs: f64,
}

pub struct PythonMetricsCalculator {
    complexity_calculator: ComplexityCalculator,
    halstead_calculator: HalsteadCalculator,
}

impl PythonMetricsCalculator {
    pub fn new() -> Self {
        Self {
            complexity_calculator: ComplexityCalculator::new(),
            halstead_calculator: HalsteadCalculator::new(),
        }
    }
    
    pub fn calculate_metrics(&self, analysis_result: &PythonAnalysisResult) -> PythonMetrics {
        let mut metrics = PythonMetrics::default();
        
        // Basic line metrics
        let content = std::fs::read_to_string(&analysis_result.file_path).unwrap_or_default();
        metrics.lines_of_code = content.lines().count() as u32;
        metrics.comment_lines = self.count_comment_lines(&content);
        metrics.blank_lines = self.count_blank_lines(&content);
        metrics.logical_lines_of_code = metrics.lines_of_code - metrics.comment_lines - metrics.blank_lines;
        
        if let Some(ast) = &analysis_result.rustpython_ast {
            // Complexity metrics
            metrics.cyclomatic_complexity = self.complexity_calculator.calculate_cyclomatic_complexity(ast);
            metrics.cognitive_complexity = self.complexity_calculator.calculate_cognitive_complexity(ast);
            
            // Halstead metrics
            metrics.halstead_metrics = self.halstead_calculator.calculate_halstead_metrics(ast);
            
            // Structure metrics
            metrics.function_count = self.count_functions(ast);
            metrics.class_count = self.count_classes(ast);
            metrics.method_count = self.count_methods(ast);
            metrics.import_count = self.count_imports(ast);
            metrics.decorator_count = self.count_decorators(ast);
            metrics.comprehension_count = self.count_comprehensions(ast);
            metrics.lambda_count = self.count_lambdas(ast);
            metrics.generator_count = self.count_generators(ast);
            
            // Nesting and inheritance
            metrics.nesting_depth = self.calculate_max_nesting_depth(ast);
            metrics.inheritance_depth = self.calculate_inheritance_depth(ast);
        }
        
        if let Some(semantic_info) = &analysis_result.semantic_info {
            // Documentation coverage
            metrics.docstring_coverage = self.calculate_docstring_coverage(semantic_info);
            
            // Function complexity statistics
            let complexities: Vec<u32> = semantic_info.function_definitions
                .iter()
                .map(|f| f.complexity)
                .collect();
            
            metrics.max_function_complexity = complexities.iter().max().copied().unwrap_or(0);
            metrics.average_function_complexity = if !complexities.is_empty() {
                complexities.iter().sum::<u32>() as f64 / complexities.len() as f64
            } else {
                0.0
            };
        }
        
        if let Some(type_info) = &analysis_result.type_info {
            // Type annotation coverage
            metrics.type_annotation_coverage = self.calculate_type_annotation_coverage(type_info);
        }
        
        // Maintainability index
        metrics.maintainability_index = self.calculate_maintainability_index(&metrics);
        
        metrics
    }
    
    fn calculate_maintainability_index(&self, metrics: &PythonMetrics) -> f64 {
        // Microsoft's Maintainability Index formula (adapted for Python)
        let halstead_volume = metrics.halstead_metrics.volume.max(1.0);
        let cyclomatic_complexity = metrics.cyclomatic_complexity.max(1) as f64;
        let lines_of_code = metrics.logical_lines_of_code.max(1) as f64;
        
        let mi = 171.0 
            - 5.2 * halstead_volume.ln()
            - 0.23 * cyclomatic_complexity
            - 16.2 * lines_of_code.ln();
        
        // Normalize to 0-100 scale
        (mi * 100.0 / 171.0).max(0.0).min(100.0)
    }
    
    fn calculate_docstring_coverage(&self, semantic_info: &PythonSemanticInfo) -> f64 {
        let total_functions = semantic_info.function_definitions.len() + semantic_info.class_definitions.len();
        if total_functions == 0 {
            return 100.0;
        }
        
        let documented_functions = semantic_info.function_definitions
            .iter()
            .filter(|f| f.docstring.is_some())
            .count() + semantic_info.class_definitions
            .iter()
            .filter(|c| c.docstring.is_some())
            .count();
        
        (documented_functions as f64 / total_functions as f64) * 100.0
    }
    
    fn calculate_type_annotation_coverage(&self, type_info: &PythonTypeInfo) -> f64 {
        let total_annotations = type_info.type_annotations.len() + type_info.inferred_types.len();
        if total_annotations == 0 {
            return 0.0;
        }
        
        let explicit_annotations = type_info.type_annotations.len();
        (explicit_annotations as f64 / total_annotations as f64) * 100.0
    }
}
```

### 7.9 Criterios de Completitud

#### 7.9.1 Entregables de la Fase
- [ ] Parser Python especializado implementado
- [ ] Análisis semántico profundo funcionando
- [ ] Sistema de inferencia de tipos
- [ ] Resolución de imports y dependencias
- [ ] Análisis de flujo de datos
- [ ] Detección de patrones Python-específicos
- [ ] Cálculo de métricas avanzadas
- [ ] Integration con parser universal
- [ ] Tests comprehensivos para Python
- [ ] Documentación completa

#### 7.9.2 Criterios de Aceptación
- [ ] Parse correctamente código Python complejo
- [ ] Análisis semántico identifica scopes y símbolos
- [ ] Inferencia de tipos funciona en casos típicos
- [ ] Resolución de imports maneja casos complejos
- [ ] Data flow analysis detecta variables no utilizadas
- [ ] Pattern detection encuentra antipatrones comunes
- [ ] Métricas calculadas son precisas
- [ ] Performance acceptable para archivos grandes
- [ ] Error handling robusto
- [ ] Integration seamless con sistema principal

### 7.10 Performance Targets

#### 7.10.1 Benchmarks Específicos Python
- **Parsing speed**: >500 lines/second para Python
- **Semantic analysis**: <2x overhead sobre parsing básico
- **Type inference**: <3x overhead sobre parsing básico
- **Memory usage**: <20MB por archivo Python típico
- **Pattern detection**: <1 segundo para archivos <1000 lines

### 7.11 Estimación de Tiempo

#### 7.11.1 Breakdown de Tareas
- Setup RustPython integration: 3 días
- Semantic analysis engine: 7 días
- Type inference system: 8 días
- Import resolution system: 6 días
- Data flow analysis: 7 días
- Pattern detection framework: 6 días
- Python-specific patterns (10+): 8 días
- Metrics calculation: 4 días
- Integration con parser universal: 3 días
- Testing comprehensivo: 6 días
- Performance optimization: 4 días
- Documentación: 3 días

**Total estimado: 65 días de desarrollo**

### 7.12 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Análisis Python extremadamente profundo
- Capacidades de análisis semántico avanzado
- Detección de patrones específicos de Python
- Foundation para análisis de calidad de código Python
- Base para implementar reglas de análisis específicas

La Fase 8 implementará capacidades similares para TypeScript/JavaScript, adaptando las técnicas desarrolladas aquí.
