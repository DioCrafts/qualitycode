# Fase 8: Parser Especializado para TypeScript/JavaScript

## Objetivo General
Implementar un parser especializado para TypeScript y JavaScript que proporcione análisis semántico profundo, inferencia de tipos, análisis de módulos ES6+, detección de patrones específicos del ecosistema JS/TS, y capacidades avanzadas de inspección de código moderno necesarias para el agente CodeAnt.

## Descripción Técnica Detallada

### 8.1 Arquitectura del Parser TypeScript/JavaScript

#### 8.1.1 Diseño del Sistema JS/TS
```
┌─────────────────────────────────────────┐
│      TypeScript/JavaScript Parser      │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │    SWC      │ │   TypeScript        │ │
│  │   Parser    │ │   Compiler API      │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Module    │ │    Type System      │ │
│  │  Resolver   │ │    Analysis         │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   React     │ │     Node.js         │ │
│  │  Analysis   │ │    Analysis         │ │
│  └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
```

#### 8.1.2 Componentes Especializados
- **SWC Parser**: Parser de alta performance para JS/TS
- **TypeScript Compiler API**: Análisis de tipos TypeScript
- **Module Resolver**: Resolución de módulos ES6/CommonJS
- **Type System Analysis**: Análisis del sistema de tipos
- **React Analysis**: Análisis específico para React
- **Node.js Analysis**: Análisis para código Node.js
- **Modern JS Features**: Soporte para ES2015+

### 8.2 Core Parser Implementation

#### 8.2.1 TypeScript/JavaScript Specialized Parser
```rust
use swc_core::{
    common::{SourceMap, Span, DUMMY_SP},
    ecma::{
        ast::*,
        parser::{lexer::Lexer, Parser, StringInput, Syntax, TsConfig},
        transforms::base::resolver,
        visit::{Visit, VisitWith},
    },
};

pub struct TypeScriptJavaScriptParser {
    swc_parser: SwcParser,
    typescript_analyzer: Arc<TypeScriptAnalyzer>,
    module_resolver: Arc<ModuleResolver>,
    react_analyzer: Arc<ReactAnalyzer>,
    nodejs_analyzer: Arc<NodeJSAnalyzer>,
    config: JSParserConfig,
}

#[derive(Debug, Clone)]
pub struct JSParserConfig {
    pub language: JSLanguage,
    pub syntax_config: JSSyntaxConfig,
    pub enable_type_checking: bool,
    pub enable_module_resolution: bool,
    pub enable_react_analysis: bool,
    pub enable_nodejs_analysis: bool,
    pub target_version: ECMAScriptVersion,
    pub module_system: ModuleSystem,
    pub jsx_factory: String,
    pub strict_mode: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JSLanguage {
    JavaScript,
    TypeScript,
    JSX,
    TSX,
}

#[derive(Debug, Clone)]
pub struct JSSyntaxConfig {
    pub jsx: bool,
    pub typescript: bool,
    pub decorators: bool,
    pub dynamic_import: bool,
    pub private_methods: bool,
    pub class_properties: bool,
    pub optional_chaining: bool,
    pub nullish_coalescing: bool,
    pub top_level_await: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ECMAScriptVersion {
    ES5,
    ES2015,
    ES2016,
    ES2017,
    ES2018,
    ES2019,
    ES2020,
    ES2021,
    ES2022,
    ES2023,
    ESNext,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModuleSystem {
    CommonJS,
    ES6,
    AMD,
    UMD,
    SystemJS,
}

impl TypeScriptJavaScriptParser {
    pub async fn new(config: JSParserConfig) -> Result<Self, JSParserError> {
        let swc_parser = SwcParser::new(config.clone());
        let typescript_analyzer = Arc::new(TypeScriptAnalyzer::new());
        let module_resolver = Arc::new(ModuleResolver::new(config.module_system.clone()));
        let react_analyzer = Arc::new(ReactAnalyzer::new());
        let nodejs_analyzer = Arc::new(NodeJSAnalyzer::new());
        
        Ok(Self {
            swc_parser,
            typescript_analyzer,
            module_resolver,
            react_analyzer,
            nodejs_analyzer,
            config,
        })
    }
    
    pub async fn parse_js_file(&self, file_path: &Path) -> Result<JSAnalysisResult, JSParserError> {
        let start_time = Instant::now();
        
        // Read and detect file type
        let content = tokio::fs::read_to_string(file_path).await
            .map_err(|e| JSParserError::FileReadError(e.to_string()))?;
        
        let detected_language = self.detect_js_language(file_path, &content)?;
        
        // Parse with SWC
        let swc_result = self.swc_parser.parse(&content, detected_language.clone())?;
        
        // Create analysis result
        let mut analysis_result = JSAnalysisResult {
            file_path: file_path.to_path_buf(),
            language: detected_language,
            swc_ast: swc_result.clone(),
            type_info: None,
            module_info: None,
            react_info: None,
            nodejs_info: None,
            semantic_info: None,
            patterns: Vec::new(),
            metrics: JSMetrics::default(),
            parse_duration_ms: 0,
        };
        
        // TypeScript analysis
        if self.config.enable_type_checking && matches!(detected_language, JSLanguage::TypeScript | JSLanguage::TSX) {
            analysis_result.type_info = Some(
                self.typescript_analyzer.analyze_types(&swc_result, &content).await?
            );
        }
        
        // Module resolution
        if self.config.enable_module_resolution {
            analysis_result.module_info = Some(
                self.module_resolver.resolve_modules(&swc_result, file_path).await?
            );
        }
        
        // React analysis
        if self.config.enable_react_analysis && self.is_react_file(&swc_result) {
            analysis_result.react_info = Some(
                self.react_analyzer.analyze_react(&swc_result, &content).await?
            );
        }
        
        // Node.js analysis
        if self.config.enable_nodejs_analysis && self.is_nodejs_file(&swc_result) {
            analysis_result.nodejs_info = Some(
                self.nodejs_analyzer.analyze_nodejs(&swc_result, &content).await?
            );
        }
        
        // Semantic analysis
        analysis_result.semantic_info = Some(
            self.analyze_semantics(&swc_result, &content).await?
        );
        
        // Pattern detection
        analysis_result.patterns = self.detect_js_patterns(&analysis_result).await?;
        
        // Calculate metrics
        analysis_result.metrics = self.calculate_js_metrics(&analysis_result);
        analysis_result.parse_duration_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(analysis_result)
    }
    
    fn detect_js_language(&self, file_path: &Path, content: &str) -> Result<JSLanguage, JSParserError> {
        // Check file extension first
        if let Some(extension) = file_path.extension().and_then(|ext| ext.to_str()) {
            match extension.to_lowercase().as_str() {
                "ts" => return Ok(JSLanguage::TypeScript),
                "tsx" => return Ok(JSLanguage::TSX),
                "jsx" => return Ok(JSLanguage::JSX),
                "js" | "mjs" | "cjs" => {
                    // Check content for TypeScript features
                    if self.has_typescript_syntax(content) {
                        return Ok(JSLanguage::TypeScript);
                    }
                    if self.has_jsx_syntax(content) {
                        return Ok(JSLanguage::JSX);
                    }
                    return Ok(JSLanguage::JavaScript);
                }
                _ => {}
            }
        }
        
        // Fallback to content analysis
        if self.has_typescript_syntax(content) {
            if self.has_jsx_syntax(content) {
                Ok(JSLanguage::TSX)
            } else {
                Ok(JSLanguage::TypeScript)
            }
        } else if self.has_jsx_syntax(content) {
            Ok(JSLanguage::JSX)
        } else {
            Ok(JSLanguage::JavaScript)
        }
    }
    
    fn has_typescript_syntax(&self, content: &str) -> bool {
        // Check for TypeScript-specific syntax
        let ts_patterns = [
            r":\s*\w+\s*[=;,\)]",  // Type annotations
            r"interface\s+\w+",     // Interface declarations
            r"type\s+\w+\s*=",      // Type aliases
            r"enum\s+\w+",          // Enums
            r"namespace\s+\w+",     // Namespaces
            r"declare\s+",          // Declare statements
            r"abstract\s+class",    // Abstract classes
            r"implements\s+\w+",    // Implements clause
            r"<\w+>",               // Generic type parameters
            r"as\s+\w+",            // Type assertions
        ];
        
        ts_patterns.iter().any(|pattern| {
            regex::Regex::new(pattern).unwrap().is_match(content)
        })
    }
    
    fn has_jsx_syntax(&self, content: &str) -> bool {
        // Check for JSX syntax
        let jsx_patterns = [
            r"<\w+[^>]*>",          // JSX elements
            r"</\w+>",              // JSX closing tags
            r"<\w+\s*/\s*>",        // Self-closing JSX tags
            r"{\s*\w+\s*}",         // JSX expressions (basic check)
        ];
        
        jsx_patterns.iter().any(|pattern| {
            regex::Regex::new(pattern).unwrap().is_match(content)
        })
    }
}
```

#### 8.2.2 SWC Parser Integration
```rust
pub struct SwcParser {
    source_map: Arc<SourceMap>,
    config: JSParserConfig,
}

impl SwcParser {
    pub fn new(config: JSParserConfig) -> Self {
        Self {
            source_map: Arc::new(SourceMap::default()),
            config,
        }
    }
    
    pub fn parse(&self, content: &str, language: JSLanguage) -> Result<Program, JSParserError> {
        let syntax = self.create_syntax_config(language);
        
        let lexer = Lexer::new(
            syntax,
            ECMAScriptVersion::Es2022.into(),
            StringInput::new(content, BytePos(0), BytePos(content.len() as u32)),
            None,
        );
        
        let mut parser = Parser::new_from(lexer);
        
        match parser.parse_program() {
            Ok(program) => Ok(program),
            Err(error) => Err(JSParserError::SWCParseError(error.to_string())),
        }
    }
    
    fn create_syntax_config(&self, language: JSLanguage) -> Syntax {
        match language {
            JSLanguage::TypeScript | JSLanguage::TSX => {
                Syntax::Typescript(TsConfig {
                    tsx: matches!(language, JSLanguage::TSX),
                    decorators: self.config.syntax_config.decorators,
                    dts: false,
                    no_early_errors: false,
                })
            }
            JSLanguage::JavaScript | JSLanguage::JSX => {
                Syntax::Es(EsConfig {
                    jsx: matches!(language, JSLanguage::JSX) || self.config.syntax_config.jsx,
                    fn_bind: false,
                    decorators: self.config.syntax_config.decorators,
                    decorators_before_export: true,
                    export_default_from: true,
                    import_assertions: true,
                    private_in_object: self.config.syntax_config.private_methods,
                    allow_super_outside_method: false,
                    allow_return_outside_function: false,
                })
            }
        }
    }
}
```

### 8.3 TypeScript Type System Analysis

#### 8.3.1 TypeScript Analyzer
```rust
pub struct TypeScriptAnalyzer {
    type_checker: TypeChecker,
    interface_analyzer: InterfaceAnalyzer,
    generic_analyzer: GenericAnalyzer,
}

#[derive(Debug, Clone)]
pub struct JSTypeInfo {
    pub type_annotations: HashMap<Span, TSType>,
    pub inferred_types: HashMap<Span, InferredJSType>,
    pub interfaces: Vec<InterfaceDeclaration>,
    pub type_aliases: Vec<TypeAliasDeclaration>,
    pub enums: Vec<EnumDeclaration>,
    pub generics: Vec<GenericDeclaration>,
    pub type_errors: Vec<TSTypeError>,
    pub type_guards: Vec<TypeGuard>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TSType {
    // Primitive types
    String,
    Number,
    Boolean,
    BigInt,
    Symbol,
    Undefined,
    Null,
    Void,
    Never,
    Any,
    Unknown,
    
    // Object types
    Object(HashMap<String, TSType>),
    Array(Box<TSType>),
    Tuple(Vec<TSType>),
    
    // Function types
    Function {
        parameters: Vec<Parameter>,
        return_type: Box<TSType>,
        is_async: bool,
    },
    
    // Advanced types
    Union(Vec<TSType>),
    Intersection(Vec<TSType>),
    Conditional {
        check_type: Box<TSType>,
        extends_type: Box<TSType>,
        true_type: Box<TSType>,
        false_type: Box<TSType>,
    },
    Mapped {
        key_type: Box<TSType>,
        value_type: Box<TSType>,
        optional: bool,
        readonly: bool,
    },
    
    // References
    TypeReference {
        name: String,
        type_arguments: Vec<TSType>,
    },
    
    // Literals
    StringLiteral(String),
    NumberLiteral(f64),
    BooleanLiteral(bool),
    
    // Utility types
    Partial(Box<TSType>),
    Required(Box<TSType>),
    Pick(Box<TSType>, Vec<String>),
    Omit(Box<TSType>, Vec<String>),
    Record(Box<TSType>, Box<TSType>),
    
    // Generic
    TypeParameter(String),
}

#[derive(Debug, Clone)]
pub struct InferredJSType {
    pub ts_type: TSType,
    pub confidence: f64,
    pub source: JSTypeSource,
    pub constraints: Vec<TypeConstraint>,
    pub nullable: bool,
}

#[derive(Debug, Clone)]
pub enum JSTypeSource {
    TypeAnnotation,
    LiteralValue,
    FunctionReturn,
    VariableAssignment,
    PropertyAccess,
    FunctionCall,
    ArrayAccess,
    Destructuring,
    Import,
    ControlFlowAnalysis,
}

impl TypeScriptAnalyzer {
    pub fn new() -> Self {
        Self {
            type_checker: TypeChecker::new(),
            interface_analyzer: InterfaceAnalyzer::new(),
            generic_analyzer: GenericAnalyzer::new(),
        }
    }
    
    pub async fn analyze_types(&self, program: &Program, content: &str) -> Result<JSTypeInfo, TypeAnalysisError> {
        let mut type_info = JSTypeInfo {
            type_annotations: HashMap::new(),
            inferred_types: HashMap::new(),
            interfaces: Vec::new(),
            type_aliases: Vec::new(),
            enums: Vec::new(),
            generics: Vec::new(),
            type_errors: Vec::new(),
            type_guards: Vec::new(),
        };
        
        // Extract type annotations
        let mut type_visitor = TypeAnnotationVisitor::new();
        program.visit_with(&mut type_visitor);
        type_info.type_annotations = type_visitor.annotations;
        
        // Extract interface declarations
        let mut interface_visitor = InterfaceVisitor::new();
        program.visit_with(&mut interface_visitor);
        type_info.interfaces = interface_visitor.interfaces;
        
        // Extract type aliases
        let mut type_alias_visitor = TypeAliasVisitor::new();
        program.visit_with(&mut type_alias_visitor);
        type_info.type_aliases = type_alias_visitor.type_aliases;
        
        // Extract enums
        let mut enum_visitor = EnumVisitor::new();
        program.visit_with(&mut enum_visitor);
        type_info.enums = enum_visitor.enums;
        
        // Infer types
        type_info.inferred_types = self.type_checker.infer_types(program, &type_info.type_annotations)?;
        
        // Analyze generics
        type_info.generics = self.generic_analyzer.analyze_generics(program)?;
        
        // Detect type errors
        type_info.type_errors = self.detect_type_errors(program, &type_info)?;
        
        // Extract type guards
        type_info.type_guards = self.extract_type_guards(program)?;
        
        Ok(type_info)
    }
    
    fn detect_type_errors(&self, program: &Program, type_info: &JSTypeInfo) -> Result<Vec<TSTypeError>, TypeAnalysisError> {
        let mut errors = Vec::new();
        let mut error_detector = TypeErrorDetector::new(&mut errors, type_info);
        program.visit_with(&mut error_detector);
        Ok(errors)
    }
    
    fn extract_type_guards(&self, program: &Program) -> Result<Vec<TypeGuard>, TypeAnalysisError> {
        let mut type_guards = Vec::new();
        let mut guard_visitor = TypeGuardVisitor::new(&mut type_guards);
        program.visit_with(&mut guard_visitor);
        Ok(type_guards)
    }
}

#[derive(Debug, Clone)]
pub struct InterfaceDeclaration {
    pub name: String,
    pub span: Span,
    pub properties: Vec<InterfaceProperty>,
    pub methods: Vec<InterfaceMethod>,
    pub extends: Vec<String>,
    pub type_parameters: Vec<String>,
    pub is_exported: bool,
}

#[derive(Debug, Clone)]
pub struct InterfaceProperty {
    pub name: String,
    pub property_type: TSType,
    pub optional: bool,
    pub readonly: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct InterfaceMethod {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: TSType,
    pub optional: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct TypeGuard {
    pub parameter_name: String,
    pub guarded_type: TSType,
    pub condition: String,
    pub span: Span,
}
```

### 8.4 Module Resolution System

#### 8.4.1 ES6/CommonJS Module Resolver
```rust
pub struct ModuleResolver {
    module_system: ModuleSystem,
    resolution_cache: Arc<RwLock<HashMap<String, ModuleResolution>>>,
    node_modules_paths: Vec<PathBuf>,
    tsconfig_paths: HashMap<String, Vec<PathBuf>>,
}

#[derive(Debug, Clone)]
pub struct JSModuleInfo {
    pub imports: Vec<ImportDeclaration>,
    pub exports: Vec<ExportDeclaration>,
    pub dynamic_imports: Vec<DynamicImport>,
    pub resolved_modules: HashMap<String, ResolvedModule>,
    pub unresolved_modules: Vec<UnresolvedModule>,
    pub circular_dependencies: Vec<CircularDependency>,
    pub dependency_graph: ModuleDependencyGraph,
    pub module_type: ModuleType,
}

#[derive(Debug, Clone)]
pub enum ModuleType {
    ES6Module,
    CommonJSModule,
    AMDModule,
    UMDModule,
    SystemJSModule,
    IIFEModule,
}

#[derive(Debug, Clone)]
pub struct ImportDeclaration {
    pub module_specifier: String,
    pub import_type: ImportType,
    pub imported_names: Vec<ImportedName>,
    pub span: Span,
    pub is_type_only: bool,
}

#[derive(Debug, Clone)]
pub enum ImportType {
    DefaultImport(String),
    NamespaceImport(String),
    NamedImports,
    SideEffectImport,
}

#[derive(Debug, Clone)]
pub struct ExportDeclaration {
    pub export_type: ExportType,
    pub exported_names: Vec<ExportedName>,
    pub module_specifier: Option<String>,
    pub span: Span,
    pub is_type_only: bool,
}

#[derive(Debug, Clone)]
pub enum ExportType {
    DefaultExport,
    NamedExports,
    NamespaceExport,
    ReExport,
}

#[derive(Debug, Clone)]
pub struct DynamicImport {
    pub module_specifier: String,
    pub span: Span,
    pub is_conditional: bool,
    pub condition: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedModule {
    pub specifier: String,
    pub resolved_path: PathBuf,
    pub module_type: ModuleType,
    pub exports: Vec<String>,
    pub is_external: bool,
    pub is_builtin: bool,
    pub package_json: Option<PackageJson>,
}

impl ModuleResolver {
    pub fn new(module_system: ModuleSystem) -> Self {
        Self {
            module_system,
            resolution_cache: Arc::new(RwLock::new(HashMap::new())),
            node_modules_paths: Vec::new(),
            tsconfig_paths: HashMap::new(),
        }
    }
    
    pub async fn resolve_modules(&self, program: &Program, file_path: &Path) -> Result<JSModuleInfo, ModuleResolutionError> {
        let mut module_info = JSModuleInfo {
            imports: Vec::new(),
            exports: Vec::new(),
            dynamic_imports: Vec::new(),
            resolved_modules: HashMap::new(),
            unresolved_modules: Vec::new(),
            circular_dependencies: Vec::new(),
            dependency_graph: ModuleDependencyGraph::new(),
            module_type: self.detect_module_type(program),
        };
        
        // Extract import/export declarations
        let mut module_visitor = ModuleVisitor::new();
        program.visit_with(&mut module_visitor);
        
        module_info.imports = module_visitor.imports;
        module_info.exports = module_visitor.exports;
        module_info.dynamic_imports = module_visitor.dynamic_imports;
        
        // Resolve each import
        for import in &module_info.imports {
            match self.resolve_module_specifier(&import.module_specifier, file_path).await {
                Ok(resolved) => {
                    module_info.resolved_modules.insert(import.module_specifier.clone(), resolved);
                }
                Err(e) => {
                    module_info.unresolved_modules.push(UnresolvedModule {
                        specifier: import.module_specifier.clone(),
                        error: e.to_string(),
                        span: import.span,
                    });
                }
            }
        }
        
        // Resolve dynamic imports
        for dynamic_import in &module_info.dynamic_imports {
            if !module_info.resolved_modules.contains_key(&dynamic_import.module_specifier) {
                match self.resolve_module_specifier(&dynamic_import.module_specifier, file_path).await {
                    Ok(resolved) => {
                        module_info.resolved_modules.insert(dynamic_import.module_specifier.clone(), resolved);
                    }
                    Err(e) => {
                        module_info.unresolved_modules.push(UnresolvedModule {
                            specifier: dynamic_import.module_specifier.clone(),
                            error: e.to_string(),
                            span: dynamic_import.span,
                        });
                    }
                }
            }
        }
        
        // Build dependency graph
        module_info.dependency_graph = self.build_dependency_graph(&module_info, file_path).await?;
        
        // Detect circular dependencies
        module_info.circular_dependencies = self.detect_circular_dependencies(&module_info.dependency_graph)?;
        
        Ok(module_info)
    }
    
    async fn resolve_module_specifier(&self, specifier: &str, current_file: &Path) -> Result<ResolvedModule, ModuleResolutionError> {
        // Check cache first
        let cache_key = format!("{}:{}", current_file.display(), specifier);
        if let Some(cached) = self.resolution_cache.read().await.get(&cache_key) {
            return Ok(cached.resolved_module.clone());
        }
        
        let resolved = if specifier.starts_with('.') || specifier.starts_with('/') {
            // Relative or absolute path
            self.resolve_relative_module(specifier, current_file).await?
        } else if self.is_builtin_module(specifier) {
            // Node.js built-in module
            self.resolve_builtin_module(specifier)?
        } else {
            // Package from node_modules
            self.resolve_package_module(specifier, current_file).await?
        };
        
        // Cache the result
        self.resolution_cache.write().await.insert(
            cache_key,
            ModuleResolution {
                resolved_module: resolved.clone(),
                resolved_at: Utc::now(),
            },
        );
        
        Ok(resolved)
    }
    
    async fn resolve_relative_module(&self, specifier: &str, current_file: &Path) -> Result<ResolvedModule, ModuleResolutionError> {
        let current_dir = current_file.parent()
            .ok_or(ModuleResolutionError::InvalidPath)?;
        
        let module_path = if specifier.starts_with('/') {
            PathBuf::from(specifier)
        } else {
            current_dir.join(specifier)
        };
        
        // Try different extensions
        let extensions = ["", ".js", ".ts", ".jsx", ".tsx", ".json", ".mjs", ".cjs"];
        
        for ext in &extensions {
            let candidate = if ext.is_empty() {
                module_path.clone()
            } else {
                module_path.with_extension(&ext[1..])
            };
            
            if candidate.exists() {
                return self.create_resolved_module(specifier, candidate, false).await;
            }
        }
        
        // Try as directory with index file
        if module_path.is_dir() {
            for ext in &["js", "ts", "jsx", "tsx"] {
                let index_file = module_path.join(format!("index.{}", ext));
                if index_file.exists() {
                    return self.create_resolved_module(specifier, index_file, false).await;
                }
            }
            
            // Check for package.json main field
            let package_json_path = module_path.join("package.json");
            if package_json_path.exists() {
                if let Ok(package_json) = self.read_package_json(&package_json_path).await {
                    if let Some(main) = package_json.main {
                        let main_file = module_path.join(main);
                        if main_file.exists() {
                            return self.create_resolved_module(specifier, main_file, false).await;
                        }
                    }
                }
            }
        }
        
        Err(ModuleResolutionError::ModuleNotFound(specifier.to_string()))
    }
    
    async fn resolve_package_module(&self, specifier: &str, current_file: &Path) -> Result<ResolvedModule, ModuleResolutionError> {
        // Search in node_modules directories
        let mut current_dir = current_file.parent();
        
        while let Some(dir) = current_dir {
            let node_modules = dir.join("node_modules").join(specifier);
            
            if node_modules.exists() {
                // Check package.json for main entry point
                let package_json_path = node_modules.join("package.json");
                if package_json_path.exists() {
                    if let Ok(package_json) = self.read_package_json(&package_json_path).await {
                        let main_file = if let Some(main) = package_json.main {
                            node_modules.join(main)
                        } else {
                            node_modules.join("index.js")
                        };
                        
                        if main_file.exists() {
                            return self.create_resolved_module(specifier, main_file, true).await;
                        }
                    }
                }
                
                // Fallback to index.js
                let index_file = node_modules.join("index.js");
                if index_file.exists() {
                    return self.create_resolved_module(specifier, index_file, true).await;
                }
            }
            
            current_dir = dir.parent();
        }
        
        Err(ModuleResolutionError::ModuleNotFound(specifier.to_string()))
    }
    
    fn detect_module_type(&self, program: &Program) -> ModuleType {
        let mut has_es6_imports = false;
        let mut has_es6_exports = false;
        let mut has_commonjs = false;
        
        let mut detector = ModuleTypeDetector::new(&mut has_es6_imports, &mut has_es6_exports, &mut has_commonjs);
        program.visit_with(&mut detector);
        
        if has_es6_imports || has_es6_exports {
            ModuleType::ES6Module
        } else if has_commonjs {
            ModuleType::CommonJSModule
        } else {
            ModuleType::ES6Module // Default assumption
        }
    }
}
```

### 8.5 React Analysis System

#### 8.5.1 React-Specific Analyzer
```rust
pub struct ReactAnalyzer {
    component_analyzer: ComponentAnalyzer,
    hook_analyzer: HookAnalyzer,
    jsx_analyzer: JSXAnalyzer,
}

#[derive(Debug, Clone)]
pub struct ReactInfo {
    pub components: Vec<ReactComponent>,
    pub hooks: Vec<HookUsage>,
    pub jsx_elements: Vec<JSXElement>,
    pub props_interfaces: Vec<PropsInterface>,
    pub context_usage: Vec<ContextUsage>,
    pub lifecycle_methods: Vec<LifecycleMethod>,
    pub performance_issues: Vec<ReactPerformanceIssue>,
    pub accessibility_issues: Vec<A11yIssue>,
}

#[derive(Debug, Clone)]
pub struct ReactComponent {
    pub name: String,
    pub component_type: ComponentType,
    pub span: Span,
    pub props: Vec<ComponentProp>,
    pub state: Vec<StateVariable>,
    pub hooks_used: Vec<String>,
    pub jsx_elements: Vec<JSXElement>,
    pub is_exported: bool,
    pub is_memo: bool,
    pub is_forwardref: bool,
}

#[derive(Debug, Clone)]
pub enum ComponentType {
    FunctionComponent,
    ClassComponent,
    ArrowFunctionComponent,
    ForwardRefComponent,
    MemoComponent,
    HigherOrderComponent,
}

#[derive(Debug, Clone)]
pub struct HookUsage {
    pub hook_name: String,
    pub hook_type: HookType,
    pub span: Span,
    pub dependencies: Vec<String>,
    pub is_custom_hook: bool,
    pub violations: Vec<HookViolation>,
}

#[derive(Debug, Clone)]
pub enum HookType {
    UseState,
    UseEffect,
    UseContext,
    UseReducer,
    UseCallback,
    UseMemo,
    UseRef,
    UseImperativeHandle,
    UseLayoutEffect,
    UseDebugValue,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct JSXElement {
    pub tag_name: String,
    pub span: Span,
    pub attributes: Vec<JSXAttribute>,
    pub children: Vec<JSXChild>,
    pub is_self_closing: bool,
    pub is_component: bool,
}

#[derive(Debug, Clone)]
pub struct JSXAttribute {
    pub name: String,
    pub value: Option<String>,
    pub span: Span,
    pub is_spread: bool,
    pub is_expression: bool,
}

impl ReactAnalyzer {
    pub fn new() -> Self {
        Self {
            component_analyzer: ComponentAnalyzer::new(),
            hook_analyzer: HookAnalyzer::new(),
            jsx_analyzer: JSXAnalyzer::new(),
        }
    }
    
    pub async fn analyze_react(&self, program: &Program, content: &str) -> Result<ReactInfo, ReactAnalysisError> {
        let mut react_info = ReactInfo {
            components: Vec::new(),
            hooks: Vec::new(),
            jsx_elements: Vec::new(),
            props_interfaces: Vec::new(),
            context_usage: Vec::new(),
            lifecycle_methods: Vec::new(),
            performance_issues: Vec::new(),
            accessibility_issues: Vec::new(),
        };
        
        // Analyze components
        react_info.components = self.component_analyzer.analyze_components(program)?;
        
        // Analyze hooks
        react_info.hooks = self.hook_analyzer.analyze_hooks(program)?;
        
        // Analyze JSX
        react_info.jsx_elements = self.jsx_analyzer.analyze_jsx(program)?;
        
        // Extract props interfaces
        react_info.props_interfaces = self.extract_props_interfaces(program)?;
        
        // Analyze context usage
        react_info.context_usage = self.analyze_context_usage(program)?;
        
        // Extract lifecycle methods
        react_info.lifecycle_methods = self.extract_lifecycle_methods(program)?;
        
        // Detect performance issues
        react_info.performance_issues = self.detect_performance_issues(&react_info)?;
        
        // Check accessibility
        react_info.accessibility_issues = self.check_accessibility(&react_info)?;
        
        Ok(react_info)
    }
    
    fn detect_performance_issues(&self, react_info: &ReactInfo) -> Result<Vec<ReactPerformanceIssue>, ReactAnalysisError> {
        let mut issues = Vec::new();
        
        // Check for missing React.memo on components with complex props
        for component in &react_info.components {
            if !component.is_memo && component.props.len() > 3 {
                issues.push(ReactPerformanceIssue {
                    issue_type: PerformanceIssueType::MissingMemo,
                    component_name: component.name.clone(),
                    span: component.span,
                    message: "Consider wrapping this component with React.memo for better performance".to_string(),
                    suggestion: Some(format!("export default React.memo({})", component.name)),
                });
            }
        }
        
        // Check for missing useCallback/useMemo in useEffect dependencies
        for hook in &react_info.hooks {
            if matches!(hook.hook_type, HookType::UseEffect) {
                for dep in &hook.dependencies {
                    if self.is_function_or_object_dependency(dep) {
                        issues.push(ReactPerformanceIssue {
                            issue_type: PerformanceIssueType::MissingCallback,
                            component_name: "".to_string(),
                            span: hook.span,
                            message: format!("Dependency '{}' should be wrapped with useCallback or useMemo", dep),
                            suggestion: Some(format!("const {} = useCallback(...)", dep)),
                        });
                    }
                }
            }
        }
        
        Ok(issues)
    }
    
    fn check_accessibility(&self, react_info: &ReactInfo) -> Result<Vec<A11yIssue>, ReactAnalysisError> {
        let mut issues = Vec::new();
        
        for jsx_element in &react_info.jsx_elements {
            // Check for missing alt attributes on img tags
            if jsx_element.tag_name == "img" {
                let has_alt = jsx_element.attributes.iter()
                    .any(|attr| attr.name == "alt");
                
                if !has_alt {
                    issues.push(A11yIssue {
                        issue_type: A11yIssueType::MissingAltText,
                        element: jsx_element.tag_name.clone(),
                        span: jsx_element.span,
                        message: "img elements must have an alt prop".to_string(),
                        suggestion: Some("Add an alt attribute describing the image".to_string()),
                    });
                }
            }
            
            // Check for missing labels on form inputs
            if matches!(jsx_element.tag_name.as_str(), "input" | "textarea" | "select") {
                let has_label_attr = jsx_element.attributes.iter()
                    .any(|attr| matches!(attr.name.as_str(), "aria-label" | "aria-labelledby"));
                
                if !has_label_attr {
                    issues.push(A11yIssue {
                        issue_type: A11yIssueType::MissingLabel,
                        element: jsx_element.tag_name.clone(),
                        span: jsx_element.span,
                        message: "Form controls must have labels".to_string(),
                        suggestion: Some("Add aria-label or associate with a label element".to_string()),
                    });
                }
            }
        }
        
        Ok(issues)
    }
}

#[derive(Debug, Clone)]
pub struct ReactPerformanceIssue {
    pub issue_type: PerformanceIssueType,
    pub component_name: String,
    pub span: Span,
    pub message: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone)]
pub enum PerformanceIssueType {
    MissingMemo,
    MissingCallback,
    MissingUseMemo,
    InlineObjectProp,
    InlineFunctionProp,
    UnnecessaryRerender,
}

#[derive(Debug, Clone)]
pub struct A11yIssue {
    pub issue_type: A11yIssueType,
    pub element: String,
    pub span: Span,
    pub message: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone)]
pub enum A11yIssueType {
    MissingAltText,
    MissingLabel,
    MissingRole,
    InvalidAriaAttribute,
    MissingKeyboardNavigation,
    InsufficientColorContrast,
    MissingFocusManagement,
}
```

### 8.6 Node.js Analysis System

#### 8.6.1 Node.js-Specific Analyzer
```rust
pub struct NodeJSAnalyzer {
    builtin_modules: HashSet<String>,
    security_analyzer: NodeSecurityAnalyzer,
    performance_analyzer: NodePerformanceAnalyzer,
}

#[derive(Debug, Clone)]
pub struct NodeJSInfo {
    pub builtin_module_usage: Vec<BuiltinModuleUsage>,
    pub file_system_operations: Vec<FileSystemOperation>,
    pub network_operations: Vec<NetworkOperation>,
    pub process_operations: Vec<ProcessOperation>,
    pub security_issues: Vec<NodeSecurityIssue>,
    pub performance_issues: Vec<NodePerformanceIssue>,
    pub environment_variables: Vec<EnvironmentVariable>,
    pub async_patterns: Vec<AsyncPattern>,
}

#[derive(Debug, Clone)]
pub struct BuiltinModuleUsage {
    pub module_name: String,
    pub import_span: Span,
    pub usage_spans: Vec<Span>,
    pub methods_used: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FileSystemOperation {
    pub operation_type: FSOperationType,
    pub file_path: Option<String>,
    pub span: Span,
    pub is_sync: bool,
    pub is_secure: bool,
}

#[derive(Debug, Clone)]
pub enum FSOperationType {
    Read,
    Write,
    Delete,
    Create,
    Move,
    Copy,
    Stat,
    Watch,
}

#[derive(Debug, Clone)]
pub struct NetworkOperation {
    pub operation_type: NetworkOperationType,
    pub url: Option<String>,
    pub span: Span,
    pub is_secure: bool,
    pub method: Option<String>,
}

#[derive(Debug, Clone)]
pub enum NetworkOperationType {
    HTTPRequest,
    HTTPSRequest,
    TCPConnection,
    UDPConnection,
    WebSocket,
}

impl NodeJSAnalyzer {
    pub fn new() -> Self {
        let mut builtin_modules = HashSet::new();
        builtin_modules.extend([
            "fs", "path", "os", "crypto", "http", "https", "net", "url",
            "querystring", "util", "events", "stream", "buffer", "child_process",
            "cluster", "worker_threads", "async_hooks", "perf_hooks"
        ].iter().map(|s| s.to_string()));
        
        Self {
            builtin_modules,
            security_analyzer: NodeSecurityAnalyzer::new(),
            performance_analyzer: NodePerformanceAnalyzer::new(),
        }
    }
    
    pub async fn analyze_nodejs(&self, program: &Program, content: &str) -> Result<NodeJSInfo, NodeAnalysisError> {
        let mut nodejs_info = NodeJSInfo {
            builtin_module_usage: Vec::new(),
            file_system_operations: Vec::new(),
            network_operations: Vec::new(),
            process_operations: Vec::new(),
            security_issues: Vec::new(),
            performance_issues: Vec::new(),
            environment_variables: Vec::new(),
            async_patterns: Vec::new(),
        };
        
        // Analyze builtin module usage
        nodejs_info.builtin_module_usage = self.analyze_builtin_modules(program)?;
        
        // Analyze file system operations
        nodejs_info.file_system_operations = self.analyze_fs_operations(program)?;
        
        // Analyze network operations
        nodejs_info.network_operations = self.analyze_network_operations(program)?;
        
        // Analyze process operations
        nodejs_info.process_operations = self.analyze_process_operations(program)?;
        
        // Detect security issues
        nodejs_info.security_issues = self.security_analyzer.analyze_security(program)?;
        
        // Detect performance issues
        nodejs_info.performance_issues = self.performance_analyzer.analyze_performance(program)?;
        
        // Extract environment variables
        nodejs_info.environment_variables = self.extract_env_variables(program)?;
        
        // Analyze async patterns
        nodejs_info.async_patterns = self.analyze_async_patterns(program)?;
        
        Ok(nodejs_info)
    }
    
    fn analyze_builtin_modules(&self, program: &Program) -> Result<Vec<BuiltinModuleUsage>, NodeAnalysisError> {
        let mut builtin_usage = Vec::new();
        let mut visitor = BuiltinModuleVisitor::new(&mut builtin_usage, &self.builtin_modules);
        program.visit_with(&mut visitor);
        Ok(builtin_usage)
    }
    
    fn analyze_fs_operations(&self, program: &Program) -> Result<Vec<FileSystemOperation>, NodeAnalysisError> {
        let mut fs_operations = Vec::new();
        let mut visitor = FileSystemVisitor::new(&mut fs_operations);
        program.visit_with(&mut visitor);
        Ok(fs_operations)
    }
    
    fn analyze_async_patterns(&self, program: &Program) -> Result<Vec<AsyncPattern>, NodeAnalysisError> {
        let mut async_patterns = Vec::new();
        let mut visitor = AsyncPatternVisitor::new(&mut async_patterns);
        program.visit_with(&mut visitor);
        Ok(async_patterns)
    }
}
```

### 8.7 JavaScript/TypeScript Pattern Detection

#### 8.7.1 JS/TS Pattern Detector
```rust
pub struct JSPatternDetector {
    patterns: Vec<Box<dyn JSPattern>>,
    config: JSPatternConfig,
}

#[async_trait]
pub trait JSPattern: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn category(&self) -> JSPatternCategory;
    fn severity(&self) -> PatternSeverity;
    fn language_support(&self) -> Vec<JSLanguage>;
    
    async fn detect(&self, analysis_result: &JSAnalysisResult) -> Result<Vec<JSPatternMatch>, PatternError>;
}

#[derive(Debug, Clone)]
pub enum JSPatternCategory {
    TypeScript,
    JavaScript,
    React,
    NodeJS,
    Performance,
    Security,
    Maintainability,
    BestPractices,
    ModernJS,
}

// Example: Async/Await over Promises Pattern
pub struct AsyncAwaitPattern;

#[async_trait]
impl JSPattern for AsyncAwaitPattern {
    fn name(&self) -> &str {
        "prefer_async_await"
    }
    
    fn description(&self) -> &str {
        "Prefer async/await over Promise chains for better readability"
    }
    
    fn category(&self) -> JSPatternCategory {
        JSPatternCategory::ModernJS
    }
    
    fn severity(&self) -> PatternSeverity {
        PatternSeverity::Low
    }
    
    fn language_support(&self) -> Vec<JSLanguage> {
        vec![JSLanguage::JavaScript, JSLanguage::TypeScript]
    }
    
    async fn detect(&self, analysis_result: &JSAnalysisResult) -> Result<Vec<JSPatternMatch>, PatternError> {
        let mut matches = Vec::new();
        let mut visitor = PromiseChainVisitor::new(&mut matches);
        analysis_result.swc_ast.visit_with(&mut visitor);
        Ok(matches)
    }
}

// Example: TypeScript Strict Mode Pattern
pub struct TypeScriptStrictPattern;

#[async_trait]
impl JSPattern for TypeScriptStrictPattern {
    fn name(&self) -> &str {
        "typescript_strict_mode"
    }
    
    fn description(&self) -> &str {
        "Ensure TypeScript strict mode is enabled for better type safety"
    }
    
    fn category(&self) -> JSPatternCategory {
        JSPatternCategory::TypeScript
    }
    
    fn severity(&self) -> PatternSeverity {
        PatternSeverity::Medium
    }
    
    fn language_support(&self) -> Vec<JSLanguage> {
        vec![JSLanguage::TypeScript, JSLanguage::TSX]
    }
    
    async fn detect(&self, analysis_result: &JSAnalysisResult) -> Result<Vec<JSPatternMatch>, PatternError> {
        let mut matches = Vec::new();
        
        // Check for any type usage
        if let Some(type_info) = &analysis_result.type_info {
            for (span, ts_type) in &type_info.inferred_types {
                if matches!(ts_type.ts_type, TSType::Any) {
                    matches.push(JSPatternMatch {
                        pattern_name: self.name().to_string(),
                        category: self.category(),
                        severity: self.severity(),
                        message: "Avoid using 'any' type. Use specific types instead.".to_string(),
                        span: *span,
                        suggestion: Some("Replace 'any' with a specific type".to_string()),
                        confidence: 0.9,
                        metadata: HashMap::new(),
                    });
                }
            }
        }
        
        Ok(matches)
    }
}

// Additional patterns to implement:
// - UnusedVariablePattern
// - ConsoleLogPattern (for production code)
// - DeprecatedAPIPattern
// - SecurityVulnerabilityPattern
// - PerformanceAntipatternPattern
// - ReactHooksRulesPattern
// - AccessibilityPattern
// - ESLintRulesPattern
```

### 8.8 Criterios de Completitud

#### 8.8.1 Entregables de la Fase
- [ ] Parser TypeScript/JavaScript especializado implementado
- [ ] Análisis del sistema de tipos TypeScript
- [ ] Resolución de módulos ES6/CommonJS
- [ ] Análisis específico para React
- [ ] Análisis específico para Node.js
- [ ] Detección de patrones JS/TS
- [ ] Cálculo de métricas específicas
- [ ] Integration con parser universal
- [ ] Tests comprehensivos para JS/TS
- [ ] Documentación completa

#### 8.8.2 Criterios de Aceptación
- [ ] Parse correctamente código TypeScript/JavaScript moderno
- [ ] Análisis de tipos TypeScript funciona correctamente
- [ ] Resolución de módulos maneja casos complejos
- [ ] Análisis React detecta componentes y hooks
- [ ] Análisis Node.js identifica APIs específicas
- [ ] Pattern detection encuentra issues comunes
- [ ] Métricas calculadas son precisas
- [ ] Performance acceptable para proyectos grandes
- [ ] Error handling robusto
- [ ] Integration seamless con sistema principal

### 8.9 Performance Targets

#### 8.9.1 Benchmarks Específicos JS/TS
- **Parsing speed**: >800 lines/second para TypeScript
- **Type analysis**: <3x overhead sobre parsing básico
- **Module resolution**: <500ms para proyectos típicos
- **Memory usage**: <25MB por archivo típico
- **Pattern detection**: <1.5 segundos para archivos <1500 lines

### 8.10 Estimación de Tiempo

#### 8.10.1 Breakdown de Tareas
- Setup SWC integration: 4 días
- TypeScript type system analysis: 8 días
- Module resolution system: 7 días
- React analysis engine: 6 días
- Node.js analysis engine: 5 días
- Pattern detection framework: 6 días
- JS/TS specific patterns (15+): 10 días
- Metrics calculation: 4 días
- Integration con parser universal: 3 días
- Testing comprehensivo: 7 días
- Performance optimization: 5 días
- Documentación: 3 días

**Total estimado: 68 días de desarrollo**

### 8.11 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Análisis TypeScript/JavaScript extremadamente profundo
- Capacidades de análisis de tipos avanzado
- Detección de patrones específicos del ecosistema JS/TS
- Análisis especializado para React y Node.js
- Foundation para análisis de calidad de código moderno

La Fase 9 implementará capacidades similares para Rust, completando el trío de lenguajes principales del agente CodeAnt.
