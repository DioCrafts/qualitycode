# Fase 12: Detección de Código Muerto Básico

## Objetivo General
Implementar un sistema avanzado de detección de código muerto que identifique automáticamente código no utilizado, funciones obsoletas, imports innecesarios, variables no referenciadas, y otros elementos redundantes en múltiples lenguajes, utilizando análisis de flujo de datos, análisis de dependencias, y técnicas de reachability analysis.

## Descripción Técnica Detallada

### 12.1 Arquitectura del Sistema de Detección

#### 12.1.1 Diseño del Dead Code Detector
```
┌─────────────────────────────────────────┐
│         Dead Code Detection System      │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │Reachability │ │   Data Flow         │ │
│  │  Analyzer   │ │   Analyzer          │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ Dependency  │ │    Usage            │ │
│  │   Graph     │ │   Tracker           │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Import    │ │   Cross-Module      │ │
│  │  Analyzer   │ │    Analyzer         │ │
│  └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
```

#### 12.1.2 Componentes del Sistema
- **Reachability Analyzer**: Análisis de alcanzabilidad de código
- **Data Flow Analyzer**: Análisis de flujo de datos para variables
- **Dependency Graph**: Grafo de dependencias entre módulos/funciones
- **Usage Tracker**: Seguimiento de uso de símbolos
- **Import Analyzer**: Análisis de imports/exports no utilizados
- **Cross-Module Analyzer**: Análisis entre módulos y archivos

### 12.2 Dead Code Detection Engine

#### 12.2.1 Core Detection Engine
```rust
use std::collections::{HashMap, HashSet, VecDeque};
use petgraph::{Graph, Direction};
use petgraph::graph::{NodeIndex, EdgeIndex};

pub struct DeadCodeDetector {
    reachability_analyzer: Arc<ReachabilityAnalyzer>,
    data_flow_analyzer: Arc<DataFlowAnalyzer>,
    dependency_graph_builder: Arc<DependencyGraphBuilder>,
    usage_tracker: Arc<UsageTracker>,
    import_analyzer: Arc<ImportAnalyzer>,
    cross_module_analyzer: Arc<CrossModuleAnalyzer>,
    config: DeadCodeConfig,
}

#[derive(Debug, Clone)]
pub struct DeadCodeConfig {
    pub analyze_unused_variables: bool,
    pub analyze_unused_functions: bool,
    pub analyze_unused_classes: bool,
    pub analyze_unused_imports: bool,
    pub analyze_unreachable_code: bool,
    pub analyze_dead_branches: bool,
    pub cross_module_analysis: bool,
    pub entry_points: Vec<String>,
    pub keep_patterns: Vec<String>,
    pub aggressive_mode: bool,
    pub language_specific_configs: HashMap<ProgrammingLanguage, LanguageDeadCodeConfig>,
}

#[derive(Debug, Clone)]
pub struct LanguageDeadCodeConfig {
    pub ignore_test_files: bool,
    pub ignore_main_functions: bool,
    pub ignore_exported_symbols: bool,
    pub custom_entry_patterns: Vec<String>,
    pub framework_specific_rules: Vec<FrameworkRule>,
}

impl DeadCodeDetector {
    pub async fn new(config: DeadCodeConfig) -> Result<Self, DeadCodeError> {
        Ok(Self {
            reachability_analyzer: Arc::new(ReachabilityAnalyzer::new()),
            data_flow_analyzer: Arc::new(DataFlowAnalyzer::new()),
            dependency_graph_builder: Arc::new(DependencyGraphBuilder::new()),
            usage_tracker: Arc::new(UsageTracker::new()),
            import_analyzer: Arc::new(ImportAnalyzer::new()),
            cross_module_analyzer: Arc::new(CrossModuleAnalyzer::new()),
            config,
        })
    }
    
    pub async fn detect_dead_code(&self, unified_ast: &UnifiedAST) -> Result<DeadCodeAnalysis, DeadCodeError> {
        let start_time = Instant::now();
        
        let mut analysis = DeadCodeAnalysis {
            file_path: unified_ast.file_path.clone(),
            language: unified_ast.language,
            unused_variables: Vec::new(),
            unused_functions: Vec::new(),
            unused_classes: Vec::new(),
            unused_imports: Vec::new(),
            unreachable_code: Vec::new(),
            dead_branches: Vec::new(),
            unused_parameters: Vec::new(),
            redundant_assignments: Vec::new(),
            statistics: DeadCodeStatistics::default(),
            execution_time_ms: 0,
        };
        
        // Build dependency graph
        let dependency_graph = self.dependency_graph_builder.build_graph(unified_ast).await?;
        
        // Analyze unused variables
        if self.config.analyze_unused_variables {
            analysis.unused_variables = self.detect_unused_variables(unified_ast, &dependency_graph).await?;
        }
        
        // Analyze unused functions
        if self.config.analyze_unused_functions {
            analysis.unused_functions = self.detect_unused_functions(unified_ast, &dependency_graph).await?;
        }
        
        // Analyze unused classes
        if self.config.analyze_unused_classes {
            analysis.unused_classes = self.detect_unused_classes(unified_ast, &dependency_graph).await?;
        }
        
        // Analyze unused imports
        if self.config.analyze_unused_imports {
            analysis.unused_imports = self.import_analyzer.detect_unused_imports(unified_ast).await?;
        }
        
        // Analyze unreachable code
        if self.config.analyze_unreachable_code {
            analysis.unreachable_code = self.reachability_analyzer.detect_unreachable_code(unified_ast, &dependency_graph).await?;
        }
        
        // Analyze dead branches
        if self.config.analyze_dead_branches {
            analysis.dead_branches = self.detect_dead_branches(unified_ast).await?;
        }
        
        // Analyze unused parameters
        analysis.unused_parameters = self.detect_unused_parameters(unified_ast).await?;
        
        // Analyze redundant assignments
        analysis.redundant_assignments = self.detect_redundant_assignments(unified_ast).await?;
        
        // Calculate statistics
        analysis.statistics = self.calculate_statistics(&analysis);
        analysis.execution_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(analysis)
    }
    
    pub async fn detect_dead_code_project(&self, project_asts: &[UnifiedAST]) -> Result<ProjectDeadCodeAnalysis, DeadCodeError> {
        let start_time = Instant::now();
        
        // Build cross-module dependency graph
        let global_dependency_graph = self.cross_module_analyzer.build_global_graph(project_asts).await?;
        
        // Identify entry points
        let entry_points = self.identify_entry_points(project_asts, &global_dependency_graph).await?;
        
        // Perform reachability analysis from entry points
        let reachable_symbols = self.reachability_analyzer.find_reachable_symbols(&global_dependency_graph, &entry_points).await?;
        
        // Analyze each file
        let mut file_analyses = Vec::new();
        for ast in project_asts {
            let mut file_analysis = self.detect_dead_code(ast).await?;
            
            // Filter results based on cross-module analysis
            file_analysis = self.filter_by_reachability(file_analysis, &reachable_symbols, &global_dependency_graph).await?;
            
            file_analyses.push(file_analysis);
        }
        
        // Aggregate results
        let project_analysis = ProjectDeadCodeAnalysis {
            project_path: project_asts.first().map(|ast| ast.file_path.parent().unwrap_or(&ast.file_path).to_path_buf()),
            file_analyses,
            global_statistics: self.calculate_project_statistics(&file_analyses),
            cross_module_issues: self.detect_cross_module_issues(&global_dependency_graph, &reachable_symbols).await?,
            dependency_cycles: self.detect_dependency_cycles(&global_dependency_graph).await?,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        };
        
        Ok(project_analysis)
    }
    
    async fn detect_unused_variables(&self, ast: &UnifiedAST, dependency_graph: &DependencyGraph) -> Result<Vec<UnusedVariable>, DeadCodeError> {
        let mut unused_variables = Vec::new();
        
        // Extract all variable declarations
        let variable_declarations = self.extract_variable_declarations(&ast.root_node);
        
        // For each variable, check if it's used
        for var_decl in variable_declarations {
            let usage_count = self.usage_tracker.count_variable_usage(ast, &var_decl.name).await?;
            
            if usage_count == 0 {
                // Check if it's an exception (like loop variables, error handling, etc.)
                if !self.is_variable_exception(&var_decl, ast) {
                    unused_variables.push(UnusedVariable {
                        name: var_decl.name.clone(),
                        declaration_location: var_decl.location.clone(),
                        variable_type: var_decl.variable_type.clone(),
                        scope: var_decl.scope.clone(),
                        reason: self.determine_unused_reason(&var_decl),
                        suggestion: self.generate_variable_suggestion(&var_decl),
                        confidence: self.calculate_confidence(&var_decl, usage_count),
                    });
                }
            } else if usage_count == 1 && self.is_self_assignment(&var_decl, ast).await? {
                // Variable only assigned to itself
                unused_variables.push(UnusedVariable {
                    name: var_decl.name.clone(),
                    declaration_location: var_decl.location.clone(),
                    variable_type: var_decl.variable_type.clone(),
                    scope: var_decl.scope.clone(),
                    reason: UnusedReason::SelfAssignmentOnly,
                    suggestion: "Remove this variable as it's only assigned to itself".to_string(),
                    confidence: 0.9,
                });
            }
        }
        
        Ok(unused_variables)
    }
    
    async fn detect_unused_functions(&self, ast: &UnifiedAST, dependency_graph: &DependencyGraph) -> Result<Vec<UnusedFunction>, DeadCodeError> {
        let mut unused_functions = Vec::new();
        
        // Extract all function declarations
        let function_declarations = self.extract_function_declarations(&ast.root_node);
        
        for func_decl in function_declarations {
            // Skip if it's an entry point or exported function
            if self.is_function_entry_point(&func_decl) || self.is_exported_function(&func_decl) {
                continue;
            }
            
            let usage_count = self.usage_tracker.count_function_usage(ast, &func_decl.name).await?;
            
            if usage_count == 0 {
                // Check for framework-specific patterns (like test functions, event handlers)
                if !self.is_function_exception(&func_decl, ast) {
                    unused_functions.push(UnusedFunction {
                        name: func_decl.name.clone(),
                        declaration_location: func_decl.location.clone(),
                        function_type: func_decl.function_type.clone(),
                        visibility: func_decl.visibility.clone(),
                        parameters: func_decl.parameters.clone(),
                        reason: self.determine_function_unused_reason(&func_decl),
                        suggestion: self.generate_function_suggestion(&func_decl),
                        confidence: self.calculate_function_confidence(&func_decl, usage_count),
                        potential_side_effects: self.analyze_function_side_effects(&func_decl, ast).await?,
                    });
                }
            }
        }
        
        Ok(unused_functions)
    }
}

#[derive(Debug, Clone)]
pub struct DeadCodeAnalysis {
    pub file_path: PathBuf,
    pub language: ProgrammingLanguage,
    pub unused_variables: Vec<UnusedVariable>,
    pub unused_functions: Vec<UnusedFunction>,
    pub unused_classes: Vec<UnusedClass>,
    pub unused_imports: Vec<UnusedImport>,
    pub unreachable_code: Vec<UnreachableCode>,
    pub dead_branches: Vec<DeadBranch>,
    pub unused_parameters: Vec<UnusedParameter>,
    pub redundant_assignments: Vec<RedundantAssignment>,
    pub statistics: DeadCodeStatistics,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct UnusedVariable {
    pub name: String,
    pub declaration_location: UnifiedPosition,
    pub variable_type: Option<UnifiedType>,
    pub scope: ScopeInfo,
    pub reason: UnusedReason,
    pub suggestion: String,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct UnusedFunction {
    pub name: String,
    pub declaration_location: UnifiedPosition,
    pub function_type: FunctionType,
    pub visibility: Visibility,
    pub parameters: Vec<Parameter>,
    pub reason: UnusedReason,
    pub suggestion: String,
    pub confidence: f64,
    pub potential_side_effects: Vec<SideEffect>,
}

#[derive(Debug, Clone)]
pub enum UnusedReason {
    NeverCalled,
    NeverReferenced,
    OnlyAssignedNeverRead,
    SelfAssignmentOnly,
    DeadBranch,
    UnreachableCode,
    ObsoleteFunction,
    DuplicateImplementation,
}
```

### 12.3 Reachability Analysis System

#### 12.3.1 Reachability Analyzer
```rust
pub struct ReachabilityAnalyzer {
    call_graph_builder: CallGraphBuilder,
    control_flow_analyzer: ControlFlowAnalyzer,
}

impl ReachabilityAnalyzer {
    pub async fn detect_unreachable_code(&self, ast: &UnifiedAST, dependency_graph: &DependencyGraph) -> Result<Vec<UnreachableCode>, ReachabilityError> {
        let mut unreachable_code = Vec::new();
        
        // Build control flow graph
        let cfg = self.control_flow_analyzer.build_cfg(&ast.root_node).await?;
        
        // Find unreachable nodes in CFG
        let unreachable_nodes = self.find_unreachable_nodes(&cfg).await?;
        
        for node_id in unreachable_nodes {
            if let Some(node) = cfg.get_node(node_id) {
                unreachable_code.push(UnreachableCode {
                    location: node.position.clone(),
                    code_type: node.node_type.clone(),
                    reason: self.determine_unreachability_reason(node, &cfg),
                    suggestion: self.generate_unreachability_suggestion(node),
                    confidence: self.calculate_unreachability_confidence(node, &cfg),
                    blocking_condition: self.find_blocking_condition(node, &cfg),
                });
            }
        }
        
        // Detect unreachable code after return statements
        unreachable_code.extend(self.detect_code_after_return(ast).await?);
        
        // Detect unreachable code in conditional branches
        unreachable_code.extend(self.detect_unreachable_branches(ast).await?);
        
        Ok(unreachable_code)
    }
    
    pub async fn find_reachable_symbols(&self, global_graph: &GlobalDependencyGraph, entry_points: &[EntryPoint]) -> Result<HashSet<SymbolId>, ReachabilityError> {
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();
        
        // Start from entry points
        for entry_point in entry_points {
            queue.push_back(entry_point.symbol_id.clone());
            reachable.insert(entry_point.symbol_id.clone());
        }
        
        // BFS to find all reachable symbols
        while let Some(symbol_id) = queue.pop_front() {
            if let Some(dependencies) = global_graph.get_dependencies(&symbol_id) {
                for dep_id in dependencies {
                    if !reachable.contains(dep_id) {
                        reachable.insert(dep_id.clone());
                        queue.push_back(dep_id.clone());
                    }
                }
            }
        }
        
        Ok(reachable)
    }
    
    async fn find_unreachable_nodes(&self, cfg: &ControlFlowGraph) -> Result<Vec<NodeId>, ReachabilityError> {
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();
        
        // Start from entry node
        if let Some(entry_id) = cfg.get_entry_node() {
            queue.push_back(entry_id);
            reachable.insert(entry_id);
        }
        
        // BFS traversal
        while let Some(node_id) = queue.pop_front() {
            for successor in cfg.get_successors(node_id) {
                if !reachable.contains(&successor) {
                    reachable.insert(successor);
                    queue.push_back(successor);
                }
            }
        }
        
        // Find unreachable nodes
        let all_nodes: HashSet<_> = cfg.get_all_nodes().into_iter().collect();
        let unreachable: Vec<_> = all_nodes.difference(&reachable).cloned().collect();
        
        Ok(unreachable)
    }
    
    async fn detect_code_after_return(&self, ast: &UnifiedAST) -> Result<Vec<UnreachableCode>, ReachabilityError> {
        let mut unreachable = Vec::new();
        let mut visitor = CodeAfterReturnVisitor::new(&mut unreachable);
        visitor.visit_node(&ast.root_node);
        Ok(unreachable)
    }
    
    async fn detect_unreachable_branches(&self, ast: &UnifiedAST) -> Result<Vec<UnreachableCode>, ReachabilityError> {
        let mut unreachable = Vec::new();
        
        // Look for conditions that are always true/false
        let mut visitor = UnreachableBranchVisitor::new(&mut unreachable);
        visitor.visit_node(&ast.root_node);
        
        Ok(unreachable)
    }
}

#[derive(Debug, Clone)]
pub struct UnreachableCode {
    pub location: UnifiedPosition,
    pub code_type: UnifiedNodeType,
    pub reason: UnreachabilityReason,
    pub suggestion: String,
    pub confidence: f64,
    pub blocking_condition: Option<BlockingCondition>,
}

#[derive(Debug, Clone)]
pub enum UnreachabilityReason {
    AfterReturn,
    AfterThrow,
    AfterBreak,
    AfterContinue,
    DeadBranch,
    AlwaysFalseCondition,
    AlwaysTrueCondition,
    UnreachableFromEntry,
    CircularDependency,
}

#[derive(Debug, Clone)]
pub struct BlockingCondition {
    pub condition_location: UnifiedPosition,
    pub condition_expression: String,
    pub reason: String,
}
```

### 12.4 Data Flow Analysis for Dead Code

#### 12.4.1 Data Flow Analyzer
```rust
pub struct DataFlowAnalyzer {
    def_use_analyzer: DefUseAnalyzer,
    live_variable_analyzer: LiveVariableAnalyzer,
    reaching_definitions_analyzer: ReachingDefinitionsAnalyzer,
}

impl DataFlowAnalyzer {
    pub async fn analyze_variable_liveness(&self, ast: &UnifiedAST) -> Result<LivenessAnalysis, DataFlowError> {
        // Build control flow graph
        let cfg = self.build_control_flow_graph(&ast.root_node).await?;
        
        // Perform live variable analysis
        let liveness_info = self.live_variable_analyzer.analyze(&cfg).await?;
        
        // Find dead assignments
        let dead_assignments = self.find_dead_assignments(&cfg, &liveness_info).await?;
        
        // Find unused definitions
        let unused_definitions = self.find_unused_definitions(&cfg, &liveness_info).await?;
        
        Ok(LivenessAnalysis {
            live_variables: liveness_info,
            dead_assignments,
            unused_definitions,
        })
    }
    
    async fn find_dead_assignments(&self, cfg: &ControlFlowGraph, liveness: &LivenessInfo) -> Result<Vec<DeadAssignment>, DataFlowError> {
        let mut dead_assignments = Vec::new();
        
        for node_id in cfg.get_all_nodes() {
            if let Some(node) = cfg.get_node(node_id) {
                if let UnifiedNodeType::AssignmentExpression = node.node_type {
                    // Check if the assigned variable is live after this assignment
                    if let Some(assigned_var) = self.extract_assigned_variable(node) {
                        let live_after = liveness.get_live_out(node_id);
                        
                        if !live_after.contains(&assigned_var) {
                            dead_assignments.push(DeadAssignment {
                                location: node.position.clone(),
                                variable_name: assigned_var,
                                assignment_type: self.classify_assignment_type(node),
                                reason: "Variable is not used after this assignment".to_string(),
                                confidence: 0.95,
                            });
                        }
                    }
                }
            }
        }
        
        Ok(dead_assignments)
    }
    
    pub async fn analyze_def_use_chains(&self, ast: &UnifiedAST) -> Result<DefUseAnalysis, DataFlowError> {
        let cfg = self.build_control_flow_graph(&ast.root_node).await?;
        let def_use_chains = self.def_use_analyzer.build_chains(&cfg).await?;
        
        let mut unused_definitions = Vec::new();
        
        for (def_id, uses) in &def_use_chains {
            if uses.is_empty() {
                if let Some(def_node) = cfg.get_node(*def_id) {
                    unused_definitions.push(UnusedDefinition {
                        location: def_node.position.clone(),
                        variable_name: self.extract_defined_variable(def_node).unwrap_or_default(),
                        definition_type: self.classify_definition_type(def_node),
                        reason: "Variable is defined but never used".to_string(),
                        confidence: 0.9,
                    });
                }
            }
        }
        
        Ok(DefUseAnalysis {
            def_use_chains,
            unused_definitions,
        })
    }
}

#[derive(Debug, Clone)]
pub struct LivenessAnalysis {
    pub live_variables: LivenessInfo,
    pub dead_assignments: Vec<DeadAssignment>,
    pub unused_definitions: Vec<UnusedDefinition>,
}

#[derive(Debug, Clone)]
pub struct DeadAssignment {
    pub location: UnifiedPosition,
    pub variable_name: String,
    pub assignment_type: AssignmentType,
    pub reason: String,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum AssignmentType {
    SimpleAssignment,
    CompoundAssignment,
    DestructuringAssignment,
    InitializationAssignment,
}

#[derive(Debug, Clone)]
pub struct RedundantAssignment {
    pub location: UnifiedPosition,
    pub variable_name: String,
    pub previous_assignment: UnifiedPosition,
    pub redundancy_type: RedundancyType,
    pub suggestion: String,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum RedundancyType {
    ImmediateReassignment,
    UnusedBetweenAssignments,
    SelfAssignment,
    ConstantReassignment,
}
```

### 12.5 Import Analysis System

#### 12.5.1 Import Analyzer
```rust
pub struct ImportAnalyzer {
    module_resolver: ModuleResolver,
    usage_tracker: UsageTracker,
}

impl ImportAnalyzer {
    pub async fn detect_unused_imports(&self, ast: &UnifiedAST) -> Result<Vec<UnusedImport>, ImportAnalysisError> {
        let mut unused_imports = Vec::new();
        
        // Extract all import statements
        let imports = self.extract_imports(&ast.root_node);
        
        for import in imports {
            let usage_analysis = self.analyze_import_usage(&import, ast).await?;
            
            if usage_analysis.is_unused() {
                unused_imports.push(UnusedImport {
                    import_statement: import.clone(),
                    location: import.location.clone(),
                    import_type: import.import_type.clone(),
                    module_name: import.module_name.clone(),
                    imported_symbols: import.imported_symbols.clone(),
                    reason: usage_analysis.unused_reason,
                    suggestion: self.generate_import_suggestion(&import, &usage_analysis),
                    confidence: usage_analysis.confidence,
                    side_effects_possible: self.check_for_side_effects(&import).await?,
                });
            } else if usage_analysis.has_partially_unused() {
                // Some symbols from the import are unused
                unused_imports.push(UnusedImport {
                    import_statement: import.clone(),
                    location: import.location.clone(),
                    import_type: ImportType::PartiallyUnused,
                    module_name: import.module_name.clone(),
                    imported_symbols: usage_analysis.unused_symbols,
                    reason: UnusedReason::PartiallyUnused,
                    suggestion: "Remove unused symbols from import".to_string(),
                    confidence: 0.95,
                    side_effects_possible: false,
                });
            }
        }
        
        Ok(unused_imports)
    }
    
    async fn analyze_import_usage(&self, import: &ImportStatement, ast: &UnifiedAST) -> Result<ImportUsageAnalysis, ImportAnalysisError> {
        let mut analysis = ImportUsageAnalysis {
            used_symbols: HashSet::new(),
            unused_symbols: Vec::new(),
            usage_locations: Vec::new(),
            confidence: 1.0,
            unused_reason: UnusedReason::NeverReferenced,
        };
        
        match &import.import_type {
            ImportType::DefaultImport(name) => {
                let usage_count = self.usage_tracker.count_symbol_usage(ast, name).await?;
                if usage_count == 0 {
                    analysis.unused_symbols.push(name.clone());
                } else {
                    analysis.used_symbols.insert(name.clone());
                }
            }
            ImportType::NamedImports(symbols) => {
                for symbol in symbols {
                    let usage_count = self.usage_tracker.count_symbol_usage(ast, &symbol.name).await?;
                    if usage_count == 0 {
                        analysis.unused_symbols.push(symbol.name.clone());
                    } else {
                        analysis.used_symbols.insert(symbol.name.clone());
                        let locations = self.usage_tracker.find_symbol_usage_locations(ast, &symbol.name).await?;
                        analysis.usage_locations.extend(locations);
                    }
                }
            }
            ImportType::NamespaceImport(namespace) => {
                let usage_count = self.usage_tracker.count_namespace_usage(ast, namespace).await?;
                if usage_count == 0 {
                    analysis.unused_symbols.push(namespace.clone());
                } else {
                    analysis.used_symbols.insert(namespace.clone());
                }
            }
            ImportType::SideEffectImport => {
                // Side-effect imports are always considered used
                analysis.used_symbols.insert(import.module_name.clone());
            }
        }
        
        Ok(analysis)
    }
    
    async fn check_for_side_effects(&self, import: &ImportStatement) -> Result<bool, ImportAnalysisError> {
        // Check if the imported module might have side effects
        // This is language and framework specific
        
        match import.language {
            ProgrammingLanguage::Python => {
                // Check for known side-effect patterns
                self.check_python_side_effects(&import.module_name).await
            }
            ProgrammingLanguage::JavaScript | ProgrammingLanguage::TypeScript => {
                // Check for side-effect imports, CSS imports, etc.
                self.check_js_side_effects(&import.module_name).await
            }
            _ => Ok(false),
        }
    }
    
    async fn check_python_side_effects(&self, module_name: &str) -> Result<bool, ImportAnalysisError> {
        // Known Python modules with side effects
        let side_effect_modules = [
            "matplotlib.pyplot",
            "seaborn",
            "django.setup",
            "logging.config",
            "warnings",
        ];
        
        Ok(side_effect_modules.contains(&module_name) || 
           module_name.contains("__init__") ||
           module_name.contains("setup"))
    }
    
    async fn check_js_side_effects(&self, module_name: &str) -> Result<bool, ImportAnalysisError> {
        // Check for CSS imports, polyfills, etc.
        Ok(module_name.ends_with(".css") ||
           module_name.ends_with(".scss") ||
           module_name.ends_with(".less") ||
           module_name.contains("polyfill") ||
           module_name.contains("shim"))
    }
}

#[derive(Debug, Clone)]
pub struct UnusedImport {
    pub import_statement: ImportStatement,
    pub location: UnifiedPosition,
    pub import_type: ImportType,
    pub module_name: String,
    pub imported_symbols: Vec<String>,
    pub reason: UnusedReason,
    pub suggestion: String,
    pub confidence: f64,
    pub side_effects_possible: bool,
}

#[derive(Debug, Clone)]
pub struct ImportUsageAnalysis {
    pub used_symbols: HashSet<String>,
    pub unused_symbols: Vec<String>,
    pub usage_locations: Vec<UnifiedPosition>,
    pub confidence: f64,
    pub unused_reason: UnusedReason,
}

impl ImportUsageAnalysis {
    pub fn is_unused(&self) -> bool {
        self.used_symbols.is_empty() && !self.unused_symbols.is_empty()
    }
    
    pub fn has_partially_unused(&self) -> bool {
        !self.used_symbols.is_empty() && !self.unused_symbols.is_empty()
    }
}
```

### 12.6 Cross-Module Analysis

#### 12.6.1 Cross-Module Analyzer
```rust
pub struct CrossModuleAnalyzer {
    module_graph_builder: ModuleGraphBuilder,
    export_analyzer: ExportAnalyzer,
    dependency_resolver: DependencyResolver,
}

impl CrossModuleAnalyzer {
    pub async fn build_global_graph(&self, project_asts: &[UnifiedAST]) -> Result<GlobalDependencyGraph, CrossModuleError> {
        let mut global_graph = GlobalDependencyGraph::new();
        
        // Build module graph
        let module_graph = self.module_graph_builder.build_graph(project_asts).await?;
        
        // For each module, add symbols and dependencies
        for ast in project_asts {
            let module_id = self.get_module_id(&ast.file_path);
            
            // Add all symbols from this module
            let symbols = self.extract_symbols_from_ast(ast).await?;
            for symbol in symbols {
                global_graph.add_symbol(module_id.clone(), symbol);
            }
            
            // Add dependencies
            let dependencies = self.extract_dependencies_from_ast(ast).await?;
            for dep in dependencies {
                global_graph.add_dependency(module_id.clone(), dep);
            }
        }
        
        // Resolve cross-module references
        global_graph = self.resolve_cross_module_references(global_graph, project_asts).await?;
        
        Ok(global_graph)
    }
    
    pub async fn identify_entry_points(&self, project_asts: &[UnifiedAST], global_graph: &GlobalDependencyGraph) -> Result<Vec<EntryPoint>, CrossModuleError> {
        let mut entry_points = Vec::new();
        
        for ast in project_asts {
            // Check for main functions
            let main_functions = self.find_main_functions(ast).await?;
            entry_points.extend(main_functions);
            
            // Check for exported functions (public API)
            let exported_functions = self.find_exported_functions(ast).await?;
            entry_points.extend(exported_functions);
            
            // Check for test functions
            if self.is_test_file(&ast.file_path) {
                let test_functions = self.find_test_functions(ast).await?;
                entry_points.extend(test_functions);
            }
            
            // Check for framework-specific entry points
            let framework_entries = self.find_framework_entry_points(ast).await?;
            entry_points.extend(framework_entries);
        }
        
        Ok(entry_points)
    }
    
    async fn find_main_functions(&self, ast: &UnifiedAST) -> Result<Vec<EntryPoint>, CrossModuleError> {
        let mut entry_points = Vec::new();
        
        match ast.language {
            ProgrammingLanguage::Python => {
                // Look for if __name__ == "__main__": pattern
                let main_blocks = self.find_python_main_blocks(ast).await?;
                entry_points.extend(main_blocks);
            }
            ProgrammingLanguage::Rust => {
                // Look for fn main() functions
                let main_functions = self.find_rust_main_functions(ast).await?;
                entry_points.extend(main_functions);
            }
            ProgrammingLanguage::JavaScript | ProgrammingLanguage::TypeScript => {
                // Look for entry point patterns (CLI tools, etc.)
                let entry_patterns = self.find_js_entry_patterns(ast).await?;
                entry_points.extend(entry_patterns);
            }
        }
        
        Ok(entry_points)
    }
    
    async fn find_exported_functions(&self, ast: &UnifiedAST) -> Result<Vec<EntryPoint>, CrossModuleError> {
        let exports = self.export_analyzer.find_exports(ast).await?;
        
        let entry_points = exports.into_iter()
            .filter(|export| self.is_public_api_export(export))
            .map(|export| EntryPoint {
                symbol_id: export.symbol_id,
                entry_type: EntryPointType::PublicAPI,
                location: export.location,
                confidence: 1.0,
            })
            .collect();
        
        Ok(entry_points)
    }
    
    async fn detect_cross_module_issues(&self, global_graph: &GlobalDependencyGraph, reachable_symbols: &HashSet<SymbolId>) -> Result<Vec<CrossModuleIssue>, CrossModuleError> {
        let mut issues = Vec::new();
        
        // Find unused exports
        let unused_exports = self.find_unused_exports(global_graph, reachable_symbols).await?;
        issues.extend(unused_exports.into_iter().map(|export| CrossModuleIssue::UnusedExport(export)));
        
        // Find circular dependencies
        let circular_deps = self.detect_circular_dependencies(global_graph).await?;
        issues.extend(circular_deps.into_iter().map(|cycle| CrossModuleIssue::CircularDependency(cycle)));
        
        // Find orphaned modules
        let orphaned_modules = self.find_orphaned_modules(global_graph, reachable_symbols).await?;
        issues.extend(orphaned_modules.into_iter().map(|module| CrossModuleIssue::OrphanedModule(module)));
        
        Ok(issues)
    }
}

#[derive(Debug, Clone)]
pub struct GlobalDependencyGraph {
    modules: HashMap<ModuleId, ModuleInfo>,
    symbols: HashMap<SymbolId, SymbolInfo>,
    dependencies: HashMap<SymbolId, Vec<SymbolId>>,
    reverse_dependencies: HashMap<SymbolId, Vec<SymbolId>>,
}

#[derive(Debug, Clone)]
pub struct EntryPoint {
    pub symbol_id: SymbolId,
    pub entry_type: EntryPointType,
    pub location: UnifiedPosition,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum EntryPointType {
    MainFunction,
    PublicAPI,
    TestFunction,
    EventHandler,
    WebEndpoint,
    CLICommand,
    FrameworkHook,
}

#[derive(Debug, Clone)]
pub enum CrossModuleIssue {
    UnusedExport(UnusedExport),
    CircularDependency(CircularDependency),
    OrphanedModule(OrphanedModule),
    DeadModule(DeadModule),
}
```

### 12.7 Language-Specific Dead Code Detection

#### 12.7.1 Python Dead Code Detector
```rust
pub struct PythonDeadCodeDetector {
    base_detector: DeadCodeDetector,
}

impl PythonDeadCodeDetector {
    pub async fn detect_python_specific_dead_code(&self, ast: &UnifiedAST) -> Result<Vec<PythonDeadCode>, DeadCodeError> {
        let mut dead_code = Vec::new();
        
        // Detect unused __all__ exports
        dead_code.extend(self.detect_unused_all_exports(ast).await?);
        
        // Detect unused dunder methods
        dead_code.extend(self.detect_unused_dunder_methods(ast).await?);
        
        // Detect unused decorators
        dead_code.extend(self.detect_unused_decorators(ast).await?);
        
        // Detect unused exception classes
        dead_code.extend(self.detect_unused_exception_classes(ast).await?);
        
        // Detect unused comprehensions
        dead_code.extend(self.detect_unused_comprehensions(ast).await?);
        
        Ok(dead_code)
    }
    
    async fn detect_unused_all_exports(&self, ast: &UnifiedAST) -> Result<Vec<PythonDeadCode>, DeadCodeError> {
        let mut unused_exports = Vec::new();
        
        // Find __all__ declarations
        let all_declarations = self.find_all_declarations(ast);
        
        for all_decl in all_declarations {
            for exported_name in &all_decl.exported_names {
                let usage_count = self.count_external_usage(ast, exported_name).await?;
                
                if usage_count == 0 {
                    unused_exports.push(PythonDeadCode::UnusedAllExport {
                        name: exported_name.clone(),
                        location: all_decl.location.clone(),
                        suggestion: format!("Remove '{}' from __all__", exported_name),
                    });
                }
            }
        }
        
        Ok(unused_exports)
    }
    
    async fn detect_unused_dunder_methods(&self, ast: &UnifiedAST) -> Result<Vec<PythonDeadCode>, DeadCodeError> {
        let mut unused_methods = Vec::new();
        
        let dunder_methods = self.find_dunder_methods(ast);
        
        for method in dunder_methods {
            // Some dunder methods are called implicitly
            if self.is_implicitly_called_dunder(&method.name) {
                continue;
            }
            
            let usage_count = self.count_dunder_method_usage(ast, &method).await?;
            
            if usage_count == 0 {
                unused_methods.push(PythonDeadCode::UnusedDunderMethod {
                    name: method.name.clone(),
                    location: method.location.clone(),
                    class_name: method.class_name.clone(),
                    suggestion: format!("Remove unused dunder method {}", method.name),
                });
            }
        }
        
        Ok(unused_methods)
    }
}

#[derive(Debug, Clone)]
pub enum PythonDeadCode {
    UnusedAllExport {
        name: String,
        location: UnifiedPosition,
        suggestion: String,
    },
    UnusedDunderMethod {
        name: String,
        location: UnifiedPosition,
        class_name: String,
        suggestion: String,
    },
    UnusedDecorator {
        name: String,
        location: UnifiedPosition,
        suggestion: String,
    },
    UnusedComprehension {
        location: UnifiedPosition,
        comprehension_type: String,
        suggestion: String,
    },
}
```

### 12.8 Criterios de Completitud

#### 12.8.1 Entregables de la Fase
- [ ] Sistema de detección de código muerto implementado
- [ ] Reachability analyzer funcionando
- [ ] Data flow analyzer para variables
- [ ] Import analyzer para imports no utilizados
- [ ] Cross-module analyzer para proyectos
- [ ] Detectores específicos por lenguaje
- [ ] Sistema de confidence scoring
- [ ] Generador de sugerencias de fixes
- [ ] Performance optimizado para proyectos grandes
- [ ] Tests comprehensivos

#### 12.8.2 Criterios de Aceptación
- [ ] Detecta variables no utilizadas con alta precisión
- [ ] Identifica funciones y clases no utilizadas
- [ ] Encuentra imports innecesarios correctamente
- [ ] Detecta código inalcanzable después de returns
- [ ] Análisis cross-module funciona en proyectos reales
- [ ] False positives < 5% en casos típicos
- [ ] Performance acceptable para proyectos de 100k+ LOC
- [ ] Sugerencias de fixes son útiles y precisas
- [ ] Integration seamless con motor de reglas
- [ ] Soporte robusto para múltiples lenguajes

### 12.9 Performance Targets

#### 12.9.1 Benchmarks de Detección
- **Analysis speed**: <500ms para archivos típicos (1000 LOC)
- **Memory usage**: <100MB para proyectos medianos
- **Accuracy**: >95% precision, >90% recall
- **Cross-module analysis**: <5 segundos para proyectos de 50 archivos
- **False positive rate**: <5% en código típico

### 12.10 Estimación de Tiempo

#### 12.10.1 Breakdown de Tareas
- Diseño de arquitectura de detección: 4 días
- Reachability analyzer: 8 días
- Data flow analyzer: 10 días
- Import analyzer: 6 días
- Cross-module analyzer: 12 días
- Detectores específicos por lenguaje: 10 días
- Sistema de confidence y sugerencias: 5 días
- Performance optimization: 6 días
- Integration con motor de reglas: 4 días
- Testing comprehensivo: 8 días
- Documentación: 3 días

**Total estimado: 76 días de desarrollo**

### 12.11 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Capacidades avanzadas de detección de código muerto
- Análisis cross-module y cross-language
- Base sólida para optimización de código
- Foundation para detección de duplicación
- Sistema robusto de análisis de flujo de datos

La Fase 13 construirá sobre esta base implementando la detección de código duplicado y análisis de similitud.
