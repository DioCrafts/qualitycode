# Fase 9: Parser Especializado para Rust

## Objetivo General
Implementar un parser especializado para Rust que proporcione análisis semántico profundo, análisis de ownership y borrowing, detección de patrones idiomáticos de Rust, análisis de unsafe code, y capacidades avanzadas de inspección específicas del ecosistema Rust necesarias para el agente CodeAnt.

## Descripción Técnica Detallada

### 9.1 Arquitectura del Parser Rust Especializado

#### 9.1.1 Diseño del Sistema Rust
```
┌─────────────────────────────────────────┐
│           Rust Specialized Parser      │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │    Syn      │ │    Ownership        │ │
│  │   Parser    │ │    Analysis         │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │  Lifetime   │ │    Trait System     │ │
│  │  Analysis   │ │    Analysis         │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Unsafe    │ │     Cargo           │ │
│  │  Analysis   │ │    Analysis         │ │
│  └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
```

#### 9.1.2 Componentes Especializados
- **Syn Parser**: Parser nativo de Rust para AST
- **Ownership Analysis**: Análisis de ownership y borrowing
- **Lifetime Analysis**: Análisis de lifetimes y referencias
- **Trait System Analysis**: Análisis del sistema de traits
- **Unsafe Analysis**: Análisis de código unsafe
- **Cargo Analysis**: Análisis de dependencias y workspace
- **Macro Analysis**: Análisis de macros procedurales y declarativas

### 9.2 Core Rust Parser Implementation

#### 9.2.1 Rust Specialized Parser
```rust
use syn::{
    parse_file, parse_str, File, Item, ItemFn, ItemImpl, ItemStruct, ItemEnum, ItemTrait,
    Expr, Pat, Type, Lifetime, GenericParam, WhereClause, Attribute, Meta, Visibility,
};
use quote::ToTokens;
use proc_macro2::TokenStream;

pub struct RustSpecializedParser {
    syn_parser: SynParser,
    ownership_analyzer: Arc<OwnershipAnalyzer>,
    lifetime_analyzer: Arc<LifetimeAnalyzer>,
    trait_analyzer: Arc<TraitSystemAnalyzer>,
    unsafe_analyzer: Arc<UnsafeAnalyzer>,
    cargo_analyzer: Arc<CargoAnalyzer>,
    macro_analyzer: Arc<MacroAnalyzer>,
    config: RustParserConfig,
}

#[derive(Debug, Clone)]
pub struct RustParserConfig {
    pub edition: RustEdition,
    pub enable_ownership_analysis: bool,
    pub enable_lifetime_analysis: bool,
    pub enable_trait_analysis: bool,
    pub enable_unsafe_analysis: bool,
    pub enable_cargo_analysis: bool,
    pub enable_macro_analysis: bool,
    pub analyze_dependencies: bool,
    pub strict_mode: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RustEdition {
    Edition2015,
    Edition2018,
    Edition2021,
    Edition2024,
}

impl RustSpecializedParser {
    pub async fn new(config: RustParserConfig) -> Result<Self, RustParserError> {
        let syn_parser = SynParser::new();
        let ownership_analyzer = Arc::new(OwnershipAnalyzer::new());
        let lifetime_analyzer = Arc::new(LifetimeAnalyzer::new());
        let trait_analyzer = Arc::new(TraitSystemAnalyzer::new());
        let unsafe_analyzer = Arc::new(UnsafeAnalyzer::new());
        let cargo_analyzer = Arc::new(CargoAnalyzer::new());
        let macro_analyzer = Arc::new(MacroAnalyzer::new());
        
        Ok(Self {
            syn_parser,
            ownership_analyzer,
            lifetime_analyzer,
            trait_analyzer,
            unsafe_analyzer,
            cargo_analyzer,
            macro_analyzer,
            config,
        })
    }
    
    pub async fn parse_rust_file(&self, file_path: &Path) -> Result<RustAnalysisResult, RustParserError> {
        let start_time = Instant::now();
        
        // Read file content
        let content = tokio::fs::read_to_string(file_path).await
            .map_err(|e| RustParserError::FileReadError(e.to_string()))?;
        
        // Parse with syn
        let syn_ast = self.syn_parser.parse(&content)?;
        
        // Create analysis result
        let mut analysis_result = RustAnalysisResult {
            file_path: file_path.to_path_buf(),
            syn_ast: syn_ast.clone(),
            ownership_info: None,
            lifetime_info: None,
            trait_info: None,
            unsafe_info: None,
            cargo_info: None,
            macro_info: None,
            semantic_info: None,
            patterns: Vec::new(),
            metrics: RustMetrics::default(),
            parse_duration_ms: 0,
        };
        
        // Ownership analysis
        if self.config.enable_ownership_analysis {
            analysis_result.ownership_info = Some(
                self.ownership_analyzer.analyze_ownership(&syn_ast, &content).await?
            );
        }
        
        // Lifetime analysis
        if self.config.enable_lifetime_analysis {
            analysis_result.lifetime_info = Some(
                self.lifetime_analyzer.analyze_lifetimes(&syn_ast, &content).await?
            );
        }
        
        // Trait system analysis
        if self.config.enable_trait_analysis {
            analysis_result.trait_info = Some(
                self.trait_analyzer.analyze_traits(&syn_ast, &content).await?
            );
        }
        
        // Unsafe code analysis
        if self.config.enable_unsafe_analysis {
            analysis_result.unsafe_info = Some(
                self.unsafe_analyzer.analyze_unsafe(&syn_ast, &content).await?
            );
        }
        
        // Cargo analysis
        if self.config.enable_cargo_analysis {
            analysis_result.cargo_info = Some(
                self.cargo_analyzer.analyze_cargo(file_path).await?
            );
        }
        
        // Macro analysis
        if self.config.enable_macro_analysis {
            analysis_result.macro_info = Some(
                self.macro_analyzer.analyze_macros(&syn_ast, &content).await?
            );
        }
        
        // Semantic analysis
        analysis_result.semantic_info = Some(
            self.analyze_rust_semantics(&syn_ast, &content).await?
        );
        
        // Pattern detection
        analysis_result.patterns = self.detect_rust_patterns(&analysis_result).await?;
        
        // Calculate metrics
        analysis_result.metrics = self.calculate_rust_metrics(&analysis_result);
        analysis_result.parse_duration_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(analysis_result)
    }
}

#[derive(Debug, Clone)]
pub struct RustAnalysisResult {
    pub file_path: PathBuf,
    pub syn_ast: File,
    pub ownership_info: Option<OwnershipInfo>,
    pub lifetime_info: Option<LifetimeInfo>,
    pub trait_info: Option<TraitInfo>,
    pub unsafe_info: Option<UnsafeInfo>,
    pub cargo_info: Option<CargoInfo>,
    pub macro_info: Option<MacroInfo>,
    pub semantic_info: Option<RustSemanticInfo>,
    pub patterns: Vec<RustPattern>,
    pub metrics: RustMetrics,
    pub parse_duration_ms: u64,
}
```

#### 9.2.2 Syn Parser Integration
```rust
pub struct SynParser;

impl SynParser {
    pub fn new() -> Self {
        Self
    }
    
    pub fn parse(&self, content: &str) -> Result<File, RustParserError> {
        parse_file(content)
            .map_err(|e| RustParserError::SynParseError(e.to_string()))
    }
    
    pub fn parse_item(&self, content: &str) -> Result<Item, RustParserError> {
        parse_str::<Item>(content)
            .map_err(|e| RustParserError::SynParseError(e.to_string()))
    }
    
    pub fn parse_expr(&self, content: &str) -> Result<Expr, RustParserError> {
        parse_str::<Expr>(content)
            .map_err(|e| RustParserError::SynParseError(e.to_string()))
    }
    
    pub fn parse_type(&self, content: &str) -> Result<Type, RustParserError> {
        parse_str::<Type>(content)
            .map_err(|e| RustParserError::SynParseError(e.to_string()))
    }
}
```

### 9.3 Ownership and Borrowing Analysis

#### 9.3.1 Ownership Analyzer
```rust
pub struct OwnershipAnalyzer {
    borrow_checker: BorrowChecker,
    move_analyzer: MoveAnalyzer,
    reference_analyzer: ReferenceAnalyzer,
}

#[derive(Debug, Clone)]
pub struct OwnershipInfo {
    pub ownership_transfers: Vec<OwnershipTransfer>,
    pub borrowing_patterns: Vec<BorrowingPattern>,
    pub reference_lifetimes: Vec<ReferenceLifetime>,
    pub move_semantics: Vec<MoveSemantics>,
    pub ownership_violations: Vec<OwnershipViolation>,
    pub borrow_checker_errors: Vec<BorrowCheckerError>,
    pub smart_pointer_usage: Vec<SmartPointerUsage>,
}

#[derive(Debug, Clone)]
pub struct OwnershipTransfer {
    pub variable_name: String,
    pub from_scope: ScopeId,
    pub to_scope: ScopeId,
    pub transfer_type: TransferType,
    pub span: proc_macro2::Span,
}

#[derive(Debug, Clone)]
pub enum TransferType {
    Move,
    Copy,
    Clone,
    Reference,
    MutableReference,
}

#[derive(Debug, Clone)]
pub struct BorrowingPattern {
    pub pattern_type: BorrowType,
    pub variable_name: String,
    pub borrowed_from: String,
    pub lifetime: Option<String>,
    pub span: proc_macro2::Span,
    pub is_mutable: bool,
}

#[derive(Debug, Clone)]
pub enum BorrowType {
    ImmutableBorrow,
    MutableBorrow,
    SharedReference,
    UniqueReference,
}

#[derive(Debug, Clone)]
pub struct MoveSemantics {
    pub moved_variable: String,
    pub move_location: proc_macro2::Span,
    pub subsequent_uses: Vec<proc_macro2::Span>,
    pub is_valid_move: bool,
    pub move_reason: MoveReason,
}

#[derive(Debug, Clone)]
pub enum MoveReason {
    FunctionCall,
    Assignment,
    Return,
    MethodCall,
    PatternMatch,
    Closure,
}

impl OwnershipAnalyzer {
    pub fn new() -> Self {
        Self {
            borrow_checker: BorrowChecker::new(),
            move_analyzer: MoveAnalyzer::new(),
            reference_analyzer: ReferenceAnalyzer::new(),
        }
    }
    
    pub async fn analyze_ownership(&self, ast: &File, content: &str) -> Result<OwnershipInfo, OwnershipError> {
        let mut ownership_info = OwnershipInfo {
            ownership_transfers: Vec::new(),
            borrowing_patterns: Vec::new(),
            reference_lifetimes: Vec::new(),
            move_semantics: Vec::new(),
            ownership_violations: Vec::new(),
            borrow_checker_errors: Vec::new(),
            smart_pointer_usage: Vec::new(),
        };
        
        // Analyze ownership transfers
        ownership_info.ownership_transfers = self.analyze_ownership_transfers(ast)?;
        
        // Analyze borrowing patterns
        ownership_info.borrowing_patterns = self.analyze_borrowing_patterns(ast)?;
        
        // Analyze move semantics
        ownership_info.move_semantics = self.move_analyzer.analyze_moves(ast)?;
        
        // Check for ownership violations
        ownership_info.ownership_violations = self.detect_ownership_violations(ast)?;
        
        // Simulate borrow checker
        ownership_info.borrow_checker_errors = self.borrow_checker.check_borrows(ast)?;
        
        // Analyze smart pointer usage
        ownership_info.smart_pointer_usage = self.analyze_smart_pointers(ast)?;
        
        Ok(ownership_info)
    }
    
    fn analyze_ownership_transfers(&self, ast: &File) -> Result<Vec<OwnershipTransfer>, OwnershipError> {
        let mut transfers = Vec::new();
        let mut visitor = OwnershipTransferVisitor::new(&mut transfers);
        visitor.visit_file(ast);
        Ok(transfers)
    }
    
    fn analyze_borrowing_patterns(&self, ast: &File) -> Result<Vec<BorrowingPattern>, OwnershipError> {
        let mut patterns = Vec::new();
        let mut visitor = BorrowingPatternVisitor::new(&mut patterns);
        visitor.visit_file(ast);
        Ok(patterns)
    }
    
    fn analyze_smart_pointers(&self, ast: &File) -> Result<Vec<SmartPointerUsage>, OwnershipError> {
        let mut smart_pointers = Vec::new();
        let mut visitor = SmartPointerVisitor::new(&mut smart_pointers);
        visitor.visit_file(ast);
        Ok(smart_pointers)
    }
}

#[derive(Debug, Clone)]
pub struct SmartPointerUsage {
    pub pointer_type: SmartPointerType,
    pub variable_name: String,
    pub span: proc_macro2::Span,
    pub usage_pattern: SmartPointerPattern,
    pub potential_issues: Vec<SmartPointerIssue>,
}

#[derive(Debug, Clone)]
pub enum SmartPointerType {
    Box,
    Rc,
    Arc,
    RefCell,
    Mutex,
    RwLock,
    Weak,
    Cow,
}

#[derive(Debug, Clone)]
pub enum SmartPointerPattern {
    SingleOwnership,
    SharedOwnership,
    InteriorMutability,
    ThreadSafety,
    CopyOnWrite,
}

#[derive(Debug, Clone)]
pub struct SmartPointerIssue {
    pub issue_type: SmartPointerIssueType,
    pub message: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone)]
pub enum SmartPointerIssueType {
    UnnecessaryBoxing,
    PotentialCycle,
    ThreadSafetyViolation,
    PerformanceImpact,
    MemoryLeak,
}
```

### 9.4 Lifetime Analysis System

#### 9.4.1 Lifetime Analyzer
```rust
pub struct LifetimeAnalyzer {
    lifetime_checker: LifetimeChecker,
    elision_analyzer: ElisionAnalyzer,
    variance_analyzer: VarianceAnalyzer,
}

#[derive(Debug, Clone)]
pub struct LifetimeInfo {
    pub explicit_lifetimes: Vec<ExplicitLifetime>,
    pub elided_lifetimes: Vec<ElidedLifetime>,
    pub lifetime_bounds: Vec<LifetimeBound>,
    pub lifetime_relationships: Vec<LifetimeRelationship>,
    pub lifetime_errors: Vec<LifetimeError>,
    pub variance_annotations: Vec<VarianceAnnotation>,
}

#[derive(Debug, Clone)]
pub struct ExplicitLifetime {
    pub name: String,
    pub span: proc_macro2::Span,
    pub scope: LifetimeScope,
    pub constraints: Vec<LifetimeConstraint>,
}

#[derive(Debug, Clone)]
pub enum LifetimeScope {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Generic,
}

#[derive(Debug, Clone)]
pub struct ElidedLifetime {
    pub location: proc_macro2::Span,
    pub inferred_lifetime: String,
    pub elision_rule: ElisionRule,
}

#[derive(Debug, Clone)]
pub enum ElisionRule {
    SingleInput,
    SelfReference,
    MultipleInputs,
    StaticLifetime,
}

#[derive(Debug, Clone)]
pub struct LifetimeRelationship {
    pub lifetime_a: String,
    pub lifetime_b: String,
    pub relationship: RelationshipType,
    pub span: proc_macro2::Span,
}

#[derive(Debug, Clone)]
pub enum RelationshipType {
    Outlives,
    Equal,
    Contains,
    Disjoint,
}

impl LifetimeAnalyzer {
    pub fn new() -> Self {
        Self {
            lifetime_checker: LifetimeChecker::new(),
            elision_analyzer: ElisionAnalyzer::new(),
            variance_analyzer: VarianceAnalyzer::new(),
        }
    }
    
    pub async fn analyze_lifetimes(&self, ast: &File, content: &str) -> Result<LifetimeInfo, LifetimeError> {
        let mut lifetime_info = LifetimeInfo {
            explicit_lifetimes: Vec::new(),
            elided_lifetimes: Vec::new(),
            lifetime_bounds: Vec::new(),
            lifetime_relationships: Vec::new(),
            lifetime_errors: Vec::new(),
            variance_annotations: Vec::new(),
        };
        
        // Extract explicit lifetimes
        lifetime_info.explicit_lifetimes = self.extract_explicit_lifetimes(ast)?;
        
        // Analyze lifetime elision
        lifetime_info.elided_lifetimes = self.elision_analyzer.analyze_elision(ast)?;
        
        // Extract lifetime bounds
        lifetime_info.lifetime_bounds = self.extract_lifetime_bounds(ast)?;
        
        // Analyze lifetime relationships
        lifetime_info.lifetime_relationships = self.analyze_lifetime_relationships(ast)?;
        
        // Check for lifetime errors
        lifetime_info.lifetime_errors = self.lifetime_checker.check_lifetimes(ast)?;
        
        // Analyze variance
        lifetime_info.variance_annotations = self.variance_analyzer.analyze_variance(ast)?;
        
        Ok(lifetime_info)
    }
    
    fn extract_explicit_lifetimes(&self, ast: &File) -> Result<Vec<ExplicitLifetime>, LifetimeError> {
        let mut lifetimes = Vec::new();
        let mut visitor = LifetimeVisitor::new(&mut lifetimes);
        visitor.visit_file(ast);
        Ok(lifetimes)
    }
    
    fn extract_lifetime_bounds(&self, ast: &File) -> Result<Vec<LifetimeBound>, LifetimeError> {
        let mut bounds = Vec::new();
        let mut visitor = LifetimeBoundVisitor::new(&mut bounds);
        visitor.visit_file(ast);
        Ok(bounds)
    }
}
```

### 9.5 Trait System Analysis

#### 9.5.1 Trait System Analyzer
```rust
pub struct TraitSystemAnalyzer {
    trait_resolver: TraitResolver,
    coherence_checker: CoherenceChecker,
    associated_type_analyzer: AssociatedTypeAnalyzer,
}

#[derive(Debug, Clone)]
pub struct TraitInfo {
    pub trait_definitions: Vec<TraitDefinition>,
    pub trait_implementations: Vec<TraitImplementation>,
    pub associated_types: Vec<AssociatedType>,
    pub trait_bounds: Vec<TraitBound>,
    pub trait_objects: Vec<TraitObject>,
    pub coherence_violations: Vec<CoherenceViolation>,
    pub orphan_rule_violations: Vec<OrphanRuleViolation>,
    pub higher_ranked_trait_bounds: Vec<HRTB>,
}

#[derive(Debug, Clone)]
pub struct TraitDefinition {
    pub name: String,
    pub span: proc_macro2::Span,
    pub associated_types: Vec<String>,
    pub associated_functions: Vec<AssociatedFunction>,
    pub default_implementations: Vec<DefaultImplementation>,
    pub super_traits: Vec<String>,
    pub is_object_safe: bool,
    pub visibility: Visibility,
}

#[derive(Debug, Clone)]
pub struct TraitImplementation {
    pub trait_name: String,
    pub implementing_type: String,
    pub span: proc_macro2::Span,
    pub implementation_type: ImplementationType,
    pub associated_type_bindings: Vec<AssociatedTypeBinding>,
    pub implemented_methods: Vec<String>,
    pub is_blanket_impl: bool,
    pub where_clause: Option<WhereClause>,
}

#[derive(Debug, Clone)]
pub enum ImplementationType {
    InherentImpl,
    TraitImpl,
    BlanketImpl,
    ConditionalImpl,
}

#[derive(Debug, Clone)]
pub struct AssociatedType {
    pub name: String,
    pub trait_name: String,
    pub span: proc_macro2::Span,
    pub bounds: Vec<TypeBound>,
    pub default_type: Option<Type>,
}

#[derive(Debug, Clone)]
pub struct TraitObject {
    pub traits: Vec<String>,
    pub span: proc_macro2::Span,
    pub is_dyn: bool,
    pub lifetime: Option<String>,
    pub is_object_safe: bool,
}

impl TraitSystemAnalyzer {
    pub fn new() -> Self {
        Self {
            trait_resolver: TraitResolver::new(),
            coherence_checker: CoherenceChecker::new(),
            associated_type_analyzer: AssociatedTypeAnalyzer::new(),
        }
    }
    
    pub async fn analyze_traits(&self, ast: &File, content: &str) -> Result<TraitInfo, TraitError> {
        let mut trait_info = TraitInfo {
            trait_definitions: Vec::new(),
            trait_implementations: Vec::new(),
            associated_types: Vec::new(),
            trait_bounds: Vec::new(),
            trait_objects: Vec::new(),
            coherence_violations: Vec::new(),
            orphan_rule_violations: Vec::new(),
            higher_ranked_trait_bounds: Vec::new(),
        };
        
        // Extract trait definitions
        trait_info.trait_definitions = self.extract_trait_definitions(ast)?;
        
        // Extract trait implementations
        trait_info.trait_implementations = self.extract_trait_implementations(ast)?;
        
        // Analyze associated types
        trait_info.associated_types = self.associated_type_analyzer.analyze(ast)?;
        
        // Extract trait bounds
        trait_info.trait_bounds = self.extract_trait_bounds(ast)?;
        
        // Extract trait objects
        trait_info.trait_objects = self.extract_trait_objects(ast)?;
        
        // Check coherence
        trait_info.coherence_violations = self.coherence_checker.check_coherence(&trait_info)?;
        
        // Check orphan rule
        trait_info.orphan_rule_violations = self.check_orphan_rule(&trait_info)?;
        
        // Extract HRTB
        trait_info.higher_ranked_trait_bounds = self.extract_hrtb(ast)?;
        
        Ok(trait_info)
    }
    
    fn extract_trait_definitions(&self, ast: &File) -> Result<Vec<TraitDefinition>, TraitError> {
        let mut traits = Vec::new();
        
        for item in &ast.items {
            if let Item::Trait(trait_item) = item {
                let mut associated_types = Vec::new();
                let mut associated_functions = Vec::new();
                let mut default_implementations = Vec::new();
                
                for trait_item in &trait_item.items {
                    match trait_item {
                        syn::TraitItem::Type(type_item) => {
                            associated_types.push(type_item.ident.to_string());
                        }
                        syn::TraitItem::Method(method_item) => {
                            let func = AssociatedFunction {
                                name: method_item.sig.ident.to_string(),
                                signature: method_item.sig.clone(),
                                has_default: method_item.default.is_some(),
                            };
                            associated_functions.push(func);
                            
                            if method_item.default.is_some() {
                                default_implementations.push(DefaultImplementation {
                                    method_name: method_item.sig.ident.to_string(),
                                    implementation: method_item.default.as_ref().unwrap().clone(),
                                });
                            }
                        }
                        _ => {}
                    }
                }
                
                let super_traits = trait_item.supertraits.iter()
                    .filter_map(|bound| {
                        if let syn::TypeParamBound::Trait(trait_bound) = bound {
                            Some(trait_bound.path.segments.last()?.ident.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();
                
                let trait_def = TraitDefinition {
                    name: trait_item.ident.to_string(),
                    span: trait_item.ident.span(),
                    associated_types,
                    associated_functions,
                    default_implementations,
                    super_traits,
                    is_object_safe: self.check_object_safety(trait_item),
                    visibility: trait_item.vis.clone(),
                };
                
                traits.push(trait_def);
            }
        }
        
        Ok(traits)
    }
    
    fn check_object_safety(&self, trait_item: &syn::ItemTrait) -> bool {
        // Simplified object safety check
        for item in &trait_item.items {
            match item {
                syn::TraitItem::Method(method) => {
                    // Check if method has Self in return type or parameters
                    if self.method_uses_self_type(&method.sig) {
                        return false;
                    }
                }
                syn::TraitItem::Type(_) => {
                    // Associated types make traits non-object-safe in some cases
                    // This is a simplified check
                }
                _ => {}
            }
        }
        true
    }
    
    fn method_uses_self_type(&self, sig: &syn::Signature) -> bool {
        // Simplified check for Self usage in method signature
        // In reality, this would need more sophisticated type analysis
        false
    }
}
```

### 9.6 Unsafe Code Analysis

#### 9.6.1 Unsafe Analyzer
```rust
pub struct UnsafeAnalyzer {
    unsafe_detector: UnsafeDetector,
    safety_checker: SafetyChecker,
    ffi_analyzer: FFIAnalyzer,
}

#[derive(Debug, Clone)]
pub struct UnsafeInfo {
    pub unsafe_blocks: Vec<UnsafeBlock>,
    pub unsafe_functions: Vec<UnsafeFunction>,
    pub unsafe_traits: Vec<UnsafeTrait>,
    pub raw_pointer_usage: Vec<RawPointerUsage>,
    pub ffi_declarations: Vec<FFIDeclaration>,
    pub transmute_usage: Vec<TransmuteUsage>,
    pub inline_assembly: Vec<InlineAssembly>,
    pub safety_violations: Vec<SafetyViolation>,
    pub safety_comments: Vec<SafetyComment>,
}

#[derive(Debug, Clone)]
pub struct UnsafeBlock {
    pub span: proc_macro2::Span,
    pub operations: Vec<UnsafeOperation>,
    pub has_safety_comment: bool,
    pub safety_justification: Option<String>,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone)]
pub enum UnsafeOperation {
    RawPointerDereference,
    CallUnsafeFunction,
    AccessMutableStatic,
    ImplementUnsafeTrait,
    TransmuteCall,
    InlineAssembly,
    UnionFieldAccess,
}

#[derive(Debug, Clone)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct UnsafeFunction {
    pub name: String,
    pub span: proc_macro2::Span,
    pub safety_requirements: Vec<SafetyRequirement>,
    pub has_safety_doc: bool,
    pub is_ffi: bool,
}

#[derive(Debug, Clone)]
pub struct RawPointerUsage {
    pub pointer_type: RawPointerType,
    pub span: proc_macro2::Span,
    pub operation: PointerOperation,
    pub safety_analysis: PointerSafetyAnalysis,
}

#[derive(Debug, Clone)]
pub enum RawPointerType {
    Const,
    Mut,
}

#[derive(Debug, Clone)]
pub enum PointerOperation {
    Creation,
    Dereference,
    Arithmetic,
    Cast,
    Comparison,
}

#[derive(Debug, Clone)]
pub struct PointerSafetyAnalysis {
    pub is_null_checked: bool,
    pub is_bounds_checked: bool,
    pub alignment_verified: bool,
    pub lifetime_guaranteed: bool,
}

impl UnsafeAnalyzer {
    pub fn new() -> Self {
        Self {
            unsafe_detector: UnsafeDetector::new(),
            safety_checker: SafetyChecker::new(),
            ffi_analyzer: FFIAnalyzer::new(),
        }
    }
    
    pub async fn analyze_unsafe(&self, ast: &File, content: &str) -> Result<UnsafeInfo, UnsafeError> {
        let mut unsafe_info = UnsafeInfo {
            unsafe_blocks: Vec::new(),
            unsafe_functions: Vec::new(),
            unsafe_traits: Vec::new(),
            raw_pointer_usage: Vec::new(),
            ffi_declarations: Vec::new(),
            transmute_usage: Vec::new(),
            inline_assembly: Vec::new(),
            safety_violations: Vec::new(),
            safety_comments: Vec::new(),
        };
        
        // Detect unsafe blocks
        unsafe_info.unsafe_blocks = self.unsafe_detector.detect_unsafe_blocks(ast, content)?;
        
        // Detect unsafe functions
        unsafe_info.unsafe_functions = self.detect_unsafe_functions(ast)?;
        
        // Detect unsafe traits
        unsafe_info.unsafe_traits = self.detect_unsafe_traits(ast)?;
        
        // Analyze raw pointer usage
        unsafe_info.raw_pointer_usage = self.analyze_raw_pointers(ast)?;
        
        // Analyze FFI
        unsafe_info.ffi_declarations = self.ffi_analyzer.analyze_ffi(ast)?;
        
        // Detect transmute usage
        unsafe_info.transmute_usage = self.detect_transmute_usage(ast)?;
        
        // Detect inline assembly
        unsafe_info.inline_assembly = self.detect_inline_assembly(ast)?;
        
        // Check safety violations
        unsafe_info.safety_violations = self.safety_checker.check_safety(&unsafe_info)?;
        
        // Extract safety comments
        unsafe_info.safety_comments = self.extract_safety_comments(content)?;
        
        Ok(unsafe_info)
    }
    
    fn detect_unsafe_functions(&self, ast: &File) -> Result<Vec<UnsafeFunction>, UnsafeError> {
        let mut unsafe_functions = Vec::new();
        
        for item in &ast.items {
            match item {
                Item::Fn(func) => {
                    if func.sig.unsafety.is_some() {
                        let safety_requirements = self.extract_safety_requirements(&func.attrs);
                        let has_safety_doc = self.has_safety_documentation(&func.attrs);
                        
                        unsafe_functions.push(UnsafeFunction {
                            name: func.sig.ident.to_string(),
                            span: func.sig.ident.span(),
                            safety_requirements,
                            has_safety_doc,
                            is_ffi: self.is_ffi_function(func),
                        });
                    }
                }
                Item::Impl(impl_item) => {
                    for impl_item in &impl_item.items {
                        if let syn::ImplItem::Method(method) = impl_item {
                            if method.sig.unsafety.is_some() {
                                let safety_requirements = self.extract_safety_requirements(&method.attrs);
                                let has_safety_doc = self.has_safety_documentation(&method.attrs);
                                
                                unsafe_functions.push(UnsafeFunction {
                                    name: method.sig.ident.to_string(),
                                    span: method.sig.ident.span(),
                                    safety_requirements,
                                    has_safety_doc,
                                    is_ffi: false,
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        
        Ok(unsafe_functions)
    }
    
    fn extract_safety_requirements(&self, attrs: &[Attribute]) -> Vec<SafetyRequirement> {
        let mut requirements = Vec::new();
        
        for attr in attrs {
            if let Ok(Meta::NameValue(meta)) = attr.parse_meta() {
                if meta.path.is_ident("safety") {
                    if let syn::Lit::Str(lit_str) = meta.lit {
                        requirements.push(SafetyRequirement {
                            description: lit_str.value(),
                            requirement_type: SafetyRequirementType::General,
                        });
                    }
                }
            }
        }
        
        requirements
    }
    
    fn has_safety_documentation(&self, attrs: &[Attribute]) -> bool {
        for attr in attrs {
            if attr.path.is_ident("doc") {
                if let Ok(Meta::NameValue(meta)) = attr.parse_meta() {
                    if let syn::Lit::Str(lit_str) = meta.lit {
                        let doc = lit_str.value().to_lowercase();
                        if doc.contains("safety") || doc.contains("unsafe") {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

#[derive(Debug, Clone)]
pub struct SafetyRequirement {
    pub description: String,
    pub requirement_type: SafetyRequirementType,
}

#[derive(Debug, Clone)]
pub enum SafetyRequirementType {
    General,
    NullPointerCheck,
    BoundsCheck,
    AlignmentCheck,
    LifetimeGuarantee,
    ThreadSafety,
    MemoryLayout,
}
```

### 9.7 Cargo and Dependency Analysis

#### 9.7.1 Cargo Analyzer
```rust
pub struct CargoAnalyzer {
    manifest_parser: ManifestParser,
    dependency_analyzer: DependencyAnalyzer,
    workspace_analyzer: WorkspaceAnalyzer,
}

#[derive(Debug, Clone)]
pub struct CargoInfo {
    pub manifest: Option<CargoManifest>,
    pub dependencies: Vec<Dependency>,
    pub dev_dependencies: Vec<Dependency>,
    pub build_dependencies: Vec<Dependency>,
    pub features: Vec<Feature>,
    pub workspace_info: Option<WorkspaceInfo>,
    pub security_advisories: Vec<SecurityAdvisory>,
    pub outdated_dependencies: Vec<OutdatedDependency>,
    pub license_issues: Vec<LicenseIssue>,
}

#[derive(Debug, Clone)]
pub struct CargoManifest {
    pub package: PackageInfo,
    pub edition: RustEdition,
    pub rust_version: Option<String>,
    pub categories: Vec<String>,
    pub keywords: Vec<String>,
    pub readme: Option<String>,
    pub repository: Option<String>,
    pub license: Option<String>,
    pub authors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub source: DependencySource,
    pub features: Vec<String>,
    pub optional: bool,
    pub default_features: bool,
    pub security_rating: SecurityRating,
    pub maintenance_status: MaintenanceStatus,
}

#[derive(Debug, Clone)]
pub enum DependencySource {
    CratesIo,
    Git { url: String, branch: Option<String> },
    Path(String),
    Registry(String),
}

#[derive(Debug, Clone)]
pub enum SecurityRating {
    Secure,
    LowRisk,
    MediumRisk,
    HighRisk,
    Critical,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum MaintenanceStatus {
    ActivelyMaintained,
    Maintained,
    MinimalMaintenance,
    Unmaintained,
    Deprecated,
    Unknown,
}

impl CargoAnalyzer {
    pub fn new() -> Self {
        Self {
            manifest_parser: ManifestParser::new(),
            dependency_analyzer: DependencyAnalyzer::new(),
            workspace_analyzer: WorkspaceAnalyzer::new(),
        }
    }
    
    pub async fn analyze_cargo(&self, file_path: &Path) -> Result<CargoInfo, CargoError> {
        let cargo_toml_path = self.find_cargo_toml(file_path)?;
        
        let mut cargo_info = CargoInfo {
            manifest: None,
            dependencies: Vec::new(),
            dev_dependencies: Vec::new(),
            build_dependencies: Vec::new(),
            features: Vec::new(),
            workspace_info: None,
            security_advisories: Vec::new(),
            outdated_dependencies: Vec::new(),
            license_issues: Vec::new(),
        };
        
        // Parse Cargo.toml
        if let Some(cargo_path) = cargo_toml_path {
            cargo_info.manifest = Some(self.manifest_parser.parse_manifest(&cargo_path).await?);
            
            // Analyze dependencies
            let dep_analysis = self.dependency_analyzer.analyze_dependencies(&cargo_path).await?;
            cargo_info.dependencies = dep_analysis.dependencies;
            cargo_info.dev_dependencies = dep_analysis.dev_dependencies;
            cargo_info.build_dependencies = dep_analysis.build_dependencies;
            
            // Extract features
            cargo_info.features = self.extract_features(&cargo_path).await?;
            
            // Analyze workspace
            cargo_info.workspace_info = self.workspace_analyzer.analyze_workspace(&cargo_path).await?;
            
            // Security analysis
            cargo_info.security_advisories = self.check_security_advisories(&cargo_info.dependencies).await?;
            
            // Check for outdated dependencies
            cargo_info.outdated_dependencies = self.check_outdated_dependencies(&cargo_info.dependencies).await?;
            
            // License analysis
            cargo_info.license_issues = self.analyze_licenses(&cargo_info).await?;
        }
        
        Ok(cargo_info)
    }
    
    fn find_cargo_toml(&self, file_path: &Path) -> Result<Option<PathBuf>, CargoError> {
        let mut current = file_path.parent();
        
        while let Some(dir) = current {
            let cargo_toml = dir.join("Cargo.toml");
            if cargo_toml.exists() {
                return Ok(Some(cargo_toml));
            }
            current = dir.parent();
        }
        
        Ok(None)
    }
    
    async fn check_security_advisories(&self, dependencies: &[Dependency]) -> Result<Vec<SecurityAdvisory>, CargoError> {
        let mut advisories = Vec::new();
        
        // This would integrate with RustSec Advisory Database
        for dependency in dependencies {
            if let Some(advisory) = self.check_rustsec_advisory(dependency).await? {
                advisories.push(advisory);
            }
        }
        
        Ok(advisories)
    }
    
    async fn check_rustsec_advisory(&self, dependency: &Dependency) -> Result<Option<SecurityAdvisory>, CargoError> {
        // Integration with RustSec Advisory Database
        // This would make HTTP requests to check for known vulnerabilities
        Ok(None)
    }
}
```

### 9.8 Rust Pattern Detection

#### 9.8.1 Rust-Specific Pattern Detector
```rust
pub struct RustPatternDetector {
    patterns: Vec<Box<dyn RustPattern>>,
    config: RustPatternConfig,
}

#[async_trait]
pub trait RustPattern: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn category(&self) -> RustPatternCategory;
    fn severity(&self) -> PatternSeverity;
    
    async fn detect(&self, analysis_result: &RustAnalysisResult) -> Result<Vec<RustPatternMatch>, PatternError>;
}

#[derive(Debug, Clone)]
pub enum RustPatternCategory {
    Ownership,
    Borrowing,
    Lifetimes,
    Safety,
    Performance,
    Idioms,
    ErrorHandling,
    Concurrency,
    Macros,
    FFI,
}

// Example: Unnecessary Clone Pattern
pub struct UnnecessaryClonePattern;

#[async_trait]
impl RustPattern for UnnecessaryClonePattern {
    fn name(&self) -> &str {
        "unnecessary_clone"
    }
    
    fn description(&self) -> &str {
        "Detects unnecessary .clone() calls that could be avoided"
    }
    
    fn category(&self) -> RustPatternCategory {
        RustPatternCategory::Performance
    }
    
    fn severity(&self) -> PatternSeverity {
        PatternSeverity::Medium
    }
    
    async fn detect(&self, analysis_result: &RustAnalysisResult) -> Result<Vec<RustPatternMatch>, PatternError> {
        let mut matches = Vec::new();
        let mut visitor = CloneVisitor::new(&mut matches);
        visitor.visit_file(&analysis_result.syn_ast);
        Ok(matches)
    }
}

// Example: Unwrap Usage Pattern
pub struct UnwrapUsagePattern;

#[async_trait]
impl RustPattern for UnwrapUsagePattern {
    fn name(&self) -> &str {
        "unwrap_usage"
    }
    
    fn description(&self) -> &str {
        "Detects usage of .unwrap() which can cause panics"
    }
    
    fn category(&self) -> RustPatternCategory {
        RustPatternCategory::ErrorHandling
    }
    
    fn severity(&self) -> PatternSeverity {
        PatternSeverity::High
    }
    
    async fn detect(&self, analysis_result: &RustAnalysisResult) -> Result<Vec<RustPatternMatch>, PatternError> {
        let mut matches = Vec::new();
        let mut visitor = UnwrapVisitor::new(&mut matches);
        visitor.visit_file(&analysis_result.syn_ast);
        Ok(matches)
    }
}

// Additional patterns to implement:
// - DerefCoercionPattern
// - IteratorChainPattern
// - MatchErgonomicsPattern
// - BorrowCheckerBypassPattern
// - LifetimeElisionPattern
// - TraitObjectPattern
// - ZeroCostAbstractionPattern
// - RustIdiomPattern
// - ConcurrencyPattern
// - UnsafePatternPattern
```

### 9.9 Criterios de Completitud

#### 9.9.1 Entregables de la Fase
- [ ] Parser Rust especializado implementado
- [ ] Análisis de ownership y borrowing
- [ ] Análisis de lifetimes
- [ ] Análisis del sistema de traits
- [ ] Análisis de código unsafe
- [ ] Análisis de Cargo y dependencias
- [ ] Detección de patrones Rust-específicos
- [ ] Cálculo de métricas específicas de Rust
- [ ] Integration con parser universal
- [ ] Tests comprehensivos para Rust
- [ ] Documentación completa

#### 9.9.2 Criterios de Aceptación
- [ ] Parse correctamente código Rust complejo
- [ ] Análisis de ownership detecta violaciones
- [ ] Análisis de lifetimes identifica problemas
- [ ] Análisis de traits funciona correctamente
- [ ] Análisis unsafe detecta riesgos de seguridad
- [ ] Análisis Cargo identifica dependencias problemáticas
- [ ] Pattern detection encuentra antipatrones comunes
- [ ] Métricas calculadas son precisas
- [ ] Performance acceptable para proyectos grandes
- [ ] Integration seamless con sistema principal

### 9.10 Performance Targets

#### 9.10.1 Benchmarks Específicos Rust
- **Parsing speed**: >600 lines/second para Rust
- **Ownership analysis**: <4x overhead sobre parsing básico
- **Trait analysis**: <3x overhead sobre parsing básico
- **Memory usage**: <30MB por archivo Rust típico
- **Pattern detection**: <2 segundos para archivos <2000 lines

### 9.11 Estimación de Tiempo

#### 9.11.1 Breakdown de Tareas
- Setup Syn integration: 3 días
- Ownership and borrowing analysis: 10 días
- Lifetime analysis system: 8 días
- Trait system analysis: 9 días
- Unsafe code analysis: 7 días
- Cargo analysis engine: 6 días
- Macro analysis (basic): 5 días
- Pattern detection framework: 6 días
- Rust-specific patterns (12+): 12 días
- Metrics calculation: 4 días
- Integration con parser universal: 3 días
- Testing comprehensivo: 8 días
- Performance optimization: 6 días
- Documentación: 3 días

**Total estimado: 90 días de desarrollo**

### 9.12 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Análisis Rust extremadamente profundo y especializado
- Capacidades únicas de análisis de ownership y borrowing
- Detección de patrones idiomáticos de Rust
- Análisis de seguridad para código unsafe
- Foundation completa para análisis de calidad de código Rust

La Fase 10 unificará todos los parsers especializados en un sistema coherente de representación AST cross-language, completando la base de parsing del agente CodeAnt.
