# Fase 11: Motor de Reglas Estáticas Configurable

## Objetivo General
Implementar un motor de reglas estáticas robusto y altamente configurable que pueda ejecutar más de 30,000 reglas de análisis de código, aprovechando el sistema AST unificado para proporcionar análisis cross-language, detección de antipatrones, y verificación de estándares de calidad con performance excepcional.

## Descripción Técnica Detallada

### 11.1 Arquitectura del Motor de Reglas

#### 11.1.1 Diseño del Sistema de Reglas
```
┌─────────────────────────────────────────┐
│           Rules Engine Core             │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │    Rule     │ │      Rule           │ │
│  │  Registry   │ │    Executor         │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │    Rule     │ │    Performance      │ │
│  │   Cache     │ │    Optimizer        │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Result    │ │    Configuration    │ │
│  │ Aggregator  │ │     Manager         │ │
│  └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
```

#### 11.1.2 Componentes del Motor
- **Rule Registry**: Registro centralizado de todas las reglas
- **Rule Executor**: Motor de ejecución paralela de reglas
- **Rule Cache**: Cache inteligente para optimizar performance
- **Performance Optimizer**: Optimizador de ejecución de reglas
- **Result Aggregator**: Agregador y procesador de resultados
- **Configuration Manager**: Gestor de configuraciones por proyecto

### 11.2 Sistema de Definición de Reglas

#### 11.2.1 Rule Definition Framework
```rust
use serde::{Deserialize, Serialize};
use async_trait::async_trait;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    pub id: RuleId,
    pub name: String,
    pub description: String,
    pub category: RuleCategory,
    pub severity: RuleSeverity,
    pub languages: Vec<ProgrammingLanguage>,
    pub tags: Vec<String>,
    pub implementation: RuleImplementation,
    pub configuration: RuleConfiguration,
    pub metadata: RuleMetadata,
    pub version: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleCategory {
    // Code Quality
    BestPractices,
    CodeSmell,
    Maintainability,
    Readability,
    
    // Security
    Security,
    Vulnerability,
    CryptographicIssues,
    InputValidation,
    
    // Performance
    Performance,
    MemoryUsage,
    AlgorithmicComplexity,
    ResourceLeaks,
    
    // Reliability
    BugProne,
    ErrorHandling,
    NullPointer,
    ConcurrencyIssues,
    
    // Design
    DesignPatterns,
    Architecture,
    SOLID,
    DRY,
    
    // Language Specific
    PythonSpecific,
    JavaScriptSpecific,
    TypeScriptSpecific,
    RustSpecific,
    
    // Cross-Language
    CrossLanguage,
    Migration,
    Consistency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleImplementation {
    Pattern {
        ast_pattern: ASTPattern,
        conditions: Vec<RuleCondition>,
        actions: Vec<RuleAction>,
    },
    Query {
        unified_query: String,
        post_processors: Vec<QueryPostProcessor>,
    },
    Procedural {
        analyzer_function: String,
        parameters: HashMap<String, serde_json::Value>,
    },
    Composite {
        sub_rules: Vec<RuleId>,
        combination_logic: CombinationLogic,
    },
    MachineLearning {
        model_id: String,
        confidence_threshold: f64,
        feature_extractors: Vec<FeatureExtractor>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleConfiguration {
    pub parameters: HashMap<String, ConfigParameter>,
    pub thresholds: HashMap<String, ThresholdConfig>,
    pub exclusions: Vec<ExclusionPattern>,
    pub custom_settings: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigParameter {
    pub name: String,
    pub parameter_type: ParameterType,
    pub default_value: serde_json::Value,
    pub description: String,
    pub validation: Option<ParameterValidation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
    Enum(Vec<String>),
}
```

#### 11.2.2 AST Pattern Matching System
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTPattern {
    pub pattern_type: PatternType,
    pub node_selector: NodeSelector,
    pub constraints: Vec<PatternConstraint>,
    pub capture_groups: HashMap<String, CaptureGroup>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Exact,
    Fuzzy,
    Structural,
    Semantic,
    Behavioral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSelector {
    pub node_type: Option<UnifiedNodeType>,
    pub semantic_type: Option<SemanticNodeType>,
    pub name_pattern: Option<String>,
    pub value_pattern: Option<String>,
    pub attribute_filters: HashMap<String, AttributeFilter>,
    pub position_constraints: Option<PositionConstraint>,
    pub depth_range: Option<(u32, u32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternConstraint {
    HasChild {
        selector: NodeSelector,
        quantifier: Quantifier,
    },
    HasParent {
        selector: NodeSelector,
    },
    HasSibling {
        selector: NodeSelector,
        direction: SiblingDirection,
    },
    InScope {
        scope_type: ScopeType,
    },
    HasAttribute {
        name: String,
        value_pattern: Option<String>,
    },
    CustomPredicate {
        predicate_name: String,
        parameters: Vec<serde_json::Value>,
    },
    CrossLanguageEquivalent {
        languages: Vec<ProgrammingLanguage>,
        equivalence_type: EquivalenceType,
    },
}

// Example rule definition in YAML/JSON:
/*
rules:
  - id: "python-mutable-default-arg"
    name: "Mutable Default Arguments"
    description: "Avoid mutable default arguments in function definitions"
    category: "BugProne"
    severity: "High"
    languages: ["Python"]
    implementation:
      pattern:
        ast_pattern:
          node_selector:
            node_type: "FunctionDeclaration"
          constraints:
            - has_child:
                selector:
                  semantic_type: "Parameter"
                  attribute_filters:
                    has_default: true
                quantifier: "one_or_more"
        conditions:
          - type: "custom_predicate"
            predicate: "is_mutable_default_value"
        actions:
          - type: "report_violation"
            message: "Mutable default argument detected"
            suggestion: "Use None as default and check inside function"
*/
```

### 11.3 Rule Engine Core

#### 11.3.1 Rules Engine Implementation
```rust
pub struct RulesEngine {
    rule_registry: Arc<RuleRegistry>,
    rule_executor: Arc<RuleExecutor>,
    rule_cache: Arc<RuleCache>,
    performance_optimizer: Arc<PerformanceOptimizer>,
    result_aggregator: Arc<ResultAggregator>,
    config_manager: Arc<ConfigurationManager>,
    metrics_collector: Arc<RuleMetricsCollector>,
}

impl RulesEngine {
    pub async fn new(config: RulesEngineConfig) -> Result<Self, RulesEngineError> {
        let rule_registry = Arc::new(RuleRegistry::new());
        let rule_cache = Arc::new(RuleCache::new(config.cache_config.clone()));
        let performance_optimizer = Arc::new(PerformanceOptimizer::new());
        let result_aggregator = Arc::new(ResultAggregator::new());
        let config_manager = Arc::new(ConfigurationManager::new());
        let metrics_collector = Arc::new(RuleMetricsCollector::new());
        
        let rule_executor = Arc::new(RuleExecutor::new(
            rule_registry.clone(),
            rule_cache.clone(),
            performance_optimizer.clone(),
            config.executor_config.clone(),
        ));
        
        // Load default rules
        let engine = Self {
            rule_registry,
            rule_executor,
            rule_cache,
            performance_optimizer,
            result_aggregator,
            config_manager,
            metrics_collector,
        };
        
        engine.load_default_rules().await?;
        
        Ok(engine)
    }
    
    pub async fn analyze_code(&self, unified_ast: &UnifiedAST, project_config: &ProjectConfig) -> Result<AnalysisResult, RulesEngineError> {
        let start_time = Instant::now();
        
        // Get applicable rules for this language and project
        let applicable_rules = self.rule_registry.get_applicable_rules(
            &unified_ast.language,
            &project_config.enabled_categories,
            &project_config.severity_threshold,
        )?;
        
        self.metrics_collector.record_analysis_start(unified_ast, applicable_rules.len());
        
        // Optimize rule execution order
        let optimized_rules = self.performance_optimizer.optimize_execution_order(&applicable_rules, unified_ast).await?;
        
        // Execute rules
        let rule_results = self.rule_executor.execute_rules(&optimized_rules, unified_ast, project_config).await?;
        
        // Aggregate results
        let aggregated_results = self.result_aggregator.aggregate_results(rule_results, project_config).await?;
        
        let analysis_result = AnalysisResult {
            file_path: unified_ast.file_path.clone(),
            language: unified_ast.language,
            violations: aggregated_results.violations,
            metrics: aggregated_results.metrics,
            suggestions: aggregated_results.suggestions,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            rules_executed: optimized_rules.len(),
            cache_hits: self.rule_cache.get_hit_count(),
            performance_stats: aggregated_results.performance_stats,
        };
        
        self.metrics_collector.record_analysis_complete(&analysis_result);
        
        Ok(analysis_result)
    }
    
    pub async fn analyze_project(&self, project_path: &Path, project_config: &ProjectConfig) -> Result<ProjectAnalysisResult, RulesEngineError> {
        let mut project_results = Vec::new();
        let mut summary_metrics = ProjectMetrics::default();
        
        // Discover all files in project
        let files = self.discover_analyzable_files(project_path, project_config).await?;
        
        // Analyze files in parallel batches
        let batch_size = project_config.parallel_analysis_batch_size.unwrap_or(10);
        
        for batch in files.chunks(batch_size) {
            let batch_futures: Vec<_> = batch.iter()
                .map(|file_path| self.analyze_file(file_path, project_config))
                .collect();
            
            let batch_results = futures::future::try_join_all(batch_futures).await?;
            
            for result in batch_results {
                summary_metrics.aggregate(&result.metrics);
                project_results.push(result);
            }
        }
        
        Ok(ProjectAnalysisResult {
            project_path: project_path.to_path_buf(),
            file_results: project_results,
            summary_metrics,
            total_violations: summary_metrics.total_violations,
            critical_violations: summary_metrics.critical_violations,
            high_violations: summary_metrics.high_violations,
            quality_score: self.calculate_quality_score(&summary_metrics),
            recommendations: self.generate_project_recommendations(&summary_metrics),
        })
    }
    
    async fn analyze_file(&self, file_path: &Path, project_config: &ProjectConfig) -> Result<AnalysisResult, RulesEngineError> {
        // This would integrate with the unified parser system from previous phases
        let unified_ast = self.parse_file_to_unified_ast(file_path).await?;
        self.analyze_code(&unified_ast, project_config).await
    }
    
    async fn load_default_rules(&self) -> Result<(), RulesEngineError> {
        // Load built-in rules
        self.load_builtin_rules().await?;
        
        // Load community rules
        self.load_community_rules().await?;
        
        // Load custom rules if configured
        self.load_custom_rules().await?;
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RulesEngineConfig {
    pub cache_config: CacheConfig,
    pub executor_config: ExecutorConfig,
    pub parallel_execution: bool,
    pub max_concurrent_rules: usize,
    pub rule_timeout_ms: u64,
    pub enable_performance_optimization: bool,
    pub enable_rule_metrics: bool,
}

#[derive(Debug, Clone)]
pub struct ProjectConfig {
    pub enabled_categories: Vec<RuleCategory>,
    pub severity_threshold: RuleSeverity,
    pub custom_rule_configs: HashMap<RuleId, RuleConfiguration>,
    pub exclusion_patterns: Vec<String>,
    pub parallel_analysis_batch_size: Option<usize>,
    pub quality_gates: QualityGates,
}
```

### 11.4 Rule Executor System

#### 11.4.1 Parallel Rule Execution
```rust
pub struct RuleExecutor {
    rule_registry: Arc<RuleRegistry>,
    rule_cache: Arc<RuleCache>,
    performance_optimizer: Arc<PerformanceOptimizer>,
    thread_pool: Arc<ThreadPool>,
    config: ExecutorConfig,
}

#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub max_concurrent_rules: usize,
    pub rule_timeout_ms: u64,
    pub enable_early_termination: bool,
    pub batch_size: usize,
    pub memory_limit_mb: usize,
}

impl RuleExecutor {
    pub async fn execute_rules(&self, rules: &[Rule], ast: &UnifiedAST, config: &ProjectConfig) -> Result<Vec<RuleResult>, ExecutorError> {
        let mut results = Vec::new();
        
        // Group rules by execution strategy
        let (pattern_rules, query_rules, procedural_rules, ml_rules) = self.group_rules_by_type(rules);
        
        // Execute pattern rules (most common, optimized path)
        if !pattern_rules.is_empty() {
            let pattern_results = self.execute_pattern_rules(&pattern_rules, ast, config).await?;
            results.extend(pattern_results);
        }
        
        // Execute query rules
        if !query_rules.is_empty() {
            let query_results = self.execute_query_rules(&query_rules, ast, config).await?;
            results.extend(query_results);
        }
        
        // Execute procedural rules
        if !procedural_rules.is_empty() {
            let procedural_results = self.execute_procedural_rules(&procedural_rules, ast, config).await?;
            results.extend(procedural_results);
        }
        
        // Execute ML rules
        if !ml_rules.is_empty() {
            let ml_results = self.execute_ml_rules(&ml_rules, ast, config).await?;
            results.extend(ml_results);
        }
        
        Ok(results)
    }
    
    async fn execute_pattern_rules(&self, rules: &[Rule], ast: &UnifiedAST, config: &ProjectConfig) -> Result<Vec<RuleResult>, ExecutorError> {
        let mut results = Vec::new();
        
        // Create pattern matcher
        let pattern_matcher = PatternMatcher::new();
        
        // Execute rules in parallel batches
        for batch in rules.chunks(self.config.batch_size) {
            let batch_futures: Vec<_> = batch.iter()
                .map(|rule| self.execute_single_pattern_rule(rule, ast, &pattern_matcher, config))
                .collect();
            
            let batch_results = futures::future::try_join_all(batch_futures).await?;
            results.extend(batch_results.into_iter().flatten());
        }
        
        Ok(results)
    }
    
    async fn execute_single_pattern_rule(&self, rule: &Rule, ast: &UnifiedAST, pattern_matcher: &PatternMatcher, config: &ProjectConfig) -> Result<Vec<RuleResult>, ExecutorError> {
        let rule_start = Instant::now();
        
        // Check cache first
        let cache_key = self.generate_cache_key(rule, ast);
        if let Some(cached_result) = self.rule_cache.get(&cache_key).await {
            return Ok(cached_result);
        }
        
        // Execute rule with timeout
        let result = tokio::time::timeout(
            Duration::from_millis(self.config.rule_timeout_ms),
            self.execute_pattern_rule_impl(rule, ast, pattern_matcher, config)
        ).await;
        
        let rule_results = match result {
            Ok(Ok(results)) => results,
            Ok(Err(e)) => {
                return Err(ExecutorError::RuleExecutionFailed {
                    rule_id: rule.id.clone(),
                    error: e.to_string(),
                });
            }
            Err(_) => {
                return Err(ExecutorError::RuleTimeout {
                    rule_id: rule.id.clone(),
                    timeout_ms: self.config.rule_timeout_ms,
                });
            }
        };
        
        // Cache successful results
        self.rule_cache.set(cache_key, &rule_results).await;
        
        // Record metrics
        let execution_time = rule_start.elapsed();
        self.record_rule_metrics(rule, execution_time, rule_results.len());
        
        Ok(rule_results)
    }
    
    async fn execute_pattern_rule_impl(&self, rule: &Rule, ast: &UnifiedAST, pattern_matcher: &PatternMatcher, config: &ProjectConfig) -> Result<Vec<RuleResult>, ExecutorError> {
        let mut results = Vec::new();
        
        if let RuleImplementation::Pattern { ast_pattern, conditions, actions } = &rule.implementation {
            // Find all nodes matching the pattern
            let matches = pattern_matcher.find_matches(ast, ast_pattern).await?;
            
            for pattern_match in matches {
                // Check additional conditions
                let mut all_conditions_met = true;
                for condition in conditions {
                    if !self.evaluate_condition(condition, &pattern_match, ast, config).await? {
                        all_conditions_met = false;
                        break;
                    }
                }
                
                if all_conditions_met {
                    // Execute actions
                    for action in actions {
                        let action_result = self.execute_action(action, &pattern_match, rule, ast, config).await?;
                        if let Some(result) = action_result {
                            results.push(result);
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    async fn evaluate_condition(&self, condition: &RuleCondition, pattern_match: &PatternMatch, ast: &UnifiedAST, config: &ProjectConfig) -> Result<bool, ExecutorError> {
        match condition {
            RuleCondition::NodeCount { min, max } => {
                let count = pattern_match.matched_nodes.len();
                Ok(count >= min.unwrap_or(0) && count <= max.unwrap_or(usize::MAX))
            }
            RuleCondition::AttributeValue { attribute, expected_value } => {
                if let Some(node) = pattern_match.matched_nodes.first() {
                    if let Some(attr_value) = node.attributes.get(attribute) {
                        Ok(attr_value.to_string() == *expected_value)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            RuleCondition::CustomPredicate { predicate_name, parameters } => {
                self.evaluate_custom_predicate(predicate_name, parameters, pattern_match, ast, config).await
            }
            RuleCondition::CrossLanguageCheck { target_languages, equivalence_check } => {
                self.evaluate_cross_language_condition(target_languages, equivalence_check, pattern_match, ast).await
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuleResult {
    pub rule_id: RuleId,
    pub violation: Option<Violation>,
    pub suggestions: Vec<Suggestion>,
    pub metrics: RuleExecutionMetrics,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct Violation {
    pub severity: RuleSeverity,
    pub message: String,
    pub location: ViolationLocation,
    pub rule_category: RuleCategory,
    pub confidence: f64,
    pub fix_suggestions: Vec<FixSuggestion>,
}

#[derive(Debug, Clone)]
pub struct ViolationLocation {
    pub file_path: PathBuf,
    pub start_line: u32,
    pub start_column: u32,
    pub end_line: u32,
    pub end_column: u32,
    pub context_lines: Vec<String>,
}
```

### 11.5 Built-in Rules Library

#### 11.5.1 Core Rules Categories
```rust
pub struct BuiltinRulesLibrary {
    categories: HashMap<RuleCategory, Vec<Rule>>,
}

impl BuiltinRulesLibrary {
    pub fn new() -> Self {
        let mut library = Self {
            categories: HashMap::new(),
        };
        
        library.load_best_practices_rules();
        library.load_security_rules();
        library.load_performance_rules();
        library.load_maintainability_rules();
        library.load_bug_prone_rules();
        library.load_language_specific_rules();
        library.load_cross_language_rules();
        
        library
    }
    
    fn load_best_practices_rules(&mut self) {
        let mut rules = Vec::new();
        
        // Example: Function naming convention
        rules.push(Rule {
            id: "naming-function-convention".into(),
            name: "Function Naming Convention".into(),
            description: "Functions should follow language-specific naming conventions".into(),
            category: RuleCategory::BestPractices,
            severity: RuleSeverity::Medium,
            languages: vec![ProgrammingLanguage::Python, ProgrammingLanguage::JavaScript, ProgrammingLanguage::TypeScript],
            implementation: RuleImplementation::Pattern {
                ast_pattern: ASTPattern {
                    pattern_type: PatternType::Structural,
                    node_selector: NodeSelector {
                        node_type: Some(UnifiedNodeType::FunctionDeclaration { 
                            is_async: false, 
                            is_generator: false, 
                            visibility: Visibility::Public 
                        }),
                        semantic_type: Some(SemanticNodeType::Declaration),
                        name_pattern: Some("^[A-Z]".to_string()), // Matches PascalCase (violation for most languages)
                        ..Default::default()
                    },
                    constraints: vec![],
                    capture_groups: HashMap::new(),
                },
                conditions: vec![
                    RuleCondition::CustomPredicate {
                        predicate_name: "violates_naming_convention".into(),
                        parameters: vec![],
                    }
                ],
                actions: vec![
                    RuleAction::ReportViolation {
                        message: "Function name should follow {language} naming convention".into(),
                        suggestion: Some("Use snake_case for Python, camelCase for JavaScript/TypeScript".into()),
                    }
                ],
            },
            configuration: RuleConfiguration::default(),
            metadata: RuleMetadata::default(),
            version: "1.0.0".into(),
            enabled: true,
        });
        
        // Example: Magic numbers
        rules.push(Rule {
            id: "avoid-magic-numbers".into(),
            name: "Avoid Magic Numbers".into(),
            description: "Replace magic numbers with named constants".into(),
            category: RuleCategory::BestPractices,
            severity: RuleSeverity::Medium,
            languages: vec![
                ProgrammingLanguage::Python,
                ProgrammingLanguage::JavaScript,
                ProgrammingLanguage::TypeScript,
                ProgrammingLanguage::Rust,
            ],
            implementation: RuleImplementation::Pattern {
                ast_pattern: ASTPattern {
                    pattern_type: PatternType::Semantic,
                    node_selector: NodeSelector {
                        node_type: Some(UnifiedNodeType::NumberLiteral),
                        ..Default::default()
                    },
                    constraints: vec![
                        PatternConstraint::CustomPredicate {
                            predicate_name: "is_magic_number".into(),
                            parameters: vec![],
                        }
                    ],
                    capture_groups: HashMap::new(),
                },
                conditions: vec![],
                actions: vec![
                    RuleAction::ReportViolation {
                        message: "Magic number detected. Consider using a named constant.".into(),
                        suggestion: Some("Replace with a descriptive constant name".into()),
                    }
                ],
            },
            configuration: RuleConfiguration {
                parameters: HashMap::from([
                    ("allowed_numbers".into(), ConfigParameter {
                        name: "allowed_numbers".into(),
                        parameter_type: ParameterType::Array,
                        default_value: serde_json::json!([0, 1, -1, 2]),
                        description: "Numbers that are not considered magic".into(),
                        validation: None,
                    })
                ]),
                ..Default::default()
            },
            metadata: RuleMetadata::default(),
            version: "1.0.0".into(),
            enabled: true,
        });
        
        self.categories.insert(RuleCategory::BestPractices, rules);
    }
    
    fn load_security_rules(&mut self) {
        let mut rules = Vec::new();
        
        // Example: SQL Injection detection
        rules.push(Rule {
            id: "sql-injection-risk".into(),
            name: "SQL Injection Risk".into(),
            description: "Detect potential SQL injection vulnerabilities".into(),
            category: RuleCategory::Security,
            severity: RuleSeverity::Critical,
            languages: vec![
                ProgrammingLanguage::Python,
                ProgrammingLanguage::JavaScript,
                ProgrammingLanguage::TypeScript,
            ],
            implementation: RuleImplementation::Pattern {
                ast_pattern: ASTPattern {
                    pattern_type: PatternType::Behavioral,
                    node_selector: NodeSelector {
                        node_type: Some(UnifiedNodeType::CallExpression),
                        ..Default::default()
                    },
                    constraints: vec![
                        PatternConstraint::CustomPredicate {
                            predicate_name: "is_sql_execution".into(),
                            parameters: vec![],
                        },
                        PatternConstraint::CustomPredicate {
                            predicate_name: "uses_string_concatenation".into(),
                            parameters: vec![],
                        }
                    ],
                    capture_groups: HashMap::new(),
                },
                conditions: vec![],
                actions: vec![
                    RuleAction::ReportViolation {
                        message: "Potential SQL injection vulnerability detected".into(),
                        suggestion: Some("Use parameterized queries or prepared statements".into()),
                    }
                ],
            },
            configuration: RuleConfiguration::default(),
            metadata: RuleMetadata {
                cwe_ids: vec!["CWE-89".into()],
                owasp_categories: vec!["A03:2021-Injection".into()],
                references: vec![
                    "https://owasp.org/www-community/attacks/SQL_Injection".into()
                ],
                ..Default::default()
            },
            version: "1.0.0".into(),
            enabled: true,
        });
        
        // Example: Hardcoded secrets
        rules.push(Rule {
            id: "hardcoded-secrets".into(),
            name: "Hardcoded Secrets".into(),
            description: "Detect hardcoded passwords, API keys, and other secrets".into(),
            category: RuleCategory::Security,
            severity: RuleSeverity::High,
            languages: vec![
                ProgrammingLanguage::Python,
                ProgrammingLanguage::JavaScript,
                ProgrammingLanguage::TypeScript,
                ProgrammingLanguage::Rust,
            ],
            implementation: RuleImplementation::Pattern {
                ast_pattern: ASTPattern {
                    pattern_type: PatternType::Semantic,
                    node_selector: NodeSelector {
                        node_type: Some(UnifiedNodeType::StringLiteral),
                        ..Default::default()
                    },
                    constraints: vec![
                        PatternConstraint::CustomPredicate {
                            predicate_name: "looks_like_secret".into(),
                            parameters: vec![],
                        }
                    ],
                    capture_groups: HashMap::new(),
                },
                conditions: vec![],
                actions: vec![
                    RuleAction::ReportViolation {
                        message: "Potential hardcoded secret detected".into(),
                        suggestion: Some("Move secrets to environment variables or secure configuration".into()),
                    }
                ],
            },
            configuration: RuleConfiguration {
                parameters: HashMap::from([
                    ("secret_patterns".into(), ConfigParameter {
                        name: "secret_patterns".into(),
                        parameter_type: ParameterType::Array,
                        default_value: serde_json::json!([
                            "password\\s*=\\s*['\"]\\w+['\"]",
                            "api_?key\\s*=\\s*['\"]\\w+['\"]",
                            "secret\\s*=\\s*['\"]\\w+['\"]",
                            "token\\s*=\\s*['\"]\\w+['\"]"
                        ]),
                        description: "Regex patterns for detecting secrets".into(),
                        validation: None,
                    })
                ]),
                ..Default::default()
            },
            metadata: RuleMetadata {
                cwe_ids: vec!["CWE-798".into()],
                owasp_categories: vec!["A07:2021-Identification and Authentication Failures".into()],
                ..Default::default()
            },
            version: "1.0.0".into(),
            enabled: true,
        });
        
        self.categories.insert(RuleCategory::Security, rules);
    }
    
    fn load_cross_language_rules(&mut self) {
        let mut rules = Vec::new();
        
        // Example: Inconsistent error handling patterns
        rules.push(Rule {
            id: "consistent-error-handling".into(),
            name: "Consistent Error Handling".into(),
            description: "Ensure consistent error handling patterns across languages in a project".into(),
            category: RuleCategory::CrossLanguage,
            severity: RuleSeverity::Medium,
            languages: vec![
                ProgrammingLanguage::Python,
                ProgrammingLanguage::JavaScript,
                ProgrammingLanguage::TypeScript,
                ProgrammingLanguage::Rust,
            ],
            implementation: RuleImplementation::Composite {
                sub_rules: vec![
                    "python-error-handling".into(),
                    "js-error-handling".into(),
                    "rust-error-handling".into(),
                ],
                combination_logic: CombinationLogic::ConsistencyCheck,
            },
            configuration: RuleConfiguration::default(),
            metadata: RuleMetadata::default(),
            version: "1.0.0".into(),
            enabled: true,
        });
        
        self.categories.insert(RuleCategory::CrossLanguage, rules);
    }
}

#[derive(Debug, Clone)]
pub struct RuleMetadata {
    pub author: String,
    pub created_date: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
    pub cwe_ids: Vec<String>,
    pub owasp_categories: Vec<String>,
    pub references: Vec<String>,
    pub tags: Vec<String>,
    pub difficulty_to_fix: DifficultyLevel,
    pub false_positive_rate: f64,
}

#[derive(Debug, Clone)]
pub enum DifficultyLevel {
    Trivial,
    Easy,
    Medium,
    Hard,
    Expert,
}
```

### 11.6 Performance Optimization System

#### 11.6.1 Rule Execution Optimizer
```rust
pub struct PerformanceOptimizer {
    execution_history: Arc<RwLock<HashMap<RuleId, ExecutionStats>>>,
    dependency_graph: Arc<RwLock<RuleDependencyGraph>>,
    resource_monitor: ResourceMonitor,
}

#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub average_execution_time: Duration,
    pub success_rate: f64,
    pub memory_usage: usize,
    pub cache_hit_rate: f64,
    pub last_executed: DateTime<Utc>,
    pub execution_count: u64,
}

impl PerformanceOptimizer {
    pub async fn optimize_execution_order(&self, rules: &[Rule], ast: &UnifiedAST) -> Result<Vec<Rule>, OptimizerError> {
        // Get execution statistics
        let stats = self.execution_history.read().await;
        
        // Create execution plan
        let mut execution_plan = Vec::new();
        
        // Group rules by estimated execution time
        let (fast_rules, medium_rules, slow_rules) = self.categorize_rules_by_speed(rules, &stats);
        
        // Optimize based on different strategies
        match self.determine_optimization_strategy(ast, rules.len()) {
            OptimizationStrategy::FastFirst => {
                execution_plan.extend(fast_rules);
                execution_plan.extend(medium_rules);
                execution_plan.extend(slow_rules);
            }
            OptimizationStrategy::HighImpactFirst => {
                let ordered = self.order_by_impact_and_speed(rules, &stats);
                execution_plan.extend(ordered);
            }
            OptimizationStrategy::DependencyBased => {
                let dependency_graph = self.dependency_graph.read().await;
                let ordered = self.topological_sort(rules, &dependency_graph)?;
                execution_plan.extend(ordered);
            }
            OptimizationStrategy::ResourceAware => {
                let ordered = self.optimize_for_resource_usage(rules, &stats, ast).await?;
                execution_plan.extend(ordered);
            }
        }
        
        Ok(execution_plan)
    }
    
    fn categorize_rules_by_speed(&self, rules: &[Rule], stats: &HashMap<RuleId, ExecutionStats>) -> (Vec<Rule>, Vec<Rule>, Vec<Rule>) {
        let mut fast_rules = Vec::new();
        let mut medium_rules = Vec::new();
        let mut slow_rules = Vec::new();
        
        for rule in rules {
            let avg_time = stats.get(&rule.id)
                .map(|s| s.average_execution_time)
                .unwrap_or(Duration::from_millis(100)); // Default estimate
            
            if avg_time < Duration::from_millis(10) {
                fast_rules.push(rule.clone());
            } else if avg_time < Duration::from_millis(100) {
                medium_rules.push(rule.clone());
            } else {
                slow_rules.push(rule.clone());
            }
        }
        
        (fast_rules, medium_rules, slow_rules)
    }
    
    fn determine_optimization_strategy(&self, ast: &UnifiedAST, rule_count: usize) -> OptimizationStrategy {
        // Heuristics for choosing optimization strategy
        if rule_count < 100 {
            OptimizationStrategy::FastFirst
        } else if ast.metadata.node_count < 1000 {
            OptimizationStrategy::HighImpactFirst
        } else if rule_count > 1000 {
            OptimizationStrategy::DependencyBased
        } else {
            OptimizationStrategy::ResourceAware
        }
    }
    
    pub async fn should_cache_result(&self, rule: &Rule, execution_time: Duration, result_size: usize) -> bool {
        // Cache if execution time is significant and result is not too large
        execution_time > Duration::from_millis(50) && result_size < 1024 * 1024 // 1MB
    }
    
    pub async fn predict_execution_time(&self, rules: &[Rule], ast: &UnifiedAST) -> Duration {
        let stats = self.execution_history.read().await;
        
        let mut total_time = Duration::from_millis(0);
        
        for rule in rules {
            let estimated_time = if let Some(stat) = stats.get(&rule.id) {
                // Adjust based on AST size
                let size_factor = (ast.metadata.node_count as f64 / 1000.0).max(1.0);
                Duration::from_nanos((stat.average_execution_time.as_nanos() as f64 * size_factor) as u64)
            } else {
                // Default estimate based on rule type
                self.estimate_rule_execution_time(rule)
            };
            
            total_time += estimated_time;
        }
        
        total_time
    }
}

#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    FastFirst,
    HighImpactFirst,
    DependencyBased,
    ResourceAware,
}
```

### 11.7 Rule Configuration System

#### 11.7.1 Dynamic Configuration Manager
```rust
pub struct ConfigurationManager {
    global_config: Arc<RwLock<GlobalRuleConfig>>,
    project_configs: Arc<RwLock<HashMap<PathBuf, ProjectRuleConfig>>>,
    rule_customizations: Arc<RwLock<HashMap<RuleId, RuleCustomization>>>,
    config_watchers: Vec<ConfigWatcher>,
}

#[derive(Debug, Clone)]
pub struct GlobalRuleConfig {
    pub default_severity_threshold: RuleSeverity,
    pub enabled_categories: Vec<RuleCategory>,
    pub disabled_rules: Vec<RuleId>,
    pub performance_settings: PerformanceSettings,
    pub output_format: OutputFormat,
    pub enable_auto_fix: bool,
}

#[derive(Debug, Clone)]
pub struct ProjectRuleConfig {
    pub project_path: PathBuf,
    pub rule_overrides: HashMap<RuleId, RuleOverride>,
    pub custom_thresholds: HashMap<String, f64>,
    pub exclusion_patterns: Vec<glob::Pattern>,
    pub language_specific_configs: HashMap<ProgrammingLanguage, LanguageConfig>,
    pub quality_gates: QualityGates,
}

#[derive(Debug, Clone)]
pub struct RuleOverride {
    pub enabled: Option<bool>,
    pub severity: Option<RuleSeverity>,
    pub custom_parameters: HashMap<String, serde_json::Value>,
    pub custom_message: Option<String>,
}

impl ConfigurationManager {
    pub async fn load_project_config(&self, project_path: &Path) -> Result<ProjectRuleConfig, ConfigError> {
        // Look for configuration files in order of preference
        let config_files = [
            project_path.join(".codeant.toml"),
            project_path.join(".codeant.yaml"),
            project_path.join(".codeant.json"),
            project_path.join("codeant.config.toml"),
        ];
        
        for config_file in &config_files {
            if config_file.exists() {
                return self.parse_config_file(config_file).await;
            }
        }
        
        // Use default configuration
        Ok(ProjectRuleConfig::default_for_project(project_path))
    }
    
    pub async fn get_effective_rule_config(&self, rule: &Rule, project_path: &Path) -> Result<EffectiveRuleConfig, ConfigError> {
        let global_config = self.global_config.read().await;
        let project_config = self.get_or_load_project_config(project_path).await?;
        
        let mut effective_config = EffectiveRuleConfig {
            rule_id: rule.id.clone(),
            enabled: rule.enabled,
            severity: rule.severity,
            parameters: rule.configuration.parameters.clone(),
            thresholds: rule.configuration.thresholds.clone(),
        };
        
        // Apply global overrides
        if global_config.disabled_rules.contains(&rule.id) {
            effective_config.enabled = false;
        }
        
        // Apply project-specific overrides
        if let Some(override_config) = project_config.rule_overrides.get(&rule.id) {
            if let Some(enabled) = override_config.enabled {
                effective_config.enabled = enabled;
            }
            if let Some(severity) = override_config.severity {
                effective_config.severity = severity;
            }
            
            // Merge custom parameters
            for (key, value) in &override_config.custom_parameters {
                effective_config.parameters.insert(key.clone(), ConfigParameter {
                    name: key.clone(),
                    parameter_type: ParameterType::from_json_value(value),
                    default_value: value.clone(),
                    description: "Custom override".to_string(),
                    validation: None,
                });
            }
        }
        
        Ok(effective_config)
    }
    
    async fn parse_config_file(&self, config_file: &Path) -> Result<ProjectRuleConfig, ConfigError> {
        let content = tokio::fs::read_to_string(config_file).await?;
        
        match config_file.extension().and_then(|ext| ext.to_str()) {
            Some("toml") => {
                let config: ProjectRuleConfigRaw = toml::from_str(&content)?;
                Ok(config.into())
            }
            Some("yaml") | Some("yml") => {
                let config: ProjectRuleConfigRaw = serde_yaml::from_str(&content)?;
                Ok(config.into())
            }
            Some("json") => {
                let config: ProjectRuleConfigRaw = serde_json::from_str(&content)?;
                Ok(config.into())
            }
            _ => Err(ConfigError::UnsupportedFormat),
        }
    }
}

// Example configuration file format (.codeant.toml):
/*
[rules]
enabled_categories = ["BestPractices", "Security", "Performance"]
severity_threshold = "Medium"

[rules.overrides]
"python-mutable-default-arg" = { enabled = true, severity = "High" }
"avoid-magic-numbers" = { enabled = true, parameters = { allowed_numbers = [0, 1, -1, 2, 10, 100] } }

[quality_gates]
max_critical_violations = 0
max_high_violations = 5
max_medium_violations = 20
min_quality_score = 80.0

[languages.python]
naming_convention = "snake_case"
max_line_length = 88

[languages.javascript]
naming_convention = "camelCase"
max_line_length = 100

[exclusions]
patterns = ["**/node_modules/**", "**/dist/**", "**/*.min.js"]
*/
```

### 11.8 Criterios de Completitud

#### 11.8.1 Entregables de la Fase
- [ ] Motor de reglas estáticas implementado
- [ ] Biblioteca de 30,000+ reglas integradas
- [ ] Sistema de ejecución paralela optimizado
- [ ] Cache inteligente de resultados
- [ ] Sistema de configuración flexible
- [ ] Optimizador de performance
- [ ] Agregador de resultados
- [ ] Sistema de métricas de ejecución
- [ ] API de reglas personalizadas
- [ ] Tests comprehensivos del motor

#### 11.8.2 Criterios de Aceptación
- [ ] Ejecuta 30,000+ reglas en tiempo razonable
- [ ] Performance escalable con tamaño de código
- [ ] Sistema de cache mejora performance >50%
- [ ] Configuración flexible por proyecto
- [ ] Reglas cross-language funcionan correctamente
- [ ] Paralelización eficiente de ejecución
- [ ] Resultados precisos y consistentes
- [ ] Memory usage controlado durante ejecución
- [ ] Integration seamless con AST unificado
- [ ] API permite reglas personalizadas

### 11.9 Performance Targets

#### 11.9.1 Benchmarks del Motor de Reglas
- **Rule execution**: <5ms promedio por regla simple
- **Parallel execution**: >80% utilización de cores
- **Cache hit rate**: >90% para archivos similares
- **Memory usage**: <2GB para análisis de proyectos grandes
- **Throughput**: >10,000 reglas/segundo en hardware típico

### 11.10 Estimación de Tiempo

#### 11.10.1 Breakdown de Tareas
- Diseño de arquitectura del motor: 4 días
- Implementación del core engine: 8 días
- Sistema de definición de reglas: 6 días
- Rule executor con paralelización: 8 días
- Sistema de cache inteligente: 5 días
- Performance optimizer: 7 días
- Configuration manager: 6 días
- Biblioteca de reglas built-in: 15 días
- Result aggregator: 4 días
- Integration con AST unificado: 5 días
- Testing comprehensivo: 8 días
- Performance optimization: 6 días
- Documentación y API: 4 días

**Total estimado: 86 días de desarrollo**

### 11.11 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Motor de reglas estáticas de clase enterprise
- Capacidad de ejecutar decenas de miles de reglas
- Performance optimizada y escalable
- Base sólida para análisis de calidad de código
- Foundation para las siguientes fases de detección

La Fase 12 construirá sobre este motor implementando la detección específica de código muerto básico.
