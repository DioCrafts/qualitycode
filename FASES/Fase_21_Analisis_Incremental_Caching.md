# Fase 21: Sistema de Análisis Incremental y Caching Inteligente

## Objetivo General
Implementar un sistema avanzado de análisis incremental y caching inteligente que optimice dramáticamente el rendimiento del agente CodeAnt, permitiendo análisis en tiempo real de cambios de código, cache multi-nivel inteligente, invalidación selectiva, y capacidades de análisis diferencial que escalen eficientemente para repositorios masivos y equipos de desarrollo grandes.

## Descripción Técnica Detallada

### 21.1 Arquitectura del Sistema Incremental

#### 21.1.1 Diseño del Incremental Analysis System
```
┌─────────────────────────────────────────┐
│       Incremental Analysis System      │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Change    │ │    Dependency       │ │
│  │  Detector   │ │    Tracker          │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ Incremental │ │   Smart Cache       │ │
│  │  Analyzer   │ │    Manager          │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ Delta Cache │ │   Invalidation      │ │
│  │  Engine     │ │    Engine           │ │
│  └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
```

#### 21.1.2 Componentes del Sistema
- **Change Detector**: Detección granular de cambios
- **Dependency Tracker**: Seguimiento de dependencias entre archivos
- **Incremental Analyzer**: Análisis solo de cambios y dependencias
- **Smart Cache Manager**: Gestión inteligente de cache multi-nivel
- **Delta Cache Engine**: Cache de deltas y análisis incrementales
- **Invalidation Engine**: Invalidación selectiva e inteligente

### 21.2 Change Detection System

#### 21.2.1 Granular Change Detector
```rust
use std::collections::{HashMap, HashSet, BTreeMap};
use git2::{Repository, Diff, DiffOptions, DiffHunk, DiffLine};
use blake3::Hasher;

pub struct GranularChangeDetector {
    git_analyzer: Arc<GitAnalyzer>,
    file_watcher: Arc<FileWatcher>,
    ast_differ: Arc<ASTDiffer>,
    semantic_differ: Arc<SemanticDiffer>,
    dependency_tracker: Arc<DependencyTracker>,
    config: ChangeDetectionConfig,
}

#[derive(Debug, Clone)]
pub struct ChangeDetectionConfig {
    pub enable_git_integration: bool,
    pub enable_file_watching: bool,
    pub enable_ast_diffing: bool,
    pub enable_semantic_diffing: bool,
    pub granularity_level: GranularityLevel,
    pub change_aggregation_window_ms: u64,
    pub ignore_whitespace_changes: bool,
    pub ignore_comment_changes: bool,
    pub track_dependencies: bool,
}

#[derive(Debug, Clone)]
pub enum GranularityLevel {
    File,        // Track changes at file level
    Function,    // Track changes at function level
    Statement,   // Track changes at statement level
    Expression,  // Track changes at expression level
    Token,       // Track changes at token level
}

impl GranularChangeDetector {
    pub async fn new(config: ChangeDetectionConfig) -> Result<Self, ChangeDetectionError> {
        Ok(Self {
            git_analyzer: Arc::new(GitAnalyzer::new()),
            file_watcher: Arc::new(FileWatcher::new()),
            ast_differ: Arc::new(ASTDiffer::new()),
            semantic_differ: Arc::new(SemanticDiffer::new()),
            dependency_tracker: Arc::new(DependencyTracker::new()),
            config,
        })
    }
    
    pub async fn detect_changes(&self, repository_path: &Path, from_commit: Option<&str>, to_commit: Option<&str>) -> Result<ChangeSet, ChangeDetectionError> {
        let start_time = Instant::now();
        
        // Detect file-level changes using Git
        let file_changes = if self.config.enable_git_integration {
            self.git_analyzer.analyze_git_changes(repository_path, from_commit, to_commit).await?
        } else {
            Vec::new()
        };
        
        // Detect granular changes within modified files
        let mut granular_changes = Vec::new();
        
        for file_change in &file_changes {
            if file_change.change_type != FileChangeType::Deleted {
                let file_granular_changes = self.detect_file_granular_changes(file_change).await?;
                granular_changes.extend(file_granular_changes);
            }
        }
        
        // Analyze dependency impact
        let dependency_impact = if self.config.track_dependencies {
            self.dependency_tracker.analyze_dependency_impact(&granular_changes).await?
        } else {
            DependencyImpact::default()
        };
        
        // Calculate affected analysis scope
        let analysis_scope = self.calculate_affected_scope(&granular_changes, &dependency_impact).await?;
        
        Ok(ChangeSet {
            id: ChangeSetId::new(),
            file_changes,
            granular_changes,
            dependency_impact,
            analysis_scope,
            change_statistics: self.calculate_change_statistics(&file_changes, &granular_changes),
            detection_time_ms: start_time.elapsed().as_millis() as u64,
            timestamp: Utc::now(),
        })
    }
    
    async fn detect_file_granular_changes(&self, file_change: &FileChange) -> Result<Vec<GranularChange>, ChangeDetectionError> {
        let mut granular_changes = Vec::new();
        
        match self.config.granularity_level {
            GranularityLevel::File => {
                // File-level granularity (already have this)
                granular_changes.push(GranularChange {
                    id: GranularChangeId::new(),
                    change_type: ChangeType::FileModified,
                    location: ChangeLocation::File(file_change.file_path.clone()),
                    content_hash: self.calculate_file_hash(&file_change.file_path)?,
                    affected_symbols: Vec::new(),
                    impact_score: 1.0,
                });
            }
            GranularityLevel::Function => {
                granular_changes.extend(self.detect_function_level_changes(file_change).await?);
            }
            GranularityLevel::Statement => {
                granular_changes.extend(self.detect_statement_level_changes(file_change).await?);
            }
            GranularityLevel::Expression => {
                granular_changes.extend(self.detect_expression_level_changes(file_change).await?);
            }
            GranularityLevel::Token => {
                granular_changes.extend(self.detect_token_level_changes(file_change).await?);
            }
        }
        
        Ok(granular_changes)
    }
    
    async fn detect_function_level_changes(&self, file_change: &FileChange) -> Result<Vec<GranularChange>, ChangeDetectionError> {
        let mut function_changes = Vec::new();
        
        // Parse old and new versions
        let old_ast = if let Some(old_content) = &file_change.old_content {
            Some(self.parse_to_unified_ast(old_content, &file_change.file_path).await?)
        } else {
            None
        };
        
        let new_ast = if let Some(new_content) = &file_change.new_content {
            Some(self.parse_to_unified_ast(new_content, &file_change.file_path).await?)
        } else {
            None
        };
        
        // Compare functions
        let function_diff = self.ast_differ.compare_functions(old_ast.as_ref(), new_ast.as_ref()).await?;
        
        for func_diff in function_diff.function_changes {
            let granular_change = GranularChange {
                id: GranularChangeId::new(),
                change_type: match func_diff.change_type {
                    FunctionChangeType::Added => ChangeType::FunctionAdded,
                    FunctionChangeType::Removed => ChangeType::FunctionRemoved,
                    FunctionChangeType::Modified => ChangeType::FunctionModified,
                    FunctionChangeType::Moved => ChangeType::FunctionMoved,
                    FunctionChangeType::Renamed => ChangeType::FunctionRenamed,
                },
                location: ChangeLocation::Function {
                    file_path: file_change.file_path.clone(),
                    function_name: func_diff.function_name.clone(),
                    line_range: func_diff.line_range,
                },
                content_hash: func_diff.content_hash,
                affected_symbols: func_diff.affected_symbols,
                impact_score: self.calculate_function_impact_score(&func_diff),
            };
            
            function_changes.push(granular_change);
        }
        
        Ok(function_changes)
    }
    
    async fn detect_semantic_changes(&self, file_change: &FileChange) -> Result<Vec<SemanticChange>, ChangeDetectionError> {
        let mut semantic_changes = Vec::new();
        
        if self.config.enable_semantic_diffing {
            let semantic_diff = self.semantic_differ.analyze_semantic_changes(file_change).await?;
            semantic_changes.extend(semantic_diff.changes);
        }
        
        Ok(semantic_changes)
    }
}

#[derive(Debug, Clone)]
pub struct ChangeSet {
    pub id: ChangeSetId,
    pub file_changes: Vec<FileChange>,
    pub granular_changes: Vec<GranularChange>,
    pub dependency_impact: DependencyImpact,
    pub analysis_scope: AnalysisScope,
    pub change_statistics: ChangeStatistics,
    pub detection_time_ms: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct GranularChange {
    pub id: GranularChangeId,
    pub change_type: ChangeType,
    pub location: ChangeLocation,
    pub content_hash: String,
    pub affected_symbols: Vec<String>,
    pub impact_score: f64,
}

#[derive(Debug, Clone)]
pub enum ChangeType {
    FileAdded,
    FileRemoved,
    FileModified,
    FileMoved,
    FileRenamed,
    FunctionAdded,
    FunctionRemoved,
    FunctionModified,
    FunctionMoved,
    FunctionRenamed,
    ClassAdded,
    ClassRemoved,
    ClassModified,
    StatementAdded,
    StatementRemoved,
    StatementModified,
    ExpressionModified,
    TokenModified,
}

#[derive(Debug, Clone)]
pub enum ChangeLocation {
    File(PathBuf),
    Function {
        file_path: PathBuf,
        function_name: String,
        line_range: (u32, u32),
    },
    Class {
        file_path: PathBuf,
        class_name: String,
        line_range: (u32, u32),
    },
    Statement {
        file_path: PathBuf,
        line_number: u32,
        column_range: (u32, u32),
    },
    Expression {
        file_path: PathBuf,
        line_number: u32,
        column_range: (u32, u32),
        expression_type: String,
    },
}

#[derive(Debug, Clone)]
pub struct DependencyImpact {
    pub directly_affected_files: Vec<PathBuf>,
    pub transitively_affected_files: Vec<PathBuf>,
    pub affected_symbols: HashMap<String, Vec<PathBuf>>,
    pub impact_radius: u32,
    pub estimated_analysis_scope: f64, // Percentage of project that needs re-analysis
}

#[derive(Debug, Clone)]
pub struct AnalysisScope {
    pub files_to_analyze: Vec<PathBuf>,
    pub functions_to_analyze: Vec<FunctionIdentifier>,
    pub classes_to_analyze: Vec<ClassIdentifier>,
    pub rules_to_execute: Vec<RuleId>,
    pub scope_justification: Vec<ScopeReason>,
    pub estimated_analysis_time: Duration,
}

#[derive(Debug, Clone)]
pub enum ScopeReason {
    DirectChange,
    DependencyChange,
    SemanticChange,
    CrossFileImpact,
    RuleSpecificRequirement,
}
```

### 21.3 Smart Cache Management System

#### 21.3.1 Multi-Level Cache Manager
```rust
use lru::LruCache;
use std::num::NonZeroUsize;

pub struct SmartCacheManager {
    l1_cache: Arc<RwLock<L1Cache>>, // Hot data in memory
    l2_cache: Arc<L2Cache>,         // Warm data in Redis
    l3_cache: Arc<L3Cache>,         // Cold data in disk/object storage
    cache_coordinator: Arc<CacheCoordinator>,
    invalidation_engine: Arc<InvalidationEngine>,
    cache_analytics: Arc<CacheAnalytics>,
    config: CacheConfig,
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub l1_cache_size_mb: usize,
    pub l2_cache_size_mb: usize,
    pub l3_cache_size_gb: usize,
    pub enable_compression: bool,
    pub enable_encryption: bool,
    pub ttl_policies: HashMap<CacheType, Duration>,
    pub eviction_policies: HashMap<CacheLevel, EvictionPolicy>,
    pub enable_predictive_loading: bool,
    pub enable_cache_warming: bool,
    pub cache_hit_ratio_target: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheType {
    ParsedAST,
    AnalysisResult,
    AIEmbedding,
    RuleResult,
    MetricsResult,
    DependencyGraph,
    SemanticAnalysis,
    AntipatternDetection,
}

#[derive(Debug, Clone)]
pub enum CacheLevel {
    L1, // Memory
    L2, // Redis
    L3, // Disk/Object Storage
}

impl SmartCacheManager {
    pub async fn new(config: CacheConfig) -> Result<Self, CacheError> {
        let l1_cache = Arc::new(RwLock::new(L1Cache::new(config.l1_cache_size_mb)));
        let l2_cache = Arc::new(L2Cache::new(config.l2_cache_size_mb).await?);
        let l3_cache = Arc::new(L3Cache::new(config.l3_cache_size_gb).await?);
        
        Ok(Self {
            l1_cache,
            l2_cache,
            l3_cache,
            cache_coordinator: Arc::new(CacheCoordinator::new()),
            invalidation_engine: Arc::new(InvalidationEngine::new()),
            cache_analytics: Arc::new(CacheAnalytics::new()),
            config,
        })
    }
    
    pub async fn get<T>(&self, key: &CacheKey) -> Result<Option<T>, CacheError>
    where
        T: CacheableItem + Clone + Send + Sync,
    {
        self.cache_analytics.record_access(key);
        
        // Try L1 cache first
        if let Some(item) = self.get_from_l1(key).await? {
            self.cache_analytics.record_hit(CacheLevel::L1);
            return Ok(Some(item));
        }
        
        // Try L2 cache
        if let Some(item) = self.l2_cache.get(key).await? {
            self.cache_analytics.record_hit(CacheLevel::L2);
            
            // Promote to L1 if frequently accessed
            if self.should_promote_to_l1(key).await? {
                self.set_l1(key.clone(), &item).await?;
            }
            
            return Ok(Some(item));
        }
        
        // Try L3 cache
        if let Some(item) = self.l3_cache.get(key).await? {
            self.cache_analytics.record_hit(CacheLevel::L3);
            
            // Promote to L2 if valuable
            if self.should_promote_to_l2(key).await? {
                self.l2_cache.set(key.clone(), &item).await?;
            }
            
            return Ok(Some(item));
        }
        
        self.cache_analytics.record_miss();
        Ok(None)
    }
    
    pub async fn set<T>(&self, key: CacheKey, item: &T) -> Result<(), CacheError>
    where
        T: CacheableItem + Clone + Send + Sync,
    {
        let cache_level = self.determine_optimal_cache_level(&key, item).await?;
        
        match cache_level {
            CacheLevel::L1 => {
                self.set_l1(key, item).await?;
            }
            CacheLevel::L2 => {
                self.l2_cache.set(key, item).await?;
            }
            CacheLevel::L3 => {
                self.l3_cache.set(key, item).await?;
            }
        }
        
        self.cache_analytics.record_insertion(&key, cache_level);
        Ok(())
    }
    
    pub async fn invalidate(&self, invalidation_request: &InvalidationRequest) -> Result<InvalidationResult, CacheError> {
        let start_time = Instant::now();
        
        // Calculate what needs to be invalidated
        let invalidation_plan = self.invalidation_engine.create_invalidation_plan(invalidation_request).await?;
        
        let mut invalidated_keys = Vec::new();
        
        // Execute invalidation plan
        for invalidation_action in &invalidation_plan.actions {
            match invalidation_action {
                InvalidationAction::InvalidateKey(key) => {
                    self.invalidate_key(key).await?;
                    invalidated_keys.push(key.clone());
                }
                InvalidationAction::InvalidatePattern(pattern) => {
                    let matching_keys = self.find_keys_matching_pattern(pattern).await?;
                    for key in matching_keys {
                        self.invalidate_key(&key).await?;
                        invalidated_keys.push(key);
                    }
                }
                InvalidationAction::InvalidateDependencies(base_key) => {
                    let dependent_keys = self.find_dependent_keys(base_key).await?;
                    for key in dependent_keys {
                        self.invalidate_key(&key).await?;
                        invalidated_keys.push(key);
                    }
                }
            }
        }
        
        Ok(InvalidationResult {
            invalidated_keys,
            invalidation_time_ms: start_time.elapsed().as_millis() as u64,
            cache_hit_ratio_impact: self.calculate_hit_ratio_impact(&invalidated_keys).await?,
        })
    }
    
    async fn determine_optimal_cache_level<T>(&self, key: &CacheKey, item: &T) -> Result<CacheLevel, CacheError>
    where
        T: CacheableItem,
    {
        // Factors to consider:
        // 1. Item size
        // 2. Access frequency prediction
        // 3. Computation cost
        // 4. Cache space availability
        
        let item_size = item.size_estimate();
        let access_frequency = self.predict_access_frequency(key).await?;
        let computation_cost = self.estimate_computation_cost(key).await?;
        
        // Decision logic
        if item_size < 1024 * 1024 && access_frequency > 0.8 {
            // Small, frequently accessed items go to L1
            Ok(CacheLevel::L1)
        } else if item_size < 10 * 1024 * 1024 && (access_frequency > 0.3 || computation_cost > 1000.0) {
            // Medium items with decent access or high computation cost go to L2
            Ok(CacheLevel::L2)
        } else {
            // Everything else goes to L3
            Ok(CacheLevel::L3)
        }
    }
    
    async fn predict_access_frequency(&self, key: &CacheKey) -> Result<f64, CacheError> {
        // Use historical data and ML to predict access frequency
        let historical_data = self.cache_analytics.get_historical_data(key).await?;
        
        if historical_data.is_empty() {
            // No historical data, use heuristics
            return Ok(self.estimate_initial_access_frequency(key));
        }
        
        // Simple exponential smoothing for prediction
        let mut prediction = 0.0;
        let alpha = 0.3; // Smoothing factor
        
        for access_data in historical_data {
            prediction = alpha * access_data.access_rate + (1.0 - alpha) * prediction;
        }
        
        Ok(prediction)
    }
}

pub struct L1Cache {
    parsed_asts: LruCache<String, UnifiedAST>,
    analysis_results: LruCache<String, AnalysisResult>,
    embeddings: LruCache<String, CodeEmbedding>,
    rule_results: LruCache<String, RuleResult>,
    metrics: LruCache<String, CodeMetrics>,
}

impl L1Cache {
    pub fn new(size_mb: usize) -> Self {
        let cache_entries = (size_mb * 1024 * 1024) / 10240; // Estimate 10KB per entry
        
        Self {
            parsed_asts: LruCache::new(NonZeroUsize::new(cache_entries / 5).unwrap()),
            analysis_results: LruCache::new(NonZeroUsize::new(cache_entries / 5).unwrap()),
            embeddings: LruCache::new(NonZeroUsize::new(cache_entries / 5).unwrap()),
            rule_results: LruCache::new(NonZeroUsize::new(cache_entries / 5).unwrap()),
            metrics: LruCache::new(NonZeroUsize::new(cache_entries / 5).unwrap()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheKey {
    pub cache_type: CacheType,
    pub primary_key: String,
    pub secondary_keys: Vec<String>,
    pub version: String,
    pub language: Option<ProgrammingLanguage>,
}

impl CacheKey {
    pub fn for_ast(file_path: &Path, content_hash: &str) -> Self {
        Self {
            cache_type: CacheType::ParsedAST,
            primary_key: file_path.to_string_lossy().to_string(),
            secondary_keys: vec![content_hash.to_string()],
            version: "1.0".to_string(),
            language: None,
        }
    }
    
    pub fn for_analysis_result(file_path: &Path, rules_hash: &str, content_hash: &str) -> Self {
        Self {
            cache_type: CacheType::AnalysisResult,
            primary_key: file_path.to_string_lossy().to_string(),
            secondary_keys: vec![rules_hash.to_string(), content_hash.to_string()],
            version: "1.0".to_string(),
            language: None,
        }
    }
    
    pub fn for_embedding(code_snippet: &str, model_id: &str, language: ProgrammingLanguage) -> Self {
        let snippet_hash = Self::hash_string(code_snippet);
        
        Self {
            cache_type: CacheType::AIEmbedding,
            primary_key: snippet_hash,
            secondary_keys: vec![model_id.to_string()],
            version: "1.0".to_string(),
            language: Some(language),
        }
    }
    
    fn hash_string(input: &str) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(input.as_bytes());
        hasher.finalize().to_hex().to_string()
    }
}
```

### 21.4 Incremental Analysis Engine

#### 21.4.1 Incremental Analyzer
```rust
pub struct IncrementalAnalyzer {
    change_detector: Arc<GranularChangeDetector>,
    cache_manager: Arc<SmartCacheManager>,
    dependency_analyzer: Arc<DependencyAnalyzer>,
    scope_calculator: Arc<ScopeCalculator>,
    delta_processor: Arc<DeltaProcessor>,
    config: IncrementalConfig,
}

#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    pub enable_aggressive_caching: bool,
    pub enable_predictive_analysis: bool,
    pub enable_background_warming: bool,
    pub max_incremental_scope_percentage: f64,
    pub force_full_analysis_threshold: f64,
    pub enable_delta_compression: bool,
    pub parallel_incremental_analysis: bool,
    pub incremental_batch_size: usize,
}

impl IncrementalAnalyzer {
    pub async fn analyze_incremental(&self, change_set: &ChangeSet, project_context: &ProjectContext) -> Result<IncrementalAnalysisResult, IncrementalAnalysisError> {
        let start_time = Instant::now();
        
        // Determine if incremental analysis is beneficial
        let scope_percentage = change_set.analysis_scope.estimated_analysis_time.as_secs_f64() / 
                              project_context.baseline_analysis_time.as_secs_f64();
        
        if scope_percentage > self.config.max_incremental_scope_percentage {
            // Too many changes, fall back to full analysis
            return Ok(IncrementalAnalysisResult::FullAnalysisRecommended {
                reason: "Change scope too large for incremental analysis".to_string(),
                estimated_benefit: 0.0,
            });
        }
        
        let mut incremental_result = IncrementalAnalysisResult::Success {
            change_set_id: change_set.id.clone(),
            files_analyzed: Vec::new(),
            cache_performance: CachePerformance::default(),
            analysis_results: HashMap::new(),
            delta_results: HashMap::new(),
            invalidated_cache_keys: Vec::new(),
            time_saved_ms: 0,
            total_analysis_time_ms: 0,
        };
        
        // Process each file in the analysis scope
        for file_path in &change_set.analysis_scope.files_to_analyze {
            let file_result = self.analyze_file_incremental(file_path, change_set, project_context).await?;
            
            if let IncrementalAnalysisResult::Success { 
                ref mut files_analyzed, 
                ref mut analysis_results, 
                ref mut cache_performance,
                .. 
            } = incremental_result {
                files_analyzed.push(file_path.clone());
                analysis_results.insert(file_path.clone(), file_result.analysis_result);
                cache_performance.merge(&file_result.cache_performance);
            }
        }
        
        // Calculate time saved
        if let IncrementalAnalysisResult::Success { ref mut time_saved_ms, ref mut total_analysis_time_ms, .. } = incremental_result {
            *total_analysis_time_ms = start_time.elapsed().as_millis() as u64;
            *time_saved_ms = self.calculate_time_saved(change_set, *total_analysis_time_ms, project_context).await?;
        }
        
        Ok(incremental_result)
    }
    
    async fn analyze_file_incremental(&self, file_path: &Path, change_set: &ChangeSet, project_context: &ProjectContext) -> Result<FileIncrementalResult, IncrementalAnalysisError> {
        let mut cache_performance = CachePerformance::default();
        
        // Check what analysis components we can reuse from cache
        let reusable_components = self.identify_reusable_components(file_path, change_set).await?;
        
        // Load cached components
        let mut cached_ast = None;
        let mut cached_analysis = None;
        let mut cached_embeddings = None;
        
        if reusable_components.can_reuse_ast {
            let ast_key = CacheKey::for_ast(file_path, &self.calculate_file_hash(file_path)?);
            cached_ast = self.cache_manager.get::<UnifiedAST>(&ast_key).await?;
            
            if cached_ast.is_some() {
                cache_performance.ast_cache_hit = true;
            }
        }
        
        if reusable_components.can_reuse_analysis {
            let analysis_key = self.build_analysis_cache_key(file_path, &project_context.active_rules)?;
            cached_analysis = self.cache_manager.get::<AnalysisResult>(&analysis_key).await?;
            
            if cached_analysis.is_some() {
                cache_performance.analysis_cache_hit = true;
            }
        }
        
        // Perform incremental analysis
        let analysis_result = if let (Some(ast), Some(analysis)) = (&cached_ast, &cached_analysis) {
            // We have both AST and analysis cached, perform delta analysis
            self.perform_delta_analysis(ast, analysis, change_set, file_path).await?
        } else if let Some(ast) = cached_ast {
            // We have AST cached, perform analysis only
            self.perform_analysis_only(ast, change_set, project_context).await?
        } else {
            // Need to parse and analyze from scratch
            self.perform_full_file_analysis(file_path, project_context).await?
        };
        
        // Cache the new results
        self.cache_new_results(file_path, &analysis_result, project_context).await?;
        
        Ok(FileIncrementalResult {
            file_path: file_path.to_path_buf(),
            analysis_result,
            cache_performance,
            components_reused: reusable_components,
        })
    }
    
    async fn perform_delta_analysis(&self, cached_ast: &UnifiedAST, cached_analysis: &AnalysisResult, change_set: &ChangeSet, file_path: &Path) -> Result<AnalysisResult, IncrementalAnalysisError> {
        // Find changes that affect this file
        let file_changes: Vec<_> = change_set.granular_changes.iter()
            .filter(|change| self.change_affects_file(change, file_path))
            .collect();
        
        if file_changes.is_empty() {
            // No changes affect this file, return cached result
            return Ok(cached_analysis.clone());
        }
        
        // Process delta changes
        let mut delta_analysis = cached_analysis.clone();
        
        for change in file_changes {
            let change_impact = self.delta_processor.process_change(change, cached_ast, &delta_analysis).await?;
            delta_analysis = self.apply_change_impact(delta_analysis, change_impact).await?;
        }
        
        // Update timestamps and metadata
        delta_analysis.analysis_timestamp = Utc::now();
        delta_analysis.is_incremental = true;
        delta_analysis.base_analysis_id = Some(cached_analysis.id.clone());
        
        Ok(delta_analysis)
    }
    
    async fn identify_reusable_components(&self, file_path: &Path, change_set: &ChangeSet) -> Result<ReusableComponents, IncrementalAnalysisError> {
        let mut components = ReusableComponents {
            can_reuse_ast: true,
            can_reuse_analysis: true,
            can_reuse_embeddings: true,
            can_reuse_metrics: true,
            reuse_confidence: 1.0,
        };
        
        // Check if file has direct changes
        let has_direct_changes = change_set.file_changes.iter()
            .any(|fc| fc.file_path == file_path);
        
        if has_direct_changes {
            components.can_reuse_ast = false;
            components.can_reuse_analysis = false;
            components.reuse_confidence *= 0.1;
        }
        
        // Check if file has dependency changes
        let has_dependency_changes = change_set.dependency_impact.transitively_affected_files
            .contains(file_path);
        
        if has_dependency_changes {
            components.can_reuse_analysis = false;
            components.can_reuse_embeddings = false;
            components.reuse_confidence *= 0.5;
        }
        
        Ok(components)
    }
}

#[derive(Debug, Clone)]
pub enum IncrementalAnalysisResult {
    Success {
        change_set_id: ChangeSetId,
        files_analyzed: Vec<PathBuf>,
        cache_performance: CachePerformance,
        analysis_results: HashMap<PathBuf, AnalysisResult>,
        delta_results: HashMap<PathBuf, DeltaAnalysisResult>,
        invalidated_cache_keys: Vec<CacheKey>,
        time_saved_ms: u64,
        total_analysis_time_ms: u64,
    },
    FullAnalysisRecommended {
        reason: String,
        estimated_benefit: f64,
    },
}

#[derive(Debug, Clone, Default)]
pub struct CachePerformance {
    pub ast_cache_hit: bool,
    pub analysis_cache_hit: bool,
    pub embedding_cache_hit: bool,
    pub metrics_cache_hit: bool,
    pub l1_hits: u32,
    pub l2_hits: u32,
    pub l3_hits: u32,
    pub total_accesses: u32,
    pub hit_ratio: f64,
}

impl CachePerformance {
    pub fn merge(&mut self, other: &CachePerformance) {
        self.l1_hits += other.l1_hits;
        self.l2_hits += other.l2_hits;
        self.l3_hits += other.l3_hits;
        self.total_accesses += other.total_accesses;
        
        let total_hits = self.l1_hits + self.l2_hits + self.l3_hits;
        self.hit_ratio = if self.total_accesses > 0 {
            total_hits as f64 / self.total_accesses as f64
        } else {
            0.0
        };
    }
}

#[derive(Debug, Clone)]
pub struct ReusableComponents {
    pub can_reuse_ast: bool,
    pub can_reuse_analysis: bool,
    pub can_reuse_embeddings: bool,
    pub can_reuse_metrics: bool,
    pub reuse_confidence: f64,
}

#[derive(Debug, Clone)]
pub struct DeltaAnalysisResult {
    pub file_path: PathBuf,
    pub changes_processed: Vec<GranularChangeId>,
    pub invalidated_results: Vec<String>,
    pub new_results: Vec<String>,
    pub delta_size_bytes: usize,
    pub processing_time_ms: u64,
}
```

### 21.5 Predictive Cache Warming

#### 21.5.1 Predictive Cache System
```rust
pub struct PredictiveCacheSystem {
    usage_predictor: Arc<UsagePredictor>,
    cache_warmer: Arc<CacheWarmer>,
    pattern_analyzer: Arc<UsagePatternAnalyzer>,
    ml_predictor: Option<Arc<MLUsagePredictor>>,
    config: PredictiveConfig,
}

#[derive(Debug, Clone)]
pub struct PredictiveConfig {
    pub enable_ml_prediction: bool,
    pub enable_pattern_based_prediction: bool,
    pub enable_time_based_prediction: bool,
    pub prediction_horizon_hours: u32,
    pub warming_batch_size: usize,
    pub max_warming_time_ms: u64,
    pub warming_priority_threshold: f64,
}

impl PredictiveCacheSystem {
    pub async fn predict_and_warm_cache(&self, project_context: &ProjectContext) -> Result<CacheWarmingResult, PredictiveError> {
        let start_time = Instant::now();
        
        // Predict what will be accessed soon
        let predictions = self.predict_upcoming_accesses(project_context).await?;
        
        // Filter predictions by priority
        let high_priority_predictions: Vec<_> = predictions.into_iter()
            .filter(|p| p.priority_score > self.config.warming_priority_threshold)
            .collect();
        
        // Warm cache for high-priority predictions
        let warming_results = self.cache_warmer.warm_cache(&high_priority_predictions).await?;
        
        Ok(CacheWarmingResult {
            predictions_made: high_priority_predictions.len(),
            cache_entries_warmed: warming_results.len(),
            warming_time_ms: start_time.elapsed().as_millis() as u64,
            estimated_time_savings: self.estimate_time_savings(&warming_results).await?,
        })
    }
    
    async fn predict_upcoming_accesses(&self, project_context: &ProjectContext) -> Result<Vec<AccessPrediction>, PredictiveError> {
        let mut predictions = Vec::new();
        
        // Pattern-based predictions
        if self.config.enable_pattern_based_prediction {
            let pattern_predictions = self.pattern_analyzer.predict_based_on_patterns(project_context).await?;
            predictions.extend(pattern_predictions);
        }
        
        // Time-based predictions
        if self.config.enable_time_based_prediction {
            let time_predictions = self.predict_based_on_time_patterns(project_context).await?;
            predictions.extend(time_predictions);
        }
        
        // ML-based predictions
        if let Some(ml_predictor) = &self.ml_predictor {
            let ml_predictions = ml_predictor.predict_accesses(project_context).await?;
            predictions.extend(ml_predictions);
        }
        
        // Deduplicate and prioritize
        predictions.sort_by(|a, b| b.priority_score.partial_cmp(&a.priority_score).unwrap());
        predictions.dedup_by(|a, b| a.cache_key == b.cache_key);
        
        Ok(predictions)
    }
    
    async fn predict_based_on_time_patterns(&self, project_context: &ProjectContext) -> Result<Vec<AccessPrediction>, PredictiveError> {
        let mut predictions = Vec::new();
        
        // Predict based on development patterns
        let current_hour = Utc::now().hour();
        
        // During working hours, predict analysis of recently modified files
        if current_hour >= 9 && current_hour <= 17 {
            let recent_files = self.get_recently_modified_files(project_context, Duration::from_hours(2)).await?;
            
            for file_path in recent_files {
                predictions.push(AccessPrediction {
                    cache_key: CacheKey::for_ast(&file_path, "latest"),
                    access_probability: 0.8,
                    priority_score: 0.7,
                    predicted_access_time: Utc::now() + Duration::from_minutes(30),
                    prediction_source: PredictionSource::TimePattern,
                });
            }
        }
        
        // Predict analysis of files that are typically modified together
        let co_modification_patterns = self.analyze_co_modification_patterns(project_context).await?;
        
        for pattern in co_modification_patterns {
            if pattern.confidence > 0.6 {
                for file_path in pattern.associated_files {
                    predictions.push(AccessPrediction {
                        cache_key: CacheKey::for_analysis_result(&file_path, "current_rules", "latest"),
                        access_probability: pattern.confidence,
                        priority_score: pattern.confidence * 0.6,
                        predicted_access_time: Utc::now() + Duration::from_minutes(15),
                        prediction_source: PredictionSource::CoModificationPattern,
                    });
                }
            }
        }
        
        Ok(predictions)
    }
}

pub struct CacheWarmer {
    analyzer: Arc<dyn CodeAnalyzer>,
    cache_manager: Arc<SmartCacheManager>,
    resource_monitor: Arc<ResourceMonitor>,
}

impl CacheWarmer {
    pub async fn warm_cache(&self, predictions: &[AccessPrediction]) -> Result<Vec<CacheWarmingEntry>, CacheWarmingError> {
        let mut warming_results = Vec::new();
        
        // Process predictions in priority order
        for prediction in predictions {
            // Check resource availability
            if !self.resource_monitor.has_sufficient_resources().await? {
                break; // Stop warming if resources are low
            }
            
            let warming_start = Instant::now();
            
            match self.warm_single_cache_entry(prediction).await {
                Ok(entry) => {
                    warming_results.push(CacheWarmingEntry {
                        cache_key: prediction.cache_key.clone(),
                        warming_time_ms: warming_start.elapsed().as_millis() as u64,
                        success: true,
                        data_size_bytes: entry.size_bytes,
                        prediction_accuracy: 0.0, // Will be updated when actually accessed
                    });
                }
                Err(error) => {
                    tracing::warn!("Failed to warm cache for key {:?}: {}", prediction.cache_key, error);
                    warming_results.push(CacheWarmingEntry {
                        cache_key: prediction.cache_key.clone(),
                        warming_time_ms: warming_start.elapsed().as_millis() as u64,
                        success: false,
                        data_size_bytes: 0,
                        prediction_accuracy: 0.0,
                    });
                }
            }
        }
        
        Ok(warming_results)
    }
    
    async fn warm_single_cache_entry(&self, prediction: &AccessPrediction) -> Result<CachedItem, CacheWarmingError> {
        match prediction.cache_key.cache_type {
            CacheType::ParsedAST => {
                let file_path = PathBuf::from(&prediction.cache_key.primary_key);
                let ast = self.analyzer.parse_file(&file_path).await?;
                self.cache_manager.set(prediction.cache_key.clone(), &ast).await?;
                Ok(CachedItem { size_bytes: ast.size_estimate() })
            }
            CacheType::AnalysisResult => {
                let file_path = PathBuf::from(&prediction.cache_key.primary_key);
                let analysis = self.analyzer.analyze_file(&file_path).await?;
                self.cache_manager.set(prediction.cache_key.clone(), &analysis).await?;
                Ok(CachedItem { size_bytes: analysis.size_estimate() })
            }
            CacheType::AIEmbedding => {
                // This would involve generating embeddings for predicted code
                self.warm_embedding_cache(prediction).await
            }
            _ => Err(CacheWarmingError::UnsupportedCacheType(prediction.cache_key.cache_type.clone())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AccessPrediction {
    pub cache_key: CacheKey,
    pub access_probability: f64,
    pub priority_score: f64,
    pub predicted_access_time: DateTime<Utc>,
    pub prediction_source: PredictionSource,
}

#[derive(Debug, Clone)]
pub enum PredictionSource {
    HistoricalPattern,
    TimePattern,
    CoModificationPattern,
    UserBehaviorPattern,
    MachineLearning,
    DependencyAnalysis,
}

#[derive(Debug, Clone)]
pub struct CacheWarmingResult {
    pub predictions_made: usize,
    pub cache_entries_warmed: usize,
    pub warming_time_ms: u64,
    pub estimated_time_savings: u64,
}
```

### 21.6 Criterios de Completitud

#### 21.6.1 Entregables de la Fase
- [ ] Sistema de análisis incremental implementado
- [ ] Cache inteligente multi-nivel funcionando
- [ ] Detector de cambios granular
- [ ] Motor de invalidación selectiva
- [ ] Sistema de cache warming predictivo
- [ ] Analizador de dependencias para invalidación
- [ ] Motor de análisis delta
- [ ] Sistema de métricas de cache
- [ ] API de análisis incremental
- [ ] Tests de performance y escalabilidad

#### 21.6.2 Criterios de Aceptación
- [ ] Análisis incremental es 10x+ más rápido que full analysis
- [ ] Cache hit ratio > 85% en uso típico
- [ ] Invalidación selectiva es precisa y eficiente
- [ ] Predictive warming mejora performance significativamente
- [ ] Sistema escala para repositorios de 1M+ LOC
- [ ] Memory usage se mantiene constante con cache
- [ ] Delta analysis preserva precisión del análisis
- [ ] Integration seamless con todas las fases anteriores
- [ ] Performance degradation < 5% con cache enabled
- [ ] Real-time analysis para cambios pequeños

### 21.7 Performance Targets

#### 21.7.1 Benchmarks de Performance
- **Incremental analysis**: <2 segundos para cambios típicos
- **Cache lookup**: <10ms para items hot
- **Cache warming**: >100 items/segundo
- **Memory usage**: <1GB cache overhead
- **Invalidation**: <100ms para invalidaciones complejas

### 21.8 Estimación de Tiempo

#### 21.8.1 Breakdown de Tareas
- Diseño de arquitectura incremental: 6 días
- Change detector granular: 12 días
- Smart cache manager multi-nivel: 15 días
- Incremental analyzer: 12 días
- Delta processor: 10 días
- Invalidation engine: 10 días
- Predictive cache system: 12 días
- Cache analytics y métricas: 8 días
- Performance optimization: 10 días
- Integration y testing: 12 días
- Documentación: 5 días

**Total estimado: 112 días de desarrollo**

### 21.9 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Performance optimizado para uso enterprise
- Análisis en tiempo real de cambios
- Eficiencia de cache de clase mundial
- Escalabilidad para equipos grandes
- Foundation para procesamiento distribuido

La Fase 22 construirá sobre esta optimización implementando procesamiento distribuido y paralelización para escalar a repositorios masivos.
