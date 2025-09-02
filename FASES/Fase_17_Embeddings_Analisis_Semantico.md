# Fase 17: Sistema de Embeddings de Código y Análisis Semántico

## Objetivo General
Desarrollar un sistema avanzado de embeddings de código y análisis semántico que utilice representaciones vectoriales densas para comprender el significado profundo del código, detectar similitudes semánticas, realizar búsquedas de código por intención, y proporcionar insights semánticos que van más allá del análisis sintáctico tradicional.

## Descripción Técnica Detallada

### 17.1 Arquitectura del Sistema de Embeddings

#### 17.1.1 Diseño del Semantic Analysis System
```
┌─────────────────────────────────────────┐
│        Semantic Analysis System        │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ Multi-Level │ │   Contextual        │ │
│  │ Embeddings  │ │   Embeddings        │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │  Semantic   │ │    Intent           │ │
│  │  Search     │ │   Detection         │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ Knowledge   │ │   Concept           │ │
│  │   Graph     │ │   Mapping           │ │
│  └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
```

#### 17.1.2 Niveles de Embeddings
- **Token-Level**: Embeddings de tokens individuales
- **Expression-Level**: Embeddings de expresiones y statements
- **Function-Level**: Embeddings de funciones completas
- **Class-Level**: Embeddings de clases y módulos
- **File-Level**: Embeddings de archivos completos
- **Project-Level**: Embeddings de proyectos enteros

### 17.2 Multi-Level Embedding System

#### 17.2.1 Multi-Level Embedding Engine
```rust
use std::collections::{HashMap, BTreeMap};
use nalgebra::{DVector, DMatrix};

pub struct MultiLevelEmbeddingEngine {
    token_embedder: Arc<TokenEmbedder>,
    expression_embedder: Arc<ExpressionEmbedder>,
    function_embedder: Arc<FunctionEmbedder>,
    class_embedder: Arc<ClassEmbedder>,
    file_embedder: Arc<FileEmbedder>,
    project_embedder: Arc<ProjectEmbedder>,
    hierarchical_aggregator: Arc<HierarchicalAggregator>,
    config: MultiLevelConfig,
}

#[derive(Debug, Clone)]
pub struct MultiLevelConfig {
    pub enable_token_embeddings: bool,
    pub enable_expression_embeddings: bool,
    pub enable_function_embeddings: bool,
    pub enable_class_embeddings: bool,
    pub enable_file_embeddings: bool,
    pub enable_project_embeddings: bool,
    pub aggregation_strategy: AggregationStrategy,
    pub context_window_size: usize,
    pub embedding_dimensions: HashMap<EmbeddingLevel, usize>,
    pub enable_hierarchical_attention: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EmbeddingLevel {
    Token,
    Expression,
    Function,
    Class,
    File,
    Project,
}

#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    Mean,
    WeightedMean,
    Attention,
    Hierarchical,
    GraphConvolution,
}

impl MultiLevelEmbeddingEngine {
    pub async fn new(config: MultiLevelConfig) -> Result<Self, EmbeddingError> {
        Ok(Self {
            token_embedder: Arc::new(TokenEmbedder::new()),
            expression_embedder: Arc::new(ExpressionEmbedder::new()),
            function_embedder: Arc::new(FunctionEmbedder::new()),
            class_embedder: Arc::new(ClassEmbedder::new()),
            file_embedder: Arc::new(FileEmbedder::new()),
            project_embedder: Arc::new(ProjectEmbedder::new()),
            hierarchical_aggregator: Arc::new(HierarchicalAggregator::new()),
            config,
        })
    }
    
    pub async fn generate_multi_level_embeddings(&self, ast: &UnifiedAST) -> Result<MultiLevelEmbeddings, EmbeddingError> {
        let start_time = Instant::now();
        
        let mut embeddings = MultiLevelEmbeddings {
            file_path: ast.file_path.clone(),
            language: ast.language,
            token_embeddings: HashMap::new(),
            expression_embeddings: HashMap::new(),
            function_embeddings: HashMap::new(),
            class_embeddings: HashMap::new(),
            file_embedding: None,
            hierarchical_structure: HierarchicalStructure::new(),
            semantic_relationships: Vec::new(),
            generation_time_ms: 0,
        };
        
        // Generate token-level embeddings
        if self.config.enable_token_embeddings {
            embeddings.token_embeddings = self.token_embedder.generate_token_embeddings(ast).await?;
        }
        
        // Generate expression-level embeddings
        if self.config.enable_expression_embeddings {
            embeddings.expression_embeddings = self.expression_embedder.generate_expression_embeddings(ast, &embeddings.token_embeddings).await?;
        }
        
        // Generate function-level embeddings
        if self.config.enable_function_embeddings {
            embeddings.function_embeddings = self.function_embedder.generate_function_embeddings(ast, &embeddings.expression_embeddings).await?;
        }
        
        // Generate class-level embeddings
        if self.config.enable_class_embeddings {
            embeddings.class_embeddings = self.class_embedder.generate_class_embeddings(ast, &embeddings.function_embeddings).await?;
        }
        
        // Generate file-level embedding
        if self.config.enable_file_embeddings {
            embeddings.file_embedding = Some(self.file_embedder.generate_file_embedding(ast, &embeddings).await?);
        }
        
        // Build hierarchical structure
        embeddings.hierarchical_structure = self.hierarchical_aggregator.build_hierarchy(&embeddings).await?;
        
        // Analyze semantic relationships
        embeddings.semantic_relationships = self.analyze_semantic_relationships(&embeddings).await?;
        
        embeddings.generation_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(embeddings)
    }
    
    pub async fn generate_contextual_embedding(&self, node: &UnifiedNode, context: &UnifiedAST) -> Result<ContextualEmbedding, EmbeddingError> {
        // Generate embedding considering surrounding context
        let context_nodes = self.extract_context_nodes(node, context, self.config.context_window_size)?;
        
        // Generate embeddings for context nodes
        let mut context_embeddings = Vec::new();
        for context_node in &context_nodes {
            let node_embedding = self.generate_node_embedding(context_node, context).await?;
            context_embeddings.push(node_embedding);
        }
        
        // Generate target node embedding
        let target_embedding = self.generate_node_embedding(node, context).await?;
        
        // Combine with attention mechanism
        let contextual_embedding = if self.config.enable_hierarchical_attention {
            self.apply_hierarchical_attention(&target_embedding, &context_embeddings).await?
        } else {
            self.apply_simple_attention(&target_embedding, &context_embeddings).await?
        };
        
        Ok(ContextualEmbedding {
            id: ContextualEmbeddingId::new(),
            target_node_id: node.id.clone(),
            target_embedding,
            context_embeddings,
            contextual_embedding,
            context_window_size: context_nodes.len(),
            attention_weights: self.extract_attention_weights(&context_embeddings),
            created_at: Utc::now(),
        })
    }
    
    async fn apply_hierarchical_attention(&self, target: &[f32], context: &[Vec<f32>]) -> Result<Vec<f32>, EmbeddingError> {
        // Implement hierarchical attention mechanism
        let mut attention_weights = Vec::new();
        
        // Calculate attention scores
        for context_embedding in context {
            let attention_score = self.calculate_attention_score(target, context_embedding)?;
            attention_weights.push(attention_score);
        }
        
        // Normalize attention weights
        let sum_weights: f32 = attention_weights.iter().sum();
        if sum_weights > 0.0 {
            for weight in &mut attention_weights {
                *weight /= sum_weights;
            }
        }
        
        // Apply attention to create contextual embedding
        let mut contextual_embedding = vec![0.0; target.len()];
        
        // Add target embedding
        for (i, &val) in target.iter().enumerate() {
            contextual_embedding[i] += val * 0.5; // 50% weight for target
        }
        
        // Add weighted context embeddings
        for (context_embedding, &weight) in context.iter().zip(attention_weights.iter()) {
            for (i, &val) in context_embedding.iter().enumerate() {
                if i < contextual_embedding.len() {
                    contextual_embedding[i] += val * weight * 0.5; // 50% weight for context
                }
            }
        }
        
        Ok(contextual_embedding)
    }
    
    fn calculate_attention_score(&self, query: &[f32], key: &[f32]) -> Result<f32, EmbeddingError> {
        // Simplified attention score calculation (dot product)
        let score: f32 = query.iter().zip(key.iter()).map(|(q, k)| q * k).sum();
        Ok(score)
    }
}

#[derive(Debug, Clone)]
pub struct MultiLevelEmbeddings {
    pub file_path: PathBuf,
    pub language: ProgrammingLanguage,
    pub token_embeddings: HashMap<TokenId, TokenEmbedding>,
    pub expression_embeddings: HashMap<ExpressionId, ExpressionEmbedding>,
    pub function_embeddings: HashMap<FunctionId, FunctionEmbedding>,
    pub class_embeddings: HashMap<ClassId, ClassEmbedding>,
    pub file_embedding: Option<FileEmbedding>,
    pub hierarchical_structure: HierarchicalStructure,
    pub semantic_relationships: Vec<SemanticRelationship>,
    pub generation_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ContextualEmbedding {
    pub id: ContextualEmbeddingId,
    pub target_node_id: NodeId,
    pub target_embedding: Vec<f32>,
    pub context_embeddings: Vec<Vec<f32>>,
    pub contextual_embedding: Vec<f32>,
    pub context_window_size: usize,
    pub attention_weights: Vec<f32>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct HierarchicalStructure {
    pub levels: BTreeMap<EmbeddingLevel, LevelInfo>,
    pub parent_child_relationships: Vec<HierarchicalRelationship>,
    pub aggregation_weights: HashMap<EmbeddingLevel, f64>,
}

#[derive(Debug, Clone)]
pub struct LevelInfo {
    pub level: EmbeddingLevel,
    pub embedding_count: usize,
    pub average_dimension: usize,
    pub quality_score: f64,
}

#[derive(Debug, Clone)]
pub struct HierarchicalRelationship {
    pub parent_id: String,
    pub child_id: String,
    pub parent_level: EmbeddingLevel,
    pub child_level: EmbeddingLevel,
    pub relationship_strength: f64,
}
```

### 17.3 Semantic Search Engine

#### 17.3.1 Semantic Code Search
```rust
pub struct SemanticSearchEngine {
    embedding_engine: Arc<MultiLevelEmbeddingEngine>,
    vector_store: Arc<VectorStore>,
    query_processor: Arc<QueryProcessor>,
    result_ranker: Arc<ResultRanker>,
    semantic_matcher: Arc<SemanticMatcher>,
    config: SemanticSearchConfig,
}

#[derive(Debug, Clone)]
pub struct SemanticSearchConfig {
    pub max_results: usize,
    pub similarity_threshold: f64,
    pub enable_query_expansion: bool,
    pub enable_semantic_filtering: bool,
    pub enable_cross_language_search: bool,
    pub ranking_algorithm: RankingAlgorithm,
    pub search_timeout_ms: u64,
}

#[derive(Debug, Clone)]
pub enum RankingAlgorithm {
    CosineSimilarity,
    EuclideanDistance,
    HybridRanking,
    LearnedRanking,
}

impl SemanticSearchEngine {
    pub async fn search_by_natural_language(&self, query: &str, languages: &[ProgrammingLanguage]) -> Result<SemanticSearchResult, SearchError> {
        let start_time = Instant::now();
        
        // Process natural language query
        let processed_query = self.query_processor.process_natural_language_query(query).await?;
        
        // Generate query embedding
        let query_embedding = self.generate_query_embedding(&processed_query).await?;
        
        // Search in vector store
        let mut search_results = Vec::new();
        
        for &language in languages {
            let lang_results = self.vector_store.search_similar_embeddings(
                &query_embedding,
                self.config.max_results / languages.len(),
                Some(language),
            ).await?;
            search_results.extend(lang_results);
        }
        
        // Rank and filter results
        let ranked_results = self.result_ranker.rank_results(search_results, &processed_query).await?;
        
        // Apply semantic filtering
        let filtered_results = if self.config.enable_semantic_filtering {
            self.semantic_matcher.filter_by_semantic_relevance(ranked_results, &processed_query).await?
        } else {
            ranked_results
        };
        
        Ok(SemanticSearchResult {
            query: query.to_string(),
            processed_query,
            results: filtered_results,
            total_results: search_results.len(),
            search_time_ms: start_time.elapsed().as_millis() as u64,
            languages_searched: languages.to_vec(),
        })
    }
    
    pub async fn search_by_code_example(&self, example_code: &str, language: ProgrammingLanguage) -> Result<SemanticSearchResult, SearchError> {
        // Generate embedding for example code
        let example_embedding = self.embedding_engine.generate_code_embedding(example_code, language).await?;
        
        // Search for similar code
        let similar_results = self.vector_store.search_similar_embeddings(
            &example_embedding.embedding_vector,
            self.config.max_results,
            None, // Search across all languages
        ).await?;
        
        // Analyze semantic similarity
        let semantic_results = self.analyze_semantic_similarity(&example_embedding, &similar_results).await?;
        
        Ok(SemanticSearchResult {
            query: format!("Code example: {}", example_code.lines().next().unwrap_or("")),
            processed_query: ProcessedQuery {
                original_query: example_code.to_string(),
                intent: QueryIntent::FindSimilarCode,
                concepts: self.extract_concepts_from_code(example_code, language).await?,
                constraints: Vec::new(),
                expanded_terms: Vec::new(),
            },
            results: semantic_results,
            total_results: similar_results.len(),
            search_time_ms: 0,
            languages_searched: vec![language],
        })
    }
    
    pub async fn search_by_intent(&self, intent: CodeIntent, context: &SearchContext) -> Result<IntentSearchResult, SearchError> {
        // Convert intent to searchable embedding
        let intent_embedding = self.generate_intent_embedding(&intent, context).await?;
        
        // Search for code that matches the intent
        let matching_code = self.find_code_by_intent(&intent_embedding, &intent, context).await?;
        
        // Analyze how well each result matches the intent
        let intent_matches = self.analyze_intent_matches(&intent, &matching_code).await?;
        
        Ok(IntentSearchResult {
            intent: intent.clone(),
            context: context.clone(),
            matches: intent_matches,
            confidence_distribution: self.calculate_confidence_distribution(&intent_matches),
            alternative_implementations: self.find_alternative_implementations(&intent, context).await?,
        })
    }
    
    async fn generate_query_embedding(&self, query: &ProcessedQuery) -> Result<Vec<f32>, SearchError> {
        match &query.intent {
            QueryIntent::FindSimilarCode => {
                // Use code embedding directly
                self.embedding_engine.generate_code_embedding(&query.original_query, ProgrammingLanguage::Python).await
                    .map(|emb| emb.embedding_vector)
                    .map_err(SearchError::EmbeddingError)
            }
            QueryIntent::FindByFunction => {
                // Generate function-focused embedding
                self.generate_function_query_embedding(query).await
            }
            QueryIntent::FindByPattern => {
                // Generate pattern-focused embedding
                self.generate_pattern_query_embedding(query).await
            }
            QueryIntent::FindByBehavior => {
                // Generate behavior-focused embedding
                self.generate_behavior_query_embedding(query).await
            }
        }
    }
    
    async fn analyze_semantic_similarity(&self, query_embedding: &CodeEmbedding, results: &[EmbeddingSearchResult]) -> Result<Vec<SemanticSearchResultItem>, SearchError> {
        let mut semantic_results = Vec::new();
        
        for result in results {
            // Calculate multiple similarity metrics
            let cosine_similarity = self.calculate_cosine_similarity(&query_embedding.embedding_vector, &result.embedding_vector)?;
            let semantic_similarity = self.calculate_semantic_similarity(query_embedding, result).await?;
            let structural_similarity = self.calculate_structural_similarity(query_embedding, result).await?;
            
            // Combine similarities with weights
            let combined_similarity = (
                cosine_similarity * 0.4 +
                semantic_similarity * 0.4 +
                structural_similarity * 0.2
            );
            
            // Extract semantic features
            let semantic_features = self.extract_semantic_features(result).await?;
            
            // Generate explanation
            let explanation = self.generate_similarity_explanation(query_embedding, result, combined_similarity).await?;
            
            semantic_results.push(SemanticSearchResultItem {
                id: result.id.clone(),
                code_snippet: result.code_snippet.clone(),
                language: result.language,
                cosine_similarity,
                semantic_similarity,
                structural_similarity,
                combined_similarity,
                semantic_features,
                explanation,
                confidence: self.calculate_result_confidence(combined_similarity, semantic_features.len()),
            });
        }
        
        // Sort by combined similarity
        semantic_results.sort_by(|a, b| b.combined_similarity.partial_cmp(&a.combined_similarity).unwrap());
        
        Ok(semantic_results)
    }
}

#[derive(Debug, Clone)]
pub struct SemanticSearchResult {
    pub query: String,
    pub processed_query: ProcessedQuery,
    pub results: Vec<SemanticSearchResultItem>,
    pub total_results: usize,
    pub search_time_ms: u64,
    pub languages_searched: Vec<ProgrammingLanguage>,
}

#[derive(Debug, Clone)]
pub struct ProcessedQuery {
    pub original_query: String,
    pub intent: QueryIntent,
    pub concepts: Vec<CodeConcept>,
    pub constraints: Vec<QueryConstraint>,
    pub expanded_terms: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum QueryIntent {
    FindSimilarCode,
    FindByFunction,
    FindByPattern,
    FindByBehavior,
    FindByPurpose,
    FindImplementations,
    FindAlternatives,
}

#[derive(Debug, Clone)]
pub struct CodeConcept {
    pub concept_type: ConceptType,
    pub name: String,
    pub confidence: f64,
    pub related_concepts: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ConceptType {
    Algorithm,
    DataStructure,
    DesignPattern,
    ProgrammingConstruct,
    Domain,
    Framework,
    Library,
}

#[derive(Debug, Clone)]
pub struct SemanticSearchResultItem {
    pub id: EmbeddingId,
    pub code_snippet: String,
    pub language: ProgrammingLanguage,
    pub cosine_similarity: f64,
    pub semantic_similarity: f64,
    pub structural_similarity: f64,
    pub combined_similarity: f64,
    pub semantic_features: Vec<SemanticFeature>,
    pub explanation: SimilarityExplanation,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct SemanticFeature {
    pub feature_type: FeatureType,
    pub value: f64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum FeatureType {
    Complexity,
    Abstraction,
    Coupling,
    Cohesion,
    Reusability,
    Testability,
    Performance,
    Security,
}
```

### 17.4 Intent Detection System

#### 17.4.1 Code Intent Detector
```rust
pub struct CodeIntentDetector {
    intent_classifier: Arc<IntentClassifier>,
    purpose_analyzer: Arc<PurposeAnalyzer>,
    behavior_analyzer: Arc<BehaviorAnalyzer>,
    domain_analyzer: Arc<DomainAnalyzer>,
    config: IntentDetectionConfig,
}

#[derive(Debug, Clone)]
pub struct IntentDetectionConfig {
    pub enable_ml_intent_detection: bool,
    pub enable_pattern_based_detection: bool,
    pub enable_domain_analysis: bool,
    pub confidence_threshold: f64,
    pub max_intents_per_code: usize,
    pub enable_multi_intent_detection: bool,
}

impl CodeIntentDetector {
    pub async fn detect_code_intent(&self, code_embedding: &CodeEmbedding, ast_node: &UnifiedNode) -> Result<CodeIntentAnalysis, IntentDetectionError> {
        let mut intent_analysis = CodeIntentAnalysis {
            code_id: code_embedding.id.clone(),
            detected_intents: Vec::new(),
            primary_purpose: None,
            behavioral_characteristics: Vec::new(),
            domain_concepts: Vec::new(),
            confidence_scores: HashMap::new(),
        };
        
        // ML-based intent detection
        if self.config.enable_ml_intent_detection {
            let ml_intents = self.intent_classifier.classify_intent(code_embedding).await?;
            intent_analysis.detected_intents.extend(ml_intents);
        }
        
        // Pattern-based intent detection
        if self.config.enable_pattern_based_detection {
            let pattern_intents = self.detect_pattern_based_intents(ast_node).await?;
            intent_analysis.detected_intents.extend(pattern_intents);
        }
        
        // Purpose analysis
        intent_analysis.primary_purpose = self.purpose_analyzer.analyze_purpose(code_embedding, ast_node).await?;
        
        // Behavioral analysis
        intent_analysis.behavioral_characteristics = self.behavior_analyzer.analyze_behavior(ast_node).await?;
        
        // Domain analysis
        if self.config.enable_domain_analysis {
            intent_analysis.domain_concepts = self.domain_analyzer.analyze_domain(code_embedding, ast_node).await?;
        }
        
        // Calculate confidence scores
        for intent in &intent_analysis.detected_intents {
            let confidence = self.calculate_intent_confidence(intent, code_embedding, ast_node).await?;
            intent_analysis.confidence_scores.insert(intent.intent_type.clone(), confidence);
        }
        
        // Filter by confidence threshold
        intent_analysis.detected_intents.retain(|intent| {
            intent_analysis.confidence_scores.get(&intent.intent_type)
                .map(|&conf| conf >= self.config.confidence_threshold)
                .unwrap_or(false)
        });
        
        Ok(intent_analysis)
    }
    
    async fn detect_pattern_based_intents(&self, node: &UnifiedNode) -> Result<Vec<DetectedIntent>, IntentDetectionError> {
        let mut intents = Vec::new();
        
        // Analyze node structure to infer intent
        match &node.node_type {
            UnifiedNodeType::FunctionDeclaration { .. } => {
                // Analyze function to determine its intent
                let function_intent = self.analyze_function_intent(node).await?;
                intents.push(function_intent);
            }
            UnifiedNodeType::ClassDeclaration { .. } => {
                // Analyze class to determine its intent
                let class_intent = self.analyze_class_intent(node).await?;
                intents.push(class_intent);
            }
            UnifiedNodeType::ForStatement | UnifiedNodeType::WhileStatement => {
                intents.push(DetectedIntent {
                    intent_type: IntentType::Iteration,
                    description: "Code performs iteration/looping".to_string(),
                    evidence: vec!["Contains loop construct".to_string()],
                    confidence: 0.9,
                });
            }
            UnifiedNodeType::IfStatement => {
                intents.push(DetectedIntent {
                    intent_type: IntentType::ConditionalLogic,
                    description: "Code performs conditional branching".to_string(),
                    evidence: vec!["Contains conditional statement".to_string()],
                    confidence: 0.9,
                });
            }
            UnifiedNodeType::TryStatement => {
                intents.push(DetectedIntent {
                    intent_type: IntentType::ErrorHandling,
                    description: "Code handles errors/exceptions".to_string(),
                    evidence: vec!["Contains try-catch block".to_string()],
                    confidence: 0.95,
                });
            }
            _ => {}
        }
        
        Ok(intents)
    }
    
    async fn analyze_function_intent(&self, function_node: &UnifiedNode) -> Result<DetectedIntent, IntentDetectionError> {
        let function_name = function_node.name.as_ref().unwrap_or(&"unknown".to_string()).to_lowercase();
        
        // Analyze function name for intent clues
        let intent_type = if function_name.starts_with("get") || function_name.starts_with("fetch") || function_name.starts_with("read") {
            IntentType::DataRetrieval
        } else if function_name.starts_with("set") || function_name.starts_with("update") || function_name.starts_with("write") {
            IntentType::DataModification
        } else if function_name.starts_with("create") || function_name.starts_with("make") || function_name.starts_with("build") {
            IntentType::ObjectCreation
        } else if function_name.starts_with("delete") || function_name.starts_with("remove") || function_name.starts_with("destroy") {
            IntentType::ObjectDestruction
        } else if function_name.starts_with("validate") || function_name.starts_with("check") || function_name.starts_with("verify") {
            IntentType::Validation
        } else if function_name.starts_with("calculate") || function_name.starts_with("compute") || function_name.contains("math") {
            IntentType::Calculation
        } else if function_name.starts_with("transform") || function_name.starts_with("convert") || function_name.contains("parse") {
            IntentType::DataTransformation
        } else if function_name.starts_with("test") || function_name.ends_with("test") {
            IntentType::Testing
        } else {
            IntentType::General
        };
        
        // Analyze function body for additional evidence
        let body_evidence = self.analyze_function_body_for_intent(function_node).await?;
        
        Ok(DetectedIntent {
            intent_type,
            description: format!("Function '{}' appears to perform {}", function_name, intent_type.description()),
            evidence: body_evidence,
            confidence: self.calculate_function_intent_confidence(&function_name, &body_evidence),
        })
    }
    
    async fn analyze_function_body_for_intent(&self, function_node: &UnifiedNode) -> Result<Vec<String>, IntentDetectionError> {
        let mut evidence = Vec::new();
        
        // Look for specific patterns in function body
        let mut visitor = IntentEvidenceVisitor::new(&mut evidence);
        visitor.visit_node(function_node);
        
        Ok(evidence)
    }
}

#[derive(Debug, Clone)]
pub struct CodeIntentAnalysis {
    pub code_id: EmbeddingId,
    pub detected_intents: Vec<DetectedIntent>,
    pub primary_purpose: Option<CodePurpose>,
    pub behavioral_characteristics: Vec<BehavioralCharacteristic>,
    pub domain_concepts: Vec<DomainConcept>,
    pub confidence_scores: HashMap<IntentType, f64>,
}

#[derive(Debug, Clone)]
pub struct DetectedIntent {
    pub intent_type: IntentType,
    pub description: String,
    pub evidence: Vec<String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntentType {
    DataRetrieval,
    DataModification,
    DataTransformation,
    ObjectCreation,
    ObjectDestruction,
    Validation,
    Calculation,
    Communication,
    ErrorHandling,
    Testing,
    Logging,
    Configuration,
    Security,
    Performance,
    ConditionalLogic,
    Iteration,
    General,
}

impl IntentType {
    pub fn description(&self) -> &'static str {
        match self {
            Self::DataRetrieval => "data retrieval operations",
            Self::DataModification => "data modification operations",
            Self::DataTransformation => "data transformation operations",
            Self::ObjectCreation => "object creation operations",
            Self::ObjectDestruction => "object destruction operations",
            Self::Validation => "validation operations",
            Self::Calculation => "calculation operations",
            Self::Communication => "communication operations",
            Self::ErrorHandling => "error handling operations",
            Self::Testing => "testing operations",
            Self::Logging => "logging operations",
            Self::Configuration => "configuration operations",
            Self::Security => "security operations",
            Self::Performance => "performance operations",
            Self::ConditionalLogic => "conditional logic",
            Self::Iteration => "iteration operations",
            Self::General => "general operations",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CodePurpose {
    pub purpose_type: PurposeType,
    pub description: String,
    pub abstraction_level: AbstractionLevel,
    pub domain_specificity: f64,
    pub reusability_score: f64,
}

#[derive(Debug, Clone)]
pub enum PurposeType {
    BusinessLogic,
    DataAccess,
    UserInterface,
    Infrastructure,
    Utility,
    Framework,
    Library,
    Application,
    Service,
    Component,
}

#[derive(Debug, Clone)]
pub enum AbstractionLevel {
    Low,      // Hardware/system level
    Medium,   // Framework/library level
    High,     // Business/domain level
    VeryHigh, // Conceptual/abstract level
}
```

### 17.5 Knowledge Graph Construction

#### 17.5.1 Code Knowledge Graph
```rust
pub struct CodeKnowledgeGraph {
    graph: Graph<KnowledgeNode, KnowledgeEdge>,
    node_index: HashMap<String, NodeIndex>,
    embedding_index: HashMap<EmbeddingId, NodeIndex>,
    concept_index: HashMap<ConceptId, NodeIndex>,
    relationship_analyzer: Arc<RelationshipAnalyzer>,
    graph_builder: Arc<GraphBuilder>,
}

#[derive(Debug, Clone)]
pub struct KnowledgeNode {
    pub id: String,
    pub node_type: KnowledgeNodeType,
    pub embedding: Option<Vec<f32>>,
    pub metadata: NodeMetadata,
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum KnowledgeNodeType {
    Function,
    Class,
    Module,
    Concept,
    Pattern,
    Domain,
    Framework,
    Library,
}

#[derive(Debug, Clone)]
pub struct KnowledgeEdge {
    pub edge_type: EdgeType,
    pub weight: f64,
    pub confidence: f64,
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum EdgeType {
    CallsFunction,
    ImplementsInterface,
    ExtendsClass,
    UsesLibrary,
    SimilarTo,
    ConceptuallyRelated,
    FunctionallyEquivalent,
    DependsOn,
    Aggregates,
    Composes,
}

impl CodeKnowledgeGraph {
    pub async fn build_from_embeddings(&mut self, embeddings: &[MultiLevelEmbeddings]) -> Result<(), KnowledgeGraphError> {
        // Add nodes for each code entity
        for embedding_set in embeddings {
            self.add_nodes_from_embeddings(embedding_set).await?;
        }
        
        // Analyze relationships between nodes
        self.analyze_and_add_relationships(embeddings).await?;
        
        // Build concept clusters
        self.build_concept_clusters().await?;
        
        // Calculate centrality measures
        self.calculate_centrality_measures().await?;
        
        Ok(())
    }
    
    async fn add_nodes_from_embeddings(&mut self, embeddings: &MultiLevelEmbeddings) -> Result<(), KnowledgeGraphError> {
        // Add function nodes
        for (func_id, func_embedding) in &embeddings.function_embeddings {
            let node = KnowledgeNode {
                id: func_id.to_string(),
                node_type: KnowledgeNodeType::Function,
                embedding: Some(func_embedding.code_embedding.clone()),
                metadata: NodeMetadata {
                    file_path: embeddings.file_path.clone(),
                    language: embeddings.language,
                    created_at: Utc::now(),
                },
                properties: HashMap::from([
                    ("function_name".to_string(), serde_json::Value::String(func_embedding.function_name.clone())),
                    ("complexity".to_string(), serde_json::Value::Number(func_embedding.complexity_metrics.cyclomatic_complexity.into())),
                    ("lines_of_code".to_string(), serde_json::Value::Number(func_embedding.function_features.lines_of_code.into())),
                ]),
            };
            
            let node_index = self.graph.add_node(node);
            self.node_index.insert(func_id.to_string(), node_index);
            self.embedding_index.insert(func_embedding.id.clone(), node_index);
        }
        
        // Add class nodes
        for (class_id, class_embedding) in &embeddings.class_embeddings {
            let node = KnowledgeNode {
                id: class_id.to_string(),
                node_type: KnowledgeNodeType::Class,
                embedding: Some(class_embedding.code_embedding.clone()),
                metadata: NodeMetadata {
                    file_path: embeddings.file_path.clone(),
                    language: embeddings.language,
                    created_at: Utc::now(),
                },
                properties: HashMap::from([
                    ("class_name".to_string(), serde_json::Value::String(class_embedding.class_name.clone())),
                    ("method_count".to_string(), serde_json::Value::Number(class_embedding.class_features.method_count.into())),
                ]),
            };
            
            let node_index = self.graph.add_node(node);
            self.node_index.insert(class_id.to_string(), node_index);
            self.embedding_index.insert(class_embedding.id.clone(), node_index);
        }
        
        Ok(())
    }
    
    async fn analyze_and_add_relationships(&mut self, embeddings: &[MultiLevelEmbeddings]) -> Result<(), KnowledgeGraphError> {
        // Analyze call relationships
        self.add_call_relationships(embeddings).await?;
        
        // Analyze inheritance relationships
        self.add_inheritance_relationships(embeddings).await?;
        
        // Analyze similarity relationships
        self.add_similarity_relationships(embeddings).await?;
        
        // Analyze conceptual relationships
        self.add_conceptual_relationships(embeddings).await?;
        
        Ok(())
    }
    
    async fn add_similarity_relationships(&mut self, embeddings: &[MultiLevelEmbeddings]) -> Result<(), KnowledgeGraphError> {
        // Find similar functions across files
        for embedding_set in embeddings {
            for (func_id, func_embedding) in &embedding_set.function_embeddings {
                // Find similar functions in other files
                let similar_functions = self.find_similar_functions(&func_embedding.code_embedding).await?;
                
                for similar in similar_functions {
                    if similar.similarity_score > 0.8 {
                        if let (Some(&from_idx), Some(&to_idx)) = (
                            self.embedding_index.get(&func_embedding.id),
                            self.embedding_index.get(&similar.id)
                        ) {
                            let edge = KnowledgeEdge {
                                edge_type: EdgeType::SimilarTo,
                                weight: similar.similarity_score,
                                confidence: similar.similarity_score,
                                properties: HashMap::from([
                                    ("similarity_type".to_string(), serde_json::Value::String("semantic".to_string())),
                                ]),
                            };
                            
                            self.graph.add_edge(from_idx, to_idx, edge);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    pub async fn query_knowledge_graph(&self, query: &KnowledgeGraphQuery) -> Result<KnowledgeGraphResult, KnowledgeGraphError> {
        match query {
            KnowledgeGraphQuery::FindSimilarFunctions { function_id, similarity_threshold } => {
                self.find_similar_functions_in_graph(function_id, *similarity_threshold).await
            }
            KnowledgeGraphQuery::FindConceptuallyRelated { concept, max_results } => {
                self.find_conceptually_related_nodes(concept, *max_results).await
            }
            KnowledgeGraphQuery::AnalyzeDependencies { node_id, depth } => {
                self.analyze_dependencies_in_graph(node_id, *depth).await
            }
            KnowledgeGraphQuery::FindCentralNodes { centrality_type, top_k } => {
                self.find_central_nodes(centrality_type, *top_k).await
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum KnowledgeGraphQuery {
    FindSimilarFunctions {
        function_id: String,
        similarity_threshold: f64,
    },
    FindConceptuallyRelated {
        concept: ConceptId,
        max_results: usize,
    },
    AnalyzeDependencies {
        node_id: String,
        depth: usize,
    },
    FindCentralNodes {
        centrality_type: CentralityType,
        top_k: usize,
    },
}

#[derive(Debug, Clone)]
pub enum CentralityType {
    Degree,
    Betweenness,
    Closeness,
    PageRank,
    Eigenvector,
}

#[derive(Debug, Clone)]
pub struct KnowledgeGraphResult {
    pub query_type: String,
    pub nodes: Vec<KnowledgeNode>,
    pub edges: Vec<KnowledgeEdge>,
    pub insights: Vec<GraphInsight>,
    pub metrics: GraphMetrics,
}

#[derive(Debug, Clone)]
pub struct GraphInsight {
    pub insight_type: InsightType,
    pub description: String,
    pub confidence: f64,
    pub supporting_evidence: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum InsightType {
    HighCoupling,
    LowCohesion,
    CentralFunction,
    IsolatedComponent,
    PotentialRefactoring,
    ArchitecturalIssue,
    DesignPatternOpportunity,
}
```

### 17.6 Criterios de Completitud

#### 17.6.1 Entregables de la Fase
- [ ] Sistema multi-level de embeddings implementado
- [ ] Motor de búsqueda semántica funcionando
- [ ] Detector de intención de código
- [ ] Knowledge graph de código construido
- [ ] Sistema de análisis contextual
- [ ] Agregador jerárquico de embeddings
- [ ] Motor de análisis de relaciones semánticas
- [ ] API de búsqueda por lenguaje natural
- [ ] Sistema de explicaciones de similitud
- [ ] Tests comprehensivos de embeddings

#### 17.6.2 Criterios de Aceptación
- [ ] Embeddings multi-level son consistentes y útiles
- [ ] Búsqueda semántica encuentra código relevante
- [ ] Detección de intención es precisa >80%
- [ ] Knowledge graph proporciona insights valiosos
- [ ] Análisis contextual mejora precisión
- [ ] Performance acceptable para uso interactivo
- [ ] Búsqueda por lenguaje natural funciona
- [ ] Explicaciones de similitud son comprensibles
- [ ] Integration seamless con fases anteriores
- [ ] Escalabilidad para proyectos grandes

### 17.7 Performance Targets

#### 17.7.1 Benchmarks de Embeddings
- **Multi-level generation**: <2 segundos para archivos típicos
- **Semantic search**: <500ms para queries típicas
- **Intent detection**: <200ms por función
- **Knowledge graph queries**: <1 segundo para queries complejas
- **Memory usage**: <1GB para proyectos medianos

### 17.8 Estimación de Tiempo

#### 17.8.1 Breakdown de Tareas
- Diseño de arquitectura multi-level: 5 días
- Token y expression embedders: 8 días
- Function y class embedders: 10 días
- File y project embedders: 8 días
- Hierarchical aggregator: 8 días
- Semantic search engine: 12 días
- Intent detection system: 10 días
- Knowledge graph construction: 12 días
- Contextual analysis system: 8 días
- Performance optimization: 8 días
- Integration y testing: 10 días
- Documentación: 4 días

**Total estimado: 103 días de desarrollo**

### 17.9 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Comprensión semántica profunda del código
- Capacidades de búsqueda por intención
- Knowledge graph para insights arquitectónicos
- Base sólida para detección de antipatrones con IA
- Foundation para análisis de comportamiento

La Fase 18 construirá sobre esta base implementando la detección de antipatrones usando IA, aprovechando las capacidades semánticas desarrolladas aquí.

