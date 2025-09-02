# Fase 15: Sistema de Categorización y Priorización de Issues

## Objetivo General
Implementar un sistema inteligente de categorización y priorización de issues que clasifique automáticamente los problemas detectados por el motor de reglas, asigne prioridades basadas en impacto y urgencia, agrupe issues relacionados, proporcione recomendaciones de fixes, y genere planes de remediación optimizados para maximizar el retorno de inversión en calidad de código.

## Descripción Técnica Detallada

### 15.1 Arquitectura del Sistema de Categorización

#### 15.1.1 Diseño del Issue Management System
```
┌─────────────────────────────────────────┐
│       Issue Management System          │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Issue     │ │    Priority         │ │
│  │Categorizer  │ │   Calculator        │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Impact    │ │    Clustering       │ │
│  │  Analyzer   │ │    Engine           │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │Remediation  │ │     ROI             │ │
│  │  Planner    │ │   Calculator        │ │
│  └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
```

#### 15.1.2 Componentes del Sistema
- **Issue Categorizer**: Clasificación automática de issues
- **Priority Calculator**: Cálculo de prioridades inteligente
- **Impact Analyzer**: Análisis de impacto en calidad/negocio
- **Clustering Engine**: Agrupación de issues relacionados
- **Remediation Planner**: Planificación de fixes
- **ROI Calculator**: Cálculo de retorno de inversión

### 15.2 Issue Classification System

#### 15.2.1 Issue Categorizer Implementation
```rust
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

pub struct IssueCategorizer {
    classification_rules: Vec<ClassificationRule>,
    ml_classifier: Option<Arc<MLClassifier>>,
    domain_knowledge: Arc<DomainKnowledge>,
    similarity_calculator: Arc<SimilarityCalculator>,
    config: CategorizationConfig,
}

#[derive(Debug, Clone)]
pub struct CategorizationConfig {
    pub enable_ml_classification: bool,
    pub enable_similarity_grouping: bool,
    pub enable_context_analysis: bool,
    pub similarity_threshold: f64,
    pub confidence_threshold: f64,
    pub max_categories_per_issue: usize,
    pub enable_auto_tagging: bool,
    pub custom_classification_rules: Vec<CustomClassificationRule>,
}

impl IssueCategorizer {
    pub async fn new(config: CategorizationConfig) -> Result<Self, CategorizationError> {
        let classification_rules = Self::load_default_classification_rules();
        let ml_classifier = if config.enable_ml_classification {
            Some(Arc::new(MLClassifier::new().await?))
        } else {
            None
        };
        
        Ok(Self {
            classification_rules,
            ml_classifier,
            domain_knowledge: Arc::new(DomainKnowledge::new()),
            similarity_calculator: Arc::new(SimilarityCalculator::new()),
            config,
        })
    }
    
    pub async fn categorize_issues(&self, issues: Vec<RawIssue>) -> Result<Vec<CategorizedIssue>, CategorizationError> {
        let mut categorized_issues = Vec::new();
        
        for issue in issues {
            let categorized = self.categorize_single_issue(issue).await?;
            categorized_issues.push(categorized);
        }
        
        // Group similar issues if enabled
        if self.config.enable_similarity_grouping {
            categorized_issues = self.group_similar_issues(categorized_issues).await?;
        }
        
        Ok(categorized_issues)
    }
    
    async fn categorize_single_issue(&self, issue: RawIssue) -> Result<CategorizedIssue, CategorizationError> {
        let mut categories = Vec::new();
        let mut tags = Vec::new();
        let mut confidence_scores = HashMap::new();
        
        // Apply rule-based classification
        for rule in &self.classification_rules {
            if rule.matches(&issue) {
                categories.push(rule.category.clone());
                confidence_scores.insert(rule.category.clone(), rule.confidence);
                
                if self.config.enable_auto_tagging {
                    tags.extend(rule.auto_tags.clone());
                }
            }
        }
        
        // Apply ML classification if available
        if let Some(ml_classifier) = &self.ml_classifier {
            let ml_predictions = ml_classifier.classify(&issue).await?;
            for prediction in ml_predictions {
                if prediction.confidence > self.config.confidence_threshold {
                    categories.push(prediction.category);
                    confidence_scores.insert(prediction.category.clone(), prediction.confidence);
                }
            }
        }
        
        // Apply domain knowledge
        let domain_categories = self.domain_knowledge.suggest_categories(&issue).await?;
        categories.extend(domain_categories);
        
        // Remove duplicates and limit categories
        categories.dedup();
        categories.truncate(self.config.max_categories_per_issue);
        
        // Determine primary category
        let primary_category = self.determine_primary_category(&categories, &confidence_scores);
        
        // Extract metadata
        let metadata = self.extract_issue_metadata(&issue).await?;
        
        Ok(CategorizedIssue {
            id: IssueId::new(),
            original_issue: issue,
            primary_category,
            secondary_categories: categories.into_iter().skip(1).collect(),
            tags,
            confidence_scores,
            metadata,
            context_info: self.analyze_context(&issue).await?,
            categorization_timestamp: Utc::now(),
        })
    }
    
    fn load_default_classification_rules() -> Vec<ClassificationRule> {
        vec![
            // Security-related rules
            ClassificationRule {
                id: "security-sql-injection".to_string(),
                category: IssueCategory::Security,
                rule_type: ClassificationRuleType::Pattern,
                pattern: Some("sql.*injection|concatenat.*sql|string.*sql".to_string()),
                conditions: vec![
                    ClassificationCondition::RuleIdContains("sql-injection".to_string()),
                    ClassificationCondition::SeverityAtLeast(IssueSeverity::High),
                ],
                confidence: 0.95,
                auto_tags: vec!["sql-injection".to_string(), "security".to_string()],
            },
            
            ClassificationRule {
                id: "security-hardcoded-secrets".to_string(),
                category: IssueCategory::Security,
                rule_type: ClassificationRuleType::Pattern,
                pattern: Some("hardcoded|secret|password|api.?key".to_string()),
                conditions: vec![
                    ClassificationCondition::RuleIdContains("hardcoded".to_string()),
                ],
                confidence: 0.90,
                auto_tags: vec!["hardcoded-secrets".to_string(), "security".to_string()],
            },
            
            // Performance-related rules
            ClassificationRule {
                id: "performance-complexity".to_string(),
                category: IssueCategory::Performance,
                rule_type: ClassificationRuleType::Metric,
                pattern: None,
                conditions: vec![
                    ClassificationCondition::ComplexityAbove(15),
                    ClassificationCondition::CategoryContains("Performance".to_string()),
                ],
                confidence: 0.85,
                auto_tags: vec!["high-complexity".to_string(), "performance".to_string()],
            },
            
            ClassificationRule {
                id: "performance-memory-leak".to_string(),
                category: IssueCategory::Performance,
                rule_type: ClassificationRuleType::Pattern,
                pattern: Some("memory.*leak|resource.*leak|close.*resource".to_string()),
                conditions: vec![
                    ClassificationCondition::MessageContains("leak".to_string()),
                ],
                confidence: 0.80,
                auto_tags: vec!["memory-leak".to_string(), "performance".to_string()],
            },
            
            // Maintainability rules
            ClassificationRule {
                id: "maintainability-code-smell".to_string(),
                category: IssueCategory::Maintainability,
                rule_type: ClassificationRuleType::Pattern,
                pattern: Some("duplicate|long.?method|large.?class|magic.?number".to_string()),
                conditions: vec![
                    ClassificationCondition::CategoryContains("CodeSmell".to_string()),
                ],
                confidence: 0.75,
                auto_tags: vec!["code-smell".to_string(), "maintainability".to_string()],
            },
            
            // Reliability rules
            ClassificationRule {
                id: "reliability-error-handling".to_string(),
                category: IssueCategory::Reliability,
                rule_type: ClassificationRuleType::Pattern,
                pattern: Some("error.?handling|exception|try.?catch|null.?pointer".to_string()),
                conditions: vec![
                    ClassificationCondition::CategoryContains("ErrorHandling".to_string()),
                ],
                confidence: 0.80,
                auto_tags: vec!["error-handling".to_string(), "reliability".to_string()],
            },
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueCategory {
    Security,
    Performance,
    Maintainability,
    Reliability,
    Compatibility,
    Usability,
    Testability,
    Documentation,
    BestPractices,
    CodeStyle,
    Architecture,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct ClassificationRule {
    pub id: String,
    pub category: IssueCategory,
    pub rule_type: ClassificationRuleType,
    pub pattern: Option<String>,
    pub conditions: Vec<ClassificationCondition>,
    pub confidence: f64,
    pub auto_tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ClassificationRuleType {
    Pattern,
    Metric,
    Context,
    Hybrid,
}

#[derive(Debug, Clone)]
pub enum ClassificationCondition {
    RuleIdContains(String),
    MessageContains(String),
    CategoryContains(String),
    SeverityAtLeast(IssueSeverity),
    ComplexityAbove(u32),
    FileTypeIs(String),
    LanguageIs(ProgrammingLanguage),
    Custom(String),
}

impl ClassificationRule {
    pub fn matches(&self, issue: &RawIssue) -> bool {
        // Check pattern match if present
        if let Some(pattern) = &self.pattern {
            let regex = regex::Regex::new(pattern).unwrap_or_else(|_| regex::Regex::new("").unwrap());
            let text_to_match = format!("{} {} {}", 
                issue.rule_id, 
                issue.message, 
                issue.category.as_ref().unwrap_or(&"".to_string())
            );
            
            if !regex.is_match(&text_to_match.to_lowercase()) {
                return false;
            }
        }
        
        // Check all conditions
        for condition in &self.conditions {
            if !self.check_condition(condition, issue) {
                return false;
            }
        }
        
        true
    }
    
    fn check_condition(&self, condition: &ClassificationCondition, issue: &RawIssue) -> bool {
        match condition {
            ClassificationCondition::RuleIdContains(substring) => {
                issue.rule_id.to_lowercase().contains(&substring.to_lowercase())
            }
            ClassificationCondition::MessageContains(substring) => {
                issue.message.to_lowercase().contains(&substring.to_lowercase())
            }
            ClassificationCondition::CategoryContains(substring) => {
                issue.category.as_ref()
                    .map(|c| c.to_lowercase().contains(&substring.to_lowercase()))
                    .unwrap_or(false)
            }
            ClassificationCondition::SeverityAtLeast(min_severity) => {
                issue.severity >= *min_severity
            }
            ClassificationCondition::ComplexityAbove(threshold) => {
                issue.complexity_metrics.as_ref()
                    .map(|m| m.cyclomatic_complexity > *threshold)
                    .unwrap_or(false)
            }
            ClassificationCondition::FileTypeIs(file_type) => {
                issue.file_path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case(file_type))
                    .unwrap_or(false)
            }
            ClassificationCondition::LanguageIs(language) => {
                issue.language == *language
            }
            ClassificationCondition::Custom(_) => {
                // Custom conditions would be implemented based on specific needs
                false
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CategorizedIssue {
    pub id: IssueId,
    pub original_issue: RawIssue,
    pub primary_category: IssueCategory,
    pub secondary_categories: Vec<IssueCategory>,
    pub tags: Vec<String>,
    pub confidence_scores: HashMap<IssueCategory, f64>,
    pub metadata: IssueMetadata,
    pub context_info: ContextInfo,
    pub categorization_timestamp: DateTime<Utc>,
}
```

### 15.3 Priority Calculation System

#### 15.3.1 Priority Calculator
```rust
pub struct PriorityCalculator {
    impact_analyzer: Arc<ImpactAnalyzer>,
    urgency_calculator: Arc<UrgencyCalculator>,
    business_value_assessor: Arc<BusinessValueAssessor>,
    risk_analyzer: Arc<RiskAnalyzer>,
    config: PriorityConfig,
}

#[derive(Debug, Clone)]
pub struct PriorityConfig {
    pub impact_weight: f64,
    pub urgency_weight: f64,
    pub business_value_weight: f64,
    pub risk_weight: f64,
    pub complexity_penalty_factor: f64,
    pub fix_time_factor: f64,
    pub enable_dynamic_priorities: bool,
    pub priority_decay_factor: f64,
    pub custom_priority_rules: Vec<CustomPriorityRule>,
}

impl PriorityCalculator {
    pub async fn calculate_priorities(&self, issues: &mut [CategorizedIssue]) -> Result<(), PriorityError> {
        for issue in issues.iter_mut() {
            let priority = self.calculate_single_priority(issue).await?;
            issue.metadata.priority = priority;
        }
        
        // Sort by priority (highest first)
        issues.sort_by(|a, b| {
            b.metadata.priority.score.partial_cmp(&a.metadata.priority.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(())
    }
    
    async fn calculate_single_priority(&self, issue: &CategorizedIssue) -> Result<IssuePriority, PriorityError> {
        // Calculate impact score
        let impact = self.impact_analyzer.calculate_impact(issue).await?;
        
        // Calculate urgency score
        let urgency = self.urgency_calculator.calculate_urgency(issue).await?;
        
        // Calculate business value
        let business_value = self.business_value_assessor.assess_business_value(issue).await?;
        
        // Calculate risk score
        let risk = self.risk_analyzer.calculate_risk(issue).await?;
        
        // Apply custom rules
        let custom_adjustments = self.apply_custom_priority_rules(issue).await?;
        
        // Calculate weighted priority score
        let base_score = (
            impact.score * self.config.impact_weight +
            urgency.score * self.config.urgency_weight +
            business_value.score * self.config.business_value_weight +
            risk.score * self.config.risk_weight
        ) / (self.config.impact_weight + self.config.urgency_weight + 
             self.config.business_value_weight + self.config.risk_weight);
        
        // Apply adjustments
        let adjusted_score = base_score + custom_adjustments;
        
        // Apply complexity penalty
        let complexity_penalty = self.calculate_complexity_penalty(issue);
        let final_score = (adjusted_score - complexity_penalty).max(0.0).min(100.0);
        
        // Determine priority level
        let priority_level = self.score_to_priority_level(final_score);
        
        Ok(IssuePriority {
            score: final_score,
            level: priority_level,
            impact,
            urgency,
            business_value,
            risk,
            reasoning: self.generate_priority_reasoning(issue, final_score, impact, urgency, business_value, risk),
            calculated_at: Utc::now(),
        })
    }
    
    fn score_to_priority_level(&self, score: f64) -> PriorityLevel {
        match score {
            s if s >= 80.0 => PriorityLevel::Critical,
            s if s >= 60.0 => PriorityLevel::High,
            s if s >= 40.0 => PriorityLevel::Medium,
            s if s >= 20.0 => PriorityLevel::Low,
            _ => PriorityLevel::Lowest,
        }
    }
    
    fn calculate_complexity_penalty(&self, issue: &CategorizedIssue) -> f64 {
        // Penalize issues that are complex to fix
        let estimated_fix_time = issue.metadata.estimated_fix_time_hours.unwrap_or(1.0);
        let complexity_penalty = (estimated_fix_time - 1.0) * self.config.complexity_penalty_factor;
        complexity_penalty.max(0.0).min(20.0) // Cap penalty at 20 points
    }
}

pub struct ImpactAnalyzer {
    code_impact_calculator: CodeImpactCalculator,
    user_impact_calculator: UserImpactCalculator,
    system_impact_calculator: SystemImpactCalculator,
}

impl ImpactAnalyzer {
    pub async fn calculate_impact(&self, issue: &CategorizedIssue) -> Result<ImpactScore, ImpactError> {
        let code_impact = self.code_impact_calculator.calculate(issue).await?;
        let user_impact = self.user_impact_calculator.calculate(issue).await?;
        let system_impact = self.system_impact_calculator.calculate(issue).await?;
        
        let overall_score = (code_impact + user_impact + system_impact) / 3.0;
        
        Ok(ImpactScore {
            score: overall_score,
            code_impact,
            user_impact,
            system_impact,
            affected_components: self.identify_affected_components(issue).await?,
            blast_radius: self.calculate_blast_radius(issue).await?,
        })
    }
}

pub struct UrgencyCalculator {
    temporal_analyzer: TemporalAnalyzer,
    frequency_analyzer: FrequencyAnalyzer,
    trend_analyzer: TrendAnalyzer,
}

impl UrgencyCalculator {
    pub async fn calculate_urgency(&self, issue: &CategorizedIssue) -> Result<UrgencyScore, UrgencyError> {
        let mut urgency_score = 0.0;
        
        // Security issues are always urgent
        if matches!(issue.primary_category, IssueCategory::Security) {
            urgency_score += 30.0;
        }
        
        // Performance issues affecting critical paths
        if matches!(issue.primary_category, IssueCategory::Performance) {
            if self.affects_critical_path(issue).await? {
                urgency_score += 25.0;
            }
        }
        
        // Issues in frequently changed files are more urgent
        let change_frequency = self.frequency_analyzer.get_file_change_frequency(&issue.original_issue.file_path).await?;
        urgency_score += change_frequency * 10.0;
        
        // Issues in recently modified files
        let recency_score = self.temporal_analyzer.calculate_recency_score(&issue.original_issue.file_path).await?;
        urgency_score += recency_score * 5.0;
        
        // Trending issues (getting worse over time)
        let trend_score = self.trend_analyzer.calculate_trend_score(issue).await?;
        urgency_score += trend_score * 15.0;
        
        Ok(UrgencyScore {
            score: urgency_score.min(100.0),
            change_frequency,
            recency_score,
            trend_score,
            is_critical_path: self.affects_critical_path(issue).await?,
            temporal_factors: self.analyze_temporal_factors(issue).await?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct IssuePriority {
    pub score: f64,
    pub level: PriorityLevel,
    pub impact: ImpactScore,
    pub urgency: UrgencyScore,
    pub business_value: BusinessValueScore,
    pub risk: RiskScore,
    pub reasoning: String,
    pub calculated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PriorityLevel {
    Critical = 5,
    High = 4,
    Medium = 3,
    Low = 2,
    Lowest = 1,
}

#[derive(Debug, Clone)]
pub struct ImpactScore {
    pub score: f64,
    pub code_impact: f64,
    pub user_impact: f64,
    pub system_impact: f64,
    pub affected_components: Vec<String>,
    pub blast_radius: BlastRadius,
}

#[derive(Debug, Clone)]
pub struct UrgencyScore {
    pub score: f64,
    pub change_frequency: f64,
    pub recency_score: f64,
    pub trend_score: f64,
    pub is_critical_path: bool,
    pub temporal_factors: Vec<TemporalFactor>,
}

#[derive(Debug, Clone)]
pub enum BlastRadius {
    File,
    Module,
    Component,
    System,
    Global,
}
```

### 15.4 Issue Clustering and Grouping

#### 15.4.1 Clustering Engine
```rust
pub struct ClusteringEngine {
    similarity_calculator: Arc<SimilarityCalculator>,
    clustering_algorithm: ClusteringAlgorithm,
    feature_extractor: Arc<FeatureExtractor>,
    config: ClusteringConfig,
}

#[derive(Debug, Clone)]
pub struct ClusteringConfig {
    pub similarity_threshold: f64,
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub clustering_method: ClusteringMethod,
    pub feature_weights: FeatureWeights,
    pub enable_hierarchical_clustering: bool,
    pub distance_metric: DistanceMetric,
}

#[derive(Debug, Clone)]
pub enum ClusteringMethod {
    Hierarchical,
    KMeans,
    DBSCAN,
    AffinityPropagation,
}

impl ClusteringEngine {
    pub async fn cluster_issues(&self, issues: &[CategorizedIssue]) -> Result<Vec<IssueCluster>, ClusteringError> {
        // Extract features from issues
        let features = self.extract_features_from_issues(issues).await?;
        
        // Calculate similarity matrix
        let similarity_matrix = self.calculate_similarity_matrix(&features).await?;
        
        // Apply clustering algorithm
        let clusters = match self.config.clustering_method {
            ClusteringMethod::Hierarchical => {
                self.hierarchical_clustering(&similarity_matrix, issues).await?
            }
            ClusteringMethod::DBSCAN => {
                self.dbscan_clustering(&similarity_matrix, issues).await?
            }
            ClusteringMethod::KMeans => {
                self.kmeans_clustering(&features, issues).await?
            }
            ClusteringMethod::AffinityPropagation => {
                self.affinity_propagation_clustering(&similarity_matrix, issues).await?
            }
        };
        
        // Post-process clusters
        let processed_clusters = self.post_process_clusters(clusters).await?;
        
        Ok(processed_clusters)
    }
    
    async fn extract_features_from_issues(&self, issues: &[CategorizedIssue]) -> Result<Vec<IssueFeatureVector>, ClusteringError> {
        let mut features = Vec::new();
        
        for issue in issues {
            let feature_vector = self.feature_extractor.extract_features(issue).await?;
            features.push(feature_vector);
        }
        
        Ok(features)
    }
    
    async fn hierarchical_clustering(&self, similarity_matrix: &SimilarityMatrix, issues: &[CategorizedIssue]) -> Result<Vec<IssueCluster>, ClusteringError> {
        let mut clusters = Vec::new();
        let mut cluster_assignments = vec![None; issues.len()];
        let mut current_cluster_id = 0;
        
        // Build dendrogram using agglomerative clustering
        let mut distance_matrix = self.similarity_to_distance_matrix(similarity_matrix);
        let mut active_clusters: Vec<Vec<usize>> = (0..issues.len()).map(|i| vec![i]).collect();
        
        while active_clusters.len() > 1 {
            // Find closest pair of clusters
            let (cluster1_idx, cluster2_idx, distance) = self.find_closest_clusters(&distance_matrix, &active_clusters)?;
            
            // Check if distance is below threshold
            if distance > (1.0 - self.config.similarity_threshold) {
                break;
            }
            
            // Merge clusters
            let mut merged_cluster = active_clusters[cluster1_idx].clone();
            merged_cluster.extend(active_clusters[cluster2_idx].clone());
            
            // Create new cluster if it meets size requirements
            if merged_cluster.len() >= self.config.min_cluster_size && 
               merged_cluster.len() <= self.config.max_cluster_size {
                
                let cluster_issues: Vec<_> = merged_cluster.iter()
                    .map(|&idx| issues[idx].clone())
                    .collect();
                
                let cluster = IssueCluster {
                    id: ClusterId::new(),
                    cluster_type: ClusterType::Similar,
                    issues: cluster_issues,
                    centroid: self.calculate_cluster_centroid(&merged_cluster, issues).await?,
                    cohesion_score: self.calculate_cohesion_score(&merged_cluster, similarity_matrix),
                    common_characteristics: self.identify_common_characteristics(&merged_cluster, issues).await?,
                    suggested_fix_strategy: self.suggest_cluster_fix_strategy(&merged_cluster, issues).await?,
                    priority_distribution: self.calculate_priority_distribution(&merged_cluster, issues),
                };
                
                clusters.push(cluster);
                
                // Mark issues as clustered
                for &issue_idx in &merged_cluster {
                    cluster_assignments[issue_idx] = Some(current_cluster_id);
                }
                
                current_cluster_id += 1;
            }
            
            // Remove merged clusters and add new one
            let new_cluster = merged_cluster;
            active_clusters.retain(|cluster| 
                cluster != &active_clusters[cluster1_idx] && 
                cluster != &active_clusters[cluster2_idx]
            );
            active_clusters.push(new_cluster);
            
            // Update distance matrix
            distance_matrix = self.update_distance_matrix(distance_matrix, cluster1_idx, cluster2_idx, &active_clusters);
        }
        
        // Handle unclustered issues
        for (idx, assignment) in cluster_assignments.iter().enumerate() {
            if assignment.is_none() {
                // Create singleton cluster for important issues
                if issues[idx].metadata.priority.level >= PriorityLevel::High {
                    let singleton_cluster = IssueCluster {
                        id: ClusterId::new(),
                        cluster_type: ClusterType::Singleton,
                        issues: vec![issues[idx].clone()],
                        centroid: self.calculate_single_issue_centroid(&issues[idx]).await?,
                        cohesion_score: 1.0,
                        common_characteristics: self.extract_issue_characteristics(&issues[idx]).await?,
                        suggested_fix_strategy: FixStrategy::Individual,
                        priority_distribution: PriorityDistribution::single(issues[idx].metadata.priority.level.clone()),
                    };
                    
                    clusters.push(singleton_cluster);
                }
            }
        }
        
        Ok(clusters)
    }
    
    async fn identify_common_characteristics(&self, cluster_indices: &[usize], issues: &[CategorizedIssue]) -> Result<CommonCharacteristics, ClusteringError> {
        let cluster_issues: Vec<_> = cluster_indices.iter()
            .map(|&idx| &issues[idx])
            .collect();
        
        // Find common categories
        let mut category_counts = HashMap::new();
        for issue in &cluster_issues {
            *category_counts.entry(issue.primary_category.clone()).or_insert(0) += 1;
        }
        
        let dominant_category = category_counts.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(category, _)| category.clone());
        
        // Find common tags
        let mut tag_counts = HashMap::new();
        for issue in &cluster_issues {
            for tag in &issue.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }
        
        let common_tags: Vec<_> = tag_counts.iter()
            .filter(|(_, count)| **count > cluster_issues.len() / 2)
            .map(|(tag, _)| tag.clone())
            .collect();
        
        // Find common file patterns
        let file_patterns = self.extract_file_patterns(&cluster_issues);
        
        // Find common rule patterns
        let rule_patterns = self.extract_rule_patterns(&cluster_issues);
        
        Ok(CommonCharacteristics {
            dominant_category,
            common_tags,
            file_patterns,
            rule_patterns,
            severity_distribution: self.calculate_severity_distribution(&cluster_issues),
            language_distribution: self.calculate_language_distribution(&cluster_issues),
        })
    }
}

#[derive(Debug, Clone)]
pub struct IssueCluster {
    pub id: ClusterId,
    pub cluster_type: ClusterType,
    pub issues: Vec<CategorizedIssue>,
    pub centroid: ClusterCentroid,
    pub cohesion_score: f64,
    pub common_characteristics: CommonCharacteristics,
    pub suggested_fix_strategy: FixStrategy,
    pub priority_distribution: PriorityDistribution,
}

#[derive(Debug, Clone)]
pub enum ClusterType {
    Similar,
    Related,
    Singleton,
    Outlier,
}

#[derive(Debug, Clone)]
pub struct CommonCharacteristics {
    pub dominant_category: Option<IssueCategory>,
    pub common_tags: Vec<String>,
    pub file_patterns: Vec<String>,
    pub rule_patterns: Vec<String>,
    pub severity_distribution: HashMap<IssueSeverity, usize>,
    pub language_distribution: HashMap<ProgrammingLanguage, usize>,
}

#[derive(Debug, Clone)]
pub enum FixStrategy {
    Individual,
    Batch,
    Template,
    Refactoring,
    Architectural,
}
```

### 15.5 Remediation Planning System

#### 15.5.1 Remediation Planner
```rust
pub struct RemediationPlanner {
    fix_strategy_generator: Arc<FixStrategyGenerator>,
    resource_estimator: Arc<ResourceEstimator>,
    dependency_analyzer: Arc<DependencyAnalyzer>,
    roi_calculator: Arc<ROICalculator>,
    config: RemediationConfig,
}

#[derive(Debug, Clone)]
pub struct RemediationConfig {
    pub available_developer_hours: f64,
    pub sprint_duration_weeks: u32,
    pub team_velocity_points: u32,
    pub risk_tolerance: RiskTolerance,
    pub prefer_quick_wins: bool,
    pub max_parallel_fixes: usize,
    pub quality_gates: Vec<QualityGate>,
}

impl RemediationPlanner {
    pub async fn create_remediation_plan(&self, clusters: &[IssueCluster]) -> Result<RemediationPlan, RemediationError> {
        // Analyze fix dependencies
        let dependencies = self.dependency_analyzer.analyze_fix_dependencies(clusters).await?;
        
        // Generate fix strategies for each cluster
        let mut fix_plans = Vec::new();
        for cluster in clusters {
            let fix_plan = self.fix_strategy_generator.generate_fix_plan(cluster).await?;
            fix_plans.push(fix_plan);
        }
        
        // Estimate resources needed
        let resource_estimates = self.resource_estimator.estimate_resources(&fix_plans).await?;
        
        // Calculate ROI for each fix
        let roi_analyses = self.roi_calculator.calculate_roi_for_fixes(&fix_plans).await?;
        
        // Optimize fix order based on ROI and dependencies
        let optimized_order = self.optimize_fix_order(&fix_plans, &dependencies, &roi_analyses).await?;
        
        // Create sprint plans
        let sprint_plans = self.create_sprint_plans(&optimized_order, &resource_estimates).await?;
        
        Ok(RemediationPlan {
            id: RemediationPlanId::new(),
            fix_plans,
            dependencies,
            resource_estimates,
            roi_analyses,
            optimized_execution_order: optimized_order,
            sprint_plans,
            total_estimated_effort: resource_estimates.total_hours,
            expected_quality_improvement: self.calculate_expected_quality_improvement(&fix_plans).await?,
            risk_assessment: self.assess_remediation_risks(&fix_plans).await?,
            created_at: Utc::now(),
        })
    }
    
    async fn optimize_fix_order(&self, fix_plans: &[FixPlan], dependencies: &[FixDependency], roi_analyses: &[ROIAnalysis]) -> Result<Vec<FixExecutionStep>, RemediationError> {
        let mut execution_steps = Vec::new();
        let mut completed_fixes = HashSet::new();
        let mut available_fixes: Vec<_> = fix_plans.iter().enumerate().collect();
        
        while !available_fixes.is_empty() {
            // Find fixes with no unmet dependencies
            let mut ready_fixes = Vec::new();
            
            for (idx, fix_plan) in &available_fixes {
                let dependencies_met = dependencies.iter()
                    .filter(|dep| dep.dependent_fix_id == fix_plan.id)
                    .all(|dep| completed_fixes.contains(&dep.prerequisite_fix_id));
                
                if dependencies_met {
                    ready_fixes.push((*idx, fix_plan));
                }
            }
            
            if ready_fixes.is_empty() {
                // Break circular dependencies by selecting highest ROI fix
                if let Some((idx, fix_plan)) = available_fixes.iter()
                    .max_by_key(|(idx, _)| {
                        roi_analyses.get(*idx)
                            .map(|roi| (roi.roi_score * 1000.0) as i64)
                            .unwrap_or(0)
                    }) {
                    ready_fixes.push((*idx, fix_plan));
                }
            }
            
            // Sort ready fixes by ROI and priority
            ready_fixes.sort_by_key(|(idx, fix_plan)| {
                let roi_score = roi_analyses.get(*idx)
                    .map(|roi| (roi.roi_score * 1000.0) as i64)
                    .unwrap_or(0);
                let priority_score = fix_plan.priority_score as i64;
                
                -(roi_score + priority_score) // Negative for descending order
            });
            
            // Select fixes for this execution step (considering parallelism)
            let mut step_fixes = Vec::new();
            let mut step_effort = 0.0;
            
            for (idx, fix_plan) in ready_fixes.iter().take(self.config.max_parallel_fixes) {
                if step_effort + fix_plan.estimated_effort_hours <= self.config.available_developer_hours {
                    step_fixes.push(fix_plan.id.clone());
                    step_effort += fix_plan.estimated_effort_hours;
                    completed_fixes.insert(fix_plan.id.clone());
                }
            }
            
            if !step_fixes.is_empty() {
                execution_steps.push(FixExecutionStep {
                    step_number: execution_steps.len() + 1,
                    fix_ids: step_fixes.clone(),
                    estimated_duration_hours: step_effort,
                    parallel_execution: step_fixes.len() > 1,
                    prerequisites: self.get_prerequisites_for_step(&step_fixes, dependencies),
                    risk_level: self.calculate_step_risk_level(&step_fixes, fix_plans),
                });
                
                // Remove completed fixes from available list
                available_fixes.retain(|(_, fix_plan)| !step_fixes.contains(&fix_plan.id));
            } else {
                // No progress possible - break to avoid infinite loop
                break;
            }
        }
        
        Ok(execution_steps)
    }
    
    async fn create_sprint_plans(&self, execution_order: &[FixExecutionStep], resource_estimates: &ResourceEstimate) -> Result<Vec<SprintPlan>, RemediationError> {
        let mut sprint_plans = Vec::new();
        let mut current_sprint = 1;
        let mut current_sprint_effort = 0.0;
        let mut current_sprint_fixes = Vec::new();
        
        let max_sprint_effort = self.config.available_developer_hours * self.config.sprint_duration_weeks as f64;
        
        for step in execution_order {
            if current_sprint_effort + step.estimated_duration_hours > max_sprint_effort {
                // Create sprint plan for current sprint
                if !current_sprint_fixes.is_empty() {
                    sprint_plans.push(SprintPlan {
                        sprint_number: current_sprint,
                        execution_steps: current_sprint_fixes.clone(),
                        total_effort_hours: current_sprint_effort,
                        expected_completion_date: self.calculate_sprint_completion_date(current_sprint),
                        quality_goals: self.define_sprint_quality_goals(&current_sprint_fixes),
                        risk_mitigation_strategies: self.identify_sprint_risks(&current_sprint_fixes),
                    });
                }
                
                // Start new sprint
                current_sprint += 1;
                current_sprint_effort = 0.0;
                current_sprint_fixes.clear();
            }
            
            current_sprint_fixes.push(step.clone());
            current_sprint_effort += step.estimated_duration_hours;
        }
        
        // Add final sprint if it has content
        if !current_sprint_fixes.is_empty() {
            sprint_plans.push(SprintPlan {
                sprint_number: current_sprint,
                execution_steps: current_sprint_fixes,
                total_effort_hours: current_sprint_effort,
                expected_completion_date: self.calculate_sprint_completion_date(current_sprint),
                quality_goals: self.define_sprint_quality_goals(&current_sprint_fixes),
                risk_mitigation_strategies: self.identify_sprint_risks(&current_sprint_fixes),
            });
        }
        
        Ok(sprint_plans)
    }
}

#[derive(Debug, Clone)]
pub struct RemediationPlan {
    pub id: RemediationPlanId,
    pub fix_plans: Vec<FixPlan>,
    pub dependencies: Vec<FixDependency>,
    pub resource_estimates: ResourceEstimate,
    pub roi_analyses: Vec<ROIAnalysis>,
    pub optimized_execution_order: Vec<FixExecutionStep>,
    pub sprint_plans: Vec<SprintPlan>,
    pub total_estimated_effort: f64,
    pub expected_quality_improvement: QualityImprovement,
    pub risk_assessment: RiskAssessment,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct FixPlan {
    pub id: FixPlanId,
    pub cluster_id: ClusterId,
    pub fix_type: FixType,
    pub description: String,
    pub affected_issues: Vec<IssueId>,
    pub estimated_effort_hours: f64,
    pub confidence_level: f64,
    pub priority_score: f64,
    pub implementation_steps: Vec<ImplementationStep>,
    pub testing_strategy: TestingStrategy,
    pub rollback_plan: RollbackPlan,
}

#[derive(Debug, Clone)]
pub enum FixType {
    CodeChange,
    Refactoring,
    Configuration,
    Documentation,
    Architectural,
    ProcessChange,
}

#[derive(Debug, Clone)]
pub struct SprintPlan {
    pub sprint_number: u32,
    pub execution_steps: Vec<FixExecutionStep>,
    pub total_effort_hours: f64,
    pub expected_completion_date: DateTime<Utc>,
    pub quality_goals: Vec<QualityGoal>,
    pub risk_mitigation_strategies: Vec<RiskMitigationStrategy>,
}
```

### 15.6 Criterios de Completitud

#### 15.6.1 Entregables de la Fase
- [ ] Sistema de categorización automática de issues
- [ ] Calculadora de prioridades inteligente
- [ ] Motor de clustering de issues similares
- [ ] Analizador de impacto y urgencia
- [ ] Planificador de remediación
- [ ] Calculadora de ROI para fixes
- [ ] Generador de planes de sprint
- [ ] Sistema de dependencias entre fixes
- [ ] Estimador de recursos y esfuerzo
- [ ] Tests comprehensivos del sistema

#### 15.6.2 Criterios de Aceptación
- [ ] Categoriza issues con >90% precisión
- [ ] Prioridades reflejan impacto real de negocio
- [ ] Clustering agrupa issues relacionados efectivamente
- [ ] Planes de remediación son factibles y optimizados
- [ ] ROI calculations son realistas y útiles
- [ ] Sprint plans respetan restricciones de recursos
- [ ] Dependencies entre fixes se manejan correctamente
- [ ] Performance acceptable para miles de issues
- [ ] Integration seamless con motor de reglas
- [ ] UI/UX intuitiva para gestión de issues

### 15.7 Performance Targets

#### 15.7.1 Benchmarks del Sistema
- **Categorization speed**: <50ms por issue
- **Priority calculation**: <100ms por issue
- **Clustering**: <5 segundos para 1000 issues
- **Remediation planning**: <30 segundos para proyectos típicos
- **Memory usage**: <200MB para análisis grandes

### 15.8 Estimación de Tiempo

#### 15.8.1 Breakdown de Tareas
- Diseño de arquitectura de categorización: 4 días
- Issue categorizer con ML: 10 días
- Priority calculator: 8 días
- Impact y urgency analyzers: 8 días
- Clustering engine: 10 días
- Remediation planner: 12 días
- ROI calculator: 6 días
- Resource estimator: 6 días
- Sprint planning system: 8 días
- Integration con motor de reglas: 5 días
- Performance optimization: 6 días
- Testing comprehensivo: 10 días
- Documentación: 4 días

**Total estimado: 97 días de desarrollo**

### 15.9 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Sistema completo de gestión inteligente de issues
- Capacidades avanzadas de priorización y planificación
- Optimización de ROI para fixes de código
- Foundation sólida para toma de decisiones
- Completitud del motor de reglas y detección básica

Con la finalización de la Fase 15, se completa el **Motor de Reglas y Detección Básica**, proporcionando una base sólida para las siguientes fases que se enfocarán en **Inteligencia Artificial y Análisis Avanzado** (Fases 16-20).

<function_calls>
<invoke name="todo_write">
<parameter name="merge">true
