# Fase 13: Detección de Código Duplicado y Similitud

## Objetivo General
Implementar un sistema avanzado de detección de código duplicado que identifique duplicación exacta, similitud estructural, similitud semántica, y clones de código a través de múltiples lenguajes, utilizando técnicas de hashing, análisis AST, machine learning, y algoritmos de similitud para proporcionar detección precisa y sugerencias de refactoring.

## Descripción Técnica Detallada

### 13.1 Arquitectura del Sistema de Detección de Duplicación

#### 13.1.1 Diseño del Clone Detection System
```
┌─────────────────────────────────────────┐
│        Clone Detection System          │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Exact     │ │    Structural       │ │
│  │ Duplicate   │ │    Similarity       │ │
│  │ Detector    │ │    Detector         │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │  Semantic   │ │    Cross-Lang       │ │
│  │ Similarity  │ │    Clone            │ │
│  │  Detector   │ │    Detector         │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ Refactoring │ │   Similarity        │ │
│  │ Suggester   │ │   Metrics           │ │
│  └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
```

#### 13.1.2 Tipos de Clones Detectados
- **Type 1 (Exact Clones)**: Código idéntico excepto por espacios en blanco y comentarios
- **Type 2 (Renamed Clones)**: Código idéntico excepto por nombres de variables/funciones
- **Type 3 (Near-miss Clones)**: Código similar con pequeñas diferencias (líneas añadidas/eliminadas)
- **Type 4 (Semantic Clones)**: Código funcionalmente equivalente pero sintácticamente diferente
- **Cross-Language Clones**: Código equivalente en diferentes lenguajes

### 13.2 Clone Detection Engine

#### 13.2.1 Core Clone Detector
```rust
use std::collections::{HashMap, HashSet};
use sha2::{Sha256, Digest};
use similar::{ChangeTag, TextDiff};

pub struct CloneDetector {
    exact_detector: Arc<ExactCloneDetector>,
    structural_detector: Arc<StructuralCloneDetector>,
    semantic_detector: Arc<SemanticCloneDetector>,
    cross_language_detector: Arc<CrossLanguageCloneDetector>,
    similarity_calculator: Arc<SimilarityCalculator>,
    refactoring_suggester: Arc<RefactoringSuggester>,
    config: CloneDetectionConfig,
}

#[derive(Debug, Clone)]
pub struct CloneDetectionConfig {
    pub min_clone_size: usize,
    pub min_similarity_threshold: f64,
    pub enable_exact_detection: bool,
    pub enable_structural_detection: bool,
    pub enable_semantic_detection: bool,
    pub enable_cross_language_detection: bool,
    pub ignore_whitespace: bool,
    pub ignore_comments: bool,
    pub ignore_variable_names: bool,
    pub max_gap_size: usize,
    pub similarity_algorithms: Vec<SimilarityAlgorithm>,
    pub language_specific_configs: HashMap<ProgrammingLanguage, LanguageCloneConfig>,
}

#[derive(Debug, Clone)]
pub struct LanguageCloneConfig {
    pub min_token_count: usize,
    pub ignore_imports: bool,
    pub ignore_boilerplate: bool,
    pub custom_normalizations: Vec<NormalizationRule>,
    pub framework_specific_rules: Vec<FrameworkRule>,
}

impl CloneDetector {
    pub async fn new(config: CloneDetectionConfig) -> Result<Self, CloneDetectionError> {
        Ok(Self {
            exact_detector: Arc::new(ExactCloneDetector::new()),
            structural_detector: Arc::new(StructuralCloneDetector::new()),
            semantic_detector: Arc::new(SemanticCloneDetector::new()),
            cross_language_detector: Arc::new(CrossLanguageCloneDetector::new()),
            similarity_calculator: Arc::new(SimilarityCalculator::new()),
            refactoring_suggester: Arc::new(RefactoringSuggester::new()),
            config,
        })
    }
    
    pub async fn detect_clones(&self, unified_ast: &UnifiedAST) -> Result<CloneAnalysis, CloneDetectionError> {
        let start_time = Instant::now();
        
        let mut analysis = CloneAnalysis {
            file_path: unified_ast.file_path.clone(),
            language: unified_ast.language,
            exact_clones: Vec::new(),
            structural_clones: Vec::new(),
            semantic_clones: Vec::new(),
            cross_language_clones: Vec::new(),
            clone_classes: Vec::new(),
            duplication_metrics: DuplicationMetrics::default(),
            refactoring_opportunities: Vec::new(),
            execution_time_ms: 0,
        };
        
        // Detect exact clones
        if self.config.enable_exact_detection {
            analysis.exact_clones = self.exact_detector.detect_exact_clones(unified_ast).await?;
        }
        
        // Detect structural clones
        if self.config.enable_structural_detection {
            analysis.structural_clones = self.structural_detector.detect_structural_clones(unified_ast).await?;
        }
        
        // Detect semantic clones
        if self.config.enable_semantic_detection {
            analysis.semantic_clones = self.semantic_detector.detect_semantic_clones(unified_ast).await?;
        }
        
        // Group clones into clone classes
        analysis.clone_classes = self.group_clones_into_classes(&analysis).await?;
        
        // Calculate duplication metrics
        analysis.duplication_metrics = self.calculate_duplication_metrics(&analysis, unified_ast);
        
        // Generate refactoring suggestions
        analysis.refactoring_opportunities = self.refactoring_suggester.suggest_refactorings(&analysis).await?;
        
        analysis.execution_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(analysis)
    }
    
    pub async fn detect_clones_project(&self, project_asts: &[UnifiedAST]) -> Result<ProjectCloneAnalysis, CloneDetectionError> {
        let start_time = Instant::now();
        
        // Detect clones within each file
        let mut file_analyses = Vec::new();
        for ast in project_asts {
            let file_analysis = self.detect_clones(ast).await?;
            file_analyses.push(file_analysis);
        }
        
        // Detect inter-file clones
        let inter_file_clones = self.detect_inter_file_clones(project_asts).await?;
        
        // Detect cross-language clones
        let cross_language_clones = if self.config.enable_cross_language_detection {
            self.cross_language_detector.detect_cross_language_clones(project_asts).await?
        } else {
            Vec::new()
        };
        
        // Build global clone classes
        let global_clone_classes = self.build_global_clone_classes(&file_analyses, &inter_file_clones).await?;
        
        // Calculate project-wide metrics
        let project_metrics = self.calculate_project_duplication_metrics(&file_analyses, &inter_file_clones);
        
        // Generate project-wide refactoring opportunities
        let project_refactoring_opportunities = self.generate_project_refactoring_opportunities(&global_clone_classes).await?;
        
        Ok(ProjectCloneAnalysis {
            project_path: project_asts.first().map(|ast| ast.file_path.parent().unwrap_or(&ast.file_path).to_path_buf()),
            file_analyses,
            inter_file_clones,
            cross_language_clones,
            global_clone_classes,
            project_metrics,
            project_refactoring_opportunities,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }
    
    async fn group_clones_into_classes(&self, analysis: &CloneAnalysis) -> Result<Vec<CloneClass>, CloneDetectionError> {
        let mut clone_classes = Vec::new();
        let mut processed_clones = HashSet::new();
        
        // Group exact clones
        for exact_clone in &analysis.exact_clones {
            if processed_clones.contains(&exact_clone.id) {
                continue;
            }
            
            let similar_clones = self.find_similar_clones(exact_clone, &analysis.exact_clones).await?;
            
            if similar_clones.len() > 1 {
                clone_classes.push(CloneClass {
                    id: CloneClassId::new(),
                    clone_type: CloneType::Exact,
                    instances: similar_clones.iter().map(|c| c.clone()).collect(),
                    similarity_score: 1.0,
                    size_metrics: self.calculate_clone_class_metrics(&similar_clones),
                    refactoring_potential: self.assess_refactoring_potential(&similar_clones),
                });
                
                for clone in &similar_clones {
                    processed_clones.insert(clone.id.clone());
                }
            }
        }
        
        // Group structural clones
        for structural_clone in &analysis.structural_clones {
            if processed_clones.contains(&structural_clone.id) {
                continue;
            }
            
            let similar_clones = self.find_similar_structural_clones(structural_clone, &analysis.structural_clones).await?;
            
            if similar_clones.len() > 1 {
                let avg_similarity = similar_clones.iter()
                    .map(|c| c.similarity_score)
                    .sum::<f64>() / similar_clones.len() as f64;
                
                clone_classes.push(CloneClass {
                    id: CloneClassId::new(),
                    clone_type: CloneType::Structural,
                    instances: similar_clones.iter().map(|c| c.clone()).collect(),
                    similarity_score: avg_similarity,
                    size_metrics: self.calculate_clone_class_metrics(&similar_clones),
                    refactoring_potential: self.assess_refactoring_potential(&similar_clones),
                });
                
                for clone in &similar_clones {
                    processed_clones.insert(clone.id.clone());
                }
            }
        }
        
        Ok(clone_classes)
    }
}

#[derive(Debug, Clone)]
pub struct CloneAnalysis {
    pub file_path: PathBuf,
    pub language: ProgrammingLanguage,
    pub exact_clones: Vec<ExactClone>,
    pub structural_clones: Vec<StructuralClone>,
    pub semantic_clones: Vec<SemanticClone>,
    pub cross_language_clones: Vec<CrossLanguageClone>,
    pub clone_classes: Vec<CloneClass>,
    pub duplication_metrics: DuplicationMetrics,
    pub refactoring_opportunities: Vec<RefactoringOpportunity>,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct CloneClass {
    pub id: CloneClassId,
    pub clone_type: CloneType,
    pub instances: Vec<Clone>,
    pub similarity_score: f64,
    pub size_metrics: CloneClassMetrics,
    pub refactoring_potential: RefactoringPotential,
}

#[derive(Debug, Clone)]
pub enum CloneType {
    Exact,
    Structural,
    Semantic,
    CrossLanguage,
}
```

### 13.3 Exact Clone Detection

#### 13.3.1 Exact Clone Detector
```rust
pub struct ExactCloneDetector {
    hasher: CodeHasher,
    normalizer: CodeNormalizer,
}

impl ExactCloneDetector {
    pub async fn detect_exact_clones(&self, ast: &UnifiedAST) -> Result<Vec<ExactClone>, ExactCloneError> {
        let mut exact_clones = Vec::new();
        
        // Extract code blocks of minimum size
        let code_blocks = self.extract_code_blocks(ast, MIN_CLONE_SIZE).await?;
        
        // Normalize and hash each block
        let mut hash_to_blocks: HashMap<String, Vec<CodeBlock>> = HashMap::new();
        
        for block in code_blocks {
            let normalized_code = self.normalizer.normalize(&block.content, ast.language)?;
            let hash = self.hasher.hash_code(&normalized_code);
            
            hash_to_blocks.entry(hash).or_default().push(block);
        }
        
        // Find blocks with identical hashes
        for (hash, blocks) in hash_to_blocks {
            if blocks.len() > 1 {
                // Create clone instances for each duplicate
                for i in 0..blocks.len() {
                    for j in (i + 1)..blocks.len() {
                        let clone = ExactClone {
                            id: CloneId::new(),
                            clone_type: CloneType::Exact,
                            original_location: blocks[i].location.clone(),
                            duplicate_location: blocks[j].location.clone(),
                            content: blocks[i].content.clone(),
                            size_lines: blocks[i].size_lines,
                            size_tokens: blocks[i].size_tokens,
                            hash: hash.clone(),
                            similarity_score: 1.0,
                            confidence: 1.0,
                        };
                        
                        exact_clones.push(clone);
                    }
                }
            }
        }
        
        Ok(exact_clones)
    }
    
    async fn extract_code_blocks(&self, ast: &UnifiedAST, min_size: usize) -> Result<Vec<CodeBlock>, ExactCloneError> {
        let mut blocks = Vec::new();
        let mut visitor = CodeBlockVisitor::new(&mut blocks, min_size);
        visitor.visit_node(&ast.root_node);
        Ok(blocks)
    }
}

#[derive(Debug, Clone)]
pub struct ExactClone {
    pub id: CloneId,
    pub clone_type: CloneType,
    pub original_location: CodeLocation,
    pub duplicate_location: CodeLocation,
    pub content: String,
    pub size_lines: usize,
    pub size_tokens: usize,
    pub hash: String,
    pub similarity_score: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct CodeLocation {
    pub file_path: PathBuf,
    pub start_line: u32,
    pub end_line: u32,
    pub start_column: u32,
    pub end_column: u32,
    pub function_context: Option<String>,
    pub class_context: Option<String>,
}

pub struct CodeHasher {
    algorithm: HashAlgorithm,
}

#[derive(Debug, Clone)]
pub enum HashAlgorithm {
    SHA256,
    MD5,
    XXHash,
    SimHash,
}

impl CodeHasher {
    pub fn hash_code(&self, code: &str) -> String {
        match self.algorithm {
            HashAlgorithm::SHA256 => {
                let mut hasher = Sha256::new();
                hasher.update(code.as_bytes());
                format!("{:x}", hasher.finalize())
            }
            HashAlgorithm::XXHash => {
                // Implementation for XXHash
                self.xxhash_code(code)
            }
            HashAlgorithm::SimHash => {
                // Implementation for SimHash (for near-duplicate detection)
                self.simhash_code(code)
            }
            _ => {
                // Default to SHA256
                let mut hasher = Sha256::new();
                hasher.update(code.as_bytes());
                format!("{:x}", hasher.finalize())
            }
        }
    }
    
    fn simhash_code(&self, code: &str) -> String {
        // SimHash implementation for fuzzy matching
        let tokens = self.tokenize_code(code);
        let mut feature_vector = vec![0i32; 64];
        
        for token in tokens {
            let token_hash = self.hash_token(&token);
            for i in 0..64 {
                if (token_hash >> i) & 1 == 1 {
                    feature_vector[i] += 1;
                } else {
                    feature_vector[i] -= 1;
                }
            }
        }
        
        let mut simhash = 0u64;
        for i in 0..64 {
            if feature_vector[i] > 0 {
                simhash |= 1u64 << i;
            }
        }
        
        format!("{:016x}", simhash)
    }
}

pub struct CodeNormalizer {
    normalization_rules: Vec<NormalizationRule>,
}

impl CodeNormalizer {
    pub fn normalize(&self, code: &str, language: ProgrammingLanguage) -> Result<String, NormalizationError> {
        let mut normalized = code.to_string();
        
        // Apply language-agnostic normalizations
        normalized = self.remove_whitespace(&normalized);
        normalized = self.remove_comments(&normalized, language);
        normalized = self.normalize_string_literals(&normalized);
        normalized = self.normalize_numeric_literals(&normalized);
        
        // Apply language-specific normalizations
        normalized = self.apply_language_specific_normalizations(normalized, language)?;
        
        Ok(normalized)
    }
    
    fn remove_whitespace(&self, code: &str) -> String {
        // Remove extra whitespace but preserve structure
        code.lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
    
    fn remove_comments(&self, code: &str, language: ProgrammingLanguage) -> String {
        match language {
            ProgrammingLanguage::Python => self.remove_python_comments(code),
            ProgrammingLanguage::JavaScript | ProgrammingLanguage::TypeScript => self.remove_js_comments(code),
            ProgrammingLanguage::Rust => self.remove_rust_comments(code),
            _ => code.to_string(),
        }
    }
    
    fn normalize_string_literals(&self, code: &str) -> String {
        // Replace string literals with placeholders
        let string_regex = regex::Regex::new(r#""[^"]*"|'[^']*'"#).unwrap();
        string_regex.replace_all(code, "STRING_LITERAL").to_string()
    }
    
    fn normalize_numeric_literals(&self, code: &str) -> String {
        // Replace numeric literals with placeholders
        let number_regex = regex::Regex::new(r"\b\d+\.?\d*\b").unwrap();
        number_regex.replace_all(code, "NUMERIC_LITERAL").to_string()
    }
}
```

### 13.4 Structural Clone Detection

#### 13.4.1 Structural Clone Detector
```rust
pub struct StructuralCloneDetector {
    ast_comparer: ASTComparer,
    tree_matcher: TreeMatcher,
    similarity_calculator: SimilarityCalculator,
}

impl StructuralCloneDetector {
    pub async fn detect_structural_clones(&self, ast: &UnifiedAST) -> Result<Vec<StructuralClone>, StructuralCloneError> {
        let mut structural_clones = Vec::new();
        
        // Extract AST subtrees of minimum size
        let subtrees = self.extract_subtrees(ast, MIN_TREE_SIZE).await?;
        
        // Compare each pair of subtrees
        for i in 0..subtrees.len() {
            for j in (i + 1)..subtrees.len() {
                let similarity = self.ast_comparer.compare_subtrees(&subtrees[i], &subtrees[j]).await?;
                
                if similarity.structural_similarity > MIN_STRUCTURAL_SIMILARITY {
                    let structural_clone = StructuralClone {
                        id: CloneId::new(),
                        clone_type: CloneType::Structural,
                        original_subtree: subtrees[i].clone(),
                        duplicate_subtree: subtrees[j].clone(),
                        similarity_score: similarity.structural_similarity,
                        differences: similarity.differences,
                        mapping: similarity.node_mapping,
                        confidence: self.calculate_structural_confidence(&similarity),
                        refactoring_potential: self.assess_structural_refactoring(&similarity),
                    };
                    
                    structural_clones.push(structural_clone);
                }
            }
        }
        
        Ok(structural_clones)
    }
    
    async fn extract_subtrees(&self, ast: &UnifiedAST, min_size: usize) -> Result<Vec<ASTSubtree>, StructuralCloneError> {
        let mut subtrees = Vec::new();
        let mut visitor = SubtreeVisitor::new(&mut subtrees, min_size);
        visitor.visit_node(&ast.root_node);
        Ok(subtrees)
    }
}

pub struct ASTComparer {
    node_comparer: NodeComparer,
    structure_comparer: StructureComparer,
}

impl ASTComparer {
    pub async fn compare_subtrees(&self, subtree1: &ASTSubtree, subtree2: &ASTSubtree) -> Result<StructuralSimilarity, ComparisonError> {
        // Compare node types and structure
        let node_similarity = self.node_comparer.compare_nodes(&subtree1.root, &subtree2.root).await?;
        let structure_similarity = self.structure_comparer.compare_structures(subtree1, subtree2).await?;
        
        // Calculate overall similarity
        let overall_similarity = (node_similarity.similarity + structure_similarity.similarity) / 2.0;
        
        // Find node mappings
        let node_mapping = self.find_node_mapping(subtree1, subtree2).await?;
        
        // Identify differences
        let differences = self.identify_differences(subtree1, subtree2, &node_mapping).await?;
        
        Ok(StructuralSimilarity {
            structural_similarity: overall_similarity,
            node_similarity: node_similarity.similarity,
            structure_similarity: structure_similarity.similarity,
            differences,
            node_mapping,
        })
    }
    
    async fn find_node_mapping(&self, subtree1: &ASTSubtree, subtree2: &ASTSubtree) -> Result<NodeMapping, ComparisonError> {
        // Use tree edit distance algorithm to find optimal mapping
        let mut mapping = NodeMapping::new();
        
        // Implement tree edit distance with node mapping
        let edit_operations = self.calculate_tree_edit_distance(&subtree1.root, &subtree2.root).await?;
        
        for operation in edit_operations {
            match operation {
                EditOperation::Match(node1_id, node2_id) => {
                    mapping.add_mapping(node1_id, node2_id);
                }
                EditOperation::Substitute(node1_id, node2_id) => {
                    mapping.add_mapping(node1_id, node2_id);
                }
                _ => {
                    // Insert or Delete operations don't create mappings
                }
            }
        }
        
        Ok(mapping)
    }
}

#[derive(Debug, Clone)]
pub struct StructuralClone {
    pub id: CloneId,
    pub clone_type: CloneType,
    pub original_subtree: ASTSubtree,
    pub duplicate_subtree: ASTSubtree,
    pub similarity_score: f64,
    pub differences: Vec<StructuralDifference>,
    pub mapping: NodeMapping,
    pub confidence: f64,
    pub refactoring_potential: RefactoringPotential,
}

#[derive(Debug, Clone)]
pub struct StructuralSimilarity {
    pub structural_similarity: f64,
    pub node_similarity: f64,
    pub structure_similarity: f64,
    pub differences: Vec<StructuralDifference>,
    pub node_mapping: NodeMapping,
}

#[derive(Debug, Clone)]
pub enum StructuralDifference {
    NodeTypeChange {
        original: UnifiedNodeType,
        duplicate: UnifiedNodeType,
        location: UnifiedPosition,
    },
    NodeInserted {
        node_type: UnifiedNodeType,
        location: UnifiedPosition,
    },
    NodeDeleted {
        node_type: UnifiedNodeType,
        location: UnifiedPosition,
    },
    AttributeChange {
        attribute_name: String,
        original_value: String,
        duplicate_value: String,
        location: UnifiedPosition,
    },
    StructureChange {
        change_type: String,
        description: String,
        location: UnifiedPosition,
    },
}

#[derive(Debug, Clone)]
pub enum EditOperation {
    Match(NodeId, NodeId),
    Substitute(NodeId, NodeId),
    Insert(NodeId),
    Delete(NodeId),
}
```

### 13.5 Semantic Clone Detection

#### 13.5.1 Semantic Clone Detector
```rust
pub struct SemanticCloneDetector {
    semantic_analyzer: SemanticAnalyzer,
    behavior_analyzer: BehaviorAnalyzer,
    data_flow_comparer: DataFlowComparer,
    ml_model: Option<Arc<SemanticSimilarityModel>>,
}

impl SemanticCloneDetector {
    pub async fn detect_semantic_clones(&self, ast: &UnifiedAST) -> Result<Vec<SemanticClone>, SemanticCloneError> {
        let mut semantic_clones = Vec::new();
        
        // Extract semantic units (functions, methods, etc.)
        let semantic_units = self.extract_semantic_units(ast).await?;
        
        // Compare each pair of semantic units
        for i in 0..semantic_units.len() {
            for j in (i + 1)..semantic_units.len() {
                let similarity = self.compare_semantic_units(&semantic_units[i], &semantic_units[j]).await?;
                
                if similarity.overall_similarity > MIN_SEMANTIC_SIMILARITY {
                    let semantic_clone = SemanticClone {
                        id: CloneId::new(),
                        clone_type: CloneType::Semantic,
                        original_unit: semantic_units[i].clone(),
                        duplicate_unit: semantic_units[j].clone(),
                        similarity_score: similarity.overall_similarity,
                        semantic_similarity: similarity.semantic_similarity,
                        behavioral_similarity: similarity.behavioral_similarity,
                        data_flow_similarity: similarity.data_flow_similarity,
                        evidence: similarity.evidence,
                        confidence: similarity.confidence,
                    };
                    
                    semantic_clones.push(semantic_clone);
                }
            }
        }
        
        Ok(semantic_clones)
    }
    
    async fn compare_semantic_units(&self, unit1: &SemanticUnit, unit2: &SemanticUnit) -> Result<SemanticSimilarity, SemanticCloneError> {
        // Semantic analysis
        let semantic_sim = self.semantic_analyzer.compare_semantics(unit1, unit2).await?;
        
        // Behavioral analysis
        let behavioral_sim = self.behavior_analyzer.compare_behavior(unit1, unit2).await?;
        
        // Data flow analysis
        let data_flow_sim = self.data_flow_comparer.compare_data_flows(unit1, unit2).await?;
        
        // ML-based similarity (if model is available)
        let ml_similarity = if let Some(model) = &self.ml_model {
            model.calculate_similarity(unit1, unit2).await?
        } else {
            0.0
        };
        
        // Combine similarities with weights
        let overall_similarity = (
            semantic_sim * 0.4 +
            behavioral_sim * 0.3 +
            data_flow_sim * 0.2 +
            ml_similarity * 0.1
        );
        
        // Collect evidence
        let evidence = self.collect_similarity_evidence(unit1, unit2, &semantic_sim, &behavioral_sim).await?;
        
        // Calculate confidence based on evidence strength
        let confidence = self.calculate_semantic_confidence(&evidence);
        
        Ok(SemanticSimilarity {
            overall_similarity,
            semantic_similarity: semantic_sim,
            behavioral_similarity: behavioral_sim,
            data_flow_similarity: data_flow_sim,
            ml_similarity,
            evidence,
            confidence,
        })
    }
    
    async fn extract_semantic_units(&self, ast: &UnifiedAST) -> Result<Vec<SemanticUnit>, SemanticCloneError> {
        let mut units = Vec::new();
        
        // Extract functions
        let functions = self.extract_functions(&ast.root_node);
        for func in functions {
            let semantic_info = self.semantic_analyzer.analyze_function(&func, ast).await?;
            units.push(SemanticUnit {
                id: SemanticUnitId::new(),
                unit_type: SemanticUnitType::Function,
                ast_node: func.clone(),
                semantic_info,
                behavior_signature: self.behavior_analyzer.create_signature(&func, ast).await?,
                data_flow_graph: self.data_flow_comparer.build_data_flow_graph(&func, ast).await?,
            });
        }
        
        // Extract classes/structs
        let classes = self.extract_classes(&ast.root_node);
        for class in classes {
            let semantic_info = self.semantic_analyzer.analyze_class(&class, ast).await?;
            units.push(SemanticUnit {
                id: SemanticUnitId::new(),
                unit_type: SemanticUnitType::Class,
                ast_node: class.clone(),
                semantic_info,
                behavior_signature: self.behavior_analyzer.create_signature(&class, ast).await?,
                data_flow_graph: self.data_flow_comparer.build_data_flow_graph(&class, ast).await?,
            });
        }
        
        Ok(units)
    }
}

#[derive(Debug, Clone)]
pub struct SemanticClone {
    pub id: CloneId,
    pub clone_type: CloneType,
    pub original_unit: SemanticUnit,
    pub duplicate_unit: SemanticUnit,
    pub similarity_score: f64,
    pub semantic_similarity: f64,
    pub behavioral_similarity: f64,
    pub data_flow_similarity: f64,
    pub evidence: Vec<SimilarityEvidence>,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct SemanticUnit {
    pub id: SemanticUnitId,
    pub unit_type: SemanticUnitType,
    pub ast_node: UnifiedNode,
    pub semantic_info: SemanticInfo,
    pub behavior_signature: BehaviorSignature,
    pub data_flow_graph: DataFlowGraph,
}

#[derive(Debug, Clone)]
pub enum SemanticUnitType {
    Function,
    Method,
    Class,
    Module,
    CodeBlock,
}

#[derive(Debug, Clone)]
pub struct SemanticSimilarity {
    pub overall_similarity: f64,
    pub semantic_similarity: f64,
    pub behavioral_similarity: f64,
    pub data_flow_similarity: f64,
    pub ml_similarity: f64,
    pub evidence: Vec<SimilarityEvidence>,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum SimilarityEvidence {
    SameInputOutputTypes,
    SimilarControlFlow,
    SimilarDataFlow,
    SameAlgorithmicComplexity,
    SimilarVariableUsage,
    SameExternalDependencies,
    SimilarErrorHandling,
    EquivalentMathOperations,
}
```

### 13.6 Cross-Language Clone Detection

#### 13.6.1 Cross-Language Clone Detector
```rust
pub struct CrossLanguageCloneDetector {
    concept_mapper: ConceptMapper,
    semantic_translator: SemanticTranslator,
    pattern_matcher: CrossLanguagePatternMatcher,
    similarity_calculator: CrossLanguageSimilarityCalculator,
}

impl CrossLanguageCloneDetector {
    pub async fn detect_cross_language_clones(&self, project_asts: &[UnifiedAST]) -> Result<Vec<CrossLanguageClone>, CrossLanguageCloneError> {
        let mut cross_language_clones = Vec::new();
        
        // Group ASTs by language
        let mut language_groups: HashMap<ProgrammingLanguage, Vec<&UnifiedAST>> = HashMap::new();
        for ast in project_asts {
            language_groups.entry(ast.language).or_default().push(ast);
        }
        
        // Compare between different language groups
        let languages: Vec<_> = language_groups.keys().cloned().collect();
        
        for i in 0..languages.len() {
            for j in (i + 1)..languages.len() {
                let lang1 = languages[i];
                let lang2 = languages[j];
                
                let clones = self.detect_clones_between_languages(
                    &language_groups[&lang1],
                    &language_groups[&lang2],
                    lang1,
                    lang2,
                ).await?;
                
                cross_language_clones.extend(clones);
            }
        }
        
        Ok(cross_language_clones)
    }
    
    async fn detect_clones_between_languages(
        &self,
        asts1: &[&UnifiedAST],
        asts2: &[&UnifiedAST],
        lang1: ProgrammingLanguage,
        lang2: ProgrammingLanguage,
    ) -> Result<Vec<CrossLanguageClone>, CrossLanguageCloneError> {
        let mut clones = Vec::new();
        
        // Extract semantic units from both language groups
        let units1 = self.extract_cross_language_units(asts1, lang1).await?;
        let units2 = self.extract_cross_language_units(asts2, lang2).await?;
        
        // Compare units across languages
        for unit1 in &units1 {
            for unit2 in &units2 {
                if self.are_comparable_units(unit1, unit2) {
                    let similarity = self.compare_cross_language_units(unit1, unit2).await?;
                    
                    if similarity.overall_similarity > MIN_CROSS_LANGUAGE_SIMILARITY {
                        clones.push(CrossLanguageClone {
                            id: CloneId::new(),
                            clone_type: CloneType::CrossLanguage,
                            unit1: unit1.clone(),
                            unit2: unit2.clone(),
                            language1: lang1,
                            language2: lang2,
                            similarity_score: similarity.overall_similarity,
                            concept_mapping: similarity.concept_mapping,
                            translation_evidence: similarity.translation_evidence,
                            confidence: similarity.confidence,
                        });
                    }
                }
            }
        }
        
        Ok(clones)
    }
    
    async fn compare_cross_language_units(&self, unit1: &CrossLanguageUnit, unit2: &CrossLanguageUnit) -> Result<CrossLanguageSimilarity, CrossLanguageCloneError> {
        // Map concepts between languages
        let concept_mapping = self.concept_mapper.map_concepts(unit1, unit2).await?;
        
        // Translate semantic representations
        let translated_unit1 = self.semantic_translator.translate_to_common_representation(unit1).await?;
        let translated_unit2 = self.semantic_translator.translate_to_common_representation(unit2).await?;
        
        // Compare translated representations
        let semantic_similarity = self.similarity_calculator.compare_semantic_representations(&translated_unit1, &translated_unit2).await?;
        
        // Look for common patterns
        let pattern_similarity = self.pattern_matcher.find_common_patterns(unit1, unit2).await?;
        
        // Combine similarities
        let overall_similarity = (semantic_similarity + pattern_similarity) / 2.0;
        
        // Collect translation evidence
        let translation_evidence = self.collect_translation_evidence(unit1, unit2, &concept_mapping).await?;
        
        Ok(CrossLanguageSimilarity {
            overall_similarity,
            semantic_similarity,
            pattern_similarity,
            concept_mapping,
            translation_evidence,
            confidence: self.calculate_cross_language_confidence(&translation_evidence),
        })
    }
    
    fn are_comparable_units(&self, unit1: &CrossLanguageUnit, unit2: &CrossLanguageUnit) -> bool {
        // Check if units are of comparable types and complexity
        unit1.unit_type.is_comparable_to(&unit2.unit_type) &&
        self.complexity_difference_acceptable(&unit1.complexity_metrics, &unit2.complexity_metrics)
    }
}

#[derive(Debug, Clone)]
pub struct CrossLanguageClone {
    pub id: CloneId,
    pub clone_type: CloneType,
    pub unit1: CrossLanguageUnit,
    pub unit2: CrossLanguageUnit,
    pub language1: ProgrammingLanguage,
    pub language2: ProgrammingLanguage,
    pub similarity_score: f64,
    pub concept_mapping: ConceptMapping,
    pub translation_evidence: Vec<TranslationEvidence>,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct CrossLanguageUnit {
    pub id: CrossLanguageUnitId,
    pub unit_type: CrossLanguageUnitType,
    pub language: ProgrammingLanguage,
    pub ast_node: UnifiedNode,
    pub semantic_representation: SemanticRepresentation,
    pub complexity_metrics: ComplexityMetrics,
    pub behavioral_signature: BehaviorSignature,
}

#[derive(Debug, Clone)]
pub enum CrossLanguageUnitType {
    Algorithm,
    DataStructure,
    DesignPattern,
    BusinessLogic,
    UtilityFunction,
}

impl CrossLanguageUnitType {
    pub fn is_comparable_to(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (Self::Algorithm, Self::Algorithm) |
            (Self::DataStructure, Self::DataStructure) |
            (Self::DesignPattern, Self::DesignPattern) |
            (Self::BusinessLogic, Self::BusinessLogic) |
            (Self::UtilityFunction, Self::UtilityFunction)
        )
    }
}
```

### 13.7 Refactoring Suggestions

#### 13.7.1 Refactoring Suggester
```rust
pub struct RefactoringSuggester {
    pattern_analyzer: PatternAnalyzer,
    complexity_analyzer: ComplexityAnalyzer,
    dependency_analyzer: DependencyAnalyzer,
}

impl RefactoringSuggester {
    pub async fn suggest_refactorings(&self, clone_analysis: &CloneAnalysis) -> Result<Vec<RefactoringOpportunity>, RefactoringError> {
        let mut opportunities = Vec::new();
        
        // Analyze clone classes for refactoring potential
        for clone_class in &clone_analysis.clone_classes {
            let refactoring_suggestions = self.analyze_clone_class_for_refactoring(clone_class).await?;
            opportunities.extend(refactoring_suggestions);
        }
        
        // Look for extract method opportunities
        opportunities.extend(self.suggest_extract_method_refactorings(clone_analysis).await?);
        
        // Look for extract class opportunities
        opportunities.extend(self.suggest_extract_class_refactorings(clone_analysis).await?);
        
        // Look for template method pattern opportunities
        opportunities.extend(self.suggest_template_method_refactorings(clone_analysis).await?);
        
        // Look for strategy pattern opportunities
        opportunities.extend(self.suggest_strategy_pattern_refactorings(clone_analysis).await?);
        
        Ok(opportunities)
    }
    
    async fn analyze_clone_class_for_refactoring(&self, clone_class: &CloneClass) -> Result<Vec<RefactoringOpportunity>, RefactoringError> {
        let mut opportunities = Vec::new();
        
        match clone_class.clone_type {
            CloneType::Exact => {
                // Suggest extract method for exact clones
                if clone_class.instances.len() >= 2 {
                    opportunities.push(RefactoringOpportunity {
                        id: RefactoringOpportunityId::new(),
                        refactoring_type: RefactoringType::ExtractMethod,
                        description: "Extract duplicated code into a common method".to_string(),
                        affected_clones: clone_class.instances.clone(),
                        estimated_effort: self.estimate_extract_method_effort(clone_class),
                        potential_benefits: self.calculate_extract_method_benefits(clone_class),
                        implementation_steps: self.generate_extract_method_steps(clone_class),
                        confidence: 0.95,
                    });
                }
            }
            CloneType::Structural => {
                // Analyze structural differences to suggest appropriate refactoring
                let differences = self.analyze_structural_differences(clone_class).await?;
                
                if differences.are_parameter_variations() {
                    opportunities.push(RefactoringOpportunity {
                        id: RefactoringOpportunityId::new(),
                        refactoring_type: RefactoringType::ParameterizeMethod,
                        description: "Parameterize method to handle variations".to_string(),
                        affected_clones: clone_class.instances.clone(),
                        estimated_effort: EstimatedEffort::Medium,
                        potential_benefits: vec![
                            RefactoringBenefit::ReducedDuplication,
                            RefactoringBenefit::ImprovedMaintainability,
                        ],
                        implementation_steps: self.generate_parameterize_method_steps(clone_class, &differences),
                        confidence: 0.8,
                    });
                }
                
                if differences.suggest_template_method() {
                    opportunities.push(RefactoringOpportunity {
                        id: RefactoringOpportunityId::new(),
                        refactoring_type: RefactoringType::TemplateMethod,
                        description: "Apply template method pattern for common structure".to_string(),
                        affected_clones: clone_class.instances.clone(),
                        estimated_effort: EstimatedEffort::High,
                        potential_benefits: vec![
                            RefactoringBenefit::ReducedDuplication,
                            RefactoringBenefit::ImprovedDesign,
                            RefactoringBenefit::BetterExtensibility,
                        ],
                        implementation_steps: self.generate_template_method_steps(clone_class, &differences),
                        confidence: 0.7,
                    });
                }
            }
            CloneType::Semantic => {
                // Suggest higher-level design patterns for semantic clones
                opportunities.push(RefactoringOpportunity {
                    id: RefactoringOpportunityId::new(),
                    refactoring_type: RefactoringType::StrategyPattern,
                    description: "Consider strategy pattern for similar algorithms".to_string(),
                    affected_clones: clone_class.instances.clone(),
                    estimated_effort: EstimatedEffort::High,
                    potential_benefits: vec![
                        RefactoringBenefit::ImprovedDesign,
                        RefactoringBenefit::BetterTestability,
                        RefactoringBenefit::ReducedComplexity,
                    ],
                    implementation_steps: self.generate_strategy_pattern_steps(clone_class),
                    confidence: 0.6,
                });
            }
            CloneType::CrossLanguage => {
                // Suggest creating common interfaces or libraries
                opportunities.push(RefactoringOpportunity {
                    id: RefactoringOpportunityId::new(),
                    refactoring_type: RefactoringType::ExtractLibrary,
                    description: "Consider extracting common functionality into a shared library".to_string(),
                    affected_clones: clone_class.instances.clone(),
                    estimated_effort: EstimatedEffort::VeryHigh,
                    potential_benefits: vec![
                        RefactoringBenefit::ReducedDuplication,
                        RefactoringBenefit::ConsistentBehavior,
                        RefactoringBenefit::CentralizedMaintenance,
                    ],
                    implementation_steps: self.generate_extract_library_steps(clone_class),
                    confidence: 0.5,
                });
            }
        }
        
        Ok(opportunities)
    }
}

#[derive(Debug, Clone)]
pub struct RefactoringOpportunity {
    pub id: RefactoringOpportunityId,
    pub refactoring_type: RefactoringType,
    pub description: String,
    pub affected_clones: Vec<Clone>,
    pub estimated_effort: EstimatedEffort,
    pub potential_benefits: Vec<RefactoringBenefit>,
    pub implementation_steps: Vec<RefactoringStep>,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum RefactoringType {
    ExtractMethod,
    ExtractClass,
    ParameterizeMethod,
    TemplateMethod,
    StrategyPattern,
    ExtractLibrary,
    MergeClasses,
    ReplaceConditionalWithPolymorphism,
}

#[derive(Debug, Clone)]
pub enum EstimatedEffort {
    Low,       // < 1 hour
    Medium,    // 1-4 hours
    High,      // 4-16 hours
    VeryHigh,  // > 16 hours
}

#[derive(Debug, Clone)]
pub enum RefactoringBenefit {
    ReducedDuplication,
    ImprovedMaintainability,
    ImprovedDesign,
    BetterTestability,
    ReducedComplexity,
    BetterExtensibility,
    ConsistentBehavior,
    CentralizedMaintenance,
}

#[derive(Debug, Clone)]
pub struct RefactoringStep {
    pub step_number: u32,
    pub description: String,
    pub code_changes: Vec<CodeChange>,
    pub validation_steps: Vec<String>,
}
```

### 13.8 Criterios de Completitud

#### 13.8.1 Entregables de la Fase
- [ ] Sistema de detección de código duplicado implementado
- [ ] Detector de clones exactos funcionando
- [ ] Detector de clones estructurales
- [ ] Detector de clones semánticos
- [ ] Detector de clones cross-language
- [ ] Sistema de métricas de duplicación
- [ ] Generador de sugerencias de refactoring
- [ ] Calculador de similitud multi-algoritmo
- [ ] Performance optimizado para proyectos grandes
- [ ] Tests comprehensivos

#### 13.8.2 Criterios de Aceptación
- [ ] Detecta clones exactos con 100% precisión
- [ ] Identifica clones estructurales con >90% precisión
- [ ] Encuentra clones semánticos con >80% precisión
- [ ] Detecta clones cross-language con >70% precisión
- [ ] False positives < 10% en código típico
- [ ] Performance acceptable para proyectos de 500k+ LOC
- [ ] Sugerencias de refactoring son útiles y factibles
- [ ] Métricas de duplicación son precisas
- [ ] Integration seamless con motor de reglas
- [ ] Soporte robusto para múltiples lenguajes

### 13.9 Performance Targets

#### 13.9.1 Benchmarks de Detección de Duplicación
- **Exact clone detection**: <100ms para archivos de 5k LOC
- **Structural clone detection**: <500ms para archivos típicos
- **Semantic clone detection**: <2 segundos para archivos complejos
- **Cross-language detection**: <30 segundos para proyectos medianos
- **Memory usage**: <500MB para proyectos grandes

### 13.10 Estimación de Tiempo

#### 13.10.1 Breakdown de Tareas
- Diseño de arquitectura de detección: 5 días
- Exact clone detector: 6 días
- Structural clone detector: 10 días
- Semantic clone detector: 12 días
- Cross-language clone detector: 15 días
- Sistema de métricas de duplicación: 4 días
- Refactoring suggester: 8 días
- Similarity calculator: 6 días
- Performance optimization: 8 días
- Integration con motor de reglas: 4 días
- Testing comprehensivo: 10 días
- Documentación: 4 días

**Total estimado: 92 días de desarrollo**

### 13.11 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Capacidades avanzadas de detección de duplicación
- Análisis de similitud multi-dimensional
- Sugerencias inteligentes de refactoring
- Detección cross-language única en la industria
- Base sólida para análisis de complejidad

La Fase 14 construirá sobre esta base implementando el análisis de complejidad ciclomática y métricas básicas.
