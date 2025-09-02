# Fase 14: Análisis de Complejidad Ciclomática y Métricas Básicas

## Objetivo General
Implementar un sistema completo de análisis de complejidad y cálculo de métricas de código que proporcione mediciones precisas de complejidad ciclomática, complejidad cognitiva, métricas de Halstead, métricas de cohesión y acoplamiento, y otras métricas de calidad de código esenciales para evaluar la mantenibilidad, legibilidad y calidad general del código.

## Descripción Técnica Detallada

### 14.1 Arquitectura del Sistema de Métricas

#### 14.1.1 Diseño del Code Metrics System
```
┌─────────────────────────────────────────┐
│          Code Metrics System           │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ Complexity  │ │    Halstead         │ │
│  │ Analyzers   │ │    Metrics          │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ Cohesion &  │ │    Size &           │ │
│  │ Coupling    │ │    Volume           │ │
│  │ Analyzers   │ │    Metrics          │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   Quality   │ │   Maintainability   │ │
│  │  Metrics    │ │     Index           │ │
│  └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
```

#### 14.1.2 Tipos de Métricas Implementadas
- **Complejidad Ciclomática**: McCabe, modificada, esencial
- **Complejidad Cognitiva**: Basada en esfuerzo mental para entender código
- **Métricas de Halstead**: Volumen, dificultad, esfuerzo, tiempo, bugs predichos
- **Métricas de Tamaño**: LOC, SLOC, comentarios, líneas en blanco
- **Métricas de Cohesión**: LCOM, TCC, LCC
- **Métricas de Acoplamiento**: CBO, RFC, DIT, NOC
- **Métricas de Calidad**: Índice de mantenibilidad, deuda técnica

### 14.2 Core Metrics Engine

#### 14.2.1 Metrics Calculator Implementation
```rust
use std::collections::{HashMap, HashSet};
use petgraph::{Graph, Direction};

pub struct MetricsCalculator {
    complexity_analyzer: Arc<ComplexityAnalyzer>,
    halstead_calculator: Arc<HalsteadCalculator>,
    cohesion_analyzer: Arc<CohesionAnalyzer>,
    coupling_analyzer: Arc<CouplingAnalyzer>,
    size_analyzer: Arc<SizeAnalyzer>,
    quality_analyzer: Arc<QualityAnalyzer>,
    config: MetricsConfig,
}

#[derive(Debug, Clone)]
pub struct MetricsConfig {
    pub enable_cyclomatic_complexity: bool,
    pub enable_cognitive_complexity: bool,
    pub enable_halstead_metrics: bool,
    pub enable_cohesion_metrics: bool,
    pub enable_coupling_metrics: bool,
    pub enable_size_metrics: bool,
    pub enable_quality_metrics: bool,
    pub complexity_thresholds: ComplexityThresholds,
    pub language_specific_configs: HashMap<ProgrammingLanguage, LanguageMetricsConfig>,
}

#[derive(Debug, Clone)]
pub struct ComplexityThresholds {
    pub cyclomatic_low: u32,
    pub cyclomatic_medium: u32,
    pub cyclomatic_high: u32,
    pub cognitive_low: u32,
    pub cognitive_medium: u32,
    pub cognitive_high: u32,
    pub halstead_volume_threshold: f64,
    pub maintainability_low: f64,
    pub maintainability_medium: f64,
}

impl MetricsCalculator {
    pub async fn new(config: MetricsConfig) -> Result<Self, MetricsError> {
        Ok(Self {
            complexity_analyzer: Arc::new(ComplexityAnalyzer::new()),
            halstead_calculator: Arc::new(HalsteadCalculator::new()),
            cohesion_analyzer: Arc::new(CohesionAnalyzer::new()),
            coupling_analyzer: Arc::new(CouplingAnalyzer::new()),
            size_analyzer: Arc::new(SizeAnalyzer::new()),
            quality_analyzer: Arc::new(QualityAnalyzer::new()),
            config,
        })
    }
    
    pub async fn calculate_metrics(&self, unified_ast: &UnifiedAST) -> Result<CodeMetrics, MetricsError> {
        let start_time = Instant::now();
        
        let mut metrics = CodeMetrics {
            file_path: unified_ast.file_path.clone(),
            language: unified_ast.language,
            complexity_metrics: ComplexityMetrics::default(),
            halstead_metrics: HalsteadMetrics::default(),
            size_metrics: SizeMetrics::default(),
            cohesion_metrics: CohesionMetrics::default(),
            coupling_metrics: CouplingMetrics::default(),
            quality_metrics: QualityMetrics::default(),
            function_metrics: Vec::new(),
            class_metrics: Vec::new(),
            overall_quality_score: 0.0,
            calculation_time_ms: 0,
        };
        
        // Calculate complexity metrics
        if self.config.enable_cyclomatic_complexity || self.config.enable_cognitive_complexity {
            metrics.complexity_metrics = self.complexity_analyzer.calculate_complexity(unified_ast).await?;
        }
        
        // Calculate Halstead metrics
        if self.config.enable_halstead_metrics {
            metrics.halstead_metrics = self.halstead_calculator.calculate_halstead_metrics(unified_ast).await?;
        }
        
        // Calculate size metrics
        if self.config.enable_size_metrics {
            metrics.size_metrics = self.size_analyzer.calculate_size_metrics(unified_ast).await?;
        }
        
        // Calculate cohesion metrics
        if self.config.enable_cohesion_metrics {
            metrics.cohesion_metrics = self.cohesion_analyzer.calculate_cohesion_metrics(unified_ast).await?;
        }
        
        // Calculate coupling metrics
        if self.config.enable_coupling_metrics {
            metrics.coupling_metrics = self.coupling_analyzer.calculate_coupling_metrics(unified_ast).await?;
        }
        
        // Calculate function-level metrics
        metrics.function_metrics = self.calculate_function_metrics(unified_ast).await?;
        
        // Calculate class-level metrics
        metrics.class_metrics = self.calculate_class_metrics(unified_ast).await?;
        
        // Calculate quality metrics and overall score
        if self.config.enable_quality_metrics {
            metrics.quality_metrics = self.quality_analyzer.calculate_quality_metrics(&metrics).await?;
            metrics.overall_quality_score = self.calculate_overall_quality_score(&metrics);
        }
        
        metrics.calculation_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(metrics)
    }
    
    pub async fn calculate_project_metrics(&self, project_asts: &[UnifiedAST]) -> Result<ProjectMetrics, MetricsError> {
        let start_time = Instant::now();
        
        // Calculate metrics for each file
        let mut file_metrics = Vec::new();
        for ast in project_asts {
            let metrics = self.calculate_metrics(ast).await?;
            file_metrics.push(metrics);
        }
        
        // Aggregate project-level metrics
        let project_metrics = ProjectMetrics {
            project_path: project_asts.first()
                .map(|ast| ast.file_path.parent().unwrap_or(&ast.file_path).to_path_buf()),
            file_metrics,
            aggregated_complexity: self.aggregate_complexity_metrics(project_asts).await?,
            aggregated_halstead: self.aggregate_halstead_metrics(project_asts).await?,
            aggregated_size: self.aggregate_size_metrics(project_asts).await?,
            project_cohesion: self.calculate_project_cohesion(project_asts).await?,
            project_coupling: self.calculate_project_coupling(project_asts).await?,
            hotspots: self.identify_complexity_hotspots(project_asts).await?,
            quality_distribution: self.calculate_quality_distribution(project_asts).await?,
            technical_debt_estimate: self.estimate_technical_debt(project_asts).await?,
            maintainability_index: self.calculate_project_maintainability_index(project_asts).await?,
            calculation_time_ms: start_time.elapsed().as_millis() as u64,
        };
        
        Ok(project_metrics)
    }
    
    async fn calculate_function_metrics(&self, ast: &UnifiedAST) -> Result<Vec<FunctionMetrics>, MetricsError> {
        let mut function_metrics = Vec::new();
        
        // Extract all functions from AST
        let functions = self.extract_functions(&ast.root_node);
        
        for function in functions {
            let metrics = FunctionMetrics {
                name: function.name.clone(),
                location: function.location.clone(),
                cyclomatic_complexity: self.complexity_analyzer.calculate_function_cyclomatic_complexity(&function).await?,
                cognitive_complexity: self.complexity_analyzer.calculate_function_cognitive_complexity(&function).await?,
                lines_of_code: self.size_analyzer.count_function_lines(&function),
                parameter_count: function.parameters.len(),
                local_variable_count: self.count_local_variables(&function),
                return_points: self.count_return_points(&function),
                nested_depth: self.calculate_nested_depth(&function),
                halstead_metrics: self.halstead_calculator.calculate_function_halstead(&function).await?,
                maintainability_index: self.calculate_function_maintainability(&function).await?,
                complexity_rating: self.rate_function_complexity(&function).await?,
            };
            
            function_metrics.push(metrics);
        }
        
        Ok(function_metrics)
    }
    
    async fn calculate_class_metrics(&self, ast: &UnifiedAST) -> Result<Vec<ClassMetrics>, MetricsError> {
        let mut class_metrics = Vec::new();
        
        // Extract all classes from AST
        let classes = self.extract_classes(&ast.root_node);
        
        for class in classes {
            let metrics = ClassMetrics {
                name: class.name.clone(),
                location: class.location.clone(),
                lines_of_code: self.size_analyzer.count_class_lines(&class),
                method_count: class.methods.len(),
                attribute_count: class.attributes.len(),
                public_method_count: class.methods.iter().filter(|m| m.is_public).count(),
                private_method_count: class.methods.iter().filter(|m| !m.is_public).count(),
                inheritance_depth: self.calculate_inheritance_depth(&class),
                coupling_between_objects: self.coupling_analyzer.calculate_cbo(&class).await?,
                response_for_class: self.coupling_analyzer.calculate_rfc(&class).await?,
                lack_of_cohesion: self.cohesion_analyzer.calculate_lcom(&class).await?,
                weighted_methods_per_class: self.calculate_wmc(&class).await?,
                complexity_rating: self.rate_class_complexity(&class).await?,
            };
            
            class_metrics.push(metrics);
        }
        
        Ok(class_metrics)
    }
}

#[derive(Debug, Clone, Default)]
pub struct CodeMetrics {
    pub file_path: PathBuf,
    pub language: ProgrammingLanguage,
    pub complexity_metrics: ComplexityMetrics,
    pub halstead_metrics: HalsteadMetrics,
    pub size_metrics: SizeMetrics,
    pub cohesion_metrics: CohesionMetrics,
    pub coupling_metrics: CouplingMetrics,
    pub quality_metrics: QualityMetrics,
    pub function_metrics: Vec<FunctionMetrics>,
    pub class_metrics: Vec<ClassMetrics>,
    pub overall_quality_score: f64,
    pub calculation_time_ms: u64,
}
```

### 14.3 Complexity Analysis System

#### 14.3.1 Complexity Analyzer
```rust
pub struct ComplexityAnalyzer {
    cyclomatic_calculator: CyclomaticComplexityCalculator,
    cognitive_calculator: CognitiveComplexityCalculator,
    essential_calculator: EssentialComplexityCalculator,
}

impl ComplexityAnalyzer {
    pub async fn calculate_complexity(&self, ast: &UnifiedAST) -> Result<ComplexityMetrics, ComplexityError> {
        let mut complexity_metrics = ComplexityMetrics::default();
        
        // Calculate cyclomatic complexity
        complexity_metrics.cyclomatic_complexity = self.cyclomatic_calculator.calculate(ast).await?;
        complexity_metrics.modified_cyclomatic_complexity = self.cyclomatic_calculator.calculate_modified(ast).await?;
        
        // Calculate cognitive complexity
        complexity_metrics.cognitive_complexity = self.cognitive_calculator.calculate(ast).await?;
        
        // Calculate essential complexity
        complexity_metrics.essential_complexity = self.essential_calculator.calculate(ast).await?;
        
        // Calculate per-function complexity distribution
        complexity_metrics.function_complexity_distribution = self.calculate_function_complexity_distribution(ast).await?;
        
        // Calculate nesting depth metrics
        complexity_metrics.max_nesting_depth = self.calculate_max_nesting_depth(&ast.root_node);
        complexity_metrics.average_nesting_depth = self.calculate_average_nesting_depth(&ast.root_node);
        
        Ok(complexity_metrics)
    }
}

pub struct CyclomaticComplexityCalculator;

impl CyclomaticComplexityCalculator {
    pub async fn calculate(&self, ast: &UnifiedAST) -> Result<u32, ComplexityError> {
        let mut complexity = 1; // Base complexity
        let mut visitor = CyclomaticComplexityVisitor::new(&mut complexity);
        visitor.visit_node(&ast.root_node);
        Ok(complexity)
    }
    
    pub async fn calculate_modified(&self, ast: &UnifiedAST) -> Result<u32, ComplexityError> {
        // Modified cyclomatic complexity considers switch statements differently
        let mut complexity = 1;
        let mut visitor = ModifiedCyclomaticComplexityVisitor::new(&mut complexity);
        visitor.visit_node(&ast.root_node);
        Ok(complexity)
    }
    
    pub async fn calculate_function_cyclomatic_complexity(&self, function: &FunctionNode) -> Result<u32, ComplexityError> {
        let mut complexity = 1; // Base complexity for function
        let mut visitor = CyclomaticComplexityVisitor::new(&mut complexity);
        visitor.visit_node(&function.body);
        Ok(complexity)
    }
}

struct CyclomaticComplexityVisitor<'a> {
    complexity: &'a mut u32,
}

impl<'a> CyclomaticComplexityVisitor<'a> {
    fn new(complexity: &'a mut u32) -> Self {
        Self { complexity }
    }
    
    fn visit_node(&mut self, node: &UnifiedNode) {
        match &node.node_type {
            // Decision points that increase cyclomatic complexity
            UnifiedNodeType::IfStatement => {
                *self.complexity += 1;
            }
            UnifiedNodeType::ForStatement => {
                *self.complexity += 1;
            }
            UnifiedNodeType::WhileStatement => {
                *self.complexity += 1;
            }
            UnifiedNodeType::MatchStatement => {
                // Each case in match adds complexity
                let case_count = self.count_match_cases(node);
                *self.complexity += case_count.max(1);
            }
            UnifiedNodeType::ConditionalExpression => {
                *self.complexity += 1;
            }
            UnifiedNodeType::TryStatement => {
                // Each catch block adds complexity
                let catch_count = self.count_catch_blocks(node);
                *self.complexity += catch_count.max(1);
            }
            UnifiedNodeType::BinaryExpression { operator } => {
                if matches!(operator, BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr) {
                    *self.complexity += 1;
                }
            }
            _ => {}
        }
        
        // Recursively visit children
        for child in &node.children {
            self.visit_node(child);
        }
    }
    
    fn count_match_cases(&self, node: &UnifiedNode) -> u32 {
        // Count the number of case/when branches in match/switch statement
        node.children.iter()
            .filter(|child| matches!(child.node_type, UnifiedNodeType::MatchCase))
            .count() as u32
    }
    
    fn count_catch_blocks(&self, node: &UnifiedNode) -> u32 {
        // Count the number of catch/except blocks
        node.children.iter()
            .filter(|child| matches!(child.node_type, UnifiedNodeType::CatchBlock))
            .count() as u32
    }
}

pub struct CognitiveComplexityCalculator {
    nesting_increment: u32,
    binary_sequence_increment: u32,
}

impl CognitiveComplexityCalculator {
    pub fn new() -> Self {
        Self {
            nesting_increment: 1,
            binary_sequence_increment: 1,
        }
    }
    
    pub async fn calculate(&self, ast: &UnifiedAST) -> Result<u32, ComplexityError> {
        let mut complexity = 0;
        let mut visitor = CognitiveComplexityVisitor::new(&mut complexity);
        visitor.visit_node(&ast.root_node, 0);
        Ok(complexity)
    }
    
    pub async fn calculate_function_cognitive_complexity(&self, function: &FunctionNode) -> Result<u32, ComplexityError> {
        let mut complexity = 0;
        let mut visitor = CognitiveComplexityVisitor::new(&mut complexity);
        visitor.visit_node(&function.body, 0);
        Ok(complexity)
    }
}

struct CognitiveComplexityVisitor<'a> {
    complexity: &'a mut u32,
}

impl<'a> CognitiveComplexityVisitor<'a> {
    fn new(complexity: &'a mut u32) -> Self {
        Self { complexity }
    }
    
    fn visit_node(&mut self, node: &UnifiedNode, nesting_level: u32) {
        let (increment, new_nesting_level) = match &node.node_type {
            // Control flow structures
            UnifiedNodeType::IfStatement => (1 + nesting_level, nesting_level + 1),
            UnifiedNodeType::ForStatement => (1 + nesting_level, nesting_level + 1),
            UnifiedNodeType::WhileStatement => (1 + nesting_level, nesting_level + 1),
            UnifiedNodeType::MatchStatement => (1 + nesting_level, nesting_level + 1),
            
            // Exception handling
            UnifiedNodeType::TryStatement => (0, nesting_level + 1), // Try itself doesn't add complexity
            UnifiedNodeType::CatchBlock => (1 + nesting_level, nesting_level),
            
            // Logical operators in sequences
            UnifiedNodeType::BinaryExpression { operator } => {
                if matches!(operator, BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr) {
                    if self.is_in_binary_sequence(node) {
                        (1, nesting_level) // Only first in sequence counts
                    } else {
                        (0, nesting_level)
                    }
                } else {
                    (0, nesting_level)
                }
            }
            
            // Recursion adds complexity
            UnifiedNodeType::CallExpression => {
                if self.is_recursive_call(node) {
                    (1 + nesting_level, nesting_level)
                } else {
                    (0, nesting_level)
                }
            }
            
            _ => (0, nesting_level),
        };
        
        *self.complexity += increment;
        
        // Visit children with updated nesting level
        for child in &node.children {
            self.visit_node(child, new_nesting_level);
        }
    }
    
    fn is_in_binary_sequence(&self, node: &UnifiedNode) -> bool {
        // Check if this logical operator is the first in a sequence
        // This is a simplified implementation
        true // TODO: Implement proper sequence detection
    }
    
    fn is_recursive_call(&self, node: &UnifiedNode) -> bool {
        // Check if this is a recursive function call
        // This would require context about the current function
        false // TODO: Implement recursive call detection
    }
}

#[derive(Debug, Clone, Default)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: u32,
    pub modified_cyclomatic_complexity: u32,
    pub cognitive_complexity: u32,
    pub essential_complexity: u32,
    pub max_nesting_depth: u32,
    pub average_nesting_depth: f64,
    pub function_complexity_distribution: ComplexityDistribution,
    pub complexity_density: f64, // Complexity per line of code
}

#[derive(Debug, Clone, Default)]
pub struct ComplexityDistribution {
    pub low_complexity_functions: u32,
    pub medium_complexity_functions: u32,
    pub high_complexity_functions: u32,
    pub very_high_complexity_functions: u32,
    pub complexity_histogram: Vec<u32>,
}
```

### 14.4 Halstead Metrics System

#### 14.4.1 Halstead Calculator
```rust
pub struct HalsteadCalculator {
    operator_extractor: OperatorExtractor,
    operand_extractor: OperandExtractor,
}

impl HalsteadCalculator {
    pub async fn calculate_halstead_metrics(&self, ast: &UnifiedAST) -> Result<HalsteadMetrics, HalsteadError> {
        // Extract operators and operands
        let operators = self.operator_extractor.extract_operators(&ast.root_node);
        let operands = self.operand_extractor.extract_operands(&ast.root_node);
        
        // Calculate basic counts
        let n1 = operators.distinct_count(); // Number of distinct operators
        let n2 = operands.distinct_count();  // Number of distinct operands
        let N1 = operators.total_count();    // Total number of operators
        let N2 = operands.total_count();     // Total number of operands
        
        // Calculate derived metrics
        let vocabulary = n1 + n2;
        let length = N1 + N2;
        let calculated_length = n1 as f64 * (n1 as f64).log2() + n2 as f64 * (n2 as f64).log2();
        let volume = length as f64 * (vocabulary as f64).log2();
        let difficulty = (n1 as f64 / 2.0) * (N2 as f64 / n2 as f64);
        let effort = difficulty * volume;
        let time = effort / 18.0; // Stroud number
        let bugs = volume / 3000.0; // Empirical formula
        
        Ok(HalsteadMetrics {
            distinct_operators: n1,
            distinct_operands: n2,
            total_operators: N1,
            total_operands: N2,
            vocabulary,
            length,
            calculated_length,
            volume,
            difficulty,
            effort,
            time,
            bugs,
            level: 1.0 / difficulty,
            intelligence: volume / difficulty,
        })
    }
    
    pub async fn calculate_function_halstead(&self, function: &FunctionNode) -> Result<HalsteadMetrics, HalsteadError> {
        // Calculate Halstead metrics for a specific function
        let operators = self.operator_extractor.extract_operators(&function.body);
        let operands = self.operand_extractor.extract_operands(&function.body);
        
        // Same calculation as above but scoped to function
        self.calculate_from_counts(operators, operands)
    }
}

pub struct OperatorExtractor {
    language_operators: HashMap<ProgrammingLanguage, Vec<String>>,
}

impl OperatorExtractor {
    pub fn extract_operators(&self, node: &UnifiedNode) -> OperatorCollection {
        let mut operators = OperatorCollection::new();
        self.extract_operators_recursive(node, &mut operators);
        operators
    }
    
    fn extract_operators_recursive(&self, node: &UnifiedNode, operators: &mut OperatorCollection) {
        match &node.node_type {
            UnifiedNodeType::BinaryExpression { operator } => {
                operators.add_operator(operator.to_string());
            }
            UnifiedNodeType::UnaryExpression { operator } => {
                operators.add_operator(operator.to_string());
            }
            UnifiedNodeType::AssignmentExpression => {
                operators.add_operator("=".to_string());
            }
            UnifiedNodeType::CallExpression => {
                operators.add_operator("()".to_string());
            }
            UnifiedNodeType::MemberExpression => {
                operators.add_operator(".".to_string());
            }
            UnifiedNodeType::ArrayExpression => {
                operators.add_operator("[]".to_string());
            }
            UnifiedNodeType::IfStatement => {
                operators.add_operator("if".to_string());
            }
            UnifiedNodeType::ForStatement => {
                operators.add_operator("for".to_string());
            }
            UnifiedNodeType::WhileStatement => {
                operators.add_operator("while".to_string());
            }
            UnifiedNodeType::ReturnStatement => {
                operators.add_operator("return".to_string());
            }
            _ => {}
        }
        
        // Recursively process children
        for child in &node.children {
            self.extract_operators_recursive(child, operators);
        }
    }
}

pub struct OperandExtractor;

impl OperandExtractor {
    pub fn extract_operands(&self, node: &UnifiedNode) -> OperandCollection {
        let mut operands = OperandCollection::new();
        self.extract_operands_recursive(node, &mut operands);
        operands
    }
    
    fn extract_operands_recursive(&self, node: &UnifiedNode, operands: &mut OperandCollection) {
        match &node.node_type {
            UnifiedNodeType::Identifier => {
                if let Some(name) = &node.name {
                    operands.add_operand(name.clone());
                }
            }
            UnifiedNodeType::StringLiteral => {
                if let Some(value) = &node.value {
                    operands.add_operand(value.raw_value.clone());
                }
            }
            UnifiedNodeType::NumberLiteral => {
                if let Some(value) = &node.value {
                    operands.add_operand(value.raw_value.clone());
                }
            }
            UnifiedNodeType::BooleanLiteral => {
                if let Some(value) = &node.value {
                    operands.add_operand(value.raw_value.clone());
                }
            }
            _ => {}
        }
        
        // Recursively process children
        for child in &node.children {
            self.extract_operands_recursive(child, operands);
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HalsteadMetrics {
    pub distinct_operators: u32,      // n1
    pub distinct_operands: u32,       // n2
    pub total_operators: u32,         // N1
    pub total_operands: u32,          // N2
    pub vocabulary: u32,              // n = n1 + n2
    pub length: u32,                  // N = N1 + N2
    pub calculated_length: f64,       // N̂ = n1*log2(n1) + n2*log2(n2)
    pub volume: f64,                  // V = N * log2(n)
    pub difficulty: f64,              // D = (n1/2) * (N2/n2)
    pub effort: f64,                  // E = D * V
    pub time: f64,                    // T = E / 18
    pub bugs: f64,                    // B = V / 3000
    pub level: f64,                   // L = 1 / D
    pub intelligence: f64,            // I = V / D
}

#[derive(Debug)]
pub struct OperatorCollection {
    operators: HashMap<String, u32>,
}

impl OperatorCollection {
    pub fn new() -> Self {
        Self {
            operators: HashMap::new(),
        }
    }
    
    pub fn add_operator(&mut self, operator: String) {
        *self.operators.entry(operator).or_insert(0) += 1;
    }
    
    pub fn distinct_count(&self) -> u32 {
        self.operators.len() as u32
    }
    
    pub fn total_count(&self) -> u32 {
        self.operators.values().sum()
    }
}

#[derive(Debug)]
pub struct OperandCollection {
    operands: HashMap<String, u32>,
}

impl OperandCollection {
    pub fn new() -> Self {
        Self {
            operands: HashMap::new(),
        }
    }
    
    pub fn add_operand(&mut self, operand: String) {
        *self.operands.entry(operand).or_insert(0) += 1;
    }
    
    pub fn distinct_count(&self) -> u32 {
        self.operands.len() as u32
    }
    
    pub fn total_count(&self) -> u32 {
        self.operands.values().sum()
    }
}
```

### 14.5 Cohesion and Coupling Analysis

#### 14.5.1 Cohesion Analyzer
```rust
pub struct CohesionAnalyzer {
    method_analyzer: MethodAnalyzer,
    attribute_analyzer: AttributeAnalyzer,
}

impl CohesionAnalyzer {
    pub async fn calculate_cohesion_metrics(&self, ast: &UnifiedAST) -> Result<CohesionMetrics, CohesionError> {
        let classes = self.extract_classes(&ast.root_node);
        let mut cohesion_metrics = CohesionMetrics::default();
        
        if !classes.is_empty() {
            let mut total_lcom = 0.0;
            let mut total_tcc = 0.0;
            let mut total_lcc = 0.0;
            
            for class in &classes {
                let lcom = self.calculate_lcom(class).await?;
                let tcc = self.calculate_tcc(class).await?;
                let lcc = self.calculate_lcc(class).await?;
                
                total_lcom += lcom;
                total_tcc += tcc;
                total_lcc += lcc;
            }
            
            cohesion_metrics.average_lcom = total_lcom / classes.len() as f64;
            cohesion_metrics.average_tcc = total_tcc / classes.len() as f64;
            cohesion_metrics.average_lcc = total_lcc / classes.len() as f64;
            cohesion_metrics.class_count = classes.len();
        }
        
        Ok(cohesion_metrics)
    }
    
    pub async fn calculate_lcom(&self, class: &ClassNode) -> Result<f64, CohesionError> {
        // LCOM (Lack of Cohesion of Methods)
        // Measures how well the methods of a class are related to each other
        
        let methods = &class.methods;
        let attributes = &class.attributes;
        
        if methods.is_empty() || attributes.is_empty() {
            return Ok(0.0);
        }
        
        // Create attribute usage matrix
        let mut method_attribute_usage = Vec::new();
        
        for method in methods {
            let mut used_attributes = HashSet::new();
            self.find_attribute_usage(method, attributes, &mut used_attributes);
            method_attribute_usage.push(used_attributes);
        }
        
        // Calculate LCOM using the Henderson-Sellers method
        let m = methods.len() as f64; // Number of methods
        let a = attributes.len() as f64; // Number of attributes
        
        let mut total_attribute_usage = 0.0;
        for usage_set in &method_attribute_usage {
            total_attribute_usage += usage_set.len() as f64;
        }
        
        let average_usage = total_attribute_usage / m;
        let lcom = (average_usage - a) / (1.0 - a);
        
        Ok(lcom.max(0.0))
    }
    
    pub async fn calculate_tcc(&self, class: &ClassNode) -> Result<f64, CohesionError> {
        // TCC (Tight Class Cohesion)
        // Measures the relative number of directly connected methods
        
        let methods = &class.methods;
        
        if methods.len() < 2 {
            return Ok(1.0);
        }
        
        let mut direct_connections = 0;
        let total_possible_connections = methods.len() * (methods.len() - 1) / 2;
        
        for i in 0..methods.len() {
            for j in (i + 1)..methods.len() {
                if self.methods_are_directly_connected(&methods[i], &methods[j]).await? {
                    direct_connections += 1;
                }
            }
        }
        
        Ok(direct_connections as f64 / total_possible_connections as f64)
    }
    
    pub async fn calculate_lcc(&self, class: &ClassNode) -> Result<f64, CohesionError> {
        // LCC (Loose Class Cohesion)
        // Measures both direct and indirect connections between methods
        
        let methods = &class.methods;
        
        if methods.len() < 2 {
            return Ok(1.0);
        }
        
        // Build connection graph
        let mut connection_graph = Graph::new_undirected();
        let mut method_indices = HashMap::new();
        
        // Add nodes for each method
        for (i, method) in methods.iter().enumerate() {
            let node_index = connection_graph.add_node(i);
            method_indices.insert(i, node_index);
        }
        
        // Add edges for connected methods
        for i in 0..methods.len() {
            for j in (i + 1)..methods.len() {
                if self.methods_are_connected(&methods[i], &methods[j]).await? {
                    connection_graph.add_edge(method_indices[&i], method_indices[&j], ());
                }
            }
        }
        
        // Count connected components
        let connected_pairs = self.count_connected_pairs(&connection_graph);
        let total_possible_pairs = methods.len() * (methods.len() - 1) / 2;
        
        Ok(connected_pairs as f64 / total_possible_pairs as f64)
    }
    
    async fn methods_are_directly_connected(&self, method1: &MethodNode, method2: &MethodNode) -> Result<bool, CohesionError> {
        // Two methods are directly connected if they access the same attribute
        let attrs1 = self.get_accessed_attributes(method1);
        let attrs2 = self.get_accessed_attributes(method2);
        
        Ok(!attrs1.is_disjoint(&attrs2))
    }
    
    async fn methods_are_connected(&self, method1: &MethodNode, method2: &MethodNode) -> Result<bool, CohesionError> {
        // Two methods are connected if they are directly connected or call each other
        if self.methods_are_directly_connected(method1, method2).await? {
            return Ok(true);
        }
        
        // Check if one method calls the other
        let calls1 = self.get_method_calls(method1);
        let calls2 = self.get_method_calls(method2);
        
        Ok(calls1.contains(&method2.name) || calls2.contains(&method1.name))
    }
}

pub struct CouplingAnalyzer {
    dependency_analyzer: DependencyAnalyzer,
    call_graph_builder: CallGraphBuilder,
}

impl CouplingAnalyzer {
    pub async fn calculate_coupling_metrics(&self, ast: &UnifiedAST) -> Result<CouplingMetrics, CouplingError> {
        let classes = self.extract_classes(&ast.root_node);
        let mut coupling_metrics = CouplingMetrics::default();
        
        if !classes.is_empty() {
            let mut total_cbo = 0.0;
            let mut total_rfc = 0.0;
            let mut total_dit = 0.0;
            let mut total_noc = 0.0;
            
            for class in &classes {
                let cbo = self.calculate_cbo(class).await?;
                let rfc = self.calculate_rfc(class).await?;
                let dit = self.calculate_dit(class).await?;
                let noc = self.calculate_noc(class, &classes).await?;
                
                total_cbo += cbo as f64;
                total_rfc += rfc as f64;
                total_dit += dit as f64;
                total_noc += noc as f64;
            }
            
            coupling_metrics.average_cbo = total_cbo / classes.len() as f64;
            coupling_metrics.average_rfc = total_rfc / classes.len() as f64;
            coupling_metrics.average_dit = total_dit / classes.len() as f64;
            coupling_metrics.average_noc = total_noc / classes.len() as f64;
        }
        
        Ok(coupling_metrics)
    }
    
    pub async fn calculate_cbo(&self, class: &ClassNode) -> Result<u32, CouplingError> {
        // CBO (Coupling Between Objects)
        // Count the number of classes this class depends on
        
        let mut coupled_classes = HashSet::new();
        
        // Analyze method dependencies
        for method in &class.methods {
            let dependencies = self.dependency_analyzer.analyze_method_dependencies(method).await?;
            for dep in dependencies {
                if dep.dependency_type == DependencyType::ClassDependency {
                    coupled_classes.insert(dep.target_class);
                }
            }
        }
        
        // Analyze attribute dependencies
        for attribute in &class.attributes {
            if let Some(attr_type) = &attribute.attribute_type {
                if let Some(class_name) = self.extract_class_name_from_type(attr_type) {
                    coupled_classes.insert(class_name);
                }
            }
        }
        
        // Analyze inheritance dependencies
        for parent in &class.parent_classes {
            coupled_classes.insert(parent.clone());
        }
        
        Ok(coupled_classes.len() as u32)
    }
    
    pub async fn calculate_rfc(&self, class: &ClassNode) -> Result<u32, CouplingError> {
        // RFC (Response For Class)
        // Count the number of methods that can be invoked in response to a message
        
        let mut response_set = HashSet::new();
        
        // Add all methods of the class
        for method in &class.methods {
            response_set.insert(method.name.clone());
        }
        
        // Add all methods called by the class methods
        for method in &class.methods {
            let called_methods = self.get_called_methods(method).await?;
            for called_method in called_methods {
                response_set.insert(called_method);
            }
        }
        
        Ok(response_set.len() as u32)
    }
    
    pub async fn calculate_dit(&self, class: &ClassNode) -> Result<u32, CouplingError> {
        // DIT (Depth of Inheritance Tree)
        // Maximum length from the class to the root of the inheritance hierarchy
        
        self.calculate_inheritance_depth_recursive(class, &HashSet::new()).await
    }
    
    pub async fn calculate_noc(&self, class: &ClassNode, all_classes: &[ClassNode]) -> Result<u32, CouplingError> {
        // NOC (Number of Children)
        // Number of immediate subclasses
        
        let mut children_count = 0;
        
        for other_class in all_classes {
            if other_class.parent_classes.contains(&class.name) {
                children_count += 1;
            }
        }
        
        Ok(children_count)
    }
}

#[derive(Debug, Clone, Default)]
pub struct CohesionMetrics {
    pub average_lcom: f64,
    pub average_tcc: f64,
    pub average_lcc: f64,
    pub class_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct CouplingMetrics {
    pub average_cbo: f64,
    pub average_rfc: f64,
    pub average_dit: f64,
    pub average_noc: f64,
    pub total_dependencies: u32,
    pub circular_dependencies: u32,
}
```

### 14.6 Quality Metrics and Maintainability Index

#### 14.6.1 Quality Analyzer
```rust
pub struct QualityAnalyzer {
    maintainability_calculator: MaintainabilityCalculator,
    technical_debt_estimator: TechnicalDebtEstimator,
    quality_gate_checker: QualityGateChecker,
}

impl QualityAnalyzer {
    pub async fn calculate_quality_metrics(&self, code_metrics: &CodeMetrics) -> Result<QualityMetrics, QualityError> {
        let maintainability_index = self.maintainability_calculator.calculate_maintainability_index(code_metrics).await?;
        let technical_debt = self.technical_debt_estimator.estimate_technical_debt(code_metrics).await?;
        let quality_gates = self.quality_gate_checker.check_quality_gates(code_metrics).await?;
        
        Ok(QualityMetrics {
            maintainability_index,
            technical_debt_hours: technical_debt.total_hours,
            technical_debt_cost: technical_debt.estimated_cost,
            code_smells: technical_debt.code_smells,
            quality_gate_status: quality_gates.overall_status,
            quality_gate_results: quality_gates.individual_results,
            testability_score: self.calculate_testability_score(code_metrics).await?,
            readability_score: self.calculate_readability_score(code_metrics).await?,
            reliability_score: self.calculate_reliability_score(code_metrics).await?,
        })
    }
}

pub struct MaintainabilityCalculator;

impl MaintainabilityCalculator {
    pub async fn calculate_maintainability_index(&self, metrics: &CodeMetrics) -> Result<f64, QualityError> {
        // Microsoft's Maintainability Index formula (adapted)
        let halstead_volume = metrics.halstead_metrics.volume.max(1.0);
        let cyclomatic_complexity = metrics.complexity_metrics.cyclomatic_complexity.max(1) as f64;
        let lines_of_code = metrics.size_metrics.logical_lines_of_code.max(1) as f64;
        let comment_ratio = metrics.size_metrics.comment_lines as f64 / metrics.size_metrics.total_lines as f64;
        
        // Base maintainability index
        let mut mi = 171.0 
            - 5.2 * halstead_volume.ln()
            - 0.23 * cyclomatic_complexity
            - 16.2 * lines_of_code.ln();
        
        // Adjust for comment ratio
        mi += 50.0 * (2.4 * comment_ratio).sin();
        
        // Normalize to 0-100 scale
        let normalized_mi = (mi * 100.0 / 171.0).max(0.0).min(100.0);
        
        Ok(normalized_mi)
    }
}

pub struct TechnicalDebtEstimator {
    debt_rules: Vec<TechnicalDebtRule>,
}

impl TechnicalDebtEstimator {
    pub async fn estimate_technical_debt(&self, metrics: &CodeMetrics) -> Result<TechnicalDebtEstimate, QualityError> {
        let mut total_debt_minutes = 0.0;
        let mut code_smells = Vec::new();
        
        // Complexity-based debt
        if metrics.complexity_metrics.cyclomatic_complexity > 10 {
            let excess_complexity = metrics.complexity_metrics.cyclomatic_complexity - 10;
            total_debt_minutes += excess_complexity as f64 * 5.0; // 5 minutes per excess complexity point
            
            code_smells.push(CodeSmell {
                smell_type: CodeSmellType::HighComplexity,
                severity: self.rate_complexity_severity(metrics.complexity_metrics.cyclomatic_complexity),
                description: format!("Cyclomatic complexity is {}", metrics.complexity_metrics.cyclomatic_complexity),
                estimated_fix_time_minutes: excess_complexity as f64 * 5.0,
            });
        }
        
        // Size-based debt
        for function_metric in &metrics.function_metrics {
            if function_metric.lines_of_code > 50 {
                let excess_lines = function_metric.lines_of_code - 50;
                total_debt_minutes += excess_lines as f64 * 0.5; // 0.5 minutes per excess line
                
                code_smells.push(CodeSmell {
                    smell_type: CodeSmellType::LongFunction,
                    severity: self.rate_function_size_severity(function_metric.lines_of_code),
                    description: format!("Function '{}' has {} lines", function_metric.name, function_metric.lines_of_code),
                    estimated_fix_time_minutes: excess_lines as f64 * 0.5,
                });
            }
        }
        
        // Cohesion-based debt
        if metrics.cohesion_metrics.average_lcom > 0.8 {
            total_debt_minutes += 30.0; // 30 minutes for low cohesion
            
            code_smells.push(CodeSmell {
                smell_type: CodeSmellType::LowCohesion,
                severity: CodeSmellSeverity::Medium,
                description: format!("Low class cohesion (LCOM: {:.2})", metrics.cohesion_metrics.average_lcom),
                estimated_fix_time_minutes: 30.0,
            });
        }
        
        // Coupling-based debt
        if metrics.coupling_metrics.average_cbo > 10.0 {
            total_debt_minutes += 20.0; // 20 minutes for high coupling
            
            code_smells.push(CodeSmell {
                smell_type: CodeSmellType::HighCoupling,
                severity: CodeSmellSeverity::Medium,
                description: format!("High coupling (CBO: {:.1})", metrics.coupling_metrics.average_cbo),
                estimated_fix_time_minutes: 20.0,
            });
        }
        
        let total_hours = total_debt_minutes / 60.0;
        let estimated_cost = total_hours * 75.0; // $75/hour developer rate
        
        Ok(TechnicalDebtEstimate {
            total_minutes: total_debt_minutes,
            total_hours,
            estimated_cost,
            code_smells,
            debt_ratio: total_debt_minutes / (metrics.size_metrics.logical_lines_of_code as f64 * 0.5), // Minutes per LOC
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    pub maintainability_index: f64,
    pub technical_debt_hours: f64,
    pub technical_debt_cost: f64,
    pub code_smells: Vec<CodeSmell>,
    pub quality_gate_status: QualityGateStatus,
    pub quality_gate_results: Vec<QualityGateResult>,
    pub testability_score: f64,
    pub readability_score: f64,
    pub reliability_score: f64,
}

#[derive(Debug, Clone)]
pub struct CodeSmell {
    pub smell_type: CodeSmellType,
    pub severity: CodeSmellSeverity,
    pub description: String,
    pub estimated_fix_time_minutes: f64,
}

#[derive(Debug, Clone)]
pub enum CodeSmellType {
    HighComplexity,
    LongFunction,
    LargeClass,
    LowCohesion,
    HighCoupling,
    DeepInheritance,
    ManyParameters,
    DuplicatedCode,
    DeadCode,
    MagicNumbers,
    LongParameterList,
    FeatureEnvy,
    DataClumps,
}

#[derive(Debug, Clone)]
pub enum CodeSmellSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct TechnicalDebtEstimate {
    pub total_minutes: f64,
    pub total_hours: f64,
    pub estimated_cost: f64,
    pub code_smells: Vec<CodeSmell>,
    pub debt_ratio: f64,
}
```

### 14.7 Criterios de Completitud

#### 14.7.1 Entregables de la Fase
- [ ] Sistema completo de análisis de complejidad
- [ ] Calculadora de métricas de Halstead
- [ ] Analizador de cohesión y acoplamiento
- [ ] Calculadora de métricas de tamaño
- [ ] Analizador de calidad y mantenibilidad
- [ ] Estimador de deuda técnica
- [ ] Sistema de quality gates
- [ ] Métricas por función y clase
- [ ] Agregación de métricas de proyecto
- [ ] Tests comprehensivos

#### 14.7.2 Criterios de Aceptación
- [ ] Calcula complejidad ciclomática correctamente
- [ ] Métricas de Halstead son precisas
- [ ] Análisis de cohesión/acoplamiento funciona
- [ ] Índice de mantenibilidad es consistente
- [ ] Estimación de deuda técnica es realista
- [ ] Performance acceptable para proyectos grandes
- [ ] Métricas son comparables entre lenguajes
- [ ] Quality gates funcionan correctamente
- [ ] Integration seamless con motor de reglas
- [ ] Documentación completa de métricas

### 14.8 Performance Targets

#### 14.8.1 Benchmarks de Métricas
- **Calculation speed**: <200ms para archivos de 2k LOC
- **Memory usage**: <50MB para proyectos medianos
- **Accuracy**: >95% consistencia con herramientas estándar
- **Project analysis**: <10 segundos para proyectos de 100 archivos
- **Concurrent processing**: >4x speedup en sistemas multi-core

### 14.9 Estimación de Tiempo

#### 14.9.1 Breakdown de Tareas
- Diseño de arquitectura de métricas: 4 días
- Complexity analyzer (ciclomática y cognitiva): 8 días
- Halstead metrics calculator: 6 días
- Cohesion analyzer: 8 días
- Coupling analyzer: 8 días
- Size metrics analyzer: 4 días
- Quality analyzer y maintainability index: 8 días
- Technical debt estimator: 6 días
- Quality gates system: 5 días
- Project-level aggregation: 6 días
- Performance optimization: 6 días
- Integration con motor de reglas: 4 días
- Testing comprehensivo: 8 días
- Documentación: 4 días

**Total estimado: 85 días de desarrollo**

### 14.10 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Sistema completo de métricas de calidad de código
- Capacidades avanzadas de análisis de complejidad
- Estimación precisa de deuda técnica
- Foundation sólida para quality gates
- Base para análisis de tendencias de calidad

La Fase 15 construirá sobre esta base implementando el sistema de categorización y priorización de issues, completando el motor de reglas y detección básica.
