# Fase 6: Implementación del Parser Universal usando Tree-sitter

## Objetivo General
Implementar un sistema de parsing universal basado en Tree-sitter que pueda analizar sintácticamente código en múltiples lenguajes de programación, generando Abstract Syntax Trees (ASTs) consistentes y proporcionando la base para todos los análisis posteriores del agente CodeAnt.

## Descripción Técnica Detallada

### 6.1 Arquitectura del Parser Universal

#### 6.1.1 Diseño del Sistema de Parsing
```
┌─────────────────────────────────────────┐
│           Universal Parser              │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ Tree-sitter │ │    Language         │ │
│  │   Core      │ │   Grammars          │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │    AST      │ │    Query            │ │
│  │ Normalizer  │ │   Engine            │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│         Language Detection Layer        │
└─────────────────────────────────────────┘
```

#### 6.1.2 Componentes Principales
- **Tree-sitter Core**: Motor de parsing incremental
- **Language Grammars**: Gramáticas específicas por lenguaje
- **AST Normalizer**: Normalización de ASTs cross-language
- **Query Engine**: Sistema de consultas sobre ASTs
- **Language Detector**: Detección automática de lenguajes
- **Cache System**: Cache de ASTs parseados

### 6.2 Tree-sitter Integration

#### 6.2.1 Core Parser Implementation
```rust
use tree_sitter::{Language, Parser, Tree, Node, Query, QueryCursor};
use std::collections::HashMap;
use std::sync::Arc;

pub struct UniversalParser {
    parsers: HashMap<ProgrammingLanguage, Parser>,
    languages: HashMap<ProgrammingLanguage, Language>,
    queries: HashMap<ProgrammingLanguage, Vec<Query>>,
    config: ParserConfig,
    cache: Arc<ParseCache>,
}

#[derive(Debug, Clone)]
pub struct ParserConfig {
    pub enable_incremental_parsing: bool,
    pub max_file_size_mb: usize,
    pub timeout_seconds: u64,
    pub enable_error_recovery: bool,
    pub cache_parsed_trees: bool,
    pub parallel_parsing: bool,
    pub max_concurrent_parsers: usize,
}

impl UniversalParser {
    pub async fn new(config: ParserConfig) -> Result<Self, ParserError> {
        let mut parser = Self {
            parsers: HashMap::new(),
            languages: HashMap::new(),
            queries: HashMap::new(),
            config,
            cache: Arc::new(ParseCache::new()),
        };
        
        // Initialize supported languages
        parser.initialize_languages().await?;
        parser.load_queries().await?;
        
        Ok(parser)
    }
    
    async fn initialize_languages(&mut self) -> Result<(), ParserError> {
        // Python
        let python_language = tree_sitter_python::language();
        self.languages.insert(ProgrammingLanguage::Python, python_language);
        let mut python_parser = Parser::new();
        python_parser.set_language(python_language)?;
        self.parsers.insert(ProgrammingLanguage::Python, python_parser);
        
        // TypeScript
        let typescript_language = tree_sitter_typescript::language_typescript();
        self.languages.insert(ProgrammingLanguage::TypeScript, typescript_language);
        let mut ts_parser = Parser::new();
        ts_parser.set_language(typescript_language)?;
        self.parsers.insert(ProgrammingLanguage::TypeScript, ts_parser);
        
        // JavaScript
        let javascript_language = tree_sitter_javascript::language();
        self.languages.insert(ProgrammingLanguage::JavaScript, javascript_language);
        let mut js_parser = Parser::new();
        js_parser.set_language(javascript_language)?;
        self.parsers.insert(ProgrammingLanguage::JavaScript, js_parser);
        
        // Rust
        let rust_language = tree_sitter_rust::language();
        self.languages.insert(ProgrammingLanguage::Rust, rust_language);
        let mut rust_parser = Parser::new();
        rust_parser.set_language(rust_language)?;
        self.parsers.insert(ProgrammingLanguage::Rust, rust_parser);
        
        // Add more languages as needed
        self.add_additional_languages().await?;
        
        Ok(())
    }
    
    async fn add_additional_languages(&mut self) -> Result<(), ParserError> {
        // Java
        if let Ok(java_lang) = tree_sitter_java::language() {
            self.languages.insert(ProgrammingLanguage::Java, java_lang);
            let mut java_parser = Parser::new();
            java_parser.set_language(java_lang)?;
            self.parsers.insert(ProgrammingLanguage::Java, java_parser);
        }
        
        // Go
        if let Ok(go_lang) = tree_sitter_go::language() {
            self.languages.insert(ProgrammingLanguage::Go, go_lang);
            let mut go_parser = Parser::new();
            go_parser.set_language(go_lang)?;
            self.parsers.insert(ProgrammingLanguage::Go, go_parser);
        }
        
        // C++
        if let Ok(cpp_lang) = tree_sitter_cpp::language() {
            self.languages.insert(ProgrammingLanguage::Cpp, cpp_lang);
            let mut cpp_parser = Parser::new();
            cpp_parser.set_language(cpp_lang)?;
            self.parsers.insert(ProgrammingLanguage::Cpp, cpp_parser);
        }
        
        // C#
        if let Ok(csharp_lang) = tree_sitter_c_sharp::language() {
            self.languages.insert(ProgrammingLanguage::CSharp, csharp_lang);
            let mut csharp_parser = Parser::new();
            csharp_parser.set_language(csharp_lang)?;
            self.parsers.insert(ProgrammingLanguage::CSharp, csharp_parser);
        }
        
        Ok(())
    }
}
```

#### 6.2.2 Parsing Interface
```rust
#[async_trait]
pub trait UniversalParsingInterface: Send + Sync {
    async fn parse_file(&self, file_path: &Path) -> Result<ParseResult, ParserError>;
    async fn parse_content(&self, content: &str, language: ProgrammingLanguage) -> Result<ParseResult, ParserError>;
    async fn parse_incremental(&self, old_tree: &Tree, content: &str, edits: &[InputEdit]) -> Result<ParseResult, ParserError>;
    async fn get_ast_json(&self, tree: &Tree) -> Result<String, ParserError>;
    async fn query_ast(&self, tree: &Tree, query: &str, language: ProgrammingLanguage) -> Result<Vec<QueryMatch>, ParserError>;
}

#[derive(Debug, Clone)]
pub struct ParseResult {
    pub tree: Tree,
    pub language: ProgrammingLanguage,
    pub file_path: Option<PathBuf>,
    pub content_hash: String,
    pub parse_duration_ms: u64,
    pub node_count: usize,
    pub error_count: usize,
    pub warnings: Vec<ParseWarning>,
    pub metadata: ParseMetadata,
}

#[derive(Debug, Clone)]
pub struct ParseMetadata {
    pub file_size_bytes: usize,
    pub line_count: usize,
    pub character_count: usize,
    pub encoding: String,
    pub has_syntax_errors: bool,
    pub complexity_estimate: f64,
    pub parsed_at: DateTime<Utc>,
}

impl UniversalParsingInterface for UniversalParser {
    async fn parse_file(&self, file_path: &Path) -> Result<ParseResult, ParserError> {
        let start_time = Instant::now();
        
        // Read file content
        let content = tokio::fs::read_to_string(file_path).await
            .map_err(|e| ParserError::FileReadError(e.to_string()))?;
        
        // Detect language
        let language = self.detect_language(file_path, &content).await?;
        
        // Check cache first
        let content_hash = self.calculate_content_hash(&content);
        if let Some(cached_result) = self.cache.get(&content_hash).await {
            return Ok(cached_result);
        }
        
        // Parse content
        let mut result = self.parse_content(&content, language).await?;
        result.file_path = Some(file_path.to_path_buf());
        result.parse_duration_ms = start_time.elapsed().as_millis() as u64;
        
        // Cache result
        if self.config.cache_parsed_trees {
            self.cache.insert(content_hash, result.clone()).await;
        }
        
        Ok(result)
    }
    
    async fn parse_content(&self, content: &str, language: ProgrammingLanguage) -> Result<ParseResult, ParserError> {
        let start_time = Instant::now();
        
        // Validate file size
        if content.len() > self.config.max_file_size_mb * 1024 * 1024 {
            return Err(ParserError::FileTooLarge(content.len()));
        }
        
        // Get parser for language
        let parser = self.parsers.get(&language)
            .ok_or(ParserError::UnsupportedLanguage(language))?;
        
        // Parse with timeout
        let tree = tokio::time::timeout(
            Duration::from_secs(self.config.timeout_seconds),
            self.parse_with_error_recovery(parser, content)
        ).await
        .map_err(|_| ParserError::ParseTimeout)?
        .map_err(ParserError::TreeSitterError)?;
        
        // Analyze parse result
        let root_node = tree.root_node();
        let error_count = self.count_error_nodes(&root_node);
        let warnings = self.analyze_parse_warnings(&root_node, content);
        
        let metadata = ParseMetadata {
            file_size_bytes: content.len(),
            line_count: content.lines().count(),
            character_count: content.chars().count(),
            encoding: "UTF-8".to_string(), // TODO: Detect actual encoding
            has_syntax_errors: error_count > 0,
            complexity_estimate: self.estimate_complexity(&root_node),
            parsed_at: Utc::now(),
        };
        
        Ok(ParseResult {
            tree,
            language,
            file_path: None,
            content_hash: self.calculate_content_hash(content),
            parse_duration_ms: start_time.elapsed().as_millis() as u64,
            node_count: self.count_nodes(&root_node),
            error_count,
            warnings,
            metadata,
        })
    }
    
    async fn parse_incremental(&self, old_tree: &Tree, content: &str, edits: &[InputEdit]) -> Result<ParseResult, ParserError> {
        if !self.config.enable_incremental_parsing {
            // Fallback to full parse
            let language = self.detect_language_from_tree(old_tree)?;
            return self.parse_content(content, language).await;
        }
        
        let start_time = Instant::now();
        let language = self.detect_language_from_tree(old_tree)?;
        
        let parser = self.parsers.get(&language)
            .ok_or(ParserError::UnsupportedLanguage(language))?;
        
        // Apply edits to old tree
        let mut new_tree = old_tree.clone();
        for edit in edits {
            new_tree.edit(edit);
        }
        
        // Parse incrementally
        let tree = parser.parse(content, Some(&new_tree))
            .ok_or(ParserError::IncrementalParseFailed)?;
        
        let root_node = tree.root_node();
        let error_count = self.count_error_nodes(&root_node);
        let warnings = self.analyze_parse_warnings(&root_node, content);
        
        let metadata = ParseMetadata {
            file_size_bytes: content.len(),
            line_count: content.lines().count(),
            character_count: content.chars().count(),
            encoding: "UTF-8".to_string(),
            has_syntax_errors: error_count > 0,
            complexity_estimate: self.estimate_complexity(&root_node),
            parsed_at: Utc::now(),
        };
        
        Ok(ParseResult {
            tree,
            language,
            file_path: None,
            content_hash: self.calculate_content_hash(content),
            parse_duration_ms: start_time.elapsed().as_millis() as u64,
            node_count: self.count_nodes(&root_node),
            error_count,
            warnings,
            metadata,
        })
    }
}
```

### 6.3 Language Detection System

#### 6.3.1 Multi-Strategy Language Detection
```rust
pub struct LanguageDetector {
    extension_map: HashMap<String, ProgrammingLanguage>,
    shebang_patterns: Vec<(Regex, ProgrammingLanguage)>,
    content_patterns: Vec<(Regex, ProgrammingLanguage, f64)>, // pattern, language, confidence
    filename_patterns: Vec<(Regex, ProgrammingLanguage)>,
}

impl LanguageDetector {
    pub fn new() -> Self {
        let mut detector = Self {
            extension_map: HashMap::new(),
            shebang_patterns: Vec::new(),
            content_patterns: Vec::new(),
            filename_patterns: Vec::new(),
        };
        
        detector.initialize_detection_rules();
        detector
    }
    
    fn initialize_detection_rules(&mut self) {
        // Extension mappings
        self.extension_map.insert("py".to_string(), ProgrammingLanguage::Python);
        self.extension_map.insert("pyw".to_string(), ProgrammingLanguage::Python);
        self.extension_map.insert("ts".to_string(), ProgrammingLanguage::TypeScript);
        self.extension_map.insert("tsx".to_string(), ProgrammingLanguage::TypeScript);
        self.extension_map.insert("js".to_string(), ProgrammingLanguage::JavaScript);
        self.extension_map.insert("jsx".to_string(), ProgrammingLanguage::JavaScript);
        self.extension_map.insert("rs".to_string(), ProgrammingLanguage::Rust);
        self.extension_map.insert("java".to_string(), ProgrammingLanguage::Java);
        self.extension_map.insert("go".to_string(), ProgrammingLanguage::Go);
        self.extension_map.insert("cpp".to_string(), ProgrammingLanguage::Cpp);
        self.extension_map.insert("cxx".to_string(), ProgrammingLanguage::Cpp);
        self.extension_map.insert("cc".to_string(), ProgrammingLanguage::Cpp);
        self.extension_map.insert("cs".to_string(), ProgrammingLanguage::CSharp);
        
        // Shebang patterns
        self.shebang_patterns.push((
            Regex::new(r"^#!/usr/bin/env python").unwrap(),
            ProgrammingLanguage::Python
        ));
        self.shebang_patterns.push((
            Regex::new(r"^#!/usr/bin/python").unwrap(),
            ProgrammingLanguage::Python
        ));
        self.shebang_patterns.push((
            Regex::new(r"^#!/usr/bin/env node").unwrap(),
            ProgrammingLanguage::JavaScript
        ));
        
        // Content patterns with confidence scores
        self.content_patterns.push((
            Regex::new(r"import\s+\{[^}]*\}\s+from\s+['\"]").unwrap(),
            ProgrammingLanguage::TypeScript,
            0.8
        ));
        self.content_patterns.push((
            Regex::new(r"interface\s+\w+\s*\{").unwrap(),
            ProgrammingLanguage::TypeScript,
            0.9
        ));
        self.content_patterns.push((
            Regex::new(r"fn\s+\w+\s*\([^)]*\)\s*->").unwrap(),
            ProgrammingLanguage::Rust,
            0.9
        ));
        self.content_patterns.push((
            Regex::new(r"def\s+\w+\s*\([^)]*\)\s*:").unwrap(),
            ProgrammingLanguage::Python,
            0.8
        ));
        self.content_patterns.push((
            Regex::new(r"public\s+class\s+\w+").unwrap(),
            ProgrammingLanguage::Java,
            0.8
        ));
        self.content_patterns.push((
            Regex::new(r"package\s+\w+").unwrap(),
            ProgrammingLanguage::Go,
            0.7
        ));
        
        // Filename patterns
        self.filename_patterns.push((
            Regex::new(r"^Cargo\.toml$").unwrap(),
            ProgrammingLanguage::Rust
        ));
        self.filename_patterns.push((
            Regex::new(r"^package\.json$").unwrap(),
            ProgrammingLanguage::JavaScript
        ));
        self.filename_patterns.push((
            Regex::new(r"^tsconfig\.json$").unwrap(),
            ProgrammingLanguage::TypeScript
        ));
    }
    
    pub async fn detect_language(&self, file_path: &Path, content: &str) -> Result<ProgrammingLanguage, DetectionError> {
        let mut candidates: HashMap<ProgrammingLanguage, f64> = HashMap::new();
        
        // 1. Check file extension (high confidence)
        if let Some(extension) = file_path.extension().and_then(|ext| ext.to_str()) {
            if let Some(&language) = self.extension_map.get(&extension.to_lowercase()) {
                candidates.insert(language, 0.9);
            }
        }
        
        // 2. Check filename patterns (high confidence)
        if let Some(filename) = file_path.file_name().and_then(|name| name.to_str()) {
            for (pattern, language) in &self.filename_patterns {
                if pattern.is_match(filename) {
                    *candidates.entry(*language).or_insert(0.0) += 0.85;
                }
            }
        }
        
        // 3. Check shebang (very high confidence)
        if let Some(first_line) = content.lines().next() {
            for (pattern, language) in &self.shebang_patterns {
                if pattern.is_match(first_line) {
                    *candidates.entry(*language).or_insert(0.0) += 0.95;
                }
            }
        }
        
        // 4. Check content patterns (medium confidence)
        for (pattern, language, confidence) in &self.content_patterns {
            if pattern.is_match(content) {
                *candidates.entry(*language).or_insert(0.0) += confidence;
            }
        }
        
        // 5. Statistical analysis for ambiguous cases
        if candidates.is_empty() || self.has_ambiguous_detection(&candidates) {
            let statistical_result = self.statistical_analysis(content).await?;
            for (language, confidence) in statistical_result {
                *candidates.entry(language).or_insert(0.0) += confidence * 0.6;
            }
        }
        
        // Return language with highest confidence
        candidates.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(language, _)| language)
            .ok_or(DetectionError::UnknownLanguage)
    }
    
    async fn statistical_analysis(&self, content: &str) -> Result<Vec<(ProgrammingLanguage, f64)>, DetectionError> {
        let mut features = self.extract_statistical_features(content);
        
        // Simple keyword-based classification
        let mut scores: HashMap<ProgrammingLanguage, f64> = HashMap::new();
        
        // Python indicators
        let python_keywords = ["def", "import", "from", "class", "if __name__", "self"];
        let python_score = python_keywords.iter()
            .map(|&kw| if content.contains(kw) { 1.0 } else { 0.0 })
            .sum::<f64>() / python_keywords.len() as f64;
        scores.insert(ProgrammingLanguage::Python, python_score);
        
        // JavaScript/TypeScript indicators
        let js_keywords = ["function", "const", "let", "var", "=>", "console.log"];
        let js_score = js_keywords.iter()
            .map(|&kw| if content.contains(kw) { 1.0 } else { 0.0 })
            .sum::<f64>() / js_keywords.len() as f64;
        scores.insert(ProgrammingLanguage::JavaScript, js_score);
        
        // TypeScript specific
        let ts_keywords = ["interface", "type", "enum", "namespace", "declare"];
        let ts_score = ts_keywords.iter()
            .map(|&kw| if content.contains(kw) { 1.0 } else { 0.0 })
            .sum::<f64>() / ts_keywords.len() as f64;
        if ts_score > 0.0 {
            scores.insert(ProgrammingLanguage::TypeScript, js_score + ts_score);
        }
        
        // Rust indicators
        let rust_keywords = ["fn", "let mut", "match", "impl", "trait", "pub"];
        let rust_score = rust_keywords.iter()
            .map(|&kw| if content.contains(kw) { 1.0 } else { 0.0 })
            .sum::<f64>() / rust_keywords.len() as f64;
        scores.insert(ProgrammingLanguage::Rust, rust_score);
        
        Ok(scores.into_iter().collect())
    }
}
```

### 6.4 AST Normalization System

#### 6.4.1 Cross-Language AST Representation
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedAST {
    pub root: NormalizedNode,
    pub language: ProgrammingLanguage,
    pub metadata: ASTMetadata,
    pub symbol_table: SymbolTable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedNode {
    pub node_type: NodeType,
    pub name: Option<String>,
    pub value: Option<String>,
    pub position: SourcePosition,
    pub children: Vec<NormalizedNode>,
    pub attributes: HashMap<String, String>,
    pub semantic_info: Option<SemanticInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    // Program structure
    Program,
    Module,
    Package,
    
    // Declarations
    FunctionDeclaration,
    ClassDeclaration,
    InterfaceDeclaration,
    VariableDeclaration,
    ConstantDeclaration,
    TypeDeclaration,
    
    // Statements
    ExpressionStatement,
    IfStatement,
    ForStatement,
    WhileStatement,
    ReturnStatement,
    BreakStatement,
    ContinueStatement,
    TryStatement,
    ThrowStatement,
    
    // Expressions
    BinaryExpression,
    UnaryExpression,
    CallExpression,
    MemberExpression,
    AssignmentExpression,
    ConditionalExpression,
    ArrayExpression,
    ObjectExpression,
    
    // Literals
    StringLiteral,
    NumberLiteral,
    BooleanLiteral,
    NullLiteral,
    Identifier,
    
    // Comments and documentation
    Comment,
    Documentation,
    
    // Language-specific (preserved for detailed analysis)
    LanguageSpecific(String),
}

pub struct ASTNormalizer {
    language_normalizers: HashMap<ProgrammingLanguage, Box<dyn LanguageNormalizer>>,
}

#[async_trait]
pub trait LanguageNormalizer: Send + Sync {
    async fn normalize(&self, tree: &Tree, content: &str) -> Result<NormalizedAST, NormalizationError>;
    fn get_language(&self) -> ProgrammingLanguage;
    fn extract_symbols(&self, node: &Node, content: &str) -> SymbolTable;
    fn map_node_type(&self, tree_sitter_type: &str) -> NodeType;
}
```

#### 6.4.2 Python AST Normalizer
```rust
pub struct PythonNormalizer;

#[async_trait]
impl LanguageNormalizer for PythonNormalizer {
    async fn normalize(&self, tree: &Tree, content: &str) -> Result<NormalizedAST, NormalizationError> {
        let root_node = tree.root_node();
        let normalized_root = self.normalize_node(&root_node, content)?;
        let symbol_table = self.extract_symbols(&root_node, content);
        
        Ok(NormalizedAST {
            root: normalized_root,
            language: ProgrammingLanguage::Python,
            metadata: ASTMetadata {
                node_count: self.count_nodes(&root_node),
                depth: self.calculate_depth(&root_node),
                complexity: self.calculate_complexity(&root_node),
                created_at: Utc::now(),
            },
            symbol_table,
        })
    }
    
    fn normalize_node(&self, node: &Node, content: &str) -> Result<NormalizedNode, NormalizationError> {
        let node_type = self.map_node_type(node.kind());
        let name = self.extract_node_name(node, content);
        let value = self.extract_node_value(node, content);
        let position = SourcePosition {
            start_line: node.start_position().row,
            start_column: node.start_position().column,
            end_line: node.end_position().row,
            end_column: node.end_position().column,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        };
        
        let mut children = Vec::new();
        let mut cursor = node.walk();
        
        for child in node.children(&mut cursor) {
            if !child.is_error() && !self.should_skip_node(&child) {
                children.push(self.normalize_node(&child, content)?);
            }
        }
        
        let attributes = self.extract_attributes(node, content);
        let semantic_info = self.extract_semantic_info(node, content);
        
        Ok(NormalizedNode {
            node_type,
            name,
            value,
            position,
            children,
            attributes,
            semantic_info,
        })
    }
    
    fn map_node_type(&self, tree_sitter_type: &str) -> NodeType {
        match tree_sitter_type {
            "module" => NodeType::Program,
            "function_definition" => NodeType::FunctionDeclaration,
            "class_definition" => NodeType::ClassDeclaration,
            "assignment" => NodeType::AssignmentExpression,
            "if_statement" => NodeType::IfStatement,
            "for_statement" => NodeType::ForStatement,
            "while_statement" => NodeType::WhileStatement,
            "return_statement" => NodeType::ReturnStatement,
            "call" => NodeType::CallExpression,
            "binary_operator" => NodeType::BinaryExpression,
            "unary_operator" => NodeType::UnaryExpression,
            "identifier" => NodeType::Identifier,
            "string" => NodeType::StringLiteral,
            "integer" => NodeType::NumberLiteral,
            "float" => NodeType::NumberLiteral,
            "true" | "false" => NodeType::BooleanLiteral,
            "none" => NodeType::NullLiteral,
            "comment" => NodeType::Comment,
            _ => NodeType::LanguageSpecific(tree_sitter_type.to_string()),
        }
    }
    
    fn extract_symbols(&self, node: &Node, content: &str) -> SymbolTable {
        let mut symbol_table = SymbolTable::new();
        self.extract_symbols_recursive(node, content, &mut symbol_table, ScopeLevel::Global);
        symbol_table
    }
    
    fn extract_symbols_recursive(&self, node: &Node, content: &str, symbols: &mut SymbolTable, scope: ScopeLevel) {
        match node.kind() {
            "function_definition" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    let name = self.get_node_text(&name_node, content);
                    let symbol = Symbol {
                        name,
                        symbol_type: SymbolType::Function,
                        scope: scope.clone(),
                        position: SourcePosition::from_node(&name_node),
                        visibility: self.extract_visibility(node),
                        metadata: self.extract_symbol_metadata(node, content),
                    };
                    symbols.add_symbol(symbol);
                }
                
                // Process function body in new scope
                let function_scope = ScopeLevel::Function(symbols.get_current_scope_id());
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.extract_symbols_recursive(&child, content, symbols, function_scope.clone());
                }
            }
            "class_definition" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    let name = self.get_node_text(&name_node, content);
                    let symbol = Symbol {
                        name,
                        symbol_type: SymbolType::Class,
                        scope: scope.clone(),
                        position: SourcePosition::from_node(&name_node),
                        visibility: self.extract_visibility(node),
                        metadata: self.extract_symbol_metadata(node, content),
                    };
                    symbols.add_symbol(symbol);
                }
                
                // Process class body in new scope
                let class_scope = ScopeLevel::Class(symbols.get_current_scope_id());
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.extract_symbols_recursive(&child, content, symbols, class_scope.clone());
                }
            }
            _ => {
                // Continue with children in same scope
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.extract_symbols_recursive(&child, content, symbols, scope.clone());
                }
            }
        }
    }
}
```

### 6.5 Query Engine Implementation

#### 6.5.1 AST Query System
```rust
pub struct ASTQueryEngine {
    compiled_queries: HashMap<String, CompiledQuery>,
    query_cache: Arc<RwLock<HashMap<String, QueryResult>>>,
}

#[derive(Debug, Clone)]
pub struct CompiledQuery {
    pub name: String,
    pub pattern: String,
    pub tree_sitter_query: Query,
    pub post_processors: Vec<Box<dyn QueryPostProcessor>>,
    pub language: ProgrammingLanguage,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub matches: Vec<QueryMatch>,
    pub execution_time_ms: u64,
    pub node_count_examined: usize,
    pub query_name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryMatch {
    pub captures: HashMap<String, QueryCapture>,
    pub pattern_index: usize,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryCapture {
    pub node_type: String,
    pub text: String,
    pub position: SourcePosition,
    pub metadata: HashMap<String, String>,
}

impl ASTQueryEngine {
    pub fn new() -> Self {
        Self {
            compiled_queries: HashMap::new(),
            query_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn load_predefined_queries(&mut self) -> Result<(), QueryError> {
        // Load common code pattern queries
        self.add_query("function_calls", r#"
            (call
              function: (identifier) @function_name
              arguments: (argument_list) @arguments)
        "#, ProgrammingLanguage::Python).await?;
        
        self.add_query("class_methods", r#"
            (class_definition
              name: (identifier) @class_name
              body: (block
                (function_definition
                  name: (identifier) @method_name)))
        "#, ProgrammingLanguage::Python).await?;
        
        self.add_query("variable_assignments", r#"
            (assignment
              left: (identifier) @variable_name
              right: (_) @value)
        "#, ProgrammingLanguage::Python).await?;
        
        // TypeScript/JavaScript queries
        self.add_query("interface_definitions", r#"
            (interface_declaration
              name: (type_identifier) @interface_name
              body: (object_type) @body)
        "#, ProgrammingLanguage::TypeScript).await?;
        
        self.add_query("arrow_functions", r#"
            (arrow_function
              parameters: (formal_parameters) @params
              body: (_) @body)
        "#, ProgrammingLanguage::TypeScript).await?;
        
        // Rust queries
        self.add_query("rust_functions", r#"
            (function_item
              name: (identifier) @function_name
              parameters: (parameters) @params
              return_type: (type_identifier)? @return_type
              body: (block) @body)
        "#, ProgrammingLanguage::Rust).await?;
        
        Ok(())
    }
    
    pub async fn add_query(&mut self, name: &str, pattern: &str, language: ProgrammingLanguage) -> Result<(), QueryError> {
        let tree_sitter_language = self.get_tree_sitter_language(language)?;
        let query = Query::new(tree_sitter_language, pattern)
            .map_err(|e| QueryError::CompilationError(e.to_string()))?;
        
        let compiled_query = CompiledQuery {
            name: name.to_string(),
            pattern: pattern.to_string(),
            tree_sitter_query: query,
            post_processors: Vec::new(),
            language,
        };
        
        self.compiled_queries.insert(name.to_string(), compiled_query);
        Ok(())
    }
    
    pub async fn execute_query(&self, query_name: &str, tree: &Tree, content: &str) -> Result<QueryResult, QueryError> {
        let start_time = Instant::now();
        
        // Check cache first
        let cache_key = format!("{}:{}", query_name, self.calculate_tree_hash(tree));
        if let Some(cached_result) = self.query_cache.read().await.get(&cache_key) {
            return Ok(cached_result.clone());
        }
        
        let compiled_query = self.compiled_queries.get(query_name)
            .ok_or(QueryError::QueryNotFound(query_name.to_string()))?;
        
        let mut query_cursor = QueryCursor::new();
        let root_node = tree.root_node();
        let matches = query_cursor.matches(&compiled_query.tree_sitter_query, root_node, content.as_bytes());
        
        let mut result_matches = Vec::new();
        let mut nodes_examined = 0;
        
        for match_ in matches {
            nodes_examined += 1;
            let mut captures = HashMap::new();
            
            for capture in match_.captures {
                let capture_name = compiled_query.tree_sitter_query
                    .capture_names()[capture.index as usize]
                    .clone();
                
                let node_text = self.get_node_text(&capture.node, content);
                let query_capture = QueryCapture {
                    node_type: capture.node.kind().to_string(),
                    text: node_text,
                    position: SourcePosition::from_node(&capture.node),
                    metadata: self.extract_capture_metadata(&capture.node, content),
                };
                
                captures.insert(capture_name, query_capture);
            }
            
            let query_match = QueryMatch {
                captures,
                pattern_index: match_.pattern_index,
                score: 1.0, // TODO: Implement scoring algorithm
            };
            
            result_matches.push(query_match);
        }
        
        // Apply post-processors
        for processor in &compiled_query.post_processors {
            result_matches = processor.process(result_matches)?;
        }
        
        let result = QueryResult {
            matches: result_matches,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            node_count_examined: nodes_examined,
            query_name: query_name.to_string(),
        };
        
        // Cache result
        self.query_cache.write().await.insert(cache_key, result.clone());
        
        Ok(result)
    }
    
    pub async fn execute_custom_query(&self, pattern: &str, language: ProgrammingLanguage, tree: &Tree, content: &str) -> Result<QueryResult, QueryError> {
        let tree_sitter_language = self.get_tree_sitter_language(language)?;
        let query = Query::new(tree_sitter_language, pattern)
            .map_err(|e| QueryError::CompilationError(e.to_string()))?;
        
        let start_time = Instant::now();
        let mut query_cursor = QueryCursor::new();
        let root_node = tree.root_node();
        let matches = query_cursor.matches(&query, root_node, content.as_bytes());
        
        let mut result_matches = Vec::new();
        let mut nodes_examined = 0;
        
        for match_ in matches {
            nodes_examined += 1;
            let mut captures = HashMap::new();
            
            for capture in match_.captures {
                let capture_name = query.capture_names()[capture.index as usize].clone();
                let node_text = self.get_node_text(&capture.node, content);
                
                let query_capture = QueryCapture {
                    node_type: capture.node.kind().to_string(),
                    text: node_text,
                    position: SourcePosition::from_node(&capture.node),
                    metadata: self.extract_capture_metadata(&capture.node, content),
                };
                
                captures.insert(capture_name, query_capture);
            }
            
            let query_match = QueryMatch {
                captures,
                pattern_index: match_.pattern_index,
                score: 1.0,
            };
            
            result_matches.push(query_match);
        }
        
        Ok(QueryResult {
            matches: result_matches,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            node_count_examined: nodes_examined,
            query_name: "custom".to_string(),
        })
    }
}
```

### 6.6 Parse Cache System

#### 6.6.1 Intelligent Caching
```rust
pub struct ParseCache {
    memory_cache: Arc<RwLock<LruCache<String, CachedParseResult>>>,
    disk_cache: Arc<DiskCache>,
    config: CacheConfig,
    metrics: Arc<CacheMetrics>,
}

#[derive(Debug, Clone)]
pub struct CachedParseResult {
    pub parse_result: ParseResult,
    pub cached_at: DateTime<Utc>,
    pub access_count: u64,
    pub last_accessed: DateTime<Utc>,
    pub file_hash: String,
    pub file_size: usize,
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub memory_cache_size: usize,
    pub disk_cache_size_mb: usize,
    pub ttl_seconds: u64,
    pub enable_disk_cache: bool,
    pub compression_enabled: bool,
    pub eviction_policy: EvictionPolicy,
}

#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    TTL,
    Hybrid,
}

impl ParseCache {
    pub fn new(config: CacheConfig) -> Self {
        let memory_cache = Arc::new(RwLock::new(
            LruCache::new(NonZeroUsize::new(config.memory_cache_size).unwrap())
        ));
        
        let disk_cache = Arc::new(DiskCache::new(
            config.disk_cache_size_mb,
            config.compression_enabled,
        ));
        
        Self {
            memory_cache,
            disk_cache,
            config,
            metrics: Arc::new(CacheMetrics::new()),
        }
    }
    
    pub async fn get(&self, content_hash: &str) -> Option<ParseResult> {
        self.metrics.record_access();
        
        // Try memory cache first
        if let Some(cached_result) = self.get_from_memory(content_hash).await {
            self.metrics.record_hit(CacheLevel::Memory);
            return Some(cached_result.parse_result);
        }
        
        // Try disk cache
        if self.config.enable_disk_cache {
            if let Some(cached_result) = self.disk_cache.get(content_hash).await {
                self.metrics.record_hit(CacheLevel::Disk);
                
                // Promote to memory cache
                self.insert_to_memory(content_hash.to_string(), cached_result.clone()).await;
                return Some(cached_result.parse_result);
            }
        }
        
        self.metrics.record_miss();
        None
    }
    
    pub async fn insert(&self, content_hash: String, parse_result: ParseResult) {
        let cached_result = CachedParseResult {
            parse_result: parse_result.clone(),
            cached_at: Utc::now(),
            access_count: 1,
            last_accessed: Utc::now(),
            file_hash: content_hash.clone(),
            file_size: parse_result.metadata.file_size_bytes,
        };
        
        // Insert to memory cache
        self.insert_to_memory(content_hash.clone(), cached_result.clone()).await;
        
        // Insert to disk cache if enabled
        if self.config.enable_disk_cache {
            self.disk_cache.insert(content_hash, cached_result).await;
        }
        
        self.metrics.record_insertion();
    }
    
    async fn get_from_memory(&self, content_hash: &str) -> Option<CachedParseResult> {
        let mut cache = self.memory_cache.write().await;
        if let Some(cached_result) = cache.get_mut(content_hash) {
            // Check TTL
            let age = Utc::now().signed_duration_since(cached_result.cached_at);
            if age.num_seconds() > self.config.ttl_seconds as i64 {
                cache.pop(content_hash);
                return None;
            }
            
            // Update access statistics
            cached_result.access_count += 1;
            cached_result.last_accessed = Utc::now();
            
            return Some(cached_result.clone());
        }
        
        None
    }
    
    async fn insert_to_memory(&self, content_hash: String, cached_result: CachedParseResult) {
        let mut cache = self.memory_cache.write().await;
        cache.put(content_hash, cached_result);
    }
    
    pub async fn invalidate(&self, content_hash: &str) {
        // Remove from memory cache
        self.memory_cache.write().await.pop(content_hash);
        
        // Remove from disk cache
        if self.config.enable_disk_cache {
            self.disk_cache.remove(content_hash).await;
        }
        
        self.metrics.record_invalidation();
    }
    
    pub async fn get_cache_stats(&self) -> CacheStats {
        let memory_cache = self.memory_cache.read().await;
        
        CacheStats {
            memory_cache_size: memory_cache.len(),
            memory_cache_capacity: memory_cache.cap().get(),
            disk_cache_size_mb: if self.config.enable_disk_cache {
                self.disk_cache.get_size_mb().await
            } else {
                0
            },
            hit_rate: self.metrics.get_hit_rate(),
            total_accesses: self.metrics.get_total_accesses(),
            total_hits: self.metrics.get_total_hits(),
            total_misses: self.metrics.get_total_misses(),
        }
    }
}
```

### 6.7 Performance Optimization

#### 6.7.1 Parallel Processing
```rust
pub struct ParallelParser {
    parser_pool: Arc<UniversalParser>,
    semaphore: Arc<Semaphore>,
    config: ParallelConfig,
}

#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pub max_concurrent_parses: usize,
    pub chunk_size: usize,
    pub enable_work_stealing: bool,
    pub priority_scheduling: bool,
}

impl ParallelParser {
    pub fn new(parser_pool: Arc<UniversalParser>, config: ParallelConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_parses));
        
        Self {
            parser_pool,
            semaphore,
            config,
        }
    }
    
    pub async fn parse_files_parallel(&self, file_paths: Vec<PathBuf>) -> Result<Vec<ParseResult>, ParserError> {
        let chunks: Vec<_> = file_paths.chunks(self.config.chunk_size).collect();
        let mut results = Vec::new();
        
        let futures: Vec<_> = chunks.into_iter()
            .map(|chunk| self.parse_chunk(chunk.to_vec()))
            .collect();
        
        let chunk_results = futures::future::try_join_all(futures).await?;
        
        for chunk_result in chunk_results {
            results.extend(chunk_result);
        }
        
        Ok(results)
    }
    
    async fn parse_chunk(&self, file_paths: Vec<PathBuf>) -> Result<Vec<ParseResult>, ParserError> {
        let mut results = Vec::new();
        
        let parse_futures: Vec<_> = file_paths.into_iter()
            .map(|path| self.parse_single_file(path))
            .collect();
        
        let parse_results = futures::future::try_join_all(parse_futures).await?;
        results.extend(parse_results);
        
        Ok(results)
    }
    
    async fn parse_single_file(&self, file_path: PathBuf) -> Result<ParseResult, ParserError> {
        let _permit = self.semaphore.acquire().await
            .map_err(|_| ParserError::SemaphoreError)?;
        
        self.parser_pool.parse_file(&file_path).await
    }
    
    pub async fn parse_directory_parallel(&self, directory: &Path, extensions: &[String]) -> Result<Vec<ParseResult>, ParserError> {
        let file_paths = self.collect_files_recursive(directory, extensions).await?;
        self.parse_files_parallel(file_paths).await
    }
    
    async fn collect_files_recursive(&self, directory: &Path, extensions: &[String]) -> Result<Vec<PathBuf>, ParserError> {
        let mut file_paths = Vec::new();
        let mut stack = vec![directory.to_path_buf()];
        
        while let Some(current_dir) = stack.pop() {
            let mut entries = tokio::fs::read_dir(&current_dir).await
                .map_err(|e| ParserError::DirectoryReadError(e.to_string()))?;
            
            while let Some(entry) = entries.next_entry().await
                .map_err(|e| ParserError::DirectoryReadError(e.to_string()))? {
                
                let path = entry.path();
                
                if path.is_dir() {
                    stack.push(path);
                } else if path.is_file() {
                    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
                        if extensions.contains(&extension.to_lowercase()) {
                            file_paths.push(path);
                        }
                    }
                }
            }
        }
        
        Ok(file_paths)
    }
}
```

### 6.8 Error Handling y Recovery

#### 6.8.1 Robust Error Recovery
```rust
#[derive(Debug, thiserror::Error)]
pub enum ParserError {
    #[error("Tree-sitter error: {0}")]
    TreeSitterError(#[from] tree_sitter::LanguageError),
    
    #[error("File read error: {0}")]
    FileReadError(String),
    
    #[error("File too large: {0} bytes")]
    FileTooLarge(usize),
    
    #[error("Parse timeout")]
    ParseTimeout,
    
    #[error("Unsupported language: {0:?}")]
    UnsupportedLanguage(ProgrammingLanguage),
    
    #[error("Incremental parse failed")]
    IncrementalParseFailed,
    
    #[error("Language detection failed")]
    LanguageDetectionFailed,
    
    #[error("Normalization error: {0}")]
    NormalizationError(String),
    
    #[error("Query error: {0}")]
    QueryError(String),
    
    #[error("Cache error: {0}")]
    CacheError(String),
}

pub struct ErrorRecoveryHandler {
    fallback_strategies: Vec<Box<dyn FallbackStrategy>>,
    error_metrics: Arc<ErrorMetrics>,
}

#[async_trait]
pub trait FallbackStrategy: Send + Sync {
    async fn can_handle(&self, error: &ParserError) -> bool;
    async fn handle(&self, error: ParserError, context: &ParseContext) -> Result<ParseResult, ParserError>;
    fn priority(&self) -> u8;
}

pub struct ParseContext {
    pub file_path: PathBuf,
    pub content: String,
    pub detected_language: Option<ProgrammingLanguage>,
    pub previous_attempts: Vec<ParserError>,
}

// Fallback strategy for syntax errors
pub struct SyntaxErrorFallback;

#[async_trait]
impl FallbackStrategy for SyntaxErrorFallback {
    async fn can_handle(&self, error: &ParserError) -> bool {
        matches!(error, ParserError::TreeSitterError(_))
    }
    
    async fn handle(&self, _error: ParserError, context: &ParseContext) -> Result<ParseResult, ParserError> {
        // Try to parse with error recovery enabled
        let mut parser = Parser::new();
        
        if let Some(language) = context.detected_language {
            let tree_sitter_lang = self.get_tree_sitter_language(language)?;
            parser.set_language(tree_sitter_lang)?;
            
            // Enable error recovery
            parser.set_timeout_micros(Some(30_000_000)); // 30 seconds
            
            if let Some(tree) = parser.parse(&context.content, None) {
                return Ok(ParseResult {
                    tree,
                    language,
                    file_path: Some(context.file_path.clone()),
                    content_hash: self.calculate_hash(&context.content),
                    parse_duration_ms: 0,
                    node_count: 0,
                    error_count: self.count_error_nodes(&tree.root_node()),
                    warnings: vec![ParseWarning::SyntaxErrorsRecovered],
                    metadata: ParseMetadata::default(),
                });
            }
        }
        
        Err(ParserError::IncrementalParseFailed)
    }
    
    fn priority(&self) -> u8 {
        10
    }
}
```

### 6.9 Criterios de Completitud

#### 6.9.1 Entregables de la Fase
- [ ] Parser universal Tree-sitter implementado
- [ ] Soporte para Python, TypeScript, JavaScript, Rust
- [ ] Sistema de detección automática de lenguajes
- [ ] AST normalization cross-language
- [ ] Query engine para consultas sobre ASTs
- [ ] Sistema de cache inteligente
- [ ] Procesamiento paralelo de archivos
- [ ] Error recovery robusto
- [ ] Métricas y monitoring de parsing
- [ ] Tests comprehensivos

#### 6.9.2 Criterios de Aceptación
- [ ] Parse correctamente archivos en todos los lenguajes soportados
- [ ] Detección de lenguajes con >95% precisión
- [ ] ASTs normalizados consistentes entre lenguajes
- [ ] Queries funcionan correctamente en todos los lenguajes
- [ ] Cache reduce tiempo de parsing en >50%
- [ ] Procesamiento paralelo escala linealmente
- [ ] Error recovery maneja archivos con errores sintácticos
- [ ] Performance: <100ms para archivos <10KB
- [ ] Memory usage: <50MB para repositorios típicos
- [ ] Tests cubren >90% del código

### 6.10 Performance Targets

#### 6.10.1 Benchmarks de Performance
- **Parsing speed**: >1000 lines/second por parser
- **Memory usage**: <10MB por AST cached
- **Cache hit rate**: >80% en uso típico
- **Parallel efficiency**: >70% utilización de cores
- **Error recovery**: <5% overhead adicional

### 6.11 Estimación de Tiempo

#### 6.11.1 Breakdown de Tareas
- Setup Tree-sitter y configuración básica: 3 días
- Implementación parser universal: 5 días
- Sistema de detección de lenguajes: 4 días
- AST normalization system: 6 días
- Query engine implementation: 5 días
- Sistema de cache: 4 días
- Procesamiento paralelo: 4 días
- Error handling y recovery: 3 días
- Testing e integración: 5 días
- Performance optimization: 3 días
- Documentación: 2 días

**Total estimado: 44 días de desarrollo**

### 6.12 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Capacidad de parsing universal para múltiples lenguajes
- ASTs normalizados y queryables
- Performance optimizada con caching y paralelización
- Error recovery robusto
- Foundation sólida para análisis específicos por lenguaje

La Fase 7 construirá sobre esta base implementando el parser especializado para Python con análisis AST avanzado.
