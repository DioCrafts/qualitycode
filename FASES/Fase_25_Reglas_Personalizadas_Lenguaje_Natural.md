# Fase 25: Sistema de Reglas Personalizadas en Lenguaje Natural

## Objetivo General
Desarrollar un sistema revolucionario que permita a los usuarios crear reglas de análisis personalizadas escribiendo en lenguaje natural (español e inglés), utilizando procesamiento de lenguaje natural avanzado y modelos de IA para traducir descripciones humanas en reglas ejecutables, proporcionando flexibilidad total para organizaciones que necesitan reglas específicas de dominio o estándares internos únicos.

## Descripción Técnica Detallada

### 25.1 Arquitectura del Sistema de Reglas Naturales

#### 25.1.1 Diseño del Natural Language Rule System
```
┌─────────────────────────────────────────┐
│    Natural Language Rule System        │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │   NLP       │ │    Intent           │ │
│  │ Processor   │ │   Extractor         │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │    Rule     │ │    Code             │ │
│  │ Generator   │ │   Generator         │ │
│  └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────────────┐ │
│  │ Validation  │ │   Learning          │ │
│  │   Engine    │ │   System            │ │
│  └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
```

#### 25.1.2 Componentes del Sistema
- **NLP Processor**: Procesamiento avanzado de lenguaje natural
- **Intent Extractor**: Extracción de intención de reglas
- **Rule Generator**: Generación automática de reglas ejecutables
- **Code Generator**: Generación de código de reglas
- **Validation Engine**: Validación de reglas generadas
- **Learning System**: Aprendizaje continuo de patrones

### 25.2 Natural Language Processing Engine

#### 25.2.1 NLP Rule Processor
```rust
use tokenizers::Tokenizer;
use candle_transformers::models::bert::BertModel;
use regex::Regex;
use std::collections::HashMap;

pub struct NaturalLanguageRuleProcessor {
    nlp_models: HashMap<Language, Arc<NLPModel>>,
    intent_classifier: Arc<IntentClassifier>,
    entity_extractor: Arc<EntityExtractor>,
    pattern_matcher: Arc<PatternMatcher>,
    rule_template_engine: Arc<RuleTemplateEngine>,
    validation_engine: Arc<RuleValidationEngine>,
    config: NLPConfig,
}

#[derive(Debug, Clone)]
pub struct NLPConfig {
    pub supported_languages: Vec<Language>,
    pub enable_entity_extraction: bool,
    pub enable_intent_classification: bool,
    pub enable_pattern_matching: bool,
    pub confidence_threshold: f64,
    pub max_rule_complexity: u32,
    pub enable_ambiguity_detection: bool,
    pub enable_context_awareness: bool,
    pub custom_domain_vocabulary: HashMap<String, Vec<String>>,
}

impl NaturalLanguageRuleProcessor {
    pub async fn new(config: NLPConfig) -> Result<Self, NLPError> {
        let mut nlp_models = HashMap::new();
        
        // Load NLP models for supported languages
        for language in &config.supported_languages {
            let model = Self::load_nlp_model_for_language(language).await?;
            nlp_models.insert(language.clone(), Arc::new(model));
        }
        
        Ok(Self {
            nlp_models,
            intent_classifier: Arc::new(IntentClassifier::new()),
            entity_extractor: Arc::new(EntityExtractor::new()),
            pattern_matcher: Arc::new(PatternMatcher::new()),
            rule_template_engine: Arc::new(RuleTemplateEngine::new()),
            validation_engine: Arc::new(RuleValidationEngine::new()),
            config,
        })
    }
    
    pub async fn process_natural_language_rule(&self, rule_text: &str, language: Language, context: &RuleCreationContext) -> Result<GeneratedRule, NLPError> {
        let start_time = Instant::now();
        
        // Preprocess the rule text
        let preprocessed_text = self.preprocess_rule_text(rule_text, language).await?;
        
        // Extract intent from the rule
        let intent_analysis = self.intent_classifier.classify_rule_intent(&preprocessed_text, language).await?;
        
        // Extract entities (code elements, conditions, actions)
        let entities = if self.config.enable_entity_extraction {
            self.entity_extractor.extract_entities(&preprocessed_text, language).await?
        } else {
            Vec::new()
        };
        
        // Match against known patterns
        let pattern_matches = if self.config.enable_pattern_matching {
            self.pattern_matcher.find_patterns(&preprocessed_text, language).await?
        } else {
            Vec::new()
        };
        
        // Detect ambiguities
        let ambiguities = if self.config.enable_ambiguity_detection {
            self.detect_ambiguities(&preprocessed_text, &intent_analysis, &entities).await?
        } else {
            Vec::new()
        };
        
        // Generate rule structure
        let rule_structure = self.generate_rule_structure(&intent_analysis, &entities, &pattern_matches, context).await?;
        
        // Generate executable rule code
        let executable_rule = self.rule_template_engine.generate_executable_rule(&rule_structure, context).await?;
        
        // Validate generated rule
        let validation_result = self.validation_engine.validate_generated_rule(&executable_rule, context).await?;
        
        if !validation_result.is_valid {
            return Err(NLPError::InvalidGeneratedRule(validation_result.errors));
        }
        
        Ok(GeneratedRule {
            id: GeneratedRuleId::new(),
            original_text: rule_text.to_string(),
            language,
            preprocessed_text,
            intent_analysis,
            extracted_entities: entities,
            pattern_matches,
            ambiguities,
            rule_structure,
            executable_rule,
            validation_result,
            confidence_score: self.calculate_generation_confidence(&intent_analysis, &entities, &validation_result),
            generation_time_ms: start_time.elapsed().as_millis() as u64,
            created_at: Utc::now(),
        })
    }
    
    async fn preprocess_rule_text(&self, text: &str, language: Language) -> Result<String, NLPError> {
        let mut processed = text.to_string();
        
        // Normalize text
        processed = processed.to_lowercase();
        
        // Handle language-specific preprocessing
        match language {
            Language::Spanish => {
                processed = self.preprocess_spanish_text(&processed).await?;
            }
            Language::English => {
                processed = self.preprocess_english_text(&processed).await?;
            }
            _ => {}
        }
        
        // Remove stop words but preserve important programming terms
        processed = self.remove_stop_words(&processed, language).await?;
        
        // Normalize programming terminology
        processed = self.normalize_programming_terms(&processed, language).await?;
        
        Ok(processed)
    }
    
    async fn preprocess_spanish_text(&self, text: &str) -> Result<String, NLPError> {
        let mut processed = text.to_string();
        
        // Handle Spanish-specific patterns
        let spanish_patterns = [
            (r"\bque\s+no\s+", "that does not "),
            (r"\bque\s+", "that "),
            (r"\bsi\s+", "if "),
            (r"\bcuando\s+", "when "),
            (r"\bdonde\s+", "where "),
            (r"\bfunciones?\s+que\s+", "functions that "),
            (r"\bclases?\s+que\s+", "classes that "),
            (r"\bvariables?\s+que\s+", "variables that "),
            (r"\bmétodos?\s+que\s+", "methods that "),
            (r"\bno\s+debe\b", "must not"),
            (r"\bdebe\b", "must"),
            (r"\bpuede\b", "can"),
            (r"\btiene\s+que\b", "has to"),
        ];
        
        for (pattern, replacement) in &spanish_patterns {
            let regex = Regex::new(pattern)?;
            processed = regex.replace_all(&processed, *replacement).to_string();
        }
        
        Ok(processed)
    }
    
    async fn normalize_programming_terms(&self, text: &str, language: Language) -> Result<String, NLPError> {
        let mut normalized = text.to_string();
        
        let term_mappings = match language {
            Language::Spanish => HashMap::from([
                ("función", "function"),
                ("funciones", "functions"),
                ("clase", "class"),
                ("clases", "classes"),
                ("método", "method"),
                ("métodos", "methods"),
                ("variable", "variable"),
                ("variables", "variables"),
                ("parámetro", "parameter"),
                ("parámetros", "parameters"),
                ("bucle", "loop"),
                ("bucles", "loops"),
                ("condicional", "conditional"),
                ("condición", "condition"),
                ("retorno", "return"),
                ("excepción", "exception"),
                ("error", "error"),
                ("seguridad", "security"),
                ("rendimiento", "performance"),
                ("complejidad", "complexity"),
            ]),
            Language::English => HashMap::new(), // No mapping needed for English
            _ => HashMap::new(),
        };
        
        for (from, to) in term_mappings {
            let pattern = format!(r"\b{}\b", regex::escape(&from));
            let regex = Regex::new(&pattern)?;
            normalized = regex.replace_all(&normalized, to).to_string();
        }
        
        Ok(normalized)
    }
}

#[derive(Debug, Clone)]
pub struct GeneratedRule {
    pub id: GeneratedRuleId,
    pub original_text: String,
    pub language: Language,
    pub preprocessed_text: String,
    pub intent_analysis: IntentAnalysis,
    pub extracted_entities: Vec<ExtractedEntity>,
    pub pattern_matches: Vec<PatternMatch>,
    pub ambiguities: Vec<Ambiguity>,
    pub rule_structure: RuleStructure,
    pub executable_rule: ExecutableRule,
    pub validation_result: RuleValidationResult,
    pub confidence_score: f64,
    pub generation_time_ms: u64,
    pub created_at: DateTime<Utc>,
}

// Example natural language rules:
/*
Spanish Examples:
- "Las funciones no deben tener más de 50 líneas de código"
- "Todas las clases que manejan datos sensibles deben tener validación de entrada"
- "Los métodos que acceden a la base de datos deben usar consultas parametrizadas"
- "Las funciones que contienen la palabra 'password' no deben hacer logging del contenido"
- "Si una función tiene más de 5 parámetros, debe ser refactorizada"
- "Los bucles anidados con más de 3 niveles son problemáticos"
- "Las variables que contienen 'secret' o 'key' no deben ser hardcodeadas"

English Examples:
- "Functions should not exceed 50 lines of code"
- "All classes handling sensitive data must have input validation"
- "Methods accessing the database must use parameterized queries"
- "Functions containing the word 'password' must not log content"
- "If a function has more than 5 parameters, it should be refactored"
- "Nested loops with more than 3 levels are problematic"
- "Variables containing 'secret' or 'key' must not be hardcoded"
*/
```

### 25.3 Intent Classification System

#### 25.3.1 Rule Intent Classifier
```rust
pub struct IntentClassifier {
    classification_model: Arc<ClassificationModel>,
    intent_patterns: HashMap<Language, Vec<IntentPattern>>,
    domain_classifier: Arc<DomainClassifier>,
}

impl IntentClassifier {
    pub async fn classify_rule_intent(&self, text: &str, language: Language) -> Result<IntentAnalysis, IntentClassificationError> {
        // Extract primary intent
        let primary_intent = self.classify_primary_intent(text, language).await?;
        
        // Extract secondary intents
        let secondary_intents = self.classify_secondary_intents(text, language).await?;
        
        // Classify domain
        let domain = self.domain_classifier.classify_domain(text, language).await?;
        
        // Extract rule type
        let rule_type = self.extract_rule_type(text, language).await?;
        
        // Extract conditions and actions
        let conditions = self.extract_conditions(text, language).await?;
        let actions = self.extract_actions(text, language).await?;
        
        Ok(IntentAnalysis {
            primary_intent,
            secondary_intents,
            domain,
            rule_type,
            conditions,
            actions,
            confidence: self.calculate_intent_confidence(&primary_intent, &conditions, &actions),
        })
    }
    
    async fn classify_primary_intent(&self, text: &str, language: Language) -> Result<RuleIntent, IntentClassificationError> {
        // Use pattern matching for common intents
        let patterns = self.intent_patterns.get(&language)
            .ok_or(IntentClassificationError::UnsupportedLanguage(language))?;
        
        for pattern in patterns {
            if pattern.regex.is_match(text) {
                return Ok(pattern.intent.clone());
            }
        }
        
        // Fallback to ML classification
        if let Some(ml_intent) = self.classification_model.classify_intent(text, language).await? {
            Ok(ml_intent)
        } else {
            Ok(RuleIntent::Unknown)
        }
    }
    
    async fn extract_conditions(&self, text: &str, language: Language) -> Result<Vec<RuleCondition>, IntentClassificationError> {
        let mut conditions = Vec::new();
        
        let condition_patterns = match language {
            Language::Spanish => vec![
                (r"si\s+(.+?)\s+entonces", ConditionType::If),
                (r"cuando\s+(.+)", ConditionType::When),
                (r"que\s+(.+)", ConditionType::That),
                (r"con\s+más\s+de\s+(\d+)", ConditionType::GreaterThan),
                (r"con\s+menos\s+de\s+(\d+)", ConditionType::LessThan),
                (r"que\s+contenga\s+(.+)", ConditionType::Contains),
                (r"que\s+no\s+contenga\s+(.+)", ConditionType::NotContains),
                (r"en\s+(.+)", ConditionType::InLocation),
            ],
            Language::English => vec![
                (r"if\s+(.+?)\s+then", ConditionType::If),
                (r"when\s+(.+)", ConditionType::When),
                (r"that\s+(.+)", ConditionType::That),
                (r"with\s+more\s+than\s+(\d+)", ConditionType::GreaterThan),
                (r"with\s+less\s+than\s+(\d+)", ConditionType::LessThan),
                (r"containing\s+(.+)", ConditionType::Contains),
                (r"not\s+containing\s+(.+)", ConditionType::NotContains),
                (r"in\s+(.+)", ConditionType::InLocation),
            ],
            _ => vec![],
        };
        
        for (pattern_str, condition_type) in condition_patterns {
            let regex = Regex::new(&pattern_str)?;
            
            for capture in regex.captures_iter(text) {
                if let Some(condition_text) = capture.get(1) {
                    conditions.push(RuleCondition {
                        condition_type,
                        condition_text: condition_text.as_str().to_string(),
                        parameters: self.extract_condition_parameters(condition_text.as_str()).await?,
                        confidence: 0.8,
                    });
                }
            }
        }
        
        Ok(conditions)
    }
    
    async fn extract_actions(&self, text: &str, language: Language) -> Result<Vec<RuleAction>, IntentClassificationError> {
        let mut actions = Vec::new();
        
        let action_patterns = match language {
            Language::Spanish => vec![
                (r"debe\s+ser\s+(.+)", ActionType::MustBe),
                (r"no\s+debe\s+(.+)", ActionType::MustNotBe),
                (r"debería\s+(.+)", ActionType::Should),
                (r"reportar\s+(.+)", ActionType::Report),
                (r"sugerir\s+(.+)", ActionType::Suggest),
                (r"avisar\s+(.+)", ActionType::Warn),
                (r"fallar\s+(.+)", ActionType::Fail),
                (r"refactorizar", ActionType::Refactor),
                (r"corregir", ActionType::Fix),
            ],
            Language::English => vec![
                (r"must\s+be\s+(.+)", ActionType::MustBe),
                (r"must\s+not\s+(.+)", ActionType::MustNotBe),
                (r"should\s+(.+)", ActionType::Should),
                (r"report\s+(.+)", ActionType::Report),
                (r"suggest\s+(.+)", ActionType::Suggest),
                (r"warn\s+(.+)", ActionType::Warn),
                (r"fail\s+(.+)", ActionType::Fail),
                (r"refactor", ActionType::Refactor),
                (r"fix", ActionType::Fix),
            ],
            _ => vec![],
        };
        
        for (pattern_str, action_type) in action_patterns {
            let regex = Regex::new(&pattern_str)?;
            
            for capture in regex.captures_iter(text) {
                let action_text = if let Some(action_match) = capture.get(1) {
                    action_match.as_str().to_string()
                } else {
                    String::new()
                };
                
                actions.push(RuleAction {
                    action_type,
                    action_text,
                    parameters: self.extract_action_parameters(&action_text).await?,
                    severity: self.infer_action_severity(&action_type, &action_text),
                });
            }
        }
        
        Ok(actions)
    }
}

#[derive(Debug, Clone)]
pub struct IntentAnalysis {
    pub primary_intent: RuleIntent,
    pub secondary_intents: Vec<RuleIntent>,
    pub domain: RuleDomain,
    pub rule_type: RuleType,
    pub conditions: Vec<RuleCondition>,
    pub actions: Vec<RuleAction>,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum RuleIntent {
    Prohibit,           // "no debe", "must not"
    Require,            // "debe", "must"
    Recommend,          // "debería", "should"
    Limit,              // "no más de", "not more than"
    Ensure,             // "asegurar", "ensure"
    Validate,           // "validar", "validate"
    Check,              // "verificar", "check"
    Count,              // "contar", "count"
    Measure,            // "medir", "measure"
    Detect,             // "detectar", "detect"
    Unknown,
}

#[derive(Debug, Clone)]
pub enum RuleDomain {
    Security,
    Performance,
    Maintainability,
    BestPractices,
    Naming,
    Structure,
    Complexity,
    Documentation,
    Testing,
    Architecture,
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum RuleType {
    Constraint,         // Limits or restrictions
    Quality,            // Quality measurements
    Pattern,            // Pattern detection
    Metric,             // Metric calculation
    Validation,         // Validation rules
    Transformation,     // Code transformation
    Detection,          // Issue detection
}

#[derive(Debug, Clone)]
pub struct RuleCondition {
    pub condition_type: ConditionType,
    pub condition_text: String,
    pub parameters: HashMap<String, String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum ConditionType {
    If,
    When,
    That,
    GreaterThan,
    LessThan,
    EqualTo,
    Contains,
    NotContains,
    InLocation,
    OfType,
    WithAttribute,
}

#[derive(Debug, Clone)]
pub struct RuleAction {
    pub action_type: ActionType,
    pub action_text: String,
    pub parameters: HashMap<String, String>,
    pub severity: ActionSeverity,
}

#[derive(Debug, Clone, Copy)]
pub enum ActionType {
    MustBe,
    MustNotBe,
    Should,
    Report,
    Suggest,
    Warn,
    Fail,
    Refactor,
    Fix,
    Count,
    Measure,
}

#[derive(Debug, Clone)]
pub enum ActionSeverity {
    Info,
    Warning,
    Error,
    Critical,
}
```

### 25.4 Rule Generation Engine

#### 25.4.1 Executable Rule Generator
```rust
pub struct ExecutableRuleGenerator {
    code_generator: Arc<RuleCodeGenerator>,
    template_engine: Arc<RuleTemplateEngine>,
    ast_pattern_generator: Arc<ASTPatternGenerator>,
    query_generator: Arc<QueryGenerator>,
    validator: Arc<RuleValidator>,
}

impl ExecutableRuleGenerator {
    pub async fn generate_executable_rule(&self, rule_structure: &RuleStructure, context: &RuleCreationContext) -> Result<ExecutableRule, RuleGenerationError> {
        // Determine the best implementation approach
        let implementation_strategy = self.determine_implementation_strategy(rule_structure, context).await?;
        
        let executable_rule = match implementation_strategy {
            ImplementationStrategy::ASTPattern => {
                self.generate_ast_pattern_rule(rule_structure, context).await?
            }
            ImplementationStrategy::Query => {
                self.generate_query_rule(rule_structure, context).await?
            }
            ImplementationStrategy::Procedural => {
                self.generate_procedural_rule(rule_structure, context).await?
            }
            ImplementationStrategy::Hybrid => {
                self.generate_hybrid_rule(rule_structure, context).await?
            }
        };
        
        // Validate the generated rule
        let validation_result = self.validator.validate_rule(&executable_rule, context).await?;
        
        if !validation_result.is_valid {
            return Err(RuleGenerationError::ValidationFailed(validation_result.errors));
        }
        
        Ok(executable_rule)
    }
    
    async fn generate_ast_pattern_rule(&self, rule_structure: &RuleStructure, context: &RuleCreationContext) -> Result<ExecutableRule, RuleGenerationError> {
        // Generate AST pattern based on rule structure
        let ast_pattern = self.ast_pattern_generator.generate_pattern(rule_structure).await?;
        
        // Generate conditions
        let conditions = self.generate_pattern_conditions(rule_structure).await?;
        
        // Generate actions
        let actions = self.generate_pattern_actions(rule_structure).await?;
        
        Ok(ExecutableRule {
            id: ExecutableRuleId::new(),
            rule_name: self.generate_rule_name(rule_structure),
            description: rule_structure.description.clone(),
            implementation: RuleImplementation::Pattern {
                ast_pattern,
                conditions,
                actions,
            },
            languages: self.determine_applicable_languages(rule_structure),
            category: self.determine_rule_category(rule_structure),
            severity: self.determine_rule_severity(rule_structure),
            configuration: self.generate_rule_configuration(rule_structure),
            metadata: self.generate_rule_metadata(rule_structure, context),
        })
    }
    
    async fn generate_procedural_rule(&self, rule_structure: &RuleStructure, context: &RuleCreationContext) -> Result<ExecutableRule, RuleGenerationError> {
        // Generate Rust code for procedural rule
        let rust_code = self.code_generator.generate_rust_analyzer_function(rule_structure).await?;
        
        Ok(ExecutableRule {
            id: ExecutableRuleId::new(),
            rule_name: self.generate_rule_name(rule_structure),
            description: rule_structure.description.clone(),
            implementation: RuleImplementation::Procedural {
                analyzer_function: rust_code,
                parameters: self.extract_procedural_parameters(rule_structure),
            },
            languages: self.determine_applicable_languages(rule_structure),
            category: self.determine_rule_category(rule_structure),
            severity: self.determine_rule_severity(rule_structure),
            configuration: self.generate_rule_configuration(rule_structure),
            metadata: self.generate_rule_metadata(rule_structure, context),
        })
    }
}

pub struct RuleCodeGenerator {
    template_engine: TemplateEngine,
    code_templates: HashMap<RuleIntent, CodeTemplate>,
}

impl RuleCodeGenerator {
    pub async fn generate_rust_analyzer_function(&self, rule_structure: &RuleStructure) -> Result<String, CodeGenerationError> {
        let template = self.select_template_for_intent(&rule_structure.intent_analysis.primary_intent)?;
        
        let template_vars = self.build_template_variables(rule_structure)?;
        
        let generated_code = self.template_engine.render(&template.content, &template_vars)?;
        
        // Format and validate the generated Rust code
        let formatted_code = self.format_rust_code(&generated_code)?;
        let validated_code = self.validate_rust_syntax(&formatted_code)?;
        
        Ok(validated_code)
    }
    
    fn select_template_for_intent(&self, intent: &RuleIntent) -> Result<&CodeTemplate, CodeGenerationError> {
        self.code_templates.get(intent)
            .ok_or(CodeGenerationError::NoTemplateForIntent(intent.clone()))
    }
    
    fn build_template_variables(&self, rule_structure: &RuleStructure) -> Result<HashMap<String, String>, CodeGenerationError> {
        let mut vars = HashMap::new();
        
        // Extract target elements
        if let Some(target) = &rule_structure.target_element {
            vars.insert("target_type".to_string(), target.element_type.to_string());
            vars.insert("target_name".to_string(), target.name.clone());
        }
        
        // Extract conditions
        for (i, condition) in rule_structure.conditions.iter().enumerate() {
            vars.insert(format!("condition_{}", i), condition.condition_text.clone());
            
            for (key, value) in &condition.parameters {
                vars.insert(format!("condition_{}_{}", i, key), value.clone());
            }
        }
        
        // Extract thresholds
        for threshold in &rule_structure.thresholds {
            vars.insert(format!("threshold_{}", threshold.name), threshold.value.to_string());
        }
        
        // Generate function name
        vars.insert("function_name".to_string(), self.generate_function_name(rule_structure));
        
        Ok(vars)
    }
    
    fn generate_function_name(&self, rule_structure: &RuleStructure) -> String {
        let intent_name = match rule_structure.intent_analysis.primary_intent {
            RuleIntent::Prohibit => "check_prohibition",
            RuleIntent::Require => "check_requirement",
            RuleIntent::Limit => "check_limit",
            RuleIntent::Validate => "validate",
            RuleIntent::Count => "count",
            RuleIntent::Measure => "measure",
            _ => "analyze",
        };
        
        let target_name = rule_structure.target_element.as_ref()
            .map(|t| t.element_type.to_string().to_lowercase())
            .unwrap_or_else(|| "code".to_string());
        
        format!("{}_{}", intent_name, target_name)
    }
}

// Example code templates
const FUNCTION_LINE_LIMIT_TEMPLATE: &str = r#"
pub async fn check_function_line_limit(ast: &UnifiedAST) -> Result<Vec<RuleViolation>, RuleError> {
    let mut violations = Vec::new();
    let max_lines = {{threshold_max_lines}};
    
    let functions = extract_functions(&ast.root_node);
    
    for function in functions {
        let line_count = count_function_lines(&function);
        
        if line_count > max_lines {
            violations.push(RuleViolation {
                severity: RuleSeverity::{{severity}},
                message: format!("Function '{}' has {} lines, exceeding the limit of {}", 
                    function.name.unwrap_or_default(), line_count, max_lines),
                location: function.position.clone(),
                rule_id: "{{rule_id}}".to_string(),
                suggestions: vec![
                    "Consider breaking this function into smaller functions".to_string(),
                    "Extract complex logic into separate methods".to_string(),
                ],
            });
        }
    }
    
    Ok(violations)
}
"#;

#[derive(Debug, Clone)]
pub struct RuleStructure {
    pub intent_analysis: IntentAnalysis,
    pub target_element: Option<TargetElement>,
    pub conditions: Vec<RuleCondition>,
    pub actions: Vec<RuleAction>,
    pub thresholds: Vec<Threshold>,
    pub scope: RuleScope,
    pub description: String,
    pub examples: Vec<RuleExample>,
}

#[derive(Debug, Clone)]
pub struct TargetElement {
    pub element_type: ElementType,
    pub name: String,
    pub attributes: HashMap<String, String>,
    pub filters: Vec<ElementFilter>,
}

#[derive(Debug, Clone)]
pub enum ElementType {
    Function,
    Class,
    Method,
    Variable,
    Parameter,
    Loop,
    Conditional,
    Expression,
    Statement,
    File,
    Module,
}

#[derive(Debug, Clone)]
pub struct Threshold {
    pub name: String,
    pub value: f64,
    pub operator: ThresholdOperator,
    pub unit: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ThresholdOperator {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

#[derive(Debug, Clone)]
pub struct ExecutableRule {
    pub id: ExecutableRuleId,
    pub rule_name: String,
    pub description: String,
    pub implementation: RuleImplementation,
    pub languages: Vec<ProgrammingLanguage>,
    pub category: RuleCategory,
    pub severity: RuleSeverity,
    pub configuration: RuleConfiguration,
    pub metadata: RuleMetadata,
}
```

### 25.5 Learning and Improvement System

#### 25.5.1 Rule Learning System
```rust
pub struct RuleLearningSystem {
    feedback_collector: Arc<FeedbackCollector>,
    pattern_learner: Arc<PatternLearner>,
    rule_optimizer: Arc<RuleOptimizer>,
    performance_tracker: Arc<RulePerformanceTracker>,
    accuracy_monitor: Arc<AccuracyMonitor>,
    config: LearningConfig,
}

#[derive(Debug, Clone)]
pub struct LearningConfig {
    pub enable_feedback_learning: bool,
    pub enable_pattern_learning: bool,
    pub enable_performance_optimization: bool,
    pub feedback_weight: f64,
    pub learning_rate: f64,
    pub min_feedback_count: usize,
    pub accuracy_threshold: f64,
    pub enable_rule_evolution: bool,
}

impl RuleLearningSystem {
    pub async fn learn_from_feedback(&self, rule_id: &ExecutableRuleId, feedback: &RuleFeedback) -> Result<LearningResult, LearningError> {
        // Collect feedback
        self.feedback_collector.record_feedback(rule_id, feedback).await?;
        
        // Analyze feedback patterns
        let feedback_analysis = self.analyze_feedback_patterns(rule_id).await?;
        
        // Generate improvement suggestions
        let improvements = self.generate_rule_improvements(rule_id, &feedback_analysis).await?;
        
        // Apply improvements if confidence is high enough
        let applied_improvements = if feedback_analysis.confidence > 0.8 {
            self.apply_rule_improvements(rule_id, &improvements).await?
        } else {
            Vec::new()
        };
        
        Ok(LearningResult {
            rule_id: rule_id.clone(),
            feedback_analyzed: true,
            improvements_suggested: improvements.len(),
            improvements_applied: applied_improvements.len(),
            new_accuracy: feedback_analysis.estimated_new_accuracy,
            learning_confidence: feedback_analysis.confidence,
        })
    }
    
    pub async fn optimize_rule_performance(&self, rule_id: &ExecutableRuleId) -> Result<OptimizationResult, LearningError> {
        // Analyze rule performance
        let performance_data = self.performance_tracker.get_performance_data(rule_id).await?;
        
        // Identify optimization opportunities
        let optimizations = self.rule_optimizer.identify_optimizations(&performance_data).await?;
        
        // Apply optimizations
        let optimization_results = self.rule_optimizer.apply_optimizations(rule_id, &optimizations).await?;
        
        Ok(OptimizationResult {
            rule_id: rule_id.clone(),
            optimizations_applied: optimization_results.len(),
            performance_improvement: self.calculate_performance_improvement(&optimization_results),
            accuracy_impact: self.calculate_accuracy_impact(&optimization_results),
        })
    }
    
    pub async fn evolve_rule(&self, rule_id: &ExecutableRuleId, evolution_context: &EvolutionContext) -> Result<EvolvedRule, LearningError> {
        // Analyze rule usage patterns
        let usage_patterns = self.analyze_rule_usage_patterns(rule_id).await?;
        
        // Identify evolution opportunities
        let evolution_opportunities = self.identify_evolution_opportunities(&usage_patterns, evolution_context).await?;
        
        // Generate evolved rule
        let evolved_rule = self.generate_evolved_rule(rule_id, &evolution_opportunities, evolution_context).await?;
        
        // Validate evolved rule
        let validation_result = self.validate_evolved_rule(&evolved_rule, evolution_context).await?;
        
        if !validation_result.is_valid {
            return Err(LearningError::EvolutionValidationFailed(validation_result.errors));
        }
        
        Ok(evolved_rule)
    }
}

#[derive(Debug, Clone)]
pub struct RuleFeedback {
    pub feedback_type: FeedbackType,
    pub accuracy_rating: f64,
    pub usefulness_rating: f64,
    pub false_positive: bool,
    pub false_negative: bool,
    pub suggested_improvements: Vec<String>,
    pub context: FeedbackContext,
    pub provided_by: UserId,
    pub provided_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub enum FeedbackType {
    Positive,
    Negative,
    Neutral,
    Suggestion,
    BugReport,
}

#[derive(Debug, Clone)]
pub struct FeedbackContext {
    pub file_path: PathBuf,
    pub language: ProgrammingLanguage,
    pub project_type: ProjectType,
    pub team_size: TeamSize,
    pub domain: ApplicationDomain,
}

#[derive(Debug, Clone)]
pub enum ProjectType {
    WebApplication,
    MobileApplication,
    DesktopApplication,
    Library,
    Framework,
    Microservice,
    Monolith,
    CLI,
    API,
}

#[derive(Debug, Clone)]
pub enum TeamSize {
    Individual,
    Small,      // 2-5 developers
    Medium,     // 6-20 developers
    Large,      // 21-100 developers
    Enterprise, // 100+ developers
}

#[derive(Debug, Clone)]
pub enum ApplicationDomain {
    Finance,
    Healthcare,
    Education,
    Ecommerce,
    Gaming,
    Enterprise,
    Startup,
    OpenSource,
    Government,
    Research,
}
```

### 25.6 Natural Language Rule Examples

#### 25.6.1 Example Rule Translations

```rust
// Example 1: Spanish to Executable Rule
// Input: "Las funciones no deben tener más de 50 líneas de código"
// Generated Rule:
pub struct FunctionLineLimitRule {
    pub max_lines: u32,
}

impl Rule for FunctionLineLimitRule {
    async fn analyze(&self, ast: &UnifiedAST) -> Result<Vec<RuleViolation>, RuleError> {
        let functions = extract_functions(&ast.root_node);
        let mut violations = Vec::new();
        
        for function in functions {
            let line_count = count_function_lines(&function);
            if line_count > self.max_lines {
                violations.push(RuleViolation {
                    severity: RuleSeverity::Medium,
                    message: format!("La función '{}' tiene {} líneas, excediendo el límite de {}", 
                        function.name.unwrap_or_default(), line_count, self.max_lines),
                    location: function.position.clone(),
                    // ... other fields
                });
            }
        }
        
        Ok(violations)
    }
}

// Example 2: English to Executable Rule  
// Input: "Methods accessing the database must use parameterized queries"
// Generated Rule:
pub struct DatabaseParameterizedQueryRule;

impl Rule for DatabaseParameterizedQueryRule {
    async fn analyze(&self, ast: &UnifiedAST) -> Result<Vec<RuleViolation>, RuleError> {
        let mut violations = Vec::new();
        
        // Find database access methods
        let db_methods = find_database_access_methods(ast);
        
        for method in db_methods {
            if !uses_parameterized_queries(&method) {
                violations.push(RuleViolation {
                    severity: RuleSeverity::High,
                    message: format!("Method '{}' accesses database without parameterized queries", 
                        method.name.unwrap_or_default()),
                    location: method.position.clone(),
                    suggestions: vec![
                        "Use prepared statements or parameterized queries".to_string(),
                        "Avoid string concatenation in SQL queries".to_string(),
                    ],
                    // ... other fields
                });
            }
        }
        
        Ok(violations)
    }
}

// Example 3: Complex Spanish Rule
// Input: "Si una clase tiene más de 20 métodos y maneja datos sensibles, debe implementar logging de auditoría"
// This would generate a more complex rule with multiple conditions
```

### 25.7 Criterios de Completitud

#### 25.7.1 Entregables de la Fase
- [ ] Sistema de procesamiento de lenguaje natural implementado
- [ ] Clasificador de intención de reglas
- [ ] Generador de reglas ejecutables
- [ ] Sistema de validación de reglas generadas
- [ ] Motor de aprendizaje y mejora continua
- [ ] Soporte completo para español e inglés
- [ ] Templates de código para diferentes tipos de reglas
- [ ] Sistema de feedback y optimización
- [ ] API de creación de reglas naturales
- [ ] Tests comprehensivos con ejemplos reales

#### 25.7.2 Criterios de Aceptación
- [ ] Procesa reglas en lenguaje natural con >80% precisión
- [ ] Genera reglas ejecutables válidas >90% del tiempo
- [ ] Soporte robusto para español e inglés
- [ ] Sistema de feedback mejora reglas continuamente
- [ ] Validación detecta errores antes de deployment
- [ ] Performance acceptable para creación interactiva
- [ ] Reglas generadas son mantenibles y legibles
- [ ] Integration seamless con motor de reglas existente
- [ ] Documentación automática de reglas generadas
- [ ] Escalabilidad para cientos de reglas personalizadas

### 25.8 Performance Targets

#### 25.8.1 Benchmarks de NLP
- **Rule processing**: <5 segundos para reglas típicas
- **Code generation**: <10 segundos para reglas complejas
- **Validation**: <2 segundos por regla generada
- **Intent classification**: <1 segundo por regla
- **Entity extraction**: <500ms por regla

### 25.9 Estimación de Tiempo

#### 25.9.1 Breakdown de Tareas
- Diseño de arquitectura NLP: 8 días
- NLP processor core: 15 días
- Intent classifier: 12 días
- Entity extractor: 10 días
- Rule generator: 18 días
- Code generator con templates: 15 días
- Validation engine: 10 días
- Learning system: 12 días
- Soporte multiidioma: 12 días
- Performance optimization: 8 días
- Integration y testing: 15 días
- Documentación: 6 días

**Total estimado: 141 días de desarrollo**

### 25.10 Próximos Pasos

Al completar esta fase, el sistema tendrá:
- Capacidades únicas de creación de reglas en lenguaje natural
- Flexibilidad total para organizaciones enterprise
- Soporte multiidioma robusto
- Sistema de aprendizaje continuo
- Completitud de las características avanzadas y optimización

<function_calls>
<invoke name="todo_write">
<parameter name="merge">true
