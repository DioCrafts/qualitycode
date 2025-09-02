"""
Validator for generated fixes and refactored code.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import asyncio
import logging
import ast
import re
from datetime import datetime

from ...domain.entities.fix_generation import (
    ValidationResult, ValidationStatus, GeneratedFix,
    FixType, RefactoringResult
)
from ...domain.entities.language import ProgrammingLanguage
from .exceptions import ValidationError, UnsupportedLanguageError
from .ai_code_generator import FixContext


logger = logging.getLogger(__name__)


@dataclass
class SyntaxValidation:
    """Result of syntax validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    line_numbers: List[int] = field(default_factory=list)


@dataclass
class SemanticValidation:
    """Result of semantic validation."""
    is_valid: bool
    semantic_errors: List[str] = field(default_factory=list)
    type_errors: List[str] = field(default_factory=list)
    undefined_references: List[str] = field(default_factory=list)
    unused_variables: List[str] = field(default_factory=list)


@dataclass
class FunctionalValidation:
    """Result of functional validation."""
    preserves_functionality: bool
    behavior_changes: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    api_changes: List[str] = field(default_factory=list)


@dataclass
class StyleValidation:
    """Result of style validation."""
    follows_conventions: bool
    style_issues: List[str] = field(default_factory=list)
    naming_issues: List[str] = field(default_factory=list)
    formatting_issues: List[str] = field(default_factory=list)


@dataclass
class SecurityValidation:
    """Result of security validation."""
    is_secure: bool
    vulnerabilities: List[str] = field(default_factory=list)
    security_improvements: List[str] = field(default_factory=list)
    risk_level: str = "low"


class FixValidator:
    """Validates generated fixes and refactored code."""
    
    def __init__(self):
        self.syntax_validators = self._initialize_syntax_validators()
        self.semantic_analyzer = SemanticAnalyzer()
        self.functional_analyzer = FunctionalAnalyzer()
        self.style_checker = StyleChecker()
        self.security_scanner = SecurityScanner()
    
    def _initialize_syntax_validators(self) -> Dict[ProgrammingLanguage, Any]:
        """Initialize language-specific syntax validators."""
        return {
            ProgrammingLanguage.PYTHON: PythonSyntaxValidator(),
            ProgrammingLanguage.JAVASCRIPT: JavaScriptSyntaxValidator(),
            ProgrammingLanguage.TYPESCRIPT: TypeScriptSyntaxValidator(),
            ProgrammingLanguage.JAVA: JavaSyntaxValidator(),
            ProgrammingLanguage.GO: GoSyntaxValidator()
        }
    
    async def validate_fix(
        self,
        fix: GeneratedFix,
        context: Optional[FixContext] = None
    ) -> List[ValidationResult]:
        """Validate a generated fix comprehensively."""
        validation_results = []
        
        # Syntax validation
        syntax_result = await self.validate_syntax(fix.fixed_code, fix.language)
        validation_results.append(self._create_validation_result(
            "syntax",
            syntax_result.is_valid,
            syntax_result.errors,
            syntax_result.warnings
        ))
        
        # Only continue if syntax is valid
        if not syntax_result.is_valid:
            return validation_results
        
        # Semantic validation
        if context:
            semantic_result = await self.validate_semantics(
                fix.fixed_code,
                context
            )
            validation_results.append(self._create_validation_result(
                "semantic",
                semantic_result.is_valid,
                semantic_result.semantic_errors + semantic_result.type_errors,
                [f"Unused: {var}" for var in semantic_result.unused_variables]
            ))
        
        # Functional validation
        if context:
            functional_result = await self.validate_functionality(
                fix.fixed_code,
                context
            )
            validation_results.append(self._create_validation_result(
                "functional",
                functional_result.preserves_functionality,
                functional_result.behavior_changes,
                functional_result.api_changes
            ))
        
        # Style validation
        style_result = await self.validate_style(fix.fixed_code, fix.language)
        validation_results.append(self._create_validation_result(
            "style",
            style_result.follows_conventions,
            style_result.style_issues + style_result.naming_issues,
            style_result.formatting_issues
        ))
        
        # Security validation
        security_result = await self.validate_security(fix.fixed_code, fix.language)
        validation_results.append(self._create_validation_result(
            "security",
            security_result.is_secure,
            security_result.vulnerabilities,
            security_result.security_improvements
        ))
        
        return validation_results
    
    async def validate_syntax(
        self,
        code: str,
        language: ProgrammingLanguage
    ) -> SyntaxValidation:
        """Validate syntax of the code."""
        if language not in self.syntax_validators:
            raise UnsupportedLanguageError(f"No syntax validator for {language}")
        
        validator = self.syntax_validators[language]
        return await validator.validate(code)
    
    async def validate_semantics(
        self,
        code: str,
        context: FixContext
    ) -> SemanticValidation:
        """Validate semantic correctness."""
        return await self.semantic_analyzer.analyze(code, context)
    
    async def validate_functionality(
        self,
        code: str,
        context: FixContext
    ) -> FunctionalValidation:
        """Validate functional correctness."""
        return await self.functional_analyzer.analyze(code, context)
    
    async def validate_style(
        self,
        code: str,
        language: ProgrammingLanguage
    ) -> StyleValidation:
        """Validate code style and conventions."""
        return await self.style_checker.check(code, language)
    
    async def validate_security(
        self,
        code: str,
        language: ProgrammingLanguage
    ) -> SecurityValidation:
        """Validate security aspects."""
        return await self.security_scanner.scan(code, language)
    
    def _create_validation_result(
        self,
        validation_type: str,
        is_valid: bool,
        errors: List[str],
        warnings: List[str]
    ) -> ValidationResult:
        """Create a validation result object."""
        if is_valid and not warnings:
            status = ValidationStatus.VALID
        elif is_valid and warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.INVALID
        
        confidence = 1.0 if not errors else 0.0
        if warnings and not errors:
            confidence = 0.8
        
        return ValidationResult(
            validation_type=validation_type,
            status=status,
            errors=errors,
            warnings=warnings,
            confidence=confidence
        )
    
    async def validate_refactoring(
        self,
        result: RefactoringResult,
        context: Optional[Any] = None
    ) -> List[ValidationResult]:
        """Validate a refactoring result."""
        # Similar to validate_fix but with refactoring-specific checks
        validation_results = []
        
        # Basic syntax validation
        syntax_result = await self.validate_syntax(
            result.refactored_code,
            ProgrammingLanguage.PYTHON  # Default, should be determined from context
        )
        
        validation_results.append(self._create_validation_result(
            "syntax",
            syntax_result.is_valid,
            syntax_result.errors,
            syntax_result.warnings
        ))
        
        # Check if refactoring preserves behavior
        if context:
            # Additional refactoring-specific validations
            pass
        
        return validation_results


class PythonSyntaxValidator:
    """Python-specific syntax validator."""
    
    async def validate(self, code: str) -> SyntaxValidation:
        """Validate Python syntax."""
        try:
            # Parse the code using Python's ast module
            tree = ast.parse(code)
            
            # Additional checks
            warnings = []
            errors = []
            
            # Check for common issues
            visitor = PythonASTVisitor()
            visitor.visit(tree)
            
            warnings.extend(visitor.warnings)
            errors.extend(visitor.errors)
            
            return SyntaxValidation(
                is_valid=True,
                errors=errors,
                warnings=warnings
            )
            
        except SyntaxError as e:
            return SyntaxValidation(
                is_valid=False,
                errors=[f"Syntax error at line {e.lineno}: {e.msg}"],
                line_numbers=[e.lineno] if e.lineno else []
            )
        except Exception as e:
            return SyntaxValidation(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )


class PythonASTVisitor(ast.NodeVisitor):
    """AST visitor for Python-specific checks."""
    
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.defined_names = set()
        self.used_names = set()
    
    def visit_FunctionDef(self, node):
        """Check function definitions."""
        # Check for too many parameters
        if len(node.args.args) > 7:
            self.warnings.append(
                f"Function '{node.name}' has {len(node.args.args)} parameters (consider using fewer)"
            )
        
        # Check for missing docstring
        if not ast.get_docstring(node):
            self.warnings.append(f"Function '{node.name}' lacks a docstring")
        
        self.defined_names.add(node.name)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Check class definitions."""
        # Check for missing docstring
        if not ast.get_docstring(node):
            self.warnings.append(f"Class '{node.name}' lacks a docstring")
        
        self.defined_names.add(node.name)
        self.generic_visit(node)
    
    def visit_Name(self, node):
        """Track name usage."""
        if isinstance(node.ctx, ast.Store):
            self.defined_names.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)


class JavaScriptSyntaxValidator:
    """JavaScript-specific syntax validator."""
    
    async def validate(self, code: str) -> SyntaxValidation:
        """Validate JavaScript syntax."""
        # Simplified validation using regex patterns
        errors = []
        warnings = []
        
        # Check for basic syntax patterns
        if not self._check_balanced_braces(code):
            errors.append("Unbalanced braces detected")
        
        # Check for common issues
        if "var " in code:
            warnings.append("Consider using 'let' or 'const' instead of 'var'")
        
        # Check for missing semicolons (simplified)
        lines = code.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.endswith((';', '{', '}', ',')) and not line.startswith('//'):
                warnings.append(f"Line {i+1}: Consider adding semicolon")
        
        return SyntaxValidation(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _check_balanced_braces(self, code: str) -> bool:
        """Check if braces are balanced."""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in code:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack[-1]] != char:
                    return False
                stack.pop()
        
        return len(stack) == 0


class TypeScriptSyntaxValidator:
    """TypeScript-specific syntax validator."""
    
    async def validate(self, code: str) -> SyntaxValidation:
        """Validate TypeScript syntax."""
        # Inherit JavaScript validation and add TypeScript-specific checks
        js_validator = JavaScriptSyntaxValidator()
        result = await js_validator.validate(code)
        
        # Additional TypeScript checks
        if ": any" in code:
            result.warnings.append("Avoid using 'any' type - be more specific")
        
        return result


class JavaSyntaxValidator:
    """Java-specific syntax validator."""
    
    async def validate(self, code: str) -> SyntaxValidation:
        """Validate Java syntax."""
        errors = []
        warnings = []
        
        # Check for class declaration
        if "class " not in code and "interface " not in code:
            errors.append("No class or interface declaration found")
        
        # Check for main method if it's a main class
        if "public static void main" in code:
            if not re.search(r'public\s+static\s+void\s+main\s*\(\s*String\s*\[\]\s*\w+\s*\)', code):
                errors.append("Invalid main method signature")
        
        # Check for proper imports
        import_pattern = r'import\s+(?:static\s+)?[\w.]+(?:\.\*)?;'
        imports = re.findall(import_pattern, code)
        
        # Basic syntax checks
        if not self._check_balanced_braces(code):
            errors.append("Unbalanced braces detected")
        
        return SyntaxValidation(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _check_balanced_braces(self, code: str) -> bool:
        """Check if braces are balanced."""
        # Similar to JavaScript validator
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in code:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack or pairs[stack[-1]] != char:
                    return False
                stack.pop()
        
        return len(stack) == 0


class GoSyntaxValidator:
    """Go-specific syntax validator."""
    
    async def validate(self, code: str) -> SyntaxValidation:
        """Validate Go syntax."""
        errors = []
        warnings = []
        
        # Check for package declaration
        if not re.match(r'^\s*package\s+\w+', code, re.MULTILINE):
            errors.append("Missing package declaration")
        
        # Check for proper imports
        if "import" in code:
            # Go imports should be in parentheses for multiple imports
            if re.search(r'import\s+"[^"]+"', code) and code.count('import') > 1:
                warnings.append("Consider grouping imports in parentheses")
        
        # Check for exported names (capitalized)
        func_pattern = r'func\s+([a-z]\w*)\s*\('
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1)
            if func_name[0].islower():
                warnings.append(f"Function '{func_name}' is not exported (lowercase)")
        
        return SyntaxValidation(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class SemanticAnalyzer:
    """Analyzes semantic correctness of code."""
    
    async def analyze(self, code: str, context: FixContext) -> SemanticValidation:
        """Perform semantic analysis."""
        errors = []
        type_errors = []
        undefined_refs = []
        unused_vars = []
        
        if context.language == ProgrammingLanguage.PYTHON:
            # Python semantic analysis
            try:
                tree = ast.parse(code)
                analyzer = PythonSemanticAnalyzer()
                analyzer.analyze(tree)
                
                undefined_refs = analyzer.undefined_references
                unused_vars = analyzer.unused_variables
                type_errors = analyzer.type_errors
                
            except Exception as e:
                errors.append(f"Semantic analysis failed: {str(e)}")
        
        # Check for logical errors
        if context.original_code and code:
            # Compare original and fixed code semantics
            original_vars = self._extract_variables(context.original_code)
            fixed_vars = self._extract_variables(code)
            
            # Check for removed functionality
            removed_vars = original_vars - fixed_vars
            if removed_vars:
                errors.append(f"Variables removed: {', '.join(removed_vars)}")
        
        return SemanticValidation(
            is_valid=len(errors) == 0 and len(type_errors) == 0,
            semantic_errors=errors,
            type_errors=type_errors,
            undefined_references=undefined_refs,
            unused_variables=unused_vars
        )
    
    def _extract_variables(self, code: str) -> Set[str]:
        """Extract variable names from code."""
        # Simple regex-based extraction
        var_pattern = r'\b(\w+)\s*='
        return set(re.findall(var_pattern, code))


class PythonSemanticAnalyzer:
    """Python-specific semantic analyzer."""
    
    def __init__(self):
        self.defined_names = set()
        self.used_names = set()
        self.undefined_references = []
        self.unused_variables = []
        self.type_errors = []
        self.scope_stack = [set()]  # Stack of scopes
    
    def analyze(self, tree):
        """Analyze Python AST for semantic issues."""
        # First pass: collect definitions
        self._collect_definitions(tree)
        
        # Second pass: check usage
        self._check_usage(tree)
        
        # Find undefined references
        self.undefined_references = list(self.used_names - self.defined_names)
        
        # Find unused variables
        self.unused_variables = list(self.defined_names - self.used_names - {'self', '__init__'})
    
    def _collect_definitions(self, node):
        """Collect all defined names."""
        if isinstance(node, ast.FunctionDef):
            self.defined_names.add(node.name)
            # Add parameters
            for arg in node.args.args:
                self.defined_names.add(arg.arg)
        elif isinstance(node, ast.ClassDef):
            self.defined_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.defined_names.add(target.id)
        
        # Recurse
        for child in ast.iter_child_nodes(node):
            self._collect_definitions(child)
    
    def _check_usage(self, node):
        """Check name usage."""
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        
        # Recurse
        for child in ast.iter_child_nodes(node):
            self._check_usage(child)


class FunctionalAnalyzer:
    """Analyzes functional correctness."""
    
    async def analyze(self, code: str, context: FixContext) -> FunctionalValidation:
        """Analyze if functionality is preserved."""
        behavior_changes = []
        side_effects = []
        api_changes = []
        
        # Extract function signatures
        original_functions = self._extract_functions(context.original_code)
        fixed_functions = self._extract_functions(code)
        
        # Check for removed functions
        removed_functions = set(original_functions.keys()) - set(fixed_functions.keys())
        if removed_functions:
            api_changes.append(f"Functions removed: {', '.join(removed_functions)}")
        
        # Check for changed signatures
        for func_name in original_functions:
            if func_name in fixed_functions:
                if original_functions[func_name] != fixed_functions[func_name]:
                    api_changes.append(f"Function signature changed: {func_name}")
        
        # Check for behavioral changes (simplified)
        if len(code) < len(context.original_code) * 0.5:
            behavior_changes.append("Significant code reduction detected")
        
        # Check for side effects
        if "global " in code and "global " not in context.original_code:
            side_effects.append("New global variable usage detected")
        
        preserves_functionality = (
            len(behavior_changes) == 0 and
            len(api_changes) == 0
        )
        
        return FunctionalValidation(
            preserves_functionality=preserves_functionality,
            behavior_changes=behavior_changes,
            side_effects=side_effects,
            api_changes=api_changes
        )
    
    def _extract_functions(self, code: str) -> Dict[str, str]:
        """Extract function signatures from code."""
        functions = {}
        
        # Python function pattern
        func_pattern = r'def\s+(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1)
            params = match.group(2)
            functions[func_name] = params.strip()
        
        return functions


class StyleChecker:
    """Checks code style and conventions."""
    
    async def check(self, code: str, language: ProgrammingLanguage) -> StyleValidation:
        """Check code style."""
        style_issues = []
        naming_issues = []
        formatting_issues = []
        
        if language == ProgrammingLanguage.PYTHON:
            # PEP 8 style checks
            lines = code.splitlines()
            
            for i, line in enumerate(lines):
                # Line length
                if len(line) > 79:
                    formatting_issues.append(f"Line {i+1} exceeds 79 characters")
                
                # Trailing whitespace
                if line.endswith(' '):
                    formatting_issues.append(f"Line {i+1} has trailing whitespace")
            
            # Naming conventions
            # Check for camelCase (should be snake_case in Python)
            camel_case_pattern = r'\b[a-z]+[A-Z]\w*\b'
            camel_case_vars = re.findall(camel_case_pattern, code)
            for var in camel_case_vars:
                if var not in ['setUp', 'tearDown']:  # Common exceptions
                    naming_issues.append(f"Variable '{var}' should use snake_case")
            
            # Check for uppercase constants
            const_pattern = r'^[A-Z_]+\s*='
            if not re.search(const_pattern, code, re.MULTILINE):
                style_issues.append("No constants found (use UPPER_CASE for constants)")
        
        elif language == ProgrammingLanguage.JAVASCRIPT:
            # JavaScript style checks
            if not code.strip().endswith(';') and not code.strip().endswith('}'):
                formatting_issues.append("Code should end with semicolon or closing brace")
            
            # Check for camelCase
            snake_case_pattern = r'\b[a-z]+_[a-z]+\b'
            snake_case_vars = re.findall(snake_case_pattern, code)
            for var in snake_case_vars:
                naming_issues.append(f"Variable '{var}' should use camelCase")
        
        follows_conventions = (
            len(style_issues) == 0 and
            len(naming_issues) == 0 and
            len(formatting_issues) <= 2  # Allow minor formatting issues
        )
        
        return StyleValidation(
            follows_conventions=follows_conventions,
            style_issues=style_issues,
            naming_issues=naming_issues,
            formatting_issues=formatting_issues
        )


class SecurityScanner:
    """Scans code for security issues."""
    
    async def scan(self, code: str, language: ProgrammingLanguage) -> SecurityValidation:
        """Scan for security vulnerabilities."""
        vulnerabilities = []
        improvements = []
        
        # Common security patterns to check
        security_patterns = {
            'sql_injection': r'(query|execute)\s*\(\s*["\'].*?\+.*?["\']',
            'hardcoded_password': r'(password|passwd|pwd)\s*=\s*["\'][^"\']+["\']',
            'eval_usage': r'\beval\s*\(',
            'exec_usage': r'\bexec\s*\(',
            'pickle_usage': r'pickle\.loads?\s*\(',
            'shell_injection': r'os\.system\s*\(|subprocess\.call\s*\(',
        }
        
        for vuln_type, pattern in security_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                vulnerabilities.append(f"Potential {vuln_type.replace('_', ' ')}")
        
        # Check for improvements
        if 'parameterized' in code or '?' in code:
            improvements.append("Uses parameterized queries")
        
        if 'environ' in code or 'getenv' in code:
            improvements.append("Uses environment variables for configuration")
        
        # Determine risk level
        if len(vulnerabilities) >= 3:
            risk_level = "high"
        elif len(vulnerabilities) >= 1:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return SecurityValidation(
            is_secure=len(vulnerabilities) == 0,
            vulnerabilities=vulnerabilities,
            security_improvements=improvements,
            risk_level=risk_level
        )
