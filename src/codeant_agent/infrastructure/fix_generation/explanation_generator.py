"""
Generator for fix explanations and educational content.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import logging
from datetime import datetime
import re

from ...domain.entities.fix_generation import (
    GeneratedFix, FixExplanation, CodeChange, AlternativeApproach,
    ComplexityLevel, FixType, RefactoringType, ChangeType,
    RefactoringResult
)
from ...domain.entities.antipattern_analysis import AntipatternType, UnifiedPosition
from ...domain.entities.language import ProgrammingLanguage


logger = logging.getLogger(__name__)


@dataclass
class ExplanationTemplate:
    """Template for generating explanations."""
    template: str
    variables: Dict[str, str] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


class FixExplanationGenerator:
    """Generates comprehensive explanations for fixes."""
    
    def __init__(self):
        self.templates = self._load_explanation_templates()
        self.educational_content = self._load_educational_content()
        self.best_practices = self._load_best_practices()
        
    def _load_explanation_templates(self) -> Dict[AntipatternType, ExplanationTemplate]:
        """Load explanation templates for different antipattern types."""
        return {
            AntipatternType.SQL_INJECTION: ExplanationTemplate(
                template="""
SQL Injection Vulnerability Fixed

What was the problem?
{problem_description}

Why is this dangerous?
SQL injection allows attackers to execute arbitrary SQL commands, potentially:
- Accessing sensitive data
- Modifying or deleting data
- Bypassing authentication
- Taking control of the database server

How was it fixed?
{fix_description}

Key improvements:
- User input is now properly sanitized
- Parameterized queries prevent SQL command injection
- Database operations are safer and more secure
""",
                references=[
                    "OWASP SQL Injection Prevention Cheat Sheet",
                    "CWE-89: SQL Injection",
                    "SQL Injection Prevention Best Practices"
                ]
            ),
            
            AntipatternType.HARDCODED_SECRETS: ExplanationTemplate(
                template="""
Hardcoded Secrets Removed

What was the problem?
{problem_description}

Security implications:
- Secrets exposed in source code
- Risk of unauthorized access if code is leaked
- Difficult to rotate credentials
- Poor security practice

How was it fixed?
{fix_description}

Best practices applied:
- Secrets moved to environment variables
- Configuration externalized from code
- Easier credential rotation
- Follows the principle of least privilege
""",
                references=[
                    "OWASP Secrets Management Cheat Sheet",
                    "12-Factor App: Config",
                    "Security Best Practices for API Keys"
                ]
            ),
            
            AntipatternType.GOD_OBJECT: ExplanationTemplate(
                template="""
God Object Refactored

What was the problem?
{problem_description}

Design issues:
- Single class doing too much (violates SRP)
- High coupling and low cohesion
- Difficult to test and maintain
- Hard to understand and modify

How was it fixed?
{fix_description}

Design improvements:
- Responsibilities separated into focused classes
- Each class now has a single responsibility
- Improved testability and maintainability
- Better code organization
""",
                references=[
                    "SOLID Principles: Single Responsibility",
                    "Refactoring: Improving the Design of Existing Code",
                    "Clean Code: A Handbook of Agile Software Craftsmanship"
                ]
            ),
            
            AntipatternType.N_PLUS_ONE_QUERY: ExplanationTemplate(
                template="""
N+1 Query Problem Resolved

What was the problem?
{problem_description}

Performance impact:
- Excessive database queries (1 + N queries)
- Poor application performance
- Increased database load
- Higher latency for users

How was it fixed?
{fix_description}

Optimization benefits:
- Single optimized query instead of N+1
- Significantly improved performance
- Reduced database load
- Better scalability
""",
                references=[
                    "ORM Performance Anti-Patterns",
                    "Database Query Optimization",
                    "Eager Loading vs Lazy Loading"
                ]
            ),
            
            AntipatternType.LONG_METHOD: ExplanationTemplate(
                template="""
Long Method Decomposed

What was the problem?
{problem_description}

Maintainability issues:
- Method too complex to understand
- Multiple responsibilities in one method
- Difficult to test individual parts
- High cognitive load

How was it fixed?
{fix_description}

Code quality improvements:
- Method broken into smaller, focused methods
- Each method has a clear purpose
- Improved readability and testability
- Easier to understand and modify
""",
                references=[
                    "Clean Code: Functions",
                    "Refactoring: Extract Method",
                    "Code Complete: High-Quality Routines"
                ]
            )
        }
    
    def _load_educational_content(self) -> Dict[str, str]:
        """Load educational content for different concepts."""
        return {
            "parameterized_queries": """
Parameterized queries (also called prepared statements) separate SQL logic from data.
Instead of concatenating user input into SQL strings, placeholders are used.
The database engine ensures data is properly escaped, preventing injection attacks.

Example:
BAD:  query = "SELECT * FROM users WHERE id = " + user_id
GOOD: query = "SELECT * FROM users WHERE id = ?"
      execute(query, [user_id])
""",
            
            "environment_variables": """
Environment variables store configuration outside your codebase.
This follows the 12-factor app methodology for better security and portability.

Benefits:
- Secrets never committed to version control
- Different configs for different environments
- Easy to change without code modifications
- Better security posture

Example:
BAD:  api_key = "sk-1234567890abcdef"
GOOD: api_key = os.environ.get('API_KEY')
""",
            
            "single_responsibility": """
The Single Responsibility Principle (SRP) states that a class should have only one reason to change.
This leads to more maintainable, testable, and understandable code.

Benefits:
- Easier to understand each class's purpose
- Changes are localized to specific classes
- Better testability with focused responsibilities
- Reduced coupling between components
""",
            
            "eager_loading": """
Eager loading fetches related data in a single query, preventing N+1 problems.
Instead of loading related data on-demand (lazy loading), it's loaded upfront.

Example:
BAD:  users = User.all()
      for user in users:
          print(user.posts)  # N queries

GOOD: users = User.includes(:posts).all()  # 1 query
      for user in users:
          print(user.posts)  # No additional queries
"""
        }
    
    def _load_best_practices(self) -> Dict[str, List[str]]:
        """Load best practices for different scenarios."""
        return {
            "security": [
                "Never trust user input - always validate and sanitize",
                "Use parameterized queries for all database operations",
                "Store secrets in environment variables or secret management systems",
                "Apply the principle of least privilege",
                "Keep dependencies updated for security patches"
            ],
            
            "design": [
                "Follow SOLID principles for better design",
                "Keep classes and methods small and focused",
                "Favor composition over inheritance",
                "Write code that is easy to test",
                "Refactor regularly to maintain code quality"
            ],
            
            "performance": [
                "Profile before optimizing",
                "Use appropriate data structures",
                "Minimize database queries",
                "Cache expensive computations",
                "Consider async operations for I/O"
            ],
            
            "maintainability": [
                "Write self-documenting code",
                "Keep functions small and focused",
                "Use meaningful names",
                "Avoid deep nesting",
                "Follow consistent coding standards"
            ]
        }
    
    async def generate_explanation(
        self,
        fix: GeneratedFix,
        context: Optional[Dict[str, Any]] = None
    ) -> FixExplanation:
        """Generate comprehensive explanation for a fix."""
        # Get template for the antipattern type
        antipattern_type = self._determine_antipattern_type(fix)
        template = self.templates.get(antipattern_type)
        
        if not template:
            # Fallback to generic explanation
            return await self._generate_generic_explanation(fix)
        
        # Generate components
        summary = self._generate_summary(fix, antipattern_type)
        detailed = self._generate_detailed_explanation(fix, template, context)
        changes = self._analyze_changes(fix)
        why = self._explain_why_this_fix(fix, antipattern_type)
        impacts = self._analyze_potential_impacts(fix)
        testing = self._generate_testing_recommendations(fix)
        alternatives = self._suggest_alternatives(fix, antipattern_type)
        educational = self._generate_educational_content(fix, antipattern_type)
        
        return FixExplanation(
            summary=summary,
            detailed_explanation=detailed,
            changes_made=changes,
            why_this_fix=why,
            potential_impacts=impacts,
            testing_recommendations=testing,
            alternative_approaches=alternatives,
            educational_content=educational,
            references=template.references if template else []
        )
    
    def _determine_antipattern_type(self, fix: GeneratedFix) -> Optional[AntipatternType]:
        """Determine antipattern type from fix."""
        # Simple heuristic based on fix type and content
        if "sql" in fix.fixed_code.lower() and "parameterized" in fix.fixed_code.lower():
            return AntipatternType.SQL_INJECTION
        elif "environ" in fix.fixed_code or "getenv" in fix.fixed_code:
            return AntipatternType.HARDCODED_SECRETS
        elif fix.fix_type == FixType.REFACTORING and "class" in fix.fixed_code:
            return AntipatternType.GOD_OBJECT
        elif "eager" in fix.fixed_code.lower() or "prefetch" in fix.fixed_code.lower():
            return AntipatternType.N_PLUS_ONE_QUERY
        else:
            return None
    
    def _generate_summary(self, fix: GeneratedFix, antipattern_type: Optional[AntipatternType]) -> str:
        """Generate a concise summary."""
        if antipattern_type:
            summaries = {
                AntipatternType.SQL_INJECTION: "Fixed SQL injection vulnerability using parameterized queries",
                AntipatternType.HARDCODED_SECRETS: "Removed hardcoded secrets and moved to environment variables",
                AntipatternType.GOD_OBJECT: "Refactored God Object to improve design and maintainability",
                AntipatternType.N_PLUS_ONE_QUERY: "Optimized database queries to eliminate N+1 problem",
                AntipatternType.LONG_METHOD: "Decomposed long method into smaller, focused methods"
            }
            return summaries.get(antipattern_type, f"Applied fix for {antipattern_type.value}")
        else:
            return f"Applied {fix.fix_type.value} fix to improve code quality"
    
    def _generate_detailed_explanation(
        self,
        fix: GeneratedFix,
        template: ExplanationTemplate,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate detailed explanation using template."""
        # Extract problem description
        problem_desc = "The code contained a vulnerability that could be exploited."
        if context and "issue_description" in context:
            problem_desc = context["issue_description"]
        
        # Extract fix description
        fix_desc = self._describe_fix_changes(fix)
        
        # Fill template
        explanation = template.template.format(
            problem_description=problem_desc,
            fix_description=fix_desc
        )
        
        return explanation.strip()
    
    def _describe_fix_changes(self, fix: GeneratedFix) -> str:
        """Describe what changes were made."""
        descriptions = []
        
        if fix.diff and fix.diff.stats:
            stats = fix.diff.stats
            descriptions.append(
                f"Modified {stats.lines_modified} lines, "
                f"added {stats.lines_added} lines, "
                f"removed {stats.lines_removed} lines"
            )
        
        # Analyze specific changes
        if "parameterized" in fix.fixed_code.lower():
            descriptions.append("Implemented parameterized queries")
        if "environ" in fix.fixed_code:
            descriptions.append("Moved configuration to environment variables")
        if fix.fix_type == FixType.REFACTORING:
            descriptions.append("Refactored code structure for better organization")
        
        return " | ".join(descriptions) if descriptions else "Applied targeted fixes to resolve the issue"
    
    def _analyze_changes(self, fix: GeneratedFix) -> List[CodeChange]:
        """Analyze and describe code changes."""
        # If we have actual changes from the fix, use those
        if hasattr(fix, 'changes') and fix.changes:
            return fix.changes
        
        # Otherwise, generate high-level change descriptions
        changes = []
        
        # Analyze based on fix type
        if fix.fix_type == FixType.DIRECT_REPLACEMENT:
            changes.append(CodeChange(
                change_type=ChangeType.LOGIC_CHANGE,
                location=UnifiedPosition(line=1, column=0),
                description="Direct code replacement to fix the issue",
                original_content=fix.original_code[:100] + "...",
                new_content=fix.fixed_code[:100] + "..."
            ))
        elif fix.fix_type == FixType.REFACTORING:
            changes.append(CodeChange(
                change_type=ChangeType.STRUCTURAL_CHANGE,
                location=UnifiedPosition(line=1, column=0),
                description="Code structure refactored for better design",
                original_content="Original structure",
                new_content="Refactored structure"
            ))
        
        return changes
    
    def _explain_why_this_fix(self, fix: GeneratedFix, antipattern_type: Optional[AntipatternType]) -> str:
        """Explain why this specific fix was chosen."""
        if antipattern_type == AntipatternType.SQL_INJECTION:
            return """
This fix uses parameterized queries because they:
- Completely prevent SQL injection by separating code from data
- Are the industry standard for secure database operations
- Maintain query performance while ensuring security
- Are supported by all major database systems
"""
        elif antipattern_type == AntipatternType.HARDCODED_SECRETS:
            return """
Environment variables were chosen because they:
- Keep secrets out of source code
- Allow different values for different environments
- Enable easy credential rotation
- Follow security best practices and compliance requirements
"""
        elif antipattern_type == AntipatternType.GOD_OBJECT:
            return """
Class extraction was used because it:
- Separates concerns into focused, manageable classes
- Improves code testability and maintainability
- Makes the codebase easier to understand
- Allows independent evolution of different features
"""
        else:
            return "This approach was selected as the most effective solution for the identified issue."
    
    def _analyze_potential_impacts(self, fix: GeneratedFix) -> List[str]:
        """Analyze potential impacts of the fix."""
        impacts = []
        
        # General impacts
        impacts.append("Code behavior preserved with improved implementation")
        
        # Specific impacts based on fix type
        if fix.fix_type == FixType.REFACTORING:
            impacts.extend([
                "API compatibility maintained",
                "Internal structure improved",
                "May require updating imports in dependent code"
            ])
        elif fix.fix_type == FixType.DIRECT_REPLACEMENT:
            impacts.append("Minimal impact on surrounding code")
        
        # Security improvements
        if fix.risk_level.value in ["low", "medium"]:
            impacts.append("Security posture improved")
        
        # Performance impacts
        antipattern_type = self._determine_antipattern_type(fix)
        if antipattern_type == AntipatternType.N_PLUS_ONE_QUERY:
            impacts.append("Significant performance improvement expected")
        
        return impacts
    
    def _generate_testing_recommendations(self, fix: GeneratedFix) -> List[str]:
        """Generate testing recommendations."""
        recommendations = [
            "Run all existing unit tests to ensure no regressions",
            "Add new tests to cover the fixed functionality",
            "Perform integration testing with dependent components"
        ]
        
        # Specific recommendations based on fix type
        antipattern_type = self._determine_antipattern_type(fix)
        
        if antipattern_type == AntipatternType.SQL_INJECTION:
            recommendations.extend([
                "Test with various SQL injection patterns to verify protection",
                "Verify parameterized queries work with all input types",
                "Test edge cases like null values and special characters"
            ])
        elif antipattern_type == AntipatternType.HARDCODED_SECRETS:
            recommendations.extend([
                "Verify environment variables are properly loaded",
                "Test behavior when environment variables are missing",
                "Ensure no secrets remain in the codebase"
            ])
        elif antipattern_type == AntipatternType.GOD_OBJECT:
            recommendations.extend([
                "Test each extracted class independently",
                "Verify interactions between refactored classes",
                "Ensure all original functionality is preserved"
            ])
        elif antipattern_type == AntipatternType.N_PLUS_ONE_QUERY:
            recommendations.extend([
                "Benchmark query performance before and after",
                "Verify data is correctly loaded with eager loading",
                "Test with various data sizes to ensure scalability"
            ])
        
        return recommendations
    
    def _suggest_alternatives(self, fix: GeneratedFix, antipattern_type: Optional[AntipatternType]) -> List[AlternativeApproach]:
        """Suggest alternative approaches."""
        alternatives = []
        
        if antipattern_type == AntipatternType.SQL_INJECTION:
            alternatives.append(AlternativeApproach(
                approach_name="Stored Procedures",
                description="Use database stored procedures for complex queries",
                pros=["Additional security layer", "Better performance for complex operations"],
                cons=["Database-specific", "Harder to version control"],
                complexity=ComplexityLevel.MODERATE
            ))
            alternatives.append(AlternativeApproach(
                approach_name="ORM with Query Builder",
                description="Use an ORM that provides safe query building",
                pros=["Type safety", "Database abstraction", "Prevents most injection attacks"],
                cons=["Learning curve", "May be overkill for simple queries"],
                complexity=ComplexityLevel.COMPLEX
            ))
        
        elif antipattern_type == AntipatternType.HARDCODED_SECRETS:
            alternatives.append(AlternativeApproach(
                approach_name="Secret Management Service",
                description="Use a dedicated service like HashiCorp Vault or AWS Secrets Manager",
                pros=["Enterprise-grade security", "Audit trails", "Automatic rotation"],
                cons=["Additional infrastructure", "Complexity", "Cost"],
                complexity=ComplexityLevel.COMPLEX
            ))
            alternatives.append(AlternativeApproach(
                approach_name="Configuration Files",
                description="Use encrypted configuration files",
                pros=["Simple to implement", "Version controllable (encrypted)"],
                cons=["Need to manage encryption keys", "Less flexible than env vars"],
                complexity=ComplexityLevel.MODERATE
            ))
        
        elif antipattern_type == AntipatternType.GOD_OBJECT:
            alternatives.append(AlternativeApproach(
                approach_name="Facade Pattern",
                description="Keep the God Object but create a facade for cleaner interface",
                pros=["Minimal changes to existing code", "Gradual refactoring possible"],
                cons=["Doesn't solve underlying problem", "Adds another layer"],
                complexity=ComplexityLevel.SIMPLE
            ))
            alternatives.append(AlternativeApproach(
                approach_name="Microservices",
                description="Break into separate microservices",
                pros=["Ultimate separation", "Independent scaling", "Technology diversity"],
                cons=["Significant complexity", "Network overhead", "Operational challenges"],
                complexity=ComplexityLevel.VERY_COMPLEX
            ))
        
        return alternatives
    
    def _generate_educational_content(self, fix: GeneratedFix, antipattern_type: Optional[AntipatternType]) -> Optional[str]:
        """Generate educational content about the fix."""
        if not antipattern_type:
            return None
        
        # Map antipatterns to educational topics
        topic_map = {
            AntipatternType.SQL_INJECTION: "parameterized_queries",
            AntipatternType.HARDCODED_SECRETS: "environment_variables",
            AntipatternType.GOD_OBJECT: "single_responsibility",
            AntipatternType.N_PLUS_ONE_QUERY: "eager_loading"
        }
        
        topic = topic_map.get(antipattern_type)
        if topic and topic in self.educational_content:
            content = self.educational_content[topic]
            
            # Add relevant best practices
            if antipattern_type in [AntipatternType.SQL_INJECTION, AntipatternType.HARDCODED_SECRETS]:
                practices = self.best_practices.get("security", [])
            elif antipattern_type == AntipatternType.GOD_OBJECT:
                practices = self.best_practices.get("design", [])
            else:
                practices = self.best_practices.get("performance", [])
            
            content += "\n\nBest Practices:\n"
            content += "\n".join(f"- {practice}" for practice in practices[:3])
            
            return content
        
        return None
    
    async def _generate_generic_explanation(self, fix: GeneratedFix) -> FixExplanation:
        """Generate generic explanation when no specific template exists."""
        summary = f"Applied {fix.fix_type.value} to improve code quality"
        
        detailed = f"""
Code Improvement Applied

What changed:
The code has been modified to address identified issues and improve overall quality.

Type of fix: {fix.fix_type.value}
Confidence level: {fix.confidence_level.value}
Risk level: {fix.risk_level.value}

The changes focus on:
- Improving code structure and organization
- Following best practices
- Enhancing maintainability
- Reducing potential issues
"""
        
        return FixExplanation(
            summary=summary,
            detailed_explanation=detailed.strip(),
            why_this_fix="This fix addresses identified code quality issues",
            potential_impacts=["Improved code quality", "Better maintainability"],
            testing_recommendations=["Test modified functionality", "Run regression tests"]
        )
    
    async def generate_refactoring_explanation(
        self,
        result: RefactoringResult
    ) -> FixExplanation:
        """Generate explanation for refactoring results."""
        summary = f"Applied {result.refactoring_type.value} refactoring"
        
        detailed = f"""
Refactoring Applied: {result.refactoring_type.value}

What was refactored:
{self._describe_refactoring(result)}

Impact Analysis:
- Complexity reduction: {result.impact_analysis.complexity_reduction:.1f}%
- Maintainability improvement: {result.impact_analysis.maintainability_improvement:.1f}%
- Testability improvement: {result.impact_analysis.testability_improvement:.1f}%

The refactoring improves code quality by:
- Applying proven design patterns
- Reducing code complexity
- Improving separation of concerns
- Making the code easier to understand and modify
"""
        
        return FixExplanation(
            summary=summary,
            detailed_explanation=detailed.strip(),
            changes_made=result.changes,
            why_this_fix=f"This refactoring improves code structure using {result.refactoring_type.value}",
            potential_impacts=[
                f"Affects {result.impact_analysis.affected_files} file(s)",
                f"Modifies {result.impact_analysis.affected_functions} function(s)",
                "Breaking changes: " + ("Yes" if result.impact_analysis.breaking_changes else "No")
            ],
            testing_recommendations=[
                "Test all refactored components",
                "Verify behavior is preserved",
                "Update documentation if needed"
            ]
        )
    
    def _describe_refactoring(self, result: RefactoringResult) -> str:
        """Describe what was refactored."""
        descriptions = {
            RefactoringType.EXTRACT_METHOD: "Extracted complex code into a separate method for better organization",
            RefactoringType.EXTRACT_CLASS: "Extracted related functionality into a new class following Single Responsibility Principle",
            RefactoringType.INTRODUCE_PARAMETER_OBJECT: "Replaced long parameter list with a parameter object",
            RefactoringType.MOVE_METHOD: "Moved method to more appropriate class based on feature envy",
            RefactoringType.INLINE_METHOD: "Inlined simple method to reduce unnecessary abstraction"
        }
        
        return descriptions.get(
            result.refactoring_type,
            f"Applied {result.refactoring_type.value} refactoring pattern"
        )
