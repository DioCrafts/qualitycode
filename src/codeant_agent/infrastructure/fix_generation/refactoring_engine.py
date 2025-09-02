"""
Automated refactoring engine for code transformation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable
import asyncio
import logging
from datetime import datetime
import re
from abc import ABC, abstractmethod

from ...domain.entities.fix_generation import (
    RefactoringType, RefactoringConfig, RefactoringPlan,
    RefactoringImpact, RefactoringResult, CodeChange,
    ChangeType, ComplexityLevel, RiskLevel, FixExplanation,
    ValidationResult, ValidationStatus
)
from ...domain.entities.antipattern_analysis import (
    DetectedAntipattern, AntipatternType, UnifiedPosition
)
from ...domain.entities.language import ProgrammingLanguage
from .exceptions import (
    RefactoringError, RefactoringNotPossibleError,
    UnsafeRefactoringError
)


logger = logging.getLogger(__name__)


@dataclass
class RefactoringContext:
    """Context for refactoring operations."""
    original_code: str
    language: ProgrammingLanguage
    file_path: str
    issue: DetectedAntipattern
    project_structure: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    test_coverage: Optional[float] = None


@dataclass
class ExtractMethodDetails:
    """Details for extract method refactoring."""
    method_name: str
    start_position: UnifiedPosition
    end_position: UnifiedPosition
    parameters: List[Dict[str, str]]  # name, type
    return_type: Optional[str]
    extracted_code: str
    method_call: str


@dataclass
class ExtractClassDetails:
    """Details for extract class refactoring."""
    class_name: str
    extracted_methods: List[str]
    extracted_fields: List[str]
    file_path: Optional[str]  # If creating new file
    imports_needed: List[str]


@runtime_checkable
class RefactoringStrategy(Protocol):
    """Protocol for refactoring strategies."""
    
    def antipattern_type(self) -> AntipatternType:
        """Get the antipattern type this strategy handles."""
        ...
    
    async def can_refactor(
        self,
        issue: DetectedAntipattern,
        context: RefactoringContext
    ) -> bool:
        """Check if refactoring is possible."""
        ...
    
    async def generate_plan(
        self,
        issue: DetectedAntipattern,
        context: RefactoringContext
    ) -> RefactoringPlan:
        """Generate refactoring plan."""
        ...
    
    async def estimate_impact(
        self,
        plan: RefactoringPlan,
        context: RefactoringContext
    ) -> RefactoringImpact:
        """Estimate the impact of refactoring."""
        ...


class BaseRefactoringStrategy(ABC):
    """Base class for refactoring strategies."""
    
    @abstractmethod
    def antipattern_type(self) -> AntipatternType:
        """Get the antipattern type this strategy handles."""
        pass
    
    @abstractmethod
    async def can_refactor(
        self,
        issue: DetectedAntipattern,
        context: RefactoringContext
    ) -> bool:
        """Check if refactoring is possible."""
        pass
    
    @abstractmethod
    async def generate_plan(
        self,
        issue: DetectedAntipattern,
        context: RefactoringContext
    ) -> RefactoringPlan:
        """Generate refactoring plan."""
        pass
    
    async def estimate_impact(
        self,
        plan: RefactoringPlan,
        context: RefactoringContext
    ) -> RefactoringImpact:
        """Estimate the impact of refactoring."""
        # Default implementation
        return RefactoringImpact(
            complexity_reduction=10.0,
            maintainability_improvement=15.0,
            testability_improvement=20.0,
            reusability_increase=10.0,
            performance_impact="neutral",
            breaking_changes=False,
            affected_files=1,
            affected_functions=1,
            affected_classes=0,
            risk_level=RiskLevel.MEDIUM
        )


class ExtractMethodStrategy(BaseRefactoringStrategy):
    """Strategy for extract method refactoring."""
    
    def antipattern_type(self) -> AntipatternType:
        return AntipatternType.LONG_METHOD
    
    async def can_refactor(
        self,
        issue: DetectedAntipattern,
        context: RefactoringContext
    ) -> bool:
        """Check if method extraction is possible."""
        # Check if we can identify extractable code blocks
        extractable_blocks = await self._identify_extractable_blocks(context)
        return len(extractable_blocks) > 0
    
    async def generate_plan(
        self,
        issue: DetectedAntipattern,
        context: RefactoringContext
    ) -> RefactoringPlan:
        """Generate extract method refactoring plan."""
        # Identify the best block to extract
        extractable_blocks = await self._identify_extractable_blocks(context)
        
        if not extractable_blocks:
            raise RefactoringNotPossibleError("No extractable blocks found")
        
        # Select the best block (simplified - take the largest)
        best_block = max(extractable_blocks, key=lambda b: b['lines'])
        
        # Generate method name
        method_name = await self._generate_method_name(best_block, context)
        
        # Analyze parameters and return type
        params, return_type = await self._analyze_method_signature(best_block, context)
        
        steps = [
            f"Extract lines {best_block['start']}-{best_block['end']} into method '{method_name}'",
            f"Add method with parameters: {params}",
            f"Replace extracted code with method call",
            "Update any variable references",
            "Verify functionality is preserved"
        ]
        
        return RefactoringPlan(
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            description=f"Extract method '{method_name}' to reduce complexity",
            target_code=context.original_code,
            estimated_effort="low",
            confidence=0.85,
            complexity=ComplexityLevel.SIMPLE,
            affected_files=[context.file_path],
            preconditions=["Code block is self-contained", "No side effects"],
            steps=steps,
            expected_outcome="Reduced method complexity and improved readability"
        )
    
    async def _identify_extractable_blocks(
        self,
        context: RefactoringContext
    ) -> List[Dict[str, Any]]:
        """Identify code blocks that can be extracted."""
        blocks = []
        lines = context.original_code.splitlines()
        
        # Simple heuristic: look for blocks with consistent indentation
        current_block = None
        base_indent = None
        
        for i, line in enumerate(lines):
            if line.strip():  # Non-empty line
                indent = len(line) - len(line.lstrip())
                
                if current_block is None:
                    # Start new block
                    if indent > 0:  # Not at top level
                        current_block = {
                            'start': i + 1,
                            'indent': indent,
                            'lines': 1
                        }
                        base_indent = indent
                elif indent == base_indent:
                    # Continue block
                    current_block['lines'] += 1
                else:
                    # End block
                    if current_block['lines'] >= 3:  # Minimum extractable size
                        current_block['end'] = i
                        current_block['code'] = '\n'.join(
                            lines[current_block['start']-1:i]
                        )
                        blocks.append(current_block)
                    current_block = None
                    base_indent = None
        
        # Handle last block
        if current_block and current_block['lines'] >= 3:
            current_block['end'] = len(lines)
            current_block['code'] = '\n'.join(
                lines[current_block['start']-1:]
            )
            blocks.append(current_block)
        
        return blocks
    
    async def _generate_method_name(
        self,
        block: Dict[str, Any],
        context: RefactoringContext
    ) -> str:
        """Generate appropriate method name for extracted block."""
        # Simple heuristic based on code content
        code = block['code'].lower()
        
        if 'calculate' in code:
            return "calculate_result"
        elif 'validate' in code:
            return "validate_input"
        elif 'process' in code:
            return "process_data"
        elif 'format' in code:
            return "format_output"
        else:
            return f"extracted_method_{block['start']}"
    
    async def _analyze_method_signature(
        self,
        block: Dict[str, Any],
        context: RefactoringContext
    ) -> tuple[List[str], Optional[str]]:
        """Analyze required parameters and return type."""
        # Simplified analysis
        params = []
        
        # Look for variable usage in the block
        variable_pattern = r'\b(\w+)\b'
        used_vars = set(re.findall(variable_pattern, block['code']))
        
        # Filter to likely parameters (simplified)
        for var in used_vars:
            if var not in ['self', 'def', 'return', 'if', 'else', 'for', 'while']:
                params.append(var)
        
        # Detect if block has return statement
        return_type = "Any" if 'return' in block['code'] else None
        
        return params[:5], return_type  # Limit to 5 parameters


class ExtractClassStrategy(BaseRefactoringStrategy):
    """Strategy for extract class refactoring."""
    
    def antipattern_type(self) -> AntipatternType:
        return AntipatternType.GOD_OBJECT
    
    async def can_refactor(
        self,
        issue: DetectedAntipattern,
        context: RefactoringContext
    ) -> bool:
        """Check if class extraction is possible."""
        # Check if we can identify cohesive groups of methods/fields
        groups = await self._identify_cohesive_groups(context)
        return len(groups) > 0
    
    async def generate_plan(
        self,
        issue: DetectedAntipattern,
        context: RefactoringContext
    ) -> RefactoringPlan:
        """Generate extract class refactoring plan."""
        groups = await self._identify_cohesive_groups(context)
        
        if not groups:
            raise RefactoringNotPossibleError("No cohesive groups found for extraction")
        
        # Select the best group to extract
        best_group = max(groups, key=lambda g: len(g['methods']) + len(g['fields']))
        
        # Generate class name
        class_name = await self._generate_class_name(best_group, context)
        
        steps = [
            f"Create new class '{class_name}'",
            f"Move methods: {', '.join(best_group['methods'])}",
            f"Move fields: {', '.join(best_group['fields'])}",
            "Update original class to use the new class",
            "Add delegation methods if needed",
            "Update all references"
        ]
        
        return RefactoringPlan(
            refactoring_type=RefactoringType.EXTRACT_CLASS,
            description=f"Extract class '{class_name}' to reduce God Object",
            target_code=context.original_code,
            estimated_effort="medium",
            confidence=0.75,
            complexity=ComplexityLevel.MODERATE,
            affected_files=[context.file_path],
            preconditions=["Methods and fields are cohesive", "No circular dependencies"],
            steps=steps,
            expected_outcome="Reduced class complexity and improved single responsibility"
        )
    
    async def _identify_cohesive_groups(
        self,
        context: RefactoringContext
    ) -> List[Dict[str, Any]]:
        """Identify cohesive groups of methods and fields."""
        groups = []
        
        # Parse class structure (simplified)
        class_match = re.search(r'class\s+(\w+)', context.original_code)
        if not class_match:
            return groups
        
        # Extract methods
        method_pattern = r'def\s+(\w+)\s*\([^)]*\):'
        methods = re.findall(method_pattern, context.original_code)
        
        # Group methods by naming patterns (simplified heuristic)
        method_groups = {}
        
        for method in methods:
            # Skip special methods
            if method.startswith('__'):
                continue
            
            # Group by prefix
            prefix = method.split('_')[0] if '_' in method else method[:4]
            if prefix not in method_groups:
                method_groups[prefix] = []
            method_groups[prefix].append(method)
        
        # Create groups with at least 2 methods
        for prefix, method_list in method_groups.items():
            if len(method_list) >= 2:
                groups.append({
                    'name': prefix,
                    'methods': method_list,
                    'fields': [],  # Simplified - not analyzing fields
                    'cohesion_score': len(method_list) / len(methods)
                })
        
        return groups
    
    async def _generate_class_name(
        self,
        group: Dict[str, Any],
        context: RefactoringContext
    ) -> str:
        """Generate appropriate class name for extracted group."""
        prefix = group['name']
        
        # Common naming patterns
        if prefix in ['get', 'set']:
            return "DataAccessor"
        elif prefix == 'validate':
            return "Validator"
        elif prefix == 'calculate':
            return "Calculator"
        elif prefix == 'process':
            return "Processor"
        else:
            return f"{prefix.capitalize()}Handler"


class IntroduceParameterObjectStrategy(BaseRefactoringStrategy):
    """Strategy for introduce parameter object refactoring."""
    
    def antipattern_type(self) -> AntipatternType:
        return AntipatternType.LONG_PARAMETER_LIST
    
    async def can_refactor(
        self,
        issue: DetectedAntipattern,
        context: RefactoringContext
    ) -> bool:
        """Check if parameter object introduction is possible."""
        # Find methods with many parameters
        long_param_methods = await self._find_long_parameter_methods(context)
        return len(long_param_methods) > 0
    
    async def generate_plan(
        self,
        issue: DetectedAntipattern,
        context: RefactoringContext
    ) -> RefactoringPlan:
        """Generate parameter object introduction plan."""
        long_param_methods = await self._find_long_parameter_methods(context)
        
        if not long_param_methods:
            raise RefactoringNotPossibleError("No methods with long parameter lists found")
        
        # Select method with most parameters
        target_method = max(long_param_methods, key=lambda m: len(m['params']))
        
        # Generate parameter object name
        object_name = f"{target_method['name'].capitalize()}Parameters"
        
        steps = [
            f"Create parameter object class '{object_name}'",
            f"Add fields for parameters: {', '.join(target_method['params'])}",
            f"Update method '{target_method['name']}' to accept parameter object",
            "Update all method calls",
            "Consider adding builder pattern if needed"
        ]
        
        return RefactoringPlan(
            refactoring_type=RefactoringType.INTRODUCE_PARAMETER_OBJECT,
            description=f"Introduce parameter object for method '{target_method['name']}'",
            target_code=context.original_code,
            estimated_effort="low",
            confidence=0.80,
            complexity=ComplexityLevel.SIMPLE,
            affected_files=[context.file_path],
            preconditions=["Parameters are cohesive", "Method signature can be changed"],
            steps=steps,
            expected_outcome="Simplified method signature and improved readability"
        )
    
    async def _find_long_parameter_methods(
        self,
        context: RefactoringContext
    ) -> List[Dict[str, Any]]:
        """Find methods with long parameter lists."""
        methods = []
        
        # Pattern to match function definitions with parameters
        pattern = r'def\s+(\w+)\s*\((.*?)\):'
        
        for match in re.finditer(pattern, context.original_code, re.DOTALL):
            method_name = match.group(1)
            params_str = match.group(2)
            
            # Parse parameters (simplified)
            params = [p.strip() for p in params_str.split(',') if p.strip()]
            
            # Filter out 'self' for class methods
            params = [p for p in params if not p.startswith('self')]
            
            if len(params) >= 4:  # Threshold for long parameter list
                methods.append({
                    'name': method_name,
                    'params': params,
                    'position': match.start()
                })
        
        return methods


class AutomatedRefactoringEngine:
    """Main engine for automated refactoring."""
    
    def __init__(self, config: RefactoringConfig = None):
        self.config = config or RefactoringConfig()
        self.strategies = self._initialize_strategies()
        self.code_transformer = CodeTransformer()
        self.safety_checker = RefactoringSafetyChecker()
        
    def _initialize_strategies(self) -> Dict[AntipatternType, RefactoringStrategy]:
        """Initialize refactoring strategies."""
        strategies = {}
        
        # Register strategies
        for strategy_class in [
            ExtractMethodStrategy,
            ExtractClassStrategy,
            IntroduceParameterObjectStrategy
        ]:
            strategy = strategy_class()
            strategies[strategy.antipattern_type()] = strategy
        
        return strategies
    
    async def generate_refactoring(
        self,
        issue: DetectedAntipattern,
        context: RefactoringContext
    ) -> RefactoringResult:
        """Generate and execute refactoring for the issue."""
        start_time = datetime.now()
        
        try:
            # Check if we have a strategy
            if issue.pattern_type not in self.strategies:
                raise RefactoringError(f"No strategy available for {issue.pattern_type}")
            
            strategy = self.strategies[issue.pattern_type]
            
            # Check if refactoring is possible
            if not await strategy.can_refactor(issue, context):
                raise RefactoringNotPossibleError(
                    f"Cannot refactor {issue.pattern_type} in this context"
                )
            
            # Generate refactoring plan
            plan = await strategy.generate_plan(issue, context)
            
            # Estimate impact
            impact = await strategy.estimate_impact(plan, context)
            
            # Check safety
            safety_check = await self.safety_checker.check_safety(plan, context)
            
            if safety_check.risk_score > self.config.safety_threshold:
                raise UnsafeRefactoringError(
                    f"Refactoring risk too high: {safety_check.risk_score}",
                    safety_check.risks
                )
            
            # Execute refactoring
            refactored_code, changes = await self.code_transformer.apply_refactoring(
                plan,
                context
            )
            
            # Generate explanation
            explanation = await self._generate_explanation(plan, changes, impact)
            
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return RefactoringResult(
                plan_id=plan.id,
                refactoring_type=plan.refactoring_type,
                original_code=context.original_code,
                refactored_code=refactored_code,
                changes=changes,
                impact_analysis=impact,
                explanation=explanation,
                execution_time_ms=execution_time_ms,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Refactoring failed: {str(e)}")
            return RefactoringResult(
                refactoring_type=RefactoringType.EXTRACT_METHOD,
                original_code=context.original_code,
                refactored_code=context.original_code,
                success=False,
                error_message=str(e),
                execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
    
    async def _generate_explanation(
        self,
        plan: RefactoringPlan,
        changes: List[CodeChange],
        impact: RefactoringImpact
    ) -> FixExplanation:
        """Generate explanation for the refactoring."""
        summary = f"Applied {plan.refactoring_type.value} refactoring"
        
        detailed = f"""
Refactoring: {plan.description}

What was done:
{chr(10).join(f'- {step}' for step in plan.steps)}

Impact Analysis:
- Complexity reduction: {impact.complexity_reduction:.1f}%
- Maintainability improvement: {impact.maintainability_improvement:.1f}%
- Testability improvement: {impact.testability_improvement:.1f}%
- Risk level: {impact.risk_level.value}

Changes made:
{chr(10).join(f'- {change.description}' for change in changes[:5])}
"""
        
        return FixExplanation(
            summary=summary,
            detailed_explanation=detailed.strip(),
            changes_made=changes,
            why_this_fix=f"This refactoring improves code quality by addressing {plan.refactoring_type.value}",
            potential_impacts=[
                f"Affects {impact.affected_files} file(s)",
                f"Modifies {impact.affected_functions} function(s)",
                "Breaking changes: " + ("Yes" if impact.breaking_changes else "No")
            ],
            testing_recommendations=[
                "Run all existing tests",
                "Add tests for refactored components",
                "Verify functionality is preserved"
            ]
        )


class CodeTransformer:
    """Transforms code based on refactoring plans."""
    
    async def apply_refactoring(
        self,
        plan: RefactoringPlan,
        context: RefactoringContext
    ) -> tuple[str, List[CodeChange]]:
        """Apply refactoring transformations to code."""
        changes = []
        refactored_code = context.original_code
        
        if plan.refactoring_type == RefactoringType.EXTRACT_METHOD:
            refactored_code, method_changes = await self._apply_extract_method(
                plan,
                context
            )
            changes.extend(method_changes)
        elif plan.refactoring_type == RefactoringType.EXTRACT_CLASS:
            refactored_code, class_changes = await self._apply_extract_class(
                plan,
                context
            )
            changes.extend(class_changes)
        elif plan.refactoring_type == RefactoringType.INTRODUCE_PARAMETER_OBJECT:
            refactored_code, param_changes = await self._apply_parameter_object(
                plan,
                context
            )
            changes.extend(param_changes)
        
        return refactored_code, changes
    
    async def _apply_extract_method(
        self,
        plan: RefactoringPlan,
        context: RefactoringContext
    ) -> tuple[str, List[CodeChange]]:
        """Apply extract method refactoring."""
        # Simplified implementation
        changes = []
        
        # For demonstration, just add a comment
        refactored_code = f"""# Refactored: Extract Method Applied
# TODO: Implement actual method extraction

{context.original_code}"""
        
        changes.append(CodeChange(
            change_type=ChangeType.METHOD_EXTRACTED,
            location=UnifiedPosition(line=1, column=0),
            description="Extract method refactoring applied",
            original_content=context.original_code[:100],
            new_content=refactored_code[:100]
        ))
        
        return refactored_code, changes
    
    async def _apply_extract_class(
        self,
        plan: RefactoringPlan,
        context: RefactoringContext
    ) -> tuple[str, List[CodeChange]]:
        """Apply extract class refactoring."""
        # Simplified implementation
        changes = []
        
        refactored_code = f"""# Refactored: Extract Class Applied
# TODO: Implement actual class extraction

{context.original_code}"""
        
        changes.append(CodeChange(
            change_type=ChangeType.CLASS_ADDED,
            location=UnifiedPosition(line=1, column=0),
            description="Extract class refactoring applied",
            original_content=context.original_code[:100],
            new_content=refactored_code[:100]
        ))
        
        return refactored_code, changes
    
    async def _apply_parameter_object(
        self,
        plan: RefactoringPlan,
        context: RefactoringContext
    ) -> tuple[str, List[CodeChange]]:
        """Apply parameter object refactoring."""
        # Simplified implementation
        changes = []
        
        refactored_code = f"""# Refactored: Parameter Object Introduced
# TODO: Implement actual parameter object

{context.original_code}"""
        
        changes.append(CodeChange(
            change_type=ChangeType.STRUCTURAL_CHANGE,
            location=UnifiedPosition(line=1, column=0),
            description="Parameter object refactoring applied",
            original_content=context.original_code[:100],
            new_content=refactored_code[:100]
        ))
        
        return refactored_code, changes


@dataclass
class SafetyCheckResult:
    """Result of safety check."""
    is_safe: bool
    risk_score: float
    risks: List[str]
    recommendations: List[str]


class RefactoringSafetyChecker:
    """Checks safety of refactoring operations."""
    
    async def check_safety(
        self,
        plan: RefactoringPlan,
        context: RefactoringContext
    ) -> SafetyCheckResult:
        """Check if refactoring is safe to apply."""
        risks = []
        risk_score = 0.0
        
        # Check for breaking changes
        if plan.refactoring_type in [
            RefactoringType.EXTRACT_CLASS,
            RefactoringType.MOVE_METHOD
        ]:
            risks.append("May introduce breaking changes to public API")
            risk_score += 0.3
        
        # Check test coverage
        if context.test_coverage is not None and context.test_coverage < 0.5:
            risks.append(f"Low test coverage ({context.test_coverage*100:.0f}%)")
            risk_score += 0.2
        
        # Check complexity
        if plan.complexity == ComplexityLevel.VERY_COMPLEX:
            risks.append("Complex refactoring with high chance of errors")
            risk_score += 0.3
        
        # Generate recommendations
        recommendations = []
        if risk_score > 0.3:
            recommendations.append("Review changes carefully before applying")
            recommendations.append("Ensure comprehensive test coverage")
            recommendations.append("Consider applying in smaller steps")
        
        return SafetyCheckResult(
            is_safe=risk_score < 0.7,
            risk_score=risk_score,
            risks=risks,
            recommendations=recommendations
        )
