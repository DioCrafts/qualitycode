"""
Implementación del generador de recomendaciones de fixes.

Este módulo implementa la generación automática de recomendaciones
de fixes, análisis de automatización y guidance de implementación.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import time

from ...domain.entities.issue_management import (
    CategorizedIssue, IssueCluster, FixPlan, FixType, IssueCategory,
    ImplementationStep, TestingStrategy
)

logger = logging.getLogger(__name__)


@dataclass
class FixRecommendation:
    """Recomendación de fix."""
    fix_id: str
    title: str
    description: str
    fix_type: FixType
    automation_potential: float  # 0-1
    complexity_score: float  # 0-100
    effort_estimate_hours: float
    confidence_level: float  # 0-1
    implementation_guidance: List[str]
    code_examples: List[str] = field(default_factory=list)
    tool_suggestions: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)


@dataclass
class AutomationAssessment:
    """Evaluación de potencial de automatización."""
    automatable_percentage: float  # 0-100
    recommended_tools: List[str]
    automation_benefits: List[str]
    automation_risks: List[str]
    implementation_complexity: str  # "low", "medium", "high"


class FixRecommendationEngine:
    """Motor de recomendaciones de fixes."""
    
    def __init__(self):
        self.fix_templates = self._initialize_fix_templates()
        self.automation_tools = self._initialize_automation_tools()
    
    def _initialize_fix_templates(self) -> Dict[IssueCategory, Dict[str, Any]]:
        """Inicializa templates de fix por categoría."""
        return {
            IssueCategory.SECURITY: {
                "sql_injection": {
                    "title": "Fix SQL Injection Vulnerability",
                    "description": "Implement parameterized queries and input validation",
                    "steps": [
                        "Replace string concatenation with parameterized queries",
                        "Add input validation and sanitization",
                        "Implement proper error handling",
                        "Add security tests"
                    ],
                    "code_examples": [
                        "# Before: query = f\"SELECT * FROM users WHERE id = {user_id}\"",
                        "# After: cursor.execute(\"SELECT * FROM users WHERE id = %s\", (user_id,))"
                    ],
                    "tools": ["SQLAlchemy", "parameterized queries", "input validators"]
                },
                "hardcoded_secrets": {
                    "title": "Remove Hardcoded Secrets",
                    "description": "Move secrets to secure configuration management",
                    "steps": [
                        "Identify all hardcoded secrets",
                        "Move to environment variables or secret manager",
                        "Update code to read from secure source",
                        "Add secret scanning to CI/CD"
                    ],
                    "tools": ["environment variables", "AWS Secrets Manager", "HashiCorp Vault"]
                }
            },
            IssueCategory.PERFORMANCE: {
                "high_complexity": {
                    "title": "Reduce Algorithmic Complexity", 
                    "description": "Optimize algorithms and data structures",
                    "steps": [
                        "Profile code to identify bottlenecks",
                        "Analyze algorithm complexity",
                        "Implement more efficient algorithms",
                        "Add performance tests"
                    ],
                    "tools": ["profilers", "performance monitors", "benchmarking tools"]
                },
                "memory_leak": {
                    "title": "Fix Memory Leak",
                    "description": "Identify and resolve memory leaks",
                    "steps": [
                        "Use memory profiling tools",
                        "Identify leak sources",
                        "Implement proper resource management",
                        "Add memory monitoring"
                    ],
                    "tools": ["memory profilers", "leak detectors", "monitoring tools"]
                }
            },
            IssueCategory.MAINTAINABILITY: {
                "code_duplication": {
                    "title": "Eliminate Code Duplication",
                    "description": "Extract common functionality to reduce duplication",
                    "steps": [
                        "Identify duplicated code patterns",
                        "Extract common functionality",
                        "Create reusable functions/classes",
                        "Update all call sites"
                    ],
                    "tools": ["refactoring tools", "IDE extractors", "static analysis"]
                },
                "high_complexity": {
                    "title": "Reduce Function Complexity",
                    "description": "Break down complex functions into smaller units",
                    "steps": [
                        "Identify complex functions",
                        "Extract smaller functions",
                        "Simplify conditional logic",
                        "Add unit tests for each function"
                    ],
                    "tools": ["complexity analyzers", "refactoring tools"]
                }
            },
            IssueCategory.RELIABILITY: {
                "error_handling": {
                    "title": "Improve Error Handling",
                    "description": "Add comprehensive error handling",
                    "steps": [
                        "Identify error-prone code paths",
                        "Add try-catch blocks",
                        "Implement proper error logging",
                        "Add recovery mechanisms"
                    ],
                    "tools": ["exception handling frameworks", "logging libraries"]
                }
            },
            IssueCategory.DOCUMENTATION: {
                "missing_docs": {
                    "title": "Add Missing Documentation",
                    "description": "Create comprehensive documentation",
                    "steps": [
                        "Analyze code to understand functionality",
                        "Write clear docstrings/comments",
                        "Create usage examples",
                        "Update API documentation"
                    ],
                    "tools": ["documentation generators", "markdown editors", "API doc tools"]
                }
            }
        }
    
    def _initialize_automation_tools(self) -> Dict[IssueCategory, List[str]]:
        """Inicializa herramientas de automatización por categoría."""
        return {
            IssueCategory.SECURITY: [
                "Bandit (Python security linter)",
                "ESLint security plugins",
                "SonarQube security rules",
                "Semgrep security patterns"
            ],
            IssueCategory.PERFORMANCE: [
                "Performance profilers",
                "Code complexity analyzers",
                "Benchmarking frameworks",
                "Memory analyzers"
            ],
            IssueCategory.MAINTAINABILITY: [
                "Refactoring tools (IDE)",
                "Code duplication detectors",
                "Complexity analyzers",
                "Design pattern extractors"
            ],
            IssueCategory.CODE_STYLE: [
                "Formatters (Black, Prettier)",
                "Linters (ESLint, Pylint)",
                "Style checkers",
                "Auto-fixers"
            ],
            IssueCategory.DOCUMENTATION: [
                "Documentation generators",
                "Docstring tools",
                "API documentation tools",
                "Comment generators"
            ]
        }
    
    async def generate_fix_recommendations(self, issues: List[CategorizedIssue]) -> List[FixRecommendation]:
        """
        Genera recomendaciones de fix para lista de issues.
        
        Args:
            issues: Lista de issues categorizados
            
        Returns:
            Lista de recomendaciones de fix
        """
        recommendations = []
        
        # Agrupar issues por categoría para recomendaciones batch
        category_groups = {}
        for issue in issues:
            category = issue.primary_category
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(issue)
        
        # Generar recomendaciones por grupo
        for category, category_issues in category_groups.items():
            if len(category_issues) >= 3:
                # Recomendación batch
                batch_recommendation = await self._generate_batch_recommendation(category, category_issues)
                recommendations.append(batch_recommendation)
            else:
                # Recomendaciones individuales
                for issue in category_issues:
                    individual_recommendation = await self._generate_individual_recommendation(issue)
                    recommendations.append(individual_recommendation)
        
        # Ordenar por valor (automation potential + inverse complexity)
        recommendations.sort(key=lambda r: r.automation_potential - r.complexity_score / 100.0, reverse=True)
        
        return recommendations
    
    async def _generate_batch_recommendation(self, category: IssueCategory, issues: List[CategorizedIssue]) -> FixRecommendation:
        """Genera recomendación para batch de issues similares."""
        # Obtener template para la categoría
        category_templates = self.fix_templates.get(category, {})
        
        # Determinar template más apropiado
        template_key = self._select_best_template(issues, category_templates)
        template = category_templates.get(template_key, {})
        
        # Calcular métricas agregadas
        total_effort = sum(issue.metadata.estimated_fix_time_hours or 2.0 for issue in issues)
        avg_complexity = sum(issue.metadata.fix_complexity_score for issue in issues) / len(issues)
        
        # Batch efficiency (20-40% savings)
        batch_efficiency = 0.3
        adjusted_effort = total_effort * (1.0 - batch_efficiency)
        
        # Calcular automation potential
        automation_potential = await self._assess_automation_potential(category, issues)
        
        # Generar guidance
        implementation_guidance = template.get("steps", [
            f"Analyze all {len(issues)} {category.value} issues",
            "Create unified fix approach",
            "Implement fixes systematically",
            "Test all changes together"
        ])
        
        return FixRecommendation(
            fix_id=f"batch_{category.value}_{len(issues)}",
            title=template.get("title", f"Batch Fix: {len(issues)} {category.value} Issues"),
            description=template.get("description", f"Systematic approach to resolve {len(issues)} {category.value} issues"),
            fix_type=self._category_to_fix_type(category),
            automation_potential=automation_potential,
            complexity_score=avg_complexity,
            effort_estimate_hours=adjusted_effort,
            confidence_level=0.8,  # High confidence for batch fixes
            implementation_guidance=implementation_guidance,
            code_examples=template.get("code_examples", []),
            tool_suggestions=template.get("tools", []),
            risk_warnings=self._generate_risk_warnings(category, len(issues))
        )
    
    async def _generate_individual_recommendation(self, issue: CategorizedIssue) -> FixRecommendation:
        """Genera recomendación para issue individual."""
        category = issue.primary_category
        category_templates = self.fix_templates.get(category, {})
        
        # Seleccionar template
        template_key = self._select_template_for_issue(issue, category_templates)
        template = category_templates.get(template_key, {})
        
        # Automation potential para issue individual
        automation_potential = await self._assess_individual_automation_potential(issue)
        
        return FixRecommendation(
            fix_id=f"individual_{issue.id.value}",
            title=template.get("title", f"Fix {category.value} Issue"),
            description=template.get("description", f"Resolve {category.value} issue in {issue.original_issue.file_path.name if issue.original_issue else 'code'}"),
            fix_type=self._category_to_fix_type(category),
            automation_potential=automation_potential,
            complexity_score=issue.metadata.fix_complexity_score,
            effort_estimate_hours=issue.metadata.estimated_fix_time_hours or 2.0,
            confidence_level=0.7,
            implementation_guidance=template.get("steps", ["Analyze issue", "Implement fix", "Test solution"]),
            code_examples=template.get("code_examples", []),
            tool_suggestions=template.get("tools", []),
            risk_warnings=self._generate_individual_risk_warnings(issue)
        )
    
    def _select_best_template(self, issues: List[CategorizedIssue], templates: Dict[str, Any]) -> str:
        """Selecciona mejor template para grupo de issues."""
        if not templates:
            return "generic"
        
        # Analizar mensajes para encontrar patrones comunes
        all_messages = [issue.original_issue.message.lower() for issue in issues if issue.original_issue]
        message_text = " ".join(all_messages)
        
        # Scoring por keywords en templates
        template_scores = {}
        for template_key, template in templates.items():
            score = 0
            # Buscar keywords del template en mensajes
            for keyword in template_key.split("_"):
                score += message_text.count(keyword)
            template_scores[template_key] = score
        
        # Retornar template con mayor score
        if template_scores:
            return max(template_scores.keys(), key=lambda k: template_scores[k])
        
        return list(templates.keys())[0]  # Primer template como fallback
    
    def _select_template_for_issue(self, issue: CategorizedIssue, templates: Dict[str, Any]) -> str:
        """Selecciona template para issue individual."""
        if not templates or not issue.original_issue:
            return "generic"
        
        message = issue.original_issue.message.lower()
        
        # Encontrar template que mejor match con el mensaje
        best_match = "generic"
        best_score = 0
        
        for template_key in templates.keys():
            score = 0
            for keyword in template_key.split("_"):
                if keyword in message:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = template_key
        
        return best_match
    
    def _category_to_fix_type(self, category: IssueCategory) -> FixType:
        """Convierte categoría a tipo de fix."""
        mapping = {
            IssueCategory.SECURITY: FixType.CODE_CHANGE,
            IssueCategory.PERFORMANCE: FixType.REFACTORING,
            IssueCategory.MAINTAINABILITY: FixType.REFACTORING,
            IssueCategory.RELIABILITY: FixType.CODE_CHANGE,
            IssueCategory.DOCUMENTATION: FixType.DOCUMENTATION,
            IssueCategory.CODE_STYLE: FixType.CODE_CHANGE,
            IssueCategory.ARCHITECTURE: FixType.ARCHITECTURAL
        }
        
        return mapping.get(category, FixType.CODE_CHANGE)
    
    async def _assess_automation_potential(self, category: IssueCategory, issues: List[CategorizedIssue]) -> float:
        """Evalúa potencial de automatización para batch."""
        base_automation = {
            IssueCategory.CODE_STYLE: 0.9,
            IssueCategory.DOCUMENTATION: 0.6,
            IssueCategory.MAINTAINABILITY: 0.4,
            IssueCategory.PERFORMANCE: 0.3,
            IssueCategory.RELIABILITY: 0.2,
            IssueCategory.SECURITY: 0.1,
            IssueCategory.ARCHITECTURE: 0.05
        }.get(category, 0.2)
        
        # Boost para batches grandes (más eficiente automatizar)
        if len(issues) >= 10:
            base_automation += 0.2
        elif len(issues) >= 5:
            base_automation += 0.1
        
        # Boost si todos son del mismo tipo
        all_same_message_pattern = len(set(
            issue.original_issue.rule_id for issue in issues if issue.original_issue
        )) == 1
        
        if all_same_message_pattern:
            base_automation += 0.3
        
        return min(1.0, base_automation)
    
    async def _assess_individual_automation_potential(self, issue: CategorizedIssue) -> float:
        """Evalúa potencial de automatización para issue individual."""
        category = issue.primary_category
        
        base_automation = {
            IssueCategory.CODE_STYLE: 0.8,
            IssueCategory.DOCUMENTATION: 0.5,
            IssueCategory.MAINTAINABILITY: 0.3,
            IssueCategory.PERFORMANCE: 0.2,
            IssueCategory.RELIABILITY: 0.1,
            IssueCategory.SECURITY: 0.05
        }.get(category, 0.1)
        
        # Reducir para issues complejos
        if issue.metadata.fix_complexity_score > 70:
            base_automation *= 0.5
        
        return base_automation
    
    def _generate_risk_warnings(self, category: IssueCategory, issue_count: int) -> List[str]:
        """Genera warnings de riesgo para batch fix."""
        warnings = []
        
        if category == IssueCategory.SECURITY:
            warnings.extend([
                "Security fixes require careful review",
                "Test thoroughly in isolated environment",
                "Consider security team consultation"
            ])
        
        elif category == IssueCategory.PERFORMANCE:
            warnings.extend([
                "Performance changes may have unexpected side effects",
                "Benchmark before and after changes",
                "Monitor production performance closely"
            ])
        
        elif category == IssueCategory.ARCHITECTURE:
            warnings.extend([
                "Architectural changes have high impact",
                "Plan migration strategy carefully",
                "Consider backward compatibility"
            ])
        
        if issue_count > 10:
            warnings.append("Large batch - consider splitting into smaller groups")
        
        if issue_count > 20:
            warnings.append("Very large batch - high coordination complexity")
        
        return warnings
    
    def _generate_individual_risk_warnings(self, issue: CategorizedIssue) -> List[str]:
        """Genera warnings para issue individual."""
        warnings = []
        
        if issue.metadata.fix_complexity_score > 80:
            warnings.append("High fix complexity - consider breaking down")
        
        if issue.metadata.regression_risk_score > 70:
            warnings.append("High regression risk - thorough testing required")
        
        if issue.context_info.test_coverage_percentage < 50:
            warnings.append("Low test coverage - add tests before fixing")
        
        if issue.context_info.module_criticality == "critical":
            warnings.append("Critical module - extra caution required")
        
        return warnings
    
    async def assess_automation_opportunities(self, issues: List[CategorizedIssue]) -> AutomationAssessment:
        """
        Evalúa oportunidades de automatización.
        
        Args:
            issues: Lista de issues
            
        Returns:
            AutomationAssessment completo
        """
        # Analizar issues por automatizabilidad
        automatable_issues = []
        total_automation_potential = 0.0
        
        for issue in issues:
            automation_potential = await self._assess_individual_automation_potential(issue)
            total_automation_potential += automation_potential
            
            if automation_potential > 0.6:
                automatable_issues.append(issue)
        
        automatable_percentage = (len(automatable_issues) / len(issues)) * 100.0 if issues else 0.0
        
        # Recomendar herramientas
        recommended_tools = set()
        for issue in automatable_issues:
            category_tools = self.automation_tools.get(issue.primary_category, [])
            recommended_tools.update(category_tools[:2])  # Top 2 tools por categoría
        
        # Beneficios de automatización
        automation_benefits = [
            f"Automate {len(automatable_issues)} issues",
            "Reduce manual fix effort by 60-80%",
            "Improve consistency of fixes",
            "Enable continuous quality improvement"
        ]
        
        # Riesgos de automatización
        automation_risks = [
            "Automated fixes may miss edge cases",
            "Initial setup effort required",
            "Tool learning curve for team"
        ]
        
        # Complejidad de implementación
        complexity = "low"
        if len(automatable_issues) > 20:
            complexity = "medium"
        if any(issue.primary_category in [IssueCategory.SECURITY, IssueCategory.ARCHITECTURE] 
               for issue in automatable_issues):
            complexity = "high"
        
        return AutomationAssessment(
            automatable_percentage=automatable_percentage,
            recommended_tools=list(recommended_tools),
            automation_benefits=automation_benefits,
            automation_risks=automation_risks,
            implementation_complexity=complexity
        )
