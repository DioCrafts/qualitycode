"""
Content Generator for the Explanation Engine.

This module is responsible for generating the core content of explanations
based on analysis results, adapting to different languages and audiences.
"""
from typing import Dict, List, Optional, Any, Tuple
import logging
import re
import json
from datetime import datetime

from ...domain.entities.explanation import (
    Language, Audience, ExplanationDepth, ExplanationRequest,
    ExplanationSection, SectionType, SectionImportance,
    ComprehensiveExplanation
)
from ...domain.entities.antipattern_analysis import (
    DetectedAntipattern, AntipatternType
)
from ...domain.entities.code_metrics import (
    CodeMetrics, ComplexityMetrics, QualityMetrics
)
from .exceptions import ContentGenerationError, UnsupportedLanguageError


logger = logging.getLogger(__name__)


class ContentGenerator:
    """
    Generates explanation content based on analysis results.
    
    This class is responsible for creating the core textual content
    of explanations, which can then be adapted for different languages
    and audiences.
    """
    
    def __init__(self):
        """Initialize the content generator."""
        self.templates = self._load_templates()
        self.educational_resources = self._load_educational_resources()
        
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load explanation templates for different types of content."""
        return {
            "summary": {
                "en": "Analysis of {file_count} files found {issue_count} issues " +
                      "({critical_count} critical, {high_count} high severity). " +
                      "Overall quality score: {quality_score}/100.",
                "es": "El análisis de {file_count} archivos encontró {issue_count} problemas " +
                      "({critical_count} críticos, {high_count} de alta severidad). " +
                      "Puntuación de calidad general: {quality_score}/100."
            },
            "issue": {
                "en": "**{issue_title}** - {issue_description}\n\n" +
                      "Location: {file_path}, lines {start_line}-{end_line}\n\n" +
                      "Severity: {severity}",
                "es": "**{issue_title}** - {issue_description}\n\n" +
                      "Ubicación: {file_path}, líneas {start_line}-{end_line}\n\n" +
                      "Severidad: {severity}"
            },
            "antipattern": {
                "en": "**{antipattern_name}** detected in {file_path}\n\n" +
                      "{antipattern_description}\n\n" +
                      "Impact: {impact_description}",
                "es": "**{antipattern_name}** detectado en {file_path}\n\n" +
                      "{antipattern_description}\n\n" +
                      "Impacto: {impact_description}"
            },
            "metrics": {
                "en": "**Code Metrics**\n\n" +
                      "- Complexity: {complexity_score}\n" +
                      "- Maintainability: {maintainability_score}\n" +
                      "- Test Coverage: {test_coverage}%\n" +
                      "- Documentation: {documentation_score}\n" +
                      "- Overall Quality: {quality_score}/100",
                "es": "**Métricas de Código**\n\n" +
                      "- Complejidad: {complexity_score}\n" +
                      "- Mantenibilidad: {maintainability_score}\n" +
                      "- Cobertura de Pruebas: {test_coverage}%\n" +
                      "- Documentación: {documentation_score}\n" +
                      "- Calidad General: {quality_score}/100"
            },
            "recommendation": {
                "en": "**Recommendation: {recommendation_title}**\n\n" +
                      "{recommendation_description}\n\n" +
                      "Priority: {priority}\n" +
                      "Effort: {effort}\n" +
                      "Impact: {impact}",
                "es": "**Recomendación: {recommendation_title}**\n\n" +
                      "{recommendation_description}\n\n" +
                      "Prioridad: {priority}\n" +
                      "Esfuerzo: {effort}\n" +
                      "Impacto: {impact}"
            }
        }
    
    def _load_educational_resources(self) -> Dict[str, Dict[str, Any]]:
        """Load educational resources for different topics."""
        return {
            "sql_injection": {
                "description": {
                    "en": "SQL injection is a code injection technique that exploits vulnerabilities " +
                          "in the interface between web applications and database servers.",
                    "es": "La inyección SQL es una técnica de inyección de código que explota vulnerabilidades " +
                          "en la interfaz entre aplicaciones web y servidores de bases de datos."
                },
                "impact": {
                    "en": "Attackers can potentially read, modify, or delete database data, " +
                          "bypass authentication, or even execute administrative operations.",
                    "es": "Los atacantes pueden potencialmente leer, modificar o eliminar datos de la base de datos, " +
                          "eludir la autenticación o incluso ejecutar operaciones administrativas."
                },
                "solution": {
                    "en": "Use parameterized queries or prepared statements instead of concatenating user input.",
                    "es": "Use consultas parametrizadas o sentencias preparadas en lugar de concatenar la entrada del usuario."
                },
                "examples": {
                    "vulnerable": "query = \"SELECT * FROM users WHERE username = '\" + username + \"'\";",
                    "fixed": "query = \"SELECT * FROM users WHERE username = ?\"; execute(query, [username]);"
                }
            },
            "god_object": {
                "description": {
                    "en": "A God Object is a class that knows too much or does too much, " +
                          "violating the Single Responsibility Principle.",
                    "es": "Un Objeto Dios es una clase que sabe demasiado o hace demasiado, " +
                          "violando el Principio de Responsabilidad Única."
                },
                "impact": {
                    "en": "Makes code hard to maintain, test, and reuse. Changes to one aspect " +
                          "can affect unrelated functionality.",
                    "es": "Hace que el código sea difícil de mantener, probar y reutilizar. Los cambios en un aspecto " +
                          "pueden afectar a funcionalidades no relacionadas."
                },
                "solution": {
                    "en": "Refactor by extracting cohesive groups of methods and fields into separate classes.",
                    "es": "Refactorice extrayendo grupos cohesivos de métodos y campos en clases separadas."
                }
            }
        }
    
    async def generate_explanation(
        self,
        analysis_result: Any,
        request: ExplanationRequest
    ) -> ComprehensiveExplanation:
        """
        Generate a comprehensive explanation from analysis results.
        
        Args:
            analysis_result: The analysis results to explain
            request: Configuration for the explanation
            
        Returns:
            A comprehensive explanation of the analysis results
        """
        try:
            # Start with an empty explanation
            explanation = ComprehensiveExplanation(
                language=request.language,
                audience=request.audience
            )
            
            # Generate summary
            explanation.summary = await self._generate_summary(analysis_result, request)
            
            # Generate detailed sections
            explanation.detailed_sections = await self._generate_detailed_sections(
                analysis_result, request
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise ContentGenerationError(f"Failed to generate explanation: {str(e)}")
    
    async def _generate_summary(
        self,
        analysis_result: Any,
        request: ExplanationRequest
    ) -> str:
        """Generate a summary of the analysis results."""
        # Extract basic metrics from analysis result
        file_count = getattr(analysis_result, 'file_count', 0)
        issue_count = len(getattr(analysis_result, 'issues', []))
        critical_count = sum(1 for issue in getattr(analysis_result, 'issues', []) 
                           if getattr(issue, 'severity', '') == 'critical')
        high_count = sum(1 for issue in getattr(analysis_result, 'issues', []) 
                       if getattr(issue, 'severity', '') == 'high')
        
        # Get quality score
        metrics = getattr(analysis_result, 'metrics', None)
        quality_score = getattr(metrics, 'overall_quality_score', 0) if metrics else 0
        
        # Format quality score to one decimal place
        quality_score_formatted = f"{quality_score:.1f}"
        
        # Get the appropriate template for the language
        lang_code = 'es' if request.language == Language.SPANISH else 'en'
        if lang_code not in self.templates['summary']:
            raise UnsupportedLanguageError(f"Language {request.language} not supported")
        
        template = self.templates['summary'][lang_code]
        
        # Fill in the template
        summary = template.format(
            file_count=file_count,
            issue_count=issue_count,
            critical_count=critical_count,
            high_count=high_count,
            quality_score=quality_score_formatted
        )
        
        return summary
    
    async def _generate_detailed_sections(
        self,
        analysis_result: Any,
        request: ExplanationRequest
    ) -> List[ExplanationSection]:
        """Generate detailed sections for the explanation."""
        sections = []
        
        # Issues section
        issues = getattr(analysis_result, 'issues', [])
        if issues:
            issues_section = await self._generate_issues_section(issues, request)
            sections.append(issues_section)
        
        # Metrics section
        metrics = getattr(analysis_result, 'metrics', None)
        if metrics:
            metrics_section = await self._generate_metrics_section(metrics, request)
            sections.append(metrics_section)
        
        # Antipatterns section
        antipatterns = getattr(analysis_result, 'antipatterns', [])
        if antipatterns:
            antipatterns_section = await self._generate_antipatterns_section(antipatterns, request)
            sections.append(antipatterns_section)
        
        # Recommendations section
        recommendations = getattr(analysis_result, 'recommendations', [])
        if recommendations:
            recommendations_section = await self._generate_recommendations_section(
                recommendations, request
            )
            sections.append(recommendations_section)
        
        return sections
    
    async def _generate_issues_section(
        self,
        issues: List[Any],
        request: ExplanationRequest
    ) -> ExplanationSection:
        """Generate a section explaining the issues found."""
        # Get the appropriate language
        lang_code = 'es' if request.language == Language.SPANISH else 'en'
        
        # Section title
        title = "Problemas Detectados" if lang_code == 'es' else "Detected Issues"
        
        # Group issues by severity
        issues_by_severity = {}
        for issue in issues:
            severity = getattr(issue, 'severity', 'medium')
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append(issue)
        
        # Generate content
        content = ""
        
        # Order by severity
        severity_order = ['critical', 'high', 'medium', 'low']
        for severity in severity_order:
            if severity in issues_by_severity:
                severity_issues = issues_by_severity[severity]
                
                # Add severity header
                if lang_code == 'es':
                    severity_name = {
                        'critical': 'Críticos',
                        'high': 'Altos',
                        'medium': 'Medios',
                        'low': 'Bajos'
                    }.get(severity, severity.capitalize())
                    content += f"\n## Problemas {severity_name} ({len(severity_issues)})\n\n"
                else:
                    content += f"\n## {severity.capitalize()} Issues ({len(severity_issues)})\n\n"
                
                # Add each issue
                for i, issue in enumerate(severity_issues):
                    issue_content = await self._format_issue(issue, request)
                    content += f"{i+1}. {issue_content}\n\n"
        
        return ExplanationSection(
            title=title,
            content=content,
            section_type=SectionType.ISSUES,
            importance=SectionImportance.HIGH
        )
    
    async def _format_issue(self, issue: Any, request: ExplanationRequest) -> str:
        """Format a single issue for display."""
        lang_code = 'es' if request.language == Language.SPANISH else 'en'
        template = self.templates['issue'][lang_code]
        
        # Extract issue properties
        issue_title = getattr(issue, 'title', 'Unknown Issue')
        issue_description = getattr(issue, 'description', '')
        file_path = getattr(issue, 'file_path', 'Unknown')
        start_line = getattr(issue, 'start_line', 0)
        end_line = getattr(issue, 'end_line', 0)
        severity = getattr(issue, 'severity', 'medium')
        
        # Translate severity
        if lang_code == 'es':
            severity = {
                'critical': 'Crítica',
                'high': 'Alta',
                'medium': 'Media',
                'low': 'Baja'
            }.get(severity, severity)
        else:
            severity = severity.capitalize()
        
        # Fill in the template
        return template.format(
            issue_title=issue_title,
            issue_description=issue_description,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            severity=severity
        )
    
    async def _generate_metrics_section(
        self,
        metrics: Any,
        request: ExplanationRequest
    ) -> ExplanationSection:
        """Generate a section explaining code metrics."""
        # Get the appropriate language
        lang_code = 'es' if request.language == Language.SPANISH else 'en'
        
        # Section title
        title = "Métricas de Código" if lang_code == 'es' else "Code Metrics"
        
        # Extract metrics
        complexity_score = getattr(metrics, 'complexity_score', 0)
        maintainability_score = getattr(metrics, 'maintainability_score', 0)
        test_coverage = getattr(metrics, 'test_coverage', 0)
        documentation_score = getattr(metrics, 'documentation_score', 0)
        quality_score = getattr(metrics, 'overall_quality_score', 0)
        
        # Format scores
        complexity_score = f"{complexity_score:.1f}"
        maintainability_score = f"{maintainability_score:.1f}"
        test_coverage = f"{test_coverage:.1f}"
        documentation_score = f"{documentation_score:.1f}"
        quality_score = f"{quality_score:.1f}"
        
        # Fill in the template
        template = self.templates['metrics'][lang_code]
        content = template.format(
            complexity_score=complexity_score,
            maintainability_score=maintainability_score,
            test_coverage=test_coverage,
            documentation_score=documentation_score,
            quality_score=quality_score
        )
        
        # Add interpretation based on audience
        content += "\n\n"
        if request.audience == Audience.JUNIOR_DEVELOPER:
            content += await self._generate_metrics_interpretation_for_junior(metrics, request)
        elif request.audience == Audience.PROJECT_MANAGER:
            content += await self._generate_metrics_interpretation_for_manager(metrics, request)
        else:
            content += await self._generate_metrics_interpretation_for_senior(metrics, request)
        
        return ExplanationSection(
            title=title,
            content=content,
            section_type=SectionType.METRICS,
            importance=SectionImportance.MEDIUM
        )
    
    async def _generate_metrics_interpretation_for_junior(
        self,
        metrics: Any,
        request: ExplanationRequest
    ) -> str:
        """Generate metrics interpretation for junior developers."""
        lang_code = 'es' if request.language == Language.SPANISH else 'en'
        
        if lang_code == 'es':
            return ("**Interpretación para Desarrolladores:**\n\n"
                   "Estas métricas indican la calidad general del código. "
                   "Una alta complejidad puede hacer que el código sea difícil de entender. "
                   "La mantenibilidad indica qué tan fácil es modificar el código. "
                   "Una buena cobertura de pruebas ayuda a prevenir regresiones. "
                   "La documentación adecuada facilita entender el propósito del código.")
        else:
            return ("**Interpretation for Developers:**\n\n"
                   "These metrics indicate the overall code quality. "
                   "High complexity can make code difficult to understand. "
                   "Maintainability indicates how easy it is to modify the code. "
                   "Good test coverage helps prevent regressions. "
                   "Proper documentation makes it easier to understand the code's purpose.")
    
    async def _generate_metrics_interpretation_for_manager(
        self,
        metrics: Any,
        request: ExplanationRequest
    ) -> str:
        """Generate metrics interpretation for project managers."""
        lang_code = 'es' if request.language == Language.SPANISH else 'en'
        
        quality_score = getattr(metrics, 'overall_quality_score', 0)
        test_coverage = getattr(metrics, 'test_coverage', 0)
        
        if lang_code == 'es':
            interpretation = "**Interpretación para Gestión:**\n\n"
            
            if quality_score < 50:
                interpretation += ("La calidad del código es preocupante y puede impactar negativamente "
                                 "la velocidad de desarrollo y la estabilidad del producto. "
                                 "Se recomienda invertir en mejoras técnicas.")
            elif quality_score < 70:
                interpretation += ("La calidad del código es aceptable pero hay margen de mejora. "
                                 "Considere asignar tiempo para refactorización en próximos sprints.")
            else:
                interpretation += ("La calidad del código es buena, lo que facilita el mantenimiento "
                                 "y la incorporación de nuevas características.")
                
            if test_coverage < 50:
                interpretation += (" La baja cobertura de pruebas aumenta el riesgo de regresiones "
                                 "y puede ralentizar futuras entregas.")
        else:
            interpretation = "**Interpretation for Management:**\n\n"
            
            if quality_score < 50:
                interpretation += ("Code quality is concerning and may negatively impact "
                                 "development velocity and product stability. "
                                 "Investment in technical improvements is recommended.")
            elif quality_score < 70:
                interpretation += ("Code quality is acceptable but there's room for improvement. "
                                 "Consider allocating time for refactoring in upcoming sprints.")
            else:
                interpretation += ("Code quality is good, facilitating maintenance "
                                 "and the addition of new features.")
                
            if test_coverage < 50:
                interpretation += (" Low test coverage increases the risk of regressions "
                                 "and may slow down future deliveries.")
        
        return interpretation
    
    async def _generate_metrics_interpretation_for_senior(
        self,
        metrics: Any,
        request: ExplanationRequest
    ) -> str:
        """Generate metrics interpretation for senior developers."""
        lang_code = 'es' if request.language == Language.SPANISH else 'en'
        
        complexity_score = getattr(metrics, 'complexity_score', 0)
        maintainability_score = getattr(metrics, 'maintainability_score', 0)
        
        if lang_code == 'es':
            interpretation = "**Análisis Técnico:**\n\n"
            
            if complexity_score > 25:
                interpretation += ("La alta complejidad ciclomática indica posibles problemas de diseño. "
                                 "Considere refactorizar métodos complejos y extraer componentes reutilizables. ")
            
            if maintainability_score < 65:
                interpretation += ("La baja mantenibilidad sugiere acumulación de deuda técnica. "
                                 "Evalúe la aplicación de patrones de diseño y mejore la cohesión de clases.")
        else:
            interpretation = "**Technical Analysis:**\n\n"
            
            if complexity_score > 25:
                interpretation += ("High cyclomatic complexity indicates potential design issues. "
                                 "Consider refactoring complex methods and extracting reusable components. ")
            
            if maintainability_score < 65:
                interpretation += ("Low maintainability suggests technical debt accumulation. "
                                 "Evaluate applying design patterns and improving class cohesion.")
        
        return interpretation
    
    async def _generate_antipatterns_section(
        self,
        antipatterns: List[Any],
        request: ExplanationRequest
    ) -> ExplanationSection:
        """Generate a section explaining detected antipatterns."""
        # Get the appropriate language
        lang_code = 'es' if request.language == Language.SPANISH else 'en'
        
        # Section title
        title = "Antipatrones Detectados" if lang_code == 'es' else "Detected Antipatterns"
        
        # Generate content
        content = ""
        
        for i, antipattern in enumerate(antipatterns):
            antipattern_content = await self._format_antipattern(antipattern, request)
            content += f"{i+1}. {antipattern_content}\n\n"
        
        return ExplanationSection(
            title=title,
            content=content,
            section_type=SectionType.ANTIPATTERNS,
            importance=SectionImportance.HIGH
        )
    
    async def _format_antipattern(self, antipattern: Any, request: ExplanationRequest) -> str:
        """Format a single antipattern for display."""
        lang_code = 'es' if request.language == Language.SPANISH else 'en'
        template = self.templates['antipattern'][lang_code]
        
        # Extract antipattern properties
        antipattern_name = getattr(antipattern, 'pattern_type', 'Unknown Antipattern')
        if isinstance(antipattern_name, AntipatternType):
            antipattern_name = antipattern_name.value
        
        file_path = getattr(antipattern, 'file_path', 'Unknown')
        
        # Get description and impact from educational resources
        antipattern_key = self._get_antipattern_key(antipattern_name)
        
        if antipattern_key in self.educational_resources:
            resource = self.educational_resources[antipattern_key]
            description = resource['description'][lang_code]
            impact = resource['impact'][lang_code]
        else:
            # Default descriptions
            if lang_code == 'es':
                description = f"Un antipatrón que afecta la calidad del código."
                impact = "Puede dificultar el mantenimiento y la evolución del código."
            else:
                description = f"An antipattern that affects code quality."
                impact = "May make code maintenance and evolution difficult."
        
        # Fill in the template
        return template.format(
            antipattern_name=self._format_antipattern_name(antipattern_name, lang_code),
            file_path=file_path,
            antipattern_description=description,
            impact_description=impact
        )
    
    def _get_antipattern_key(self, antipattern_name: str) -> str:
        """Convert antipattern name to a key for educational resources."""
        # Convert to lowercase and replace spaces with underscores
        key = antipattern_name.lower().replace(' ', '_')
        
        # Map common variations
        mapping = {
            'sql_injection': 'sql_injection',
            'god_object': 'god_object',
            'god_class': 'god_object',
            'long_method': 'long_method',
            'n+1_query': 'n_plus_one_query',
            'n_plus_one_query': 'n_plus_one_query'
        }
        
        return mapping.get(key, key)
    
    def _format_antipattern_name(self, name: str, lang_code: str) -> str:
        """Format antipattern name for display."""
        # Convert snake_case or kebab-case to Title Case
        formatted = name.replace('_', ' ').replace('-', ' ').title()
        
        # Special cases for Spanish
        if lang_code == 'es':
            translations = {
                'God Object': 'Objeto Dios',
                'Long Method': 'Método Largo',
                'Long Parameter List': 'Lista Larga de Parámetros',
                'Duplicate Code': 'Código Duplicado',
                'Feature Envy': 'Envidia de Características',
                'Shotgun Surgery': 'Cirugía de Escopeta',
                'Sql Injection': 'Inyección SQL',
                'N Plus One Query': 'Consulta N+1'
            }
            
            if formatted in translations:
                return translations[formatted]
        
        return formatted
    
    async def _generate_recommendations_section(
        self,
        recommendations: List[Any],
        request: ExplanationRequest
    ) -> ExplanationSection:
        """Generate a section with recommendations."""
        # Get the appropriate language
        lang_code = 'es' if request.language == Language.SPANISH else 'en'
        
        # Section title
        title = "Recomendaciones" if lang_code == 'es' else "Recommendations"
        
        # Generate content
        content = ""
        
        for i, recommendation in enumerate(recommendations):
            recommendation_content = await self._format_recommendation(recommendation, request)
            content += f"{i+1}. {recommendation_content}\n\n"
        
        return ExplanationSection(
            title=title,
            content=content,
            section_type=SectionType.RECOMMENDATIONS,
            importance=SectionImportance.HIGH
        )
    
    async def _format_recommendation(
        self,
        recommendation: Any,
        request: ExplanationRequest
    ) -> str:
        """Format a single recommendation for display."""
        lang_code = 'es' if request.language == Language.SPANISH else 'en'
        template = self.templates['recommendation'][lang_code]
        
        # Extract recommendation properties
        title = getattr(recommendation, 'title', 'Unknown Recommendation')
        description = getattr(recommendation, 'description', '')
        priority = getattr(recommendation, 'priority', 'medium')
        effort = getattr(recommendation, 'effort', 'medium')
        impact = getattr(recommendation, 'impact', 'medium')
        
        # Translate values
        if lang_code == 'es':
            priority = {
                'high': 'Alta',
                'medium': 'Media',
                'low': 'Baja'
            }.get(priority, priority)
            
            effort = {
                'high': 'Alto',
                'medium': 'Medio',
                'low': 'Bajo'
            }.get(effort, effort)
            
            impact = {
                'high': 'Alto',
                'medium': 'Medio',
                'low': 'Bajo'
            }.get(impact, impact)
        else:
            priority = priority.capitalize()
            effort = effort.capitalize()
            impact = impact.capitalize()
        
        # Fill in the template
        return template.format(
            recommendation_title=title,
            recommendation_description=description,
            priority=priority,
            effort=effort,
            impact=impact
        )
