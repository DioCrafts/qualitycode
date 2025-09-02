"""
Implementación del analizador de calidad y mantenibilidad.

Este módulo implementa el cálculo del índice de mantenibilidad,
análisis de calidad general y métricas de testabilidad/legibilidad.
"""

import logging
import asyncio
import math
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import time

from ...domain.entities.code_metrics import (
    QualityMetrics, CodeMetrics, HalsteadMetrics, ComplexityMetrics, 
    SizeMetrics, CohesionMetrics, CouplingMetrics
)
from ...domain.entities.parse_result import ParseResult
from ...domain.entities.ast_normalization import NormalizedNode as UnifiedNode, NodeType as UnifiedNodeType
from ...domain.value_objects.programming_language import ProgrammingLanguage

logger = logging.getLogger(__name__)


@dataclass
class QualityAnalysisResult:
    """Resultado del análisis de calidad."""
    quality_metrics: QualityMetrics
    maintainability_breakdown: Dict[str, float]
    testability_analysis: Dict[str, Any]
    readability_analysis: Dict[str, Any]
    reliability_analysis: Dict[str, Any]
    analysis_time_ms: int


class MaintainabilityCalculator:
    """Calculadora del índice de mantenibilidad."""
    
    def calculate_maintainability_index(self, code_metrics: CodeMetrics) -> float:
        """
        Calcula índice de mantenibilidad usando fórmula adaptada de Microsoft.
        
        MI = 171 - 5.2*ln(HalsteadVolume) - 0.23*CyclomaticComplexity - 16.2*ln(LinesOfCode) + 50*sin(sqrt(2.4*CommentRatio))
        
        Args:
            code_metrics: Métricas de código calculadas
            
        Returns:
            Índice de mantenibilidad (0-100, donde 100 es mejor)
        """
        # Valores base con mínimos para evitar log(0)
        halstead_volume = max(1.0, code_metrics.halstead_metrics.volume)
        cyclomatic_complexity = max(1, code_metrics.complexity_metrics.cyclomatic_complexity)
        lines_of_code = max(1, code_metrics.size_metrics.logical_lines_of_code)
        comment_ratio = code_metrics.size_metrics.get_comment_ratio()
        
        try:
            # Fórmula base de mantenibilidad
            mi = (171.0 
                  - 5.2 * math.log(halstead_volume)
                  - 0.23 * cyclomatic_complexity 
                  - 16.2 * math.log(lines_of_code))
            
            # Ajuste por comentarios
            if comment_ratio > 0:
                comment_factor = 50.0 * math.sin(math.sqrt(2.4 * comment_ratio))
                mi += comment_factor
            
            # Normalizar a escala 0-100
            normalized_mi = max(0.0, min(100.0, mi * 100.0 / 171.0))
            
            return normalized_mi
            
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Error calculando índice de mantenibilidad: {e}")
            return 50.0  # Valor neutro por defecto
    
    def get_maintainability_breakdown(self, code_metrics: CodeMetrics) -> Dict[str, float]:
        """
        Obtiene breakdown detallado de factores de mantenibilidad.
        
        Returns:
            Diccionario con contribución de cada factor
        """
        halstead_volume = max(1.0, code_metrics.halstead_metrics.volume)
        cyclomatic_complexity = max(1, code_metrics.complexity_metrics.cyclomatic_complexity)
        lines_of_code = max(1, code_metrics.size_metrics.logical_lines_of_code)
        comment_ratio = code_metrics.size_metrics.get_comment_ratio()
        
        return {
            "base_score": 171.0,
            "halstead_penalty": -5.2 * math.log(halstead_volume),
            "complexity_penalty": -0.23 * cyclomatic_complexity,
            "size_penalty": -16.2 * math.log(lines_of_code),
            "comment_bonus": 50.0 * math.sin(math.sqrt(2.4 * comment_ratio)) if comment_ratio > 0 else 0.0,
            "comment_ratio": comment_ratio
        }


class TestabilityAnalyzer:
    """Analizador de testabilidad del código."""
    
    def analyze_testability(self, code_metrics: CodeMetrics, parse_result: ParseResult) -> Dict[str, Any]:
        """
        Analiza testabilidad del código.
        
        Returns:
            Diccionario con análisis de testabilidad
        """
        analysis = {
            "testability_score": 0.0,
            "factors": {},
            "suggestions": []
        }
        
        # Factor 1: Complejidad (menos complejo = más testeable)
        complexity_factor = self._calculate_complexity_testability_factor(code_metrics.complexity_metrics)
        analysis["factors"]["complexity"] = complexity_factor
        
        # Factor 2: Acoplamiento (menos acoplado = más testeable)
        coupling_factor = self._calculate_coupling_testability_factor(code_metrics.coupling_metrics)
        analysis["factors"]["coupling"] = coupling_factor
        
        # Factor 3: Cohesión (más cohesivo = más testeable)
        cohesion_factor = self._calculate_cohesion_testability_factor(code_metrics.cohesion_metrics)
        analysis["factors"]["cohesion"] = cohesion_factor
        
        # Factor 4: Tamaño de funciones (funciones pequeñas = más testeables)
        size_factor = self._calculate_size_testability_factor(code_metrics.function_metrics)
        analysis["factors"]["function_size"] = size_factor
        
        # Factor 5: Dependencias externas
        dependency_factor = self._calculate_dependency_testability_factor(parse_result)
        analysis["factors"]["dependencies"] = dependency_factor
        
        # Calcular score general (promedio ponderado)
        weights = {
            "complexity": 0.3,
            "coupling": 0.25,
            "cohesion": 0.2,
            "function_size": 0.15,
            "dependencies": 0.1
        }
        
        testability_score = sum(
            analysis["factors"][factor] * weight 
            for factor, weight in weights.items()
        )
        
        analysis["testability_score"] = min(100.0, max(0.0, testability_score))
        
        # Generar sugerencias
        analysis["suggestions"] = self._generate_testability_suggestions(analysis["factors"])
        
        return analysis
    
    def _calculate_complexity_testability_factor(self, complexity_metrics: ComplexityMetrics) -> float:
        """Calcula factor de testabilidad basado en complejidad."""
        # Complejidad alta reduce testabilidad
        max_acceptable_complexity = 10
        complexity = complexity_metrics.cyclomatic_complexity
        
        if complexity <= max_acceptable_complexity:
            return 100.0
        else:
            # Penalización exponencial para alta complejidad
            penalty = min(90.0, (complexity - max_acceptable_complexity) * 5.0)
            return max(10.0, 100.0 - penalty)
    
    def _calculate_coupling_testability_factor(self, coupling_metrics: CouplingMetrics) -> float:
        """Calcula factor de testabilidad basado en acoplamiento."""
        # Acoplamiento alto reduce testabilidad
        max_acceptable_cbo = 5.0
        cbo = coupling_metrics.average_cbo
        
        if cbo <= max_acceptable_cbo:
            return 100.0
        else:
            penalty = min(80.0, (cbo - max_acceptable_cbo) * 10.0)
            return max(20.0, 100.0 - penalty)
    
    def _calculate_cohesion_testability_factor(self, cohesion_metrics: CohesionMetrics) -> float:
        """Calcula factor de testabilidad basado en cohesión."""
        # Alta cohesión mejora testabilidad
        lcom = cohesion_metrics.average_lcom
        
        if lcom <= 0.3:
            return 100.0
        elif lcom <= 0.6:
            return 80.0
        else:
            return 50.0
    
    def _calculate_size_testability_factor(self, function_metrics: List) -> float:
        """Calcula factor de testabilidad basado en tamaño."""
        if not function_metrics:
            return 100.0
        
        # Funciones grandes son difíciles de testear
        large_functions = sum(1 for f in function_metrics if hasattr(f, 'lines_of_code') and f.lines_of_code > 50)
        total_functions = len(function_metrics)
        
        large_function_ratio = large_functions / total_functions
        
        return max(20.0, 100.0 - (large_function_ratio * 80.0))
    
    def _calculate_dependency_testability_factor(self, parse_result: ParseResult) -> float:
        """Calcula factor de testabilidad basado en dependencias."""
        # Análisis simplificado de imports/dependencias
        content = self._get_file_content(parse_result)
        
        # Contar imports/dependencias
        import_count = content.count('import ') + content.count('from ') + content.count('require(')
        
        if import_count <= 10:
            return 100.0
        elif import_count <= 20:
            return 80.0
        else:
            return max(40.0, 100.0 - (import_count - 20) * 2.0)
    
    def _generate_testability_suggestions(self, factors: Dict[str, float]) -> List[str]:
        """Genera sugerencias para mejorar testabilidad."""
        suggestions = []
        
        if factors.get("complexity", 100) < 70:
            suggestions.extend([
                "Reduce cyclomatic complexity by extracting methods",
                "Simplify conditional logic",
                "Use polymorphism instead of complex conditionals"
            ])
        
        if factors.get("coupling", 100) < 70:
            suggestions.extend([
                "Reduce coupling by using dependency injection",
                "Apply interfaces to decouple implementations",
                "Use composition over inheritance"
            ])
        
        if factors.get("cohesion", 100) < 70:
            suggestions.extend([
                "Improve class cohesion by grouping related functionality",
                "Split classes with multiple responsibilities"
            ])
        
        if factors.get("function_size", 100) < 70:
            suggestions.append("Break large functions into smaller, testable units")
        
        return suggestions
    
    def _get_file_content(self, parse_result: ParseResult) -> str:
        """Obtiene contenido del archivo."""
        if hasattr(parse_result.tree, 'root_node') and hasattr(parse_result.tree.root_node, 'text'):
            return parse_result.tree.root_node.text.decode('utf-8')
        return ""


class ReadabilityAnalyzer:
    """Analizador de legibilidad del código."""
    
    def analyze_readability(self, code_metrics: CodeMetrics, parse_result: ParseResult) -> Dict[str, Any]:
        """
        Analiza legibilidad del código.
        
        Returns:
            Diccionario con análisis de legibilidad
        """
        analysis = {
            "readability_score": 0.0,
            "factors": {},
            "suggestions": []
        }
        
        content = self._get_file_content(parse_result)
        
        # Factor 1: Nombres descriptivos
        naming_factor = self._analyze_naming_quality(content, parse_result.language)
        analysis["factors"]["naming"] = naming_factor
        
        # Factor 2: Comentarios y documentación
        documentation_factor = self._analyze_documentation_quality(content, parse_result.language)
        analysis["factors"]["documentation"] = documentation_factor
        
        # Factor 3: Longitud de líneas
        line_length_factor = self._analyze_line_length(content)
        analysis["factors"]["line_length"] = line_length_factor
        
        # Factor 4: Complejidad de expresiones
        expression_factor = self._analyze_expression_complexity(code_metrics)
        analysis["factors"]["expression_complexity"] = expression_factor
        
        # Factor 5: Consistencia de estilo
        style_factor = self._analyze_style_consistency(content, parse_result.language)
        analysis["factors"]["style_consistency"] = style_factor
        
        # Score general
        weights = {
            "naming": 0.3,
            "documentation": 0.25,
            "line_length": 0.15,
            "expression_complexity": 0.2,
            "style_consistency": 0.1
        }
        
        readability_score = sum(
            analysis["factors"][factor] * weight 
            for factor, weight in weights.items()
        )
        
        analysis["readability_score"] = min(100.0, max(0.0, readability_score))
        analysis["suggestions"] = self._generate_readability_suggestions(analysis["factors"])
        
        return analysis
    
    def _analyze_naming_quality(self, content: str, language: ProgrammingLanguage) -> float:
        """Analiza calidad de nombres."""
        import re
        
        # Buscar patrones de nombres
        if language == ProgrammingLanguage.PYTHON:
            # Convenciones Python
            functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            variables = re.findall(r'(\b[a-z_][a-zA-Z0-9_]*)\s*=', content)
        else:
            # Convenciones generales
            functions = re.findall(r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)|fn\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            functions = [f for group in functions for f in group if f]
            classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)|struct\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            classes = [c for group in classes for c in group if c]
            variables = re.findall(r'(?:let|var|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
        
        # Evaluar calidad de nombres
        all_names = functions + classes + variables
        if not all_names:
            return 100.0
        
        descriptive_names = 0
        for name in all_names:
            if self._is_descriptive_name(name):
                descriptive_names += 1
        
        return (descriptive_names / len(all_names)) * 100.0
    
    def _is_descriptive_name(self, name: str) -> bool:
        """Verifica si un nombre es descriptivo."""
        # Criterios para nombre descriptivo
        if len(name) < 3:
            return False  # Muy corto
        
        if name in ['a', 'b', 'c', 'x', 'y', 'z', 'i', 'j', 'k', 'data', 'info', 'temp', 'tmp']:
            return False  # Nombres genéricos
        
        # Nombres con palabras completas son mejores
        if '_' in name or any(c.isupper() for c in name[1:]):  # snake_case o camelCase
            return True
        
        return len(name) >= 5  # Al menos 5 caracteres para nombres simples
    
    def _analyze_documentation_quality(self, content: str, language: ProgrammingLanguage) -> float:
        """Analiza calidad de documentación."""
        lines = content.split('\n')
        total_lines = len(lines)
        
        if total_lines == 0:
            return 100.0
        
        # Contar líneas de comentarios/documentación
        comment_lines = 0
        docstring_lines = 0
        
        for line in lines:
            stripped = line.strip()
            
            if language == ProgrammingLanguage.PYTHON:
                if stripped.startswith('#'):
                    comment_lines += 1
                elif '"""' in stripped or "'''" in stripped:
                    docstring_lines += 1
            
            elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
                if stripped.startswith('//') or stripped.startswith('*') or '/**' in stripped:
                    comment_lines += 1
            
            elif language == ProgrammingLanguage.RUST:
                if stripped.startswith('//') or stripped.startswith('///') or stripped.startswith('//!'):
                    comment_lines += 1
        
        # Calcular score basado en ratio de documentación
        doc_ratio = (comment_lines + docstring_lines) / total_lines
        
        if doc_ratio >= 0.2:
            return 100.0
        elif doc_ratio >= 0.1:
            return 80.0
        elif doc_ratio >= 0.05:
            return 60.0
        else:
            return max(20.0, doc_ratio * 400.0)  # Escalar linealmente
    
    def _analyze_line_length(self, content: str) -> float:
        """Analiza longitud de líneas."""
        lines = content.split('\n')
        if not lines:
            return 100.0
        
        # Calcular estadísticas de longitud
        line_lengths = [len(line) for line in lines]
        avg_length = sum(line_lengths) / len(line_lengths)
        max_length = max(line_lengths) if line_lengths else 0
        long_lines = sum(1 for length in line_lengths if length > 120)
        
        # Score basado en criterios
        score = 100.0
        
        if avg_length > 80:
            score -= 20.0
        
        if max_length > 150:
            score -= 30.0
        
        long_line_ratio = long_lines / len(lines)
        if long_line_ratio > 0.1:
            score -= long_line_ratio * 50.0
        
        return max(0.0, score)
    
    def _analyze_expression_complexity(self, code_metrics: CodeMetrics) -> float:
        """Analiza complejidad de expresiones."""
        # Basado en complejidad cognitiva
        cognitive_complexity = code_metrics.complexity_metrics.cognitive_complexity
        
        if cognitive_complexity <= 5:
            return 100.0
        elif cognitive_complexity <= 15:
            return 80.0
        elif cognitive_complexity <= 25:
            return 60.0
        else:
            return max(20.0, 100.0 - (cognitive_complexity - 25) * 2.0)
    
    def _analyze_style_consistency(self, content: str, language: ProgrammingLanguage) -> float:
        """Analiza consistencia de estilo."""
        score = 100.0
        
        # Análisis básico de consistencia
        lines = content.split('\n')
        
        # Verificar consistencia de indentación
        indentations = []
        for line in lines:
            if line.strip():  # Líneas no vacías
                leading_spaces = len(line) - len(line.lstrip(' '))
                if leading_spaces > 0:
                    indentations.append(leading_spaces)
        
        if indentations:
            # Verificar si hay patrón consistente
            import math
            # Calcular desviación estándar de indentación
            avg_indent = sum(indentations) / len(indentations)
            variance = sum((x - avg_indent) ** 2 for x in indentations) / len(indentations)
            std_dev = math.sqrt(variance)
            
            # Penalizar alta variabilidad
            if std_dev > 4.0:
                score -= 20.0
        
        # Verificar trailing whitespace
        lines_with_trailing = sum(1 for line in lines if line.endswith(' ') or line.endswith('\t'))
        if lines_with_trailing > len(lines) * 0.1:
            score -= 10.0
        
        return max(0.0, score)
    
    def _generate_readability_suggestions(self, factors: Dict[str, float]) -> List[str]:
        """Genera sugerencias para mejorar legibilidad."""
        suggestions = []
        
        if factors.get("naming", 100) < 70:
            suggestions.extend([
                "Use more descriptive variable and function names",
                "Avoid abbreviations and single-letter variables",
                "Follow language naming conventions"
            ])
        
        if factors.get("documentation", 100) < 60:
            suggestions.extend([
                "Add more comments to explain complex logic",
                "Write docstrings for public functions and classes",
                "Document non-obvious business logic"
            ])
        
        if factors.get("line_length", 100) < 70:
            suggestions.extend([
                "Break long lines into multiple shorter lines",
                "Consider extracting complex expressions into variables"
            ])
        
        if factors.get("expression_complexity", 100) < 70:
            suggestions.extend([
                "Simplify complex expressions",
                "Extract nested conditions into separate variables"
            ])
        
        return suggestions
    
    def _get_file_content(self, parse_result: ParseResult) -> str:
        """Obtiene contenido del archivo."""
        if hasattr(parse_result.tree, 'root_node') and hasattr(parse_result.tree.root_node, 'text'):
            return parse_result.tree.root_node.text.decode('utf-8')
        return ""


class ReliabilityAnalyzer:
    """Analizador de confiabilidad del código."""
    
    def analyze_reliability(self, code_metrics: CodeMetrics, parse_result: ParseResult) -> Dict[str, Any]:
        """
        Analiza confiabilidad del código.
        
        Returns:
            Diccionario con análisis de confiabilidad
        """
        analysis = {
            "reliability_score": 0.0,
            "factors": {},
            "risk_indicators": [],
            "suggestions": []
        }
        
        content = self._get_file_content(parse_result)
        
        # Factor 1: Manejo de errores
        error_handling_factor = self._analyze_error_handling(content, parse_result.language)
        analysis["factors"]["error_handling"] = error_handling_factor
        
        # Factor 2: Validación de entrada
        input_validation_factor = self._analyze_input_validation(content)
        analysis["factors"]["input_validation"] = input_validation_factor
        
        # Factor 3: Complejidad (alta complejidad = mayor riesgo de bugs)
        complexity_reliability_factor = self._calculate_complexity_reliability_factor(code_metrics.complexity_metrics)
        analysis["factors"]["complexity_reliability"] = complexity_reliability_factor
        
        # Factor 4: Uso de recursos
        resource_usage_factor = self._analyze_resource_usage(content, parse_result.language)
        analysis["factors"]["resource_usage"] = resource_usage_factor
        
        # Factor 5: Patrones de riesgo
        risk_pattern_factor = self._analyze_risk_patterns(content, parse_result.language)
        analysis["factors"]["risk_patterns"] = risk_pattern_factor
        
        # Score general
        weights = {
            "error_handling": 0.3,
            "input_validation": 0.25,
            "complexity_reliability": 0.2,
            "resource_usage": 0.15,
            "risk_patterns": 0.1
        }
        
        reliability_score = sum(
            analysis["factors"][factor] * weight 
            for factor, weight in weights.items()
        )
        
        analysis["reliability_score"] = min(100.0, max(0.0, reliability_score))
        
        # Identificar indicadores de riesgo
        analysis["risk_indicators"] = self._identify_risk_indicators(analysis["factors"], content)
        analysis["suggestions"] = self._generate_reliability_suggestions(analysis["factors"])
        
        return analysis
    
    def _analyze_error_handling(self, content: str, language: ProgrammingLanguage) -> float:
        """Analiza calidad del manejo de errores."""
        # Contar estructuras de manejo de errores
        if language == ProgrammingLanguage.PYTHON:
            try_count = content.count('try:')
            except_count = content.count('except')
            raise_count = content.count('raise ')
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            try_count = content.count('try {')
            except_count = content.count('catch')
            raise_count = content.count('throw ')
        elif language == ProgrammingLanguage.RUST:
            try_count = content.count('Result<') + content.count('Option<')
            except_count = content.count('match') + content.count('if let')
            raise_count = content.count('panic!') + content.count('unwrap()')
        else:
            return 70.0  # Score neutro para lenguajes no soportados
        
        # Calcular score basado en uso de manejo de errores
        total_statements = content.count('\n') + 1
        error_handling_ratio = (try_count + except_count + raise_count) / max(1, total_statements)
        
        if error_handling_ratio >= 0.1:
            return 100.0
        elif error_handling_ratio >= 0.05:
            return 80.0
        elif error_handling_ratio >= 0.02:
            return 60.0
        else:
            return max(30.0, error_handling_ratio * 1500.0)
    
    def _analyze_input_validation(self, content: str) -> float:
        """Analiza validación de entrada."""
        # Buscar patrones de validación
        validation_patterns = [
            'if not ', 'if !', 'assert', 'validate', 'check', 'verify',
            'isinstance', 'typeof', 'is_valid', 'ensure'
        ]
        
        validation_count = sum(content.count(pattern) for pattern in validation_patterns)
        function_count = content.count('def ') + content.count('function ') + content.count('fn ')
        
        if function_count == 0:
            return 100.0
        
        validation_ratio = validation_count / function_count
        
        if validation_ratio >= 1.0:
            return 100.0
        elif validation_ratio >= 0.5:
            return 80.0
        else:
            return max(40.0, validation_ratio * 160.0)
    
    def _calculate_complexity_reliability_factor(self, complexity_metrics: ComplexityMetrics) -> float:
        """Calcula factor de confiabilidad basado en complejidad."""
        # Alta complejidad aumenta probabilidad de bugs
        complexity = complexity_metrics.cyclomatic_complexity
        
        if complexity <= 5:
            return 100.0
        elif complexity <= 10:
            return 85.0
        elif complexity <= 20:
            return 65.0
        else:
            return max(20.0, 100.0 - (complexity - 20) * 2.0)
    
    def _analyze_resource_usage(self, content: str, language: ProgrammingLanguage) -> float:
        """Analiza uso de recursos."""
        risky_patterns = []
        
        if language == ProgrammingLanguage.PYTHON:
            risky_patterns = ['open(', 'file(', 'socket(', 'thread', 'Process(']
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            risky_patterns = ['new Promise', 'setTimeout', 'setInterval', 'fetch(', 'XMLHttpRequest']
        elif language == ProgrammingLanguage.RUST:
            risky_patterns = ['unsafe', 'Box::leak', 'mem::transmute', 'ptr::']
        
        risky_usage = sum(content.count(pattern) for pattern in risky_patterns)
        
        if risky_usage == 0:
            return 100.0
        elif risky_usage <= 3:
            return 80.0
        else:
            return max(40.0, 100.0 - risky_usage * 10.0)
    
    def _analyze_risk_patterns(self, content: str, language: ProgrammingLanguage) -> float:
        """Analiza patrones de riesgo."""
        risk_score = 100.0
        
        # Patrones de riesgo generales
        if 'TODO' in content or 'FIXME' in content or 'HACK' in content:
            risk_score -= 10.0
        
        # Magic numbers
        import re
        magic_numbers = re.findall(r'\b\d{2,}\b', content)
        if len(magic_numbers) > 5:
            risk_score -= 15.0
        
        # Líneas muy largas (> 150 caracteres)
        long_lines = sum(1 for line in content.split('\n') if len(line) > 150)
        if long_lines > 0:
            risk_score -= min(20.0, long_lines * 2.0)
        
        return max(0.0, risk_score)
    
    def _identify_risk_indicators(self, factors: Dict[str, float], content: str) -> List[str]:
        """Identifica indicadores de riesgo."""
        indicators = []
        
        if factors.get("error_handling", 100) < 50:
            indicators.append("Insufficient error handling")
        
        if factors.get("input_validation", 100) < 50:
            indicators.append("Lack of input validation")
        
        if factors.get("complexity_reliability", 100) < 50:
            indicators.append("High complexity increases bug risk")
        
        if 'TODO' in content or 'FIXME' in content:
            indicators.append("Contains TODO/FIXME comments")
        
        return indicators
    
    def _generate_reliability_suggestions(self, factors: Dict[str, float]) -> List[str]:
        """Genera sugerencias para mejorar confiabilidad."""
        suggestions = []
        
        if factors.get("error_handling", 100) < 70:
            suggestions.extend([
                "Add proper error handling with try-catch blocks",
                "Handle edge cases and invalid inputs",
                "Use specific exception types"
            ])
        
        if factors.get("input_validation", 100) < 70:
            suggestions.extend([
                "Validate all input parameters",
                "Check for null/undefined values",
                "Validate data types and ranges"
            ])
        
        if factors.get("complexity_reliability", 100) < 70:
            suggestions.extend([
                "Reduce complexity to minimize bug risk",
                "Add unit tests for complex functions"
            ])
        
        return suggestions
    
    def _get_file_content(self, parse_result: ParseResult) -> str:
        """Obtiene contenido del archivo."""
        if hasattr(parse_result.tree, 'root_node') and hasattr(parse_result.tree.root_node, 'text'):
            return parse_result.tree.root_node.text.decode('utf-8')
        return ""


class QualityAnalyzer:
    """Analizador principal de calidad."""
    
    def __init__(self):
        self.maintainability_calculator = MaintainabilityCalculator()
        self.testability_analyzer = TestabilityAnalyzer()
        self.readability_analyzer = ReadabilityAnalyzer()
        self.reliability_analyzer = ReliabilityAnalyzer()
    
    async def calculate_quality_metrics(
        self, 
        code_metrics: CodeMetrics,
        parse_result: ParseResult,
        config: Optional[Dict[str, Any]] = None
    ) -> QualityAnalysisResult:
        """
        Calcula métricas de calidad completas.
        
        Args:
            code_metrics: Métricas de código ya calculadas
            parse_result: Resultado del parsing original
            config: Configuración opcional
            
        Returns:
            QualityAnalysisResult completo
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Calculando métricas de calidad para {parse_result.file_path}")
            
            # Calcular índice de mantenibilidad
            maintainability_index = self.maintainability_calculator.calculate_maintainability_index(code_metrics)
            maintainability_breakdown = self.maintainability_calculator.get_maintainability_breakdown(code_metrics)
            
            # Analizar testabilidad
            testability_analysis = self.testability_analyzer.analyze_testability(code_metrics, parse_result)
            
            # Analizar legibilidad
            readability_analysis = self.readability_analyzer.analyze_readability(code_metrics, parse_result)
            
            # Analizar confiabilidad
            reliability_analysis = self.reliability_analyzer.analyze_reliability(code_metrics, parse_result)
            
            # Crear métricas de calidad
            quality_metrics = QualityMetrics(
                maintainability_index=maintainability_index,
                testability_score=testability_analysis["testability_score"],
                readability_score=readability_analysis["readability_score"],
                reliability_score=reliability_analysis["reliability_score"],
                security_score=70.0,  # Placeholder - sería calculado por SecurityAnalyzer
                performance_score=75.0  # Placeholder - sería calculado por PerformanceAnalyzer
            )
            
            total_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Análisis de calidad completado para {parse_result.file_path}: "
                f"MI={maintainability_index:.1f}, Testability={quality_metrics.testability_score:.1f}, "
                f"Readability={quality_metrics.readability_score:.1f} en {total_time}ms"
            )
            
            return QualityAnalysisResult(
                quality_metrics=quality_metrics,
                maintainability_breakdown=maintainability_breakdown,
                testability_analysis=testability_analysis,
                readability_analysis=readability_analysis,
                reliability_analysis=reliability_analysis,
                analysis_time_ms=total_time
            )
            
        except Exception as e:
            logger.error(f"Error calculando métricas de calidad: {e}")
            raise
