"""
Implementación del Generador Multimedia.

Este módulo implementa la generación de visualizaciones, diagramas
y contenido multimedia para las explicaciones.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from uuid import uuid4

from ...domain.entities.explanation import Language
from ..ports.explanation_ports import MultimediaGeneratorPort


logger = logging.getLogger(__name__)


class MultimediaGenerator(MultimediaGeneratorPort):
    """Generador de contenido multimedia."""
    
    def __init__(self):
        self.chart_templates = self._initialize_chart_templates()
        self.diagram_templates = self._initialize_diagram_templates()
        self.visualization_configs = self._initialize_visualization_configs()
    
    def _initialize_chart_templates(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar plantillas de gráficos."""
        return {
            "quality_metrics": {
                "type": "radar",
                "title_es": "Métricas de Calidad",
                "title_en": "Quality Metrics",
                "config": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "scales": {
                        "r": {
                            "beginAtZero": True,
                            "max": 100
                        }
                    }
                }
            },
            "issue_distribution": {
                "type": "doughnut",
                "title_es": "Distribución de Problemas",
                "title_en": "Issue Distribution",
                "config": {
                    "responsive": True,
                    "plugins": {
                        "legend": {
                            "position": "bottom"
                        }
                    }
                }
            },
            "complexity_trend": {
                "type": "line",
                "title_es": "Tendencia de Complejidad",
                "title_en": "Complexity Trend",
                "config": {
                    "responsive": True,
                    "scales": {
                        "y": {
                            "beginAtZero": True
                        }
                    }
                }
            },
            "file_metrics": {
                "type": "bar",
                "title_es": "Métricas por Archivo",
                "title_en": "Metrics by File",
                "config": {
                    "responsive": True,
                    "scales": {
                        "y": {
                            "beginAtZero": True
                        }
                    }
                }
            }
        }
    
    def _initialize_diagram_templates(self) -> Dict[str, Dict[str, str]]:
        """Inicializar plantillas de diagramas."""
        return {
            "dependency_graph": {
                "type": "mermaid",
                "template_es": """
graph TD
    {nodes}
    {edges}
""",
                "template_en": """
graph TD
    {nodes}
    {edges}
"""
            },
            "flow_diagram": {
                "type": "mermaid",
                "template_es": """
flowchart TD
    {flow_nodes}
    {flow_edges}
""",
                "template_en": """
flowchart TD
    {flow_nodes}
    {flow_edges}
"""
            },
            "architecture_diagram": {
                "type": "mermaid",
                "template_es": """
graph LR
    {components}
    {connections}
""",
                "template_en": """
graph LR
    {components}
    {connections}
"""
            }
        }
    
    def _initialize_visualization_configs(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar configuraciones de visualización."""
        return {
            "colors": {
                "primary": "#3498db",
                "secondary": "#2ecc71",
                "warning": "#f39c12",
                "danger": "#e74c3c",
                "info": "#9b59b6",
                "success": "#27ae60"
            },
            "chart_colors": [
                "#3498db", "#2ecc71", "#f39c12", "#e74c3c", 
                "#9b59b6", "#27ae60", "#1abc9c", "#34495e"
            ],
            "severity_colors": {
                "critical": "#e74c3c",
                "high": "#f39c12",
                "medium": "#f1c40f",
                "low": "#2ecc71"
            }
        }
    
    async def generate_visualizations(
        self, 
        analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generar visualizaciones."""
        try:
            visualizations = []
            
            # Generar gráfico de métricas de calidad
            quality_chart = await self._generate_quality_metrics_chart(analysis_result)
            if quality_chart:
                visualizations.append(quality_chart)
            
            # Generar gráfico de distribución de problemas
            issue_chart = await self._generate_issue_distribution_chart(analysis_result)
            if issue_chart:
                visualizations.append(issue_chart)
            
            # Generar gráfico de tendencia de complejidad
            complexity_chart = await self._generate_complexity_trend_chart(analysis_result)
            if complexity_chart:
                visualizations.append(complexity_chart)
            
            # Generar gráfico de métricas por archivo
            file_chart = await self._generate_file_metrics_chart(analysis_result)
            if file_chart:
                visualizations.append(file_chart)
            
            logger.info(f"Visualizaciones generadas: {len(visualizations)}")
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generando visualizaciones: {str(e)}")
            return []
    
    async def generate_diagrams(
        self, 
        analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generar diagramas."""
        try:
            diagrams = []
            
            # Generar diagrama de dependencias si está disponible
            dependency_diagram = await self._generate_dependency_diagram(analysis_result)
            if dependency_diagram:
                diagrams.append(dependency_diagram)
            
            # Generar diagramas de flujo para funciones complejas
            flow_diagrams = await self._generate_flow_diagrams(analysis_result)
            diagrams.extend(flow_diagrams)
            
            # Generar diagrama de arquitectura
            architecture_diagram = await self._generate_architecture_diagram(analysis_result)
            if architecture_diagram:
                diagrams.append(architecture_diagram)
            
            logger.info(f"Diagramas generados: {len(diagrams)}")
            return diagrams
            
        except Exception as e:
            logger.error(f"Error generando diagramas: {str(e)}")
            return []
    
    async def generate_code_comparisons(
        self, 
        original_code: str, 
        fixed_code: str,
        language: str
    ) -> Dict[str, Any]:
        """Generar comparaciones de código."""
        try:
            comparison = {
                "id": str(uuid4()),
                "type": "code_comparison",
                "original_code": {
                    "content": original_code,
                    "language": language,
                    "highlights": await self._highlight_problematic_lines(original_code)
                },
                "fixed_code": {
                    "content": fixed_code,
                    "language": language,
                    "highlights": await self._highlight_improvements(fixed_code)
                },
                "differences": await self._calculate_differences(original_code, fixed_code),
                "improvements": await self._identify_improvements(original_code, fixed_code),
                "explanation": await self._generate_comparison_explanation(original_code, fixed_code)
            }
            
            logger.info("Comparación de código generada exitosamente")
            return comparison
            
        except Exception as e:
            logger.error(f"Error generando comparación de código: {str(e)}")
            return {}
    
    async def _generate_quality_metrics_chart(self, analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generar gráfico de métricas de calidad."""
        try:
            metrics = analysis_result.get('metrics', {})
            if not metrics:
                return None
            
            # Preparar datos para el gráfico radar
            chart_data = {
                "labels": ["Calidad General", "Mantenibilidad", "Legibilidad", "Rendimiento", "Seguridad"],
                "datasets": [{
                    "label": "Puntuación",
                    "data": [
                        metrics.get('overall_quality_score', 0),
                        metrics.get('maintainability_score', 0),
                        metrics.get('readability_score', 0),
                        metrics.get('performance_score', 0),
                        metrics.get('security_score', 0)
                    ],
                    "backgroundColor": "rgba(52, 152, 219, 0.2)",
                    "borderColor": "rgba(52, 152, 219, 1)",
                    "borderWidth": 2
                }]
            }
            
            template = self.chart_templates["quality_metrics"]
            
            return {
                "id": str(uuid4()),
                "type": "chart",
                "chart_type": template["type"],
                "title": template["title_es"],  # Por defecto en español
                "data": chart_data,
                "config": template["config"],
                "description": "Gráfico radar mostrando las métricas de calidad del código"
            }
            
        except Exception as e:
            logger.error(f"Error generando gráfico de métricas de calidad: {str(e)}")
            return None
    
    async def _generate_issue_distribution_chart(self, analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generar gráfico de distribución de problemas."""
        try:
            violations = analysis_result.get('violations', [])
            if not violations:
                return None
            
            # Contar problemas por severidad
            severity_counts = {}
            for violation in violations:
                severity = violation.get('severity', 'unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Preparar datos para el gráfico doughnut
            labels = list(severity_counts.keys())
            data = list(severity_counts.values())
            colors = [self.visualization_configs["severity_colors"].get(severity, "#95a5a6") for severity in labels]
            
            chart_data = {
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": colors,
                    "borderWidth": 2,
                    "borderColor": "#ffffff"
                }]
            }
            
            template = self.chart_templates["issue_distribution"]
            
            return {
                "id": str(uuid4()),
                "type": "chart",
                "chart_type": template["type"],
                "title": template["title_es"],
                "data": chart_data,
                "config": template["config"],
                "description": "Distribución de problemas por severidad"
            }
            
        except Exception as e:
            logger.error(f"Error generando gráfico de distribución de problemas: {str(e)}")
            return None
    
    async def _generate_complexity_trend_chart(self, analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generar gráfico de tendencia de complejidad."""
        try:
            # Simular datos de tendencia (en un caso real vendrían del análisis histórico)
            complexity_data = {
                "labels": ["Archivo 1", "Archivo 2", "Archivo 3", "Archivo 4", "Archivo 5"],
                "datasets": [{
                    "label": "Complejidad Ciclomática",
                    "data": [5, 8, 12, 6, 9],
                    "borderColor": self.visualization_configs["colors"]["warning"],
                    "backgroundColor": "rgba(243, 156, 18, 0.2)",
                    "borderWidth": 2,
                    "fill": False
                }]
            }
            
            template = self.chart_templates["complexity_trend"]
            
            return {
                "id": str(uuid4()),
                "type": "chart",
                "chart_type": template["type"],
                "title": template["title_es"],
                "data": complexity_data,
                "config": template["config"],
                "description": "Tendencia de complejidad ciclomática por archivo"
            }
            
        except Exception as e:
            logger.error(f"Error generando gráfico de tendencia de complejidad: {str(e)}")
            return None
    
    async def _generate_file_metrics_chart(self, analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generar gráfico de métricas por archivo."""
        try:
            # Simular datos de métricas por archivo
            file_data = {
                "labels": ["main.py", "utils.py", "models.py", "views.py", "tests.py"],
                "datasets": [
                    {
                        "label": "Líneas de Código",
                        "data": [150, 200, 300, 180, 120],
                        "backgroundColor": self.visualization_configs["colors"]["primary"],
                        "borderColor": self.visualization_configs["colors"]["primary"],
                        "borderWidth": 1
                    },
                    {
                        "label": "Complejidad",
                        "data": [8, 12, 15, 6, 4],
                        "backgroundColor": self.visualization_configs["colors"]["warning"],
                        "borderColor": self.visualization_configs["colors"]["warning"],
                        "borderWidth": 1
                    }
                ]
            }
            
            template = self.chart_templates["file_metrics"]
            
            return {
                "id": str(uuid4()),
                "type": "chart",
                "chart_type": template["type"],
                "title": template["title_es"],
                "data": file_data,
                "config": template["config"],
                "description": "Métricas de código por archivo"
            }
            
        except Exception as e:
            logger.error(f"Error generando gráfico de métricas por archivo: {str(e)}")
            return None
    
    async def _generate_dependency_diagram(self, analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generar diagrama de dependencias."""
        try:
            dependency_info = analysis_result.get('dependency_analysis')
            if not dependency_info:
                return None
            
            # Generar diagrama Mermaid de dependencias
            mermaid_code = await self._create_dependency_mermaid(dependency_info)
            
            return {
                "id": str(uuid4()),
                "type": "diagram",
                "diagram_type": "mermaid",
                "title": "Diagrama de Dependencias",
                "content": mermaid_code,
                "description": "Diagrama mostrando las dependencias entre módulos del código"
            }
            
        except Exception as e:
            logger.error(f"Error generando diagrama de dependencias: {str(e)}")
            return None
    
    async def _generate_flow_diagrams(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generar diagramas de flujo para funciones complejas."""
        try:
            flow_diagrams = []
            
            # Buscar funciones con alta complejidad
            function_metrics = analysis_result.get('function_metrics', [])
            complex_functions = [f for f in function_metrics if f.get('cyclomatic_complexity', 0) > 10]
            
            for function in complex_functions[:3]:  # Limitar a 3 diagramas
                mermaid_code = await self._create_flow_mermaid(function)
                
                flow_diagram = {
                    "id": str(uuid4()),
                    "type": "diagram",
                    "diagram_type": "mermaid",
                    "title": f"Flujo de Control - {function.get('name', 'Función')}",
                    "content": mermaid_code,
                    "description": f"Diagrama de flujo para la función {function.get('name', 'desconocida')}"
                }
                
                flow_diagrams.append(flow_diagram)
            
            return flow_diagrams
            
        except Exception as e:
            logger.error(f"Error generando diagramas de flujo: {str(e)}")
            return []
    
    async def _generate_architecture_diagram(self, analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generar diagrama de arquitectura."""
        try:
            # Generar diagrama Mermaid de arquitectura
            mermaid_code = await self._create_architecture_mermaid(analysis_result)
            
            return {
                "id": str(uuid4()),
                "type": "diagram",
                "diagram_type": "mermaid",
                "title": "Diagrama de Arquitectura",
                "content": mermaid_code,
                "description": "Diagrama mostrando la arquitectura general del sistema"
            }
            
        except Exception as e:
            logger.error(f"Error generando diagrama de arquitectura: {str(e)}")
            return None
    
    async def _create_dependency_mermaid(self, dependency_info: Dict[str, Any]) -> str:
        """Crear código Mermaid para diagrama de dependencias."""
        template = self.diagram_templates["dependency_graph"]
        
        # Simular nodos y conexiones (en un caso real vendrían del análisis)
        nodes = """
    A[main.py] --> B[utils.py]
    A --> C[models.py]
    B --> D[helpers.py]
    C --> E[database.py]
    F[tests.py] --> A
    F --> B
    F --> C
"""
        
        edges = ""  # Ya incluido en nodes
        
        return template["template_es"].format(nodes=nodes, edges=edges)
    
    async def _create_flow_mermaid(self, function: Dict[str, Any]) -> str:
        """Crear código Mermaid para diagrama de flujo."""
        template = self.diagram_templates["flow_diagram"]
        
        # Simular flujo de control (en un caso real vendría del análisis del AST)
        flow_nodes = """
    Start([Inicio]) --> Check{¿Condición?}
    Check -->|Sí| Process1[Proceso 1]
    Check -->|No| Process2[Proceso 2]
    Process1 --> End([Fin])
    Process2 --> End
"""
        
        flow_edges = ""  # Ya incluido en flow_nodes
        
        return template["template_es"].format(flow_nodes=flow_nodes, flow_edges=flow_edges)
    
    async def _create_architecture_mermaid(self, analysis_result: Dict[str, Any]) -> str:
        """Crear código Mermaid para diagrama de arquitectura."""
        template = self.diagram_templates["architecture_diagram"]
        
        # Simular componentes de arquitectura
        components = """
    UI[Interfaz de Usuario] --> API[API Layer]
    API --> BL[Business Logic]
    BL --> DB[(Base de Datos)]
    BL --> EXT[Servicios Externos]
"""
        
        connections = ""  # Ya incluido en components
        
        return template["template_es"].format(components=components, connections=connections)
    
    async def _highlight_problematic_lines(self, code: str) -> List[Dict[str, Any]]:
        """Resaltar líneas problemáticas en el código."""
        # Simular resaltado de líneas problemáticas
        lines = code.split('\n')
        highlights = []
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['if', 'for', 'while', 'try']):
                highlights.append({
                    "line": i + 1,
                    "type": "warning",
                    "message": "Línea con lógica compleja"
                })
        
        return highlights
    
    async def _highlight_improvements(self, code: str) -> List[Dict[str, Any]]:
        """Resaltar mejoras en el código."""
        # Simular resaltado de mejoras
        lines = code.split('\n')
        highlights = []
        
        for i, line in enumerate(lines):
            if 'return' in line.lower() and i < len(lines) - 1:
                highlights.append({
                    "line": i + 1,
                    "type": "success",
                    "message": "Early return implementado"
                })
        
        return highlights
    
    async def _calculate_differences(self, original_code: str, fixed_code: str) -> Dict[str, Any]:
        """Calcular diferencias entre código original y corregido."""
        original_lines = original_code.split('\n')
        fixed_lines = fixed_code.split('\n')
        
        return {
            "lines_added": max(0, len(fixed_lines) - len(original_lines)),
            "lines_removed": max(0, len(original_lines) - len(fixed_lines)),
            "lines_modified": len([i for i in range(min(len(original_lines), len(fixed_lines))) 
                                 if original_lines[i] != fixed_lines[i]]),
            "total_changes": abs(len(fixed_lines) - len(original_lines))
        }
    
    async def _identify_improvements(self, original_code: str, fixed_code: str) -> List[str]:
        """Identificar mejoras en el código corregido."""
        improvements = []
        
        # Analizar mejoras específicas
        if 'if' in original_code and 'return' in fixed_code:
            improvements.append("Implementación de early returns")
        
        if original_code.count('if') > fixed_code.count('if'):
            improvements.append("Reducción de anidamiento")
        
        if len(fixed_code.split('\n')) < len(original_code.split('\n')):
            improvements.append("Reducción de líneas de código")
        
        return improvements
    
    async def _generate_comparison_explanation(self, original_code: str, fixed_code: str) -> str:
        """Generar explicación de la comparación."""
        improvements = await self._identify_improvements(original_code, fixed_code)
        
        if improvements:
            explanation = "**Mejoras implementadas:**\n\n"
            for improvement in improvements:
                explanation += f"• {improvement}\n"
            explanation += "\n**Beneficios:**\n"
            explanation += "• Código más legible y mantenible\n"
            explanation += "• Reducción de complejidad ciclomática\n"
            explanation += "• Mejor rendimiento y claridad"
        else:
            explanation = "El código ha sido refactorizado para mejorar su estructura y legibilidad."
        
        return explanation
