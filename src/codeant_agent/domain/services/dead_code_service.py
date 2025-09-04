"""
Servicios de dominio para código muerto.
"""
from typing import Dict, Any, List

from ..entities.dead_code_analysis import DeadCodeAnalysis

class DeadCodeClassificationService:
    """
    Servicio para clasificar y generar sugerencias para issues de código muerto.
    """
    
    def classify_dead_code_analysis(self, analysis: DeadCodeAnalysis) -> Dict[str, List[Any]]:
        """
        Clasificar issues encontrados en el análisis.
        
        Args:
            analysis: Análisis de código muerto
            
        Returns:
            Diccionario con issues clasificados
        """
        # Clasificar en categorías
        classified = {}
        
        # Variables no utilizadas
        if analysis.unused_variables:
            classified["unused_variables"] = analysis.unused_variables
        
        # Funciones no utilizadas
        if analysis.unused_functions:
            classified["unused_functions"] = analysis.unused_functions
        
        # Clases no utilizadas
        if analysis.unused_classes:
            classified["unused_classes"] = analysis.unused_classes
        
        # Imports no utilizados
        if analysis.unused_imports:
            classified["unused_imports"] = analysis.unused_imports
        
        # Código inalcanzable
        if analysis.unreachable_code:
            classified["unreachable_code"] = analysis.unreachable_code
        
        # Ramas muertas
        if analysis.dead_branches:
            classified["dead_branches"] = analysis.dead_branches
        
        # Parámetros no utilizados
        if analysis.unused_parameters:
            classified["unused_parameters"] = analysis.unused_parameters
        
        # Asignaciones redundantes
        if analysis.redundant_assignments:
            classified["redundant_assignments"] = analysis.redundant_assignments
        
        return classified
    
    def generate_removal_suggestions(self, classified_issues: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generar sugerencias para eliminar código muerto.
        
        Args:
            classified_issues: Issues clasificados
            
        Returns:
            Lista de sugerencias
        """
        suggestions = []
        
        # Sugerencias para imports
        if "unused_imports" in classified_issues and classified_issues["unused_imports"]:
            suggestions.append({
                'action': 'remove_imports',
                'items': classified_issues["unused_imports"],
                'description': 'Eliminar imports no utilizados',
                'risk_level': 'low',
                'priority': 'high'
            })
        
        # Sugerencias para variables
        if "unused_variables" in classified_issues and classified_issues["unused_variables"]:
            suggestions.append({
                'action': 'remove_variables',
                'items': classified_issues["unused_variables"],
                'description': 'Eliminar variables no utilizadas',
                'risk_level': 'low',
                'priority': 'medium'
            })
        
        # Sugerencias para código inalcanzable
        if "unreachable_code" in classified_issues and classified_issues["unreachable_code"]:
            suggestions.append({
                'action': 'remove_unreachable',
                'items': classified_issues["unreachable_code"],
                'description': 'Eliminar código inalcanzable',
                'risk_level': 'low',
                'priority': 'medium'
            })
        
        # Sugerencias para funciones
        if "unused_functions" in classified_issues and classified_issues["unused_functions"]:
            suggestions.append({
                'action': 'remove_functions',
                'items': classified_issues["unused_functions"],
                'description': 'Eliminar funciones no utilizadas',
                'risk_level': 'medium',
                'priority': 'medium'
            })
        
        # Sugerencias para clases
        if "unused_classes" in classified_issues and classified_issues["unused_classes"]:
            suggestions.append({
                'action': 'remove_classes',
                'items': classified_issues["unused_classes"],
                'description': 'Eliminar clases no utilizadas',
                'risk_level': 'medium',
                'priority': 'medium'
            })
        
        # Sugerencias para parámetros
        if "unused_parameters" in classified_issues and classified_issues["unused_parameters"]:
            suggestions.append({
                'action': 'remove_parameters',
                'items': classified_issues["unused_parameters"],
                'description': 'Eliminar parámetros no utilizados',
                'risk_level': 'medium',
                'priority': 'low'
            })
        
        return suggestions