"""
Motor de Explicaciones en Lenguaje Natural.

Este módulo implementa el motor de explicaciones que traduce automáticamente
los hallazgos técnicos de análisis de código en explicaciones claras y accionables
adaptadas a diferentes audiencias e idiomas.

Componentes principales:
- NaturalLanguageExplanationEngine: Motor principal de explicaciones
- ContentGenerator: Generador de contenido explicativo
- LanguageAdapter: Adaptador para múltiples idiomas
- AudienceAdapter: Adaptador para diferentes audiencias
- ExplanationTemplateEngine: Motor de plantillas
- EducationalContentGenerator: Generador de contenido educativo
- InteractiveExplainer: Sistema de explicaciones interactivas
- MultimediaGenerator: Generador de visualizaciones y diagramas

Características:
- Soporte multiidioma (español, inglés)
- Adaptación por audiencia (desarrolladores, managers, etc.)
- Explicaciones interactivas con Q&A
- Contenido educativo y ejemplos
- Visualizaciones y diagramas
- Personalización basada en contexto
- Performance optimizada (<3s generación, <1s respuesta interactiva)
"""

from .explanation_engine import NaturalLanguageExplanationEngine
from .content_generator import ContentGenerator
from .language_adapter import LanguageAdapter
from .audience_adapter import AudienceAdapter
from .template_engine import ExplanationTemplateEngine
from .educational_content_generator import EducationalContentGenerator
from .interactive_explainer import InteractiveExplainer
from .multimedia_generator import MultimediaGenerator
from .explanation_config import (
    CompleteExplanationConfig,
    DEFAULT_CONFIG,
    DEVELOPMENT_CONFIG,
    PRODUCTION_CONFIG,
    TESTING_CONFIG
)

__all__ = [
    # Componentes principales
    'NaturalLanguageExplanationEngine',
    'ContentGenerator',
    'LanguageAdapter',
    'AudienceAdapter',
    'ExplanationTemplateEngine',
    'EducationalContentGenerator',
    'InteractiveExplainer',
    'MultimediaGenerator',
    
    # Configuraciones
    'CompleteExplanationConfig',
    'DEFAULT_CONFIG',
    'DEVELOPMENT_CONFIG',
    'PRODUCTION_CONFIG',
    'TESTING_CONFIG'
]

__version__ = "1.0.0"
__author__ = "CodeAnt Team"
__description__ = "Motor de Explicaciones en Lenguaje Natural para Análisis de Código"
