# Sistema de Reglas Personalizadas en Lenguaje Natural

## Descripción

Este módulo implementa un sistema revolucionario que permite a los usuarios crear reglas de análisis personalizadas escribiendo en lenguaje natural (español e inglés). Utiliza procesamiento de lenguaje natural avanzado y modelos de IA para traducir descripciones humanas en reglas ejecutables, proporcionando flexibilidad total para organizaciones que necesitan reglas específicas de dominio o estándares internos únicos.

## Arquitectura

El sistema sigue una arquitectura hexagonal con las siguientes capas:

### Capa de Dominio

- **Entidades**: Definiciones de reglas, intenciones, condiciones, acciones, etc.
- **Repositorios**: Interfaces para persistencia de reglas y feedback.
- **Servicios**: Lógica de negocio para procesamiento de lenguaje natural y generación de reglas.

### Capa de Aplicación

- **Puertos**: Interfaces para componentes externos.
- **DTOs**: Objetos de transferencia de datos para la API.
- **Casos de Uso**: Orquestación de la lógica de negocio.

### Capa de Infraestructura

- **NLP Processor**: Procesamiento de lenguaje natural.
- **Intent Extractor**: Extracción de intenciones.
- **Entity Extractor**: Extracción de entidades.
- **Pattern Matcher**: Búsqueda de patrones.
- **Rule Generator**: Generación de reglas ejecutables.
- **Code Generator**: Generación de código.
- **Template Engine**: Motor de plantillas.
- **Rule Validator**: Validación de reglas.
- **Feedback Collector**: Recolección de feedback.
- **Rule Optimizer**: Optimización de reglas.
- **Pattern Learner**: Aprendizaje de patrones.
- **Accuracy Monitor**: Monitoreo de precisión.

### Capa de Presentación

- **API REST**: Endpoints para interactuar con el sistema.

## Flujo de Procesamiento

1. El usuario proporciona una descripción en lenguaje natural de una regla.
2. El sistema procesa el texto utilizando NLP para extraer intenciones, entidades y patrones.
3. Se genera una estructura de regla basada en el análisis.
4. Se genera una regla ejecutable a partir de la estructura.
5. La regla se valida para asegurar su corrección.
6. La regla se puede ejecutar para analizar código.
7. El usuario puede proporcionar feedback para mejorar la regla.
8. El sistema aprende del feedback para mejorar futuras reglas.

## Ejemplos de Reglas en Lenguaje Natural

### Español

- "Las funciones no deben tener más de 50 líneas de código"
- "Todas las clases que manejan datos sensibles deben tener validación de entrada"
- "Los métodos que acceden a la base de datos deben usar consultas parametrizadas"
- "Las funciones que contienen la palabra 'password' no deben hacer logging del contenido"
- "Si una función tiene más de 5 parámetros, debe ser refactorizada"
- "Los bucles anidados con más de 3 niveles son problemáticos"
- "Las variables que contienen 'secret' o 'key' no deben ser hardcodeadas"

### Inglés

- "Functions should not exceed 50 lines of code"
- "All classes handling sensitive data must have input validation"
- "Methods accessing the database must use parameterized queries"
- "Functions containing the word 'password' must not log content"
- "If a function has more than 5 parameters, it should be refactored"
- "Nested loops with more than 3 levels are problematic"
- "Variables containing 'secret' or 'key' must not be hardcoded"

## API REST

### Endpoints

- `POST /api/v1/natural-rules/process-text`: Procesa un texto en lenguaje natural.
- `POST /api/v1/natural-rules/`: Crea una regla en lenguaje natural.
- `GET /api/v1/natural-rules/{rule_id}`: Obtiene una regla por su ID.
- `POST /api/v1/natural-rules/generate`: Genera una regla ejecutable a partir de una estructura.
- `POST /api/v1/natural-rules/validate`: Valida una regla ejecutable.
- `POST /api/v1/natural-rules/generate-code`: Genera código para una regla.
- `POST /api/v1/natural-rules/learning/feedback`: Procesa feedback para una regla.
- `POST /api/v1/natural-rules/learning/optimize/{rule_id}`: Optimiza una regla.
- `GET /api/v1/natural-rules/learning/analyze-feedback/{rule_id}`: Analiza el feedback para una regla.
- `GET /api/v1/natural-rules/learning/improvements/{rule_id}`: Obtiene mejoras sugeridas para una regla.

## Tests

El sistema incluye tests unitarios y de integración para asegurar su correcto funcionamiento:

- **Tests Unitarios**: Verifican el funcionamiento de componentes individuales.
- **Tests de Integración**: Verifican la interacción entre componentes y el flujo completo.

## Requisitos

- Python 3.8+
- FastAPI
- Pytest (para tests)
- Httpx (para tests de API)
- Regex (para procesamiento de texto)

## Uso

```python
# Ejemplo de uso del sistema
from codeant_agent.domain.entities.natural_rules.language import Language
from codeant_agent.infrastructure.natural_rules.nlp_processor import NLPProcessor
from codeant_agent.infrastructure.natural_rules.intent_extractor import IntentExtractor
from codeant_agent.infrastructure.natural_rules.entity_extractor import EntityExtractor
from codeant_agent.infrastructure.natural_rules.pattern_matcher import PatternMatcher
from codeant_agent.infrastructure.natural_rules.rule_generator import RuleGenerator
from codeant_agent.infrastructure.natural_rules.code_generator import CodeGenerator
from codeant_agent.infrastructure.natural_rules.template_engine import TemplateEngine
from codeant_agent.infrastructure.natural_rules.rule_validator import RuleValidator

async def process_natural_rule(text, language_str):
    # Inicializar componentes
    nlp_processor = NLPProcessor()
    intent_extractor = IntentExtractor()
    entity_extractor = EntityExtractor()
    pattern_matcher = PatternMatcher()
    template_engine = TemplateEngine()
    code_generator = CodeGenerator(template_engine)
    rule_generator = RuleGenerator(code_generator, template_engine)
    rule_validator = RuleValidator()
    
    # Convertir string de idioma a enum
    language = Language.from_string(language_str)
    
    # Procesar texto
    preprocessed_text = await nlp_processor.preprocess_text(text, language)
    intent_analysis = await intent_extractor.extract_intent(preprocessed_text, language)
    entities = await entity_extractor.extract_entities(preprocessed_text, language)
    pattern_matches = await pattern_matcher.find_patterns(preprocessed_text, language)
    
    # Crear estructura de regla
    rule_structure = RuleStructure(
        intent_analysis=intent_analysis,
        description=text
    )
    
    # Generar regla ejecutable
    executable_rule = await rule_generator.generate_rule(rule_structure)
    
    # Validar regla
    validation_result = await rule_validator.validate_rule(executable_rule)
    
    return {
        "rule": executable_rule,
        "validation": validation_result
    }
```
