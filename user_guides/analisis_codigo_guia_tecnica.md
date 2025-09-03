# Guía Técnica: Análisis de Código con CodeAnt

## Índice
1. [Arquitectura del sistema de análisis](#arquitectura-del-sistema-de-análisis)
2. [Flujo de trabajo técnico](#flujo-de-trabajo-técnico)
3. [Motor de reglas estáticas](#motor-de-reglas-estáticas)
4. [Detección de clones y código duplicado](#detección-de-clones-y-código-duplicado)
5. [Análisis incremental](#análisis-incremental)
6. [Configuración avanzada](#configuración-avanzada)
7. [Integración con CI/CD](#integración-con-cicd)

## Arquitectura del sistema de análisis

CodeAnt implementa una arquitectura hexagonal (también conocida como puertos y adaptadores) que separa claramente:

```
┌────────────────────────────────────────────┐
│               Dominio                      │
│  ┌─────────────┐       ┌─────────────┐     │
│  │   Entities  │       │   Services  │     │
│  └─────────────┘       └─────────────┘     │
└────────────────────────────────────────────┘
              │                 ▲
              ▼                 │
┌────────────────────────────────────────────┐
│              Aplicación                    │
│  ┌─────────────┐       ┌─────────────┐     │
│  │  Use Cases  │       │    DTOs     │     │
│  └─────────────┘       └─────────────┘     │
└────────────────────────────────────────────┘
              │                 ▲
              ▼                 │
┌────────────────────────────────────────────┐
│           Infraestructura                  │
│  ┌─────────────┐       ┌─────────────┐     │
│  │Repositories │       │  Adapters   │     │
│  └─────────────┘       └─────────────┘     │
└────────────────────────────────────────────┘
              │                 ▲
              ▼                 │
┌────────────────────────────────────────────┐
│           Presentación                     │
│  ┌─────────────┐       ┌─────────────┐     │
│  │Controllers  │       │    Views    │     │
│  └─────────────┘       └─────────────┘     │
└────────────────────────────────────────────┘
```

El sistema de análisis de código es modular y se compone de varios subsistemas:

1. **Sistema de parseo**: Convierte el código fuente en AST (Abstract Syntax Tree).
2. **Motor de reglas**: Aplica reglas estáticas para detectar problemas.
3. **Analizador de duplicación**: Detecta código duplicado.
4. **Analizador de métricas**: Calcula métricas de complejidad, cohesión, acoplamiento, etc.
5. **Sistema de recomendaciones**: Sugiere mejoras basadas en patrones detectados.

## Flujo de trabajo técnico

Cuando se inicia un análisis de código desde la interfaz de usuario, ocurren los siguientes pasos:

1. **Clonación del repositorio**: Si es la primera vez, se clona el repositorio completo; si no, se actualiza mediante `git pull`.

2. **Descubrimiento de archivos**: Se indexan todos los archivos relevantes, excluyendo los que coinciden con patrones de ignorado (similar a `.gitignore`).

3. **Análisis incremental**: Se determinan los archivos que han cambiado desde el último análisis para optimizar el rendimiento.

4. **Parseo de código**: Cada archivo se parsea para generar su representación AST.

5. **Aplicación de reglas**: Se ejecutan las reglas de análisis estático sobre los AST.

6. **Detección de duplicación**: Se buscan fragmentos de código similares o idénticos.

7. **Cálculo de métricas**: Se calculan diversas métricas de calidad de código.

8. **Generación de informe**: Se compilan todos los resultados en un informe estructurado.

9. **Persistencia de resultados**: Los resultados se almacenan en la base de datos para seguimiento.

10. **Notificación**: Se notifica a la interfaz de usuario que el análisis ha finalizado.

## Motor de reglas estáticas

El motor de reglas de CodeAnt es extensible y configurable. Las reglas se organizan en categorías:

- **Estilo de código**: Convenciones de nombrado, indentación, etc.
- **Complejidad**: Complejidad ciclomática, funciones demasiado largas, etc.
- **Rendimiento**: Operaciones ineficientes, uso excesivo de memoria, etc.
- **Seguridad**: Vulnerabilidades conocidas, prácticas inseguras.
- **Mantenibilidad**: Código duplicado, acoplamiento excesivo, etc.

Cada regla tiene los siguientes atributos:

- **ID**: Identificador único
- **Nombre**: Nombre descriptivo
- **Descripción**: Explicación del problema
- **Severidad**: Crítica, Alta, Media, Baja, Informativa
- **Etiquetas**: Categorías a las que pertenece
- **Impacto**: Áreas afectadas (mantenibilidad, rendimiento, etc.)
- **Soluciones**: Recomendaciones para resolver el problema

## Detección de clones y código duplicado

CodeAnt utiliza varios algoritmos para detectar código duplicado:

1. **Detección basada en tokens**: Para encontrar código sintácticamente similar.
2. **Detección semántica**: Para identificar código funcionalmente equivalente aunque sintácticamente diferente.
3. **Detección de clones de API**: Para encontrar uso repetitivo de APIs externas.

Los clones se clasifican en:

- **Tipo 1**: Clones exactos (copy-paste).
- **Tipo 2**: Clones con cambios en nombres de variables, literales, etc.
- **Tipo 3**: Clones con cambios estructurales menores (líneas añadidas/eliminadas).
- **Tipo 4**: Clones semánticos (misma funcionalidad, diferente implementación).

## Análisis incremental

Para mejorar el rendimiento, CodeAnt implementa análisis incremental que:

1. Mantiene un índice de archivos previamente analizados.
2. Utiliza la información de git para identificar archivos modificados.
3. Realiza análisis completo solo en archivos que han cambiado.
4. Conserva resultados previos para archivos no modificados.
5. Actualiza solo las métricas globales que se ven afectadas.

Esto reduce significativamente el tiempo de análisis en repositorios grandes.

## Configuración avanzada

Los usuarios pueden personalizar el análisis mediante archivos de configuración `.codeant.json` en la raíz del repositorio:

```json
{
  "ignore_patterns": [
    "node_modules/**",
    "dist/**",
    "**/*.min.js"
  ],
  "include_patterns": [
    "src/**",
    "lib/**"
  ],
  "rules": {
    "complexity": {
      "max_cyclomatic_complexity": 15,
      "max_function_length": 100
    },
    "duplications": {
      "min_lines": 5,
      "min_tokens": 100,
      "ignore_literals": true
    }
  },
  "max_file_size_mb": 2,
  "enable_incremental_analysis": true,
  "parallel_analysis_batch_size": 20
}
```

## Integración con CI/CD

CodeAnt se puede integrar en flujos de trabajo CI/CD para análisis automático:

### GitHub Actions

```yaml
name: CodeAnt Analysis

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
      
      - name: Run CodeAnt Analysis
        uses: codeant/github-action@v1
        with:
          api_key: ${{ secrets.CODEANT_API_KEY }}
          project_id: "your-project-id"
          fail_on_severity: "critical"
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    stages {
        stage('Analyze') {
            steps {
                sh 'codeant-cli analyze --project-id=your-project-id --api-key=$CODEANT_API_KEY'
            }
            post {
                always {
                    publishCodeAnt()
                }
            }
        }
    }
}
```

### GitLab CI

```yaml
analyze:
  stage: test
  image: codeant/cli:latest
  script:
    - codeant-cli analyze --project-id=your-project-id --api-key=$CODEANT_API_KEY
  artifacts:
    reports:
      codeant: codeant-report.json
```

---

Para más información técnica, consulta la documentación completa de la API en https://docs.codeant.example.com o contacta con el equipo técnico en tech-support@codeant.example.com.

© 2025 CodeAnt - Análisis de Código Inteligente
