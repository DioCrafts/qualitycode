# 🔍 Diagrama Completo de Análisis - CodeAnt Agent

Este diagrama muestra todos los análisis que se ejecutan cuando se pulsa el botón "Analizar Proyecto" en el dashboard.

```mermaid
graph TB
    subgraph "🚀 Análisis al Pulsar 'Analizar Proyecto'"
        A[Analizar Proyecto] --> B{Análisis en Paralelo asyncio.gather}
        
        B --> C["🔧 Análisis de Complejidad (include_complexity)"]
        B --> D["📊 Métricas de Calidad (include_metrics)"]
        B --> E["💀 Código Muerto (include_dead_code)"]
        B --> F["🔒 Seguridad (include_security)"]
        B --> G["📑 Duplicados (include_duplicates)"]
        B --> H["🐛 Bugs Potenciales (include_bugs)"]
        B --> I["📦 Dependencias (include_dependencies)"]
        B --> J["🧪 Cobertura de Tests (include_test_coverage)"]
        B --> K["⚡ Performance (include_performance)"]
        B --> L["🏗️ Arquitectura (include_architecture)"]
        B --> M["📝 Documentación (include_documentation)"]
        
        %% Análisis de Complejidad
        C --> C1["Métricas Calculadas: • Complejidad Ciclomática • Complejidad Cognitiva • Número de parámetros • Profundidad de anidamiento • Líneas de código por función"]
        C1 --> C2["Tecnologías: • AST Parser por lenguaje • Tree-sitter (Python/JS/TS/Rust) • Cross-language analysis • Unified AST"]
        
        %% Métricas de Calidad
        D --> D1["Índices Calculados: • Índice de Mantenibilidad • Deuda Técnica (TODOs/FIXMEs) • Cobertura de Documentación • Code Smells detectados • Ratio comentarios/código"]
        D1 --> D2["Fórmulas: • MI = 171 - 5.2*ln(V) - 0.23*CC - 16.2*ln(LOC) • Tech Debt = TODOs * 30min • Doc Coverage = (documented/total) * 100"]
        
        %% Código Muerto
        E --> E1["Detección de: • Variables no usadas • Funciones no usadas • Clases no usadas • Imports no usados • Código inalcanzable"]
        E1 --> E2["Análisis AST: • Control Flow Graph • Scope analysis • Reference tracking • Cross-module analysis"]
        
        %% Seguridad
        F --> F1["Vulnerabilidades: • Secretos hardcodeados (API keys, passwords) • Funciones inseguras (eval, exec) • SQL Injection patterns • XSS vulnerabilities • Path traversal"]
        F1 --> F2["Compliance: • OWASP Top 10 • CWE categorization • Severity levels (CRITICAL/HIGH/MEDIUM/LOW) • Security hotspots"]
        
        %% Duplicados
        G --> G1["Análisis: • Archivos con nombres similares • Bloques de código duplicados • Funciones similares • Porcentaje de duplicación • Clone detection"]
        G1 --> G2["Algoritmos: • Levenshtein distance • Token-based comparison • AST similarity • Hash-based detection"]
        
        %% Bugs Potenciales
        H --> H1["Patrones Detectados: • Null pointer exceptions • División por cero • Índices fuera de rango • Bucles infinitos • Race conditions"]
        H1 --> H2["Análisis por Lenguaje: • Python: .get().method, except: • JS/TS: JSON.parse sin try-catch • TypeScript: uso de 'any' • Memory leaks: archivos sin cerrar"]
        
        %% Dependencias
        I --> I1["Información: • Total de dependencias • Directas vs desarrollo • Dependencias obsoletas • Vulnerabilidades conocidas • Licencias problemáticas"]
        I1 --> I2["Archivos Analizados: • package.json • pyproject.toml • requirements.txt • Cargo.toml • go.mod"]
        
        %% Cobertura de Tests
        J --> J1["Métricas: • Archivos con tests • Funciones de test • Cobertura estimada % • Tipos: unit/integration/e2e • Test quality score"]
        J1 --> J2["Detección: • test_*.py, *_test.py • *.test.js, *.spec.ts • describe(), it(), test() • pytest, jest, mocha"]
        
        %% Performance
        K --> K1["Problemas Detectados: • Algoritmos O(n²) • N+1 queries • Operaciones bloqueantes • DOM manipulation ineficiente • Sync en contexto async"]
        K1 --> K2["Patrones: • Bucles anidados • querySelector en loops • innerHTML += • Array.filter().map() • time.sleep(), requests síncronos"]
        
        %% Arquitectura
        L --> L1["Violaciones: • Violaciones de capas • God classes (>500 líneas) • God functions (>50 líneas) • Alto acoplamiento (>15 imports) • Dependencias circulares"]
        L1 --> L2["Análisis Hexagonal: • Domain → Infrastructure ❌ • Application → Domain ✅ • Infrastructure → Application ✅ • Presentation → Application ✅"]
        
        %% Documentación
        M --> M1["Cobertura: • Funciones documentadas % • Clases documentadas % • README quality score • Inline comments ratio • TODOs/FIXMEs obsoletos"]
        M1 --> M2["Verificación: • Python: docstrings • JS/TS: JSDoc • README sections • API documentation • Code examples"]
        
        %% Resultado Final
        C2 --> R[📊 Resultado Consolidado]
        D2 --> R
        E2 --> R
        F2 --> R
        G2 --> R
        H2 --> R
        I2 --> R
        J2 --> R
        K2 --> R
        L2 --> R
        M2 --> R
        
        R --> R1["AnalysisResults: • Total violations • Critical/High/Medium/Low • Quality score (0-100) • Files analyzed • Execution time"]
    end
    
    %% Estilos
    style A fill:#4a6cf7,stroke:#3955d8,stroke-width:3px,color:#fff
    style B fill:#1e293b,stroke:#0f172a,stroke-width:2px,color:#fff
    style C fill:#8b5cf6,stroke:#7c3aed,stroke-width:2px,color:#fff
    style D fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    style E fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    style F fill:#ef4444,stroke:#dc2626,stroke-width:2px,color:#fff
    style G fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#fff
    style H fill:#ec4899,stroke:#db2777,stroke-width:2px,color:#fff
    style I fill:#14b8a6,stroke:#0d9488,stroke-width:2px,color:#fff
    style J fill:#a855f7,stroke:#9333ea,stroke-width:2px,color:#fff
    style K fill:#f97316,stroke:#ea580c,stroke-width:2px,color:#fff
    style L fill:#6366f1,stroke:#4f46e5,stroke-width:2px,color:#fff
    style M fill:#84cc16,stroke:#65a30d,stroke-width:2px,color:#fff
    style R fill:#0ea5e9,stroke:#0284c7,stroke-width:3px,color:#fff
    style R1 fill:#0f172a,stroke:#020617,stroke-width:2px,color:#fff
    
```

## 📋 Resumen de Análisis Implementados

### Análisis Originales (5)
1. **🔧 Complejidad** - Análisis AST multi-lenguaje con métricas avanzadas
2. **📊 Métricas de Calidad** - Índice de mantenibilidad y deuda técnica
3. **💀 Código Muerto** - Detección de código no utilizado con Tree-sitter
4. **🔒 Seguridad** - Escaneo de vulnerabilidades y OWASP compliance
5. **📑 Duplicados** - Detección de código duplicado

### Nuevos Análisis Implementados (6)
6. **🐛 Bugs Potenciales** - Detección proactiva de errores comunes
7. **📦 Dependencias** - Análisis de vulnerabilidades y obsolescencia
8. **🧪 Cobertura de Tests** - Evaluación de la calidad de testing
9. **⚡ Performance** - Detección de problemas de rendimiento
10. **🏗️ Arquitectura** - Verificación de arquitectura hexagonal
11. **📝 Documentación** - Análisis de calidad de documentación

## 🔄 Flujo de Ejecución

1. **Usuario** pulsa "Analizar Proyecto" en el dashboard
2. **Frontend** envía POST a `/api/analysis/run`
3. **Backend** ejecuta todos los análisis en paralelo usando `asyncio.gather()`
4. **Cada análisis** procesa los archivos según su especialidad
5. **Resultados** se consolidan en `AnalysisResults`
6. **Dashboard** muestra los resultados actualizados

## 📊 Métricas Finales

El sistema calcula un **Quality Score** global basado en:
- Número de violaciones por severidad
- Densidad de problemas por archivo
- Cobertura de tests y documentación
- Cumplimiento de mejores prácticas

```
Quality Score = 100 - (weighted_violations / files_analyzed * 10)
```

Donde las violaciones se ponderan:
- Critical: 10 puntos
- High: 5 puntos
- Medium: 2 puntos
- Low: 1 punto
