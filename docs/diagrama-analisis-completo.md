# 🔍 Diagrama Completo de Análisis - CodeAnt Agent

Este diagrama muestra todos los análisis que se ejecutan cuando se pulsa el botón "Analizar Proyecto" en el dashboard.

```mermaid
graph TB
    subgraph "🚀 Análisis al Pulsar 'Analizar Proyecto'"
        A[Analizar Proyecto] --> B{Análisis en Paralelo<br/>asyncio.gather()}
        
        B --> C["🔧 Análisis de Complejidad<br/>(include_complexity)"]
        B --> D["📊 Métricas de Calidad<br/>(include_metrics)"]
        B --> E["💀 Código Muerto<br/>(include_dead_code)"]
        B --> F["🔒 Seguridad<br/>(include_security)"]
        B --> G["📑 Duplicados<br/>(include_duplicates)"]
        B --> H["🐛 Bugs Potenciales<br/>(include_bugs)"]
        B --> I["📦 Dependencias<br/>(include_dependencies)"]
        B --> J["🧪 Cobertura de Tests<br/>(include_test_coverage)"]
        B --> K["⚡ Performance<br/>(include_performance)"]
        B --> L["🏗️ Arquitectura<br/>(include_architecture)"]
        B --> M["📝 Documentación<br/>(include_documentation)"]
        
        %% Análisis de Complejidad
        C --> C1["<b>Métricas Calculadas:</b><br/>• Complejidad Ciclomática<br/>• Complejidad Cognitiva<br/>• Número de parámetros<br/>• Profundidad de anidamiento<br/>• Líneas de código por función"]
        C1 --> C2["<b>Tecnologías:</b><br/>• AST Parser por lenguaje<br/>• Tree-sitter (Python/JS/TS/Rust)<br/>• Cross-language analysis<br/>• Unified AST"]
        
        %% Métricas de Calidad
        D --> D1["<b>Índices Calculados:</b><br/>• Índice de Mantenibilidad<br/>• Deuda Técnica (TODOs/FIXMEs)<br/>• Cobertura de Documentación<br/>• Code Smells detectados<br/>• Ratio comentarios/código"]
        D1 --> D2["<b>Fórmulas:</b><br/>• MI = 171 - 5.2*ln(V) - 0.23*CC - 16.2*ln(LOC)<br/>• Tech Debt = TODOs * 30min<br/>• Doc Coverage = (documented/total) * 100"]
        
        %% Código Muerto
        E --> E1["<b>Detección de:</b><br/>• Variables no usadas<br/>• Funciones no usadas<br/>• Clases no usadas<br/>• Imports no usados<br/>• Código inalcanzable"]
        E1 --> E2["<b>Análisis AST:</b><br/>• Control Flow Graph<br/>• Scope analysis<br/>• Reference tracking<br/>• Cross-module analysis"]
        
        %% Seguridad
        F --> F1["<b>Vulnerabilidades:</b><br/>• Secretos hardcodeados (API keys, passwords)<br/>• Funciones inseguras (eval, exec)<br/>• SQL Injection patterns<br/>• XSS vulnerabilities<br/>• Path traversal"]
        F1 --> F2["<b>Compliance:</b><br/>• OWASP Top 10<br/>• CWE categorization<br/>• Severity levels (CRITICAL/HIGH/MEDIUM/LOW)<br/>• Security hotspots"]
        
        %% Duplicados
        G --> G1["<b>Análisis:</b><br/>• Archivos con nombres similares<br/>• Bloques de código duplicados<br/>• Funciones similares<br/>• Porcentaje de duplicación<br/>• Clone detection"]
        G1 --> G2["<b>Algoritmos:</b><br/>• Levenshtein distance<br/>• Token-based comparison<br/>• AST similarity<br/>• Hash-based detection"]
        
        %% Bugs Potenciales
        H --> H1["<b>Patrones Detectados:</b><br/>• Null pointer exceptions<br/>• División por cero<br/>• Índices fuera de rango<br/>• Bucles infinitos<br/>• Race conditions"]
        H1 --> H2["<b>Análisis por Lenguaje:</b><br/>• Python: .get().method, except:<br/>• JS/TS: JSON.parse sin try-catch<br/>• TypeScript: uso de 'any'<br/>• Memory leaks: archivos sin cerrar"]
        
        %% Dependencias
        I --> I1["<b>Información:</b><br/>• Total de dependencias<br/>• Directas vs desarrollo<br/>• Dependencias obsoletas<br/>• Vulnerabilidades conocidas<br/>• Licencias problemáticas"]
        I1 --> I2["<b>Archivos Analizados:</b><br/>• package.json<br/>• pyproject.toml<br/>• requirements.txt<br/>• Cargo.toml<br/>• go.mod"]
        
        %% Cobertura de Tests
        J --> J1["<b>Métricas:</b><br/>• Archivos con tests<br/>• Funciones de test<br/>• Cobertura estimada %<br/>• Tipos: unit/integration/e2e<br/>• Test quality score"]
        J1 --> J2["<b>Detección:</b><br/>• test_*.py, *_test.py<br/>• *.test.js, *.spec.ts<br/>• describe(), it(), test()<br/>• pytest, jest, mocha"]
        
        %% Performance
        K --> K1["<b>Problemas Detectados:</b><br/>• Algoritmos O(n²)<br/>• N+1 queries<br/>• Operaciones bloqueantes<br/>• DOM manipulation ineficiente<br/>• Sync en contexto async"]
        K1 --> K2["<b>Patrones:</b><br/>• Bucles anidados<br/>• querySelector en loops<br/>• innerHTML +=<br/>• Array.filter().map()<br/>• time.sleep(), requests síncronos"]
        
        %% Arquitectura
        L --> L1["<b>Violaciones:</b><br/>• Violaciones de capas<br/>• God classes (>500 líneas)<br/>• God functions (>50 líneas)<br/>• Alto acoplamiento (>15 imports)<br/>• Dependencias circulares"]
        L1 --> L2["<b>Análisis Hexagonal:</b><br/>• Domain → Infrastructure ❌<br/>• Application → Domain ✅<br/>• Infrastructure → Application ✅<br/>• Presentation → Application ✅"]
        
        %% Documentación
        M --> M1["<b>Cobertura:</b><br/>• Funciones documentadas %<br/>• Clases documentadas %<br/>• README quality score<br/>• Inline comments ratio<br/>• TODOs/FIXMEs obsoletos"]
        M1 --> M2["<b>Verificación:</b><br/>• Python: docstrings<br/>• JS/TS: JSDoc<br/>• README sections<br/>• API documentation<br/>• Code examples"]
        
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
        
        R --> R1["<b>AnalysisResults:</b><br/>• Total violations<br/>• Critical/High/Medium/Low<br/>• Quality score (0-100)<br/>• Files analyzed<br/>• Execution time"]
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
