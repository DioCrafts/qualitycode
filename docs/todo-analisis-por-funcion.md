# 📋 TODO: Análisis Detallado por Función

## 🎯 Objetivo
Implementar análisis detallado por función para las pestañas de Calidad y Complejidad, reemplazando los datos simulados actuales con análisis reales del backend.

## 🔍 Estado Actual
- **Frontend:** Pestañas de Calidad y Complejidad mejoradas con datos simulados
- **Backend:** Solo genera métricas básicas (total_functions, complex_functions, etc.)
- **Falta:** Análisis detallado por función individual

## 📊 Datos Simulados Actuales (a reemplazar)

### Pestaña de Calidad:
- Problemas específicos de funciones (función muy larga, código duplicado)
- Ubicaciones exactas (archivo:línea)
- Recomendaciones específicas por problema
- Plan de mejora priorizado

### Pestaña de Complejidad:
- Hotspots de complejidad por función
- Factores de complejidad (condiciones, bucles, anidamiento)
- Patrones de complejidad detectados
- Plan de refactorización por prioridad

## 🚀 Implementación Futura

### 1. Backend - Análisis por Función
```python
# Estructura de datos a implementar
class FunctionAnalysis:
    name: str
    file: str
    line: int
    end_line: int
    complexity: int
    cognitive_complexity: int
    conditions: int
    loops: int
    switches: int
    max_nesting: int
    quality_issues: List[QualityIssue]
    recommendations: List[str]
    code_preview: str
```

### 2. Backend - Problemas de Calidad
```python
class QualityIssue:
    type: str  # "long_function", "duplicate_code", "bad_naming", etc.
    severity: str  # "critical", "warning", "info"
    description: str
    file: str
    line: int
    impact: str
    recommendations: List[str]
```

### 3. Backend - Patrones de Complejidad
```python
class ComplexityPattern:
    type: str  # "nested_loops", "deep_conditions", "complex_switch", etc.
    count: int
    impact: str
    affected_functions: List[str]
    recommendations: List[str]
```

## 🛠️ Tareas Técnicas

### Backend:
- [ ] Extender `analyze_project_use_case.py` para análisis por función
- [ ] Implementar detección de problemas de calidad específicos
- [ ] Agregar análisis de patrones de complejidad
- [ ] Generar recomendaciones automáticas
- [ ] Crear endpoints para datos detallados por función

### Frontend:
- [ ] Conectar con nuevos endpoints del backend
- [ ] Reemplazar datos simulados con datos reales
- [ ] Implementar filtros y búsqueda por función
- [ ] Agregar funcionalidad "Ver archivo" para problemas de calidad
- [ ] Mejorar visualización de patrones de complejidad

## 📈 Beneficios Esperados

1. **Análisis real:** Datos basados en el código actual del proyecto
2. **Accionable:** Recomendaciones específicas para cada función
3. **Priorizado:** Problemas ordenados por impacto real
4. **Medible:** Métricas de mejora cuantificables
5. **Profesional:** Herramienta enterprise de calidad

## 🎯 Prioridad
**Alta** - Mejora significativa en la utilidad del dashboard

## 📅 Estimación
**2-3 sprints** para implementación completa

## 🔗 Archivos Relacionados
- `src/codeant_agent/application/use_cases/analyze_project_use_case.py`
- `src/codeant_agent/presentation/api/routers/analysis.py`
- `dashboard/src/routes/analysis/[id]/+page.svelte`

## 📝 Notas
- Mantener compatibilidad con análisis actual
- Considerar performance para proyectos grandes
- Implementar paginación para listas largas de funciones
- Agregar tests unitarios para nuevos análisis
