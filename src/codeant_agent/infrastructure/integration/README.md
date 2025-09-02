# 🔗 Servicio de Integración Incremental

Este módulo proporciona la integración del sistema de análisis incremental con el sistema principal de CodeAnt, ofreciendo una interfaz unificada y simplificada para el análisis de código.

## 🎯 Características Principales

### ✨ Funcionalidades
- **Gestión de Sesiones**: Manejo completo del ciclo de vida de sesiones de análisis
- **Análisis Incremental**: Ejecución de análisis optimizados con caché inteligente
- **Detección de Cambios**: Identificación granular de modificaciones en el código
- **Predicción de Caché**: Anticipación de necesidades de caché basada en patrones
- **Métricas de Rendimiento**: Seguimiento detallado del rendimiento del sistema
- **Limpieza Automática**: Gestión automática de sesiones expiradas

### 🏗️ Arquitectura
```
┌─────────────────────────────────────────────────────────────┐
│                IncrementalIntegrationService                │
├─────────────────────────────────────────────────────────────┤
│  • Gestión de Sesiones Activas                              │
│  • Coordinación de Casos de Uso                             │
│  • Seguimiento de Métricas                                  │
│  • Limpieza Automática                                      │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Casos de Uso                             │
├─────────────────────────────────────────────────────────────┤
│  • IncrementalAnalysisUseCase                               │
│  • CacheManagementUseCase                                   │
│  • ChangeDetectionUseCase                                   │
│  • PredictiveCacheUseCase                                   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Uso del Servicio

### Inicialización
```python
from codeant_agent.infrastructure.integration.incremental_integration_service import IncrementalIntegrationService
from codeant_agent.application.use_cases.incremental_use_cases import (
    IncrementalAnalysisUseCase,
    CacheManagementUseCase,
    ChangeDetectionUseCase,
    PredictiveCacheUseCase
)

# Crear instancias de los casos de uso
incremental_use_case = IncrementalAnalysisUseCase()
cache_use_case = CacheManagementUseCase()
change_use_case = ChangeDetectionUseCase()
predictive_use_case = PredictiveCacheUseCase()

# Crear servicio de integración
integration_service = IncrementalIntegrationService(
    incremental_analysis_use_case=incremental_use_case,
    cache_management_use_case=cache_use_case,
    change_detection_use_case=change_use_case,
    predictive_cache_use_case=predictive_use_case
)
```

### Flujo de Trabajo Típico

#### 1. Iniciar Sesión de Análisis
```python
# Iniciar una nueva sesión
session_id = await integration_service.start_analysis_session(
    project_path="/path/to/project",
    session_config={
        "cache_strategy": "aggressive",
        "file_patterns": ["*.py", "*.js", "*.ts"]
    }
)
```

#### 2. Realizar Análisis Incremental
```python
# Ejecutar análisis incremental
result = await integration_service.perform_incremental_analysis(
    session_id=session_id,
    force_full_analysis=False,
    include_metrics=True
)

print(f"Archivos analizados: {result.files_analyzed}")
print(f"Tasa de acierto del caché: {result.cache_hit_rate}")
```

#### 3. Detectar Cambios
```python
# Detectar cambios en el código
changes = await integration_service.detect_changes(
    session_id=session_id,
    previous_commit="abc123",
    current_commit="def456"
)

print(f"Archivos modificados: {len(changes.changed_files)}")
print(f"Funciones modificadas: {len(changes.changed_functions)}")
```

#### 4. Gestionar Caché
```python
# Obtener estado del caché
cache_status = await integration_service.get_cache_status(session_id)
print(f"Tamaño del caché L1: {cache_status['l1_cache_size']}")

# Predecir uso del caché
prediction = await integration_service.predict_cache_usage(
    session_id=session_id,
    analysis_history_days=7,
    prediction_horizon_hours=24
)
print(f"Archivos predichos: {len(prediction.predicted_files)}")
```

#### 5. Finalizar Sesión
```python
# Finalizar la sesión y obtener resumen
summary = await integration_service.end_analysis_session(session_id)
print(f"Duración de la sesión: {summary['duration_seconds']} segundos")
print(f"Análisis realizados: {summary['analysis_count']}")
```

## 📊 Métricas y Monitoreo

### Métricas de Sesión
- **Duración**: Tiempo total de la sesión
- **Análisis**: Número de análisis realizados
- **Cambios**: Número de cambios detectados
- **Rendimiento**: Tiempo promedio de análisis

### Métricas del Sistema
- **Sesiones Activas**: Número de sesiones en curso
- **Análisis Totales**: Total de análisis realizados
- **Cambios Totales**: Total de cambios detectados
- **Tiempo Promedio**: Tiempo promedio de análisis

### Obtener Métricas
```python
# Métricas de una sesión específica
sessions = await integration_service.get_active_sessions()
for session in sessions:
    print(f"Sesión: {session['session_id']}")
    print(f"Análisis: {session['analysis_count']}")

# Métricas del sistema
system_metrics = await integration_service.get_system_metrics()
print(f"Sesiones activas: {system_metrics['active_sessions']}")
print(f"Análisis totales: {system_metrics['total_analyses']}")
```

## 🧹 Gestión de Recursos

### Limpieza Automática
El servicio incluye funcionalidad para limpiar sesiones expiradas automáticamente:

```python
# Limpiar sesiones expiradas (más de 24 horas)
cleaned_count = await integration_service.cleanup_expired_sessions(max_duration_hours=24)
print(f"Sesiones limpiadas: {cleaned_count}")
```

### Gestión de Memoria
- **Sesiones Activas**: Limitadas por configuración
- **Métricas**: Almacenadas en memoria con límites
- **Limpieza**: Automática y manual disponible

## 🔧 Configuración

### Configuración de Sesión
```python
session_config = {
    "cache_strategy": "smart",  # smart, aggressive, conservative
    "file_patterns": ["*.py", "*.js", "*.ts"],
    "max_analysis_time": 300,  # segundos
    "enable_predictive_cache": True,
    "metrics_collection": True
}
```

### Configuración del Sistema
- **Duración Máxima de Sesión**: 24 horas (configurable)
- **Límite de Sesiones Activas**: Sin límite (configurable)
- **Intervalo de Limpieza**: Manual (configurable)

## 🧪 Testing

### Tests de Integración
```bash
# Ejecutar tests de integración
pytest src/codeant_agent/tests/integration/test_incremental_integration.py -v

# Tests específicos
pytest src/codeant_agent/tests/integration/test_incremental_integration.py::TestIncrementalIntegrationService::test_start_analysis_session_success -v
```

### Tests de Rendimiento
```bash
# Tests de rendimiento
pytest src/codeant_agent/tests/performance/test_incremental_performance.py -v
```

## 📈 Rendimiento

### Optimizaciones Implementadas
- **Gestión de Memoria**: Limpieza automática de sesiones expiradas
- **Caché Inteligente**: Reutilización de resultados de análisis
- **Análisis Incremental**: Solo procesa archivos modificados
- **Predicción**: Anticipa necesidades de caché

### Métricas de Rendimiento
- **Tiempo de Análisis**: Reducción del 60-80% con caché
- **Uso de Memoria**: Optimizado con límites configurables
- **Escalabilidad**: Soporte para múltiples sesiones concurrentes

## 🔒 Seguridad

### Validaciones
- **Rutas de Proyecto**: Validación de existencia y permisos
- **IDs de Sesión**: Validación de formato y existencia
- **Configuración**: Validación de parámetros de entrada

### Manejo de Errores
- **Errores de Validación**: Retornados con códigos 400
- **Errores del Dominio**: Retornados con códigos 500
- **Errores Inesperados**: Capturados y loggeados

## 🚀 Próximos Pasos

### Mejoras Planificadas
1. **Persistencia**: Almacenamiento de sesiones en base de datos
2. **Distribución**: Soporte para múltiples instancias
3. **ML Avanzado**: Mejores predicciones de caché
4. **UI**: Interfaz web para gestión de sesiones

### Integraciones Futuras
- **CI/CD**: Integración con pipelines de integración continua
- **IDEs**: Plugins para editores de código
- **APIs**: Endpoints REST para integración externa

## 📚 Referencias

- [Documentación del Sistema Incremental](../incremental/README.md)
- [Casos de Uso de Análisis Incremental](../../application/use_cases/incremental_use_cases.py)
- [Entidades del Dominio](../../domain/entities/incremental.py)
- [Tests de Integración](../../../tests/integration/test_incremental_integration.py)
