# Sistema de Análisis Incremental y Caching

## 🎯 Descripción General

El Sistema de Análisis Incremental es un componente crítico de CodeAnt Agent que optimiza el rendimiento del análisis de código mediante:

- **Detección Granular de Cambios**: Identifica cambios a nivel de archivo, función, declaración y expresión
- **Cache Multi-nivel Inteligente**: Sistema de cache L1 (memoria), L2 (Redis), L3 (disco) con promoción automática
- **Análisis Incremental**: Reutiliza resultados previos y solo analiza lo que cambió
- **Cache Predictivo con ML**: Predice qué análisis se necesitarán y los precarga
- **Tracking de Dependencias**: Identifica y analiza el impacto de cambios en dependencias

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Incremental Analysis System               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌──────────────────┐              │
│  │ Change Detector │───▶│ Dependency       │              │
│  │   (Granular)    │    │   Tracker        │              │
│  └─────────────────┘    └──────────────────┘              │
│           │                      │                          │
│           ▼                      ▼                          │
│  ┌─────────────────┐    ┌──────────────────┐              │
│  │  Incremental    │◀──▶│  Smart Cache     │              │
│  │    Analyzer     │    │   Manager        │              │
│  └─────────────────┘    └──────────────────┘              │
│           │                      │                          │
│           ▼                      ▼                          │
│  ┌─────────────────┐    ┌──────────────────┐              │
│  │ Delta Processor │    │ Predictive Cache │              │
│  └─────────────────┘    └──────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Características Principales

### 1. Detección de Cambios Granular

- **Niveles de Granularidad**:
  - `FILE`: Cambios a nivel de archivo completo
  - `FUNCTION`: Cambios en funciones/métodos
  - `STATEMENT`: Cambios en declaraciones
  - `EXPRESSION`: Cambios en expresiones
  - `TOKEN`: Cambios a nivel de token

- **Tipos de Cambios Detectados**:
  - Adiciones, eliminaciones y modificaciones
  - Cambios semánticos (imports, signatures)
  - Refactorizaciones mayores

### 2. Cache Multi-nivel

- **L1 (Memoria)**: 
  - Cache LRU ultra-rápido
  - Para items pequeños y frecuentemente accedidos
  - TTL: 1 hora por defecto

- **L2 (Redis)**:
  - Cache distribuido
  - Para items medianos
  - TTL: 24 horas por defecto

- **L3 (Disco)**:
  - Almacenamiento persistente
  - Para items grandes o poco frecuentes
  - TTL: 7 días por defecto

### 3. Análisis Incremental

- **Reutilización Inteligente**: Identifica componentes sin cambios
- **Delta Processing**: Aplica solo los cambios necesarios
- **Paralelización**: Análisis concurrente de múltiples archivos

### 4. Cache Predictivo

- **Modelos de Predicción**:
  - Patrones temporales
  - Secuencias de acceso
  - Correlaciones entre archivos
  - Machine Learning (Linear Regression)

- **Cache Warming**: Precarga proactiva de análisis predichos

### 5. Tracking de Dependencias

- **Grafo de Dependencias**: NetworkX para análisis eficiente
- **Detección de Ciclos**: Identifica dependencias circulares
- **Análisis de Impacto**: Calcula el alcance de los cambios

## 📖 Uso

### Configuración Básica

```python
from codeant_agent.infrastructure.incremental import (
    create_incremental_system,
    IncrementalConfig
)

# Crear configuración
config = IncrementalConfig(
    l1_cache_size=1000,
    enable_predictive_caching=True,
    max_parallel_analyses=4
)

# Crear sistema
system = create_incremental_system(config)

# Acceder a componentes
engine = system['engine']
cache_manager = system['cache_manager']
change_detector = system['change_detector']
```

### Detectar Cambios

```python
# Detectar cambios entre commits
change_set = await change_detector.detect_changes(
    repository_path=Path("/path/to/repo"),
    from_commit="HEAD~1",
    to_commit="HEAD"
)

# Analizar impacto
impact = await dependency_tracker.analyze_dependency_impact(
    changes=change_set.file_changes[0].granular_changes
)
```

### Análisis Incremental

```python
# Realizar análisis incremental
results = await engine.analyze_incremental(
    change_set=change_set,
    project_context={'total_files': 1000}
)

# Los resultados reutilizan análisis cacheados cuando es posible
```

### Cache Predictivo

```python
# Inicializar sistema predictivo
await predictive_cache.initialize()

# Predecir accesos futuros
predictions = await predictive_cache.predict_future_accesses(
    time_horizon=timedelta(minutes=30)
)

# Calentar cache basado en predicciones
warmed = await predictive_cache.warm_cache(predictions)
```

## 🔧 Configuración Avanzada

### Ajustar por Recursos

```python
config = IncrementalConfig()
config.adjust_for_environment(
    available_memory_mb=4096,
    cpu_count=8
)
```

### Configuración de Cache

```python
cache_config = CacheConfig(
    l1_max_memory_mb=200,
    l2_enabled=True,
    compress_l3_items=True,
    promotion_threshold=3
)
```

### Configuración de Dependencias

```python
dep_config = DependencyConfig(
    analyze_third_party_deps=True,
    max_graph_size=10000,
    parallel_dependency_analysis=True
)
```

## 📊 Métricas y Monitoreo

El sistema recolecta métricas detalladas:

- **Cache Metrics**: Hit rate, tamaño, evictions
- **Change Detection Metrics**: Tiempo de detección, número de cambios
- **Analysis Performance**: Speedup incremental, tiempo ahorrado
- **Prediction Accuracy**: Precisión de predicciones, false positives

```python
# Obtener métricas
cache_metrics = await cache_manager.get_cache_metrics()
print(f"Cache hit rate: {cache_metrics.hit_rate:.2%}")

# Analizar patrones de acceso
patterns = await predictive_cache.analyze_access_patterns(
    time_window=timedelta(hours=24)
)
```

## 🧪 Testing

```python
# Test de detección de cambios
def test_granular_change_detection():
    detector = GranularChangeDetector(config)
    changes = await detector.detect_file_changes(
        file_path=Path("test.py"),
        old_content="def foo(): pass",
        new_content="def foo():\n    return 42"
    )
    assert len(changes[0].granular_changes) == 1
    assert changes[0].granular_changes[0].change_type == ChangeType.FUNCTION_MODIFIED

# Test de cache multi-nivel
def test_cache_promotion():
    cache = SmartCacheManager(config)
    
    # Almacenar en L3
    await cache.store_in_l3("key", "value", ttl=3600)
    
    # Acceder varias veces para promover
    for _ in range(3):
        value = await cache.get_cached_item(CacheKey(path="key"))
    
    # Verificar promoción a L1
    assert cache.l1_cache.get("key") is not None
```

## 🔍 Troubleshooting

### Cache Miss Alto

1. Verificar configuración de TTL
2. Aumentar tamaño de cache L1
3. Habilitar cache predictivo
4. Revisar patrones de invalidación

### Análisis Lento

1. Verificar umbral incremental
2. Aumentar paralelismo
3. Optimizar detección de cambios
4. Revisar profundidad de dependencias

### Predicciones Incorrectas

1. Reentrenar modelo predictivo
2. Ajustar umbral de confianza
3. Ampliar ventana de features
4. Verificar calidad de datos históricos

## 🛠️ Optimización de Performance

### Tips de Optimización

1. **Granularidad Apropiada**: Use FUNCTION para balance entre precisión y performance
2. **Cache Warming Estratégico**: Programe warming en horas de baja actividad
3. **Invalidación Inteligente**: Use patrones específicos en lugar de invalidación masiva
4. **Monitoreo Continuo**: Revise métricas regularmente para identificar bottlenecks

### Configuración para Proyectos Grandes

```python
config = IncrementalConfig(
    l1_cache_size=5000,
    max_parallel_analyses=8,
    incremental_threshold=0.1,  # Más agresivo
    enable_predictive_caching=True,
    max_dependency_depth=2  # Limitar profundidad
)
```

## 📚 Referencias

- [Documentación de NetworkX](https://networkx.org/)
- [Cache Algorithms](https://en.wikipedia.org/wiki/Cache_replacement_policies)
- [Incremental Computation](https://en.wikipedia.org/wiki/Incremental_computing)
- [Predictive Caching Papers](https://scholar.google.com/scholar?q=predictive+caching)

