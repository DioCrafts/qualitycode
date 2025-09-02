# 🐜 CodeAnt Agent

> **Agente inteligente de análisis de código con arquitectura hexagonal**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Hexagonal-orange.svg)](https://en.wikipedia.org/wiki/Hexagonal_architecture_(software))
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 ¿Qué hace este proyecto?

CodeAnt Agent es un sistema inteligente de análisis de código que utiliza **arquitectura hexagonal** para proporcionar análisis estático, detección de problemas de calidad, y sugerencias de mejora para múltiples lenguajes de programación.

**En palabras simples**: Es como tener un experto en código que revisa tu proyecto 24/7, encuentra problemas y te sugiere cómo mejorarlo.

## 🏗️ Arquitectura

```
┌─────────────────────────────────────┐
│           Presentation Layer        │
│     (REST API, GraphQL, CLI)        │
├─────────────────────────────────────┤
│          Application Layer          │
│    (Use Cases, Command Handlers)    │
├─────────────────────────────────────┤
│            Domain Layer             │
│  (Entities, Value Objects, Rules)   │
├─────────────────────────────────────┤
│         Infrastructure Layer        │
│ (Database, File System, External)   │
└─────────────────────────────────────┘
```

### 🎨 Principios de Diseño

- **🏛️ Arquitectura Hexagonal**: Separación clara entre dominio y infraestructura
- **🧪 Test-First Development**: TDD con cobertura >95%
- **🔒 SOLID Principles**: Código mantenible y extensible
- **📊 Observabilidad**: Logging estructurado, métricas y health checks
- **🐳 Container-First**: Docker y Docker Compose para desarrollo y producción

## 🚀 Cómo ejecutarlo

### Prerrequisitos

- Python 3.9+
- Docker y Docker Compose (opcional)
- Git

### Setup Rápido

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/codeant/codeant-agent.git
   cd codeant-agent
   ```

2. **Ejecutar setup automático**
   ```bash
   ./scripts/setup.sh
   ```

3. **Activar entorno virtual**
   ```bash
   source venv/bin/activate
   ```

4. **Ejecutar la aplicación**
   ```bash
   python -m codeant_agent.main
   ```

### 🐳 Con Docker

```bash
# Levantar todos los servicios
docker-compose up -d

# Ver logs de la aplicación
docker-compose logs -f codeant-agent

# Acceder a la aplicación
curl http://localhost:8080/health
```

## 🧪 Cómo probarlo

### Tests Unitarios
```bash
# Todos los tests
python -m pytest

# Tests unitarios específicos
python -m pytest tests/unit/ -v

# Con cobertura
python -m pytest --cov=src --cov-report=html
```

### Tests de Integración
```bash
# Tests de integración
python -m pytest tests/integration/ -v

# Tests E2E
python -m pytest tests/e2e/ -v
```

### Quality Checks
```bash
# Linting
python -m flake8 src/
python -m black --check src/

# Type checking
python -m mypy src/

# Security audit
python -m safety check
```

## 📊 Endpoints Disponibles

| Endpoint | Descripción | Método |
|----------|-------------|---------|
| `/` | Información básica de la aplicación | GET |
| `/health` | Health check completo | GET |
| `/health/live` | Liveness check (Kubernetes) | GET |
| `/health/ready` | Readiness check (Kubernetes) | GET |
| `/metrics` | Métricas Prometheus | GET |
| `/info` | Información detallada | GET |
| `/config` | Configuración actual (dev only) | GET |
| `/docs` | Documentación API (Swagger) | GET |

## ⚙️ Configuración

### Variables de Entorno

```bash
# Entorno
ENVIRONMENT=development  # development, testing, staging, production
DEBUG=true

# Base de datos
DATABASE__URL=postgresql://user:pass@host:5432/db
DATABASE__MAX_CONNECTIONS=20

# Redis
REDIS__URL=redis://localhost:6379

# Servidor
SERVER__HOST=0.0.0.0
SERVER__PORT=8080

# Logging
LOGGING__LEVEL=DEBUG
LOGGING__FORMAT=text

# IA
AI__ENABLE_AI_ANALYSIS=true
AI__MAX_TOKENS=4096
```

### Archivos de Configuración

- `pyproject.toml` - Configuración principal del proyecto
- `.env` - Variables de entorno (creado automáticamente)
- `docker-compose.yml` - Servicios de infraestructura

## 🏗️ Estructura del Proyecto

```
codeant-agent/
├── src/codeant_agent/           # Código fuente principal
│   ├── config/                  # Configuración
│   ├── domain/                  # Lógica de dominio
│   ├── application/             # Casos de uso
│   ├── infrastructure/          # Implementaciones externas
│   ├── presentation/            # Controllers y API
│   ├── parsers/                 # Parsers de código
│   ├── analysis/                # Motor de análisis
│   └── utils/                   # Utilidades comunes
├── tests/                       # Tests organizados por capas
├── scripts/                     # Scripts de utilidad
├── docs/                        # Documentación técnica
├── monitoring/                  # Configuración de monitoreo
└── docker-compose.yml           # Servicios de infraestructura
```

## 🔧 Desarrollo

### Comandos Útiles

```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Formatear código
python -m black src/
python -m isort src/

# Linting
python -m flake8 src/

# Type checking
python -m mypy src/

# Pre-commit hooks
pre-commit run --all-files
```

### Workflow de Desarrollo

1. **Fork y clone** el repositorio
2. **Crear rama** para tu feature: `git checkout -b feature/nueva-funcionalidad`
3. **Implementar** siguiendo TDD
4. **Ejecutar tests**: `python -m pytest`
5. **Formatear código**: `python -m black src/`
6. **Commit**: `git commit -m "feat: nueva funcionalidad"`
7. **Push y Pull Request**

## 📈 Monitoreo y Observabilidad

### Métricas Prometheus
- **HTTP Requests**: Total, duración, códigos de estado
- **Base de Datos**: Conexiones activas, duración de queries
- **Análisis de Código**: Total, duración, por lenguaje
- **Sistema**: Memoria, CPU, cache hits/misses

### Health Checks
- **Basic**: Verificación básica del sistema
- **Memory**: Uso de memoria del sistema
- **Database**: Conexión a PostgreSQL
- **Redis**: Conexión a Redis

### Logging Estructurado
- Formato JSON para producción
- Formato legible para desarrollo
- Contexto automático (request ID, user ID)
- Niveles configurables

## 🚀 Despliegue

### Producción

```bash
# Build de la imagen
docker build -t codeant-agent:latest .

# Ejecutar
docker run -d \
  -p 8080:8080 \
  -e ENVIRONMENT=production \
  -e DATABASE__URL=postgresql://... \
  codeant-agent:latest
```

### Kubernetes

```bash
# Aplicar manifiestos
kubectl apply -f k8s/

# Verificar deployment
kubectl get pods -l app=codeant-agent
```

## 🤝 Contribuir

### Guías de Contribución

1. **Sigue las convenciones** del proyecto
2. **Escribe tests** para nueva funcionalidad
3. **Mantén cobertura** >95%
4. **Documenta** cambios importantes
5. **Usa conventional commits**

### Convenciones de Código

- **Naming**: Descriptivo y autoexplicativo
- **Funciones**: Máximo 20 líneas
- **Comentarios**: Solo cuando el "por qué" no sea obvio
- **Tests**: AAA pattern (Arrange, Act, Assert)

## 📚 Documentación

- [📖 Documentación Técnica](docs/)
- [🏗️ Arquitectura](docs/architecture.md)
- [🔌 API Reference](docs/api.md)
- [🚀 Deployment](docs/deployment.md)

## 🐛 Reportar Issues

1. **Buscar** si ya existe un issue similar
2. **Crear** issue con template completo
3. **Incluir** logs, pasos de reproducción
4. **Etiquetar** correctamente

## 📄 Licencia

Este proyecto está bajo la licencia [MIT](LICENSE).

## 🙏 Agradecimientos

- **FastAPI** por el framework web increíble
- **Pydantic** por la validación de datos
- **Pytest** por el framework de testing
- **Docker** por la containerización
- **Prometheus** por las métricas

## 📞 Contacto

- **GitHub Issues**: [Reportar bugs/features](https://github.com/codeant/codeant-agent/issues)
- **Discussions**: [Discusiones generales](https://github.com/codeant/codeant-agent/discussions)
- **Email**: team@codeant.dev

---

**⭐ Si este proyecto te es útil, ¡dale una estrella!**

**🐜 CodeAnt Agent - Haciendo el código mejor, un commit a la vez.**