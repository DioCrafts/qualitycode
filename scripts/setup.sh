#!/bin/bash

# Script de setup inicial para CodeAnt Agent
# Este script configura el entorno de desarrollo

set -e  # Exit on any error

echo "🚀 Configurando CodeAnt Agent..."

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar que estamos en el directorio correcto
if [ ! -f "pyproject.toml" ]; then
    log_error "Este script debe ejecutarse desde el directorio raíz del proyecto"
    exit 1
fi

# Verificar Python
log_info "Verificando Python..."
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 no está instalado"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log_info "Python version: $PYTHON_VERSION"

# Verificar pip
log_info "Verificando pip..."
if ! command -v pip3 &> /dev/null; then
    log_error "pip3 no está instalado"
    exit 1
fi

# Verificar Docker
log_info "Verificando Docker..."
if ! command -v docker &> /dev/null; then
    log_warning "Docker no está instalado. Algunas funcionalidades no estarán disponibles."
    DOCKER_AVAILABLE=false
else
    DOCKER_VERSION=$(docker --version)
    log_info "Docker version: $DOCKER_VERSION"
    DOCKER_AVAILABLE=true
fi

# Verificar Docker Compose
if [ "$DOCKER_AVAILABLE" = true ]; then
    log_info "Verificando Docker Compose..."
    if ! command -v docker-compose &> /dev/null; then
        log_warning "Docker Compose no está instalado"
        DOCKER_COMPOSE_AVAILABLE=false
    else
        DOCKER_COMPOSE_VERSION=$(docker-compose --version)
        log_info "Docker Compose version: $DOCKER_COMPOSE_VERSION"
        DOCKER_COMPOSE_AVAILABLE=true
    fi
fi

# Crear entorno virtual
log_info "Creando entorno virtual..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_success "Entorno virtual creado"
else
    log_info "Entorno virtual ya existe"
fi

# Activar entorno virtual
log_info "Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
log_info "Actualizando pip..."
pip install --upgrade pip setuptools wheel

# Instalar dependencias de desarrollo
log_info "Instalando dependencias de desarrollo..."
pip install -e ".[dev]"

# Instalar pre-commit hooks
log_info "Instalando pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    log_success "Pre-commit hooks instalados"
else
    log_warning "pre-commit no está disponible"
fi

# Crear directorios necesarios
log_info "Creando directorios necesarios..."
mkdir -p logs models cache monitoring/{prometheus,grafana/{dashboards,datasources}}

# Crear archivo .env si no existe
if [ ! -f ".env" ]; then
    log_info "Creando archivo .env..."
    cat > .env << EOF
# CodeAnt Agent - Variables de entorno
ENVIRONMENT=development
DEBUG=true

# Base de datos
DATABASE__URL=postgresql://codeant:dev_password@localhost:5432/codeant_dev
DATABASE__MAX_CONNECTIONS=20
DATABASE__MIN_CONNECTIONS=5

# Redis
REDIS__URL=redis://localhost:6379
REDIS__MAX_CONNECTIONS=10

# Servidor
SERVER__HOST=0.0.0.0
SERVER__PORT=8080
SERVER__WORKERS=4
SERVER__RELOAD=true

# Logging
LOGGING__LEVEL=DEBUG
LOGGING__FORMAT=text

# IA
AI__ENABLE_AI_ANALYSIS=true
AI__MAX_TOKENS=4096

# Features
FEATURES__ENABLE_TELEMETRY=true
FEATURES__ENABLE_DEBUG_ENDPOINTS=true
FEATURES__ENABLE_HEALTH_CHECKS=true
EOF
    log_success "Archivo .env creado"
else
    log_info "Archivo .env ya existe"
fi

# Crear archivo de configuración de Prometheus
log_info "Creando configuración de Prometheus..."
cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'codeant-agent'
    static_configs:
      - targets: ['codeant-agent:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
log_success "Configuración de Prometheus creada"

# Crear configuración de Grafana
log_info "Creando configuración de Grafana..."
cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
log_success "Configuración de Grafana creada"

# Verificar que todo funciona
log_info "Verificando instalación..."
python -c "import codeant_agent; print('✅ Módulo codeant_agent importado correctamente')"

# Ejecutar tests básicos
log_info "Ejecutando tests básicos..."
if python -m pytest tests/unit/test_config.py -v; then
    log_success "Tests de configuración pasaron"
else
    log_warning "Algunos tests fallaron"
fi

# Mostrar información final
echo ""
log_success "🎉 Setup completado exitosamente!"
echo ""
echo "📋 Próximos pasos:"
echo "1. Activar el entorno virtual: source venv/bin/activate"
echo "2. Ejecutar tests: python -m pytest"
echo "3. Ejecutar la aplicación: python -m codeant_agent.main"
echo ""

if [ "$DOCKER_COMPOSE_AVAILABLE" = true ]; then
    echo "🐳 Para usar Docker Compose:"
    echo "   docker-compose up -d"
    echo "   docker-compose logs -f codeant-agent"
    echo ""
fi

echo "📚 Documentación disponible en:"
echo "   - README.md"
echo "   - docs/"
echo ""

log_info "¡Happy coding! 🚀"
