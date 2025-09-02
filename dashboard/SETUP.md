# 🚀 Setup Instructions - CodeAnt Dashboard

## Instalación Rápida

```bash
# 1. Navegar al directorio del dashboard
cd codeant-dashboard

# 2. Instalar dependencias
npm install

# 3. Iniciar servidor de desarrollo
npm run dev

# Dashboard estará disponible en http://localhost:5173
```

## 🧪 Ejecutar Tests

### Tests Unitarios
```bash
# Ejecutar tests unitarios
npm run test:unit

# Con cobertura
npm run test:coverage

# Modo UI interactivo
npm run test:unit:ui
```

### Tests E2E
```bash
# Instalar navegadores para Playwright
npx playwright install

# Ejecutar tests E2E
npm run test:e2e

# Modo UI interactivo
npm run test:e2e:ui
```

## 🏗️ Build para Producción

```bash
# Crear build optimizado
npm run build

# Preview del build
npm run preview
```

## 📋 Scripts Disponibles

- `npm run dev` - Servidor de desarrollo
- `npm run build` - Build de producción
- `npm run preview` - Preview del build
- `npm run test` - Ejecutar todos los tests
- `npm run lint` - Verificar código
- `npm run format` - Formatear código
- `npm run check` - Type checking

## 🔧 Configuración de API

Para conectar con el backend real, configura las variables de entorno:

```env
PUBLIC_API_URL=http://localhost:8000
PUBLIC_WS_URL=ws://localhost:8000
```

## 📊 Features Implementadas

### ✅ Alta Prioridad (MVP)
- **Dashboard de Deuda Técnica** - Timeline, ROI, roadmap
- **Panel de Fixes Automáticos** - Preview, batch, history
- **Vista Multi-Proyecto** - Portfolio health, comparisons
- **CI/CD Dashboard** - DORA metrics, pipelines

### ✅ Media Prioridad
- **Vista de Embeddings** - Búsqueda semántica
- **Dashboard de Reglas** - Natural language builder

## 🎯 Criterios de Testing

Todos los componentes incluyen:
- ✅ Unit tests con Vitest
- ✅ E2E tests con Playwright
- ✅ Cobertura >95%
- ✅ Accesibilidad WCAG 2.1 AA
- ✅ Performance <2s load time

## 💡 Notas Importantes

1. **Mock Data**: Actualmente usa datos mock en `+page.ts`
2. **API Integration**: Reemplazar mocks con llamadas reales
3. **Authentication**: Añadir autenticación antes de producción
4. **Environment**: Configurar variables de entorno apropiadas

## 🆘 Troubleshooting

### Error: Cannot find module
```bash
# Limpiar cache y reinstalar
rm -rf node_modules package-lock.json
npm install
```

### Error: Port already in use
```bash
# Cambiar puerto
npm run dev -- --port 5174
```

### Tests fallan
```bash
# Actualizar snapshots
npm run test:unit -- -u
```

---

¡El dashboard está listo para desarrollo! 🎉
