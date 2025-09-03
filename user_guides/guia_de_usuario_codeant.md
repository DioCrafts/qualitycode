# Guía del Usuario - CodeAnt

## Índice
1. [Introducción a CodeAnt](#introducción-a-codeant)
2. [Primeros pasos](#primeros-pasos)
3. [Gestión de Proyectos](#gestión-de-proyectos)
   - [Crear un nuevo proyecto](#crear-un-nuevo-proyecto)
   - [Ver detalles del proyecto](#ver-detalles-del-proyecto)
4. [Análisis de Código](#análisis-de-código)
   - [Iniciar un análisis](#iniciar-un-análisis)
   - [Ver resultados del análisis](#ver-resultados-del-análisis)
5. [Solución de problemas comunes](#solución-de-problemas-comunes)

## Introducción a CodeAnt

CodeAnt es una plataforma de análisis de código que te permite evaluar la calidad del código fuente de tus proyectos de software. Con CodeAnt puedes:

- **Detectar problemas de código**: Identifica patrones problemáticos, anti-patrones y deuda técnica.
- **Mejorar la calidad**: Obtén recomendaciones para mejorar la estructura y calidad del código.
- **Realizar seguimiento**: Monitorea la evolución de la calidad del código a lo largo del tiempo.

La arquitectura de CodeAnt sigue los principios de la arquitectura hexagonal, lo que garantiza un código limpio, mantenible y extensible.

## Primeros pasos

Para comenzar a usar CodeAnt, necesitas acceder a la interfaz web a través de tu navegador. Una vez que hayas iniciado sesión, te encontrarás con el Dashboard principal que te dará acceso a todas las funcionalidades.

## Gestión de Proyectos

### Crear un nuevo proyecto

Para añadir un nuevo proyecto a CodeAnt:

1. Desde el Dashboard, navega a la sección de **Proyectos** en el menú lateral.
2. Haz clic en el botón **Nuevo Proyecto**.
3. Completa el formulario con la siguiente información:
   - **Nombre del Proyecto**: Un nombre descriptivo para tu proyecto.
   - **Slug**: Identificador único del proyecto (se generará automáticamente si lo dejas en blanco).
   - **Descripción** (opcional): Una breve descripción del proyecto.
   - **URL del Repositorio**: La URL del repositorio de código (por ejemplo, GitHub, GitLab, etc.).
   - **Tipo de Repositorio**: Selecciona el tipo de sistema de control de versiones (Git por defecto).
   - **Rama Principal**: La rama principal del repositorio (main o master por defecto).

4. Haz clic en **Crear Proyecto**.

Una vez creado, CodeAnt clonará el repositorio automáticamente y lo preparará para el análisis.

### Ver detalles del proyecto

Para ver los detalles de un proyecto:

1. En la página de Proyectos, haz clic en la tarjeta del proyecto que deseas explorar.
2. Se abrirá la página de detalles que muestra:
   - Información básica del proyecto
   - Detalles del repositorio
   - Historial de análisis (si existe)
   - Métricas generales

## Análisis de Código

### Iniciar un análisis

Para analizar el código de un proyecto:

1. Navega a la página de detalles del proyecto.
2. Haz clic en el botón **Analizar proyecto**.
3. Espera mientras CodeAnt procesa el código:
   - Clona o actualiza el repositorio
   - Analiza cada archivo
   - Detecta patrones y problemas
   - Genera métricas y recomendaciones

El tiempo de análisis dependerá del tamaño del proyecto. Durante este proceso, verás una barra de progreso indicando el estado.

### Ver resultados del análisis

Una vez completado el análisis, podrás ver:

1. **Resumen de resultados**:
   - Número total de problemas encontrados
   - Puntuación de calidad general
   - Cantidad de archivos analizados

2. Para ver el análisis detallado, haz clic en **Ver análisis completo**, donde encontrarás:
   - Lista completa de problemas detectados
   - Recomendaciones de mejora
   - Métricas detalladas por archivo y tipo de problema
   - Visualización de tendencias (si hay múltiples análisis)

## Solución de problemas comunes

### El botón "Analizar proyecto" no responde

- **Problema**: Haces clic en el botón de análisis pero parece que no ocurre nada.
- **Solución**: Verifica tu conexión a internet. Si persiste, actualiza la página e intenta nuevamente.

### Error en la creación del proyecto

- **Problema**: Aparece un error al intentar crear un nuevo proyecto.
- **Solución**: Verifica que la URL del repositorio sea correcta y accesible públicamente. Para repositorios privados, asegúrate de haber configurado las credenciales correctamente.

### El análisis falla o se queda atascado

- **Problema**: El análisis comienza pero nunca termina o termina con errores.
- **Solución**: 
  1. Verifica que el repositorio sea accesible
  2. Comprueba que el repositorio no sea demasiado grande
  3. Asegúrate de que la rama principal especificada existe
  4. Contacta con soporte si el problema persiste

---

Para más información o asistencia adicional, contacta con el equipo de soporte a través del correo electrónico support@codeant.example.com o abre un ticket de soporte desde el Dashboard.

© 2025 CodeAnt - Análisis de Código Inteligente
