import type { RequestHandler } from '@sveltejs/kit';
import { json } from '@sveltejs/kit';

// Esta función intenta descubrir los endpoints disponibles en el backend
export const GET = (async ({ fetch }) => {
    try {
        const apiUrl = process.env.PUBLIC_API_URL || 'http://backend:8000';

        // Lista de posibles rutas a probar
        const routesToCheck = [
            '/',
            '/api',
            '/api/v1',
            '/docs',
            '/redoc',
            '/openapi.json',
            '/health',
            '/info'
        ];

        // Probamos cada ruta
        const results = await Promise.allSettled(
            routesToCheck.map(route =>
                fetch(`${apiUrl}${route}`)
                    .then(async response => ({
                        route,
                        status: response.status,
                        ok: response.ok,
                        type: response.headers.get('content-type')
                    }))
                    .catch(error => ({
                        route,
                        error: error.message,
                        status: 'error',
                        ok: false
                    }))
            )
        );

        const availableEndpoints = results
            .filter(result => result.status === 'fulfilled' && result.value.ok)
            .map(result => result.value);

        const unavailableEndpoints = results
            .filter(result => result.status === 'fulfilled' && !result.value.ok)
            .map(result => result.value);

        return json({
            available: availableEndpoints,
            unavailable: unavailableEndpoints,
            message: "Comprobación de endpoints completada"
        });
    } catch (error) {
        console.error('Error al comprobar endpoints:', error);
        return json({
            error: error.message,
            message: "Error al comprobar endpoints"
        }, { status: 500 });
    }
}) satisfies RequestHandler;