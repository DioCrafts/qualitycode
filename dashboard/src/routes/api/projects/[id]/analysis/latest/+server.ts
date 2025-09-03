import type { RequestHandler } from '@sveltejs/kit';
import { json } from '@sveltejs/kit';

// Endpoint para obtener el último análisis de un proyecto
export const GET = (async ({ params, fetch }) => {
    try {
        const projectId = params.id;
        console.log(`Obteniendo último análisis para el proyecto ${projectId}`);

        // Intentar obtener del backend real
        const apiUrl = process.env.PUBLIC_API_URL || 'http://backend:8000';
        const response = await fetch(`${apiUrl}/api/projects/${projectId}/analysis/latest`, {
            headers: {
                'Accept': 'application/json'
            }
        });

        if (response.ok) {
            const analysis = await response.json();
            return json(analysis);
        } else if (response.status === 404) {
            // No hay análisis para este proyecto
            return new Response(null, { status: 404 });
        } else {
            console.error('Error obteniendo análisis:', response.status);
            return json(
                { error: 'Error obteniendo análisis' },
                { status: response.status }
            );
        }
    } catch (error) {
        console.error('Error obteniendo último análisis:', error);
        return json(
            { error: 'Error interno del servidor' },
            { status: 500 }
        );
    }
}) satisfies RequestHandler;
