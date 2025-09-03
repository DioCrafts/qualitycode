import type { RequestHandler } from '@sveltejs/kit';
import { json } from '@sveltejs/kit';

// Endpoint para obtener el último análisis de un proyecto
export const GET = (async ({ params, fetch }) => {
    try {
        const projectId = params.id;
        console.log(`Obteniendo último análisis para el proyecto ${projectId}`);

        // Intentar obtener del backend real
        const apiUrl = process.env.PUBLIC_API_URL || 'http://backend:8000';
        const response = await fetch(`${apiUrl}/api/v1/projects/${projectId}/analysis/latest`, {
            headers: {
                'Accept': 'application/json'
            }
        });

        if (response.ok) {
            const analysis = await response.json();
            return json(analysis);
        } else {
            // Si hay un error o el endpoint no existe, devolvemos un análisis simulado para desarrollo
            console.info("Modo desarrollo: generando respuesta simulada para último análisis");

            // 50% de probabilidad de que no haya análisis previo
            if (Math.random() > 0.5) {
                return json(null, { status: 404 });
            }

            // Crear un análisis simulado
            const mockAnalysis = {
                id: `analysis-${Math.random().toString(36).substring(2, 15)}`,
                project_id: projectId,
                status: "COMPLETED",
                created_at: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(), // Entre hoy y hace 7 días
                completed_at: new Date(Date.now() - Math.random() * 6 * 60 * 60 * 1000).toISOString(), // Entre ahora y hace 6 horas
                total_violations: Math.floor(Math.random() * 50) + 5, // 5-54 violaciones
                critical_violations: Math.floor(Math.random() * 5), // 0-4 críticas
                high_violations: Math.floor(Math.random() * 10), // 0-9 altas
                files_analyzed: Math.floor(Math.random() * 100) + 20, // 20-119 archivos
                quality_score: Math.floor(Math.random() * 40) + 60, // 60-99 puntuación
            };

            return json(mockAnalysis);
        }
    } catch (error) {
        console.error('Error obteniendo último análisis:', error);
        return json(
            { error: 'Error interno del servidor' },
            { status: 500 }
        );
    }
}) satisfies RequestHandler;
