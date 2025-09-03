import type { RequestHandler } from '@sveltejs/kit';
import { json } from '@sveltejs/kit';

// Obtener un proyecto por su ID
export const GET = (async ({ params, fetch }) => {
    try {
        const projectId = params.id;
        console.log(`Obteniendo proyecto ${projectId}`);

        // Intentar obtener del backend real
        const apiUrl = process.env.PUBLIC_API_URL || 'http://backend:8000';
        const response = await fetch(`${apiUrl}/api/projects/${projectId}`, {
            headers: {
                'Accept': 'application/json'
            }
        });

        if (response.ok) {
            const project = await response.json();
            return json(project);
        } else {
            // Si el backend devuelve error o el endpoint no existe
            console.info("Modo desarrollo: generando proyecto simulado");

            // Si el ID es "1", devolvemos el proyecto demo
            if (projectId === "1") {
                return json({
                    id: "1",
                    name: "Proyecto Demo",
                    slug: "demo-project",
                    description: "Un proyecto de ejemplo para mostrar la funcionalidad",
                    repository_url: "https://github.com/example/demo-project",
                    repository_type: "git",
                    default_branch: "main",
                    status: "ACTIVE",
                    created_at: "2025-09-01T00:00:00Z",
                    updated_at: "2025-09-02T00:00:00Z",
                    metadata: {
                        stars: 15,
                        forks: 5,
                        language_stats: {
                            Python: 60,
                            JavaScript: 30,
                            HTML: 10,
                        },
                    },
                    settings: {
                        analysis_config: {},
                        ignore_patterns: [],
                        include_patterns: [],
                        max_file_size_mb: 10,
                        enable_incremental_analysis: true
                    }
                });
            } else {
                return json(
                    { error: `Proyecto no encontrado: ${projectId}` },
                    { status: 404 }
                );
            }
        }
    } catch (error) {
        console.error('Error obteniendo proyecto:', error);
        return json(
            { error: 'Error interno del servidor' },
            { status: 500 }
        );
    }
}) satisfies RequestHandler;
