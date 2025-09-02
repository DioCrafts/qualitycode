import type { RequestHandler } from '@sveltejs/kit';
import { json } from '@sveltejs/kit';

// Función para obtener proyectos
export const GET = (async () => {
    try {
        console.log("Obteniendo proyectos - modo desarrollo activo");

        // En modo desarrollo usamos datos de ejemplo directamente
        // Nota: Backend no tiene la ruta configurada aún
        return json([
            {
                id: '1',
                name: 'Proyecto Demo',
                description: 'Un proyecto de ejemplo para mostrar la funcionalidad',
                slug: 'demo-project',
                status: 'ACTIVE',
                metadata: {
                    stars: 15,
                    forks: 5,
                    language_stats: {
                        'Python': 60,
                        'JavaScript': 30,
                        'HTML': 10
                    }
                }
            },
            {
                id: '2',
                name: 'API REST',
                description: 'Servicio backend con arquitectura hexagonal',
                slug: 'api-rest',
                status: 'ACTIVE',
                metadata: {
                    stars: 42,
                    forks: 12,
                    language_stats: {
                        'Python': 80,
                        'YAML': 15,
                        'Dockerfile': 5
                    }
                }
            }
        ]);
    } catch (error) {
        console.error('Error en API de proyectos:', error);
        // Retorna proyectos de ejemplo para desarrollo
        return json([
            {
                id: '1',
                name: 'Proyecto Demo',
                description: 'Un proyecto de ejemplo para mostrar la funcionalidad',
                slug: 'demo-project',
                status: 'ACTIVE',
                metadata: {
                    stars: 15,
                    forks: 5,
                    language_stats: {
                        'Python': 60,
                        'JavaScript': 30,
                        'HTML': 10
                    }
                }
            },
            {
                id: '2',
                name: 'API REST',
                description: 'Servicio backend con arquitectura hexagonal',
                slug: 'api-rest',
                status: 'ACTIVE',
                metadata: {
                    stars: 42,
                    forks: 12,
                    language_stats: {
                        'Python': 80,
                        'YAML': 15,
                        'Dockerfile': 5
                    }
                }
            }
        ]);
    }
}) satisfies RequestHandler;

// Función para crear un nuevo proyecto
export const POST = (async ({ request }) => {
    try {
        const projectData = await request.json();
        console.log("Creando proyecto - modo desarrollo activo:", projectData);

        // En modo desarrollo, simulamos la creación exitosa
        // Nota: Backend no tiene la ruta configurada aún
        const mockProject = {
            id: crypto.randomUUID(),
            name: projectData?.name || 'Nuevo proyecto',
            slug: projectData?.slug || 'nuevo-proyecto',
            description: projectData?.description || '',
            repository_url: projectData?.repository_url || 'https://github.com/ejemplo/repo',
            repository_type: projectData?.repository_type || 'GIT',
            default_branch: projectData?.default_branch || 'main',
            status: 'ACTIVE',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            metadata: {
                stars: 0,
                forks: 0,
                language_stats: {}
            }
        };

        return json(mockProject);
    } catch (error) {
        console.error('Error al crear proyecto:', error);
        return json({ error: 'Error al crear proyecto' }, { status: 500 });
    }
}) satisfies RequestHandler;