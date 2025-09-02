import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// Función para obtener proyectos
export const GET: RequestHandler = async ({ fetch }) => {
    try {
        // Intenta obtener los proyectos del backend real
        const apiUrl = process.env.PUBLIC_API_URL || 'http://localhost:8001';
        const response = await fetch(`${apiUrl}/api/v1/projects`);

        if (response.ok) {
            const projects = await response.json();
            return json(projects);
        } else {
            console.error('Error al obtener proyectos del backend:', response.status);
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
};

// Función para crear un nuevo proyecto
export const POST: RequestHandler = async ({ request, fetch }) => {
    try {
        const projectData = await request.json();

        // Intenta crear el proyecto en el backend real
        const apiUrl = process.env.PUBLIC_API_URL || 'http://localhost:8001';
        const response = await fetch(`${apiUrl}/api/v1/projects`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(projectData)
        });

        if (response.ok) {
            const createdProject = await response.json();
            return json(createdProject);
        } else {
            const errorData = await response.json().catch(() => ({ message: 'Error desconocido' }));
            return json(
                { error: errorData.message || `Error del servidor: ${response.status}` },
                { status: response.status }
            );
        }
    } catch (error) {
        console.error('Error al crear proyecto:', error);

        // En desarrollo, simula la creación exitosa
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
    }
};
