import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch }) => {
    try {
        // Intentamos cargar los proyectos desde la API
        const response = await fetch('/api/projects');

        if (response.ok) {
            const projects = await response.json();
            return {
                projects,
                status: 'success'
            };
        } else {
            // Si el backend no está disponible o responde con error, devolvemos datos simulados
            console.warn(`Error cargando proyectos: ${response.status}`);
            return {
                projects: [],
                status: 'error',
                error: `Error del servidor: ${response.status}`
            };
        }
    } catch (e) {
        console.error('Error en la carga de proyectos:', e);
        // Devolvemos datos simulados
        return {
            projects: [],
            status: 'error',
            error: e.message || 'Error desconocido'
        };
    }
};
