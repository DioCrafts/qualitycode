import type { PageLoad } from './$types';

export const load = (async ({ params, fetch }) => {
    // Nota: La carga principal de datos se realiza en el componente
    // debido a la necesidad de actualizar el estado al realizar acciones
    return {
        projectId: params.id
    };
}) satisfies PageLoad;
