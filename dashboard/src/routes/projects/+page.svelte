<script lang="ts">
    import { goto } from "$app/navigation";
    import { onMount } from "svelte";

    // Datos de la página
    // Convertimos a const ya que no lo usamos directamente como let
    export const data = {};

    let projects = [];
    let loading = true;
    let error = null;
    let showNewProjectModal = false;

    // Formulario para nuevo proyecto
    let newProject = {
        name: "",
        slug: "",
        description: "",
        repository_url: "",
        repository_type: "git", // Cambiado a minúsculas para coincidir con lo que espera el backend
        default_branch: "main",
    };

    onMount(async () => {
        try {
            const response = await fetch("/api/projects");
            if (response.ok) {
                projects = await response.json();
            } else {
                error = `Error: ${response.status}`;
                // Para desarrollo, usamos proyectos de ejemplo
                projects = [
                    {
                        id: "1",
                        name: "Proyecto Demo",
                        description:
                            "Un proyecto de ejemplo para mostrar la funcionalidad",
                        slug: "demo-project",
                        status: "ACTIVE",
                        metadata: {
                            stars: 15,
                            forks: 5,
                            language_stats: {
                                Python: 60,
                                JavaScript: 30,
                                HTML: 10,
                            },
                        },
                    },
                ];
            }
        } catch (e) {
            console.error("Error cargando proyectos:", e);
            error = e.message;
            // Para desarrollo, usamos proyectos de ejemplo
            projects = [
                {
                    id: "1",
                    name: "Proyecto Demo",
                    description:
                        "Un proyecto de ejemplo para mostrar la funcionalidad",
                    slug: "demo-project",
                    status: "ACTIVE",
                    metadata: {
                        stars: 15,
                        forks: 5,
                        language_stats: {
                            Python: 60,
                            JavaScript: 30,
                            HTML: 10,
                        },
                    },
                },
            ];
        } finally {
            loading = false;
        }
    });

    async function createProject() {
        try {
            loading = true;
            // Generar slug a partir del nombre si está vacío
            if (!newProject.slug) {
                newProject.slug = newProject.name
                    .toLowerCase()
                    .replace(/[^\w\s-]/g, "")
                    .replace(/\s+/g, "-");
            }

            const response = await fetch("/api/projects", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(newProject),
            });

            if (response.ok) {
                const createdProject = await response.json();
                projects = [...projects, createdProject];
                showNewProjectModal = false;
                newProject = {
                    name: "",
                    slug: "",
                    description: "",
                    repository_url: "",
                    repository_type: "git", // Cambiado a minúsculas para coincidir con lo que espera el backend
                    default_branch: "main",
                };
            } else {
                error = `Error creando proyecto: ${response.status}`;
            }
        } catch (e) {
            console.error("Error creando proyecto:", e);
            error = e.message;
        } finally {
            loading = false;
        }
    }

    function openProjectDetails(projectId) {
        console.log(`Navegando a los detalles del proyecto ${projectId}`);
        goto(`/projects/${projectId}`);
    }
</script>

<div class="projects-container">
    <div class="projects-header">
        <h1>Proyectos</h1>
        <button
            class="btn-primary"
            on:click={() => (showNewProjectModal = true)}
        >
            Nuevo Proyecto
        </button>
    </div>

    {#if loading}
        <div class="loading-spinner">Cargando proyectos...</div>
    {:else if error}
        <div class="error-message">
            <p>Error al cargar proyectos: {error}</p>
            <button
                class="btn-secondary"
                on:click={() => window.location.reload()}
            >
                Reintentar
            </button>
        </div>
    {:else if projects.length === 0}
        <div class="empty-state">
            <h2>No hay proyectos</h2>
            <p>
                Para empezar, crea tu primer proyecto haciendo clic en "Nuevo
                Proyecto".
            </p>
        </div>
    {:else}
        <div class="projects-grid">
            {#each projects as project}
                <div
                    class="project-card"
                    on:click={() => openProjectDetails(project.id)}
                    on:keydown={(e) => e.key === 'Enter' && openProjectDetails(project.id)}
                    tabindex="0"
                    role="button"
                    aria-label="Ver detalles de {project.name}"
                >
                    <h2>{project.name}</h2>
                    <p>{project.description || "Sin descripción"}</p>
                    <div class="project-meta">
                        <span class="status {project.status.toLowerCase()}">
                            {project.status}
                        </span>
                        {#if project.metadata?.language_stats}
                            <div class="language-bar">
                                {#each Object.entries(project.metadata.language_stats) as [lang, percentage]}
                                    <div
                                        class="language-segment {lang.toLowerCase()}"
                                        style="width: {percentage}%"
                                        title="{lang}: {percentage}%"
                                    ></div>
                                {/each}
                            </div>
                        {/if}
                    </div>
                </div>
            {/each}
        </div>
    {/if}

    {#if showNewProjectModal}
        <div class="modal-backdrop">
            <div class="modal">
                <div class="modal-header">
                    <h2>Nuevo Proyecto</h2>
                    <button
                        class="close-btn"
                        on:click={() => (showNewProjectModal = false)}
                    >
                        ✕
                    </button>
                </div>
                <div class="modal-body">
                    <form on:submit|preventDefault={createProject}>
                        <div class="form-group">
                            <label for="project-name"
                                >Nombre del Proyecto*</label
                            >
                            <input
                                id="project-name"
                                type="text"
                                bind:value={newProject.name}
                                required
                                placeholder="Mi Proyecto"
                            />
                        </div>
                        <div class="form-group">
                            <label for="project-slug">Slug</label>
                            <input
                                id="project-slug"
                                type="text"
                                bind:value={newProject.slug}
                                placeholder="mi-proyecto"
                            />
                            <small
                                >Generado automáticamente si se deja en blanco</small
                            >
                        </div>
                        <div class="form-group">
                            <label for="project-description">Descripción</label>
                            <textarea
                                id="project-description"
                                bind:value={newProject.description}
                                placeholder="Descripción del proyecto"
                            ></textarea>
                        </div>
                        <div class="form-group">
                            <label for="repository-url"
                                >URL del Repositorio*</label
                            >
                            <input
                                id="repository-url"
                                type="url"
                                bind:value={newProject.repository_url}
                                required
                                placeholder="https://github.com/usuario/repositorio"
                            />
                        </div>
                        <div class="form-group">
                            <label for="repository-type"
                                >Tipo de Repositorio</label
                            >
                            <select
                                id="repository-type"
                                bind:value={newProject.repository_type}
                            >
                                <option value="git">Git</option>
                                <option value="subversion">SVN</option>
                                <option value="mercurial">Mercurial</option>
                                <option value="perforce">Perforce</option>
                                <option value="unknown">Otro</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="default-branch">Rama Principal</label>
                            <input
                                id="default-branch"
                                type="text"
                                bind:value={newProject.default_branch}
                                placeholder="main"
                            />
                        </div>
                        <div class="form-actions">
                            <button
                                type="button"
                                class="btn-secondary"
                                on:click={() => (showNewProjectModal = false)}
                            >
                                Cancelar
                            </button>
                            <button
                                type="submit"
                                class="btn-primary"
                                disabled={loading}
                            >
                                {loading ? "Creando..." : "Crear Proyecto"}
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    .projects-container {
        padding: 1.5rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    .projects-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }

    .projects-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1.5rem;
    }

    .project-card {
        background-color: var(--color-bg-primary, #ffffff);
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 8px;
        padding: 1.5rem;
        transition:
            transform 0.2s,
            box-shadow 0.2s;
        cursor: pointer;
    }

    .project-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .project-card h2 {
        margin-top: 0;
        margin-bottom: 0.5rem;
    }

    .project-meta {
        margin-top: 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .status {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .status.active {
        background-color: #e6f7e6;
        color: #2e7d32;
    }

    .status.inactive {
        background-color: #f7e6e6;
        color: #c62828;
    }

    .language-bar {
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        display: flex;
        margin-top: 0.5rem;
        background-color: #f5f5f5;
    }

    .language-segment {
        height: 100%;
    }

    .language-segment.python {
        background-color: #3572a5;
    }

    .language-segment.javascript {
        background-color: #f1e05a;
    }

    .language-segment.html {
        background-color: #e34c26;
    }

    .language-segment.css {
        background-color: #563d7c;
    }

    .language-segment.java {
        background-color: #b07219;
    }

    .language-segment.typescript {
        background-color: #2b7489;
    }

    .empty-state {
        text-align: center;
        padding: 3rem;
        background-color: var(--color-bg-secondary, #f5f5f5);
        border-radius: 8px;
    }

    .modal-backdrop {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    }

    .modal {
        background-color: white;
        border-radius: 8px;
        width: 90%;
        max-width: 600px;
        max-height: 90vh;
        overflow-y: auto;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .modal-header {
        padding: 1rem 1.5rem;
        border-bottom: 1px solid var(--color-border, #e5e5e5);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .modal-header h2 {
        margin: 0;
    }

    .close-btn {
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        padding: 0;
        color: #666;
    }

    .modal-body {
        padding: 1.5rem;
    }

    .form-group {
        margin-bottom: 1.5rem;
    }

    .form-group label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }

    .form-group input,
    .form-group textarea,
    .form-group select {
        width: 100%;
        padding: 0.75rem;
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 4px;
        font-family: inherit;
        font-size: 1rem;
    }

    .form-group textarea {
        min-height: 100px;
        resize: vertical;
    }

    .form-group small {
        display: block;
        margin-top: 0.25rem;
        color: #666;
    }

    .form-actions {
        display: flex;
        justify-content: flex-end;
        gap: 1rem;
    }

    .btn-primary {
        background-color: #4a6cf7;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s;
    }

    .btn-primary:hover {
        background-color: #3955d8;
    }

    .btn-primary:disabled {
        background-color: #a4b0e6;
        cursor: not-allowed;
    }

    .btn-secondary {
        background-color: white;
        color: #4a6cf7;
        border: 1px solid #4a6cf7;
        border-radius: 4px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s;
    }

    .btn-secondary:hover {
        background-color: #f5f7ff;
    }

    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }

    .error-message {
        padding: 1.5rem;
        background-color: #ffebee;
        border: 1px solid #ffcdd2;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* Dark mode */
    :global(.dark) .project-card {
        background-color: var(--color-bg-primary-dark, #1e1e1e);
        border-color: var(--color-border-dark, #333);
    }

    :global(.dark) .empty-state {
        background-color: var(--color-bg-secondary-dark, #2a2a2a);
    }

    :global(.dark) .modal {
        background-color: var(--color-bg-primary-dark, #1e1e1e);
    }

    :global(.dark) .modal-header {
        border-color: var(--color-border-dark, #333);
    }

    :global(.dark) .form-group input,
    :global(.dark) .form-group textarea,
    :global(.dark) .form-group select {
        background-color: var(--color-bg-secondary-dark, #2a2a2a);
        border-color: var(--color-border-dark, #333);
        color: white;
    }

    :global(.dark) .btn-secondary {
        background-color: transparent;
        color: #6d8eff;
        border-color: #6d8eff;
    }

    :global(.dark) .btn-secondary:hover {
        background-color: rgba(109, 142, 255, 0.1);
    }

    :global(.dark) .error-message {
        background-color: rgba(255, 82, 82, 0.1);
        border-color: rgba(255, 82, 82, 0.3);
    }
</style>
