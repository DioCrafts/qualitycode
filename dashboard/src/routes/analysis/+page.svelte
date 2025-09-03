<script lang="ts">
    import { onMount } from "svelte";

    let analyses = [];
    let loading = true;
    let error = null;

    onMount(async () => {
        try {
            // Por ahora mostramos un mensaje, ya que no hay endpoint de listado de análisis
            loading = false;
        } catch (e) {
            console.error("Error:", e);
            error = e.message;
            loading = false;
        }
    });
</script>

<div class="container">
    <h1>Análisis de Código</h1>

    {#if loading}
        <div class="loading">Cargando análisis...</div>
    {:else if error}
        <div class="error">Error: {error}</div>
    {:else}
        <div class="content">
            <p>
                Esta sección mostrará un resumen de todos los análisis
                realizados.
            </p>

            <div class="info-card">
                <h3>Próximamente:</h3>
                <ul>
                    <li>Historial de análisis por proyecto</li>
                    <li>Métricas agregadas de todos los proyectos</li>
                    <li>Tendencias de calidad de código</li>
                    <li>Comparación entre proyectos</li>
                </ul>
            </div>

            <div class="action">
                <a href="/projects" class="btn-primary">Ver proyectos</a>
            </div>
        </div>
    {/if}
</div>

<style>
    .container {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    .content {
        margin-top: 2rem;
    }

    .info-card {
        background: var(--color-bg-primary, #fff);
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 2rem 0;
    }

    .info-card h3 {
        margin-bottom: 1rem;
    }

    .info-card ul {
        list-style-position: inside;
        color: #666;
    }

    .action {
        margin-top: 2rem;
    }

    .btn-primary {
        background-color: #4a6cf7;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 4px;
        text-decoration: none;
        display: inline-block;
    }

    .btn-primary:hover {
        background-color: #3955d8;
    }

    .loading,
    .error {
        padding: 1rem;
        text-align: center;
    }

    .error {
        color: #d32f2f;
    }
</style>
