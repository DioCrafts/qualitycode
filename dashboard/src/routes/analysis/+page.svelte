<script lang="ts">
    import { goto } from "$app/navigation";
    import { dashboardStore } from "$lib/stores/dashboard";
    import { onMount } from "svelte";
    import { get } from "svelte/store";

    let selectedProjectId: string | null = null;

    onMount(() => {
        const state = get(dashboardStore) as any;
        selectedProjectId = state?.selectedProject?.id || null;
        if (selectedProjectId) {
            goto(`/analysis/${selectedProjectId}`);
        }
    });
</script>

<div class="container">
    <h1>Análisis</h1>
    {#if !selectedProjectId}
        <p>No hay proyecto seleccionado.</p>
        <a class="btn" href="/projects">Seleccionar proyecto</a>
    {/if}
</div>

<style>
    .container {
        padding: 1.5rem;
        max-width: 900px;
        margin: 0 auto;
    }
    .btn {
        display: inline-block;
        padding: 0.6rem 1rem;
        border: 1px solid #4a6cf7;
        color: #4a6cf7;
        border-radius: 6px;
        text-decoration: none;
    }
</style>
