<script lang="ts">
    import { page } from "$app/stores";
    import { onMount } from "svelte";

    const projectId = $page.params.id;

    let loading = true;
    let error: string | null = null;
    let analysis: any = null;

    onMount(async () => {
        try {
            const res = await fetch(`/api/projects/${projectId}/analysis/latest`);
            if (!res.ok) {
                error = `Error ${res.status}`;
                return;
            }
            const data = await res.json();
            // Unificar estructura: algunos campos vienen en result
            analysis = data?.result ? { ...data, ...data.result } : data;
        } catch (e: any) {
            error = e?.message ?? "Error desconocido";
        } finally {
            loading = false;
        }
    });
</script>

<div class="analysis-page">
    <div class="header">
        <h1>Análisis del proyecto</h1>
        <a class="btn-secondary" href={`/projects/${projectId}`}>Volver al proyecto</a>
    </div>

    {#if loading}
        <div class="loading">Cargando análisis...</div>
    {:else if error}
        <div class="error">{error}</div>
    {:else if analysis}
        <section class="summary">
            <div class="card">
                <div class="value">{analysis.files_analyzed ?? 0}</div>
                <div class="label">Archivos analizados</div>
            </div>
            <div class="card">
                <div class="value">{analysis.total_violations ?? 0}</div>
                <div class="label">Problemas encontrados</div>
            </div>
            <div class="card">
                <div class="value">{analysis.quality_score ?? "N/A"}</div>
                <div class="label">Puntuación de calidad</div>
            </div>
        </section>

        <section class="section">
            <h2>Complejidad</h2>
            {#if analysis.complexity_metrics}
                <div class="grid">
                    <div class="card"><strong>Funciones totales:</strong> {analysis.complexity_metrics.total_functions ?? 0}</div>
                    <div class="card"><strong>Funciones complejas:</strong> {analysis.complexity_metrics.complex_functions ?? 0}</div>
                    <div class="card"><strong>Complejidad media:</strong> {analysis.complexity_metrics.average_complexity ?? "-"}</div>
                    <div class="card"><strong>Complejidad máx.:</strong> {analysis.complexity_metrics.max_complexity ?? "-"}</div>
                </div>

                {#if analysis.complexity_metrics.complexity_hotspots?.length}
                    <div class="card">
                        <strong>Hotspots:</strong>
                        <ul>
                            {#each analysis.complexity_metrics.complexity_hotspots as h}
                                <li>{h.file} — {h.complexity}</li>
                            {/each}
                        </ul>
                    </div>
                {/if}
            {/if}
        </section>

        <section class="section">
            <h2>Cross-language</h2>
            {#if analysis.complexity_metrics?.cross_language_analysis}
                {#let x = analysis.complexity_metrics.cross_language_analysis}
                <div class="grid">
                    <div class="card"><strong>Lenguajes analizados:</strong> {x.languages_analyzed?.join(", ") ?? "-"}</div>
                    <div class="card">
                        <strong>Archivos por lenguaje:</strong>
                        <ul>
                            {#each Object.entries(x.files_per_language ?? {}) as [lang, count]}
                                <li>{lang}: {count}</li>
                            {/each}
                        </ul>
                    </div>
                </div>

                {#if x.high_similarity_pairs?.length}
                    <div class="card">
                        <strong>Pares con alta similitud:</strong>
                        <ul>
                            {#each x.high_similarity_pairs as p}
                                <li>{p.file1} ({p.lang1}) ↔ {p.file2} ({p.lang2}) — {p.similarity}</li>
                            {/each}
                        </ul>
                    </div>
                {/if}

                {#if x.cross_language_patterns?.length}
                    <div class="card">
                        <strong>Patrones cross-language:</strong>
                        <ul>
                            {#each x.cross_language_patterns as c}
                                <li>{c.concept}: {Object.keys(c.languages).join(", ")}</li>
                            {/each}
                        </ul>
                    </div>
                {/if}
            {:else}
                <div class="card">No hay datos cross-language disponibles.</div>
            {/if}
        </section>

        <section class="section">
            <h2>Calidad</h2>
            {#if analysis.quality_metrics}
                <pre class="pre">{JSON.stringify(analysis.quality_metrics, null, 2)}</pre>
            {:else}
                <div class="card">Sin métricas de calidad.</div>
            {/if}
        </section>

        <section class="section">
            <h2>Duplicados</h2>
            {#if analysis.duplicate_results}
                <pre class="pre">{JSON.stringify(analysis.duplicate_results, null, 2)}</pre>
            {:else}
                <div class="card">Sin resultados de duplicados.</div>
            {/if}
        </section>
    {:else}
        <div class="error">No hay análisis disponible.</div>
    {/if}
</div>

<style>
    .analysis-page { padding: 1.5rem; max-width: 1200px; margin: 0 auto; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
    .section { margin-bottom: 2rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }
    .card { background: var(--color-bg-primary, #fff); border: 1px solid var(--color-border, #e5e5e5); border-radius: 8px; padding: 1rem; }
    .value { font-size: 2rem; font-weight: 700; }
    .label { color: #666; }
    .loading, .error { padding: 1rem; }
    .pre { background: var(--color-bg-secondary, #f7f7f7); padding: 1rem; border-radius: 6px; overflow: auto; }
</style>

