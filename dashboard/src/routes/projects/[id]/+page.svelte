<script lang="ts">
    import { page } from "$app/stores";
    import { onMount } from "svelte";

    // Obtener el ID del proyecto de la URL
    const projectId = $page.params.id;

    // Estado del proyecto y análisis
    let project: any = null;
    let loading = true;
    let error: string | null = null;
    let analyzing = false;
    let analysisResults: any = null;
    let analysisInProgress = false;

    // Cargar los detalles del proyecto
    onMount(async () => {
        try {
            loading = true;
            const response = await fetch(`/api/projects/${projectId}`);

            if (response.ok) {
                project = await response.json();
                // Comprobar si hay análisis previos
                checkExistingAnalysis();
            } else {
                error = `Error: ${response.status}`;
            }
        } catch (e) {
            console.error("Error cargando proyecto:", e);
            error = (e as Error).message;
        } finally {
            loading = false;
        }
    });

    // Verificar si ya existen análisis para este proyecto
    async function checkExistingAnalysis() {
        try {
            const response = await fetch(
                `/api/projects/${projectId}/analysis/latest`,
            );
            if (response.ok) {
                analysisResults = await response.json();
                // Si el análisis existe y tiene resultados, extraerlos
                if (analysisResults && analysisResults.result) {
                    // Fusionar los resultados del análisis con el objeto principal
                    analysisResults = {
                        ...analysisResults,
                        ...analysisResults.result,
                    };
                }
            }
        } catch (e) {
            console.error("Error comprobando análisis existentes:", e);
            // No establecemos error aquí porque no es crítico
        }
    }

    // Iniciar análisis del proyecto
    async function runAnalysis() {
        try {
            analyzing = true;
            analysisInProgress = true;

            const response = await fetch(`/api/analysis/run`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    projectId: projectId,
                    config: {
                        forceFullAnalysis: true,
                        includeMetrics: true,
                    },
                }),
            });

            if (response.ok) {
                const initialResponse = await response.json();
                console.log("Análisis iniciado:", initialResponse);

                // Esperar un poco y luego recuperar los resultados
                setTimeout(async () => {
                    await checkExistingAnalysis();
                    analysisInProgress = false;
                }, 5000); // Esperar 5 segundos para que termine el análisis
            } else {
                const errorData = await response.json().catch(() => ({}));
                error = `Error iniciando análisis: ${errorData.detail || response.status}`;
                analysisInProgress = false;
            }
        } catch (e) {
            console.error("Error al iniciar análisis:", e);
            error = (e as Error).message;
            analysisInProgress = false;
        } finally {
            analyzing = false;
        }
    }
</script>

<div class="project-detail-container">
    {#if loading}
        <div class="loading-spinner">Cargando detalles del proyecto...</div>
    {:else if error}
        <div class="error-message">
            <p>{error}</p>
            <button
                class="btn-secondary"
                on:click={() => window.location.reload()}
            >
                Reintentar
            </button>
        </div>
    {:else if project}
        <div class="project-header">
            <div>
                <h1>{project.name}</h1>
                <p class="project-slug">{project.slug}</p>
                <p class="project-description">
                    {project.description || "Sin descripción"}
                </p>
            </div>
            <div class="project-actions">
                <button
                    class="btn-primary"
                    on:click={runAnalysis}
                    disabled={analyzing || analysisInProgress}
                >
                    {#if analyzing}
                        Iniciando análisis...
                    {:else if analysisInProgress}
                        Análisis en progreso...
                    {:else}
                        Analizar proyecto
                    {/if}
                </button>
            </div>
        </div>

        <div class="project-details">
            <div class="details-section">
                <h2>Detalles del repositorio</h2>
                <div class="detail-card">
                    <div class="detail-item">
                        <span class="label">URL:</span>
                        <span class="value">
                            <a
                                href={project.repository_url}
                                target="_blank"
                                rel="noopener noreferrer"
                            >
                                {project.repository_url}
                            </a>
                        </span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Tipo:</span>
                        <span class="value"
                            >{project.repository_type || "Git"}</span
                        >
                    </div>
                    <div class="detail-item">
                        <span class="label">Rama principal:</span>
                        <span class="value"
                            >{project.default_branch || "main"}</span
                        >
                    </div>
                    <div class="detail-item">
                        <span class="label">Estado:</span>
                        <span
                            class="value status {project.status?.toLowerCase() ||
                                'active'}"
                        >
                            {project.status || "ACTIVE"}
                        </span>
                    </div>
                </div>
            </div>

            {#if analysisInProgress}
                <div class="details-section">
                    <h2>Análisis en progreso</h2>
                    <div class="progress-card">
                        <div class="progress-bar">
                            <div class="progress-indicator"></div>
                        </div>
                        <p>
                            El análisis está en progreso. Esto puede tardar unos
                            minutos...
                        </p>
                    </div>
                </div>
            {/if}

            {#if analysisResults && !analysisInProgress}
                <div class="details-section">
                    <h2>Resultados del análisis</h2>
                    <div class="analysis-results">
                        <div class="metrics-summary">
                            <div class="metric-card">
                                <div class="metric-value">
                                    {analysisResults.total_violations || 0}
                                </div>
                                <div class="metric-label">
                                    Problemas encontrados
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">
                                    {analysisResults.quality_score || "N/A"}
                                </div>
                                <div class="metric-label">
                                    Puntuación de calidad
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">
                                    {analysisResults.files_analyzed || 0}
                                </div>
                                <div class="metric-label">
                                    Archivos analizados
                                </div>
                            </div>
                        </div>

                        <!-- Análisis Cross-Language -->
                        {#if analysisResults.complexity_metrics?.cross_language_analysis}
                            {@const crossAnalysis =
                                analysisResults.complexity_metrics
                                    .cross_language_analysis}
                            <div class="cross-language-section">
                                <h3>🌐 Análisis Cross-Language</h3>

                                <!-- Lenguajes analizados -->
                                {#if crossAnalysis.languages_analyzed?.length > 0}
                                    <div class="languages-info">
                                        <h4>Lenguajes detectados:</h4>
                                        <div class="language-tags">
                                            {#each crossAnalysis.languages_analyzed as lang}
                                                <span
                                                    class="language-tag language-{lang}"
                                                    >{lang}</span
                                                >
                                            {/each}
                                        </div>
                                    </div>
                                {/if}

                                <!-- Archivos por lenguaje -->
                                {#if crossAnalysis.files_per_language}
                                    <div class="files-per-language">
                                        <h4>Archivos por lenguaje:</h4>
                                        <div class="language-stats">
                                            {#each Object.entries(crossAnalysis.files_per_language) as [lang, count]}
                                                <div class="language-stat">
                                                    <span class="lang-name"
                                                        >{lang}</span
                                                    >
                                                    <span class="file-count"
                                                        >{count} archivo{count !==
                                                        1
                                                            ? "s"
                                                            : ""}</span
                                                    >
                                                </div>
                                            {/each}
                                        </div>
                                    </div>
                                {/if}

                                <!-- Similitudes entre lenguajes -->
                                {#if crossAnalysis.high_similarity_pairs?.length > 0}
                                    <div class="similarities-section">
                                        <h4>🔍 Similitudes detectadas:</h4>
                                        <div class="similarities-list">
                                            {#each crossAnalysis.high_similarity_pairs.slice(0, 5) as pair}
                                                <div class="similarity-item">
                                                    <div
                                                        class="similarity-files"
                                                    >
                                                        <span class="file-path"
                                                            >{pair.file1
                                                                .split("/")
                                                                .pop()}</span
                                                        >
                                                        <span
                                                            class="similarity-arrow"
                                                            >↔</span
                                                        >
                                                        <span class="file-path"
                                                            >{pair.file2
                                                                .split("/")
                                                                .pop()}</span
                                                        >
                                                    </div>
                                                    <div
                                                        class="similarity-details"
                                                    >
                                                        <span
                                                            class="similarity-score"
                                                            >{Math.round(
                                                                pair.similarity *
                                                                    100,
                                                            )}%</span
                                                        >
                                                        <span class="languages"
                                                            >{pair.lang1} ↔ {pair.lang2}</span
                                                        >
                                                    </div>
                                                </div>
                                            {/each}
                                        </div>
                                    </div>
                                {/if}

                                <!-- Patrones cross-language -->
                                {#if crossAnalysis.cross_language_patterns?.length > 0}
                                    <div class="patterns-section">
                                        <h4>🎯 Patrones cross-language:</h4>
                                        <div class="patterns-list">
                                            {#each crossAnalysis.cross_language_patterns.slice(0, 3) as pattern}
                                                <div class="pattern-item">
                                                    <span
                                                        class="pattern-concept"
                                                        >{pattern.concept}</span
                                                    >
                                                    <div
                                                        class="pattern-languages"
                                                    >
                                                        {#each Object.entries(pattern.languages) as [lang, count]}
                                                            <span
                                                                class="pattern-lang"
                                                                >{lang}: {count}</span
                                                            >
                                                        {/each}
                                                    </div>
                                                </div>
                                            {/each}
                                        </div>
                                    </div>
                                {/if}
                            </div>
                        {/if}

                        <a
                            href="/analysis/{projectId}"
                            class="btn-secondary view-details-btn"
                        >
                            Ver análisis completo
                        </a>
                    </div>
                </div>
            {:else if !analysisInProgress}
                <div class="details-section">
                    <h2>Análisis de código</h2>
                    <div class="empty-analysis">
                        <p>Este proyecto aún no ha sido analizado.</p>
                        <p>
                            Haz clic en "Analizar proyecto" para iniciar el
                            primer análisis de código.
                        </p>
                    </div>
                </div>
            {/if}
        </div>
    {:else}
        <div class="error-message">
            <p>No se encontró el proyecto.</p>
            <a href="/projects" class="btn-secondary"
                >Volver a la lista de proyectos</a
            >
        </div>
    {/if}
</div>

<style>
    .project-detail-container {
        padding: 1.5rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    .project-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--color-border, #e5e5e5);
    }

    .project-slug {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.25rem;
        margin-bottom: 0.75rem;
    }

    .project-description {
        margin-top: 0.5rem;
        max-width: 800px;
        line-height: 1.6;
    }

    .project-actions {
        display: flex;
        gap: 1rem;
    }

    .details-section {
        margin-bottom: 2rem;
    }

    .details-section h2 {
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }

    .detail-card {
        background-color: var(--color-bg-primary, #ffffff);
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 8px;
        padding: 1.5rem;
    }

    .detail-item {
        display: flex;
        margin-bottom: 1rem;
    }

    .detail-item:last-child {
        margin-bottom: 0;
    }

    .label {
        flex-basis: 30%;
        font-weight: 500;
    }

    .value {
        flex-basis: 70%;
    }

    .value a {
        color: #4a6cf7;
        text-decoration: none;
    }

    .value a:hover {
        text-decoration: underline;
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

    .progress-card {
        background-color: var(--color-bg-primary, #ffffff);
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }

    .progress-bar {
        height: 8px;
        background-color: #f0f0f0;
        border-radius: 4px;
        margin-bottom: 1rem;
        overflow: hidden;
    }

    .progress-indicator {
        height: 100%;
        width: 30%;
        background-color: #4a6cf7;
        border-radius: 4px;
        animation: progress-animation 2s infinite;
    }

    @keyframes progress-animation {
        0% {
            width: 10%;
            margin-left: 0%;
        }
        50% {
            width: 30%;
            margin-left: 70%;
        }
        100% {
            width: 10%;
            margin-left: 0%;
        }
    }

    .analysis-results {
        background-color: var(--color-bg-primary, #ffffff);
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 8px;
        padding: 1.5rem;
    }

    .metrics-summary {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background-color: var(--color-bg-secondary, #f5f5f5);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }

    .empty-analysis {
        background-color: var(--color-bg-secondary, #f5f5f5);
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
    }

    .view-details-btn {
        display: block;
        text-align: center;
        text-decoration: none;
        padding: 0.75rem;
    }

    /* Cross-Language Analysis Styles */
    .cross-language-section {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--color-border, #e5e5e5);
    }

    .cross-language-section h3 {
        margin-bottom: 1.5rem;
        color: #4a6cf7;
        font-size: 1.3rem;
    }

    .cross-language-section h4 {
        margin-bottom: 0.75rem;
        font-size: 1rem;
        color: #333;
    }

    .languages-info {
        margin-bottom: 1.5rem;
    }

    .language-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .language-tag {
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
    }

    .language-python {
        background-color: #3776ab;
        color: white;
    }

    .language-typescript {
        background-color: #3178c6;
        color: white;
    }

    .language-javascript {
        background-color: #f7df1e;
        color: #000;
    }

    .language-rust {
        background-color: #ce422b;
        color: white;
    }

    .files-per-language {
        margin-bottom: 1.5rem;
    }

    .language-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.75rem;
    }

    .language-stat {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0.75rem;
        background-color: var(--color-bg-secondary, #f5f5f5);
        border-radius: 6px;
        font-size: 0.9rem;
    }

    .lang-name {
        font-weight: 500;
        text-transform: capitalize;
    }

    .file-count {
        color: #666;
        font-size: 0.85rem;
    }

    .similarities-section {
        margin-bottom: 1.5rem;
    }

    .similarities-list {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .similarity-item {
        padding: 0.75rem;
        background-color: var(--color-bg-secondary, #f5f5f5);
        border-radius: 6px;
        border-left: 3px solid #4a6cf7;
    }

    .similarity-files {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .file-path {
        font-family: monospace;
        font-size: 0.85rem;
        background-color: rgba(74, 108, 247, 0.1);
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
    }

    .similarity-arrow {
        color: #4a6cf7;
        font-weight: bold;
    }

    .similarity-details {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
    }

    .similarity-score {
        font-weight: bold;
        color: #4a6cf7;
    }

    .languages {
        color: #666;
    }

    .patterns-section {
        margin-bottom: 1.5rem;
    }

    .patterns-list {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .pattern-item {
        padding: 0.75rem;
        background-color: var(--color-bg-secondary, #f5f5f5);
        border-radius: 6px;
        border-left: 3px solid #28a745;
    }

    .pattern-concept {
        display: block;
        font-weight: 500;
        margin-bottom: 0.5rem;
        color: #28a745;
    }

    .pattern-languages {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .pattern-lang {
        font-size: 0.8rem;
        color: #666;
        background-color: rgba(40, 167, 69, 0.1);
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
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

    /* Dark mode */
    :global(.dark) .detail-card,
    :global(.dark) .progress-card,
    :global(.dark) .analysis-results {
        background-color: var(--color-bg-primary-dark, #1e1e1e);
        border-color: var(--color-border-dark, #333);
    }

    :global(.dark) .metric-card,
    :global(.dark) .empty-analysis {
        background-color: var(--color-bg-secondary-dark, #2a2a2a);
    }

    :global(.dark) .progress-bar {
        background-color: #333;
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

    /* Dark mode for cross-language analysis */
    :global(.dark) .cross-language-section {
        border-top-color: var(--color-border-dark, #333);
    }

    :global(.dark) .cross-language-section h3 {
        color: #6d8eff;
    }

    :global(.dark) .cross-language-section h4 {
        color: #e0e0e0;
    }

    :global(.dark) .language-stat,
    :global(.dark) .similarity-item,
    :global(.dark) .pattern-item {
        background-color: var(--color-bg-secondary-dark, #2a2a2a);
    }

    :global(.dark) .file-path {
        background-color: rgba(109, 142, 255, 0.2);
        color: #e0e0e0;
    }

    :global(.dark) .similarity-score {
        color: #6d8eff;
    }

    :global(.dark) .pattern-concept {
        color: #4caf50;
    }

    :global(.dark) .pattern-lang {
        background-color: rgba(76, 175, 80, 0.2);
        color: #b0b0b0;
    }
</style>
