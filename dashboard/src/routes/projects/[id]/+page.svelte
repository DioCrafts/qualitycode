<script lang="ts">
    import { page } from "$app/stores";
    import { onMount } from "svelte";

    // Obtener el ID del proyecto de la URL
    const projectId = $page.params.id;

    // Estado del proyecto y análisis
    let project = null;
    let loading = true;
    let error = null;
    let analyzing = false;
    let analysisResults = null;
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
            error = e.message;
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
            error = e.message;
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
                    <h2>📊 Resultados del Análisis Completo</h2>
                    <div class="analysis-results">
                        <div class="metrics-summary">
                            <div class="metric-card primary">
                                <div class="metric-icon">📈</div>
                                <div class="metric-value">
                                    {analysisResults.quality_score || 0}
                                </div>
                                <div class="metric-label">
                                    Puntuación Global
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-icon">📁</div>
                                <div class="metric-value">
                                    {analysisResults.files_analyzed || 0}
                                </div>
                                <div class="metric-label">
                                    Archivos Analizados
                                </div>
                            </div>
                            <div class="metric-card warning">
                                <div class="metric-icon">⚠️</div>
                                <div class="metric-value">
                                    {analysisResults.total_violations || 0}
                                </div>
                                <div class="metric-label">
                                    Problemas Encontrados
                                </div>
                            </div>
                            <div class="metric-card danger">
                                <div class="metric-icon">🚨</div>
                                <div class="metric-value">
                                    {analysisResults.critical_issues || 0}
                                </div>
                                <div class="metric-label">Críticos</div>
                            </div>
                        </div>

                        <!-- Mini resumen de cada análisis -->
                        <div class="analysis-grid">
                            <div class="analysis-item">
                                <span class="analysis-icon">🔧</span>
                                <span class="analysis-name">Complejidad</span>
                                <span class="analysis-value"
                                    >{analysisResults.complexity_metrics
                                        ?.total_functions || 0} funciones</span
                                >
                            </div>
                            <div class="analysis-item">
                                <span class="analysis-icon">💀</span>
                                <span class="analysis-name">Código Muerto</span>
                                <span class="analysis-value"
                                    >{analysisResults.dead_code_results
                                        ?.total_issues || 0} issues</span
                                >
                            </div>
                            <div class="analysis-item">
                                <span class="analysis-icon">🔒</span>
                                <span class="analysis-name">Seguridad</span>
                                <span class="analysis-value"
                                    >{analysisResults.security_results
                                        ?.total_vulnerabilities || 0} vulns</span
                                >
                            </div>
                            <div class="analysis-item">
                                <span class="analysis-icon">🐛</span>
                                <span class="analysis-name">Bugs</span>
                                <span class="analysis-value"
                                    >{analysisResults.bug_analysis_results
                                        ?.total_bugs || 0} detectados</span
                                >
                            </div>
                            <div class="analysis-item">
                                <span class="analysis-icon">📦</span>
                                <span class="analysis-name">Dependencias</span>
                                <span class="analysis-value"
                                    >{analysisResults.dependency_results
                                        ?.total_dependencies || 0} deps</span
                                >
                            </div>
                            <div class="analysis-item">
                                <span class="analysis-icon">🧪</span>
                                <span class="analysis-name">Tests</span>
                                <span class="analysis-value"
                                    >{analysisResults.test_coverage_results
                                        ?.coverage_percentage || 0}%</span
                                >
                            </div>
                            <div class="analysis-item">
                                <span class="analysis-icon">⚡</span>
                                <span class="analysis-name">Performance</span>
                                <span class="analysis-value"
                                    >{analysisResults.performance_results
                                        ?.total_issues || 0} issues</span
                                >
                            </div>
                            <div class="analysis-item">
                                <span class="analysis-icon">🏗️</span>
                                <span class="analysis-name">Arquitectura</span>
                                <span class="analysis-value"
                                    >{analysisResults.architecture_results
                                        ?.violations || 0} violaciones</span
                                >
                            </div>
                            <div class="analysis-item">
                                <span class="analysis-icon">📝</span>
                                <span class="analysis-name">Documentación</span>
                                <span class="analysis-value"
                                    >{analysisResults.documentation_results
                                        ?.coverage_percentage || 0}%</span
                                >
                            </div>
                        </div>

                        {#if analysisResults.dead_code_results?.advanced_analysis}
                            <div class="ai-highlight">
                                <span class="ai-badge">🤖 IA</span>
                                <span
                                    >Análisis inteligente detectó {analysisResults
                                        .dead_code_results.advanced_analysis
                                        .safe_to_delete || 0} items seguros para
                                    eliminar con 99% certeza</span
                                >
                            </div>
                        {/if}

                        <a
                            href={`/analysis/${projectId}`}
                            class="btn-primary view-details-btn"
                        >
                            Ver análisis detallado →
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
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        transition:
            transform 0.2s,
            box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }

    .metric-card.primary {
        background: linear-gradient(135deg, #4a6cf7 0%, #3955d8 100%);
        color: white;
    }

    .metric-card.warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }

    .metric-card.danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }

    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        line-height: 1;
    }

    .metric-label {
        color: #666;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .metric-card.primary .metric-label,
    .metric-card.warning .metric-label,
    .metric-card.danger .metric-label {
        color: rgba(255, 255, 255, 0.9);
    }

    /* Analysis Grid */
    .analysis-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
        padding: 1.5rem;
        background: var(--color-bg-secondary, #f8f9fa);
        border-radius: 12px;
    }

    .analysis-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        transition: all 0.2s;
        border: 1px solid transparent;
    }

    .analysis-item:hover {
        border-color: #4a6cf7;
        box-shadow: 0 4px 12px rgba(74, 108, 247, 0.1);
        transform: translateY(-2px);
    }

    .analysis-icon {
        font-size: 1.5rem;
    }

    .analysis-name {
        flex: 1;
        font-weight: 600;
        color: #334155;
    }

    .analysis-value {
        font-weight: 500;
        color: #64748b;
        font-size: 0.9rem;
    }

    /* AI Highlight */
    .ai-highlight {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.5rem;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #86efac;
        border-radius: 12px;
        margin: 1.5rem 0;
        font-weight: 500;
        color: #16a34a;
    }

    .ai-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background: white;
        border-radius: 6px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
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
</style>
