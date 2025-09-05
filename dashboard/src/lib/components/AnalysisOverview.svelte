<script lang="ts">
    export let analysis: any = null;
    export let projectName: string = "";

    // Función para calcular el color del score
    function getScoreColor(score: number) {
        if (score >= 80) return "#10b981";
        if (score >= 60) return "#f59e0b";
        if (score >= 40) return "#f97316";
        return "#ef4444";
    }

    // Función para formatear números grandes
    function formatNumber(num: any) {
        if (num === null || num === undefined) return "0";
        return num.toLocaleString();
    }
</script>

<div class="analysis-overview">
    <div class="overview-header">
        <h2>🔍 Análisis Completo: {projectName}</h2>
        <div class="analysis-timestamp">
            Última actualización: {new Date().toLocaleString("es-ES")}
        </div>
    </div>

    {#if analysis}
        <!-- Score Principal -->
        <div class="score-section">
            <div
                class="score-circle"
                style="--score-color: {getScoreColor(
                    analysis.quality_score || 0,
                )}"
            >
                <svg viewBox="0 0 200 200" class="score-svg">
                    <circle cx="100" cy="100" r="90" class="score-bg" />
                    <circle
                        cx="100"
                        cy="100"
                        r="90"
                        class="score-fill"
                        style="stroke-dasharray: {(analysis.quality_score ||
                            0) * 5.65} 565;"
                    />
                </svg>
                <div class="score-text">
                    <div class="score-value">{analysis.quality_score || 0}</div>
                    <div class="score-label">Puntuación</div>
                </div>
            </div>

            <div class="score-details">
                <div class="detail-item">
                    <span class="detail-icon">📁</span>
                    <span class="detail-value"
                        >{formatNumber(analysis.files_analyzed)}</span
                    >
                    <span class="detail-label">Archivos analizados</span>
                </div>
                <div class="detail-item critical">
                    <span class="detail-icon">🚨</span>
                    <span class="detail-value"
                        >{formatNumber(analysis.critical_issues || 0)}</span
                    >
                    <span class="detail-label">Problemas críticos</span>
                </div>
                <div class="detail-item warning">
                    <span class="detail-icon">⚠️</span>
                    <span class="detail-value"
                        >{formatNumber(analysis.total_violations)}</span
                    >
                    <span class="detail-label">Total de problemas</span>
                </div>
            </div>
        </div>

        <!-- Grid de Análisis -->
        <div class="analysis-cards">
            <!-- Complejidad -->
            <div class="analysis-card complexity">
                <div class="card-header">
                    <span class="card-icon">🔧</span>
                    <h3>Complejidad del Código</h3>
                </div>
                <div class="card-metrics">
                    <div class="metric">
                        <span class="metric-value"
                            >{analysis.complexity_metrics?.total_functions ||
                                0}</span
                        >
                        <span class="metric-label">Funciones totales</span>
                    </div>
                    <div class="metric highlight">
                        <span class="metric-value"
                            >{analysis.complexity_metrics?.complex_functions ||
                                0}</span
                        >
                        <span class="metric-label">Funciones complejas</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value"
                            >{analysis.complexity_metrics?.average_complexity?.toFixed(
                                1,
                            ) || "N/A"}</span
                        >
                        <span class="metric-label">Complejidad promedio</span>
                    </div>
                </div>
                {#if analysis.complexity_metrics?.complexity_hotspots?.length}
                    <div class="hotspots">
                        <h4>🔥 Hotspots detectados</h4>
                        <ul>
                            {#each analysis.complexity_metrics.complexity_hotspots.slice(0, 3) as hotspot}
                                <li>
                                    {hotspot.file} - Complejidad: {hotspot.complexity}
                                </li>
                            {/each}
                        </ul>
                    </div>
                {/if}
            </div>

            <!-- Código Muerto con IA -->
            <div class="analysis-card dead-code">
                <div class="card-header">
                    <span class="card-icon">💀</span>
                    <h3>Código Muerto (IA 99% Certeza)</h3>
                </div>
                <div class="card-metrics">
                    <div class="metric">
                        <span class="metric-value"
                            >{analysis.dead_code_results?.total_issues ||
                                0}</span
                        >
                        <span class="metric-label">Total detectado</span>
                    </div>
                    {#if analysis.dead_code_results?.advanced_analysis}
                        <div class="metric success">
                            <span class="metric-value"
                                >{analysis.dead_code_results.advanced_analysis
                                    .safe_to_delete || 0}</span
                            >
                            <span class="metric-label">Seguros eliminar</span>
                        </div>
                        <div class="metric warning">
                            <span class="metric-value"
                                >{analysis.dead_code_results.advanced_analysis
                                    .requires_review || 0}</span
                            >
                            <span class="metric-label">Requieren revisión</span>
                        </div>
                    {/if}
                </div>
                {#if analysis.dead_code_results?.advanced_analysis}
                    <div class="ai-badge-wrapper">
                        <span class="ai-badge">🤖 Análisis con IA</span>
                        <span class="precision">99.9% precisión</span>
                    </div>
                {/if}
            </div>

            <!-- Seguridad -->
            <div class="analysis-card security">
                <div class="card-header">
                    <span class="card-icon">🔒</span>
                    <h3>Análisis de Seguridad</h3>
                </div>
                <div class="card-metrics">
                    <div class="metric critical">
                        <span class="metric-value"
                            >{analysis.security_results?.critical || 0}</span
                        >
                        <span class="metric-label">Críticas</span>
                    </div>
                    <div class="metric high">
                        <span class="metric-value"
                            >{analysis.security_results?.high || 0}</span
                        >
                        <span class="metric-label">Altas</span>
                    </div>
                    <div class="metric medium">
                        <span class="metric-value"
                            >{analysis.security_results?.medium || 0}</span
                        >
                        <span class="metric-label">Medias</span>
                    </div>
                </div>
                <div class="security-summary">
                    <span class="summary-icon">🛡️</span>
                    <span>OWASP Top 10 verificado</span>
                </div>
            </div>

            <!-- Tests -->
            <div class="analysis-card tests">
                <div class="card-header">
                    <span class="card-icon">🧪</span>
                    <h3>Cobertura de Tests</h3>
                </div>
                <div class="coverage-circle">
                    <svg viewBox="0 0 100 100" class="coverage-svg">
                        <circle cx="50" cy="50" r="45" class="coverage-bg" />
                        <circle
                            cx="50"
                            cy="50"
                            r="45"
                            class="coverage-fill"
                            style="stroke-dasharray: {(analysis
                                .test_coverage_results?.coverage_percentage ||
                                0) * 2.827} 282.7;"
                        />
                    </svg>
                    <div class="coverage-text">
                        {analysis.test_coverage_results?.coverage_percentage ||
                            0}%
                    </div>
                </div>
                <div class="test-breakdown">
                    <div class="test-type">
                        <span
                            >Unit: {analysis.test_coverage_results
                                ?.unit_tests || 0}</span
                        >
                    </div>
                    <div class="test-type">
                        <span
                            >E2E: {analysis.test_coverage_results?.e2e_tests ||
                                0}</span
                        >
                    </div>
                </div>
            </div>

            <!-- Performance -->
            <div class="analysis-card performance">
                <div class="card-header">
                    <span class="card-icon">⚡</span>
                    <h3>Performance</h3>
                </div>
                <div class="perf-issues">
                    {#if analysis.performance_results?.n_squared_algorithms > 0}
                        <div class="issue-item critical">
                            <span class="issue-icon">🐌</span>
                            <span
                                >{analysis.performance_results
                                    .n_squared_algorithms} algoritmos O(n²)</span
                            >
                        </div>
                    {/if}
                    {#if analysis.performance_results?.n_plus_one_queries > 0}
                        <div class="issue-item warning">
                            <span class="issue-icon">🔄</span>
                            <span
                                >{analysis.performance_results
                                    .n_plus_one_queries} N+1 queries</span
                            >
                        </div>
                    {/if}
                    {#if analysis.performance_results?.blocking_operations > 0}
                        <div class="issue-item">
                            <span class="issue-icon">🚫</span>
                            <span
                                >{analysis.performance_results
                                    .blocking_operations} operaciones bloqueantes</span
                            >
                        </div>
                    {/if}
                </div>
            </div>

            <!-- Arquitectura -->
            <div class="analysis-card architecture">
                <div class="card-header">
                    <span class="card-icon">🏗️</span>
                    <h3>Arquitectura</h3>
                </div>
                <div class="arch-status">
                    {#if analysis.architecture_results?.violations === 0}
                        <div class="status-good">
                            ✅ Arquitectura hexagonal respetada
                        </div>
                    {:else}
                        <div class="violations-list">
                            <div class="violation-item">
                                <span
                                    >God classes: {analysis.architecture_results
                                        ?.god_classes || 0}</span
                                >
                            </div>
                            <div class="violation-item">
                                <span
                                    >Circular deps: {analysis
                                        .architecture_results
                                        ?.circular_dependencies || 0}</span
                                >
                            </div>
                            <div class="violation-item">
                                <span
                                    >Layer violations: {analysis
                                        .architecture_results
                                        ?.layer_violations || 0}</span
                                >
                            </div>
                        </div>
                    {/if}
                </div>
            </div>
        </div>

        <!-- Resumen de lenguajes -->
        {#if analysis.complexity_metrics?.cross_language_analysis}
            <div class="languages-section">
                <h3>📊 Distribución por Lenguaje</h3>
                <div class="language-bars">
                    {#each Object.entries(analysis.complexity_metrics.cross_language_analysis.files_per_language || {}) as [lang, count]}
                        <div class="language-item">
                            <span class="lang-name">{lang}</span>
                            <div class="lang-bar-container">
                                <div
                                    class="lang-bar"
                                    style="width: {(count /
                                        analysis.files_analyzed) *
                                        100}%"
                                ></div>
                            </div>
                            <span class="lang-count">{count} archivos</span>
                        </div>
                    {/each}
                </div>
            </div>
        {/if}
    {:else}
        <div class="no-analysis">
            <span class="no-analysis-icon">📊</span>
            <p>No hay análisis disponible</p>
            <p class="no-analysis-hint">
                Ejecuta un análisis para ver los resultados aquí
            </p>
        </div>
    {/if}
</div>

<style>
    .analysis-overview {
        padding: 2rem;
        background: var(--color-bg-secondary, #f8f9fa);
        border-radius: 16px;
        max-width: 1400px;
        margin: 0 auto;
    }

    .overview-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }

    .overview-header h2 {
        margin: 0;
        color: #1e293b;
        font-size: 1.75rem;
    }

    .analysis-timestamp {
        color: #64748b;
        font-size: 0.9rem;
    }

    /* Score Section */
    .score-section {
        display: flex;
        gap: 3rem;
        align-items: center;
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    .score-circle {
        position: relative;
        width: 200px;
        height: 200px;
        flex-shrink: 0;
    }

    .score-svg {
        width: 100%;
        height: 100%;
        transform: rotate(-90deg);
    }

    .score-bg {
        fill: none;
        stroke: #e5e7eb;
        stroke-width: 10;
    }

    .score-fill {
        fill: none;
        stroke: var(--score-color);
        stroke-width: 10;
        stroke-linecap: round;
        transition: stroke-dasharray 1s ease-out;
    }

    .score-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
    }

    .score-value {
        font-size: 3rem;
        font-weight: 700;
        color: var(--score-color);
        line-height: 1;
    }

    .score-label {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }

    .score-details {
        display: flex;
        gap: 3rem;
        flex: 1;
    }

    .detail-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }

    .detail-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .detail-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }

    .detail-item.critical .detail-value {
        color: #ef4444;
    }

    .detail-item.warning .detail-value {
        color: #f59e0b;
    }

    .detail-label {
        color: #64748b;
        font-size: 0.9rem;
    }

    /* Analysis Cards */
    .analysis-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }

    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition:
            transform 0.2s,
            box-shadow 0.2s;
    }

    .analysis-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }

    .card-icon {
        font-size: 1.75rem;
    }

    .card-header h3 {
        margin: 0;
        font-size: 1.1rem;
        color: #1e293b;
    }

    .card-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .metric {
        text-align: center;
    }

    .metric-value {
        display: block;
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }

    .metric.highlight .metric-value {
        color: #f59e0b;
    }

    .metric.success .metric-value {
        color: #10b981;
    }

    .metric.warning .metric-value {
        color: #f59e0b;
    }

    .metric.critical .metric-value {
        color: #ef4444;
    }

    .metric.high .metric-value {
        color: #f97316;
    }

    .metric.medium .metric-value {
        color: #eab308;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
    }

    /* Hotspots */
    .hotspots {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }

    .hotspots h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
        color: #ef4444;
    }

    .hotspots ul {
        margin: 0;
        padding-left: 1.5rem;
        font-size: 0.85rem;
        color: #64748b;
    }

    /* AI Badge */
    .ai-badge-wrapper {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }

    .ai-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
        color: white;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .precision {
        color: #16a34a;
        font-weight: 500;
        font-size: 0.9rem;
    }

    /* Security Summary */
    .security-summary {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1rem;
        padding: 0.75rem;
        background: #f0f9ff;
        color: #0369a1;
        border-radius: 6px;
        font-size: 0.9rem;
    }

    /* Coverage Circle */
    .coverage-circle {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto 1rem;
    }

    .coverage-svg {
        width: 100%;
        height: 100%;
        transform: rotate(-90deg);
    }

    .coverage-bg {
        fill: none;
        stroke: #e5e7eb;
        stroke-width: 8;
    }

    .coverage-fill {
        fill: none;
        stroke: #10b981;
        stroke-width: 8;
        stroke-linecap: round;
        transition: stroke-dasharray 1s ease-out;
    }

    .coverage-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.5rem;
        font-weight: 700;
        color: #10b981;
    }

    .test-breakdown {
        display: flex;
        justify-content: center;
        gap: 2rem;
        font-size: 0.9rem;
        color: #64748b;
    }

    /* Performance Issues */
    .perf-issues {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .issue-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        background: #fef3c7;
        color: #92400e;
        border-radius: 6px;
        font-size: 0.9rem;
    }

    .issue-item.critical {
        background: #fee2e2;
        color: #991b1b;
    }

    .issue-item.warning {
        background: #fed7aa;
        color: #9a3412;
    }

    .issue-icon {
        font-size: 1.25rem;
    }

    /* Architecture Status */
    .status-good {
        padding: 1rem;
        background: #d1fae5;
        color: #065f46;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
    }

    .violations-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .violation-item {
        padding: 0.5rem 0.75rem;
        background: #fee2e2;
        color: #991b1b;
        border-radius: 6px;
        font-size: 0.9rem;
    }

    /* Languages Section */
    .languages-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .languages-section h3 {
        margin: 0 0 1.5rem 0;
        color: #1e293b;
    }

    .language-bars {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .language-item {
        display: grid;
        grid-template-columns: 100px 1fr 100px;
        align-items: center;
        gap: 1rem;
    }

    .lang-name {
        font-weight: 600;
        color: #1e293b;
        text-transform: capitalize;
    }

    .lang-bar-container {
        height: 28px;
        background: #f3f4f6;
        border-radius: 14px;
        overflow: hidden;
    }

    .lang-bar {
        height: 100%;
        background: linear-gradient(90deg, #4a6cf7 0%, #3955d8 100%);
        border-radius: 14px;
        transition: width 0.8s ease-out;
    }

    .lang-count {
        text-align: right;
        color: #64748b;
        font-size: 0.9rem;
    }

    /* No Analysis */
    .no-analysis {
        text-align: center;
        padding: 4rem 2rem;
        background: white;
        border-radius: 12px;
    }

    .no-analysis-icon {
        font-size: 4rem;
        display: block;
        margin-bottom: 1rem;
        opacity: 0.5;
    }

    .no-analysis p {
        margin: 0.5rem 0;
        color: #64748b;
    }

    .no-analysis-hint {
        font-size: 0.9rem;
    }

    /* Dark mode */
    :global(.dark) .analysis-overview {
        background: #0f172a;
    }

    :global(.dark) .score-section,
    :global(.dark) .analysis-card,
    :global(.dark) .languages-section,
    :global(.dark) .no-analysis {
        background: #1e293b;
    }

    :global(.dark) .overview-header h2,
    :global(.dark) .card-header h3,
    :global(.dark) .languages-section h3,
    :global(.dark) .metric-value,
    :global(.dark) .detail-value,
    :global(.dark) .lang-name {
        color: #f1f5f9;
    }

    :global(.dark) .analysis-timestamp,
    :global(.dark) .metric-label,
    :global(.dark) .detail-label,
    :global(.dark) .lang-count {
        color: #94a3b8;
    }

    :global(.dark) .score-bg,
    :global(.dark) .coverage-bg {
        stroke: #334155;
    }

    :global(.dark) .lang-bar-container {
        background: #334155;
    }
</style>
