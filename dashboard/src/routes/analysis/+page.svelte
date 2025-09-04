<script lang="ts">
    import { onMount } from "svelte";
    import { fade } from "svelte/transition";

    let loading = true;
    let error = null;
    
    // Datos mock para la demostración
    const qualityScore = 90;
    const codeSmells = 23;
    const duplicatedLines = 2.1;
    const testCoverage = 87.3;
    const avgComplexity = 4.2;
    
    // Datos para la visualización de distribución de complejidad
    const complexityBars = [20, 45, 85, 60, 30, 70, 40, 55, 25, 65];
    
    // Lista de problemas
    const issues = [
        { severity: 'high', title: 'Unused variable \'authToken\'', file: 'src/auth/middleware.js:42', action: 'Auto-fix' },
        { severity: 'high', title: 'Function complexity too high (12)', file: 'src/utils/validator.js:156', action: 'Review' },
        { severity: 'medium', title: 'Duplicated code block', file: 'src/components/Form.tsx:89', action: 'Refactor' },
        { severity: 'medium', title: 'Missing error handling', file: 'src/api/endpoints.js:203', action: 'Fix' },
        { severity: 'low', title: 'Magic number should be constant', file: 'src/config/database.js:15', action: 'Fix' }
    ];
    
    // Datos de cobertura de código
    const coverageData = [
        { title: 'Overall Coverage', percentage: 87.3, lines: '12,847 / 14,732', level: 'high' },
        { title: 'Frontend Components', percentage: 92.1, lines: '4,234 / 4,598', level: 'high' },
        { title: 'API Endpoints', percentage: 76.4, lines: '2,847 / 3,724', level: 'medium' },
        { title: 'Utility Functions', percentage: 94.6, lines: '1,892 / 2,001', level: 'high' },
        { title: 'Database Layers', percentage: 43.2, lines: '987 / 2,287', level: 'low' },
        { title: 'Authentication', percentage: 89.7, lines: '1,145 / 1,276', level: 'high' }
    ];
    
    // Métricas DORA
    const doraMetrics = [
        { title: 'Deployment Frequency', value: '3.2x/day', level: 'elite' },
        { title: 'Lead Time for Changes', value: '2.4 days', level: 'high' },
        { title: 'Mean Time to Recovery', value: '45 min', level: 'elite' },
        { title: 'Change Failure Rate', value: '4.2%', level: 'high' }
    ];

    onMount(async () => {
        // Simulando carga de datos
        setTimeout(() => {
            loading = false;
        }, 1000);
    });
    
    // Función para manejar acciones sobre issues
    const handleIssueAction = (issue) => {
        // Aquí se implementaría la lógica para resolver cada issue
        console.log('Action on issue:', issue);
    };
</script>

<div class="main-content">
    {#if loading}
        <div class="loading">Cargando análisis...</div>
    {:else if error}
        <div class="error">Error: {error}</div>
    {:else}
        <header class="header" transition:fade={{ duration: 300 }}>
            <div class="header-left">
                <h2>Code Quality</h2>
                <p class="header-subtitle">Análisis estático, complejidad y métricas de calidad</p>
            </div>
            <div class="header-actions">
                <button class="btn btn-secondary">
                    <i class="fas fa-download"></i> Export Report
                </button>
                <button class="btn btn-primary">
                    <i class="fas fa-play"></i> Run Analysis
                </button>
            </div>
        </header>

        <section class="quality-overview" transition:fade={{ duration: 300, delay: 100 }}>
            <div class="quality-score-card">
                <div class="score-header">
                    <div>
                        <h3 class="score-title">Overall Quality Score</h3>
                        <p class="score-subtitle">Basado en análisis estático y métricas de complejidad</p>
                    </div>
                </div>
                
                <div class="score-visual">
                    <div class="score-circle" style="background: conic-gradient(from 0deg, var(--color-success) 0deg {qualityScore * 3.6}deg, var(--color-bg-light) {qualityScore * 3.6}deg 360deg);">
                        <span class="score-value">{qualityScore}</span>
                    </div>
                    
                    <div class="score-details">
                        <div class="score-metric">
                            <div class="metric-name">
                                <div class="metric-icon" style="background: var(--color-success);">
                                    <i class="fas fa-check"></i>
                                </div>
                                <span>Maintainability</span>
                            </div>
                            <span>A</span>
                        </div>
                        
                        <div class="score-metric">
                            <div class="metric-name">
                                <div class="metric-icon" style="background: var(--color-primary);">
                                    <i class="fas fa-code"></i>
                                </div>
                                <span>Reliability</span>
                            </div>
                            <span>A</span>
                        </div>
                        
                        <div class="score-metric">
                            <div class="metric-name">
                                <div class="metric-icon" style="background: var(--color-warning);">
                                    <i class="fas fa-tachometer-alt"></i>
                                </div>
                                <span>Performance</span>
                            </div>
                            <span>B</span>
                        </div>
                        
                        <div class="score-metric">
                            <div class="metric-name">
                                <div class="metric-icon" style="background: var(--color-purple);">
                                    <i class="fas fa-layer-group"></i>
                                </div>
                                <span>Complexity</span>
                            </div>
                            <span>B</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon" style="background: linear-gradient(135deg, var(--color-danger-light), var(--color-danger));">
                        <i class="fas fa-bug"></i>
                    </div>
                    <div class="stat-title">Code Smells</div>
                    <div class="stat-value">{codeSmells}</div>
                </div>

                <div class="stat-card">
                    <div class="stat-icon" style="background: linear-gradient(135deg, var(--color-warning-light), var(--color-warning));">
                        <i class="fas fa-exclamation-triangle"></i>
                    </div>
                    <div class="stat-title">Duplicated Lines</div>
                    <div class="stat-value">{duplicatedLines}%</div>
                </div>

                <div class="stat-card">
                    <div class="stat-icon" style="background: linear-gradient(135deg, var(--color-success-light), var(--color-success));">
                        <i class="fas fa-shield-alt"></i>
                    </div>
                    <div class="stat-title">Test Coverage</div>
                    <div class="stat-value">{testCoverage}%</div>
                </div>

                <div class="stat-card">
                    <div class="stat-icon" style="background: linear-gradient(135deg, var(--color-purple-light), var(--color-purple));">
                        <i class="fas fa-project-diagram"></i>
                    </div>
                    <div class="stat-title">Complexity</div>
                    <div class="stat-value">{avgComplexity}</div>
                </div>
            </div>
        </section>

        <section class="content-grid" transition:fade={{ duration: 300, delay: 200 }}>
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Code Complexity Distribution</h3>
                    <button class="filter-btn">
                        <i class="fas fa-filter"></i>
                    </button>
                </div>
                <div class="complexity-chart">
                    <div class="complexity-bars">
                        {#each complexityBars as height, i}
                            <div class="bar" style="height: {height}%;"></div>
                        {/each}
                    </div>
                </div>
                <div class="chart-labels">
                    <span>Baja (1-5)</span>
                    <span>Media (6-10)</span>
                    <span>Alta (11+)</span>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Top Issues to Fix</h3>
                    <span class="issue-count">{issues.length} total</span>
                </div>
                <div class="issues-list">
                    {#each issues as issue}
                        <div class="issue-item">
                            <div class="issue-severity severity-{issue.severity}"></div>
                            <div class="issue-details">
                                <div class="issue-title">{issue.title}</div>
                                <div class="issue-file">{issue.file}</div>
                            </div>
                            <button class="issue-action" on:click={() => handleIssueAction(issue)}>
                                {issue.action}
                            </button>
                        </div>
                    {/each}
                </div>
            </div>
        </section>

        <section class="coverage-section" transition:fade={{ duration: 300, delay: 300 }}>
            <div class="coverage-header">
                <h3 class="card-title">Test Coverage Analysis</h3>
                <div class="coverage-legend">
                    <span class="legend-item">
                        <div class="legend-dot high"></div>
                        > 80% (Good)
                    </span>
                    <span class="legend-item">
                        <div class="legend-dot medium"></div>
                        50-80% (Fair)
                    </span>
                    <span class="legend-item">
                        <div class="legend-dot low"></div>
                        &lt; 50% (Poor)
                    </span>
                </div>
            </div>
            
            <div class="coverage-grid">
                {#each coverageData as coverage}
                    <div class="coverage-item">
                        <div class="coverage-title">{coverage.title}</div>
                        <div class="coverage-bar">
                            <div class="coverage-fill {coverage.level}" style="width: {coverage.percentage}%;"></div>
                        </div>
                        <div class="coverage-stats">
                            <span>{coverage.percentage}%</span>
                            <span>{coverage.lines}</span>
                        </div>
                    </div>
                {/each}
            </div>
        </section>

        <section class="dora-metrics" transition:fade={{ duration: 300, delay: 400 }}>
            <h3 class="card-title">DORA Metrics</h3>
            <p class="dora-subtitle">DevOps Research and Assessment key performance indicators</p>
            
            <div class="dora-grid">
                {#each doraMetrics as metric}
                    <div class="dora-metric {metric.level}">
                        <div class="dora-title">{metric.title}</div>
                        <div class="dora-value">{metric.value}</div>
                        <span class="dora-label label-{metric.level}">{metric.level}</span>
                    </div>
                {/each}
            </div>
        </section>
    {/if}
</div>

<style>
    /* Variables para tema claro */
    :root {
        --color-bg: #f8fafc;
        --color-bg-light: #e2e8f0;
        --color-bg-card: #ffffff;
        --color-text: #334155;
        --color-text-light: #64748b;
        --color-border: #e2e8f0;
        --color-primary: #3b82f6;
        --color-primary-light: #60a5fa;
        --color-success: #10b981;
        --color-success-light: #34d399;
        --color-warning: #f59e0b;
        --color-warning-light: #fbbf24;
        --color-danger: #ef4444;
        --color-danger-light: #f87171;
        --color-purple: #8b5cf6;
        --color-purple-light: #a78bfa;
    }
    
    /* Estilos base */
    .main-content {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
        color: var(--color-text);
        background-color: var(--color-bg);
        min-height: 100vh;
    }

    /* Header */
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }

    .header-left h2 {
        font-size: 2rem;
        font-weight: 600;
        background: linear-gradient(135deg, var(--color-primary), var(--color-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }

    .header-subtitle {
        color: var(--color-text-light);
        font-size: 1rem;
    }

    .header-actions {
        display: flex;
        gap: 1rem;
        align-items: center;
    }

    .btn {
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }

    .btn-primary {
        background: linear-gradient(135deg, var(--color-primary), var(--color-purple));
        color: white;
    }

    .btn-secondary {
        background: var(--color-bg-light);
        color: var(--color-text);
        border: 1px solid var(--color-border);
    }

    .btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }

    /* Quality overview section */
    .quality-overview {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 2rem;
        margin-bottom: 2rem;
    }

    .quality-score-card {
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: 1rem;
        padding: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .quality-score-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--color-success), var(--color-primary), var(--color-purple));
    }

    .score-header {
        display: flex;
        justify-content: between;
        align-items: center;
        margin-bottom: 2rem;
    }

    .score-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--color-text);
    }

    .score-subtitle {
        color: var(--color-text-light);
    }

    .score-visual {
        display: flex;
        align-items: center;
        gap: 2rem;
        margin: 2rem 0;
    }

    .score-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }

    .score-circle::before {
        content: '';
        width: 90px;
        height: 90px;
        border-radius: 50%;
        background: var(--color-bg-card);
        position: absolute;
    }

    .score-value {
        font-size: 2rem;
        font-weight: 700;
        z-index: 2;
        color: var(--color-success);
    }

    .score-details {
        flex: 1;
    }

    .score-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.75rem 0;
        padding: 0.75rem;
        background: var(--color-bg-light);
        border-radius: 0.5rem;
    }

    .metric-name {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .metric-icon {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        color: white;
    }

    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }

    .stat-card {
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: 1rem;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .stat-icon {
        width: 48px;
        height: 48px;
        border-radius: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        color: white;
        margin: 0 auto 1rem;
    }

    .stat-title {
        font-size: 0.875rem;
        color: var(--color-text-light);
        margin-bottom: 0.5rem;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--color-text);
    }

    /* Content Grid */
    .content-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin-bottom: 2rem;
    }

    .card {
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }

    .card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--color-text);
    }
    
    .issue-count {
        font-size: 0.875rem; 
        color: var(--color-text-light);
    }

    /* Complexity Chart */
    .complexity-chart {
        height: 200px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(139, 92, 246, 0.05));
        border-radius: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }

    .complexity-bars {
        display: flex;
        align-items: flex-end;
        gap: 0.5rem;
        height: 120px;
    }

    .bar {
        width: 20px;
        background: linear-gradient(to top, var(--color-danger), var(--color-warning), var(--color-success));
        border-radius: 4px 4px 0 0;
    }
    
    .chart-labels {
        display: flex;
        justify-content: space-around;
        font-size: 0.75rem;
        color: var(--color-text-light);
    }

    /* Issues List */
    .issues-list {
        max-height: 300px;
        overflow-y: auto;
    }

    .issue-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        border: 1px solid var(--color-border);
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
    }

    .issue-item:hover {
        background: var(--color-bg-light);
    }

    .issue-severity {
        width: 8px;
        height: 40px;
        border-radius: 4px;
    }

    .severity-high { background: var(--color-danger); }
    .severity-medium { background: var(--color-warning); }
    .severity-low { background: var(--color-success); }

    .issue-details {
        flex: 1;
    }

    .issue-title {
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
        color: var(--color-text);
    }

    .issue-file {
        font-size: 0.75rem;
        color: var(--color-text-light);
        font-family: 'Monaco', 'Courier', monospace;
    }

    .issue-action {
        padding: 0.25rem 0.75rem;
        background: rgba(59, 130, 246, 0.1);
        color: var(--color-primary);
        border-radius: 9999px;
        font-size: 0.75rem;
        border: none;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .issue-action:hover {
        background: rgba(59, 130, 246, 0.2);
    }

    /* Coverage Section */
    .coverage-section {
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .coverage-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }
    
    .coverage-legend {
        display: flex;
        gap: 1rem;
        font-size: 0.875rem;
        color: var(--color-text-light);
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    .legend-dot.high { background: var(--color-success); }
    .legend-dot.medium { background: var(--color-warning); }
    .legend-dot.low { background: var(--color-danger); }

    .coverage-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
    }

    .coverage-item {
        padding: 1.5rem;
        background: var(--color-bg-light);
        border-radius: 0.75rem;
    }

    .coverage-title {
        font-size: 0.875rem;
        color: var(--color-text-light);
        margin-bottom: 1rem;
    }

    .coverage-bar {
        width: 100%;
        height: 8px;
        background: rgba(100, 116, 139, 0.2);
        border-radius: 4px;
        overflow: hidden;
        margin-bottom: 0.75rem;
    }

    .coverage-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .coverage-fill.high { background: linear-gradient(90deg, var(--color-success), var(--color-success-light)); }
    .coverage-fill.medium { background: linear-gradient(90deg, var(--color-warning), var(--color-warning-light)); }
    .coverage-fill.low { background: linear-gradient(90deg, var(--color-danger), var(--color-danger-light)); }

    .coverage-stats {
        display: flex;
        justify-content: space-between;
        font-size: 0.875rem;
        color: var(--color-text);
    }

    /* DORA Metrics */
    .dora-metrics {
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .dora-subtitle {
        color: var(--color-text-light); 
        margin-bottom: 1rem;
    }

    .dora-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-top: 1.5rem;
    }

    .dora-metric {
        text-align: center;
        padding: 1.5rem;
        background: var(--color-bg-light);
        border-radius: 0.75rem;
        position: relative;
        overflow: hidden;
    }

    .dora-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }

    .dora-metric.elite::before { background: var(--color-success); }
    .dora-metric.high::before { background: var(--color-primary); }
    .dora-metric.medium::before { background: var(--color-warning); }
    .dora-metric.low::before { background: var(--color-danger); }

    .dora-title {
        font-size: 0.875rem;
        color: var(--color-text-light);
        margin-bottom: 0.75rem;
    }

    .dora-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--color-text);
    }

    .dora-label {
        font-size: 0.75rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 500;
        text-transform: capitalize;
    }

    .label-elite {
        background: rgba(16, 185, 129, 0.1);
        color: var(--color-success);
    }

    .label-high {
        background: rgba(59, 130, 246, 0.1);
        color: var(--color-primary);
    }

    .label-medium {
        background: rgba(245, 158, 11, 0.1);
        color: var(--color-warning);
    }

    .label-low {
        background: rgba(239, 68, 68, 0.1);
        color: var(--color-danger);
    }

    /* Responsive */
    @media (max-width: 1024px) {
        .quality-overview,
        .content-grid {
            grid-template-columns: 1fr;
        }

        .coverage-grid,
        .dora-grid {
            grid-template-columns: 1fr 1fr;
        }
    }

    @media (max-width: 768px) {
        .header {
            flex-direction: column;
            align-items: flex-start;
            gap: 1rem;
        }

        .header-actions {
            width: 100%;
            justify-content: space-between;
        }

        .coverage-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 1rem;
        }
        
        .coverage-grid,
        .dora-grid {
            grid-template-columns: 1fr;
        }
        
        .score-visual {
            flex-direction: column;
            align-items: center;
        }
    }
    
    /* Loading y Error */
    .loading,
    .error {
        padding: 2rem;
        text-align: center;
        background: var(--color-bg-card);
        border-radius: 1rem;
        margin: 2rem 0;
        border: 1px solid var(--color-border);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .error {
        color: var(--color-danger);
        border-color: var(--color-danger-light);
    }
</style>
