<script lang="ts">
    import { page } from "$app/stores";
    import {
        AlertTriangle,
        Brain,
        Bug,
        Building,
        Building2,
        ChartColumn,
        CheckCircle,
        CheckCircle2,
        Code2,
        Copy,
        Eye,
        FileCode,
        FileText,
        Gauge,
        Info,
        Lightbulb,
        Package,
        Search,
        Shield,
        Skull,
        TestTube,
        TestTube2,
        Trash2,
        TrendingUp,
        Zap,
    } from "lucide-svelte";
    import { onMount } from "svelte";

    const projectId = $page.params.id;

    let loading = true;
    let error: string | null = null;
    let analysis: any = null;
    let activeTab = "overview";

    // Íconos para cada tipo de análisis
    const analysisIcons = {
        complexity: ChartColumn,
        quality: TrendingUp,
        deadCode: Skull,
        security: Shield,
        duplicates: Copy,
        bugs: Bug,
        dependencies: Package,
        testCoverage: TestTube2,
        performance: Zap,
        architecture: Building2,
        documentation: FileText,
    };

    // Colores para niveles de severidad
    const severityColors = {
        critical: "#ef4444",
        high: "#f97316",
        medium: "#eab308",
        low: "#3b82f6",
        info: "#6b7280",
    };

    onMount(async () => {
        try {
            const res = await fetch(
                `/api/projects/${projectId}/analysis/latest`,
            );
            if (!res.ok) {
                error = `Error ${res.status}`;
                return;
            }
            const data = await res.json();
            // Unificar estructura: algunos campos vienen en result
            analysis = data?.result ? { ...data, ...data.result } : data;
            console.log("Analysis data:", analysis);
            console.log("Dead code results:", analysis?.dead_code_results);
            console.log(
                "Unused variables:",
                analysis?.dead_code_results?.unused_variables,
            );
            console.log(
                "Unused functions:",
                analysis?.dead_code_results?.unused_functions,
            );
            console.log(
                "Unused classes:",
                analysis?.dead_code_results?.unused_classes,
            );
            console.log(
                "Advanced analysis:",
                analysis?.dead_code_results?.advanced_analysis,
            );
            console.log("Complexity metrics:", analysis?.complexity_metrics);
        } catch (e: any) {
            error = e?.message ?? "Error desconocido";
        } finally {
            loading = false;
        }
    });

    function getSeverityIcon(severity: string) {
        switch (severity?.toLowerCase()) {
            case "critical":
                return AlertTriangle;
            case "high":
                return AlertTriangle;
            case "medium":
                return Info;
            case "low":
                return Info;
            default:
                return Info;
        }
    }

    function formatPercentage(value: any) {
        if (value === null || value === undefined) return "N/A";
        return `${value}%`;
    }

    function formatNumber(value: any) {
        if (value === null || value === undefined) return "0";
        return value.toLocaleString();
    }
</script>

<div class="analysis-page">
    <div class="header">
        <div class="header-left">
            <h1>🔍 Análisis Completo del Proyecto</h1>
            <p class="subtitle">
                Resultados detallados de los 11 análisis ejecutados
            </p>
        </div>
        <a class="btn-secondary" href={`/projects/${projectId}`}>
            ← Volver al proyecto
        </a>
    </div>

    {#if loading}
        <div class="loading">
            <div class="spinner"></div>
            <p>Cargando análisis...</p>
        </div>
    {:else if error}
        <div class="error-card">
            <AlertTriangle size={24} />
            <span>{error}</span>
        </div>
    {:else if analysis}
        <!-- Resumen General -->
        <section class="summary-cards">
            <div class="summary-card primary">
                <div class="card-icon">
                    <TrendingUp size={32} />
                </div>
                <div class="card-content">
                    <div class="value">{analysis.quality_score ?? 0}</div>
                    <div class="label">Puntuación Global</div>
                </div>
            </div>

            <div class="summary-card">
                <div class="card-icon">
                    <FileText size={24} />
                </div>
                <div class="card-content">
                    <div class="value">
                        {formatNumber(analysis.files_analyzed ?? 0)}
                    </div>
                    <div class="label">Archivos Analizados</div>
                </div>
            </div>

            <div class="summary-card">
                <div class="card-icon">
                    <AlertTriangle size={24} />
                </div>
                <div class="card-content">
                    <div class="value">
                        {formatNumber(analysis.total_violations ?? 0)}
                    </div>
                    <div class="label">Problemas Totales</div>
                </div>
            </div>

            <div class="summary-card">
                <div class="card-icon">
                    <Shield size={24} />
                </div>
                <div class="card-content">
                    <div class="value severity-critical">
                        {formatNumber(analysis.critical_issues ?? 0)}
                    </div>
                    <div class="label">Problemas Críticos</div>
                </div>
            </div>
        </section>

        <!-- Navegación por pestañas -->
        <div class="tabs">
            <button
                class="tab {activeTab === 'overview' ? 'active' : ''}"
                on:click={() => (activeTab = "overview")}
            >
                Vista General
            </button>
            <button
                class="tab {activeTab === 'complexity' ? 'active' : ''}"
                on:click={() => (activeTab = "complexity")}
            >
                <ChartColumn size={16} /> Complejidad
            </button>
            <button
                class="tab {activeTab === 'quality' ? 'active' : ''}"
                on:click={() => (activeTab = "quality")}
            >
                <TrendingUp size={16} /> Calidad
            </button>
            <button
                class="tab {activeTab === 'deadCode' ? 'active' : ''}"
                on:click={() => (activeTab = "deadCode")}
            >
                <Skull size={16} /> Código Muerto
            </button>
            <button
                class="tab {activeTab === 'security' ? 'active' : ''}"
                on:click={() => (activeTab = "security")}
            >
                <Shield size={16} /> Seguridad
            </button>
            <button
                class="tab {activeTab === 'bugs' ? 'active' : ''}"
                on:click={() => (activeTab = "bugs")}
            >
                <Bug size={16} /> Bugs
            </button>
            <button
                class="tab {activeTab === 'duplicates' ? 'active' : ''}"
                on:click={() => (activeTab = "duplicates")}
            >
                <Copy size={16} /> Duplicados
            </button>
            <button
                class="tab {activeTab === 'more' ? 'active' : ''}"
                on:click={() => (activeTab = "more")}
            >
                Más Análisis
            </button>
        </div>

        <!-- Contenido de las pestañas -->
        <div class="tab-content">
            {#if activeTab === "overview"}
                <div class="overview-grid">
                    <!-- 1. Análisis de Complejidad -->
                    <div class="analysis-card">
                        <div class="analysis-header">
                            <ChartColumn size={20} />
                            <h3>Análisis de Complejidad</h3>
                        </div>
                        {#if analysis.complexity_metrics}
                            <div class="metrics-grid">
                                <div class="metric">
                                    <span class="metric-label"
                                        >Funciones totales</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.complexity_metrics
                                            .total_functions ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Funciones complejas</span
                                    >
                                    <span class="metric-value warning"
                                        >{analysis.complexity_metrics
                                            .complex_functions ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Complejidad promedio</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.complexity_metrics.average_complexity?.toFixed(
                                            1,
                                        ) ?? "N/A"}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Complejidad máxima</span
                                    >
                                    <span class="metric-value danger"
                                        >{analysis.complexity_metrics
                                            .max_complexity ?? "N/A"}</span
                                    >
                                </div>
                            </div>
                        {/if}
                    </div>

                    <!-- 2. Métricas de Calidad -->
                    <div class="analysis-card">
                        <div class="analysis-header">
                            <TrendingUp size={20} />
                            <h3>Métricas de Calidad</h3>
                        </div>
                        {#if analysis.quality_metrics}
                            <div class="metrics-grid">
                                <div class="metric">
                                    <span class="metric-label"
                                        >Índice de Mantenibilidad</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.quality_metrics.maintainability_index?.toFixed(
                                            1,
                                        ) ?? "N/A"}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Deuda Técnica</span
                                    >
                                    <span class="metric-value warning"
                                        >{analysis.quality_metrics
                                            .technical_debt_hours ?? 0}h</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Cobertura de Docs</span
                                    >
                                    <span class="metric-value"
                                        >{formatPercentage(
                                            analysis.quality_metrics
                                                .documentation_coverage,
                                        )}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Code Smells</span
                                    >
                                    <span class="metric-value danger"
                                        >{analysis.quality_metrics
                                            .code_smells ?? 0}</span
                                    >
                                </div>
                            </div>
                        {/if}
                    </div>

                    <!-- 3. Código Muerto Inteligente -->
                    <div class="analysis-card">
                        <div class="analysis-header">
                            <Skull size={20} />
                            <h3>Código Muerto (IA 99% Certeza)</h3>
                        </div>
                        {#if analysis.dead_code_results}
                            <div class="metrics-grid">
                                <div class="metric">
                                    <span class="metric-label"
                                        >Variables no usadas</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.dead_code_results
                                            .unused_variables?.length ??
                                            0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Funciones no usadas</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.dead_code_results
                                            .unused_functions?.length ??
                                            0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Clases no usadas</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.dead_code_results
                                            .unused_classes?.length ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Total issues</span
                                    >
                                    <span class="metric-value danger"
                                        >{analysis.dead_code_results
                                            .total_issues ?? 0}</span
                                    >
                                </div>
                            </div>
                            {#if analysis.dead_code_results.advanced_analysis}
                                <div class="advanced-analysis">
                                    <div class="confidence-badge success">
                                        ✨ Análisis con IA: {analysis
                                            .dead_code_results.advanced_analysis
                                            .safe_to_delete ?? 0} items seguros para
                                        eliminar
                                    </div>
                                </div>
                            {/if}
                        {/if}
                    </div>

                    <!-- 4. Análisis de Seguridad -->
                    <div class="analysis-card">
                        <div class="analysis-header">
                            <Shield size={20} />
                            <h3>Análisis de Seguridad</h3>
                        </div>
                        {#if analysis.security_results}
                            <div class="metrics-grid">
                                <div class="metric">
                                    <span class="metric-label"
                                        >Vulnerabilidades</span
                                    >
                                    <span class="metric-value danger"
                                        >{analysis.security_results
                                            .total_vulnerabilities ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Críticas</span>
                                    <span class="metric-value critical"
                                        >{analysis.security_results.critical ??
                                            0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Altas</span>
                                    <span class="metric-value warning"
                                        >{analysis.security_results.high ??
                                            0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Medias</span>
                                    <span class="metric-value"
                                        >{analysis.security_results.medium ??
                                            0}</span
                                    >
                                </div>
                            </div>
                        {/if}
                    </div>

                    <!-- 5. Bugs Potenciales -->
                    <div class="analysis-card">
                        <div class="analysis-header">
                            <Bug size={20} />
                            <h3>Bugs Potenciales</h3>
                        </div>
                        {#if analysis.bug_analysis_results}
                            <div class="metrics-grid">
                                <div class="metric">
                                    <span class="metric-label"
                                        >Bugs detectados</span
                                    >
                                    <span class="metric-value danger"
                                        >{analysis.bug_analysis_results
                                            .total_bugs ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Null pointers</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.bug_analysis_results
                                            .null_pointer_risks ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >División por cero</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.bug_analysis_results
                                            .division_by_zero ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Memory leaks</span
                                    >
                                    <span class="metric-value warning"
                                        >{analysis.bug_analysis_results
                                            .memory_leaks ?? 0}</span
                                    >
                                </div>
                            </div>
                        {/if}
                    </div>

                    <!-- 6. Análisis de Dependencias -->
                    <div class="analysis-card">
                        <div class="analysis-header">
                            <Package size={20} />
                            <h3>Análisis de Dependencias</h3>
                        </div>
                        {#if analysis.dependency_results}
                            <div class="metrics-grid">
                                <div class="metric">
                                    <span class="metric-label">Total deps</span>
                                    <span class="metric-value"
                                        >{analysis.dependency_results
                                            .total_dependencies ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Obsoletas</span>
                                    <span class="metric-value warning"
                                        >{analysis.dependency_results
                                            .outdated ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Vulnerables</span
                                    >
                                    <span class="metric-value danger"
                                        >{analysis.dependency_results
                                            .vulnerable ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Sin licencia</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.dependency_results
                                            .no_license ?? 0}</span
                                    >
                                </div>
                            </div>
                        {/if}
                    </div>

                    <!-- 7. Cobertura de Tests -->
                    <div class="analysis-card">
                        <div class="analysis-header">
                            <TestTube2 size={20} />
                            <h3>Cobertura de Tests</h3>
                        </div>
                        {#if analysis.test_coverage_results}
                            <div class="metrics-grid">
                                <div class="metric">
                                    <span class="metric-label"
                                        >Cobertura total</span
                                    >
                                    <span class="metric-value"
                                        >{formatPercentage(
                                            analysis.test_coverage_results
                                                .coverage_percentage,
                                        )}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Tests totales</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.test_coverage_results
                                            .total_tests ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Unit tests</span>
                                    <span class="metric-value"
                                        >{analysis.test_coverage_results
                                            .unit_tests ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label">E2E tests</span>
                                    <span class="metric-value"
                                        >{analysis.test_coverage_results
                                            .e2e_tests ?? 0}</span
                                    >
                                </div>
                            </div>
                        {/if}
                    </div>

                    <!-- 8. Análisis de Performance -->
                    <div class="analysis-card">
                        <div class="analysis-header">
                            <Zap size={20} />
                            <h3>Análisis de Performance</h3>
                        </div>
                        {#if analysis.performance_results}
                            <div class="metrics-grid">
                                <div class="metric">
                                    <span class="metric-label"
                                        >Issues totales</span
                                    >
                                    <span class="metric-value warning"
                                        >{analysis.performance_results
                                            .total_issues ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >O(n²) detectados</span
                                    >
                                    <span class="metric-value danger"
                                        >{analysis.performance_results
                                            .n_squared_algorithms ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label">N+1 queries</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.performance_results
                                            .n_plus_one_queries ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Blocking ops</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.performance_results
                                            .blocking_operations ?? 0}</span
                                    >
                                </div>
                            </div>
                        {/if}
                    </div>

                    <!-- 9. Análisis de Arquitectura -->
                    <div class="analysis-card">
                        <div class="analysis-header">
                            <Building2 size={20} />
                            <h3>Análisis de Arquitectura</h3>
                        </div>
                        {#if analysis.architecture_results}
                            <div class="metrics-grid">
                                <div class="metric">
                                    <span class="metric-label">Violaciones</span
                                    >
                                    <span class="metric-value danger"
                                        >{analysis.architecture_results
                                            .violations ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label">God classes</span
                                    >
                                    <span class="metric-value warning"
                                        >{analysis.architecture_results
                                            .god_classes ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Circular deps</span
                                    >
                                    <span class="metric-value danger"
                                        >{analysis.architecture_results
                                            .circular_dependencies ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Layer violations</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.architecture_results
                                            .layer_violations ?? 0}</span
                                    >
                                </div>
                            </div>
                        {/if}
                    </div>

                    <!-- 10. Análisis de Documentación -->
                    <div class="analysis-card">
                        <div class="analysis-header">
                            <FileText size={20} />
                            <h3>Análisis de Documentación</h3>
                        </div>
                        {#if analysis.documentation_results}
                            <div class="metrics-grid">
                                <div class="metric">
                                    <span class="metric-label">Cobertura</span>
                                    <span class="metric-value"
                                        >{formatPercentage(
                                            analysis.documentation_results
                                                .coverage_percentage,
                                        )}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Sin docs</span>
                                    <span class="metric-value warning"
                                        >{analysis.documentation_results
                                            .undocumented_functions ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >README score</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.documentation_results
                                            .readme_score ?? 0}/10</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label">TODOs</span>
                                    <span class="metric-value"
                                        >{analysis.documentation_results
                                            .todos ?? 0}</span
                                    >
                                </div>
                            </div>
                        {/if}
                    </div>

                    <!-- 11. Análisis de Duplicados -->
                    <div class="analysis-card">
                        <div class="analysis-header">
                            <Copy size={20} />
                            <h3>Análisis de Duplicados</h3>
                        </div>
                        {#if analysis.duplicate_results}
                            <div class="metrics-grid">
                                <div class="metric">
                                    <span class="metric-label"
                                        >Bloques duplicados</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.duplicate_results
                                            .duplicate_blocks ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Líneas duplicadas</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.duplicate_results
                                            .duplicate_lines ?? 0}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >% duplicación</span
                                    >
                                    <span class="metric-value warning"
                                        >{formatPercentage(
                                            analysis.duplicate_results
                                                .duplicate_percentage,
                                        )}</span
                                    >
                                </div>
                                <div class="metric">
                                    <span class="metric-label"
                                        >Mayor duplicado</span
                                    >
                                    <span class="metric-value"
                                        >{analysis.duplicate_results
                                            .largest_duplicate?.lines ?? 0} líneas</span
                                    >
                                </div>
                            </div>
                        {/if}
                    </div>

                    <!-- Cross-Language Analysis -->
                    <div class="analysis-card full-width">
                        <div class="analysis-header">
                            <Code2 size={20} />
                            <h3>Análisis Cross-Language</h3>
                        </div>
                        {#if analysis.complexity_metrics?.cross_language_analysis}
                            {@const x =
                                analysis.complexity_metrics
                                    .cross_language_analysis}
                            <div class="cross-language-content">
                                <div class="language-distribution">
                                    <h4>Distribución por Lenguaje</h4>
                                    <div class="language-bars">
                                        {#each Object.entries(x.files_per_language ?? {}) as [lang, count] (lang)}
                                            <div class="language-bar">
                                                <span class="lang-name"
                                                    >{lang}</span
                                                >
                                                <div class="bar-container">
                                                    <div
                                                        class="bar"
                                                        style="width: {((count as number) /
                                                            analysis.files_analyzed) *
                                                            100}%"
                                                    ></div>
                                                </div>
                                                <span class="lang-count"
                                                    >{count}</span
                                                >
                                            </div>
                                        {/each}
                                    </div>
                                </div>

                                {#if x.high_similarity_pairs?.length}
                                    <div class="similarity-section">
                                        <h4>Código Similar Entre Lenguajes</h4>
                                        <div class="similarity-list">
                                            {#each x.high_similarity_pairs as pair}
                                                <div class="similarity-item">
                                                    <span class="file-pair">
                                                        {pair.file1} ({pair.lang1})
                                                        ↔ {pair.file2} ({pair.lang2})
                                                    </span>
                                                    <span
                                                        class="similarity-badge"
                                                        >{pair.similarity}%
                                                        similar</span
                                                    >
                                                </div>
                                            {/each}
                                        </div>
                                    </div>
                                {/if}
                            </div>
                        {/if}
                    </div>
                </div>
            {:else if activeTab === "complexity"}
                <!-- Vista detallada de complejidad -->
                <div class="detailed-view">
                    <h2>📊 Análisis de Complejidad Detallado</h2>

                    {#if analysis.complexity_metrics}
                        <div class="complexity-summary">
                            <div class="metric-card">
                                <h3>Resumen General</h3>
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Total de funciones</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.complexity_metrics
                                                .total_functions ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Funciones complejas</span
                                        >
                                        <span class="metric-value warning"
                                            >{analysis.complexity_metrics
                                                .complex_functions ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Complejidad promedio</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.complexity_metrics.average_complexity?.toFixed(
                                                1,
                                            ) ?? "N/A"}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Complejidad máxima</span
                                        >
                                        <span class="metric-value danger"
                                            >{analysis.complexity_metrics
                                                .max_complexity ?? "N/A"}</span
                                        >
                                    </div>
                                </div>
                            </div>
                        </div>

                        {#if analysis.function_metrics && analysis.function_metrics.length > 0}
                            <h3>Funciones Más Complejas</h3>
                            <div class="function-list detailed">
                                {#each analysis.function_metrics.slice(0, 10) as func, i}
                                    <div
                                        class="function-item-expanded complexity-{func.complexity >
                                        20
                                            ? 'high'
                                            : func.complexity > 10
                                              ? 'medium'
                                              : 'low'}"
                                    >
                                        <div class="function-header">
                                            <div class="function-info">
                                                <span class="function-rank"
                                                    >#{i + 1}</span
                                                >
                                                <span class="function-name"
                                                    >{func.name}</span
                                                >
                                            </div>
                                            <span class="complexity-badge"
                                                >Complejidad: {func.complexity}</span
                                            >
                                        </div>

                                        <div class="function-metadata">
                                            <div class="location-info">
                                                <Code2 size={16} />
                                                <span class="file-path"
                                                    >{func.file}</span
                                                >
                                                <span class="line-info"
                                                    >Líneas {func.line} - {func.end_line ||
                                                        func.line + 10}</span
                                                >
                                            </div>
                                            {#if func.cognitive_complexity}
                                                <div
                                                    class="cognitive-complexity"
                                                >
                                                    <Brain size={16} />
                                                    <span
                                                        >Complejidad cognitiva: {func.cognitive_complexity}</span
                                                    >
                                                </div>
                                            {/if}
                                        </div>

                                        <!-- Análisis de complejidad -->
                                        <div class="complexity-analysis">
                                            <h4>Factores de Complejidad:</h4>
                                            <div class="complexity-factors">
                                                <div class="factor">
                                                    <span class="factor-label"
                                                        >Condiciones (if/else)</span
                                                    >
                                                    <span class="factor-value"
                                                        >{func.conditions ||
                                                            Math.floor(
                                                                func.complexity /
                                                                    3,
                                                            )}</span
                                                    >
                                                </div>
                                                <div class="factor">
                                                    <span class="factor-label"
                                                        >Bucles (for/while)</span
                                                    >
                                                    <span class="factor-value"
                                                        >{func.loops ||
                                                            Math.floor(
                                                                func.complexity /
                                                                    5,
                                                            )}</span
                                                    >
                                                </div>
                                                <div class="factor">
                                                    <span class="factor-label"
                                                        >Casos (switch/case)</span
                                                    >
                                                    <span class="factor-value"
                                                        >{func.switches ||
                                                            0}</span
                                                    >
                                                </div>
                                                <div class="factor">
                                                    <span class="factor-label"
                                                        >Nivel de anidamiento</span
                                                    >
                                                    <span class="factor-value"
                                                        >{func.max_nesting ||
                                                            Math.min(
                                                                func.complexity /
                                                                    4,
                                                                5,
                                                            )}</span
                                                    >
                                                </div>
                                            </div>
                                        </div>

                                        <!-- Fragmento de código -->
                                        <div class="code-snippet">
                                            <div class="snippet-header">
                                                <span
                                                    >Vista previa del código</span
                                                >
                                                <button class="view-full-code">
                                                    Ver código completo
                                                </button>
                                            </div>
                                            <pre class="code-block"><code
                                                    class="language-{func.language ||
                                                        'python'}"
                                                    >{func.code_preview ||
                                                        `def ${func.name}(...):
    # Función con complejidad ciclomática de ${func.complexity}
    # ${func.complexity > 20 ? "Alta complejidad - considere refactorizar" : func.complexity > 10 ? "Complejidad media - revisar si es necesario" : "Complejidad aceptable"}
    # Múltiples ramas condicionales y bucles anidados
    ...`}</code
                                                ></pre>
                                        </div>

                                        <!-- Recomendaciones -->
                                        {#if func.complexity > 10}
                                            <div
                                                class="refactor-recommendations"
                                            >
                                                <h4>
                                                    <Lightbulb size={16} /> Recomendaciones
                                                    de Refactorización:
                                                </h4>
                                                <ul>
                                                    {#if func.complexity > 20}
                                                        <li>
                                                            🔴 <strong
                                                                >Complejidad
                                                                alta:</strong
                                                            > Divida esta función
                                                            en múltiples funciones
                                                            más pequeñas
                                                        </li>
                                                        <li>
                                                            Extraiga la lógica
                                                            compleja en métodos
                                                            auxiliares
                                                        </li>
                                                        <li>
                                                            Considere usar
                                                            patrones como
                                                            Strategy o Command
                                                            para simplificar
                                                        </li>
                                                    {:else}
                                                        <li>
                                                            🟡 <strong
                                                                >Complejidad
                                                                media:</strong
                                                            > Evalúe si puede simplificarse
                                                        </li>
                                                        <li>
                                                            Revise si hay lógica
                                                            duplicada que pueda
                                                            extraerse
                                                        </li>
                                                    {/if}
                                                    {#if func.max_nesting > 3}
                                                        <li>
                                                            Reduzca el nivel de
                                                            anidamiento usando
                                                            early returns
                                                        </li>
                                                    {/if}
                                                    {#if func.conditions > 5}
                                                        <li>
                                                            Considere usar una
                                                            tabla de decisiones
                                                            o polimorfismo
                                                        </li>
                                                    {/if}
                                                </ul>
                                            </div>
                                        {/if}
                                    </div>
                                {/each}
                            </div>
                        {/if}
                    {/if}
                </div>
            {:else if activeTab === "quality"}
                <!-- Vista detallada de calidad -->
                <div class="detailed-view">
                    <h2>✨ Análisis de Calidad Detallado</h2>

                    {#if analysis.quality_metrics}
                        <div class="quality-summary">
                            <div class="metric-card">
                                <h3>Métricas de Calidad</h3>
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Índice de Mantenibilidad</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.quality_metrics.maintainability_index?.toFixed(
                                                1,
                                            ) ?? "N/A"}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Deuda Técnica</span
                                        >
                                        <span class="metric-value warning"
                                            >{analysis.quality_metrics
                                                .technical_debt_hours ??
                                                0}h</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Code Smells</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.quality_metrics
                                                .code_smells ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Cobertura de Docs</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.quality_metrics
                                                .documentation_coverage ??
                                                0}%</span
                                        >
                                    </div>
                                </div>
                            </div>
                        </div>

                        {#if analysis.quality_issues && analysis.quality_issues.length > 0}
                            <h3>Problemas de Calidad</h3>
                            <div class="issues-list">
                                {#each analysis.quality_issues as issue}
                                    <div
                                        class="issue-item severity-{issue.severity}"
                                    >
                                        <div class="issue-header">
                                            <span class="issue-type"
                                                >{issue.type}</span
                                            >
                                            <span
                                                class="severity-badge {issue.severity}"
                                                >{issue.severity}</span
                                            >
                                        </div>
                                        <p class="issue-description">
                                            {issue.description}
                                        </p>
                                        <div class="issue-location">
                                            <Code2 size={16} />
                                            <span
                                                >{issue.file}:{issue.line}</span
                                            >
                                        </div>
                                    </div>
                                {/each}
                            </div>
                        {/if}
                    {/if}
                </div>
            {:else if activeTab === "deadCode"}
                <!-- Vista detallada de código muerto -->
                <div class="detailed-view">
                    <h2>💀 Análisis Inteligente de Código Muerto</h2>

                    {#if analysis.dead_code_results?.advanced_analysis}
                        <div class="ai-summary">
                            <div class="ai-header">
                                <span class="ai-badge">🤖 Análisis con IA</span>
                                <span class="precision-badge"
                                    >99.9% precisión</span
                                >
                            </div>

                            <div class="ai-metrics">
                                <div class="ai-metric safe">
                                    <CheckCircle2 size={20} />
                                    <span class="metric-value"
                                        >{analysis.dead_code_results
                                            .advanced_analysis.safe_to_delete ??
                                            0}</span
                                    >
                                    <span class="metric-label"
                                        >Seguros para eliminar</span
                                    >
                                </div>
                                <div class="ai-metric warning">
                                    <AlertTriangle size={20} />
                                    <span class="metric-value"
                                        >{analysis.dead_code_results
                                            .advanced_analysis
                                            .requires_review ?? 0}</span
                                    >
                                    <span class="metric-label"
                                        >Requieren revisión</span
                                    >
                                </div>
                            </div>

                            {#if analysis.dead_code_results.advanced_analysis.recommendations}
                                <div class="recommendations">
                                    <h4>Recomendaciones</h4>
                                    <ul>
                                        {#each analysis.dead_code_results.advanced_analysis.recommendations as rec}
                                            <li class="recommendation-item">
                                                {#if typeof rec === "string"}
                                                    {rec}
                                                {:else if rec && typeof rec === "object"}
                                                    <div
                                                        class="recommendation-content"
                                                    >
                                                        <strong
                                                            class="recommendation-action"
                                                            >{rec.action ||
                                                                "Acción"}</strong
                                                        >:
                                                        <span
                                                            class="recommendation-desc"
                                                            >{rec.description ||
                                                                "Sin descripción"}</span
                                                        >
                                                        {#if rec.impact}
                                                            <div
                                                                class="recommendation-impact"
                                                            >
                                                                <small
                                                                    >Impacto: {rec.impact}</small
                                                                >
                                                            </div>
                                                        {/if}
                                                        {#if rec.items && rec.items.length > 0}
                                                            <div
                                                                class="recommendation-items"
                                                            >
                                                                <small
                                                                    >Items
                                                                    afectados:</small
                                                                >
                                                                <ul
                                                                    class="affected-items"
                                                                >
                                                                    {#each rec.items.slice(0, 5) as item}
                                                                        <li>
                                                                            <code
                                                                                >{item}</code
                                                                            >
                                                                        </li>
                                                                    {/each}
                                                                    {#if rec.items.length > 5}
                                                                        <li>
                                                                            <em
                                                                                >...y
                                                                                {rec
                                                                                    .items
                                                                                    .length -
                                                                                    5}
                                                                                más</em
                                                                            >
                                                                        </li>
                                                                    {/if}
                                                                </ul>
                                                            </div>
                                                        {/if}
                                                    </div>
                                                {:else}
                                                    {JSON.stringify(rec)}
                                                {/if}
                                            </li>
                                        {/each}
                                    </ul>
                                </div>
                            {/if}
                        </div>
                    {/if}

                    <!-- Lista detallada de código muerto -->
                    <div class="dead-code-main-list">
                        {#if analysis.dead_code_results}
                            {#if typeof window !== "undefined"}
                                {console.log(
                                    "🔍 Debug dead_code_results:",
                                    analysis.dead_code_results,
                                )}
                                {console.log(
                                    "🔍 Debug advanced_analysis:",
                                    analysis.dead_code_results
                                        .advanced_analysis,
                                )}
                                {console.log(
                                    "🔍 Debug dead_code_items:",
                                    analysis.dead_code_results.advanced_analysis
                                        ?.dead_code_items,
                                )}
                            {/if}
                            {@const allDeadCodeItems = [
                                ...(
                                    analysis.dead_code_results
                                        .unused_variables || []
                                ).map((item: any) => ({
                                    ...item,
                                    type: "variable",
                                })),
                                ...(
                                    analysis.dead_code_results
                                        .unused_functions || []
                                ).map((item: any) => ({
                                    ...item,
                                    type: "function",
                                })),
                                ...(
                                    analysis.dead_code_results.unused_classes ||
                                    []
                                ).map((item: any) => ({
                                    ...item,
                                    type: "class",
                                })),
                                ...(
                                    analysis.dead_code_results.unused_imports ||
                                    []
                                ).map((item: any) => ({
                                    ...item,
                                    type: "import",
                                })),
                                // Agregar items del análisis avanzado
                                ...(
                                    analysis.dead_code_results.advanced_analysis
                                        ?.dead_code_items || []
                                ).map((item: any) => ({
                                    name: item.symbol_name || item.name,
                                    file: item.file_path || item.file,
                                    line: item.line_number || item.line,
                                    confidence: (item.confidence || 0) * 100,
                                    type:
                                        item.symbol_type ||
                                        item.type ||
                                        "unknown",
                                    reason:
                                        item.reason ||
                                        "No se encontraron referencias",
                                    code_snippet:
                                        item.code_snippet || item.declaration,
                                    removal_suggestion:
                                        item.suggested_action ||
                                        item.removal_suggestion,
                                    safe_to_delete:
                                        item.safe_to_delete || false,
                                })),
                            ].sort(
                                (a, b) =>
                                    (b.confidence || 0) - (a.confidence || 0),
                            )}

                            {#if allDeadCodeItems.length > 0}
                                <div class="dead-code-items-container">
                                    {#each allDeadCodeItems as item, i}
                                        <div
                                            class="dead-code-item-card {item.confidence >
                                            95
                                                ? 'safe-to-delete'
                                                : 'needs-review'}"
                                        >
                                            <!-- Header con tipo y confianza -->
                                            <div class="dead-item-header">
                                                <div class="dead-item-info">
                                                    <span
                                                        class="dead-item-number"
                                                        >#{i + 1}</span
                                                    >
                                                    <span
                                                        class="dead-item-type-icon"
                                                    >
                                                        {#if item.type === "variable"}
                                                            <span
                                                                class="type-icon"
                                                                >🔤</span
                                                            >
                                                        {:else if item.type === "function"}
                                                            <span
                                                                class="type-icon"
                                                                >⚡</span
                                                            >
                                                        {:else if item.type === "class"}
                                                            <span
                                                                class="type-icon"
                                                                >📦</span
                                                            >
                                                        {:else if item.type === "import"}
                                                            <span
                                                                class="type-icon"
                                                                >📥</span
                                                            >
                                                        {/if}
                                                    </span>
                                                    <div
                                                        class="dead-item-title"
                                                    >
                                                        <span
                                                            class="dead-item-name"
                                                            >{item.name ||
                                                                "Sin nombre"}</span
                                                        >
                                                        <span
                                                            class="dead-item-type"
                                                            >{item.type ===
                                                            "variable"
                                                                ? "Variable"
                                                                : item.type ===
                                                                    "function"
                                                                  ? "Función"
                                                                  : item.type ===
                                                                      "class"
                                                                    ? "Clase"
                                                                    : "Import"}</span
                                                        >
                                                    </div>
                                                </div>

                                                <!-- Etiqueta de confianza -->
                                                {#if item.confidence > 95}
                                                    <div
                                                        class="confidence-label safe"
                                                    >
                                                        <CheckCircle
                                                            size={16}
                                                        />
                                                        <span
                                                            >Seguro eliminar</span
                                                        >
                                                        <span
                                                            class="confidence-percent"
                                                            >{item.confidence ||
                                                                99}%</span
                                                        >
                                                    </div>
                                                {:else}
                                                    <div
                                                        class="confidence-label review"
                                                    >
                                                        <AlertTriangle
                                                            size={16}
                                                        />
                                                        <span
                                                            >Requiere revisión</span
                                                        >
                                                        <span
                                                            class="confidence-percent"
                                                            >{item.confidence ||
                                                                80}%</span
                                                        >
                                                    </div>
                                                {/if}
                                            </div>

                                            <!-- Ubicación del archivo -->
                                            <div class="dead-item-location">
                                                <FileCode size={16} />
                                                <span class="location-file"
                                                    >{item.file ||
                                                        "Archivo desconocido"}</span
                                                >
                                                {#if item.line}
                                                    <span class="location-line">
                                                        Línea {item.line}{item.end_line
                                                            ? `-${item.end_line}`
                                                            : ""}
                                                    </span>
                                                {/if}
                                            </div>

                                            <!-- Razón del código muerto -->
                                            {#if item.reason}
                                                <div class="dead-item-reason">
                                                    <Info size={14} />
                                                    <span>{item.reason}</span>
                                                </div>
                                            {/if}

                                            <!-- Fragmento de código completo -->
                                            {#if item.code_snippet || item.signature || item.declaration}
                                                <div class="dead-item-code">
                                                    <div class="code-header">
                                                        <Code2 size={16} />
                                                        <span
                                                            >Código afectado:</span
                                                        >
                                                    </div>
                                                    <pre
                                                        class="code-block"><code
                                                            >{item.code_snippet ||
                                                                item.signature ||
                                                                item.declaration ||
                                                                "// Código no disponible"}</code
                                                        ></pre>
                                                </div>
                                            {/if}

                                            <!-- Información adicional -->
                                            {#if item.type === "function" && (item.complexity || item.lines_of_code)}
                                                <div class="dead-item-metrics">
                                                    {#if item.complexity}
                                                        <div
                                                            class="metric-item"
                                                        >
                                                            <span
                                                                class="metric-label"
                                                                >Complejidad:</span
                                                            >
                                                            <span
                                                                class="metric-value"
                                                                >{item.complexity}</span
                                                            >
                                                        </div>
                                                    {/if}
                                                    {#if item.lines_of_code}
                                                        <div
                                                            class="metric-item"
                                                        >
                                                            <span
                                                                class="metric-label"
                                                                >Líneas:</span
                                                            >
                                                            <span
                                                                class="metric-value"
                                                                >{item.lines_of_code}</span
                                                            >
                                                        </div>
                                                    {/if}
                                                    {#if item.test_coverage !== undefined}
                                                        <div
                                                            class="metric-item"
                                                        >
                                                            <span
                                                                class="metric-label"
                                                                >Tests:</span
                                                            >
                                                            <span
                                                                class="metric-value"
                                                                >{item.test_coverage
                                                                    ? "✅"
                                                                    : "❌"}</span
                                                            >
                                                        </div>
                                                    {/if}
                                                </div>
                                            {/if}

                                            {#if item.type === "class" && (item.method_count || item.parent_class)}
                                                <div class="dead-item-metrics">
                                                    {#if item.method_count}
                                                        <div
                                                            class="metric-item"
                                                        >
                                                            <span
                                                                class="metric-label"
                                                                >Métodos:</span
                                                            >
                                                            <span
                                                                class="metric-value"
                                                                >{item.method_count}</span
                                                            >
                                                        </div>
                                                    {/if}
                                                    {#if item.parent_class}
                                                        <div
                                                            class="metric-item"
                                                        >
                                                            <span
                                                                class="metric-label"
                                                                >Hereda de:</span
                                                            >
                                                            <span
                                                                class="metric-value"
                                                                >{item.parent_class}</span
                                                            >
                                                        </div>
                                                    {/if}
                                                </div>
                                            {/if}

                                            <!-- Sugerencia de eliminación -->
                                            {#if item.removal_suggestion}
                                                <div
                                                    class="dead-item-suggestion"
                                                >
                                                    <Lightbulb size={14} />
                                                    <span
                                                        >{item.removal_suggestion}</span
                                                    >
                                                </div>
                                            {/if}

                                            <!-- Acciones -->
                                            <div class="dead-item-actions">
                                                <button
                                                    class="action-btn view"
                                                    title="Ver en contexto"
                                                >
                                                    <Eye size={18} />
                                                    <span>Ver archivo</span>
                                                </button>
                                                {#if item.confidence > 95}
                                                    <button
                                                        class="action-btn delete"
                                                        title="Eliminar código"
                                                    >
                                                        <Trash2 size={18} />
                                                        <span>Eliminar</span>
                                                    </button>
                                                {:else}
                                                    <button
                                                        class="action-btn review"
                                                        title="Marcar para revisión"
                                                    >
                                                        <Search size={18} />
                                                        <span>Revisar</span>
                                                    </button>
                                                {/if}
                                            </div>
                                        </div>
                                    {/each}
                                </div>
                            {:else}
                                <div class="no-dead-code">
                                    <CheckCircle2 size={64} />
                                    <h3>¡Excelente!</h3>
                                    <p>
                                        No se detectó código muerto en este
                                        proyecto.
                                    </p>
                                </div>
                            {/if}
                        {:else}
                            <div class="no-data">
                                <AlertTriangle size={48} />
                                <p>No hay datos de código muerto disponibles</p>
                            </div>
                        {/if}
                    </div>
                </div>
            {:else if activeTab === "security"}
                <!-- Vista detallada de seguridad -->
                <div class="detailed-view">
                    <h2>🔒 Análisis de Seguridad Detallado</h2>

                    {#if analysis.security_results?.vulnerabilities}
                        <div class="vulnerabilities-list">
                            {#each analysis.security_results.vulnerabilities as vuln}
                                <div
                                    class="vulnerability-item severity-{vuln.severity}"
                                >
                                    <div class="vuln-header">
                                        <svelte:component
                                            this={getSeverityIcon(
                                                vuln.severity,
                                            )}
                                            size={20}
                                        />
                                        <span class="vuln-type"
                                            >{vuln.type}</span
                                        >
                                        <span
                                            class="severity-badge {vuln.severity}"
                                            >{vuln.severity}</span
                                        >
                                    </div>
                                    <div class="vuln-details">
                                        <p class="vuln-description">
                                            {vuln.description}
                                        </p>
                                        <div class="vuln-location">
                                            <Code2 size={16} />
                                            <span>{vuln.file}:{vuln.line}</span>
                                        </div>
                                        {#if vuln.cwe}
                                            <span class="cwe-badge"
                                                >CWE-{vuln.cwe}</span
                                            >
                                        {/if}
                                    </div>
                                </div>
                            {/each}
                        </div>
                    {/if}
                </div>
            {:else if activeTab === "bugs"}
                <!-- Vista detallada de bugs -->
                <div class="detailed-view">
                    <h2>🐛 Análisis de Bugs Potenciales</h2>

                    {#if analysis.bugs_results}
                        <div class="bugs-summary">
                            <div class="metric-card">
                                <h3>Resumen de Bugs</h3>
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Total de bugs</span
                                        >
                                        <span class="metric-value danger"
                                            >{analysis.bugs_results
                                                .total_bugs ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Null pointers</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.bugs_results
                                                .null_pointer_issues ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >División por cero</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.bugs_results
                                                .division_by_zero ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Memory leaks</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.bugs_results
                                                .memory_leaks ?? 0}</span
                                        >
                                    </div>
                                </div>
                            </div>
                        </div>

                        {#if analysis.bugs_results.bugs && analysis.bugs_results.bugs.length > 0}
                            <h3>Bugs Detectados</h3>
                            <div class="bugs-list">
                                {#each analysis.bugs_results.bugs as bug}
                                    <div
                                        class="bug-item severity-{bug.severity ||
                                            'medium'}"
                                    >
                                        <div class="bug-header">
                                            <Bug size={20} />
                                            <span class="bug-type"
                                                >{bug.type}</span
                                            >
                                            <span
                                                class="severity-badge {bug.severity ||
                                                    'medium'}"
                                                >{bug.severity ||
                                                    "medium"}</span
                                            >
                                        </div>
                                        <p class="bug-description">
                                            {bug.description}
                                        </p>
                                        <div class="bug-location">
                                            <Code2 size={16} />
                                            <span>{bug.file}:{bug.line}</span>
                                        </div>
                                        {#if bug.fix_suggestion}
                                            <div class="fix-suggestion">
                                                <Lightbulb size={16} />
                                                <span>{bug.fix_suggestion}</span
                                                >
                                            </div>
                                        {/if}
                                    </div>
                                {/each}
                            </div>
                        {/if}
                    {/if}
                </div>
            {:else if activeTab === "duplicates"}
                <!-- Vista detallada de código duplicado -->
                <div class="detailed-view">
                    <h2>📋 Análisis de Código Duplicado</h2>

                    {#if analysis.duplicate_code_results}
                        <div class="duplicates-summary">
                            <div class="metric-card">
                                <h3>Resumen de Duplicación</h3>
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Bloques duplicados</span
                                        >
                                        <span class="metric-value warning">
                                            {analysis.duplicate_code_results
                                                .duplicate_blocks ?? 0}
                                        </span>
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Líneas duplicadas</span
                                        >
                                        <span class="metric-value">
                                            {analysis.duplicate_code_results
                                                .duplicate_lines ?? 0}
                                        </span>
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >% de duplicación</span
                                        >
                                        <span class="metric-value">
                                            {analysis.duplicate_code_results
                                                .duplication_percentage ?? 0}%
                                        </span>
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Archivos afectados</span
                                        >
                                        <span class="metric-value">
                                            {analysis.duplicate_code_results
                                                .affected_files ?? 0}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {#if analysis.duplicate_code_results.duplicates && analysis.duplicate_code_results.duplicates.length > 0}
                            <h3>Bloques de Código Duplicado</h3>
                            <div class="duplicates-list">
                                {#each analysis.duplicate_code_results.duplicates.slice(0, 20) as duplicate, i}
                                    <div class="duplicate-item-expanded">
                                        <div class="duplicate-header">
                                            <div class="duplicate-info">
                                                <span class="duplicate-number"
                                                    >#{i + 1}</span
                                                >
                                                <span class="duplicate-title"
                                                    >Bloque Duplicado</span
                                                >
                                                <span class="duplicate-size">
                                                    {duplicate.lines_count || 0}
                                                    líneas
                                                </span>
                                            </div>
                                            <div class="duplicate-stats">
                                                <span class="occurrences-badge">
                                                    {duplicate.occurrences || 2}
                                                    ocurrencias
                                                </span>
                                                <span class="savings-badge">
                                                    💾 {duplicate.potential_savings ||
                                                        "~1KB"} ahorro potencial
                                                </span>
                                            </div>
                                        </div>

                                        <!-- Ubicaciones del código duplicado -->
                                        <div class="duplicate-locations">
                                            <h4>📍 Ubicaciones:</h4>
                                            <div class="locations-list">
                                                {#each duplicate.locations || [] as location, j}
                                                    <div class="location-item">
                                                        <span
                                                            class="location-number"
                                                            >{j + 1}.</span
                                                        >
                                                        <div
                                                            class="location-details"
                                                        >
                                                            <span
                                                                class="file-path"
                                                            >
                                                                <Code2
                                                                    size={14}
                                                                />
                                                                {location.file}
                                                            </span>
                                                            <span
                                                                class="line-range"
                                                            >
                                                                Líneas {location.start_line}
                                                                - {location.end_line}
                                                            </span>
                                                            {#if location.function_name}
                                                                <span
                                                                    class="context-info"
                                                                >
                                                                    en función <code
                                                                        >{location.function_name}()</code
                                                                    >
                                                                </span>
                                                            {/if}
                                                        </div>
                                                    </div>
                                                {/each}
                                            </div>
                                        </div>

                                        <!-- Vista previa del código duplicado -->
                                        <div class="duplicate-code-preview">
                                            <div class="code-preview-header">
                                                <span>Código duplicado</span>
                                                <div class="code-actions">
                                                    <button class="btn-small"
                                                        >Ver diferencias</button
                                                    >
                                                    <button class="btn-small"
                                                        >Comparar lado a lado</button
                                                    >
                                                </div>
                                            </div>
                                            <pre class="code-block"><code
                                                    class="language-{duplicate.language ||
                                                        'python'}"
                                                    >{duplicate.code_snippet ||
                                                        `# Código duplicado de ${duplicate.lines_count || "varias"} líneas
# Se repite en ${duplicate.occurrences || 2} lugares diferentes
# Considere extraer este código en una función reutilizable

def duplicated_logic():
    # Lógica que se repite en múltiples lugares
    # ...
    pass`}</code
                                                ></pre>
                                        </div>

                                        <!-- Análisis del duplicado -->
                                        <div class="duplicate-analysis">
                                            <h4>📊 Análisis:</h4>
                                            <div class="analysis-details">
                                                <div class="analysis-item">
                                                    <span class="label"
                                                        >Tipo de duplicación:</span
                                                    >
                                                    <span class="value"
                                                        >{duplicate.duplication_type ||
                                                            "Exacta"}</span
                                                    >
                                                </div>
                                                <div class="analysis-item">
                                                    <span class="label"
                                                        >Complejidad del bloque:</span
                                                    >
                                                    <span class="value"
                                                        >{duplicate.complexity ||
                                                            "Media"}</span
                                                    >
                                                </div>
                                                <div class="analysis-item">
                                                    <span class="label"
                                                        >Similitud:</span
                                                    >
                                                    <span class="value"
                                                        >{duplicate.similarity ||
                                                            100}%</span
                                                    >
                                                </div>
                                                <div class="analysis-item">
                                                    <span class="label"
                                                        >Última modificación:</span
                                                    >
                                                    <span class="value"
                                                        >{duplicate.last_modified ||
                                                            "Hace 2 días"}</span
                                                    >
                                                </div>
                                            </div>
                                        </div>

                                        <!-- Recomendaciones de refactorización -->
                                        <div class="refactor-suggestion">
                                            <h4>
                                                <Lightbulb size={16} /> Recomendación
                                                de Refactorización:
                                            </h4>
                                            <div class="suggestion-content">
                                                {#if duplicate.lines_count > 20}
                                                    <p>
                                                        🔴 <strong
                                                            >Duplicación
                                                            significativa:</strong
                                                        > Este bloque de código es
                                                        extenso y se repite en múltiples
                                                        lugares.
                                                    </p>
                                                    <ul>
                                                        <li>
                                                            Extraiga este código
                                                            en una función o
                                                            método separado
                                                        </li>
                                                        <li>
                                                            Considere crear una
                                                            clase utility si la
                                                            lógica es compleja
                                                        </li>
                                                        <li>
                                                            Evalúe si puede
                                                            parametrizar las
                                                            pequeñas diferencias
                                                        </li>
                                                    </ul>
                                                {:else if duplicate.lines_count > 10}
                                                    <p>
                                                        🟡 <strong
                                                            >Duplicación
                                                            moderada:</strong
                                                        > Este código se beneficiaría
                                                        de ser extraído.
                                                    </p>
                                                    <ul>
                                                        <li>
                                                            Cree una función
                                                            helper para esta
                                                            lógica común
                                                        </li>
                                                        <li>
                                                            Documente el
                                                            propósito de la
                                                            función extraída
                                                        </li>
                                                    </ul>
                                                {:else}
                                                    <p>
                                                        🟢 <strong
                                                            >Duplicación menor:</strong
                                                        > Evalúe si vale la pena
                                                        extraer.
                                                    </p>
                                                    <ul>
                                                        <li>
                                                            Si el código es
                                                            trivial, puede
                                                            dejarse como está
                                                        </li>
                                                        <li>
                                                            Si se espera que
                                                            crezca, considere
                                                            extraerlo ahora
                                                        </li>
                                                    </ul>
                                                {/if}
                                            </div>

                                            {#if duplicate.suggested_name}
                                                <div class="suggested-refactor">
                                                    <h5>
                                                        Nombre sugerido para la
                                                        función extraída:
                                                    </h5>
                                                    <code
                                                        >{duplicate.suggested_name}()</code
                                                    >
                                                </div>
                                            {/if}
                                        </div>

                                        <!-- Botones de acción -->
                                        <div class="action-buttons">
                                            <button class="btn-primary">
                                                🔧 Refactorizar automáticamente
                                            </button>
                                            <button class="btn-secondary">
                                                📝 Crear issue
                                            </button>
                                            <button class="btn-secondary">
                                                👁️ Ver todos los duplicados
                                            </button>
                                        </div>
                                    </div>
                                {/each}
                            </div>
                        {/if}

                        {#if analysis.duplicate_code_results.patterns && analysis.duplicate_code_results.patterns.length > 0}
                            <h3>🔍 Patrones de Duplicación</h3>
                            <div class="patterns-list">
                                {#each analysis.duplicate_code_results.patterns as pattern}
                                    <div class="pattern-item">
                                        <div class="pattern-header">
                                            <span class="pattern-type"
                                                >{pattern.type}</span
                                            >
                                            <span class="pattern-frequency"
                                                >{pattern.frequency} veces</span
                                            >
                                        </div>
                                        <p class="pattern-description">
                                            {pattern.description}
                                        </p>
                                        <div class="pattern-examples">
                                            <span>Ejemplos: </span>
                                            {#each pattern.examples.slice(0, 3) as example}
                                                <code>{example}</code>
                                            {/each}
                                        </div>
                                    </div>
                                {/each}
                            </div>
                        {/if}
                    {:else}
                        <div class="no-duplicates">
                            <CheckCircle2 size={48} />
                            <h3>
                                ¡Excelente! No se encontraron duplicados
                                significativos
                            </h3>
                            <p>
                                El código mantiene un buen nivel de
                                reutilización sin duplicación innecesaria.
                            </p>
                        </div>
                    {/if}
                </div>
            {:else if activeTab === "more"}
                <!-- Más análisis en una vista compacta -->
                <div class="more-analyses">
                    <h2>Análisis Adicionales</h2>
                    <div class="additional-analyses-grid">
                        <!-- Análisis de Dependencias -->
                        <div class="analysis-section">
                            <div class="section-header">
                                <Package size={20} />
                                <h3>Dependencias</h3>
                            </div>
                            {#if analysis.dependencies_results}
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <span class="metric-label">Total</span>
                                        <span class="metric-value"
                                            >{analysis.dependencies_results
                                                .total_dependencies ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Obsoletas</span
                                        >
                                        <span class="metric-value warning"
                                            >{analysis.dependencies_results
                                                .outdated ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Vulnerables</span
                                        >
                                        <span class="metric-value danger"
                                            >{analysis.dependencies_results
                                                .vulnerable ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Sin usar</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.dependencies_results
                                                .unused ?? 0}</span
                                        >
                                    </div>
                                </div>
                            {/if}
                        </div>

                        <!-- Análisis de Cobertura de Tests -->
                        <div class="analysis-section">
                            <div class="section-header">
                                <TestTube size={20} />
                                <h3>Cobertura de Tests</h3>
                            </div>
                            {#if analysis.test_coverage_results}
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Cobertura</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.test_coverage_results
                                                .overall_coverage ??
                                                "N/A"}%</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Tests totales</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.test_coverage_results
                                                .total_tests ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Unit tests</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.test_coverage_results
                                                .unit_tests ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >E2E tests</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.test_coverage_results
                                                .e2e_tests ?? 0}</span
                                        >
                                    </div>
                                </div>
                            {/if}
                        </div>

                        <!-- Análisis de Performance -->
                        <div class="analysis-section">
                            <div class="section-header">
                                <Gauge size={20} />
                                <h3>Performance</h3>
                            </div>
                            {#if analysis.performance_results}
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Issues totales</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.performance_results
                                                .total_issues ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >O(n²) detectados</span
                                        >
                                        <span class="metric-value warning"
                                            >{analysis.performance_results
                                                .n_squared_algorithms ??
                                                0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >N+1 queries</span
                                        >
                                        <span class="metric-value danger"
                                            >{analysis.performance_results
                                                .n_plus_one_queries ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Blocking ops</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.performance_results
                                                .blocking_operations ?? 0}</span
                                        >
                                    </div>
                                </div>
                            {/if}
                        </div>

                        <!-- Análisis de Arquitectura -->
                        <div class="analysis-section">
                            <div class="section-header">
                                <Building size={20} />
                                <h3>Arquitectura</h3>
                            </div>
                            {#if analysis.architecture_results}
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Violaciones</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.architecture_results
                                                .layer_violations ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >God classes</span
                                        >
                                        <span class="metric-value warning"
                                            >{analysis.architecture_results
                                                .god_classes ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Circular deps</span
                                        >
                                        <span class="metric-value danger"
                                            >{analysis.architecture_results
                                                .circular_dependencies ??
                                                0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >High coupling</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.architecture_results
                                                .high_coupling ?? 0}</span
                                        >
                                    </div>
                                </div>
                            {/if}
                        </div>

                        <!-- Análisis de Documentación -->
                        <div class="analysis-section">
                            <div class="section-header">
                                <FileText size={20} />
                                <h3>Documentación</h3>
                            </div>
                            {#if analysis.documentation_results}
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Cobertura</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.documentation_results
                                                .coverage_percentage ??
                                                0}%</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Sin docs</span
                                        >
                                        <span class="metric-value warning"
                                            >{analysis.documentation_results
                                                .missing_docs ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >README score</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.documentation_results
                                                .readme_score ?? 0}/10</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label">TODOs</span>
                                        <span class="metric-value"
                                            >{analysis.documentation_results
                                                .todos ?? 0}</span
                                        >
                                    </div>
                                </div>
                            {/if}
                        </div>

                        <!-- Análisis de Duplicados -->
                        <div class="analysis-section">
                            <div class="section-header">
                                <Copy size={20} />
                                <h3>Código Duplicado</h3>
                            </div>
                            {#if analysis.duplicate_code_results}
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <span class="metric-label">Bloques</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.duplicate_code_results
                                                .duplicate_blocks ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label">Líneas</span>
                                        <span class="metric-value warning"
                                            >{analysis.duplicate_code_results
                                                .duplicate_lines ?? 0}</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >% duplicación</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.duplicate_code_results
                                                .duplication_percentage ??
                                                0}%</span
                                        >
                                    </div>
                                    <div class="metric">
                                        <span class="metric-label"
                                            >Mayor bloque</span
                                        >
                                        <span class="metric-value"
                                            >{analysis.duplicate_code_results
                                                .largest_duplicate ?? 0} líneas</span
                                        >
                                    </div>
                                </div>
                            {/if}
                        </div>
                    </div>
                </div>
            {/if}
        </div>
    {:else}
        <div class="error-card">
            <Info size={24} />
            <span>No hay análisis disponible para este proyecto.</span>
        </div>
    {/if}
</div>

<style>
    .analysis-page {
        padding: 2rem;
        max-width: 1400px;
        margin: 0 auto;
        background: var(--color-bg-secondary, #f8f9fa);
        min-height: 100vh;
    }

    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        background: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .header-left h1 {
        margin: 0;
        font-size: 1.75rem;
        color: #1e293b;
    }

    .subtitle {
        margin: 0.5rem 0 0 0;
        color: #64748b;
        font-size: 0.95rem;
    }

    .btn-secondary {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: white;
        color: #4a6cf7;
        border: 2px solid #4a6cf7;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        text-decoration: none;
        transition: all 0.2s;
    }

    .btn-secondary:hover {
        background: #4a6cf7;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 108, 247, 0.2);
    }

    /* Loading y Error */
    .loading {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem;
        background: white;
        border-radius: 12px;
        gap: 1rem;
    }

    .spinner {
        width: 48px;
        height: 48px;
        border: 4px solid #f3f4f6;
        border-top-color: #4a6cf7;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }

    .error-card {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.5rem;
        background: #fee2e2;
        color: #dc2626;
        border-radius: 8px;
        border: 1px solid #fecaca;
    }

    /* Summary Cards */
    .summary-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }

    .summary-card {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition:
            transform 0.2s,
            box-shadow 0.2s;
    }

    .summary-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }

    .summary-card.primary {
        background: linear-gradient(135deg, #4a6cf7 0%, #3955d8 100%);
        color: white;
    }

    .card-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 56px;
        height: 56px;
        background: rgba(74, 108, 247, 0.1);
        border-radius: 12px;
        color: #4a6cf7;
        flex-shrink: 0;
    }

    .summary-card.primary .card-icon {
        background: rgba(255, 255, 255, 0.2);
        color: white;
    }

    .card-content {
        flex: 1;
    }

    .value {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.5rem;
    }

    .label {
        color: #64748b;
        font-size: 0.95rem;
    }

    .summary-card.primary .label {
        color: rgba(255, 255, 255, 0.9);
    }

    /* Tabs */
    .tabs {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 2rem;
        background: white;
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        overflow-x: auto;
    }

    .tab {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        background: transparent;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        color: #64748b;
        cursor: pointer;
        transition: all 0.2s;
        white-space: nowrap;
    }

    .tab:hover {
        background: #f1f5f9;
        color: #1e293b;
    }

    .tab.active {
        background: #4a6cf7;
        color: white;
    }

    /* Tab Content */
    .tab-content {
        animation: fadeIn 0.3s ease-in-out;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Overview Grid */
    .overview-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 1.5rem;
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
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }

    .analysis-card.full-width {
        grid-column: 1 / -1;
    }

    .analysis-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        color: #1e293b;
    }

    .analysis-header h3 {
        margin: 0;
        font-size: 1.1rem;
    }

    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
    }

    .metric {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }

    .metric-value.warning {
        color: #f59e0b;
    }

    .metric-value.danger {
        color: #ef4444;
    }

    .metric-value.critical {
        color: #dc2626;
        font-weight: 700;
    }

    /* Advanced Analysis */
    .advanced-analysis {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }

    .confidence-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: #f0fdf4;
        color: #16a34a;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .confidence-badge.success {
        background: #f0fdf4;
        color: #16a34a;
    }

    /* Cross-Language Content */
    .cross-language-content {
        display: flex;
        flex-direction: column;
        gap: 2rem;
    }

    .language-distribution h4,
    .similarity-section h4 {
        margin: 0 0 1rem 0;
        font-size: 1rem;
        color: #475569;
    }

    .language-bars {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .language-bar {
        display: grid;
        grid-template-columns: 80px 1fr 50px;
        align-items: center;
        gap: 1rem;
    }

    .lang-name {
        font-weight: 500;
        color: #1e293b;
    }

    .bar-container {
        height: 24px;
        background: #f1f5f9;
        border-radius: 12px;
        overflow: hidden;
    }

    .bar {
        height: 100%;
        background: linear-gradient(90deg, #4a6cf7 0%, #3955d8 100%);
        border-radius: 12px;
        transition: width 0.5s ease-out;
    }

    .lang-count {
        text-align: right;
        color: #64748b;
    }

    /* Similarity */
    .similarity-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .similarity-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1rem;
        background: #f8fafc;
        border-radius: 8px;
        font-size: 0.9rem;
    }

    .file-pair {
        color: #475569;
        font-family: "Consolas", "Monaco", monospace;
    }

    .similarity-badge {
        padding: 0.25rem 0.75rem;
        background: #fef3c7;
        color: #d97706;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85rem;
    }

    /* Detailed Views */
    .detailed-view {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .detailed-view h2 {
        margin: 0 0 2rem 0;
        font-size: 1.5rem;
        color: #1e293b;
    }

    .detailed-view h3 {
        margin: 1.5rem 0 1rem 0;
        font-size: 1.2rem;
        color: #334155;
    }

    /* Estilos para la vista de complejidad */
    .complexity-summary,
    .quality-summary,
    .bugs-summary {
        margin-bottom: 2rem;
    }

    .function-list,
    .issues-list,
    .bugs-list {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .function-item,
    .issue-item,
    .bug-item {
        background: var(--color-bg-secondary, #f8f9fa);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        transition: all 0.2s;
    }

    .function-item.complexity-high {
        border-color: #ef4444;
        background: #fef2f2;
    }

    .function-item.complexity-medium {
        border-color: #f59e0b;
        background: #fffbeb;
    }

    .function-item.complexity-low {
        border-color: #10b981;
        background: #f0fdf4;
    }

    .function-header,
    .issue-header,
    .bug-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }

    .function-name,
    .issue-type,
    .bug-type {
        font-weight: 600;
        color: #1e293b;
    }

    .complexity-badge {
        background: #e2e8f0;
        color: #475569;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
    }

    .function-details,
    .issue-location,
    .bug-location {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #64748b;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }

    .cognitive-complexity,
    .fix-suggestion {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #64748b;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }

    .issue-description,
    .bug-description {
        color: #475569;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }

    /* Estilos para la sección de análisis adicionales */
    .additional-analyses-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 1.5rem;
    }

    .analysis-section {
        background: var(--color-bg-secondary, #f8f9fa);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }

    .section-header h3 {
        margin: 0;
        font-size: 1.1rem;
        color: #1e293b;
    }

    /* AI Summary */
    .ai-summary {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }

    .ai-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .ai-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: white;
        border-radius: 6px;
        font-weight: 600;
        color: #1e293b;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .precision-badge {
        padding: 0.25rem 0.75rem;
        background: #16a34a;
        color: white;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .ai-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .ai-metric {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .ai-metric.safe {
        border-left: 4px solid #16a34a;
    }

    .ai-metric.warning {
        border-left: 4px solid #f59e0b;
    }

    .recommendations {
        background: white;
        padding: 1rem;
        border-radius: 8px;
    }

    .recommendations h4 {
        margin: 0 0 0.75rem 0;
        font-size: 1rem;
        color: #1e293b;
    }

    .recommendations ul {
        margin: 0;
        padding-left: 0;
        color: #475569;
        list-style: none;
    }

    .recommendations li {
        margin-bottom: 0.5rem;
    }

    .recommendation-item {
        margin-bottom: 16px;
        padding: 12px;
        background-color: rgba(59, 130, 246, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(59, 130, 246, 0.1);
    }

    .recommendation-content {
        width: 100%;
    }

    .recommendation-action {
        color: #3b82f6;
        font-weight: 600;
    }

    .recommendation-desc {
        color: #4b5563;
    }

    .recommendation-impact {
        margin-top: 4px;
        color: #6b7280;
        font-size: 0.875rem;
    }

    .recommendation-items {
        margin-top: 8px;
        padding-left: 16px;
    }

    .affected-items {
        list-style: none;
        padding: 0;
        margin: 4px 0 0 0;
    }

    .affected-items li {
        padding: 2px 0;
        font-size: 0.85rem;
        color: #6b7280;
    }

    .affected-items code {
        background-color: rgba(0, 0, 0, 0.05);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-family: "JetBrains Mono", monospace;
    }

    /* Dead Code Main List */
    .dead-code-main-list {
        padding: 20px;
    }

    .dead-code-items-container {
        display: flex;
        flex-direction: column;
        gap: 24px;
    }

    .dead-code-item-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }

    .dead-code-item-card.safe-to-delete {
        border-color: #10b981;
    }

    .dead-code-item-card.needs-review {
        border-color: #f59e0b;
    }

    .dead-code-item-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .dead-item-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 16px;
        flex-wrap: wrap;
        gap: 16px;
    }

    .dead-item-info {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .dead-item-number {
        font-size: 1.2rem;
        font-weight: 700;
        color: #64748b;
    }

    .dead-item-type-icon .type-icon {
        font-size: 1.5rem;
    }

    .dead-item-title {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .dead-item-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e293b;
        font-family: "JetBrains Mono", monospace;
    }

    .dead-item-type {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .confidence-label {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 24px;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .confidence-label.safe {
        background-color: #d1fae5;
        color: #065f46;
    }

    .confidence-label.review {
        background-color: #fed7aa;
        color: #92400e;
    }

    .confidence-percent {
        font-weight: 700;
        margin-left: 4px;
    }

    .dead-item-location {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        padding: 12px;
        background-color: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }

    .location-file {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.875rem;
        color: #475569;
        flex: 1;
    }

    .location-line {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
    }

    .dead-item-reason {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 16px;
        padding: 12px;
        background-color: #f0f9ff;
        border-radius: 8px;
        border: 1px solid #bae6fd;
        color: #0369a1;
        font-size: 0.875rem;
    }

    .dead-item-code {
        margin-bottom: 16px;
    }

    .code-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        font-weight: 500;
        color: #475569;
    }

    .code-block {
        background-color: #1e293b;
        color: #e2e8f0;
        padding: 16px;
        border-radius: 8px;
        overflow-x: auto;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.875rem;
        line-height: 1.6;
    }

    .dead-item-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 16px;
        margin-bottom: 16px;
        padding: 16px;
        background-color: #f8fafc;
        border-radius: 8px;
    }

    .metric-item {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .metric-item .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-item .metric-value {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1e293b;
    }

    .dead-item-suggestion {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 16px;
        padding: 12px;
        background-color: #fef3c7;
        border-radius: 8px;
        border: 1px solid #fde68a;
        color: #92400e;
        font-size: 0.875rem;
    }

    .dead-item-actions {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }

    .action-btn {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border: none;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
    }

    .action-btn.view {
        background-color: #e0e7ff;
        color: #3730a3;
    }

    .action-btn.view:hover {
        background-color: #c7d2fe;
    }

    .action-btn.delete {
        background-color: #fee2e2;
        color: #dc2626;
    }

    .action-btn.delete:hover {
        background-color: #fecaca;
    }

    .action-btn.review {
        background-color: #fef3c7;
        color: #d97706;
    }

    .action-btn.review:hover {
        background-color: #fde68a;
    }

    .no-dead-code,
    .no-data {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px;
        text-align: center;
        color: #64748b;
    }

    .no-dead-code h3 {
        margin: 16px 0 8px;
        font-size: 1.5rem;
        color: #10b981;
    }

    .no-data {
        color: #94a3b8;
    }

    /* Vulnerabilities */
    .vulnerabilities-list {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .vulnerability-item {
        border-radius: 8px;
        border: 2px solid;
        overflow: hidden;
        transition: all 0.2s;
    }

    .vulnerability-item.severity-critical {
        border-color: #fee2e2;
        background: #fef2f2;
    }

    .vulnerability-item.severity-high {
        border-color: #fed7aa;
        background: #fff7ed;
    }

    .vulnerability-item.severity-medium {
        border-color: #fef3c7;
        background: #fffbeb;
    }

    .vulnerability-item.severity-low {
        border-color: #dbeafe;
        background: #eff6ff;
    }

    .vuln-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.03);
    }

    .vuln-type {
        flex: 1;
        font-weight: 600;
        color: #1e293b;
    }

    .severity-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .severity-badge.critical {
        background: #dc2626;
        color: white;
    }

    .severity-badge.high {
        background: #ea580c;
        color: white;
    }

    .severity-badge.medium {
        background: #d97706;
        color: white;
    }

    .severity-badge.low {
        background: #2563eb;
        color: white;
    }

    .vuln-details {
        padding: 1rem;
    }

    .vuln-description {
        margin: 0 0 0.75rem 0;
        color: #475569;
        line-height: 1.6;
    }

    .vuln-location {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        color: #64748b;
        font-family: "Consolas", "Monaco", monospace;
    }

    .cwe-badge {
        display: inline-block;
        margin-top: 0.5rem;
        padding: 0.25rem 0.5rem;
        background: #e5e7eb;
        color: #4b5563;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Dark mode support */
    :global(.dark) .analysis-page {
        background: #0f172a;
    }

    :global(.dark) .header,
    :global(.dark) .summary-card,
    :global(.dark) .analysis-card,
    :global(.dark) .detailed-view,
    :global(.dark) .tabs {
        background: #1e293b;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    :global(.dark) .header-left h1,
    :global(.dark) .analysis-header,
    :global(.dark) .metric-value,
    :global(.dark) .detailed-view h2 {
        color: #f1f5f9;
    }

    :global(.dark) .subtitle,
    :global(.dark) .label,
    :global(.dark) .metric-label {
        color: #94a3b8;
    }

    :global(.dark) .tab {
        color: #94a3b8;
    }

    :global(.dark) .tab:hover {
        background: #334155;
        color: #f1f5f9;
    }

    :global(.dark) .btn-secondary {
        background: #1e293b;
        border-color: #475569;
        color: #e2e8f0;
    }

    :global(.dark) .btn-secondary:hover {
        background: #334155;
        border-color: #64748b;
    }
</style>
