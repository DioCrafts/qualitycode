<script lang="ts">
    import { cicdStore, dashboardStore } from "$lib/stores/dashboard";
    import type { Pipeline } from "$lib/types/dashboard";
    import { formatDuration } from "$lib/utils/formatters";
    import { onMount } from "svelte";
    import DORAMetricsDisplay from "./DORAMetricsDisplay.svelte";
    import DeploymentHistory from "./DeploymentHistory.svelte";
    import FailureAnalysis from "./FailureAnalysis.svelte";
    import PipelineMonitor from "./PipelineMonitor.svelte";

    $: ({ selectedProject, selectedTimeRange } = $dashboardStore);
    $: ({ pipelines, doraMetrics, loading, error } = $cicdStore);

    let activeTab: "overview" | "pipelines" | "deployments" | "failures" =
        "overview";
    let selectedPipeline: Pipeline | null = null;

    onMount(() => {
        if (selectedProject) {
            cicdStore.loadCICDData(selectedProject.id, selectedTimeRange);
        }
    });

    $: if (selectedProject) {
        cicdStore.loadCICDData(selectedProject.id, selectedTimeRange);
    }

    $: activePipelines = pipelines.filter((p) => p.status === "running");
    $: failedPipelines = pipelines.filter((p) => p.status === "failed");
    $: successRate =
        pipelines.length > 0
            ? (pipelines.filter((p) => p.status === "success").length /
                  pipelines.length) *
              100
            : 0;

    function getPipelineStatusColor(status: string): string {
        const colors = {
            success: "text-green-600 bg-green-100",
            failed: "text-red-600 bg-red-100",
            running: "text-blue-600 bg-blue-100",
            pending: "text-gray-600 bg-gray-100",
        };
        return (
            colors[status as keyof typeof colors] || "text-gray-600 bg-gray-100"
        );
    }

    function getPipelineStatusIcon(status: string): string {
        const icons = {
            success: "✅",
            failed: "❌",
            running: "🔄",
            pending: "⏳",
        };
        return icons[status as keyof typeof icons] || "❓";
    }

    function getTrendIcon(trend: "up" | "down" | "stable"): string {
        const icons = {
            up: "↗️",
            down: "↘️",
            stable: "→",
        };
        return icons[trend] || "→";
    }

    function getTrendColor(
        trend: "up" | "down" | "stable",
        metric: string,
    ): string {
        // For some metrics, up is good (deployment frequency), for others down is good (failure rate)
        const upIsGood = ["deploymentFrequency"];
        const downIsGood = [
            "changeFailureRate",
            "timeToRestoreService",
            "leadTimeForChanges",
        ];

        if (trend === "stable") return "text-gray-600";

        if (upIsGood.includes(metric)) {
            return trend === "up" ? "text-green-600" : "text-red-600";
        } else if (downIsGood.includes(metric)) {
            return trend === "down" ? "text-green-600" : "text-red-600";
        }

        return "text-gray-600";
    }
</script>

<div class="space-y-6">
    <!-- Header -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div class="flex justify-between items-start">
            <div>
                <h2
                    class="text-2xl font-bold text-gray-900 dark:text-white mb-2"
                >
                    CI/CD Integration & DORA Metrics
                </h2>
                <p class="text-gray-600 dark:text-gray-400">
                    Pipeline monitoring and DevOps performance metrics
                </p>
            </div>
            <div class="flex items-center space-x-2">
                <span class="flex items-center space-x-1 text-sm text-gray-500">
                    <span
                        class="w-2 h-2 bg-green-500 rounded-full animate-pulse"
                    ></span>
                    <span>Live</span>
                </span>
            </div>
        </div>
    </div>

    {#if loading}
        <div class="flex items-center justify-center h-96">
            <div
                class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"
            ></div>
        </div>
    {:else if error}
        <div
            class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4"
        >
            <p class="text-red-800 dark:text-red-200">Error: {error}</p>
        </div>
    {:else if !selectedProject}
        <div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-8 text-center">
            <p class="text-gray-600 dark:text-gray-400">
                Select a project to view CI/CD metrics
            </p>
        </div>
    {:else}
        <!-- DORA Metrics Overview -->
        {#if doraMetrics}
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <!-- Deployment Frequency -->
                <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <div class="flex items-center justify-between mb-2">
                        <h3
                            class="text-sm font-medium text-gray-600 dark:text-gray-400"
                        >
                            Deployment Frequency
                        </h3>
                        <span
                            class={getTrendColor(
                                doraMetrics.deploymentFrequency.trend,
                                "deploymentFrequency",
                            )}
                        >
                            {getTrendIcon(
                                doraMetrics.deploymentFrequency.trend,
                            )}
                        </span>
                    </div>
                    <p class="text-2xl font-bold text-gray-900 dark:text-white">
                        {doraMetrics.deploymentFrequency.value}
                    </p>
                    <p class="text-sm text-gray-500 dark:text-gray-400">
                        {doraMetrics.deploymentFrequency.unit.replace("_", " ")}
                    </p>
                </div>

                <!-- Lead Time for Changes -->
                <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <div class="flex items-center justify-between mb-2">
                        <h3
                            class="text-sm font-medium text-gray-600 dark:text-gray-400"
                        >
                            Lead Time for Changes
                        </h3>
                        <span
                            class={getTrendColor(
                                doraMetrics.leadTimeForChanges.trend,
                                "leadTimeForChanges",
                            )}
                        >
                            {getTrendIcon(doraMetrics.leadTimeForChanges.trend)}
                        </span>
                    </div>
                    <p class="text-2xl font-bold text-gray-900 dark:text-white">
                        {doraMetrics.leadTimeForChanges.value}
                    </p>
                    <p class="text-sm text-gray-500 dark:text-gray-400">
                        {doraMetrics.leadTimeForChanges.unit}
                    </p>
                </div>

                <!-- Change Failure Rate -->
                <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <div class="flex items-center justify-between mb-2">
                        <h3
                            class="text-sm font-medium text-gray-600 dark:text-gray-400"
                        >
                            Change Failure Rate
                        </h3>
                        <span
                            class={getTrendColor(
                                doraMetrics.changeFailureRate.trend,
                                "changeFailureRate",
                            )}
                        >
                            {getTrendIcon(doraMetrics.changeFailureRate.trend)}
                        </span>
                    </div>
                    <p class="text-2xl font-bold text-gray-900 dark:text-white">
                        {doraMetrics.changeFailureRate.percentage}%
                    </p>
                    <p class="text-sm text-gray-500 dark:text-gray-400">
                        {doraMetrics.changeFailureRate.value} failures
                    </p>
                </div>

                <!-- Time to Restore Service -->
                <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <div class="flex items-center justify-between mb-2">
                        <h3
                            class="text-sm font-medium text-gray-600 dark:text-gray-400"
                        >
                            Time to Restore Service
                        </h3>
                        <span
                            class={getTrendColor(
                                doraMetrics.timeToRestoreService.trend,
                                "timeToRestoreService",
                            )}
                        >
                            {getTrendIcon(
                                doraMetrics.timeToRestoreService.trend,
                            )}
                        </span>
                    </div>
                    <p class="text-2xl font-bold text-gray-900 dark:text-white">
                        {doraMetrics.timeToRestoreService.value}
                    </p>
                    <p class="text-sm text-gray-500 dark:text-gray-400">
                        {doraMetrics.timeToRestoreService.unit}
                    </p>
                </div>
            </div>
        {/if}

        <!-- Tabs -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow">
            <div class="border-b border-gray-200 dark:border-gray-700">
                <nav class="flex -mb-px">
                    <button
                        on:click={() => (activeTab = "overview")}
                        class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                   {activeTab === 'overview'
                            ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                            : 'border-transparent text-gray-500 hover:text-gray-700'}"
                    >
                        Overview
                    </button>
                    <button
                        on:click={() => (activeTab = "pipelines")}
                        class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                   {activeTab === 'pipelines'
                            ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                            : 'border-transparent text-gray-500 hover:text-gray-700'}"
                    >
                        Pipelines ({pipelines.length})
                    </button>
                    <button
                        on:click={() => (activeTab = "deployments")}
                        class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                   {activeTab === 'deployments'
                            ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                            : 'border-transparent text-gray-500 hover:text-gray-700'}"
                    >
                        Deployments
                    </button>
                    <button
                        on:click={() => (activeTab = "failures")}
                        class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                   {activeTab === 'failures'
                            ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                            : 'border-transparent text-gray-500 hover:text-gray-700'}"
                    >
                        Failures ({failedPipelines.length})
                    </button>
                </nav>
            </div>

            <div class="p-6">
                {#if activeTab === "overview"}
                    <!-- Pipeline Status Overview -->
                    <div class="space-y-6">
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div
                                class="bg-green-50 dark:bg-green-900/20 rounded-lg p-4"
                            >
                                <h4
                                    class="text-sm font-medium text-green-800 dark:text-green-200 mb-2"
                                >
                                    Success Rate
                                </h4>
                                <p
                                    class="text-3xl font-bold text-green-600 dark:text-green-400"
                                >
                                    {successRate.toFixed(1)}%
                                </p>
                                <p
                                    class="text-sm text-green-600 dark:text-green-400 mt-1"
                                >
                                    Last {pipelines.length} pipelines
                                </p>
                            </div>

                            <div
                                class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4"
                            >
                                <h4
                                    class="text-sm font-medium text-blue-800 dark:text-blue-200 mb-2"
                                >
                                    Active Pipelines
                                </h4>
                                <p
                                    class="text-3xl font-bold text-blue-600 dark:text-blue-400"
                                >
                                    {activePipelines.length}
                                </p>
                                <p
                                    class="text-sm text-blue-600 dark:text-blue-400 mt-1"
                                >
                                    Currently running
                                </p>
                            </div>

                            <div
                                class="bg-red-50 dark:bg-red-900/20 rounded-lg p-4"
                            >
                                <h4
                                    class="text-sm font-medium text-red-800 dark:text-red-200 mb-2"
                                >
                                    Failed Pipelines
                                </h4>
                                <p
                                    class="text-3xl font-bold text-red-600 dark:text-red-400"
                                >
                                    {failedPipelines.length}
                                </p>
                                <p
                                    class="text-sm text-red-600 dark:text-red-400 mt-1"
                                >
                                    Need attention
                                </p>
                            </div>
                        </div>

                        {#if doraMetrics}
                            <DORAMetricsDisplay
                                metrics={doraMetrics}
                                timeRange={selectedTimeRange}
                            />
                        {/if}
                    </div>
                {:else if activeTab === "pipelines"}
                    <PipelineMonitor
                        {pipelines}
                        on:select={(e) => (selectedPipeline = e.detail)}
                    />
                {:else if activeTab === "deployments"}
                    <DeploymentHistory
                        projectId={selectedProject.id}
                        timeRange={selectedTimeRange}
                    />
                {:else if activeTab === "failures"}
                    <FailureAnalysis
                        failures={failedPipelines}
                        projectId={selectedProject.id}
                    />
                {/if}
            </div>
        </div>

        <!-- Recent Pipeline Activity -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3
                class="text-lg font-semibold mb-4 text-gray-900 dark:text-white"
            >
                Recent Pipeline Activity
            </h3>
            <div class="space-y-3">
                {#each pipelines.slice(0, 5) as pipeline (pipeline.id)}
                    <div
                        class="flex items-center justify-between p-3 border border-gray-200
                      dark:border-gray-700 rounded-lg hover:bg-gray-50
                      dark:hover:bg-gray-700 transition-colors"
                    >
                        <div class="flex items-center space-x-3">
                            <span class="text-lg"
                                >{getPipelineStatusIcon(pipeline.status)}</span
                            >
                            <div>
                                <p
                                    class="font-medium text-gray-900 dark:text-white"
                                >
                                    {pipeline.name}
                                </p>
                                <p
                                    class="text-sm text-gray-500 dark:text-gray-400"
                                >
                                    {new Date(
                                        pipeline.lastRun,
                                    ).toLocaleString()}
                                </p>
                            </div>
                        </div>
                        <div class="flex items-center space-x-3">
                            <span
                                class="px-2 py-1 text-xs font-medium rounded {getPipelineStatusColor(
                                    pipeline.status,
                                )}"
                            >
                                {pipeline.status}
                            </span>
                            <span class="text-sm text-gray-500">
                                {formatDuration(pipeline.duration)}
                            </span>
                        </div>
                    </div>
                {/each}
            </div>
        </div>
    {/if}
</div>
