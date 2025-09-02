<script lang="ts">
    import type { Pipeline } from "$lib/types/dashboard";
    import { formatDate, formatDuration } from "$lib/utils/formatters";

    export let failures: Pipeline[];
    export let projectId: string;

    // Mock failure data
    let loading = false;
    let error: string | null = null;

    $: failureCategories = categorizeFailures(failures);

    // Create failure analysis
    function categorizeFailures(failures: Pipeline[]): Record<string, number> {
        const categories: Record<string, number> = {
            "Build Error": 0,
            "Test Failure": 0,
            "Dependency Issue": 0,
            "Configuration Error": 0,
            "Performance Issue": 0,
            "Security Check": 0,
            Other: 0,
        };

        // In a real app, we'd analyze the failures
        // Here we'll just generate random data
        failures.forEach((failure) => {
            // Get a random category
            const allCategories = Object.keys(categories);
            const category =
                allCategories[Math.floor(Math.random() * allCategories.length)];
            categories[category]++;
        });

        return categories;
    }

    // Calculate most common failure
    $: mostCommonFailure = Object.entries(failureCategories).sort(
        (a, b) => b[1] - a[1],
    )[0];

    // Find stages with most failures
    $: stageFailures = failures
        .flatMap((p) => p.stages.filter((s) => s.status === "failed"))
        .reduce(
            (acc, stage) => {
                if (!acc[stage.name]) acc[stage.name] = 0;
                acc[stage.name]++;
                return acc;
            },
            {} as Record<string, number>,
        );

    $: mostFailedStage = Object.entries(stageFailures).sort(
        (a, b) => b[1] - a[1],
    )[0];

    // Calculate recent trends (mocked)
    $: failureRate = (failures.length / 15) * 100; // Assuming 15 total runs
    $: failureTrend = Math.random() > 0.5 ? "improving" : "worsening";
</script>

<div class="space-y-6">
    <div class="flex justify-between items-center">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">
            Failure Analysis
        </h3>

        <div class="text-sm text-gray-500">
            {failures.length} failures in the last 30 days
        </div>
    </div>

    {#if loading}
        <div class="flex items-center justify-center h-64">
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
    {:else if failures.length === 0}
        <div
            class="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 text-center"
        >
            <svg
                class="w-12 h-12 text-green-600 dark:text-green-400 mx-auto mb-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
            >
                <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
            </svg>
            <h4
                class="text-lg font-medium text-green-800 dark:text-green-200 mb-1"
            >
                No failures detected!
            </h4>
            <p class="text-green-700 dark:text-green-300">
                All pipelines are running successfully.
            </p>
        </div>
    {:else}
        <!-- Overview -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                <h4 class="font-medium text-gray-900 dark:text-white mb-3">
                    Failure Rate
                </h4>
                <div class="text-3xl font-bold text-red-600 dark:text-red-400">
                    {failureRate.toFixed(1)}%
                </div>
                <p class="mt-2 text-sm text-gray-600 dark:text-gray-400">
                    Trend: <span
                        class={failureTrend === "improving"
                            ? "text-green-600"
                            : "text-red-600"}
                    >
                        {failureTrend === "improving"
                            ? "↘️ Improving"
                            : "↗️ Worsening"}
                    </span>
                </p>
            </div>

            <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 class="font-medium text-gray-900 dark:text-white mb-3">
                    Most Common Failure
                </h4>
                <div class="text-xl font-bold text-gray-900 dark:text-white">
                    {mostCommonFailure?.[0]}
                </div>
                <p class="mt-2 text-sm text-gray-600 dark:text-gray-400">
                    {mostCommonFailure?.[1]} occurrences ({Math.round(
                        (mostCommonFailure?.[1] / failures.length) * 100,
                    )}%)
                </p>
            </div>

            <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 class="font-medium text-gray-900 dark:text-white mb-3">
                    Most Failed Stage
                </h4>
                <div class="text-xl font-bold text-gray-900 dark:text-white">
                    {mostFailedStage?.[0]}
                </div>
                <p class="mt-2 text-sm text-gray-600 dark:text-gray-400">
                    Failed in {mostFailedStage?.[1]} pipelines
                </p>
            </div>
        </div>

        <!-- Failure Categories -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h4 class="font-medium text-gray-900 dark:text-white mb-4">
                Failure Categories
            </h4>

            <div class="space-y-3">
                {#each Object.entries(failureCategories).filter(([_, count]) => count > 0) as [category, count]}
                    <div>
                        <div class="flex justify-between text-sm mb-1">
                            <span class="text-gray-700 dark:text-gray-300"
                                >{category}</span
                            >
                            <span class="text-gray-700 dark:text-gray-300">
                                {count} failure{count > 1 ? "s" : ""}
                            </span>
                        </div>
                        <div
                            class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2"
                        >
                            <div
                                class="h-2 rounded-full bg-red-500"
                                style="width: {(count / failures.length) *
                                    100}%"
                            ></div>
                        </div>
                    </div>
                {/each}
            </div>
        </div>

        <!-- Recent Failures -->
        <div>
            <h4 class="font-medium text-gray-900 dark:text-white mb-4">
                Recent Failures
            </h4>

            <div class="space-y-4">
                {#each failures.slice(0, 5) as failure (failure.id)}
                    {@const failedStage = failure.stages.find(
                        (s) => s.status === "failed",
                    )}
                    <div
                        class="bg-white dark:bg-gray-800 rounded-lg shadow p-4 border-l-4 border-red-500"
                    >
                        <div class="flex justify-between items-start mb-3">
                            <div>
                                <h5
                                    class="font-medium text-gray-900 dark:text-white"
                                >
                                    {failure.name}
                                </h5>
                                <p class="text-sm text-gray-500">
                                    Failed on {formatDate(failure.lastRun)}
                                </p>
                            </div>
                            <div class="text-sm text-gray-500">
                                Duration: {formatDuration(failure.duration)}
                            </div>
                        </div>

                        {#if failedStage}
                            <div
                                class="bg-red-50 dark:bg-red-900/20 p-3 rounded-lg text-sm"
                            >
                                <div
                                    class="font-medium text-red-800 dark:text-red-200 mb-1"
                                >
                                    Failed in stage: {failedStage.name}
                                </div>
                                {#if failedStage.logs && failedStage.logs.length > 0}
                                    <pre
                                        class="text-red-700 dark:text-red-300 overflow-x-auto whitespace-pre-wrap text-xs">{failedStage
                                            .logs[0]}</pre>
                                {:else}
                                    <p class="text-red-700 dark:text-red-300">
                                        No logs available. Check pipeline
                                        details for more information.
                                    </p>
                                {/if}
                            </div>

                            <div class="mt-3 flex justify-end space-x-3">
                                <button
                                    class="text-sm text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200"
                                >
                                    View Full Logs
                                </button>
                                <button
                                    class="text-sm text-blue-600 hover:text-blue-800"
                                >
                                    Retry Pipeline
                                </button>
                            </div>
                        {/if}
                    </div>
                {/each}
            </div>
        </div>

        <!-- Recommendations -->
        <div class="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h4 class="font-medium text-gray-900 dark:text-white mb-4">
                Recommendations
            </h4>

            <div class="space-y-4">
                {#if mostCommonFailure}
                    <div class="flex items-start space-x-3">
                        <div
                            class="flex-shrink-0 bg-indigo-100 dark:bg-indigo-900 p-1 rounded-full"
                        >
                            <svg
                                class="w-5 h-5 text-indigo-600 dark:text-indigo-400"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                    stroke-width="2"
                                    d="M13 10V3L4 14h7v7l9-11h-7z"
                                />
                            </svg>
                        </div>
                        <div>
                            <h5
                                class="font-medium text-gray-900 dark:text-white mb-1"
                            >
                                Address {mostCommonFailure[0]} Issues
                            </h5>
                            <p class="text-gray-700 dark:text-gray-300 text-sm">
                                This category accounts for {Math.round(
                                    (mostCommonFailure[1] / failures.length) *
                                        100,
                                )}% of your failures. Consider {mostCommonFailure[0] ===
                                "Test Failure"
                                    ? "reviewing test stability"
                                    : mostCommonFailure[0] === "Build Error"
                                      ? "checking build dependencies"
                                      : mostCommonFailure[0] ===
                                          "Dependency Issue"
                                        ? "updating dependency management"
                                        : "investigating root causes"}.
                            </p>
                        </div>
                    </div>
                {/if}

                {#if mostFailedStage}
                    <div class="flex items-start space-x-3">
                        <div
                            class="flex-shrink-0 bg-indigo-100 dark:bg-indigo-900 p-1 rounded-full"
                        >
                            <svg
                                class="w-5 h-5 text-indigo-600 dark:text-indigo-400"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                    stroke-width="2"
                                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                                />
                            </svg>
                        </div>
                        <div>
                            <h5
                                class="font-medium text-gray-900 dark:text-white mb-1"
                            >
                                Focus on {mostFailedStage[0]} Stage
                            </h5>
                            <p class="text-gray-700 dark:text-gray-300 text-sm">
                                This stage is failing most frequently. Consider
                                adding additional logging or tests to identify
                                the specific issues.
                            </p>
                        </div>
                    </div>
                {/if}

                <div class="flex items-start space-x-3">
                    <div
                        class="flex-shrink-0 bg-indigo-100 dark:bg-indigo-900 p-1 rounded-full"
                    >
                        <svg
                            class="w-5 h-5 text-indigo-600 dark:text-indigo-400"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                stroke-width="2"
                                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                        </svg>
                    </div>
                    <div>
                        <h5
                            class="font-medium text-gray-900 dark:text-white mb-1"
                        >
                            Improve Error Handling
                        </h5>
                        <p class="text-gray-700 dark:text-gray-300 text-sm">
                            Consider implementing more robust error handling and
                            retry mechanisms in your CI/CD pipeline,
                            particularly for network-related operations.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    {/if}
</div>
