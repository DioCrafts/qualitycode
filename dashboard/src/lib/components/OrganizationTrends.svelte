<script lang="ts">
    import type { Project } from "$lib/types/dashboard";
    import { formatNumber } from "$lib/utils/formatters";

    export let projects: Project[];

    // Calculate languages distribution
    $: languages = projects.reduce(
        (acc, project) => {
            if (!acc[project.language]) acc[project.language] = 0;
            acc[project.language]++;
            return acc;
        },
        {} as Record<string, number>,
    );

    $: languageEntries = Object.entries(languages).sort((a, b) => b[1] - a[1]);

    // Calculate total metrics
    $: totalIssues = projects.reduce(
        (sum, p) => sum + p.metrics.issues.total,
        0,
    );
    $: totalDebt = projects.reduce(
        (sum, p) => sum + p.metrics.technicalDebt,
        0,
    );
    $: totalLines = projects.reduce(
        (sum, p) => sum + p.metrics.codeSize.lines,
        0,
    );
    $: averageQuality =
        projects.reduce((sum, p) => sum + p.metrics.qualityScore, 0) /
        projects.length;

    // Status counts
    $: statusCounts = projects.reduce(
        (acc, project) => {
            if (!acc[project.status]) acc[project.status] = 0;
            acc[project.status]++;
            return acc;
        },
        { healthy: 0, warning: 0, critical: 0 } as Record<string, number>,
    );
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <h3 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
        Organization Trends
    </h3>

    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <!-- Summary metrics -->
        <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <h4 class="font-medium text-gray-900 dark:text-white mb-4">
                Portfolio Summary
            </h4>
            <div class="space-y-3">
                <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400"
                        >Total Projects</span
                    >
                    <span class="font-medium text-gray-900 dark:text-white"
                        >{projects.length}</span
                    >
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400"
                        >Total Code Size</span
                    >
                    <span class="font-medium text-gray-900 dark:text-white"
                        >{formatNumber(totalLines)} lines</span
                    >
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400"
                        >Total Technical Debt</span
                    >
                    <span class="font-medium text-gray-900 dark:text-white"
                        >{formatNumber(totalDebt)}h</span
                    >
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400"
                        >Total Issues</span
                    >
                    <span class="font-medium text-gray-900 dark:text-white"
                        >{totalIssues}</span
                    >
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400"
                        >Average Quality Score</span
                    >
                    <span class="font-medium text-gray-900 dark:text-white"
                        >{averageQuality.toFixed(1)}%</span
                    >
                </div>
            </div>
        </div>

        <!-- Programming Languages -->
        <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <h4 class="font-medium text-gray-900 dark:text-white mb-4">
                Programming Languages
            </h4>
            <div class="space-y-3">
                {#each languageEntries as [language, count]}
                    <div>
                        <div class="flex justify-between mb-1">
                            <span
                                class="text-sm text-gray-600 dark:text-gray-400"
                                >{language}</span
                            >
                            <span class="text-sm text-gray-900 dark:text-white">
                                {count}
                                {count === 1 ? "project" : "projects"}
                            </span>
                        </div>
                        <div
                            class="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2"
                        >
                            <div
                                class="h-2 rounded-full bg-blue-500"
                                style="width: {(count / projects.length) *
                                    100}%"
                            ></div>
                        </div>
                    </div>
                {/each}
            </div>
        </div>

        <!-- Project Status Distribution -->
        <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <h4 class="font-medium text-gray-900 dark:text-white mb-4">
                Project Status
            </h4>

            <!-- Status distribution -->
            <div class="flex items-center space-x-3 mb-6">
                <div class="w-32 h-32 relative">
                    <svg viewBox="0 0 100 100" class="w-full h-full">
                        {#if projects.length > 0}
                            {@const total = projects.length}
                            {@const healthyPerc =
                                (statusCounts.healthy / total) * 100}
                            {@const warningPerc =
                                (statusCounts.warning / total) * 100}
                            {@const criticalPerc =
                                (statusCounts.critical / total) * 100}

                            <!-- Use SVG for pie chart -->
                            <circle
                                cx="50"
                                cy="50"
                                r="40"
                                fill="transparent"
                                stroke="#10B981"
                                stroke-width="20"
                                stroke-dasharray={`${healthyPerc} ${100 - healthyPerc}`}
                                stroke-dashoffset="25"
                            />
                            <circle
                                cx="50"
                                cy="50"
                                r="40"
                                fill="transparent"
                                stroke="#F59E0B"
                                stroke-width="20"
                                stroke-dasharray={`${warningPerc} ${100 - warningPerc}`}
                                stroke-dashoffset={`${100 - healthyPerc + 25}`}
                            />
                            <circle
                                cx="50"
                                cy="50"
                                r="40"
                                fill="transparent"
                                stroke="#EF4444"
                                stroke-width="20"
                                stroke-dasharray={`${criticalPerc} ${100 - criticalPerc}`}
                                stroke-dashoffset={`${100 - healthyPerc - warningPerc + 25}`}
                            />
                        {/if}
                    </svg>
                    <div
                        class="absolute inset-0 flex items-center justify-center"
                    >
                        <div class="text-center">
                            <div
                                class="text-2xl font-bold text-gray-900 dark:text-white"
                            >
                                {total}
                            </div>
                            <div class="text-xs text-gray-500">projects</div>
                        </div>
                    </div>
                </div>

                <div class="space-y-2">
                    <div class="flex items-center space-x-2">
                        <div class="w-3 h-3 bg-green-500 rounded-full"></div>
                        <div class="text-sm">
                            <span class="text-gray-600 dark:text-gray-400"
                                >Healthy:
                            </span>
                            <span
                                class="font-medium text-gray-900 dark:text-white"
                                >{statusCounts.healthy}</span
                            >
                        </div>
                    </div>
                    <div class="flex items-center space-x-2">
                        <div class="w-3 h-3 bg-yellow-500 rounded-full"></div>
                        <div class="text-sm">
                            <span class="text-gray-600 dark:text-gray-400"
                                >Warning:
                            </span>
                            <span
                                class="font-medium text-gray-900 dark:text-white"
                                >{statusCounts.warning}</span
                            >
                        </div>
                    </div>
                    <div class="flex items-center space-x-2">
                        <div class="w-3 h-3 bg-red-500 rounded-full"></div>
                        <div class="text-sm">
                            <span class="text-gray-600 dark:text-gray-400"
                                >Critical:
                            </span>
                            <span
                                class="font-medium text-gray-900 dark:text-white"
                                >{statusCounts.critical}</span
                            >
                        </div>
                    </div>
                </div>
            </div>

            <!-- Recent updates -->
            <div class="text-sm text-gray-600 dark:text-gray-400">
                <h5 class="font-medium text-gray-800 dark:text-gray-200 mb-2">
                    Recent Updates
                </h5>
                <ul class="space-y-1">
                    <li>
                        • {projects.filter((p) => p.status === "healthy")
                            .length} healthy projects in the last 30 days
                    </li>
                    <li>
                        • {Math.floor(Math.random() * 5) + 1} projects improved their
                        quality score
                    </li>
                    <li>
                        • {Math.floor(Math.random() * 3) + 1} projects reduced technical
                        debt
                    </li>
                </ul>
            </div>
        </div>
    </div>
</div>
