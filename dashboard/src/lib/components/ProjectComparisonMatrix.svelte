<script lang="ts">
    import type { Project } from "$lib/types/dashboard";

    export let projects: Project[];

    // Metrics to compare
    const metrics = [
        {
            name: "Quality Score",
            key: "qualityScore",
            format: (v: number) => `${v.toFixed(1)}%`,
            higher: true,
        },
        {
            name: "Technical Debt",
            key: "technicalDebt",
            format: (v: number) => `${v}h`,
            higher: false,
        },
        {
            name: "Code Coverage",
            key: "coverage",
            format: (v: number) => `${v}%`,
            higher: true,
        },
        {
            name: "Critical Issues",
            key: "issues.critical",
            format: (v: number) => `${v}`,
            higher: false,
        },
        {
            name: "Total Issues",
            key: "issues.total",
            format: (v: number) => `${v}`,
            higher: false,
        },
        {
            name: "Code Size (LOC)",
            key: "codeSize.lines",
            format: (v: number) => `${v.toLocaleString()}`,
            higher: null,
        },
    ];

    // Get value from nested path
    function getValue(obj: any, path: string) {
        return path.split(".").reduce((p, c) => p && p[c], obj);
    }

    // Get best and worst values for each metric
    function getBestWorst(metric: (typeof metrics)[0]) {
        const values = projects.map((p) => getValue(p.metrics, metric.key));
        if (metric.higher === null) return { best: null, worst: null };

        const best = metric.higher ? Math.max(...values) : Math.min(...values);

        const worst = metric.higher ? Math.min(...values) : Math.max(...values);

        return { best, worst };
    }

    function getColorClass(value: number, metric: (typeof metrics)[0]) {
        if (metric.higher === null) return "";

        const { best, worst } = getBestWorst(metric);
        if (best === worst) return "";

        if (value === best) return "bg-green-100 dark:bg-green-900/30";
        if (value === worst) return "bg-red-100 dark:bg-red-900/30";
        return "";
    }
</script>

<div class="space-y-6">
    <div class="flex justify-between items-center">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">
            Project Comparison Matrix
        </h3>
        <div class="text-sm text-gray-500">
            {projects.length} projects selected
        </div>
    </div>

    {#if projects.length < 2}
        <div class="text-center py-8">
            <p class="text-gray-500">Select at least 2 projects to compare</p>
            <p class="text-sm text-gray-500 mt-2">
                Use the grid or table view to select multiple projects
            </p>
        </div>
    {:else}
        <div class="overflow-x-auto">
            <table
                class="min-w-full divide-y divide-gray-200 dark:divide-gray-700"
            >
                <thead class="bg-gray-50 dark:bg-gray-700">
                    <tr>
                        <th
                            scope="col"
                            class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider"
                        >
                            Metric
                        </th>
                        {#each projects as project}
                            <th
                                scope="col"
                                class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider"
                            >
                                {project.name}
                                <div
                                    class="text-xs font-normal text-gray-500 normal-case"
                                >
                                    {project.language}
                                </div>
                            </th>
                        {/each}
                    </tr>
                </thead>
                <tbody
                    class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700"
                >
                    {#each metrics as metric}
                        <tr>
                            <td
                                class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white"
                            >
                                {metric.name}
                            </td>
                            {#each projects as project}
                                {@const value = getValue(
                                    project.metrics,
                                    metric.key,
                                )}
                                <td
                                    class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white {getColorClass(
                                        value,
                                        metric,
                                    )}"
                                >
                                    {metric.format(value)}
                                </td>
                            {/each}
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Quality Score Comparison -->
            <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
                <h4 class="font-medium text-gray-900 dark:text-white mb-4">
                    Quality Score Comparison
                </h4>
                <div class="space-y-3">
                    {#each projects as project}
                        <div>
                            <div class="flex justify-between mb-1">
                                <span
                                    class="text-sm font-medium text-gray-700 dark:text-gray-300"
                                    >{project.name}</span
                                >
                                <span
                                    class="text-sm font-medium text-gray-700 dark:text-gray-300"
                                    >{project.metrics.qualityScore}%</span
                                >
                            </div>
                            <div
                                class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2"
                            >
                                <div
                                    class="h-2 rounded-full"
                                    class:bg-red-500={project.metrics
                                        .qualityScore < 60}
                                    class:bg-yellow-500={project.metrics
                                        .qualityScore >= 60 &&
                                        project.metrics.qualityScore < 80}
                                    class:bg-green-500={project.metrics
                                        .qualityScore >= 80}
                                    style="width: {project.metrics
                                        .qualityScore}%"
                                ></div>
                            </div>
                        </div>
                    {/each}
                </div>
            </div>

            <!-- Issues Distribution -->
            <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
                <h4 class="font-medium text-gray-900 dark:text-white mb-4">
                    Issue Distribution
                </h4>
                <div class="space-y-3">
                    {#each projects as project}
                        <div>
                            <div class="flex justify-between mb-1">
                                <span
                                    class="text-sm font-medium text-gray-700 dark:text-gray-300"
                                    >{project.name}</span
                                >
                                <span
                                    class="text-sm font-medium text-gray-700 dark:text-gray-300"
                                    >{project.metrics.issues.total} issues</span
                                >
                            </div>
                            {#if project.metrics.issues.total > 0}
                                {@const critical =
                                    (project.metrics.issues.critical /
                                        project.metrics.issues.total) *
                                    100}
                                {@const high =
                                    (project.metrics.issues.high /
                                        project.metrics.issues.total) *
                                    100}
                                {@const other = 100 - critical - high}
                                <div
                                    class="w-full h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden flex"
                                >
                                    <div
                                        class="h-full bg-red-500"
                                        style="width: {critical}%"
                                    ></div>
                                    <div
                                        class="h-full bg-orange-500"
                                        style="width: {high}%"
                                    ></div>
                                    <div
                                        class="h-full bg-blue-500"
                                        style="width: {other}%"
                                    ></div>
                                </div>
                                <div class="flex justify-between text-xs mt-1">
                                    <span class="text-red-600 dark:text-red-400"
                                        >Critical: {project.metrics.issues
                                            .critical}</span
                                    >
                                    <span
                                        class="text-orange-600 dark:text-orange-400"
                                        >High: {project.metrics.issues
                                            .high}</span
                                    >
                                    <span
                                        class="text-blue-600 dark:text-blue-400"
                                        >Other: {project.metrics.issues.total -
                                            project.metrics.issues.critical -
                                            project.metrics.issues.high}</span
                                    >
                                </div>
                            {/if}
                        </div>
                    {/each}
                </div>
            </div>
        </div>
    {/if}
</div>
