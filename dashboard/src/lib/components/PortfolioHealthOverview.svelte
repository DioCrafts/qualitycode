<script lang="ts">
    import type { PortfolioHealth } from "$lib/types/dashboard";
    import { formatNumber } from "$lib/utils/formatters";

    export let health: PortfolioHealth;

    function getTrendIcon(trend: "up" | "down" | "stable"): string {
        return trend === "up" ? "↗️" : trend === "down" ? "↘️" : "→";
    }

    function getTrendColor(
        trend: "up" | "down" | "stable",
        metric: string,
    ): string {
        // For some metrics, up is good (quality), for others down is good (debt, issues)
        const upIsGood = ["quality"];
        const downIsGood = ["debt", "issues"];

        if (trend === "stable") return "text-gray-600 dark:text-gray-400";

        if (upIsGood.includes(metric)) {
            return trend === "up"
                ? "text-green-600 dark:text-green-400"
                : "text-red-600 dark:text-red-400";
        } else if (downIsGood.includes(metric)) {
            return trend === "down"
                ? "text-green-600 dark:text-green-400"
                : "text-red-600 dark:text-red-400";
        }

        return "text-gray-600 dark:text-gray-400";
    }
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <h3 class="text-xl font-semibold text-gray-900 dark:text-white mb-4">
        Portfolio Health Overview
    </h3>

    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div>
            <div class="flex flex-col items-center">
                <div
                    class="mb-2 text-3xl font-bold text-gray-900 dark:text-white"
                >
                    {health.averageQualityScore.toFixed(1)}%
                </div>
                <div
                    class="text-sm font-medium text-gray-500 dark:text-gray-400"
                >
                    Average Quality Score
                </div>
                <div class="mt-1 flex items-center">
                    <span
                        class={getTrendColor(health.trends.quality, "quality")}
                    >
                        {getTrendIcon(health.trends.quality)}
                    </span>
                    <span
                        class="ml-1 text-sm {getTrendColor(
                            health.trends.quality,
                            'quality',
                        )}"
                    >
                        {health.trends.quality === "up"
                            ? "Improving"
                            : health.trends.quality === "down"
                              ? "Declining"
                              : "Stable"}
                    </span>
                </div>
            </div>

            <div
                class="mt-4 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3"
            >
                <div
                    class="h-3 rounded-full"
                    class:bg-red-500={health.averageQualityScore < 60}
                    class:bg-yellow-500={health.averageQualityScore >= 60 &&
                        health.averageQualityScore < 80}
                    class:bg-green-500={health.averageQualityScore >= 80}
                    style="width: {health.averageQualityScore}%"
                ></div>
            </div>
        </div>

        <div class="space-y-4">
            <div class="flex justify-between items-center">
                <div>
                    <div
                        class="text-sm font-medium text-gray-500 dark:text-gray-400"
                    >
                        Total Projects
                    </div>
                    <div
                        class="text-2xl font-bold text-gray-900 dark:text-white"
                    >
                        {health.totalProjects}
                    </div>
                </div>
                <div>
                    <div
                        class="text-sm font-medium text-gray-500 dark:text-gray-400"
                    >
                        Healthy Projects
                    </div>
                    <div
                        class="text-2xl font-bold text-green-600 dark:text-green-400"
                    >
                        {health.healthyProjects}
                    </div>
                </div>
            </div>

            <div>
                <div class="flex justify-between text-sm mb-1">
                    <span class="text-gray-500 dark:text-gray-400"
                        >Project Health Distribution</span
                    >
                    <span class="text-gray-700 dark:text-gray-300">
                        {Math.round(
                            (health.healthyProjects / health.totalProjects) *
                                100,
                        )}% healthy
                    </span>
                </div>
                {#if health.totalProjects > 0}
                    {@const healthyPercent =
                        (health.healthyProjects / health.totalProjects) * 100}
                    {@const warningPercent =
                        ((health.totalProjects -
                            health.healthyProjects -
                            health.criticalIssues / 5) /
                            health.totalProjects) *
                        100}
                    {@const criticalPercent =
                        100 - healthyPercent - warningPercent}
                    <div
                        class="w-full h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden flex"
                    >
                        <div
                            class="h-full bg-green-500"
                            style="width: {healthyPercent}%"
                        ></div>
                        <div
                            class="h-full bg-yellow-500"
                            style="width: {warningPercent}%"
                        ></div>
                        <div
                            class="h-full bg-red-500"
                            style="width: {criticalPercent}%"
                        ></div>
                    </div>
                    <div class="flex justify-between text-xs mt-1">
                        <span class="text-green-600 dark:text-green-400"
                            >Healthy: {health.healthyProjects}</span
                        >
                        <span class="text-yellow-600 dark:text-yellow-400"
                            >Warning: {health.totalProjects -
                                health.healthyProjects -
                                Math.ceil(health.criticalIssues / 5)}</span
                        >
                        <span class="text-red-600 dark:text-red-400"
                            >Critical: {Math.ceil(
                                health.criticalIssues / 5,
                            )}</span
                        >
                    </div>
                {/if}
            </div>
        </div>

        <div class="space-y-4">
            <div>
                <div
                    class="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1"
                >
                    Total Technical Debt
                </div>
                <div
                    class="text-2xl font-bold text-gray-900 dark:text-white mb-1"
                >
                    {formatNumber(health.totalTechnicalDebt)}h
                </div>
                <div class="flex items-center">
                    <span class={getTrendColor(health.trends.debt, "debt")}>
                        {getTrendIcon(health.trends.debt)}
                    </span>
                    <span
                        class="ml-1 text-sm {getTrendColor(
                            health.trends.debt,
                            'debt',
                        )}"
                    >
                        {health.trends.debt === "down"
                            ? "Decreasing"
                            : health.trends.debt === "up"
                              ? "Increasing"
                              : "Stable"}
                    </span>
                </div>
            </div>

            <div>
                <div
                    class="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1"
                >
                    Critical Issues
                </div>
                <div
                    class="text-2xl font-bold text-red-600 dark:text-red-400 mb-1"
                >
                    {health.criticalIssues}
                </div>
                <div class="flex items-center">
                    <span class={getTrendColor(health.trends.issues, "issues")}>
                        {getTrendIcon(health.trends.issues)}
                    </span>
                    <span
                        class="ml-1 text-sm {getTrendColor(
                            health.trends.issues,
                            'issues',
                        )}"
                    >
                        {health.trends.issues === "down"
                            ? "Decreasing"
                            : health.trends.issues === "up"
                              ? "Increasing"
                              : "Stable"}
                    </span>
                </div>
            </div>
        </div>
    </div>
</div>
