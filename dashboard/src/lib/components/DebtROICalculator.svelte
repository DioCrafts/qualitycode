<script lang="ts">
    import type {
        DebtComponent,
        TechnicalDebtMetrics,
    } from "$lib/types/dashboard";
    import { formatCurrency } from "$lib/utils/formatters";

    export let components: DebtComponent[];
    export let metrics: TechnicalDebtMetrics;

    $: roiData = components
        .map((component) => ({
            component: component.name,
            currentDebt: component.debt,
            remediationCost: component.debt * 100, // $100 per hour
            expectedBenefit: component.debt * 150, // 1.5x ROI
            roi: 0.5,
            priority:
                component.criticalIssues > 5
                    ? "critical"
                    : component.criticalIssues > 2
                      ? "high"
                      : component.debt > 100
                        ? "medium"
                        : "low",
        }))
        .sort((a, b) => {
            const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
            return priorityOrder[a.priority] - priorityOrder[b.priority];
        });

    function getPriorityColor(priority: string): string {
        const colors = {
            critical: "text-red-600 bg-red-100",
            high: "text-orange-600 bg-orange-100",
            medium: "text-yellow-600 bg-yellow-100",
            low: "text-green-600 bg-green-100",
        };
        return (
            colors[priority as keyof typeof colors] ||
            "text-gray-600 bg-gray-100"
        );
    }
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <h3 class="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
        ROI Calculator
    </h3>

    <div class="space-y-4">
        <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <div class="grid grid-cols-2 gap-4 text-sm">
                <div>
                    <p class="text-gray-600 dark:text-gray-400">
                        Total Remediation Cost
                    </p>
                    <p
                        class="text-xl font-bold text-blue-600 dark:text-blue-400"
                    >
                        {formatCurrency(metrics.estimatedCost)}
                    </p>
                </div>
                <div>
                    <p class="text-gray-600 dark:text-gray-400">Expected ROI</p>
                    <p
                        class="text-xl font-bold text-green-600 dark:text-green-400"
                    >
                        {formatCurrency(metrics.estimatedCost * 1.5)}
                    </p>
                </div>
            </div>
        </div>

        <div class="space-y-3">
            <h4 class="font-medium text-gray-700 dark:text-gray-300">
                Priority Components
            </h4>
            {#each roiData.slice(0, 5) as item}
                <div
                    class="flex items-center justify-between p-3 border border-gray-200
                    dark:border-gray-700 rounded-lg"
                >
                    <div class="flex items-center space-x-3">
                        <span
                            class="px-2 py-1 text-xs font-medium rounded {getPriorityColor(
                                item.priority,
                            )}"
                        >
                            {item.priority}
                        </span>
                        <span class="font-medium text-gray-900 dark:text-white">
                            {item.component}
                        </span>
                    </div>
                    <div class="text-right text-sm">
                        <p class="text-gray-600 dark:text-gray-400">
                            Cost: {formatCurrency(item.remediationCost)}
                        </p>
                        <p
                            class="text-green-600 dark:text-green-400 font-medium"
                        >
                            ROI: {(item.roi * 100).toFixed(0)}%
                        </p>
                    </div>
                </div>
            {/each}
        </div>
    </div>
</div>
