<script lang="ts">
    import type {
        DebtComponent,
        TechnicalDebtMetrics,
    } from "$lib/types/dashboard";
    import { formatNumber } from "$lib/utils/formatters";

    export let components: DebtComponent[];
    export let metrics: TechnicalDebtMetrics;

    // Create payment phases based on priority
    $: paymentPhases = [
        {
            phase: "Phase 1 - Critical",
            duration: "2-4 weeks",
            components: components
                .filter((c) => c.criticalIssues > 0)
                .slice(0, 3),
            totalDebt: 0,
            impact: "Immediate risk mitigation",
        },
        {
            phase: "Phase 2 - High Impact",
            duration: "1-2 months",
            components: components
                .filter((c) => c.debt > 100 && c.criticalIssues === 0)
                .slice(0, 3),
            totalDebt: 0,
            impact: "Performance improvement",
        },
        {
            phase: "Phase 3 - Maintenance",
            duration: "2-3 months",
            components: components
                .filter((c) => c.debt <= 100 && c.criticalIssues === 0)
                .slice(0, 3),
            totalDebt: 0,
            impact: "Long-term maintainability",
        },
    ].map((phase) => ({
        ...phase,
        totalDebt: phase.components.reduce((sum, c) => sum + c.debt, 0),
    }));

    $: totalPlannedDebt = paymentPhases.reduce(
        (sum, phase) => sum + phase.totalDebt,
        0,
    );
    $: completionPercentage = (totalPlannedDebt / metrics.totalDebt) * 100;
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <h3 class="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
        Debt Payment Roadmap
    </h3>

    <div class="mb-6">
        <div class="flex justify-between text-sm mb-2">
            <span class="text-gray-600 dark:text-gray-400"
                >Roadmap Coverage</span
            >
            <span class="font-medium text-gray-900 dark:text-white">
                {completionPercentage.toFixed(0)}% of total debt
            </span>
        </div>
        <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
                class="h-2 rounded-full bg-gradient-to-r from-blue-500 to-green-500"
                style="width: {completionPercentage}%"
            />
        </div>
    </div>

    <div class="space-y-4">
        {#each paymentPhases as phase, index}
            <div class="relative">
                {#if index < paymentPhases.length - 1}
                    <div
                        class="absolute left-6 top-12 bottom-0 w-0.5 bg-gray-300 dark:bg-gray-600"
                    ></div>
                {/if}

                <div class="flex items-start space-x-3">
                    <div
                        class="flex-shrink-0 w-12 h-12 bg-blue-100 dark:bg-blue-900
                      rounded-full flex items-center justify-center"
                    >
                        <span
                            class="text-blue-600 dark:text-blue-400 font-bold"
                        >
                            {index + 1}
                        </span>
                    </div>

                    <div
                        class="flex-1 bg-gray-50 dark:bg-gray-700 rounded-lg p-4"
                    >
                        <div class="flex justify-between items-start mb-2">
                            <div>
                                <h4
                                    class="font-medium text-gray-900 dark:text-white"
                                >
                                    {phase.phase}
                                </h4>
                                <p
                                    class="text-sm text-gray-500 dark:text-gray-400"
                                >
                                    {phase.duration} • {formatNumber(
                                        phase.totalDebt,
                                    )}h debt
                                </p>
                            </div>
                            <span
                                class="text-sm text-green-600 dark:text-green-400"
                            >
                                {phase.impact}
                            </span>
                        </div>

                        {#if phase.components.length > 0}
                            <div class="mt-3 space-y-1">
                                {#each phase.components as component}
                                    <div
                                        class="text-sm text-gray-600 dark:text-gray-400 flex justify-between"
                                    >
                                        <span>• {component.name}</span>
                                        <span>{component.debt}h</span>
                                    </div>
                                {/each}
                            </div>
                        {/if}
                    </div>
                </div>
            </div>
        {/each}
    </div>

    <div class="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
        <p class="text-sm text-green-800 dark:text-green-200">
            <strong>Estimated completion:</strong> 5-8 months with dedicated resources
        </p>
    </div>
</div>
