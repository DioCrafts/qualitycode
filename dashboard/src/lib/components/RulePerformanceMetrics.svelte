<script lang="ts">
    import type { CustomRule } from "$lib/types/dashboard";

    export let rules: CustomRule[];

    // Compute metrics
    $: totalFindings = rules.reduce(
        (sum, rule) => sum + (rule.metrics?.findings || 0),
        0,
    );
    $: activatedRules = rules.filter((rule) => rule.status === "active").length;
    $: inactiveRules = rules.filter((rule) => rule.status !== "active").length;

    // Group by category
    $: rulesByCategory = rules.reduce(
        (acc, rule) => {
            if (!acc[rule.category]) {
                acc[rule.category] = {
                    count: 0,
                    findings: 0,
                    active: 0,
                    inactive: 0,
                };
            }
            acc[rule.category].count++;
            acc[rule.category].findings += rule.metrics?.findings || 0;

            if (rule.status === "active") {
                acc[rule.category].active++;
            } else {
                acc[rule.category].inactive++;
            }

            return acc;
        },
        {} as Record<
            string,
            {
                count: number;
                findings: number;
                active: number;
                inactive: number;
            }
        >,
    );

    $: categoryEntries = Object.entries(rulesByCategory).sort(
        (a, b) => b[1].findings - a[1].findings,
    );

    // Find top rules by findings
    $: topRulesByFindings = [...rules]
        .filter((rule) => rule.metrics?.findings > 0)
        .sort((a, b) => (b.metrics?.findings || 0) - (a.metrics?.findings || 0))
        .slice(0, 5);

    function getCategoryIcon(category: string): string {
        const icons: Record<string, string> = {
            security: "🔒",
            performance: "⚡",
            quality: "✨",
            style: "🎨",
            accessibility: "♿",
            testing: "🧪",
            documentation: "📝",
        };
        return icons[category] || "📋";
    }
</script>

<div class="space-y-6">
    <!-- Overview metrics -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4
                class="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1"
            >
                Total Rules
            </h4>
            <div class="text-3xl font-bold text-gray-900 dark:text-white">
                {rules.length}
            </div>
        </div>

        <div class="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h4
                class="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1"
            >
                Active Rules
            </h4>
            <div class="text-3xl font-bold text-green-600 dark:text-green-400">
                {activatedRules}
            </div>
        </div>

        <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <h4
                class="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1"
            >
                Inactive Rules
            </h4>
            <div class="text-3xl font-bold text-gray-500 dark:text-gray-400">
                {inactiveRules}
            </div>
        </div>

        <div class="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-4">
            <h4
                class="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1"
            >
                Total Findings
            </h4>
            <div class="text-3xl font-bold text-amber-600 dark:text-amber-400">
                {totalFindings}
            </div>
        </div>
    </div>

    <!-- Rules by category -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Rules by Category
        </h3>

        <div class="space-y-4">
            {#if categoryEntries.length === 0}
                <p class="text-center text-gray-500 dark:text-gray-400 py-4">
                    No rule categories found
                </p>
            {:else}
                {#each categoryEntries as [category, data]}
                    <div
                        class="border-b border-gray-100 dark:border-gray-700 last:border-b-0 pb-4 last:pb-0"
                    >
                        <div class="flex justify-between items-start mb-2">
                            <div class="flex items-center space-x-2">
                                <span class="text-2xl"
                                    >{getCategoryIcon(category)}</span
                                >
                                <h4
                                    class="text-lg font-medium text-gray-900 dark:text-white capitalize"
                                >
                                    {category}
                                </h4>
                            </div>
                            <div class="flex items-center space-x-3">
                                <div class="text-center">
                                    <div
                                        class="text-2xl font-bold text-gray-900 dark:text-white"
                                    >
                                        {data.count}
                                    </div>
                                    <div class="text-xs text-gray-500">
                                        rules
                                    </div>
                                </div>
                                <div class="text-center">
                                    <div
                                        class="text-2xl font-bold text-amber-600 dark:text-amber-400"
                                    >
                                        {data.findings}
                                    </div>
                                    <div class="text-xs text-gray-500">
                                        findings
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Activity bars -->
                        <div class="mt-3">
                            <div class="flex justify-between text-sm mb-1">
                                <span class="text-gray-500 dark:text-gray-400"
                                    >Rule Status</span
                                >
                                <span class="text-gray-700 dark:text-gray-300"
                                    >{data.active} active, {data.inactive} inactive</span
                                >
                            </div>
                            <div
                                class="h-2 w-full bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden"
                            >
                                {#if data.count > 0}
                                    {@const activePercent =
                                        (data.active / data.count) * 100}
                                    <div
                                        class="h-full bg-green-500"
                                        style="width: {activePercent}%"
                                    ></div>
                                {/if}
                            </div>
                        </div>
                    </div>
                {/each}
            {/if}
        </div>
    </div>

    <!-- Top rules by findings -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Top Rules by Findings
        </h3>

        {#if topRulesByFindings.length === 0}
            <p class="text-center text-gray-500 dark:text-gray-400 py-4">
                No findings detected
            </p>
        {:else}
            <div class="space-y-4">
                {#each topRulesByFindings as rule}
                    <div
                        class="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 last:border-b-0 pb-3 last:pb-0"
                    >
                        <div class="flex items-center space-x-3">
                            <span class="text-xl"
                                >{getCategoryIcon(rule.category)}</span
                            >
                            <div>
                                <h5
                                    class="font-medium text-gray-900 dark:text-white"
                                >
                                    {rule.name}
                                </h5>
                                <p
                                    class="text-sm text-gray-500 dark:text-gray-400"
                                >
                                    {rule.description}
                                </p>
                            </div>
                        </div>
                        <div class="text-right">
                            <div
                                class="text-lg font-bold text-amber-600 dark:text-amber-400"
                            >
                                {rule.metrics?.findings || 0}
                            </div>
                            <div class="text-xs text-gray-500">findings</div>
                        </div>
                    </div>
                {/each}

                <div class="mt-4 flex justify-center">
                    <button
                        class="text-blue-600 hover:text-blue-800 text-sm flex items-center space-x-1"
                    >
                        <span>View All Rules</span>
                        <svg
                            class="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                stroke-width="2"
                                d="M13 7l5 5m0 0l-5 5m5-5H6"
                            />
                        </svg>
                    </button>
                </div>
            </div>
        {/if}
    </div>

    <!-- Impact metrics -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Rule Impact Metrics
        </h3>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Quality impact -->
            <div
                class="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/10 dark:to-indigo-900/10 rounded-lg p-4"
            >
                <h4 class="font-medium text-gray-900 dark:text-white mb-2">
                    Quality Impact
                </h4>
                <p class="text-gray-600 dark:text-gray-400 text-sm mb-4">
                    Estimated quality improvement from active rules
                </p>

                <div class="flex items-center">
                    <div
                        class="text-3xl font-bold text-blue-600 dark:text-blue-400 mr-3"
                    >
                        +{(totalFindings / 100).toFixed(1)}%
                    </div>
                    <div class="text-sm text-gray-600 dark:text-gray-400">
                        quality score increase from baseline
                    </div>
                </div>
            </div>

            <!-- Time savings -->
            <div
                class="bg-gradient-to-br from-green-50 to-teal-50 dark:from-green-900/10 dark:to-teal-900/10 rounded-lg p-4"
            >
                <h4 class="font-medium text-gray-900 dark:text-white mb-2">
                    Developer Time Savings
                </h4>
                <p class="text-gray-600 dark:text-gray-400 text-sm mb-4">
                    Estimated time saved through early issue detection
                </p>

                <div class="flex items-center">
                    <div
                        class="text-3xl font-bold text-green-600 dark:text-green-400 mr-3"
                    >
                        {(totalFindings * 0.25).toFixed(1)}h
                    </div>
                    <div class="text-sm text-gray-600 dark:text-gray-400">
                        developer hours saved per month
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
