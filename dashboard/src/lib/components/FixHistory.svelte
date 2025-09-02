<script lang="ts">
    import type { FixHistory } from "$lib/types/dashboard";
    import { formatDate } from "$lib/utils/formatters";
    import { onMount } from "svelte";

    export let projectId: string | undefined;

    let history: FixHistory[] = [];
    let loading = false;
    let error: string | null = null;

    onMount(async () => {
        if (projectId) {
            await loadHistory();
        }
    });

    async function loadHistory() {
        loading = true;
        error = null;

        try {
            // In a real app, this would fetch from API
            // For demo, we'll create mock data
            await new Promise((resolve) => setTimeout(resolve, 500));

            history = Array(5)
                .fill(0)
                .map((_, index) => ({
                    id: `history-${index}`,
                    fixId: `fix-${index}`,
                    appliedAt: new Date(
                        Date.now() - index * 24 * 60 * 60 * 1000,
                    ),
                    appliedBy: "Developer",
                    status:
                        index % 3 === 0
                            ? "failed"
                            : index % 5 === 0
                              ? "rolled_back"
                              : "success",
                    beforeMetrics: { qualityScore: 75 - index * 2 },
                    afterMetrics: { qualityScore: 78 - index },
                    rollbackAvailable: index < 3,
                }));

            loading = false;
        } catch (err) {
            error =
                err instanceof Error ? err.message : "Failed to load history";
            loading = false;
        }
    }

    function getStatusColor(status: string): string {
        const colors = {
            success: "bg-green-100 text-green-800",
            failed: "bg-red-100 text-red-800",
            rolled_back: "bg-yellow-100 text-yellow-800",
        };
        return (
            colors[status as keyof typeof colors] || "bg-gray-100 text-gray-800"
        );
    }
</script>

<div class="space-y-4">
    <div class="flex justify-between items-center mb-4">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">
            Fix History
        </h3>

        <button
            class="text-sm text-blue-600 hover:text-blue-800"
            on:click={loadHistory}
        >
            Refresh
        </button>
    </div>

    {#if loading}
        <div class="flex items-center justify-center h-32">
            <div
                class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"
            ></div>
        </div>
    {:else if error}
        <div class="bg-red-50 border border-red-200 rounded-lg p-4">
            <p class="text-red-800">{error}</p>
            <button
                class="mt-2 text-sm text-red-600 hover:text-red-800"
                on:click={loadHistory}
            >
                Try again
            </button>
        </div>
    {:else if history.length === 0}
        <div class="text-center py-8">
            <p class="text-gray-500 dark:text-gray-400">
                No fix history available
            </p>
            <p class="text-sm text-gray-500 dark:text-gray-400 mt-2">
                Fix history will appear here after applying fixes
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
                            Date
                        </th>
                        <th
                            scope="col"
                            class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider"
                        >
                            Fix
                        </th>
                        <th
                            scope="col"
                            class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider"
                        >
                            Status
                        </th>
                        <th
                            scope="col"
                            class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider"
                        >
                            Quality Change
                        </th>
                        <th
                            scope="col"
                            class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider"
                        >
                            Actions
                        </th>
                    </tr>
                </thead>
                <tbody
                    class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700"
                >
                    {#each history as item (item.id)}
                        <tr class="hover:bg-gray-50 dark:hover:bg-gray-700">
                            <td
                                class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white"
                            >
                                {formatDate(item.appliedAt)}
                            </td>
                            <td
                                class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white"
                            >
                                Fix #{item.fixId.replace("fix-", "")}
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span
                                    class="px-2 py-1 text-xs font-medium rounded capitalize {getStatusColor(
                                        item.status,
                                    )}"
                                >
                                    {item.status.replace("_", " ")}
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="flex items-center space-x-2">
                                    <span class="text-sm text-gray-500"
                                        >{item.beforeMetrics
                                            .qualityScore}%</span
                                    >
                                    <svg
                                        class="w-4 h-4 text-gray-400"
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
                                    <span
                                        class="text-sm font-medium text-green-600"
                                        >{item.afterMetrics.qualityScore}%</span
                                    >
                                    <span class="text-xs text-green-600"
                                        >(+{item.afterMetrics.qualityScore -
                                            item.beforeMetrics
                                                .qualityScore}%)</span
                                    >
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                {#if item.rollbackAvailable}
                                    <button
                                        class="text-blue-600 hover:text-blue-800"
                                    >
                                        Rollback
                                    </button>
                                {:else}
                                    <span class="text-gray-400"
                                        >Rollback unavailable</span
                                    >
                                {/if}
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>
    {/if}
</div>
