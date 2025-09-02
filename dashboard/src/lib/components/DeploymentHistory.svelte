<script lang="ts">
    import type { TimeRange } from "$lib/types/dashboard";
    import { formatDate } from "$lib/utils/formatters";
    import { onMount } from "svelte";

    export let projectId: string;
    export let timeRange: TimeRange;

    // Mock deployment data
    interface Deployment {
        id: string;
        version: string;
        environment: "production" | "staging" | "development";
        timestamp: Date;
        duration: number; // in seconds
        status: "success" | "failed" | "in_progress";
        changes: number;
        author: string;
        commit: string;
    }

    let deployments: Deployment[] = [];
    let loading = false;
    let error: string | null = null;

    onMount(async () => {
        await loadDeployments();
    });

    async function loadDeployments() {
        loading = true;
        error = null;

        try {
            // In a real app, this would be an API call
            // Simulating API delay
            await new Promise((resolve) => setTimeout(resolve, 500));

            // Generate mock data
            deployments = Array(10)
                .fill(0)
                .map((_, i) => {
                    const date = new Date();
                    date.setDate(date.getDate() - i);

                    return {
                        id: `deploy-${i}`,
                        version: `v1.${Math.floor(Math.random() * 10)}.${Math.floor(Math.random() * 100)}`,
                        environment:
                            i % 5 === 0
                                ? "development"
                                : i % 3 === 0
                                  ? "staging"
                                  : "production",
                        timestamp: date,
                        duration: Math.floor(Math.random() * 300) + 60, // 1-6 minutes
                        status:
                            i === 0
                                ? "in_progress"
                                : Math.random() > 0.8
                                  ? "failed"
                                  : "success",
                        changes: Math.floor(Math.random() * 20) + 1,
                        author: "Developer",
                        commit: `${Math.random().toString(16).slice(2, 10)}`,
                    };
                });

            loading = false;
        } catch (err) {
            error =
                err instanceof Error
                    ? err.message
                    : "Failed to load deployments";
            loading = false;
        }
    }

    function getStatusColor(status: string): string {
        const colors = {
            success: "bg-green-100 text-green-800",
            failed: "bg-red-100 text-red-800",
            in_progress: "bg-blue-100 text-blue-800",
        };
        return (
            colors[status as keyof typeof colors] || "bg-gray-100 text-gray-800"
        );
    }

    function getEnvironmentColor(env: string): string {
        const colors = {
            production: "bg-purple-100 text-purple-800",
            staging: "bg-amber-100 text-amber-800",
            development: "bg-teal-100 text-teal-800",
        };
        return (
            colors[env as keyof typeof colors] || "bg-gray-100 text-gray-800"
        );
    }
</script>

<div class="space-y-6">
    <div class="flex justify-between items-center">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">
            Deployment History
        </h3>

        <button
            class="text-sm text-blue-600 hover:text-blue-800 flex items-center"
            on:click={loadDeployments}
        >
            <svg
                class="w-4 h-4 mr-1"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
            >
                <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
            </svg>
            Refresh
        </button>
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
            <p class="text-red-800 dark:text-red-200">
                Error loading deployments: {error}
            </p>
            <button
                class="mt-2 text-sm text-red-600 hover:text-red-800"
                on:click={loadDeployments}
            >
                Try again
            </button>
        </div>
    {:else if deployments.length === 0}
        <div class="text-center py-8">
            <p class="text-gray-500 dark:text-gray-400">
                No deployments found in the selected time range
            </p>
        </div>
    {:else}
        <!-- Deployment timeline -->
        <div class="relative">
            <div
                class="absolute left-9 top-0 bottom-0 w-0.5 bg-gray-200 dark:bg-gray-700"
            ></div>

            <div class="space-y-8">
                {#each deployments as deployment (deployment.id)}
                    <div class="relative flex gap-6">
                        <div class="flex flex-col items-center">
                            <div
                                class={`w-8 h-8 rounded-full flex items-center justify-center 
                                          ${
                                              deployment.status === "success"
                                                  ? "bg-green-100 text-green-600 dark:bg-green-900/20 dark:text-green-400"
                                                  : deployment.status ===
                                                      "failed"
                                                    ? "bg-red-100 text-red-600 dark:bg-red-900/20 dark:text-red-400"
                                                    : "bg-blue-100 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400"
                                          }`}
                            >
                                {#if deployment.status === "success"}
                                    <svg
                                        class="w-5 h-5"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                    >
                                        <path
                                            stroke-linecap="round"
                                            stroke-linejoin="round"
                                            stroke-width="2"
                                            d="M5 13l4 4L19 7"
                                        />
                                    </svg>
                                {:else if deployment.status === "failed"}
                                    <svg
                                        class="w-5 h-5"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                    >
                                        <path
                                            stroke-linecap="round"
                                            stroke-linejoin="round"
                                            stroke-width="2"
                                            d="M6 18L18 6M6 6l12 12"
                                        />
                                    </svg>
                                {:else}
                                    <svg
                                        class="w-5 h-5 animate-spin"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                    >
                                        <path
                                            stroke-linecap="round"
                                            stroke-linejoin="round"
                                            stroke-width="2"
                                            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                                        />
                                    </svg>
                                {/if}
                            </div>
                            <div class="mt-2 text-xs text-gray-500">
                                {formatDate(deployment.timestamp, "short")}
                            </div>
                        </div>

                        <div
                            class="flex-1 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4"
                        >
                            <div class="flex justify-between items-start">
                                <div>
                                    <div class="flex items-center space-x-2">
                                        <h4
                                            class="font-medium text-gray-900 dark:text-white"
                                        >
                                            {deployment.version}
                                        </h4>
                                        <span
                                            class="px-2 py-0.5 text-xs font-medium rounded capitalize {getStatusColor(
                                                deployment.status,
                                            )}"
                                        >
                                            {deployment.status.replace(
                                                "_",
                                                " ",
                                            )}
                                        </span>
                                        <span
                                            class="px-2 py-0.5 text-xs font-medium rounded capitalize {getEnvironmentColor(
                                                deployment.environment,
                                            )}"
                                        >
                                            {deployment.environment}
                                        </span>
                                    </div>
                                    <p
                                        class="text-sm text-gray-600 dark:text-gray-400 mt-1"
                                    >
                                        {deployment.changes}
                                        {deployment.changes === 1
                                            ? "change"
                                            : "changes"} by {deployment.author}
                                    </p>
                                </div>
                                <div class="text-sm text-gray-500">
                                    {deployment.duration}s
                                </div>
                            </div>

                            <div
                                class="mt-3 flex items-center space-x-4 text-sm"
                            >
                                <span class="text-gray-600 dark:text-gray-400">
                                    Commit: <code
                                        class="text-xs bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded"
                                        >{deployment.commit}</code
                                    >
                                </span>

                                <div class="flex space-x-2">
                                    <button
                                        class="text-blue-600 hover:text-blue-800"
                                        >View Details</button
                                    >
                                    {#if deployment.status === "failed"}
                                        <button
                                            class="text-blue-600 hover:text-blue-800"
                                            >View Logs</button
                                        >
                                    {:else if deployment.environment === "production"}
                                        <button
                                            class="text-green-600 hover:text-green-800"
                                            >Release Notes</button
                                        >
                                    {/if}
                                </div>
                            </div>
                        </div>
                    </div>
                {/each}
            </div>
        </div>

        <!-- Pagination -->
        <div class="flex justify-center mt-8">
            <nav class="flex items-center space-x-1">
                <button
                    class="px-3 py-1 rounded border border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                    Previous
                </button>
                <button class="px-3 py-1 rounded bg-blue-600 text-white"
                    >1</button
                >
                <button
                    class="px-3 py-1 rounded border border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                    2
                </button>
                <button
                    class="px-3 py-1 rounded border border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                    3
                </button>
                <span class="px-2 text-gray-500 dark:text-gray-400">...</span>
                <button
                    class="px-3 py-1 rounded border border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                    Next
                </button>
            </nav>
        </div>
    {/if}
</div>
