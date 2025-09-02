<script lang="ts">
    import type { Pipeline } from "$lib/types/dashboard";
    import { formatDuration } from "$lib/utils/formatters";
    import { createEventDispatcher } from "svelte";

    export let pipelines: Pipeline[];

    const dispatch = createEventDispatcher();
    let selectedPipeline: Pipeline | null = null;
    let filterStatus: string = "all";

    $: filteredPipelines =
        filterStatus === "all"
            ? pipelines
            : pipelines.filter((p) => p.status === filterStatus);

    function selectPipeline(pipeline: Pipeline) {
        selectedPipeline = pipeline;
        dispatch("select", pipeline);
    }

    function getStatusColor(status: string): string {
        const colors = {
            success: "bg-green-100 text-green-800",
            failed: "bg-red-100 text-red-800",
            running: "bg-blue-100 text-blue-800",
            pending: "bg-gray-100 text-gray-800",
        };
        return (
            colors[status as keyof typeof colors] || "bg-gray-100 text-gray-800"
        );
    }

    function getStageStatusColor(status: string): string {
        const colors = {
            success: "text-green-600",
            failed: "text-red-600",
            running: "text-blue-600",
            pending: "text-gray-600",
            skipped: "text-gray-400",
        };
        return colors[status as keyof typeof colors] || "text-gray-600";
    }
</script>

<div class="space-y-6">
    <div class="flex justify-between items-center">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">
            Pipeline Monitor
        </h3>

        <div class="flex items-center space-x-3">
            <div class="flex items-center space-x-2">
                <span class="text-sm text-gray-500 dark:text-gray-400"
                    >Filter:</span
                >
                <select
                    bind:value={filterStatus}
                    class="text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-2 py-1 focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
                >
                    <option value="all">All Status</option>
                    <option value="running">Running</option>
                    <option value="success">Success</option>
                    <option value="failed">Failed</option>
                    <option value="pending">Pending</option>
                </select>
            </div>

            <button
                class="text-sm text-blue-600 hover:text-blue-800 flex items-center"
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
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Pipeline list -->
        <div>
            {#if filteredPipelines.length === 0}
                <div class="text-center py-8">
                    <p class="text-gray-500 dark:text-gray-400">
                        No pipelines match your filter
                    </p>
                </div>
            {:else}
                <div class="space-y-3 max-h-96 overflow-y-auto pr-2">
                    {#each filteredPipelines as pipeline (pipeline.id)}
                        <div
                            class="border border-gray-200 dark:border-gray-700 rounded-lg p-4 cursor-pointer transition-all hover:shadow-md"
                            class:bg-blue-50={selectedPipeline?.id ===
                                pipeline.id}
                            class:border-blue-300={selectedPipeline?.id ===
                                pipeline.id}
                            class:dark:bg-blue-900={selectedPipeline?.id ===
                                pipeline.id}
                            class:dark:border-blue-700={selectedPipeline?.id ===
                                pipeline.id}
                            on:click={() => selectPipeline(pipeline)}
                        >
                            <div class="flex justify-between items-center">
                                <div>
                                    <h4
                                        class="font-medium text-gray-900 dark:text-white flex items-center"
                                    >
                                        {pipeline.name}
                                        {#if pipeline.status === "running"}
                                            <div
                                                class="ml-2 flex items-center space-x-1"
                                            >
                                                <div
                                                    class="animate-pulse w-1.5 h-1.5 bg-blue-600 rounded-full"
                                                ></div>
                                                <div
                                                    class="animate-pulse w-1.5 h-1.5 bg-blue-600 rounded-full delay-75"
                                                ></div>
                                                <div
                                                    class="animate-pulse w-1.5 h-1.5 bg-blue-600 rounded-full delay-150"
                                                ></div>
                                            </div>
                                        {/if}
                                    </h4>
                                    <p
                                        class="text-sm text-gray-500 dark:text-gray-400"
                                    >
                                        Last run: {new Date(
                                            pipeline.lastRun,
                                        ).toLocaleString()}
                                    </p>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <span
                                        class="px-2 py-1 text-xs font-medium rounded capitalize {getStatusColor(
                                            pipeline.status,
                                        )}"
                                    >
                                        {pipeline.status}
                                    </span>
                                    <span
                                        class="text-sm text-gray-500 dark:text-gray-400"
                                    >
                                        {formatDuration(pipeline.duration)}
                                    </span>
                                </div>
                            </div>

                            <div class="mt-3 flex space-x-1">
                                {#each pipeline.stages as stage, i}
                                    <div
                                        class="flex-1 h-1.5 rounded-full {getStageStatusColor(
                                            stage.status,
                                        )} bg-current opacity-50"
                                        class:opacity-100={stage.status ===
                                            "running" ||
                                            stage.status === "failed"}
                                        title={`${stage.name}: ${stage.status}`}
                                    ></div>
                                {/each}
                            </div>
                        </div>
                    {/each}
                </div>
            {/if}
        </div>

        <!-- Pipeline details -->
        <div>
            {#if selectedPipeline}
                <div
                    class="border border-gray-200 dark:border-gray-700 rounded-lg p-4 bg-white dark:bg-gray-800"
                >
                    <div class="flex justify-between items-start mb-4">
                        <div>
                            <h4
                                class="font-medium text-lg text-gray-900 dark:text-white"
                            >
                                {selectedPipeline.name}
                            </h4>
                            <p class="text-sm text-gray-500 dark:text-gray-400">
                                ID: {selectedPipeline.id}
                            </p>
                        </div>
                        <span
                            class="px-2 py-1 text-xs font-medium rounded capitalize {getStatusColor(
                                selectedPipeline.status,
                            )}"
                        >
                            {selectedPipeline.status}
                        </span>
                    </div>

                    <!-- Pipeline timeline -->
                    <div class="space-y-4">
                        <h5
                            class="font-medium text-gray-800 dark:text-gray-200"
                        >
                            Pipeline Stages
                        </h5>
                        <div class="space-y-3">
                            {#each selectedPipeline.stages as stage, index}
                                <div class="relative">
                                    {#if index < selectedPipeline.stages.length - 1}
                                        <div
                                            class="absolute left-2.5 top-6 bottom-0 w-0.5 bg-gray-300 dark:bg-gray-600"
                                        ></div>
                                    {/if}

                                    <div class="flex items-start space-x-3">
                                        <div
                                            class={`flex-shrink-0 w-5 h-5 mt-1.5 rounded-full ${getStageStatusColor(stage.status)} bg-current`}
                                        ></div>

                                        <div class="flex-1">
                                            <div
                                                class="flex justify-between items-center"
                                            >
                                                <h6
                                                    class="font-medium text-gray-900 dark:text-white"
                                                >
                                                    {stage.name}
                                                </h6>
                                                <div
                                                    class="text-sm text-gray-500 dark:text-gray-400 flex items-center space-x-2"
                                                >
                                                    <span class="capitalize"
                                                        >{stage.status}</span
                                                    >
                                                    <span>•</span>
                                                    <span
                                                        >{formatDuration(
                                                            stage.duration,
                                                        )}</span
                                                    >
                                                </div>
                                            </div>

                                            {#if stage.status === "failed" && stage.logs && stage.logs.length > 0}
                                                <div
                                                    class="mt-2 bg-red-50 dark:bg-red-900/20 p-3 rounded-lg text-sm"
                                                >
                                                    <div
                                                        class="font-medium text-red-800 dark:text-red-200 mb-1"
                                                    >
                                                        Error:
                                                    </div>
                                                    <pre
                                                        class="text-red-700 dark:text-red-300 overflow-x-auto whitespace-pre-wrap">{stage
                                                            .logs[0]}</pre>
                                                </div>
                                            {:else if stage.status === "running"}
                                                <div
                                                    class="mt-2 bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg text-sm"
                                                >
                                                    <div
                                                        class="font-medium text-blue-800 dark:text-blue-200 flex items-center"
                                                    >
                                                        <svg
                                                            class="animate-spin -ml-1 mr-2 h-4 w-4"
                                                            fill="none"
                                                            viewBox="0 0 24 24"
                                                        >
                                                            <circle
                                                                class="opacity-25"
                                                                cx="12"
                                                                cy="12"
                                                                r="10"
                                                                stroke="currentColor"
                                                                stroke-width="4"
                                                            ></circle>
                                                            <path
                                                                class="opacity-75"
                                                                fill="currentColor"
                                                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                                            ></path>
                                                        </svg>
                                                        Running...
                                                    </div>
                                                </div>
                                            {/if}
                                        </div>
                                    </div>
                                </div>
                            {/each}
                        </div>
                    </div>

                    <!-- Actions -->
                    <div class="mt-6 flex justify-end space-x-3">
                        <button
                            class="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                            View Logs
                        </button>
                        {#if selectedPipeline.status !== "running"}
                            <button
                                class="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded"
                            >
                                Rerun Pipeline
                            </button>
                        {:else}
                            <button
                                class="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded"
                            >
                                Cancel Run
                            </button>
                        {/if}
                    </div>
                </div>
            {:else}
                <div
                    class="border border-gray-200 dark:border-gray-700 border-dashed rounded-lg p-8 text-center h-full flex flex-col items-center justify-center"
                >
                    <svg
                        class="w-12 h-12 text-gray-400 mb-4"
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
                    <p class="text-gray-600 dark:text-gray-400">
                        Select a pipeline to view details
                    </p>
                </div>
            {/if}
        </div>
    </div>
</div>
