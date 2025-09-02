<script lang="ts">
    import type { FixBatch } from "$lib/types/dashboard";

    export let batches: FixBatch[];

    let selectedBatch: FixBatch | null = null;

    function getBatchStatusColor(status: string): string {
        const colors = {
            pending: "bg-yellow-100 text-yellow-800",
            in_progress: "bg-blue-100 text-blue-800",
            completed: "bg-green-100 text-green-800",
            failed: "bg-red-100 text-red-800",
        };
        return (
            colors[status as keyof typeof colors] || "bg-gray-100 text-gray-800"
        );
    }
</script>

<div class="space-y-4">
    <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
        Fix Batches
    </h3>

    {#if batches.length === 0}
        <div class="text-center py-8">
            <p class="text-gray-500 dark:text-gray-400">
                No fix batches available
            </p>
            <p class="text-sm text-gray-500 dark:text-gray-400 mt-2">
                Select fixes from the Available Fixes tab and apply them to
                create a batch
            </p>
        </div>
    {:else}
        <div class="grid gap-4 grid-cols-1 md:grid-cols-2">
            {#each batches as batch (batch.id)}
                <div
                    class="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                    class:bg-blue-50={selectedBatch?.id === batch.id}
                    on:click={() => (selectedBatch = batch)}
                >
                    <div class="flex justify-between items-start mb-3">
                        <div>
                            <h4
                                class="font-medium text-gray-900 dark:text-white"
                            >
                                {batch.name}
                            </h4>
                            <p class="text-sm text-gray-500 dark:text-gray-400">
                                {batch.fixes.length} fixes
                            </p>
                        </div>
                        <span
                            class="px-2 py-1 text-xs font-medium rounded capitalize {getBatchStatusColor(
                                batch.status,
                            )}"
                        >
                            {batch.status.replace("_", " ")}
                        </span>
                    </div>

                    <div class="grid grid-cols-3 gap-2 text-center text-sm">
                        <div>
                            <p class="text-gray-500 dark:text-gray-400">
                                Files
                            </p>
                            <p class="font-medium">
                                {batch.totalImpact.filesAffected}
                            </p>
                        </div>
                        <div>
                            <p class="text-gray-500 dark:text-gray-400">
                                Lines
                            </p>
                            <p class="font-medium">
                                {batch.totalImpact.linesChanged}
                            </p>
                        </div>
                        <div>
                            <p class="text-gray-500 dark:text-gray-400">
                                Issues
                            </p>
                            <p class="font-medium">
                                {batch.totalImpact.issuesResolved}
                            </p>
                        </div>
                    </div>
                </div>
            {/each}
        </div>

        {#if selectedBatch}
            <div
                class="mt-6 border-t border-gray-200 dark:border-gray-700 pt-6"
            >
                <h4
                    class="text-lg font-medium text-gray-900 dark:text-white mb-4"
                >
                    Batch Details: {selectedBatch.name}
                </h4>

                <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mb-4">
                    <div
                        class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center"
                    >
                        <div>
                            <p class="text-sm text-gray-500 dark:text-gray-400">
                                Files Affected
                            </p>
                            <p class="text-lg font-medium">
                                {selectedBatch.totalImpact.filesAffected}
                            </p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-500 dark:text-gray-400">
                                Lines Changed
                            </p>
                            <p class="text-lg font-medium">
                                {selectedBatch.totalImpact.linesChanged}
                            </p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-500 dark:text-gray-400">
                                Issues Resolved
                            </p>
                            <p class="text-lg font-medium">
                                {selectedBatch.totalImpact.issuesResolved}
                            </p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-500 dark:text-gray-400">
                                Est. Time
                            </p>
                            <p class="text-lg font-medium">
                                {selectedBatch.estimatedTime} mins
                            </p>
                        </div>
                    </div>
                </div>

                <div class="space-y-3">
                    <h5 class="font-medium text-gray-800 dark:text-gray-200">
                        Fixes in this batch
                    </h5>
                    {#each selectedBatch.fixes as fix (fix.id)}
                        <div
                            class="border border-gray-200 dark:border-gray-700 rounded-lg p-3 bg-white dark:bg-gray-800"
                        >
                            <div class="flex justify-between items-center mb-1">
                                <span
                                    class="font-medium text-gray-900 dark:text-white"
                                    >{fix.title}</span
                                >
                                <span
                                    class="text-xs px-2 py-0.5 bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full"
                                >
                                    {fix.type}
                                </span>
                            </div>
                            <p class="text-sm text-gray-600 dark:text-gray-400">
                                {fix.filePath}
                            </p>
                        </div>
                    {/each}
                </div>

                <div class="flex justify-end mt-4 space-x-3">
                    <button
                        class="px-4 py-2 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-100 dark:text-gray-300 dark:border-gray-600 dark:hover:bg-gray-700"
                    >
                        View Details
                    </button>
                    <button
                        class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                    >
                        Apply Batch
                    </button>
                </div>
            </div>
        {/if}
    {/if}
</div>
