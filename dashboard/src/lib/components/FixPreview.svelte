<script lang="ts">
    import type { AutoFix } from "$lib/types/dashboard";
    import { createEventDispatcher } from "svelte";

    export let fix: AutoFix;

    const dispatch = createEventDispatcher();

    function close() {
        dispatch("close");
    }

    function apply() {
        dispatch("apply");
    }
</script>

<!-- Modal Overlay -->
<div
    class="fixed inset-0 bg-black bg-opacity-50 z-40 flex items-center justify-center p-4"
>
    <!-- Modal Content -->
    <div
        class="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col"
    >
        <!-- Modal Header -->
        <div
            class="border-b border-gray-200 dark:border-gray-700 p-4 flex items-center justify-between"
        >
            <div>
                <h3 class="text-lg font-medium text-gray-900 dark:text-white">
                    Fix Preview
                </h3>
                <p class="text-sm text-gray-500 dark:text-gray-400">
                    {fix.filePath}
                </p>
            </div>
            <div>
                <span
                    class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                >
                    {fix.type}
                </span>
                <span
                    class="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
                >
                    {fix.severity}
                </span>
                <button
                    class="ml-4 text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
                    on:click={close}
                >
                    <svg
                        class="h-6 w-6"
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
                </button>
            </div>
        </div>

        <!-- Modal Body -->
        <div class="p-6 overflow-y-auto flex-grow">
            <!-- Fix Description -->
            <div class="mb-6">
                <h4
                    class="text-lg font-medium text-gray-900 dark:text-white mb-2"
                >
                    {fix.title}
                </h4>
                <p class="text-gray-700 dark:text-gray-300 mb-4">
                    {fix.description}
                </p>

                <div class="flex items-center space-x-2 text-sm text-gray-500">
                    <span>Confidence: {Math.round(fix.confidence * 100)}%</span>
                    <span class="text-gray-400">•</span>
                    <span>Lines: {fix.startLine}-{fix.endLine}</span>
                    <span class="text-gray-400">•</span>
                    <span>Impact: {fix.impact.linesChanged} lines</span>
                </div>
            </div>

            <!-- Code Comparison -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <h5
                        class="font-medium text-gray-700 dark:text-gray-300 mb-2"
                    >
                        Original Code
                    </h5>
                    <pre
                        class="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-auto max-h-80 whitespace-pre-wrap"><code
                            class="language-javascript">{fix.originalCode}</code
                        ></pre>
                </div>
                <div>
                    <h5
                        class="font-medium text-green-700 dark:text-green-300 mb-2"
                    >
                        Fixed Code
                    </h5>
                    <pre
                        class="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg text-sm overflow-auto max-h-80 whitespace-pre-wrap"><code
                            class="language-javascript">{fix.proposedCode}</code
                        ></pre>
                </div>
            </div>

            <!-- Impact Analysis -->
            <div class="mt-6">
                <h5 class="font-medium text-gray-900 dark:text-white mb-3">
                    Impact Analysis
                </h5>
                <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <p class="text-sm text-gray-500 dark:text-gray-400">
                                Lines Changed
                            </p>
                            <p
                                class="text-lg font-medium text-gray-900 dark:text-white"
                            >
                                {fix.impact.linesChanged}
                            </p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-500 dark:text-gray-400">
                                Tests Affected
                            </p>
                            <p
                                class="text-lg font-medium text-gray-900 dark:text-white"
                            >
                                {fix.impact.testsAffected}
                            </p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-500 dark:text-gray-400">
                                Dependencies
                            </p>
                            <p
                                class="text-lg font-medium text-gray-900 dark:text-white"
                            >
                                {fix.impact.dependencies.length}
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Explanation -->
            {#if fix.explanation}
                <div class="mt-6">
                    <h5 class="font-medium text-gray-900 dark:text-white mb-2">
                        Explanation
                    </h5>
                    <div class="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                        <p class="text-gray-700 dark:text-gray-300">
                            {fix.explanation}
                        </p>
                    </div>
                </div>
            {/if}

            <!-- Educational Content -->
            {#if fix.educationalContent}
                <div class="mt-6">
                    <h5 class="font-medium text-gray-900 dark:text-white mb-2">
                        Learn More
                    </h5>
                    <div
                        class="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg"
                    >
                        <p class="text-gray-700 dark:text-gray-300">
                            {fix.educationalContent}
                        </p>
                    </div>
                </div>
            {/if}
        </div>

        <!-- Modal Footer -->
        <div
            class="border-t border-gray-200 dark:border-gray-700 p-4 flex justify-end space-x-3"
        >
            <button
                class="px-4 py-2 bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 rounded-lg"
                on:click={close}
            >
                Close
            </button>
            <button
                class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center space-x-1"
                on:click={apply}
            >
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
                        d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                    />
                </svg>
                <span>Add to Selection</span>
            </button>
        </div>
    </div>
</div>
