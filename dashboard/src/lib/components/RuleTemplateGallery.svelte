<script lang="ts">
    import type { RuleTemplate } from "$lib/types/dashboard";
    import { createEventDispatcher } from "svelte";

    export let templates: RuleTemplate[] = [];

    const dispatch = createEventDispatcher();
    let selectedCategory = "all";

    function useTemplate(template: RuleTemplate) {
        dispatch("useTemplate", template);
    }

    $: filteredTemplates =
        selectedCategory === "all"
            ? templates
            : templates.filter((t) => t.category === selectedCategory);

    $: categories = ["all", ...new Set(templates.map((t) => t.category))];

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
    <div class="flex flex-wrap items-center justify-between gap-4">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">
            Rule Templates Gallery
        </h3>

        <!-- Category filter -->
        <div class="flex flex-wrap gap-2">
            {#each categories as category}
                <button
                    on:click={() => (selectedCategory = category)}
                    class="px-3 py-1.5 rounded-full text-sm font-medium transition-colors
                           {selectedCategory === category
                        ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                        : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'}"
                >
                    {#if category !== "all"}
                        <span class="mr-1">{getCategoryIcon(category)}</span>
                    {/if}
                    <span class="capitalize">{category}</span>
                </button>
            {/each}
        </div>
    </div>

    {#if templates.length === 0}
        <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-8 text-center">
            <svg
                class="w-16 h-16 mx-auto text-gray-400 mb-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
            >
                <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
                />
            </svg>
            <p class="text-gray-600 dark:text-gray-400 mb-2">
                No templates available
            </p>
            <p class="text-sm text-gray-500">
                Templates will help you create rules quickly
            </p>
        </div>
    {:else if filteredTemplates.length === 0}
        <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-8 text-center">
            <p class="text-gray-600 dark:text-gray-400">
                No templates found for the selected category
            </p>
            <button
                on:click={() => (selectedCategory = "all")}
                class="mt-3 text-blue-600 hover:text-blue-800"
            >
                Show all templates
            </button>
        </div>
    {:else}
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {#each filteredTemplates as template (template.id)}
                <div
                    class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm overflow-hidden"
                >
                    <!-- Template header -->
                    <div
                        class="p-4 border-b border-gray-200 dark:border-gray-700"
                    >
                        <div class="flex items-start justify-between">
                            <div class="flex items-center space-x-3">
                                <span class="text-2xl"
                                    >{getCategoryIcon(template.category)}</span
                                >
                                <h4
                                    class="font-medium text-gray-900 dark:text-white"
                                >
                                    {template.name}
                                </h4>
                            </div>
                            <span
                                class="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full"
                            >
                                {template.language}
                            </span>
                        </div>
                    </div>

                    <!-- Template body -->
                    <div class="p-4">
                        <p
                            class="text-gray-600 dark:text-gray-400 text-sm mb-4"
                        >
                            {template.description}
                        </p>

                        <div class="mb-4">
                            <h5
                                class="text-xs uppercase tracking-wider text-gray-500 mb-2"
                            >
                                Pattern Example
                            </h5>
                            <div
                                class="bg-gray-50 dark:bg-gray-700 rounded p-3 text-sm font-mono overflow-x-auto"
                            >
                                {template.patternExample}
                            </div>
                        </div>

                        <div class="flex items-center justify-between mt-4">
                            <div class="flex items-center space-x-1">
                                <span
                                    class="text-xs px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full capitalize"
                                >
                                    {template.category}
                                </span>
                                <span
                                    class="text-xs px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full capitalize"
                                >
                                    {template.severity}
                                </span>
                            </div>

                            <button
                                on:click={() => useTemplate(template)}
                                class="text-sm px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                            >
                                Use Template
                            </button>
                        </div>
                    </div>
                </div>
            {/each}
        </div>
    {/if}

    <!-- Tips section -->
    <div
        class="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-lg p-6 mt-8"
    >
        <h4 class="font-medium text-gray-900 dark:text-white mb-3">
            Tips for Creating Effective Rules
        </h4>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
                <h5
                    class="text-sm font-medium text-gray-800 dark:text-gray-200 mb-2"
                >
                    Rule Creation Best Practices
                </h5>
                <ul class="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                    <li>• Be specific about what pattern to detect</li>
                    <li>• Start with a small scope and expand if needed</li>
                    <li>• Test rules against real code examples</li>
                    <li>• Focus on high-impact issues first</li>
                </ul>
            </div>

            <div>
                <h5
                    class="text-sm font-medium text-gray-800 dark:text-gray-200 mb-2"
                >
                    Things to Avoid
                </h5>
                <ul class="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                    <li>• Overly broad patterns causing false positives</li>
                    <li>• Very complex rules that are hard to maintain</li>
                    <li>• Rules that contradict coding standards</li>
                    <li>• Duplicating existing built-in rules</li>
                </ul>
            </div>
        </div>
    </div>
</div>
