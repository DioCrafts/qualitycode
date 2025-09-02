<script lang="ts">
    import type { CustomRule, RuleTemplate } from "$lib/types/dashboard";
    import { onMount } from "svelte";
    import RulePerformanceMetrics from "./RulePerformanceMetrics.svelte";
    import RuleTemplateGallery from "./RuleTemplateGallery.svelte";
    import RuleTestingSandbox from "./RuleTestingSandbox.svelte";

    export let projectId: string;

    let activeTab: "builder" | "performance" | "templates" | "sandbox" =
        "builder";
    let rules: CustomRule[] = [];
    let templates: RuleTemplate[] = [];
    let selectedRule: CustomRule | null = null;
    let loading = false;
    let error: string | null = null;

    // Rule builder state
    let naturalLanguageInput = "";
    let isGenerating = false;
    let generatedRule: Partial<CustomRule> | null = null;

    async function loadRules() {
        loading = true;
        error = null;

        try {
            const response = await fetch(`/api/rules/custom/${projectId}`);
            if (!response.ok) throw new Error("Failed to load rules");

            const data = await response.json();
            rules = data.rules;
        } catch (err) {
            error = err instanceof Error ? err.message : "Failed to load rules";
        } finally {
            loading = false;
        }
    }

    async function loadTemplates() {
        try {
            const response = await fetch("/api/rules/templates");
            if (!response.ok) throw new Error("Failed to load templates");

            const data = await response.json();
            templates = data.templates;
        } catch (err) {
            console.error("Failed to load templates:", err);
        }
    }

    async function generateRuleFromNaturalLanguage() {
        if (!naturalLanguageInput.trim()) return;

        isGenerating = true;
        error = null;

        try {
            const response = await fetch("/api/rules/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    naturalLanguageInput,
                    projectId,
                    language: "auto", // Auto-detect from input language
                }),
            });

            if (!response.ok) throw new Error("Failed to generate rule");

            const data = await response.json();
            generatedRule = data.rule;
        } catch (err) {
            error =
                err instanceof Error ? err.message : "Failed to generate rule";
        } finally {
            isGenerating = false;
        }
    }

    async function saveRule(rule: Partial<CustomRule>) {
        try {
            const response = await fetch("/api/rules/custom", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ...rule,
                    projectId,
                }),
            });

            if (!response.ok) throw new Error("Failed to save rule");

            await loadRules(); // Refresh rules list
            generatedRule = null;
            naturalLanguageInput = "";
        } catch (err) {
            error = err instanceof Error ? err.message : "Failed to save rule";
        }
    }

    async function toggleRuleStatus(ruleId: string) {
        const rule = rules.find((r) => r.id === ruleId);
        if (!rule) return;

        try {
            const response = await fetch(`/api/rules/custom/${ruleId}/status`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    status: rule.status === "active" ? "inactive" : "active",
                }),
            });

            if (!response.ok) throw new Error("Failed to update rule status");

            await loadRules();
        } catch (err) {
            error =
                err instanceof Error ? err.message : "Failed to update rule";
        }
    }

    onMount(() => {
        loadRules();
        loadTemplates();
    });

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

    function getStatusColor(status: string): string {
        const colors = {
            active: "text-green-600 bg-green-100",
            inactive: "text-gray-600 bg-gray-100",
            testing: "text-yellow-600 bg-yellow-100",
        };
        return (
            colors[status as keyof typeof colors] || "text-gray-600 bg-gray-100"
        );
    }
</script>

<div class="space-y-6">
    <!-- Header -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div class="flex justify-between items-start">
            <div>
                <h2
                    class="text-2xl font-bold text-gray-900 dark:text-white mb-2"
                >
                    Custom Rules Dashboard
                </h2>
                <p class="text-gray-600 dark:text-gray-400">
                    Create and manage custom analysis rules using natural
                    language
                </p>
            </div>
            <div class="flex items-center space-x-2">
                <span
                    class="px-3 py-1 bg-blue-100 text-blue-800 dark:bg-blue-900
                     dark:text-blue-200 rounded-full text-sm font-medium"
                >
                    {rules.filter((r) => r.status === "active").length} Active Rules
                </span>
            </div>
        </div>
    </div>

    <!-- Tabs -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div class="border-b border-gray-200 dark:border-gray-700">
            <nav class="flex -mb-px">
                <button
                    on:click={() => (activeTab = "builder")}
                    class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                 {activeTab === 'builder'
                        ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700'}"
                >
                    Rule Builder
                </button>
                <button
                    on:click={() => (activeTab = "performance")}
                    class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                 {activeTab === 'performance'
                        ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700'}"
                >
                    Performance ({rules.length})
                </button>
                <button
                    on:click={() => (activeTab = "templates")}
                    class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                 {activeTab === 'templates'
                        ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700'}"
                >
                    Templates ({templates.length})
                </button>
                <button
                    on:click={() => (activeTab = "sandbox")}
                    class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                 {activeTab === 'sandbox'
                        ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700'}"
                >
                    Testing Sandbox
                </button>
            </nav>
        </div>

        <div class="p-6">
            {#if loading && activeTab === "performance"}
                <div class="flex items-center justify-center h-64">
                    <div
                        class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"
                    ></div>
                </div>
            {:else if error}
                <div
                    class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-4"
                >
                    <p class="text-red-800 dark:text-red-200">Error: {error}</p>
                </div>
            {/if}

            {#if activeTab === "builder"}
                <!-- Natural Language Rule Builder -->
                <div class="space-y-6">
                    <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
                        <h3
                            class="text-lg font-medium text-gray-900 dark:text-white mb-4"
                        >
                            Create Rule from Natural Language
                        </h3>

                        <div class="space-y-4">
                            <div>
                                <label
                                    class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
                                >
                                    Describe your rule in natural language:
                                </label>
                                <textarea
                                    bind:value={naturalLanguageInput}
                                    placeholder="Example: 'No permitir funciones con más de 20 líneas de código' or 'Functions should not have more than 3 parameters'"
                                    rows="4"
                                    class="w-full px-4 py-3 border border-gray-300 dark:border-gray-600
                         rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-800
                         resize-none"
                                />
                            </div>

                            <div class="flex items-center justify-between">
                                <p
                                    class="text-sm text-gray-500 dark:text-gray-400"
                                >
                                    💡 Tip: You can write in Spanish or English.
                                    Be specific about what you want to detect.
                                </p>

                                <button
                                    on:click={generateRuleFromNaturalLanguage}
                                    disabled={isGenerating ||
                                        !naturalLanguageInput.trim()}
                                    class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700
                         disabled:opacity-50 disabled:cursor-not-allowed transition-colors
                         flex items-center space-x-2"
                                >
                                    {#if isGenerating}
                                        <svg
                                            class="animate-spin h-4 w-4"
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
                                        <span>Generating...</span>
                                    {:else}
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
                                                d="M13 10V3L4 14h7v7l9-11h-7z"
                                            />
                                        </svg>
                                        <span>Generate Rule</span>
                                    {/if}
                                </button>
                            </div>
                        </div>
                    </div>

                    {#if generatedRule}
                        <div
                            class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200
                        dark:border-blue-800 rounded-lg p-6"
                        >
                            <h4
                                class="text-lg font-medium text-blue-900 dark:text-blue-100 mb-4"
                            >
                                Generated Rule Preview
                            </h4>

                            <div class="space-y-4">
                                <div>
                                    <label
                                        class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                                    >
                                        Rule Name
                                    </label>
                                    <input
                                        type="text"
                                        bind:value={generatedRule.name}
                                        class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600
                           rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
                                    />
                                </div>

                                <div>
                                    <label
                                        class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                                    >
                                        Description
                                    </label>
                                    <textarea
                                        bind:value={generatedRule.description}
                                        rows="2"
                                        class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600
                           rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700
                           resize-none"
                                    />
                                </div>

                                <div>
                                    <label
                                        class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                                    >
                                        Generated Pattern
                                    </label>
                                    <pre
                                        class="bg-gray-100 dark:bg-gray-800 rounded p-3 text-sm overflow-x-auto">
                    <code>{generatedRule.generatedPattern}</code>
                  </pre>
                                </div>

                                <div class="grid grid-cols-2 gap-4">
                                    <div>
                                        <label
                                            class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                                        >
                                            Category
                                        </label>
                                        <select
                                            bind:value={generatedRule.category}
                                            class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600
                             rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
                                        >
                                            <option value="quality"
                                                >Quality</option
                                            >
                                            <option value="security"
                                                >Security</option
                                            >
                                            <option value="performance"
                                                >Performance</option
                                            >
                                            <option value="style">Style</option>
                                            <option value="documentation"
                                                >Documentation</option
                                            >
                                        </select>
                                    </div>

                                    <div>
                                        <label
                                            class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                                        >
                                            Severity
                                        </label>
                                        <select
                                            bind:value={generatedRule.severity}
                                            class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600
                             rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
                                        >
                                            <option value="error">Error</option>
                                            <option value="warning"
                                                >Warning</option
                                            >
                                            <option value="info">Info</option>
                                        </select>
                                    </div>
                                </div>

                                <div class="flex justify-end space-x-3 pt-4">
                                    <button
                                        on:click={() => (generatedRule = null)}
                                        class="px-4 py-2 border border-gray-300 dark:border-gray-600
                           text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50
                           dark:hover:bg-gray-700 transition-colors"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        on:click={() => (activeTab = "sandbox")}
                                        class="px-4 py-2 border border-blue-500 text-blue-600
                           rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20
                           transition-colors"
                                    >
                                        Test in Sandbox
                                    </button>
                                    <button
                                        on:click={() => saveRule(generatedRule)}
                                        class="px-4 py-2 bg-blue-600 text-white rounded-lg
                           hover:bg-blue-700 transition-colors"
                                    >
                                        Save Rule
                                    </button>
                                </div>
                            </div>
                        </div>
                    {/if}

                    <!-- Recent Rules -->
                    {#if rules.length > 0}
                        <div>
                            <h4
                                class="text-lg font-medium text-gray-900 dark:text-white mb-4"
                            >
                                Recent Custom Rules
                            </h4>
                            <div class="space-y-3">
                                {#each rules.slice(0, 5) as rule (rule.id)}
                                    <div
                                        class="border border-gray-200 dark:border-gray-700 rounded-lg p-4
                              hover:shadow transition-shadow"
                                    >
                                        <div
                                            class="flex items-center justify-between"
                                        >
                                            <div
                                                class="flex items-center space-x-3"
                                            >
                                                <span class="text-xl"
                                                    >{getCategoryIcon(
                                                        rule.category,
                                                    )}</span
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
                                            <div
                                                class="flex items-center space-x-2"
                                            >
                                                <span
                                                    class="px-2 py-1 text-xs font-medium rounded {getStatusColor(
                                                        rule.status,
                                                    )}"
                                                >
                                                    {rule.status}
                                                </span>
                                                <button
                                                    on:click={() =>
                                                        toggleRuleStatus(
                                                            rule.id,
                                                        )}
                                                    class="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                                                >
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
                                                            d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                                                        />
                                                        <path
                                                            stroke-linecap="round"
                                                            stroke-linejoin="round"
                                                            stroke-width="2"
                                                            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                                        />
                                                    </svg>
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                {/each}
                            </div>
                        </div>
                    {/if}
                </div>
            {:else if activeTab === "performance"}
                <RulePerformanceMetrics {rules} />
            {:else if activeTab === "templates"}
                <RuleTemplateGallery
                    {templates}
                    on:useTemplate={(e) => {
                        naturalLanguageInput = e.detail.description;
                        activeTab = "builder";
                    }}
                />
            {:else if activeTab === "sandbox"}
                <RuleTestingSandbox
                    rule={generatedRule || selectedRule}
                    {projectId}
                />
            {/if}
        </div>
    </div>
</div>
