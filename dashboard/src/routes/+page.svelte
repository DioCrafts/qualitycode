<script lang="ts">
    import { page } from "$app/stores";
    import AutoFixPanel from "$lib/components/AutoFixPanel.svelte";
    import CICDDashboard from "$lib/components/CICDDashboard.svelte";
    import CustomRulesDashboard from "$lib/components/CustomRulesDashboard.svelte";
    import EmbeddingsView from "$lib/components/EmbeddingsView.svelte";
    import MultiProjectView from "$lib/components/MultiProjectView.svelte";
    import TechnicalDebtDashboard from "$lib/components/TechnicalDebtDashboard.svelte";
    import { dashboardStore } from "$lib/stores/dashboard";
    // Import the PageData type from +page.ts
    import type { PageData as LoadedPageData } from "./+page";

    export let data: LoadedPageData;

    type ViewType =
        | "debt"
        | "fixes"
        | "multi"
        | "cicd"
        | "embeddings"
        | "rules";
    let activeView: ViewType = "debt";

    // Get view from URL params if present
    $: if ($page.url.searchParams.has("view")) {
        const view = $page.url.searchParams.get("view");
        if (
            view &&
            ["debt", "fixes", "multi", "cicd", "embeddings", "rules"].includes(
                view,
            )
        ) {
            activeView = view as ViewType;
        }
    }

    function getViewIcon(view: string): string {
        const icons = {
            debt: "💸",
            fixes: "🔧",
            multi: "🏢",
            cicd: "🚀",
            embeddings: "🧠",
            rules: "📝",
        };
        return icons[view as keyof typeof icons] || "📊";
    }

    function getViewTitle(view: string): string {
        const titles = {
            debt: "Technical Debt",
            fixes: "Auto Fixes",
            multi: "Multi-Project",
            cicd: "CI/CD & DORA",
            embeddings: "Semantic Analysis",
            rules: "Custom Rules",
        };
        return titles[view as keyof typeof titles] || "Dashboard";
    }
</script>

<svelte:head>
    <title>CodeAnt Dashboard - {getViewTitle(activeView)}</title>
</svelte:head>

<div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Top Navigation -->
    <nav
        class="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700"
    >
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between h-16">
                <div class="flex items-center">
                    <h1
                        class="text-xl font-bold text-gray-900 dark:text-white flex items-center space-x-2"
                    >
                        <span class="text-2xl">🐜</span>
                        <span>CodeAnt Dashboard</span>
                    </h1>
                </div>

                <div class="flex items-center space-x-4">
                    <!-- Project Selector -->
                    {#if data.projects && data.projects.length > 0}
                        <select
                            value={$dashboardStore.selectedProject?.id || ""}
                            on:change={(e) => {
                                const project = data.projects.find(
                                    (p) => p.id === e.currentTarget.value,
                                );
                                dashboardStore.setProject(project || null);
                            }}
                            class="px-3 py-2 border border-gray-300 dark:border-gray-600
                     rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
                        >
                            <option value="">Select Project</option>
                            {#each data.projects as project}
                                <option value={project.id}
                                    >{project.name}</option
                                >
                            {/each}
                        </select>
                    {/if}

                    <!-- Time Range Selector -->
                    <select
                        value={$dashboardStore.selectedTimeRange.period}
                        on:change={(e) => {
                            const period = e.currentTarget.value as
                                | "1d"
                                | "7d"
                                | "30d"
                                | "90d";
                            const end = new Date();
                            const start = new Date();

                            switch (period) {
                                case "1d":
                                    start.setDate(end.getDate() - 1);
                                    break;
                                case "7d":
                                    start.setDate(end.getDate() - 7);
                                    break;
                                case "30d":
                                    start.setDate(end.getDate() - 30);
                                    break;
                                case "90d":
                                    start.setDate(end.getDate() - 90);
                                    break;
                            }

                            dashboardStore.setTimeRange({ period, start, end });
                        }}
                        class="px-3 py-2 border border-gray-300 dark:border-gray-600
                   rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
                    >
                        <option value="1d">Last 24 hours</option>
                        <option value="7d">Last 7 days</option>
                        <option value="30d">Last 30 days</option>
                        <option value="90d">Last 90 days</option>
                    </select>

                    <!-- User Menu -->
                    <div class="flex items-center space-x-2">
                        <span class="text-sm text-gray-600 dark:text-gray-400">
                            {data.user?.name || "User"}
                        </span>
                        <div
                            class="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center"
                        >
                            <span class="text-white text-sm font-medium">
                                {data.user?.name?.charAt(0) || "U"}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </nav>

    <!-- Dashboard Navigation Tabs -->
    <div class="bg-white dark:bg-gray-800 shadow-sm">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <nav class="flex space-x-8 overflow-x-auto">
                {#each ["debt", "fixes", "multi", "cicd", "embeddings", "rules"] as view}
                    <button
                        on:click={() => (activeView = view as ViewType)}
                        class="whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm
                   transition-colors flex items-center space-x-2
                   {activeView === view
                            ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}"
                    >
                        <span class="text-lg">{getViewIcon(view)}</span>
                        <span>{getViewTitle(view)}</span>
                    </button>
                {/each}
            </nav>
        </div>
    </div>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {#if activeView === "debt"}
            <TechnicalDebtDashboard />
        {:else if activeView === "fixes"}
            <AutoFixPanel />
        {:else if activeView === "multi"}
            <MultiProjectView organizationId={data.organizationId} />
        {:else if activeView === "cicd"}
            <CICDDashboard />
        {:else if activeView === "embeddings"}
            <EmbeddingsView
                projectId={$dashboardStore.selectedProject?.id || ""}
            />
        {:else if activeView === "rules"}
            <CustomRulesDashboard
                projectId={$dashboardStore.selectedProject?.id || ""}
            />
        {/if}
    </main>
</div>

<style>
    /* Custom scrollbar for navigation */
    nav::-webkit-scrollbar {
        height: 4px;
    }

    nav::-webkit-scrollbar-track {
        background: transparent;
    }

    nav::-webkit-scrollbar-thumb {
        background: #cbd5e0;
        border-radius: 2px;
    }

    :global(.dark) nav::-webkit-scrollbar-thumb {
        background: #4a5568;
    }
</style>
