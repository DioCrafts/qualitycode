<script lang="ts">
    import type { CodeCluster } from "$lib/types/dashboard";

    export let clusters: CodeCluster[] = [];
    export let loading = false;
    export let selectedCluster: CodeCluster | null = null;
    export let similarityThreshold = 0.7;

    // Calculate node positions for a simple visualization
    // In a real app, this would use D3 force layout or similar
    function calculateNodePositions(clusters: CodeCluster[]) {
        const nodes: Array<{
            x: number;
            y: number;
            r: number;
            cluster: CodeCluster;
        }> = [];
        const centerX = 400;
        const centerY = 300;

        clusters.forEach((cluster, i) => {
            const angle = (i / clusters.length) * Math.PI * 2;
            const radius = 200;
            const x = centerX + Math.cos(angle) * radius;
            const y = centerY + Math.sin(angle) * radius;
            const r = 30 + cluster.files.length / 3;

            nodes.push({
                x,
                y,
                r,
                cluster,
            });

            // Add child nodes for files in cluster
            cluster.files.forEach((file, j) => {
                const childAngle = angle + (j - cluster.files.length / 2) * 0.2;
                const childRadius = radius + 80;
                const childX = centerX + Math.cos(childAngle) * childRadius;
                const childY = centerY + Math.sin(childAngle) * childRadius;

                nodes.push({
                    x: childX,
                    y: childY,
                    r: 8,
                    cluster,
                });
            });
        });

        return nodes;
    }

    // Calculate connections between nodes
    function calculateConnections(clusters: CodeCluster[], threshold: number) {
        const connections: Array<{
            source: number;
            target: number;
            strength: number;
        }> = [];

        // Connect between main cluster nodes
        for (let i = 0; i < clusters.length; i++) {
            for (let j = i + 1; j < clusters.length; j++) {
                // Simulate similarity score
                const similarity = Math.random();
                if (similarity > threshold) {
                    connections.push({
                        source: i,
                        target: j,
                        strength: similarity,
                    });
                }
            }
        }

        return connections;
    }
</script>

<div class="space-y-6">
    <div class="flex justify-between items-center">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">
            Code Similarity Map
        </h3>

        <div class="flex items-center space-x-3">
            <div class="flex items-center space-x-2">
                <span class="text-sm text-gray-600 dark:text-gray-400"
                    >Threshold:</span
                >
                <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.1"
                    bind:value={similarityThreshold}
                    class="w-24"
                />
                <span class="text-sm text-gray-900 dark:text-white"
                    >{similarityThreshold.toFixed(1)}</span
                >
            </div>
        </div>
    </div>

    {#if loading}
        <div class="flex items-center justify-center h-96">
            <div
                class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"
            ></div>
        </div>
    {:else if clusters.length === 0}
        <div
            class="bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8 text-center"
        >
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
                    d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
            </svg>
            <p class="text-gray-600 dark:text-gray-400 mb-2">
                No code similarity clusters available
            </p>
            <p class="text-sm text-gray-500 dark:text-gray-500">
                The similarity analysis may not have been run on this project
                yet.
            </p>
        </div>
    {:else}
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Visualization -->
            <div
                class="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden"
            >
                <div class="p-4 border-b border-gray-200 dark:border-gray-700">
                    <h4 class="font-medium text-gray-900 dark:text-white">
                        Cluster Visualization
                    </h4>
                </div>

                <div class="relative" style="height: 600px;">
                    <svg width="100%" height="100%" viewBox="0 0 800 600">
                        {#if clusters.length > 0}
                            {@const nodes = calculateNodePositions(clusters)}
                            {@const connections = calculateConnections(
                                clusters,
                                similarityThreshold,
                            )}

                            <!-- Draw connections -->
                            {#each connections as conn}
                                <line
                                    x1={nodes[conn.source].x}
                                    y1={nodes[conn.source].y}
                                    x2={nodes[conn.target].x}
                                    y2={nodes[conn.target].y}
                                    stroke-width={conn.strength * 3}
                                    stroke="rgba(79, 70, 229, 0.3)"
                                    stroke-dasharray="3,3"
                                />
                            {/each}

                            <!-- Draw nodes -->
                            {#each nodes as node, i}
                                <!-- Only draw main cluster nodes and selected cluster's file nodes -->
                                {#if i < clusters.length || node.cluster.id === selectedCluster?.id}
                                    <circle
                                        cx={node.x}
                                        cy={node.y}
                                        r={node.r}
                                        fill={node.cluster.id ===
                                        selectedCluster?.id
                                            ? "rgba(79, 70, 229, 0.8)"
                                            : "rgba(79, 70, 229, 0.2)"}
                                        stroke={node.cluster.id ===
                                        selectedCluster?.id
                                            ? "rgba(79, 70, 229, 1)"
                                            : "rgba(79, 70, 229, 0.5)"}
                                        stroke-width="2"
                                        class="cursor-pointer hover:opacity-80 transition-opacity"
                                        on:click={() =>
                                            (selectedCluster = node.cluster)}
                                    />

                                    <!-- Label for main cluster nodes -->
                                    {#if i < clusters.length}
                                        <text
                                            x={node.x}
                                            y={node.y}
                                            text-anchor="middle"
                                            dominant-baseline="middle"
                                            fill="currentColor"
                                            class="text-xs pointer-events-none select-none"
                                        >
                                            {node.cluster.name}
                                        </text>
                                    {/if}
                                {/if}
                            {/each}
                        {/if}
                    </svg>
                </div>
            </div>

            <!-- Cluster Details -->
            <div class="bg-white dark:bg-gray-800 rounded-lg shadow">
                <div class="p-4 border-b border-gray-200 dark:border-gray-700">
                    <h4 class="font-medium text-gray-900 dark:text-white">
                        Cluster Details
                    </h4>
                </div>

                <div class="p-4">
                    {#if selectedCluster}
                        <div class="space-y-4">
                            <div>
                                <h5
                                    class="font-medium text-gray-900 dark:text-white text-lg mb-1"
                                >
                                    {selectedCluster.name}
                                </h5>
                                <p class="text-gray-600 dark:text-gray-400">
                                    {selectedCluster.description}
                                </p>
                            </div>

                            <div>
                                <h6
                                    class="font-medium text-gray-700 dark:text-gray-300 mb-2"
                                >
                                    Cluster Metrics
                                </h6>
                                <div class="grid grid-cols-2 gap-4">
                                    <div
                                        class="bg-gray-50 dark:bg-gray-700 rounded p-3"
                                    >
                                        <div
                                            class="text-sm text-gray-500 dark:text-gray-400"
                                        >
                                            Files
                                        </div>
                                        <div
                                            class="text-lg font-medium text-gray-900 dark:text-white"
                                        >
                                            {selectedCluster.files.length}
                                        </div>
                                    </div>
                                    <div
                                        class="bg-gray-50 dark:bg-gray-700 rounded p-3"
                                    >
                                        <div
                                            class="text-sm text-gray-500 dark:text-gray-400"
                                        >
                                            Cohesion
                                        </div>
                                        <div
                                            class="text-lg font-medium text-gray-900 dark:text-white"
                                        >
                                            {(
                                                selectedCluster.cohesion * 100
                                            ).toFixed(0)}%
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div>
                                <h6
                                    class="font-medium text-gray-700 dark:text-gray-300 mb-2"
                                >
                                    Main Purpose
                                </h6>
                                <div
                                    class="bg-blue-50 dark:bg-blue-900/20 rounded p-3 text-gray-800 dark:text-gray-200"
                                >
                                    {selectedCluster.mainPurpose}
                                </div>
                            </div>

                            <div>
                                <h6
                                    class="font-medium text-gray-700 dark:text-gray-300 mb-2"
                                >
                                    Files in Cluster
                                </h6>
                                <div class="max-h-40 overflow-y-auto pr-2">
                                    {#each selectedCluster.files as file, i}
                                        <div
                                            class="flex items-center text-sm py-1 border-b border-gray-100 dark:border-gray-700 last:border-0"
                                        >
                                            <span
                                                class="w-6 text-right text-gray-400 mr-2"
                                                >{i + 1}.</span
                                            >
                                            <span
                                                class="text-gray-800 dark:text-gray-200 truncate"
                                                >{file}</span
                                            >
                                        </div>
                                    {/each}
                                </div>
                            </div>

                            <div class="flex justify-end">
                                <button
                                    class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
                                >
                                    View Cluster Details
                                </button>
                            </div>
                        </div>
                    {:else}
                        <div class="text-center py-10">
                            <p class="text-gray-600 dark:text-gray-400">
                                Select a cluster to view details
                            </p>
                        </div>
                    {/if}
                </div>
            </div>
        </div>

        <!-- Cluster List -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow">
            <div class="p-4 border-b border-gray-200 dark:border-gray-700">
                <h4 class="font-medium text-gray-900 dark:text-white">
                    All Clusters
                </h4>
            </div>

            <div class="p-4">
                <div
                    class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
                >
                    {#each clusters as cluster (cluster.id)}
                        <div
                            class="border border-gray-200 dark:border-gray-700 rounded-lg p-4 cursor-pointer transition-all"
                            class:bg-blue-50={selectedCluster?.id ===
                                cluster.id}
                            class:border-blue-300={selectedCluster?.id ===
                                cluster.id}
                            on:click={() => (selectedCluster = cluster)}
                        >
                            <h5
                                class="font-medium text-gray-900 dark:text-white mb-1 flex items-center justify-between"
                            >
                                <span>{cluster.name}</span>
                                <span
                                    class="text-xs px-2 py-0.5 bg-gray-100 dark:bg-gray-700 rounded-full"
                                >
                                    {cluster.files.length} files
                                </span>
                            </h5>
                            <p
                                class="text-sm text-gray-600 dark:text-gray-400 line-clamp-2"
                            >
                                {cluster.description}
                            </p>
                            <div class="mt-2 text-xs">
                                <span
                                    class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200"
                                >
                                    {cluster.mainPurpose}
                                </span>
                            </div>
                        </div>
                    {/each}
                </div>
            </div>
        </div>
    {/if}
</div>
