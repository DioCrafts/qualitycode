<script lang="ts">
    import { onMount } from "svelte";
    import type { SemanticSearchResult } from "$lib/types/dashboard";
    
    export let projectId: string;
    
    let searchQuery = '';
    let sourceLanguage = 'any';
    let targetLanguage = 'any';
    let results: SemanticSearchResult[] = [];
    let loading = false;
    let error: string | null = null;
    
    const languageOptions = [
        { value: 'any', label: 'Any Language' },
        { value: 'typescript', label: 'TypeScript' },
        { value: 'javascript', label: 'JavaScript' },
        { value: 'python', label: 'Python' },
        { value: 'java', label: 'Java' },
        { value: 'csharp', label: 'C#' },
        { value: 'go', label: 'Go' },
        { value: 'rust', label: 'Rust' }
    ];
    
    async function performSearch() {
        if (!searchQuery.trim()) return;
        
        loading = true;
        error = null;
        
        try {
            // In a real app, this would be an API call
            // Simulating API delay
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Generate mock results
            results = Array(5).fill(0).map((_, i) => {
                const isSource = Math.random() > 0.5;
                return {
                    id: `result-${i}`,
                    filePath: `src/${isSource ? 'frontend' : 'backend'}/example_${i + 1}.${isSource ? 'ts' : 'py'}`,
                    functionName: `${isSource ? 'process' : 'handle'}Data${i + 1}`,
                    similarity: 0.9 - (i * 0.1),
                    snippet: isSource 
                        ? `function processData${i + 1}(data: Record<string, any>) {\n  return data.map(item => {\n    return {\n      id: item.id,\n      value: item.value * 2\n    };\n  });\n}`
                        : `def handle_data${i + 1}(data):\n    return [\n        {"id": item["id"], "value": item["value"] * 2}\n        for item in data\n    ]`,
                    explanation: `This function processes a list of items by doubling their values. It is semantically equivalent to ${isSource ? 'Python' : 'TypeScript'} functions that perform the same transformation.`
                };
            });
            
            loading = false;
        } catch (err) {
            error = err instanceof Error ? err.message : 'Search failed';
            loading = false;
        }
    }
</script>

<div class="space-y-6">
    <div class="flex justify-between items-center">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">Cross-Language Search</h3>
    </div>
    
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div class="space-y-4">
            <div>
                <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Search for semantically similar code across languages
                </label>
                <div class="relative">
                    <input
                        type="text"
                        bind:value={searchQuery}
                        placeholder="Search for code functionality, e.g., 'filter array of objects'"
                        class="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 
                               rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
                        on:keydown={(e) => e.key === 'Enter' && performSearch()}
                    />
                    <svg class="absolute left-3 top-3.5 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Source Language
                    </label>
                    <select
                        bind:value={sourceLanguage}
                        class="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 
                               rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
                    >
                        {#each languageOptions as option}
                            <option value={option.value}>{option.label}</option>
                        {/each}
                    </select>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Target Language
                    </label>
                    <select
                        bind:value={targetLanguage}
                        class="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 
                               rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
                    >
                        {#each languageOptions as option}
                            <option value={option.value}>{option.label}</option>
                        {/each}
                    </select>
                </div>
            </div>
            
            <div class="flex justify-end">
                <button
                    on:click={performSearch}
                    disabled={loading || !searchQuery.trim()}
                    class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 
                           disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                    {#if loading}
                        <span class="flex items-center space-x-2">
                            <svg class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <span>Searching...</span>
                        </span>
                    {:else}
                        Search
                    {/if}
                </button>
            </div>
        </div>
    </div>
    
    <!-- Results section -->
    {#if loading}
        <div class="flex items-center justify-center h-64">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
    {:else if error}
        <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-4">
            <p class="text-red-800 dark:text-red-200">Error: {error}</p>
        </div>
    {:else if results.length > 0}
        <h4 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Cross-Language Search Results
        </h4>
        
        <div class="space-y-6">
            {#each results as result (result.id)}
                <div class="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
                    <div class="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 p-4 border-b border-gray-200 dark:border-gray-700">
                        <div class="flex justify-between items-start mb-1">
                            <h5 class="font-medium text-gray-900 dark:text-white">
                                {result.functionName || 'Unnamed function'}
                            </h5>
                            <span class="text-sm text-gray-500 dark:text-gray-400">
                                Similarity: {Math.round(result.similarity * 100)}%
                            </span>
                        </div>
                        <p class="text-sm text-gray-600 dark:text-gray-400">
                            {result.filePath}
                        </p>
                    </div>
                    
                    <div class="p-4">
                        <pre class="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm">
                            <code class={`language-${result.filePath.endsWith('.ts') || result.filePath.endsWith('.js') ? 'javascript' : result.filePath.endsWith('.py') ? 'python' : 'plaintext'}`}>
{result.snippet}
                            </code>
                        </pre>
                        
                        <div class="mt-4 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                            <h6 class="text-sm font-medium text-gray-900 dark:text-white mb-2">
                                Semantic Analysis
                            </h6>
                            <p class="text-gray-700 dark:text-gray-300 text-sm">
                                {result.explanation}
                            </p>
                        </div>
                        
                        <div class="mt-4 flex justify-end space-x-3">
                            <button class="text-sm text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200">
                                Find More Like This
                            </button>
                            <button class="text-sm text-blue-600 hover:text-blue-800">
                                View in Context
                            </button>
                        </div>
                    </div>
                </div>
            {/each}
        </div>
    {:else if searchQuery.trim()}
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-8 text-center">
            <svg class="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p class="text-gray-700 dark:text-gray-300 mb-2">No cross-language matches found</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">
                Try a different search query or adjust your language filters.
            </p>
        </div>
    {:else}
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-8 text-center">
            <svg class="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16l2.879-2.879m0 0a3 3 0 104.243-4.242 3 3 0 00-4.243 4.242zM21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p class="text-gray-700 dark:text-gray-300 mb-2">Cross-language search uses AI embeddings</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">
                Find semantically equivalent code across different programming languages.
            </p>
        </div>
    {/if}
    
    <!-- How it works -->
    <div class="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
        <h4 class="text-lg font-medium text-gray-900 dark:text-white mb-4">How Cross-Language Search Works</h4>
        
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
                <div class="flex items-center space-x-3 mb-3">
                    <div class="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
                        <span class="font-medium">1</span>
                    </div>
                    <h5 class="font-medium text-gray-900 dark:text-white">Semantic Encoding</h5>
                </div>
                <p class="text-sm text-gray-600 dark:text-gray-400">
                    Code from all languages is encoded into a universal semantic embedding space using AI models like CodeBERT.
                </p>
            </div>
            
            <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
                <div class="flex items-center space-x-3 mb-3">
                    <div class="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
                        <span class="font-medium">2</span>
                    </div>
                    <h5 class="font-medium text-gray-900 dark:text-white">Intent Mapping</h5>
                </div>
                <p class="text-sm text-gray-600 dark:text-gray-400">
                    We map code intent and functionality rather than syntax, allowing cross-language matches based on behavior.
                </p>
            </div>
            
            <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
                <div class="flex items-center space-x-3 mb-3">
                    <div class="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
                        <span class="font-medium">3</span>
                    </div>
                    <h5 class="font-medium text-gray-900 dark:text-white">Similarity Matching</h5>
                </div>
                <p class="text-sm text-gray-600 dark:text-gray-400">
                    Your search is transformed into the same embedding space and we find the closest matches regardless of language.
                </p>
            </div>
        </div>
    </div>
</div>
