<script lang="ts">
  import type { CodeCluster, EmbeddingVisualization, SemanticSearchResult } from '$lib/types/dashboard';
  import { onMount } from 'svelte';
  import CodeSimilarityMap from './CodeSimilarityMap.svelte';
  import CrossLanguageSearch from './CrossLanguageSearch.svelte';
  import EmbeddingsExplorer from './EmbeddingsExplorer.svelte';
  
  export let projectId: string;
  
  let activeTab: 'search' | 'similarity' | 'explorer' | 'cross-language' = 'search';
  let loading = false;
  let error: string | null = null;
  
  // Search state
  let searchQuery = '';
  let searchResults: SemanticSearchResult[] = [];
  let searchMode: 'natural' | 'code' = 'natural';
  let searching = false;
  
  // Similarity state
  let clusters: CodeCluster[] = [];
  let selectedCluster: CodeCluster | null = null;
  let similarityThreshold = 0.7;
  
  // Embeddings visualization
  let embeddingViz: EmbeddingVisualization | null = null;
  let selectedNode: string | null = null;
  
  async function performSemanticSearch() {
    if (!searchQuery.trim()) return;
    
    searching = true;
    error = null;
    
    try {
      const response = await fetch('/api/embeddings/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId,
          query: searchQuery,
          mode: searchMode,
          limit: 20
        })
      });
      
      if (!response.ok) throw new Error('Search failed');
      
      const data = await response.json();
      searchResults = data.results;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Search failed';
    } finally {
      searching = false;
    }
  }
  
  async function loadClusters() {
    loading = true;
    error = null;
    
    try {
      const response = await fetch(`/api/embeddings/clusters/${projectId}`);
      if (!response.ok) throw new Error('Failed to load clusters');
      
      const data = await response.json();
      clusters = data.clusters;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load clusters';
    } finally {
      loading = false;
    }
  }
  
  async function loadEmbeddingVisualization() {
    loading = true;
    error = null;
    
    try {
      const response = await fetch(`/api/embeddings/visualization/${projectId}`);
      if (!response.ok) throw new Error('Failed to load visualization');
      
      const data = await response.json();
      embeddingViz = data;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load visualization';
    } finally {
      loading = false;
    }
  }
  
  onMount(() => {
    if (activeTab === 'similarity') {
      loadClusters();
    } else if (activeTab === 'explorer') {
      loadEmbeddingVisualization();
    }
  });
  
  $: if (activeTab === 'similarity' && clusters.length === 0) {
    loadClusters();
  }
  
  $: if (activeTab === 'explorer' && !embeddingViz) {
    loadEmbeddingVisualization();
  }
  
  function getSearchModeIcon(mode: string): string {
    return mode === 'natural' ? '💬' : '💻';
  }
</script>

<div class="space-y-6">
  <!-- Header -->
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <div class="flex justify-between items-start">
      <div>
        <h2 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Semantic Code Analysis
        </h2>
        <p class="text-gray-600 dark:text-gray-400">
          AI-powered code understanding and semantic search
        </p>
      </div>
      <div class="flex items-center space-x-2 text-sm text-gray-500">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        <span>Powered by CodeBERT</span>
      </div>
    </div>
  </div>
  
  <!-- Tabs -->
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow">
    <div class="border-b border-gray-200 dark:border-gray-700">
      <nav class="flex -mb-px">
        <button
          on:click={() => activeTab = 'search'}
          class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                 {activeTab === 'search' 
                   ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                   : 'border-transparent text-gray-500 hover:text-gray-700'}"
        >
          Semantic Search
        </button>
        <button
          on:click={() => activeTab = 'similarity'}
          class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                 {activeTab === 'similarity' 
                   ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                   : 'border-transparent text-gray-500 hover:text-gray-700'}"
        >
          Code Similarity Map
        </button>
        <button
          on:click={() => activeTab = 'explorer'}
          class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                 {activeTab === 'explorer' 
                   ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                   : 'border-transparent text-gray-500 hover:text-gray-700'}"
        >
          Embeddings Explorer
        </button>
        <button
          on:click={() => activeTab = 'cross-language'}
          class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                 {activeTab === 'cross-language' 
                   ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                   : 'border-transparent text-gray-500 hover:text-gray-700'}"
        >
          Cross-Language
        </button>
      </nav>
    </div>
    
    <div class="p-6">
      {#if error}
        <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-4">
          <p class="text-red-800 dark:text-red-200">Error: {error}</p>
        </div>
      {/if}
      
      {#if activeTab === 'search'}
        <!-- Semantic Search -->
        <div class="space-y-6">
          <div class="flex gap-4">
            <div class="flex-1">
              <div class="relative">
                <input
                  type="text"
                  bind:value={searchQuery}
                  on:keydown={(e) => e.key === 'Enter' && performSemanticSearch()}
                  placeholder={searchMode === 'natural' 
                    ? "Search by intent: 'function that handles user authentication'" 
                    : "Search by code pattern: 'async function that calls API'"}
                  class="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 
                         rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
                />
                <svg class="absolute left-3 top-3.5 w-5 h-5 text-gray-400" 
                     fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
            </div>
            
            <div class="flex items-center space-x-2">
              <div class="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
                <button
                  on:click={() => searchMode = 'natural'}
                  class="px-3 py-1.5 rounded text-sm font-medium transition-colors
                         {searchMode === 'natural' 
                           ? 'bg-white dark:bg-gray-600 text-blue-600 dark:text-blue-400 shadow' 
                           : 'text-gray-600 dark:text-gray-400'}"
                >
                  {getSearchModeIcon('natural')} Natural
                </button>
                <button
                  on:click={() => searchMode = 'code'}
                  class="px-3 py-1.5 rounded text-sm font-medium transition-colors
                         {searchMode === 'code' 
                           ? 'bg-white dark:bg-gray-600 text-blue-600 dark:text-blue-400 shadow' 
                           : 'text-gray-600 dark:text-gray-400'}"
                >
                  {getSearchModeIcon('code')} Code
                </button>
              </div>
              
              <button
                on:click={performSemanticSearch}
                disabled={searching || !searchQuery.trim()}
                class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 
                       disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {#if searching}
                  <span class="flex items-center space-x-2">
                    <svg class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" 
                              stroke="currentColor" stroke-width="4"></circle>
                      <path class="opacity-75" fill="currentColor" 
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Searching...</span>
                  </span>
                {:else}
                  Search
                {/if}
              </button>
            </div>
          </div>
          
          <!-- Search Results -->
          {#if searchResults.length > 0}
            <div class="space-y-4">
              <div class="flex items-center justify-between">
                <h3 class="text-lg font-medium text-gray-900 dark:text-white">
                  Found {searchResults.length} results
                </h3>
                <button class="text-sm text-blue-600 hover:text-blue-800">
                  Export Results
                </button>
              </div>
              
              <div class="space-y-3">
                {#each searchResults as result (result.id)}
                  <div class="border border-gray-200 dark:border-gray-700 rounded-lg p-4 
                              hover:shadow-lg transition-shadow">
                    <div class="flex justify-between items-start mb-2">
                      <div>
                        <h4 class="font-medium text-gray-900 dark:text-white">
                          {result.functionName || 'Code snippet'}
                        </h4>
                        <p class="text-sm text-gray-500">{result.filePath}</p>
                      </div>
                      <div class="flex items-center space-x-2">
                        <div class="flex items-center space-x-1">
                          <div class="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div 
                              class="h-2 rounded-full bg-gradient-to-r from-blue-500 to-green-500"
                              style="width: {result.similarity * 100}%"
                            />
                          </div>
                          <span class="text-sm font-medium text-gray-600 dark:text-gray-400">
                            {Math.round(result.similarity * 100)}%
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <pre class="bg-gray-50 dark:bg-gray-900 rounded p-3 text-sm overflow-x-auto">
                      <code class="language-javascript">{result.snippet}</code>
                    </pre>
                    
                    <p class="mt-3 text-sm text-gray-600 dark:text-gray-400">
                      {result.explanation}
                    </p>
                    
                    <div class="mt-3 flex items-center space-x-4">
                      <button class="text-sm text-blue-600 hover:text-blue-800">
                        View in context →
                      </button>
                      <button class="text-sm text-gray-600 hover:text-gray-800">
                        Find similar
                      </button>
                    </div>
                  </div>
                {/each}
              </div>
            </div>
          {:else if searchQuery && !searching}
            <div class="text-center py-12 text-gray-500 dark:text-gray-400">
              <svg class="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" 
                   stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p>No results found for "{searchQuery}"</p>
              <p class="text-sm mt-2">Try a different search query or mode</p>
            </div>
          {:else}
            <div class="text-center py-12 text-gray-500 dark:text-gray-400">
              <svg class="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" 
                   stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <p>Start typing to search your codebase semantically</p>
              <p class="text-sm mt-2">
                Use natural language to find code by intent or functionality
              </p>
            </div>
          {/if}
        </div>
      {:else if activeTab === 'similarity'}
        <CodeSimilarityMap 
          {clusters} 
          {loading}
          bind:selectedCluster
          bind:similarityThreshold
        />
      {:else if activeTab === 'explorer'}
        <EmbeddingsExplorer 
          visualization={embeddingViz}
          {loading}
          bind:selectedNode
        />
      {:else if activeTab === 'cross-language'}
        <CrossLanguageSearch {projectId} />
      {/if}
    </div>
  </div>
</div>

<style>
  /* Syntax highlighting for code snippets */
  :global(.language-javascript) {
    color: #383a42;
  }
  
  :global(.dark .language-javascript) {
    color: #abb2bf;
  }
</style>
