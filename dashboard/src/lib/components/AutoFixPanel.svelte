<script lang="ts">
  import { autoFixStore, dashboardStore } from '$lib/stores/dashboard';
  import type { AutoFix } from '$lib/types/dashboard';
  import { onMount } from 'svelte';
  import FixBatchManager from './FixBatchManager.svelte';
  import FixHistory from './FixHistory.svelte';
  import FixPreview from './FixPreview.svelte';
  
  $: ({ selectedProject } = $dashboardStore);
  $: ({ availableFixes, batches, selectedFixes, previewFix, loading, error } = $autoFixStore);
  
  let activeTab: 'available' | 'batches' | 'history' = 'available';
  let filterSeverity: string = 'all';
  let filterType: string = 'all';
  let searchQuery: string = '';
  let showPreview: boolean = false;
  
  onMount(() => {
    if (selectedProject) {
      autoFixStore.loadFixes(selectedProject.id);
    }
  });
  
  $: if (selectedProject) {
    autoFixStore.loadFixes(selectedProject.id);
  }
  
  $: filteredFixes = availableFixes.filter(fix => {
    const matchesSeverity = filterSeverity === 'all' || fix.severity === filterSeverity;
    const matchesType = filterType === 'all' || fix.type === filterType;
    const matchesSearch = searchQuery === '' || 
      fix.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      fix.description.toLowerCase().includes(searchQuery.toLowerCase());
    
    return matchesSeverity && matchesType && matchesSearch;
  });
  
  function getSeverityColor(severity: string): string {
    const colors = {
      critical: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
      high: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
      medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
      low: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    };
    return colors[severity as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  }
  
  function getTypeIcon(type: string): string {
    const icons = {
      security: '🔒',
      performance: '⚡',
      quality: '✨',
      style: '🎨'
    };
    return icons[type as keyof typeof icons] || '🔧';
  }
  
  async function applySelectedFixes() {
    if (selectedFixes.size === 0) return;
    
    const confirmed = confirm(`Apply ${selectedFixes.size} fixes? This action can be rolled back.`);
    if (!confirmed) return;
    
    try {
      await autoFixStore.applyFixes(Array.from(selectedFixes));
      alert('Fixes applied successfully!');
    } catch (error) {
      alert('Failed to apply fixes: ' + error);
    }
  }
  
  function previewFix(fix: AutoFix) {
    autoFixStore.setPreviewFix(fix);
    showPreview = true;
  }
</script>

<div class="space-y-6">
  <!-- Header -->
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <div class="flex justify-between items-start">
      <div>
        <h2 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Automatic Fix Generator
        </h2>
        <p class="text-gray-600 dark:text-gray-400">
          AI-powered code fixes with one-click application
        </p>
      </div>
      {#if selectedFixes.size > 0}
        <button
          on:click={applySelectedFixes}
          class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 
                 transition-colors flex items-center space-x-2"
        >
          <span>Apply {selectedFixes.size} Fixes</span>
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M5 13l4 4L19 7" />
          </svg>
        </button>
      {/if}
    </div>
  </div>
  
  <!-- Tabs -->
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow">
    <div class="border-b border-gray-200 dark:border-gray-700">
      <nav class="flex -mb-px">
        <button
          on:click={() => activeTab = 'available'}
          class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                 {activeTab === 'available' 
                   ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                   : 'border-transparent text-gray-500 hover:text-gray-700'}"
        >
          Available Fixes ({availableFixes.length})
        </button>
        <button
          on:click={() => activeTab = 'batches'}
          class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                 {activeTab === 'batches' 
                   ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                   : 'border-transparent text-gray-500 hover:text-gray-700'}"
        >
          Fix Batches ({batches.length})
        </button>
        <button
          on:click={() => activeTab = 'history'}
          class="px-6 py-3 text-sm font-medium border-b-2 transition-colors
                 {activeTab === 'history' 
                   ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                   : 'border-transparent text-gray-500 hover:text-gray-700'}"
        >
          History
        </button>
      </nav>
    </div>
    
    <div class="p-6">
      {#if loading}
        <div class="flex items-center justify-center h-64">
          <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      {:else if error}
        <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p class="text-red-800 dark:text-red-200">Error: {error}</p>
        </div>
      {:else if activeTab === 'available'}
        <!-- Filters -->
        <div class="mb-6 space-y-4">
          <div class="flex flex-wrap gap-4">
            <div class="flex-1 min-w-[200px]">
              <input
                type="text"
                bind:value={searchQuery}
                placeholder="Search fixes..."
                class="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 
                       rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
              />
            </div>
            <select
              bind:value={filterSeverity}
              class="px-4 py-2 border border-gray-300 dark:border-gray-600 
                     rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
            >
              <option value="all">All Severities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
            <select
              bind:value={filterType}
              class="px-4 py-2 border border-gray-300 dark:border-gray-600 
                     rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
            >
              <option value="all">All Types</option>
              <option value="security">Security</option>
              <option value="performance">Performance</option>
              <option value="quality">Quality</option>
              <option value="style">Style</option>
            </select>
          </div>
          
          <!-- Quick Stats -->
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center">
              <p class="text-2xl font-bold text-gray-900 dark:text-white">
                {filteredFixes.length}
              </p>
              <p class="text-sm text-gray-600 dark:text-gray-400">Available</p>
            </div>
            <div class="bg-red-50 dark:bg-red-900/20 rounded-lg p-3 text-center">
              <p class="text-2xl font-bold text-red-600 dark:text-red-400">
                {filteredFixes.filter(f => f.severity === 'critical').length}
              </p>
              <p class="text-sm text-gray-600 dark:text-gray-400">Critical</p>
            </div>
            <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 text-center">
              <p class="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {selectedFixes.size}
              </p>
              <p class="text-sm text-gray-600 dark:text-gray-400">Selected</p>
            </div>
            <div class="bg-green-50 dark:bg-green-900/20 rounded-lg p-3 text-center">
              <p class="text-2xl font-bold text-green-600 dark:text-green-400">
                {filteredFixes.filter(f => f.confidence > 0.9).length}
              </p>
              <p class="text-sm text-gray-600 dark:text-gray-400">High Confidence</p>
            </div>
          </div>
        </div>
        
        <!-- Fix List -->
        <div class="space-y-4">
          {#each filteredFixes as fix (fix.id)}
            <div class="border border-gray-200 dark:border-gray-700 rounded-lg p-4 
                        hover:shadow-lg transition-shadow">
              <div class="flex items-start space-x-4">
                <input
                  type="checkbox"
                  checked={selectedFixes.has(fix.id)}
                  on:change={() => autoFixStore.toggleFixSelection(fix.id)}
                  class="mt-1 w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                />
                
                <div class="flex-1">
                  <div class="flex items-center justify-between mb-2">
                    <h4 class="font-medium text-gray-900 dark:text-white flex items-center space-x-2">
                      <span class="text-xl">{getTypeIcon(fix.type)}</span>
                      <span>{fix.title}</span>
                    </h4>
                    <div class="flex items-center space-x-2">
                      <span class="px-2 py-1 text-xs font-medium rounded {getSeverityColor(fix.severity)}">
                        {fix.severity.toUpperCase()}
                      </span>
                      <span class="text-sm text-gray-500">
                        {Math.round(fix.confidence * 100)}% confidence
                      </span>
                    </div>
                  </div>
                  
                  <p class="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    {fix.description}
                  </p>
                  
                  <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-4 text-sm text-gray-500">
                      <span>{fix.filePath}</span>
                      <span>Lines {fix.startLine}-{fix.endLine}</span>
                      <span>{fix.impact.linesChanged} lines changed</span>
                    </div>
                    
                    <button
                      on:click={() => previewFix(fix)}
                      class="text-blue-600 hover:text-blue-800 text-sm font-medium"
                    >
                      Preview →
                    </button>
                  </div>
                </div>
              </div>
            </div>
          {/each}
          
          {#if filteredFixes.length === 0}
            <div class="text-center py-8 text-gray-500 dark:text-gray-400">
              No fixes found matching your criteria
            </div>
          {/if}
        </div>
      {:else if activeTab === 'batches'}
        <FixBatchManager {batches} />
      {:else if activeTab === 'history'}
        <FixHistory projectId={selectedProject?.id} />
      {/if}
    </div>
  </div>
  
  <!-- Fix Preview Modal -->
  {#if showPreview && previewFix}
    <FixPreview 
      fix={previewFix} 
      on:close={() => {
        showPreview = false;
        autoFixStore.setPreviewFix(null);
      }}
      on:apply={() => {
        autoFixStore.toggleFixSelection(previewFix.id);
        showPreview = false;
      }}
    />
  {/if}
</div>
