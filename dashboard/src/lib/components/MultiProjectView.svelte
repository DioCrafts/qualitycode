<script lang="ts">
  import { multiProjectStore } from '$lib/stores/dashboard';
  import { formatNumber } from '$lib/utils/formatters';
  import { onMount } from 'svelte';
  import OrganizationTrends from './OrganizationTrends.svelte';
  import PortfolioHealthOverview from './PortfolioHealthOverview.svelte';
  import ProjectComparisonMatrix from './ProjectComparisonMatrix.svelte';
  
  export let organizationId: string;
  
  $: ({ projects, portfolioHealth, loading, error } = $multiProjectStore);
  
  let selectedProjects: Set<string> = new Set();
  let sortBy: 'name' | 'quality' | 'debt' | 'issues' = 'quality';
  let sortDirection: 'asc' | 'desc' = 'desc';
  let filterLanguage: string = 'all';
  let viewMode: 'grid' | 'table' | 'comparison' = 'grid';
  
  onMount(() => {
    multiProjectStore.loadProjects(organizationId);
  });
  
  $: sortedProjects = [...projects].sort((a, b) => {
    let aVal: any, bVal: any;
    
    switch (sortBy) {
      case 'name':
        aVal = a.name;
        bVal = b.name;
        break;
      case 'quality':
        aVal = a.metrics.qualityScore;
        bVal = b.metrics.qualityScore;
        break;
      case 'debt':
        aVal = a.metrics.technicalDebt;
        bVal = b.metrics.technicalDebt;
        break;
      case 'issues':
        aVal = a.metrics.issues.total;
        bVal = b.metrics.issues.total;
        break;
    }
    
    const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
    return sortDirection === 'asc' ? comparison : -comparison;
  });
  
  $: filteredProjects = filterLanguage === 'all' 
    ? sortedProjects 
    : sortedProjects.filter(p => p.language === filterLanguage);
  
  $: languages = [...new Set(projects.map(p => p.language))];
  
  function getStatusColor(status: string): string {
    const colors = {
      healthy: 'text-green-600 bg-green-100',
      warning: 'text-yellow-600 bg-yellow-100',
      critical: 'text-red-600 bg-red-100'
    };
    return colors[status as keyof typeof colors] || 'text-gray-600 bg-gray-100';
  }
  
  function getStatusIcon(status: string): string {
    const icons = {
      healthy: '✅',
      warning: '⚠️',
      critical: '🚨'
    };
    return icons[status as keyof typeof icons] || '❓';
  }
  
  function toggleProjectSelection(projectId: string) {
    if (selectedProjects.has(projectId)) {
      selectedProjects.delete(projectId);
    } else {
      selectedProjects.add(projectId);
    }
    selectedProjects = selectedProjects; // Trigger reactivity
  }
  
  function selectAllProjects() {
    selectedProjects = new Set(projects.map(p => p.id));
  }
  
  function clearSelection() {
    selectedProjects = new Set();
  }
</script>

<div class="space-y-6">
  <!-- Header -->
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <div class="flex justify-between items-start">
      <div>
        <h2 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Multi-Project Portfolio View
        </h2>
        <p class="text-gray-600 dark:text-gray-400">
          Comprehensive analysis across all projects in your organization
        </p>
      </div>
      <div class="flex space-x-2">
        <button
          on:click={() => viewMode = 'grid'}
          class="p-2 rounded {viewMode === 'grid' ? 'bg-blue-100 text-blue-600' : 'text-gray-500'}"
          title="Grid View"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
          </svg>
        </button>
        <button
          on:click={() => viewMode = 'table'}
          class="p-2 rounded {viewMode === 'table' ? 'bg-blue-100 text-blue-600' : 'text-gray-500'}"
          title="Table View"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        </button>
        <button
          on:click={() => viewMode = 'comparison'}
          class="p-2 rounded {viewMode === 'comparison' ? 'bg-blue-100 text-blue-600' : 'text-gray-500'}"
          title="Comparison View"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </button>
      </div>
    </div>
  </div>
  
  {#if loading}
    <div class="flex items-center justify-center h-96">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
    </div>
  {:else if error}
    <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
      <p class="text-red-800 dark:text-red-200">Error: {error}</p>
    </div>
  {:else}
    <!-- Portfolio Health Overview -->
    {#if portfolioHealth}
      <PortfolioHealthOverview health={portfolioHealth} />
    {/if}
    
    <!-- Controls -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
      <div class="flex flex-wrap gap-4 items-center justify-between">
        <div class="flex flex-wrap gap-4">
          <select
            bind:value={filterLanguage}
            class="px-4 py-2 border border-gray-300 dark:border-gray-600 
                   rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
          >
            <option value="all">All Languages</option>
            {#each languages as lang}
              <option value={lang}>{lang}</option>
            {/each}
          </select>
          
          <select
            bind:value={sortBy}
            class="px-4 py-2 border border-gray-300 dark:border-gray-600 
                   rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700"
          >
            <option value="name">Sort by Name</option>
            <option value="quality">Sort by Quality Score</option>
            <option value="debt">Sort by Technical Debt</option>
            <option value="issues">Sort by Issues</option>
          </select>
          
          <button
            on:click={() => sortDirection = sortDirection === 'asc' ? 'desc' : 'asc'}
            class="p-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                   hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            {#if sortDirection === 'asc'}
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 11l5-5m0 0l5 5m-5-5v12" />
              </svg>
            {:else}
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 13l-5 5m0 0l-5-5m5 5V6" />
              </svg>
            {/if}
          </button>
        </div>
        
        {#if viewMode !== 'comparison'}
          <div class="flex gap-2">
            <button
              on:click={selectAllProjects}
              class="px-3 py-1 text-sm text-blue-600 hover:text-blue-800"
            >
              Select All
            </button>
            <button
              on:click={clearSelection}
              class="px-3 py-1 text-sm text-gray-600 hover:text-gray-800"
            >
              Clear Selection
            </button>
          </div>
        {/if}
      </div>
    </div>
    
    <!-- Project Views -->
    {#if viewMode === 'grid'}
      <!-- Grid View -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {#each filteredProjects as project (project.id)}
          <div 
            class="bg-white dark:bg-gray-800 rounded-lg shadow hover:shadow-lg 
                   transition-shadow p-6 cursor-pointer border-2
                   {selectedProjects.has(project.id) ? 'border-blue-500' : 'border-transparent'}"
            on:click={() => toggleProjectSelection(project.id)}
          >
            <div class="flex justify-between items-start mb-4">
              <div>
                <h3 class="font-semibold text-lg text-gray-900 dark:text-white">
                  {project.name}
                </h3>
                <p class="text-sm text-gray-500">{project.language}</p>
              </div>
              <span class="px-2 py-1 text-xs font-medium rounded {getStatusColor(project.status)}">
                {getStatusIcon(project.status)} {project.status}
              </span>
            </div>
            
            <div class="space-y-3">
              <div class="flex justify-between items-center">
                <span class="text-sm text-gray-600 dark:text-gray-400">Quality Score</span>
                <span class="font-medium {project.metrics.qualityScore >= 80 ? 'text-green-600' : project.metrics.qualityScore >= 60 ? 'text-yellow-600' : 'text-red-600'}">
                  {project.metrics.qualityScore.toFixed(1)}%
                </span>
              </div>
              
              <div class="flex justify-between items-center">
                <span class="text-sm text-gray-600 dark:text-gray-400">Technical Debt</span>
                <span class="font-medium text-gray-900 dark:text-white">
                  {formatNumber(project.metrics.technicalDebt)}h
                </span>
              </div>
              
              <div class="flex justify-between items-center">
                <span class="text-sm text-gray-600 dark:text-gray-400">Issues</span>
                <div class="flex items-center space-x-2">
                  <span class="text-xs px-1.5 py-0.5 bg-red-100 text-red-800 rounded">
                    {project.metrics.issues.critical}
                  </span>
                  <span class="text-xs px-1.5 py-0.5 bg-orange-100 text-orange-800 rounded">
                    {project.metrics.issues.high}
                  </span>
                  <span class="text-sm font-medium text-gray-900 dark:text-white">
                    {project.metrics.issues.total}
                  </span>
                </div>
              </div>
              
              <div class="pt-3 border-t border-gray-200 dark:border-gray-700">
                <div class="grid grid-cols-3 gap-2 text-center">
                  <div>
                    <p class="text-xs text-gray-500">Files</p>
                    <p class="font-medium text-sm">{formatNumber(project.metrics.codeSize.files)}</p>
                  </div>
                  <div>
                    <p class="text-xs text-gray-500">Lines</p>
                    <p class="font-medium text-sm">{formatNumber(project.metrics.codeSize.lines)}</p>
                  </div>
                  <div>
                    <p class="text-xs text-gray-500">Coverage</p>
                    <p class="font-medium text-sm">{project.metrics.coverage}%</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        {/each}
      </div>
    {:else if viewMode === 'table'}
      <!-- Table View -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <table class="w-full">
          <thead class="bg-gray-50 dark:bg-gray-700">
            <tr>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Project
              </th>
              <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                Quality
              </th>
              <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                Tech Debt
              </th>
              <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                Issues
              </th>
              <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                Coverage
              </th>
              <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                Size
              </th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-200 dark:divide-gray-700">
            {#each filteredProjects as project (project.id)}
              <tr 
                class="hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer
                       {selectedProjects.has(project.id) ? 'bg-blue-50 dark:bg-blue-900/20' : ''}"
                on:click={() => toggleProjectSelection(project.id)}
              >
                <td class="px-6 py-4 whitespace-nowrap">
                  <div>
                    <div class="text-sm font-medium text-gray-900 dark:text-white">
                      {project.name}
                    </div>
                    <div class="text-sm text-gray-500">{project.language}</div>
                  </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-center">
                  <span class="px-2 py-1 text-xs font-medium rounded {getStatusColor(project.status)}">
                    {project.status}
                  </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-center">
                  <span class="font-medium {project.metrics.qualityScore >= 80 ? 'text-green-600' : project.metrics.qualityScore >= 60 ? 'text-yellow-600' : 'text-red-600'}">
                    {project.metrics.qualityScore.toFixed(1)}%
                  </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-center">
                  {formatNumber(project.metrics.technicalDebt)}h
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-center">
                  <div class="flex items-center justify-center space-x-1">
                    <span class="text-xs px-1.5 py-0.5 bg-red-100 text-red-800 rounded">
                      {project.metrics.issues.critical}
                    </span>
                    <span class="text-xs px-1.5 py-0.5 bg-orange-100 text-orange-800 rounded">
                      {project.metrics.issues.high}
                    </span>
                    <span class="text-sm">
                      {project.metrics.issues.total}
                    </span>
                  </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-center">
                  {project.metrics.coverage}%
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-center text-sm">
                  {formatNumber(project.metrics.codeSize.lines)} lines
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else}
      <!-- Comparison View -->
      <ProjectComparisonMatrix 
        projects={selectedProjects.size >= 2 
          ? projects.filter(p => selectedProjects.has(p.id))
          : projects.slice(0, 4)} 
      />
    {/if}
    
    <!-- Organization Trends -->
    <OrganizationTrends projects={projects} />
  {/if}
</div>
