<script lang="ts">
  import { dashboardStore, technicalDebtStore } from '$lib/stores/dashboard';
  import { formatCurrency, formatNumber, formatPercentage } from '$lib/utils/formatters';
  import { onMount } from 'svelte';
  import DebtByComponentChart from './charts/DebtByComponentChart.svelte';
  import DebtEvolutionChart from './charts/DebtEvolutionChart.svelte';
  import DebtPaymentRoadmap from './DebtPaymentRoadmap.svelte';
  import DebtROICalculator from './DebtROICalculator.svelte';
  
  $: ({ selectedProject, selectedTimeRange } = $dashboardStore);
  $: ({ metrics, evolution, components, loading, error } = $technicalDebtStore);
  
  onMount(() => {
    if (selectedProject) {
      technicalDebtStore.loadDebtData(selectedProject.id, selectedTimeRange);
    }
  });
  
  // Reactive loading when project or time range changes
  $: if (selectedProject) {
    technicalDebtStore.loadDebtData(selectedProject.id, selectedTimeRange);
  }
  
  function getSeverityColor(severity: string): string {
    const colors = {
      critical: 'text-red-600 bg-red-100',
      high: 'text-orange-600 bg-orange-100',
      medium: 'text-yellow-600 bg-yellow-100',
      low: 'text-green-600 bg-green-100'
    };
    return colors[severity as keyof typeof colors] || 'text-gray-600 bg-gray-100';
  }
  
  function getCategoryIcon(category: string): string {
    const icons = {
      codeSmells: '🐛',
      bugs: '🪲',
      vulnerabilities: '🔓',
      duplications: '📋'
    };
    return icons[category as keyof typeof icons] || '📊';
  }
</script>

<div class="space-y-6">
  <!-- Header -->
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <h2 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">
      Technical Debt Dashboard
    </h2>
    <p class="text-gray-600 dark:text-gray-400">
      Comprehensive analysis of technical debt, evolution trends, and remediation strategies
    </p>
  </div>
  
  {#if loading}
    <div class="flex items-center justify-center h-96">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
    </div>
  {:else if error}
    <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
      <p class="text-red-800 dark:text-red-200">Error loading technical debt data: {error}</p>
    </div>
  {:else if metrics}
    <!-- Key Metrics Overview -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Total Debt</p>
            <p class="text-2xl font-semibold text-gray-900 dark:text-white">
              {formatNumber(metrics.totalDebt)}
            </p>
            <p class="text-sm text-gray-500 dark:text-gray-400">
              ~{formatNumber(metrics.estimatedHours)} hours
            </p>
          </div>
          <div class="text-3xl">💸</div>
        </div>
      </div>
      
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Debt Ratio</p>
            <p class="text-2xl font-semibold text-gray-900 dark:text-white">
              {formatPercentage(metrics.debtRatio)}
            </p>
            <p class="text-sm text-gray-500 dark:text-gray-400">
              of total codebase
            </p>
          </div>
          <div class="text-3xl">📊</div>
        </div>
      </div>
      
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Estimated Cost</p>
            <p class="text-2xl font-semibold text-gray-900 dark:text-white">
              {formatCurrency(metrics.estimatedCost)}
            </p>
            <p class="text-sm text-gray-500 dark:text-gray-400">
              remediation cost
            </p>
          </div>
          <div class="text-3xl">💰</div>
        </div>
      </div>
      
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Critical Issues</p>
            <p class="text-2xl font-semibold text-red-600 dark:text-red-400">
              {formatNumber(metrics.severity.critical)}
            </p>
            <p class="text-sm text-gray-500 dark:text-gray-400">
              need immediate attention
            </p>
          </div>
          <div class="text-3xl">🚨</div>
        </div>
      </div>
    </div>
    
    <!-- Debt by Severity and Category -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Severity Breakdown -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 class="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
          Debt by Severity
        </h3>
        <div class="space-y-3">
          {#each Object.entries(metrics.severity) as [severity, count]}
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-3">
                <span class="px-2 py-1 text-xs font-medium rounded {getSeverityColor(severity)}">
                  {severity.toUpperCase()}
                </span>
                <span class="text-sm text-gray-600 dark:text-gray-400">{count} issues</span>
              </div>
              <div class="flex items-center space-x-2">
                <div class="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div 
                    class="h-2 rounded-full transition-all duration-300"
                    class:bg-red-600={severity === 'critical'}
                    class:bg-orange-500={severity === 'high'}
                    class:bg-yellow-500={severity === 'medium'}
                    class:bg-green-500={severity === 'low'}
                    style="width: {(count / metrics.totalDebt) * 100}%"
                  ></div>
                </div>
                <span class="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {formatPercentage(count / metrics.totalDebt)}
                </span>
              </div>
            </div>
          {/each}
        </div>
      </div>
      
      <!-- Category Breakdown -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 class="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
          Debt by Category
        </h3>
        <div class="grid grid-cols-2 gap-4">
          {#each Object.entries(metrics.categories) as [category, count]}
            <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <div class="flex items-center justify-between mb-2">
                <span class="text-2xl">{getCategoryIcon(category)}</span>
                <span class="text-2xl font-bold text-gray-900 dark:text-white">
                  {formatNumber(count)}
                </span>
              </div>
              <p class="text-sm text-gray-600 dark:text-gray-400 capitalize">
                {category.replace(/([A-Z])/g, ' $1').trim()}
              </p>
            </div>
          {/each}
        </div>
      </div>
    </div>
    
    <!-- Debt Evolution Timeline -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 class="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
        Debt Evolution Timeline
      </h3>
      <DebtEvolutionChart data={evolution} />
    </div>
    
    <!-- Debt by Component -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 class="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
        Debt Distribution by Component
      </h3>
      <DebtByComponentChart components={components} />
    </div>
    
    <!-- ROI Calculator and Payment Roadmap -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <DebtROICalculator 
        components={components} 
        metrics={metrics}
      />
      
      <DebtPaymentRoadmap 
        components={components}
        metrics={metrics}
      />
    </div>
  {:else}
    <div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-8 text-center">
      <p class="text-gray-600 dark:text-gray-400">
        Select a project to view technical debt metrics
      </p>
    </div>
  {/if}
</div>

<style>
  /* Add any component-specific styles here */
  :global(.debt-chart) {
    min-height: 300px;
  }
</style>
