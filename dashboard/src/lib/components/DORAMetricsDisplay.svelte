<script lang="ts">
    import type { DORAMetrics, TimeRange } from "$lib/types/dashboard";
    
    export let metrics: DORAMetrics;
    export let timeRange: TimeRange;
    
    function getTrendIcon(trend: 'up' | 'down' | 'stable'): string {
        return trend === 'up' ? '↗️' : trend === 'down' ? '↘️' : '→';
    }
    
    function getTrendColor(trend: 'up' | 'down' | 'stable', metric: string): string {
        // For some metrics, up is good (deployment frequency), for others down is good (failure rate)
        const upIsGood = ['deploymentFrequency'];
        const downIsGood = ['changeFailureRate', 'timeToRestoreService', 'leadTimeForChanges'];
        
        if (trend === 'stable') return 'text-gray-600';
        
        if (upIsGood.includes(metric)) {
            return trend === 'up' ? 'text-green-600' : 'text-red-600';
        } else if (downIsGood.includes(metric)) {
            return trend === 'down' ? 'text-green-600' : 'text-red-600';
        }
        
        return 'text-gray-600';
    }
    
    // Performance levels based on DORA research
    // https://cloud.google.com/blog/products/devops-sre/using-the-four-keys-to-measure-your-devops-performance
    function getPerformanceLevel(metric: string, value: number): string {
        if (metric === 'deploymentFrequency') {
            const deploysPerDay = value / (metric === 'per_day' ? 1 : metric === 'per_week' ? 7 : 30);
            if (deploysPerDay >= 1) return 'Elite';
            if (deploysPerDay >= 1/7) return 'High';
            if (deploysPerDay >= 1/30) return 'Medium';
            return 'Low';
        }
        
        if (metric === 'leadTimeForChanges') {
            const hours = value;
            if (hours < 24) return 'Elite';
            if (hours < 168) return 'High'; // 1 week
            if (hours < 672) return 'Medium'; // 1 month
            return 'Low';
        }
        
        if (metric === 'changeFailureRate') {
            const percentage = value;
            if (percentage <= 0.15) return 'Elite';
            if (percentage <= 0.30) return 'High';
            if (percentage <= 0.45) return 'Medium';
            return 'Low';
        }
        
        if (metric === 'timeToRestoreService') {
            const minutes = value;
            if (minutes < 60) return 'Elite';
            if (minutes < 1440) return 'High'; // 1 day
            if (minutes < 10080) return 'Medium'; // 1 week
            return 'Low';
        }
        
        return 'N/A';
    }
    
    function getPerformanceLevelColor(level: string): string {
        const colors = {
            Elite: 'text-indigo-600 dark:text-indigo-400',
            High: 'text-green-600 dark:text-green-400',
            Medium: 'text-yellow-600 dark:text-yellow-400',
            Low: 'text-red-600 dark:text-red-400'
        };
        return colors[level as keyof typeof colors] || 'text-gray-600';
    }
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <div class="flex justify-between items-center mb-4">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">DORA Metrics</h3>
        <div class="text-sm text-gray-500 dark:text-gray-400">
            {timeRange.start.toLocaleDateString()} - {timeRange.end.toLocaleDateString()}
        </div>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Deployment Frequency -->
        <div class="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-700 dark:to-gray-800 rounded-lg p-4">
            <div class="flex justify-between items-center mb-3">
                <h4 class="font-medium text-gray-900 dark:text-white">Deployment Frequency</h4>
                <span class={getTrendColor(metrics.deploymentFrequency.trend, 'deploymentFrequency')}>
                    {getTrendIcon(metrics.deploymentFrequency.trend)}
                </span>
            </div>
            
            <div class="flex items-baseline mb-1">
                <div class="text-3xl font-bold text-gray-900 dark:text-white">
                    {metrics.deploymentFrequency.value}
                </div>
                <div class="ml-2 text-sm text-gray-600 dark:text-gray-400">
                    {metrics.deploymentFrequency.unit.replace('_', ' ')}
                </div>
            </div>
            
            <div class="flex justify-between items-center mt-4">
                <p class="text-sm text-gray-600 dark:text-gray-400">
                    Performance Level
                </p>
                <p class={`text-sm font-medium ${getPerformanceLevelColor(getPerformanceLevel('deploymentFrequency', metrics.deploymentFrequency.value))}`}>
                    {getPerformanceLevel('deploymentFrequency', metrics.deploymentFrequency.value)}
                </p>
            </div>
            
            <div class="mt-2 text-xs text-gray-500 dark:text-gray-400">
                Elite teams deploy multiple times per day
            </div>
        </div>
        
        <!-- Lead Time for Changes -->
        <div class="bg-gradient-to-br from-green-50 to-teal-50 dark:from-gray-700 dark:to-gray-800 rounded-lg p-4">
            <div class="flex justify-between items-center mb-3">
                <h4 class="font-medium text-gray-900 dark:text-white">Lead Time for Changes</h4>
                <span class={getTrendColor(metrics.leadTimeForChanges.trend, 'leadTimeForChanges')}>
                    {getTrendIcon(metrics.leadTimeForChanges.trend)}
                </span>
            </div>
            
            <div class="flex items-baseline mb-1">
                <div class="text-3xl font-bold text-gray-900 dark:text-white">
                    {metrics.leadTimeForChanges.value}
                </div>
                <div class="ml-2 text-sm text-gray-600 dark:text-gray-400">
                    {metrics.leadTimeForChanges.unit}
                </div>
            </div>
            
            <div class="flex justify-between items-center mt-4">
                <p class="text-sm text-gray-600 dark:text-gray-400">
                    Performance Level
                </p>
                <p class={`text-sm font-medium ${getPerformanceLevelColor(getPerformanceLevel('leadTimeForChanges', metrics.leadTimeForChanges.value))}`}>
                    {getPerformanceLevel('leadTimeForChanges', metrics.leadTimeForChanges.value)}
                </p>
            </div>
            
            <div class="mt-2 text-xs text-gray-500 dark:text-gray-400">
                Measures time from commit to production
            </div>
        </div>
        
        <!-- Change Failure Rate -->
        <div class="bg-gradient-to-br from-red-50 to-pink-50 dark:from-gray-700 dark:to-gray-800 rounded-lg p-4">
            <div class="flex justify-between items-center mb-3">
                <h4 class="font-medium text-gray-900 dark:text-white">Change Failure Rate</h4>
                <span class={getTrendColor(metrics.changeFailureRate.trend, 'changeFailureRate')}>
                    {getTrendIcon(metrics.changeFailureRate.trend)}
                </span>
            </div>
            
            <div class="flex items-baseline mb-1">
                <div class="text-3xl font-bold text-gray-900 dark:text-white">
                    {metrics.changeFailureRate.percentage}%
                </div>
                <div class="ml-2 text-sm text-gray-600 dark:text-gray-400">
                    ({metrics.changeFailureRate.value} failures)
                </div>
            </div>
            
            <div class="flex justify-between items-center mt-4">
                <p class="text-sm text-gray-600 dark:text-gray-400">
                    Performance Level
                </p>
                <p class={`text-sm font-medium ${getPerformanceLevelColor(getPerformanceLevel('changeFailureRate', metrics.changeFailureRate.percentage / 100))}`}>
                    {getPerformanceLevel('changeFailureRate', metrics.changeFailureRate.percentage / 100)}
                </p>
            </div>
            
            <div class="mt-2 text-xs text-gray-500 dark:text-gray-400">
                Percentage of deployments causing a failure in production
            </div>
        </div>
        
        <!-- Time to Restore Service -->
        <div class="bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-gray-700 dark:to-gray-800 rounded-lg p-4">
            <div class="flex justify-between items-center mb-3">
                <h4 class="font-medium text-gray-900 dark:text-white">Time to Restore Service</h4>
                <span class={getTrendColor(metrics.timeToRestoreService.trend, 'timeToRestoreService')}>
                    {getTrendIcon(metrics.timeToRestoreService.trend)}
                </span>
            </div>
            
            <div class="flex items-baseline mb-1">
                <div class="text-3xl font-bold text-gray-900 dark:text-white">
                    {metrics.timeToRestoreService.value}
                </div>
                <div class="ml-2 text-sm text-gray-600 dark:text-gray-400">
                    {metrics.timeToRestoreService.unit}
                </div>
            </div>
            
            <div class="flex justify-between items-center mt-4">
                <p class="text-sm text-gray-600 dark:text-gray-400">
                    Performance Level
                </p>
                <p class={`text-sm font-medium ${getPerformanceLevelColor(getPerformanceLevel('timeToRestoreService', metrics.timeToRestoreService.value))}`}>
                    {getPerformanceLevel('timeToRestoreService', metrics.timeToRestoreService.value)}
                </p>
            </div>
            
            <div class="mt-2 text-xs text-gray-500 dark:text-gray-400">
                How quickly service is restored after a failure
            </div>
        </div>
    </div>
    
    <!-- Overall Assessment -->
    <div class="mt-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <h4 class="font-medium text-gray-900 dark:text-white mb-2">Overall DORA Performance</h4>
        
        <div class="flex flex-wrap gap-2">
            <div class="px-3 py-1 bg-indigo-100 text-indigo-800 dark:bg-indigo-900/20 dark:text-indigo-300 rounded-lg text-sm">
                Elite: {['deploymentFrequency', 'leadTimeForChanges', 'changeFailureRate', 'timeToRestoreService'].filter(m => getPerformanceLevel(m, m === 'changeFailureRate' ? metrics.changeFailureRate.percentage / 100 : metrics[m as keyof DORAMetrics].value) === 'Elite').length}/4 metrics
            </div>
            <div class="px-3 py-1 bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300 rounded-lg text-sm">
                High: {['deploymentFrequency', 'leadTimeForChanges', 'changeFailureRate', 'timeToRestoreService'].filter(m => getPerformanceLevel(m, m === 'changeFailureRate' ? metrics.changeFailureRate.percentage / 100 : metrics[m as keyof DORAMetrics].value) === 'High').length}/4 metrics
            </div>
            <div class="px-3 py-1 bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-300 rounded-lg text-sm">
                Medium: {['deploymentFrequency', 'leadTimeForChanges', 'changeFailureRate', 'timeToRestoreService'].filter(m => getPerformanceLevel(m, m === 'changeFailureRate' ? metrics.changeFailureRate.percentage / 100 : metrics[m as keyof DORAMetrics].value) === 'Medium').length}/4 metrics
            </div>
            <div class="px-3 py-1 bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300 rounded-lg text-sm">
                Low: {['deploymentFrequency', 'leadTimeForChanges', 'changeFailureRate', 'timeToRestoreService'].filter(m => getPerformanceLevel(m, m === 'changeFailureRate' ? metrics.changeFailureRate.percentage / 100 : metrics[m as keyof DORAMetrics].value) === 'Low').length}/4 metrics
            </div>
        </div>
    </div>
</div>
