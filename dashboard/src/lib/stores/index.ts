// Export all stores from a single location
export * from './analysis';
export * from './auth';
export * from './dashboard';
export * from './preferences';
export * from './realtime';
export * from './security';

// Re-export commonly used stores for convenience
export { analysisStore, criticalIssues, filteredIssues, issueStats } from './analysis';
export { authStore, currentOrganization, currentUser, userRole } from './auth';
export { dashboardStore } from './dashboard';
export { dashboardCache, favoriteFilters, savedLayouts, userPreferences } from './preferences';
export { connectionStatus, notificationCount, realTimeStore, unreadNotifications } from './realtime';
export { complianceScore, criticalVulnerabilities, securityStore, threatSummary, vulnerabilityStats } from './security';

// Helper function to reset all stores
export function resetAllStores() {
    // TypeScript issue with recognizing reset method
    // TODO: Fix type inference for store reset methods
    (dashboardStore as any).reset();
    (analysisStore as any).reset();
    (securityStore as any).reset();
    (authStore as any).reset();
    (realTimeStore as any).reset();
}
