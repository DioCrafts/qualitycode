/**
 * Dashboard Stores - Estado global del dashboard
 */

import type {
    AutoFix,
    DORAMetrics,
    DashboardFilter,
    DebtComponent,
    DebtEvolution,
    FixBatch,
    Pipeline,
    PortfolioHealth,
    Project,
    TechnicalDebtMetrics,
    TimeRange
} from '$lib/types/dashboard';
import { derived, get, writable } from 'svelte/store';

// Store principal del dashboard
interface DashboardState {
    selectedProject: Project | null;
    selectedTimeRange: TimeRange;
    filters: DashboardFilter;
    loading: boolean;
    error: string | null;
    realTimeEnabled: boolean;
}

function createDashboardStore() {
    const { subscribe, set, update } = writable<DashboardState>({
        selectedProject: null,
        selectedTimeRange: {
            period: '30d',
            start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
            end: new Date()
        },
        filters: {},
        loading: false,
        error: null,
        realTimeEnabled: true
    });

    return {
        subscribe,

        setProject: (project: Project | null) =>
            update(state => ({ ...state, selectedProject: project })),

        setTimeRange: (timeRange: TimeRange) =>
            update(state => ({ ...state, selectedTimeRange: timeRange })),

        setFilters: (filters: DashboardFilter) =>
            update(state => ({ ...state, filters })),

        toggleRealTime: () =>
            update(state => ({ ...state, realTimeEnabled: !state.realTimeEnabled })),

        setLoading: (loading: boolean) =>
            update(state => ({ ...state, loading })),

        setError: (error: string | null) =>
            update(state => ({ ...state, error }))
    };
}

export const dashboardStore = createDashboardStore();

// Store para Deuda Técnica
interface TechnicalDebtState {
    metrics: TechnicalDebtMetrics | null;
    evolution: DebtEvolution[];
    components: DebtComponent[];
    loading: boolean;
    error: string | null;
}

function createTechnicalDebtStore() {
    const { subscribe, set, update } = writable<TechnicalDebtState>({
        metrics: null,
        evolution: [],
        components: [],
        loading: false,
        error: null
    });

    return {
        subscribe,

        async loadDebtData(projectId: string, timeRange: TimeRange) {
            update(state => ({ ...state, loading: true, error: null }));

            try {
                const response = await fetch('/api/technical-debt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ projectId, timeRange })
                });

                if (!response.ok) throw new Error('Failed to load technical debt data');

                const data = await response.json();
                update(state => ({
                    ...state,
                    metrics: data.metrics,
                    evolution: data.evolution,
                    components: data.components,
                    loading: false
                }));
            } catch (error) {
                update(state => ({
                    ...state,
                    error: error instanceof Error ? error.message : 'Unknown error',
                    loading: false
                }));
            }
        },

        reset: () => set({
            metrics: null,
            evolution: [],
            components: [],
            loading: false,
            error: null
        })
    };
}

export const technicalDebtStore = createTechnicalDebtStore();

// Store para Fixes Automáticos
interface AutoFixState {
    availableFixes: AutoFix[];
    batches: FixBatch[];
    selectedFixes: Set<string>;
    previewFix: AutoFix | null;
    loading: boolean;
    error: string | null;
}

function createAutoFixStore() {
    const { subscribe, set, update } = writable<AutoFixState>({
        availableFixes: [],
        batches: [],
        selectedFixes: new Set(),
        previewFix: null,
        loading: false,
        error: null
    });

    return {
        subscribe,

        async loadFixes(projectId: string) {
            update(state => ({ ...state, loading: true, error: null }));

            try {
                const response = await fetch(`/api/fixes/${projectId}`);
                if (!response.ok) throw new Error('Failed to load fixes');

                const data = await response.json();
                update(state => ({
                    ...state,
                    availableFixes: data.fixes,
                    batches: data.batches,
                    loading: false
                }));
            } catch (error) {
                update(state => ({
                    ...state,
                    error: error instanceof Error ? error.message : 'Unknown error',
                    loading: false
                }));
            }
        },

        toggleFixSelection: (fixId: string) => {
            update(state => {
                const newSelected = new Set(state.selectedFixes);
                if (newSelected.has(fixId)) {
                    newSelected.delete(fixId);
                } else {
                    newSelected.add(fixId);
                }
                return { ...state, selectedFixes: newSelected };
            });
        },

        setPreviewFix: (fix: AutoFix | null) =>
            update(state => ({ ...state, previewFix: fix })),

        async applyFixes(fixIds: string[]) {
            update(state => ({ ...state, loading: true, error: null }));

            try {
                const response = await fetch('/api/fixes/apply', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fixIds })
                });

                if (!response.ok) throw new Error('Failed to apply fixes');

                const result = await response.json();

                // Refresh fixes after application
                const currentState = get(dashboardStore);
                if (currentState.selectedProject) {
                    await this.loadFixes(currentState.selectedProject.id);
                }

                return result;
            } catch (error) {
                update(state => ({
                    ...state,
                    error: error instanceof Error ? error.message : 'Unknown error',
                    loading: false
                }));
                throw error;
            }
        }
    };
}

export const autoFixStore = createAutoFixStore();

// Store para Multi-Proyecto
interface MultiProjectState {
    projects: Project[];
    portfolioHealth: PortfolioHealth | null;
    loading: boolean;
    error: string | null;
}

function createMultiProjectStore() {
    const { subscribe, set, update } = writable<MultiProjectState>({
        projects: [],
        portfolioHealth: null,
        loading: false,
        error: null
    });

    return {
        subscribe,

        async loadProjects(organizationId: string) {
            update(state => ({ ...state, loading: true, error: null }));

            try {
                const [projectsRes, healthRes] = await Promise.all([
                    fetch(`/api/organizations/${organizationId}/projects`),
                    fetch(`/api/organizations/${organizationId}/health`)
                ]);

                if (!projectsRes.ok || !healthRes.ok) {
                    throw new Error('Failed to load multi-project data');
                }

                const projects = await projectsRes.json();
                const health = await healthRes.json();

                update(state => ({
                    ...state,
                    projects,
                    portfolioHealth: health,
                    loading: false
                }));
            } catch (error) {
                update(state => ({
                    ...state,
                    error: error instanceof Error ? error.message : 'Unknown error',
                    loading: false
                }));
            }
        }
    };
}

export const multiProjectStore = createMultiProjectStore();

// Store para CI/CD y DORA
interface CICDState {
    pipelines: Pipeline[];
    doraMetrics: DORAMetrics | null;
    loading: boolean;
    error: string | null;
}

function createCICDStore() {
    const { subscribe, set, update } = writable<CICDState>({
        pipelines: [],
        doraMetrics: null,
        loading: false,
        error: null
    });

    return {
        subscribe,

        async loadCICDData(projectId: string, timeRange: TimeRange) {
            update(state => ({ ...state, loading: true, error: null }));

            try {
                const [pipelinesRes, doraRes] = await Promise.all([
                    fetch(`/api/projects/${projectId}/pipelines`),
                    fetch('/api/dora-metrics', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ projectId, timeRange })
                    })
                ]);

                if (!pipelinesRes.ok || !doraRes.ok) {
                    throw new Error('Failed to load CI/CD data');
                }

                const pipelines = await pipelinesRes.json();
                const doraMetrics = await doraRes.json();

                update(state => ({
                    ...state,
                    pipelines,
                    doraMetrics,
                    loading: false
                }));
            } catch (error) {
                update(state => ({
                    ...state,
                    error: error instanceof Error ? error.message : 'Unknown error',
                    loading: false
                }));
            }
        }
    };
}

export const cicdStore = createCICDStore();

// Derived stores
export const isLoading = derived(
    [dashboardStore, technicalDebtStore, autoFixStore, multiProjectStore, cicdStore],
    ([$dashboard, $debt, $fixes, $projects, $cicd]) =>
        $dashboard.loading || $debt.loading || $fixes.loading || $projects.loading || $cicd.loading
);

export const hasErrors = derived(
    [dashboardStore, technicalDebtStore, autoFixStore, multiProjectStore, cicdStore],
    ([$dashboard, $debt, $fixes, $projects, $cicd]) => {
        const errors = [
            $dashboard.error,
            $debt.error,
            $fixes.error,
            $projects.error,
            $cicd.error
        ].filter(Boolean);

        return errors.length > 0 ? errors : null;
    }
);