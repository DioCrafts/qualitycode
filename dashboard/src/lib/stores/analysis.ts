import type {
    AnalysisResult,
    IssueCategory,
    IssueFilters,
    IssueSeverity
} from '$lib/types';
import { derived, writable } from 'svelte/store';

interface AnalysisConfig {
    rulesets: string[];
    skipPatterns: string[];
    autoFix: boolean;
    deepAnalysis: boolean;
}

interface AnalysisState {
    currentAnalysis: AnalysisResult | null;
    analysisHistory: AnalysisResult[];
    selectedIssues: string[];
    issueFilters: IssueFilters;
}

function getDefaultIssueFilters(): IssueFilters {
    return {
        search: '',
        severity: [],
        category: [],
        status: [],
        assignee: []
    };
}

function createAnalysisStore() {
    const { subscribe, set, update } = writable<AnalysisState>({
        currentAnalysis: null,
        analysisHistory: [],
        selectedIssues: [],
        issueFilters: getDefaultIssueFilters()
    });

    return {
        subscribe,

        setCurrentAnalysis: (analysis: AnalysisResult) =>
            update(state => ({ ...state, currentAnalysis: analysis })),

        addToHistory: (analysis: AnalysisResult) =>
            update(state => ({
                ...state,
                analysisHistory: [analysis, ...state.analysisHistory].slice(0, 50)
            })),

        selectIssues: (issueIds: string[]) =>
            update(state => ({ ...state, selectedIssues: issueIds })),

        toggleIssueSelection: (issueId: string) =>
            update(state => ({
                ...state,
                selectedIssues: state.selectedIssues.includes(issueId)
                    ? state.selectedIssues.filter(id => id !== issueId)
                    : [...state.selectedIssues, issueId]
            })),

        selectAllIssues: () =>
            update(state => ({
                ...state,
                selectedIssues: state.currentAnalysis?.violations.map(v => v.id) || []
            })),

        clearSelection: () =>
            update(state => ({ ...state, selectedIssues: [] })),

        setIssueFilters: (filters: IssueFilters) =>
            update(state => ({ ...state, issueFilters: filters })),

        async runAnalysis(projectId: string, config: AnalysisConfig) {
            try {
                const response = await fetch('/api/analysis/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ projectId, config })
                });

                if (!response.ok) throw new Error('Analysis failed');

                const analysis = await response.json();

                update(state => ({
                    ...state,
                    currentAnalysis: analysis,
                    analysisHistory: [analysis, ...state.analysisHistory].slice(0, 50),
                    selectedIssues: []
                }));

                return analysis;
            } catch (error) {
                console.error('Analysis failed:', error);
                throw error;
            }
        },

        async applyFixes(issueIds: string[]) {
            try {
                const response = await fetch('/api/fixes/apply', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ issueIds })
                });

                if (!response.ok) throw new Error('Failed to apply fixes');

                const results = await response.json();

                update(state => {
                    if (state.currentAnalysis) {
                        return {
                            ...state,
                            currentAnalysis: {
                                ...state.currentAnalysis,
                                violations: state.currentAnalysis.violations.filter(
                                    v => !issueIds.includes(v.id)
                                )
                            },
                            selectedIssues: state.selectedIssues.filter(
                                id => !issueIds.includes(id)
                            )
                        };
                    }
                    return state;
                });

                return results;
            } catch (error) {
                console.error('Failed to apply fixes:', error);
                throw error;
            }
        },

        reset() {
            set({
                currentAnalysis: null,
                analysisHistory: [],
                selectedIssues: [],
                issueFilters: getDefaultIssueFilters()
            });
        }
    };
}

export const analysisStore = createAnalysisStore();

// Derived stores
export const criticalIssues = derived(
    analysisStore,
    $analysis => $analysis.currentAnalysis?.violations.filter(v => v.severity === 'critical') || []
);

export const filteredIssues = derived(
    analysisStore,
    $analysis => {
        if (!$analysis.currentAnalysis) return [];

        const { violations } = $analysis.currentAnalysis;
        const { search, severity, category } = $analysis.issueFilters;

        return violations.filter(issue => {
            // Search filter
            if (search && !issue.title.toLowerCase().includes(search.toLowerCase()) &&
                !issue.description.toLowerCase().includes(search.toLowerCase())) {
                return false;
            }

            // Severity filter
            if (severity && severity.length > 0 && !severity.includes(issue.severity)) {
                return false;
            }

            // Category filter
            if (category && category.length > 0 && !category.includes(issue.category)) {
                return false;
            }

            return true;
        });
    }
);

export const issueStats = derived(
    analysisStore,
    $analysis => {
        if (!$analysis.currentAnalysis) {
            return {
                total: 0,
                bySeverity: {} as Record<IssueSeverity, number>,
                byCategory: {} as Record<IssueCategory, number>
            };
        }

        const { violations } = $analysis.currentAnalysis;

        const bySeverity = violations.reduce((acc, issue) => {
            acc[issue.severity] = (acc[issue.severity] || 0) + 1;
            return acc;
        }, {} as Record<IssueSeverity, number>);

        const byCategory = violations.reduce((acc, issue) => {
            acc[issue.category] = (acc[issue.category] || 0) + 1;
            return acc;
        }, {} as Record<IssueCategory, number>);

        return {
            total: violations.length,
            bySeverity,
            byCategory
        };
    }
);
