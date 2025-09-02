import { dashboardStore, technicalDebtStore } from '$lib/stores/dashboard';
import { render, screen, waitFor } from '@testing-library/svelte';
import { writable } from 'svelte/store';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import TechnicalDebtDashboard from '../TechnicalDebtDashboard.svelte';

// Mock the stores
vi.mock('$lib/stores/dashboard', () => ({
    technicalDebtStore: {
        subscribe: vi.fn(),
        loadDebtData: vi.fn(),
        reset: vi.fn()
    },
    dashboardStore: {
        subscribe: vi.fn()
    }
}));

describe('TechnicalDebtDashboard', () => {
    const mockTechnicalDebtData = {
        metrics: {
            totalDebt: 1500,
            debtRatio: 0.15,
            estimatedHours: 150,
            estimatedCost: 15000,
            severity: {
                critical: 10,
                high: 25,
                medium: 40,
                low: 75
            },
            categories: {
                codeSmells: 80,
                bugs: 30,
                vulnerabilities: 15,
                duplications: 25
            }
        },
        evolution: [
            { date: new Date('2024-01-01'), totalDebt: 1200, debtRatio: 0.12, categories: {} },
            { date: new Date('2024-02-01'), totalDebt: 1350, debtRatio: 0.13, categories: {} },
            { date: new Date('2024-03-01'), totalDebt: 1500, debtRatio: 0.15, categories: {} }
        ],
        components: [
            { name: 'API Module', path: '/src/api', debt: 400, debtRatio: 0.25, issues: 40, criticalIssues: 5 },
            { name: 'UI Components', path: '/src/components', debt: 300, debtRatio: 0.20, issues: 30, criticalIssues: 3 },
            { name: 'Utils', path: '/src/utils', debt: 200, debtRatio: 0.10, issues: 20, criticalIssues: 2 }
        ],
        loading: false,
        error: null
    };

    const mockDashboardData = {
        selectedProject: { id: 'project-1', name: 'Test Project' },
        selectedTimeRange: { period: '30d', start: new Date(), end: new Date() }
    };

    beforeEach(() => {
        vi.clearAllMocks();

        // Setup store mocks
        const technicalDebtMock = writable(mockTechnicalDebtData);
        const dashboardMock = writable(mockDashboardData);

        technicalDebtStore.subscribe.mockImplementation(technicalDebtMock.subscribe);
        dashboardStore.subscribe.mockImplementation(dashboardMock.subscribe);
    });

    it('renders dashboard header correctly', () => {
        render(TechnicalDebtDashboard);

        expect(screen.getByText('Technical Debt Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Comprehensive analysis of technical debt, evolution trends, and remediation strategies')).toBeInTheDocument();
    });

    it('displays key metrics overview', () => {
        render(TechnicalDebtDashboard);

        // Check total debt
        expect(screen.getByText('Total Debt')).toBeInTheDocument();
        expect(screen.getByText('1.5K')).toBeInTheDocument();
        expect(screen.getByText('~150 hours')).toBeInTheDocument();

        // Check debt ratio
        expect(screen.getByText('Debt Ratio')).toBeInTheDocument();
        expect(screen.getByText('15.0%')).toBeInTheDocument();

        // Check estimated cost
        expect(screen.getByText('Estimated Cost')).toBeInTheDocument();
        expect(screen.getByText('$15,000')).toBeInTheDocument();

        // Check critical issues
        expect(screen.getByText('Critical Issues')).toBeInTheDocument();
        expect(screen.getByText('10')).toBeInTheDocument();
    });

    it('shows severity breakdown correctly', () => {
        render(TechnicalDebtDashboard);

        expect(screen.getByText('Debt by Severity')).toBeInTheDocument();
        expect(screen.getByText('10 issues')).toBeInTheDocument(); // Critical
        expect(screen.getByText('25 issues')).toBeInTheDocument(); // High
        expect(screen.getByText('40 issues')).toBeInTheDocument(); // Medium
        expect(screen.getByText('75 issues')).toBeInTheDocument(); // Low
    });

    it('displays category breakdown', () => {
        render(TechnicalDebtDashboard);

        expect(screen.getByText('Debt by Category')).toBeInTheDocument();
        expect(screen.getByText('80')).toBeInTheDocument(); // Code Smells
        expect(screen.getByText('30')).toBeInTheDocument(); // Bugs
        expect(screen.getByText('15')).toBeInTheDocument(); // Vulnerabilities
        expect(screen.getByText('25')).toBeInTheDocument(); // Duplications
    });

    it('loads debt data when project is selected', async () => {
        render(TechnicalDebtDashboard);

        await waitFor(() => {
            expect(technicalDebtStore.loadDebtData).toHaveBeenCalledWith(
                'project-1',
                mockDashboardData.selectedTimeRange
            );
        });
    });

    it('shows loading state', () => {
        const loadingData = { ...mockTechnicalDebtData, loading: true, metrics: null };
        const technicalDebtMock = writable(loadingData);
        technicalDebtStore.subscribe.mockImplementation(technicalDebtMock.subscribe);

        render(TechnicalDebtDashboard);

        expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
    });

    it('displays error state', () => {
        const errorData = { ...mockTechnicalDebtData, error: 'Failed to load data', metrics: null };
        const technicalDebtMock = writable(errorData);
        technicalDebtStore.subscribe.mockImplementation(technicalDebtMock.subscribe);

        render(TechnicalDebtDashboard);

        expect(screen.getByText('Error loading technical debt data: Failed to load data')).toBeInTheDocument();
    });

    it('shows empty state when no project selected', () => {
        const noDashboardData = { ...mockDashboardData, selectedProject: null };
        const dashboardMock = writable(noDashboardData);
        dashboardStore.subscribe.mockImplementation(dashboardMock.subscribe);

        render(TechnicalDebtDashboard);

        expect(screen.getByText('Select a project to view technical debt metrics')).toBeInTheDocument();
    });

    it('reloads data when time range changes', async () => {
        const { component } = render(TechnicalDebtDashboard);

        // Clear previous calls
        vi.clearAllMocks();

        // Update time range
        const newTimeRange = { period: '7d', start: new Date(), end: new Date() };
        const updatedDashboardData = { ...mockDashboardData, selectedTimeRange: newTimeRange };
        const dashboardMock = writable(updatedDashboardData);
        dashboardStore.subscribe.mockImplementation(dashboardMock.subscribe);

        // Trigger update
        component.$set({});

        await waitFor(() => {
            expect(technicalDebtStore.loadDebtData).toHaveBeenCalledWith(
                'project-1',
                newTimeRange
            );
        });
    });
});
