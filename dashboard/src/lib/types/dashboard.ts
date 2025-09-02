/**
 * Dashboard Types - Tipos para el sistema de dashboard interactivo
 */

// Programming language type
export type ProgrammingLanguage = 'TypeScript' | 'JavaScript' | 'Python' | 'Java' | 'Go' | 'Rust' | 'C++' | 'C#' | 'Ruby' | 'PHP' | 'Swift' | 'Kotlin' | 'Unknown';

// Tipos para Deuda Técnica
export interface TechnicalDebtMetrics {
    totalDebt: number;
    debtRatio: number;
    estimatedHours: number;
    estimatedCost: number;
    severity: {
        critical: number;
        high: number;
        medium: number;
        low: number;
    };
    categories: {
        codeSmells: number;
        bugs: number;
        vulnerabilities: number;
        duplications: number;
    };
}

export interface DebtEvolution {
    date: Date;
    totalDebt: number;
    debtRatio: number;
    categories: Record<string, number>;
}

export interface DebtComponent {
    name: string;
    path: string;
    debt: number;
    debtRatio: number;
    issues: number;
    criticalIssues: number;
}

export interface DebtROI {
    component: string;
    currentDebt: number;
    remediationCost: number;
    expectedBenefit: number;
    roi: number;
    priority: 'critical' | 'high' | 'medium' | 'low';
}

// Tipos para Fixes Automáticos
export interface AutoFix {
    id: string;
    issueId: string;
    type: 'security' | 'performance' | 'quality' | 'style';
    severity: 'critical' | 'high' | 'medium' | 'low';
    title: string;
    description: string;
    filePath: string;
    startLine: number;
    endLine: number;
    originalCode: string;
    proposedCode: string;
    confidence: number;
    impact: {
        linesChanged: number;
        testsAffected: number;
        dependencies: string[];
    };
    explanation: string;
    educationalContent?: string;
}

export interface FixBatch {
    id: string;
    name: string;
    fixes: AutoFix[];
    totalImpact: {
        filesAffected: number;
        linesChanged: number;
        issuesResolved: number;
    };
    estimatedTime: number;
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
}

export interface FixHistory {
    id: string;
    fixId: string;
    appliedAt: Date;
    appliedBy: string;
    status: 'success' | 'failed' | 'rolled_back';
    beforeMetrics: Record<string, any>;
    afterMetrics: Record<string, any>;
    rollbackAvailable: boolean;
}

// Tipos para Multi-Proyecto
export interface Project {
    id: string;
    name: string;
    language: ProgrammingLanguage;
    lastAnalysis: Date;
    metrics: ProjectMetrics;
    status: 'healthy' | 'warning' | 'critical';
}

export interface ProjectMetrics {
    qualityScore: number;
    technicalDebt: number;
    coverage: number;
    issues: {
        total: number;
        critical: number;
        high: number;
    };
    codeSize: {
        lines: number;
        files: number;
        functions: number;
    };
}

export interface PortfolioHealth {
    totalProjects: number;
    healthyProjects: number;
    averageQualityScore: number;
    totalTechnicalDebt: number;
    criticalIssues: number;
    trends: {
        quality: TrendDirection;
        debt: TrendDirection;
        issues: TrendDirection;
    };
}

// Tipos para CI/CD Integration
export interface Pipeline {
    id: string;
    name: string;
    status: 'success' | 'failed' | 'running' | 'pending';
    lastRun: Date;
    duration: number;
    stages: PipelineStage[];
}

export interface PipelineStage {
    name: string;
    status: 'success' | 'failed' | 'running' | 'pending' | 'skipped';
    duration: number;
    logs?: string[];
}

export interface DORAMetrics {
    deploymentFrequency: {
        value: number;
        unit: 'per_day' | 'per_week' | 'per_month';
        trend: TrendDirection;
    };
    leadTimeForChanges: {
        value: number;
        unit: 'hours' | 'days';
        trend: TrendDirection;
    };
    changeFailureRate: {
        value: number;
        percentage: number;
        trend: TrendDirection;
    };
    timeToRestoreService: {
        value: number;
        unit: 'minutes' | 'hours';
        trend: TrendDirection;
    };
}

// Tipos para Embeddings y Búsqueda Semántica
export interface SemanticSearchResult {
    id: string;
    filePath: string;
    functionName?: string;
    similarity: number;
    snippet: string;
    explanation: string;
    embeddings?: number[];
}

export interface CodeCluster {
    id: string;
    name: string;
    description: string;
    files: string[];
    centroid: number[];
    cohesion: number;
    mainPurpose: string;
}

export interface EmbeddingVisualization {
    nodes: Array<{
        id: string;
        label: string;
        x: number;
        y: number;
        type: 'file' | 'function' | 'class';
        cluster?: string;
    }>;
    edges: Array<{
        source: string;
        target: string;
        weight: number;
    }>;
}

// Tipos para Reglas Personalizadas
export interface CustomRule {
    id: string;
    name: string;
    description: string;
    naturalLanguageInput: string;
    generatedPattern: string;
    category: string;
    severity: 'info' | 'warning' | 'error';
    language: ProgrammingLanguage[];
    effectiveness: {
        truePositives: number;
        falsePositives: number;
        accuracy: number;
    };
    status: 'active' | 'inactive' | 'testing';
    createdBy: string;
    createdAt: Date;
}

export interface RuleTemplate {
    id: string;
    name: string;
    description: string;
    category: string;
    basePattern: string;
    parameters: RuleParameter[];
    examples: string[];
    popularity: number;
}

export interface RuleParameter {
    name: string;
    type: 'string' | 'number' | 'boolean' | 'regex';
    description: string;
    defaultValue?: any;
    required: boolean;
}

// Tipos comunes
export type TrendDirection = 'up' | 'down' | 'stable';

export interface TimeRange {
    start: Date;
    end: Date;
    period?: '1d' | '7d' | '30d' | '90d' | '1y' | 'custom';
}

export interface DashboardFilter {
    projects?: string[];
    severity?: ('critical' | 'high' | 'medium' | 'low')[];
    categories?: string[];
    languages?: ProgrammingLanguage[];
    timeRange?: TimeRange;
}

export interface DashboardWidget {
    id: string;
    type: string;
    title: string;
    position: {
        x: number;
        y: number;
        w: number;
        h: number;
    };
    config: Record<string, any>;
    data?: any;
}

export interface UserDashboardLayout {
    id: string;
    userId: string;
    name: string;
    widgets: DashboardWidget[];
    filters: DashboardFilter;
    isDefault: boolean;
}
