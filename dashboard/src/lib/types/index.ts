// Core types for CodeAnt Dashboard

export interface User {
    id: string;
    name: string;
    email: string;
    role: UserRole;
    organization: Organization;
    preferences: UserPreferences;
}

export type UserRole = 'developer' | 'tech_lead' | 'manager' | 'security' | 'qa';

export interface Organization {
    id: string;
    name: string;
    plan: 'starter' | 'professional' | 'enterprise';
    projects: Project[];
}

export interface Project {
    id: string;
    name: string;
    description: string;
    language: string;
    repository: string;
    lastAnalysis?: AnalysisResult;
    qualityScore: number;
    issueCount: number;
    criticalIssues: number;
}

export interface TimeRange {
    period: '24h' | '7d' | '30d' | '90d' | 'custom';
    start: Date | null;
    end: Date | null;
}

export interface DashboardData {
    overview: OverviewData;
    qualityTrends: QualityTrendData[];
    issuesDistribution: IssueDistribution;
    vulnerabilities: Vulnerability[];
    compliance: ComplianceStatus;
    aiInsights: AIInsight[];
    recentAnalyses: AnalysisResult[];
}

export interface OverviewData {
    qualityScore: number;
    totalIssues: number;
    criticalIssues: number;
    securityScore: number;
    technicalDebt: number;
    complexityAverage: number;
}

export interface QualityTrendData {
    timestamp: string;
    qualityScore: number;
    issueCount: number;
    coverage: number;
    complexity: number;
}

export interface Issue {
    id: string;
    category: IssueCategory;
    severity: IssueSeverity;
    title: string;
    description: string;
    file: string;
    line: number;
    column: number;
    codeSnippet?: string;
    suggestedFix?: string;
    aiExplanation?: string;
}

export type IssueCategory = 'security' | 'performance' | 'maintainability' | 'style' | 'bug' | 'vulnerability';
export type IssueSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';

export interface IssueDistribution {
    byCategory: Record<IssueCategory, number>;
    bySeverity: Record<IssueSeverity, number>;
    byFile: Array<{ file: string; count: number }>;
}

export interface Vulnerability {
    id: string;
    title: string;
    severity: IssueSeverity;
    cve?: string;
    cvssScore?: number;
    description: string;
    recommendation: string;
    affectedFiles: string[];
    status: 'open' | 'in_progress' | 'resolved' | 'false_positive';
    type?: string;
}

export interface ComplianceStatus {
    owasp: ComplianceReport;
    pci: ComplianceReport;
    hipaa: ComplianceReport;
    gdpr: ComplianceReport;
    checks?: ComplianceCheck[];
}

export interface ComplianceReport {
    name: string;
    status: 'compliant' | 'partial' | 'non_compliant';
    score: number;
    issues: string[];
}

export interface AIInsight {
    id: string;
    type: 'recommendation' | 'pattern' | 'prediction' | 'anomaly';
    title: string;
    description: string;
    confidence: number;
    impact: 'high' | 'medium' | 'low';
    actionable: boolean;
    suggestedAction?: string;
}

export interface AnalysisResult {
    id: string;
    projectId: string;
    timestamp: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress?: number;
    violations: Issue[];
    metrics: CodeMetrics;
    duration: number;
}

export interface CodeMetrics {
    linesOfCode: number;
    coverage: number;
    duplicateCode: number;
    complexity: number;
    maintainabilityIndex: number;
}

export interface DashboardLayout {
    columns: number;
    rows: 'auto' | number;
    breakpoints: {
        lg: number;
        md: number;
        sm: number;
        xs: number;
    };
}

export interface DashboardWidget {
    id: string;
    component: string;
    position: {
        x: number;
        y: number;
        w: number;
        h: number;
    };
    props: Record<string, any>;
}

export interface DashboardConfig {
    layout: DashboardLayout;
    widgets: DashboardWidget[];
    permissions: UserPermissions;
}

export interface UserPermissions {
    canViewCode: boolean;
    canApplyFixes: boolean;
    canCreateRules: boolean;
    canViewMetrics: boolean;
    canExportReports: boolean;
    canManageTeam?: boolean;
    canViewBusinessMetrics?: boolean;
    canViewSecurityDetails?: boolean;
}

export interface UserPreferences {
    theme: 'light' | 'dark' | 'system';
    language: 'es' | 'en';
    dashboardLayout: string;
    notificationsEnabled: boolean;
}

export interface DashboardFilters {
    projects?: string[];
    severity?: IssueSeverity[];
    category?: IssueCategory[];
    dateRange?: TimeRange;
    search?: string;
}

export interface ActiveAnalysis {
    id: string;
    projectId: string;
    projectName: string;
    status: 'initializing' | 'scanning' | 'analyzing' | 'finalizing';
    progress: number;
    startTime: string;
    estimatedCompletion?: string;
}

export interface CompletedAnalysis {
    id: string;
    projectId: string;
    projectName: string;
    completedAt: string;
    result: 'success' | 'failed' | 'partial';
    issuesFound: number;
    qualityScore: number;
}

export interface Notification {
    id: string;
    type: 'info' | 'success' | 'warning' | 'error';
    title: string;
    message: string;
    timestamp: string;
    read: boolean;
    actionUrl?: string;
}

export interface RealTimeMessage {
    type: string;
    data: any;
    timestamp: string;
    payload?: any;
}

export interface FileQualityData {
    fileName: string;
    filePath: string;
    qualityScore: number;
    linesOfCode: number;
    issueCount: number;
    complexity: number;
    coverage?: number;
}

export interface IssueFilters {
    search?: string;
    severity?: IssueSeverity[];
    category?: IssueCategory[];
    status?: string[];
    assignee?: string[];
}

export interface SortConfig {
    field: string;
    direction: 'asc' | 'desc';
}

export type BulkAction = 'fix' | 'ignore' | 'assign' | 'export';

export interface ChartConfig {
    type: 'line' | 'bar' | 'pie' | 'scatter' | 'heatmap' | 'treemap';
    title: string;
    description: string;
    colorScheme?: 'quality' | 'complexity' | 'security';
    interactive?: boolean;
}

export interface SecurityMetrics {
    criticalCount: number;
    criticalTrend: number;
    securityScore: number;
    scoreTrend: number;
    compliancePercentage: number;
    complianceTrend: number;
    attackSurfaceSize: number;
    attackSurfaceTrend: number;
}

export interface ThreatModel {
    id: string;
    name: string;
    components: Array<{
        id: string;
        name: string;
        type: string;
        threats: Threat[];
    }>;
    dataFlows: Array<{
        from: string;
        to: string;
        data: string;
        protocol: string;
    }>;
}

export interface SecurityFix {
    id: string;
    vulnerabilityId: string;
    type: 'patch' | 'configuration' | 'code_change';
    description: string;
    code?: string;
    impact: string;
    estimatedTime: number;
}

// Additional types for stores
export interface DashboardPermissions {
    canViewAnalytics: boolean;
    canManageProjects: boolean;
    canConfigureSecurity: boolean;
    canViewTeamMetrics: boolean;
    canExportData: boolean;
    canManageUsers: boolean;
}

export interface WidgetConfig {
    id: string;
    type: string;
    title: string;
    position: { x: number; y: number; w: number; h: number };
    settings: Record<string, any>;
}

export interface IssueDistributionData {
    byCategory: Record<IssueCategory, number>;
    bySeverity: Record<IssueSeverity, number>;
}

export interface ThreatModelComponent {
    id: string;
    name: string;
    type: 'process' | 'datastore' | 'external_entity' | 'data_flow';
    threats: Threat[];
}

export interface Threat {
    id: string;
    type: 'spoofing' | 'tampering' | 'repudiation' | 'information_disclosure' | 'denial_of_service' | 'elevation_of_privilege';
    title: string;
    description: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    mitigations: Mitigation[];
}

export interface Mitigation {
    id: string;
    description: string;
    implemented: boolean;
    implementationDetails?: string;
}

export interface ComplianceCheck {
    id: string;
    standard: string;
    requirement: string;
    status: 'passed' | 'failed' | 'not_applicable';
    details: string;
}

export interface ComplianceStatusExtended {
    standard: string;
    checks: ComplianceCheck[];
    overallStatus: 'compliant' | 'partial' | 'non_compliant';
    score: number;
}

// Redefine ComplianceStatus to be more detailed
export interface DetailedComplianceStatus {
    checks: ComplianceCheck[];
    standards: ComplianceStatusExtended[];
}

// Type for vulnerability with additional properties
export interface VulnerabilityExtended extends Vulnerability {
    type: string;
    fixAvailable: boolean;
    exploitAvailable: boolean;
    affectedVersions?: string[];
}

// Enhanced RealTimeMessage
export interface RealTimeMessageTyped extends RealTimeMessage {
    type: 'notification' | 'analysis_update' | 'security_alert' | 'system_message' | 'heartbeat';
    payload: any;
}
