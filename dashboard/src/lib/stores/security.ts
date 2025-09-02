import type {
  ComplianceStatus,
  Mitigation,
  SecurityFix,
  Threat,
  ThreatModel,
  Vulnerability
} from '$lib/types';
import { derived, writable } from 'svelte/store';

interface SecurityState {
  vulnerabilities: Vulnerability[];
  complianceStatus: ComplianceStatus | null;
  threatModel: ThreatModel | null;
  securityFixes: SecurityFix[];
  loading: boolean;
  error: Error | null;
}

function createSecurityStore() {
  const { subscribe, set, update } = writable<SecurityState>({
    vulnerabilities: [],
    complianceStatus: null,
    threatModel: null,
    securityFixes: [],
    loading: false,
    error: null
  });

  return {
    subscribe,

    async loadVulnerabilities(projectId: string) {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch(`/api/security/vulnerabilities/${projectId}`);
        if (!response.ok) throw new Error('Failed to load vulnerabilities');

        const vulnerabilities = await response.json();
        update(state => ({ ...state, vulnerabilities, loading: false }));
      } catch (error) {
        update(state => ({
          ...state,
          error: error as Error,
          loading: false
        }));
      }
    },

    async checkCompliance(projectId: string, standards: string[]) {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch('/api/security/compliance', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ projectId, standards })
        });

        if (!response.ok) throw new Error('Failed to check compliance');

        const complianceStatus = await response.json();
        update(state => ({ ...state, complianceStatus, loading: false }));
      } catch (error) {
        update(state => ({
          ...state,
          error: error as Error,
          loading: false
        }));
      }
    },

    async generateThreatModel(projectId: string) {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch('/api/security/threat-model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ projectId })
        });

        if (!response.ok) throw new Error('Failed to generate threat model');

        const threatModel = await response.json();
        update(state => ({ ...state, threatModel, loading: false }));
      } catch (error) {
        update(state => ({
          ...state,
          error: error as Error,
          loading: false
        }));
      }
    },

    async generateSecurityFixes(vulnerabilityIds: string[]) {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch('/api/security/fixes/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ vulnerabilityIds })
        });

        if (!response.ok) throw new Error('Failed to generate fixes');

        const fixes = await response.json();
        update(state => ({
          ...state,
          securityFixes: fixes,
          loading: false
        }));

        return fixes;
      } catch (error) {
        update(state => ({
          ...state,
          error: error as Error,
          loading: false
        }));
        throw error;
      }
    },

    async applySecurityFix(fixId: string) {
      try {
        const response = await fetch(`/api/security/fixes/${fixId}/apply`, {
          method: 'POST'
        });

        if (!response.ok) throw new Error('Failed to apply fix');

        const result = await response.json();

        // Remove the fixed vulnerability
        update(state => ({
          ...state,
          vulnerabilities: state.vulnerabilities.filter(
            v => v.id !== result.vulnerabilityId
          ),
          securityFixes: state.securityFixes.filter(f => f.id !== fixId)
        }));

        return result;
      } catch (error) {
        console.error('Failed to apply security fix:', error);
        throw error;
      }
    },

    addThreat(componentId: string, threat: Threat) {
      update(state => {
        if (!state.threatModel) return state;

        return {
          ...state,
          threatModel: {
            ...state.threatModel,
            components: state.threatModel.components.map(comp =>
              comp.id === componentId
                ? { ...comp, threats: [...comp.threats, threat] }
                : comp
            )
          }
        };
      });
    },

    addMitigation(threatId: string, mitigation: Mitigation) {
      update(state => {
        if (!state.threatModel) return state;

        return {
          ...state,
          threatModel: {
            ...state.threatModel,
            components: state.threatModel.components.map(comp => ({
              ...comp,
              threats: comp.threats.map(threat =>
                threat.id === threatId
                  ? { ...threat, mitigations: [...threat.mitigations, mitigation] }
                  : threat
              )
            }))
          }
        };
      });
    },

    reset() {
      set({
        vulnerabilities: [],
        complianceStatus: null,
        threatModel: null,
        securityFixes: [],
        loading: false,
        error: null
      });
    }
  };
}

export const securityStore = createSecurityStore();

// Derived stores
export const criticalVulnerabilities = derived(
  securityStore,
  $security => $security.vulnerabilities.filter(v => v.severity === 'critical')
);

export const vulnerabilityStats = derived(
  securityStore,
  $security => {
    const stats = {
      total: $security.vulnerabilities.length,
      bySeverity: {
        critical: 0,
        high: 0,
        medium: 0,
        low: 0,
        info: 0
      },
      byType: {} as Record<string, number>
    };

    $security.vulnerabilities.forEach(vuln => {
      stats.bySeverity[vuln.severity]++;
      if (vuln.type) {
        stats.byType[vuln.type] = (stats.byType[vuln.type] || 0) + 1;
      }
    });

    return stats;
  }
);

export const complianceScore = derived(
  securityStore,
  $security => {
    if (!$security.complianceStatus) return null;

    const checks = $security.complianceStatus.checks || [];
    const passedChecks = checks.filter(check => check.status === 'passed').length;

    return {
      score: checks.length > 0 ? Math.round((passedChecks / checks.length) * 100) : 0,
      passed: passedChecks,
      total: checks.length
    };
  }
);

export const threatSummary = derived(
  securityStore,
  $security => {
    if (!$security.threatModel) return null;

    const threats = $security.threatModel.components.flatMap(c => c.threats);
    const mitigations = threats.flatMap(t => t.mitigations);

    return {
      totalThreats: threats.length,
      totalMitigations: mitigations.length,
      unmitigatedThreats: threats.filter(t => t.mitigations.length === 0).length,
      threatsBySeverity: threats.reduce((acc, threat) => {
        acc[threat.severity] = (acc[threat.severity] || 0) + 1;
        return acc;
      }, {} as Record<string, number>)
    };
  }
);
