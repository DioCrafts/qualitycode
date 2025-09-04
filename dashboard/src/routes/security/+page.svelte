<script lang="ts">
    import { onMount } from "svelte";
    import { fade } from "svelte/transition";

    let loading = true;
    let error = null;
    
    // Datos mock para la demostración
    const criticalVulns = 2;
    const highVulns = 15;
    const mediumVulns = 34;
    const lowVulns = 8;
    
    // Timeline items
    const timelineItems = [
        {
            type: 'critical',
            icon: 'fas fa-exclamation-triangle',
            event: 'Critical SQL Injection detectado',
            details: 'api/auth/login.js - Línea 47',
            time: 'Hace 15 minutos'
        },
        {
            type: 'high',
            icon: 'fas fa-key',
            event: 'API Key expuesta en código',
            details: 'config/database.js',
            time: 'Hace 2 horas'
        },
        {
            type: 'medium',
            icon: 'fas fa-shield-alt',
            event: 'Dependencia vulnerable actualizada',
            details: 'lodash@4.17.20 → 4.17.21',
            time: 'Hace 4 horas'
        },
        {
            type: 'low',
            icon: 'fas fa-check-circle',
            event: 'Scan completado exitosamente',
            details: '23 archivos analizados',
            time: 'Hace 6 horas'
        }
    ];
    
    // Stats cards
    const statsCards = [
        {
            type: 'secrets',
            icon: 'fas fa-key',
            title: 'Secrets Detectados',
            value: 7,
            change: '+2 esta semana',
            trend: 'negative'
        },
        {
            type: 'dependencies',
            icon: 'fas fa-cube',
            title: 'Dependencias Vulnerables',
            value: 12,
            change: '-3 desde la semana pasada',
            trend: 'positive'
        },
        {
            type: 'compliance',
            icon: 'fas fa-clipboard-check',
            title: 'Compliance Score',
            value: '78%',
            change: '+5% esta semana',
            trend: 'positive'
        },
        {
            type: 'iac',
            icon: 'fas fa-server',
            title: 'IaC Misconfigurations',
            value: 5,
            change: '-2 esta semana',
            trend: 'positive'
        }
    ];
    
    // Vulnerabilities
    const vulnerabilities = [
        {
            severity: 'critical',
            title: 'SQL Injection in Authentication',
            description: 'Unsanitized user input in login endpoint allows SQL injection attacks',
            file: 'api/auth/login.js:47',
            cwe: 'CWE-89',
            time: '15 min ago'
        },
        {
            severity: 'critical',
            title: 'Hard-coded Encryption Key',
            description: 'Cryptographic key stored directly in source code',
            file: 'utils/encryption.js:12',
            cwe: 'CWE-798',
            time: '1 hour ago'
        },
        {
            severity: 'high',
            title: 'Cross-Site Scripting (XSS)',
            description: 'User input reflected in HTML without proper encoding',
            file: 'components/Search.tsx:89',
            cwe: 'CWE-79',
            time: '2 hours ago'
        },
        {
            severity: 'high',
            title: 'Weak Cryptographic Hash',
            description: 'Use of MD5 hash function for sensitive data',
            file: 'auth/password.js:34',
            cwe: 'CWE-327',
            time: '3 hours ago'
        },
        {
            severity: 'medium',
            title: 'Missing Rate Limiting',
            description: 'API endpoints lack proper rate limiting controls',
            file: 'api/routes/user.js:156',
            cwe: 'CWE-770',
            time: '4 hours ago'
        }
    ];
    
    // Secret scanning results
    const secretsDetected = [
        {
            type: 'AWS Access Key',
            severity: 'critical',
            file: 'config/aws.js:23',
            time: 'Hace 2 horas'
        },
        {
            type: 'Database Password',
            severity: 'high',
            file: '.env.production:15',
            time: 'Hace 6 horas'
        },
        {
            type: 'JWT Secret',
            severity: 'high',
            file: 'auth/token.js:8',
            time: 'Hace 1 día'
        }
    ];
    
    // Compliance data
    const complianceData = [
        { title: 'OWASP Top 10', status: 'pass', percentage: 87, remaining: 2 },
        { title: 'CWE Top 25', status: 'warning', percentage: 72, remaining: 7 },
        { title: 'SOC 2', status: 'pass', percentage: 94, remaining: 1 },
        { title: 'GDPR', status: 'fail', percentage: 45, remaining: 12 },
        { title: 'NIST', status: 'warning', percentage: 68, remaining: 8 },
        { title: 'HIPAA', status: 'pass', percentage: 91, remaining: 2 }
    ];
    
    // Filtro activo para vulnerabilidades
    let activeFilter = 'all';
    
    // Función para filtrar vulnerabilidades
    function filterVulnerabilities(filter) {
        activeFilter = filter;
    }
    
    // Función para manejar acciones sobre vulnerabilidades
    const handleVulnAction = (vuln, action) => {
        // Aquí se implementaría la lógica para resolver vulnerabilidades
        console.log('Action on vulnerability:', vuln, action);
    };

    onMount(async () => {
        // Simulando carga de datos
        setTimeout(() => {
            loading = false;
        }, 1000);
    });
</script>

<div class="main-content">
    {#if loading}
        <div class="loading">Cargando análisis de seguridad...</div>
    {:else if error}
        <div class="error">Error: {error}</div>
    {:else}
        <header class="header" transition:fade={{ duration: 300 }}>
            <div class="header-left">
                <h2>Security Overview</h2>
                <p class="header-subtitle">SAST, secret scanning, dependencies y compliance</p>
            </div>
            <div class="header-actions">
                <button class="btn btn-secondary">
                    <i class="fas fa-download"></i> Security Report
                </button>
                <button class="btn btn-primary">
                    <i class="fas fa-play"></i> Run Security Scan
                </button>
            </div>
        </header>

        <section class="security-overview" transition:fade={{ duration: 300, delay: 100 }}>
            <div class="security-score-card">
                <div class="score-header">
                    <div>
                        <h3 class="score-title">Security Posture</h3>
                        <p class="score-subtitle">Análisis de vulnerabilidades y riesgos</p>
                    </div>
                    <span class="security-level">Needs Attention</span>
                </div>
                
                <div class="vuln-stats">
                    <div class="vuln-stat">
                        <div class="vuln-count critical">{criticalVulns}</div>
                        <div class="vuln-label">Critical</div>
                    </div>
                    <div class="vuln-stat">
                        <div class="vuln-count high">{highVulns}</div>
                        <div class="vuln-label">High</div>
                    </div>
                    <div class="vuln-stat">
                        <div class="vuln-count medium">{mediumVulns}</div>
                        <div class="vuln-label">Medium</div>
                    </div>
                    <div class="vuln-stat">
                        <div class="vuln-count low">{lowVulns}</div>
                        <div class="vuln-label">Low</div>
                    </div>
                </div>
            </div>

            <div class="threat-timeline">
                <div class="timeline-header">
                    <h3 class="timeline-title">Security Timeline</h3>
                    <p class="timeline-subtitle">Eventos de seguridad recientes</p>
                </div>
                
                {#each timelineItems as item}
                    <div class="timeline-item">
                        <div class="timeline-icon {item.type}">
                            <i class="{item.icon}"></i>
                        </div>
                        <div class="timeline-content">
                            <div class="timeline-event">{item.event}</div>
                            <div class="timeline-details">{item.details}</div>
                            <div class="timeline-time">{item.time}</div>
                        </div>
                    </div>
                {/each}
            </div>
        </section>

        <section class="stats-grid" transition:fade={{ duration: 300, delay: 150 }}>
            {#each statsCards as stat}
                <div class="stat-card {stat.type}">
                    <div class="stat-header">
                        <div class="stat-icon">
                            <i class="{stat.icon}"></i>
                        </div>
                    </div>
                    <div class="stat-title">{stat.title}</div>
                    <div class="stat-value">{stat.value}</div>
                    <div class="stat-change {stat.trend}">
                        <i class="fas fa-arrow-{stat.trend === 'positive' ? 'down' : 'up'}"></i> {stat.change}
                    </div>
                </div>
            {/each}
        </section>

        <section class="content-grid" transition:fade={{ duration: 300, delay: 200 }}>
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Active Security Issues</h3>
                    <div class="filter-tabs">
                        <button class="filter-tab {activeFilter === 'all' ? 'active' : ''}" on:click={() => filterVulnerabilities('all')}>
                            All <span class="tab-count">{vulnerabilities.length}</span>
                        </button>
                        <button class="filter-tab {activeFilter === 'critical' ? 'active' : ''}" on:click={() => filterVulnerabilities('critical')}>
                            Critical <span class="tab-count">{vulnerabilities.filter(v => v.severity === 'critical').length}</span>
                        </button>
                        <button class="filter-tab {activeFilter === 'high' ? 'active' : ''}" on:click={() => filterVulnerabilities('high')}>
                            High <span class="tab-count">{vulnerabilities.filter(v => v.severity === 'high').length}</span>
                        </button>
                    </div>
                </div>
                
                <div class="vulnerabilities-list">
                    {#each vulnerabilities as vuln}
                        <div class="vuln-item" class:hidden={activeFilter !== 'all' && vuln.severity !== activeFilter}>
                            <div class="vuln-severity severity-{vuln.severity}"></div>
                            <div class="vuln-details">
                                <div class="vuln-title">{vuln.title}</div>
                                <div class="vuln-description">{vuln.description}</div>
                                <div class="vuln-meta">
                                    <span><i class="fas fa-file"></i> {vuln.file}</span>
                                    <span><i class="fas fa-bug"></i> {vuln.cwe}</span>
                                    <span><i class="fas fa-clock"></i> {vuln.time}</span>
                                </div>
                            </div>
                            <div class="vuln-actions">
                                <button class="vuln-btn btn-fix" on:click={() => handleVulnAction(vuln, 'fix')}>Auto-fix</button>
                                <button class="vuln-btn btn-suppress" on:click={() => handleVulnAction(vuln, 'suppress')}>Suppress</button>
                            </div>
                        </div>
                    {/each}
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Secret Scanning Results</h3>
                    <i class="fas fa-key secret-icon"></i>
                </div>
                
                <div class="secrets-summary">
                    <div class="secrets-header">
                        <span>Secrets detectados en los últimos 30 días</span>
                        <span class="secrets-count">{secretsDetected.length} activos</span>
                    </div>
                    <div class="secrets-progress">
                        <div class="secrets-progress-fill" style="width: 70%;"></div>
                    </div>
                </div>

                <div class="secrets-list">
                    {#each secretsDetected as secret}
                        <div class="secret-item {secret.severity}">
                            <div class="secret-header">
                                <span class="secret-type">{secret.type}</span>
                                <span class="secret-severity-badge {secret.severity}">{secret.severity}</span>
                            </div>
                            <div class="secret-file">{secret.file}</div>
                            <div class="secret-time">{secret.time}</div>
                        </div>
                    {/each}
                </div>

                <div class="secrets-footer">
                    <button class="btn-secondary">
                        View All Secrets
                    </button>
                </div>
            </div>
        </section>

        <section class="compliance-section" transition:fade={{ duration: 300, delay: 250 }}>
            <h3 class="card-title">Compliance & Standards</h3>
            <p class="compliance-subtitle">Cumplimiento con estándares de seguridad</p>
            
            <div class="compliance-grid">
                {#each complianceData as compliance}
                    <div class="compliance-item {compliance.status}">
                        <div class="compliance-header">
                            <h4 class="compliance-title">{compliance.title}</h4>
                            <span class="compliance-status status-{compliance.status}">{compliance.status}</span>
                        </div>
                        <div class="compliance-progress">
                            <div class="progress-fill progress-{compliance.status}" style="width: {compliance.percentage}%;"></div>
                        </div>
                        <div class="compliance-details">
                            <span>{compliance.percentage}% compliance</span>
                            <span>{compliance.remaining} issues remaining</span>
                        </div>
                    </div>
                {/each}
            </div>
        </section>
    {/if}
</div>

<style>
    /* Variables para tema claro */
    :root {
        --color-bg: #f8fafc;
        --color-bg-light: #e2e8f0;
        --color-bg-card: #ffffff;
        --color-text: #334155;
        --color-text-light: #64748b;
        --color-border: #e2e8f0;
        --color-primary: #ef4444;
        --color-primary-light: #f87171;
        --color-success: #10b981;
        --color-success-light: #34d399;
        --color-warning: #f59e0b;
        --color-warning-light: #fbbf24;
        --color-danger: #dc2626;
        --color-danger-light: #ef4444;
        --color-critical: #7c2d12;
        --color-critical-light: #9a3412;
        --color-purple: #8b5cf6;
        --color-purple-light: #a78bfa;
    }
    
    /* Estilos base */
    .main-content {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
        color: var(--color-text);
        background-color: var(--color-bg);
        min-height: 100vh;
    }

    /* Header */
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }

    .header-left h2 {
        font-size: 2rem;
        font-weight: 600;
        background: linear-gradient(135deg, var(--color-danger), var(--color-critical));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }

    .header-subtitle {
        color: var(--color-text-light);
        font-size: 1rem;
    }

    .header-actions {
        display: flex;
        gap: 1rem;
        align-items: center;
    }

    .btn {
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }

    .btn-primary {
        background: linear-gradient(135deg, var(--color-danger), var(--color-critical));
        color: white;
    }

    .btn-secondary {
        background: var(--color-bg-light);
        color: var(--color-text);
        border: 1px solid var(--color-border);
    }

    .btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
    }

    /* Security overview section */
    .security-overview {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin-bottom: 2rem;
    }

    .security-score-card {
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: 1rem;
        padding: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .security-score-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--color-danger), var(--color-critical));
    }

    .score-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 2rem;
    }

    .score-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--color-text);
    }

    .score-subtitle {
        color: var(--color-text-light);
    }

    .security-level {
        padding: 0.5rem 1rem;
        background: rgba(239, 68, 68, 0.1);
        color: var(--color-danger);
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    .security-level.good {
        background: rgba(16, 185, 129, 0.1);
        color: var(--color-success);
        border-color: rgba(16, 185, 129, 0.2);
    }

    .security-level.fair {
        background: rgba(245, 158, 11, 0.1);
        color: var(--color-warning);
        border-color: rgba(245, 158, 11, 0.2);
    }

    .vuln-stats {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 1.5rem;
    }

    .vuln-stat {
        text-align: center;
        padding: 1rem;
        background: var(--color-bg-light);
        border-radius: 0.75rem;
    }

    .vuln-count {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .vuln-label {
        font-size: 0.75rem;
        color: var(--color-text-light);
    }

    .critical { color: var(--color-critical); }
    .high { color: var(--color-danger); }
    .medium { color: var(--color-warning); }
    .low { color: var(--color-success); }

    /* Timeline section */
    .threat-timeline {
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .timeline-header {
        margin-bottom: 1.5rem;
    }

    .timeline-title {
        font-size: 1.125rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--color-text);
    }

    .timeline-subtitle {
        color: var(--color-text-light);
    }

    .timeline-item {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 1rem 0;
        border-bottom: 1px solid var(--color-border);
    }

    .timeline-item:last-child {
        border-bottom: none;
    }

    .timeline-icon {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        color: white;
        margin-top: 0.25rem;
    }
    
    .timeline-icon.critical {
        background: linear-gradient(135deg, var(--color-critical), var(--color-critical-light));
    }
    
    .timeline-icon.high {
        background: linear-gradient(135deg, var(--color-danger), var(--color-danger-light));
    }
    
    .timeline-icon.medium {
        background: linear-gradient(135deg, var(--color-warning), var(--color-warning-light));
    }
    
    .timeline-icon.low {
        background: linear-gradient(135deg, var(--color-success), var(--color-success-light));
    }

    .timeline-content {
        flex: 1;
    }

    .timeline-event {
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
        color: var(--color-text);
    }

    .timeline-details {
        font-size: 0.75rem;
        color: var(--color-text-light);
        margin-bottom: 0.5rem;
    }

    .timeline-time {
        font-size: 0.75rem;
        color: var(--color-text-light);
    }

    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }

    .stat-card {
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: 1rem;
        padding: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .stat-card:hover::before {
        opacity: 1;
    }

    .stat-card.secrets::before { background: var(--color-critical); }
    .stat-card.dependencies::before { background: var(--color-warning); }
    .stat-card.compliance::before { background: var(--color-success); }
    .stat-card.iac::before { background: var(--color-purple); }

    .stat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .stat-icon {
        width: 48px;
        height: 48px;
        border-radius: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        color: white;
    }
    
    .secrets .stat-icon {
        background: linear-gradient(135deg, var(--color-critical), var(--color-critical-light));
    }
    
    .dependencies .stat-icon {
        background: linear-gradient(135deg, var(--color-warning), var(--color-warning-light));
    }
    
    .compliance .stat-icon {
        background: linear-gradient(135deg, var(--color-success), var(--color-success-light));
    }
    
    .iac .stat-icon {
        background: linear-gradient(135deg, var(--color-purple), var(--color-purple-light));
    }

    .stat-title {
        font-size: 0.875rem;
        color: var(--color-text-light);
        margin-bottom: 0.5rem;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--color-text);
    }

    .stat-change {
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }

    .positive { color: var(--color-success); }
    .negative { color: var(--color-danger); }

    /* Content Grid */
    .content-grid {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 2rem;
        margin-bottom: 2rem;
    }

    .card {
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }

    .card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--color-text);
    }
    
    .secret-icon {
        color: var(--color-danger);
    }

    /* Filter tabs */
    .filter-tabs {
        display: flex;
        gap: 0.5rem;
    }

    .filter-tab {
        padding: 0.5rem 1rem;
        background: var(--color-bg-light);
        border: none;
        border-radius: 0.5rem;
        color: var(--color-text-light);
        cursor: pointer;
        font-size: 0.875rem;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .filter-tab.active,
    .filter-tab:hover {
        background: rgba(239, 68, 68, 0.1);
        color: var(--color-danger);
    }

    .tab-count {
        background: rgba(239, 68, 68, 0.1);
        color: var(--color-danger);
        padding: 0.125rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
    }

    /* Vulnerabilities list */
    .vulnerabilities-list {
        max-height: 400px;
        overflow-y: auto;
    }
    
    .hidden {
        display: none;
    }

    .vuln-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        border: 1px solid var(--color-border);
        border-radius: 0.75rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
        background: var(--color-bg-card);
    }

    .vuln-item:hover {
        background: var(--color-bg-light);
    }

    .vuln-severity {
        width: 8px;
        height: 50px;
        border-radius: 4px;
    }

    .severity-critical { background: linear-gradient(to bottom, var(--color-critical), var(--color-critical-light)); }
    .severity-high { background: linear-gradient(to bottom, var(--color-danger), var(--color-danger-light)); }
    .severity-medium { background: linear-gradient(to bottom, var(--color-warning), var(--color-warning-light)); }
    .severity-low { background: linear-gradient(to bottom, var(--color-success), var(--color-success-light)); }

    .vuln-details {
        flex: 1;
    }

    .vuln-title {
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
        color: var(--color-text);
    }

    .vuln-description {
        font-size: 0.75rem;
        color: var(--color-text-light);
        margin-bottom: 0.5rem;
    }

    .vuln-meta {
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: var(--color-text-light);
    }

    .vuln-actions {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .vuln-btn {
        padding: 0.25rem 0.75rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 500;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .btn-fix {
        background: rgba(16, 185, 129, 0.1);
        color: var(--color-success);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .btn-fix:hover {
        background: rgba(16, 185, 129, 0.2);
    }

    .btn-suppress {
        background: rgba(100, 116, 139, 0.1);
        color: var(--color-text-light);
        border: 1px solid rgba(100, 116, 139, 0.2);
    }
    
    .btn-suppress:hover {
        background: rgba(100, 116, 139, 0.2);
    }

    /* Secrets section */
    .secrets-summary {
        margin-bottom: 1.5rem;
    }

    .secrets-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
        color: var(--color-text);
    }

    .secrets-count {
        font-weight: 600;
        color: var(--color-danger);
    }

    .secrets-progress {
        width: 100%;
        height: 8px;
        background: rgba(148, 163, 184, 0.2);
        border-radius: 4px;
        overflow: hidden;
    }

    .secrets-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--color-danger), var(--color-danger-light));
    }

    .secrets-list {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }

    .secret-item {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid var(--color-border);
    }

    .secret-item.critical {
        background: rgba(124, 45, 18, 0.05);
        border-color: rgba(124, 45, 18, 0.1);
    }

    .secret-item.high {
        background: rgba(239, 68, 68, 0.05);
        border-color: rgba(239, 68, 68, 0.1);
    }

    .secret-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .secret-type {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--color-text);
    }

    .secret-severity-badge {
        padding: 0.125rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
    }

    .secret-severity-badge.critical {
        background: rgba(124, 45, 18, 0.1);
        color: var(--color-critical);
    }

    .secret-severity-badge.high {
        background: rgba(239, 68, 68, 0.1);
        color: var(--color-danger);
    }

    .secret-file {
        font-size: 0.75rem;
        color: var(--color-text-light);
        font-family: monospace;
        margin-bottom: 0.25rem;
    }

    .secret-time {
        font-size: 0.75rem;
        color: var(--color-text-light);
    }

    .secrets-footer {
        text-align: center;
        margin-top: 1.5rem;
    }

    /* Compliance section */
    .compliance-section {
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .compliance-subtitle {
        color: var(--color-text-light);
        margin-bottom: 1rem;
    }

    .compliance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 1.5rem;
    }

    .compliance-item {
        padding: 1.5rem;
        background: var(--color-bg-light);
        border-radius: 0.75rem;
        position: relative;
    }

    .compliance-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: 0.75rem 0.75rem 0 0;
    }

    .compliance-item.pass::before { background: var(--color-success); }
    .compliance-item.warning::before { background: var(--color-warning); }
    .compliance-item.fail::before { background: var(--color-danger); }

    .compliance-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .compliance-title {
        font-size: 1rem;
        font-weight: 500;
        color: var(--color-text);
    }

    .compliance-status {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: capitalize;
    }

    .status-pass {
        background: rgba(16, 185, 129, 0.1);
        color: var(--color-success);
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--color-warning);
    }

    .status-fail {
        background: rgba(239, 68, 68, 0.1);
        color: var(--color-danger);
    }

    .compliance-progress {
        width: 100%;
        height: 6px;
        background: rgba(148, 163, 184, 0.2);
        border-radius: 3px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .progress-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }

    .progress-pass { background: linear-gradient(90deg, var(--color-success), var(--color-success-light)); }
    .progress-warning { background: linear-gradient(90deg, var(--color-warning), var(--color-warning-light)); }
    .progress-fail { background: linear-gradient(90deg, var(--color-danger), var(--color-danger-light)); }

    .compliance-details {
        font-size: 0.875rem;
        color: var(--color-text);
        display: flex;
        justify-content: space-between;
    }

    /* Loading y Error */
    .loading,
    .error {
        padding: 2rem;
        text-align: center;
        background: var(--color-bg-card);
        border-radius: 1rem;
        margin: 2rem 0;
        border: 1px solid var(--color-border);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .error {
        color: var(--color-danger);
        border-color: var(--color-danger-light);
    }
    
    /* Responsive */
    @media (max-width: 1024px) {
        .security-overview,
        .content-grid {
            grid-template-columns: 1fr;
        }
        
        .vuln-stats {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .compliance-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 768px) {
        .header {
            flex-direction: column;
            align-items: flex-start;
            gap: 1rem;
        }

        .header-actions {
            width: 100%;
            justify-content: space-between;
        }
        
        .vuln-stats {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .compliance-grid {
            grid-template-columns: 1fr;
        }
        
        .filter-tabs {
            margin-top: 1rem;
            width: 100%;
            justify-content: space-between;
        }
    }

    @media (max-width: 480px) {
        .vuln-stats {
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }
        
        .vuln-item {
            flex-direction: column;
            align-items: flex-start;
        }
        
        .vuln-severity {
            width: 100%;
            height: 4px;
            margin-bottom: 0.5rem;
        }
        
        .vuln-actions {
            width: 100%;
            flex-direction: row;
            margin-top: 0.5rem;
        }
    }
</style>
