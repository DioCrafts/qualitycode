<script lang="ts">
    let loading = false;
</script>

<div class="container">
    <h1>Métricas de Código</h1>

    <div class="content">
        <p>Dashboard de métricas y estadísticas de calidad de código.</p>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-icon">📊</div>
                <h3>Complejidad</h3>
                <p>Análisis de complejidad ciclomática y cognitiva</p>
            </div>

            <div class="metric-card">
                <div class="metric-icon">📈</div>
                <h3>Tendencias</h3>
                <p>Evolución de la calidad del código en el tiempo</p>
            </div>

            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <h3>Cobertura</h3>
                <p>Cobertura de tests y documentación</p>
            </div>

            <div class="metric-card">
                <div class="metric-icon">⚡</div>
                <h3>Performance</h3>
                <p>Métricas de rendimiento y optimización</p>
            </div>
        </div>

        <div class="info-card">
            <h3>Próximamente:</h3>
            <ul>
                <li>Dashboard interactivo con gráficos</li>
                <li>Métricas DORA (DevOps Research and Assessment)</li>
                <li>Índice de mantenibilidad por proyecto</li>
                <li>Benchmarks contra estándares de la industria</li>
                <li>Exportación de reportes</li>
            </ul>
        </div>

        <div class="action">
            <a href="/projects" class="btn-primary">Ver proyectos</a>
        </div>
    </div>
</div>

<style>
    .container {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    .content {
        margin-top: 2rem;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .metric-card {
        background: var(--color-bg-primary, #fff);
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }

    .metric-card h3 {
        margin-bottom: 0.5rem;
    }

    .metric-card p {
        color: #666;
        font-size: 0.9rem;
    }

    .info-card {
        background: var(--color-bg-primary, #fff);
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 2rem 0;
    }

    .info-card h3 {
        margin-bottom: 1rem;
    }

    .info-card ul {
        list-style-position: inside;
        color: #666;
    }

    .action {
        margin-top: 2rem;
    }

    .btn-primary {
        background-color: #4a6cf7;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 4px;
        text-decoration: none;
        display: inline-block;
    }

    .btn-primary:hover {
        background-color: #3955d8;
    }
</style>
