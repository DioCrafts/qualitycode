<script lang="ts">
    let loading = false;
</script>

<div class="container">
    <h1>Análisis de Seguridad</h1>

    <div class="content">
        <p>
            Esta sección mostrará análisis de seguridad detallados de tus
            proyectos.
        </p>

        <div class="info-card">
            <h3>Características próximas:</h3>
            <ul>
                <li>Detección de vulnerabilidades conocidas (CVE)</li>
                <li>Análisis de dependencias inseguras</li>
                <li>Detección de secretos y credenciales expuestas</li>
                <li>Análisis de permisos y configuraciones</li>
                <li>Recomendaciones OWASP</li>
            </ul>
        </div>

        <div class="security-tips">
            <h3>Tips de seguridad:</h3>
            <div class="tip">
                <strong>🔒 Secretos:</strong> Nunca hardcodees credenciales en el
                código
            </div>
            <div class="tip">
                <strong>🛡️ Dependencias:</strong> Mantén tus dependencias actualizadas
            </div>
            <div class="tip">
                <strong>🔍 Validación:</strong> Siempre valida y sanitiza las entradas
                de usuario
            </div>
        </div>

        <div class="action">
            <a href="/projects" class="btn-primary">Analizar proyectos</a>
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

    .info-card,
    .security-tips {
        background: var(--color-bg-primary, #fff);
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 2rem 0;
    }

    .info-card h3,
    .security-tips h3 {
        margin-bottom: 1rem;
    }

    .info-card ul {
        list-style-position: inside;
        color: #666;
    }

    .tip {
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: var(--color-bg-secondary, #f5f5f5);
        border-radius: 4px;
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
