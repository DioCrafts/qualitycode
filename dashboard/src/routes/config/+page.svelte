<script lang="ts">
    let loading = false;
    let settings = {
        theme: "light",
        notifications: true,
        autoAnalysis: false,
        analysisFrequency: "daily",
    };
</script>

<div class="container">
    <h1>Configuración</h1>

    <div class="content">
        <div class="settings-section">
            <h2>Preferencias generales</h2>

            <div class="setting-item">
                <label>
                    <span>Tema de la aplicación</span>
                    <select bind:value={settings.theme}>
                        <option value="light">Claro</option>
                        <option value="dark">Oscuro</option>
                        <option value="system">Sistema</option>
                    </select>
                </label>
            </div>

            <div class="setting-item">
                <label>
                    <input
                        type="checkbox"
                        bind:checked={settings.notifications}
                    />
                    <span>Activar notificaciones</span>
                </label>
            </div>
        </div>

        <div class="settings-section">
            <h2>Análisis automático</h2>

            <div class="setting-item">
                <label>
                    <input
                        type="checkbox"
                        bind:checked={settings.autoAnalysis}
                    />
                    <span>Ejecutar análisis automáticamente</span>
                </label>
            </div>

            <div class="setting-item">
                <label>
                    <span>Frecuencia de análisis</span>
                    <select
                        bind:value={settings.analysisFrequency}
                        disabled={!settings.autoAnalysis}
                    >
                        <option value="hourly">Cada hora</option>
                        <option value="daily">Diario</option>
                        <option value="weekly">Semanal</option>
                        <option value="monthly">Mensual</option>
                    </select>
                </label>
            </div>
        </div>

        <div class="info-card">
            <h3>Próximamente:</h3>
            <ul>
                <li>Configuración de reglas de análisis personalizadas</li>
                <li>Integración con CI/CD</li>
                <li>Webhooks y notificaciones</li>
                <li>Gestión de equipos y permisos</li>
                <li>Exportación e importación de configuraciones</li>
            </ul>
        </div>

        <div class="actions">
            <button class="btn-primary">Guardar cambios</button>
            <button class="btn-secondary">Cancelar</button>
        </div>
    </div>
</div>

<style>
    .container {
        padding: 2rem;
        max-width: 800px;
        margin: 0 auto;
    }

    .content {
        margin-top: 2rem;
    }

    .settings-section {
        background: var(--color-bg-primary, #fff);
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }

    .settings-section h2 {
        margin-bottom: 1.5rem;
        font-size: 1.2rem;
    }

    .setting-item {
        margin-bottom: 1.5rem;
    }

    .setting-item:last-child {
        margin-bottom: 0;
    }

    .setting-item label {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .setting-item input[type="checkbox"] {
        width: 20px;
        height: 20px;
    }

    .setting-item select {
        padding: 0.5rem;
        border: 1px solid var(--color-border, #ddd);
        border-radius: 4px;
        background: white;
    }

    .setting-item select:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .info-card {
        background: var(--color-bg-secondary, #f5f5f5);
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

    .actions {
        margin-top: 2rem;
        display: flex;
        gap: 1rem;
    }

    .btn-primary,
    .btn-secondary {
        padding: 0.75rem 1.5rem;
        border-radius: 4px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
    }

    .btn-primary {
        background-color: #4a6cf7;
        color: white;
        border: none;
    }

    .btn-primary:hover {
        background-color: #3955d8;
    }

    .btn-secondary {
        background-color: white;
        color: #4a6cf7;
        border: 1px solid #4a6cf7;
    }

    .btn-secondary:hover {
        background-color: #f5f7ff;
    }
</style>
