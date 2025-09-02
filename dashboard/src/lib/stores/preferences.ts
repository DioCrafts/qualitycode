import { browser } from '$app/environment';
import { writable, type Writable } from 'svelte/store';

interface UserPreferences {
    theme: 'light' | 'dark' | 'system';
    language: 'es' | 'en';
    accessibility: {
        highContrast: boolean;
        reducedMotion: boolean;
        fontSize: 'normal' | 'large' | 'x-large';
    };
    notifications: {
        desktop: boolean;
        email: boolean;
        severity: string[];
    };
    dashboard: {
        autoRefresh: boolean;
        refreshInterval: number; // en segundos
        compactView: boolean;
    };
}

const defaultPreferences: UserPreferences = {
    theme: 'system',
    language: 'es',
    accessibility: {
        highContrast: false,
        reducedMotion: false,
        fontSize: 'normal'
    },
    notifications: {
        desktop: true,
        email: true,
        severity: ['critical', 'high']
    },
    dashboard: {
        autoRefresh: true,
        refreshInterval: 30,
        compactView: false
    }
};

// Función genérica para crear stores persistidos
function createPersistedStore<T>(key: string, initialValue: T): Writable<T> {
    // Si no estamos en el navegador, devolver un store normal
    if (!browser) {
        return writable(initialValue);
    }

    // Intentar cargar desde localStorage
    let storedValue: T = initialValue;
    try {
        const item = localStorage.getItem(key);
        if (item) {
            storedValue = JSON.parse(item);
        }
    } catch (error) {
        console.error(`Error loading persisted value for ${key}:`, error);
    }

    // Crear el store con el valor inicial o el valor almacenado
    const store = writable(storedValue);

    // Suscribirse a cambios para persistir
    store.subscribe(value => {
        if (browser) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
            } catch (error) {
                console.error(`Error persisting value for ${key}:`, error);
            }
        }
    });

    return store;
}

// Store de preferencias de usuario
export const userPreferences = createPersistedStore('user-preferences', defaultPreferences);

// Stores derivados para acceso rápido
export function createThemeStore() {
    const { subscribe } = userPreferences;

    return {
        subscribe: (fn: (value: UserPreferences['theme']) => void) => {
            return subscribe(prefs => fn(prefs.theme));
        },

        setTheme(theme: UserPreferences['theme']) {
            userPreferences.update(prefs => ({ ...prefs, theme }));
        },

        applyTheme() {
            if (!browser) return;

            userPreferences.subscribe(prefs => {
                const { theme } = prefs;
                const root = document.documentElement;

                // Remover clases existentes
                root.classList.remove('light', 'dark');

                if (theme === 'system') {
                    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                    root.classList.add(prefersDark ? 'dark' : 'light');
                } else {
                    root.classList.add(theme);
                }
            });
        }
    };
}

export function createAccessibilityStore() {
    const { subscribe } = userPreferences;

    return {
        subscribe: (fn: (value: UserPreferences['accessibility']) => void) => {
            return subscribe(prefs => fn(prefs.accessibility));
        },

        toggleHighContrast() {
            userPreferences.update(prefs => ({
                ...prefs,
                accessibility: {
                    ...prefs.accessibility,
                    highContrast: !prefs.accessibility.highContrast
                }
            }));
        },

        toggleReducedMotion() {
            userPreferences.update(prefs => ({
                ...prefs,
                accessibility: {
                    ...prefs.accessibility,
                    reducedMotion: !prefs.accessibility.reducedMotion
                }
            }));
        },

        setFontSize(fontSize: UserPreferences['accessibility']['fontSize']) {
            userPreferences.update(prefs => ({
                ...prefs,
                accessibility: {
                    ...prefs.accessibility,
                    fontSize
                }
            }));
        },

        applyAccessibilitySettings() {
            if (!browser) return;

            userPreferences.subscribe(prefs => {
                const { accessibility } = prefs;
                const root = document.documentElement;

                // High contrast
                root.classList.toggle('high-contrast', accessibility.highContrast);

                // Reduced motion
                root.classList.toggle('reduced-motion', accessibility.reducedMotion);

                // Font size
                root.setAttribute('data-font-size', accessibility.fontSize);
            });
        }
    };
}

// Función para inicializar preferencias al cargar la aplicación
export function initializePreferences() {
    if (!browser) return;

    const themeStore = createThemeStore();
    const accessibilityStore = createAccessibilityStore();

    themeStore.applyTheme();
    accessibilityStore.applyAccessibilitySettings();

    // Escuchar cambios del sistema
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        themeStore.applyTheme();
    });
}

// Store para caché de datos del dashboard
export const dashboardCache = createPersistedStore('dashboard-cache', {
    lastUpdate: null as string | null,
    cachedData: null as any
});

// Store para layouts guardados
export const savedLayouts = createPersistedStore('saved-layouts', [] as Array<{
    id: string;
    name: string;
    layout: any;
    createdAt: string;
}>);

// Store para filtros favoritos
export const favoriteFilters = createPersistedStore('favorite-filters', [] as Array<{
    id: string;
    name: string;
    filters: any;
    createdAt: string;
}>);
