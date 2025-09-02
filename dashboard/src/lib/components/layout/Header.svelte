<script lang="ts">
    import { goto } from "$app/navigation";
    import { authStore } from "$lib/stores";
    import type { User } from "$lib/types";
    import { createEventDispatcher } from "svelte";
    import NotificationBell from "../ui/NotificationBell.svelte";
    import SearchBar from "../ui/SearchBar.svelte";
    import UserMenu from "../ui/UserMenu.svelte";

    export let toggleSidebar: () => void;
    export let sidebarOpen = true;
    export let user: User | null;
    export let notificationCount = 0;

    const dispatch = createEventDispatcher();

    let searchQuery = "";

    async function handleSearch() {
        if (searchQuery.trim()) {
            await goto(`/search?q=${encodeURIComponent(searchQuery)}`);
        }
    }

    function handleLogout() {
        authStore.logout();
        goto("/login");
    }
</script>

<header class="app-header">
    <div class="header-left">
        <button
            class="menu-toggle"
            on:click={toggleSidebar}
            aria-label="Toggle sidebar"
        >
            <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
            >
                {#if sidebarOpen}
                    <line x1="3" y1="12" x2="21" y2="12"></line>
                    <line x1="3" y1="6" x2="21" y2="6"></line>
                    <line x1="3" y1="18" x2="21" y2="18"></line>
                {:else}
                    <line x1="3" y1="12" x2="21" y2="12"></line>
                    <line x1="3" y1="6" x2="21" y2="6"></line>
                    <line x1="3" y1="18" x2="21" y2="18"></line>
                {/if}
            </svg>
        </button>

        <div class="search-container">
            <SearchBar
                bind:value={searchQuery}
                on:search={handleSearch}
                placeholder="Buscar código, issues, proyectos..."
            />
        </div>
    </div>

    <div class="header-right">
        <NotificationBell count={notificationCount} />

        {#if user}
            <UserMenu {user} on:logout={handleLogout} />
        {:else}
            <a href="/login" class="login-link">Iniciar sesión</a>
        {/if}
    </div>
</header>

<style>
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: var(--header-height, 64px);
        padding: 0 1rem;
        background-color: var(--color-bg-primary, #ffffff);
        border-bottom: 1px solid var(--color-border, #e5e5e5);
        gap: 1rem;
    }

    .header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
        flex: 1;
        max-width: 600px;
    }

    .menu-toggle {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border: none;
        background: none;
        color: var(--color-text-secondary, #666);
        cursor: pointer;
        border-radius: 8px;
        transition: all 0.2s ease;
    }

    .menu-toggle:hover {
        background-color: var(--color-bg-hover, #f0f0f0);
        color: var(--color-text-primary, #333);
    }

    .menu-toggle svg {
        width: 20px;
        height: 20px;
    }

    .search-container {
        flex: 1;
        max-width: 400px;
    }

    .header-right {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .login-link {
        padding: 0.5rem 1rem;
        background-color: var(--color-primary, #3b82f6);
        color: white;
        text-decoration: none;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }

    .login-link:hover {
        background-color: var(--color-primary-hover, #2563eb);
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .search-container {
            display: none;
        }

        .app-header {
            padding: 0 0.5rem;
        }
    }

    /* Dark mode */
    :global(.dark) .app-header {
        background-color: var(--color-bg-primary-dark, #0f0f0f);
        border-bottom-color: var(--color-border-dark, #333);
    }

    :global(.dark) .menu-toggle {
        color: var(--color-text-secondary-dark, #aaa);
    }

    :global(.dark) .menu-toggle:hover {
        background-color: var(--color-bg-hover-dark, #2a2a2a);
        color: var(--color-text-primary-dark, #fff);
    }

    /* High contrast */
    :global(.high-contrast) .app-header {
        border-bottom-width: 2px;
        border-bottom-color: #000;
    }

    /* Reduced motion */
    :global(.reduced-motion) .menu-toggle,
    :global(.reduced-motion) .login-link {
        transition: none !important;
    }
</style>
