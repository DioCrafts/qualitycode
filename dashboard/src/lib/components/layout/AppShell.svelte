<script lang="ts">
    import { page } from "$app/stores";
    import { currentUser } from "$lib/stores";
    import { notificationCount } from "$lib/stores/realtime";
    import { fade, fly } from "svelte/transition";
    import Header from "./Header.svelte";
    import Sidebar from "./Sidebar.svelte";

    export let sidebar = true;

    let sidebarOpen = true;
    let mobileSidebarOpen = false;

    // Responsive behavior
    function handleResize() {
        if (window.innerWidth < 768) {
            sidebarOpen = false;
        } else {
            sidebarOpen = true;
            mobileSidebarOpen = false;
        }
    }

    // Toggle sidebar
    function toggleSidebar() {
        if (window.innerWidth < 768) {
            mobileSidebarOpen = !mobileSidebarOpen;
        } else {
            sidebarOpen = !sidebarOpen;
        }
    }

    // Close mobile sidebar on route change
    $: if ($page) {
        mobileSidebarOpen = false;
    }
</script>

<svelte:window on:resize={handleResize} />

<div class="app-shell" class:sidebar-collapsed={!sidebarOpen}>
    {#if sidebar}
        <!-- Desktop Sidebar -->
        <aside
            class="sidebar desktop-sidebar"
            class:collapsed={!sidebarOpen}
            transition:fly={{ x: -250, duration: 200 }}
        >
            <Sidebar {sidebarOpen} />
        </aside>

        <!-- Mobile Sidebar with Overlay -->
        {#if mobileSidebarOpen}
            <div
                class="mobile-overlay"
                on:click={() => (mobileSidebarOpen = false)}
                on:keydown={(e) => e.key === 'Escape' && (mobileSidebarOpen = false)}
                role="button"
                tabindex="0"
                aria-label="Cerrar menú lateral"
                transition:fade={{ duration: 200 }}
            ></div>
            <aside
                class="sidebar mobile-sidebar"
                transition:fly={{ x: -250, duration: 200 }}
            >
                <Sidebar sidebarOpen={true} mobile={true} />
            </aside>
        {/if}
    {/if}

    <div class="main-container">
        <Header
            {toggleSidebar}
            {sidebarOpen}
            user={$currentUser}
            notificationCount={$notificationCount}
        />

        <main class="main-content">
            <slot />
        </main>
    </div>
</div>

<style>
    .app-shell {
        display: flex;
        height: 100vh;
        overflow: hidden;
        background-color: var(--color-bg-secondary, #f5f5f5);
    }

    .sidebar {
        width: var(--sidebar-width, 250px);
        background-color: var(--color-bg-primary, #ffffff);
        border-right: 1px solid var(--color-border, #e5e5e5);
        transition:
            width 0.2s ease,
            transform 0.2s ease;
        z-index: 100;
    }

    .desktop-sidebar {
        position: relative;
        flex-shrink: 0;
    }

    .desktop-sidebar.collapsed {
        width: 64px;
    }

    .mobile-sidebar {
        position: fixed;
        top: 0;
        left: 0;
        height: 100vh;
        z-index: 1000;
    }

    .mobile-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: 999;
    }

    .main-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    .main-content {
        flex: 1;
        overflow-y: auto;
        padding: var(--content-padding, 2rem);
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .desktop-sidebar {
            display: none;
        }

        .main-content {
            padding: 1rem;
        }
    }

    /* Dark mode */
    :global(.dark) .app-shell {
        background-color: var(--color-bg-secondary-dark, #1a1a1a);
    }

    :global(.dark) .sidebar {
        background-color: var(--color-bg-primary-dark, #0f0f0f);
        border-right-color: var(--color-border-dark, #333);
    }

    /* High contrast mode */
    :global(.high-contrast) .sidebar {
        border-right-width: 2px;
        border-right-color: #000;
    }

    /* Reduced motion */
    :global(.reduced-motion) .sidebar,
    :global(.reduced-motion) .mobile-overlay {
        transition: none !important;
    }
</style>
