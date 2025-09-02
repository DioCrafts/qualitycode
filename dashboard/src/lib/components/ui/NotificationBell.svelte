<script lang="ts">
    import { realTimeStore, unreadNotifications } from "$lib/stores";
    import { createEventDispatcher } from "svelte";
    import { fade, fly } from "svelte/transition";

    export let count = 0;

    const dispatch = createEventDispatcher();

    let showDropdown = false;

    function toggleDropdown() {
        showDropdown = !showDropdown;
        if (showDropdown) {
            dispatch("open");
        }
    }

    function handleClickOutside(event: MouseEvent) {
        if (!(event.target as HTMLElement).closest(".notification-container")) {
            showDropdown = false;
        }
    }

    function markAsRead(id: string) {
        realTimeStore.markNotificationRead(id);
    }

    function clearAll() {
        realTimeStore.clearNotifications();
        showDropdown = false;
    }
</script>

<svelte:window on:click={handleClickOutside} />

<div class="notification-container">
    <button
        class="notification-bell"
        on:click|stopPropagation={toggleDropdown}
        aria-label="Notifications"
        aria-expanded={showDropdown}
    >
        <svg
            width="20"
            height="20"
            viewBox="0 0 20 20"
            fill="none"
            stroke="currentColor"
        >
            <path d="M15 7a5 5 0 1 0-10 0c0 3.5-1 5.5-1 5.5h12s-1-2-1-5.5z" />
            <path d="M11.73 16a2 2 0 0 1-3.46 0" />
        </svg>

        {#if count > 0}
            <span
                class="notification-badge"
                transition:fade={{ duration: 200 }}
            >
                {count > 99 ? "99+" : count}
            </span>
        {/if}
    </button>

    {#if showDropdown}
        <div
            class="notification-dropdown"
            transition:fly={{ y: -10, duration: 200 }}
            on:click|stopPropagation
        >
            <div class="dropdown-header">
                <h3>Notificaciones</h3>
                {#if $unreadNotifications.length > 0}
                    <button class="clear-all" on:click={clearAll}>
                        Marcar todas como leídas
                    </button>
                {/if}
            </div>

            <div class="notification-list">
                {#if $unreadNotifications.length === 0}
                    <div class="empty-state">
                        <p>No tienes notificaciones nuevas</p>
                    </div>
                {:else}
                    {#each $unreadNotifications as notification (notification.id)}
                        <div
                            class="notification-item"
                            class:unread={!notification.read}
                            on:click={() => markAsRead(notification.id)}
                        >
                            <div class="notification-icon {notification.type}">
                                {#if notification.type === "error"}
                                    ⚠️
                                {:else if notification.type === "success"}
                                    ✅
                                {:else if notification.type === "warning"}
                                    ⚡
                                {:else}
                                    ℹ️
                                {/if}
                            </div>

                            <div class="notification-content">
                                <h4>{notification.title}</h4>
                                <p>{notification.message}</p>
                                <time
                                    >{new Date(
                                        notification.timestamp,
                                    ).toLocaleTimeString()}</time
                                >
                            </div>
                        </div>
                    {/each}
                {/if}
            </div>
        </div>
    {/if}
</div>

<style>
    .notification-container {
        position: relative;
    }

    .notification-bell {
        position: relative;
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

    .notification-bell:hover {
        background-color: var(--color-bg-hover, #f0f0f0);
        color: var(--color-text-primary, #333);
    }

    .notification-badge {
        position: absolute;
        top: 6px;
        right: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 18px;
        height: 18px;
        padding: 0 4px;
        background-color: var(--color-danger, #ef4444);
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9px;
    }

    .notification-dropdown {
        position: absolute;
        top: calc(100% + 8px);
        right: 0;
        width: 320px;
        max-height: 400px;
        background-color: var(--color-bg-primary, #ffffff);
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        z-index: 1000;
    }

    .dropdown-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem;
        border-bottom: 1px solid var(--color-border, #e5e5e5);
    }

    .dropdown-header h3 {
        margin: 0;
        font-size: 0.875rem;
        font-weight: 600;
    }

    .clear-all {
        padding: 0.25rem 0.5rem;
        border: none;
        background: none;
        color: var(--color-primary, #3b82f6);
        font-size: 0.75rem;
        cursor: pointer;
        transition: opacity 0.2s ease;
    }

    .clear-all:hover {
        opacity: 0.8;
    }

    .notification-list {
        max-height: 300px;
        overflow-y: auto;
    }

    .empty-state {
        padding: 3rem 1rem;
        text-align: center;
        color: var(--color-text-muted, #999);
    }

    .empty-state p {
        margin: 0;
        font-size: 0.875rem;
    }

    .notification-item {
        display: flex;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--color-border, #e5e5e5);
        cursor: pointer;
        transition: background-color 0.2s ease;
    }

    .notification-item:hover {
        background-color: var(--color-bg-hover, #f0f0f0);
    }

    .notification-item:last-child {
        border-bottom: none;
    }

    .notification-item.unread {
        background-color: var(--color-primary-light, #eff6ff);
    }

    .notification-icon {
        flex-shrink: 0;
        font-size: 1.25rem;
    }

    .notification-icon.error {
        color: var(--color-danger);
    }

    .notification-icon.success {
        color: var(--color-success);
    }

    .notification-icon.warning {
        color: var(--color-warning);
    }

    .notification-icon.info {
        color: var(--color-info);
    }

    .notification-content h4 {
        margin: 0 0 0.25rem;
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--color-text-primary, #333);
    }

    .notification-content p {
        margin: 0 0 0.25rem;
        font-size: 0.75rem;
        color: var(--color-text-secondary, #666);
    }

    .notification-content time {
        font-size: 0.6875rem;
        color: var(--color-text-muted, #999);
    }

    /* Dark mode */
    :global(.dark) .notification-bell {
        color: var(--color-text-secondary-dark, #aaa);
    }

    :global(.dark) .notification-bell:hover {
        background-color: var(--color-bg-hover-dark, #2a2a2a);
        color: var(--color-text-primary-dark, #fff);
    }

    :global(.dark) .notification-dropdown {
        background-color: var(--color-bg-primary-dark, #0f0f0f);
        border-color: var(--color-border-dark, #333);
    }

    :global(.dark) .dropdown-header {
        border-bottom-color: var(--color-border-dark, #333);
    }

    :global(.dark) .notification-item {
        border-bottom-color: var(--color-border-dark, #333);
    }

    :global(.dark) .notification-item:hover {
        background-color: var(--color-bg-hover-dark, #2a2a2a);
    }

    :global(.dark) .notification-item.unread {
        background-color: var(--color-primary-dark, #1e40af);
    }

    /* High contrast */
    :global(.high-contrast) .notification-dropdown {
        border-width: 2px;
        border-color: #000;
    }

    /* Reduced motion */
    :global(.reduced-motion) .notification-bell,
    :global(.reduced-motion) .notification-item,
    :global(.reduced-motion) .clear-all {
        transition: none !important;
    }
</style>
