<script lang="ts">
    import { goto } from "$app/navigation";
    import type { User } from "$lib/types";
    import { createEventDispatcher } from "svelte";
    import { fly } from "svelte/transition";

    export let user: User;

    const dispatch = createEventDispatcher();

    let showDropdown = false;

    function toggleDropdown() {
        showDropdown = !showDropdown;
    }

    function handleClickOutside(event: MouseEvent) {
        if (!(event.target as HTMLElement).closest(".user-menu-container")) {
            showDropdown = false;
        }
    }

    function handleLogout() {
        dispatch("logout");
        showDropdown = false;
    }

    function navigateTo(path: string) {
        goto(path);
        showDropdown = false;
    }

    // Get user initials
    $: initials = user.name
        .split(" ")
        .map((n) => n[0])
        .join("")
        .toUpperCase()
        .slice(0, 2);

    // Role display names
    const roleNames = {
        developer: "Desarrollador",
        tech_lead: "Tech Lead",
        manager: "Manager",
        security: "Seguridad",
        qa: "QA",
    };
</script>

<svelte:window on:click={handleClickOutside} />

<div class="user-menu-container">
    <button
        class="user-menu-trigger"
        on:click|stopPropagation={toggleDropdown}
        aria-label="User menu"
        aria-expanded={showDropdown}
    >
        <div class="user-avatar">
            {initials}
        </div>
        <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="currentColor"
            class="chevron"
            class:open={showDropdown}
        >
            <path
                d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"
            />
        </svg>
    </button>

    {#if showDropdown}
        <div
            class="user-dropdown"
            transition:fly={{ y: -10, duration: 200 }}
            on:click|stopPropagation
        >
            <div class="user-info">
                <div class="user-avatar large">
                    {initials}
                </div>
                <div class="user-details">
                    <h4>{user.name}</h4>
                    <p>{user.email}</p>
                    <span class="user-role">{roleNames[user.role]}</span>
                </div>
            </div>

            <div class="dropdown-divider"></div>

            <nav class="dropdown-menu">
                <button
                    class="menu-item"
                    on:click={() => navigateTo("/profile")}
                >
                    <svg
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="currentColor"
                    >
                        <path
                            d="M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM12.735 14c.618 0 1.093-.561.872-1.139a6.002 6.002 0 0 0-11.215 0c-.22.578.254 1.139.872 1.139h9.47z"
                        />
                    </svg>
                    Mi Perfil
                </button>

                <button
                    class="menu-item"
                    on:click={() => navigateTo("/settings")}
                >
                    <svg
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="currentColor"
                    >
                        <path
                            d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492zM5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0z"
                        />
                        <path
                            d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52l-.094-.319zm-2.633.283c.246-.835 1.428-.835 1.674 0l.094.319a1.873 1.873 0 0 0 2.693 1.115l.291-.16c.764-.415 1.6.42 1.184 1.185l-.159.292a1.873 1.873 0 0 0 1.116 2.692l.318.094c.835.246.835 1.428 0 1.674l-.319.094a1.873 1.873 0 0 0-1.115 2.693l.16.291c.415.764-.42 1.6-1.185 1.184l-.291-.159a1.873 1.873 0 0 0-2.693 1.116l-.094.318c-.246.835-1.428.835-1.674 0l-.094-.319a1.873 1.873 0 0 0-2.692-1.115l-.292.16c-.764.415-1.6-.42-1.184-1.185l.159-.291A1.873 1.873 0 0 0 1.945 8.93l-.319-.094c-.835-.246-.835-1.428 0-1.674l.319-.094A1.873 1.873 0 0 0 3.06 4.377l-.16-.292c-.415-.764.42-1.6 1.185-1.184l.292.159a1.873 1.873 0 0 0 2.692-1.115l.094-.319z"
                        />
                    </svg>
                    Configuración
                </button>

                <button
                    class="menu-item"
                    on:click={() => navigateTo("/organization")}
                >
                    <svg
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="currentColor"
                    >
                        <path
                            d="M6.5 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM14 7.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0zm-8.5 5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3z"
                        />
                        <path
                            d="M10.823.473a.5.5 0 0 1 .354 0l2.5 1A.5.5 0 0 1 14 2v12a.5.5 0 0 1-.323.473l-2.5 1a.5.5 0 0 1-.354 0L8.5 14.28l-2.323 1.194a.5.5 0 0 1-.354 0l-2.5-1A.5.5 0 0 1 3 14V2a.5.5 0 0 1 .323-.473l2.5-1a.5.5 0 0 1 .354 0L8.5 1.72l2.323-1.194z"
                        />
                    </svg>
                    {user.organization.name}
                </button>
            </nav>

            <div class="dropdown-divider"></div>

            <button class="menu-item logout" on:click={handleLogout}>
                <svg
                    width="16"
                    height="16"
                    viewBox="0 0 16 16"
                    fill="currentColor"
                >
                    <path
                        fill-rule="evenodd"
                        d="M10 12.5a.5.5 0 0 1-.5.5h-8a.5.5 0 0 1-.5-.5v-9a.5.5 0 0 1 .5-.5h8a.5.5 0 0 1 .5.5v2a.5.5 0 0 0 1 0v-2A1.5 1.5 0 0 0 9.5 2h-8A1.5 1.5 0 0 0 0 3.5v9A1.5 1.5 0 0 0 1.5 14h8a1.5 1.5 0 0 0 1.5-1.5v-2a.5.5 0 0 0-1 0v2z"
                    />
                    <path
                        fill-rule="evenodd"
                        d="M15.854 8.354a.5.5 0 0 0 0-.708l-3-3a.5.5 0 0 0-.708.708L14.293 7.5H5.5a.5.5 0 0 0 0 1h8.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3z"
                    />
                </svg>
                Cerrar sesión
            </button>
        </div>
    {/if}
</div>

<style>
    .user-menu-container {
        position: relative;
    }

    .user-menu-trigger {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem;
        border: none;
        background: none;
        cursor: pointer;
        border-radius: 8px;
        transition: background-color 0.2s ease;
    }

    .user-menu-trigger:hover {
        background-color: var(--color-bg-hover, #f0f0f0);
    }

    .user-avatar {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background-color: var(--color-primary, #3b82f6);
        color: white;
        font-size: 0.875rem;
        font-weight: 600;
        border-radius: 50%;
    }

    .user-avatar.large {
        width: 48px;
        height: 48px;
        font-size: 1.125rem;
    }

    .chevron {
        transition: transform 0.2s ease;
    }

    .chevron.open {
        transform: rotate(180deg);
    }

    .user-dropdown {
        position: absolute;
        top: calc(100% + 8px);
        right: 0;
        width: 280px;
        background-color: var(--color-bg-primary, #ffffff);
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        z-index: 1000;
    }

    .user-info {
        display: flex;
        gap: 1rem;
        padding: 1rem;
    }

    .user-details h4 {
        margin: 0 0 0.25rem;
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--color-text-primary, #333);
    }

    .user-details p {
        margin: 0 0 0.25rem;
        font-size: 0.75rem;
        color: var(--color-text-secondary, #666);
    }

    .user-role {
        display: inline-block;
        padding: 0.125rem 0.5rem;
        background-color: var(--color-primary-light, #eff6ff);
        color: var(--color-primary, #3b82f6);
        font-size: 0.6875rem;
        font-weight: 500;
        border-radius: 4px;
    }

    .dropdown-divider {
        height: 1px;
        background-color: var(--color-border, #e5e5e5);
    }

    .dropdown-menu {
        padding: 0.5rem 0;
    }

    .menu-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        width: 100%;
        padding: 0.5rem 1rem;
        border: none;
        background: none;
        color: var(--color-text-primary, #333);
        font-size: 0.875rem;
        text-align: left;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }

    .menu-item:hover {
        background-color: var(--color-bg-hover, #f0f0f0);
    }

    .menu-item svg {
        flex-shrink: 0;
        color: var(--color-text-secondary, #666);
    }

    .menu-item.logout {
        color: var(--color-danger, #ef4444);
    }

    .menu-item.logout svg {
        color: var(--color-danger, #ef4444);
    }

    /* Dark mode */
    :global(.dark) .user-menu-trigger:hover {
        background-color: var(--color-bg-hover-dark, #2a2a2a);
    }

    :global(.dark) .user-dropdown {
        background-color: var(--color-bg-primary-dark, #0f0f0f);
        border-color: var(--color-border-dark, #333);
    }

    :global(.dark) .user-details h4 {
        color: var(--color-text-primary-dark, #fff);
    }

    :global(.dark) .user-details p {
        color: var(--color-text-secondary-dark, #aaa);
    }

    :global(.dark) .dropdown-divider {
        background-color: var(--color-border-dark, #333);
    }

    :global(.dark) .menu-item {
        color: var(--color-text-primary-dark, #fff);
    }

    :global(.dark) .menu-item:hover {
        background-color: var(--color-bg-hover-dark, #2a2a2a);
    }

    :global(.dark) .menu-item svg {
        color: var(--color-text-secondary-dark, #aaa);
    }

    /* High contrast */
    :global(.high-contrast) .user-dropdown {
        border-width: 2px;
        border-color: #000;
    }

    /* Reduced motion */
    :global(.reduced-motion) .user-menu-trigger,
    :global(.reduced-motion) .chevron,
    :global(.reduced-motion) .menu-item {
        transition: none !important;
    }
</style>
