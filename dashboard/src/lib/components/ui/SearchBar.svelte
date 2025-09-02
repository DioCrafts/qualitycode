<script lang="ts">
    import { createEventDispatcher } from "svelte";

    export let value = "";
    export let placeholder = "Buscar...";
    export let debounce = 300;

    const dispatch = createEventDispatcher();

    let timeoutId: ReturnType<typeof setTimeout>;

    function handleInput() {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            dispatch("search", value);
        }, debounce);
    }

    function handleKeydown(event: KeyboardEvent) {
        if (event.key === "Enter") {
            clearTimeout(timeoutId);
            dispatch("search", value);
        }
    }

    function clearSearch() {
        value = "";
        dispatch("search", value);
    }
</script>

<div class="search-bar">
    <svg
        class="search-icon"
        width="20"
        height="20"
        viewBox="0 0 20 20"
        fill="none"
        stroke="currentColor"
    >
        <circle cx="9" cy="9" r="7"></circle>
        <path d="M14 14l3.5 3.5"></path>
    </svg>

    <input
        type="search"
        bind:value
        on:input={handleInput}
        on:keydown={handleKeydown}
        {placeholder}
        class="search-input"
        aria-label="Search"
    />

    {#if value}
        <button
            class="clear-button"
            on:click={clearSearch}
            aria-label="Clear search"
        >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path
                    d="M12.354 4.354a.5.5 0 0 0-.708-.708L8 7.293 4.354 3.646a.5.5 0 1 0-.708.708L7.293 8l-3.647 3.646a.5.5 0 0 0 .708.708L8 8.707l3.646 3.647a.5.5 0 0 0 .708-.708L8.707 8l3.647-3.646z"
                />
            </svg>
        </button>
    {/if}
</div>

<style>
    .search-bar {
        position: relative;
        display: flex;
        align-items: center;
        width: 100%;
    }

    .search-icon {
        position: absolute;
        left: 12px;
        color: var(--color-text-muted, #999);
        pointer-events: none;
    }

    .search-input {
        width: 100%;
        padding: 0.5rem 2.5rem 0.5rem 2.5rem;
        border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 8px;
        font-size: 0.875rem;
        background-color: var(--color-bg-secondary, #f5f5f5);
        color: var(--color-text-primary, #333);
        transition: all 0.2s ease;
    }

    .search-input:focus {
        outline: none;
        border-color: var(--color-primary, #3b82f6);
        background-color: var(--color-bg-primary, #ffffff);
    }

    .search-input::placeholder {
        color: var(--color-text-muted, #999);
    }

    .clear-button {
        position: absolute;
        right: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        padding: 0;
        border: none;
        background: none;
        color: var(--color-text-muted, #999);
        cursor: pointer;
        border-radius: 4px;
        transition: all 0.2s ease;
    }

    .clear-button:hover {
        background-color: var(--color-bg-hover, #f0f0f0);
        color: var(--color-text-primary, #333);
    }

    /* Dark mode */
    :global(.dark) .search-input {
        background-color: var(--color-bg-secondary-dark, #1a1a1a);
        border-color: var(--color-border-dark, #333);
        color: var(--color-text-primary-dark, #fff);
    }

    :global(.dark) .search-input:focus {
        background-color: var(--color-bg-primary-dark, #0f0f0f);
    }

    :global(.dark) .clear-button:hover {
        background-color: var(--color-bg-hover-dark, #2a2a2a);
        color: var(--color-text-primary-dark, #fff);
    }

    /* High contrast */
    :global(.high-contrast) .search-input {
        border-width: 2px;
    }

    :global(.high-contrast) .search-input:focus {
        border-color: #000;
    }

    /* Reduced motion */
    :global(.reduced-motion) .search-input,
    :global(.reduced-motion) .clear-button {
        transition: none !important;
    }
</style>
