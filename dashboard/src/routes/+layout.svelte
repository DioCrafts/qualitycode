<script lang="ts">
	import { browser } from "$app/environment";
	import AppShell from "$lib/components/layout/AppShell.svelte";
	import { initializePreferences } from "$lib/stores/preferences";
	import { onMount } from "svelte";
	import "../app.css";

	// Initialize user preferences on mount
	onMount(() => {
		if (browser) {
			initializePreferences();
		}
	});
</script>

<AppShell>
	<slot />
</AppShell>
