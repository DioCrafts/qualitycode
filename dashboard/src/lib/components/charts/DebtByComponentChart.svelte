<script lang="ts">
    import type { DebtComponent } from "$lib/types/dashboard";

    export let components: DebtComponent[];
</script>

<div class="space-y-4">
    {#each components as component (component.path)}
        <div class="relative">
            <div class="flex justify-between items-center mb-2">
                <div>
                    <h4 class="font-medium text-gray-900 dark:text-white">
                        {component.name}
                    </h4>
                    <p class="text-sm text-gray-500">{component.path}</p>
                </div>
                <div class="text-right">
                    <p class="font-medium text-gray-900 dark:text-white">
                        {component.debt}h
                    </p>
                    <p class="text-sm text-gray-500">
                        {component.issues} issues
                    </p>
                </div>
            </div>
            <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                    class="h-2 rounded-full bg-gradient-to-r from-yellow-500 to-red-500"
                    style="width: {(component.debt /
                        Math.max(...components.map((c) => c.debt))) *
                        100}%"
                />
            </div>
        </div>
    {/each}
</div>
