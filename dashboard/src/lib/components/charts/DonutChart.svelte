<script lang="ts">
    import { scaleOrdinal } from "d3-scale";
    import { arc, pie } from "d3-shape";
    import { cubicOut } from "svelte/easing";
    import { tweened } from "svelte/motion";

    export let data: Array<{ label: string; value: number; color?: string }> =
        [];
    export let width = 300;
    export let height = 300;
    export let innerRadius = 60;
    export let outerRadius = 100;

    // Center of the chart
    $: centerX = width / 2;
    $: centerY = height / 2;

    // Color scale
    const defaultColors = [
        "var(--color-primary)",
        "var(--color-secondary)",
        "var(--color-info)",
        "var(--color-warning)",
        "var(--color-danger)",
        "#8b5cf6",
        "#ec4899",
        "#14b8a6",
    ];

    $: colorScale = scaleOrdinal()
        .domain(data.map((d) => d.label))
        .range(defaultColors);

    // Pie generator
    $: pieGenerator = pie<(typeof data)[0]>()
        .value((d) => d.value)
        .sort(null);

    // Arc generator
    $: arcGenerator = arc<any>()
        .innerRadius(innerRadius)
        .outerRadius(outerRadius);

    // Arc generator for hover effect
    $: hoverArcGenerator = arc<any>()
        .innerRadius(innerRadius)
        .outerRadius(outerRadius + 10);

    // Generate pie data
    $: pieData = pieGenerator(data);

    // Total value for center display
    $: totalValue = data.reduce((sum, d) => sum + d.value, 0);

    // Animated angles
    const animatedData = pieData.map(() => ({
        startAngle: tweened(0, { duration: 600, easing: cubicOut }),
        endAngle: tweened(0, { duration: 600, easing: cubicOut }),
    }));

    // Update animations when data changes
    $: pieData.forEach((d, i) => {
        if (animatedData[i]) {
            animatedData[i].startAngle.set(d.startAngle);
            animatedData[i].endAngle.set(d.endAngle);
        }
    });

    let hoveredSlice: any = null;

    function getSliceColor(item: (typeof data)[0], index: number) {
        return item.color || colorScale(item.label);
    }
</script>

<div class="chart-container">
    <svg {width} {height}>
        <g transform="translate({centerX},{centerY})">
            <!-- Slices -->
            {#each pieData as slice, index}
                <path
                    d={arcGenerator({
                        startAngle: $animatedData[index].startAngle,
                        endAngle: $animatedData[index].endAngle,
                    })}
                    fill={getSliceColor(slice.data, index)}
                    stroke="white"
                    stroke-width="2"
                    opacity={hoveredSlice && hoveredSlice !== slice ? 0.6 : 1}
                    on:mouseenter={() => (hoveredSlice = slice)}
                    on:mouseleave={() => (hoveredSlice = null)}
                    style="transition: all 0.2s ease; cursor: pointer;"
                    transform={hoveredSlice === slice
                        ? `translate(${Math.cos((slice.startAngle + slice.endAngle) / 2) * 5},
                                   ${Math.sin((slice.startAngle + slice.endAngle) / 2) * 5})`
                        : ""}
                />
            {/each}

            <!-- Center text -->
            <text
                text-anchor="middle"
                dominant-baseline="middle"
                class="center-text"
            >
                <tspan x="0" y="-10" class="total-label">Total</tspan>
                <tspan x="0" y="10" class="total-value">{totalValue}</tspan>
            </text>
        </g>
    </svg>

    <!-- Legend -->
    <div class="legend">
        {#each data as item, index}
            <div
                class="legend-item"
                class:hovered={hoveredSlice && hoveredSlice.data === item}
            >
                <div
                    class="legend-color"
                    style="background-color: {getSliceColor(item, index)}"
                ></div>
                <span class="legend-label">{item.label}</span>
                <span class="legend-value">{item.value}</span>
                <span class="legend-percentage">
                    ({((item.value / totalValue) * 100).toFixed(1)}%)
                </span>
            </div>
        {/each}
    </div>
</div>

<style>
    .chart-container {
        display: flex;
        align-items: center;
        gap: 2rem;
        font-family: var(--font-sans);
    }

    .center-text {
        fill: var(--color-text);
    }

    .total-label {
        font-size: 14px;
        fill: var(--color-text-secondary);
    }

    .total-value {
        font-size: 24px;
        font-weight: 700;
    }

    .legend {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        transition: background-color 0.2s ease;
    }

    .legend-item.hovered {
        background-color: var(--color-bg-secondary);
    }

    .legend-color {
        width: 12px;
        height: 12px;
        border-radius: 2px;
        flex-shrink: 0;
    }

    .legend-label {
        font-size: 14px;
        color: var(--color-text);
        min-width: 100px;
    }

    .legend-value {
        font-size: 14px;
        font-weight: 600;
        color: var(--color-text);
        margin-left: auto;
    }

    .legend-percentage {
        font-size: 12px;
        color: var(--color-text-secondary);
    }

    @media (max-width: 768px) {
        .chart-container {
            flex-direction: column;
        }
    }
</style>
