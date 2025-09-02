<script lang="ts">
    import { scaleBand, scaleLinear } from "d3-scale";
    import { cubicOut } from "svelte/easing";
    import { tweened } from "svelte/motion";

    export let data: Array<{ label: string; value: number; color?: string }> =
        [];
    export let width = 600;
    export let height = 300;
    export let margin = { top: 20, right: 30, bottom: 40, left: 50 };

    // Reactive calculations
    $: innerWidth = width - margin.left - margin.right;
    $: innerHeight = height - margin.top - margin.bottom;

    // Scales
    $: xScale = scaleBand()
        .domain(data.map((d) => d.label))
        .range([0, innerWidth])
        .padding(0.2);

    $: yScale = scaleLinear()
        .domain([0, Math.max(...data.map((d) => d.value)) * 1.1])
        .range([innerHeight, 0]);

    // Animated bar heights
    let barHeights: Array<any> = [];
    $: barHeights = data.map((_, i) => {
        if (!barHeights[i]) {
            return tweened(0, {
                duration: 600,
                easing: cubicOut,
            });
        }
        return barHeights[i];
    });

    // Update animations when data changes
    $: data.forEach((d, i) => {
        if (barHeights[i]) {
            barHeights[i].set(innerHeight - yScale(d.value));
        }
    });

    let hoveredBar: (typeof data)[0] | null = null;

    function getBarColor(item: (typeof data)[0], index: number) {
        if (item.color) return item.color;
        const colors = [
            "var(--color-primary)",
            "var(--color-secondary)",
            "var(--color-info)",
            "var(--color-warning)",
            "var(--color-danger)",
        ];
        return colors[index % colors.length];
    }
</script>

<div class="chart-container">
    <svg {width} {height}>
        <g transform="translate({margin.left},{margin.top})">
            <!-- Grid lines -->
            <g class="grid y-grid">
                {#each yScale.ticks(5) as tick}
                    <line
                        x1={0}
                        x2={innerWidth}
                        y1={yScale(tick)}
                        y2={yScale(tick)}
                        stroke="var(--color-border)"
                        stroke-dasharray="2,2"
                        opacity="0.5"
                    />
                {/each}
            </g>

            <!-- X Axis -->
            <g class="x-axis" transform="translate(0,{innerHeight})">
                <line
                    x1={0}
                    x2={innerWidth}
                    y1={0}
                    y2={0}
                    stroke="var(--color-border)"
                />
                {#each data as item}
                    <g
                        transform="translate({(xScale(item.label) || 0) +
                            xScale.bandwidth() / 2},0)"
                    >
                        <line y1={0} y2={6} stroke="var(--color-border)" />
                        <text
                            y={20}
                            text-anchor="middle"
                            fill="var(--color-text-secondary)"
                            font-size="12"
                        >
                            {item.label}
                        </text>
                    </g>
                {/each}
            </g>

            <!-- Y Axis -->
            <g class="y-axis">
                <line
                    x1={0}
                    x2={0}
                    y1={0}
                    y2={innerHeight}
                    stroke="var(--color-border)"
                />
                {#each yScale.ticks(5) as tick}
                    <g transform="translate(0,{yScale(tick)})">
                        <line x1={-6} x2={0} stroke="var(--color-border)" />
                        <text
                            x={-10}
                            text-anchor="end"
                            dominant-baseline="middle"
                            fill="var(--color-text-secondary)"
                            font-size="12"
                        >
                            {tick}
                        </text>
                    </g>
                {/each}
            </g>

            <!-- Bars -->
            {#each data as item, index}
                <rect
                    x={xScale(item.label)}
                    y={yScale(item.value)}
                    width={xScale.bandwidth()}
                    height={$barHeights[index]}
                    fill={getBarColor(item, index)}
                    opacity={hoveredBar && hoveredBar !== item ? 0.6 : 1}
                    on:mouseenter={() => (hoveredBar = item)}
                    on:mouseleave={() => (hoveredBar = null)}
                    style="transition: opacity 0.2s ease; cursor: pointer;"
                />

                <!-- Value labels -->
                {#if hoveredBar === item}
                    <text
                        x={(xScale(item.label) || 0) + xScale.bandwidth() / 2}
                        y={yScale(item.value) - 5}
                        text-anchor="middle"
                        fill="var(--color-text)"
                        font-size="14"
                        font-weight="600"
                    >
                        {item.value}
                    </text>
                {/if}
            {/each}
        </g>
    </svg>
</div>

<style>
    .chart-container {
        position: relative;
        font-family: var(--font-sans);
    }

    svg {
        overflow: visible;
    }
</style>
