<script lang="ts">
    import { extent } from "d3-array";
    import { scaleLinear, scaleTime } from "d3-scale";
    import { curveMonotoneX, line } from "d3-shape";
    import { cubicOut } from "svelte/easing";
    import { tweened } from "svelte/motion";

    export let data: Array<{ date: Date; value: number }> = [];
    export let width = 600;
    export let height = 300;
    export let margin = { top: 20, right: 30, bottom: 40, left: 50 };

    let svgElement: SVGSVGElement;

    // Reactive calculations
    $: innerWidth = width - margin.left - margin.right;
    $: innerHeight = height - margin.top - margin.bottom;

    // Scales
    $: xScale = scaleTime()
        .domain(extent(data, (d) => d.date) as [Date, Date])
        .range([0, innerWidth]);

    $: yScale = scaleLinear()
        .domain([0, Math.max(...data.map((d) => d.value)) * 1.1])
        .range([innerHeight, 0]);

    // Line generator
    $: lineGenerator = line<{ date: Date; value: number }>()
        .x((d) => xScale(d.date))
        .y((d) => yScale(d.value))
        .curve(curveMonotoneX);

    // Path data
    $: pathData = lineGenerator(data) || "";

    // Animated values
    const animatedValue = tweened(0, {
        duration: 400,
        easing: cubicOut,
    });

    let hoveredPoint: { date: Date; value: number } | null = null;
    let mousePosition = { x: 0, y: 0 };

    function handleMouseMove(event: MouseEvent) {
        if (!svgElement) return;

        const rect = svgElement.getBoundingClientRect();
        const x = event.clientX - rect.left - margin.left;
        const y = event.clientY - rect.top - margin.top;

        mousePosition = { x, y };

        // Find closest data point
        const xDate = xScale.invert(x);
        const bisector = (a: (typeof data)[0], b: (typeof data)[0]) =>
            a.date.getTime() - b.date.getTime();

        const index = data.findIndex((d, i) => {
            if (i === data.length - 1) return true;
            return xDate >= d.date && xDate < data[i + 1].date;
        });

        if (index >= 0) {
            hoveredPoint = data[index];
            animatedValue.set(data[index].value);
        }
    }

    function handleMouseLeave() {
        hoveredPoint = null;
    }
</script>

<div class="chart-container">
    <svg
        bind:this={svgElement}
        {width}
        {height}
        on:mousemove={handleMouseMove}
        on:mouseleave={handleMouseLeave}
    >
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

            <!-- Axes -->
            <g class="x-axis" transform="translate(0,{innerHeight})">
                <line
                    x1={0}
                    x2={innerWidth}
                    y1={0}
                    y2={0}
                    stroke="var(--color-border)"
                />
                {#each xScale.ticks(5) as tick}
                    <g transform="translate({xScale(tick)},0)">
                        <line y1={0} y2={6} stroke="var(--color-border)" />
                        <text
                            y={20}
                            text-anchor="middle"
                            fill="var(--color-text-secondary)"
                            font-size="12"
                        >
                            {tick.toLocaleDateString()}
                        </text>
                    </g>
                {/each}
            </g>

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

            <!-- Line -->
            <path
                d={pathData}
                fill="none"
                stroke="var(--color-primary)"
                stroke-width="2"
            />

            <!-- Data points -->
            {#each data as point}
                <circle
                    cx={xScale(point.date)}
                    cy={yScale(point.value)}
                    r="4"
                    fill="var(--color-primary)"
                    stroke="white"
                    stroke-width="2"
                />
            {/each}

            <!-- Hover overlay -->
            {#if hoveredPoint}
                <g
                    transform="translate({xScale(hoveredPoint.date)},{yScale(
                        hoveredPoint.value,
                    )})"
                >
                    <circle
                        r="6"
                        fill="var(--color-primary)"
                        stroke="white"
                        stroke-width="3"
                    />
                    <rect
                        x={-40}
                        y={-40}
                        width="80"
                        height="30"
                        fill="var(--color-bg-primary)"
                        stroke="var(--color-border)"
                        rx="4"
                    />
                    <text
                        y={-20}
                        text-anchor="middle"
                        fill="var(--color-text)"
                        font-size="14"
                        font-weight="600"
                    >
                        {$animatedValue.toFixed(1)}
                    </text>
                </g>
            {/if}
        </g>
    </svg>

    {#if hoveredPoint}
        <div
            class="tooltip"
            style="left: {mousePosition.x + margin.left}px; top: {height +
                10}px;"
        >
            <div class="tooltip-date">
                {hoveredPoint.date.toLocaleDateString()}
            </div>
            <div class="tooltip-value">
                Valor: {hoveredPoint.value.toFixed(1)}
            </div>
        </div>
    {/if}
</div>

<style>
    .chart-container {
        position: relative;
        font-family: var(--font-sans);
    }

    svg {
        overflow: visible;
    }

    .tooltip {
        position: absolute;
        background: var(--color-bg-primary);
        border: 1px solid var(--color-border);
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 12px;
        pointer-events: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transform: translateX(-50%);
    }

    .tooltip-date {
        color: var(--color-text-secondary);
        margin-bottom: 4px;
    }

    .tooltip-value {
        color: var(--color-text);
        font-weight: 600;
    }
</style>
