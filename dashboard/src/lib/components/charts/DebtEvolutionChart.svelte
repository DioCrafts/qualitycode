<script lang="ts">
    import type { DebtEvolution } from "$lib/types/dashboard";
    import { scaleLinear, scaleTime } from "d3-scale";
    import { area, curveMonotoneX, line } from "d3-shape";
    import { Html, LayerCake, Svg } from "layercake";

    export let data: DebtEvolution[];

    $: xDomain = [
        Math.min(...data.map((d) => d.date.getTime())),
        Math.max(...data.map((d) => d.date.getTime())),
    ];

    $: yDomain = [0, Math.max(...data.map((d) => d.totalDebt)) * 1.1];

    let hoveredPoint: DebtEvolution | null = null;
</script>

<div class="w-full h-64">
    <LayerCake
        padding={{ top: 10, right: 10, bottom: 40, left: 50 }}
        x={(d) => d.date.getTime()}
        y={(d) => d.totalDebt}
        xScale={scaleTime()}
        yScale={scaleLinear()}
        {xDomain}
        {yDomain}
        {data}
    >
        <Svg>
            <defs>
                <linearGradient
                    id="debtGradient"
                    x1="0%"
                    y1="0%"
                    x2="0%"
                    y2="100%"
                >
                    <stop
                        offset="0%"
                        style="stop-color:#ef4444;stop-opacity:0.8"
                    />
                    <stop
                        offset="100%"
                        style="stop-color:#ef4444;stop-opacity:0.1"
                    />
                </linearGradient>
            </defs>

            <!-- Area -->
            <path
                d={area()
                    .x((d) => d.x)
                    .y0((d) => d.yScale(0))
                    .y1((d) => d.y)
                    .curve(curveMonotoneX)(
                    data.map((d) => ({
                        x: d.xScale(d.date.getTime()),
                        y: d.yScale(d.totalDebt),
                        yScale: d.yScale,
                    })),
                )}
                fill="url(#debtGradient)"
            />

            <!-- Line -->
            <path
                d={line()
                    .x((d) => d.xScale(d.date.getTime()))
                    .y((d) => d.yScale(d.totalDebt))
                    .curve(curveMonotoneX)(data)}
                fill="none"
                stroke="#ef4444"
                stroke-width="2"
            />

            <!-- Points -->
            {#each data as point}
                <circle
                    cx={point.xScale(point.date.getTime())}
                    cy={point.yScale(point.totalDebt)}
                    r="4"
                    fill="#ef4444"
                    on:mouseenter={() => (hoveredPoint = point)}
                    on:mouseleave={() => (hoveredPoint = null)}
                    class="cursor-pointer hover:r-6 transition-all"
                />
            {/each}
        </Svg>

        <Html>
            {#if hoveredPoint}
                <div
                    class="absolute bg-gray-900 text-white p-2 rounded shadow-lg text-sm"
                    style="left: {hoveredPoint.xScale(
                        hoveredPoint.date.getTime(),
                    )}px; 
                 top: {hoveredPoint.yScale(hoveredPoint.totalDebt) - 40}px;
                 transform: translateX(-50%);"
                >
                    <div class="font-medium">
                        {hoveredPoint.date.toLocaleDateString()}
                    </div>
                    <div>Debt: {hoveredPoint.totalDebt.toFixed(0)} hours</div>
                    <div>
                        Ratio: {(hoveredPoint.debtRatio * 100).toFixed(1)}%
                    </div>
                </div>
            {/if}
        </Html>
    </LayerCake>
</div>
