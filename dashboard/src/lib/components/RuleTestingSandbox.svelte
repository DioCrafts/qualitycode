<script lang="ts">
    import type { CustomRule } from "$lib/types/dashboard";
    import { onMount } from "svelte";

    export let rule: CustomRule | null;
    export let projectId: string;

    let isRunning = false;
    let testCode = "";
    let results: { lineNumber: number; message: string; severity: string }[] =
        [];
    let error: string | null = null;

    onMount(() => {
        // Load sample code based on rule type if available
        if (rule) {
            loadSampleCode();
        }
    });

    function loadSampleCode() {
        // This would ideally load code from an API based on rule's language and type
        // For now, we'll just set default examples
        if (
            rule?.language === "typescript" ||
            rule?.language === "javascript"
        ) {
            testCode = `// Sample TypeScript/JavaScript code
function processList(items) {
  // Missing type annotations
  let result = [];
  
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    result.push({
      id: item.id,
      value: item.value * 2,
      processed: true
    });
  }
  
  // Could be a map operation
  return result;
}

// Function with too many parameters
function doSomethingComplex(a, b, c, d, e, f, g) {
  // Complex logic here
  if (a > 0) {
    if (b > 0) {
      if (c > 0) {
        // Deeply nested conditionals
        return a + b + c;
      }
    }
  }
  
  return 0;
}
`;
        } else if (rule?.language === "python") {
            testCode = `# Sample Python code
def process_list(items):
    # Could use list comprehension
    result = []
    
    for i in range(len(items)):
        item = items[i]
        result.append({
            'id': item['id'],
            'value': item['value'] * 2,
            'processed': True
        })
    
    return result

# Function with too many parameters
def do_something_complex(a, b, c, d, e, f, g):
    # Complex logic here
    if a > 0:
        if b > 0:
            if c > 0:
                # Deeply nested conditionals
                return a + b + c
    
    return 0
`;
        } else {
            testCode = `// Sample code\n// Paste code to test against your rule here`;
        }
    }

    async function runTest() {
        if (!testCode.trim()) return;

        isRunning = true;
        error = null;
        results = [];

        try {
            // In a real app, this would be an API call to test the rule
            // For now, we'll simulate results
            await new Promise((resolve) => setTimeout(resolve, 1000));

            if (rule) {
                // Generate simulated results based on the rule
                generateMockResults();
            } else {
                error = "No rule selected for testing";
            }
        } catch (err) {
            error =
                err instanceof Error ? err.message : "Failed to run rule test";
        } finally {
            isRunning = false;
        }
    }

    function generateMockResults() {
        // This simulates finding issues in the code based on the rule
        // In a real app, this would be done by an actual rule engine

        if (!rule) return;

        const lines = testCode.split("\n");

        // Find patterns in the code to create mock results
        // These are simplified examples based on common issues
        if (
            rule.name?.toLowerCase().includes("parameter") ||
            rule.description?.toLowerCase().includes("parameter")
        ) {
            // Find functions with too many parameters
            const paramRegex = /function\s+\w+\s*\(([^)]*)\)/g;
            let match;

            while ((match = paramRegex.exec(testCode)) !== null) {
                const params = match[1].split(",").filter((p) => p.trim());
                if (params.length > 3) {
                    const lineNumber = testCode
                        .substring(0, match.index)
                        .split("\n").length;
                    results.push({
                        lineNumber,
                        message: `Function has ${params.length} parameters, which exceeds the recommended limit`,
                        severity: rule.severity || "warning",
                    });
                }
            }
        }

        if (
            rule.name?.toLowerCase().includes("nested") ||
            rule.description?.toLowerCase().includes("nested")
        ) {
            // Find nested conditionals
            const lines = testCode.split("\n");
            let indentLevel = 0;
            let ifCount = 0;

            lines.forEach((line, index) => {
                const trimmedLine = line.trim();

                if (
                    trimmedLine.startsWith("if ") ||
                    trimmedLine.startsWith("if(")
                ) {
                    ifCount++;
                    if (ifCount > 2) {
                        results.push({
                            lineNumber: index + 1,
                            message: "Deeply nested conditional detected",
                            severity: rule.severity || "warning",
                        });
                    }
                }

                if (trimmedLine.endsWith("{")) {
                    indentLevel++;
                }

                if (trimmedLine.startsWith("}")) {
                    indentLevel--;
                    if (ifCount > 0) ifCount--;
                }
            });
        }

        // Add at least one result if we didn't find any yet
        if (results.length === 0) {
            // Find a long line
            const lines = testCode.split("\n");
            for (let i = 0; i < lines.length; i++) {
                if (lines[i].length > 80) {
                    results.push({
                        lineNumber: i + 1,
                        message:
                            "Line exceeds recommended length of 80 characters",
                        severity: rule.severity || "info",
                    });
                    break;
                }
            }
        }

        // Still no results? Add a placeholder
        if (results.length === 0) {
            results.push({
                lineNumber: 1,
                message: "This is a sample issue for demonstration purposes",
                severity: rule.severity || "info",
            });
        }

        // Sort by line number
        results.sort((a, b) => a.lineNumber - b.lineNumber);
    }
</script>

<div class="space-y-6">
    <!-- Sandbox header -->
    <div class="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
        <div class="flex items-start justify-between">
            <div>
                <h3
                    class="text-lg font-medium text-gray-900 dark:text-white mb-1"
                >
                    Rule Testing Sandbox
                </h3>
                <p class="text-gray-600 dark:text-gray-400">
                    Test your custom rules against code samples
                </p>
            </div>

            {#if rule}
                <div
                    class="px-3 py-1.5 bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300 rounded-lg text-sm"
                >
                    Testing: {rule.name || "Unnamed Rule"}
                </div>
            {:else}
                <div
                    class="px-3 py-1.5 bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300 rounded-lg text-sm"
                >
                    No rule selected
                </div>
            {/if}
        </div>
    </div>

    <!-- Code editor and results -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Code input panel -->
        <div>
            <div class="mb-4 flex justify-between items-center">
                <label
                    class="block text-sm font-medium text-gray-700 dark:text-gray-300"
                >
                    Test code
                </label>
                <button
                    on:click={loadSampleCode}
                    class="text-sm text-blue-600 hover:text-blue-800"
                >
                    Load Sample Code
                </button>
            </div>

            <div
                class="border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden mb-4"
            >
                <textarea
                    bind:value={testCode}
                    rows="15"
                    class="w-full p-4 font-mono text-sm bg-gray-50 dark:bg-gray-800 focus:outline-none"
                ></textarea>
            </div>

            <div class="flex justify-end">
                <button
                    on:click={runTest}
                    disabled={isRunning || !testCode.trim() || !rule}
                    class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700
                           disabled:opacity-50 disabled:cursor-not-allowed
                           transition-colors flex items-center space-x-2"
                >
                    {#if isRunning}
                        <svg
                            class="animate-spin h-4 w-4"
                            fill="none"
                            viewBox="0 0 24 24"
                        >
                            <circle
                                class="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                stroke-width="4"
                            ></circle>
                            <path
                                class="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            ></path>
                        </svg>
                        <span>Running...</span>
                    {:else}
                        <svg
                            class="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                stroke-width="2"
                                d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                            />
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                stroke-width="2"
                                d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                        </svg>
                        <span>Run Test</span>
                    {/if}
                </button>
            </div>
        </div>

        <!-- Results panel -->
        <div>
            <h4
                class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4"
            >
                Results
            </h4>

            {#if error}
                <div
                    class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-4"
                >
                    <p class="text-red-800 dark:text-red-200">{error}</p>
                </div>
            {/if}

            {#if !isRunning && !error && results.length === 0}
                <div
                    class="bg-gray-50 dark:bg-gray-700 rounded-lg p-6 text-center"
                >
                    <svg
                        class="w-12 h-12 text-gray-400 mx-auto mb-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            stroke-width="2"
                            d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                        />
                    </svg>
                    <p class="text-gray-600 dark:text-gray-400">
                        Click "Run Test" to analyze your code
                    </p>
                </div>
            {:else if !isRunning && results.length > 0}
                <div
                    class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden"
                >
                    <div
                        class="p-4 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600 flex justify-between items-center"
                    >
                        <h4 class="font-medium text-gray-900 dark:text-white">
                            Found {results.length} issue{results.length !== 1
                                ? "s"
                                : ""}
                        </h4>
                    </div>

                    <div class="divide-y divide-gray-100 dark:divide-gray-700">
                        {#each results as result}
                            <div class="p-4">
                                <div class="flex items-start">
                                    <div
                                        class="flex-shrink-0 w-12 text-right mr-4 text-sm text-gray-500"
                                    >
                                        Line {result.lineNumber}:
                                    </div>
                                    <div class="flex-1">
                                        <div class="flex items-start">
                                            <span
                                                class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium capitalize
                                                {result.severity === 'error'
                                                    ? 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300'
                                                    : result.severity ===
                                                        'warning'
                                                      ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-300'
                                                      : 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-300'}"
                                            >
                                                {result.severity}
                                            </span>
                                            <p
                                                class="ml-2 text-gray-700 dark:text-gray-300"
                                            >
                                                {result.message}
                                            </p>
                                        </div>

                                        <div
                                            class="mt-2 bg-gray-50 dark:bg-gray-700 p-2 rounded text-sm font-mono overflow-x-auto"
                                        >
                                            {testCode.split("\n")[
                                                result.lineNumber - 1
                                            ] || ""}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        {/each}
                    </div>
                </div>

                {#if rule}
                    <div
                        class="mt-6 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-4"
                    >
                        <h5
                            class="text-sm font-medium text-gray-800 dark:text-gray-200 mb-2"
                        >
                            Rule Details
                        </h5>
                        <div class="space-y-2">
                            <div class="grid grid-cols-3 gap-2 text-sm">
                                <div class="text-gray-600 dark:text-gray-400">
                                    Name:
                                </div>
                                <div
                                    class="col-span-2 text-gray-900 dark:text-white"
                                >
                                    {rule.name || "Unnamed"}
                                </div>

                                <div class="text-gray-600 dark:text-gray-400">
                                    Category:
                                </div>
                                <div
                                    class="col-span-2 text-gray-900 dark:text-white capitalize"
                                >
                                    {rule.category || "N/A"}
                                </div>

                                <div class="text-gray-600 dark:text-gray-400">
                                    Severity:
                                </div>
                                <div
                                    class="col-span-2 text-gray-900 dark:text-white capitalize"
                                >
                                    {rule.severity || "warning"}
                                </div>
                            </div>
                        </div>
                    </div>
                {/if}
            {/if}
        </div>
    </div>
</div>
