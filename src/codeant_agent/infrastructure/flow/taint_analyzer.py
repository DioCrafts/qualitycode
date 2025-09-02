"""
Taint analyzer mínimo: user input -> sinks simples (print, exec, innerHTML).
"""

import re
from typing import Dict, Any, List

from ...application.ports.flow_ports import TaintAnalyzerPort


SOURCE_RE = re.compile(r"(input\(|request\.|argv|stdin)")
SINK_RE = re.compile(r"\b(print|exec|system|innerHTML|dangerouslySetInnerHTML)\b")


class SimpleTaintAnalyzer(TaintAnalyzerPort):
    async def analyze(self, project_path: str, file_path: str) -> Dict[str, Any]:
        path = file_path if file_path.startswith("/") else f"{project_path}/{file_path}"
        sources: List[int] = []
        sinks: List[int] = []
        issues: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for idx, line in enumerate(fh, start=1):
                    if SOURCE_RE.search(line):
                        sources.append(idx)
                    if SINK_RE.search(line):
                        sinks.append(idx)
            # naive pairing: any source before any sink
            for s in sources:
                for k in sinks:
                    if s < k:
                        issues.append({
                            'source_line': s,
                            'sink_line': k,
                            'source': 'user_input',
                            'sink': 'sink',
                            'vulnerability_type': 'taint_flow',
                            'severity': 3
                        })
        except FileNotFoundError:
            pass
        return {'issues': issues}


