"""
Data flow analyzer mínimo: detecta dependencias triviales var = X; uso posterior.
"""

import re
from typing import Dict, Any, List

from ...application.ports.flow_ports import DataFlowAnalyzerPort


ASSIGN_RE = re.compile(r"^(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<rhs>.+)$")
USE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


class SimpleDataFlowAnalyzer(DataFlowAnalyzerPort):
    async def analyze(self, project_path: str, file_path: str) -> Dict[str, Any]:
        path = file_path if file_path.startswith("/") else f"{project_path}/{file_path}"
        definitions: Dict[str, int] = {}
        variables: List[str] = []
        dependencies: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for idx, line in enumerate(fh, start=1):
                    stripped = line.strip()
                    if not stripped or stripped.startswith(('#', '//')):
                        continue
                    m = ASSIGN_RE.match(stripped)
                    if m:
                        var = m.group('var')
                        variables.append(var)
                        definitions[var] = idx
                        # record uses in rhs
                        for tok in USE_RE.findall(m.group('rhs')):
                            if tok in definitions:
                                dependencies.append({
                                    'variable': tok,
                                    'definition_line': definitions[tok],
                                    'use_line': idx,
                                    'dependency_type': 'true'
                                })
                        continue
                    # Non-assignment line: record uses
                    for tok in USE_RE.findall(stripped):
                        if tok in definitions:
                            dependencies.append({
                                'variable': tok,
                                'definition_line': definitions[tok],
                                'use_line': idx,
                                'dependency_type': 'true'
                            })
        except FileNotFoundError:
            pass
        return {'variables': sorted(set(variables)), 'dependencies': dependencies}


