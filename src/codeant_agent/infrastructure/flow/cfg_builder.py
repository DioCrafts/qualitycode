"""
CFG Builder mínimo: crea nodos por línea con statements simples.
"""

from typing import Dict, Any, List

from ...application.ports.flow_ports import CFGBuilderPort


class SimpleCFGBuilder(CFGBuilderPort):
    async def build_cfg(self, project_path: str, file_path: str) -> Dict[str, Any]:
        path = file_path if file_path.startswith("/") else f"{project_path}/{file_path}"
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        last_node_id = None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for idx, line in enumerate(fh, start=1):
                    label = line.strip()
                    if not label:
                        continue
                    node_id = f"n{idx}"
                    nodes.append({"id": node_id, "label": label, "line": idx})
                    if last_node_id is not None:
                        edges.append({"source": last_node_id, "target": node_id, "edge_type": "sequential"})
                    last_node_id = node_id
        except FileNotFoundError:
            pass
        return {"nodes": nodes, "edges": edges}


