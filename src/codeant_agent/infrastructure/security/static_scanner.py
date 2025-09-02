"""
Escáner estático mínimo (MVP) para detección básica de vulnerabilidades.
"""

import re
from typing import List, Dict, Any

from ...application.ports.security_ports import StaticSecurityScannerPort


SQL_PATTERN = re.compile(r"(SELECT\s+.*\s+FROM\s+.*\+)|(\.format\()|(f[\"'].*\{.*\}.*[\"'])", re.IGNORECASE)
XSS_PATTERN = re.compile(r"innerHTML\s*=|dangerouslySetInnerHTML", re.IGNORECASE)


class SimpleStaticSecurityScanner(StaticSecurityScannerPort):
    async def scan_files(self, project_path: str, files: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for file_rel in files:
            file_path = f"{project_path}/{file_rel}" if not file_rel.startswith("/") else file_rel
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                    for idx, line in enumerate(fh, start=1):
                        if SQL_PATTERN.search(line):
                            results.append({
                                "category": "Injection",
                                "title": "Posible SQL Injection",
                                "description": "Construcción dinámica de SQL detectada",
                                "severity": 4,
                                "file_path": file_path,
                                "line": idx,
                                "cwe_id": "CWE-89",
                            })
                        if XSS_PATTERN.search(line):
                            results.append({
                                "category": "CrossSiteScripting",
                                "title": "Posible XSS",
                                "description": "Uso de innerHTML/dangerouslySetInnerHTML",
                                "severity": 3,
                                "file_path": file_path,
                                "line": idx,
                                "cwe_id": "CWE-79",
                            })
            except FileNotFoundError:
                continue
        return results


