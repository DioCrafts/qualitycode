"""
Compliance checker mínimo para OWASP (MVP Fase 23).
"""

from typing import List, Dict, Any

from ...application.ports.security_ports import ComplianceCheckerPort


class SimpleComplianceChecker(ComplianceCheckerPort):
    async def check(self, framework: str, vulnerabilities: List[Dict[str, Any]]) -> Dict[str, Any]:
        framework_name = framework.upper()
        score = 100.0
        violations: List[str] = []

        # Penalizar por categorías OWASP relevantes
        for v in vulnerabilities:
            cat = (v.get("category") or "").lower()
            sev = int(v.get("severity", 0))
            penalty = 0.0
            if "injection" in cat:
                penalty = 10.0 * sev
                violations.append("OWASP-A03 Injection")
            elif "crosssitescripting" in cat or "xss" in cat:
                penalty = 8.0 * sev
                violations.append("OWASP-XSS")
            elif "brokenaccess" in cat:
                penalty = 12.0 * sev
                violations.append("OWASP-A01 Broken Access Control")
            score -= penalty

        score = max(0.0, score)
        overall = "compliant" if score >= 80.0 else ("partially_compliant" if score >= 50.0 else "non_compliant")

        return {
            "framework": framework_name,
            "overall_status": overall,
            "score": score,
            "violations": sorted(set(violations)),
        }


