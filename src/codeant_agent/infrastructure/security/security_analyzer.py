"""
Analizador de seguridad para el análisis de proyectos.
"""

import os
import re
from typing import Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SecurityAnalyzer:
    """
    Analizador básico de seguridad que busca patrones comunes de vulnerabilidades.
    """

    def __init__(self):
        # Patrones de seguridad comunes
        self.unsafe_patterns = {
            'python': [
                (re.compile(r'\beval\s*\('), 'Uso de eval() - Función insegura'),
                (re.compile(r'\bexec\s*\('), 'Uso de exec() - Función insegura'),
                (re.compile(r'\b__import__\s*\('), 'Uso de __import__() - Función insegura'),
                (re.compile(r'\binput\s*\('), 'Uso de input() sin validación'),
                (re.compile(r'\bos\.system\s*\('), 'Uso de os.system() - Función insegura'),
                (re.compile(r'\bsubprocess\..*\('), 'Uso de subprocess sin validación'),
            ],
            'javascript': [
                (re.compile(r'\beval\s*\('), 'Uso de eval() - Función insegura'),
                (re.compile(r'\binnerHTML\s*='), 'Uso de innerHTML - Posible XSS'),
                (re.compile(r'\bdangerouslySetInnerHTML'), 'Uso de dangerouslySetInnerHTML - Posible XSS'),
                (re.compile(r'\bdocument\.write\s*\('), 'Uso de document.write() - Función insegura'),
                (re.compile(r'\blocalStorage\.|\bsessionStorage\.'), 'Uso de storage sin validación'),
            ],
            'typescript': [
                (re.compile(r'\beval\s*\('), 'Uso de eval() - Función insegura'),
                (re.compile(r'\binnerHTML\s*='), 'Uso de innerHTML - Posible XSS'),
                (re.compile(r'\bdangerouslySetInnerHTML'), 'Uso de dangerouslySetInnerHTML - Posible XSS'),
                (re.compile(r'\bdocument\.write\s*\('), 'Uso de document.write() - Función insegura'),
            ],
            'rust': [
                (re.compile(r'\bunsafe\s*\{'), 'Uso de bloque unsafe'),
                (re.compile(r'\bmem::transmute'), 'Uso de mem::transmute - Función insegura'),
            ]
        }

        # Patrones para detectar secretos hardcodeados
        self.secret_patterns = [
            (re.compile(r'password\s*[:=]\s*["\'][^"\']+["\']', re.IGNORECASE), 'Password hardcodeado'),
            (re.compile(r'api_key\s*[:=]\s*["\'][^"\']+["\']', re.IGNORECASE), 'API Key hardcodeado'),
            (re.compile(r'secret\s*[:=]\s*["\'][^"\']+["\']', re.IGNORECASE), 'Secret hardcodeado'),
            (re.compile(r'token\s*[:=]\s*["\'][^"\']+["\']', re.IGNORECASE), 'Token hardcodeado'),
            (re.compile(r'Bearer\s+[A-Za-z0-9\-_\.]+', re.IGNORECASE), 'Token Bearer expuesto'),
        ]

        # Patrones de SQL Injection
        self.sql_injection_patterns = [
            (re.compile(r'(SELECT|INSERT|UPDATE|DELETE).*\+.*FROM', re.IGNORECASE), 'Posible SQL Injection con concatenación'),
            (re.compile(r'.*\%s.*\%.*', re.IGNORECASE), 'Posible SQL Injection con formateo'),
            (re.compile(r'.*\{.*\}.*', re.IGNORECASE), 'Posible SQL Injection con f-strings'),
        ]

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analiza un archivo individual en busca de vulnerabilidades.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (FileNotFoundError, UnicodeDecodeError):
            return {'vulnerabilities': [], 'issues': 0}

        vulnerabilities = []
        file_extension = Path(file_path).suffix.lower()

        # Determinar el lenguaje basado en la extensión
        language = self._get_language_from_extension(file_extension)

        # Buscar patrones inseguros del lenguaje específico
        if language in self.unsafe_patterns:
            for pattern, description in self.unsafe_patterns[language]:
                matches = pattern.findall(content)
                if matches:
                    vulnerabilities.append({
                        'type': 'unsafe_function',
                        'description': description,
                        'severity': 'HIGH' if 'eval' in description or 'exec' in description else 'MEDIUM',
                        'file': file_path,
                        'matches': len(matches)
                    })

        # Buscar secretos hardcodeados
        for pattern, description in self.secret_patterns:
            if pattern.search(content):
                vulnerabilities.append({
                    'type': 'hardcoded_secret',
                    'description': description,
                    'severity': 'CRITICAL',
                    'file': file_path,
                    'matches': 1
                })

        # Buscar patrones de SQL Injection
        for pattern, description in self.sql_injection_patterns:
            if pattern.search(content):
                vulnerabilities.append({
                    'type': 'sql_injection',
                    'description': description,
                    'severity': 'HIGH',
                    'file': file_path,
                    'matches': 1
                })

        return {
            'vulnerabilities': vulnerabilities,
            'issues': len(vulnerabilities)
        }

    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """
        Analiza todo el proyecto en busca de vulnerabilidades.
        """
        logger.info("Ejecutando análisis de seguridad básico...")

        all_vulnerabilities = []
        total_files = 0

        # Extensiones de archivos a analizar
        extensions = ['.py', '.js', '.ts', '.tsx', '.rs']

        for root, dirs, files in os.walk(project_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    result = self.analyze_file(file_path)

                    if result['vulnerabilities']:
                        all_vulnerabilities.extend(result['vulnerabilities'])
                        total_files += 1

        # Agrupar por severidad
        severity_count = {
            'CRITICAL': len([v for v in all_vulnerabilities if v['severity'] == 'CRITICAL']),
            'HIGH': len([v for v in all_vulnerabilities if v['severity'] == 'HIGH']),
            'MEDIUM': len([v for v in all_vulnerabilities if v['severity'] == 'MEDIUM']),
            'LOW': len([v for v in all_vulnerabilities if v['severity'] == 'LOW'])
        }

        # Calcular OWASP compliance score basado en vulnerabilidades encontradas
        total_vulnerabilities = len(all_vulnerabilities)
        owasp_compliance = max(0, 100 - (total_vulnerabilities * 5))

        logger.info(f"Análisis de seguridad completado: {total_vulnerabilities} vulnerabilidades encontradas")

        return {
            "vulnerabilities": severity_count,
            "security_hotspots": total_vulnerabilities,
            "owasp_compliance": owasp_compliance,
            "cve_matches": 0,  # No implementado aún
            "details": all_vulnerabilities[:10]  # Top 10 vulnerabilidades
        }

    def _get_language_from_extension(self, extension: str) -> str:
        """Determina el lenguaje basado en la extensión del archivo."""
        mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.rs': 'rust'
        }
        return mapping.get(extension, 'unknown')
