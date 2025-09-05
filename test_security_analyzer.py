#!/usr/bin/env python3
"""
Script de prueba para verificar que el SecurityAnalyzer funciona correctamente.
"""

import asyncio
import tempfile
import os
from pathlib import Path

# Agregar el directorio src al path
import sys
sys.path.insert(0, '/home/torrefacto/qualitycode/src')

from codeant_agent.infrastructure.security.security_analyzer import SecurityAnalyzer


async def test_security_analyzer():
    """Prueba el SecurityAnalyzer con archivos de ejemplo."""

    # Crear directorio temporal con archivos de prueba
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creando archivos de prueba en: {temp_dir}")

        # Crear archivo Python con vulnerabilidades
        python_file = Path(temp_dir) / "test.py"
        python_file.write_text("""
# Archivo de prueba con vulnerabilidades
password = "hardcoded_password"
api_key = "sk-123456789"

def dangerous_function():
    user_input = input("Enter something: ")  # Uso de input sin validación
    result = eval(user_input)  # Uso de eval - MUY PELIGROSO
    return result

def sql_query(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"  # SQL Injection
    return query
""")

        # Crear archivo JavaScript con vulnerabilidades
        js_file = Path(temp_dir) / "test.js"
        js_file.write_text("""
function unsafeFunction() {
    const userInput = document.getElementById('input').value;
    document.getElementById('output').innerHTML = userInput; // XSS vulnerability

    // Uso de eval
    const result = eval(userInput);
    return result;
}
""")

        # Crear archivo Rust seguro
        rust_file = Path(temp_dir) / "safe.rs"
        rust_file.write_text("""
fn safe_function() {
    println!("This is safe Rust code");
}
""")

        # Ejecutar el análisis de seguridad
        analyzer = SecurityAnalyzer()
        results = analyzer.analyze_project(temp_dir)

        print("\n=== RESULTADOS DEL ANÁLISIS DE SEGURIDAD ===")
        print(f"Vulnerabilidades encontradas: {results['security_hotspots']}")
        print(f"OWASP Compliance Score: {results['owasp_compliance']}%")

        print("\n=== DETALLE POR SEVERIDAD ===")
        vulnerabilities = results['vulnerabilities']
        print(f"CRÍTICAS: {vulnerabilities['CRITICAL']}")
        print(f"ALTAS: {vulnerabilities['HIGH']}")
        print(f"MEDIAS: {vulnerabilities['MEDIUM']}")
        print(f"BAJAS: {vulnerabilities['LOW']}")

        print("\n=== DETALLES DE VULNERABILIDADES ===")
        for vuln in results.get('details', []):
            print(f"- {vuln['type']}: {vuln['description']} (Severidad: {vuln['severity']})")
            print(f"  Archivo: {vuln['file']}")
            print(f"  Coincidencias: {vuln['matches']}")
            print()

        print("✅ Test completado exitosamente!")


if __name__ == "__main__":
    asyncio.run(test_security_analyzer())
