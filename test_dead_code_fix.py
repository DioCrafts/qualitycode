#!/usr/bin/env python3
"""Script para verificar que el error DeadCodeResult está corregido."""

import sys
import os

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Intentar importar la clase DeadCodeResult
    from codeant_agent.infrastructure.dead_code.advanced_dead_code_engine import DeadCodeResult
    
    # Intentar crear una instancia con los parámetros correctos
    result = DeadCodeResult(
        file_path="test.py",
        symbol_name="test_function",
        symbol_type="function",
        line_number=10,
        confidence=0.95,
        severity="high",
        reason="Sin uso detectado",
        suggested_action="Eliminar con seguridad",
        safe_to_delete=True,
        dependencies=[],
        used_in_tests=False,
        potentially_dynamic=False
    )
    
    print("✅ ÉXITO: DeadCodeResult funciona correctamente")
    print(f"   - file_path: {result.file_path}")
    print(f"   - symbol_name: {result.symbol_name}")
    print(f"   - confidence: {result.confidence}")
    print(f"   - safe_to_delete: {result.safe_to_delete}")
    
except TypeError as e:
    print(f"❌ ERROR: {e}")
    print("   El error de 'symbol_id' aún existe")
    
except Exception as e:
    print(f"❌ ERROR inesperado: {type(e).__name__}: {e}")
