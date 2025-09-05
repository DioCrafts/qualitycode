#!/usr/bin/env python3
"""
Script de prueba para verificar el análisis AST con Tree-sitter.
"""
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

    TREE_SITTER_AVAILABLE = True
except ImportError:
    print("❌ Tree-sitter no está disponible")
    TREE_SITTER_AVAILABLE = False

if TREE_SITTER_AVAILABLE:
    print("🎉 Tree-sitter está disponible! Probando análisis AST real...")
else:
    print("⚠️  Tree-sitter no disponible. Usando análisis básico.")
    print("Para habilitar análisis AST real, reconstruye el contenedor Docker:")
    print("docker-compose build --no-cache backend")
    sys.exit(1)

async def test_tree_sitter_analysis():
    """Prueba el análisis con Tree-sitter."""
    print("=== Probando análisis AST con Tree-sitter ===\n")

    # Código de prueba con elementos muertos
    test_code_python = '''
import os
import sys
import json  # Import no utilizado
from typing import List, Dict  # Dict no es utilizado

# Variable global no utilizada
UNUSED_CONSTANT = 42

def main():
    """Función principal."""
    used_var = 10
    unused_var = 20  # Variable no utilizada

    result = calculate(used_var)
    print(f"Result: {result}")

    # Más variables no utilizadas
    temp = 100
    another_unused = "hello"

    return result

def calculate(value):
    """Función que calcula algo."""
    return value * 2

def unused_function():
    """Esta función nunca es llamada."""
    return "Never called"

def helper_function(param1, param2, unused_param):
    """Función con parámetro no utilizado."""
    return param1 + param2

class UsedClass:
    """Clase que sí es utilizada."""
    def __init__(self):
        self.value = 10

class UnusedClass:
    """Esta clase nunca es instanciada."""
    def __init__(self):
        self.data = []

    def method(self):
        return "unused"

# Código principal
if __name__ == "__main__":
    obj = UsedClass()
    main()
'''

    # Crear analizador para Python
    analyzer = TreeSitterAnalyzer("python")

    # Parsear el código
    if analyzer.parse(test_code_python):
        print("✅ Código parseado exitosamente\n")

        # Analizar código muerto
        dead_code = analyzer.analyze()

        # Mostrar resultados
        print("📊 Resultados del análisis:\n")

        # Variables no utilizadas
        unused_vars = dead_code.get("unused_variables", [])
        print(f"🔸 Variables no utilizadas: {len(unused_vars)}")
        for var in unused_vars:
            print(f"   - {var['name']} (línea {var['line']})")

        # Funciones no utilizadas
        unused_funcs = dead_code.get("unused_functions", [])
        print(f"\n🔸 Funciones no utilizadas: {len(unused_funcs)}")
        for func in unused_funcs:
            print(f"   - {func['name']} (línea {func['line']})")

        # Clases no utilizadas
        unused_classes = dead_code.get("unused_classes", [])
        print(f"\n🔸 Clases no utilizadas: {len(unused_classes)}")
        for cls in unused_classes:
            print(f"   - {cls['name']} (línea {cls['line']})")

        # Imports no utilizados
        unused_imports = dead_code.get("unused_imports", [])
        print(f"\n🔸 Imports no utilizados: {len(unused_imports)}")
        for imp in unused_imports:
            print(f"   - {imp['name']} (línea {imp['line']})")

        # Parámetros no utilizados
        unused_params = dead_code.get("unused_parameters", [])
        print(f"\n🔸 Parámetros no utilizados: {len(unused_params)}")
        for param in unused_params:
            print(f"   - {param['name']} (línea {param['line']})")

        print("\n✅ Análisis completado exitosamente!")

    else:
        print("❌ Error al parsear el código")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_tree_sitter_analysis())
    
    return result

def calculate(value):
    """Función que calcula algo."""
    return value * 2

def unused_function():
    """Esta función nunca es llamada."""
    return "Never called"

def helper_function(param1, param2, unused_param):
    """Función con parámetro no utilizado."""
    return param1 + param2

class UsedClass:
    """Clase que sí es utilizada."""
    def __init__(self):
        self.value = 10

class UnusedClass:
    """Esta clase nunca es instanciada."""
    def __init__(self):
        self.data = []
    
    def method(self):
        return "unused"

# Código principal
if __name__ == "__main__":
    obj = UsedClass()
    main()
'''

async def test_tree_sitter_analysis():
    """Prueba el análisis con Tree-sitter."""
    print("=== Probando análisis AST con Tree-sitter ===\n")
    
    # Crear analizador para Python
    analyzer = TreeSitterAnalyzer("python")
    
    # Parsear el código
    if analyzer.parse(test_code_python):
        print("✅ Código parseado exitosamente\n")
        
        # Analizar código muerto
        dead_code = analyzer.analyze()
        
        # Mostrar resultados
        print("📊 Resultados del análisis:\n")
        
        # Variables no utilizadas
        unused_vars = dead_code.get("unused_variables", [])
        print(f"🔸 Variables no utilizadas: {len(unused_vars)}")
        for var in unused_vars:
            print(f"   - {var['name']} (línea {var['line']})")
        
        # Funciones no utilizadas
        unused_funcs = dead_code.get("unused_functions", [])
        print(f"\n🔸 Funciones no utilizadas: {len(unused_funcs)}")
        for func in unused_funcs:
            print(f"   - {func['name']} (línea {func['line']})")
        
        # Clases no utilizadas
        unused_classes = dead_code.get("unused_classes", [])
        print(f"\n🔸 Clases no utilizadas: {len(unused_classes)}")
        for cls in unused_classes:
            print(f"   - {cls['name']} (línea {cls['line']})")
        
        # Imports no utilizados
        unused_imports = dead_code.get("unused_imports", [])
        print(f"\n🔸 Imports no utilizados: {len(unused_imports)}")
        for imp in unused_imports:
            print(f"   - {imp['name']} (línea {imp['line']})")
        
        # Parámetros no utilizados
        unused_params = dead_code.get("unused_parameters", [])
        print(f"\n🔸 Parámetros no utilizados: {len(unused_params)}")
        for param in unused_params:
            print(f"   - {param['name']} (línea {param['line']})")
        
        print("\n✅ Análisis completado exitosamente!")
        
    else:
        print("❌ Error al parsear el código")

if __name__ == "__main__":
    asyncio.run(test_tree_sitter_analysis())
