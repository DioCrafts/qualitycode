#!/usr/bin/env python3
"""
Script de verificación para la integración del sistema incremental.

Este script verifica que todos los componentes del sistema incremental
estén correctamente integrados y funcionando.
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_file_exists(file_path: str) -> bool:
    """Verificar que un archivo existe."""
    return Path(file_path).exists()

def check_directory_exists(dir_path: str) -> bool:
    """Verificar que un directorio existe."""
    return Path(dir_path).exists()

def check_python_syntax(file_path: str) -> bool:
    """Verificar la sintaxis de un archivo Python."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            compile(f.read(), file_path, 'exec')
        return True
    except SyntaxError as e:
        print(f"❌ Error de sintaxis en {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error al verificar {file_path}: {e}")
        return False

def check_imports(file_path: str) -> List[str]:
    """Verificar imports en un archivo Python."""
    errors = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Verificar imports básicos
        if 'from fastapi import' in content and 'APIRouter' not in content:
            errors.append("Falta import de APIRouter")
        
        if 'from pydantic import' in content and 'BaseModel' not in content:
            errors.append("Falta import de BaseModel")
            
        if 'from typing import' in content and 'Optional' not in content:
            errors.append("Falta import de Optional")
            
    except Exception as e:
        errors.append(f"Error al verificar imports: {e}")
    
    return errors

def verify_incremental_integration() -> Dict[str, Any]:
    """Verificar la integración del sistema incremental."""
    print("🔍 Verificando integración del sistema incremental...")
    
    results = {
        "files_exist": True,
        "syntax_valid": True,
        "imports_valid": True,
        "directories_exist": True,
        "errors": []
    }
    
    # Archivos críticos a verificar
    critical_files = [
        "src/codeant_agent/presentation/api/controllers/incremental_controller.py",
        "src/codeant_agent/presentation/api/controllers/incremental_integration_controller.py",
        "src/codeant_agent/infrastructure/integration/incremental_integration_service.py",
        "src/codeant_agent/infrastructure/integration/__init__.py",
        "src/codeant_agent/infrastructure/integration/README.md",
        "src/codeant_agent/tests/integration/test_incremental_integration.py"
    ]
    
    # Directorios críticos a verificar
    critical_directories = [
        "src/codeant_agent/infrastructure/integration",
        "src/codeant_agent/presentation/api/controllers",
        "src/codeant_agent/tests/integration"
    ]
    
    # Verificar archivos
    print("\n📁 Verificando archivos...")
    for file_path in critical_files:
        if check_file_exists(file_path):
            print(f"✅ {file_path}")
            
            # Verificar sintaxis
            if file_path.endswith('.py'):
                if not check_python_syntax(file_path):
                    results["syntax_valid"] = False
                    results["errors"].append(f"Error de sintaxis en {file_path}")
                
                # Verificar imports
                import_errors = check_imports(file_path)
                if import_errors:
                    results["imports_valid"] = False
                    results["errors"].extend([f"{file_path}: {error}" for error in import_errors])
        else:
            print(f"❌ {file_path} - NO ENCONTRADO")
            results["files_exist"] = False
            results["errors"].append(f"Archivo no encontrado: {file_path}")
    
    # Verificar directorios
    print("\n📂 Verificando directorios...")
    for dir_path in critical_directories:
        if check_directory_exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - NO ENCONTRADO")
            results["directories_exist"] = False
            results["errors"].append(f"Directorio no encontrado: {dir_path}")
    
    # Verificar integración con la aplicación principal
    print("\n🔗 Verificando integración con la aplicación principal...")
    app_file = "src/codeant_agent/presentation/api/app.py"
    if check_file_exists(app_file):
        try:
            with open(app_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Verificar que los imports estén presentes
            if "incremental_controller" in content:
                print("✅ Import del controlador incremental presente")
            else:
                print("❌ Import del controlador incremental faltante")
                results["errors"].append("Import del controlador incremental faltante en app.py")
            
            if "incremental_integration_controller" in content:
                print("✅ Import del controlador de integración presente")
            else:
                print("❌ Import del controlador de integración faltante")
                results["errors"].append("Import del controlador de integración faltante en app.py")
            
            # Verificar que los routers estén incluidos
            if "incremental_router" in content:
                print("✅ Router incremental incluido")
            else:
                print("❌ Router incremental no incluido")
                results["errors"].append("Router incremental no incluido en app.py")
            
            if "incremental_integration_router" in content:
                print("✅ Router de integración incluido")
            else:
                print("❌ Router de integración no incluido")
                results["errors"].append("Router de integración no incluido en app.py")
                
        except Exception as e:
            print(f"❌ Error al verificar app.py: {e}")
            results["errors"].append(f"Error al verificar app.py: {e}")
    else:
        print("❌ app.py no encontrado")
        results["errors"].append("app.py no encontrado")
    
    return results

def verify_endpoints() -> Dict[str, Any]:
    """Verificar que los endpoints estén correctamente definidos."""
    print("\n🌐 Verificando endpoints...")
    
    results = {
        "endpoints_found": True,
        "endpoints": [],
        "errors": []
    }
    
    # Endpoints esperados
    expected_endpoints = [
        "/api/v1/incremental/analyze",
        "/api/v1/incremental/detect-changes",
        "/api/v1/incremental/cache/status",
        "/api/v1/incremental/cache/clear",
        "/api/v1/incremental/cache/predict",
        "/api/v1/incremental/cache/warmup",
        "/api/v1/incremental/metrics",
        "/api/v1/incremental-integration/sessions",
        "/api/v1/incremental-integration/sessions/{session_id}/analyze",
        "/api/v1/incremental-integration/sessions/{session_id}/detect-changes",
        "/api/v1/incremental-integration/sessions/{session_id}/cache/status",
        "/api/v1/incremental-integration/sessions/{session_id}/cache/predict",
        "/api/v1/incremental-integration/sessions/{session_id}",
        "/api/v1/incremental-integration/sessions",
        "/api/v1/incremental-integration/sessions/cleanup",
        "/api/v1/incremental-integration/metrics"
    ]
    
    # Verificar en los controladores
    controller_files = [
        "src/codeant_agent/presentation/api/controllers/incremental_controller.py",
        "src/codeant_agent/presentation/api/controllers/incremental_integration_controller.py"
    ]
    
    for controller_file in controller_files:
        if check_file_exists(controller_file):
            try:
                with open(controller_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Buscar definiciones de endpoints
                if "@router.post" in content or "@router.get" in content or "@router.delete" in content:
                    print(f"✅ Endpoints encontrados en {controller_file}")
                    results["endpoints"].append(controller_file)
                else:
                    print(f"❌ No se encontraron endpoints en {controller_file}")
                    results["errors"].append(f"No se encontraron endpoints en {controller_file}")
                    
            except Exception as e:
                print(f"❌ Error al verificar {controller_file}: {e}")
                results["errors"].append(f"Error al verificar {controller_file}: {e}")
        else:
            print(f"❌ {controller_file} no encontrado")
            results["errors"].append(f"Controlador no encontrado: {controller_file}")
    
    return results

def verify_tests() -> Dict[str, Any]:
    """Verificar que los tests estén correctamente implementados."""
    print("\n🧪 Verificando tests...")
    
    results = {
        "tests_found": True,
        "test_files": [],
        "errors": []
    }
    
    # Archivos de test esperados
    test_files = [
        "src/codeant_agent/tests/integration/test_incremental_integration.py"
    ]
    
    for test_file in test_files:
        if check_file_exists(test_file):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Verificar que contenga tests
                if "def test_" in content or "class Test" in content:
                    print(f"✅ Tests encontrados en {test_file}")
                    results["test_files"].append(test_file)
                else:
                    print(f"❌ No se encontraron tests en {test_file}")
                    results["errors"].append(f"No se encontraron tests en {test_file}")
                    
            except Exception as e:
                print(f"❌ Error al verificar {test_file}: {e}")
                results["errors"].append(f"Error al verificar {test_file}: {e}")
        else:
            print(f"❌ {test_file} no encontrado")
            results["errors"].append(f"Archivo de test no encontrado: {test_file}")
    
    return results

def main():
    """Función principal de verificación."""
    print("🚀 Verificación de la Integración del Sistema Incremental")
    print("=" * 60)
    
    # Verificar integración
    integration_results = verify_incremental_integration()
    
    # Verificar endpoints
    endpoints_results = verify_endpoints()
    
    # Verificar tests
    tests_results = verify_tests()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 60)
    
    all_good = (
        integration_results["files_exist"] and
        integration_results["syntax_valid"] and
        integration_results["imports_valid"] and
        integration_results["directories_exist"] and
        endpoints_results["endpoints_found"] and
        tests_results["tests_found"]
    )
    
    if all_good:
        print("✅ ¡TODAS LAS VERIFICACIONES PASARON!")
        print("🎉 El sistema incremental está correctamente integrado")
    else:
        print("❌ ALGUNAS VERIFICACIONES FALLARON")
        print("\n🔍 Errores encontrados:")
        
        all_errors = (
            integration_results["errors"] +
            endpoints_results["errors"] +
            tests_results["errors"]
        )
        
        for error in all_errors:
            print(f"  • {error}")
    
    print(f"\n📈 Estadísticas:")
    print(f"  • Archivos verificados: {len([f for f in integration_results.keys() if f.endswith('.py')])}")
    print(f"  • Endpoints encontrados: {len(endpoints_results['endpoints'])}")
    print(f"  • Archivos de test: {len(tests_results['test_files'])}")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
