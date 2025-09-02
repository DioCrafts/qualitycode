"""
Módulo que define la aplicación FastAPI para el sistema de reglas en lenguaje natural.
"""
import logging
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from codeant_agent.presentation.api.controllers.natural_rule_controller import router as natural_rule_router
from codeant_agent.presentation.api.controllers.learning_controller import router as learning_router


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Crea y configura la aplicación FastAPI.
    
    Returns:
        Aplicación FastAPI configurada
    """
    app = FastAPI(
        title="CodeAnt Natural Rules API",
        description="API para el sistema de reglas en lenguaje natural",
        version="1.0.0",
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En producción, especificar orígenes permitidos
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Registrar rutas
    app.include_router(natural_rule_router)
    app.include_router(learning_router)
    
    @app.get("/", tags=["health"])
    async def health_check() -> Dict[str, str]:
        """Endpoint para verificar el estado de la API.
        
        Returns:
            Diccionario con el estado de la API
        """
        return {"status": "ok", "version": "1.0.0"}
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)