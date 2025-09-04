"""
Módulo de infraestructura para análisis de código muerto.

Este módulo proporciona las implementaciones concretas de los
repositorios definidos en el dominio para análisis de código muerto.
"""

from .dead_code_repository_impl import DeadCodeRepositoryImpl

__all__ = ['DeadCodeRepositoryImpl']
