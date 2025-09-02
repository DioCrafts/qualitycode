"""
Entidades para detección automática de lenguajes.

Este módulo define las entidades que representan el proceso
de detección automática de lenguajes de programación.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

from ..value_objects.programming_language import ProgrammingLanguage


class DetectionStrategy(Enum):
    """Estrategias de detección de lenguajes."""
    EXTENSION = "extension"
    FILENAME = "filename"
    SHEBANG = "shebang"
    CONTENT_PATTERN = "content_pattern"
    STATISTICAL = "statistical"
    HEURISTIC = "heuristic"
    FALLBACK = "fallback"


class DetectionConfidence(Enum):
    """Niveles de confianza en la detección."""
    VERY_HIGH = "very_high"      # 95-100%
    HIGH = "high"                 # 80-94%
    MEDIUM = "medium"             # 60-79%
    LOW = "low"                   # 40-59%
    VERY_LOW = "very_low"         # 0-39%
    UNKNOWN = "unknown"           # No determinado


@dataclass
class DetectionResult:
    """Resultado de la detección de lenguaje."""
    
    language: ProgrammingLanguage
    confidence: DetectionConfidence
    confidence_score: float  # 0.0 a 1.0
    strategy: DetectionStrategy
    evidence: List[str] = field(default_factory=list)
    alternatives: List[Tuple[ProgrammingLanguage, float]] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    detection_time_ms: int = 0
    
    def __post_init__(self) -> None:
        """Validar el resultado de detección."""
        if self.language is None:
            raise ValueError("El lenguaje detectado no puede ser None")
        
        if self.confidence is None:
            raise ValueError("El nivel de confianza no puede ser None")
        
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("El score de confianza debe estar entre 0.0 y 1.0")
        
        if self.strategy is None:
            raise ValueError("La estrategia de detección no puede ser None")
        
        if self.detection_time_ms < 0:
            raise ValueError("El tiempo de detección no puede ser negativo")
    
    @property
    def is_confident(self) -> bool:
        """Verifica si la detección es confiable."""
        return self.confidence in [DetectionConfidence.VERY_HIGH, DetectionConfidence.HIGH]
    
    @property
    def is_uncertain(self) -> bool:
        """Verifica si la detección es incierta."""
        return self.confidence in [DetectionConfidence.LOW, DetectionConfidence.VERY_LOW]
    
    @property
    def has_alternatives(self) -> bool:
        """Verifica si hay alternativas de detección."""
        return len(self.alternatives) > 0
    
    def get_primary_alternative(self) -> Optional[Tuple[ProgrammingLanguage, float]]:
        """Obtiene la alternativa principal si existe."""
        if not self.alternatives:
            return None
        
        # Ordenar por score de confianza descendente
        sorted_alternatives = sorted(self.alternatives, key=lambda x: x[1], reverse=True)
        return sorted_alternatives[0]
    
    def get_confidence_description(self) -> str:
        """Obtiene una descripción legible del nivel de confianza."""
        confidence_descriptions = {
            DetectionConfidence.VERY_HIGH: "Muy alta confianza",
            DetectionConfidence.HIGH: "Alta confianza",
            DetectionConfidence.MEDIUM: "Confianza media",
            DetectionConfidence.LOW: "Baja confianza",
            DetectionConfidence.VERY_LOW: "Muy baja confianza",
            DetectionConfidence.UNKNOWN: "Confianza desconocida",
        }
        
        return confidence_descriptions.get(self.confidence, "Confianza desconocida")
    
    def get_strategy_description(self) -> str:
        """Obtiene una descripción legible de la estrategia."""
        strategy_descriptions = {
            DetectionStrategy.EXTENSION: "Detección por extensión de archivo",
            DetectionStrategy.FILENAME: "Detección por nombre de archivo",
            DetectionStrategy.SHEBANG: "Detección por shebang",
            DetectionStrategy.CONTENT_PATTERN: "Detección por patrones de contenido",
            DetectionStrategy.STATISTICAL: "Detección estadística",
            DetectionStrategy.HEURISTIC: "Detección heurística",
            DetectionStrategy.FALLBACK: "Detección por fallback",
        }
        
        return strategy_descriptions.get(self.strategy, "Estrategia desconocida")
    
    def to_dict(self) -> Dict:
        """Convierte el resultado a diccionario."""
        return {
            "language": self.language.value,
            "language_name": self.language.get_name(),
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "confidence_percentage": int(self.confidence_score * 100),
            "strategy": self.strategy.value,
            "strategy_description": self.get_strategy_description(),
            "evidence": self.evidence,
            "alternatives": [
                {
                    "language": alt[0].value,
                    "language_name": alt[0].get_name(),
                    "confidence_score": alt[1],
                    "confidence_percentage": int(alt[1] * 100)
                }
                for alt in self.alternatives
            ],
            "detected_at": self.detected_at.isoformat(),
            "detection_time_ms": self.detection_time_ms,
            "is_confident": self.is_confident,
            "is_uncertain": self.is_uncertain,
            "has_alternatives": self.has_alternatives,
        }
    
    def __str__(self) -> str:
        """Representación string del resultado."""
        confidence_pct = int(self.confidence_score * 100)
        return f"DetectionResult({self.language.get_name()}, {confidence_pct}%, {self.strategy.value})"
    
    def __repr__(self) -> str:
        """Representación de debug del resultado."""
        return (
            f"DetectionResult("
            f"language={self.language}, "
            f"confidence={self.confidence}, "
            f"confidence_score={self.confidence_score}, "
            f"strategy={self.strategy}, "
            f"evidence={self.evidence}, "
            f"alternatives={self.alternatives}"
            f")"
        )


@dataclass
class DetectionContext:
    """Contexto para la detección de lenguajes."""
    
    file_path: Optional[str] = None
    filename: Optional[str] = None
    extension: Optional[str] = None
    content_preview: Optional[str] = None
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    additional_metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validar el contexto de detección."""
        if self.file_path is None and self.filename is None:
            raise ValueError("Se debe proporcionar file_path o filename")
        
        if self.file_size_bytes is not None and self.file_size_bytes < 0:
            raise ValueError("El tamaño del archivo no puede ser negativo")
    
    def get_extension_from_filename(self) -> Optional[str]:
        """Obtiene la extensión del nombre del archivo."""
        if self.filename:
            if '.' in self.filename:
                return self.filename.split('.')[-1].lower()
        return None
    
    def get_extension_from_path(self) -> Optional[str]:
        """Obtiene la extensión de la ruta del archivo."""
        if self.file_path:
            if '.' in self.file_path:
                return self.file_path.split('.')[-1].lower()
        return None
    
    def get_effective_extension(self) -> Optional[str]:
        """Obtiene la extensión efectiva (prioriza extension explícita)."""
        if self.extension:
            return self.extension.lower()
        
        filename_ext = self.get_extension_from_filename()
        if filename_ext:
            return filename_ext
        
        return self.get_extension_from_path()
    
    def get_content_preview_lines(self, max_lines: int = 10) -> List[str]:
        """Obtiene las primeras líneas del contenido para análisis."""
        if not self.content_preview:
            return []
        
        lines = self.content_preview.split('\n')
        return lines[:max_lines]
    
    def get_first_line(self) -> Optional[str]:
        """Obtiene la primera línea del contenido."""
        if not self.content_preview:
            return None
        
        lines = self.content_preview.split('\n')
        return lines[0] if lines else None
    
    def has_shebang(self) -> bool:
        """Verifica si el contenido tiene shebang."""
        first_line = self.get_first_line()
        return first_line is not None and first_line.startswith('#!')
    
    def get_shebang_interpreter(self) -> Optional[str]:
        """Obtiene el intérprete del shebang."""
        if not self.has_shebang():
            return None
        
        first_line = self.get_first_line()
        if first_line:
            # Extraer el intérprete del shebang
            shebang_parts = first_line[2:].strip().split()
            if shebang_parts:
                interpreter = shebang_parts[0]
                # Limpiar la ruta del intérprete
                if '/' in interpreter:
                    interpreter = interpreter.split('/')[-1]
                return interpreter.lower()
        
        return None
    
    def to_dict(self) -> Dict:
        """Convierte el contexto a diccionario."""
        return {
            "file_path": self.file_path,
            "filename": self.filename,
            "extension": self.extension,
            "effective_extension": self.get_effective_extension(),
            "content_preview_lines": self.get_content_preview_lines(),
            "first_line": self.get_first_line(),
            "has_shebang": self.has_shebang(),
            "shebang_interpreter": self.get_shebang_interpreter(),
            "file_size_bytes": self.file_size_bytes,
            "mime_type": self.mime_type,
            "encoding": self.encoding,
            "additional_metadata": self.additional_metadata,
        }
    
    def __str__(self) -> str:
        """Representación string del contexto."""
        if self.filename:
            return f"DetectionContext({self.filename})"
        elif self.file_path:
            return f"DetectionContext({self.file_path})"
        else:
            return "DetectionContext(unknown)"
    
    def __repr__(self) -> str:
        """Representación de debug del contexto."""
        return (
            f"DetectionContext("
            f"file_path={self.file_path}, "
            f"filename={self.filename}, "
            f"extension={self.extension}, "
            f"content_preview_length={len(self.content_preview) if self.content_preview else 0}, "
            f"file_size_bytes={self.file_size_bytes}"
            f")"
        )
