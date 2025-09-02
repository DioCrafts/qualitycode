"""
Clasificador especializado para antipatrones de seguridad.
"""

import re
import logging
from typing import List, Dict, Any, Optional

from .base_classifier import BaseAntipatternClassifier, DetectedPattern, ClassifierError
from ....domain.entities.antipattern_analysis import (
    AntipatternType, AntipatternFeatures, SeverityIndicator, SecurityRisk
)
from ....domain.value_objects.source_position import SourcePosition

logger = logging.getLogger(__name__)


class SecurityAntipatternClassifier(BaseAntipatternClassifier):
    """Clasificador para antipatrones de seguridad."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Patrones específicos de seguridad
        self.sql_injection_patterns = [
            r'["\'].*\+.*["\'].*SELECT',
            r'SELECT.*\+.*["\']',
            r'INSERT.*\+.*VALUES',
            r'UPDATE.*SET.*\+.*["\']',
            r'DELETE.*FROM.*\+.*["\']',
            r'query\s*\(\s*["\'].*\+',
            r'execute\s*\(\s*["\'].*\+',
        ]
        
        self.hardcoded_secrets_patterns = [
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'api[_-]?key\s*=\s*["\'][A-Za-z0-9]{20,}["\']',
            r'secret[_-]?key\s*=\s*["\'][A-Za-z0-9]{16,}["\']',
            r'access[_-]?token\s*=\s*["\'][A-Za-z0-9]{20,}["\']',
            r'private[_-]?key\s*=\s*["\'][A-Za-z0-9+/]{40,}["\']',
            r'["\'][A-Za-z0-9]{32}["\']',  # Posibles MD5 hashes
            r'["\'][A-Za-z0-9]{40}["\']',  # Posibles SHA1 hashes
            r'["\']sk_[A-Za-z0-9]{24,}["\']',  # Stripe secret keys
            r'["\']AKIA[A-Z0-9]{16}["\']',  # AWS access keys
        ]
        
        self.weak_crypto_patterns = [
            r'md5\s*\(',
            r'sha1\s*\(',
            r'des\s*\(',
            r'3des\s*\(',
            r'rc4\s*\(',
            r'Random\(\)',  # Weak random
            r'Math\.random\(\)',  # JavaScript weak random
        ]
        
        self.xss_patterns = [
            r'innerHTML\s*=.*\+',
            r'document\.write\s*\(.*\+',
            r'eval\s*\(.*input',
            r'setTimeout\s*\(.*input',
            r'setInterval\s*\(.*input',
        ]
        
        self.path_traversal_patterns = [
            r'\.\./',
            r'\.\.\\\/',
            r'os\.path\.join\(.*input',
            r'open\(.*input.*\)',
            r'file\(.*input.*\)',
        ]
    
    def _initialize_feature_weights(self) -> Dict[str, float]:
        """Inicializar pesos de features de seguridad."""
        return {
            'sql_injection_indicators': 3.0,
            'hardcoded_secrets': 2.5,
            'weak_crypto': 2.0,
            'xss_vulnerabilities': 2.5,
            'path_traversal': 2.0,
            'input_validation': 1.5,
            'user_input_usage': 1.0,
            'file_operations': 1.0,
            'network_operations': 0.8,
        }
    
    async def detect_patterns(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> List[DetectedPattern]:
        """Detectar antipatrones de seguridad."""
        
        detected_patterns = []
        
        try:
            # Detectar SQL Injection
            sql_injection_pattern = await self._detect_sql_injection(features, threshold)
            if sql_injection_pattern:
                detected_patterns.append(sql_injection_pattern)
            
            # Detectar secrets hardcodeados
            secrets_pattern = await self._detect_hardcoded_secrets(features, threshold)
            if secrets_pattern:
                detected_patterns.append(secrets_pattern)
            
            # Detectar criptografía débil
            weak_crypto_pattern = await self._detect_weak_cryptography(features, threshold)
            if weak_crypto_pattern:
                detected_patterns.append(weak_crypto_pattern)
            
            # Detectar XSS vulnerabilities
            xss_pattern = await self._detect_xss_vulnerabilities(features, threshold)
            if xss_pattern:
                detected_patterns.append(xss_pattern)
            
            # Detectar path traversal
            path_traversal_pattern = await self._detect_path_traversal(features, threshold)
            if path_traversal_pattern:
                detected_patterns.append(path_traversal_pattern)
                
        except Exception as e:
            logger.error(f"Error in security pattern detection: {e}")
            raise ClassifierError(f"Security classification failed: {e}")
        
        return detected_patterns
    
    async def _detect_sql_injection(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar patrones de SQL Injection."""
        
        indicators = {}
        evidence = []
        
        # Verificar si hay operaciones SQL
        if not features.has_sql_operations:
            return None
        
        indicators['has_sql_operations'] = 1.0
        evidence.append("Code contains SQL operations")
        
        # Verificar entrada de usuario
        if features.has_user_input:
            indicators['user_input_in_sql'] = 0.8
            evidence.append("Code accepts user input")
        
        # TODO: Aquí se analizaría el código fuente real para detectar patrones específicos
        # Por ahora, usamos heurísticas basadas en las features disponibles
        
        # Simular detección de string concatenation en SQL
        if features.has_sql_operations and features.has_user_input:
            indicators['string_concatenation'] = 0.7
            evidence.append("Potential string concatenation in SQL queries")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.SQL_INJECTION,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.SQL_INJECTION),
                description="Potential SQL injection vulnerability detected",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="security_risk",
                        value=SecurityRisk.CRITICAL,
                        description="SQL injection can lead to complete database compromise"
                    ),
                    SeverityIndicator(
                        indicator_type="exploitability",
                        value="high",
                        description="Easily exploitable if user input reaches SQL queries"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_hardcoded_secrets(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar secrets hardcodeados."""
        
        indicators = {}
        evidence = []
        
        # Verificar features de seguridad personalizadas
        if features.custom_features:
            secret_patterns = features.custom_features.get('secret_patterns_found', [])
            if secret_patterns:
                indicators['hardcoded_secrets'] = min(1.0, len(secret_patterns) * 0.3)
                evidence.extend([f"Found potential secret pattern: {pattern[:20]}..." 
                               for pattern in secret_patterns[:3]])
        
        # Heurística: archivos con muchas constantes string pueden tener secrets
        if features.lines_of_code > 50:
            # Simular análisis de strings constants
            indicators['string_constants'] = 0.4
            evidence.append("File contains multiple string constants")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.HARDCODED_SECRETS,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.HARDCODED_SECRETS),
                description="Hardcoded secrets or credentials detected",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="security_risk",
                        value=SecurityRisk.HIGH,
                        description="Hardcoded secrets can be extracted from source code"
                    ),
                    SeverityIndicator(
                        indicator_type="compliance_issue",
                        value=True,
                        description="Violates security best practices and compliance requirements"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_weak_cryptography(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar uso de criptografía débil."""
        
        indicators = {}
        evidence = []
        
        # Verificar si hay operaciones criptográficas
        if not features.has_crypto_operations:
            return None
        
        indicators['has_crypto_operations'] = 0.5
        evidence.append("Code contains cryptographic operations")
        
        # Verificar features de seguridad personalizadas
        if features.custom_features:
            crypto_algorithms = features.custom_features.get('crypto_algorithms_found', [])
            if crypto_algorithms:
                # Simular detección de algoritmos débiles
                weak_algos = ['md5', 'sha1', 'des', 'rc4']
                weak_found = [algo for algo in crypto_algorithms if any(weak in algo.lower() for weak in weak_algos)]
                
                if weak_found:
                    indicators['weak_algorithms'] = min(1.0, len(weak_found) * 0.4)
                    evidence.extend([f"Uses weak algorithm: {algo}" for algo in weak_found])
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.WEAK_CRYPTOGRAPHY,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.WEAK_CRYPTOGRAPHY),
                description="Weak cryptographic practices detected",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="security_risk",
                        value=SecurityRisk.MEDIUM,
                        description="Weak cryptography can be broken by attackers"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_xss_vulnerabilities(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar vulnerabilidades XSS."""
        
        indicators = {}
        evidence = []
        
        # Solo relevante para aplicaciones web
        if features.language.value not in ['javascript', 'typescript']:
            return None
        
        # Verificar entrada de usuario
        if features.has_user_input:
            indicators['user_input'] = 0.6
            evidence.append("Code accepts user input")
        
        # Simular detección de innerHTML usage
        if features.has_user_input and features.language.value in ['javascript', 'typescript']:
            indicators['dynamic_html'] = 0.7
            evidence.append("Potential dynamic HTML manipulation with user input")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.XSS_VULNERABILITY,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.XSS_VULNERABILITY),
                description="Potential XSS vulnerability detected",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="security_risk",
                        value=SecurityRisk.HIGH,
                        description="XSS can lead to user session hijacking"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
    
    async def _detect_path_traversal(
        self, 
        features: AntipatternFeatures, 
        threshold: float
    ) -> Optional[DetectedPattern]:
        """Detectar vulnerabilidades de path traversal."""
        
        indicators = {}
        evidence = []
        
        # Verificar operaciones de archivo
        if not features.has_file_operations:
            return None
        
        indicators['file_operations'] = 0.5
        evidence.append("Code performs file operations")
        
        # Verificar entrada de usuario
        if features.has_user_input:
            indicators['user_input_files'] = 0.8
            evidence.append("User input may influence file operations")
        
        # Calcular confianza
        confidence = await self.calculate_confidence_score(features, indicators)
        
        if confidence >= threshold:
            return DetectedPattern(
                pattern_type=AntipatternType.PATH_TRAVERSAL,
                confidence=confidence,
                locations=self._identify_pattern_locations(features, AntipatternType.PATH_TRAVERSAL),
                description="Potential path traversal vulnerability detected",
                evidence=evidence,
                severity_indicators=[
                    SeverityIndicator(
                        indicator_type="security_risk",
                        value=SecurityRisk.MEDIUM,
                        description="Path traversal can allow access to unauthorized files"
                    )
                ],
                feature_importance=indicators
            )
        
        return None
