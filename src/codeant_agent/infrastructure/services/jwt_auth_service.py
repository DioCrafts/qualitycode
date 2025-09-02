"""Implementación del servicio de autenticación con JWT."""

import re
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from codeant_agent.domain.entities.user import User
from codeant_agent.domain.services.auth_service import AuthDomainService
from codeant_agent.domain.value_objects import UserId
from codeant_agent.utils.result import Result


class JWTAuthService(AuthDomainService):
    """Implementación del servicio de autenticación usando JWT."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7
    ):
        """Inicializar el servicio de autenticación."""
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
        # Configurar context para hashing de contraseñas
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash_password(self, password: str) -> str:
        """Hashear una contraseña usando bcrypt."""
        return self.pwd_context.hash(password)

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verificar una contraseña contra su hash."""
        return self.pwd_context.verify(password, hashed_password)

    def validate_password_strength(self, password: str) -> Result[bool, str]:
        """Validar la fortaleza de una contraseña."""
        errors = []
        
        # Longitud mínima
        if len(password) < 8:
            errors.append("La contraseña debe tener al menos 8 caracteres")
        
        # Longitud máxima
        if len(password) > 128:
            errors.append("La contraseña no puede exceder 128 caracteres")
        
        # Al menos una letra minúscula
        if not re.search(r"[a-z]", password):
            errors.append("La contraseña debe contener al menos una letra minúscula")
        
        # Al menos una letra mayúscula
        if not re.search(r"[A-Z]", password):
            errors.append("La contraseña debe contener al menos una letra mayúscula")
        
        # Al menos un número
        if not re.search(r"\d", password):
            errors.append("La contraseña debe contener al menos un número")
        
        # Al menos un carácter especial
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password):
            errors.append("La contraseña debe contener al menos un carácter especial")
        
        # Verificar patrones comunes débiles
        weak_patterns = [
            r"123456",
            r"password",
            r"qwerty",
            r"admin",
            r"letmein",
            r"welcome"
        ]
        
        for pattern in weak_patterns:
            if re.search(pattern, password.lower()):
                errors.append("La contraseña contiene patrones comunes débiles")
                break
        
        if errors:
            return Result.failure("; ".join(errors))
        
        return Result.success(True)

    def generate_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Generar un JWT token para el usuario."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode = {
            "sub": str(user.id.value),
            "email": user.email.value,
            "username": user.username.value,
            "org_id": str(user.organization_id.value) if user.organization_id else None,
            "roles": user.roles,
            "permissions": user.permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "codeant-server",
            "aud": "codeant-api",
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Result[dict, str]:
        """Verificar y decodificar un JWT token."""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                audience="codeant-api",
                issuer="codeant-server"
            )
            
            # Verificar que sea un token de acceso
            if payload.get("type") != "access":
                return Result.failure("Token inválido: tipo incorrecto")
            
            # Verificar expiración
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                return Result.failure("Token expirado")
            
            return Result.success(payload)
            
        except JWTError as e:
            return Result.failure(f"Token inválido: {str(e)}")

    def generate_refresh_token(self, user_id: UserId) -> str:
        """Generar un refresh token."""
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode = {
            "sub": str(user_id.value),
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "codeant-server",
            "aud": "codeant-api",
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_refresh_token(self, token: str) -> Result[UserId, str]:
        """Verificar un refresh token."""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                audience="codeant-api",
                issuer="codeant-server"
            )
            
            # Verificar que sea un refresh token
            if payload.get("type") != "refresh":
                return Result.failure("Token inválido: tipo incorrecto")
            
            # Verificar expiración
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                return Result.failure("Refresh token expirado")
            
            user_id = payload.get("sub")
            if not user_id:
                return Result.failure("Token inválido: falta user ID")
            
            return Result.success(UserId.from_str(user_id))
            
        except JWTError as e:
            return Result.failure(f"Refresh token inválido: {str(e)}")
        except Exception as e:
            return Result.failure(f"Error al procesar refresh token: {str(e)}")

    def create_tokens(self, user: User) -> dict:
        """Crear tanto access token como refresh token para un usuario."""
        access_token = self.generate_token(user)
        refresh_token = self.generate_refresh_token(user.id)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60  # en segundos
        }
