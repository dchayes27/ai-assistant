"""
Authentication middleware and utilities for MCP server
"""

import os
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from loguru import logger

from .models import AuthRequest, AuthResponse


class AuthConfig:
    """Authentication configuration"""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", self._generate_secret_key())
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.refresh_token_expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        self.api_key_header = "X-API-Key"
        self.enable_api_key_auth = os.getenv("ENABLE_API_KEY_AUTH", "true").lower() == "true"
        self.enable_jwt_auth = os.getenv("ENABLE_JWT_AUTH", "true").lower() == "true"
        self.api_keys = self._load_api_keys()
        
    def _generate_secret_key(self) -> str:
        """Generate a random secret key"""
        return secrets.token_urlsafe(32)
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from environment or config"""
        api_keys = {}
        
        # Load from environment variables
        for i in range(1, 11):  # Support up to 10 API keys
            key = os.getenv(f"API_KEY_{i}")
            name = os.getenv(f"API_KEY_{i}_NAME", f"key_{i}")
            permissions = os.getenv(f"API_KEY_{i}_PERMISSIONS", "read,write").split(",")
            
            if key:
                api_keys[key] = {
                    "name": name,
                    "permissions": permissions,
                    "created_at": datetime.utcnow().isoformat()
                }
        
        # Default API key for development
        if not api_keys and os.getenv("ENVIRONMENT", "development") == "development":
            default_key = "dev-key-12345"
            api_keys[default_key] = {
                "name": "development",
                "permissions": ["read", "write", "admin"],
                "created_at": datetime.utcnow().isoformat()
            }
            logger.warning(f"Using default development API key: {default_key}")
        
        return api_keys


class AuthManager:
    """Authentication manager"""
    
    def __init__(self):
        self.config = AuthConfig()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer(auto_error=False)
        
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.config.secret_key, algorithm=self.config.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.config.secret_key, algorithm=self.config.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.config.secret_key, algorithms=[self.config.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key"""
        if api_key in self.config.api_keys:
            key_info = self.config.api_keys[api_key].copy()
            key_info["key"] = api_key
            return key_info
        return None
    
    async def authenticate_request(self, request: Request, credentials: Optional[HTTPAuthorizationCredentials] = None) -> Dict[str, Any]:
        """Authenticate incoming request"""
        # Try API key authentication first
        if self.config.enable_api_key_auth:
            api_key = request.headers.get(self.config.api_key_header)
            if api_key:
                key_info = self.verify_api_key(api_key)
                if key_info:
                    logger.debug(f"Authenticated with API key: {key_info['name']}")
                    return {
                        "type": "api_key",
                        "user_id": key_info["name"],
                        "permissions": key_info["permissions"],
                        "key_info": key_info
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid API key"
                    )
        
        # Try JWT authentication
        if self.config.enable_jwt_auth and credentials:
            payload = self.verify_token(credentials.credentials)
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            logger.debug(f"Authenticated with JWT: {payload.get('sub')}")
            return {
                "type": "jwt",
                "user_id": payload.get("sub"),
                "permissions": payload.get("permissions", []),
                "payload": payload
            }
        
        # No authentication provided
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    def check_permission(self, auth_info: Dict[str, Any], required_permission: str) -> bool:
        """Check if user has required permission"""
        permissions = auth_info.get("permissions", [])
        return required_permission in permissions or "admin" in permissions


# Global auth manager instance
auth_manager = AuthManager()


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(HTTPBearer(auto_error=False))
) -> Dict[str, Any]:
    """Dependency to get current authenticated user"""
    return await auth_manager.authenticate_request(request, credentials)


async def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency to ensure user has admin permissions"""
    if not auth_manager.check_permission(current_user, "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required"
        )
    return current_user


async def get_write_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency to ensure user has write permissions"""
    if not auth_manager.check_permission(current_user, "write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write permissions required"
        )
    return current_user


async def get_read_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency to ensure user has read permissions"""
    if not auth_manager.check_permission(current_user, "read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Read permissions required"
        )
    return current_user


class AuthMiddleware:
    """Authentication middleware for FastAPI"""
    
    def __init__(self, app):
        self.app = app
        self.auth_manager = auth_manager
    
    async def __call__(self, scope, receive, send):
        """Process ASGI request through authentication middleware"""
        if scope["type"] != "http":
            # Pass through non-HTTP requests (like WebSocket)
            await self.app(scope, receive, send)
            return
        
        # Extract path from scope
        path = scope.get("path", "")
        
        # Skip authentication for health check and docs
        if path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            await self.app(scope, receive, send)
            return
        
        # Add request ID for tracking
        request_id = secrets.token_hex(8)
        
        # Add authentication info to scope
        try:
            credentials = None
            headers = dict(scope.get("headers", []))
            
            # Look for authorization header
            auth_header = headers.get(b"authorization")
            if auth_header:
                auth_str = auth_header.decode("utf-8")
                if auth_str.startswith("Bearer "):
                    credentials = HTTPAuthorizationCredentials(
                        scheme="Bearer",
                        credentials=auth_str[7:]
                    )
            
            # Create a minimal request object for authentication
            from starlette.requests import Request
            request = Request(scope, receive)
            
            auth_info = await self.auth_manager.authenticate_request(request, credentials)
            scope["auth_info"] = auth_info
            scope["authenticated"] = True
            scope["request_id"] = request_id
            
            logger.debug(f"Request {request_id} authenticated: {auth_info['user_id']}")
            
        except HTTPException as e:
            logger.warning(f"Request {request_id} authentication failed: {e.detail}")
            scope["authenticated"] = False
            scope["auth_info"] = None
            scope["request_id"] = request_id
        
        # Add request ID to response headers wrapper
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


def create_test_token(user_id: str = "test_user", permissions: list = None) -> str:
    """Create a test JWT token (for development/testing only)"""
    if permissions is None:
        permissions = ["read", "write"]
    
    data = {
        "sub": user_id,
        "permissions": permissions,
        "iat": datetime.utcnow()
    }
    
    return auth_manager.create_access_token(data)


def hash_api_key(api_key: str) -> str:
    """Hash API key for storage"""
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a new API key"""
    return f"ak_{secrets.token_urlsafe(32)}"


# Authentication utilities for external use
def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user with username/password (placeholder implementation)"""
    # This would typically check against a user database
    # For now, return None to indicate not implemented
    logger.warning("Username/password authentication not implemented")
    return None


async def login(auth_request: AuthRequest) -> AuthResponse:
    """Login endpoint handler"""
    if auth_request.api_key:
        # API key authentication
        key_info = auth_manager.verify_api_key(auth_request.api_key)
        if not key_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Create JWT token for API key
        token_data = {
            "sub": key_info["name"],
            "permissions": key_info["permissions"],
            "api_key": True
        }
        
        access_token = auth_manager.create_access_token(token_data)
        refresh_token = auth_manager.create_refresh_token(token_data)
        
        return AuthResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=auth_manager.config.access_token_expire_minutes * 60,
            user_id=key_info["name"]
        )
    
    elif auth_request.username and auth_request.password:
        # Username/password authentication
        user = authenticate_user(auth_request.username, auth_request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        token_data = {
            "sub": user["username"],
            "permissions": user.get("permissions", ["read"]),
        }
        
        access_token = auth_manager.create_access_token(token_data)
        refresh_token = auth_manager.create_refresh_token(token_data)
        
        return AuthResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=auth_manager.config.access_token_expire_minutes * 60,
            user_id=user["username"]
        )
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either API key or username/password must be provided"
        )