"""
MCP Server module for AI Assistant
Provides FastAPI-based REST API and WebSocket endpoints for agent interactions
"""

from .main import app, create_app
from .models import *
from .auth import AuthMiddleware
from .ollama_client import OllamaClient

__all__ = [
    "app",
    "create_app", 
    "AuthMiddleware",
    "OllamaClient"
]