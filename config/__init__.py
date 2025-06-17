"""
Configuration package for AI Assistant
Provides configuration loading, validation, and management
"""

from .validator import ConfigValidator, ConfigLoader, ValidationResult
from pathlib import Path
import os

# Default configuration directory
DEFAULT_CONFIG_DIR = Path(__file__).parent

# Environment detection
def get_environment() -> str:
    """Detect the current environment"""
    env = os.getenv("AI_ASSISTANT_ENV", "development")
    return env.lower()

# Global configuration loader instance
_config_loader = None

def get_config_loader(config_dir: str = None) -> ConfigLoader:
    """Get the global configuration loader instance"""
    global _config_loader
    
    if _config_loader is None:
        config_dir = config_dir or str(DEFAULT_CONFIG_DIR)
        _config_loader = ConfigLoader(config_dir)
    
    return _config_loader

def load_config(environment: str = None) -> dict:
    """Load configuration for the specified environment"""
    if environment is None:
        environment = get_environment()
    
    loader = get_config_loader()
    return loader.load_config(environment)

def validate_config(config: dict) -> ValidationResult:
    """Validate a configuration dictionary"""
    validator = ConfigValidator()
    return validator.validate_config(config)

__all__ = [
    "ConfigValidator",
    "ConfigLoader", 
    "ValidationResult",
    "get_environment",
    "get_config_loader",
    "load_config",
    "validate_config"
]