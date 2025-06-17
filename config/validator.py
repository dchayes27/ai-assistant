"""
Configuration Validator
Validates configuration files and ensures all settings are correct
"""

import os
import yaml
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import socket
from urllib.parse import urlparse


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class ConfigValidator:
    """Configuration validation and validation rules"""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict]:
        """Initialize validation rules for different configuration sections"""
        return {
            "app": {
                "required_fields": ["name", "version"],
                "field_types": {
                    "name": str,
                    "version": str,
                    "debug": bool,
                    "log_level": str,
                    "temp_dir": str,
                    "max_concurrent_requests": int
                },
                "validators": {
                    "log_level": lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    "max_concurrent_requests": lambda x: 1 <= x <= 100,
                    "version": lambda x: re.match(r'^\d+\.\d+\.\d+$', x) is not None
                }
            },
            "server": {
                "required_fields": ["host", "port"],
                "field_types": {
                    "host": str,
                    "port": int,
                    "reload": bool,
                    "workers": int,
                    "cors_origins": list,
                    "api_prefix": str,
                    "timeout": (int, float)
                },
                "validators": {
                    "port": lambda x: 1024 <= x <= 65535,
                    "workers": lambda x: 1 <= x <= 16,
                    "timeout": lambda x: 1 <= x <= 300,
                    "api_prefix": lambda x: x.startswith("/")
                }
            },
            "database": {
                "required_fields": ["type", "path"],
                "field_types": {
                    "type": str,
                    "path": str,
                    "pool_size": int,
                    "pool_recycle": int,
                    "backup_interval": int,
                    "backup_retention": int,
                    "enable_fts": bool
                },
                "validators": {
                    "type": lambda x: x in ["sqlite", "postgresql", "mysql"],
                    "pool_size": lambda x: 1 <= x <= 20,
                    "backup_retention": lambda x: 1 <= x <= 365
                }
            },
            "llm": {
                "required_fields": ["provider", "default_model"],
                "field_types": {
                    "provider": str,
                    "base_url": str,
                    "default_model": str,
                    "timeout": (int, float),
                    "max_tokens": int,
                    "temperature": (int, float),
                    "top_p": (int, float),
                    "stream": bool,
                    "retry_attempts": int,
                    "retry_delay": (int, float)
                },
                "validators": {
                    "provider": lambda x: x in ["ollama", "openai", "anthropic", "huggingface"],
                    "base_url": self._validate_url,
                    "temperature": lambda x: 0.0 <= x <= 2.0,
                    "top_p": lambda x: 0.0 <= x <= 1.0,
                    "max_tokens": lambda x: 1 <= x <= 32768,
                    "retry_attempts": lambda x: 0 <= x <= 10
                }
            },
            "tts": {
                "required_fields": ["provider"],
                "field_types": {
                    "provider": str,
                    "fallback_provider": str,
                    "output_format": str,
                    "sample_rate": int
                },
                "validators": {
                    "provider": lambda x: x in ["coqui", "pyttsx3", "azure", "google"],
                    "output_format": lambda x: x in ["wav", "mp3", "ogg"],
                    "sample_rate": lambda x: x in [8000, 16000, 22050, 44100, 48000]
                }
            },
            "stt": {
                "required_fields": ["provider", "model"],
                "field_types": {
                    "provider": str,
                    "model": str,
                    "language": str,
                    "temperature": (int, float),
                    "device": str
                },
                "validators": {
                    "provider": lambda x: x in ["whisper", "azure", "google", "amazon"],
                    "device": lambda x: x in ["cpu", "cuda", "auto"],
                    "temperature": lambda x: 0.0 <= x <= 1.0
                }
            },
            "security": {
                "required_fields": [],
                "field_types": {
                    "enable_auth": bool,
                    "api_key": (str, type(None)),
                    "jwt_secret": str,
                    "jwt_expiry": int,
                    "rate_limit": int,
                    "cors_enabled": bool,
                    "https_only": bool
                },
                "validators": {
                    "jwt_expiry": lambda x: 60 <= x <= 86400,  # 1 minute to 1 day
                    "rate_limit": lambda x: 1 <= x <= 10000,
                    "jwt_secret": lambda x: len(x) >= 32 if x else True
                }
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate entire configuration"""
        result = ValidationResult(is_valid=True)
        
        # Validate each section
        for section_name, section_config in config.items():
            if section_name in self.validation_rules:
                section_result = self._validate_section(
                    section_name, section_config, self.validation_rules[section_name]
                )
                result.errors.extend(section_result.errors)
                result.warnings.extend(section_result.warnings)
                result.suggestions.extend(section_result.suggestions)
                
                if not section_result.is_valid:
                    result.is_valid = False
        
        # Cross-section validations
        cross_validation = self._validate_cross_sections(config)
        result.errors.extend(cross_validation.errors)
        result.warnings.extend(cross_validation.warnings)
        result.suggestions.extend(cross_validation.suggestions)
        
        if not cross_validation.is_valid:
            result.is_valid = False
        
        return result
    
    def _validate_section(self, section_name: str, section_config: Dict[str, Any], 
                         rules: Dict[str, Any]) -> ValidationResult:
        """Validate a single configuration section"""
        result = ValidationResult(is_valid=True)
        
        # Check required fields
        for required_field in rules.get("required_fields", []):
            if required_field not in section_config:
                result.errors.append(
                    f"{section_name}.{required_field}: Required field is missing"
                )
                result.is_valid = False
        
        # Check field types and validators
        field_types = rules.get("field_types", {})
        validators = rules.get("validators", {})
        
        for field_name, field_value in section_config.items():
            # Type validation
            if field_name in field_types:
                expected_type = field_types[field_name]
                if not self._check_type(field_value, expected_type):
                    result.errors.append(
                        f"{section_name}.{field_name}: Expected type {expected_type}, "
                        f"got {type(field_value).__name__}"
                    )
                    result.is_valid = False
            
            # Custom validators
            if field_name in validators:
                try:
                    if not validators[field_name](field_value):
                        result.errors.append(
                            f"{section_name}.{field_name}: Value '{field_value}' "
                            f"failed validation"
                        )
                        result.is_valid = False
                except Exception as e:
                    result.errors.append(
                        f"{section_name}.{field_name}: Validation error - {str(e)}"
                    )
                    result.is_valid = False
        
        return result
    
    def _validate_cross_sections(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate relationships between different configuration sections"""
        result = ValidationResult(is_valid=True)
        
        # Check if server port is available
        if "server" in config and "port" in config["server"]:
            port = config["server"]["port"]
            if not self._is_port_available(port):
                result.warnings.append(
                    f"server.port: Port {port} may already be in use"
                )
        
        # Check database path accessibility
        if "database" in config and "path" in config["database"]:
            db_path = config["database"]["path"]
            if not self._check_database_path(db_path):
                result.warnings.append(
                    f"database.path: Database directory '{os.path.dirname(db_path)}' "
                    f"may not be writable"
                )
        
        # Check LLM and voice model compatibility
        if "llm" in config and "tts" in config:
            llm_provider = config["llm"].get("provider")
            tts_provider = config["tts"].get("provider")
            
            if llm_provider == "ollama" and tts_provider not in ["coqui", "pyttsx3"]:
                result.suggestions.append(
                    "For ollama LLM provider, coqui or pyttsx3 TTS providers "
                    "are recommended for better local integration"
                )
        
        # Check memory settings consistency
        if "memory" in config:
            memory_config = config["memory"]
            context_limit = memory_config.get("conversation_context_limit", 20)
            summarization_threshold = memory_config.get("summarization_threshold", 50)
            
            if context_limit >= summarization_threshold:
                result.warnings.append(
                    "memory: conversation_context_limit should be less than "
                    "summarization_threshold for effective memory management"
                )
        
        return result
    
    def _check_type(self, value: Any, expected_type: Union[type, tuple]) -> bool:
        """Check if value matches expected type(s)"""
        if isinstance(expected_type, tuple):
            return any(isinstance(value, t) for t in expected_type)
        return isinstance(value, expected_type)
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    def _is_port_available(self, port: int) -> bool:
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def _check_database_path(self, db_path: str) -> bool:
        """Check if database path is accessible"""
        try:
            db_dir = os.path.dirname(db_path)
            if not db_dir:
                return True  # Relative path
            
            if not os.path.exists(db_dir):
                # Try to create directory
                os.makedirs(db_dir, exist_ok=True)
            
            return os.access(db_dir, os.W_OK)
        except Exception:
            return False


class ConfigLoader:
    """Configuration loader with validation and environment support"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.validator = ConfigValidator()
    
    def load_config(self, environment: str = "development") -> Dict[str, Any]:
        """Load configuration with environment-specific overrides"""
        # Load base configuration
        config = self._load_yaml_file("config.yaml")
        
        # Load environment-specific overrides
        env_config_file = f"config.{environment}.yaml"
        if (self.config_dir / env_config_file).exists():
            env_config = self._load_yaml_file(env_config_file)
            config = self._merge_configs(config, env_config)
        
        # Load voice profiles
        if (self.config_dir / "voice_profiles.yaml").exists():
            config["voice_profiles"] = self._load_yaml_file("voice_profiles.yaml")
        
        # Load prompt templates
        if (self.config_dir / "prompt_templates.yaml").exists():
            config["prompt_templates"] = self._load_yaml_file("prompt_templates.yaml")
        
        # Apply environment variables
        config = self._apply_env_variables(config)
        
        # Validate configuration
        validation_result = self.validator.validate_config(config)
        if not validation_result.is_valid:
            raise ValueError(f"Configuration validation failed: {validation_result.errors}")
        
        # Log warnings and suggestions
        if validation_result.warnings:
            print("Configuration warnings:")
            for warning in validation_result.warnings:
                print(f"  - {warning}")
        
        if validation_result.suggestions:
            print("Configuration suggestions:")
            for suggestion in validation_result.suggestions:
                print(f"  - {suggestion}")
        
        return config
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        file_path = self.config_dir / filename
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {file_path}: {e}")
    
    def _merge_configs(self, base_config: Dict[str, Any], 
                      override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries, with override taking precedence"""
        result = base_config.copy()
        
        for key, value in override_config.items():
            if (key in result and isinstance(result[key], dict) 
                and isinstance(value, dict)):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        env_mappings = {
            "AI_ASSISTANT_DEBUG": ("app", "debug", bool),
            "AI_ASSISTANT_LOG_LEVEL": ("app", "log_level", str),
            "AI_ASSISTANT_PORT": ("server", "port", int),
            "AI_ASSISTANT_HOST": ("server", "host", str),
            "AI_ASSISTANT_DB_PATH": ("database", "path", str),
            "AI_ASSISTANT_LLM_URL": ("llm", "base_url", str),
            "AI_ASSISTANT_LLM_MODEL": ("llm", "default_model", str),
            "AI_ASSISTANT_API_KEY": ("security", "api_key", str),
            "AI_ASSISTANT_JWT_SECRET": ("security", "jwt_secret", str),
            "AI_ASSISTANT_TTS_PROVIDER": ("tts", "provider", str),
            "AI_ASSISTANT_STT_PROVIDER": ("stt", "provider", str),
        }
        
        for env_var, (section, key, value_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                if section not in config:
                    config[section] = {}
                
                # Convert to appropriate type
                try:
                    if value_type == bool:
                        config[section][key] = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type == int:
                        config[section][key] = int(env_value)
                    elif value_type == float:
                        config[section][key] = float(env_value)
                    else:
                        config[section][key] = env_value
                except ValueError:
                    print(f"Warning: Invalid value for {env_var}: {env_value}")
        
        return config
    
    def save_config(self, config: Dict[str, Any], filename: str = "config.yaml"):
        """Save configuration to file"""
        file_path = self.config_dir / filename
        
        # Validate before saving
        validation_result = self.validator.validate_config(config)
        if not validation_result.is_valid:
            raise ValueError(f"Cannot save invalid configuration: {validation_result.errors}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get configuration schema for validation"""
        return self.validator.validation_rules


def validate_config_file(config_file: str) -> ValidationResult:
    """Validate a configuration file"""
    config_dir = os.path.dirname(config_file) or "."
    config_filename = os.path.basename(config_file)
    
    loader = ConfigLoader(config_dir)
    
    try:
        config = loader._load_yaml_file(config_filename)
        return loader.validator.validate_config(config)
    except Exception as e:
        result = ValidationResult(is_valid=False)
        result.errors.append(f"Failed to load configuration: {str(e)}")
        return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python validator.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    result = validate_config_file(config_file)
    
    if result.is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed")
        for error in result.errors:
            print(f"  Error: {error}")
    
    if result.warnings:
        print("\n⚠️  Warnings:")
        for warning in result.warnings:
            print(f"  {warning}")
    
    if result.suggestions:
        print("\n💡 Suggestions:")
        for suggestion in result.suggestions:
            print(f"  {suggestion}")
    
    sys.exit(0 if result.is_valid else 1)