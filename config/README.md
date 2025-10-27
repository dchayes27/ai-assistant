# AI Assistant Configuration System

A comprehensive configuration management system for the AI Assistant project, supporting environment-specific configurations, validation, hot-reloading, and runtime updates.

## 📁 Configuration Files

### Core Configuration Files

- **`config.yaml`** - Main configuration with all default settings
- **`voice_profiles.yaml`** - TTS voice profiles and settings
- **`prompt_templates.yaml`** - System prompts and templates for different use cases

### Environment-Specific Configurations

- **`config.development.yaml`** - Development environment overrides
- **`config.production.yaml`** - Production environment settings
- **`config.testing.yaml`** - Testing environment configuration

### Python Modules

- **`validator.py`** - Configuration validation and schema enforcement
- **`manager.py`** - Runtime configuration management with hot-reloading
- **`__init__.py`** - Package initialization and convenience functions

## 🚀 Quick Start

### Basic Usage

```python
from config import load_config, get_config_value

# Load configuration for current environment
config = load_config()

# Get specific configuration values
llm_model = get_config_value("llm.default_model")
server_port = get_config_value("server.port", default=8000)
```

### Using Configuration Manager

```python
from config.manager import ConfigManager

# Initialize with auto-reload
with ConfigManager(auto_reload=True) as manager:
    config = manager.load_config("development")
    
    # Get configuration values
    model = manager.get("llm.default_model")
    
    # Update configuration at runtime
    manager.set("llm.temperature", 0.8)
    
    # Register change callbacks
    def on_config_change(new_config):
        print("Configuration updated!")
    
    manager.add_change_callback(on_config_change)
```

## 🔧 Configuration Structure

### Application Settings

```yaml
app:
  name: "AI Assistant"
  version: "1.0.0"
  debug: false  # Production default (true in development)
  log_level: "INFO"  # Production default (DEBUG in development)
  temp_dir: "temp"
  max_concurrent_requests: 10
```

### Server Configuration

```yaml
server:
  host: "localhost"
  port: 8000
  reload: false
  workers: 1
  cors_origins: ["http://localhost:3000"]
  api_prefix: "/api/v1"
  timeout: 30
```

### LLM Configuration

```yaml
llm:
  provider: "ollama"
  base_url: "http://localhost:11434"
  default_model: "llama3.2:latest"
  timeout: 30
  max_tokens: 2048
  temperature: 0.7
  top_p: 0.9
```

### Database Settings

```yaml
database:
  type: "sqlite"
  path: "data/assistant.db"
  pool_size: 5
  backup_interval: 3600
  enable_fts: true
```

## 🎤 Voice Profiles

The system includes predefined voice profiles for different use cases:

### Available Profiles

- **Sarah** - Friendly female voice for general conversation
- **Alex** - Professional male voice for presentations
- **Emma** - Warm teacher voice for educational content
- **Marcus** - Clear technical voice for debugging
- **Zoe** - Enthusiastic voice for creative sessions

### Profile Configuration

```yaml
profiles:
  sarah:
    name: "Sarah"
    provider: "coqui"
    model: "tts_models/en/ljspeech/tacotron2-DDC"
    settings:
      rate: 1.0
      pitch: 1.0
      volume: 0.8
    recommended_for: ["chat", "learning", "general"]
```

### Voice Selection by Mode

```yaml
mode_voice_mapping:
  chat: "sarah"
  project: "alex" 
  learning: "emma"
  debug: "marcus"
```

## 📝 Prompt Templates

The system includes specialized prompt templates for different conversation modes:

### System Prompts

- **default_chat** - General conversation assistant
- **project_manager** - Project planning and management
- **tutor** - Educational conversations and tutoring
- **researcher** - Research assistance and analysis
- **debugger** - Technical debugging and troubleshooting

### Specialized Templates

- **code_review** - Code analysis and feedback
- **creative_writing** - Creative writing assistance
- **technical_documentation** - Documentation creation
- **data_analysis** - Data interpretation and insights

### Response Formats

```yaml
response_formats:
  structured_list: |
    Format response as numbered list with main points and sub-points
  
  step_by_step_guide: |
    Provide detailed instructions with prerequisites and verification
```

## 🌍 Environment Management

### Environment Detection

The system automatically detects the environment from:

1. `AI_ASSISTANT_ENV` environment variable
2. Defaults to "development" if not set

### Environment-Specific Features

#### Development
- Debug mode enabled
- Auto-reload on file changes
- Relaxed security settings
- Faster/smaller models for quick testing

#### Production
- Optimized performance settings
- Enhanced security
- Comprehensive logging
- Redis caching
- SSL/HTTPS configuration

#### Testing
- In-memory database
- Mock services
- Minimal logging
- Fast execution

### Switching Environments

```bash
# Set environment variable
export AI_ASSISTANT_ENV=production

# Or programmatically
from config.manager import ConfigManager
manager = ConfigManager()
manager.set_environment("production")
```

## 🔍 Configuration Validation

### Automatic Validation

All configurations are automatically validated when loaded:

```python
from config import validate_config

config = load_config()
result = validate_config(config)

if result.is_valid:
    print("✅ Configuration is valid")
else:
    print("❌ Validation errors:", result.errors)
```

### Validation Rules

The validator checks:

- **Required fields** - Ensures essential configuration keys are present
- **Type checking** - Validates data types for all fields
- **Range validation** - Ensures numeric values are within acceptable ranges
- **URL validation** - Validates URL formats and accessibility
- **Cross-section validation** - Checks relationships between configuration sections

### Command Line Validation

```bash
# Validate a configuration file
python config/validator.py config/config.yaml

# Output:
# ✅ Configuration is valid
# ⚠️  Warnings:
#   - server.port: Port 8000 may already be in use
# 💡 Suggestions:
#   - Consider using Redis for production caching
```

## 🔄 Hot-Reloading

### Automatic File Watching

When auto-reload is enabled, the configuration manager watches for file changes:

```python
# Enable auto-reload (default)
manager = ConfigManager(auto_reload=True)

# Disable auto-reload
manager = ConfigManager(auto_reload=False)
```

### Change Callbacks

Register callbacks to respond to configuration changes:

```python
def on_model_change(config):
    new_model = config.get("llm", {}).get("default_model")
    if new_model:
        # Reload LLM with new model
        reload_llm_model(new_model)

manager.add_change_callback(on_model_change)
```

## 🌐 Environment Variables

Override configuration values using environment variables:

```bash
# Application settings
export AI_ASSISTANT_DEBUG=true
export AI_ASSISTANT_LOG_LEVEL=DEBUG

# Server settings  
export AI_ASSISTANT_PORT=8080
export AI_ASSISTANT_HOST=0.0.0.0

# Database settings
export AI_ASSISTANT_DB_PATH=/custom/path/assistant.db

# LLM settings
export AI_ASSISTANT_LLM_URL=http://custom-ollama:11434
export AI_ASSISTANT_LLM_MODEL=custom-model:latest

# Security settings
export AI_ASSISTANT_API_KEY=your-secret-api-key
export AI_ASSISTANT_JWT_SECRET=your-jwt-secret
```

## 🏗️ Configuration Schema

### Getting the Schema

```python
from config.manager import ConfigManager

manager = ConfigManager()
schema = manager.get_schema()

# Schema provides validation rules for each section
print(schema["llm"]["validators"])
```

### Adding Custom Validation

Extend the validator with custom rules:

```python
from config.validator import ConfigValidator

class CustomValidator(ConfigValidator):
    def __init__(self):
        super().__init__()
        
        # Add custom validation rules
        self.validation_rules["custom_section"] = {
            "required_fields": ["custom_field"],
            "field_types": {"custom_field": str},
            "validators": {
                "custom_field": lambda x: x.startswith("custom_")
            }
        }
```

## 🔧 Runtime Configuration Updates

### Updating Individual Values

```python
# Update a single value
manager.set("llm.temperature", 0.9)

# Update with validation
success = manager.set("server.port", 8080, validate=True)
```

### Batch Updates

```python
# Update multiple values at once
updates = {
    "llm.temperature": 0.8,
    "llm.max_tokens": 1024,
    "server.timeout": 60
}

success = manager.update(updates, validate=True)
```

### Saving Changes

```python
# Save current configuration to file
manager.save_to_file("config.custom.yaml")

# Export as string
yaml_config = manager.export_config("yaml")
json_config = manager.export_config("json")
```

## 🚨 Error Handling

### Configuration Loading Errors

```python
try:
    config = load_config("production")
except FileNotFoundError:
    print("Configuration file not found")
except ValueError as e:
    print(f"Configuration validation failed: {e}")
```

### Graceful Degradation

The system provides fallback mechanisms:

- Falls back to development config if environment-specific config fails
- Uses default values for missing configuration keys
- Continues with existing config if reload fails

## 🎯 Best Practices

### 1. Environment Separation

- Keep sensitive data (API keys, secrets) in environment variables
- Use environment-specific configs for different deployment stages
- Never commit production secrets to version control

### 2. Configuration Organization

- Group related settings in logical sections
- Use descriptive names for configuration keys
- Document complex configuration options

### 3. Validation

- Always validate configurations before deployment
- Use the provided validation rules as a starting point
- Add custom validation for domain-specific requirements

### 4. Security

- Enable authentication in production environments
- Use HTTPS in production
- Rotate secrets regularly
- Limit CORS origins to trusted domains

### 5. Performance

- Use appropriate cache settings for your environment
- Configure worker counts based on available resources
- Monitor memory usage and adjust limits accordingly

## 🔧 Troubleshooting

### Common Issues

#### Configuration Not Loading

```bash
# Check file exists and is readable
ls -la config/config.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# Check environment variable
echo $AI_ASSISTANT_ENV
```

#### Validation Failures

```python
# Get detailed validation results
result = validate_config(config)
for error in result.errors:
    print(f"Error: {error}")
for warning in result.warnings:
    print(f"Warning: {warning}")
```

#### Port Conflicts

```bash
# Check if port is in use
lsof -i :8000

# Use different port
export AI_ASSISTANT_PORT=8080
```

#### Permission Issues

```bash
# Check directory permissions
ls -la data/
mkdir -p data/
chmod 755 data/
```

### Debug Mode

Enable debug mode for verbose logging:

```bash
export AI_ASSISTANT_DEBUG=true
export AI_ASSISTANT_LOG_LEVEL=DEBUG
```

## 📚 API Reference

### ConfigManager

- `load_config(environment)` - Load configuration for environment
- `get(key_path, default)` - Get configuration value using dot notation
- `set(key_path, value, validate)` - Set configuration value
- `update(updates, validate)` - Update multiple values
- `reload()` - Reload configuration from files
- `validate()` - Validate current configuration
- `add_change_callback(callback)` - Register change callback
- `save_to_file(filename)` - Save configuration to file

### ConfigValidator

- `validate_config(config)` - Validate entire configuration
- `_validate_section(name, config, rules)` - Validate specific section
- `_validate_cross_sections(config)` - Cross-section validation

### Utility Functions

- `load_config(environment)` - Load configuration
- `get_config_value(key_path, default)` - Get configuration value
- `set_config_value(key_path, value, validate)` - Set configuration value
- `validate_config(config)` - Validate configuration

## 🤝 Contributing

When adding new configuration options:

1. Add the option to the appropriate section in `config.yaml`
2. Add validation rules in `validator.py`
3. Update environment-specific configs as needed
4. Add documentation and examples
5. Test with all environments

## 📄 License

This configuration system is part of the AI Assistant project and follows the same licensing terms.