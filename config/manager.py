"""
Configuration Manager
Provides runtime configuration management and hot-reloading
"""

import os
import time
import threading
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .validator import ConfigLoader, ValidationResult


class ConfigChangeHandler(FileSystemEventHandler):
    """Handler for configuration file changes"""
    
    def __init__(self, manager: 'ConfigManager'):
        self.manager = manager
        self.last_modified = {}
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith(('.yaml', '.yml')):
            # Debounce rapid file changes
            current_time = time.time()
            if (event.src_path in self.last_modified and 
                current_time - self.last_modified[event.src_path] < 1.0):
                return
            
            self.last_modified[event.src_path] = current_time
            self.manager._on_config_changed(event.src_path)


class ConfigManager:
    """
    Configuration Manager with hot-reloading and runtime updates
    """
    
    def __init__(self, config_dir: str = "config", auto_reload: bool = True):
        self.config_dir = Path(config_dir)
        self.auto_reload = auto_reload
        self.config_loader = ConfigLoader(str(self.config_dir))
        
        self._config = {}
        self._environment = "development"
        self._change_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.RLock()
        
        # File watching
        self._observer = None
        if auto_reload:
            self._setup_file_watcher()
    
    def load_config(self, environment: str = None) -> Dict[str, Any]:
        """Load configuration for the specified environment"""
        with self._lock:
            if environment:
                self._environment = environment
            
            try:
                new_config = self.config_loader.load_config(self._environment)
                self._config = new_config
                self._notify_change_callbacks(new_config)
                return new_config
            except Exception as e:
                print(f"Error loading configuration: {e}")
                if not self._config:
                    raise
                return self._config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration"""
        with self._lock:
            return self._config.copy()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "llm.default_model")
            default: Default value if key is not found
        
        Returns:
            Configuration value or default
        """
        with self._lock:
            keys = key_path.split('.')
            value = self._config
            
            try:
                for key in keys:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                return default
    
    def set(self, key_path: str, value: Any, validate: bool = True) -> bool:
        """
        Set a configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to set
            value: Value to set
            validate: Whether to validate the updated configuration
        
        Returns:
            True if successful, False if validation failed
        """
        with self._lock:
            keys = key_path.split('.')
            config_copy = self._config.copy()
            
            # Navigate to the parent container
            current = config_copy
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            current[keys[-1]] = value
            
            # Validate if requested
            if validate:
                validation = self.config_loader.validator.validate_config(config_copy)
                if not validation.is_valid:
                    print(f"Configuration update failed validation: {validation.errors}")
                    return False
            
            # Update the configuration
            self._config = config_copy
            self._notify_change_callbacks(self._config)
            return True
    
    def update(self, updates: Dict[str, Any], validate: bool = True) -> bool:
        """
        Update multiple configuration values
        
        Args:
            updates: Dictionary of key_path -> value updates
            validate: Whether to validate after updates
        
        Returns:
            True if successful, False if validation failed
        """
        with self._lock:
            config_copy = self._config.copy()
            
            # Apply all updates
            for key_path, value in updates.items():
                keys = key_path.split('.')
                current = config_copy
                
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                current[keys[-1]] = value
            
            # Validate if requested
            if validate:
                validation = self.config_loader.validator.validate_config(config_copy)
                if not validation.is_valid:
                    print(f"Configuration update failed validation: {validation.errors}")
                    return False
            
            # Update the configuration
            self._config = config_copy
            self._notify_change_callbacks(self._config)
            return True
    
    def reload(self) -> bool:
        """Reload configuration from files"""
        try:
            self.load_config(self._environment)
            print(f"Configuration reloaded for environment: {self._environment}")
            return True
        except Exception as e:
            print(f"Failed to reload configuration: {e}")
            return False
    
    def save_to_file(self, filename: str = None) -> bool:
        """Save current configuration to file"""
        try:
            with self._lock:
                if filename is None:
                    filename = f"config.{self._environment}.yaml"
                
                self.config_loader.save_config(self._config, filename)
                return True
        except Exception as e:
            print(f"Failed to save configuration: {e}")
            return False
    
    def validate(self) -> ValidationResult:
        """Validate current configuration"""
        with self._lock:
            return self.config_loader.validator.validate_config(self._config)
    
    def add_change_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback to be called when configuration changes"""
        with self._lock:
            self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove a configuration change callback"""
        with self._lock:
            if callback in self._change_callbacks:
                self._change_callbacks.remove(callback)
    
    def _notify_change_callbacks(self, config: Dict[str, Any]):
        """Notify all registered callbacks of configuration changes"""
        for callback in self._change_callbacks:
            try:
                callback(config)
            except Exception as e:
                print(f"Error in configuration change callback: {e}")
    
    def _setup_file_watcher(self):
        """Set up file system watcher for auto-reload"""
        if not self.config_dir.exists():
            return
        
        self._observer = Observer()
        handler = ConfigChangeHandler(self)
        self._observer.schedule(handler, str(self.config_dir), recursive=False)
        self._observer.start()
    
    def _on_config_changed(self, file_path: str):
        """Handle configuration file change"""
        print(f"Configuration file changed: {file_path}")
        
        # Reload after a short delay to ensure file is fully written
        def delayed_reload():
            time.sleep(0.5)
            self.reload()
        
        threading.Thread(target=delayed_reload, daemon=True).start()
    
    def stop_file_watcher(self):
        """Stop the file system watcher"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
    
    def get_environment(self) -> str:
        """Get the current environment"""
        return self._environment
    
    def set_environment(self, environment: str) -> bool:
        """Change the environment and reload configuration"""
        try:
            self.load_config(environment)
            return True
        except Exception as e:
            print(f"Failed to switch to environment '{environment}': {e}")
            return False
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the configuration schema"""
        return self.config_loader.get_schema()
    
    def export_config(self, format: str = "yaml") -> str:
        """Export current configuration as string"""
        with self._lock:
            if format.lower() == "yaml":
                return yaml.dump(self._config, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                import json
                return json.dumps(self._config, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_file_watcher()


# Global configuration manager instance
_config_manager = None

def get_config_manager(config_dir: str = None, auto_reload: bool = True) -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        config_dir = config_dir or "config"
        _config_manager = ConfigManager(config_dir, auto_reload)
    
    return _config_manager

def init_config(environment: str = None, config_dir: str = None) -> Dict[str, Any]:
    """Initialize the global configuration"""
    manager = get_config_manager(config_dir)
    return manager.load_config(environment)

def get_config() -> Dict[str, Any]:
    """Get the current global configuration"""
    manager = get_config_manager()
    return manager.get_config()

def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation"""
    manager = get_config_manager()
    return manager.get(key_path, default)

def set_config_value(key_path: str, value: Any, validate: bool = True) -> bool:
    """Set a configuration value using dot notation"""
    manager = get_config_manager()
    return manager.set(key_path, value, validate)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        env = sys.argv[1]
    else:
        env = "development"
    
    # Initialize configuration manager
    with ConfigManager(auto_reload=True) as manager:
        config = manager.load_config(env)
        print(f"Loaded configuration for environment: {env}")
        
        # Add a change callback
        def on_config_change(new_config):
            print("Configuration changed!")
        
        manager.add_change_callback(on_config_change)
        
        # Keep running to demonstrate auto-reload
        try:
            print("Configuration manager is running. Modify config files to see auto-reload in action.")
            print("Press Ctrl+C to exit.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down configuration manager...")