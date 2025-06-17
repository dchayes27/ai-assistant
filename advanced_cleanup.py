#!/usr/bin/env python3
"""
Advanced package cleanup - finds packages with no imports
"""

import subprocess
import sys
import os
from pathlib import Path

def get_all_imports():
    """Find all import statements in the codebase"""
    imports = set()
    
    # Scan all Python files
    for py_file in Path("/Users/danielchayes/ai-assistant").rglob("*.py"):
        if "venv" in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        # Extract module name
                        if line.startswith('import '):
                            module = line.replace('import ', '').split('.')[0].split(' as ')[0].split(',')[0].strip()
                        elif line.startswith('from '):
                            module = line.replace('from ', '').split('.')[0].split(' ')[0].strip()
                        
                        if module and not module.startswith('.'):
                            imports.add(module.lower().replace('-', '_'))
        except Exception:
            continue
    
    return imports

def map_imports_to_packages():
    """Map import names to actual package names"""
    mapping = {
        'cv2': 'opencv-python',
        'pil': 'pillow', 
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
        'yaml': 'pyyaml',
        'dotenv': 'python-dotenv',
        'jwt': 'pyjwt',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'gradio': 'gradio',
        'httpx': 'httpx',
        'sqlalchemy': 'sqlalchemy',
        'whisper': 'openai-whisper',
        'torch': 'torch',
        'torchaudio': 'torchaudio',
        'transformers': 'transformers',
        'TTS': 'TTS',
        'tts': 'TTS',
        'ollama': 'ollama',
        'loguru': 'loguru',
        'pydantic': 'pydantic',
        'chromadb': 'chromadb',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'librosa': 'librosa',
        'sounddevice': 'sounddevice',
        'pyaudio': 'pyaudio',
        'gtts': 'gtts',
        'pyttsx3': 'pyttsx3',
        'edge_tts': 'edge-tts',
        'pytest': 'pytest',
        'black': 'black',
        'flake8': 'flake8',
        'mypy': 'mypy',
        'click': 'click',
        'rich': 'rich',
        'schedule': 'schedule',
        'passlib': 'passlib',
        'bcrypt': 'passlib',
        'alembic': 'alembic',
        'websockets': 'websockets',
        'mcp': 'mcp',
        'accelerate': 'accelerate',
        'optimum': 'optimum',
        'onnxruntime': 'onnxruntime',
        'numba': 'numba',
        'llvmlite': 'llvmlite',
        'sympy': 'sympy',
        'pre_commit': 'pre-commit',
        'PIL': 'pillow',
        'audioread': 'audioread',
    }
    return mapping

def main():
    print("🔍 Advanced package analysis...")
    
    # Get all imports from codebase
    print("📖 Scanning codebase for imports...")
    imports = get_all_imports()
    print(f"   Found {len(imports)} unique imports")
    
    # Map to package names
    mapping = map_imports_to_packages()
    used_packages = set()
    
    for imp in imports:
        package_name = mapping.get(imp, imp)
        used_packages.add(package_name.lower().replace('_', '-'))
    
    # Always keep essential packages
    essential = {
        'pip', 'setuptools', 'wheel', 'packaging', 'certifi', 
        'charset-normalizer', 'idna', 'requests', 'urllib3', 
        'six', 'python-dateutil', 'pytz', 'typing-extensions',
        'filelock', 'fsspec', 'markupsafe', 'jinja2', 'itsdangerous',
        'werkzeug', 'blinker', 'dnspython', 'email-validator'
    }
    used_packages.update(essential)
    
    # Get installed packages
    result = subprocess.run([
        '/Users/danielchayes/ai-assistant/venv/bin/pip', 'list'
    ], capture_output=True, text=True)
    
    installed = set()
    for line in result.stdout.strip().split('\n')[2:]:
        if line.strip():
            package = line.split()[0].lower()
            installed.add(package)
    
    print(f"📦 Total installed: {len(installed)}")
    print(f"✅ Actually used: {len(used_packages)}")
    
    # Find unused
    unused = installed - used_packages
    
    # Filter out packages that might be dependencies
    definitely_unused = set()
    for package in unused:
        # Skip if it looks like a dependency
        if any(dep in package for dep in ['dist', 'info', 'wheel', 'pip']):
            continue
        if package.startswith('_'):
            continue
        definitely_unused.add(package)
    
    print(f"🗑️  Definitely unused: {len(definitely_unused)}")
    
    if definitely_unused:
        print(f"\nUnused packages:")
        for package in sorted(definitely_unused):
            print(f"  - {package}")
        
        # Create comprehensive removal script
        with open('/Users/danielchayes/ai-assistant/remove_all_unused.sh', 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Remove all unused packages\n")
            f.write("echo '🗑️  Removing all unused packages...'\n")
            f.write("source /Users/danielchayes/ai-assistant/venv/bin/activate\n")
            
            # Remove in chunks to avoid command line length limits
            packages = list(definitely_unused)
            chunk_size = 20
            for i in range(0, len(packages), chunk_size):
                chunk = packages[i:i+chunk_size]
                f.write(f"pip uninstall -y {' '.join(chunk)}\n")
            
            f.write("echo '✅ Deep cleanup complete!'\n")
            f.write("pip list | wc -l\n")
        
        os.chmod('/Users/danielchayes/ai-assistant/remove_all_unused.sh', 0o755)
        
        print(f"\n💡 For comprehensive cleanup:")
        print(f"   /Users/danielchayes/ai-assistant/remove_all_unused.sh")
        print(f"\n💾 Potential package reduction: {len(installed)} → {len(used_packages)} packages")

if __name__ == "__main__":
    main()