#!/usr/bin/env python3
"""
Clean unused packages from virtual environment
Removes packages that are installed but not actually imported/used
"""

import subprocess
import sys
import os
import ast
import re
from pathlib import Path
from typing import Set, Dict, List

def get_installed_packages() -> Dict[str, str]:
    """Get all installed packages and their versions"""
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                          capture_output=True, text=True)
    packages = {}
    
    for line in result.stdout.strip().split('\n')[2:]:  # Skip header
        if line.strip():
            parts = line.split()
            if len(parts) >= 2:
                packages[parts[0].lower().replace('-', '_')] = parts[1]
    
    return packages

def get_requirements_packages() -> Set[str]:
    """Get packages explicitly listed in requirements.txt"""
    req_packages = set()
    req_file = Path("requirements.txt")
    
    if req_file.exists():
        with open(req_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before ==, >=, etc.)
                    package = re.split(r'[=<>!]', line)[0].strip()
                    req_packages.add(package.lower().replace('-', '_'))
    
    return req_packages

def find_imported_packages(directory: str) -> Set[str]:
    """Find all packages imported in Python files"""
    imported = set()
    
    # Standard library modules to ignore
    stdlib_modules = {
        'os', 'sys', 'json', 'time', 'datetime', 'asyncio', 'threading', 
        'logging', 'pathlib', 'typing', 'collections', 'itertools', 
        'functools', 'contextlib', 'uuid', 'hashlib', 'base64', 'urllib',
        'http', 'socket', 'ssl', 'email', 'html', 'xml', 'sqlite3',
        'csv', 'configparser', 'argparse', 'subprocess', 'platform',
        'tempfile', 'shutil', 'glob', 'io', 'math', 'random', 'string',
        'enum', 'dataclasses', 'abc', 're', 'queue', 'multiprocessing'
    }
    
    for py_file in Path(directory).rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module = alias.name.split('.')[0]
                            if module not in stdlib_modules:
                                imported.add(module.lower().replace('-', '_'))
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            module = node.module.split('.')[0]
                            if module not in stdlib_modules:
                                imported.add(module.lower().replace('-', '_'))
        except Exception as e:
            print(f"Error processing {py_file}: {e}")
            continue
    
    return imported

def get_package_dependencies(package: str) -> Set[str]:
    """Get dependencies of a package"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', package], 
                              capture_output=True, text=True)
        
        dependencies = set()
        for line in result.stdout.split('\n'):
            if line.startswith('Requires:'):
                deps = line.replace('Requires:', '').strip()
                if deps:
                    for dep in deps.split(','):
                        dep = dep.strip().lower().replace('-', '_')
                        if dep:
                            dependencies.add(dep)
        
        return dependencies
    except Exception:
        return set()

def find_unused_packages():
    """Find packages that are installed but not used"""
    
    print("🔍 Analyzing package usage...")
    
    # Get all installed packages
    installed = get_installed_packages()
    print(f"📦 Total installed packages: {len(installed)}")
    
    # Get explicitly required packages
    required = get_requirements_packages()
    print(f"📋 Explicitly required packages: {len(required)}")
    
    # Find packages imported in code
    print("🔎 Scanning code for imports...")
    imported = find_imported_packages(".")
    
    # Add common package name mappings
    package_mappings = {
        'pil': 'pillow',
        'cv2': 'opencv_python',
        'sklearn': 'scikit_learn',
        'yaml': 'pyyaml',
        'dotenv': 'python_dotenv',
        'jwt': 'pyjwt',
        'bcrypt': 'passlib',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'gradio': 'gradio',
        'httpx': 'httpx',
        'sqlalchemy': 'sqlalchemy',
        'alembic': 'alembic',
        'whisper': 'openai_whisper',
        'torch': 'torch',
        'torchaudio': 'torchaudio',
        'transformers': 'transformers',
        'tts': 'tts',
        'ollama': 'ollama',
        'loguru': 'loguru',
        'pydantic': 'pydantic',
        'chromadb': 'chromadb',
        'numpy': 'numpy',
        'scipy': 'scipy',
    }
    
    # Map imported names to package names
    mapped_imports = set()
    for imp in imported:
        mapped_imports.add(package_mappings.get(imp, imp))
    
    print(f"🔗 Found imports: {len(mapped_imports)}")
    
    # Build dependency tree
    needed_packages = set()
    needed_packages.update(required)
    needed_packages.update(mapped_imports)
    
    # Add dependencies of needed packages
    print("🕸️  Building dependency tree...")
    to_check = list(needed_packages)
    checked = set()
    
    while to_check:
        package = to_check.pop()
        if package in checked:
            continue
        checked.add(package)
        
        deps = get_package_dependencies(package)
        for dep in deps:
            if dep not in needed_packages:
                needed_packages.add(dep)
                to_check.append(dep)
    
    print(f"✅ Total needed packages (including dependencies): {len(needed_packages)}")
    
    # Find unused packages
    unused = []
    essential_packages = {
        'pip', 'setuptools', 'wheel', 'packaging', 'certifi', 'charset_normalizer',
        'idna', 'requests', 'urllib3', 'six', 'python_dateutil', 'pytz'
    }
    
    for package in installed:
        if (package not in needed_packages and 
            package not in essential_packages and
            not package.startswith('_')):
            unused.append(package)
    
    return unused, needed_packages, installed

def main():
    """Main cleanup function"""
    
    # Change to project directory
    os.chdir('/Users/danielchayes/ai-assistant')
    
    print("🧹 AI Assistant Package Cleanup")
    print("=" * 40)
    
    unused, needed, installed = find_unused_packages()
    
    print(f"\n📊 Summary:")
    print(f"  Total installed: {len(installed)}")
    print(f"  Actually needed: {len(needed)}")
    print(f"  Unused packages: {len(unused)}")
    print(f"  Potential space savings: {len(unused)} packages")
    
    if unused:
        print(f"\n🗑️  Unused packages to remove:")
        for i, package in enumerate(sorted(unused)[:20]):  # Show first 20
            print(f"  {i+1:2}. {package}")
        
        if len(unused) > 20:
            print(f"  ... and {len(unused) - 20} more")
        
        print(f"\n💡 To remove unused packages:")
        print(f"   pip uninstall {' '.join(sorted(unused))}")
        
        # Create removal script
        with open('remove_unused.sh', 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Remove unused packages\n")
            f.write("echo '🗑️  Removing unused packages...'\n")
            f.write(f"pip uninstall -y {' '.join(sorted(unused))}\n")
            f.write("echo '✅ Cleanup complete!'\n")
        
        os.chmod('remove_unused.sh', 0o755)
        print(f"   OR run: ./remove_unused.sh")
    
    else:
        print(f"\n✨ No unused packages found!")

if __name__ == "__main__":
    main()