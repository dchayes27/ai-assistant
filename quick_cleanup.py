#!/usr/bin/env python3
"""
Quick cleanup of obviously unused packages
"""

import subprocess
import sys

def get_installed_packages():
    """Get list of installed packages"""
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                          capture_output=True, text=True)
    packages = []
    for line in result.stdout.strip().split('\n')[2:]:
        if line.strip():
            parts = line.split()
            if len(parts) >= 2:
                packages.append(parts[0])
    return packages

def main():
    print("🔍 Finding obviously unused packages...")
    
    installed = get_installed_packages()
    print(f"📦 Total packages: {len(installed)}")
    
    # Common unused packages that often get installed as dependencies
    commonly_unused = [
        # Language packages we probably don't need
        'sudachidict-core', 'gruut-lang-es', 'gruut-lang-de', 'gruut-lang-fr',
        'mecab-python3', 'unidic-lite', 'jaconv',
        
        # Development tools if not developing
        'black', 'flake8', 'mypy', 'pre-commit', 'isort', 'bandit',
        
        # Heavy ML packages we might not use
        'onnxruntime', 'accelerate', 'optimum', 'transformers',
        
        # Multiple TTS engines (keep only TTS/Coqui)
        'edge-tts', 'gtts', 'pyttsx3',
        
        # Audio packages we might not need
        'pyaudio', 'sounddevice', 'librosa', 'audioread',
        
        # Database packages we might not use
        'chromadb', 'alembic',
        
        # Testing if not testing
        'pytest', 'pytest-asyncio',
        
        # Documentation
        'sphinx', 'sphinx-rtd-theme',
        
        # Jupyter if not using notebooks
        'jupyter', 'ipython', 'notebook',
        
        # Heavy numerical packages
        'sympy', 'numba', 'llvmlite',
        
        # Computer vision if not using
        'opencv-python', 'pillow',
        
        # Web scraping if not using
        'beautifulsoup4', 'lxml', 'selenium',
    ]
    
    # Find which ones are actually installed
    to_remove = []
    for package in commonly_unused:
        if package in [p.lower() for p in installed]:
            to_remove.append(package)
    
    print(f"\n🗑️  Found {len(to_remove)} commonly unused packages:")
    for package in to_remove:
        print(f"  - {package}")
    
    if to_remove:
        # Create removal script
        with open('/Users/danielchayes/ai-assistant/remove_unused.sh', 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Remove commonly unused packages\n")
            f.write("echo '🗑️  Removing unused packages...'\n")
            f.write(f"source /Users/danielchayes/ai-assistant/venv/bin/activate\n")
            f.write(f"pip uninstall -y {' '.join(to_remove)}\n")
            f.write("echo '✅ Cleanup complete!'\n")
        
        print(f"\n💡 To remove these packages:")
        print(f"   chmod +x /Users/danielchayes/ai-assistant/remove_unused.sh")
        print(f"   /Users/danielchayes/ai-assistant/remove_unused.sh")
        
        estimated_savings = len(to_remove) * 20  # Rough estimate
        print(f"\n💾 Estimated space savings: ~{estimated_savings}MB")
    else:
        print(f"\n✨ No obvious unused packages found!")

if __name__ == "__main__":
    main()