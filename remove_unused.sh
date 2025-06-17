#!/bin/bash
# Remove commonly unused packages
echo '🗑️  Removing unused packages...'
source /Users/danielchayes/ai-assistant/venv/bin/activate
pip uninstall -y sudachidict-core black flake8 mypy onnxruntime accelerate optimum transformers edge-tts gtts pyttsx3 pyaudio sounddevice librosa audioread chromadb alembic pytest pytest-asyncio sympy numba llvmlite pillow
echo '✅ Cleanup complete!'
