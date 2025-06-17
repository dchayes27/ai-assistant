#!/bin/bash

echo "🧹 Safe Package Cleanup for AI Assistant"
echo "======================================="

cd /Users/danielchayes/ai-assistant
source venv/bin/activate

echo "📊 Before cleanup:"
pip list | wc -l | xargs echo "Total packages:"

echo ""
echo "🗑️  Removing safe-to-remove packages..."

# Remove development tools (if not actively developing)
pip uninstall -y black flake8 mypy pre-commit pytest pytest-asyncio ruff

# Remove multiple TTS engines (keep only TTS/Coqui)
pip uninstall -y edge-tts gtts pyttsx3

# Remove heavy ML packages not being used
pip uninstall -y accelerate optimum transformers onnxruntime

# Remove language-specific packages we don't use
pip uninstall -y sudachidict-core gruut-lang-es gruut-lang-de gruut-lang-fr

# Remove audio packages if not doing complex audio processing
pip uninstall -y librosa audioread soundfile

# Remove visualization packages if not creating plots
pip uninstall -y matplotlib

# Remove heavy math packages if not doing complex math
pip uninstall -y sympy numba llvmlite

# Remove database packages we're not using
pip uninstall -y chromadb alembic

# Remove all the PyObjC framework packages (macOS specific, often unused)
pip uninstall -y $(pip list | grep pyobjc-framework | cut -d' ' -f1 | tr '\n' ' ')

# Remove NLP packages if not doing advanced NLP
pip uninstall -y spacy nltk

# Remove computer vision if not using
pip uninstall -y pillow

echo ""
echo "✅ Cleanup complete!"
echo "📊 After cleanup:"
pip list | wc -l | xargs echo "Total packages:"

echo ""
echo "💾 To see space saved:"
echo "   du -sh venv"