#!/bin/bash

echo "🧹 Creating minimal virtual environment..."

# Navigate to project directory
cd /Users/danielchayes/ai-assistant

# Backup current venv (optional)
echo "📦 Backing up current venv..."
mv venv venv_backup

# Create new minimal venv
echo "🆕 Creating new virtual environment..."
python3 -m venv venv

# Activate new venv
source venv/bin/activate

# Install minimal requirements
echo "📥 Installing minimal dependencies..."
pip install --upgrade pip
pip install -r requirements-minimal.txt

# Check new size
echo "📊 New venv size:"
du -sh venv

echo "✨ Cleanup complete!"
echo "💡 To use: source venv/bin/activate"
echo "🗑️  Remove backup with: rm -rf venv_backup"