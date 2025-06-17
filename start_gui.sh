#!/bin/bash

# AI Assistant GUI Launcher
echo "🤖 Starting AI Assistant GUI..."

# Change to the ai-assistant directory
cd /Users/danielchayes/ai-assistant

# Activate virtual environment and start GUI
source venv/bin/activate
python gui/app.py --host 0.0.0.0 --port 7860

echo "GUI started at http://localhost:7860"