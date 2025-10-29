#!/bin/bash

# AI Assistant Development Environment Setup
# For MacBook Pro M3 Max
# Run this once to set up everything

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 AI Assistant Development Setup${NC}"
echo "Hardware: MacBook Pro M3 Max"
echo "Target: <200ms voice-to-voice latency"
echo ""

# 1. Check Python version
echo -e "${BLUE}Checking Python...${NC}"
if ! python3.11 --version &> /dev/null; then
    echo -e "${YELLOW}Python 3.11 not found. Please install it first.${NC}"
    echo "brew install python@3.11"
    exit 1
fi
echo -e "${GREEN}✓ Python 3.11 found${NC}"

# 2. Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3.11 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# 3. Activate venv
source venv/bin/activate

# 4. Set M3 optimizations
echo -e "${BLUE}Setting M3 optimizations...${NC}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CFLAGS="-I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib"
echo -e "${GREEN}✓ M3 environment variables set${NC}"

# 5. Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓ Pip upgraded${NC}"

# 6. Install PyAudio separately (needs special flags)
echo -e "${BLUE}Installing PyAudio...${NC}"
if ! pip show pyaudio > /dev/null 2>&1; then
    pip install pyaudio
    echo -e "${GREEN}✓ PyAudio installed${NC}"
else
    echo -e "${GREEN}✓ PyAudio already installed${NC}"
fi

# 7. Install requirements
echo -e "${BLUE}Installing requirements...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Requirements installed${NC}"

# 8. Install pre-commit hooks
echo -e "${BLUE}Setting up pre-commit hooks...${NC}"
pip install pre-commit > /dev/null 2>&1
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
    echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
else
    echo -e "${YELLOW}⚠ No .pre-commit-config.yaml found${NC}"
fi

# 9. Create .env if not exists
if [ ! -f ".env" ]; then
    echo -e "${BLUE}Creating .env from example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠ Please edit .env with your API keys${NC}"
else
    echo -e "${GREEN}✓ .env exists${NC}"
fi

# 10. Check Ollama
echo -e "${BLUE}Checking Ollama...${NC}"
if ! ollama --version &> /dev/null; then
    echo -e "${YELLOW}⚠ Ollama not installed. Install from: https://ollama.ai${NC}"
else
    echo -e "${GREEN}✓ Ollama installed${NC}"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is running${NC}"
    else
        echo -e "${YELLOW}⚠ Ollama not running. Run: ollama serve${NC}"
    fi
fi

# 11. Create necessary directories
echo -e "${BLUE}Creating project directories...${NC}"
mkdir -p data logs test-reports realtime
echo -e "${GREEN}✓ Directories created${NC}"

# 12. Initialize database
echo -e "${BLUE}Initializing database...${NC}"
python -c "
from memory.db_manager import DatabaseManager
try:
    DatabaseManager().initialize()
    print('✓ Database initialized')
except Exception as e:
    print(f'⚠ Database initialization failed: {e}')
" 2>/dev/null || echo -e "${YELLOW}⚠ Database initialization pending${NC}"

# 13. Create VERSION file
if [ ! -f "VERSION" ]; then
    echo "0.1.0" > VERSION
    echo -e "${GREEN}✓ VERSION file created${NC}"
fi

# 14. Make scripts executable
echo -e "${BLUE}Setting script permissions...${NC}"
chmod +x run_tests.sh scripts/*.sh 2>/dev/null || true
chmod +x scripts/*.py 2>/dev/null || true
echo -e "${GREEN}✓ Scripts made executable${NC}"

# 15. Test MPS availability
echo -e "${BLUE}Testing M3 GPU (MPS)...${NC}"
python -c "
import torch
if torch.backends.mps.is_available():
    print('✓ Metal Performance Shaders available')
else:
    print('⚠ MPS not available')
" 2>/dev/null || echo -e "${YELLOW}⚠ PyTorch MPS test pending${NC}"

# 16. Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Development environment setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Start Ollama: ${YELLOW}ollama serve${NC}"
echo "2. Pull model: ${YELLOW}ollama pull mistral:7b-instruct-v0.2-q4_K_M${NC}"
echo "3. Activate venv: ${YELLOW}source venv/bin/activate${NC}"
echo "4. Run tests: ${YELLOW}./run_tests.sh --unit${NC}"
echo "5. Start development: ${YELLOW}python -m gui.app${NC}"
echo ""
echo "For streaming implementation, see: ${BLUE}docs/IMPLEMENTATION_PLAN.md${NC}"