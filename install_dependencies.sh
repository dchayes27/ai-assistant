#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to handle errors
handle_error() {
    print_error "$1"
    print_error "Installation failed. Please check the error above and try again."
    exit 1
}

# Main installation script
print_status "Starting AI Assistant dependency installation..."
echo ""

# 1. Check and install Homebrew
print_status "Checking for Homebrew..."
if ! command_exists brew; then
    print_warning "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || handle_error "Failed to install Homebrew"
    
    # Add Homebrew to PATH for this session
    if [[ -d "/opt/homebrew" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -d "/usr/local/bin/brew" ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    
    print_success "Homebrew installed successfully"
else
    print_success "Homebrew is already installed"
fi

# Update Homebrew
print_status "Updating Homebrew..."
brew update || print_warning "Failed to update Homebrew, continuing anyway..."

# 2. Install Python 3.11, ffmpeg, and portaudio
print_status "Installing system dependencies via Homebrew..."

# Install Python 3.11
if ! brew list python@3.11 &>/dev/null; then
    print_status "Installing Python 3.11..."
    brew install python@3.11 || handle_error "Failed to install Python 3.11"
    print_success "Python 3.11 installed"
else
    print_success "Python 3.11 is already installed"
fi

# Install ffmpeg
if ! brew list ffmpeg &>/dev/null; then
    print_status "Installing ffmpeg..."
    brew install ffmpeg || handle_error "Failed to install ffmpeg"
    print_success "ffmpeg installed"
else
    print_success "ffmpeg is already installed"
fi

# Install portaudio
if ! brew list portaudio &>/dev/null; then
    print_status "Installing portaudio..."
    brew install portaudio || handle_error "Failed to install portaudio"
    print_success "portaudio installed"
else
    print_success "portaudio is already installed"
fi

# 3. Install or find Ollama
print_status "Checking for Ollama..."
if ! command_exists ollama; then
    # Check common locations
    OLLAMA_PATHS=(
        "$HOME/ollama/ollama"
        "$HOME/.ollama/bin/ollama"
        "/usr/local/bin/ollama"
        "/opt/homebrew/bin/ollama"
    )
    
    OLLAMA_FOUND=false
    for path in "${OLLAMA_PATHS[@]}"; do
        if [[ -x "$path" ]]; then
            print_success "Found Ollama at: $path"
            print_warning "Please add Ollama to your PATH or create a symlink"
            OLLAMA_FOUND=true
            break
        fi
    done
    
    if [[ "$OLLAMA_FOUND" == false ]]; then
        print_warning "Ollama not found. Installing Ollama..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            curl -fsSL https://ollama.com/install.sh | sh || handle_error "Failed to install Ollama"
        else
            handle_error "Please install Ollama manually from https://ollama.com"
        fi
    fi
else
    print_success "Ollama is already installed"
fi

# Check for Ollama models
print_status "Checking for Ollama models..."
if command_exists ollama; then
    MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}')
    if [[ -z "$MODELS" ]]; then
        print_warning "No Ollama models found. Downloading recommended model..."
        ollama pull llama3.2:3b || print_warning "Failed to download model, please run 'ollama pull llama3.2:3b' manually"
    else
        print_success "Found Ollama models: $(echo $MODELS | tr '\n' ' ')"
    fi
fi

# 4. Create Python virtual environment
print_status "Setting up Python virtual environment..."
cd "$(dirname "$0")" || handle_error "Failed to change to script directory"

# Use Python 3.11 specifically
PYTHON_CMD="python3.11"
if ! command_exists $PYTHON_CMD; then
    PYTHON_CMD="python3"
fi

# Create virtual environment
if [[ ! -d "venv" ]]; then
    print_status "Creating virtual environment..."
    $PYTHON_CMD -m venv venv || handle_error "Failed to create virtual environment"
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate || handle_error "Failed to activate virtual environment"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip || handle_error "Failed to upgrade pip"

# 5. Install pip requirements
print_status "Installing Python requirements..."
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt || handle_error "Failed to install Python requirements"
    print_success "Python requirements installed"
else
    handle_error "requirements.txt not found"
fi

# 6. Download Whisper medium model
print_status "Downloading Whisper medium model..."
python -c "
import whisper
import os
print('Downloading Whisper medium model...')
try:
    model = whisper.load_model('medium', download_root='models/whisper')
    print('Whisper medium model downloaded successfully')
except Exception as e:
    print(f'Warning: Failed to download Whisper model: {e}')
    print('You may need to download it manually later')
" || print_warning "Failed to download Whisper model, you may need to download it manually"

# 7. Download Coqui TTS model
print_status "Setting up TTS models..."
python -c "
try:
    from TTS.api import TTS
    import os
    
    # Create models directory
    os.makedirs('models/tts', exist_ok=True)
    
    # List available models
    print('Available TTS models:')
    models = TTS.list_models()
    
    # Download a recommended model
    print('Downloading recommended TTS model...')
    tts = TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC', progress_bar=True)
    print('TTS model downloaded successfully')
except ImportError:
    print('Note: Coqui TTS not found in requirements. Using pyttsx3 instead.')
except Exception as e:
    print(f'Warning: Failed to download TTS model: {e}')
    print('Using fallback TTS (pyttsx3)')
" || print_warning "Failed to set up advanced TTS, will use pyttsx3 as fallback"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p models/{whisper,tts,ollama} logs temp cache audio_cache conversation_history vector_store

# Final success message
echo ""
print_success "Installation completed successfully!"
echo ""
print_status "Next steps:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Start the AI assistant: python -m core.main"
echo "  3. Or launch the GUI: python -m gui.app"
echo ""
print_status "If you encountered any warnings, you may need to:"
echo "  - Install Ollama models: ollama pull llama3.2:3b"
echo "  - Configure your audio devices if using voice features"
echo ""

# Create activation helper script
cat > activate.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
echo "Virtual environment activated. You can now run:"
echo "  python -m core.main    # Start the API server"
echo "  python -m gui.app      # Launch the GUI"
EOF

chmod +x activate.sh
print_status "Created activation helper script: ./activate.sh"