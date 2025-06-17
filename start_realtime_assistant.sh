#!/bin/bash

# Real-time Streaming Voice Assistant Launcher
# Complete startup script for the streaming voice assistant

set -e  # Exit on error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
LOG_DIR="$PROJECT_DIR/logs"
VENV_PATH="$PROJECT_DIR/venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Configuration
DEFAULT_MODE="continuous"
DEFAULT_MODEL="gpt-4o"
DEFAULT_TTS="edge"
DEFAULT_LATENCY="500"

# Parse command line arguments
MODE="$DEFAULT_MODE"
MODEL="$DEFAULT_MODEL"
TTS_PROVIDER="$DEFAULT_TTS"
TARGET_LATENCY="$DEFAULT_LATENCY"
ENABLE_WEB=false
NO_INTERRUPTIONS=false
NO_TOOLS=false
VERBOSE=false
DRY_RUN=false

show_help() {
    cat << EOF
Real-time Streaming Voice Assistant Launcher

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -m, --mode MODE         Operating mode (development/production/testing/continuous)
    --model MODEL           LLM model to use (default: gpt-4o)
    --tts PROVIDER          TTS provider (openai/edge/pyttsx3, default: edge)
    --latency MS            Target latency in milliseconds (default: 500)
    --web                   Enable web interface
    --no-interruptions      Disable interruption handling
    --no-tools              Disable tool calling
    -v, --verbose           Verbose output
    -d, --dry-run           Show what would be done without executing

EXAMPLES:
    $0                              # Start with default settings
    $0 --mode development --verbose # Development mode with verbose output
    $0 --model gpt-4o --tts openai  # Use OpenAI TTS
    $0 --latency 300 --web          # Target 300ms latency with web interface

ENVIRONMENT VARIABLES:
    OPENAI_API_KEY          OpenAI API key for LLM and TTS
    AI_ASSISTANT_LOG_LEVEL  Log level (DEBUG/INFO/WARNING/ERROR)

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --tts)
            TTS_PROVIDER="$2"
            shift 2
            ;;
        --latency)
            TARGET_LATENCY="$2"
            shift 2
            ;;
        --web)
            ENABLE_WEB=true
            shift
            ;;
        --no-interruptions)
            NO_INTERRUPTIONS=true
            shift
            ;;
        --no-tools)
            NO_TOOLS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Utility functions
execute_command() {
    local cmd="$1"
    local description="$2"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would execute: $cmd"
        return 0
    fi
    
    if [[ "$VERBOSE" == true ]]; then
        log_info "Executing: $cmd"
    fi
    
    if ! eval "$cmd"; then
        log_error "Failed to $description"
        return 1
    fi
    
    return 0
}

check_audio_devices() {
    log_info "Checking audio devices..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would check audio devices"
        return 0
    fi
    
    # Activate virtual environment for audio check
    source "$VENV_PATH/bin/activate"
    
    python3 -c "
import sys
try:
    import pyaudio
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    print(f'Found {device_count} audio devices')
    
    # List compatible devices
    compatible = 0
    for i in range(device_count):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0 or device_info['maxOutputChannels'] > 0:
            compatible += 1
            if '$VERBOSE' == 'true':
                print(f'  Device {i}: {device_info[\"name\"]} - In:{device_info[\"maxInputChannels\"]} Out:{device_info[\"maxOutputChannels\"]}')
    
    print(f'Compatible devices: {compatible}')
    p.terminate()
    
    if compatible == 0:
        print('WARNING: No compatible audio devices found')
        sys.exit(1)
        
except ImportError as e:
    print(f'ERROR: Audio libraries not available: {e}')
    sys.exit(1)
except Exception as e:
    print(f'ERROR: Audio device check failed: {e}')
    sys.exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        log_success "Audio devices check passed"
    else
        log_error "Audio devices check failed"
        return 1
    fi
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if project directory exists
    if [[ ! -d "$PROJECT_DIR" ]]; then
        log_error "Project directory not found: $PROJECT_DIR"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [[ ! -d "$VENV_PATH" ]]; then
        log_error "Virtual environment not found: $VENV_PATH"
        log_info "Run ./install_dependencies.sh first to set up the environment"
        exit 1
    fi
    
    # Check if streaming components exist
    if [[ ! -f "$PROJECT_DIR/realtime/unified_interface.py" ]]; then
        log_error "Streaming components not found. Are you on the realtime-streaming branch?"
        log_info "Switch to feature/realtime-streaming branch or merge it to main"
        exit 1
    fi
    
    # Check Python dependencies
    if ! source "$VENV_PATH/bin/activate" 2>/dev/null; then
        log_error "Failed to activate virtual environment"
        exit 1
    fi
    
    # Check required packages for streaming
    local required_packages=("openai" "webrtcvad" "pyaudio" "edge-tts")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            log_warning "Package $package not found, installing..."
            if ! pip install "$package" 2>/dev/null; then
                log_warning "Failed to install $package automatically"
            fi
        fi
    done
    
    log_success "Prerequisites check passed"
}

ensure_mcp_servers() {
    log_info "Ensuring MCP servers are running..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would check MCP servers"
        return 0
    fi
    
    # Check if MCP git server is configured
    if claude mcp list 2>/dev/null | grep -q "git:"; then
        log_success "MCP git server is configured"
    else
        log_warning "MCP git server not configured - some features may be limited"
    fi
    
    # Additional MCP servers could be started here
}

check_api_keys() {
    log_info "Checking API keys..."
    
    # Check OpenAI API key for LLM
    if [[ -z "$OPENAI_API_KEY" ]]; then
        log_warning "OPENAI_API_KEY not set - LLM features will be limited"
        
        if [[ "$MODEL" == "gpt-4o" ]]; then
            log_error "OpenAI API key required for GPT-4o model"
            echo "Please set OPENAI_API_KEY environment variable or use a different model"
            exit 1
        fi
    else
        log_success "OpenAI API key found"
    fi
    
    # Check TTS provider requirements
    if [[ "$TTS_PROVIDER" == "openai" && -z "$OPENAI_API_KEY" ]]; then
        log_warning "OpenAI API key required for OpenAI TTS, falling back to Edge TTS"
        TTS_PROVIDER="edge"
    fi
}

create_directories() {
    log_info "Creating required directories..."
    
    local directories=("$LOG_DIR" "$PROJECT_DIR/temp" "$PROJECT_DIR/run")
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            execute_command "mkdir -p '$dir'" "create directory $dir"
        fi
    done
    
    log_success "Directories created successfully"
}

start_streaming_assistant() {
    log_info "Starting real-time streaming voice assistant..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would start streaming assistant with:"
        log_info "  Mode: $MODE"
        log_info "  Model: $MODEL"
        log_info "  TTS: $TTS_PROVIDER"
        log_info "  Target Latency: ${TARGET_LATENCY}ms"
        return 0
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Build command arguments
    local cmd_args="--mode $MODE --model $MODEL --tts $TTS_PROVIDER --latency $TARGET_LATENCY"
    
    if [[ "$ENABLE_WEB" == true ]]; then
        cmd_args="$cmd_args --web"
    fi
    
    if [[ "$NO_INTERRUPTIONS" == true ]]; then
        cmd_args="$cmd_args --no-interruptions"
    fi
    
    if [[ "$NO_TOOLS" == true ]]; then
        cmd_args="$cmd_args --no-tools"
    fi
    
    if [[ "$VERBOSE" == true ]]; then
        cmd_args="$cmd_args --log-level DEBUG"
    fi
    
    # Set log file
    local log_file="$LOG_DIR/realtime_assistant_$(date +%Y%m%d_%H%M%S).log"
    
    # Start the assistant
    log_info "Command: python3 realtime/unified_interface.py $cmd_args"
    log_info "Logs: $log_file"
    
    # Execute the streaming assistant
    python3 realtime/unified_interface.py $cmd_args 2>&1 | tee "$log_file"
}

start_web_interface() {
    if [[ "$ENABLE_WEB" == true ]]; then
        log_info "Starting web interface on http://localhost:3000"
        
        if [[ "$DRY_RUN" == true ]]; then
            log_info "DRY RUN: Would start web interface"
            return 0
        fi
        
        # Start simple HTTP server for web interface
        if [[ -d "web_realtime" ]]; then
            python3 -m http.server 3000 --directory web_realtime &
            echo $! > "$PROJECT_DIR/run/web_interface.pid"
        else
            log_warning "Web interface directory not found"
        fi
    fi
}

cleanup() {
    log_info "Cleaning up..."
    
    # Kill web interface if running
    if [[ -f "$PROJECT_DIR/run/web_interface.pid" ]]; then
        local pid=$(cat "$PROJECT_DIR/run/web_interface.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
        fi
        rm -f "$PROJECT_DIR/run/web_interface.pid"
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    echo -e "${CYAN}╭─────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│        Real-time Streaming Voice Assistant         │${NC}"
    echo -e "${CYAN}╰─────────────────────────────────────────────────────╯${NC}"
    echo
    
    log_info "Starting Real-time Streaming Voice Assistant"
    log_info "Mode: $MODE | Model: $MODEL | TTS: $TTS_PROVIDER | Target: ${TARGET_LATENCY}ms"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN MODE - No actual commands will be executed"
    fi
    
    # Pre-flight checks
    check_prerequisites
    create_directories
    check_audio_devices
    ensure_mcp_servers
    check_api_keys
    
    # Start services
    start_web_interface
    start_streaming_assistant
    
    log_success "Real-time voice assistant startup completed!"
}

# Run main function
main "$@"