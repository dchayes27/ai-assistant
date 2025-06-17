#!/bin/bash

# AI Assistant - Update Script
# Updates models, dependencies, and system components

set -e  # Exit on error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
VENV_PATH="$PROJECT_DIR/venv"
BACKUP_DIR="$PROJECT_DIR/backups"

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

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Configuration
UPDATE_PYTHON_DEPS=true
UPDATE_OLLAMA_MODELS=true
UPDATE_WHISPER_MODELS=false
UPDATE_TTS_MODELS=false
UPDATE_SYSTEM_PACKAGES=false
CREATE_BACKUP=true
AUTO_RESTART=false
FORCE_UPDATE=false
DRY_RUN=false
SKIP_CONFIRMATIONS=false

show_help() {
    cat << EOF
AI Assistant Update Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -a, --all               Update everything (Python deps, models, system packages)
    -p, --python-only       Update only Python dependencies
    -m, --models-only       Update only AI models
    -s, --system-only       Update only system packages
    --no-backup             Skip creating backup before update
    --auto-restart          Automatically restart services after update
    --force                 Force update even if versions are current
    --dry-run               Show what would be updated without doing it
    --skip-confirm          Skip confirmation prompts
    -v, --verbose           Verbose output

SPECIFIC UPDATES:
    --ollama-models         Update Ollama models
    --whisper-models        Update Whisper models
    --tts-models            Update TTS models
    --python-deps           Update Python dependencies

EXAMPLES:
    $0                      # Interactive update with prompts
    $0 --all                # Update everything
    $0 --python-only        # Update only Python packages
    $0 --models-only        # Update only AI models
    $0 --force --auto-restart # Force update and restart services
    $0 --dry-run            # Preview updates without executing

ENVIRONMENT VARIABLES:
    AI_ASSISTANT_NO_BACKUP  Skip backup creation
    AI_ASSISTANT_AUTO_YES   Skip all confirmations

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -a|--all)
            UPDATE_PYTHON_DEPS=true
            UPDATE_OLLAMA_MODELS=true
            UPDATE_WHISPER_MODELS=true
            UPDATE_TTS_MODELS=true
            UPDATE_SYSTEM_PACKAGES=true
            shift
            ;;
        -p|--python-only)
            UPDATE_PYTHON_DEPS=true
            UPDATE_OLLAMA_MODELS=false
            UPDATE_WHISPER_MODELS=false
            UPDATE_TTS_MODELS=false
            UPDATE_SYSTEM_PACKAGES=false
            shift
            ;;
        -m|--models-only)
            UPDATE_PYTHON_DEPS=false
            UPDATE_OLLAMA_MODELS=true
            UPDATE_WHISPER_MODELS=true
            UPDATE_TTS_MODELS=true
            UPDATE_SYSTEM_PACKAGES=false
            shift
            ;;
        -s|--system-only)
            UPDATE_PYTHON_DEPS=false
            UPDATE_OLLAMA_MODELS=false
            UPDATE_WHISPER_MODELS=false
            UPDATE_TTS_MODELS=false
            UPDATE_SYSTEM_PACKAGES=true
            shift
            ;;
        --ollama-models)
            UPDATE_OLLAMA_MODELS=true
            shift
            ;;
        --whisper-models)
            UPDATE_WHISPER_MODELS=true
            shift
            ;;
        --tts-models)
            UPDATE_TTS_MODELS=true
            shift
            ;;
        --python-deps)
            UPDATE_PYTHON_DEPS=true
            shift
            ;;
        --no-backup)
            CREATE_BACKUP=false
            shift
            ;;
        --auto-restart)
            AUTO_RESTART=true
            shift
            ;;
        --force)
            FORCE_UPDATE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-confirm)
            SKIP_CONFIRMATIONS=true
            shift
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Apply environment variables
if [[ -n "$AI_ASSISTANT_NO_BACKUP" ]]; then
    CREATE_BACKUP=false
fi

if [[ -n "$AI_ASSISTANT_AUTO_YES" ]]; then
    SKIP_CONFIRMATIONS=true
fi

# Utility functions
confirm_action() {
    local message="$1"
    
    if [[ "$SKIP_CONFIRMATIONS" == true ]]; then
        return 0
    fi
    
    echo -e "${YELLOW}$message${NC}"
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

execute_command() {
    local cmd="$1"
    local description="$2"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would execute: $cmd"
        return 0
    fi
    
    log_info "Executing: $description"
    if ! eval "$cmd"; then
        log_error "Failed to $description"
        return 1
    fi
    
    return 0
}

check_prerequisites() {
    log_step "Checking prerequisites"
    
    # Check if project directory exists
    if [[ ! -d "$PROJECT_DIR" ]]; then
        log_error "Project directory not found: $PROJECT_DIR"
        exit 1
    fi
    
    # Create required directories
    mkdir -p "$LOG_DIR" "$BACKUP_DIR"
    
    # Check if virtual environment exists
    if [[ ! -d "$VENV_PATH" ]]; then
        log_error "Virtual environment not found: $VENV_PATH"
        log_info "Run ./install_dependencies.sh first to set up the environment"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

check_if_running() {
    log_step "Checking if services are running"
    
    local status_file="$PROJECT_DIR/status.json"
    local services_running=false
    
    # Check via status file
    if [[ -f "$status_file" ]]; then
        if grep -q '"status": "running"' "$status_file" 2>/dev/null; then
            services_running=true
        fi
    fi
    
    # Check via ports
    local ports=(11434 8000 7860 8001)
    for port in "${ports[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN >/dev/null 2>&1; then
            services_running=true
            break
        fi
    done
    
    if [[ "$services_running" == true ]]; then
        log_warning "AI Assistant services appear to be running"
        
        if [[ "$AUTO_RESTART" == true ]]; then
            log_info "Auto-restart enabled, will stop and restart services"
            return 0
        fi
        
        if confirm_action "Services are running. Updates may require restart. Continue?"; then
            return 0
        else
            log_info "Update cancelled by user"
            exit 0
        fi
    fi
    
    log_success "No running services detected"
}

create_backup() {
    if [[ "$CREATE_BACKUP" != true ]]; then
        log_info "Skipping backup as requested"
        return 0
    fi
    
    log_step "Creating backup before update"
    
    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="$BACKUP_DIR/update_backup_$backup_timestamp"
    
    mkdir -p "$backup_path"
    
    # Backup critical files and directories
    local items_to_backup=(
        "config"
        "data"
        "requirements.txt"
        "status.json"
    )
    
    for item in "${items_to_backup[@]}"; do
        if [[ -e "$PROJECT_DIR/$item" ]]; then
            log_info "Backing up: $item"
            execute_command "cp -r '$PROJECT_DIR/$item' '$backup_path/'" "backup $item"
        fi
    done
    
    # Create backup manifest
    cat > "$backup_path/manifest.txt" << EOF
AI Assistant Backup
Created: $(date)
Type: Pre-update backup
Hostname: $(hostname)
User: $(whoami)
Project Directory: $PROJECT_DIR

Backed up items:
$(ls -la "$backup_path" | tail -n +2)

Update Configuration:
- Python Dependencies: $UPDATE_PYTHON_DEPS
- Ollama Models: $UPDATE_OLLAMA_MODELS
- Whisper Models: $UPDATE_WHISPER_MODELS
- TTS Models: $UPDATE_TTS_MODELS
- System Packages: $UPDATE_SYSTEM_PACKAGES
EOF
    
    log_success "Backup created: $backup_path"
    echo "$backup_path" > "$PROJECT_DIR/.last_backup"
}

update_python_dependencies() {
    if [[ "$UPDATE_PYTHON_DEPS" != true ]]; then
        return 0
    fi
    
    log_step "Updating Python dependencies"
    
    cd "$PROJECT_DIR"
    source "$VENV_PATH/bin/activate"
    
    # Check current pip version
    local current_pip_version=$(pip --version | awk '{print $2}')
    log_info "Current pip version: $current_pip_version"
    
    # Update pip first
    execute_command "pip install --upgrade pip" "update pip"
    
    # Check if requirements.txt exists
    if [[ ! -f "requirements.txt" ]]; then
        log_error "requirements.txt not found"
        return 1
    fi
    
    # Show what would be updated
    log_info "Checking for outdated packages..."
    if [[ "$DRY_RUN" != true ]]; then
        pip list --outdated --format=columns || true
    fi
    
    # Update packages
    execute_command "pip install --upgrade -r requirements.txt" "update Python dependencies"
    
    # Check for security vulnerabilities
    if command -v safety >/dev/null 2>&1; then
        log_info "Checking for security vulnerabilities..."
        execute_command "safety check" "check for security vulnerabilities"
    else
        log_info "Installing safety for security checks..."
        execute_command "pip install safety" "install safety"
        execute_command "safety check" "check for security vulnerabilities"
    fi
    
    # Generate updated requirements with versions
    log_info "Generating updated requirements..."
    execute_command "pip freeze > requirements.lock" "generate requirements.lock"
    
    log_success "Python dependencies updated successfully"
}

update_ollama_models() {
    if [[ "$UPDATE_OLLAMA_MODELS" != true ]]; then
        return 0
    fi
    
    log_step "Updating Ollama models"
    
    # Check if Ollama is available
    if ! command -v ollama >/dev/null 2>&1; then
        log_error "Ollama not found. Please install Ollama first."
        return 1
    fi
    
    # Start Ollama if not running
    local ollama_was_running=false
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        log_info "Starting Ollama for model updates..."
        execute_command "nohup ollama serve > '$LOG_DIR/ollama_update.log' 2>&1 &" "start Ollama"
        sleep 5
    else
        ollama_was_running=true
    fi
    
    # Get list of currently installed models
    log_info "Checking currently installed models..."
    local installed_models
    if [[ "$DRY_RUN" != true ]]; then
        installed_models=$(ollama list | tail -n +2 | awk '{print $1}' | grep -v "^$")
    else
        installed_models="llama3.2:latest mistral:latest"  # Mock for dry run
    fi
    
    if [[ -z "$installed_models" ]]; then
        log_warning "No Ollama models found to update"
        return 0
    fi
    
    log_info "Found models to update:"
    echo "$installed_models" | while read -r model; do
        log_info "  - $model"
    done
    
    # Update each model
    echo "$installed_models" | while read -r model; do
        if [[ -n "$model" ]]; then
            log_info "Updating model: $model"
            execute_command "ollama pull '$model'" "update model $model"
        fi
    done
    
    # Stop Ollama if we started it
    if [[ "$ollama_was_running" == false ]]; then
        log_info "Stopping Ollama..."
        pkill -f "ollama serve" 2>/dev/null || true
    fi
    
    log_success "Ollama models updated successfully"
}

update_whisper_models() {
    if [[ "$UPDATE_WHISPER_MODELS" != true ]]; then
        return 0
    fi
    
    log_step "Updating Whisper models"
    
    cd "$PROJECT_DIR"
    source "$VENV_PATH/bin/activate"
    
    # Check current Whisper version
    local whisper_version
    if python -c "import whisper; print(whisper.__version__)" 2>/dev/null; then
        whisper_version=$(python -c "import whisper; print(whisper.__version__)")
        log_info "Current Whisper version: $whisper_version"
    else
        log_error "Whisper not installed"
        return 1
    fi
    
    # Update Whisper package
    execute_command "pip install --upgrade openai-whisper" "update Whisper package"
    
    # Download/update models
    local whisper_models=("tiny" "base" "small" "medium")
    
    for model in "${whisper_models[@]}"; do
        log_info "Checking Whisper model: $model"
        
        if [[ "$DRY_RUN" != true ]]; then
            python -c "
import whisper
try:
    model = whisper.load_model('$model')
    print(f'Model $model loaded successfully')
except Exception as e:
    print(f'Error loading model $model: {e}')
    exit(1)
"
        else
            log_info "DRY RUN: Would check/download Whisper model: $model"
        fi
    done
    
    log_success "Whisper models updated successfully"
}

update_tts_models() {
    if [[ "$UPDATE_TTS_MODELS" != true ]]; then
        return 0
    fi
    
    log_step "Updating TTS models"
    
    cd "$PROJECT_DIR"
    source "$VENV_PATH/bin/activate"
    
    # Update Coqui TTS
    if python -c "import TTS" 2>/dev/null; then
        log_info "Updating Coqui TTS..."
        execute_command "pip install --upgrade TTS" "update Coqui TTS"
        
        # Update specific TTS models
        local tts_models=(
            "tts_models/en/ljspeech/tacotron2-DDC"
            "tts_models/en/ljspeech/glow-tts"
            "tts_models/en/vctk/vits"
        )
        
        for model in "${tts_models[@]}"; do
            log_info "Checking TTS model: $model"
            
            if [[ "$DRY_RUN" != true ]]; then
                python -c "
from TTS.api import TTS
try:
    tts = TTS('$model', progress_bar=True)
    print(f'Model $model loaded successfully')
except Exception as e:
    print(f'Error loading model $model: {e}')
"
            else
                log_info "DRY RUN: Would check/download TTS model: $model"
            fi
        done
    else
        log_warning "Coqui TTS not installed, skipping TTS model updates"
    fi
    
    log_success "TTS models updated successfully"
}

update_system_packages() {
    if [[ "$UPDATE_SYSTEM_PACKAGES" != true ]]; then
        return 0
    fi
    
    log_step "Updating system packages"
    
    # Detect OS and package manager
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS with Homebrew
        if command -v brew >/dev/null 2>&1; then
            log_info "Updating Homebrew packages..."
            execute_command "brew update" "update Homebrew"
            execute_command "brew upgrade" "upgrade Homebrew packages"
            execute_command "brew cleanup" "cleanup Homebrew"
        else
            log_warning "Homebrew not found, skipping macOS package updates"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt >/dev/null 2>&1; then
            log_info "Updating APT packages..."
            execute_command "sudo apt update" "update package list"
            execute_command "sudo apt upgrade -y" "upgrade packages"
            execute_command "sudo apt autoremove -y" "remove unused packages"
        elif command -v yum >/dev/null 2>&1; then
            log_info "Updating YUM packages..."
            execute_command "sudo yum update -y" "update packages"
        elif command -v dnf >/dev/null 2>&1; then
            log_info "Updating DNF packages..."
            execute_command "sudo dnf update -y" "update packages"
        else
            log_warning "No supported package manager found"
        fi
    else
        log_warning "Unsupported operating system for system package updates"
    fi
    
    log_success "System packages updated successfully"
}

update_git_repository() {
    log_step "Checking for repository updates"
    
    cd "$PROJECT_DIR"
    
    # Check if this is a git repository
    if [[ ! -d ".git" ]]; then
        log_info "Not a git repository, skipping git updates"
        return 0
    fi
    
    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        log_warning "Uncommitted changes detected in repository"
        if ! confirm_action "There are uncommitted changes. Continue with update?"; then
            log_info "Update cancelled due to uncommitted changes"
            return 1
        fi
    fi
    
    # Fetch latest changes
    execute_command "git fetch origin" "fetch latest changes"
    
    # Check if updates are available
    local current_branch=$(git branch --show-current)
    local local_commit=$(git rev-parse HEAD)
    local remote_commit=$(git rev-parse "origin/$current_branch" 2>/dev/null || echo "$local_commit")
    
    if [[ "$local_commit" != "$remote_commit" ]]; then
        log_info "Repository updates available"
        
        if confirm_action "Update repository to latest version?"; then
            execute_command "git pull origin '$current_branch'" "pull latest changes"
            log_success "Repository updated successfully"
        else
            log_info "Repository update skipped by user"
        fi
    else
        log_success "Repository is up to date"
    fi
}

restart_services() {
    if [[ "$AUTO_RESTART" != true ]]; then
        return 0
    fi
    
    log_step "Restarting AI Assistant services"
    
    # Stop services
    if [[ -f "$SCRIPT_DIR/stop_all.sh" ]]; then
        execute_command "bash '$SCRIPT_DIR/stop_all.sh' --quiet" "stop services"
    fi
    
    # Wait a moment
    sleep 3
    
    # Start services
    if [[ -f "$SCRIPT_DIR/start_all.sh" ]]; then
        execute_command "bash '$SCRIPT_DIR/start_all.sh' --skip-health" "start services"
    fi
    
    log_success "Services restarted successfully"
}

create_update_report() {
    log_step "Creating update report"
    
    local report_file="$LOG_DIR/update_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
AI Assistant Update Report
==========================

Update Date: $(date)
Hostname: $(hostname)
User: $(whoami)
Project Directory: $PROJECT_DIR

Update Configuration:
- Python Dependencies: $UPDATE_PYTHON_DEPS
- Ollama Models: $UPDATE_OLLAMA_MODELS
- Whisper Models: $UPDATE_WHISPER_MODELS
- TTS Models: $UPDATE_TTS_MODELS
- System Packages: $UPDATE_SYSTEM_PACKAGES
- Create Backup: $CREATE_BACKUP
- Auto Restart: $AUTO_RESTART
- Force Update: $FORCE_UPDATE
- Dry Run: $DRY_RUN

System Information:
- OS: $(uname -s)
- Architecture: $(uname -m)
- Kernel: $(uname -r)

EOF
    
    # Add Python environment info
    if source "$VENV_PATH/bin/activate" 2>/dev/null; then
        echo "Python Environment:" >> "$report_file"
        echo "- Python Version: $(python --version)" >> "$report_file"
        echo "- Pip Version: $(pip --version)" >> "$report_file"
        echo "- Virtual Environment: $VENV_PATH" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Add installed packages
    if [[ "$UPDATE_PYTHON_DEPS" == true ]]; then
        echo "Installed Python Packages:" >> "$report_file"
        pip list >> "$report_file" 2>/dev/null || echo "Could not list packages" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Add Ollama models
    if [[ "$UPDATE_OLLAMA_MODELS" == true ]] && command -v ollama >/dev/null 2>&1; then
        echo "Ollama Models:" >> "$report_file"
        ollama list >> "$report_file" 2>/dev/null || echo "Could not list Ollama models" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    log_success "Update report created: $report_file"
}

show_update_summary() {
    log_step "Update Summary"
    
    echo
    echo -e "${CYAN}╭─────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│                Update Completed                     │${NC}"
    echo -e "${CYAN}╰─────────────────────────────────────────────────────╯${NC}"
    echo
    
    echo -e "${GREEN}Updates Applied:${NC}"
    if [[ "$UPDATE_PYTHON_DEPS" == true ]]; then
        echo -e "  ${BLUE}✓${NC} Python dependencies updated"
    fi
    if [[ "$UPDATE_OLLAMA_MODELS" == true ]]; then
        echo -e "  ${BLUE}✓${NC} Ollama models updated"
    fi
    if [[ "$UPDATE_WHISPER_MODELS" == true ]]; then
        echo -e "  ${BLUE}✓${NC} Whisper models updated"
    fi
    if [[ "$UPDATE_TTS_MODELS" == true ]]; then
        echo -e "  ${BLUE}✓${NC} TTS models updated"
    fi
    if [[ "$UPDATE_SYSTEM_PACKAGES" == true ]]; then
        echo -e "  ${BLUE}✓${NC} System packages updated"
    fi
    echo
    
    if [[ "$CREATE_BACKUP" == true ]] && [[ -f "$PROJECT_DIR/.last_backup" ]]; then
        echo -e "${GREEN}Backup:${NC} $(cat "$PROJECT_DIR/.last_backup")"
    fi
    
    echo -e "${GREEN}Next Steps:${NC}"
    if [[ "$AUTO_RESTART" != true ]]; then
        echo -e "  ${BLUE}•${NC} Restart services: $SCRIPT_DIR/start_all.sh"
    fi
    echo -e "  ${BLUE}•${NC} Check status: $SCRIPT_DIR/monitor.sh"
    echo -e "  ${BLUE}•${NC} View logs: tail -f $LOG_DIR/*.log"
    echo
}

# Main execution
main() {
    echo -e "${CYAN}╭─────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│            AI Assistant Update Script              │${NC}"
    echo -e "${CYAN}╰─────────────────────────────────────────────────────╯${NC}"
    echo
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN MODE - No actual updates will be performed"
    fi
    
    log_info "Update configuration:"
    log_info "  Python Dependencies: $UPDATE_PYTHON_DEPS"
    log_info "  Ollama Models: $UPDATE_OLLAMA_MODELS"
    log_info "  Whisper Models: $UPDATE_WHISPER_MODELS"
    log_info "  TTS Models: $UPDATE_TTS_MODELS"
    log_info "  System Packages: $UPDATE_SYSTEM_PACKAGES"
    log_info "  Create Backup: $CREATE_BACKUP"
    log_info "  Auto Restart: $AUTO_RESTART"
    
    # Pre-flight checks
    check_prerequisites
    check_if_running
    
    # Create backup
    create_backup
    
    # Perform updates
    update_git_repository
    update_python_dependencies
    update_ollama_models
    update_whisper_models
    update_tts_models
    update_system_packages
    
    # Post-update actions
    restart_services
    create_update_report
    
    # Show summary
    show_update_summary
    
    log_success "AI Assistant update completed successfully!"
    
    return 0
}

# Error handling
trap 'log_error "Update interrupted"; exit 1' INT TERM

# Run main function
main "$@"