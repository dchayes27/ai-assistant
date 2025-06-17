#!/bin/bash

# AI Assistant - Start All Services
# Launches all components with comprehensive health checks

set -e  # Exit on error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="$PROJECT_DIR/run"
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

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Configuration
DEFAULT_ENVIRONMENT="development"
ENVIRONMENT="${AI_ASSISTANT_ENV:-$DEFAULT_ENVIRONMENT}"
CONFIG_FILE="$PROJECT_DIR/config/config.$ENVIRONMENT.yaml"

# Service configuration
OLLAMA_PORT=11434
API_PORT=8000
GUI_PORT=7860
METRICS_PORT=8001

# Timeout settings
STARTUP_TIMEOUT=60
HEALTH_CHECK_TIMEOUT=30
RETRY_ATTEMPTS=3
RETRY_DELAY=5

# Parse command line arguments
FORCE_RESTART=false
SKIP_HEALTH_CHECKS=false
VERBOSE=false
DRY_RUN=false

show_help() {
    cat << EOF
AI Assistant Startup Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -f, --force             Force restart of running services
    -s, --skip-health       Skip health checks
    -v, --verbose           Verbose output
    -d, --dry-run           Show what would be done without executing
    -e, --env ENV           Set environment (development/production/testing)

EXAMPLES:
    $0                      # Start all services normally
    $0 --force --verbose    # Force restart with verbose output
    $0 --env production     # Start in production mode
    $0 --dry-run            # Preview actions without executing

ENVIRONMENT VARIABLES:
    AI_ASSISTANT_ENV        Environment to use (development/production/testing)
    AI_ASSISTANT_LOG_LEVEL  Log level (DEBUG/INFO/WARNING/ERROR)
    AI_ASSISTANT_NO_BROWSER Skip opening browser automatically

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force)
            FORCE_RESTART=true
            shift
            ;;
        -s|--skip-health)
            SKIP_HEALTH_CHECKS=true
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
        -e|--env)
            ENVIRONMENT="$2"
            CONFIG_FILE="$PROJECT_DIR/config/config.$ENVIRONMENT.yaml"
            shift 2
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

create_directories() {
    log_step "Creating required directories"
    
    local directories=("$LOG_DIR" "$PID_DIR" "$PROJECT_DIR/data" "$PROJECT_DIR/temp" "$PROJECT_DIR/backups")
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            execute_command "mkdir -p '$dir'" "create directory $dir"
        fi
    done
    
    log_success "Directories created successfully"
}

check_prerequisites() {
    log_step "Checking prerequisites"
    
    # Check if project directory exists
    if [[ ! -d "$PROJECT_DIR" ]]; then
        log_error "Project directory not found: $PROJECT_DIR"
        exit 1
    fi
    
    # Check if configuration file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        log_info "Available environments:"
        ls -1 "$PROJECT_DIR/config/config."*.yaml 2>/dev/null | sed 's/.*config\.\(.*\)\.yaml/  - \1/' || echo "  No configuration files found"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [[ ! -d "$VENV_PATH" ]]; then
        log_error "Virtual environment not found: $VENV_PATH"
        log_info "Run ./install_dependencies.sh first to set up the environment"
        exit 1
    fi
    
    # Check Python dependencies
    if ! source "$VENV_PATH/bin/activate" 2>/dev/null; then
        log_error "Failed to activate virtual environment"
        exit 1
    fi
    
    # Check required Python packages
    local required_packages=("fastapi" "uvicorn" "gradio" "sqlalchemy" "pydantic")
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            log_error "Required package not found: $package"
            log_info "Run ./install_dependencies.sh to install missing packages"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

check_port_availability() {
    local port=$1
    local service_name=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        if [[ "$FORCE_RESTART" == true ]]; then
            log_warning "Port $port is in use by $service_name, will attempt to stop existing service"
            return 1
        else
            log_error "Port $port is already in use by $service_name"
            log_info "Use --force to restart existing services"
            return 1
        fi
    fi
    return 0
}

stop_existing_services() {
    log_step "Stopping existing services if running"
    
    # Stop services gracefully first
    if [[ -f "$SCRIPT_DIR/stop_all.sh" ]]; then
        execute_command "bash '$SCRIPT_DIR/stop_all.sh' --quiet" "stop existing services"
    fi
    
    # Kill any remaining processes on our ports
    local ports=($OLLAMA_PORT $API_PORT $GUI_PORT $METRICS_PORT)
    for port in "${ports[@]}"; do
        local pids=$(lsof -ti:$port 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            log_warning "Killing processes on port $port: $pids"
            execute_command "kill -TERM $pids" "terminate processes on port $port"
            sleep 2
            
            # Force kill if still running
            local remaining_pids=$(lsof -ti:$port 2>/dev/null || true)
            if [[ -n "$remaining_pids" ]]; then
                execute_command "kill -KILL $remaining_pids" "force kill processes on port $port"
            fi
        fi
    done
    
    log_success "Existing services stopped"
}

wait_for_service() {
    local service_name=$1
    local port=$2
    local timeout=${3:-$HEALTH_CHECK_TIMEOUT}
    local endpoint=${4:-""}
    
    log_info "Waiting for $service_name to start (port $port, timeout ${timeout}s)"
    
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))
    
    while [[ $(date +%s) -lt $end_time ]]; do
        if [[ -n "$endpoint" ]]; then
            # Check HTTP endpoint
            if curl -s -f "http://localhost:$port$endpoint" >/dev/null 2>&1; then
                log_success "$service_name is ready"
                return 0
            fi
        else
            # Check port availability
            if netstat -an 2>/dev/null | grep -q ":$port.*LISTEN" || \
               lsof -Pi :$port -sTCP:LISTEN >/dev/null 2>&1; then
                log_success "$service_name is ready"
                return 0
            fi
        fi
        
        sleep 1
        if [[ "$VERBOSE" == true ]]; then
            echo -n "."
        fi
    done
    
    log_error "$service_name failed to start within ${timeout}s"
    return 1
}

start_ollama() {
    log_step "Starting Ollama service"
    
    # Check if Ollama is already running
    if curl -s http://localhost:$OLLAMA_PORT/api/tags >/dev/null 2>&1; then
        log_success "Ollama is already running"
        return 0
    fi
    
    # Start Ollama
    if command -v ollama >/dev/null 2>&1; then
        execute_command "nohup ollama serve > '$LOG_DIR/ollama.log' 2>&1 &" "start Ollama"
        echo $! > "$PID_DIR/ollama.pid"
        
        if ! wait_for_service "Ollama" $OLLAMA_PORT 60 "/api/tags"; then
            log_error "Failed to start Ollama service"
            return 1
        fi
        
        log_success "Ollama service started successfully"
    else
        log_error "Ollama not found. Please install Ollama first."
        return 1
    fi
}

pull_default_models() {
    log_step "Ensuring default models are available"
    
    # Read default model from config
    local default_model
    if command -v python >/dev/null 2>&1; then
        default_model=$(python -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print(config.get('llm', {}).get('default_model', 'llama3.2:latest'))
except:
    print('llama3.2:latest')
" 2>/dev/null || echo "llama3.2:latest")
    else
        default_model="llama3.2:latest"
    fi
    
    log_info "Checking for model: $default_model"
    
    # Check if model exists
    if ! ollama list | grep -q "$default_model"; then
        log_info "Pulling model: $default_model (this may take a while)"
        execute_command "ollama pull '$default_model'" "pull default model"
    else
        log_success "Model $default_model is already available"
    fi
}

start_api_server() {
    log_step "Starting AI Assistant API server"
    
    cd "$PROJECT_DIR"
    source "$VENV_PATH/bin/activate"
    
    # Start the MCP server
    execute_command "nohup python -m uvicorn mcp_server.main:app --host 0.0.0.0 --port $API_PORT --env-file .env > '$LOG_DIR/api.log' 2>&1 &" "start API server"
    echo $! > "$PID_DIR/api.pid"
    
    if ! wait_for_service "API Server" $API_PORT 30 "/health"; then
        log_error "Failed to start API server"
        return 1
    fi
    
    log_success "API server started successfully"
}

start_gui() {
    log_step "Starting AI Assistant GUI"
    
    cd "$PROJECT_DIR"
    source "$VENV_PATH/bin/activate"
    
    # Start Gradio interface
    execute_command "nohup python gui/interface.py --port $GUI_PORT > '$LOG_DIR/gui.log' 2>&1 &" "start GUI"
    echo $! > "$PID_DIR/gui.pid"
    
    if ! wait_for_service "GUI" $GUI_PORT 30; then
        log_error "Failed to start GUI"
        return 1
    fi
    
    log_success "GUI started successfully"
}

start_monitoring() {
    log_step "Starting monitoring services"
    
    # Start metrics server if enabled
    if grep -q "enable_metrics.*true" "$CONFIG_FILE" 2>/dev/null; then
        cd "$PROJECT_DIR"
        source "$VENV_PATH/bin/activate"
        
        execute_command "nohup python -m prometheus_client --port $METRICS_PORT > '$LOG_DIR/metrics.log' 2>&1 &" "start metrics server"
        echo $! > "$PID_DIR/metrics.pid"
        
        if wait_for_service "Metrics Server" $METRICS_PORT 10; then
            log_success "Monitoring services started"
        else
            log_warning "Metrics server failed to start (optional service)"
        fi
    else
        log_info "Monitoring disabled in configuration"
    fi
}

run_health_checks() {
    if [[ "$SKIP_HEALTH_CHECKS" == true ]]; then
        log_info "Skipping health checks as requested"
        return 0
    fi
    
    log_step "Running comprehensive health checks"
    
    local health_check_failed=false
    
    # Check Ollama health
    log_info "Checking Ollama health..."
    if curl -s -f http://localhost:$OLLAMA_PORT/api/tags >/dev/null; then
        log_success "✓ Ollama service is healthy"
    else
        log_error "✗ Ollama service health check failed"
        health_check_failed=true
    fi
    
    # Check API server health
    log_info "Checking API server health..."
    if curl -s -f http://localhost:$API_PORT/health >/dev/null; then
        log_success "✓ API server is healthy"
    else
        log_error "✗ API server health check failed"
        health_check_failed=true
    fi
    
    # Check GUI health
    log_info "Checking GUI health..."
    if curl -s -f http://localhost:$GUI_PORT >/dev/null; then
        log_success "✓ GUI is healthy"
    else
        log_error "✗ GUI health check failed"
        health_check_failed=true
    fi
    
    # Check database connectivity
    log_info "Checking database connectivity..."
    cd "$PROJECT_DIR"
    source "$VENV_PATH/bin/activate"
    if python -c "
from memory.db_manager import DatabaseManager
import asyncio
async def test_db():
    async with DatabaseManager('data/assistant.db') as db:
        await db.initialize()
        print('Database connection successful')
asyncio.run(test_db())
" 2>/dev/null; then
        log_success "✓ Database is accessible"
    else
        log_error "✗ Database connectivity check failed"
        health_check_failed=true
    fi
    
    # Check disk space
    log_info "Checking disk space..."
    local available_space=$(df "$PROJECT_DIR" | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    if [[ $available_gb -lt 1 ]]; then
        log_warning "⚠ Low disk space: ${available_gb}GB available"
    else
        log_success "✓ Disk space: ${available_gb}GB available"
    fi
    
    # Check memory usage
    log_info "Checking memory usage..."
    local memory_usage=$(free | awk '/^Mem:/{printf "%.1f", $3/$2 * 100.0}')
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        log_warning "⚠ High memory usage: ${memory_usage}%"
    else
        log_success "✓ Memory usage: ${memory_usage}%"
    fi
    
    if [[ "$health_check_failed" == true ]]; then
        log_error "Some health checks failed"
        return 1
    else
        log_success "All health checks passed"
        return 0
    fi
}

create_status_file() {
    log_step "Creating status file"
    
    local status_file="$PROJECT_DIR/status.json"
    
    cat > "$status_file" << EOF
{
    "status": "running",
    "environment": "$ENVIRONMENT",
    "started_at": "$(date -Iseconds)",
    "services": {
        "ollama": {
            "port": $OLLAMA_PORT,
            "pid_file": "$PID_DIR/ollama.pid",
            "log_file": "$LOG_DIR/ollama.log",
            "url": "http://localhost:$OLLAMA_PORT"
        },
        "api": {
            "port": $API_PORT,
            "pid_file": "$PID_DIR/api.pid",
            "log_file": "$LOG_DIR/api.log",
            "url": "http://localhost:$API_PORT"
        },
        "gui": {
            "port": $GUI_PORT,
            "pid_file": "$PID_DIR/gui.pid",
            "log_file": "$LOG_DIR/gui.log",
            "url": "http://localhost:$GUI_PORT"
        },
        "metrics": {
            "port": $METRICS_PORT,
            "pid_file": "$PID_DIR/metrics.pid",
            "log_file": "$LOG_DIR/metrics.log",
            "url": "http://localhost:$METRICS_PORT"
        }
    },
    "config_file": "$CONFIG_FILE",
    "project_dir": "$PROJECT_DIR"
}
EOF
    
    log_success "Status file created: $status_file"
}

show_startup_summary() {
    log_step "Startup Summary"
    
    echo
    echo -e "${CYAN}╭─────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│                AI Assistant Started                 │${NC}"
    echo -e "${CYAN}╰─────────────────────────────────────────────────────╯${NC}"
    echo
    echo -e "${GREEN}Environment:${NC} $ENVIRONMENT"
    echo -e "${GREEN}Project Directory:${NC} $PROJECT_DIR"
    echo -e "${GREEN}Configuration:${NC} $CONFIG_FILE"
    echo
    echo -e "${GREEN}Services:${NC}"
    echo -e "  ${BLUE}•${NC} Ollama Service:    http://localhost:$OLLAMA_PORT"
    echo -e "  ${BLUE}•${NC} API Server:       http://localhost:$API_PORT"
    echo -e "  ${BLUE}•${NC} GUI Interface:    http://localhost:$GUI_PORT"
    echo -e "  ${BLUE}•${NC} Metrics (if enabled): http://localhost:$METRICS_PORT"
    echo
    echo -e "${GREEN}Logs:${NC}"
    echo -e "  ${BLUE}•${NC} Ollama:  $LOG_DIR/ollama.log"
    echo -e "  ${BLUE}•${NC} API:     $LOG_DIR/api.log"
    echo -e "  ${BLUE}•${NC} GUI:     $LOG_DIR/gui.log"
    echo
    echo -e "${GREEN}Management:${NC}"
    echo -e "  ${BLUE}•${NC} Stop services:    $SCRIPT_DIR/stop_all.sh"
    echo -e "  ${BLUE}•${NC} Monitor status:   $SCRIPT_DIR/monitor.sh"
    echo -e "  ${BLUE}•${NC} View logs:        tail -f $LOG_DIR/*.log"
    echo
    
    # Open browser if not disabled
    if [[ -z "$AI_ASSISTANT_NO_BROWSER" && "$ENVIRONMENT" == "development" ]]; then
        log_info "Opening GUI in browser..."
        if command -v open >/dev/null 2>&1; then
            execute_command "open http://localhost:$GUI_PORT" "open browser"
        elif command -v xdg-open >/dev/null 2>&1; then
            execute_command "xdg-open http://localhost:$GUI_PORT" "open browser"
        fi
    fi
}

# Main execution
main() {
    echo -e "${CYAN}╭─────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│            AI Assistant Startup Script             │${NC}"
    echo -e "${CYAN}╰─────────────────────────────────────────────────────╯${NC}"
    echo
    
    log_info "Starting AI Assistant in $ENVIRONMENT environment"
    log_info "Project directory: $PROJECT_DIR"
    log_info "Configuration: $CONFIG_FILE"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN MODE - No actual commands will be executed"
    fi
    
    # Pre-flight checks
    create_directories
    check_prerequisites
    
    # Stop existing services if force restart
    if [[ "$FORCE_RESTART" == true ]]; then
        stop_existing_services
    else
        # Check port availability
        local ports_in_use=false
        if ! check_port_availability $OLLAMA_PORT "Ollama"; then ports_in_use=true; fi
        if ! check_port_availability $API_PORT "API Server"; then ports_in_use=true; fi
        if ! check_port_availability $GUI_PORT "GUI"; then ports_in_use=true; fi
        
        if [[ "$ports_in_use" == true ]]; then
            log_error "Some ports are in use. Use --force to restart existing services."
            exit 1
        fi
    fi
    
    # Start services in order
    if ! start_ollama; then
        log_error "Failed to start Ollama service"
        exit 1
    fi
    
    if ! pull_default_models; then
        log_error "Failed to pull default models"
        exit 1
    fi
    
    if ! start_api_server; then
        log_error "Failed to start API server"
        exit 1
    fi
    
    if ! start_gui; then
        log_error "Failed to start GUI"
        exit 1
    fi
    
    start_monitoring  # Optional service
    
    # Health checks
    if ! run_health_checks; then
        log_warning "Some health checks failed, but services appear to be running"
    fi
    
    # Create status file
    create_status_file
    
    # Show summary
    show_startup_summary
    
    log_success "AI Assistant startup completed successfully!"
    
    return 0
}

# Error handling
trap 'log_error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"