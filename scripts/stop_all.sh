#!/bin/bash

# AI Assistant - Stop All Services
# Gracefully stops all running services with proper cleanup

set -e  # Exit on error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="$PROJECT_DIR/run"
STATUS_FILE="$PROJECT_DIR/status.json"

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
    if [[ "$QUIET" != true ]]; then
        echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

log_success() {
    if [[ "$QUIET" != true ]]; then
        echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_warning() {
    if [[ "$QUIET" != true ]]; then
        echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

log_step() {
    if [[ "$QUIET" != true ]]; then
        echo -e "${PURPLE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

# Configuration
GRACEFUL_TIMEOUT=30
FORCE_TIMEOUT=10
QUIET=false
FORCE_KILL=false
CLEANUP_LOGS=false
CLEANUP_DATA=false

# Service ports (for cleanup verification)
OLLAMA_PORT=11434
API_PORT=8000
GUI_PORT=7860
METRICS_PORT=8001

show_help() {
    cat << EOF
AI Assistant Stop Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -q, --quiet             Quiet mode (minimal output)
    -f, --force             Force kill services without graceful shutdown
    -t, --timeout SECONDS   Graceful shutdown timeout (default: 30)
    --cleanup-logs          Remove log files after stopping
    --cleanup-data          Remove temporary data files (CAUTION)
    --cleanup-all           Remove logs, temp data, and PID files

EXAMPLES:
    $0                      # Gracefully stop all services
    $0 --quiet              # Stop services quietly
    $0 --force              # Force stop without graceful shutdown
    $0 --cleanup-logs       # Stop services and clean up logs
    $0 --timeout 60         # Use 60 second graceful timeout

SIGNALS:
    The script sends SIGTERM for graceful shutdown, then SIGKILL if needed.
    Services should handle SIGTERM by saving state and shutting down cleanly.

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -f|--force)
            FORCE_KILL=true
            shift
            ;;
        -t|--timeout)
            GRACEFUL_TIMEOUT="$2"
            shift 2
            ;;
        --cleanup-logs)
            CLEANUP_LOGS=true
            shift
            ;;
        --cleanup-data)
            CLEANUP_DATA=true
            shift
            ;;
        --cleanup-all)
            CLEANUP_LOGS=true
            CLEANUP_DATA=true
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
is_process_running() {
    local pid=$1
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

get_process_by_port() {
    local port=$1
    lsof -ti:$port 2>/dev/null || true
}

stop_process_gracefully() {
    local pid=$1
    local service_name=$2
    local timeout=${3:-$GRACEFUL_TIMEOUT}
    
    if ! is_process_running "$pid"; then
        log_warning "$service_name (PID $pid) is not running"
        return 0
    fi
    
    log_info "Stopping $service_name (PID $pid) gracefully..."
    
    # Send SIGTERM for graceful shutdown
    if kill -TERM "$pid" 2>/dev/null; then
        local count=0
        while [[ $count -lt $timeout ]]; do
            if ! is_process_running "$pid"; then
                log_success "$service_name stopped gracefully"
                return 0
            fi
            sleep 1
            ((count++))
            
            if [[ $((count % 5)) -eq 0 && "$QUIET" != true ]]; then
                echo -n "."
            fi
        done
        
        # Force kill if graceful shutdown failed
        log_warning "$service_name did not stop gracefully, force killing..."
        if kill -KILL "$pid" 2>/dev/null; then
            sleep 2
            if ! is_process_running "$pid"; then
                log_success "$service_name force killed"
                return 0
            else
                log_error "Failed to kill $service_name (PID $pid)"
                return 1
            fi
        else
            log_error "Failed to send kill signal to $service_name (PID $pid)"
            return 1
        fi
    else
        log_error "Failed to send TERM signal to $service_name (PID $pid)"
        return 1
    fi
}

stop_process_by_port() {
    local port=$1
    local service_name=$2
    
    local pids=$(get_process_by_port $port)
    if [[ -n "$pids" ]]; then
        log_info "Found $service_name processes on port $port: $pids"
        for pid in $pids; do
            if [[ "$FORCE_KILL" == true ]]; then
                log_info "Force killing $service_name (PID $pid)..."
                kill -KILL "$pid" 2>/dev/null || true
            else
                stop_process_gracefully "$pid" "$service_name"
            fi
        done
        
        # Verify port is free
        sleep 1
        local remaining_pids=$(get_process_by_port $port)
        if [[ -n "$remaining_pids" ]]; then
            log_warning "Some processes still running on port $port: $remaining_pids"
            return 1
        else
            log_success "Port $port is now free"
            return 0
        fi
    else
        log_info "No processes found on port $port"
        return 0
    fi
}

stop_service_by_pid_file() {
    local pid_file=$1
    local service_name=$2
    
    if [[ ! -f "$pid_file" ]]; then
        log_info "$service_name PID file not found: $pid_file"
        return 0
    fi
    
    local pid=$(cat "$pid_file" 2>/dev/null)
    if [[ -z "$pid" ]]; then
        log_warning "$service_name PID file is empty: $pid_file"
        rm -f "$pid_file"
        return 0
    fi
    
    if is_process_running "$pid"; then
        if [[ "$FORCE_KILL" == true ]]; then
            log_info "Force killing $service_name (PID $pid)..."
            kill -KILL "$pid" 2>/dev/null || true
            sleep 1
        else
            stop_process_gracefully "$pid" "$service_name"
        fi
    else
        log_info "$service_name (PID $pid) is not running"
    fi
    
    # Remove PID file
    rm -f "$pid_file"
    log_info "Removed PID file: $pid_file"
}

stop_gui_service() {
    log_step "Stopping GUI service"
    
    # Try PID file first
    stop_service_by_pid_file "$PID_DIR/gui.pid" "GUI"
    
    # Fallback to port-based detection
    stop_process_by_port $GUI_PORT "GUI"
    
    log_success "GUI service stopped"
}

stop_api_service() {
    log_step "Stopping API service"
    
    # Try PID file first
    stop_service_by_pid_file "$PID_DIR/api.pid" "API"
    
    # Fallback to port-based detection
    stop_process_by_port $API_PORT "API"
    
    log_success "API service stopped"
}

stop_metrics_service() {
    log_step "Stopping metrics service"
    
    # Try PID file first
    stop_service_by_pid_file "$PID_DIR/metrics.pid" "Metrics"
    
    # Fallback to port-based detection
    stop_process_by_port $METRICS_PORT "Metrics"
    
    log_success "Metrics service stopped"
}

stop_ollama_service() {
    log_step "Stopping Ollama service"
    
    # Try PID file first
    if [[ -f "$PID_DIR/ollama.pid" ]]; then
        stop_service_by_pid_file "$PID_DIR/ollama.pid" "Ollama"
    fi
    
    # Try stopping Ollama via command if available
    if command -v ollama >/dev/null 2>&1; then
        log_info "Attempting to stop Ollama via ollama command..."
        if pgrep -f "ollama serve" >/dev/null; then
            pkill -TERM -f "ollama serve" 2>/dev/null || true
            sleep 3
            
            # Force kill if still running
            if pgrep -f "ollama serve" >/dev/null; then
                log_warning "Ollama still running, force killing..."
                pkill -KILL -f "ollama serve" 2>/dev/null || true
            fi
        fi
    fi
    
    # Fallback to port-based detection
    stop_process_by_port $OLLAMA_PORT "Ollama"
    
    log_success "Ollama service stopped"
}

cleanup_pid_files() {
    log_step "Cleaning up PID files"
    
    local pid_files=("$PID_DIR"/*.pid)
    if [[ -f "${pid_files[0]}" ]]; then
        for pid_file in "${pid_files[@]}"; do
            if [[ -f "$pid_file" ]]; then
                local service_name=$(basename "$pid_file" .pid)
                log_info "Removing PID file: $pid_file"
                rm -f "$pid_file"
            fi
        done
        log_success "PID files cleaned up"
    else
        log_info "No PID files to clean up"
    fi
}

cleanup_log_files() {
    if [[ "$CLEANUP_LOGS" == true ]]; then
        log_step "Cleaning up log files"
        
        if [[ -d "$LOG_DIR" ]]; then
            local log_files=("$LOG_DIR"/*.log)
            if [[ -f "${log_files[0]}" ]]; then
                for log_file in "${log_files[@]}"; do
                    if [[ -f "$log_file" ]]; then
                        log_info "Removing log file: $log_file"
                        rm -f "$log_file"
                    fi
                done
                log_success "Log files cleaned up"
            else
                log_info "No log files to clean up"
            fi
        fi
    fi
}

cleanup_temp_files() {
    if [[ "$CLEANUP_DATA" == true ]]; then
        log_step "Cleaning up temporary files"
        
        # Clean temp directory
        if [[ -d "$PROJECT_DIR/temp" ]]; then
            log_info "Removing temporary files..."
            rm -rf "$PROJECT_DIR/temp"/*
            log_success "Temporary files cleaned up"
        fi
        
        # Clean cache files
        if [[ -d "$PROJECT_DIR/__pycache__" ]]; then
            log_info "Removing Python cache files..."
            find "$PROJECT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
            find "$PROJECT_DIR" -name "*.pyc" -delete 2>/dev/null || true
            log_success "Cache files cleaned up"
        fi
        
        log_warning "Temporary data cleanup completed"
    fi
}

update_status_file() {
    log_step "Updating status file"
    
    if [[ -f "$STATUS_FILE" ]]; then
        # Update status to stopped
        if command -v python >/dev/null 2>&1; then
            python -c "
import json
import sys
from datetime import datetime

try:
    with open('$STATUS_FILE', 'r') as f:
        status = json.load(f)
    
    status['status'] = 'stopped'
    status['stopped_at'] = datetime.now().isoformat()
    
    with open('$STATUS_FILE', 'w') as f:
        json.dump(status, f, indent=2)
    
    print('Status file updated')
except Exception as e:
    print(f'Failed to update status file: {e}', file=sys.stderr)
    sys.exit(1)
"
        else
            # Simple fallback if Python is not available
            echo '{"status": "stopped", "stopped_at": "'$(date -Iseconds)'"}' > "$STATUS_FILE"
        fi
        
        log_success "Status file updated"
    else
        log_info "No status file found to update"
    fi
}

verify_all_stopped() {
    log_step "Verifying all services are stopped"
    
    local services_still_running=false
    local ports=($OLLAMA_PORT $API_PORT $GUI_PORT $METRICS_PORT)
    
    for port in "${ports[@]}"; do
        local pids=$(get_process_by_port $port)
        if [[ -n "$pids" ]]; then
            log_warning "Processes still running on port $port: $pids"
            services_still_running=true
        fi
    done
    
    # Check for any remaining AI Assistant processes
    local ai_processes=$(pgrep -f "ai.assistant\|gradio\|uvicorn.*mcp_server" 2>/dev/null || true)
    if [[ -n "$ai_processes" ]]; then
        log_warning "AI Assistant processes still running: $ai_processes"
        services_still_running=true
    fi
    
    if [[ "$services_still_running" == true ]]; then
        log_error "Some services are still running"
        return 1
    else
        log_success "All services successfully stopped"
        return 0
    fi
}

show_stop_summary() {
    if [[ "$QUIET" != true ]]; then
        log_step "Stop Summary"
        
        echo
        echo -e "${CYAN}╭─────────────────────────────────────────────────────╮${NC}"
        echo -e "${CYAN}│                AI Assistant Stopped                 │${NC}"
        echo -e "${CYAN}╰─────────────────────────────────────────────────────╯${NC}"
        echo
        echo -e "${GREEN}All services have been stopped successfully${NC}"
        echo
        
        if [[ "$CLEANUP_LOGS" == true ]]; then
            echo -e "${GREEN}Log files cleaned up${NC}"
        fi
        
        if [[ "$CLEANUP_DATA" == true ]]; then
            echo -e "${GREEN}Temporary data cleaned up${NC}"
        fi
        
        echo -e "${GREEN}Management:${NC}"
        echo -e "  ${BLUE}•${NC} Start services:   $SCRIPT_DIR/start_all.sh"
        echo -e "  ${BLUE}•${NC} Monitor status:   $SCRIPT_DIR/monitor.sh"
        echo -e "  ${BLUE}•${NC} Update system:    $SCRIPT_DIR/update.sh"
        echo
    fi
}

# Main execution
main() {
    if [[ "$QUIET" != true ]]; then
        echo -e "${CYAN}╭─────────────────────────────────────────────────────╮${NC}"
        echo -e "${CYAN}│            AI Assistant Stop Script                │${NC}"
        echo -e "${CYAN}╰─────────────────────────────────────────────────────╯${NC}"
        echo
        
        log_info "Stopping AI Assistant services..."
        
        if [[ "$FORCE_KILL" == true ]]; then
            log_warning "Force kill mode enabled - services will be terminated immediately"
        else
            log_info "Graceful shutdown timeout: ${GRACEFUL_TIMEOUT}s"
        fi
    fi
    
    # Stop services in reverse order of startup
    stop_gui_service
    stop_api_service
    stop_metrics_service
    stop_ollama_service
    
    # Cleanup operations
    cleanup_pid_files
    cleanup_log_files
    cleanup_temp_files
    
    # Update status
    update_status_file
    
    # Verify everything is stopped
    if ! verify_all_stopped; then
        if [[ "$FORCE_KILL" != true ]]; then
            log_warning "Some services are still running. Try with --force to kill them."
            exit 1
        fi
    fi
    
    # Show summary
    show_stop_summary
    
    log_success "AI Assistant shutdown completed successfully!"
    
    return 0
}

# Error handling
trap 'log_error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"