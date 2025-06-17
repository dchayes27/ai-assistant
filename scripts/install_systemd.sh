#!/bin/bash

# AI Assistant - SystemD Service Installation Script
# Installs and configures systemd services for auto-start on boot

set -e  # Exit on error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SYSTEMD_DIR="/etc/systemd/system"
INSTALL_DIR="/opt/ai-assistant"
SERVICE_USER="ai-assistant"
SERVICE_GROUP="ai-assistant"

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
INSTALL_ALL=true
INSTALL_SERVICES=true
INSTALL_TIMERS=true
CREATE_USER=true
COPY_FILES=true
START_SERVICES=false
DRY_RUN=false
FORCE_INSTALL=false

show_help() {
    cat << EOF
AI Assistant SystemD Installation Script

Usage: $0 [OPTIONS]

INSTALLATION OPTIONS:
    -h, --help              Show this help message
    -a, --all               Install all services and timers (default)
    -s, --services-only     Install only services (no timers)
    -t, --timers-only       Install only timers
    --no-user               Don't create service user
    --no-copy               Don't copy project files to /opt
    --start                 Start services after installation
    --force                 Force installation (overwrite existing)
    --dry-run               Show what would be done without executing

SPECIFIC SERVICES:
    --ollama                Install Ollama service only
    --api                   Install API service only
    --gui                   Install GUI service only
    --backup                Install backup service only

EXAMPLES:
    $0                      # Install all services and timers
    $0 --services-only      # Install only services
    $0 --start              # Install and start services
    $0 --force --start      # Force reinstall and start
    $0 --dry-run            # Preview installation

NOTES:
    - This script requires root privileges
    - The AI Assistant will be installed to $INSTALL_DIR
    - A dedicated user '$SERVICE_USER' will be created
    - Services will be configured to start on boot

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
            INSTALL_ALL=true
            INSTALL_SERVICES=true
            INSTALL_TIMERS=true
            shift
            ;;
        -s|--services-only)
            INSTALL_SERVICES=true
            INSTALL_TIMERS=false
            shift
            ;;
        -t|--timers-only)
            INSTALL_SERVICES=false
            INSTALL_TIMERS=true
            shift
            ;;
        --no-user)
            CREATE_USER=false
            shift
            ;;
        --no-copy)
            COPY_FILES=false
            shift
            ;;
        --start)
            START_SERVICES=true
            shift
            ;;
        --force)
            FORCE_INSTALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --ollama|--api|--gui|--backup)
            INSTALL_ALL=false
            SPECIFIC_SERVICE="${1#--}"
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
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        log_info "Please run: sudo $0"
        exit 1
    fi
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
    
    # Check if systemd is available
    if ! command -v systemctl >/dev/null 2>&1; then
        log_error "systemctl not found. This system doesn't appear to use systemd"
        exit 1
    fi
    
    # Check systemd version
    local systemd_version=$(systemctl --version | head -n1 | awk '{print $2}')
    log_info "SystemD version: $systemd_version"
    
    # Check if running on a supported system
    if [[ ! -d "/etc/systemd/system" ]]; then
        log_error "SystemD system directory not found: /etc/systemd/system"
        exit 1
    fi
    
    # Check if source files exist
    if [[ ! -d "$SCRIPT_DIR/systemd" ]]; then
        log_error "SystemD service files not found: $SCRIPT_DIR/systemd"
        exit 1
    fi
    
    # Check available disk space
    local available_space=$(df /opt | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    
    if [[ $available_gb -lt 2 ]]; then
        log_error "Insufficient disk space in /opt: ${available_gb}GB available (need at least 2GB)"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

create_service_user() {
    if [[ "$CREATE_USER" != true ]]; then
        log_info "Skipping user creation as requested"
        return 0
    fi
    
    log_step "Creating service user and group"
    
    # Check if group exists
    if ! getent group "$SERVICE_GROUP" >/dev/null 2>&1; then
        execute_command "groupadd --system '$SERVICE_GROUP'" "create group $SERVICE_GROUP"
    else
        log_info "Group $SERVICE_GROUP already exists"
    fi
    
    # Check if user exists
    if ! getent passwd "$SERVICE_USER" >/dev/null 2>&1; then
        execute_command "useradd --system --gid '$SERVICE_GROUP' --home-dir '$INSTALL_DIR' --shell /bin/false --comment 'AI Assistant Service User' '$SERVICE_USER'" "create user $SERVICE_USER"
    else
        log_info "User $SERVICE_USER already exists"
    fi
    
    log_success "Service user and group configured"
}

copy_project_files() {
    if [[ "$COPY_FILES" != true ]]; then
        log_info "Skipping file copy as requested"
        return 0
    fi
    
    log_step "Copying project files to $INSTALL_DIR"
    
    # Create installation directory
    execute_command "mkdir -p '$INSTALL_DIR'" "create installation directory"
    
    # Copy project files
    log_info "Copying project files..."
    execute_command "cp -r '$PROJECT_DIR'/* '$INSTALL_DIR'/" "copy project files"
    
    # Create required directories
    local required_dirs=("logs" "run" "data" "temp" "backups" "models")
    for dir in "${required_dirs[@]}"; do
        execute_command "mkdir -p '$INSTALL_DIR/$dir'" "create directory $dir"
    done
    
    # Set ownership
    execute_command "chown -R '$SERVICE_USER:$SERVICE_GROUP' '$INSTALL_DIR'" "set ownership"
    
    # Set permissions
    execute_command "chmod -R 755 '$INSTALL_DIR'" "set permissions"
    execute_command "chmod +x '$INSTALL_DIR/scripts'/*.sh" "make scripts executable"
    
    # Secure sensitive files
    execute_command "chmod 600 '$INSTALL_DIR/config'/*.yaml" "secure configuration files"
    execute_command "chmod 700 '$INSTALL_DIR/data'" "secure data directory"
    
    log_success "Project files copied and secured"
}

install_service_file() {
    local service_name="$1"
    local service_file="$SCRIPT_DIR/systemd/$service_name"
    local target_file="$SYSTEMD_DIR/$service_name"
    
    if [[ ! -f "$service_file" ]]; then
        log_error "Service file not found: $service_file"
        return 1
    fi
    
    if [[ -f "$target_file" ]] && [[ "$FORCE_INSTALL" != true ]]; then
        log_warning "Service file already exists: $target_file"
        log_info "Use --force to overwrite"
        return 0
    fi
    
    log_info "Installing service: $service_name"
    execute_command "cp '$service_file' '$target_file'" "copy service file"
    execute_command "chmod 644 '$target_file'" "set service file permissions"
    
    return 0
}

install_services() {
    if [[ "$INSTALL_SERVICES" != true ]]; then
        return 0
    fi
    
    log_step "Installing systemd services"
    
    if [[ "$INSTALL_ALL" == true ]]; then
        local services=("ai-assistant.service" "ai-assistant-ollama.service" "ai-assistant-api.service" "ai-assistant-gui.service" "ai-assistant-backup.service")
        
        for service in "${services[@]}"; do
            install_service_file "$service"
        done
    elif [[ -n "$SPECIFIC_SERVICE" ]]; then
        install_service_file "ai-assistant-${SPECIFIC_SERVICE}.service"
    fi
    
    log_success "Services installed"
}

install_timers() {
    if [[ "$INSTALL_TIMERS" != true ]]; then
        return 0
    fi
    
    log_step "Installing systemd timers"
    
    local timers=("ai-assistant-backup.timer")
    
    for timer in "${timers[@]}"; do
        install_service_file "$timer"
    done
    
    log_success "Timers installed"
}

reload_systemd() {
    log_step "Reloading systemd daemon"
    
    execute_command "systemctl daemon-reload" "reload systemd daemon"
    
    log_success "SystemD daemon reloaded"
}

enable_services() {
    log_step "Enabling services for auto-start"
    
    if [[ "$INSTALL_ALL" == true ]]; then
        execute_command "systemctl enable ai-assistant.service" "enable main service"
        execute_command "systemctl enable ai-assistant-ollama.service" "enable Ollama service"
        
        if [[ "$INSTALL_TIMERS" == true ]]; then
            execute_command "systemctl enable ai-assistant-backup.timer" "enable backup timer"
        fi
    elif [[ -n "$SPECIFIC_SERVICE" ]]; then
        execute_command "systemctl enable ai-assistant-${SPECIFIC_SERVICE}.service" "enable $SPECIFIC_SERVICE service"
    fi
    
    log_success "Services enabled for auto-start"
}

start_services_now() {
    if [[ "$START_SERVICES" != true ]]; then
        return 0
    fi
    
    log_step "Starting services"
    
    if [[ "$INSTALL_ALL" == true ]]; then
        execute_command "systemctl start ai-assistant-ollama.service" "start Ollama service"
        sleep 5
        execute_command "systemctl start ai-assistant.service" "start main service"
        
        if [[ "$INSTALL_TIMERS" == true ]]; then
            execute_command "systemctl start ai-assistant-backup.timer" "start backup timer"
        fi
    elif [[ -n "$SPECIFIC_SERVICE" ]]; then
        execute_command "systemctl start ai-assistant-${SPECIFIC_SERVICE}.service" "start $SPECIFIC_SERVICE service"
    fi
    
    log_success "Services started"
}

show_status() {
    log_step "Service Status"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would show service status"
        return 0
    fi
    
    echo
    echo -e "${WHITE}SystemD Service Status:${NC}"
    echo -e "${GRAY}─────────────────────────${NC}"
    
    local services=("ai-assistant" "ai-assistant-ollama" "ai-assistant-api" "ai-assistant-gui")
    
    for service in "${services[@]}"; do
        if systemctl list-unit-files | grep -q "^${service}.service"; then
            local status=$(systemctl is-active "$service.service" 2>/dev/null || echo "inactive")
            local enabled=$(systemctl is-enabled "$service.service" 2>/dev/null || echo "disabled")
            
            local status_color
            case "$status" in
                "active") status_color="$GREEN" ;;
                "inactive") status_color="$GRAY" ;;
                "failed") status_color="$RED" ;;
                *) status_color="$YELLOW" ;;
            esac
            
            local enabled_color
            case "$enabled" in
                "enabled") enabled_color="$GREEN" ;;
                "disabled") enabled_color="$GRAY" ;;
                *) enabled_color="$YELLOW" ;;
            esac
            
            printf "  %-25s ${status_color}%-8s${NC} ${enabled_color}%-8s${NC}\n" \
                   "$service" "$status" "$enabled"
        fi
    done
    
    # Show timer status
    if systemctl list-unit-files | grep -q "^ai-assistant-backup.timer"; then
        local timer_status=$(systemctl is-active "ai-assistant-backup.timer" 2>/dev/null || echo "inactive")
        local timer_enabled=$(systemctl is-enabled "ai-assistant-backup.timer" 2>/dev/null || echo "disabled")
        
        echo
        echo -e "${WHITE}Timer Status:${NC}"
        echo -e "${GRAY}─────────────${NC}"
        printf "  %-25s ${GREEN}%-8s${NC} ${GREEN}%-8s${NC}\n" \
               "ai-assistant-backup.timer" "$timer_status" "$timer_enabled"
    fi
    
    echo
}

create_uninstall_script() {
    log_step "Creating uninstall script"
    
    local uninstall_script="$INSTALL_DIR/scripts/uninstall_systemd.sh"
    
    cat > "$uninstall_script" << 'EOF'
#!/bin/bash

# AI Assistant - SystemD Service Uninstallation Script

set -e

echo "Stopping and disabling AI Assistant services..."

# Stop services
systemctl stop ai-assistant.service 2>/dev/null || true
systemctl stop ai-assistant-ollama.service 2>/dev/null || true
systemctl stop ai-assistant-api.service 2>/dev/null || true
systemctl stop ai-assistant-gui.service 2>/dev/null || true
systemctl stop ai-assistant-backup.timer 2>/dev/null || true
systemctl stop ai-assistant-backup.service 2>/dev/null || true

# Disable services
systemctl disable ai-assistant.service 2>/dev/null || true
systemctl disable ai-assistant-ollama.service 2>/dev/null || true
systemctl disable ai-assistant-api.service 2>/dev/null || true
systemctl disable ai-assistant-gui.service 2>/dev/null || true
systemctl disable ai-assistant-backup.timer 2>/dev/null || true
systemctl disable ai-assistant-backup.service 2>/dev/null || true

# Remove service files
rm -f /etc/systemd/system/ai-assistant*.service
rm -f /etc/systemd/system/ai-assistant*.timer

# Reload systemd
systemctl daemon-reload

echo "AI Assistant systemd services have been uninstalled."
echo "Note: Project files in /opt/ai-assistant and user accounts are preserved."
echo "To completely remove, run: sudo rm -rf /opt/ai-assistant && sudo userdel ai-assistant"
EOF
    
    execute_command "chmod +x '$uninstall_script'" "make uninstall script executable"
    
    log_success "Uninstall script created: $uninstall_script"
}

show_installation_summary() {
    log_step "Installation Summary"
    
    echo
    echo -e "${CYAN}╭─────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│              SystemD Installation Complete          │${NC}"
    echo -e "${CYAN}╰─────────────────────────────────────────────────────╯${NC}"
    echo
    
    echo -e "${GREEN}Installation Details:${NC}"
    echo -e "  ${BLUE}•${NC} Installation Directory: $INSTALL_DIR"
    echo -e "  ${BLUE}•${NC} Service User: $SERVICE_USER"
    echo -e "  ${BLUE}•${NC} Service Group: $SERVICE_GROUP"
    echo -e "  ${BLUE}•${NC} SystemD Directory: $SYSTEMD_DIR"
    echo
    
    echo -e "${GREEN}Installed Services:${NC}"
    if [[ "$INSTALL_ALL" == true ]]; then
        echo -e "  ${BLUE}•${NC} ai-assistant.service (main service)"
        echo -e "  ${BLUE}•${NC} ai-assistant-ollama.service (LLM service)"
        echo -e "  ${BLUE}•${NC} ai-assistant-api.service (API server)"
        echo -e "  ${BLUE}•${NC} ai-assistant-gui.service (web interface)"
        echo -e "  ${BLUE}•${NC} ai-assistant-backup.service (backup service)"
        
        if [[ "$INSTALL_TIMERS" == true ]]; then
            echo -e "  ${BLUE}•${NC} ai-assistant-backup.timer (scheduled backups)"
        fi
    elif [[ -n "$SPECIFIC_SERVICE" ]]; then
        echo -e "  ${BLUE}•${NC} ai-assistant-${SPECIFIC_SERVICE}.service"
    fi
    echo
    
    echo -e "${GREEN}Management Commands:${NC}"
    echo -e "  ${BLUE}•${NC} Start services:     sudo systemctl start ai-assistant"
    echo -e "  ${BLUE}•${NC} Stop services:      sudo systemctl stop ai-assistant"
    echo -e "  ${BLUE}•${NC} Restart services:   sudo systemctl restart ai-assistant"
    echo -e "  ${BLUE}•${NC} Check status:       sudo systemctl status ai-assistant"
    echo -e "  ${BLUE}•${NC} View logs:          sudo journalctl -u ai-assistant -f"
    echo -e "  ${BLUE}•${NC} Disable auto-start: sudo systemctl disable ai-assistant"
    echo -e "  ${BLUE}•${NC} Uninstall:          sudo $INSTALL_DIR/scripts/uninstall_systemd.sh"
    echo
    
    if [[ "$START_SERVICES" == true ]]; then
        echo -e "${GREEN}Services Status:${NC}"
        show_status
    else
        echo -e "${YELLOW}Note: Services are installed but not started.${NC}"
        echo -e "${YELLOW}Run 'sudo systemctl start ai-assistant' to start them.${NC}"
        echo
    fi
}

# Main execution
main() {
    echo -e "${CYAN}╭─────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│         AI Assistant SystemD Installation          │${NC}"
    echo -e "${CYAN}╰─────────────────────────────────────────────────────╯${NC}"
    echo
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN MODE - No actual changes will be made"
    fi
    
    log_info "Installation configuration:"
    log_info "  Install Directory: $INSTALL_DIR"
    log_info "  Service User: $SERVICE_USER"
    log_info "  Install Services: $INSTALL_SERVICES"
    log_info "  Install Timers: $INSTALL_TIMERS"
    log_info "  Create User: $CREATE_USER"
    log_info "  Copy Files: $COPY_FILES"
    log_info "  Start Services: $START_SERVICES"
    
    # Pre-flight checks
    check_root
    check_prerequisites
    
    # Installation steps
    create_service_user
    copy_project_files
    install_services
    install_timers
    reload_systemd
    enable_services
    create_uninstall_script
    start_services_now
    
    # Show results
    show_installation_summary
    
    log_success "AI Assistant SystemD installation completed successfully!"
    
    return 0
}

# Error handling
trap 'log_error "Installation interrupted"; exit 1' INT TERM

# Run main function
main "$@"