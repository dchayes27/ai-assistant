#!/bin/bash

# AI Assistant - Backup Script
# Creates comprehensive backups of database, configurations, and system state

set -e  # Exit on error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$PROJECT_DIR/backups"
LOG_DIR="$PROJECT_DIR/logs"

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
BACKUP_TYPE="full"
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ai_assistant_backup_$BACKUP_TIMESTAMP"
COMPRESSION=true
ENCRYPTION=false
REMOTE_BACKUP=false
RETENTION_DAYS=30
VERIFY_BACKUP=true
INCLUDE_LOGS=false
INCLUDE_TEMP=false
EXCLUDE_CACHE=true

# Remote backup settings
REMOTE_HOST=""
REMOTE_PATH=""
REMOTE_USER=""

show_help() {
    cat << EOF
AI Assistant Backup Script

Usage: $0 [OPTIONS]

BACKUP TYPES:
    -f, --full              Full backup (default) - everything except logs/temp
    -q, --quick             Quick backup - only database and configs
    -c, --config-only       Configuration files only
    -d, --database-only     Database files only
    -m, --models-only       AI models only

OPTIONS:
    -h, --help              Show this help message
    -n, --name NAME         Custom backup name
    -o, --output DIR        Custom backup directory
    --no-compression        Skip compression
    --encryption            Enable encryption (requires gpg)
    --include-logs          Include log files in backup
    --include-temp          Include temporary files
    --no-verify             Skip backup verification
    --retention DAYS        Backup retention in days (default: 30)
    -v, --verbose           Verbose output

REMOTE BACKUP:
    --remote HOST           Remote backup host
    --remote-path PATH      Remote backup path
    --remote-user USER      Remote backup user
    --ssh-key KEY           SSH private key for remote backup

EXAMPLES:
    $0                      # Full backup with default settings
    $0 --quick              # Quick backup of essentials only
    $0 --config-only        # Backup only configuration files
    $0 --name "pre-update"  # Custom backup name
    $0 --remote server.com --remote-path /backups # Remote backup
    $0 --encryption --retention 90 # Encrypted backup with 90-day retention

ENVIRONMENT VARIABLES:
    AI_ASSISTANT_BACKUP_DIR     Custom backup directory
    AI_ASSISTANT_BACKUP_REMOTE  Remote backup configuration
    AI_ASSISTANT_GPG_KEY        GPG key for encryption

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--full)
            BACKUP_TYPE="full"
            shift
            ;;
        -q|--quick)
            BACKUP_TYPE="quick"
            shift
            ;;
        -c|--config-only)
            BACKUP_TYPE="config"
            shift
            ;;
        -d|--database-only)
            BACKUP_TYPE="database"
            shift
            ;;
        -m|--models-only)
            BACKUP_TYPE="models"
            shift
            ;;
        -n|--name)
            BACKUP_NAME="$2"
            shift 2
            ;;
        -o|--output)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --no-compression)
            COMPRESSION=false
            shift
            ;;
        --encryption)
            ENCRYPTION=true
            shift
            ;;
        --include-logs)
            INCLUDE_LOGS=true
            shift
            ;;
        --include-temp)
            INCLUDE_TEMP=true
            shift
            ;;
        --no-verify)
            VERIFY_BACKUP=false
            shift
            ;;
        --retention)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --remote)
            REMOTE_HOST="$2"
            REMOTE_BACKUP=true
            shift 2
            ;;
        --remote-path)
            REMOTE_PATH="$2"
            shift 2
            ;;
        --remote-user)
            REMOTE_USER="$2"
            shift 2
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
if [[ -n "$AI_ASSISTANT_BACKUP_DIR" ]]; then
    BACKUP_DIR="$AI_ASSISTANT_BACKUP_DIR"
fi

# Utility functions
check_prerequisites() {
    log_step "Checking prerequisites"
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Check available disk space
    local available_space=$(df "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    
    if [[ $available_gb -lt 1 ]]; then
        log_error "Insufficient disk space for backup: ${available_gb}GB available"
        exit 1
    fi
    
    log_info "Available disk space: ${available_gb}GB"
    
    # Check compression tools
    if [[ "$COMPRESSION" == true ]]; then
        if ! command -v tar >/dev/null 2>&1; then
            log_error "tar command not found (required for compression)"
            exit 1
        fi
        
        if command -v gzip >/dev/null 2>&1; then
            COMPRESSION_CMD="gzip"
        elif command -v xz >/dev/null 2>&1; then
            COMPRESSION_CMD="xz"
        else
            log_warning "No compression tool found, disabling compression"
            COMPRESSION=false
        fi
    fi
    
    # Check encryption tools
    if [[ "$ENCRYPTION" == true ]]; then
        if ! command -v gpg >/dev/null 2>&1; then
            log_error "gpg command not found (required for encryption)"
            exit 1
        fi
        
        # Check for GPG key
        if [[ -z "$AI_ASSISTANT_GPG_KEY" ]]; then
            log_error "GPG key not specified. Set AI_ASSISTANT_GPG_KEY environment variable"
            exit 1
        fi
    fi
    
    # Check remote backup prerequisites
    if [[ "$REMOTE_BACKUP" == true ]]; then
        if ! command -v rsync >/dev/null 2>&1; then
            log_error "rsync command not found (required for remote backup)"
            exit 1
        fi
        
        if [[ -z "$REMOTE_HOST" ]]; then
            log_error "Remote host not specified"
            exit 1
        fi
        
        if [[ -z "$REMOTE_PATH" ]]; then
            REMOTE_PATH="/tmp/ai-assistant-backups"
            log_warning "Remote path not specified, using default: $REMOTE_PATH"
        fi
    fi
    
    log_success "Prerequisites check passed"
}

get_backup_size_estimate() {
    log_step "Estimating backup size"
    
    local total_size=0
    
    case "$BACKUP_TYPE" in
        "full")
            if [[ -d "$PROJECT_DIR/data" ]]; then
                total_size=$((total_size + $(du -sk "$PROJECT_DIR/data" | awk '{print $1}')))
            fi
            if [[ -d "$PROJECT_DIR/config" ]]; then
                total_size=$((total_size + $(du -sk "$PROJECT_DIR/config" | awk '{print $1}')))
            fi
            if [[ "$INCLUDE_LOGS" == true ]] && [[ -d "$LOG_DIR" ]]; then
                total_size=$((total_size + $(du -sk "$LOG_DIR" | awk '{print $1}')))
            fi
            ;;
        "quick"|"config")
            if [[ -d "$PROJECT_DIR/config" ]]; then
                total_size=$((total_size + $(du -sk "$PROJECT_DIR/config" | awk '{print $1}')))
            fi
            if [[ "$BACKUP_TYPE" == "quick" ]] && [[ -d "$PROJECT_DIR/data" ]]; then
                total_size=$((total_size + $(du -sk "$PROJECT_DIR/data" | awk '{print $1}')))
            fi
            ;;
        "database")
            if [[ -d "$PROJECT_DIR/data" ]]; then
                total_size=$((total_size + $(du -sk "$PROJECT_DIR/data" | awk '{print $1}')))
            fi
            ;;
    esac
    
    local size_mb=$((total_size / 1024))
    local size_gb=$((size_mb / 1024))
    
    if [[ $size_gb -gt 0 ]]; then
        log_info "Estimated backup size: ${size_gb}GB"
    else
        log_info "Estimated backup size: ${size_mb}MB"
    fi
    
    return $total_size
}

create_backup_manifest() {
    local backup_path="$1"
    local manifest_file="$backup_path/manifest.json"
    
    log_info "Creating backup manifest"
    
    cat > "$manifest_file" << EOF
{
    "backup_info": {
        "name": "$BACKUP_NAME",
        "type": "$BACKUP_TYPE",
        "timestamp": "$(date -Iseconds)",
        "hostname": "$(hostname)",
        "user": "$(whoami)",
        "project_dir": "$PROJECT_DIR",
        "compression": $COMPRESSION,
        "encryption": $ENCRYPTION
    },
    "system_info": {
        "os": "$(uname -s)",
        "arch": "$(uname -m)",
        "kernel": "$(uname -r)",
        "uptime": "$(uptime)"
    },
    "backup_contents": [
EOF
    
    # Add file listing
    find "$backup_path" -type f ! -name "manifest.json" -printf '        "%p",\n' >> "$manifest_file"
    
    # Remove trailing comma and close JSON
    sed -i '$ s/,$//' "$manifest_file"
    cat >> "$manifest_file" << EOF
    ],
    "verification": {
        "file_count": $(find "$backup_path" -type f ! -name "manifest.json" | wc -l),
        "total_size": $(du -sb "$backup_path" | awk '{print $1}'),
        "checksum": "$(find "$backup_path" -type f ! -name "manifest.json" -exec md5sum {} + | md5sum | awk '{print $1}')"
    }
}
EOF
    
    log_success "Backup manifest created"
}

backup_database() {
    local backup_path="$1"
    
    log_step "Backing up database"
    
    local db_backup_dir="$backup_path/database"
    mkdir -p "$db_backup_dir"
    
    # Backup SQLite database files
    if [[ -d "$PROJECT_DIR/data" ]]; then
        log_info "Copying database files..."
        cp -r "$PROJECT_DIR/data"/* "$db_backup_dir/" 2>/dev/null || true
        
        # Create SQL dump for SQLite databases
        for db_file in "$PROJECT_DIR/data"/*.db; do
            if [[ -f "$db_file" ]]; then
                local db_name=$(basename "$db_file" .db)
                log_info "Creating SQL dump for: $db_name"
                
                if command -v sqlite3 >/dev/null 2>&1; then
                    sqlite3 "$db_file" .dump > "$db_backup_dir/${db_name}_dump.sql"
                    log_success "SQL dump created: ${db_name}_dump.sql"
                else
                    log_warning "sqlite3 not found, skipping SQL dump"
                fi
            fi
        done
    else
        log_warning "No database directory found: $PROJECT_DIR/data"
    fi
    
    log_success "Database backup completed"
}

backup_configurations() {
    local backup_path="$1"
    
    log_step "Backing up configurations"
    
    local config_backup_dir="$backup_path/config"
    mkdir -p "$config_backup_dir"
    
    # Backup configuration directory
    if [[ -d "$PROJECT_DIR/config" ]]; then
        log_info "Copying configuration files..."
        cp -r "$PROJECT_DIR/config"/* "$config_backup_dir/"
        log_success "Configuration files backed up"
    else
        log_warning "No configuration directory found: $PROJECT_DIR/config"
    fi
    
    # Backup other important files
    local important_files=(
        "requirements.txt"
        "requirements.lock"
        ".env"
        "status.json"
        "README.md"
    )
    
    for file in "${important_files[@]}"; do
        if [[ -f "$PROJECT_DIR/$file" ]]; then
            log_info "Backing up: $file"
            cp "$PROJECT_DIR/$file" "$config_backup_dir/"
        fi
    done
    
    log_success "Configuration backup completed"
}

backup_models() {
    local backup_path="$1"
    
    log_step "Backing up AI models"
    
    local models_backup_dir="$backup_path/models"
    mkdir -p "$models_backup_dir"
    
    # Backup Ollama models (model information, not the actual models)
    if command -v ollama >/dev/null 2>&1; then
        log_info "Backing up Ollama model list..."
        
        # Start Ollama temporarily if not running
        local ollama_was_running=false
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            ollama_was_running=true
        else
            log_info "Starting Ollama temporarily..."
            nohup ollama serve > /dev/null 2>&1 &
            local ollama_pid=$!
            sleep 5
        fi
        
        # Export model list
        ollama list > "$models_backup_dir/ollama_models.txt" 2>/dev/null || true
        
        # Stop Ollama if we started it
        if [[ "$ollama_was_running" == false ]] && [[ -n "$ollama_pid" ]]; then
            kill $ollama_pid 2>/dev/null || true
        fi
        
        log_success "Ollama model list backed up"
    fi
    
    # Backup Whisper model cache info
    local whisper_cache_dir="$HOME/.cache/whisper"
    if [[ -d "$whisper_cache_dir" ]]; then
        log_info "Backing up Whisper model information..."
        mkdir -p "$models_backup_dir/whisper"
        ls -la "$whisper_cache_dir" > "$models_backup_dir/whisper/model_cache.txt" 2>/dev/null || true
    fi
    
    # Backup TTS model information
    local tts_cache_dir="$HOME/.local/share/tts"
    if [[ -d "$tts_cache_dir" ]]; then
        log_info "Backing up TTS model information..."
        mkdir -p "$models_backup_dir/tts"
        ls -la "$tts_cache_dir" > "$models_backup_dir/tts/model_cache.txt" 2>/dev/null || true
    fi
    
    log_success "Model backup completed"
}

backup_logs() {
    local backup_path="$1"
    
    if [[ "$INCLUDE_LOGS" != true ]]; then
        return 0
    fi
    
    log_step "Backing up logs"
    
    local logs_backup_dir="$backup_path/logs"
    mkdir -p "$logs_backup_dir"
    
    if [[ -d "$LOG_DIR" ]]; then
        log_info "Copying log files..."
        cp -r "$LOG_DIR"/* "$logs_backup_dir/" 2>/dev/null || true
        log_success "Log files backed up"
    else
        log_warning "No log directory found: $LOG_DIR"
    fi
}

backup_system_state() {
    local backup_path="$1"
    
    log_step "Backing up system state"
    
    local state_backup_dir="$backup_path/system"
    mkdir -p "$state_backup_dir"
    
    # System information
    log_info "Collecting system information..."
    
    cat > "$state_backup_dir/system_info.txt" << EOF
System Information
==================

Date: $(date)
Hostname: $(hostname)
User: $(whoami)
OS: $(uname -s)
Kernel: $(uname -r)
Architecture: $(uname -m)
Uptime: $(uptime)

CPU Information:
$(lscpu 2>/dev/null || system_profiler SPHardwareDataType 2>/dev/null || echo "CPU info not available")

Memory Information:
$(free -h 2>/dev/null || vm_stat 2>/dev/null || echo "Memory info not available")

Disk Usage:
$(df -h)

Network Interfaces:
$(ip addr show 2>/dev/null || ifconfig 2>/dev/null || echo "Network info not available")

Environment Variables:
$(env | grep AI_ASSISTANT || echo "No AI_ASSISTANT environment variables")

Process List:
$(ps aux | grep -E "(ollama|uvicorn|gradio|python)" | grep -v grep || echo "No relevant processes found")
EOF
    
    # Python environment
    if [[ -d "$PROJECT_DIR/venv" ]]; then
        log_info "Backing up Python environment information..."
        source "$PROJECT_DIR/venv/bin/activate" 2>/dev/null || true
        
        cat > "$state_backup_dir/python_env.txt" << EOF
Python Environment
==================

Python Version: $(python --version 2>&1)
Pip Version: $(pip --version 2>&1)
Virtual Environment: $PROJECT_DIR/venv

Installed Packages:
$(pip list 2>/dev/null || echo "Could not list packages")

Pip Freeze:
$(pip freeze 2>/dev/null || echo "Could not freeze requirements")
EOF
    fi
    
    # Git information
    if [[ -d "$PROJECT_DIR/.git" ]]; then
        log_info "Backing up Git repository information..."
        cd "$PROJECT_DIR"
        
        cat > "$state_backup_dir/git_info.txt" << EOF
Git Repository Information
==========================

Current Branch: $(git branch --show-current 2>/dev/null || echo "Unknown")
Current Commit: $(git rev-parse HEAD 2>/dev/null || echo "Unknown")
Remote URL: $(git remote get-url origin 2>/dev/null || echo "No remote")
Status: $(git status --porcelain 2>/dev/null || echo "Could not get status")

Recent Commits:
$(git log --oneline -10 2>/dev/null || echo "Could not get log")

Remotes:
$(git remote -v 2>/dev/null || echo "No remotes")
EOF
    fi
    
    log_success "System state backed up"
}

compress_backup() {
    local backup_path="$1"
    
    if [[ "$COMPRESSION" != true ]]; then
        return 0
    fi
    
    log_step "Compressing backup"
    
    local compressed_file="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"
    
    cd "$BACKUP_DIR"
    log_info "Creating compressed archive: $compressed_file"
    
    if tar -czf "$compressed_file" "$(basename "$backup_path")"; then
        log_success "Backup compressed successfully"
        
        # Remove uncompressed directory
        rm -rf "$backup_path"
        
        # Update backup path
        echo "$compressed_file"
    else
        log_error "Failed to compress backup"
        return 1
    fi
}

encrypt_backup() {
    local backup_file="$1"
    
    if [[ "$ENCRYPTION" != true ]]; then
        echo "$backup_file"
        return 0
    fi
    
    log_step "Encrypting backup"
    
    local encrypted_file="${backup_file}.gpg"
    
    log_info "Encrypting backup with GPG key: $AI_ASSISTANT_GPG_KEY"
    
    if gpg --trust-model always --encrypt -r "$AI_ASSISTANT_GPG_KEY" \
       --cipher-algo AES256 --output "$encrypted_file" "$backup_file"; then
        log_success "Backup encrypted successfully"
        
        # Remove unencrypted file
        rm -f "$backup_file"
        
        echo "$encrypted_file"
    else
        log_error "Failed to encrypt backup"
        return 1
    fi
}

verify_backup() {
    local backup_file="$1"
    
    if [[ "$VERIFY_BACKUP" != true ]]; then
        return 0
    fi
    
    log_step "Verifying backup"
    
    # Check file exists and is readable
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    if [[ ! -r "$backup_file" ]]; then
        log_error "Backup file not readable: $backup_file"
        return 1
    fi
    
    # Check file size
    local file_size=$(stat -c%s "$backup_file" 2>/dev/null || stat -f%z "$backup_file" 2>/dev/null)
    if [[ $file_size -lt 1024 ]]; then
        log_error "Backup file appears to be too small: ${file_size} bytes"
        return 1
    fi
    
    log_info "Backup file size: $(numfmt --to=iec $file_size)"
    
    # Verify compressed file integrity
    if [[ "$backup_file" == *.tar.gz ]]; then
        log_info "Verifying compressed archive integrity..."
        if tar -tzf "$backup_file" >/dev/null 2>&1; then
            log_success "Archive integrity verified"
        else
            log_error "Archive integrity check failed"
            return 1
        fi
    fi
    
    # Verify encrypted file
    if [[ "$backup_file" == *.gpg ]]; then
        log_info "Verifying encrypted file..."
        if gpg --list-packets "$backup_file" >/dev/null 2>&1; then
            log_success "Encrypted file verified"
        else
            log_error "Encrypted file verification failed"
            return 1
        fi
    fi
    
    log_success "Backup verification completed"
}

upload_remote_backup() {
    local backup_file="$1"
    
    if [[ "$REMOTE_BACKUP" != true ]]; then
        return 0
    fi
    
    log_step "Uploading to remote backup location"
    
    local remote_target="${REMOTE_USER:+$REMOTE_USER@}$REMOTE_HOST:$REMOTE_PATH/"
    
    log_info "Uploading to: $remote_target"
    
    # Create remote directory
    if [[ -n "$REMOTE_USER" ]]; then
        ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_PATH'" 2>/dev/null || true
    else
        ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_PATH'" 2>/dev/null || true
    fi
    
    # Upload file
    if rsync -avz --progress "$backup_file" "$remote_target"; then
        log_success "Remote backup uploaded successfully"
    else
        log_error "Failed to upload remote backup"
        return 1
    fi
}

cleanup_old_backups() {
    log_step "Cleaning up old backups"
    
    if [[ $RETENTION_DAYS -le 0 ]]; then
        log_info "Backup retention disabled"
        return 0
    fi
    
    log_info "Removing backups older than $RETENTION_DAYS days"
    
    # Find and remove old backups
    local old_backups=$(find "$BACKUP_DIR" -name "ai_assistant_backup_*" -type f -mtime +$RETENTION_DAYS 2>/dev/null || true)
    
    if [[ -n "$old_backups" ]]; then
        echo "$old_backups" | while read -r old_backup; do
            log_info "Removing old backup: $(basename "$old_backup")"
            rm -f "$old_backup"
        done
        
        local removed_count=$(echo "$old_backups" | wc -l)
        log_success "Removed $removed_count old backup(s)"
    else
        log_info "No old backups found to remove"
    fi
}

show_backup_summary() {
    local backup_file="$1"
    
    log_step "Backup Summary"
    
    echo
    echo -e "${CYAN}╭─────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│                Backup Completed                     │${NC}"
    echo -e "${CYAN}╰─────────────────────────────────────────────────────╯${NC}"
    echo
    
    echo -e "${GREEN}Backup Information:${NC}"
    echo -e "  ${BLUE}•${NC} Type: $BACKUP_TYPE"
    echo -e "  ${BLUE}•${NC} Name: $BACKUP_NAME"
    echo -e "  ${BLUE}•${NC} Location: $backup_file"
    
    if [[ -f "$backup_file" ]]; then
        local file_size=$(stat -c%s "$backup_file" 2>/dev/null || stat -f%z "$backup_file" 2>/dev/null)
        echo -e "  ${BLUE}•${NC} Size: $(numfmt --to=iec $file_size)"
    fi
    
    echo -e "  ${BLUE}•${NC} Compression: $([ "$COMPRESSION" == true ] && echo "Enabled" || echo "Disabled")"
    echo -e "  ${BLUE}•${NC} Encryption: $([ "$ENCRYPTION" == true ] && echo "Enabled" || echo "Disabled")"
    echo -e "  ${BLUE}•${NC} Remote Backup: $([ "$REMOTE_BACKUP" == true ] && echo "Enabled" || echo "Disabled")"
    echo
    
    echo -e "${GREEN}Management:${NC}"
    echo -e "  ${BLUE}•${NC} List backups: ls -la $BACKUP_DIR"
    echo -e "  ${BLUE}•${NC} Restore backup: $SCRIPT_DIR/restore.sh $backup_file"
    echo
}

# Main execution
main() {
    echo -e "${CYAN}╭─────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│            AI Assistant Backup Script              │${NC}"
    echo -e "${CYAN}╰─────────────────────────────────────────────────────╯${NC}"
    echo
    
    log_info "Starting backup process"
    log_info "Backup type: $BACKUP_TYPE"
    log_info "Backup name: $BACKUP_NAME"
    log_info "Backup directory: $BACKUP_DIR"
    
    # Pre-flight checks
    check_prerequisites
    get_backup_size_estimate
    
    # Create backup directory
    local backup_path="$BACKUP_DIR/$BACKUP_NAME"
    mkdir -p "$backup_path"
    
    # Perform backup based on type
    case "$BACKUP_TYPE" in
        "full")
            backup_database "$backup_path"
            backup_configurations "$backup_path"
            backup_models "$backup_path"
            backup_logs "$backup_path"
            backup_system_state "$backup_path"
            ;;
        "quick")
            backup_database "$backup_path"
            backup_configurations "$backup_path"
            backup_system_state "$backup_path"
            ;;
        "config")
            backup_configurations "$backup_path"
            ;;
        "database")
            backup_database "$backup_path"
            ;;
        "models")
            backup_models "$backup_path"
            ;;
    esac
    
    # Create manifest
    create_backup_manifest "$backup_path"
    
    # Compress backup
    local final_backup_path="$backup_path"
    if [[ "$COMPRESSION" == true ]]; then
        final_backup_path=$(compress_backup "$backup_path")
    fi
    
    # Encrypt backup
    if [[ "$ENCRYPTION" == true ]]; then
        final_backup_path=$(encrypt_backup "$final_backup_path")
    fi
    
    # Verify backup
    verify_backup "$final_backup_path"
    
    # Upload to remote location
    upload_remote_backup "$final_backup_path"
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Show summary
    show_backup_summary "$final_backup_path"
    
    log_success "Backup completed successfully!"
    
    return 0
}

# Error handling
trap 'log_error "Backup interrupted"; exit 1' INT TERM

# Run main function
main "$@"