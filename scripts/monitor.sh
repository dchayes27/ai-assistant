#!/bin/bash

# AI Assistant - Monitoring Script
# Real-time system monitoring and status dashboard

set -e  # Exit on error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
STATUS_FILE="$PROJECT_DIR/status.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Symbols for status
CHECKMARK="✓"
CROSSMARK="✗"
WARNING="⚠"
INFO="ℹ"
ARROW="→"

# Configuration
REFRESH_INTERVAL=5
COMPACT_MODE=false
SHOW_LOGS=false
SHOW_PERFORMANCE=true
SHOW_NETWORK=true
SHOW_SERVICES=true
CONTINUOUS_MODE=true
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=85
ALERT_THRESHOLD_DISK=90

# Service ports
OLLAMA_PORT=11434
API_PORT=8000
GUI_PORT=7860
METRICS_PORT=8001

show_help() {
    cat << EOF
AI Assistant Monitoring Script

Usage: $0 [OPTIONS]

MONITORING MODES:
    -d, --dashboard         Real-time dashboard (default)
    -s, --status            Show current status and exit
    -l, --logs              Show recent log entries
    -p, --performance       Show performance metrics only
    -n, --network           Show network status only
    --services              Show service status only

OPTIONS:
    -h, --help              Show this help message
    -c, --compact           Compact display mode
    -i, --interval SEC      Refresh interval in seconds (default: 5)
    --no-continuous         Show once and exit
    --no-color              Disable colored output
    --csv                   Output in CSV format
    --json                  Output in JSON format

ALERTS:
    --cpu-alert PERCENT     CPU usage alert threshold (default: 80)
    --memory-alert PERCENT  Memory usage alert threshold (default: 85)
    --disk-alert PERCENT    Disk usage alert threshold (default: 90)

EXAMPLES:
    $0                      # Real-time dashboard
    $0 --status             # Quick status check
    $0 --compact --interval 2 # Compact mode with 2s refresh
    $0 --logs               # Show recent logs
    $0 --json               # Output status in JSON format

KEYBOARD SHORTCUTS (in dashboard mode):
    q, Ctrl+C               Quit
    r                       Refresh immediately
    l                       Toggle log display
    p                       Toggle performance display
    n                       Toggle network display
    s                       Toggle service display
    c                       Toggle compact mode
    h, ?                    Show help

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dashboard)
            CONTINUOUS_MODE=true
            shift
            ;;
        -s|--status)
            CONTINUOUS_MODE=false
            shift
            ;;
        -l|--logs)
            SHOW_LOGS=true
            CONTINUOUS_MODE=false
            shift
            ;;
        -p|--performance)
            SHOW_PERFORMANCE=true
            SHOW_NETWORK=false
            SHOW_SERVICES=false
            CONTINUOUS_MODE=false
            shift
            ;;
        -n|--network)
            SHOW_PERFORMANCE=false
            SHOW_NETWORK=true
            SHOW_SERVICES=false
            CONTINUOUS_MODE=false
            shift
            ;;
        --services)
            SHOW_PERFORMANCE=false
            SHOW_NETWORK=false
            SHOW_SERVICES=true
            CONTINUOUS_MODE=false
            shift
            ;;
        -c|--compact)
            COMPACT_MODE=true
            shift
            ;;
        -i|--interval)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        --no-continuous)
            CONTINUOUS_MODE=false
            shift
            ;;
        --no-color)
            RED=''
            GREEN=''
            YELLOW=''
            BLUE=''
            PURPLE=''
            CYAN=''
            WHITE=''
            GRAY=''
            NC=''
            shift
            ;;
        --csv)
            OUTPUT_FORMAT="csv"
            CONTINUOUS_MODE=false
            shift
            ;;
        --json)
            OUTPUT_FORMAT="json"
            CONTINUOUS_MODE=false
            shift
            ;;
        --cpu-alert)
            ALERT_THRESHOLD_CPU="$2"
            shift 2
            ;;
        --memory-alert)
            ALERT_THRESHOLD_MEMORY="$2"
            shift 2
            ;;
        --disk-alert)
            ALERT_THRESHOLD_DISK="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Utility functions
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

format_bytes() {
    local bytes=$1
    if [[ $bytes -gt 1073741824 ]]; then
        echo "$(( bytes / 1073741824 ))GB"
    elif [[ $bytes -gt 1048576 ]]; then
        echo "$(( bytes / 1048576 ))MB"
    elif [[ $bytes -gt 1024 ]]; then
        echo "$(( bytes / 1024 ))KB"
    else
        echo "${bytes}B"
    fi
}

format_uptime() {
    local seconds=$1
    local days=$((seconds / 86400))
    local hours=$(((seconds % 86400) / 3600))
    local minutes=$(((seconds % 3600) / 60))
    
    if [[ $days -gt 0 ]]; then
        echo "${days}d ${hours}h ${minutes}m"
    elif [[ $hours -gt 0 ]]; then
        echo "${hours}h ${minutes}m"
    else
        echo "${minutes}m"
    fi
}

get_status_color() {
    local status=$1
    case "$status" in
        "running"|"online"|"active"|"healthy")
            echo "$GREEN"
            ;;
        "stopped"|"offline"|"inactive"|"down")
            echo "$RED"
            ;;
        "warning"|"degraded"|"partial")
            echo "$YELLOW"
            ;;
        *)
            echo "$GRAY"
            ;;
    esac
}

get_status_symbol() {
    local status=$1
    case "$status" in
        "running"|"online"|"active"|"healthy")
            echo "$CHECKMARK"
            ;;
        "stopped"|"offline"|"inactive"|"down")
            echo "$CROSSMARK"
            ;;
        "warning"|"degraded"|"partial")
            echo "$WARNING"
            ;;
        *)
            echo "$INFO"
            ;;
    esac
}

check_service_status() {
    local port=$1
    local service_name=$2
    
    if lsof -Pi :$port -sTCP:LISTEN >/dev/null 2>&1; then
        echo "running"
    else
        echo "stopped"
    fi
}

check_service_health() {
    local port=$1
    local endpoint=$2
    
    if [[ -n "$endpoint" ]]; then
        if curl -s -f "http://localhost:$port$endpoint" >/dev/null 2>&1; then
            echo "healthy"
        else
            echo "unhealthy"
        fi
    else
        if nc -z localhost $port 2>/dev/null; then
            echo "healthy"
        else
            echo "unhealthy"
        fi
    fi
}

get_process_info() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null | head -n1)
    
    if [[ -n "$pid" ]]; then
        local cpu=$(ps -o pcpu= -p $pid 2>/dev/null | tr -d ' ')
        local memory=$(ps -o pmem= -p $pid 2>/dev/null | tr -d ' ')
        local start_time=$(ps -o lstart= -p $pid 2>/dev/null | xargs)
        
        echo "$pid|${cpu:-0.0}|${memory:-0.0}|$start_time"
    else
        echo "N/A|0.0|0.0|N/A"
    fi
}

get_system_stats() {
    # CPU usage
    local cpu_usage
    if command -v top >/dev/null 2>&1; then
        cpu_usage=$(top -l 1 -n 0 | grep "CPU usage" | awk '{print $3}' | sed 's/%//' 2>/dev/null || echo "0")
    elif command -v vmstat >/dev/null 2>&1; then
        cpu_usage=$(vmstat 1 2 | tail -1 | awk '{print 100-$15}' 2>/dev/null || echo "0")
    else
        cpu_usage="N/A"
    fi
    
    # Memory usage
    local memory_info
    if command -v free >/dev/null 2>&1; then
        memory_info=$(free | awk '/^Mem:/{printf "%.1f|%.1f|%.1f", $3/1024/1024, $2/1024/1024, $3/$2*100}')
    elif command -v vm_stat >/dev/null 2>&1; then
        local pages_used=$(vm_stat | grep "Pages active\|Pages inactive\|Pages speculative\|Pages occupied by compressor" | awk '{sum+=$3} END {print sum}')
        local pages_total=$(vm_stat | grep "Pages free\|Pages active\|Pages inactive\|Pages speculative\|Pages wired down\|Pages occupied by compressor" | awk '{sum+=$3} END {print sum}')
        local page_size=$(vm_stat | grep "page size" | awk '{print $8}')
        local used_gb=$(echo "scale=1; $pages_used * $page_size / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "0")
        local total_gb=$(echo "scale=1; $pages_total * $page_size / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "0")
        local usage_percent=$(echo "scale=1; $pages_used * 100 / $pages_total" | bc 2>/dev/null || echo "0")
        memory_info="$used_gb|$total_gb|$usage_percent"
    else
        memory_info="N/A|N/A|N/A"
    fi
    
    # Disk usage
    local disk_usage=$(df "$PROJECT_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')
    local disk_available=$(df -h "$PROJECT_DIR" | awk 'NR==2 {print $4}')
    
    # Load average
    local load_avg
    if command -v uptime >/dev/null 2>&1; then
        load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    else
        load_avg="N/A"
    fi
    
    # System uptime
    local system_uptime
    if command -v uptime >/dev/null 2>&1; then
        system_uptime=$(uptime | awk '{for(i=3;i<=NF;i++) printf "%s ", $i; print ""}' | sed 's/,$//')
    else
        system_uptime="N/A"
    fi
    
    echo "$cpu_usage|$memory_info|$disk_usage|$disk_available|$load_avg|$system_uptime"
}

get_network_stats() {
    # Network connections
    local active_connections=$(netstat -an 2>/dev/null | grep ESTABLISHED | wc -l | tr -d ' ')
    local listening_ports=$(netstat -an 2>/dev/null | grep LISTEN | wc -l | tr -d ' ')
    
    # AI Assistant specific connections
    local ollama_connections=$(netstat -an 2>/dev/null | grep ":$OLLAMA_PORT" | grep ESTABLISHED | wc -l | tr -d ' ')
    local api_connections=$(netstat -an 2>/dev/null | grep ":$API_PORT" | grep ESTABLISHED | wc -l | tr -d ' ')
    local gui_connections=$(netstat -an 2>/dev/null | grep ":$GUI_PORT" | grep ESTABLISHED | wc -l | tr -d ' ')
    
    echo "$active_connections|$listening_ports|$ollama_connections|$api_connections|$gui_connections"
}

display_header() {
    if [[ "$COMPACT_MODE" == true ]]; then
        echo -e "${CYAN}AI Assistant Monitor - $(get_timestamp)${NC}"
        return
    fi
    
    clear
    echo -e "${CYAN}╭─────────────────────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│                          AI Assistant Monitor                              │${NC}"
    echo -e "${CYAN}├─────────────────────────────────────────────────────────────────────────────┤${NC}"
    echo -e "${CYAN}│${NC} Last Updated: $(get_timestamp)                                      ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC} Refresh Interval: ${REFRESH_INTERVAL}s                                               ${CYAN}│${NC}"
    echo -e "${CYAN}╰─────────────────────────────────────────────────────────────────────────────╯${NC}"
    echo
}

display_service_status() {
    if [[ "$SHOW_SERVICES" != true ]]; then
        return
    fi
    
    echo -e "${WHITE}Service Status:${NC}"
    echo -e "${GRAY}──────────────${NC}"
    
    # Check each service
    local services=("Ollama:$OLLAMA_PORT:/api/tags" "API:$API_PORT:/health" "GUI:$GUI_PORT:" "Metrics:$METRICS_PORT:")
    
    for service_info in "${services[@]}"; do
        local service_name=$(echo "$service_info" | cut -d: -f1)
        local port=$(echo "$service_info" | cut -d: -f2)
        local endpoint=$(echo "$service_info" | cut -d: -f3)
        
        local status=$(check_service_status $port "$service_name")
        local health=$(check_service_health $port "$endpoint")
        local process_info=$(get_process_info $port)
        
        local pid=$(echo "$process_info" | cut -d'|' -f1)
        local cpu=$(echo "$process_info" | cut -d'|' -f2)
        local memory=$(echo "$process_info" | cut -d'|' -f3)
        
        local status_color=$(get_status_color "$status")
        local status_symbol=$(get_status_symbol "$status")
        local health_color=$(get_status_color "$health")
        local health_symbol=$(get_status_symbol "$health")
        
        if [[ "$COMPACT_MODE" == true ]]; then
            echo -e "${status_color}${status_symbol}${NC} $service_name (${status_color}$status${NC})"
        else
            printf "  %-8s ${status_color}%-7s${NC} ${health_color}%-9s${NC} Port: %-5s" \
                   "$service_name" "$status" "$health" "$port"
            if [[ "$pid" != "N/A" ]]; then
                printf " PID: %-6s CPU: %5s%% MEM: %5s%%" "$pid" "$cpu" "$memory"
            fi
            echo
        fi
    done
    echo
}

display_system_performance() {
    if [[ "$SHOW_PERFORMANCE" != true ]]; then
        return
    fi
    
    local stats=$(get_system_stats)
    local cpu_usage=$(echo "$stats" | cut -d'|' -f1)
    local memory_used=$(echo "$stats" | cut -d'|' -f2)
    local memory_total=$(echo "$stats" | cut -d'|' -f3)
    local memory_percent=$(echo "$stats" | cut -d'|' -f4)
    local disk_usage=$(echo "$stats" | cut -d'|' -f5)
    local disk_available=$(echo "$stats" | cut -d'|' -f6)
    local load_avg=$(echo "$stats" | cut -d'|' -f7)
    local system_uptime=$(echo "$stats" | cut -d'|' -f8)
    
    echo -e "${WHITE}System Performance:${NC}"
    echo -e "${GRAY}───────────────────${NC}"
    
    # CPU usage with color coding
    local cpu_color=$GREEN
    if (( $(echo "$cpu_usage > $ALERT_THRESHOLD_CPU" | bc -l 2>/dev/null || echo 0) )); then
        cpu_color=$RED
    elif (( $(echo "$cpu_usage > 60" | bc -l 2>/dev/null || echo 0) )); then
        cpu_color=$YELLOW
    fi
    
    # Memory usage with color coding
    local memory_color=$GREEN
    if (( $(echo "$memory_percent > $ALERT_THRESHOLD_MEMORY" | bc -l 2>/dev/null || echo 0) )); then
        memory_color=$RED
    elif (( $(echo "$memory_percent > 70" | bc -l 2>/dev/null || echo 0) )); then
        memory_color=$YELLOW
    fi
    
    # Disk usage with color coding
    local disk_color=$GREEN
    if [[ $disk_usage -gt $ALERT_THRESHOLD_DISK ]]; then
        disk_color=$RED
    elif [[ $disk_usage -gt 75 ]]; then
        disk_color=$YELLOW
    fi
    
    if [[ "$COMPACT_MODE" == true ]]; then
        echo -e "  CPU: ${cpu_color}${cpu_usage}%${NC} | MEM: ${memory_color}${memory_percent}%${NC} | DISK: ${disk_color}${disk_usage}%${NC}"
    else
        echo -e "  CPU Usage:       ${cpu_color}${cpu_usage}%${NC}"
        echo -e "  Memory:          ${memory_color}${memory_used}GB / ${memory_total}GB (${memory_percent}%)${NC}"
        echo -e "  Disk Usage:      ${disk_color}${disk_usage}%${NC} (${disk_available} available)"
        echo -e "  Load Average:    $load_avg"
        echo -e "  System Uptime:   $system_uptime"
    fi
    echo
}

display_network_status() {
    if [[ "$SHOW_NETWORK" != true ]]; then
        return
    fi
    
    local network_stats=$(get_network_stats)
    local active_connections=$(echo "$network_stats" | cut -d'|' -f1)
    local listening_ports=$(echo "$network_stats" | cut -d'|' -f2)
    local ollama_connections=$(echo "$network_stats" | cut -d'|' -f3)
    local api_connections=$(echo "$network_stats" | cut -d'|' -f4)
    local gui_connections=$(echo "$network_stats" | cut -d'|' -f5)
    
    echo -e "${WHITE}Network Status:${NC}"
    echo -e "${GRAY}───────────────${NC}"
    
    if [[ "$COMPACT_MODE" == true ]]; then
        echo -e "  Connections: $active_connections active | Ollama: $ollama_connections | API: $api_connections | GUI: $gui_connections"
    else
        echo -e "  Active Connections:  $active_connections"
        echo -e "  Listening Ports:     $listening_ports"
        echo -e "  Ollama Connections:  $ollama_connections"
        echo -e "  API Connections:     $api_connections"
        echo -e "  GUI Connections:     $gui_connections"
    fi
    echo
}

display_recent_logs() {
    if [[ "$SHOW_LOGS" != true ]]; then
        return
    fi
    
    echo -e "${WHITE}Recent Logs:${NC}"
    echo -e "${GRAY}────────────${NC}"
    
    local log_files=("$LOG_DIR/ollama.log" "$LOG_DIR/api.log" "$LOG_DIR/gui.log")
    
    for log_file in "${log_files[@]}"; do
        if [[ -f "$log_file" ]]; then
            local service_name=$(basename "$log_file" .log)
            echo -e "${BLUE}$service_name:${NC}"
            
            # Show last 3 lines of each log
            tail -n 3 "$log_file" 2>/dev/null | while read -r line; do
                echo -e "  ${GRAY}$line${NC}"
            done
            echo
        fi
    done
}

display_alerts() {
    local stats=$(get_system_stats)
    local cpu_usage=$(echo "$stats" | cut -d'|' -f1)
    local memory_percent=$(echo "$stats" | cut -d'|' -f4)
    local disk_usage=$(echo "$stats" | cut -d'|' -f5)
    
    local alerts=()
    
    # Check CPU alert
    if (( $(echo "$cpu_usage > $ALERT_THRESHOLD_CPU" | bc -l 2>/dev/null || echo 0) )); then
        alerts+=("High CPU usage: ${cpu_usage}%")
    fi
    
    # Check memory alert
    if (( $(echo "$memory_percent > $ALERT_THRESHOLD_MEMORY" | bc -l 2>/dev/null || echo 0) )); then
        alerts+=("High memory usage: ${memory_percent}%")
    fi
    
    # Check disk alert
    if [[ $disk_usage -gt $ALERT_THRESHOLD_DISK ]]; then
        alerts+=("High disk usage: ${disk_usage}%")
    fi
    
    # Check for stopped services
    local services=("Ollama:$OLLAMA_PORT" "API:$API_PORT" "GUI:$GUI_PORT")
    for service_info in "${services[@]}"; do
        local service_name=$(echo "$service_info" | cut -d: -f1)
        local port=$(echo "$service_info" | cut -d: -f2)
        local status=$(check_service_status $port "$service_name")
        
        if [[ "$status" == "stopped" ]]; then
            alerts+=("$service_name service is down")
        fi
    done
    
    if [[ ${#alerts[@]} -gt 0 ]]; then
        echo -e "${RED}${WARNING} Alerts:${NC}"
        echo -e "${GRAY}────────${NC}"
        for alert in "${alerts[@]}"; do
            echo -e "  ${RED}${CROSSMARK}${NC} $alert"
        done
        echo
    fi
}

output_json() {
    local stats=$(get_system_stats)
    local network_stats=$(get_network_stats)
    
    cat << EOF
{
    "timestamp": "$(date -Iseconds)",
    "system": {
        "cpu_usage": "$(echo "$stats" | cut -d'|' -f1)",
        "memory": {
            "used_gb": "$(echo "$stats" | cut -d'|' -f2)",
            "total_gb": "$(echo "$stats" | cut -d'|' -f3)",
            "usage_percent": "$(echo "$stats" | cut -d'|' -f4)"
        },
        "disk": {
            "usage_percent": "$(echo "$stats" | cut -d'|' -f5)",
            "available": "$(echo "$stats" | cut -d'|' -f6)"
        },
        "load_average": "$(echo "$stats" | cut -d'|' -f7)",
        "uptime": "$(echo "$stats" | cut -d'|' -f8)"
    },
    "services": {
        "ollama": {
            "status": "$(check_service_status $OLLAMA_PORT "Ollama")",
            "health": "$(check_service_health $OLLAMA_PORT "/api/tags")",
            "port": $OLLAMA_PORT
        },
        "api": {
            "status": "$(check_service_status $API_PORT "API")",
            "health": "$(check_service_health $API_PORT "/health")",
            "port": $API_PORT
        },
        "gui": {
            "status": "$(check_service_status $GUI_PORT "GUI")",
            "health": "$(check_service_health $GUI_PORT "")",
            "port": $GUI_PORT
        }
    },
    "network": {
        "active_connections": "$(echo "$network_stats" | cut -d'|' -f1)",
        "listening_ports": "$(echo "$network_stats" | cut -d'|' -f2)",
        "service_connections": {
            "ollama": "$(echo "$network_stats" | cut -d'|' -f3)",
            "api": "$(echo "$network_stats" | cut -d'|' -f4)",
            "gui": "$(echo "$network_stats" | cut -d'|' -f5)"
        }
    }
}
EOF
}

output_csv() {
    echo "timestamp,service,status,health,cpu,memory,connections"
    
    local timestamp=$(date -Iseconds)
    local services=("Ollama:$OLLAMA_PORT:/api/tags" "API:$API_PORT:/health" "GUI:$GUI_PORT:")
    
    for service_info in "${services[@]}"; do
        local service_name=$(echo "$service_info" | cut -d: -f1)
        local port=$(echo "$service_info" | cut -d: -f2)
        local endpoint=$(echo "$service_info" | cut -d: -f3)
        
        local status=$(check_service_status $port "$service_name")
        local health=$(check_service_health $port "$endpoint")
        local process_info=$(get_process_info $port)
        
        local cpu=$(echo "$process_info" | cut -d'|' -f2)
        local memory=$(echo "$process_info" | cut -d'|' -f3)
        local connections=$(netstat -an 2>/dev/null | grep ":$port" | grep ESTABLISHED | wc -l | tr -d ' ')
        
        echo "$timestamp,$service_name,$status,$health,$cpu,$memory,$connections"
    done
}

show_interactive_help() {
    echo -e "${WHITE}Keyboard Shortcuts:${NC}"
    echo -e "${GRAY}───────────────────${NC}"
    echo -e "  ${BLUE}q, Ctrl+C${NC}    Quit"
    echo -e "  ${BLUE}r${NC}            Refresh immediately"
    echo -e "  ${BLUE}l${NC}            Toggle log display"
    echo -e "  ${BLUE}p${NC}            Toggle performance display"
    echo -e "  ${BLUE}n${NC}            Toggle network display"
    echo -e "  ${BLUE}s${NC}            Toggle service display"
    echo -e "  ${BLUE}c${NC}            Toggle compact mode"
    echo -e "  ${BLUE}h, ?${NC}         Show this help"
    echo
}

handle_interactive_input() {
    # Check for input without blocking
    read -t 0.1 -n 1 key 2>/dev/null || return
    
    case "$key" in
        q|Q)
            echo -e "\n${GREEN}Monitoring stopped.${NC}"
            exit 0
            ;;
        r|R)
            return 0  # Force refresh
            ;;
        l|L)
            SHOW_LOGS=$([[ "$SHOW_LOGS" == true ]] && echo false || echo true)
            ;;
        p|P)
            SHOW_PERFORMANCE=$([[ "$SHOW_PERFORMANCE" == true ]] && echo false || echo true)
            ;;
        n|N)
            SHOW_NETWORK=$([[ "$SHOW_NETWORK" == true ]] && echo false || echo true)
            ;;
        s|S)
            SHOW_SERVICES=$([[ "$SHOW_SERVICES" == true ]] && echo false || echo true)
            ;;
        c|C)
            COMPACT_MODE=$([[ "$COMPACT_MODE" == true ]] && echo false || echo true)
            ;;
        h|H|?)
            show_interactive_help
            sleep 3
            ;;
    esac
}

# Main execution functions
run_once() {
    case "$OUTPUT_FORMAT" in
        "json")
            output_json
            ;;
        "csv")
            output_csv
            ;;
        *)
            display_header
            display_alerts
            display_service_status
            display_system_performance
            display_network_status
            display_recent_logs
            ;;
    esac
}

run_continuous() {
    # Set up terminal for interactive mode
    stty -echo -icanon time 0 min 0 2>/dev/null || true
    
    # Trap to restore terminal settings
    trap 'stty echo icanon 2>/dev/null || true; echo -e "\n${GREEN}Monitoring stopped.${NC}"; exit 0' INT TERM
    
    while true; do
        display_header
        display_alerts
        display_service_status
        display_system_performance
        display_network_status
        display_recent_logs
        
        if [[ "$COMPACT_MODE" != true ]]; then
            echo -e "${GRAY}Press 'h' for help, 'q' to quit${NC}"
        fi
        
        # Wait for refresh interval or user input
        local count=0
        while [[ $count -lt $REFRESH_INTERVAL ]]; do
            handle_interactive_input
            sleep 1
            ((count++))
        done
    done
}

# Main execution
main() {
    # Check if project directory exists
    if [[ ! -d "$PROJECT_DIR" ]]; then
        echo -e "${RED}Error: Project directory not found: $PROJECT_DIR${NC}"
        exit 1
    fi
    
    # Create log directory if needed
    mkdir -p "$LOG_DIR"
    
    # Install signal handlers
    trap 'echo -e "\n${GREEN}Monitoring stopped.${NC}"; exit 0' INT TERM
    
    if [[ "$CONTINUOUS_MODE" == true ]]; then
        run_continuous
    else
        run_once
    fi
}

# Run main function
main "$@"