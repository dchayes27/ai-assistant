# AI Assistant Deployment Scripts

Comprehensive deployment and management scripts for the AI Assistant project, including startup/shutdown, monitoring, updates, backups, and systemd service management.

## 📁 Script Overview

### Core Management Scripts

- **`start_all.sh`** - Launches all components with comprehensive health checks
- **`stop_all.sh`** - Gracefully stops all services with proper cleanup
- **`monitor.sh`** - Real-time system monitoring and status dashboard
- **`update.sh`** - Updates models, dependencies, and system components
- **`backup.sh`** - Creates comprehensive backups of database and configurations

### SystemD Integration

- **`install_systemd.sh`** - Installs systemd services for auto-start on boot
- **`systemd/`** directory contains service files for production deployment

## 🚀 Quick Start

### Basic Operations

```bash
# Start all services
./scripts/start_all.sh

# Monitor system status
./scripts/monitor.sh

# Stop all services
./scripts/stop_all.sh

# Create backup
./scripts/backup.sh --quick

# Update system
./scripts/update.sh --all
```

### Production Deployment

```bash
# Install systemd services (requires root)
sudo ./scripts/install_systemd.sh --all --start

# Check service status
sudo systemctl status ai-assistant

# View logs
sudo journalctl -u ai-assistant -f
```

## 📋 Detailed Script Documentation

### start_all.sh - Service Startup

Comprehensive startup script with health checks and dependency management.

#### Features
- **Dependency Checking** - Verifies all prerequisites before starting
- **Service Orchestration** - Starts services in correct order
- **Health Monitoring** - Validates service health after startup
- **Port Management** - Checks for port conflicts and handles restarts
- **Configuration Validation** - Ensures configs are valid before startup

#### Usage Examples

```bash
# Basic startup
./scripts/start_all.sh

# Force restart existing services
./scripts/start_all.sh --force

# Start in production mode
./scripts/start_all.sh --env production

# Skip health checks for faster startup
./scripts/start_all.sh --skip-health

# Verbose output with debugging
./scripts/start_all.sh --verbose
```

#### Startup Sequence
1. **Prerequisites Check** - Validates environment and dependencies
2. **Port Availability** - Checks for conflicts and handles restarts
3. **Ollama Service** - Starts LLM service and pulls default models
4. **API Server** - Launches FastAPI backend with health checks
5. **GUI Interface** - Starts Gradio web interface
6. **Monitoring Services** - Enables metrics and monitoring (optional)
7. **Health Validation** - Comprehensive health checks
8. **Status File Creation** - Creates runtime status information

#### Health Check Details
- **Ollama**: `GET http://localhost:11434/api/tags` (200 OK with model list)
- **API Server**: `GET http://localhost:8000/health` (status: "healthy")
- **GUI**: `GET http://localhost:7860` (Gradio interface loaded)

#### Configuration
- **Default Ports**: Ollama (11434), API (8000), GUI (7860), Metrics (8001)
- **Environment Detection**: Automatically detects dev/prod environment
- **Timeout Settings**: Configurable timeouts for service startup
- **Retry Logic**: Built-in retry mechanisms for robust startup

### stop_all.sh - Service Shutdown

Graceful shutdown script with proper cleanup and verification.

#### Features
- **Graceful Shutdown** - Sends SIGTERM for clean shutdown
- **Force Kill Option** - SIGKILL for unresponsive services
- **Cleanup Operations** - Removes PID files and temporary data
- **Verification** - Ensures all services are properly stopped
- **Flexible Options** - Various cleanup and timeout options

#### Usage Examples

```bash
# Graceful shutdown
./scripts/stop_all.sh

# Quiet mode (minimal output)
./scripts/stop_all.sh --quiet

# Force kill unresponsive services
./scripts/stop_all.sh --force

# Stop with cleanup
./scripts/stop_all.sh --cleanup-logs --cleanup-data

# Custom timeout
./scripts/stop_all.sh --timeout 60
```

#### Shutdown Sequence
1. **GUI Service** - Stops web interface first
2. **API Server** - Shuts down backend services  
3. **Metrics** - Stops monitoring services
4. **Ollama** - Gracefully stops LLM service
5. **Cleanup** - Removes PID files and temporary data
6. **Verification** - Confirms all services are stopped

### monitor.sh - System Monitoring

Real-time monitoring dashboard with comprehensive system metrics.

#### Features
- **Real-time Dashboard** - Live updating status display
- **Service Health** - Monitors all AI Assistant services
- **System Performance** - CPU, memory, disk, and network metrics
- **Interactive Controls** - Keyboard shortcuts for navigation
- **Multiple Output Formats** - Dashboard, JSON, CSV outputs
- **Alerting** - Configurable thresholds for resource alerts

#### Usage Examples

```bash
# Real-time dashboard
./scripts/monitor.sh

# Quick status check
./scripts/monitor.sh --status

# JSON output for automation
./scripts/monitor.sh --json

# Compact mode with custom refresh
./scripts/monitor.sh --compact --interval 2

# Performance metrics only
./scripts/monitor.sh --performance

# CSV output for logging
./scripts/monitor.sh --csv >> monitoring.log
```

#### Interactive Dashboard

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                          AI Assistant Monitor                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Last Updated: 2024-01-15 14:30:45                                         │
│ Refresh Interval: 5s                                                       │
╰─────────────────────────────────────────────────────────────────────────────╯

Service Status:
──────────────
  Ollama    running   healthy   Port: 11434  PID: 12345  CPU:  15.2% MEM:   8.1%
  API       running   healthy   Port: 8000   PID: 12346  CPU:   5.1% MEM:   4.2%
  GUI       running   healthy   Port: 7860   PID: 12347  CPU:   2.3% MEM:   3.1%

System Performance:
───────────────────
  CPU Usage:       25.4%
  Memory:          4.2GB / 16.0GB (26.3%)
  Disk Usage:      45% (120GB available)
  Load Average:    1.2
  System Uptime:   3d 12h 45m

Keyboard Shortcuts: q=quit, r=refresh, l=logs, p=performance, h=help
```

#### Keyboard Controls
- **q, Ctrl+C** - Quit monitoring
- **r** - Force immediate refresh
- **l** - Toggle log display
- **p** - Toggle performance metrics
- **n** - Toggle network information
- **s** - Toggle service status
- **c** - Toggle compact mode
- **h, ?** - Show help

### update.sh - System Updates

Comprehensive update system for models, dependencies, and components.

#### Features
- **Selective Updates** - Choose what to update (Python, models, system)
- **Backup Integration** - Automatic backup before updates
- **Version Management** - Tracks and manages component versions
- **Rollback Capability** - Can revert to previous versions
- **Service Management** - Handles service restarts during updates

#### Usage Examples

```bash
# Update everything
./scripts/update.sh --all

# Python dependencies only
./scripts/update.sh --python-only

# AI models only
./scripts/update.sh --models-only

# Force update with auto-restart
./scripts/update.sh --force --auto-restart

# Preview updates without executing
./scripts/update.sh --dry-run

# Update specific components
./scripts/update.sh --ollama-models --whisper-models
```

#### Update Categories

**Python Dependencies**
- Updates pip and all Python packages
- Security vulnerability scanning
- Requirements.txt and lock file generation

**AI Models**
- Ollama model updates (LLM models)
- Whisper model updates (speech-to-text)
- TTS model updates (text-to-speech)

**System Packages**
- Operating system packages (Homebrew, APT, YUM)
- System dependencies and libraries

#### Update Process
1. **Pre-flight Checks** - Validates system state
2. **Backup Creation** - Creates backup before changes
3. **Service Management** - Stops services if needed
4. **Component Updates** - Updates selected components
5. **Verification** - Validates updated components
6. **Service Restart** - Restarts services if configured
7. **Report Generation** - Creates detailed update report

### backup.sh - Backup System

Comprehensive backup solution with multiple backup types and encryption.

#### Features
- **Multiple Backup Types** - Full, quick, config-only, database-only
- **Compression** - Automatic compression with multiple algorithms
- **Encryption** - GPG encryption for sensitive data
- **Remote Backup** - Automatic upload to remote locations
- **Retention Management** - Automatic cleanup of old backups
- **Verification** - Backup integrity verification

#### Usage Examples

```bash
# Full backup
./scripts/backup.sh --full

# Quick backup (essentials only)
./scripts/backup.sh --quick

# Configuration files only
./scripts/backup.sh --config-only

# Encrypted backup with remote upload
./scripts/backup.sh --encryption --remote server.com

# Custom backup with retention
./scripts/backup.sh --name "pre-update" --retention 90

# Database backup only
./scripts/backup.sh --database-only --no-compression
```

#### Backup Types

**Full Backup**
- Database files and SQL dumps
- Configuration files
- AI model information
- System state and environment info
- Log files (if enabled)

**Quick Backup**
- Database files
- Configuration files
- System state
- Essential runtime information

**Selective Backups**
- Config-only: Just configuration files
- Database-only: Database files and dumps
- Models-only: AI model metadata and cache info

#### Backup Process
1. **Prerequisites Check** - Validates backup environment
2. **Size Estimation** - Estimates backup size
3. **Content Collection** - Gathers files based on backup type
4. **Manifest Creation** - Creates detailed backup manifest
5. **Compression** - Compresses backup if enabled
6. **Encryption** - Encrypts backup if configured
7. **Verification** - Validates backup integrity
8. **Remote Upload** - Uploads to remote location if configured
9. **Cleanup** - Removes old backups based on retention policy

#### Backup Structure
```
backup_20240115_143045/
├── manifest.json           # Backup metadata and file listing
├── database/               # Database files and SQL dumps
│   ├── assistant.db
│   └── assistant_dump.sql
├── config/                 # Configuration files
│   ├── config.yaml
│   ├── voice_profiles.yaml
│   └── prompt_templates.yaml
├── models/                 # AI model information
│   ├── ollama/
│   ├── whisper/
│   └── tts/
├── system/                 # System state information
│   ├── system_info.txt
│   ├── python_env.txt
│   └── git_info.txt
└── logs/                   # Log files (if enabled)
    ├── ollama.log
    ├── api.log
    └── gui.log
```

## 🔧 SystemD Integration

### Service Architecture

The systemd integration provides production-ready service management with proper dependencies, security, and resource limits.

#### Service Hierarchy
```
ai-assistant.service (main service)
├── ai-assistant-ollama.service (LLM service)
├── ai-assistant-api.service (API server)
└── ai-assistant-gui.service (web interface)

ai-assistant-backup.timer (scheduled backups)
└── ai-assistant-backup.service (backup execution)
```

### Installation

```bash
# Install all services
sudo ./scripts/install_systemd.sh --all --start

# Install specific service
sudo ./scripts/install_systemd.sh --ollama

# Install with custom options
sudo ./scripts/install_systemd.sh --services-only --no-copy
```

### Service Management

```bash
# Basic operations
sudo systemctl start ai-assistant
sudo systemctl stop ai-assistant
sudo systemctl restart ai-assistant
sudo systemctl status ai-assistant

# Individual services
sudo systemctl start ai-assistant-ollama
sudo systemctl status ai-assistant-api

# Enable/disable auto-start
sudo systemctl enable ai-assistant
sudo systemctl disable ai-assistant

# View logs
sudo journalctl -u ai-assistant -f
sudo journalctl -u ai-assistant-ollama --since today
```

### Service Features

**Security Hardening**
- Dedicated service user with minimal privileges
- Filesystem restrictions and protections
- Network access controls
- Memory and CPU limits

**Resource Management**
- Memory limits per service
- CPU quota allocation
- File descriptor limits
- Process limits

**Monitoring Integration**
- Systemd journal logging
- Health check endpoints
- Automatic restart on failure
- Service dependency management

## 🎛️ Configuration

### Environment Variables

Common environment variables used across scripts:

```bash
# Core settings
export AI_ASSISTANT_ENV=production
export AI_ASSISTANT_LOG_LEVEL=INFO
export AI_ASSISTANT_NO_BROWSER=1

# Service configuration
export AI_ASSISTANT_PORT=8000
export AI_ASSISTANT_HOST=0.0.0.0

# Database settings
export AI_ASSISTANT_DB_PATH=/opt/ai-assistant/data/assistant.db

# LLM settings
export AI_ASSISTANT_LLM_URL=http://localhost:11434
export AI_ASSISTANT_LLM_MODEL=llama3.2:latest

# Security
export AI_ASSISTANT_API_KEY=your-secret-key
export AI_ASSISTANT_JWT_SECRET=your-jwt-secret

# Backup settings
export AI_ASSISTANT_BACKUP_DIR=/opt/ai-assistant/backups
export AI_ASSISTANT_GPG_KEY=backup@example.com
```

### Default Ports

| Service | Port | Description |
|---------|------|-------------|
| Ollama | 11434 | LLM service API |
| API Server | 8000 | FastAPI backend |
| GUI | 7860 | Gradio web interface |
| Metrics | 8001 | Prometheus metrics (optional) |

### File Locations

| Type | Location | Description |
|------|----------|-------------|
| Project | `/opt/ai-assistant` | Main installation directory |
| Logs | `/opt/ai-assistant/logs` | Service log files |
| Data | `/opt/ai-assistant/data` | Database files |
| Config | `/opt/ai-assistant/config` | Configuration files |
| Backups | `/opt/ai-assistant/backups` | Backup storage |
| SystemD | `/etc/systemd/system` | Service definitions |

## 🚨 Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check what's using a port
sudo lsof -i :8000

# Kill process using port
sudo kill $(lsof -ti:8000)

# Use different port
export AI_ASSISTANT_PORT=8080
```

#### Permission Issues
```bash
# Fix ownership
sudo chown -R ai-assistant:ai-assistant /opt/ai-assistant

# Fix permissions
sudo chmod -R 755 /opt/ai-assistant
sudo chmod +x /opt/ai-assistant/scripts/*.sh
```

#### Service Start Failures
```bash
# Check service status
sudo systemctl status ai-assistant

# View detailed logs
sudo journalctl -u ai-assistant -n 50

# Check configuration
sudo systemd-analyze verify /etc/systemd/system/ai-assistant.service
```

#### Database Issues
```bash
# Check database permissions
ls -la /opt/ai-assistant/data/

# Test database connectivity
sqlite3 /opt/ai-assistant/data/assistant.db ".tables"

# Backup and restore database
./scripts/backup.sh --database-only
```

### Debug Mode

Enable debug mode for verbose logging:

```bash
# For scripts
export AI_ASSISTANT_LOG_LEVEL=DEBUG

# For services
sudo systemctl edit ai-assistant
# Add: Environment=AI_ASSISTANT_LOG_LEVEL=DEBUG
```

### Log Analysis

```bash
# View all AI Assistant logs
sudo journalctl -u "ai-assistant*" --since today

# Follow logs in real-time
sudo journalctl -u ai-assistant -f

# Export logs for analysis
sudo journalctl -u ai-assistant --since "1 hour ago" > debug.log
```

## 🔄 Maintenance

### Regular Maintenance Tasks

**Daily**
- Monitor service health with `monitor.sh`
- Check log files for errors
- Verify backup completion

**Weekly**
- Run `update.sh --python-only` for security updates
- Clean up old log files
- Review resource usage

**Monthly**
- Run `update.sh --all` for comprehensive updates
- Test backup and restore procedures
- Review and rotate API keys

**Quarterly**
- Update AI models with `update.sh --models-only`
- Review and update configuration
- Security audit and hardening review

### Automated Maintenance

Set up cron jobs for automated maintenance:

```bash
# Daily backup (already handled by systemd timer)
0 2 * * * /opt/ai-assistant/scripts/backup.sh --quick

# Weekly updates
0 3 * * 0 /opt/ai-assistant/scripts/update.sh --python-only

# Monthly cleanup
0 4 1 * * find /opt/ai-assistant/logs -name "*.log" -mtime +30 -delete
```

## 📊 Monitoring and Alerting

### Health Checks

All scripts include comprehensive health checking:

- **Service Status** - Process and port monitoring
- **Resource Usage** - CPU, memory, disk monitoring
- **Connectivity** - Network and API endpoint testing
- **Performance** - Response time and throughput monitoring

### Integration with Monitoring Systems

Export metrics for external monitoring:

```bash
# Prometheus metrics
curl http://localhost:8001/metrics

# JSON status for automation
./scripts/monitor.sh --json | jq .

# CSV for logging systems
./scripts/monitor.sh --csv >> /var/log/ai-assistant-metrics.csv
```

### Alerting

Set up alerts based on script exit codes and metrics:

```bash
# Check service health
if ! ./scripts/monitor.sh --status --quiet; then
    echo "AI Assistant service unhealthy" | mail admin@example.com
fi

# Resource monitoring
CPU_USAGE=$(./scripts/monitor.sh --json | jq -r '.system.cpu_usage')
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "High CPU usage: $CPU_USAGE%" | mail admin@example.com
fi
```

## 🔐 Security Considerations

### Service Security

- **Dedicated User** - Services run as non-privileged `ai-assistant` user
- **Filesystem Restrictions** - Limited read/write access via systemd
- **Network Controls** - Restricted network access where possible
- **Resource Limits** - CPU, memory, and process limits

### Data Protection

- **Encryption** - Backup encryption with GPG
- **Access Controls** - Proper file permissions and ownership
- **Secrets Management** - Environment variables for sensitive data
- **Audit Logging** - Comprehensive logging for security events

### Best Practices

1. **Regular Updates** - Keep all components up to date
2. **Backup Encryption** - Always encrypt sensitive backups
3. **Access Control** - Limit SSH and service access
4. **Monitoring** - Monitor for unusual activity
5. **Incident Response** - Have procedures for security incidents

## 🤝 Contributing

When contributing to the deployment scripts:

1. **Test Thoroughly** - Test on multiple environments
2. **Document Changes** - Update this README and script help
3. **Follow Conventions** - Use consistent coding style
4. **Error Handling** - Include comprehensive error handling
5. **Security** - Consider security implications of changes

## 📄 License

These deployment scripts are part of the AI Assistant project and follow the same licensing terms.