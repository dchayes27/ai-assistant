# Troubleshooting Guide

Common issues and solutions for the AI Assistant.

---

## Installation Issues

### Python Version Mismatch

**Symptoms:**
```
ERROR: Python 3.8 is installed, but Python 3.9+ is required
```

**Solution:**
```bash
# Check Python version
python --version

# Install Python 3.9+ and create virtual environment
python3.9 -m venv venv
source venv/bin/activate
```

### Dependency Conflicts

**Symptoms:**
```
ERROR: Cannot install torch==2.5.1 and numpy==1.26.4 (version conflict)
```

**Solution:**
```bash
# Clear pip cache and reinstall
pip cache purge
pip install --no-cache-dir -r requirements.txt

# Or use fresh virtual environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
./install_dependencies.sh
```

### FFmpeg Not Found

**Symptoms:**
```
ERROR: ffmpeg not found in PATH
RuntimeError: Failed to initialize audio processing
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

---

## Runtime Issues

### Port Already in Use

**Symptoms:**
```
ERROR: Address already in use: 0.0.0.0:8000
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port
lsof -i :8000  # Check API port
lsof -i :7860  # Check GUI port
lsof -i :11434 # Check Ollama port

# Kill process
kill -9 <PID>

# Or use stop script
./scripts/stop_all.sh
```

### Ollama Not Responding

**Symptoms:**
```
ERROR: Failed to connect to Ollama at http://localhost:11434
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Or use start script
./scripts/start_all.sh

# Check Ollama logs
journalctl -u ollama -f  # If using systemd
```

### Database Locked

**Symptoms:**
```
sqlite3.OperationalError: database is locked
ERROR: Could not acquire database lock
```

**Solution:**
```bash
# Check for stale lock
lsof | grep assistant.db

# Stop all services
./scripts/stop_all.sh

# Remove lock file if exists
rm -f ~/ai-assistant/memory/assistant.db-wal
rm -f ~/ai-assistant/memory/assistant.db-shm

# Restart services
./scripts/start_all.sh
```

### Audio Device Not Found

**Symptoms:**
```
OSError: No Default Input Device Available
ERROR: Failed to initialize audio device
```

**Solution:**
```bash
# Check available audio devices
python -c "import pyaudio; p=pyaudio.PyAudio(); print([p.get_device_info_by_index(i) for i in range(p.get_device_count())])"

# On Linux, check ALSA
aplay -l
arecord -l

# Install audio libraries if missing
sudo apt-get install portaudio19-dev python3-pyaudio
pip install --force-reinstall pyaudio
```

### Memory/Performance Issues

**Symptoms:**
- Slow response times (>5 seconds)
- High memory usage (>4GB)
- System freezing during processing

**Solution:**
```bash
# Check memory usage
./scripts/monitor.sh

# Reduce model sizes in config/config.yaml
# whisper_model: "tiny" instead of "medium"
# ollama_model: "llama3.2:3b" instead of "7b"

# Reduce context length
# max_context_length: 10 instead of 20

# Check for memory leaks
ps aux | grep python
```

---

## Configuration Issues

### Environment Variables Not Loaded

**Symptoms:**
```
WARNING: JWT_SECRET_KEY not set, using generated key
ERROR: Configuration value missing
```

**Solution:**
```bash
# Create .env file from example
cp .env.example .env

# Edit .env with your values
vi .env

# Export environment variables
export $(cat .env | xargs)

# Or use with start script
./scripts/start_all.sh --env production
```

### Model Not Found

**Symptoms:**
```
ERROR: Model 'llama3.2:3b' not found
ERROR: Model 'medium' not available for Whisper
```

**Solution:**
```bash
# List Ollama models
ollama list

# Pull missing model
ollama pull llama3.2:3b

# For Whisper, model downloads automatically on first use
# Or pre-download:
python -c "import whisper; whisper.load_model('medium')"
```

### Invalid YAML Syntax

**Symptoms:**
```
yaml.scanner.ScannerError: mapping values are not allowed here
ERROR: Failed to parse config/config.yaml
```

**Solution:**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# Common issues:
# - Tabs instead of spaces
# - Missing quotes around special characters
# - Incorrect indentation
```

### Permission Denied

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'data/'
ERROR: Cannot write to log file
```

**Solution:**
```bash
# Fix directory permissions
sudo chown -R $USER:$USER ~/ai-assistant/
chmod -R 755 ~/ai-assistant/

# Create missing directories
mkdir -p data/ logs/ temp/
chmod 755 data/ logs/ temp/
```

---

## Testing Issues

### Tests Hanging

**Symptoms:**
- Tests never complete
- Stuck on "waiting for Ollama"
- Timeout errors

**Solution:**
```bash
# Skip tests requiring Ollama
./run_tests.sh -m "not requires_ollama"

# Set shorter timeouts
pytest --timeout=10

# Run in verbose mode to see where it hangs
pytest -vvv --tb=short
```

### Database Fixture Conflicts

**Symptoms:**
```
ERROR: Database fixture already in use
sqlite3.OperationalError: database is locked
```

**Solution:**
```bash
# Use separate test database
export TEST_MODE=true
pytest --db-isolation

# Clean test artifacts
rm -rf .pytest_cache/
find . -name "*.db" -path "*/tests/*" -delete
```

### Import Errors in Tests

**Symptoms:**
```
ImportError: cannot import name 'SmartAssistant'
ModuleNotFoundError: No module named 'core'
```

**Solution:**
```bash
# Install package in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Reinstall dependencies
pip install -r requirements.txt
```

---

## API Issues

### 401 Unauthorized

**Symptoms:**
```
{"detail": "Authentication required"}
Status: 401 Unauthorized
```

**Solution:**
```bash
# Get API token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-api-key"}'

# Use token in requests
curl http://localhost:8000/agent/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### 500 Internal Server Error

**Symptoms:**
```
{"detail": "Internal server error"}
Status: 500
```

**Solution:**
```bash
# Check API server logs
tail -f logs/mcp_server.log

# Enable debug mode
export AI_ASSISTANT_DEBUG=true
./scripts/start_all.sh

# Check full error traceback in logs
```

### WebSocket Connection Failed

**Symptoms:**
```
WebSocket connection failed: Error during handshake
Status: 403 Forbidden
```

**Solution:**
```bash
# Ensure authentication token in connection
ws://localhost:8000/ws/connection_id?token=YOUR_TOKEN

# Check CORS settings in config
# server.cors_origins in config/config.yaml
```

---

## GUI Issues

### Gradio Interface Not Loading

**Symptoms:**
- Browser shows "Unable to connect"
- Port 7860 not responding

**Solution:**
```bash
# Check if GUI is running
curl http://localhost:7860

# Check GUI logs
tail -f logs/gui.log

# Restart GUI
pkill -f "gui.app"
python -m gui.app
```

### Voice Input Not Working

**Symptoms:**
- Microphone button doesn't record
- "Audio device error" message

**Solution:**
```bash
# Grant microphone permissions (browser)
# Allow microphone access in browser settings

# Check audio device availability
python -c "import pyaudio; print(pyaudio.PyAudio().get_default_input_device_info())"

# Test with different browser (Chrome, Firefox, Safari)
```

---

## Database Issues

### Migration Failed

**Symptoms:**
```
ERROR: Migration failed at step 003
AlembicError: Can't locate revision
```

**Solution:**
```bash
# Check current migration version
python -c "from memory.migrations import MigrationManager; m = MigrationManager(); print(m.get_current_version())"

# Reset migrations (WARNING: data loss)
python -m memory.migrations reset
python -m memory.migrations upgrade

# Or restore from backup
python -m memory.backup restore latest
```

### Query Performance Degradation

**Symptoms:**
- Slow database queries (>1 second)
- High CPU usage during queries

**Solution:**
```bash
# Rebuild indexes
sqlite3 ~/ai-assistant/memory/assistant.db << EOF
REINDEX;
ANALYZE;
VACUUM;
EOF

# Check database size
du -h ~/ai-assistant/memory/assistant.db

# Clean old data
python -c "from memory import DatabaseManager; db = DatabaseManager(); db.cleanup_old_data(days=30)"
```

---

## General Debugging

### Enable Debug Logging

```bash
# Set debug environment variables
export AI_ASSISTANT_DEBUG=true
export AI_ASSISTANT_LOG_LEVEL=DEBUG

# Restart services
./scripts/stop_all.sh
./scripts/start_all.sh --verbose
```

### Check System Requirements

```bash
# Verify all dependencies
python --version  # Should be 3.9+
ollama --version  # Should be installed
ffmpeg -version   # Should be installed
sqlite3 --version # Should be 3.x

# Check available resources
free -h           # RAM
df -h            # Disk space
nproc            # CPU cores
```

### Collect Debug Information

When reporting issues, include:

```bash
# System info
uname -a
python --version
pip list | grep -E "torch|transformers|fastapi|gradio"

# Service status
./scripts/monitor.sh

# Recent logs
tail -n 100 logs/mcp_server.log
tail -n 100 logs/gui.log

# Error messages
grep ERROR logs/*.log | tail -20
```

---

## Getting Help

If none of these solutions work:

1. **Check logs** in `logs/` directory
2. **Search issues** on [GitHub](https://github.com/dchayes27/ai-assistant/issues)
3. **Create new issue** with debug information above
4. **Ask in discussions** for usage questions

For immediate help, see [CONTRIBUTING.md](CONTRIBUTING.md) for contact information.
