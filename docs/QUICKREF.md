# Quick Reference

Quick command reference for daily use of the AI Assistant.

---

## Starting/Stopping

```bash
./scripts/start_all.sh              # Start all services
./start_gui.sh                      # Start GUI only
./scripts/stop_all.sh               # Stop everything
./scripts/monitor.sh                # Check status
```

**Service Startup Options:**
```bash
./scripts/start_all.sh --force              # Force restart existing services
./scripts/start_all.sh --env production     # Production mode
./scripts/start_all.sh --verbose            # Debug output
```

---

## Testing

```bash
./run_tests.sh                      # All tests
./run_tests.sh --unit --fast        # Quick unit tests
./run_tests.sh --coverage           # With coverage
./run_tests.sh --integration        # Integration tests (needs Ollama)
./run_tests.sh --performance        # Performance benchmarks
```

---

## Configuration

```bash
export AI_ASSISTANT_ENV=production  # Change environment
vi config/config.yaml               # Edit main config
vi .env                             # Edit secrets
```

**Environment Variables:**
- `AI_ASSISTANT_ENV` - Environment (development/production/testing)
- `AI_ASSISTANT_DEBUG` - Enable debug mode
- `AI_ASSISTANT_LOG_LEVEL` - Set log level (DEBUG/INFO/WARNING/ERROR)
- `JWT_SECRET_KEY` - JWT signing secret (required in production)

---

## Database

```bash
# Access database
sqlite3 ~/ai-assistant/memory/assistant.db

# Common queries
.tables                             # List tables
SELECT * FROM conversations;        # View conversations
SELECT * FROM messages LIMIT 10;    # View recent messages
.schema conversations               # View table schema
```

---

## Logs

```bash
tail -f logs/gui.log                # GUI logs
tail -f logs/mcp_server.log         # API logs
tail -f logs/assistant.log          # Main assistant logs
journalctl -u ai-assistant -f       # Systemd logs (production)
```

**Log Locations:**
- `logs/` - Default log directory
- `logs/archive/` - Rotated logs

---

## Development

```bash
# Code quality
python -m black .                   # Format code
python -m flake8 .                  # Lint
python -m mypy core/ memory/        # Type check

# Pre-commit hooks
pre-commit install                  # Install hooks
pre-commit run --all-files          # Run on all files
```

---

## Common URLs

When services are running:

- **GUI**: http://localhost:7860
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **API Redoc**: http://localhost:8000/redoc
- **Ollama**: http://localhost:11434
- **Metrics** (if enabled): http://localhost:8001

---

## Troubleshooting

```bash
# Check service status
./scripts/monitor.sh

# Check port usage
lsof -i :7860                       # GUI port
lsof -i :8000                       # API port
lsof -i :11434                      # Ollama port

# View recent errors
tail -n 50 logs/mcp_server.log | grep ERROR
tail -n 50 logs/gui.log | grep ERROR

# Restart everything
./scripts/stop_all.sh && ./scripts/start_all.sh --force
```

---

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "feat: description"

# Push and create PR
git push -u origin feature/my-feature
```

---

## Backup & Restore

```bash
# Create backup
python -m memory.backup create

# Restore from backup
python -m memory.backup restore backup_20251027.tar.gz

# List backups
ls -lh ~/ai-assistant/memory/backups/
```

---

For detailed documentation, see:
- [CURRENT_STATE.md](CURRENT_STATE.md) - Project overview
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide
