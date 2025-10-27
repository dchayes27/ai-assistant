# Documentation Cleanup Guide

**Last Updated**: 2025-10-27
**Purpose**: Track documentation issues and needed improvements

---

## 🗑️ Remove

### README.md

#### Lines 89-117: Outdated Project Structure
**Why**: The directory structure diagram is completely incorrect and misleading.

**Current (Wrong)**:
```
core/
├── main.py
├── llm.py
├── speech.py
└── tts.py
memory/
├── database.py
├── context.py
└── models.py
mcp_server/
├── server.py
└── tools.py
```

**Actual Structure**:
```
core/
├── smart_assistant.py
└── tool_manager.py
memory/
├── db_manager.py
├── backup.py
├── migrations.py
├── models.py
└── vector_store.py
mcp_server/
├── main.py
├── auth.py
├── models.py
├── ollama_client.py
└── tools.py
```

**Action**: Remove the entire "Project Structure" section from README.md (lines 89-117). Reference docs/CURRENT_STATE.md instead for accurate structure.

### CLAUDE.md

#### Lines 532-540: MCP Git Integration Claims
**File**: CLAUDE.md
**Section**: "## GitHub Repository and CI/CD" → "### Using MCP Git Integration"

**Why**: Claims that "git MCP server is configured" and shows commands like `claude mcp list`, but MCP git integration is NOT actually available in the current session (verified in conversation).

**Action**: Remove or rewrite this section to reflect that git operations use built-in Bash tool, not MCP servers.

---

## ✏️ Update

### README.md

#### Line 62: Non-existent Setup Script
**Current**:
```bash
python scripts/setup.py
```

**Problem**: `scripts/setup.py` does not exist.

**Fix**: Replace with actual installation command:
```bash
./install_dependencies.sh
```

#### Lines 68-80: Invalid Usage Commands
**Current**:
```bash
python -m core.main           # Won't work - no core/main.py
python -m gui.app             # Works
python -m mcp_server.server   # Won't work - no server.py
```

**Fix**: Update to actual working commands:
```bash
# Start all services
./scripts/start_all.sh

# Start GUI only
./start_gui.sh
# OR
python -m gui.app

# Start API server only
python -m mcp_server.main
```

#### Line 84: Wrong Configuration File Reference
**Current**:
```
See `config.example.yaml` for available options.
```

**Problem**: `config.example.yaml` doesn't exist. Project uses `.env.example` and `config/config.yaml`.

**Fix**:
```
Configuration is managed through YAML files in the `config/` directory and environment variables. See `.env.example` for environment settings and `config/config.yaml` for application configuration.
```

#### Add Missing Quick Start Section
**Location**: After "## Installation" (after line 63)

**Add**:
```markdown
## Quick Start

```bash
# 1. Start all services (Ollama + API + GUI)
./scripts/start_all.sh

# 2. Open GUI in browser
# Navigate to http://localhost:7860

# 3. Stop services when done
./scripts/stop_all.sh
```

For detailed usage, see [CURRENT_STATE.md](docs/CURRENT_STATE.md).
```

### STREAMING_REQUIREMENTS.md

#### Line 1: Misleading Title
**Current**: `# Real-Time Voice Assistant Streaming Requirements`

**Problem**: This reads like current requirements, but it's a planning document for future work.

**Fix**: Rename file to `STREAMING_ROADMAP.md` and update title:
```markdown
# Real-Time Voice Assistant Streaming Roadmap (PLANNED)

**Status**: Planning Phase - Not Yet Implemented
**Target**: Future enhancement to enable real-time streaming voice interactions
```

**Add at top**:
```markdown
> ⚠️ **Note**: This is a planning document for future development. The features described here are NOT yet implemented. For current capabilities, see [CURRENT_STATE.md](docs/CURRENT_STATE.md).
```

### CLAUDE.md

#### Lines 47-56: Test Runner Documentation Incomplete
**Current**: Shows `./run_tests.sh` options but doesn't mention that some flags require external services.

**Add after line 56**:
```bash
# Note: Some tests require external services
./run_tests.sh --requires-ollama    # Requires Ollama running on localhost:11434
./run_tests.sh --audio              # Requires audio hardware/drivers
```

#### Lines 532-580: GitHub/MCP Section
**Current**: Mixes GitHub Actions info with incorrect MCP claims.

**Fix**: Split into two clear sections:

1. **GitHub Actions CI/CD** (keep lines 541-580, update heading)
2. Remove "Using MCP Git Integration" subsection entirely

**Replace lines 532-540 with**:
```markdown
### Git Operations in Claude Code

Claude Code performs git operations using the built-in Bash tool:
- Can execute git commands directly (commit, push, branch, etc.)
- No MCP git server integration in current session
- Uses standard git CLI for all operations
```

### scripts/README.md

#### Line 88: Startup Sequence - Missing Health Check Details
**Current**: Lists "7. Health Validation - Comprehensive health checks"

**Problem**: Doesn't specify what endpoints are checked or what constitutes "healthy".

**Add after line 94**:
```markdown
#### Health Check Details
- **Ollama**: `GET http://localhost:11434/api/tags` (200 OK with model list)
- **API Server**: `GET http://localhost:8000/health` (status: "healthy")
- **GUI**: `GET http://localhost:7860` (Gradio interface loaded)
```

### config/README.md

#### Lines 67-74: Example App Settings
**Problem**: Shows `debug: false` as default, but development config likely has `debug: true`.

**Fix**: Clarify that these are production defaults:
```yaml
app:
  name: "AI Assistant"
  version: "1.0.0"
  debug: false  # Production default (true in development)
  log_level: "INFO"  # Production default (DEBUG in development)
  # ... rest of config
```

### tests/README.md

#### Lines 98-109: Conditional Test Execution
**Current**: Shows markers but doesn't explain WHEN tests are skipped.

**Add after line 109**:
```markdown
**Automatic Skipping**:
- `@pytest.mark.requires_ollama` tests are skipped if Ollama is not running on localhost:11434
- `@pytest.mark.requires_audio` tests are skipped if PyAudio/audio hardware is unavailable
- Use markers to explicitly include/exclude these tests in CI/CD environments
```

---

## ➕ Add

### Missing: docs/ARCHITECTURE.md

**Why**: Detailed architecture explanation scattered across multiple files. Need single source of truth.

**Should Include**:
1. **System Architecture Diagram** (ASCII art or mermaid)
   - Show data flow between components
   - Highlight async operations
   - Show external dependencies (Ollama, FFmpeg, SQLite)

2. **Component Interactions**
   - How GUI calls core SmartAssistant
   - How MCP server provides API layer
   - How memory system integrates with all components
   - Request/response flow for voice interactions

3. **Data Models**
   - Conversation structure
   - Message format
   - Vector embeddings storage
   - Project/knowledge organization

4. **Technology Decisions**
   - Why Ollama (local-first, privacy)
   - Why SQLite (embedded, FTS5, portability)
   - Why Gradio (rapid prototyping, built-in components)
   - Why FastAPI (async, OpenAPI, modern Python)

### Missing: docs/API.md

**Why**: API documentation only available at `/docs` endpoint when server running. Need offline reference.

**Should Include**:
1. **REST Endpoints**
   - Authentication endpoints (`/auth/login`, `/auth/refresh`)
   - Agent endpoints (`/agent/query`, `/agent/stream`)
   - Memory endpoints (`/memory/search`, `/memory/save`)
   - Tool endpoints (`/tools/list`, `/tools/execute`)
   - Conversation endpoints (`/conversations`, `/conversations/{id}/messages`)

2. **WebSocket Interface**
   - Connection: `ws://localhost:8000/ws/{connection_id}`
   - Message formats (query, ping, response_chunk)
   - Event types and handling

3. **Request/Response Examples**
   - Complete curl examples for each endpoint
   - Python client examples using httpx
   - Error responses and status codes

4. **Authentication**
   - JWT token format
   - API key usage
   - Role-based access (read, write, admin)

### Missing: docs/TROUBLESHOOTING.md

**Why**: Common issues and solutions currently scattered across multiple README files.

**Should Include**:
1. **Installation Issues**
   - Python version mismatches
   - Dependency conflicts (torch, numpy versions)
   - FFmpeg not found
   - Virtual environment problems

2. **Runtime Issues**
   - Port conflicts (8000, 7860, 11434)
   - Ollama not responding
   - Database locked errors
   - Audio device not found
   - Memory/performance issues

3. **Configuration Issues**
   - Environment variable not loaded
   - Model not found (Whisper, Ollama)
   - Invalid YAML syntax
   - Permission denied errors

4. **Testing Issues**
   - Tests hanging (waiting for Ollama)
   - Database fixture conflicts
   - Import errors in tests
   - Coverage report failures

**Each Issue Should Have**:
- Symptoms (error messages, behavior)
- Root cause explanation
- Step-by-step solution
- Prevention tips

### Missing: docs/CONTRIBUTING.md

**Why**: Contributing guidelines mentioned in multiple places but no comprehensive guide.

**Should Include**:
1. **Development Setup**
   - Fork and clone workflow
   - Branch naming conventions (`feature/`, `bugfix/`, `docs/`)
   - Development environment setup
   - Pre-commit hooks setup

2. **Code Standards**
   - Python style guide (Black, flake8, mypy)
   - Type hints required for public APIs
   - Docstring format (Google style)
   - Async/await patterns

3. **Testing Requirements**
   - Unit tests required for new features
   - Integration tests for API changes
   - Performance tests for optimization work
   - Test coverage must not decrease

4. **Pull Request Process**
   - PR template (description, testing, checklist)
   - Review requirements
   - CI/CD must pass
   - Documentation updates required

5. **Adding New Features**
   - How to add new MCP tools
   - How to add new conversation modes
   - How to add new TTS providers
   - Database schema changes (migrations)

### Missing: CHANGELOG.md (root level)

**Why**: No version history or release notes. Important for tracking changes.

**Format**:
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation in docs/ directory
- GitHub Actions CI/CD pipeline
- docs/CURRENT_STATE.md for project overview

### Changed
- Reorganized configuration system
- Improved test coverage

### Fixed
- Database connection pooling issues
- Memory leaks in long-running sessions

## [1.0.0] - 2025-10-27

### Added
- Initial release
- Voice input/output with Whisper and TTS
- Local LLM integration via Ollama
- Persistent conversation memory with SQLite
- Vector semantic search
- MCP tool integration
- Gradio web GUI
- FastAPI REST API and WebSocket support
```

### Missing: examples/ Directory

**Why**: No code examples for developers integrating with the system.

**Should Include**:
```
examples/
├── README.md                       # Overview of examples
├── basic_chat.py                   # Simple text conversation
├── voice_interaction.py            # Voice input/output example
├── api_client.py                   # REST API usage
├── websocket_client.py             # WebSocket streaming
├── custom_tool.py                  # Creating custom MCP tool
├── memory_search.py                # Semantic search examples
├── conversation_export.py          # Export conversation history
└── batch_processing.py             # Batch message processing
```

Each example should be:
- Self-contained and runnable
- Well-commented
- Include error handling
- Show best practices

### Missing: Quick Reference Card

**File**: docs/QUICKREF.md

**Why**: Quick command reference for daily use.

**Should Include**:
```markdown
# Quick Reference

## Starting/Stopping
```bash
./scripts/start_all.sh              # Start all services
./start_gui.sh                      # Start GUI only
./scripts/stop_all.sh               # Stop everything
./scripts/monitor.sh                # Check status
```

## Testing
```bash
./run_tests.sh                      # All tests
./run_tests.sh --unit --fast        # Quick unit tests
./run_tests.sh --coverage           # With coverage
```

## Configuration
```bash
export AI_ASSISTANT_ENV=production  # Change environment
vi config/config.yaml               # Edit main config
vi .env                             # Edit secrets
```

## Database
```bash
sqlite3 ~/ai-assistant/memory/assistant.db
.tables                             # List tables
SELECT * FROM conversations;        # Query data
```

## Logs
```bash
tail -f logs/gui.log                # GUI logs
tail -f logs/mcp_server.log         # API logs
journalctl -u ai-assistant -f       # Systemd logs (production)
```

## Common URLs
- GUI: http://localhost:7860
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Ollama: http://localhost:11434
```

### Add: Documentation Links in README.md

**Location**: After "## Overview" section (after line 13)

**Add**:
```markdown
## Documentation

- **[Quick Start](docs/QUICKREF.md)** - Common commands and URLs
- **[Current State](docs/CURRENT_STATE.md)** - Project overview and setup
- **[Architecture](docs/ARCHITECTURE.md)** - System design and data flow
- **[API Reference](docs/API.md)** - REST and WebSocket API documentation
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Contributing](docs/CONTRIBUTING.md)** - Development guidelines
- **[Testing Guide](tests/README.md)** - Running and writing tests

For Claude Code development guidance, see [CLAUDE.md](CLAUDE.md).
```

### Add: Script Usage Documentation

**File**: Each script in scripts/ should have a `--help` flag

**Currently**: Scripts have extensive README.md but no `--help` output.

**Add to each script** (start_all.sh, stop_all.sh, etc.):
```bash
# Add near top of script
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Start all AI Assistant services with health checks.

OPTIONS:
    --force         Force restart existing services
    --env ENV       Environment (development|production|testing)
    --verbose       Enable verbose output
    --skip-health   Skip health checks
    --help          Show this help message

EXAMPLES:
    $0                              # Normal startup
    $0 --force --env production     # Production restart
    $0 --verbose                    # Debug mode

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        # ... rest of argument parsing
    esac
done
```

### Add: Code Comments in Complex Functions

**Files Needing Better Documentation**:

1. **core/smart_assistant.py:400-500** - Complex state management
   - Add docstrings explaining state transitions
   - Document async callback handling
   - Explain tool integration flow

2. **memory/db_manager.py:200-300** - Connection pooling logic
   - Explain thread-safety mechanisms
   - Document transaction handling
   - Clarify retry logic

3. **mcp_server/main.py:290-348** - Streaming response handling
   - Document SSE format details
   - Explain error recovery in streams
   - Clarify database save timing

4. **gui/interface.py:200-400** - Complex UI state management
   - Document Gradio component interactions
   - Explain async callback chains
   - Clarify file upload handling

**Standard**: All public functions should have:
- One-line summary
- Args description with types
- Returns description with type
- Raises section for exceptions
- Example usage (for complex functions)

### Add: Performance Optimization Notes

**File**: docs/PERFORMANCE.md

**Why**: Users may experience slow responses and need optimization guidance.

**Should Include**:
1. **Model Selection Trade-offs**
   - Whisper: tiny (fast) vs medium (accurate) vs large (best quality)
   - Ollama: 3B (fast) vs 7B (balanced) vs 13B+ (quality)
   - Trade-offs table with latency numbers

2. **Hardware Recommendations**
   - Minimum: 8GB RAM, 4 cores
   - Recommended: 16GB RAM, 8 cores, GPU
   - Optimal: 32GB+ RAM, 16+ cores, GPU with CUDA

3. **Configuration Tuning**
   - Database pool size settings
   - Context length impact
   - Batch size for embeddings
   - Worker count for API server

4. **Monitoring and Profiling**
   - Using monitor.sh for metrics
   - Identifying bottlenecks
   - Memory leak detection
   - Database query optimization

---

## Priority Summary

### 🔴 High Priority (Fix Immediately)
1. **README.md** - Fix incorrect project structure and commands (breaks onboarding)
2. **README.md** - Fix installation step 4 (points to non-existent file)
3. **STREAMING_REQUIREMENTS.md** - Rename and clarify status (misleading)
4. **CLAUDE.md** - Remove/fix MCP git integration claims (incorrect)

### 🟡 Medium Priority (Fix Soon)
1. **Add docs/API.md** - API documentation currently unavailable offline
2. **Add docs/TROUBLESHOOTING.md** - Common issues scattered, need consolidation
3. **Add CHANGELOG.md** - No version tracking
4. **Update scripts with --help flags** - Improve command-line UX

### 🟢 Low Priority (Nice to Have)
1. **Add docs/ARCHITECTURE.md** - Deep technical reference
2. **Add docs/CONTRIBUTING.md** - Formalize contribution process
3. **Add examples/ directory** - Help developers integrate
4. **Add docs/PERFORMANCE.md** - Optimization guide
5. **Improve code comments** - Better inline documentation

---

## Validation Checklist

After making changes, verify:
- [ ] All file paths referenced in documentation actually exist
- [ ] All commands in documentation actually work
- [ ] No contradictions between different documentation files
- [ ] docs/CURRENT_STATE.md matches actual project state
- [ ] README.md provides accurate quick start
- [ ] Each specialized README (scripts/, config/, tests/) is accurate
- [ ] All examples in documentation are tested and working
- [ ] Documentation reflects current implementation, not future plans

---

**Next Steps**:
1. Review this cleanup document with project maintainer
2. Prioritize fixes based on user impact
3. Create issues/tasks for each fix
4. Update documentation incrementally
5. Establish documentation review process for PRs
