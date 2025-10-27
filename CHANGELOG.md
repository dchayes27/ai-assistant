# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation in docs/ directory
  - CURRENT_STATE.md for project overview
  - CLEANUP.md for documentation maintenance
  - ROADMAP.md for improvement tracking
  - QUICKREF.md for quick command reference
- GitHub Actions CI/CD pipeline
- Documentation links section in README.md
- Quick Start section in README.md
- STREAMING_ROADMAP.md for future voice streaming enhancements

### Changed
- Reorganized README.md with corrected commands and file references
- Renamed STREAMING_REQUIREMENTS.md to STREAMING_ROADMAP.md for clarity
- Updated all documentation READMEs with clarifications and details
- Improved configuration documentation in config/README.md
- Enhanced test documentation in tests/README.md

### Fixed
- README.md outdated project structure diagram
- README.md incorrect installation commands
- README.md wrong configuration file references
- CLAUDE.md test runner documentation gaps
- scripts/README.md missing health check details

## [1.0.0] - 2025-10-27

### Added
- Initial release
- Voice input/output with OpenAI Whisper
- Local LLM integration via Ollama with streaming support
- Persistent conversation memory with SQLite database
- Vector semantic search with ChromaDB/sqlite-vec
- MCP (Model Context Protocol) tool integration
- Gradio web GUI interface (port 7860)
- FastAPI REST API with OpenAPI documentation (port 8000)
- WebSocket support for real-time communication
- Multiple TTS providers (pyttsx3, edge-tts, gTTS, Coqui)
- JWT-based authentication with API key support
- Conversation modes (chat, project, learning, research, debug)
- Database migrations system
- Automated backup system with compression
- Comprehensive test suite (unit, integration, performance, e2e)
- Service management scripts (start_all.sh, stop_all.sh, monitor.sh)
- Environment-based configuration (development, production, testing)
- Connection pooling for database operations
- Vector store for semantic similarity search
- Knowledge base management
- Project tracking capabilities

### Core Components
- **core/smart_assistant.py** - Main AI assistant orchestration
- **core/tool_manager.py** - MCP tool integration
- **memory/db_manager.py** - SQLite database operations
- **memory/vector_store.py** - Vector embeddings and semantic search
- **memory/backup.py** - Automated backup system
- **memory/migrations.py** - Database schema versioning
- **mcp_server/main.py** - FastAPI REST API and WebSocket server
- **mcp_server/auth.py** - JWT authentication
- **mcp_server/ollama_client.py** - Ollama LLM client
- **mcp_server/tools.py** - External tool implementations
- **gui/interface.py** - Gradio web interface
- **gui/components.py** - Reusable UI components

### Dependencies
- Python 3.9+
- Ollama v0.4.4
- OpenAI Whisper
- Transformers 4.46.3
- PyTorch 2.5.1
- FastAPI 0.115.5
- Gradio 5.8.0
- SQLAlchemy 2.0.36
- ChromaDB 0.5.23
- And more (see requirements.txt)

---

## Release Notes

### Version 1.0.0

First stable release of the Local AI Assistant. This release provides a complete voice-enabled AI assistant that runs entirely on local infrastructure with no cloud dependencies.

**Key Features:**
- Fully local operation (no external API calls required)
- Voice conversation capabilities
- Persistent memory with semantic search
- Extensible tool system via MCP
- Web GUI and API interfaces
- Production-ready deployment scripts

**Known Limitations:**
- TTS requires manual "play" button (not automatic)
- Voice pipeline is batch-based (not real-time streaming)
- Single-user focused (multi-user support minimal)
- Limited interruption support during long responses

**Future Enhancements:**
See [STREAMING_ROADMAP.md](STREAMING_ROADMAP.md) and [docs/ROADMAP.md](docs/ROADMAP.md) for planned improvements.

---

## Contributing

When adding entries to this changelog:
1. Add under `[Unreleased]` section
2. Use categories: Added, Changed, Deprecated, Removed, Fixed, Security
3. Write clear, user-focused descriptions
4. Link to relevant issues/PRs when applicable
5. Move to versioned section on release

For more details, see [CONTRIBUTING.md](docs/CONTRIBUTING.md).
