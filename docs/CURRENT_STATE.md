# Current State of AI Assistant

**Last Updated**: 2025-10-27

## Purpose

This is a comprehensive local AI assistant with voice capabilities, persistent memory, and Model Context Protocol (MCP) integration. The system processes voice input via Whisper, generates responses using locally-hosted LLMs through Ollama, converts responses to speech via TTS, and maintains conversation history with semantic search capabilities. It provides both a web GUI (Gradio) and programmatic API (FastAPI) for interacting with the AI assistant.

## Project Structure

### Core Modules

- **`core/`** - Main AI assistant logic
  - `smart_assistant.py` - Primary assistant class integrating speech, LLM, and TTS
  - `tool_manager.py` - MCP tool registry and execution management

- **`memory/`** - Persistent storage and semantic search
  - `db_manager.py` - SQLite database operations with connection pooling
  - `vector_store.py` - Vector embeddings for semantic search (ChromaDB/sqlite-vec)
  - `migrations.py` - Database schema versioning
  - `backup.py` - Automated backup utilities

- **`mcp_server/`** - FastAPI-based MCP server
  - `main.py` - REST API and WebSocket endpoints (port 8000)
  - `ollama_client.py` - Ollama LLM integration with streaming support
  - `tools.py` - External tool implementations (web search, weather, etc.)
  - `auth.py` - JWT authentication middleware
  - `models.py` - Pydantic data models and validation

- **`gui/`** - Web-based user interface
  - `app.py` - Application launcher
  - `interface.py` - Gradio interface implementation (port 7860)
  - `components.py` - Reusable UI components

- **`config/`** - Configuration management
  - `config.yaml` - Base configuration
  - `config.{development,production,testing}.yaml` - Environment-specific configs
  - `manager.py` - Configuration loader with validation
  - `prompt_templates.yaml` - System prompts for different conversation modes
  - `voice_profiles.yaml` - Voice/TTS profile settings

- **`scripts/`** - Deployment and maintenance utilities
  - `start_all.sh` / `stop_all.sh` - Service lifecycle management
  - `monitor.sh` - Health checks and metrics
  - `systemd/` - Systemd service templates for production deployment
  - `backup.sh` - Database backup automation

- **`tests/`** - Comprehensive test suite
  - `unit/` - Unit tests for individual components
  - `integration/` - Service interaction tests (requires Ollama)
  - `performance/` - Benchmarks and stress tests
  - `e2e/` - End-to-end workflow validation

### Support Files

- **`install_dependencies.sh`** - Automated dependency installation
- **`run_tests.sh`** - Test runner with filtering options (--unit, --integration, --fast, --coverage)
- **`CLAUDE.md`** - Development guide and project conventions
- **`STREAMING_REQUIREMENTS.md`** - Real-time streaming enhancement roadmap

## Tech Stack

### Languages & Frameworks
- **Python 3.9+** - Primary language
- **FastAPI** - REST API and WebSocket server
- **Gradio** - Web GUI framework
- **SQLAlchemy** - Database ORM
- **Pydantic** - Data validation

### AI/ML Components
- **Ollama (v0.4.4)** - Local LLM hosting (default: Mistral 7B)
- **OpenAI Whisper** - Speech-to-text (configurable model size)
- **Transformers (v4.46.3)** - NLP model support
- **PyTorch (v2.5.1)** - ML framework

### TTS (Text-to-Speech)
- **pyttsx3** - Offline TTS (fallback)
- **Edge-TTS** - Microsoft Edge TTS (streaming capable)
- **gTTS** - Google Text-to-Speech
- **Coqui TTS** - Local neural TTS (configurable)

### Storage & Memory
- **SQLite3** - Primary database with FTS5 full-text search
- **sqlite-vec** - Vector storage extension
- **ChromaDB** - Alternative vector store
- **Alembic** - Database migrations

### Audio Processing
- **PyAudio** - Real-time audio I/O
- **sounddevice** - Audio device management
- **librosa** - Audio analysis
- **scipy/numpy** - Signal processing

### MCP & Tools
- **MCP SDK (v1.9.4)** - Model Context Protocol integration
- **httpx** - Async HTTP client
- **websockets** - WebSocket support

### Development Tools
- **pytest** - Testing framework with async support
- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking
- **pre-commit** - Git hooks

## How It Works

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                          │
│  ┌────────────────────┐              ┌─────────────────────┐    │
│  │   Gradio Web GUI   │              │   FastAPI REST API  │    │
│  │   (port 7860)      │              │   (port 8000)       │    │
│  └────────┬───────────┘              └──────────┬──────────┘    │
└───────────┼──────────────────────────────────────┼──────────────┘
            │                                       │
            └───────────────┬───────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                      SMART ASSISTANT                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │   Whisper    │  │    Ollama    │  │   TTS Engine      │   │
│  │   (STT)      │→ │   LLM        │→ │   (Speech)        │   │
│  └──────────────┘  └───────┬──────┘  └───────────────────┘   │
└────────────────────────────┼──────────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                  │
┌───────────▼────────────┐      ┌─────────────▼──────────────┐
│    MEMORY SYSTEM       │      │     MCP TOOLS              │
│  ┌─────────────────┐   │      │  ┌──────────────────────┐  │
│  │ SQLite Database │   │      │  │  Tool Registry       │  │
│  │  - Conversations│   │      │  │  - Web Search        │  │
│  │  - Messages     │   │      │  │  - Weather           │  │
│  │  - Knowledge    │   │      │  │  - Memory Queries    │  │
│  │  - Projects     │   │      │  │  - Custom Tools      │  │
│  └─────────────────┘   │      │  └──────────────────────┘  │
│  ┌─────────────────┐   │      └─────────────────────────────┘
│  │  Vector Store   │   │
│  │  (Semantic      │   │
│  │   Search)       │   │
│  └─────────────────┘   │
└────────────────────────┘
```

### Data Flow

1. **Input Processing**
   - User speaks or types input
   - Audio captured via PyAudio → Whisper transcription
   - Text input sent directly to assistant

2. **LLM Processing**
   - Query sent to Ollama LLM (with streaming support)
   - Context retrieved from database (last N messages)
   - Vector store searches for relevant historical context
   - MCP tools executed as needed (web search, database queries, etc.)

3. **Response Generation**
   - LLM generates response (token-by-token streaming available)
   - Response stored in conversation history
   - Embeddings generated and stored for semantic search

4. **Output Processing**
   - Text response displayed in GUI/returned via API
   - TTS converts text to speech
   - Audio played to user (currently manual, streaming planned)

5. **Memory Persistence**
   - All conversations stored in SQLite with FTS5 indexing
   - Vector embeddings enable semantic search
   - Automatic cleanup based on retention policies

### Key Features

- **Conversation Modes**: Chat, Project, Learning, Research, Debug (configurable prompts)
- **Streaming Support**: Ollama streaming enabled, TTS streaming planned
- **Multi-provider TTS**: Falls back through multiple TTS engines for reliability
- **Tool Integration**: Extensible tool registry via MCP protocol
- **Semantic Search**: Vector-based similarity search across conversation history
- **Authentication**: JWT-based API authentication with role-based access
- **Monitoring**: Built-in health checks, metrics, and performance logging

## Setup

### Prerequisites

1. **System Requirements**
   - Python 3.9 or higher
   - FFmpeg (audio processing)
   - SQLite3 (usually pre-installed)

2. **Install Ollama**
   ```bash
   # Visit https://ollama.ai for installation
   curl https://ollama.ai/install.sh | sh

   # Pull a model (e.g., Mistral 7B)
   ollama pull mistral:7b
   ```

### Installation

```bash
# Clone repository
git clone https://github.com/dchayes27/ai-assistant.git
cd ai-assistant

# Install dependencies
./install_dependencies.sh

# Activate virtual environment
source venv/bin/activate

# (Optional) Configure environment
cp .env.example .env
# Edit .env with your settings (OpenAI API key for TTS, etc.)
```

### Running the Assistant

#### Option 1: GUI Interface (Recommended)
```bash
./start_gui.sh
# Opens Gradio interface at http://localhost:7860
```

#### Option 2: API Server
```bash
python -m mcp_server.main
# API available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

#### Option 3: All Services
```bash
./scripts/start_all.sh
# Starts Ollama, API server, and GUI
# Use --force to restart existing services
# Use --env production for production mode
```

### Configuration

- **Environment Selection**: Set `AI_ASSISTANT_ENV` (development/production/testing)
- **Model Configuration**: Edit `config/config.yaml`
  - Whisper model size (tiny/base/small/medium/large)
  - Ollama model selection
  - TTS provider preferences
- **Voice Profiles**: Customize in `config/voice_profiles.yaml`
- **Prompt Templates**: Modify conversation modes in `config/prompt_templates.yaml`

### Testing

```bash
# Run all tests
./run_tests.sh

# Run specific categories
./run_tests.sh --unit              # Unit tests only
./run_tests.sh --integration       # Integration tests (requires Ollama)
./run_tests.sh --performance       # Performance benchmarks
./run_tests.sh --fast              # Skip slow tests
./run_tests.sh --coverage          # Generate coverage report

# Linting and formatting
python -m black .                  # Format code
python -m flake8 .                 # Lint
python -m mypy core/ memory/       # Type check
```

### Stopping Services

```bash
./scripts/stop_all.sh              # Stop all services
pkill -f "gradio"                  # Stop GUI only
pkill -f "uvicorn"                 # Stop API server only
```

## Current Status & Next Steps

### Working Features
- ✅ Voice input/output with Whisper + TTS
- ✅ Local LLM via Ollama with streaming
- ✅ Persistent conversation memory
- ✅ Semantic search via vector store
- ✅ MCP tool integration
- ✅ Web GUI and REST API
- ✅ Authentication and authorization
- ✅ Comprehensive test suite
- ✅ CI/CD pipeline (GitHub Actions)

### In Progress / Planned
- 🔄 **Real-time streaming enhancements** (see `STREAMING_REQUIREMENTS.md`)
  - Continuous audio capture with VAD
  - Automatic TTS playback (remove manual button)
  - Gapless audio streaming
  - 1-2 second voice-to-voice latency target
- 🔄 **Enhanced tool ecosystem**
- 🔄 **Advanced memory features** (automatic summarization, knowledge graphs)

### Known Limitations
- TTS requires manual "play" button (not automatic)
- Voice pipeline is batch-based (not continuous streaming)
- Limited interruption support during long responses
- Single-user focused (multi-user support minimal)

## Development

### Key Entry Points
- GUI: `python -m gui.app --help`
- API: `python -m mcp_server.main`
- Assistant: `from core.smart_assistant import SmartAssistant`

### Adding New Tools
1. Implement tool in `mcp_server/tools.py`
2. Register in tool registry
3. Add tests in `tests/unit/test_tools.py`
4. Document in tool list endpoint

### Database Changes
1. Create migration in `memory/migrations.py`
2. Update models in `memory/models.py`
3. Test with `./run_tests.sh --integration`

### Contributing
See `CLAUDE.md` for development patterns, coding standards, and testing requirements.

## Resources

- **Documentation**: `/docs` directory
- **API Docs**: http://localhost:8000/docs (when running)
- **Logs**: `logs/` directory (auto-rotated)
- **Database**: `~/ai-assistant/memory/assistant.db` (default)
- **Configuration**: `config/*.yaml`

For detailed implementation guidance, see `CLAUDE.md`.
