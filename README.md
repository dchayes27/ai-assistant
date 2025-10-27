# Local AI Assistant

A comprehensive local AI assistant with voice capabilities, persistent memory, and MCP (Model Context Protocol) integration.

## Overview

This project implements a fully local AI assistant that can:
- Process voice input using OpenAI Whisper
- Generate responses using Ollama-hosted language models
- Speak responses using text-to-speech
- Maintain conversation history and context
- Integrate with external tools via MCP servers
- Provide both GUI and API interfaces

## Architecture

### Core Components

- **core/**: Main application logic and AI model integration
  - Ollama LLM integration
  - Whisper speech-to-text processing
  - TTS (Text-to-Speech) engine
  - Conversation management

- **memory/**: Persistent storage and context management
  - SQLite database for conversation history
  - Vector storage for semantic search
  - Context retrieval and management

- **mcp_server/**: Model Context Protocol server
  - Tool integration framework
  - External service connections
  - Custom tool implementations

- **gui/**: User interface components
  - Gradio-based web interface
  - Voice input/output controls
  - Conversation display

- **scripts/**: Utility and setup scripts
  - Installation helpers
  - Model download scripts
  - Database initialization

## Requirements

- Python 3.9+
- Ollama (for local LLM hosting)
- FFmpeg (for audio processing)
- SQLite3

## Installation

1. Clone this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Ollama and download your preferred models
4. Run the setup script:
   ```bash
   ./install_dependencies.sh
   ```

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

## Usage

### Start all services (recommended):
```bash
./scripts/start_all.sh
```

### Start GUI only:
```bash
./start_gui.sh
# OR
python -m gui.app
```

### Start API server only:
```bash
python -m mcp_server.main
```

### Stop all services:
```bash
./scripts/stop_all.sh
```

## Configuration

Configuration is managed through YAML files in the `config/` directory and environment variables. See `.env.example` for environment settings and `config/config.yaml` for application configuration.

## Development

For detailed project structure and architecture, see [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md).

## License

MIT License - See LICENSE file for details