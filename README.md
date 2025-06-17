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
   python scripts/setup.py
   ```

## Usage

### Start the API server:
```bash
python -m core.main
```

### Launch the GUI:
```bash
python -m gui.app
```

### Run as MCP server:
```bash
python -m mcp_server.server
```

## Configuration

Configuration is managed through environment variables and config files. See `config.example.yaml` for available options.

## Development

### Project Structure
```
ai-assistant/
├── core/
│   ├── __init__.py
│   ├── main.py
│   ├── llm.py
│   ├── speech.py
│   └── tts.py
├── memory/
│   ├── __init__.py
│   ├── database.py
│   ├── context.py
│   └── models.py
├── mcp_server/
│   ├── __init__.py
│   ├── server.py
│   └── tools.py
├── gui/
│   ├── __init__.py
│   ├── app.py
│   └── components.py
├── scripts/
│   ├── setup.py
│   └── download_models.py
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## License

MIT License - See LICENSE file for details