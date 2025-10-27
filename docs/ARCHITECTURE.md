# System Architecture

High-level architecture and design decisions for the AI Assistant.

---

## Overview

The AI Assistant is built as a modular, local-first system with clear separation of concerns.

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                          │
│  ┌──────────────────┐         ┌──────────────────────┐     │
│  │  Gradio Web GUI  │         │   FastAPI REST API   │     │
│  │  (Port 7860)     │         │   (Port 8000)        │     │
│  └────────┬─────────┘         └──────────┬───────────┘     │
└───────────┼────────────────────────────────┼────────────────┘
            │                                │
            └────────────┬───────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                  SMART ASSISTANT (core/)                  │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐ │
│  │   Whisper    │→ │   Ollama   │→ │   TTS Engine     │ │
│  │   (STT)      │  │    LLM     │  │   (Speech)       │ │
│  └──────────────┘  └─────┬──────┘  └──────────────────┘ │
└──────────────────────────┼────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                              │
┌───────────▼─────────┐      ┌────────────▼────────────────┐
│   MEMORY SYSTEM     │      │       MCP TOOLS             │
│   (memory/)         │      │       (mcp_server/tools.py) │
│                     │      │                             │
│  ┌──────────────┐  │      │  ┌──────────────────────┐  │
│  │   SQLite DB  │  │      │  │  - Web Search        │  │
│  │   - Convs    │  │      │  │  - Weather           │  │
│  │   - Messages │  │      │  │  - Memory Queries    │  │
│  │   - Knowledge│  │      │  │  - Custom Tools      │  │
│  └──────────────┘  │      │  └──────────────────────┘  │
│                     │      └─────────────────────────────┘
│  ┌──────────────┐  │
│  │ Vector Store │  │      ┌─────────────────────────────┐
│  │  (Semantic   │  │      │    EXTERNAL SERVICES        │
│  │   Search)    │  │      │  - Ollama (localhost:11434) │
│  └──────────────┘  │      │  - FFmpeg (audio)           │
└────────────────────┘      │  - SQLite (storage)         │
                             └─────────────────────────────┘
```

---

## Core Components

### 1. Smart Assistant (core/smart_assistant.py)

**Responsibility**: Orchestrates the entire conversation flow

**Key Features**:
- Voice input processing via Whisper
- LLM response generation via Ollama
- TTS output generation
- State management (idle, listening, processing, speaking)
- Conversation mode switching (chat, project, learning, research, debug)
- Tool execution coordination

**Async Architecture**: Uses asyncio for non-blocking I/O operations

### 2. Memory System (memory/)

**Components**:
- **DatabaseManager**: SQLite operations with connection pooling
- **VectorStore**: Semantic search via embeddings
- **Migrations**: Schema version management
- **Backup**: Automated backup with compression

**Database Schema**:
```sql
conversations (id, user_id, mode, title, created_at, updated_at)
messages (id, conversation_id, role, content, created_at)
knowledge_base (id, title, content, tags, metadata)
projects (id, name, description, status, metadata)
embeddings (id, entity_type, entity_id, embedding, model)
```

**Features**:
- FTS5 full-text search
- Vector similarity search
- Automatic embedding generation
- Connection pooling for performance

### 3. MCP Server (mcp_server/)

**Responsibility**: Provides REST API and WebSocket interfaces

**Endpoints**:
- `/agent/query` - Synchronous LLM queries
- `/agent/stream` - Streaming LLM responses
- `/ws/{connection_id}` - WebSocket for real-time
- `/auth/login` - JWT authentication
- `/health` - Health checks

**Features**:
- OpenAPI documentation auto-generation
- JWT + API key authentication
- CORS support
- Request/response validation via Pydantic

### 4. GUI (gui/)

**Technology**: Gradio (Python web framework)

**Features**:
- Voice input/output controls
- Conversation history display
- Settings management
- Project/knowledge management
- Real-time updates

---

## Data Flow

### Voice Conversation Flow

```
1. User speaks → Audio captured
2. Audio → Whisper STT → Text
3. Text → Context retrieval from DB
4. Text + Context → Ollama LLM → Response
5. Response → Saved to DB
6. Response → TTS → Audio
7. Audio → Played to user
```

### API Request Flow

```
1. HTTP Request → FastAPI endpoint
2. Authentication check (JWT/API key)
3. Request validation (Pydantic)
4. SmartAssistant.process_message()
5. Database operations (save/retrieve)
6. LLM generation (Ollama)
7. Tool execution (if needed)
8. Response formation
9. HTTP Response → Client
```

### Tool Execution Flow

```
1. LLM response contains tool call
2. ToolManager detects tool need
3. Tool execution (async)
4. Tool result → Formatted
5. Result → Added to context
6. LLM generates final response
```

---

## Technology Decisions

### Why Ollama?
- **Local-first**: No cloud dependencies
- **Privacy**: Data never leaves local machine
- **Performance**: Direct localhost communication
- **Flexibility**: Easy model switching
- **Streaming**: Native streaming support

### Why SQLite?
- **Embedded**: No separate database server
- **FTS5**: Built-in full-text search
- **Portable**: Single file database
- **Fast**: Excellent for read-heavy workloads
- **Simple**: No configuration needed

### Why Gradio?
- **Rapid prototyping**: Quick UI development
- **Python-native**: No separate frontend framework
- **Built-in components**: Audio, chat, etc.
- **Auto-reload**: Development-friendly
- **Sharing**: Easy deployment

### Why FastAPI?
- **Async**: Native async/await support
- **Fast**: High performance
- **OpenAPI**: Auto-generated docs
- **Validation**: Pydantic integration
- **Modern**: Latest Python features

### Why Whisper?
- **Accuracy**: State-of-the-art STT
- **Multilingual**: 99 languages
- **Offline**: Runs locally
- **Multiple sizes**: Tiny to large models
- **Open source**: MIT licensed

---

## Performance Considerations

### Database
- **Connection pooling**: Reuse connections
- **Indexes**: On frequently queried columns
- **FTS5**: For fast text search
- **Vacuum**: Periodic database optimization

### LLM
- **Streaming**: Token-by-token responses
- **Context management**: Limit context window
- **Model selection**: Balance speed vs quality
- **Caching**: Ollama handles model caching

### Memory
- **Vector cache**: LRU cache for embeddings
- **Conversation pruning**: Limit context length
- **Batch operations**: Reduce DB round-trips

### Async Operations
- **Non-blocking I/O**: All external calls async
- **Thread pools**: For CPU-bound work
- **Event loop**: Single-threaded async

---

## Security

### Authentication
- **JWT**: Stateless token authentication
- **API Keys**: Simple key-based auth
- **Bcrypt**: Password hashing
- **Token expiry**: Configurable timeouts

### Data Protection
- **Local only**: No external data transmission
- **Encrypted storage**: Optional DB encryption
- **Secure secrets**: Environment variables
- **CORS**: Configurable origins

### Input Validation
- **Pydantic**: Request validation
- **SQL injection**: Parameterized queries
- **XSS**: Gradio handles sanitization
- **Rate limiting**: Planned feature

---

## Scalability

### Current Architecture (Single User)
- Optimized for localhost deployment
- Single database instance
- Direct LLM connection

### Future Multi-User Support
- User ID in all tables
- Connection pooling
- Rate limiting per user
- Separate workspaces

---

## Extension Points

### Adding New Tools
1. Implement in `mcp_server/tools.py`
2. Register in tool registry
3. Define tool schema
4. Add tests

### Adding New Conversation Modes
1. Add to `ConversationMode` enum
2. Define system prompt
3. Update GUI selector
4. Add mode-specific logic

### Adding New TTS Providers
1. Implement TTS interface
2. Add to provider list
3. Implement fallback logic
4. Add configuration options

### Database Schema Changes
1. Create migration script
2. Update models
3. Test migration
4. Document changes

---

For implementation details, see source code in respective modules.
For deployment architecture, see production deployment guide (coming soon).
