# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive local AI assistant with voice capabilities, persistent memory, and MCP (Model Context Protocol) integration. The system provides both GUI and API interfaces for interacting with AI models running locally via Ollama.

## Architecture

The project follows a modular architecture with clear separation of concerns:

- **core/**: Main AI assistant logic, LLM integration, and conversation management
- **memory/**: Persistent storage with SQLite database and vector store for semantic search
- **mcp_server/**: FastAPI-based MCP server providing REST and WebSocket APIs
- **gui/**: Gradio-based web interface for user interactions
- **config/**: YAML-based configuration management with environment-specific overrides
- **scripts/**: Deployment, monitoring, and management utilities

## Common Development Commands

### Testing
```bash
# Run all tests
./run_tests.sh

# Run specific test categories
./run_tests.sh --unit                    # Unit tests only
./run_tests.sh --integration             # Integration tests only
./run_tests.sh --performance             # Performance tests only
./run_tests.sh --fast                    # Exclude slow tests
./run_tests.sh --coverage               # With coverage reporting

# Run tests requiring external services
./run_tests.sh --requires-ollama        # Tests needing Ollama
./run_tests.sh --audio                  # Audio processing tests
```

### Linting and Code Quality
```bash
# Format code
python -m black .

# Run linting
python -m flake8 .

# Type checking
python -m mypy core/ memory/ mcp_server/ gui/
```

### Service Management
```bash
# Start all services
./scripts/start_all.sh

# Start with specific options
./scripts/start_all.sh --force --verbose     # Force restart with logs
./scripts/start_all.sh --env production      # Production environment

# Stop services
./scripts/stop_all.sh

# Monitor services
./scripts/monitor.sh
```

### Development Setup
```bash
# Install dependencies
./install_dependencies.sh

# Activate virtual environment
source venv/bin/activate
```

## Key Configuration

- Configuration is managed via YAML files in `config/` directory
- Environment-specific configs: `config.development.yaml`, `config.production.yaml`, `config.testing.yaml`
- Default configuration in `config/config.yaml`
- Use `AI_ASSISTANT_ENV` environment variable to switch configurations

## Database and Memory

- SQLite database with migrations managed via `memory/migrations.py`
- Vector store for semantic search using ChromaDB or sqlite-vec
- Database initialization and management through `memory/db_manager.py`

## Testing Structure

Tests are organized by type in `tests/` directory:
- `unit/`: Unit tests for individual components
- `integration/`: Integration tests for service interactions  
- `performance/`: Performance benchmarks and stress tests
- `e2e/`: End-to-end workflow tests

Test markers:
- `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.performance`
- `@pytest.mark.slow` for tests taking >5 seconds
- `@pytest.mark.requires_ollama` for tests needing Ollama service
- `@pytest.mark.requires_audio` for audio hardware-dependent tests

## Service Dependencies

The system requires several external services:
- **Ollama**: Local LLM hosting (port 11434)
- **FFmpeg**: Audio processing
- **SQLite3**: Database storage

Services start in dependency order:
1. Ollama service
2. API server (port 8000)
3. GUI interface (port 7860)
4. Optional metrics server (port 8001)

## Development Patterns

- Use async/await for I/O operations
- Configuration via dependency injection through config managers
- Structured logging with loguru
- Type hints throughout the codebase
- Error handling with custom exception classes in `mcp_server/models.py`

## Performance Considerations

- The system uses streaming responses for LLM interactions
- Vector operations are optimized for semantic search
- Memory usage is monitored and configurable via `performance.max_memory_usage`
- Connection pooling for database operations

## Deployment

- Production deployment via systemd services (templates in `scripts/systemd/`)
- Backup automation with configurable retention
- Health checks and monitoring built into service scripts
- Environment-based configuration switching

## To-Do: Focusing on Pure Voice Assistant Real-Time Streaming

Let's concentrate solely on transforming your voice assistant into a real-time streaming system.

### Your Current Voice Pipeline
```
Voice → Whisper → GPT-4o → Sesame TTS → Audio Output
       ↓
    MCP Servers (Tools/Memory/Database)
```

### Real-Time Streaming Enhancement Plan

#### Step 1: Upgrade Your Voice Agent for Streaming

```bash
claude-code "Create realtime/streaming_voice_agent.py based on integrated_voice_agent.py:

1. Implement StreamingVoiceAgent class:
   - Stream audio input continuously (no push-to-talk)
   - Use OpenAI's streaming API for GPT-4o
   - Process Whisper in chunks for faster initial response
   - Stream TTS output as sentences complete

2. Add these core streaming methods:
   - stream_listen() - Continuous audio capture with VAD
   - stream_transcribe() - Progressive Whisper transcription  
   - stream_gpt4_response() - Token-by-token from OpenAI
   - stream_tts_output() - Sentence-level TTS generation
   
3. Implement interruption handling:
   - Stop GPT-4o generation on new input
   - Flush audio queues
   - Seamless conversation continuation"
```

#### Step 2: Create Real-Time Audio Pipeline

```bash
claude-code "Build realtime/audio_pipeline.py:

1. AudioStreamManager class with:
   - PyAudio for low-latency I/O
   - WebRTC VAD for speech detection
   - Ring buffer for continuous audio
   - Echo cancellation
   
2. Implement parallel processing:
   - Audio recording thread
   - Transcription thread  
   - LLM processing thread
   - TTS generation thread
   - Audio playback thread
   
3. Add queue management:
   - Lock-free audio queues
   - Priority queue for interruptions
   - Backpressure handling"
```

#### Step 3: Optimize GPT-4o Streaming

```bash
claude-code "Create realtime/gpt4_streaming.py:

1. Streaming GPT-4o client with:
   - Token-by-token reception
   - Sentence boundary detection
   - Function calling support (for MCP tools)
   - Context window management
   
2. Smart chunking logic:
   - Detect natural pause points
   - Group tokens for TTS
   - Handle code/lists specially
   - Maintain conversation flow
   
3. Tool streaming integration:
   - Stream tool calls to MCP servers
   - Progressive result integration
   - Parallel tool execution"
```

#### Step 4: Enhance MCP Integration for Streaming

```bash
claude-code "Update your MCP tool integration for real-time:

1. Create mcp/streaming_client.py:
   - WebSocket connections to MCP servers
   - Streaming tool execution
   - Progressive result delivery
   - Connection pooling
   
2. Modify tool responses:
   - Stream search results as found
   - Progressive memory queries
   - Real-time calculation updates
   - Incremental database results"
```

#### Step 5: TTS Streaming Implementation

```bash
claude-code "Create realtime/tts_streaming.py:

1. For Sesame TTS (if it supports streaming):
   - Chunk text by sentences
   - Generate audio progressively
   - Smooth audio transitions
   
2. Add fallback streaming TTS:
   - Edge-TTS for streaming support
   - Coqui TTS with chunking
   - pyttsx3 for lowest latency
   
3. Implement audio queue:
   - Buffer management
   - Gapless playback
   - Volume normalization"
```

#### Step 6: Build Conversation State Manager

```bash
claude-code "Create realtime/conversation_manager.py:

1. Real-time state tracking:
   - Current speaker (user/assistant)
   - Conversation context
   - Interruption state
   - Tool execution status
   
2. Memory integration:
   - Stream updates to database
   - Real-time context retrieval
   - Progressive memory building
   
3. Session management:
   - Conversation threading
   - State persistence
   - Recovery from disconnects"
```

#### Step 7: Create Unified Real-Time Interface

```bash
claude-code "Build realtime/unified_interface.py:

1. Main orchestration class:
   - Coordinate all streaming components
   - Handle state transitions
   - Manage resource lifecycle
   
2. Event-driven architecture:
   - Audio input events
   - Transcription events
   - LLM token events
   - TTS ready events
   - Playback complete events
   
3. Configuration management:
   - Hot-reloadable settings
   - Per-user preferences
   - Adaptive quality settings"
```

#### Step 8: Web Interface for Real-Time Interaction

```bash
claude-code "Create web_realtime/index.html and app.js:

1. Modern web interface with:
   - WebSocket connection to your agent
   - Real-time transcription display
   - Streaming response visualization
   - Audio waveform display
   
2. JavaScript client with:
   - MediaRecorder for audio capture
   - WebSocket for bidirectional streaming
   - Web Audio API for playback
   - Automatic reconnection
   
3. UI features:
   - Push-to-talk OR continuous mode
   - Interruption button
   - Conversation history
   - Settings panel"
```

#### Step 9: Performance Monitoring

```bash
claude-code "Add realtime/performance_monitor.py:

1. Track key metrics:
   - Voice-to-voice latency
   - Time to first token
   - Audio quality scores
   - Interruption response time
   
2. Create dashboard showing:
   - Real-time latency graph
   - Token generation speed
   - Audio processing load
   - MCP tool response times
   
3. Optimization suggestions:
   - Model size recommendations
   - Quality/speed tradeoffs
   - Resource usage alerts"
```

#### Step 10: Launch Script

```bash
claude-code "Create start_realtime_assistant.sh:

#!/bin/bash
# Start all components for real-time voice

# Check audio devices
echo 'Checking audio devices...'
python -c 'import pyaudio; p=pyaudio.PyAudio(); print(f"Found {p.get_device_count()} audio devices")'

# Start MCP servers (if needed)
echo 'Ensuring MCP servers are running...'

# Launch real-time voice assistant
echo 'Starting real-time voice assistant...'
python realtime/unified_interface.py --mode continuous --model gpt-4o

# Optional: Launch web interface
echo 'Starting web interface on http://localhost:3000'
python -m http.server 3000 --directory web_realtime &

echo 'Real-time voice assistant is ready!'"
```

### Expected Results

With this implementation, you'll achieve:
- **< 500ms voice-to-voice latency**
- **Natural interruptions and barge-in**
- **Continuous conversation flow**
- **Streaming MCP tool results**
- **Professional audio quality**

The system will feel conversational and responsive, just like talking to a real person!

## Migration Best Practices: Current to Real-Time Architecture

### Pre-Migration Planning

#### 1. Assess Current System State
```bash
# Before starting migration, document current performance
./run_tests.sh --performance --benchmark
./scripts/monitor.sh --baseline > migration/current_baseline.txt

# Document current voice pipeline latencies
python -c "
from core.smart_assistant import SmartAssistant
import time
# Test current voice-to-voice latency and document results
"
```

#### 2. Create Migration Branch Strategy
```bash
# Create feature branch for streaming development
git checkout -b feature/realtime-streaming
git push -u origin feature/realtime-streaming

# Create migration tracking directory
mkdir -p migration/{backups,config,tests,docs}
```

#### 3. Backup Current System
```bash
# Backup database and configurations
cp -r data/ migration/backups/data_$(date +%Y%m%d)
cp -r config/ migration/backups/config_$(date +%Y%m%d)
cp -r core/ migration/backups/core_$(date +%Y%m%d)

# Document current dependencies
pip freeze > migration/backups/requirements_current.txt
```

### Migration Phases

#### Phase 1: Parallel Development (Weeks 1-2)
**Goal**: Build streaming components alongside existing system

```bash
# Create realtime directory structure
mkdir -p realtime/{audio,llm,tts,state,monitoring}
mkdir -p web_realtime/{static,templates}

# Set up isolated testing environment
python -m venv venv_streaming
source venv_streaming/bin/activate
pip install -r requirements.txt

# Add streaming-specific dependencies
pip install webrtcvad pyaudio asyncio-mqtt
pip freeze > requirements-streaming.txt
```

**Key Principles:**
- Keep existing voice agent functional during development
- Use feature flags to enable/disable streaming components
- Implement streaming components as separate modules first
- Test each component in isolation before integration

#### Phase 2: Audio Pipeline Migration (Week 3)
**Goal**: Replace current audio I/O with streaming pipeline

```bash
# Test audio compatibility
python realtime/audio_pipeline.py --test-only
python -c "import pyaudio; print('Audio devices:', pyaudio.PyAudio().get_device_count())"

# Gradual rollout strategy
# 1. Add streaming audio as optional mode
# 2. Test with existing Whisper integration
# 3. Compare quality metrics
# 4. Switch default when stable
```

**Migration Steps:**
1. Implement `AudioStreamManager` with same interface as current audio
2. Add configuration flag: `audio.streaming_mode: false`
3. Test streaming audio with existing Whisper pipeline
4. Run A/B tests comparing quality and latency
5. Switch default when metrics improve

#### Phase 3: LLM Streaming Integration (Week 4)
**Goal**: Replace batch LLM calls with streaming

```bash
# Test OpenAI streaming compatibility
python realtime/gpt4_streaming.py --validate-tokens
python -c "
import openai
# Test streaming vs batch for same prompts
# Compare quality and latency
"
```

**Migration Approach:**
- Implement streaming LLM client with fallback to current system
- Use adapter pattern to maintain same interface for MCP tools
- Test tool calling compatibility with streaming responses
- Gradually increase streaming usage based on success metrics

#### Phase 4: TTS Streaming Migration (Week 5)
**Goal**: Replace current TTS with streaming output

```bash
# Test TTS streaming quality
python realtime/tts_streaming.py --quality-test
python -c "
# Compare current TTS vs streaming TTS
# Test audio quality, latency, and naturalness
"
```

**Compatibility Strategy:**
- Keep current TTS as fallback option
- Implement streaming TTS with same audio output interface  
- Test gapless playback and interruption handling
- Use configuration to control TTS mode per user

#### Phase 5: State Management Migration (Week 6)
**Goal**: Migrate conversation state to real-time manager

```bash
# Migrate existing conversations
python migration/migrate_conversations.py --dry-run
python migration/migrate_conversations.py --execute

# Test state compatibility
python realtime/conversation_manager.py --validate-existing-data
```

**Data Migration:**
- Export existing conversation history to compatible format
- Test real-time state manager with historical data
- Implement state synchronization between old and new systems
- Gradual migration of active conversations

#### Phase 6: Integration and Cutover (Week 7)
**Goal**: Replace current voice agent with streaming version

```bash
# Full system integration test
python realtime/unified_interface.py --integration-test
./run_tests.sh --e2e --streaming

# Performance comparison
python migration/compare_performance.py --current-vs-streaming
```

### Migration Safety Measures

#### 1. Feature Flags and Configuration
```yaml
# Add to config/config.yaml
experimental:
  realtime_streaming:
    enabled: false
    audio_streaming: false
    llm_streaming: false
    tts_streaming: false
    fallback_to_current: true

migration:
  preserve_current_system: true
  gradual_rollout: true
  rollback_enabled: true
```

#### 2. Rollback Plan
```bash
# Quick rollback script
cat > migration/rollback.sh << 'EOF'
#!/bin/bash
echo "Rolling back to previous voice agent..."

# Stop streaming services
pkill -f "realtime/"

# Restore original configuration
cp migration/backups/config_*/config.yaml config/
cp migration/backups/core_*/*.py core/

# Restart original services
./scripts/start_all.sh --force

echo "Rollback complete. Original system restored."
EOF

chmod +x migration/rollback.sh
```

#### 3. Monitoring During Migration
```bash
# Set up migration monitoring
python migration/monitor_migration.py &

# Track key metrics:
# - Voice-to-voice latency before/after
# - Error rates during transition
# - User experience metrics
# - Resource usage changes
```

### Testing Strategy During Migration

#### 1. Compatibility Testing
```bash
# Test existing functionality still works
./run_tests.sh --unit --integration
python tests/test_voice_compatibility.py

# Test new streaming components
./run_tests.sh --streaming --performance
python tests/test_streaming_integration.py
```

#### 2. Performance Regression Testing
```bash
# Automated performance comparison
python migration/performance_regression.py \
  --baseline migration/current_baseline.txt \
  --current-test \
  --streaming-test \
  --report migration/performance_report.html
```

#### 3. User Acceptance Testing
```bash
# A/B testing framework
python migration/ab_testing.py \
  --users 10 \
  --current-system 50% \
  --streaming-system 50% \
  --duration 24h
```

### Post-Migration Validation

#### 1. System Health Checks
```bash
# Comprehensive health validation
./scripts/monitor.sh --full-check
python realtime/performance_monitor.py --validate-targets

# Expected improvements:
# - Voice-to-voice latency < 500ms (down from ~2-3s)
# - Interruption response < 200ms
# - Audio quality maintained or improved
# - Memory usage optimized
```

#### 2. Cleanup Old System
```bash
# After 2 weeks of stable streaming operation
python migration/cleanup_old_system.py --safe-mode

# Archive old components
mkdir -p archive/pre_streaming/
mv core/voice_agent_old.py archive/pre_streaming/
mv config/voice_legacy.yaml archive/pre_streaming/

# Update documentation
sed -i 's/voice_agent.py/streaming_voice_agent.py/g' README.md
```

### Migration Timeline and Milestones

**Week 1-2**: Parallel development of streaming components
- ✅ Audio pipeline implemented and tested
- ✅ LLM streaming client working
- ✅ TTS streaming functional

**Week 3**: Audio migration
- ✅ Streaming audio replaces current input/output
- ✅ Quality metrics match or exceed current system
- ✅ Fallback mechanism working

**Week 4**: LLM streaming integration
- ✅ Streaming responses integrated with MCP tools
- ✅ Conversation flow maintained
- ✅ Tool calling compatibility verified

**Week 5**: TTS streaming deployment
- ✅ Gapless audio playback working
- ✅ Interruption handling functional
- ✅ Audio quality maintained

**Week 6**: State management migration
- ✅ All conversations migrated to new state manager
- ✅ Real-time state synchronization working
- ✅ Historical data preserved

**Week 7**: Full integration and validation
- ✅ End-to-end streaming voice assistant operational
- ✅ Performance targets achieved (<500ms latency)
- ✅ All tests passing
- ✅ Production deployment successful

### Risk Mitigation

**High-Risk Areas:**
1. **Audio compatibility**: Different hardware may have varying PyAudio support
2. **OpenAI API limits**: Streaming may hit different rate limits
3. **Memory usage**: Streaming components may increase RAM usage
4. **Database locks**: Real-time state updates may cause contention

**Mitigation Strategies:**
- Extensive hardware testing across different audio devices
- Implement queue management and backpressure handling
- Memory profiling and optimization during development
- Database connection pooling and optimistic locking
- Always maintain rollback capability until system is proven stable