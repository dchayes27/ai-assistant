# Real-Time Voice Assistant Streaming Roadmap (PLANNED)

**Status**: Planning Phase - Not Yet Implemented
**Target**: Future enhancement to enable real-time streaming voice interactions

> ⚠️ **Note**: This is a planning document for future development. The features described here are NOT yet implemented. For current capabilities, see [CURRENT_STATE.md](docs/CURRENT_STATE.md).

---

## Current System Analysis

### **Existing Architecture (Working)**
```
🎤 Voice Input → Whisper STT → Ollama LLM → TTS → 🔊 Audio Output
                                  ↓
                              MCP Server (Tools/Memory/Database)
```

### **Current Components**
- **LLM Provider**: Ollama (localhost:11434) with Mistral 7B (possibly upgrade to larger model later)
- **STT**: OpenAI Whisper (base model, CPU, int8)
- **TTS**: Coqui TTS with fallback to pyttsx3
- **MCP Integration**: Full tool registry with memory/database access
- **Storage**: SQLite database + vector store for memory
- **Interface**: Gradio GUI + FastAPI REST/WebSocket

### **Current Performance Baseline**
- Voice pipeline latency: Unknown (currently requires manual "play" button click for TTS)
- Ollama streaming: Already supported (`stream: true` in config)
- Audio processing: Batch-based (full sentences)
- Response pattern: Wait for complete audio → process → manual TTS playback

## Streaming Enhancement Goals

### **Target Architecture**
```
🎤 Continuous Audio → Real-time STT → Streaming Ollama → Streaming TTS → 🔊 Real-time Audio
                                           ↓
                                    MCP Tools (Real-time)
```

### **Performance Targets**
- **Voice-to-voice latency**: 1-2 seconds total (acceptable target)
- **TTS Auto-playback**: Automatic (no manual "play" button required)  
- **Audio quality**: Realistic sounding TTS
- **Conversation flow**: Natural, continuous
- **Database integration**: Full memory + structured data (movie ratings, music, project databases)

### **Core Requirements**

#### 1. **Audio Pipeline Enhancement**
- **Continuous audio capture** (replace batch recording)
- **WebRTC VAD** for speech detection
- **Chunked audio processing** for faster STT
- **Audio interruption handling** (stop current playback)
- **Gapless audio output** for natural flow

#### 2. **Ollama Streaming Integration** 
- **Use existing Ollama setup** (no OpenAI LLM)
- **Token-by-token streaming** from Ollama API
- **Sentence boundary detection** for TTS chunking
- **Maintain MCP tool integration** during streaming
- **Context window management** for long conversations

#### 3. **TTS Streaming Implementation**
- **Sentence-level audio generation** (not full response)
- **Multiple provider support** (keep existing + add streaming options)
- **Audio queue management** for smooth playback
- **Provider fallback** (Edge-TTS → Coqui → pyttsx3)

#### 4. **MCP Tools Integration**
- **Real-time tool execution** during streaming
- **Progressive result streaming** back to user
- **Maintain all existing tools** (memory, database, etc.)
- **Tool response interruption** support

#### 5. **System Integration**
- **Enhance existing GUI** (don't replace - keep text input/output, conversation history)
- **Add automatic TTS playback** (eliminate manual "play" button)
- **Maintain configuration system** (YAML-based)
- **Keep ALL database/memory functionality** (memory + structured data storage)

## Technical Constraints

### **Must Use Existing Components**
- ✅ **Ollama for LLM** (not OpenAI)
- ✅ **Current MCP server setup** 
- ✅ **SQLite + vector store**
- ✅ **Existing configuration system**
- ✅ **Current directory structure**

### **External Dependencies (Optional)**
- 🔧 **OpenAI API key**: Only for TTS if using OpenAI provider
- 🔧 **Edge-TTS**: Free alternative for streaming TTS
- 🔧 **WebRTC VAD**: For speech detection
- 🔧 **PyAudio**: For real-time audio I/O

### **Compatibility Requirements**
- 📱 **Works with existing GUI** (Gradio interface)
- 📡 **Works with existing API** (FastAPI endpoints) 
- 🔧 **Works with existing MCP tools**
- 💾 **Preserves conversation history/memory**
- ⚙️ **Uses existing configuration files**

## Implementation Strategy

### **Phase 1: Audio Foundation**
1. **Measure current baseline** performance
2. **Implement continuous audio capture** with VAD
3. **Add chunked Whisper processing**
4. **Test audio interruption handling**

### **Phase 2: Ollama Streaming**
1. **Create Ollama streaming client** (token-by-token)
2. **Implement sentence boundary detection**
3. **Integrate with existing MCP tool system**
4. **Test streaming + tool execution**

### **Phase 3: TTS Streaming**
1. **Add sentence-level TTS processing**
2. **Implement audio queue management**
3. **Add multiple TTS provider support**
4. **Test gapless audio playback**

### **Phase 4: Integration**
1. **Connect all streaming components**
2. **Add configuration options** for streaming mode
3. **Integrate with existing GUI/API**
4. **Performance optimization and testing**

## Key Requirements (Answered)

### **Audio Processing** ✅
- **Augment existing processing** (don't replace)
- **1-2 second total latency** is acceptable
- **Automatic TTS playback** (no manual button clicks)

### **LLM Integration** ✅  
- **Use Mistral 7B** (current Ollama model), upgrade path to larger models
- **Maintain streaming capability** already in config
- **Keep context window management** as-is

### **TTS Strategy** ✅
- **Prioritize realistic sounding TTS** over speed
- **Multiple provider support** for reliability
- **Automatic playback** on response completion

### **User Experience** ✅
- **Enhance existing GUI** (keep text input/output, conversation history)
- **Add automatic voice responses** as enhancement
- **Maintain text-based interaction** alongside voice

### **System Architecture** ✅
- **Enhance current system** rather than replace
- **Maintain all database functionality** (memory + structured data)
- **Keep existing API compatibility**

## Success Criteria

### **Technical Metrics**
- ✅ Voice-to-voice latency < 2 seconds
- ✅ First response time < 500ms  
- ✅ Audio quality maintained
- ✅ Natural conversation flow
- ✅ Successful interruption handling

### **Functional Requirements**
- ✅ All existing MCP tools work in streaming mode
- ✅ Conversation memory/history preserved
- ✅ Multi-provider TTS with fallback
- ✅ Compatible with existing GUI/API
- ✅ Configurable via existing YAML system

### **User Experience**
- ✅ Feels natural and conversational
- ✅ Clear system state indicators
- ✅ Reliable interruption support
- ✅ Graceful error handling
- ✅ Easy to switch between streaming/non-streaming modes

## Files That Need Creation/Modification

### **New Components (To Create)**
- `realtime/audio_streaming.py` - Continuous audio capture with VAD
- `realtime/ollama_streaming.py` - Token streaming from Ollama
- `realtime/tts_streaming.py` - Multi-provider streaming TTS
- `realtime/voice_agent.py` - Main orchestration class
- `realtime/conversation_state.py` - Real-time state management

### **Existing Components (To Modify)**
- `core/smart_assistant.py` - Add streaming mode option
- `mcp_server/ollama_client.py` - Add streaming methods
- `gui/interface.py` - Add streaming interface elements
- `config/config.yaml` - Add streaming configuration options

### **Integration Points**
- Existing MCP tool registry
- Current conversation memory system
- Gradio GUI components
- FastAPI REST/WebSocket endpoints

This document ensures no assumptions about external APIs, maintains compatibility with existing Ollama setup, and provides clear technical requirements for implementation.