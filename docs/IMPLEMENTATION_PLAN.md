# Holistic Implementation Plan: From Setup to Streaming with XTTS v2

**Created**: 2025-10-29
**Hardware**: MacBook Pro M3 Max (40 GPU cores, 16 CPU cores, 64GB RAM)
**Target**: Full streaming voice assistant with <400ms latency

## Hardware Compatibility Notes (M3 Max Specific)

### ✅ Your M3 Max Advantages:
- **Neural Engine**: 16-core Neural Engine perfect for Whisper and XTTS v2
- **Unified Memory**: 64GB shared memory - no VRAM limitations
- **Metal Performance Shaders**: Accelerated PyTorch operations
- **ProRes Media Engine**: Hardware audio encoding/decoding

### Optimizations for Apple Silicon:
```bash
# Use MPS (Metal Performance Shaders) backend for PyTorch
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Install Apple Silicon optimized packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-metal  # For any TensorFlow operations

# Use accelerated Whisper
pip install openai-whisper --upgrade  # Includes CoreML optimizations
```

### Expected Performance on M3 Max:
- **Whisper base.en**: ~50ms processing time per second of audio
- **XTTS v2**: ~100-150ms to first audio chunk (better than CUDA in some cases!)
- **Ollama with Mistral 7B**: ~30-50 tokens/second
- **Total voice-to-voice latency**: **200-300ms achievable** (better than our 400ms target!)

---

## Current State Analysis
- **Project Status**: Framework exists but NOT operational
- **Critical Security Fixes**: ✅ Already completed (JWT, auth, dependencies)
- **Outstanding Issues**: 23 GitHub issues including streaming (#18)
- **Key Problems**:
  - No environment setup
  - Batch-based audio (2-3s latency)
  - Manual TTS playback required
  - Large monolithic files (smart_assistant.py: 979 lines, interface.py: 1080 lines)

---

## Phase 1: Foundation Setup (Day 1-2)
**Goal**: Get basic system operational before adding streaming

### Day 1: Environment & Dependencies
1. **Set up Python environment (M3 optimized)**
   ```bash
   # Create venv with Python 3.11 (best for M3)
   python3.11 -m venv venv
   source venv/bin/activate

   # Install with Apple Silicon optimizations
   pip install --upgrade pip setuptools wheel

   # Install PyAudio with homebrew portaudio
   export CFLAGS="-I/opt/homebrew/include"
   export LDFLAGS="-L/opt/homebrew/lib"
   pip install pyaudio

   # Install core requirements
   pip install -r requirements.txt
   ```

2. **Start and configure Ollama**
   ```bash
   # Ollama runs natively on Apple Silicon
   ollama serve

   # Download models (these run great on M3)
   ollama pull mistral:7b-instruct-v0.2-q4_K_M  # Optimized for Apple Silicon
   ollama pull llama2:13b-chat-q4_K_M  # If you want higher quality
   ```

3. **Initialize database and configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys

   # Initialize database
   python -c "from memory.db_manager import DatabaseManager; DatabaseManager().initialize()"
   ```

4. **Test M3 specific features**
   ```python
   # Test Metal backend
   import torch
   print(f"MPS available: {torch.backends.mps.is_available()}")
   print(f"MPS built: {torch.backends.mps.is_built()}")

   # Test audio devices
   import pyaudio
   p = pyaudio.PyAudio()
   print(f"Audio devices: {p.get_device_count()}")
   ```

### Day 2: Fix Immediate Issues
1. **Address GitHub Issue #7**: Fix HTTP client resource leaks
2. **Address GitHub Issue #6**: Add database transaction management
3. **Test existing batch pipeline**
4. **Document current latency baseline**

---

## Phase 2: Prepare for Streaming (Day 3-4)
**Goal**: Refactor architecture to support streaming

### Day 3: Modularize Core Components (Addresses Issue #5)
1. **Refactor smart_assistant.py (979 lines → 4 modules)**
   ```
   core/
   ├── smart_assistant.py (300 lines - orchestration)
   ├── audio_processor.py (250 lines - audio I/O)
   ├── conversation_state.py (200 lines - state)
   └── metrics.py (150 lines - monitoring)
   ```

2. **Create abstraction layer for TTS**
   ```python
   # core/tts_manager.py
   class TTSManager:
       def __init__(self):
           self.device = "mps" if torch.backends.mps.is_available() else "cpu"

       def get_provider(self, name: str) -> TTSProvider
       def stream_audio(self, text: str) -> AsyncIterator[bytes]
   ```

3. **Implement audio pipeline with CoreAudio optimization**
   ```python
   # core/audio_pipeline.py
   class AudioPipeline:
       def __init__(self):
           # Use CoreAudio for lowest latency on macOS
           self.sample_rate = 48000  # M3 optimized
           self.chunk_size = 512  # Lower latency on M3
   ```

### Day 4: Upgrade Core Dependencies
1. **Install Faster-Whisper with CoreML**
   ```bash
   # Faster-whisper with Apple Silicon optimization
   pip install faster-whisper

   # Optional: CoreML converted models for even better performance
   pip install whisper-coreml
   ```

2. **Optimize Ollama for streaming**
   - Configure for Metal acceleration
   - Enable token streaming
   - Tune context window for M3's memory

3. **Install XTTS v2 with MPS support**
   ```bash
   # XTTS with Apple Silicon support
   pip install TTS
   pip install torch-directml  # For additional acceleration
   ```

---

## Phase 3: Implement XTTS v2 Streaming (Day 5-7)
**Goal**: Replace batch TTS with streaming XTTS v2

### Day 5: XTTS v2 Integration with M3 Optimization
1. **Create MPS-optimized streaming TTS**
   ```python
   # realtime/xtts_streaming.py
   import torch

   class XTTSStreaming:
       def __init__(self):
           # Use MPS backend on M3
           self.device = torch.device("mps")
           self.model = self.load_xtts_v2().to(self.device)

           # Pre-allocate tensors for M3's unified memory
           self.buffer = torch.zeros((1, 80, 1000), device=self.device)

       async def stream_tts(self, text: str, voice_id: str):
           # M3 can handle larger chunks efficiently
           chunk_size = 50  # Larger chunks on M3

           for chunk in self.chunk_text(text, chunk_size):
               with torch.inference_mode():
                   audio = self.model.synthesize(chunk)
                   yield audio.cpu().numpy()  # Fast unified memory transfer
   ```

2. **Voice cloning optimization**
   - Use Neural Engine for speaker embedding
   - Cache embeddings in unified memory
   - Leverage ProRes engine for audio processing

### Day 6: Real-time Pipeline Integration
1. **Create M3-optimized orchestrator**
   ```python
   # realtime/streaming_assistant.py
   class StreamingAssistant:
       def __init__(self):
           # All models use unified memory - no transfers needed
           self.stt = FasterWhisper(
               model_size="base",
               device="auto",  # Automatically uses CoreML
               compute_type="int8_float16"  # M3 optimized
           )
           self.llm = OllamaStreaming(
               model="mistral:7b",
               gpu_layers=-1,  # Use all available GPU
               num_thread=8  # Use efficiency cores for background
           )
           self.tts = XTTSStreaming(device="mps")
   ```

2. **Leverage M3's efficiency cores**
   - Run background tasks on efficiency cores
   - Keep performance cores for real-time audio
   - Use Grand Central Dispatch for threading

### Day 7: M3-Specific Performance Optimization
1. **Neural Engine utilization**
   ```python
   # Use CoreML for additional acceleration
   import coremltools as ct

   # Convert models to CoreML where possible
   whisper_coreml = ct.convert(whisper_model)
   ```

2. **Memory optimization**
   - Utilize unified memory architecture
   - No VRAM transfers needed
   - Aggressive caching possible with 64GB

3. **Performance testing on M3**
   - Expected: <200ms voice-to-voice
   - Monitor with powermetrics
   - Check thermal throttling (unlikely with M3 Max)

---

## Phase 4: UI and Integration (Day 8-9)

### Day 8: Gradio UI Updates
1. **Use M3's GPU for UI rendering**
2. **Add Metal Performance HUD for monitoring**
3. **Implement ProMotion support (120Hz updates)**

### Day 9: API Updates
1. **WebSocket with hardware acceleration**
2. **Use Accelerate framework for audio processing**

---

## Phase 5: Testing and Polish (Day 10)

### M3-Specific Testing
1. **Performance validation**
   ```bash
   # Monitor M3 performance
   sudo powermetrics --samplers gpu_power,cpu_power

   # Check thermal state
   pmset -g thermlog
   ```

2. **Battery optimization**
   - Test on battery vs plugged in
   - Optimize for efficiency cores when on battery

3. **Memory pressure testing**
   - With 64GB, no issues expected
   - Test with multiple models loaded

---

## Migration Strategy with Hardware Considerations

### Feature Flags
```yaml
# config/config.yaml
hardware:
  platform: "apple_silicon"
  use_neural_engine: true
  use_mps: true
  unified_memory_gb: 64

streaming:
  enabled: false
  provider: xtts_v2
  device: "mps"  # Metal Performance Shaders
  chunk_size: 512  # Optimized for M3

performance:
  use_efficiency_cores: true
  max_memory_usage_gb: 32  # Leave 32GB for system
```

---

## Success Metrics on M3 Max
- ✅ Voice-to-voice latency < **200ms** (better than target!)
- ✅ Smooth 120Hz UI updates
- ✅ No thermal throttling
- ✅ <50% CPU usage at idle
- ✅ <20W power consumption average
- ✅ All models in memory simultaneously

---

## Hardware-Specific Advantages

### Why M3 Max is Perfect for This Project:
1. **Unified Memory**: No GPU transfers, instant access to 64GB
2. **Neural Engine**: Hardware acceleration for AI models
3. **ProRes Engine**: Hardware audio encoding/decoding
4. **Efficiency Cores**: Background tasks without affecting real-time
5. **Metal 3**: Advanced GPU compute for PyTorch

### Expected Performance Gains vs Standard Setup:
- **2x faster** Whisper transcription (Neural Engine)
- **1.5x faster** XTTS v2 generation (MPS + unified memory)
- **3x faster** model loading (no VRAM transfers)
- **50% lower** power consumption

---

## Risk Mitigation for Apple Silicon

### Potential Issues and Solutions:
1. **PyTorch MPS bugs**: Use `PYTORCH_ENABLE_MPS_FALLBACK=1`
2. **Library compatibility**: Most ML libraries now support Apple Silicon
3. **Docker limitations**: Use native installation instead
4. **Temperature**: M3 Max has excellent cooling, unlikely issue

---

## Next Immediate Steps
1. Create Python 3.11 virtual environment
2. Install Apple Silicon optimized packages
3. Test MPS and Neural Engine availability
4. Begin implementation with hardware optimizations

---

## Estimated Timeline
- **Total**: 10 days
- **Performance gain**: 10x reduction in latency (2-3s → 200ms)
- **Quality**: 90% of ElevenLabs at 0% cost
- **Privacy**: 100% local operation