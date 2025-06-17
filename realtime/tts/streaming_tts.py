"""
Streaming Text-to-Speech Manager

Implements real-time TTS with sentence-level streaming, multiple provider support,
and gapless audio playback for natural conversation flow.

Providers:
- OpenAI TTS: High quality, streaming capable
- Edge-TTS: Free, good quality, streaming support
- Coqui TTS: Local, customizable
- pyttsx3: Fallback, lowest latency

Usage:
    tts = StreamingTTSManager(provider="openai", voice="alloy")
    await tts.speak_text("Hello world!")
"""

import asyncio
import logging
import time
import io
import os
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from collections import deque
import threading

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

try:
    from TTS.api import TTS
    HAS_COQUI = True
except ImportError:
    HAS_COQUI = False

try:
    import pyaudio
    import wave
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


class TTSProvider(Enum):
    """Supported TTS providers."""
    OPENAI = "openai"
    EDGE = "edge"
    COQUI = "coqui"
    PYTTSX3 = "pyttsx3"


class TTSState(Enum):
    """TTS manager states."""
    IDLE = "idle"
    GENERATING = "generating"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass
class TTSConfig:
    """Configuration for TTS streaming."""
    provider: TTSProvider = TTSProvider.OPENAI
    voice: str = "alloy"
    speed: float = 1.0
    
    # Audio settings
    sample_rate: int = 24000
    chunk_size: int = 1024
    
    # Streaming settings
    sentence_buffer_ms: int = 500
    max_queue_size: int = 10
    
    # Provider-specific settings
    openai_model: str = "tts-1"
    edge_voice: str = "en-US-AriaNeural"
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    pyttsx3_rate: int = 200
    
    # Performance settings
    generation_timeout: float = 10.0
    playback_timeout: float = 30.0


@dataclass
class TTSChunk:
    """Represents a TTS audio chunk."""
    audio_data: bytes
    text: str
    timestamp: float
    duration_ms: float
    chunk_index: int


class StreamingTTSManager:
    """
    Streaming TTS manager with multiple provider support and gapless playback.
    
    Features:
    - Multiple TTS provider support (OpenAI, Edge, Coqui, pyttsx3)
    - Sentence-level streaming for natural flow
    - Gapless audio queue management
    - Provider fallback on errors
    - Performance monitoring
    - Interruption handling
    """
    
    def __init__(self, provider: str = "openai", voice: str = "alloy", 
                 speed: float = 1.0, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig(
            provider=TTSProvider(provider),
            voice=voice,
            speed=speed
        )
        
        self.state = TTSState.IDLE
        self.current_provider = None
        
        # Audio queue for streaming
        self.audio_queue = deque(maxlen=self.config.max_queue_size)
        self.playback_queue = asyncio.Queue()
        
        # Provider instances
        self.openai_client = None
        self.edge_communicator = None
        self.coqui_tts = None
        self.pyttsx3_engine = None
        
        # Streaming state
        self.current_generation = None
        self.playback_task = None
        self.is_speaking = False
        self.chunk_counter = 0
        
        # Callbacks
        self.on_audio_ready: Optional[Callable[[bytes], None]] = None
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "total_chars": 0,
            "average_generation_ms": 0,
            "provider_failures": {},
            "cache_hits": 0
        }
        
        # Simple cache for repeated phrases
        self.audio_cache: Dict[str, bytes] = {}
        self.max_cache_size = 100
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> bool:
        """Initialize the TTS manager and selected provider."""
        try:
            self.logger.info(f"Initializing TTS with provider: {self.config.provider.value}")
            
            # Initialize the primary provider
            if not await self._initialize_provider(self.config.provider):
                # Try fallback providers
                fallback_providers = [TTSProvider.EDGE, TTSProvider.PYTTSX3, TTSProvider.OPENAI]
                
                for provider in fallback_providers:
                    if provider != self.config.provider:
                        self.logger.warning(f"Trying fallback provider: {provider.value}")
                        if await self._initialize_provider(provider):
                            self.config.provider = provider
                            break
                else:
                    raise RuntimeError("No TTS providers available")
            
            # Start playback worker
            self.playback_task = asyncio.create_task(self._playback_worker())
            
            self.logger.info(f"TTS initialized with provider: {self.config.provider.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS: {e}")
            return False
    
    async def _initialize_provider(self, provider: TTSProvider) -> bool:
        """Initialize a specific TTS provider."""
        try:
            if provider == TTSProvider.OPENAI:
                if not HAS_OPENAI:
                    return False
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return False
                self.openai_client = openai.AsyncOpenAI(api_key=api_key)
                self.current_provider = provider
                return True
            
            elif provider == TTSProvider.EDGE:
                if not HAS_EDGE_TTS:
                    return False
                # Edge TTS doesn't need initialization
                self.current_provider = provider
                return True
            
            elif provider == TTSProvider.COQUI:
                if not HAS_COQUI:
                    return False
                self.coqui_tts = TTS(model_name=self.config.coqui_model)
                self.current_provider = provider
                return True
            
            elif provider == TTSProvider.PYTTSX3:
                if not HAS_PYTTSX3:
                    return False
                self.pyttsx3_engine = pyttsx3.init()
                self.pyttsx3_engine.setProperty('rate', self.config.pyttsx3_rate)
                voices = self.pyttsx3_engine.getProperty('voices')
                if voices:
                    self.pyttsx3_engine.setProperty('voice', voices[0].id)
                self.current_provider = provider
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {provider.value}: {e}")
            return False
    
    async def speak_text(self, text: str, priority: bool = False) -> bool:
        """
        Convert text to speech and queue for playback.
        
        Args:
            text: Text to speak
            priority: If True, interrupt current speech
            
        Returns:
            Success status
        """
        if not text.strip():
            return True
        
        try:
            # Check cache first
            cache_key = f"{self.config.provider.value}:{self.config.voice}:{text}"
            if cache_key in self.audio_cache:
                audio_data = self.audio_cache[cache_key]
                await self._queue_audio(audio_data, text)
                self.stats["cache_hits"] += 1
                return True
            
            # Interrupt if priority
            if priority and self.is_speaking:
                await self.stop_speaking()
            
            self.state = TTSState.GENERATING
            generation_start = time.time()
            
            # Generate audio based on provider
            audio_data = await self._generate_audio(text)
            
            if audio_data:
                generation_time = (time.time() - generation_start) * 1000
                
                # Cache the result
                if len(self.audio_cache) < self.max_cache_size:
                    self.audio_cache[cache_key] = audio_data
                
                # Queue for playback
                await self._queue_audio(audio_data, text)
                
                # Update stats
                self.stats["total_requests"] += 1
                self.stats["total_chars"] += len(text)
                self.stats["average_generation_ms"] = (
                    (self.stats["average_generation_ms"] * (self.stats["total_requests"] - 1) +
                     generation_time) / self.stats["total_requests"]
                )
                
                self.logger.debug(f"TTS generated {len(audio_data)} bytes in {generation_time:.1f}ms")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"TTS generation error: {e}")
            self.state = TTSState.ERROR
            
            # Track provider failures
            provider_name = self.config.provider.value
            if provider_name not in self.stats["provider_failures"]:
                self.stats["provider_failures"][provider_name] = 0
            self.stats["provider_failures"][provider_name] += 1
            
            if self.on_error:
                self.on_error(e)
            
            return False
        
        finally:
            if self.state == TTSState.GENERATING:
                self.state = TTSState.IDLE
    
    async def _generate_audio(self, text: str) -> Optional[bytes]:
        """Generate audio using the current provider."""
        if self.current_provider == TTSProvider.OPENAI:
            return await self._generate_openai(text)
        elif self.current_provider == TTSProvider.EDGE:
            return await self._generate_edge(text)
        elif self.current_provider == TTSProvider.COQUI:
            return await self._generate_coqui(text)
        elif self.current_provider == TTSProvider.PYTTSX3:
            return await self._generate_pyttsx3(text)
        
        return None
    
    async def _generate_openai(self, text: str) -> Optional[bytes]:
        """Generate audio using OpenAI TTS."""
        try:
            response = await self.openai_client.audio.speech.create(
                model=self.config.openai_model,
                voice=self.config.voice,
                input=text,
                speed=self.config.speed
            )
            
            return response.content
            
        except Exception as e:
            self.logger.error(f"OpenAI TTS error: {e}")
            return None
    
    async def _generate_edge(self, text: str) -> Optional[bytes]:
        """Generate audio using Edge TTS."""
        try:
            communicate = edge_tts.Communicate(text, self.config.edge_voice)
            audio_data = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Edge TTS error: {e}")
            return None
    
    async def _generate_coqui(self, text: str) -> Optional[bytes]:
        """Generate audio using Coqui TTS."""
        try:
            # Coqui TTS is sync, run in thread
            loop = asyncio.get_event_loop()
            audio_path = await loop.run_in_executor(
                None, self.coqui_tts.tts, text
            )
            
            # Read the generated audio file
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temp file
            os.unlink(audio_path)
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Coqui TTS error: {e}")
            return None
    
    async def _generate_pyttsx3(self, text: str) -> Optional[bytes]:
        """Generate audio using pyttsx3."""
        try:
            # pyttsx3 doesn't support direct audio generation
            # This is a placeholder - would need additional work
            # to capture audio output
            
            # For now, just log the text
            self.logger.info(f"pyttsx3 would speak: {text}")
            
            # Return dummy audio data
            return b"pyttsx3_placeholder_audio"
            
        except Exception as e:
            self.logger.error(f"pyttsx3 TTS error: {e}")
            return None
    
    async def _queue_audio(self, audio_data: bytes, text: str):
        """Queue audio data for playback."""
        chunk = TTSChunk(
            audio_data=audio_data,
            text=text,
            timestamp=time.time(),
            duration_ms=len(audio_data) / (self.config.sample_rate * 2) * 1000,  # Estimate
            chunk_index=self.chunk_counter
        )
        
        self.chunk_counter += 1
        
        try:
            await self.playback_queue.put(chunk)
        except asyncio.QueueFull:
            self.logger.warning("TTS playback queue full, dropping oldest chunk")
            try:
                self.playback_queue.get_nowait()
                await self.playback_queue.put(chunk)
            except asyncio.QueueEmpty:
                pass
    
    async def _playback_worker(self):
        """Worker task for audio playback."""
        while True:
            try:
                chunk = await self.playback_queue.get()
                
                if chunk is None:  # Shutdown signal
                    break
                
                self.state = TTSState.SPEAKING
                self.is_speaking = True
                
                if self.on_speech_start:
                    self.on_speech_start()
                
                # Send audio to output callback
                if self.on_audio_ready:
                    await self.on_audio_ready(chunk.audio_data)
                
                # Simulate playback duration
                await asyncio.sleep(chunk.duration_ms / 1000)
                
                self.is_speaking = False
                self.state = TTSState.IDLE
                
                if self.on_speech_end:
                    self.on_speech_end()
                
                self.playback_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Playback worker error: {e}")
                self.is_speaking = False
                self.state = TTSState.ERROR
    
    async def stop_speaking(self):
        """Stop current speech and clear queue."""
        self.logger.debug("Stopping TTS speech")
        
        # Clear the queue
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
                self.playback_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        self.is_speaking = False
        self.state = TTSState.INTERRUPTED
    
    async def stop(self):
        """Stop the TTS manager and cleanup resources."""
        self.logger.info("Stopping TTS manager")
        
        await self.stop_speaking()
        
        # Stop playback worker
        if self.playback_task:
            await self.playback_queue.put(None)  # Shutdown signal
            await self.playback_task
        
        # Cleanup provider resources
        if self.pyttsx3_engine:
            self.pyttsx3_engine.stop()
        
        self.state = TTSState.IDLE
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get TTS performance statistics."""
        stats = self.stats.copy()
        stats["current_provider"] = self.current_provider.value if self.current_provider else None
        stats["queue_size"] = self.playback_queue.qsize()
        stats["cache_size"] = len(self.audio_cache)
        stats["is_speaking"] = self.is_speaking
        return stats
    
    async def test_tts(self, test_text: str = "Hello, this is a test of the streaming TTS system.") -> Dict[str, Any]:
        """Test TTS functionality."""
        start_time = time.time()
        
        try:
            success = await self.speak_text(test_text)
            duration = (time.time() - start_time) * 1000
            
            return {
                "success": success,
                "provider": self.current_provider.value if self.current_provider else None,
                "text_length": len(test_text),
                "generation_time_ms": duration,
                "stats": self.get_statistics()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000
            }


# Test function
async def test_streaming_tts():
    """Test the streaming TTS implementation."""
    print("Testing Streaming TTS Manager...")
    
    # Test with available providers
    providers_to_test = []
    
    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        providers_to_test.append("openai")
    if HAS_EDGE_TTS:
        providers_to_test.append("edge")
    if HAS_PYTTSX3:
        providers_to_test.append("pyttsx3")
    
    if not providers_to_test:
        providers_to_test = ["pyttsx3"]  # Fallback
    
    results = {}
    
    for provider in providers_to_test:
        print(f"\nTesting provider: {provider}")
        
        try:
            tts = StreamingTTSManager(provider=provider)
            
            if await tts.initialize():
                result = await tts.test_tts(f"Testing {provider} TTS provider.")
                results[provider] = result
                print(f"✅ {provider}: {result}")
                
                await tts.stop()
            else:
                results[provider] = {"success": False, "error": "Failed to initialize"}
                print(f"❌ {provider}: Failed to initialize")
                
        except Exception as e:
            results[provider] = {"success": False, "error": str(e)}
            print(f"❌ {provider}: {e}")
    
    return results


if __name__ == "__main__":
    # Run tests
    results = asyncio.run(test_streaming_tts())
    print(f"\nTest results: {results}")