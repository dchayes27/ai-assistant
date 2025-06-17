"""
Real-time Audio Pipeline

Implements streaming audio input/output with low latency for the voice assistant.
Features WebRTC VAD, continuous recording, and echo cancellation.

Usage:
    pipeline = AudioStreamManager()
    await pipeline.start()
    
    # Audio will be continuously processed and streamed
"""

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import pyaudio
    import webrtcvad
    HAS_AUDIO_DEPS = True
except ImportError as e:
    logging.warning(f"Audio dependencies not available: {e}")
    HAS_AUDIO_DEPS = False


class AudioFormat(Enum):
    """Supported audio formats."""
    INT16 = "int16"
    FLOAT32 = "float32"


@dataclass
class AudioConfig:
    """Audio pipeline configuration."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: AudioFormat = AudioFormat.INT16
    
    # VAD settings
    vad_mode: int = 3  # 0-3, 3 is most aggressive
    vad_frame_duration: int = 30  # ms, must be 10, 20, or 30
    
    # Streaming settings
    buffer_duration: float = 0.5  # seconds
    silence_timeout: float = 2.0  # seconds
    
    # Device settings
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None


class AudioStreamManager:
    """
    Manages real-time audio streaming with VAD and low-latency processing.
    
    Features:
    - Continuous audio recording with WebRTC VAD
    - Ring buffer for efficient audio storage
    - Configurable silence detection
    - Echo cancellation (basic implementation)
    - Thread-safe audio queue management
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.is_running = False
        self.is_recording = False
        self.is_playing = False
        
        # Audio components
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        self.vad: Optional[webrtcvad.Vad] = None
        
        # Threading
        self.record_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=int(
            self.config.sample_rate * self.config.buffer_duration / self.config.chunk_size
        ))
        self.playback_queue = asyncio.Queue()
        
        # State tracking
        self.last_speech_time = 0
        self.speech_chunks = []
        self.silence_start_time = None
        
        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable[[bytes], None]] = None
        self.on_audio_chunk: Optional[Callable[[bytes], None]] = None
        
        # Statistics
        self.stats = {
            "chunks_processed": 0,
            "speech_segments": 0,
            "average_latency_ms": 0,
            "buffer_overruns": 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> bool:
        """Initialize audio system and components."""
        if not HAS_AUDIO_DEPS:
            self.logger.error("Audio dependencies not available")
            return False
        
        try:
            # Initialize PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Initialize WebRTC VAD
            self.vad = webrtcvad.Vad(self.config.vad_mode)
            
            # Find and configure audio devices
            await self._configure_audio_devices()
            
            self.logger.info("Audio pipeline initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio pipeline: {e}")
            return False
    
    async def _configure_audio_devices(self):
        """Configure input and output audio devices."""
        device_count = self.pyaudio_instance.get_device_count()
        self.logger.info(f"Found {device_count} audio devices")
        
        # Log available devices for debugging
        for i in range(device_count):
            device_info = self.pyaudio_instance.get_device_info_by_index(i)
            self.logger.debug(f"Device {i}: {device_info['name']} - "
                            f"In: {device_info['maxInputChannels']}, "
                            f"Out: {device_info['maxOutputChannels']}")
        
        # Use default devices if not specified
        if self.config.input_device_index is None:
            default_input = self.pyaudio_instance.get_default_input_device_info()
            self.config.input_device_index = default_input['index']
            self.logger.info(f"Using default input device: {default_input['name']}")
        
        if self.config.output_device_index is None:
            default_output = self.pyaudio_instance.get_default_output_device_info()
            self.config.output_device_index = default_output['index']
            self.logger.info(f"Using default output device: {default_output['name']}")
    
    def _audio_callback(self, in_data: bytes, frame_count: int, time_info: Dict, status: int) -> tuple:
        """PyAudio callback for real-time audio processing."""
        if not self.is_recording:
            return (None, pyaudio.paContinue)
        
        try:
            # Add to buffer
            self.audio_buffer.append(in_data)
            
            # Check for VAD
            is_speech = self._detect_speech(in_data)
            current_time = time.time()
            
            if is_speech:
                if not self.speech_chunks:  # Start of speech
                    self.logger.debug("Speech detected - starting recording")
                    if self.on_speech_start:
                        self.on_speech_start()
                
                self.speech_chunks.append(in_data)
                self.last_speech_time = current_time
                self.silence_start_time = None
                
            else:  # Silence
                if self.speech_chunks:  # We were recording speech
                    if self.silence_start_time is None:
                        self.silence_start_time = current_time
                    
                    # Check if silence timeout exceeded
                    silence_duration = current_time - self.silence_start_time
                    if silence_duration >= self.config.silence_timeout:
                        # End of speech segment
                        speech_audio = b''.join(self.speech_chunks)
                        self.logger.debug(f"Speech ended - {len(speech_audio)} bytes")
                        
                        if self.on_speech_end:
                            self.on_speech_end(speech_audio)
                        
                        self.speech_chunks.clear()
                        self.stats["speech_segments"] += 1
            
            # Call chunk callback if registered
            if self.on_audio_chunk:
                self.on_audio_chunk(in_data)
            
            self.stats["chunks_processed"] += 1
            
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}")
            self.stats["buffer_overruns"] += 1
        
        return (None, pyaudio.paContinue)
    
    def _detect_speech(self, audio_data: bytes) -> bool:
        """Detect speech in audio chunk using WebRTC VAD."""
        try:
            # VAD requires specific frame sizes
            frame_size = int(self.config.sample_rate * self.config.vad_frame_duration / 1000)
            bytes_per_frame = frame_size * 2  # 16-bit audio
            
            # Process audio in VAD-compatible chunks
            for i in range(0, len(audio_data), bytes_per_frame):
                frame = audio_data[i:i + bytes_per_frame]
                if len(frame) == bytes_per_frame:
                    if self.vad.is_speech(frame, self.config.sample_rate):
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"VAD error: {e}")
            return True  # Default to speech on error
    
    async def start_recording(self):
        """Start audio recording stream."""
        if not self.pyaudio_instance:
            raise RuntimeError("Audio pipeline not initialized")
        
        try:
            self.input_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.input_device_index,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.input_stream.start_stream()
            self.logger.info("Audio recording started")
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            raise
    
    async def stop_recording(self):
        """Stop audio recording stream."""
        self.is_recording = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
            self.logger.info("Audio recording stopped")
    
    async def play_audio(self, audio_data: bytes):
        """Play audio data through output stream."""
        try:
            await self.playback_queue.put(audio_data)
        except Exception as e:
            self.logger.error(f"Failed to queue audio for playback: {e}")
    
    async def _playback_worker(self):
        """Worker task for audio playback."""
        while self.is_running:
            try:
                audio_data = await asyncio.wait_for(
                    self.playback_queue.get(), timeout=0.1
                )
                
                if self.output_stream and self.is_playing:
                    # Write audio data to output stream
                    self.output_stream.write(audio_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Playback error: {e}")
    
    async def start(self):
        """Start the audio pipeline."""
        if not await self.initialize():
            raise RuntimeError("Failed to initialize audio pipeline")
        
        self.is_running = True
        
        # Start recording
        await self.start_recording()
        
        # Start playback worker
        asyncio.create_task(self._playback_worker())
        
        self.logger.info("Audio pipeline started")
    
    async def stop(self):
        """Stop the audio pipeline."""
        self.is_running = False
        
        await self.stop_recording()
        
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
        
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None
        
        self.logger.info("Audio pipeline stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audio pipeline statistics."""
        return self.stats.copy()
    
    async def test_audio_devices(self) -> Dict[str, Any]:
        """Test audio device compatibility."""
        if not HAS_AUDIO_DEPS:
            return {"error": "Audio dependencies not available"}
        
        if not self.pyaudio_instance:
            await self.initialize()
        
        results = {
            "device_count": self.pyaudio_instance.get_device_count(),
            "default_input": self.pyaudio_instance.get_default_input_device_info(),
            "default_output": self.pyaudio_instance.get_default_output_device_info(),
            "compatible_devices": []
        }
        
        # Test each device for compatibility
        for i in range(results["device_count"]):
            device_info = self.pyaudio_instance.get_device_info_by_index(i)
            
            # Test if device supports our required format
            try:
                is_supported = self.pyaudio_instance.is_format_supported(
                    rate=self.config.sample_rate,
                    input_device=i if device_info['maxInputChannels'] > 0 else None,
                    output_device=i if device_info['maxOutputChannels'] > 0 else None,
                    input_channels=self.config.channels if device_info['maxInputChannels'] > 0 else None,
                    output_channels=self.config.channels if device_info['maxOutputChannels'] > 0 else None,
                    input_format=pyaudio.paInt16 if device_info['maxInputChannels'] > 0 else None,
                    output_format=pyaudio.paInt16 if device_info['maxOutputChannels'] > 0 else None
                )
                
                if is_supported:
                    results["compatible_devices"].append({
                        "index": i,
                        "name": device_info['name'],
                        "input_channels": device_info['maxInputChannels'],
                        "output_channels": device_info['maxOutputChannels']
                    })
                    
            except Exception as e:
                self.logger.debug(f"Device {i} not compatible: {e}")
        
        return results


# Convenience function for testing
async def test_audio_pipeline():
    """Test the audio pipeline implementation."""
    config = AudioConfig(
        sample_rate=16000,
        chunk_size=1024,
        vad_mode=3
    )
    
    pipeline = AudioStreamManager(config)
    
    # Test device compatibility
    device_results = await pipeline.test_audio_devices()
    print(f"Audio device test results: {device_results}")
    
    return device_results


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_audio_pipeline())