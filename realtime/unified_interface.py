"""
Unified Real-time Streaming Interface

Main orchestration class that coordinates all streaming components for the
real-time voice assistant. Provides a single entry point for the complete
voice-to-voice pipeline with sub-500ms latency.

Usage:
    interface = UnifiedStreamingInterface()
    await interface.start()
    
    # Real-time voice assistant now running
    # Voice → Audio Pipeline → LLM Streaming → TTS → Audio Output
"""

import asyncio
import logging
import time
import argparse
import signal
import sys
import os
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import json

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from realtime.streaming_voice_agent import StreamingVoiceAgent, StreamingAgentConfig, AgentState
    from realtime.audio.pipeline import AudioConfig
    from realtime.llm.gpt4_streaming import StreamingConfig
    from realtime.tts.streaming_tts import TTSConfig, TTSProvider
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Streaming components not available: {e}")
    COMPONENTS_AVAILABLE = False


class SystemMode(Enum):
    """System operating modes."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    CONTINUOUS = "continuous"


@dataclass
class SystemConfig:
    """Unified system configuration."""
    mode: SystemMode = SystemMode.DEVELOPMENT
    
    # Model selection
    llm_model: str = "gpt-4o"
    tts_provider: str = "edge"  # edge, openai, pyttsx3
    
    # Performance targets
    target_latency_ms: int = 500
    enable_interruptions: bool = True
    
    # Feature flags
    enable_monitoring: bool = True
    enable_web_interface: bool = False
    enable_tools: bool = True
    
    # Audio settings
    audio_input_device: Optional[int] = None
    audio_output_device: Optional[int] = None
    
    # API keys
    openai_api_key: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None


class UnifiedStreamingInterface:
    """
    Unified interface for the real-time streaming voice assistant.
    
    Coordinates:
    - Audio input/output pipeline
    - LLM streaming with OpenAI GPT-4o
    - TTS streaming with multiple providers
    - Performance monitoring
    - Web interface (optional)
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        
        # Core components
        self.voice_agent: Optional[StreamingVoiceAgent] = None
        self.web_server = None
        
        # System state
        self.is_running = False
        self.start_time = 0
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.session_metrics = {
            "start_time": 0,
            "total_interactions": 0,
            "average_latency_ms": 0,
            "uptime_seconds": 0,
            "errors": 0
        }
        
        # Callbacks
        self.on_ready: Optional[Callable] = None
        self.on_shutdown: Optional[Callable] = None
        self.on_interaction: Optional[Callable[[str, str], None]] = None  # (user_text, assistant_text)
        
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """Configure logging based on system config."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.config.log_file
        )
        
        # Also log to console in development
        if self.config.mode == SystemMode.DEVELOPMENT:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)
    
    async def initialize(self) -> bool:
        """Initialize all system components."""
        if not COMPONENTS_AVAILABLE:
            self.logger.error("Streaming components not available")
            return False
        
        try:
            self.logger.info(f"Initializing unified streaming interface in {self.config.mode.value} mode")
            
            # Configure audio
            audio_config = AudioConfig(
                input_device_index=self.config.audio_input_device,
                output_device_index=self.config.audio_output_device
            )
            
            # Configure LLM
            llm_config = StreamingConfig(
                model=self.config.llm_model,
                enable_tools=self.config.enable_tools
            )
            
            # Configure TTS
            tts_config = TTSConfig(
                provider=TTSProvider(self.config.tts_provider)
            )
            
            # Create agent configuration
            agent_config = StreamingAgentConfig(
                audio_config=audio_config,
                llm_config=llm_config,
                openai_api_key=self.config.openai_api_key or os.getenv("OPENAI_API_KEY"),
                tts_provider=self.config.tts_provider,
                target_voice_to_voice_ms=self.config.target_latency_ms,
                enable_interruptions=self.config.enable_interruptions,
                enable_tools=self.config.enable_tools,
                enable_monitoring=self.config.enable_monitoring
            )
            
            # Initialize voice agent
            self.voice_agent = StreamingVoiceAgent(agent_config)
            
            if not await self.voice_agent.initialize():
                raise RuntimeError("Failed to initialize voice agent")
            
            # Set up callbacks
            self.voice_agent.on_state_change = self._handle_state_change
            self.voice_agent.on_transcription = self._handle_transcription
            self.voice_agent.on_response_complete = self._handle_response_complete
            self.voice_agent.on_error = self._handle_error
            
            # Initialize web interface if enabled
            if self.config.enable_web_interface:
                await self._initialize_web_interface()
            
            self.logger.info("Unified streaming interface initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize unified interface: {e}")
            return False
    
    async def _initialize_web_interface(self):
        """Initialize web interface for real-time interaction."""
        # Placeholder for web interface initialization
        # Would integrate with web_realtime/ components
        self.logger.info("Web interface initialization (placeholder)")
        pass
    
    def _handle_state_change(self, new_state: AgentState):
        """Handle voice agent state changes."""
        self.logger.debug(f"Agent state: {new_state.value}")
        
        if new_state == AgentState.ERROR:
            self.session_metrics["errors"] += 1
    
    def _handle_transcription(self, text: str):
        """Handle user speech transcription."""
        self.logger.info(f"User: {text}")
        self.session_metrics["total_interactions"] += 1
    
    def _handle_response_complete(self, text: str):
        """Handle assistant response completion."""
        self.logger.info(f"Assistant: {text}")
        
        if self.on_interaction:
            # Would need to store user text to pass both
            self.on_interaction("", text)
    
    def _handle_error(self, error: Exception):
        """Handle system errors."""
        self.logger.error(f"System error: {error}")
        self.session_metrics["errors"] += 1
    
    async def start(self):
        """Start the unified streaming interface."""
        if not await self.initialize():
            raise RuntimeError("Failed to initialize system")
        
        try:
            self.is_running = True
            self.start_time = time.time()
            self.session_metrics["start_time"] = self.start_time
            
            # Set up signal handlers for graceful shutdown
            if hasattr(signal, 'SIGINT'):
                signal.signal(signal.SIGINT, self._signal_handler)
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start voice agent
            self.logger.info("🚀 Starting real-time streaming voice assistant...")
            
            # Start voice agent in background
            agent_task = asyncio.create_task(self.voice_agent.start_streaming())
            
            # Start monitoring task
            monitor_task = asyncio.create_task(self._monitoring_loop())
            
            # Start web interface if enabled
            web_task = None
            if self.config.enable_web_interface:
                web_task = asyncio.create_task(self._run_web_interface())
            
            # Notify ready
            if self.on_ready:
                self.on_ready()
            
            self._print_startup_info()
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cleanup
            self.logger.info("Shutting down...")
            
            agent_task.cancel()
            monitor_task.cancel()
            
            if web_task:
                web_task.cancel()
            
            await self._cleanup()
            
        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
            raise
        finally:
            self.is_running = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def shutdown(self):
        """Initiate graceful shutdown."""
        self.shutdown_event.set()
    
    async def _cleanup(self):
        """Cleanup system resources."""
        if self.voice_agent:
            await self.voice_agent.stop_streaming()
        
        if self.on_shutdown:
            self.on_shutdown()
        
        session_duration = time.time() - self.start_time
        self.session_metrics["uptime_seconds"] = session_duration
        
        self.logger.info(f"Session completed - Duration: {session_duration:.1f}s, "
                        f"Interactions: {self.session_metrics['total_interactions']}")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                if self.voice_agent:
                    stats = self.voice_agent.get_session_statistics()
                    
                    # Update metrics
                    self.session_metrics.update({
                        "uptime_seconds": time.time() - self.start_time,
                        "total_interactions": stats.get("total_interactions", 0),
                        "average_latency_ms": stats.get("average_response_time_ms", 0)
                    })
                    
                    # Log periodic status
                    if self.config.mode == SystemMode.DEVELOPMENT:
                        self.logger.debug(f"System status: {self.session_metrics}")
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    async def _run_web_interface(self):
        """Run web interface server."""
        # Placeholder for web interface
        self.logger.info("Web interface server (placeholder)")
        while self.is_running:
            await asyncio.sleep(1)
    
    def _print_startup_info(self):
        """Print startup information."""
        print("\n" + "="*60)
        print("🎤 REAL-TIME STREAMING VOICE ASSISTANT")
        print("="*60)
        print(f"Mode: {self.config.mode.value}")
        print(f"LLM Model: {self.config.llm_model}")
        print(f"TTS Provider: {self.config.tts_provider}")
        print(f"Target Latency: {self.config.target_latency_ms}ms")
        print(f"Interruptions: {'✅ Enabled' if self.config.enable_interruptions else '❌ Disabled'}")
        print(f"Tools: {'✅ Enabled' if self.config.enable_tools else '❌ Disabled'}")
        
        if self.config.enable_web_interface:
            print(f"Web Interface: ✅ Enabled")
        
        print("\n🎯 VOICE ASSISTANT IS READY!")
        print("   - Speak naturally for voice input")
        print("   - Assistant will respond in real-time")
        print("   - Interrupt responses by speaking")
        print("   - Press Ctrl+C to stop")
        print("="*60 + "\n")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "is_running": self.is_running,
            "mode": self.config.mode.value,
            "session_metrics": self.session_metrics.copy(),
            "config": {
                "llm_model": self.config.llm_model,
                "tts_provider": self.config.tts_provider,
                "target_latency_ms": self.config.target_latency_ms
            }
        }
        
        if self.voice_agent:
            status["agent_status"] = self.voice_agent.get_session_statistics()
        
        return status


def create_config_from_args(args) -> SystemConfig:
    """Create system config from command line arguments."""
    return SystemConfig(
        mode=SystemMode(args.mode),
        llm_model=args.model,
        tts_provider=args.tts,
        target_latency_ms=args.latency,
        enable_interruptions=not args.no_interruptions,
        enable_monitoring=not args.no_monitoring,
        enable_web_interface=args.web,
        enable_tools=not args.no_tools,
        openai_api_key=args.openai_key,
        log_level=args.log_level
    )


async def main():
    """Main entry point for the unified streaming interface."""
    parser = argparse.ArgumentParser(description="Real-time Streaming Voice Assistant")
    
    parser.add_argument("--mode", choices=["development", "production", "testing", "continuous"],
                       default="continuous", help="Operating mode")
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use")
    parser.add_argument("--tts", choices=["openai", "edge", "pyttsx3"], 
                       default="edge", help="TTS provider")
    parser.add_argument("--latency", type=int, default=500,
                       help="Target voice-to-voice latency in ms")
    parser.add_argument("--no-interruptions", action="store_true",
                       help="Disable interruption handling")
    parser.add_argument("--no-monitoring", action="store_true",
                       help="Disable performance monitoring")
    parser.add_argument("--web", action="store_true",
                       help="Enable web interface")
    parser.add_argument("--no-tools", action="store_true",
                       help="Disable tool calling")
    parser.add_argument("--openai-key", help="OpenAI API key")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Create and start interface
    interface = UnifiedStreamingInterface(config)
    
    try:
        await interface.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))