#!/usr/bin/env python3
"""
Baseline Performance Measurement

Measures current voice pipeline performance to establish baseline metrics
before implementing real-time streaming improvements.

Usage:
    python migration/baseline_performance.py --output migration/current_baseline.txt
"""

import os
import sys
import time
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import statistics

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.smart_assistant import SmartAssistant
    from memory import DatabaseManager
    from config.manager import ConfigManager
except ImportError as e:
    print(f"Warning: Could not import core components: {e}")
    print("This is expected if core components are not yet implemented.")
    SmartAssistant = None


class BaselineProfiler:
    """Profile current voice assistant performance."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "version": "current",
            "metrics": {},
            "test_scenarios": []
        }
    
    async def measure_component_latency(self, component_name: str, test_func, *args, **kwargs):
        """Measure latency of a specific component."""
        latencies = []
        
        for i in range(5):  # Run 5 tests for average
            start_time = time.perf_counter()
            try:
                await test_func(*args, **kwargs) if asyncio.iscoroutinefunction(test_func) else test_func(*args, **kwargs)
                end_time = time.perf_counter()
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            except Exception as e:
                print(f"Error testing {component_name}: {e}")
                latencies.append(float('inf'))
        
        return {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
    
    def test_audio_initialization(self):
        """Test audio system initialization time."""
        try:
            import pyaudio
            start = time.perf_counter()
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            p.terminate()
            end = time.perf_counter()
            return (end - start) * 1000, device_count
        except ImportError:
            return float('inf'), 0
    
    def test_whisper_loading(self):
        """Test Whisper model loading time."""
        try:
            import whisper
            start = time.perf_counter()
            model = whisper.load_model("base")
            end = time.perf_counter()
            return (end - start) * 1000
        except ImportError:
            return float('inf')
    
    def test_database_connection(self):
        """Test database connection time."""
        try:
            start = time.perf_counter()
            # Test database connection
            db_path = "data/assistant.db"
            if os.path.exists(db_path):
                import sqlite3
                conn = sqlite3.connect(db_path)
                conn.close()
            end = time.perf_counter()
            return (end - start) * 1000
        except Exception:
            return float('inf')
    
    def test_config_loading(self):
        """Test configuration loading time."""
        try:
            start = time.perf_counter()
            if os.path.exists(self.config_path):
                import yaml
                with open(self.config_path, 'r') as f:
                    yaml.safe_load(f)
            end = time.perf_counter()
            return (end - start) * 1000
        except Exception:
            return float('inf')
    
    async def test_llm_response_time(self):
        """Test LLM response time simulation."""
        # Simulate LLM call - replace with actual implementation when available
        await asyncio.sleep(0.5)  # Simulate 500ms LLM response
        return 500  # Return simulated latency
    
    def run_baseline_tests(self):
        """Run all baseline performance tests."""
        print("🔍 Starting baseline performance measurement...")
        
        # Audio system tests
        print("Testing audio system initialization...")
        audio_latency, device_count = self.test_audio_initialization()
        self.results["metrics"]["audio_init_ms"] = audio_latency
        self.results["metrics"]["audio_devices"] = device_count
        
        # Whisper model loading
        print("Testing Whisper model loading...")
        whisper_latency = self.test_whisper_loading()
        self.results["metrics"]["whisper_load_ms"] = whisper_latency
        
        # Database connection
        print("Testing database connection...")
        db_latency = self.test_database_connection()
        self.results["metrics"]["database_connect_ms"] = db_latency
        
        # Configuration loading
        print("Testing configuration loading...")
        config_latency = self.test_config_loading()
        self.results["metrics"]["config_load_ms"] = config_latency
        
        # System info
        try:
            import psutil
            self.results["metrics"]["cpu_count"] = psutil.cpu_count()
            self.results["metrics"]["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            pass
        
        # Estimated current voice-to-voice latency
        estimated_total = audio_latency + whisper_latency + 500 + 200  # Audio + STT + LLM + TTS estimate
        self.results["metrics"]["estimated_voice_to_voice_ms"] = estimated_total
        
        # Target improvements
        self.results["targets"] = {
            "voice_to_voice_ms": 500,
            "time_to_first_token_ms": 200,
            "interruption_response_ms": 200,
            "audio_quality_score": 0.95
        }
        
        print("✅ Baseline performance measurement complete!")
        return self.results
    
    def save_results(self, output_path: str):
        """Save baseline results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📊 Baseline results saved to: {output_path}")
    
    def print_summary(self):
        """Print a summary of baseline results."""
        metrics = self.results["metrics"]
        targets = self.results["targets"]
        
        print("\n📈 BASELINE PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Audio Initialization: {metrics.get('audio_init_ms', 'N/A'):.1f}ms")
        print(f"Whisper Model Loading: {metrics.get('whisper_load_ms', 'N/A'):.1f}ms")
        print(f"Database Connection: {metrics.get('database_connect_ms', 'N/A'):.1f}ms")
        print(f"Config Loading: {metrics.get('config_load_ms', 'N/A'):.1f}ms")
        print(f"Audio Devices Found: {metrics.get('audio_devices', 'N/A')}")
        
        print(f"\n🎯 ESTIMATED CURRENT PERFORMANCE:")
        print(f"Voice-to-Voice Latency: {metrics.get('estimated_voice_to_voice_ms', 'N/A'):.1f}ms")
        
        print(f"\n🚀 STREAMING TARGETS:")
        print(f"Target Voice-to-Voice: {targets['voice_to_voice_ms']}ms")
        print(f"Target Time-to-First-Token: {targets['time_to_first_token_ms']}ms")
        print(f"Target Interruption Response: {targets['interruption_response_ms']}ms")
        
        improvement_factor = metrics.get('estimated_voice_to_voice_ms', 1000) / targets['voice_to_voice_ms']
        print(f"\n💡 Expected Improvement: {improvement_factor:.1f}x faster")


def main():
    parser = argparse.ArgumentParser(description="Measure baseline performance")
    parser.add_argument("--output", default="migration/current_baseline.txt",
                       help="Output file for baseline results")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    profiler = BaselineProfiler(args.config)
    results = profiler.run_baseline_tests()
    profiler.print_summary()
    profiler.save_results(args.output)


if __name__ == "__main__":
    main()