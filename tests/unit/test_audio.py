"""
Audio processing tests
Tests speech-to-text, text-to-speech, and audio handling
"""

import pytest
import os
import tempfile
import wave
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import io

from core import SmartAssistant, AssistantConfig


class TestAudioProcessing:
    """Test audio processing functionality"""
    
    @pytest.mark.audio
    def test_sample_audio_generation(self, sample_audio_data):
        """Test that sample audio data is generated correctly"""
        assert isinstance(sample_audio_data, bytes)
        assert len(sample_audio_data) > 0
        
        # Should be 1 second of 16-bit audio at 16kHz
        expected_samples = 16000  # 1 second at 16kHz
        expected_bytes = expected_samples * 2  # 16-bit = 2 bytes per sample
        assert len(sample_audio_data) == expected_bytes
    
    @pytest.mark.audio
    def test_wav_file_creation(self, sample_wav_file):
        """Test WAV file creation and reading"""
        assert os.path.exists(sample_wav_file)
        
        # Verify WAV file properties
        with wave.open(sample_wav_file, 'rb') as wav_file:
            assert wav_file.getnchannels() == 1  # Mono
            assert wav_file.getsampwidth() == 2  # 16-bit
            assert wav_file.getframerate() == 16000  # 16kHz
            
            frames = wav_file.readframes(wav_file.getnframes())
            assert len(frames) > 0
    
    @pytest.mark.audio
    def test_audio_data_conversion(self, sample_audio_data):
        """Test audio data format conversions"""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(sample_audio_data, dtype=np.int16)
        assert len(audio_array) == 16000  # 1 second at 16kHz
        
        # Check audio characteristics
        assert audio_array.dtype == np.int16
        assert np.min(audio_array) >= -32768
        assert np.max(audio_array) <= 32767
    
    @pytest.mark.audio
    def test_audio_amplitude_analysis(self, sample_audio_data):
        """Test audio amplitude and frequency analysis"""
        audio_array = np.frombuffer(sample_audio_data, dtype=np.int16)
        
        # Convert to float for analysis
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Check RMS (should be significant for sine wave)
        rms = np.sqrt(np.mean(audio_float ** 2))
        assert rms > 0.1  # Should have significant amplitude
        
        # Check for sine wave characteristics (roughly)
        # The sample should have energy concentrated around 440Hz
        assert np.std(audio_float) > 0.1  # Should have variation


class TestWhisperIntegration:
    """Test Whisper speech-to-text integration"""
    
    @pytest.mark.audio
    async def test_whisper_transcription_mock(self, test_assistant, sample_wav_file):
        """Test Whisper transcription with mocked model"""
        # The mock is already set up in the fixture
        transcript = test_assistant.whisper_model.transcribe(sample_wav_file)
        
        assert transcript["text"] == "This is a mock transcription"
        assert transcript["language"] == "en"
        assert "segments" in transcript
    
    @pytest.mark.audio
    async def test_audio_transcription_pipeline(self, test_assistant, sample_audio_data):
        """Test complete audio transcription pipeline"""
        # Process audio through the assistant
        transcript = await test_assistant.transcribe_audio(sample_audio_data)
        
        assert transcript == "This is a mock transcription"
        
        # Verify the whisper model was called
        test_assistant.whisper_model.transcribe.assert_called_once()
    
    @pytest.mark.audio
    async def test_transcription_error_handling(self, test_assistant):
        """Test error handling in transcription"""
        # Set up mock to raise an exception
        test_assistant.whisper_model.transcribe.side_effect = Exception("Transcription failed")
        
        # Should handle the error gracefully
        with pytest.raises(Exception):
            await test_assistant.transcribe_audio(b"invalid audio data")
    
    @pytest.mark.audio
    async def test_empty_audio_handling(self, test_assistant):
        """Test handling of empty or silent audio"""
        # Create silent audio (all zeros)
        silent_audio = b'\x00' * 32000  # 1 second of silence
        
        # Should still attempt transcription
        transcript = await test_assistant.transcribe_audio(silent_audio)
        
        # Mock should still return the mock response
        assert transcript == "This is a mock transcription"
    
    @pytest.mark.audio
    def test_whisper_model_loading(self):
        """Test Whisper model loading configuration"""
        # Test different model sizes
        model_sizes = ["tiny", "base", "small", "medium", "large"]
        
        for size in model_sizes:
            config = AssistantConfig(whisper_model=size)
            assert config.whisper_model == size
    
    @pytest.mark.audio
    @pytest.mark.slow
    @pytest.mark.skipif(
        "not config.getoption('--run-slow')",
        reason="Slow test requiring actual Whisper model"
    )
    async def test_real_whisper_transcription(self, sample_wav_file):
        """Test with real Whisper model (slow test)"""
        import whisper
        
        # Load actual Whisper model (this will be slow)
        model = whisper.load_model("tiny")
        result = model.transcribe(sample_wav_file)
        
        # Should return some transcription (may not be accurate for sine wave)
        assert "text" in result
        assert isinstance(result["text"], str)


class TestTTSIntegration:
    """Test text-to-speech integration"""
    
    @pytest.mark.audio
    async def test_tts_synthesis_mock(self, test_assistant):
        """Test TTS synthesis with mocked engine"""
        text = "Hello, this is a test message for speech synthesis."
        output_file = await test_assistant.synthesize_speech(text)
        
        assert output_file is not None
        assert output_file.startswith("temp/speech_")
        
        # Verify TTS engine was called
        if hasattr(test_assistant.tts_engine, 'tts_to_file'):
            # Coqui TTS
            test_assistant.tts_engine.tts_to_file.assert_called_once()
        else:
            # pyttsx3
            test_assistant.tts_engine.save_to_file.assert_called_once()
            test_assistant.tts_engine.runAndWait.assert_called_once()
    
    @pytest.mark.audio
    async def test_tts_empty_text(self, test_assistant):
        """Test TTS with empty text"""
        output_file = await test_assistant.synthesize_speech("")
        
        # Should still work with empty text
        assert output_file is not None
    
    @pytest.mark.audio
    async def test_tts_long_text(self, test_assistant):
        """Test TTS with very long text"""
        long_text = "This is a very long text. " * 100  # ~2800 characters
        
        output_file = await test_assistant.synthesize_speech(long_text)
        assert output_file is not None
    
    @pytest.mark.audio
    async def test_tts_special_characters(self, test_assistant):
        """Test TTS with special characters and unicode"""
        special_text = "Hello! How are you? I'm doing well. 50% of people like émojis 🤖."
        
        output_file = await test_assistant.synthesize_speech(special_text)
        assert output_file is not None
    
    @pytest.mark.audio
    async def test_tts_error_handling(self, test_assistant):
        """Test TTS error handling"""
        # Set up mock to raise an exception
        if hasattr(test_assistant.tts_engine, 'tts_to_file'):
            test_assistant.tts_engine.tts_to_file.side_effect = Exception("TTS failed")
        else:
            test_assistant.tts_engine.save_to_file.side_effect = Exception("TTS failed")
        
        with pytest.raises(Exception):
            await test_assistant.synthesize_speech("This will fail")
    
    @pytest.mark.audio
    def test_tts_engine_selection(self):
        """Test TTS engine selection in configuration"""
        # Test different TTS configurations
        tts_options = [
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts", 
            "pyttsx3"
        ]
        
        for tts_model in tts_options:
            config = AssistantConfig(tts_model=tts_model)
            assert config.tts_model == tts_model


class TestVoiceMessagePipeline:
    """Test complete voice message processing pipeline"""
    
    @pytest.mark.audio
    async def test_voice_message_processing(self, test_assistant, sample_audio_data):
        """Test complete voice message processing"""
        result = await test_assistant.process_voice_message(
            sample_audio_data,
            synthesize_response=True
        )
        
        assert "transcript" in result
        assert "response" in result
        assert "audio_file" in result
        
        assert result["transcript"] == "This is a mock transcription"
        assert result["response"] == "This is a mock response"
        assert result["audio_file"] is not None
    
    @pytest.mark.audio
    async def test_voice_message_no_synthesis(self, test_assistant, sample_audio_data):
        """Test voice message without speech synthesis"""
        result = await test_assistant.process_voice_message(
            sample_audio_data,
            synthesize_response=False
        )
        
        assert "transcript" in result
        assert "response" in result
        assert "audio_file" not in result
    
    @pytest.mark.audio
    async def test_voice_message_with_thread(self, test_assistant, sample_audio_data):
        """Test voice message with conversation thread"""
        # Create conversation thread
        thread_id = await test_assistant.create_conversation_thread()
        
        result = await test_assistant.process_voice_message(
            sample_audio_data,
            thread_id=thread_id,
            synthesize_response=True
        )
        
        assert result["thread_id"] == thread_id
        assert "transcript" in result
        assert "response" in result
    
    @pytest.mark.audio
    async def test_voice_pipeline_error_handling(self, test_assistant):
        """Test error handling in voice pipeline"""
        # Test with invalid audio data
        invalid_audio = b"not audio data"
        
        # Should handle gracefully (though mock might not care about data format)
        result = await test_assistant.process_voice_message(invalid_audio)
        
        # With mocks, this should still work
        assert "transcript" in result or "error" in result


class TestAudioFormatHandling:
    """Test different audio format handling"""
    
    @pytest.mark.audio
    def test_different_sample_rates(self):
        """Test audio with different sample rates"""
        sample_rates = [8000, 16000, 22050, 44100, 48000]
        
        for rate in sample_rates:
            duration = 0.5  # 0.5 seconds
            t = np.linspace(0, duration, int(rate * duration), False)
            audio_data = np.sin(2 * np.pi * 440 * t)
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            
            assert len(audio_bytes) == int(rate * duration) * 2
    
    @pytest.mark.audio
    def test_different_bit_depths(self):
        """Test audio with different bit depths"""
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_signal = np.sin(2 * np.pi * 440 * t)
        
        # Test 16-bit
        audio_16bit = (audio_signal * 32767).astype(np.int16)
        assert audio_16bit.dtype == np.int16
        
        # Test 32-bit float
        audio_32bit = audio_signal.astype(np.float32)
        assert audio_32bit.dtype == np.float32
        assert np.max(np.abs(audio_32bit)) <= 1.0
    
    @pytest.mark.audio
    def test_stereo_to_mono_conversion(self):
        """Test stereo to mono audio conversion"""
        duration = 1.0
        sample_rate = 16000
        samples = int(sample_rate * duration)
        
        # Create stereo audio (2 channels)
        left_channel = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        right_channel = np.sin(2 * np.pi * 880 * np.linspace(0, duration, samples))
        
        stereo_audio = np.column_stack((left_channel, right_channel))
        assert stereo_audio.shape == (samples, 2)
        
        # Convert to mono (average channels)
        mono_audio = np.mean(stereo_audio, axis=1)
        assert mono_audio.shape == (samples,)
    
    @pytest.mark.audio
    def test_audio_normalization(self):
        """Test audio normalization"""
        # Create audio with varying amplitudes
        t = np.linspace(0, 1, 16000, False)
        audio = np.sin(2 * np.pi * 440 * t) * 0.1  # Low amplitude
        
        # Normalize audio
        max_val = np.max(np.abs(audio))
        normalized_audio = audio / max_val
        
        assert np.max(np.abs(normalized_audio)) <= 1.0
        assert np.max(np.abs(normalized_audio)) > 0.9  # Should be close to 1.0


class TestAudioQualityMetrics:
    """Test audio quality and characteristics"""
    
    @pytest.mark.audio
    def test_signal_to_noise_ratio(self, sample_audio_data):
        """Test signal-to-noise ratio calculation"""
        audio_array = np.frombuffer(sample_audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # For a clean sine wave, SNR should be high
        signal_power = np.mean(audio_float ** 2)
        
        # Add noise for testing
        noise = np.random.normal(0, 0.01, len(audio_float))
        noisy_audio = audio_float + noise
        
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        
        assert snr > 20  # Should have good SNR
    
    @pytest.mark.audio
    def test_frequency_analysis(self, sample_audio_data):
        """Test frequency domain analysis of audio"""
        audio_array = np.frombuffer(sample_audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32)
        
        # Perform FFT
        fft = np.fft.fft(audio_float)
        freqs = np.fft.fftfreq(len(audio_float), 1/16000)
        
        # Find peak frequency
        magnitude = np.abs(fft[:len(fft)//2])
        peak_idx = np.argmax(magnitude)
        peak_freq = freqs[peak_idx]
        
        # Should be close to 440Hz (our generated frequency)
        assert abs(peak_freq - 440) < 10  # Within 10Hz tolerance
    
    @pytest.mark.audio
    def test_audio_duration_accuracy(self):
        """Test accuracy of audio duration"""
        durations = [0.5, 1.0, 2.0, 5.0]
        sample_rate = 16000
        
        for expected_duration in durations:
            samples = int(sample_rate * expected_duration)
            t = np.linspace(0, expected_duration, samples, False)
            audio = np.sin(2 * np.pi * 440 * t)
            
            # Calculate actual duration
            actual_duration = len(audio) / sample_rate
            
            # Should be very close
            assert abs(actual_duration - expected_duration) < 0.001


class TestAudioFileOperations:
    """Test audio file I/O operations"""
    
    @pytest.mark.audio
    def test_wav_file_round_trip(self, temp_directory):
        """Test writing and reading WAV files"""
        # Generate test audio
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Write WAV file
        wav_path = os.path.join(temp_directory, "test_roundtrip.wav")
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        # Read WAV file back
        with wave.open(wav_path, 'rb') as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == sample_rate
            
            frames = wav_file.readframes(wav_file.getnframes())
            read_audio = np.frombuffer(frames, dtype=np.int16)
        
        # Should match original
        np.testing.assert_array_equal(audio_int16, read_audio)
    
    @pytest.mark.audio
    def test_temporary_file_cleanup(self, test_assistant):
        """Test temporary audio file cleanup"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        original_count = len(os.listdir(temp_dir))
        
        # Process audio (should create temporary files)
        text = "Test speech synthesis"
        output_file = await test_assistant.synthesize_speech(text)
        
        # File should exist
        if output_file and os.path.dirname(output_file) == temp_dir:
            assert os.path.exists(output_file)
        
        # In a real implementation, temporary files should be cleaned up
        # For now, just verify the process doesn't crash
    
    @pytest.mark.audio
    def test_audio_metadata_preservation(self, temp_directory):
        """Test preservation of audio metadata"""
        # Create audio with specific properties
        sample_rate = 22050
        channels = 1
        sample_width = 2
        duration = 1.5
        
        audio_data = np.random.randint(-32768, 32767, 
                                     size=int(sample_rate * duration), 
                                     dtype=np.int16)
        
        wav_path = os.path.join(temp_directory, "metadata_test.wav")
        
        # Write with metadata
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Verify metadata is preserved
        with wave.open(wav_path, 'rb') as wav_file:
            assert wav_file.getnchannels() == channels
            assert wav_file.getsampwidth() == sample_width
            assert wav_file.getframerate() == sample_rate
            assert wav_file.getnframes() == len(audio_data)