#!/usr/bin/env python3
"""
Test script for OpenAI TTS integration
"""

import os
import sys
import asyncio
from core.smart_assistant import SmartAssistant, AssistantConfig

async def test_openai_tts():
    """Test OpenAI TTS functionality"""
    
    # Get OpenAI API key from environment or prompt user
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("No API key provided. Testing local TTS fallback...")
        config = AssistantConfig(tts_model="tts_models/en/ljspeech/tacotron2-DDC")
    else:
        print("Testing OpenAI TTS...")
        config = AssistantConfig(
            tts_model="openai:alloy",
            openai_api_key=api_key
        )
    
    async with SmartAssistant(config) as assistant:
        # Test text
        test_text = "Hello! This is a test of the OpenAI text-to-speech integration. The voice quality should be much better than the local models."
        
        print(f"Synthesizing: {test_text}")
        
        try:
            audio_file = await assistant.synthesize_speech(test_text)
            if audio_file and os.path.exists(audio_file):
                print(f"✅ Success! Audio file created: {audio_file}")
                print(f"File size: {os.path.getsize(audio_file)} bytes")
                
                # Optionally play the audio on macOS
                if sys.platform == "darwin":
                    play = input("Play the audio file? (y/n): ").strip().lower()
                    if play == 'y':
                        os.system(f'afplay "{audio_file}"')
                
            else:
                print("❌ Failed to create audio file")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_openai_tts())