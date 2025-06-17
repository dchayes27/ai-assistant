#!/usr/bin/env python3
"""
Test external MCP tools integration
"""

import os
import sys
import asyncio
from core.tool_manager import ToolManager

async def test_tools():
    """Test the external tools"""
    
    print("🔧 Testing External MCP Tools...")
    
    # Initialize tool manager
    tool_manager = ToolManager()
    await tool_manager.initialize()
    
    # List available tools
    print(f"\n📋 Available tools: {len(tool_manager.available_tools)}")
    for tool in tool_manager.available_tools:
        print(f"  - {tool['name']}: {tool['description']}")
    
    # Test tool detection
    test_messages = [
        "What's the weather in New York?",
        "Search the web for latest AI news",
        "Calculate 25 * 47 + 123",
        "Find news about climate change",
        "What's 15% of 240?",
        "How's the weather in London?"
    ]
    
    print(f"\n🔍 Testing tool detection...")
    for message in test_messages:
        detected = tool_manager.detect_tool_needs(message)
        print(f"  '{message}' -> {[t['tool'] for t in detected]}")
    
    # Test specific tools (if API keys available)
    print(f"\n🧪 Testing tool execution...")
    
    # Test calculator (no API key needed)
    print("Testing calculator...")
    calc_result = await tool_manager.execute_tool("calculate", expression="2 + 2 * 3")
    print(f"  Calculate '2 + 2 * 3': {calc_result}")
    
    # Test web search (needs Brave API key)
    if os.environ.get('BRAVE_API_KEY'):
        print("Testing web search...")
        search_result = await tool_manager.execute_tool("web_search", 
                                                       query="latest AI developments", 
                                                       count=3)
        print(f"  Web search result: {search_result.get('success', False)}")
    else:
        print("  Skipping web search (no BRAVE_API_KEY)")
    
    # Test weather (needs OpenWeather API key)
    if os.environ.get('OPENWEATHER_API_KEY'):
        print("Testing weather...")
        weather_result = await tool_manager.execute_tool("weather", 
                                                        location="San Francisco")
        print(f"  Weather result: {weather_result.get('success', False)}")
    else:
        print("  Skipping weather (no OPENWEATHER_API_KEY)")
    
    # Test message processing with tools
    print(f"\n💬 Testing message processing...")
    test_message = "What's 15 + 25 and also what's the weather like in Paris?"
    enhanced_message, tool_results = await tool_manager.process_message_with_tools(test_message)
    
    print(f"  Original: {test_message}")
    print(f"  Enhanced: {len(enhanced_message)} chars")
    print(f"  Tools used: {len(tool_results)}")
    
    for result in tool_results:
        tool_name = result.get('tool', 'unknown')
        success = 'success' in result.get('result', {})
        print(f"    - {tool_name}: {'✅' if success else '❌'}")
    
    await tool_manager.close()
    print(f"\n✨ Tool testing complete!")

if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Note: python-dotenv not installed, reading from environment")
    
    asyncio.run(test_tools())