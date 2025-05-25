#!/usr/bin/env python3
"""
Test script for the generalized LM Studio MCP server
"""

import asyncio
from generalized_lm_studio_mcp_server import GeneralizedLMStudioServer

async def test_capabilities():
    """Test the capability detection"""
    print("Testing Generalized LM Studio MCP Server...")
    print("=" * 50)
    
    server = GeneralizedLMStudioServer()
    
    # Test capability detection
    capabilities = await server.detect_model_capabilities()
    
    print(f"Model Name: {capabilities.model_name}")
    print(f"Model ID: {capabilities.model_id}")
    print(f"Supports Vision: {capabilities.supports_vision}")
    print(f"Supports Text: {capabilities.supports_text}")
    print(f"Supports Chat: {capabilities.supports_chat}")
    print(f"Context Length: {capabilities.context_length}")
    print(f"Max Tokens: {capabilities.max_tokens}")
    
    print("\n" + "=" * 50)
    print("Testing basic functionality...")
    
    # Test health check
    try:
        health_result = await server.health_check()
        print(f"Health Check: {health_result[0].text}")
    except Exception as e:
        print(f"Health Check Error: {e}")
    
    # Test list models
    try:
        models_result = await server.list_models()
        print(f"Available Models:\n{models_result[0].text}")
    except Exception as e:
        print(f"List Models Error: {e}")
    
    # Test current model
    try:
        current_model_result = await server.get_current_model()
        print(f"Current Model: {current_model_result[0].text}")
    except Exception as e:
        print(f"Get Current Model Error: {e}")
    
    # Test chat completion
    try:
        chat_result = await server.chat_completion({
            "prompt": "Hello! Can you tell me what you are?",
            "max_tokens": 50
        })
        print(f"Chat Test: {chat_result[0].text[:100]}...")
    except Exception as e:
        print(f"Chat Test Error: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    asyncio.run(test_capabilities()) 