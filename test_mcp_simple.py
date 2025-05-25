#!/usr/bin/env python3
"""
Simple test to verify MCP server can initialize properly
"""
import asyncio
import sys
from generalized_lm_studio_mcp_server import GeneralizedLMStudioServer

async def test_server_init():
    """Test that the server can initialize and detect capabilities"""
    try:
        print("Initializing server...", file=sys.stderr)
        server = GeneralizedLMStudioServer()
        
        print("Detecting model capabilities...", file=sys.stderr)
        await server.detect_model_capabilities()
        
        print(f"✓ Model: {server.capabilities.model_name}", file=sys.stderr)
        print(f"✓ Vision: {server.capabilities.supports_vision}", file=sys.stderr)
        print(f"✓ Text: {server.capabilities.supports_text}", file=sys.stderr)
        print(f"✓ Chat: {server.capabilities.supports_chat}", file=sys.stderr)
        
        print("Server initialization test passed!", file=sys.stderr)
        return True
        
    except Exception as e:
        print(f"✗ Server initialization failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False

if __name__ == "__main__":
    result = asyncio.run(test_server_init())
    sys.exit(0 if result else 1) 