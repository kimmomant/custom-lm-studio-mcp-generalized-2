#!/usr/bin/env python3
"""
Test script to verify MCP server communication
"""
import asyncio
import json
import sys
from generalized_lm_studio_mcp_server import GeneralizedLMStudioServer

async def test_mcp_server():
    """Test basic MCP server functionality"""
    try:
        server = GeneralizedLMStudioServer()
        
        # Test model detection
        print("Testing model detection...", file=sys.stderr)
        await server.detect_model_capabilities()
        print(f"Model detected: {server.capabilities.model_name}", file=sys.stderr)
        print(f"Vision support: {server.capabilities.supports_vision}", file=sys.stderr)
        
        # Test tool listing
        print("Testing tool listing...", file=sys.stderr)
        tools = await server.server._list_tools_handler()
        print(f"Found {len(tools)} tools", file=sys.stderr)
        
        print("MCP server test completed successfully!", file=sys.stderr)
        return True
        
    except Exception as e:
        print(f"Error testing MCP server: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    result = asyncio.run(test_mcp_server())
    sys.exit(0 if result else 1) 