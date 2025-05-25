#!/usr/bin/env python3

import httpx
import asyncio
import sys

async def test_lm_studio():
    """Test if LM Studio is accessible"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:1234/v1/models")
            print(f"LM Studio status: {response.status_code}")
            if response.status_code == 200:
                models = response.json()
                print(f"Available models: {len(models.get('data', []))}")
                return True
            else:
                print("LM Studio not responding correctly")
                return False
    except Exception as e:
        print(f"Error connecting to LM Studio: {e}")
        return False

async def test_mcp_imports():
    """Test if MCP imports work"""
    try:
        import mcp
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        print("MCP imports successful")
        return True
    except Exception as e:
        print(f"MCP import error: {e}")
        return False

async def main():
    print("Testing MCP server prerequisites...")
    
    # Test imports
    imports_ok = await test_mcp_imports()
    
    # Test LM Studio connection
    lm_studio_ok = await test_lm_studio()
    
    if imports_ok and lm_studio_ok:
        print("✓ All prerequisites are met")
        return 0
    else:
        print("✗ Some prerequisites are missing")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 