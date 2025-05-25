#!/usr/bin/env python3
"""
Minimal MCP server for testing protocol communication
"""
import asyncio
import logging
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalMCPServer:
    def __init__(self):
        self.server = Server("minimal-test-server")
        self.setup_handlers()

    def setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="test_tool",
                    description="A simple test tool",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Test message"
                            }
                        },
                        "required": ["message"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name == "test_tool":
                message = arguments.get("message", "No message")
                return [TextContent(type="text", text=f"Test response: {message}")]
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def run(self):
        logger.info("Starting minimal MCP server...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="minimal-test-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    ),
                ),
            )

async def main():
    server = MinimalMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main()) 