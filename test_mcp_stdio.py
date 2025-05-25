#!/usr/bin/env python3

import asyncio
import json
import sys
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.shared.types import Tool, TextContent
from mcp.server import stdio_server

class MinimalMCPServer:
    def __init__(self):
        self.server = Server("minimal-test-server")
        self.setup_handlers()

    def setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools():
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
        async def handle_call_tool(name: str, arguments: dict):
            if name == "test_tool":
                message = arguments.get("message", "No message")
                return [TextContent(type="text", text=f"Test response: {message}")]
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def run(self):
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
    print("Starting minimal MCP server...", file=sys.stderr)
    server = MinimalMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main()) 