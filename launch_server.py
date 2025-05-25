#!/usr/bin/env python3
"""
Cross-platform launcher for the Generalized LM Studio MCP Server.
This script sets up the environment and launches the server.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Set environment variables
    os.environ['LM_STUDIO_BASE_URL'] = 'http://localhost:1234'
    os.environ['PYTHONPATH'] = str(script_dir)
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Path to the main server script
    server_script = script_dir / 'generalized_lm_studio_mcp_server.py'
    
    if not server_script.exists():
        print(f"Error: Server script not found at {server_script}")
        sys.exit(1)
    
    # Launch the server
    try:
        print(f"Starting Generalized LM Studio MCP Server...")
        print(f"Script directory: {script_dir}")
        print(f"LM Studio URL: {os.environ['LM_STUDIO_BASE_URL']}")
        print("-" * 50)
        
        subprocess.run([sys.executable, str(server_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        sys.exit(0)

if __name__ == '__main__':
    main() 