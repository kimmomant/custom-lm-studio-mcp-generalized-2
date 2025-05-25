# Generalized LM Studio MCP Server Setup Instructions

This is a generalized MCP (Model Context Protocol) server that works with any LM Studio model and automatically detects the model's capabilities (text-only or vision-enabled).

## Prerequisites

1. **Python 3.8+** installed and available in your system PATH
2. **LM Studio** running with a model loaded
3. **Required Python packages** (install with `pip install -r requirements.txt`)

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start LM Studio
- Launch LM Studio
- Load any model (text-only or vision-capable)
- Ensure the server is running on `http://localhost:1234` (default)

### 3. Test the Server
Run the test script to verify everything works:
```bash
python test_generalized_server.py
```

### 4. Configure Cursor (or other MCP client)

Copy the contents of `cursor_mcp_config_clean.json` to your Cursor MCP configuration:

```json
{
  "mcpServers": {
    "lm-studio-mcp-generalized": {
      "command": "python",
      "args": [
        "generalized_lm_studio_mcp_server.py"
      ],
      "env": {
        "LM_STUDIO_BASE_URL": "http://localhost:1234"
      }
    }
  }
}
```

**Important**: Make sure to run Cursor from the directory containing the server files, or provide the full path to `generalized_lm_studio_mcp_server.py`.

## Alternative Startup Methods

### Using Batch File (Windows)
```bash
start_generalized_mcp.bat
```

### Direct Python Execution
```bash
python generalized_lm_studio_mcp_server.py
```

## Features

The server automatically detects your model's capabilities and provides appropriate tools:

### Always Available:
- `health_check` - Check if LM Studio is accessible
- `list_models` - List available models
- `get_current_model` - Get currently loaded model info
- `chat_completion` - Generate text completions

### Vision-Enabled Models Only:
- `analyze_image` - Analyze images with custom prompts
- `chat_with_image` - Have conversations about images
- `compare_images` - Compare two images
- `extract_text` - Extract text from images (OCR-like)
- `analyze_screenshot` - Specialized screenshot analysis
- `batch_analyze_images` - Analyze multiple images

## Troubleshooting

### Common Issues:

1. **"Connection refused" errors**
   - Ensure LM Studio is running
   - Check that the server is on `http://localhost:1234`
   - Verify a model is loaded in LM Studio

2. **"Python not found" errors**
   - Ensure Python is installed and in your system PATH
   - Try using `python3` instead of `python` in the configuration

3. **Import errors**
   - Install dependencies: `pip install -r requirements.txt`
   - Ensure you're in the correct directory

4. **Vision tools not appearing**
   - The server automatically detects vision capabilities
   - If your model supports vision but tools don't appear, check LM Studio logs
   - Try restarting the MCP server

### Custom LM Studio URL
If LM Studio is running on a different port or host, set the environment variable:
```bash
set LM_STUDIO_BASE_URL=http://localhost:8080
```

## File Structure

- `generalized_lm_studio_mcp_server.py` - Main server file
- `start_generalized_mcp.bat` - Windows batch file for easy startup
- `cursor_mcp_config_clean.json` - Clean configuration template
- `requirements.txt` - Python dependencies
- `test_generalized_server.py` - Test script

## No More Hardcoded Paths!

This version has been cleaned up to remove all hardcoded paths. The server now works from any directory and uses relative paths and environment variables for maximum portability. 