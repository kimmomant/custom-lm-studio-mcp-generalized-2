# Generalized LM Studio MCP Server

A universal Model Context Protocol (MCP) server that automatically adapts to any LM Studio model's capabilities. Works with text-only models, vision models, and everything in between!

## 🎯 Why This Tool Exists

**The Problem**: Cursor AI IDE's built-in models (DeepSeek, ChatGPT, Claude, Gemini) have image analysis capabilities, but they can't use them autonomously. You can only include images manually using the `@` symbol. When you ask the AI to "analyze the screenshot I just took" or "read the chart in ./output/", it can't automatically access those files - even though the underlying models are perfectly capable of image analysis.

**The Solution**: This MCP server enables autonomous image analysis by connecting Cursor to local LM Studio models. Now when you ask your AI to analyze an image file, it can automatically read and analyze it without requiring manual `@` inclusion - something the built-in models can't do despite having the same underlying capabilities.

**Real Example**: Instead of manually dragging every auto-generated chart into Cursor with `@`, you can simply ask: *"Analyze the chart at ./output/sales_chart.png and explain the trends"* - and your local AI will automatically access and analyze the image file autonomously.

## 🚀 Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Start LM Studio** with any model loaded
3. **Test the server**: `python test_generalized_server.py`
4. **Configure your MCP client** using `cursor_mcp_config.json` (⚠️ **update the path!**)

## ✨ Features

### 🌐 Universal Compatibility
- **Any Model**: Works with text-only models (Phi-4, DeepSeek, etc.)
- **Vision Models**: Automatically detects and enables vision tools (Gemma-3-27B-IT, LLaVA, etc.)
- **Auto-Adaptation**: Tools appear/disappear based on model capabilities
- **No Restart**: Switch models in LM Studio - just refresh in Cursor settings

### 🛠️ Available Tools

**Always Available:**
- Health checks and model management
- Text completions and chat

**Vision Models Only:**
- Image analysis and description
- Screenshot analysis (UI elements, layout, accessibility)
- Text extraction from images (OCR-like)
- Image comparisons
- Batch image processing
- Conversational image chat

## 📚 Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get running in 3 steps
- **[Setup Instructions](SETUP_INSTRUCTIONS.md)** - Detailed setup and troubleshooting
- **[Technical Documentation](README_GENERALIZED.md)** - Full technical details

## 🧪 Try the Demo

```bash
python demo_capability_adaptation.py
```

See how the server automatically detects your model's capabilities!

## 🔧 Configuration

**⚠️ IMPORTANT**: Cursor has its own environment, so direct Python calls may not work.

**Windows (Recommended)**: Use the batch file:
```json
{
  "mcpServers": {
    "lm-studio-mcp-generalized": {
      "command": "C:\\path\\to\\your\\custom-lm-studio-mcp-generalized-2\\start_generalized_mcp.bat"
    }
  }
}
```

**macOS/Linux**: Use the Python launcher:
```json
{
  "mcpServers": {
    "lm-studio-mcp-generalized": {
      "command": "python3",
      "args": ["/path/to/your/custom-lm-studio-mcp-generalized-2/launch_server.py"]
    }
  }
}
```

**Example paths:**
- **Windows**: `"C:\\Users\\YourName\\PROJECTS\\custom-lm-studio-mcp-generalized-2\\start_generalized_mcp.bat"`
- **macOS**: `"/Users/YourName/Projects/custom-lm-studio-mcp-generalized-2/launch_server.py"`
- **Linux**: `"/home/YourName/Projects/custom-lm-studio-mcp-generalized-2/launch_server.py"`

## 🎯 Why This Server?

- **Enables Autonomous Image Analysis**: What built-in models can't do - automatically access and analyze image files
- **No More Manual `@` Inclusion**: AI can read images from file paths without manual intervention
- **Same Models, Better Integration**: Use LM Studio as a workaround for Cursor's autonomous limitations
- **Privacy & Cost Control**: Keep your data local and avoid API costs
- **One Server, All Models**: No need for different servers for different models
- **Zero Configuration**: Automatically detects what your model can do
- **Seamless Integration**: Works directly with Cursor's AI without workflow interruption
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Future-Proof**: Works with new models as they're released

## 📋 Requirements

- Python 3.8+
- LM Studio running on localhost:1234
- Any model loaded in LM Studio

## 🆘 Need Help?

Check the [Quick Start Guide](QUICK_START.md) for common issues like "Failed to create client" or [Setup Instructions](SETUP_INSTRUCTIONS.md) for detailed troubleshooting.

---

**No more model-specific servers!** 🎉 One server that adapts to everything. 