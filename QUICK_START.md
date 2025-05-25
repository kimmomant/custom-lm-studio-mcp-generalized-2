# Quick Start Guide - Generalized LM Studio MCP Server

## 🚀 **Get Started in 3 Steps**

### **Step 1: Prerequisites**
- ✅ LM Studio installed and running on `localhost:1234`
- ✅ Python 3.8+ installed
- ✅ A model loaded in LM Studio (any model works!)

### **Step 2: Install & Test**
```bash
# Install dependencies
pip install -r requirements.txt

# Test the server
python test_generalized_server.py
```

### **Step 3: Configure Cursor**

**⚠️ IMPORTANT**: You must use the **absolute path** to the server file in your Cursor MCP configuration.

#### **Windows Users (Recommended)**
Use the provided batch file since Cursor may not find Python directly:

```json
{
  "mcpServers": {
    "lm-studio-mcp-generalized": {
      "command": "C:\\path\\to\\your\\custom-lm-studio-mcp-generalized-2\\start_generalized_mcp.bat"
    }
  }
}
```

**Replace the path** with your actual directory:
- Example: `"C:\\Users\\YourName\\PROJECTS\\custom-lm-studio-mcp-generalized-2\\start_generalized_mcp.bat"`

#### **macOS/Linux Users**
Use the cross-platform Python launcher:

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

**Replace the path** with your actual directory:
- **macOS**: `"/Users/YourName/Projects/custom-lm-studio-mcp-generalized-2/launch_server.py"`
- **Linux**: `"/home/YourName/Projects/custom-lm-studio-mcp-generalized-2/launch_server.py"`

#### **Alternative: Direct Python (If Python is in PATH)**
If you're sure Python is available in Cursor's environment:

```json
{
  "mcpServers": {
    "lm-studio-mcp-generalized": {
      "command": "python",
      "args": ["/absolute/path/to/generalized_lm_studio_mcp_server.py"],
      "env": {
        "LM_STUDIO_BASE_URL": "http://localhost:1234"
      }
    }
  }
}
```

**Note**: This may not work reliably in Cursor due to environment isolation.

## 🎯 **What You Get**

### **With ANY Model:**
- ✅ Health checks
- ✅ Model management
- ✅ Text completions

### **With Vision Models (Gemma-3-27B-IT, LLaVA, etc.):**
- ✅ Image analysis
- ✅ Screenshot analysis
- ✅ Text extraction from images
- ✅ Image comparisons
- ✅ Batch image processing

## 🔄 **Model Switching**

1. **Load any model** in LM Studio
2. **Click the refresh button** next to the MCP server in Cursor settings
3. **Tools update** to match the new model's capabilities
4. **No server restart needed** - the MCP server keeps running

**Important**: Cursor caches the tool list, so you **must manually refresh** the MCP server in Cursor settings after switching models in LM Studio.

## 🧪 **Quick Test**

Run the demo to see capability detection in action:
```bash
python demo_capability_adaptation.py
```

## 📚 **Need More Info?**

- **Full Setup Guide**: `SETUP_INSTRUCTIONS.md`
- **Technical Documentation**: `README_GENERALIZED.md`

## 🆘 **Troubleshooting**

| Problem | Solution |
|---------|----------|
| "Failed to create client" | Check that you're using the **absolute path** to the server file |
| No tools available | Check LM Studio is running with a model loaded |
| Vision tools missing | Load a vision-capable model (Gemma-3-27B-IT, LLaVA) and **refresh in Cursor** |
| Tools don't update after model switch | **Click refresh button** next to MCP server in Cursor settings |
| Connection errors | Verify LM Studio is on `localhost:1234` |
| Python not found | Use full path to Python: `"C:\\Python\\python.exe"` (Windows) or `"/usr/bin/python3"` (Linux/macOS) |

## 🎉 **That's It!**

You now have a universal MCP server that works with any LM Studio model and automatically adapts to their capabilities! 