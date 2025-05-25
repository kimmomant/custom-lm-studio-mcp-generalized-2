# Generalized LM Studio MCP Server

A dynamic Model Context Protocol (MCP) server that automatically adapts to the capabilities of the currently loaded model in LM Studio. This server intelligently detects whether the loaded model supports vision capabilities and only exposes the appropriate tools.

## Features

### 🔍 **Automatic Capability Detection**
- Dynamically detects if the current model supports vision/multimodal capabilities
- Automatically adapts the available tools based on model capabilities
- Tests vision support by sending a small test image to the model
- Refreshes capabilities each time tools are listed

### 🛠️ **Universal Tools** (Available for all models)
- **Health Check**: Verify LM Studio API connectivity
- **List Models**: Show all available models in LM Studio
- **Get Current Model**: Display the currently loaded model
- **Chat Completion**: Generate text completions with any loaded model

### 👁️ **Vision Tools** (Only available for vision-capable models)
- **Analyze Image**: Comprehensive image analysis with custom prompts
- **Chat with Image**: Interactive conversations about images with history
- **Compare Images**: Side-by-side comparison of two images
- **Extract Text**: OCR-like text extraction from images
- **Analyze Screenshot**: Specialized UI/UX analysis for screenshots
- **Batch Analyze Images**: Process multiple images in a directory

## How It Works

### Capability Detection Process

1. **Model Discovery**: Queries LM Studio's `/v1/models` endpoint to get current model info
2. **Vision Testing**: Sends a minimal test image (1x1 pixel) to determine vision support
3. **Tool Adaptation**: Dynamically builds the tool list based on detected capabilities
4. **Runtime Refresh**: Re-detects capabilities when tools are requested

### Supported Model Types

#### ✅ **Vision-Capable Models**
- Gemma-3-27B-IT (multimodal)
- LLaVA models
- Any model that accepts `image_url` content in chat completions

#### ✅ **Text-Only Models**
- Phi-4
- DeepSeek models
- Qwen models
- Any standard language model

## Installation & Setup

### Prerequisites
- Python 3.8+
- LM Studio running on `localhost:1234`
- Required Python packages (see `requirements.txt`)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Test the server
python test_generalized_server.py
```

### Configuration for Cursor

Use the provided configuration file:
```json
{
  "mcpServers": {
    "lm-studio-mcp": {
      "command": "python",
      "args": ["generalized_lm_studio_mcp_server.py"],
      "cwd": "/path/to/your/custom-lm-studio-mcp-generalized-2",
      "env": {
        "PYTHONPATH": "/path/to/your/custom-lm-studio-mcp-generalized-2"
      }
    }
  }
}
```

## Usage Examples

### Basic Text Completion
```python
# Available for ALL models
{
  "tool": "chat_completion",
  "arguments": {
    "prompt": "Explain quantum computing in simple terms",
    "temperature": 0.7,
    "max_tokens": 500
  }
}
```

### Image Analysis (Vision models only)
```python
# Only available if model supports vision
{
  "tool": "analyze_image",
  "arguments": {
    "image_path": "/path/to/image.jpg",
    "prompt": "Describe this image in detail",
    "temperature": 0.7
  }
}
```

### Screenshot Analysis (Vision models only)
```python
{
  "tool": "analyze_screenshot",
  "arguments": {
    "image_path": "/path/to/screenshot.png",
    "analysis_focus": "ui_elements"
  }
}
```

## Model Switching

The server automatically adapts when you switch models in LM Studio:

1. **Load a vision model** (e.g., Gemma-3-27B-IT) → All tools available
2. **Switch to text-only model** (e.g., Phi-4) → Only text tools available
3. **Switch back to vision model** → Vision tools become available again

## Architecture

### Key Components

#### `ModelCapabilities` Class
Tracks the current model's capabilities:
```python
class ModelCapabilities:
    supports_vision: bool = False
    supports_text: bool = True
    supports_chat: bool = True
    model_name: str = ""
    model_id: str = ""
    context_length: int = 0
    max_tokens: int = 1024
```

#### `GeneralizedLMStudioServer` Class
Main server class that:
- Detects model capabilities on startup and tool requests
- Dynamically builds tool lists based on capabilities
- Routes tool calls to appropriate handlers
- Provides fallback behavior for unsupported tools

### Vision Detection Logic

```python
async def _test_vision_capability(self):
    """Test vision support with a minimal image"""
    # Create 1x1 pixel test image
    test_image = Image.new('RGB', (1, 1), color='white')
    
    # Send vision request
    response = await self.lm_studio_client.post("/v1/chat/completions", {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }],
        "max_tokens": 10
    })
    
    # Vision supported if request succeeds
    self.capabilities.supports_vision = (response.status_code == 200)
```

## Error Handling

- **Model Not Loaded**: Gracefully handles when no model is loaded
- **Vision Test Failure**: Falls back to text-only mode if vision test fails
- **API Errors**: Provides informative error messages for debugging
- **Tool Unavailability**: Clear messages when tools aren't supported by current model

## Logging

The server provides detailed logging for:
- Model detection results
- Capability test outcomes
- Tool availability changes
- API request/response cycles

## Comparison with Original Servers

| Feature | Original Gemma Server | Original Image Server | Generalized Server |
|---------|----------------------|----------------------|-------------------|
| Model Support | Gemma-3-27B-IT only | Fixed model | Any LM Studio model |
| Vision Detection | Assumed | Assumed | Dynamic testing |
| Tool Adaptation | Static | Static | Dynamic |
| Model Switching | Manual restart needed | Manual restart needed | Automatic |
| Error Handling | Basic | Basic | Comprehensive |

## Benefits

1. **Universal Compatibility**: Works with any model loaded in LM Studio
2. **Automatic Adaptation**: No manual configuration needed when switching models
3. **Efficient Resource Usage**: Only loads tools that the model can actually use
4. **Future-Proof**: Automatically supports new vision models as they become available
5. **Graceful Degradation**: Falls back to text-only mode for non-vision models

## Troubleshooting

### Common Issues

1. **No tools available**: Check if LM Studio is running and a model is loaded
2. **Vision tools missing**: Current model may not support vision - try loading a multimodal model
3. **Connection errors**: Verify LM Studio is running on `localhost:1234`

### Debug Mode
Enable detailed logging by setting the log level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- Support for other model capabilities (function calling, code execution, etc.)
- Integration with other LLM providers beyond LM Studio
- Caching of capability detection results
- Support for model-specific optimizations
- Advanced vision model features (multiple images, video analysis)

## Contributing

Feel free to contribute improvements, bug fixes, or new features. The modular design makes it easy to add support for new capabilities or model types. 