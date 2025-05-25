#!/usr/bin/env python3
"""
Generalized MCP Server for LM Studio Integration
Dynamically adapts to the currently loaded model's capabilities
"""

import asyncio
import base64
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import httpx
from PIL import Image
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.server.lowlevel import NotificationOptions
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LM Studio configuration
LM_STUDIO_BASE_URL = "http://localhost:1234"

class ModelCapabilities:
    """Class to track model capabilities"""
    def __init__(self):
        self.supports_vision = False
        self.supports_text = False
        self.supports_chat = False
        self.model_name = ""
        self.model_id = ""
        self.context_length = 0
        self.max_tokens = 1024

class GeneralizedLMStudioServer:
    def __init__(self):
        self.server = Server("lm-studio-mcp-server")
        self.lm_studio_client = httpx.AsyncClient(
            base_url=LM_STUDIO_BASE_URL,
            timeout=120.0
        )
        self.capabilities = ModelCapabilities()
        self.setup_handlers()

    async def detect_model_capabilities(self) -> ModelCapabilities:
        """Detect the capabilities of the currently loaded model"""
        try:
            # Get current model info
            response = await self.lm_studio_client.get("/v1/models")
            response.raise_for_status()
            models_data = response.json()
            
            if not models_data.get("data"):
                logger.warning("No models found in LM Studio")
                return self.capabilities
            
            current_model = models_data["data"][0]  # First model is usually the loaded one
            model_id = current_model.get("id", "")
            
            self.capabilities.model_name = model_id
            self.capabilities.model_id = model_id
            self.capabilities.supports_text = True  # All models support text
            self.capabilities.supports_chat = True  # All models support chat
            
            # Test vision capabilities by attempting a simple vision request
            await self._test_vision_capability()
            
            # Get context length if available
            if "context_length" in current_model:
                self.capabilities.context_length = current_model["context_length"]
            
            logger.info(f"Detected model: {model_id}")
            logger.info(f"Vision support: {self.capabilities.supports_vision}")
            logger.info(f"Text support: {self.capabilities.supports_text}")
            logger.info(f"Chat support: {self.capabilities.supports_chat}")
            
            return self.capabilities
            
        except Exception as e:
            logger.error(f"Error detecting model capabilities: {str(e)}")
            # Default to text-only capabilities
            self.capabilities.supports_text = True
            self.capabilities.supports_chat = True
            return self.capabilities

    async def _test_vision_capability(self):
        """Test if the current model supports vision by making a test request"""
        try:
            # Create a small test image (1x1 pixel)
            test_image = Image.new('RGB', (1, 1), color='white')
            buffer = BytesIO()
            test_image.save(buffer, format='JPEG')
            test_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Try to send a vision request
            test_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What do you see?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{test_image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            response = await self.lm_studio_client.post("/v1/chat/completions", json=test_payload)
            
            if response.status_code == 200:
                self.capabilities.supports_vision = True
                logger.info("Vision capability detected successfully")
            else:
                logger.info(f"Vision test failed with status {response.status_code}")
                
        except Exception as e:
            logger.info(f"Vision capability test failed: {str(e)}")
            self.capabilities.supports_vision = False

    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools based on model capabilities"""
            # Refresh capabilities each time tools are listed
            await self.detect_model_capabilities()
            
            tools = []
            
            # Always available tools
            tools.extend([
                Tool(
                    name="health_check",
                    description="Check if LM Studio API is accessible",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="list_models",
                    description="List all available models in LM Studio",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_current_model",
                    description="Get the currently loaded model in LM Studio",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                )
            ])
            
            # Text/Chat capabilities (available for all models)
            if self.capabilities.supports_text or self.capabilities.supports_chat:
                tools.append(
                    Tool(
                        name="chat_completion",
                        description="Generate a completion from the current LM Studio model",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The user's prompt to send to the model"
                                },
                                "system_prompt": {
                                    "type": "string",
                                    "default": "",
                                    "description": "Optional system instructions for the model"
                                },
                                "temperature": {
                                    "type": "number",
                                    "default": 0.7,
                                    "description": "Controls randomness (0.0 to 1.0)"
                                },
                                "max_tokens": {
                                    "type": "integer",
                                    "default": 1024,
                                    "description": "Maximum number of tokens to generate"
                                }
                            },
                            "required": ["prompt"]
                        }
                    )
                )
            
            # Vision capabilities (only if model supports vision)
            if self.capabilities.supports_vision:
                tools.extend([
                    Tool(
                        name="analyze_image",
                        description="Analyze an image using the current model's vision capabilities",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image_path": {
                                    "type": "string",
                                    "description": "Path to the image file"
                                },
                                "image_data": {
                                    "type": "string",
                                    "description": "Base64 encoded image data (alternative to image_path)"
                                },
                                "prompt": {
                                    "type": "string",
                                    "default": "Describe this image in detail.",
                                    "description": "Custom prompt for image analysis"
                                },
                                "system_prompt": {
                                    "type": "string",
                                    "description": "Custom system prompt to guide the analysis"
                                },
                                "temperature": {
                                    "type": "number",
                                    "default": 0.7,
                                    "description": "Temperature for response generation (0.0-1.0)"
                                },
                                "max_tokens": {
                                    "type": "integer",
                                    "default": 1024,
                                    "description": "Maximum tokens in response"
                                }
                            },
                            "required": []
                        }
                    ),
                    Tool(
                        name="chat_with_image",
                        description="Have a conversation about an image with the current model",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image_path": {
                                    "type": "string",
                                    "description": "Path to the image file"
                                },
                                "image_data": {
                                    "type": "string",
                                    "description": "Base64 encoded image data"
                                },
                                "message": {
                                    "type": "string",
                                    "description": "Your question or message about the image"
                                },
                                "conversation_history": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "role": {"type": "string"},
                                            "content": {"type": "string"}
                                        }
                                    },
                                    "description": "Previous conversation history"
                                },
                                "temperature": {
                                    "type": "number",
                                    "default": 0.7
                                }
                            },
                            "required": ["message"]
                        }
                    ),
                    Tool(
                        name="compare_images",
                        description="Compare two images using the current model's vision capabilities",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image1_path": {
                                    "type": "string",
                                    "description": "Path to the first image"
                                },
                                "image1_data": {
                                    "type": "string",
                                    "description": "Base64 data for first image"
                                },
                                "image2_path": {
                                    "type": "string",
                                    "description": "Path to the second image"
                                },
                                "image2_data": {
                                    "type": "string",
                                    "description": "Base64 data for second image"
                                },
                                "comparison_prompt": {
                                    "type": "string",
                                    "default": "Compare these two images. What are the similarities and differences?",
                                    "description": "Specific comparison prompt"
                                }
                            },
                            "required": []
                        }
                    ),
                    Tool(
                        name="extract_text",
                        description="Extract and analyze text from an image (OCR-like functionality)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image_path": {
                                    "type": "string",
                                    "description": "Path to the image file"
                                },
                                "image_data": {
                                    "type": "string",
                                    "description": "Base64 encoded image data"
                                },
                                "format_output": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Whether to format the extracted text nicely"
                                }
                            },
                            "required": []
                        }
                    ),
                    Tool(
                        name="analyze_screenshot",
                        description="Specialized analysis for screenshots (UI elements, layout, etc.)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image_path": {
                                    "type": "string",
                                    "description": "Path to the screenshot"
                                },
                                "image_data": {
                                    "type": "string",
                                    "description": "Base64 encoded screenshot data"
                                },
                                "analysis_focus": {
                                    "type": "string",
                                    "enum": ["ui_elements", "layout", "content", "accessibility", "general"],
                                    "default": "general",
                                    "description": "What aspect to focus on"
                                }
                            },
                            "required": []
                        }
                    ),
                    Tool(
                        name="batch_analyze_images",
                        description="Analyze multiple images and provide a summary",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "directory_path": {
                                    "type": "string",
                                    "description": "Path to directory containing images"
                                },
                                "analysis_prompt": {
                                    "type": "string",
                                    "default": "Analyze this image and describe what you see.",
                                    "description": "Prompt to use for each image"
                                },
                                "max_images": {
                                    "type": "integer",
                                    "default": 10,
                                    "description": "Maximum number of images to process"
                                },
                                "summary_prompt": {
                                    "type": "string",
                                    "default": "Provide a summary of all the analyzed images, noting patterns and key findings.",
                                    "description": "Prompt for final summary"
                                }
                            },
                            "required": ["directory_path"]
                        }
                    )
                ])
            
            return tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                # Basic tools
                if name == "health_check":
                    return await self.health_check()
                elif name == "list_models":
                    return await self.list_models()
                elif name == "get_current_model":
                    return await self.get_current_model()
                elif name == "chat_completion":
                    return await self.chat_completion(arguments)
                
                # Vision tools (only if supported)
                elif name == "analyze_image" and self.capabilities.supports_vision:
                    return await self.analyze_image(arguments)
                elif name == "chat_with_image" and self.capabilities.supports_vision:
                    return await self.chat_with_image(arguments)
                elif name == "compare_images" and self.capabilities.supports_vision:
                    return await self.compare_images(arguments)
                elif name == "extract_text" and self.capabilities.supports_vision:
                    return await self.extract_text(arguments)
                elif name == "analyze_screenshot" and self.capabilities.supports_vision:
                    return await self.analyze_screenshot(arguments)
                elif name == "batch_analyze_images" and self.capabilities.supports_vision:
                    return await self.batch_analyze_images(arguments)
                else:
                    return [TextContent(type="text", text=f"Tool '{name}' is not available for the current model or is unknown")]
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    # Basic LM Studio tools
    async def health_check(self) -> List[TextContent]:
        """Check if LM Studio API is accessible"""
        try:
            response = await self.lm_studio_client.get("/v1/models")
            if response.status_code == 200:
                return [TextContent(type="text", text="LM Studio API is running and accessible.")]
            else:
                return [TextContent(type="text", text=f"LM Studio API returned status code: {response.status_code}")]
        except Exception as e:
            return [TextContent(type="text", text=f"LM Studio API is not accessible: {str(e)}")]

    async def list_models(self) -> List[TextContent]:
        """List all available models in LM Studio"""
        try:
            response = await self.lm_studio_client.get("/v1/models")
            response.raise_for_status()
            models_data = response.json()
            
            if not models_data.get("data"):
                return [TextContent(type="text", text="No models found in LM Studio")]
            
            model_list = "Available models in LM Studio:\n\n"
            for model in models_data["data"]:
                model_list += f"- {model.get('id', 'Unknown')}\n"
            
            return [TextContent(type="text", text=model_list)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error listing models: {str(e)}")]

    async def get_current_model(self) -> List[TextContent]:
        """Get the currently loaded model in LM Studio"""
        try:
            response = await self.lm_studio_client.get("/v1/models")
            response.raise_for_status()
            models_data = response.json()
            
            if not models_data.get("data"):
                return [TextContent(type="text", text="No models currently loaded")]
            
            current_model = models_data["data"][0]
            model_name = current_model.get("id", "Unknown")
            
            return [TextContent(type="text", text=f"Currently loaded model: {model_name}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting current model: {str(e)}")]

    async def chat_completion(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Generate a completion from the current LM Studio model"""
        try:
            prompt = arguments["prompt"]
            system_prompt = arguments.get("system_prompt", "")
            temperature = arguments.get("temperature", 0.7)
            max_tokens = arguments.get("max_tokens", 1024)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = await self.lm_studio_client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return [TextContent(type="text", text=result["choices"][0]["message"]["content"])]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error generating completion: {str(e)}")]

    # Image utility methods
    def load_image_as_base64(self, image_path: Optional[str] = None, image_data: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Load image from path or base64 data and return base64 string and info"""
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Get image info
            with Image.open(image_path) as img:
                image_info = {
                    "filename": os.path.basename(image_path),
                    "size": img.size,
                    "mode": img.mode,
                    "format": img.format,
                    "file_size": len(image_bytes)
                }
            
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
        elif image_data:
            # Assume image_data is already base64 encoded
            image_b64 = image_data
            
            # Try to get basic info from the base64 data
            try:
                image_bytes = base64.b64decode(image_data)
                with Image.open(BytesIO(image_bytes)) as img:
                    image_info = {
                        "size": img.size,
                        "mode": img.mode,
                        "format": img.format,
                        "file_size": len(image_bytes)
                    }
            except Exception:
                image_info = {"source": "base64_data"}
        else:
            raise ValueError("Either image_path or image_data must be provided")
        
        return image_b64, image_info

    async def query_model_with_image(
        self, 
        image_b64: str, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Send image and prompt to the current model via LM Studio"""
        try:
            # Build messages array
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add the current message with image
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
            messages.append(user_message)
            
            payload = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            logger.info(f"Sending request to LM Studio with {len(messages)} messages")
            response = await self.lm_studio_client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error querying model with image: {str(e)}")
            return f"Error querying model: {str(e)}"

    # Vision tools (only available if model supports vision)
    async def analyze_image(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Analyze an image using the current model's vision capabilities"""
        try:
            image_path = arguments.get("image_path")
            image_data = arguments.get("image_data")
            prompt = arguments.get("prompt", "Describe this image in detail.")
            system_prompt = arguments.get("system_prompt")
            temperature = arguments.get("temperature", 0.7)
            max_tokens = arguments.get("max_tokens", 1024)
            
            # Load image
            image_b64, image_info = self.load_image_as_base64(image_path, image_data)
            
            # Query model
            response = await self.query_model_with_image(
                image_b64, prompt, system_prompt, temperature, max_tokens
            )
            
            # Format result
            result = {
                "model": self.capabilities.model_name,
                "image_info": image_info,
                "prompt": prompt,
                "analysis": response
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error analyzing image: {str(e)}")]

    async def chat_with_image(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Chat about an image with conversation history"""
        try:
            image_path = arguments.get("image_path")
            image_data = arguments.get("image_data")
            message = arguments["message"]
            conversation_history = arguments.get("conversation_history", [])
            temperature = arguments.get("temperature", 0.7)
            
            # Load image
            image_b64, image_info = self.load_image_as_base64(image_path, image_data)
            
            # Query model with conversation context
            response = await self.query_model_with_image(
                image_b64, message, None, temperature, 1024, conversation_history
            )
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error in chat: {str(e)}")]

    async def compare_images(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Compare two images"""
        try:
            # Load first image
            image1_path = arguments.get("image1_path")
            image1_data = arguments.get("image1_data")
            image1_b64, image1_info = self.load_image_as_base64(image1_path, image1_data)
            
            # Load second image
            image2_path = arguments.get("image2_path")
            image2_data = arguments.get("image2_data")
            image2_b64, image2_info = self.load_image_as_base64(image2_path, image2_data)
            
            comparison_prompt = arguments.get(
                "comparison_prompt", 
                "Compare these two images. What are the similarities and differences?"
            )
            
            # Analyze each image separately first
            analysis1 = await self.query_model_with_image(
                image1_b64, "Describe this image in detail.", None, 0.7, 512
            )
            
            analysis2 = await self.query_model_with_image(
                image2_b64, "Describe this image in detail.", None, 0.7, 512
            )
            
            # Now ask for comparison
            comparison_text = f"""
            I have analyzed two images:
            
            Image 1: {analysis1}
            
            Image 2: {analysis2}
            
            {comparison_prompt}
            """
            
            comparison = await self.query_model_with_image(
                image1_b64, comparison_text, None, 0.7, 1024
            )
            
            result = {
                "model": self.capabilities.model_name,
                "image1_info": image1_info,
                "image2_info": image2_info,
                "image1_analysis": analysis1,
                "image2_analysis": analysis2,
                "comparison": comparison
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error comparing images: {str(e)}")]

    async def extract_text(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Extract text from an image"""
        try:
            image_path = arguments.get("image_path")
            image_data = arguments.get("image_data")
            format_output = arguments.get("format_output", True)
            
            # Load image
            image_b64, image_info = self.load_image_as_base64(image_path, image_data)
            
            # Create OCR-focused prompt
            if format_output:
                prompt = """Extract all text from this image and format it clearly. 
                Preserve the structure and layout as much as possible. 
                If there are multiple sections or columns, indicate this in your response."""
            else:
                prompt = "Extract all text visible in this image."
            
            response = await self.query_model_with_image(image_b64, prompt)
            
            result = {
                "model": self.capabilities.model_name,
                "image_info": image_info,
                "extracted_text": response
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error extracting text: {str(e)}")]

    async def analyze_screenshot(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Specialized screenshot analysis"""
        try:
            image_path = arguments.get("image_path")
            image_data = arguments.get("image_data")
            analysis_focus = arguments.get("analysis_focus", "general")
            
            # Load image
            image_b64, image_info = self.load_image_as_base64(image_path, image_data)
            
            # Create focus-specific prompts
            prompts = {
                "ui_elements": "Analyze this screenshot and identify all UI elements (buttons, menus, text fields, etc.). Describe their layout and purpose.",
                "layout": "Analyze the layout and design of this screenshot. Comment on the visual hierarchy, spacing, and overall organization.",
                "content": "Focus on the content visible in this screenshot. What information is being displayed? What is the main purpose of this interface?",
                "accessibility": "Analyze this screenshot from an accessibility perspective. Are there any potential issues with contrast, text size, or navigation?",
                "general": "Analyze this screenshot comprehensively. Describe what application or website this appears to be, the main functionality visible, and any notable elements."
            }
            
            prompt = prompts.get(analysis_focus, prompts["general"])
            
            response = await self.query_model_with_image(image_b64, prompt)
            
            result = {
                "model": self.capabilities.model_name,
                "image_info": image_info,
                "analysis_focus": analysis_focus,
                "analysis": response
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error analyzing screenshot: {str(e)}")]

    async def batch_analyze_images(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Analyze multiple images in a directory"""
        try:
            directory_path = arguments["directory_path"]
            analysis_prompt = arguments.get("analysis_prompt", "Analyze this image and describe what you see.")
            summary_prompt = arguments.get("summary_prompt", "Provide a summary of all the analyzed images, noting patterns and key findings.")
            max_images = arguments.get("max_images", 10)
            
            if not os.path.exists(directory_path):
                return [TextContent(type="text", text=f"Directory not found: {directory_path}")]
            
            # Find image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
            image_files = []
            
            for file_path in Path(directory_path).iterdir():
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(str(file_path))
                    if len(image_files) >= max_images:
                        break
            
            if not image_files:
                return [TextContent(type="text", text="No image files found in directory")]
            
            # Analyze each image
            results = []
            for image_path in image_files:
                try:
                    image_b64, image_info = self.load_image_as_base64(image_path)
                    analysis = await self.query_model_with_image(image_b64, analysis_prompt)
                    
                    results.append({
                        "filename": os.path.basename(image_path),
                        "image_info": image_info,
                        "analysis": analysis
                    })
                except Exception as e:
                    results.append({
                        "filename": os.path.basename(image_path),
                        "error": str(e)
                    })
            
            # Generate summary using text-only completion
            analyses_text = "\n\n".join([
                f"Image: {r['filename']}\nAnalysis: {r.get('analysis', 'Error: ' + r.get('error', 'Unknown error'))}"
                for r in results
            ])
            
            summary_text = f"Here are the analyses of {len(results)} images:\n\n{analyses_text}\n\n{summary_prompt}"
            
            try:
                summary_payload = {
                    "messages": [
                        {"role": "user", "content": summary_text}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "stream": False
                }
                
                response = await self.lm_studio_client.post("/v1/chat/completions", json=summary_payload)
                response.raise_for_status()
                summary_result = response.json()
                summary = summary_result["choices"][0]["message"]["content"]
            except Exception as e:
                summary = f"Error generating summary: {str(e)}"
            
            batch_result = {
                "model": self.capabilities.model_name,
                "directory": directory_path,
                "total_images": len(image_files),
                "successful_analyses": len([r for r in results if "error" not in r]),
                "results": results,
                "summary": summary
            }
            
            return [TextContent(type="text", text=json.dumps(batch_result, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error in batch analysis: {str(e)}")]

    async def run(self):
        """Run the MCP server"""
        # Detect capabilities on startup
        await self.detect_model_capabilities()
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="lm-studio-mcp-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    ),
                ),
            )

async def main():
    """Main entry point"""
    server = GeneralizedLMStudioServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main()) 