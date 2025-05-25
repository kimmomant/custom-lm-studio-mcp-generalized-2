#!/usr/bin/env python3
"""
Demonstration script showing how the generalized MCP server adapts to model capabilities
"""

import asyncio
import json
from generalized_lm_studio_mcp_server import GeneralizedLMStudioServer

async def demo_capability_adaptation():
    """Demonstrate how the server adapts to different model capabilities"""
    print("🚀 Generalized LM Studio MCP Server - Capability Adaptation Demo")
    print("=" * 70)
    
    server = GeneralizedLMStudioServer()
    
    # Step 1: Detect current model capabilities
    print("📊 Step 1: Detecting Current Model Capabilities")
    print("-" * 50)
    
    capabilities = await server.detect_model_capabilities()
    
    print(f"🤖 Current Model: {capabilities.model_name}")
    print(f"📝 Text Support: {'✅' if capabilities.supports_text else '❌'}")
    print(f"💬 Chat Support: {'✅' if capabilities.supports_chat else '❌'}")
    print(f"👁️  Vision Support: {'✅' if capabilities.supports_vision else '❌'}")
    
    # Step 2: Show available tools
    print(f"\n🛠️  Step 2: Available Tools for {capabilities.model_name}")
    print("-" * 50)
    
    # Get the tools that would be available
    # We'll manually check what tools would be available based on capabilities
    universal_tools = ["health_check", "list_models", "get_current_model", "chat_completion"]
    vision_tools = []
    
    if capabilities.supports_vision:
        vision_tools = [
            "analyze_image", "chat_with_image", "compare_images", 
            "extract_text", "analyze_screenshot", "batch_analyze_images"
        ]
    
    print("🌐 Universal Tools (available for all models):")
    for tool in universal_tools:
        print(f"   ✅ {tool}")
    
    if vision_tools:
        print("\n👁️  Vision Tools (only for vision-capable models):")
        for tool in vision_tools:
            print(f"   ✅ {tool}")
    else:
        print("\n👁️  Vision Tools: ❌ Not available (model doesn't support vision)")
    
    # Step 3: Test basic functionality
    print(f"\n🧪 Step 3: Testing Basic Functionality")
    print("-" * 50)
    
    # Test chat completion (should work for all models)
    try:
        print("Testing chat completion...")
        chat_result = await server.chat_completion({
            "prompt": "In one sentence, what type of AI model are you?",
            "max_tokens": 50
        })
        print(f"✅ Chat Response: {chat_result[0].text[:100]}...")
    except Exception as e:
        print(f"❌ Chat Error: {e}")
    
    # Test vision capability if available
    if capabilities.supports_vision:
        print(f"\n👁️  Step 4: Testing Vision Capabilities")
        print("-" * 50)
        
        # Create a simple test image
        from PIL import Image
        import base64
        from io import BytesIO
        
        # Create a simple colored square
        test_image = Image.new('RGB', (100, 100), color='blue')
        buffer = BytesIO()
        test_image.save(buffer, format='JPEG')
        test_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        try:
            print("Testing image analysis with a blue square...")
            analysis_result = await server.analyze_image({
                "image_data": test_image_b64,
                "prompt": "What color is this image? Be brief.",
                "max_tokens": 30
            })
            
            result_data = json.loads(analysis_result[0].text)
            print(f"✅ Vision Response: {result_data['analysis'][:100]}...")
            
        except Exception as e:
            print(f"❌ Vision Error: {e}")
    else:
        print(f"\n👁️  Step 4: Vision Testing Skipped")
        print("-" * 50)
        print("❌ Current model doesn't support vision capabilities")
        print("💡 Try loading a vision model like Gemma-3-27B-IT or LLaVA")
    
    # Step 5: Show model switching guidance
    print(f"\n🔄 Step 5: Model Switching Demonstration")
    print("-" * 50)
    print("To see the server adapt to different capabilities:")
    print("1. 🔄 Switch to a text-only model in LM Studio (e.g., Phi-4)")
    print("2. 🔄 Click refresh button next to MCP server in Cursor settings")
    print("3. 🔄 Vision tools will disappear from Cursor")
    print("4. 🔄 Switch back to a vision model (e.g., Gemma-3-27B-IT)")
    print("5. 🔄 Refresh in Cursor again - vision tools will reappear")
    
    print(f"\n🎉 Demo Complete!")
    print("=" * 70)
    print("The generalized server automatically adapts to any model's capabilities!")

if __name__ == "__main__":
    asyncio.run(demo_capability_adaptation()) 