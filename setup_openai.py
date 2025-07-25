#!/usr/bin/env python3
"""
Setup script to enable OpenAI GPT for real AI categorization
"""

import os
import sys

def setup_openai():
    """Setup OpenAI API key for real AI categorization"""
    
    print("🤖 Setting up OpenAI GPT for Real AI Categorization")
    print("=" * 60)
    
    # Check if API key already exists
    current_key = os.getenv('OPENAI_API_KEY')
    if current_key:
        print(f"✅ OpenAI API key already found: {current_key[:10]}...")
        print("🔧 To change it, set a new environment variable")
    else:
        print("❌ No OpenAI API key found")
        print("🔑 You need to get an API key from OpenAI")
    
    print("\n📋 Steps to Enable Real AI:")
    print("1. Go to https://platform.openai.com/api-keys")
    print("2. Create a new API key")
    print("3. Set the environment variable:")
    print("   Windows: set OPENAI_API_KEY=your_api_key_here")
    print("   Linux/Mac: export OPENAI_API_KEY=your_api_key_here")
    print("   Or create a .env file with: OPENAI_API_KEY=your_api_key_here")
    
    print("\n💡 Alternative: Use Ollama (Local LLM)")
    print("1. Install Ollama from https://ollama.ai")
    print("2. Download a model: ollama pull llama2")
    print("3. The system will automatically detect it")
    
    print("\n🎯 Benefits of Real AI:")
    print("✅ Better categorization accuracy")
    print("✅ Handles complex descriptions")
    print("✅ Learns from context")
    print("✅ More intelligent analysis")
    
    return True

if __name__ == '__main__':
    setup_openai() 