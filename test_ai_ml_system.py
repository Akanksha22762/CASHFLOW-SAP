#!/usr/bin/env python3
"""
Test script to verify XGBoost and Ollama integration in the system
"""

import requests
import json
import time

def test_ai_ml_system():
    """Test the AI/ML system integration"""
    print("🧪 Testing AI/ML System Integration...")
    
    base_url = "http://localhost:5000"
    
    # Test 1: Check if the server is running
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test 2: Check if Ollama is available
    try:
        response = requests.get(f"{base_url}/check-ollama")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Ollama status: {data.get('status', 'Unknown')}")
            if data.get('available'):
                print("✅ Ollama is available and working")
            else:
                print("⚠️ Ollama is not available, will use fallback")
        else:
            print(f"⚠️ Ollama check endpoint returned {response.status_code}")
    except Exception as e:
        print(f"⚠️ Ollama check failed: {e}")
    
    # Test 3: Test transaction analysis with AI/ML
    print("\n🔍 Testing Transaction Analysis with AI/ML...")
    
    # Create test transaction data
    test_data = {
        "category_type": "Operating Activities",
        "analysis_type": "category_analysis",
        "vendor_name": None
    }
    
    try:
        response = requests.post(
            f"{base_url}/get-transaction-details",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Transaction analysis successful")
            
            # Check if AI/ML processing was used
            if 'ai_model' in data:
                print(f"✅ AI Model used: {data['ai_model']}")
            else:
                print("⚠️ No AI model information in response")
            
            # Check if patterns were generated
            if 'patterns' in data:
                patterns = data['patterns']
                print(f"✅ ML Patterns generated:")
                print(f"   - Trend: {patterns.get('trend', 'N/A')}")
                print(f"   - Volatility: {patterns.get('volatility', 'N/A')}")
                print(f"   - Consistency: {patterns.get('consistency', 'N/A')}")
                print(f"   - Amount Pattern: {patterns.get('amount_pattern', 'N/A')}")
            else:
                print("⚠️ No ML patterns in response")
            
            # Check transaction count
            if 'total_count' in data:
                print(f"✅ Transaction count: {data['total_count']}")
            else:
                print("⚠️ No transaction count in response")
                
        else:
            print(f"❌ Transaction analysis failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Transaction analysis error: {e}")
    
    # Test 4: Test vendor analysis with AI/ML
    print("\n🏢 Testing Vendor Analysis with AI/ML...")
    
    test_vendor_data = {
        "category_type": None,
        "analysis_type": "vendor_analysis",
        "vendor_name": "Test Vendor"
    }
    
    try:
        response = requests.post(
            f"{base_url}/get-transaction-details",
            json=test_vendor_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Vendor analysis successful")
            
            if 'ai_model' in data:
                print(f"✅ AI Model used: {data['ai_model']}")
            
            if 'total_count' in data:
                print(f"✅ Vendor transaction count: {data['total_count']}")
                
        else:
            print(f"❌ Vendor analysis failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Vendor analysis error: {e}")
    
    print("\n🎯 AI/ML System Test Complete!")
    return True

def test_ollama_directly():
    """Test Ollama directly if available"""
    print("\n🔍 Testing Ollama Directly...")
    
    try:
        from ollama_simple_integration import simple_ollama, check_ollama_availability
        
        # Check availability
        availability = check_ollama_availability()
        print(f"✅ Ollama availability check: {availability}")
        
        if availability.get('available'):
            # Test simple prompt
            prompt = "Analyze this financial data: 100 transactions, total amount ₹50,00,000"
            response = simple_ollama(prompt, "llama2:7b", max_tokens=50)
            
            if response:
                print(f"✅ Ollama response: {response[:100]}...")
            else:
                print("⚠️ Ollama returned empty response")
        else:
            print("⚠️ Ollama is not available")
            
    except ImportError as e:
        print(f"❌ Cannot import Ollama integration: {e}")
    except Exception as e:
        print(f"❌ Ollama test error: {e}")

if __name__ == "__main__":
    print("🚀 Starting AI/ML System Tests...")
    
    # Test the main system
    test_ai_ml_system()
    
    # Test Ollama directly
    test_ollama_directly()
    
    print("\n✨ All tests completed!")
