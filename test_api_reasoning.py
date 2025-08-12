#!/usr/bin/env python3
"""
Test API Reasoning Integration
Verifies that API endpoints return reasoning explanations
"""

import requests
import json
import time

def test_api_reasoning():
    """Test that API endpoints return reasoning explanations"""
    print("🧠 TESTING API REASONING INTEGRATION")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Check if server is running
    print("\n🔍 Test 1: Server Connection")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
        else:
            print(f"⚠️ Server responded with status: {response.status_code}")
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        print("Make sure Flask app is running: python app1.py")
        return
    
    # Test 2: Test parameter analysis endpoint
    print("\n🔍 Test 2: Parameter Analysis Endpoint")
    print("-" * 40)
    
    try:
        # Create test data
        test_data = {
            "parameter_type": "A1_historical_trends",
            "vendor_name": None,
            "analysis_depth": "detailed"
        }
        
        response = requests.post(
            f"{base_url}/run-parameter-analysis",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Parameter analysis endpoint responded successfully")
            
            # Check if reasoning is in response
            if 'reasoning_explanations' in result:
                print("✅ Reasoning explanations found in response!")
                print(f"   📊 ML Analysis: {'✅' if 'ml_analysis' in result['reasoning_explanations'] else '❌'}")
                print(f"   🧠 AI Analysis: {'✅' if 'ai_analysis' in result['reasoning_explanations'] else '❌'}")
                print(f"   🔗 Hybrid Analysis: {'✅' if 'hybrid_analysis' in result['reasoning_explanations'] else '❌'}")
                
                # Show sample reasoning
                if 'ml_analysis' in result['reasoning_explanations']:
                    ml_reasoning = result['reasoning_explanations']['ml_analysis']
                    print(f"   📝 ML Decision Logic: {ml_reasoning.get('decision_logic', 'N/A')[:100]}...")
            else:
                print("❌ No reasoning explanations found in response")
                print(f"   Response keys: {list(result.keys())}")
        else:
            print(f"❌ Parameter analysis failed with status: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Parameter analysis test failed: {e}")
    
    # Test 3: Test vendor analysis endpoint
    print("\n🔍 Test 3: Vendor Analysis Endpoint")
    print("-" * 40)
    
    try:
        test_data = {
            "vendor": "auto",
            "analysis_type": "cash_flow",
            "ai_model": "hybrid"
        }
        
        response = requests.post(
            f"{base_url}/vendor-analysis",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Vendor analysis endpoint responded successfully")
            
            # Check if reasoning is in response
            if 'reasoning_explanations' in result:
                print("✅ Reasoning explanations found in vendor response!")
                print(f"   📊 ML Analysis: {'✅' if 'ml_analysis' in result['reasoning_explanations'] else '❌'}")
                print(f"   🧠 AI Analysis: {'✅' if 'ai_analysis' in result['reasoning_explanations'] else '❌'}")
                print(f"   🔗 Hybrid Analysis: {'✅' if 'hybrid_analysis' in result['reasoning_explanations'] else '❌'}")
            else:
                print("❌ No reasoning explanations found in vendor response")
                print(f"   Response keys: {list(result.keys())}")
        else:
            print(f"❌ Vendor analysis failed with status: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Vendor analysis test failed: {e}")
    
    print("\n🎯 Next Steps:")
    print("   1. If reasoning is found in API responses, the issue is in the UI")
    print("   2. If reasoning is NOT found, the issue is in the backend")
    print("   3. Check browser console for JavaScript errors")
    print("   4. Verify data files are uploaded before running analysis")

if __name__ == "__main__":
    print("🚀 API Reasoning Integration Test")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Test the API reasoning
    test_api_reasoning()
    
    print("\n🎉 API Testing Complete!")
