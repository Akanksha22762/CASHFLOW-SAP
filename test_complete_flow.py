#!/usr/bin/env python3
"""
Complete Flow Test
Tests the entire flow from API to UI display
"""

import requests
import json
import time
import pandas as pd
import os

def test_complete_flow():
    """Test the complete flow from API to UI"""
    print("🧠 TESTING COMPLETE FLOW")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Check server
    print("\n🔍 Test 1: Server Connection")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
        else:
            print(f"⚠️ Server responded with status: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        return
    
    # Test 2: Create and upload test data
    print("\n🔍 Test 2: Create Test Data")
    print("-" * 40)
    
    try:
        # Create test data
        test_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=20, freq='D'),
            'Description': ['Test transaction ' + str(i) for i in range(1, 21)],
            'Amount': [100 + i * 10 for i in range(20)],
            'Category': ['Operating'] * 20
        })
        
        # Save to temporary file
        test_file = 'test_bank_data.xlsx'
        test_data.to_excel(test_file, index=False)
        
        print(f"✅ Test data created: {test_data.shape}")
        print(f"   File: {test_file}")
        
    except Exception as e:
        print(f"❌ Test data creation failed: {e}")
        return
    
    # Test 3: Upload test data
    print("\n🔍 Test 3: Upload Test Data")
    print("-" * 40)
    
    try:
        with open(test_file, 'rb') as f:
            files = {'bank_file': (test_file, f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            response = requests.post(f"{base_url}/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Data upload successful!")
            print(f"   Response: {result}")
        else:
            print(f"❌ Data upload failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return
            
    except Exception as e:
        print(f"❌ Data upload failed: {e}")
        return
    
    # Test 4: Test parameter analysis with reasoning
    print("\n🔍 Test 4: Parameter Analysis with Reasoning")
    print("-" * 40)
    
    try:
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
            print("✅ Parameter analysis successful!")
            
            # Check for reasoning
            if 'reasoning_explanations' in result:
                print("✅ Reasoning explanations found!")
                reasoning = result['reasoning_explanations']
                print(f"   📊 ML Analysis: {'✅' if 'ml_analysis' in reasoning else '❌'}")
                print(f"   🧠 AI Analysis: {'✅' if 'ai_analysis' in reasoning else '❌'}")
                print(f"   🔗 Hybrid Analysis: {'✅' if 'hybrid_analysis' in reasoning else '❌'}")
                
                # Show sample reasoning content
                if 'ml_analysis' in reasoning:
                    ml = reasoning['ml_analysis']
                    print(f"   📝 ML Decision Logic: {ml.get('decision_logic', 'N/A')[:100]}...")
                
                if 'ai_analysis' in reasoning:
                    ai = reasoning['ai_analysis']
                    print(f"   📝 AI Decision Logic: {ai.get('decision_logic', 'N/A')[:100]}...")
                
                if 'hybrid_analysis' in reasoning:
                    hybrid = reasoning['hybrid_analysis']
                    print(f"   📝 Hybrid Decision Logic: {hybrid.get('decision_logic', 'N/A')[:100]}...")
                
                print(f"\n🧠 Complete reasoning structure:")
                print(json.dumps(reasoning, indent=2, default=str))
                
            else:
                print("❌ No reasoning explanations found!")
                print(f"   Response keys: {list(result.keys())}")
                print(f"   Full response: {json.dumps(result, indent=2, default=str)}")
        else:
            print(f"❌ Parameter analysis failed: {response.status_code}")
            print(f"   Response: {response.text[:500]}...")
            
    except Exception as e:
        print(f"❌ Parameter analysis test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Test vendor analysis with reasoning
    print("\n🔍 Test 5: Vendor Analysis with Reasoning")
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
            print("✅ Vendor analysis successful!")
            
            # Check for reasoning
            if 'reasoning_explanations' in result:
                print("✅ Reasoning explanations found in vendor analysis!")
                reasoning = result['reasoning_explanations']
                print(f"   📊 ML Analysis: {'✅' if 'ml_analysis' in reasoning else '❌'}")
                print(f"   🧠 AI Analysis: {'✅' if 'ai_analysis' in reasoning else '❌'}")
                print(f"   🔗 Hybrid Analysis: {'✅' if 'hybrid_analysis' in reasoning else '❌'}")
            else:
                print("❌ No reasoning explanations found in vendor analysis!")
                print(f"   Response keys: {list(result.keys())}")
        else:
            print(f"❌ Vendor analysis failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Vendor analysis test failed: {e}")
    
    # Cleanup
    try:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n🧹 Cleaned up test file: {test_file}")
    except:
        pass
    
    print("\n🎯 Analysis:")
    print("   1. If reasoning is found in API responses, the issue is in the UI JavaScript")
    print("   2. If reasoning is NOT found, the issue is in the backend")
    print("   3. Check browser console for JavaScript errors")
    print("   4. Verify the UI templates are properly updated")

if __name__ == "__main__":
    print("🚀 Complete Flow Test")
    print("=" * 60)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(3)
    
    # Test the complete flow
    test_complete_flow()
    
    print("\n🎉 Testing Complete!")
