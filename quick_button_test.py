#!/usr/bin/env python3
"""
Quick test to verify button functionality
"""

import requests
import json

def test_button_functionality():
    """Test if the buttons are working properly"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 QUICK BUTTON FUNCTIONALITY TEST")
    print("=" * 40)
    
    # Test 1: Check if server is responding
    print("\n1️⃣ Testing server response...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Server is responding")
        else:
            print(f"❌ Server error: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Server connection error: {e}")
        return
    
    # Test 2: Test vendor analysis with a simple request
    print("\n2️⃣ Testing vendor analysis endpoint...")
    try:
        vendor_data = {
            'vendor': 'test_vendor',
            'analysis_type': 'cash_flow',
            'ai_model': 'hybrid'
        }
        
        response = requests.post(f"{base_url}/vendor-analysis", json=vendor_data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Vendor analysis endpoint is working")
            print(f"✅ Response: {result.get('ai_model', 'Unknown')}")
        else:
            print(f"❌ Vendor analysis failed: {response.status_code}")
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Vendor analysis error: {e}")
    
    # Test 3: Test transaction analysis with a simple request
    print("\n3️⃣ Testing transaction analysis endpoint...")
    try:
        transaction_data = {
            'transaction_type': 'all',
            'analysis_type': 'cash_flow',
            'ai_model': 'hybrid'
        }
        
        response = requests.post(f"{base_url}/transaction-analysis", json=transaction_data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Transaction analysis endpoint is working")
            print(f"✅ Response: {result.get('ai_model', 'Unknown')}")
        else:
            print(f"❌ Transaction analysis failed: {response.status_code}")
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Transaction analysis error: {e}")
    
    print("\n" + "=" * 40)
    print("🎯 BUTTON TEST COMPLETED")
    print("✅ If server is responding, the buttons should work")
    print("✅ Try clicking 'Run Cash Flow Analysis' in the interface")

if __name__ == "__main__":
    test_button_functionality() 