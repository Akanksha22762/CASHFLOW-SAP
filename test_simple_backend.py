#!/usr/bin/env python3
"""
Simple test to check backend status and data availability
"""

import requests
import json

def test_backend_status():
    """Test basic backend connectivity and data status"""
    
    base_url = "http://127.0.0.1:5000"
    
    print("🔍 Testing Backend Status")
    print("=" * 40)
    
    # Test 1: Basic connectivity
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Backend is running (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Backend connection failed: {e}")
        return
    
    # Test 2: Check if any data is uploaded
    try:
        response = requests.post(
            f"{base_url}/get-transaction-details",
            json={
                "category_type": "Investing Activities",
                "analysis_type": "category_analysis"
            },
            timeout=10
        )
        
        print(f"📊 API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API call successful")
            print(f"   Success: {data.get('success')}")
            print(f"   Transactions: {len(data.get('transactions', []))}")
            print(f"   Data Source: {data.get('data_source', 'Unknown')}")
        elif response.status_code == 400:
            data = response.json()
            print(f"⚠️ API returned error: {data.get('error')}")
            
            # Try with old parameter format
            print("\n🔄 Trying with old parameter format...")
            response2 = requests.post(
                f"{base_url}/get-transaction-details",
                json={
                    "parameter_type": "investing",
                    "vendor_name": ""
                },
                timeout=10
            )
            
            if response2.status_code == 200:
                data2 = response2.json()
                print(f"✅ Old format works! Found {len(data2.get('transactions', []))} transactions")
            else:
                print(f"❌ Old format also failed: {response2.status_code}")
                
        else:
            print(f"❌ Unexpected status: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
    
    print("\n" + "=" * 40)
    print("🏁 Backend Status Test Complete!")

if __name__ == "__main__":
    test_backend_status()
