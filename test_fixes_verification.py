#!/usr/bin/env python3
"""
Test script to verify the cross button and network error fixes
"""

import requests
import json
import time

def test_fixes():
    """Test the fixes for cross button and network errors"""
    
    base_url = "http://127.0.0.1:5000"
    
    print("🧪 Testing Fixes for Cross Button and Network Errors")
    print("=" * 60)
    
    # Test 1: Check if app is running
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ App is running successfully")
        else:
            print(f"❌ App returned status {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to app: {e}")
        return
    
    # Test 2: Test parameter analysis (should work without network errors)
    try:
        print("\n🔍 Testing parameter analysis...")
        
        response = requests.post(
            f"{base_url}/run-parameter-analysis",
            json={"parameter_type": "A1_historical_trends"},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print("✅ Parameter analysis completed successfully")
                print("✅ No network errors detected")
                
                # Check if results contain data
                results = data.get('results', {})
                if results:
                    print("✅ Results data is available")
                    print(f"📊 Result keys: {list(results.keys())}")
                else:
                    print("⚠️ No results data found")
                    
            else:
                print(f"❌ Analysis failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing parameter analysis: {e}")
    
    print("\n" + "=" * 60)
    print("📋 MANUAL TESTING INSTRUCTIONS:")
    print("=" * 60)
    print("1. Open browser: http://127.0.0.1:5000")
    print("2. Upload a bank statement file")
    print("3. Click 'Run Analysis' on any parameter card")
    print("4. Wait for completion (should show 'Completed')")
    print("5. Click 'View Results' to open modal")
    print("6. Test the red X button - should close modal")
    print("7. Test clicking outside modal - should close")
    print("8. Test pressing Escape key - should close")
    print("9. Check browser console (F12) for debug messages")
    print("\n✅ Expected Results:")
    print("   - No network errors in console")
    print("   - Modal opens with actual data")
    print("   - Red X button closes modal")
    print("   - Multiple ways to close modal work")

if __name__ == "__main__":
    test_fixes() 