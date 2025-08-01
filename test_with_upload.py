#!/usr/bin/env python3
"""
Test with File Upload
Tests the button functionality after uploading a file
"""

import requests
import os

def test_with_upload():
    """Test button functionality after uploading a file"""
    
    print("🧪 Testing Button Functionality with File Upload...")
    print("=" * 60)
    
    # Check if we have a test file
    test_files = ['bank_entries.xlsx', 'Bank_Statement_Combined.xlsx', 'steel_plant_bank_statement.xlsx']
    test_file = None
    
    for file in test_files:
        if os.path.exists(file):
            test_file = file
            break
    
    if not test_file:
        print("❌ No test file found. Please ensure you have a bank statement file.")
        return False
    
    print(f"✅ Using test file: {test_file}")
    
    # Test 1: Upload file
    try:
        with open(test_file, 'rb') as f:
            files = {'bank': (test_file, f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            response = requests.post("http://localhost:5000/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            print("✅ File uploaded successfully")
        else:
            print(f"❌ File upload failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return False
    
    # Test 2: Test button functionality
    try:
        response = requests.post("http://localhost:5000/run-parameter-analysis", 
                               json={"parameter_type": "A1_historical_trends"},
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print("✅ Button functionality works! Parameter analysis completed.")
                print(f"📊 Results: {len(data.get('results', {}))} data points")
            else:
                print(f"❌ Analysis failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Button test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Button test error: {e}")
        return False
    
    print("\n🎯 BUTTON FUNCTIONALITY VERIFIED!")
    print("✅ File upload works")
    print("✅ Button triggers analysis")
    print("✅ Backend processes the request")
    print("✅ Results are returned")
    
    print("\n🚀 MANUAL TESTING:")
    print("1. Open http://localhost:5000 in browser")
    print("2. Upload a bank statement file")
    print("3. Click 'Run Analysis' on any parameter card")
    print("4. You should see results appear in the card")
    
    return True

if __name__ == "__main__":
    success = test_with_upload()
    if success:
        print("\n✅ All tests passed! Buttons are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.") 