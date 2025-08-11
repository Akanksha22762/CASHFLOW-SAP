#!/usr/bin/env python3
"""
Test script for simplified interface with cash flow analysis
"""

import requests
import json
import time

def test_simplified_interface():
    """Test the simplified interface with cash flow analysis"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 TESTING SIMPLIFIED INTERFACE WITH CASH FLOW ANALYSIS")
    print("=" * 60)
    
    # Test 1: Transaction Analysis
    print("\n1️⃣ Testing Transaction Analysis...")
    try:
        transaction_data = {
            'transaction_type': 'all',
            'analysis_type': 'cash_flow',  # Should be ignored, always uses cash_flow
            'ai_model': 'hybrid'  # Should be ignored, always uses hybrid
        }
        
        response = requests.post(f"{base_url}/transaction-analysis", json=transaction_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Transaction Analysis: {result.get('transactions_analyzed', 0)} transactions processed")
            print(f"✅ AI Model: {result.get('ai_model', 'Unknown')}")
            print(f"✅ Analysis Type: {result.get('analysis_type', 'Unknown')}")
            
            if 'data' in result and result['data']:
                print(f"✅ Analysis completed successfully")
            else:
                print(f"⚠️ No analysis data returned")
        else:
            print(f"❌ Transaction Analysis failed: {response.status_code}")
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Transaction Analysis error: {e}")
    
    # Test 2: Vendor Analysis
    print("\n2️⃣ Testing Vendor Analysis...")
    try:
        vendor_data = {
            'vendor': 'auto',  # Extract vendors automatically
            'analysis_type': 'cash_flow',  # Should be ignored, always uses cash_flow
            'ai_model': 'hybrid'  # Should be ignored, always uses hybrid
        }
        
        response = requests.post(f"{base_url}/vendor-analysis", json=vendor_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Vendor Analysis: {result.get('vendors_analyzed', 0)} vendors processed")
            print(f"✅ AI Model: {result.get('ai_model', 'Unknown')}")
            print(f"✅ Analysis Type: {result.get('analysis_type', 'Unknown')}")
            
            if 'data' in result and result['data']:
                print(f"✅ Analysis completed successfully")
                # Show first vendor result if available
                vendor_names = list(result['data'].keys())
                if vendor_names and vendor_names[0] != 'error':
                    first_vendor = vendor_names[0]
                    print(f"✅ Sample vendor: {first_vendor}")
            else:
                print(f"⚠️ No analysis data returned")
        else:
            print(f"❌ Vendor Analysis failed: {response.status_code}")
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Vendor Analysis error: {e}")
    
    # Test 3: Check if interface is simplified
    print("\n3️⃣ Testing Interface Simplification...")
    try:
        response = requests.get(f"{base_url}/")
        
        if response.status_code == 200:
            html_content = response.text
            
            # Check if Analysis Type dropdowns are removed
            if 'vendorAnalysisDropdown' not in html_content:
                print("✅ Vendor Analysis Type dropdown removed")
            else:
                print("❌ Vendor Analysis Type dropdown still present")
            
            if 'transactionAnalysisDropdown' not in html_content:
                print("✅ Transaction Analysis Type dropdown removed")
            else:
                print("❌ Transaction Analysis Type dropdown still present")
            
            # Check if AI Model dropdowns are removed
            if 'vendorAIModelDropdown' not in html_content:
                print("✅ Vendor AI Model dropdown removed")
            else:
                print("❌ Vendor AI Model dropdown still present")
            
            if 'transactionAIModelDropdown' not in html_content:
                print("✅ Transaction AI Model dropdown removed")
            else:
                print("❌ Transaction AI Model dropdown still present")
            
            # Check if Analysis Options section is removed
            if 'Analysis Options' not in html_content:
                print("✅ Analysis Options section removed")
            else:
                print("❌ Analysis Options section still present")
            
            # Check if buttons are updated
            if 'Run Cash Flow Analysis' in html_content:
                print("✅ Buttons updated to show 'Run Cash Flow Analysis'")
            else:
                print("❌ Buttons not updated")
                
        else:
            print(f"❌ Failed to load interface: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Interface test error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 SIMPLIFIED INTERFACE TEST COMPLETED")
    print("✅ Interface should now be simplified with cash flow analysis only")
    print("✅ All dropdowns for analysis type and AI model should be removed")
    print("✅ Analysis Options section should be completely removed")
    print("✅ Both vendor and transaction analysis should use cash flow analysis with hybrid model")

if __name__ == "__main__":
    test_simplified_interface() 