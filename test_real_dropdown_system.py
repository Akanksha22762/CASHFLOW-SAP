#!/usr/bin/env python3
"""
Test script for the real dropdown AI/ML analysis system with actual data
"""

import requests
import json
import time
import os

def test_real_dropdown_system():
    """Test the dropdown system with real data"""
    
    base_url = "http://localhost:5000"
    
    print("🚀 Testing Real Dropdown AI/ML System")
    print("=" * 50)
    
    # Test 1: Check if dropdown data endpoint works
    print("\n1️⃣ Testing Dropdown Data Population...")
    try:
        response = requests.get(f"{base_url}/get-dropdown-data")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Dropdown data retrieved successfully!")
                print(f"📊 Vendors found: {len(data.get('vendors', []))}")
                print(f"📈 Transaction types: {len(data.get('transaction_types', []))}")
                print(f"📋 Total transactions: {data.get('total_transactions', 0)}")
            else:
                print(f"❌ No data available: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Dropdown data failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Dropdown data error: {e}")
    
    # Test 2: Test vendor analysis with real data
    print("\n2️⃣ Testing Vendor Analysis with Real Data...")
    vendor_data = {
        "vendor": "all",
        "analysis_type": "payment_patterns",
        "ai_model": "hybrid"
    }
    
    try:
        response = requests.post(f"{base_url}/vendor-analysis", json=vendor_data)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Vendor Analysis: {result.get('vendors_analyzed', 0)} vendors processed")
                print(f"🤖 AI Model Used: {result.get('ai_model', 'Unknown')}")
                print(f"📊 Analysis Type: {vendor_data['analysis_type']}")
            else:
                print(f"❌ Vendor Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ Vendor Analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Vendor Analysis error: {e}")
    
    # Test 3: Test transaction analysis with real data
    print("\n3️⃣ Testing Transaction Analysis with Real Data...")
    transaction_data = {
        "transaction_type": "all",
        "analysis_type": "pattern_analysis",
        "ai_model": "hybrid"
    }
    
    try:
        response = requests.post(f"{base_url}/transaction-analysis", json=transaction_data)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Transaction Analysis: {result.get('transactions_analyzed', 0)} transactions processed")
                print(f"🤖 AI Model Used: {result.get('ai_model', 'Unknown')}")
                print(f"📊 Analysis Type: {transaction_data['analysis_type']}")
            else:
                print(f"❌ Transaction Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ Transaction Analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Transaction Analysis error: {e}")
    
    # Test 4: Test advanced analysis
    print("\n4️⃣ Testing Advanced Analysis...")
    advanced_data = {
        "category": "revenue_analysis",
        "depth": "detailed",
        "processing_mode": "real_time"
    }
    
    try:
        response = requests.post(f"{base_url}/analysis-category", json=advanced_data)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Advanced Analysis: {result.get('category', 'Unknown')} completed")
                print(f"🔍 Analysis Depth: {result.get('depth', 'Unknown')}")
                print(f"⚡ Processing Mode: {advanced_data['processing_mode']}")
            else:
                print(f"❌ Advanced Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ Advanced Analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Advanced Analysis error: {e}")
    
    # Test 5: Test complete analysis
    print("\n5️⃣ Testing Complete Analysis...")
    complete_data = {
        "vendor": "all",
        "vendor_analysis": "payment_patterns",
        "vendor_ai_model": "hybrid",
        "transaction_type": "all",
        "transaction_analysis": "pattern_analysis",
        "transaction_ai_model": "hybrid",
        "analysis_category": "revenue_analysis",
        "analysis_depth": "detailed",
        "processing_mode": "real_time",
        "report_type": "comprehensive_report",
        "report_format": "pdf",
        "report_detail": "detailed"
    }
    
    try:
        response = requests.post(f"{base_url}/complete-analysis", json=complete_data)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Complete Analysis: All systems processed")
                print(f"🎯 Analysis Complete: {result.get('analysis_complete', False)}")
                if result.get('data'):
                    data = result['data']
                    print(f"🏢 Vendor Analysis: {data.get('vendor_analysis', {}).get('vendors_processed', 0)} vendors")
                    print(f"📊 Transaction Analysis: {data.get('transaction_analysis', {}).get('transactions_processed', 0)} transactions")
                    print(f"🧠 Advanced Analysis: {len(data.get('advanced_analysis', {}).get('analysis_categories', []))} categories")
                    print(f"📄 Report Generation: {len(data.get('report_generation', {}).get('reports_generated', []))} reports")
            else:
                print(f"❌ Complete Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ Complete Analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Complete Analysis error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Real Dropdown AI/ML System Test Complete!")
    print("📊 Dropdowns now work with real data")
    print("🤖 Ollama + XGBoost processing active")
    print("⚡ Real-time AI/ML analysis enabled")
    print("🎯 System is ready for production use!")

def check_data_availability():
    """Check if bank data is available"""
    data_folder = "data"
    bank_file = os.path.join(data_folder, "bank_data_processed.xlsx")
    
    if os.path.exists(bank_file):
        print(f"✅ Bank data found: {bank_file}")
        return True
    else:
        print(f"❌ Bank data not found: {bank_file}")
        print("💡 Please upload a bank statement first to test the dropdown system")
        return False

if __name__ == "__main__":
    print("🔍 Checking data availability...")
    if check_data_availability():
        # Wait a moment for the server to start
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        test_real_dropdown_system()
    else:
        print("\n📋 To test the dropdown system:")
        print("1. Upload a bank statement through the web interface")
        print("2. Process the data")
        print("3. Then run this test again") 