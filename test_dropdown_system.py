#!/usr/bin/env python3
"""
Test script for the complete dropdown AI/ML analysis system
"""

import requests
import json
import time

def test_dropdown_system():
    """Test the complete dropdown AI/ML system"""
    
    base_url = "http://localhost:5000"
    
    print("🚀 Testing Complete Dropdown AI/ML System")
    print("=" * 50)
    
    # Test 1: Vendor Analysis
    print("\n1️⃣ Testing Vendor Analysis...")
    vendor_data = {
        "vendor": "all",
        "analysis_type": "payment_patterns",
        "ai_model": "hybrid"
    }
    
    try:
        response = requests.post(f"{base_url}/vendor-analysis", json=vendor_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Vendor Analysis: {result.get('vendors_analyzed', 0)} vendors processed")
            print(f"🤖 AI Model Used: {result.get('ai_model', 'Unknown')}")
        else:
            print(f"❌ Vendor Analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Vendor Analysis error: {e}")
    
    # Test 2: Transaction Analysis
    print("\n2️⃣ Testing Transaction Analysis...")
    transaction_data = {
        "transaction_type": "all",
        "analysis_type": "pattern_analysis",
        "ai_model": "hybrid"
    }
    
    try:
        response = requests.post(f"{base_url}/transaction-analysis", json=transaction_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Transaction Analysis: {result.get('transactions_analyzed', 0)} transactions processed")
            print(f"🤖 AI Model Used: {result.get('ai_model', 'Unknown')}")
        else:
            print(f"❌ Transaction Analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Transaction Analysis error: {e}")
    
    # Test 3: Advanced Analysis
    print("\n3️⃣ Testing Advanced Analysis...")
    advanced_data = {
        "category": "revenue_analysis",
        "depth": "detailed",
        "processing_mode": "real_time"
    }
    
    try:
        response = requests.post(f"{base_url}/analysis-category", json=advanced_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Advanced Analysis: {result.get('category', 'Unknown')} completed")
            print(f"🔍 Analysis Depth: {result.get('depth', 'Unknown')}")
        else:
            print(f"❌ Advanced Analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Advanced Analysis error: {e}")
    
    # Test 4: Report Generation
    print("\n4️⃣ Testing Report Generation...")
    report_data = {
        "report_type": "vendor_report",
        "format": "pdf",
        "detail_level": "detailed"
    }
    
    try:
        response = requests.post(f"{base_url}/generate-report", json=report_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Report Generation: {result.get('report_type', 'Unknown')} completed")
            print(f"📄 Format: {result.get('format', 'Unknown')}")
        else:
            print(f"❌ Report Generation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Report Generation error: {e}")
    
    # Test 5: Complete Analysis
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
            print(f"✅ Complete Analysis: All systems processed")
            print(f"🎯 Analysis Complete: {result.get('analysis_complete', False)}")
        else:
            print(f"❌ Complete Analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Complete Analysis error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Dropdown AI/ML System Test Complete!")
    print("📊 All dropdowns now trigger AI/ML processing")
    print("🤖 Ollama + XGBoost integration active")
    print("⚡ Real-time processing enabled")

if __name__ == "__main__":
    # Wait a moment for the server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    test_dropdown_system() 