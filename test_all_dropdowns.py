#!/usr/bin/env python3
"""
Comprehensive Test for All Dropdowns and Analysis Functions
Tests every dropdown option and analysis type to ensure they work correctly
"""

import requests
import json
import pandas as pd
import os
from datetime import datetime

def test_server_connection():
    """Test if server is running"""
    try:
        response = requests.get('http://localhost:5000/')
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_dropdown_data():
    """Test dropdown data endpoint"""
    try:
        response = requests.get('http://localhost:5000/get-dropdown-data')
        if response.status_code == 200:
            data = response.json()
            print("✅ Dropdown data endpoint working")
            print(f"📊 Available vendors: {len(data.get('vendors', []))}")
            return data
        else:
            print(f"❌ Dropdown data endpoint failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Dropdown data test failed: {e}")
        return None

def test_transaction_analysis_dropdowns():
    """Test all transaction analysis dropdown combinations"""
    print("\n🔍 TESTING TRANSACTION ANALYSIS DROPDOWNS")
    print("=" * 60)
    
    # Test transaction types
    transaction_types = [
        "Operating Activities (XGBoost)",
        "Investing Activities (XGBoost)", 
        "Financing Activities (XGBoost)",
        "all"
    ]
    
    # Test analysis types
    analysis_types = [
        "pattern_analysis",
        "trend_analysis", 
        "cash_flow",
        "anomaly_detection",
        "predictive"
    ]
    
    # Test AI models
    ai_models = [
        "hybrid",
        "ollama",
        "xgboost"
    ]
    
    successful_tests = 0
    total_tests = 0
    
    for transaction_type in transaction_types:
        for analysis_type in analysis_types:
            for ai_model in ai_models:
                total_tests += 1
                print(f"\n📊 Testing: {transaction_type} | {analysis_type} | {ai_model}")
                
                try:
                    # Test transaction analysis
                    payload = {
                        'transaction_type': transaction_type,
                        'analysis_type': analysis_type,
                        'ai_model': ai_model
                    }
                    
                    response = requests.post(
                        'http://localhost:5000/transaction-analysis',
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            data = result.get('data', {})
                            transaction_count = data.get('transaction_count', 'N/A')
                            total_amount = data.get('total_amount', 0)
                            
                            print(f"  ✅ SUCCESS: {transaction_count} transactions, ₹{total_amount:,.2f}")
                            
                            # Check for specific issues
                            if transaction_count == 'N/A' or total_amount == 0:
                                print(f"  ⚠️ WARNING: No data found for {transaction_type}")
                            else:
                                successful_tests += 1
                        else:
                            print(f"  ❌ FAILED: {result.get('error', 'Unknown error')}")
                    else:
                        print(f"  ❌ HTTP ERROR: {response.status_code}")
                        
                except Exception as e:
                    print(f"  ❌ EXCEPTION: {str(e)[:50]}")
    
    print(f"\n📊 TRANSACTION ANALYSIS RESULTS:")
    print(f"  Successful: {successful_tests}/{total_tests}")
    print(f"  Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    return successful_tests > 0

def test_vendor_analysis_dropdowns():
    """Test all vendor analysis dropdown combinations"""
    print("\n🔍 TESTING VENDOR ANALYSIS DROPDOWNS")
    print("=" * 60)
    
    # Get available vendors first
    dropdown_data = test_dropdown_data()
    if not dropdown_data:
        print("❌ Cannot test vendor analysis without dropdown data")
        return False
    
    vendors = dropdown_data.get('vendors', [])
    if not vendors:
        print("❌ No vendors available for testing")
        return False
    
    # Test analysis types
    analysis_types = [
        "payment_patterns",
        "risk_assessment", 
        "cash_flow",
        "predictive"
    ]
    
    # Test AI models
    ai_models = [
        "hybrid",
        "ollama",
        "xgboost"
    ]
    
    successful_tests = 0
    total_tests = 0
    
    # Test with first few vendors
    test_vendors = vendors[:3]  # Test first 3 vendors
    
    for vendor in test_vendors:
        for analysis_type in analysis_types:
            for ai_model in ai_models:
                total_tests += 1
                print(f"\n🏢 Testing: {vendor} | {analysis_type} | {ai_model}")
                
                try:
                    payload = {
                        'vendor': vendor,
                        'analysis_type': analysis_type,
                        'ai_model': ai_model
                    }
                    
                    response = requests.post(
                        'http://localhost:5000/vendor-analysis',
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            data = result.get('data', {})
                            transaction_count = data.get('transactions_count', 'N/A')
                            total_amount = data.get('total_amount', 0)
                            
                            print(f"  ✅ SUCCESS: {transaction_count} transactions, ₹{total_amount:,.2f}")
                            
                            if transaction_count == 'N/A' or total_amount == 0:
                                print(f"  ⚠️ WARNING: No data found for {vendor}")
                            else:
                                successful_tests += 1
                        else:
                            print(f"  ❌ FAILED: {result.get('error', 'Unknown error')}")
                    else:
                        print(f"  ❌ HTTP ERROR: {response.status_code}")
                        
                except Exception as e:
                    print(f"  ❌ EXCEPTION: {str(e)[:50]}")
    
    print(f"\n📊 VENDOR ANALYSIS RESULTS:")
    print(f"  Successful: {successful_tests}/{total_tests}")
    print(f"  Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    return successful_tests > 0

def test_data_availability():
    """Test if required data files exist"""
    print("\n🔍 TESTING DATA AVAILABILITY")
    print("=" * 60)
    
    required_files = [
        'data/bank_data_processed.xlsx',
        'data/sap_data_processed.xlsx'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_excel(file_path)
                print(f"✅ {file_path}: {len(df)} rows, {len(df.columns)} columns")
                
                # Check for required columns
                if 'Amount' in df.columns:
                    total_amount = df['Amount'].sum()
                    print(f"  📊 Total Amount: ₹{total_amount:,.2f}")
                else:
                    print(f"  ❌ Missing 'Amount' column")
                    
            except Exception as e:
                print(f"❌ {file_path}: Error reading file - {e}")
        else:
            print(f"❌ {file_path}: File not found")

def test_specific_issues():
    """Test specific issues found in the image"""
    print("\n🔍 TESTING SPECIFIC ISSUES")
    print("=" * 60)
    
    # Test the specific case from the image
    print("📊 Testing cash_flow analysis with hybrid model...")
    
    try:
        payload = {
            'transaction_type': 'Investing Activities (XGBoost)',
            'analysis_type': 'cash_flow',
            'ai_model': 'hybrid'
        }
        
        response = requests.post(
            'http://localhost:5000/transaction-analysis',
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response received")
            
            if result.get('success'):
                data = result.get('data', {})
                print(f"📊 Analysis Results:")
                print(f"  AI Model: {data.get('ai_model', 'N/A')}")
                print(f"  Analysis Type: {data.get('analysis_type', 'N/A')}")
                print(f"  Transaction Count: {data.get('transaction_count', 'N/A')}")
                print(f"  Total Amount: ₹{data.get('total_amount', 0):,.2f}")
                print(f"  Average Amount: ₹{data.get('avg_amount', 0):,.2f}")
                print(f"  Max Amount: ₹{data.get('max_amount', 0):,.2f}")
                print(f"  Min Amount: ₹{data.get('min_amount', 0):,.2f}")
                
                # Check for patterns
                patterns = data.get('patterns', {})
                if patterns:
                    print(f"  Patterns: {patterns}")
                else:
                    print(f"  ⚠️ No patterns found")
                    
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    """Run comprehensive dropdown tests"""
    print("🔍 COMPREHENSIVE DROPDOWN TESTING")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    # Test server connection
    if not test_server_connection():
        print("❌ Cannot proceed without server connection")
        return
    
    # Test data availability
    test_data_availability()
    
    # Test dropdown data
    dropdown_data = test_dropdown_data()
    
    # Test transaction analysis
    transaction_success = test_transaction_analysis_dropdowns()
    
    # Test vendor analysis
    vendor_success = test_vendor_analysis_dropdowns()
    
    # Test specific issues
    test_specific_issues()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    print(f"✅ Server Connection: Working")
    print(f"✅ Dropdown Data: {'Available' if dropdown_data else 'Failed'}")
    print(f"✅ Transaction Analysis: {'Working' if transaction_success else 'Issues Found'}")
    print(f"✅ Vendor Analysis: {'Working' if vendor_success else 'Issues Found'}")
    
    print(f"\nTest completed at: {datetime.now()}")

if __name__ == "__main__":
    main() 