#!/usr/bin/env python3
"""
Test with large dataset to see if XGBoost trains properly
"""

import pandas as pd
import numpy as np

def test_with_large_dataset():
    """Test with large dataset to see if XGBoost trains"""
    print("🧪 Testing with Large Dataset...")
    
    try:
        from app1 import hybrid_categorize_transaction, lightweight_ai, universal_categorize_any_dataset
        
        # Create a large dataset similar to your bank data
        large_dataset = pd.DataFrame({
            'Description': [
                'Infrastructure Development - Warehouse Construction',
                'Customer Payment - Steel Company',
                'Equipment Purchase - Rolling Mill',
                'Bank Loan - Working Capital',
                'Salary Payment - Employee Payroll',
                'Raw Material Purchase - Iron Ore',
                'Utility Payment - Electricity',
                'Transport Cost - Freight',
                'Infrastructure Development - Plant Expansion',
                'Customer Payment - Construction Company',
                'Equipment Purchase - Quality Testing',
                'Bank Loan - Equipment Finance',
                'Salary Payment - Management',
                'Raw Material Purchase - Coal',
                'Utility Payment - Water',
                'Transport Cost - Logistics',
                'Infrastructure Development - Modernization',
                'Customer Payment - Manufacturing',
                'Equipment Purchase - Furnace Upgrade',
                'Bank Loan - Expansion Project',
                'Infrastructure Development - Warehouse Construction 2',
                'Customer Payment - Steel Company 2',
                'Equipment Purchase - Rolling Mill 2',
                'Bank Loan - Working Capital 2',
                'Salary Payment - Employee Payroll 2',
                'Raw Material Purchase - Iron Ore 2',
                'Utility Payment - Electricity 2',
                'Transport Cost - Freight 2',
                'Infrastructure Development - Plant Expansion 2',
                'Customer Payment - Construction Company 2',
                'Equipment Purchase - Quality Testing 2',
                'Bank Loan - Equipment Finance 2',
                'Salary Payment - Management 2',
                'Raw Material Purchase - Coal 2',
                'Utility Payment - Water 2',
                'Transport Cost - Logistics 2',
                'Infrastructure Development - Modernization 2',
                'Customer Payment - Manufacturing 2',
                'Equipment Purchase - Furnace Upgrade 2',
                'Bank Loan - Expansion Project 2'
            ],
            'Amount': [1000000, 500000, 2000000, 1500000, 50000, 800000, 120000, 75000, 1100000, 550000, 2100000, 1600000, 60000, 850000, 130000, 80000, 900000, 650000, 1800000, 1400000, 1200000, 600000, 2200000, 1700000, 70000, 900000, 140000, 85000, 1300000, 650000, 2300000, 1800000, 80000, 950000, 150000, 90000, 1100000, 750000, 2000000, 1600000],
            'Date': pd.date_range('2024-01-01', periods=40, freq='D'),
            'Type': ['Outward'] * 40
        })
        
        print(f"📊 Large Dataset: {len(large_dataset)} samples")
        
        # Process with universal categorization
        print("\n🤖 Processing with Universal Categorization...")
        result_df = universal_categorize_any_dataset(large_dataset)
        
        # Check if XGBoost is trained
        print(f"\n📋 XGBoost Training Status:")
        print(f"   Is Trained: {lightweight_ai.is_trained}")
        
        # Test categorization after processing
        print(f"\n🧪 Testing Categorization After Processing:")
        test_cases = [
            ("Infrastructure Development", 1000000),
            ("Customer Payment", 500000),
            ("Equipment Purchase", 2000000),
            ("Bank Loan", 1500000),
            ("Salary Payment", 50000)
        ]
        
        results = []
        for desc, amount in test_cases:
            result = hybrid_categorize_transaction(desc, amount)
            results.append(result)
            print(f"   '{desc}' → {result}")
            
            if '(XGBoost)' in result:
                print("   ✅ Using XGBoost")
            elif '(Ollama)' in result:
                print("   ✅ Using Ollama")
            elif '(Rules)' in result:
                print("   ⚠️ Using Rules")
            else:
                print("   ❌ Unknown method")
        
        # Calculate statistics
        xgboost_count = sum(1 for r in results if '(XGBoost)' in r)
        ollama_count = sum(1 for r in results if '(Ollama)' in r)
        rules_count = sum(1 for r in results if '(Rules)' in r)
        total = len(results)
        
        print(f"\n📊 Final Results:")
        print(f"   XGBoost: {xgboost_count}/{total} ({xgboost_count/total*100:.1f}%)")
        print(f"   Ollama: {ollama_count}/{total} ({ollama_count/total*100:.1f}%)")
        print(f"   Rules: {rules_count}/{total} ({rules_count/total*100:.1f}%)")
        
        if xgboost_count > 0:
            print("\n🎉 SUCCESS: XGBoost is working!")
        else:
            print("\n❌ ISSUE: Still using only rules")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_large_dataset() 