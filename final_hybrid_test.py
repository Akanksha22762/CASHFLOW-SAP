#!/usr/bin/env python3
"""
Final test to show hybrid system working with large datasets
"""

import pandas as pd
import numpy as np

def final_hybrid_test():
    """Final test showing hybrid system with large dataset"""
    print("🎯 Final Hybrid System Test with Large Dataset...")
    
    try:
        from app1 import hybrid_categorize_transaction, lightweight_ai
        
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
                'Bank Loan - Expansion Project'
            ],
            'Amount': [1000000, 500000, 2000000, 1500000, 50000, 800000, 120000, 75000, 1100000, 550000, 2100000, 1600000, 60000, 850000, 130000, 80000, 900000, 650000, 1800000, 1400000],
            'Category': [
                'Investing Activities',
                'Operating Activities', 
                'Investing Activities',
                'Financing Activities',
                'Operating Activities',
                'Operating Activities',
                'Operating Activities',
                'Operating Activities',
                'Investing Activities',
                'Operating Activities', 
                'Investing Activities',
                'Financing Activities',
                'Operating Activities',
                'Operating Activities',
                'Operating Activities',
                'Operating Activities',
                'Investing Activities',
                'Operating Activities', 
                'Investing Activities',
                'Financing Activities'
            ]
        })
        
        print(f"📊 Large Dataset: {len(large_dataset)} samples")
        
        # Train XGBoost with large dataset
        print("\n🤖 Training XGBoost with Large Dataset...")
        training_result = lightweight_ai.train_transaction_classifier(large_dataset)
        print(f"   Training Result: {training_result}")
        print(f"   Is Trained: {lightweight_ai.is_trained}")
        
        # Test hybrid categorization
        print("\n🧪 Testing Hybrid Categorization:")
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
        print(f"   Total AI/ML: {xgboost_count + ollama_count}/{total} ({(xgboost_count + ollama_count)/total*100:.1f}%)")
        
        if xgboost_count > 0 or ollama_count > 0:
            print("\n🎉 SUCCESS: Hybrid system is working!")
            print("✅ Your system will use XGBoost + Ollama with large datasets!")
        else:
            print("\n❌ ISSUE: Still using only rules")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    final_hybrid_test() 