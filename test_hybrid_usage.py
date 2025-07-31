#!/usr/bin/env python3
"""
Test to check why hybrid system is not being used
"""

import pandas as pd
import numpy as np

def test_hybrid_usage():
    """Test why hybrid system is not being used"""
    print("🧪 Testing Hybrid System Usage...")
    
    try:
        from app1 import hybrid_categorize_transaction, lightweight_ai
        
        # Test cases
        test_cases = [
            ("Infrastructure Development", 1000000),
            ("Customer Payment", 500000),
            ("Equipment Purchase", 2000000),
            ("Bank Loan", 1500000),
            ("Salary Payment", 50000)
        ]
        
        print("📋 Testing Hybrid Categorization:")
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
        
        print(f"\n📊 Results:")
        print(f"   XGBoost: {xgboost_count}/{total} ({xgboost_count/total*100:.1f}%)")
        print(f"   Ollama: {ollama_count}/{total} ({ollama_count/total*100:.1f}%)")
        print(f"   Rules: {rules_count}/{total} ({rules_count/total*100:.1f}%)")
        
        # Check if lightweight_ai is trained
        print(f"\n🔍 Lightweight AI Status:")
        print(f"   Is trained: {lightweight_ai.is_trained}")
        if hasattr(lightweight_ai, 'models'):
            print(f"   Models: {list(lightweight_ai.models.keys())}")
        
        # Test with a small dataset to see if training works
        print(f"\n🧪 Testing Training with Small Dataset:")
        test_data = pd.DataFrame({
            'Description': [
                'Infrastructure Development',
                'Customer Payment',
                'Equipment Purchase',
                'Bank Loan',
                'Salary Payment'
            ],
            'Amount': [1000000, 500000, 2000000, 1500000, 50000],
            'Category': [
                'Investing Activities',
                'Operating Activities', 
                'Investing Activities',
                'Financing Activities',
                'Operating Activities'
            ]
        })
        
        training_result = lightweight_ai.train_transaction_classifier(test_data)
        print(f"   Training result: {training_result}")
        print(f"   Is trained after: {lightweight_ai.is_trained}")
        
        # Test categorization again after training
        print(f"\n🧪 Testing After Training:")
        for desc, amount in test_cases[:3]:
            result = hybrid_categorize_transaction(desc, amount)
            print(f"   '{desc}' → {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hybrid_usage() 