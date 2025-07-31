#!/usr/bin/env python3
"""
Debug hybrid system to see why it's not using Ollama + XGBoost
"""

import pandas as pd
import numpy as np

def debug_hybrid_system():
    """Debug the hybrid system"""
    print("🔍 Debugging Hybrid System...")
    
    try:
        from app1 import hybrid_categorize_transaction, lightweight_ai
        from ollama_simple_integration import check_ollama_availability, simple_ollama
        
        print("📋 System Status:")
        print(f"   Ollama Available: {check_ollama_availability()}")
        print(f"   XGBoost Trained: {lightweight_ai.is_trained}")
        
        # Test Ollama directly
        print("\n🧪 Testing Ollama Directly:")
        try:
            test_prompt = "Categorize this transaction: Infrastructure Development"
            result = simple_ollama(test_prompt, "llama2:7b", max_tokens=20)
            print(f"   Ollama Result: {result}")
        except Exception as e:
            print(f"   Ollama Error: {e}")
        
        # Test XGBoost training
        print("\n🧪 Testing XGBoost Training:")
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
        print(f"   Training Result: {training_result}")
        print(f"   Is Trained After: {lightweight_ai.is_trained}")
        
        # Test hybrid categorization
        print("\n🧪 Testing Hybrid Categorization:")
        test_cases = [
            ("Infrastructure Development", 1000000),
            ("Customer Payment", 500000),
            ("Equipment Purchase", 2000000)
        ]
        
        for desc, amount in test_cases:
            result = hybrid_categorize_transaction(desc, amount)
            print(f"   '{desc}' → {result}")
            
            if '(XGBoost)' in result:
                print("   ✅ Using XGBoost")
            elif '(Ollama)' in result:
                print("   ✅ Using Ollama")
            elif '(Rules)' in result:
                print("   ⚠️ Using Rules")
            else:
                print("   ❌ Unknown method")
        
        # Test with larger dataset
        print("\n🧪 Testing with Larger Dataset:")
        large_test_data = pd.DataFrame({
            'Description': [
                'Infrastructure Development',
                'Customer Payment',
                'Equipment Purchase',
                'Bank Loan',
                'Salary Payment',
                'Raw Material Purchase',
                'Utility Payment',
                'Transport Cost',
                'Infrastructure Development 2',
                'Customer Payment 2'
            ],
            'Amount': [1000000, 500000, 2000000, 1500000, 50000, 800000, 120000, 75000, 1100000, 550000],
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
                'Operating Activities'
            ]
        })
        
        training_result = lightweight_ai.train_transaction_classifier(large_test_data)
        print(f"   Training Result: {training_result}")
        print(f"   Is Trained After: {lightweight_ai.is_trained}")
        
        # Test categorization after training
        print("\n🧪 Testing After Training:")
        for desc, amount in test_cases:
            result = hybrid_categorize_transaction(desc, amount)
            print(f"   '{desc}' → {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_hybrid_system() 