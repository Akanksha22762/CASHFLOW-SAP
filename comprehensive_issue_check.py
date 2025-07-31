#!/usr/bin/env python3
"""
Comprehensive check for all issues with XGBoost + Ollama system
"""

import pandas as pd
import numpy as np

def comprehensive_issue_check():
    """Check all potential issues with the system"""
    print("🔍 COMPREHENSIVE ISSUE CHECK...")
    
    try:
        from app1 import hybrid_categorize_transaction, lightweight_ai, universal_categorize_any_dataset
        from ollama_simple_integration import check_ollama_availability, simple_ollama
        
        print("\n📋 1. SYSTEM STATUS CHECK:")
        print(f"   Ollama Available: {check_ollama_availability()}")
        print(f"   XGBoost Trained: {lightweight_ai.is_trained}")
        
        # Test 1: Check if XGBoost trains with small dataset
        print("\n🧪 2. SMALL DATASET TEST (Issue: XGBoost not training):")
        small_dataset = pd.DataFrame({
            'Description': ['Infrastructure Development', 'Customer Payment', 'Equipment Purchase'],
            'Amount': [1000000, 500000, 2000000],
            'Date': pd.date_range('2024-01-01', periods=3, freq='D'),
            'Type': ['Outward'] * 3
        })
        
        print(f"   Small Dataset: {len(small_dataset)} samples")
        result = universal_categorize_any_dataset(small_dataset)
        print(f"   XGBoost Trained After: {lightweight_ai.is_trained}")
        
        # Test 2: Check Ollama timeout issue
        print("\n🧪 3. OLLAMA TIMEOUT TEST (Issue: Ollama hanging):")
        try:
            test_prompt = "Categorize this transaction: Infrastructure Development"
            result = simple_ollama(test_prompt, "llama2:7b", max_tokens=20)
            print(f"   Ollama Result: {result[:50]}..." if result else "   Ollama Result: None")
            print("   ✅ Ollama working")
        except Exception as e:
            print(f"   ❌ Ollama Error: {e}")
        
        # Test 3: Check hybrid categorization with untrained model
        print("\n🧪 4. HYBRID CATEGORIZATION TEST (Issue: Falling back to rules):")
        # Reset training to simulate untrained state
        lightweight_ai.is_trained = False
        
        test_cases = [
            ("Infrastructure Development", 1000000),
            ("Customer Payment", 500000),
            ("Equipment Purchase", 2000000)
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
        
        # Test 4: Check with large dataset (should work)
        print("\n🧪 5. LARGE DATASET TEST (Should work):")
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
                'Customer Payment - Construction Company'
            ],
            'Amount': [1000000, 500000, 2000000, 1500000, 50000, 800000, 120000, 75000, 1100000, 550000],
            'Date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'Type': ['Outward'] * 10
        })
        
        print(f"   Large Dataset: {len(large_dataset)} samples")
        result = universal_categorize_any_dataset(large_dataset)
        print(f"   XGBoost Trained After: {lightweight_ai.is_trained}")
        
        # Test 5: Check accuracy display
        print("\n🧪 6. ACCURACY DISPLAY TEST:")
        if hasattr(lightweight_ai, 'last_training_accuracy'):
            print(f"   ✅ Actual Accuracy: {lightweight_ai.last_training_accuracy:.1f}%")
        else:
            print("   ❌ No accuracy stored")
        
        # Summary
        print("\n📊 ISSUE SUMMARY:")
        xgboost_count = sum(1 for r in results if '(XGBoost)' in r)
        ollama_count = sum(1 for r in results if '(Ollama)' in r)
        rules_count = sum(1 for r in results if '(Rules)' in r)
        
        print(f"   XGBoost Usage: {xgboost_count}/3")
        print(f"   Ollama Usage: {ollama_count}/3")
        print(f"   Rules Usage: {rules_count}/3")
        
        if rules_count > 0:
            print("   ⚠️ ISSUE: System falling back to rules")
        if xgboost_count > 0:
            print("   ✅ XGBoost working")
        if ollama_count > 0:
            print("   ✅ Ollama working")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    comprehensive_issue_check() 