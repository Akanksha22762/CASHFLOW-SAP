#!/usr/bin/env python3
"""
Test script to verify ML usage fix
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Import the fixed functions
from app1 import lightweight_ai, hybrid_categorize_transaction

def test_ml_usage_fix():
    """Test that ML models are being used properly"""
    print("🧪 Testing ML Usage Fix...")
    
    # Create test data
    test_descriptions = [
        "Infrastructure Development - Warehouse Construction - 4717 sq ft",
        "VIP Customer Payment - Premium Service",
        "Equipment Purchase - New Production Line",
        "Bank Loan Disbursement - Working Capital",
        "Salary Payment - Employee Wages",
        "Utility Bill Payment - Electricity",
        "Marketing Campaign - Digital Advertising",
        "Interest Payment - Bank Loan",
        "Raw Material Purchase - Steel Components",
        "Dividend Payment - Shareholder Return"
    ]
    
    test_amounts = [1000000, 500000, 2000000, 1500000, 50000, 10000, 25000, 15000, 300000, 100000]
    
    print("📊 Testing categorization with ML models...")
    
    # Test each transaction
    results = []
    for i, (desc, amt) in enumerate(zip(test_descriptions, test_amounts)):
        result = hybrid_categorize_transaction(desc, amt)
        results.append(result)
        print(f"   {i+1}. {desc[:40]}... → {result}")
    
    # Calculate statistics
    ml_count = sum(1 for cat in results if '(XGBoost)' in cat or '(ML)' in cat)
    ollama_count = sum(1 for cat in results if '(Ollama)' in cat)
    rules_count = sum(1 for cat in results if '(Rules)' in cat)
    total_transactions = len(results)
    
    print(f"\n📈 ML Usage Statistics:")
    print(f"   ML Models (XGBoost): {ml_count}/{total_transactions} ({ml_count/total_transactions*100:.1f}%)")
    print(f"   Ollama AI: {ollama_count}/{total_transactions} ({ollama_count/total_transactions*100:.1f}%)")
    print(f"   Rule-based: {rules_count}/{total_transactions} ({rules_count/total_transactions*100:.1f}%)")
    print(f"   Total AI/ML Usage: {ml_count + ollama_count}/{total_transactions} ({(ml_count + ollama_count)/total_transactions*100:.1f}%)")
    
    # Check if ML models are being used
    if ml_count > 0:
        print("✅ SUCCESS: ML models are being used!")
    else:
        print("❌ FAILURE: ML models are not being used")
    
    # Check if training is working
    print(f"\n🔍 ML Training Status:")
    print(f"   Is trained: {lightweight_ai.is_trained}")
    print(f"   Models available: {list(lightweight_ai.models.keys())}")
    
    return ml_count > 0

if __name__ == "__main__":
    success = test_ml_usage_fix()
    if success:
        print("\n🎉 ML Usage Fix Test PASSED!")
    else:
        print("\n❌ ML Usage Fix Test FAILED!") 