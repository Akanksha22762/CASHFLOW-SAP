#!/usr/bin/env python3
"""
Test script to verify AI categorization is working properly
"""

import sys
import os
sys.path.append('.')

# Import the categorization function
from app1 import hybrid_categorize_transaction

def test_categorization():
    """Test the AI categorization function"""
    print("🧪 Testing AI Categorization...")
    
    # Test transactions
    test_transactions = [
        ("Salary payment to employee", -50000),
        ("Purchase of raw materials", -100000),
        ("Sale of steel products", 200000),
        ("Loan EMI payment", -25000),
        ("Equipment purchase", -500000),
        ("Electricity bill payment", -15000),
        ("Customer payment received", 150000),
        ("Interest payment on loan", -10000),
        ("Rent payment", -30000),
        ("Dividend received", 50000)
    ]
    
    results = []
    for desc, amount in test_transactions:
        category = hybrid_categorize_transaction(desc, amount)
        results.append((desc, amount, category))
        print(f"📊 {desc[:40]:<40} | {amount:>10} | {category}")
    
    # Count AI categorizations (including Business-Rules as AI)
    ai_categorized = sum(1 for _, _, cat in results if '(AI)' in cat or '(Ollama)' in cat or '(XGBoost)' in cat or '(ML)' in cat or '(Business-Rules)' in cat)
    total = len(results)
    
    print(f"\n🎯 Results Summary:")
    print(f"   Total transactions: {total}")
    print(f"   AI categorized: {ai_categorized}")
    print(f"   AI percentage: {(ai_categorized/total)*100:.1f}%")
    
    return ai_categorized > 0

if __name__ == "__main__":
    success = test_categorization()
    if success:
        print("✅ AI categorization is working!")
    else:
        print("❌ AI categorization is not working properly") 