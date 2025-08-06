#!/usr/bin/env python3
"""
Test script to verify AI categorization with large dataset simulation
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append('.')

# Import the categorization function
from app1 import universal_categorize_any_dataset, hybrid_categorize_transaction

def create_large_test_dataset():
    """Create a large test dataset to simulate real usage"""
    print("📊 Creating large test dataset...")
    
    # Generate diverse transaction data
    np.random.seed(42)  # For reproducible results
    
    # Transaction types and their amounts
    transaction_types = [
        ("Salary payment", -50000, "Operating Activities"),
        ("Raw material purchase", -100000, "Operating Activities"),
        ("Steel product sale", 200000, "Operating Activities"),
        ("Loan EMI payment", -25000, "Financing Activities"),
        ("Equipment purchase", -500000, "Investing Activities"),
        ("Electricity bill", -15000, "Operating Activities"),
        ("Customer payment", 150000, "Operating Activities"),
        ("Interest payment", -10000, "Financing Activities"),
        ("Rent payment", -30000, "Operating Activities"),
        ("Dividend received", 50000, "Financing Activities"),
        ("Vendor payment", -75000, "Operating Activities"),
        ("Service income", 80000, "Operating Activities"),
        ("Tax payment", -20000, "Operating Activities"),
        ("Asset sale", 300000, "Investing Activities"),
        ("Loan disbursement", 1000000, "Financing Activities"),
        ("Utility payment", -12000, "Operating Activities"),
        ("Commission earned", 25000, "Operating Activities"),
        ("Insurance premium", -18000, "Operating Activities"),
        ("Investment income", 40000, "Investing Activities"),
        ("Maintenance expense", -22000, "Operating Activities")
    ]
    
    # Create 500 transactions (large dataset)
    transactions = []
    for i in range(500):
        # Randomly select transaction type
        desc, amount, category = transaction_types[np.random.randint(0, len(transaction_types))]
        
        # Add some variation to amounts
        variation = np.random.uniform(0.8, 1.2)
        final_amount = int(amount * variation)
        
        # Add some variation to descriptions
        variations = [
            f"{desc} #{i+1}",
            f"{desc} - Transaction {i+1}",
            f"{desc} (Batch {i//10 + 1})",
            f"{desc} - {np.random.choice(['Regular', 'Special', 'Bulk', 'Standard'])}",
            f"{desc} - {np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'])}"
        ]
        final_desc = np.random.choice(variations)
        
        transactions.append({
            'Description': final_desc,
            'Amount': final_amount,
            'Date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)),
            'Type': 'DEBIT' if final_amount < 0 else 'CREDIT'
        })
    
    df = pd.DataFrame(transactions)
    print(f"✅ Created dataset with {len(df)} transactions")
    return df

def test_large_dataset_categorization():
    """Test AI categorization with large dataset"""
    print("🧪 Testing AI Categorization with Large Dataset...")
    
    # Create large test dataset
    df = create_large_test_dataset()
    
    # Test the universal categorization function
    print("🤖 Applying universal AI/ML categorization...")
    df_categorized = universal_categorize_any_dataset(df)
    
    # Analyze results
    if 'Category' in df_categorized.columns:
        categories = df_categorized['Category'].tolist()
        
        # Count different types of categorization
        ai_categorized = sum(1 for cat in categories if '(AI)' in cat or '(Ollama)' in cat or '(XGBoost)' in cat or '(ML)' in cat or '(Business-Rules)' in cat)
        xgboost_categorized = sum(1 for cat in categories if '(XGBoost)' in cat or '(ML)' in cat)
        ollama_categorized = sum(1 for cat in categories if '(Ollama)' in cat)
        business_rules_categorized = sum(1 for cat in categories if '(Business-Rules)' in cat)
        total = len(categories)
        
        print(f"\n🎯 Large Dataset Results Summary:")
        print(f"   Total transactions: {total}")
        print(f"   AI categorized: {ai_categorized} ({(ai_categorized/total)*100:.1f}%)")
        print(f"   XGBoost categorized: {xgboost_categorized} ({(xgboost_categorized/total)*100:.1f}%)")
        print(f"   Ollama categorized: {ollama_categorized} ({(ollama_categorized/total)*100:.1f}%)")
        print(f"   Business rules categorized: {business_rules_categorized} ({(business_rules_categorized/total)*100:.1f}%)")
        
        # Show category distribution
        category_counts = pd.Series(categories).value_counts()
        print(f"\n📊 Category Distribution:")
        for cat, count in category_counts.head(10).items():
            print(f"   {cat}: {count} transactions")
        
        return ai_categorized > 0
    else:
        print("❌ Category column not found after categorization")
        return False

if __name__ == "__main__":
    success = test_large_dataset_categorization()
    if success:
        print("✅ AI categorization is working with large dataset!")
    else:
        print("❌ AI categorization is not working properly with large dataset") 