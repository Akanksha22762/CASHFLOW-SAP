#!/usr/bin/env python3
"""
Test Final Output - Verify the output is working correctly
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_revenue_ai_system import AdvancedRevenueAISystem

# Create simple test data
dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
test_data = []

for i in range(30):
    if i % 3 == 0:
        test_data.append({
            'Date': dates[i],
            'Description': f'Expense {i+1} - Salary Payment',
            'Amount': -np.random.uniform(1000, 10000),
            'Type': 'Debit'
        })
    else:
        test_data.append({
            'Date': dates[i],
            'Description': f'Revenue {i+1} - Steel Sales',
            'Amount': np.random.uniform(5000, 50000),
            'Type': 'Credit'
        })

df = pd.DataFrame(test_data)

print("🎯 TESTING FINAL OUTPUT")
print("=" * 50)

ai_system = AdvancedRevenueAISystem()

# Test A1 - Historical Revenue Trends
print("\n1. Testing A1 - Historical Revenue Trends")
print("-" * 40)

try:
    result = ai_system.enhanced_analyze_historical_revenue_trends(df)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Success: {len(result)} metrics")
        
        # Show key metrics
        key_metrics = ['total_revenue', 'revenue_count', 'avg_revenue', 'revenue_growth_rate']
        print("📊 Key Metrics:")
        for metric in key_metrics:
            if metric in result:
                print(f"   {metric}: {result[metric]}")
        
        # Show advanced AI features
        if 'advanced_ai_features' in result:
            ai_features = result['advanced_ai_features']
            print(f"🤖 Advanced AI Features: {len(ai_features)}")
            for feature in ai_features:
                print(f"   - {feature}")
        else:
            print("❌ No advanced AI features")
            
        print(f"📋 Analysis Type: {result.get('analysis_type', 'Unknown')}")
        
except Exception as e:
    print(f"❌ Exception: {e}")

# Test A6 - Operating Expenses
print("\n2. Testing A6 - Operating Expenses")
print("-" * 40)

try:
    result = ai_system.enhanced_analyze_operating_expenses(df)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Success: {len(result)} metrics")
        
        # Show key metrics
        key_metrics = ['total_expenses', 'expense_count', 'avg_expense', 'expense_efficiency_score']
        print("📊 Key Metrics:")
        for metric in key_metrics:
            if metric in result:
                print(f"   {metric}: {result[metric]}")
        
        # Show advanced AI features
        if 'advanced_ai_features' in result:
            ai_features = result['advanced_ai_features']
            print(f"🤖 Advanced AI Features: {len(ai_features)}")
            for feature in ai_features:
                print(f"   - {feature}")
        else:
            print("❌ No advanced AI features")
            
        print(f"📋 Analysis Type: {result.get('analysis_type', 'Unknown')}")
        
except Exception as e:
    print(f"❌ Exception: {e}")

print("\n" + "=" * 50)
print("🎯 FINAL TEST COMPLETE")
print("✅ Check if the output cards are now showing correctly!") 