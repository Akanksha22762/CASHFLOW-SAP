#!/usr/bin/env python3
"""
Debug Output - See what's actually happening
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

print("🔍 DEBUGGING OUTPUT")
print("=" * 50)

ai_system = AdvancedRevenueAISystem()

# Test A1 - Historical Revenue Trends
print("\n1. Testing A1 - Historical Revenue Trends")
print("-" * 40)

try:
    result = ai_system.enhanced_analyze_historical_revenue_trends(df)
    print(f"✅ Function executed successfully")
    print(f"📊 Result type: {type(result)}")
    print(f"📊 Result length: {len(result) if isinstance(result, dict) else 'N/A'}")
    
    if isinstance(result, dict):
        print("📋 Keys in result:")
        for key in result.keys():
            print(f"   - {key}")
        
        if 'advanced_ai_features' in result:
            ai_features = result['advanced_ai_features']
            print(f"🤖 Advanced AI Features: {len(ai_features)}")
            for feature in ai_features:
                print(f"   - {feature}")
        else:
            print("❌ No advanced_ai_features found")
    else:
        print(f"❌ Result is not a dict: {result}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test A6 - Operating Expenses
print("\n2. Testing A6 - Operating Expenses")
print("-" * 40)

try:
    result = ai_system.enhanced_analyze_operating_expenses(df)
    print(f"✅ Function executed successfully")
    print(f"📊 Result type: {type(result)}")
    print(f"📊 Result length: {len(result) if isinstance(result, dict) else 'N/A'}")
    
    if isinstance(result, dict):
        print("📋 Keys in result:")
        for key in result.keys():
            print(f"   - {key}")
        
        if 'advanced_ai_features' in result:
            ai_features = result['advanced_ai_features']
            print(f"🤖 Advanced AI Features: {len(ai_features)}")
            for feature in ai_features:
                print(f"   - {feature}")
        else:
            print("❌ No advanced_ai_features found")
    else:
        print(f"❌ Result is not a dict: {result}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("�� DEBUG COMPLETE") 