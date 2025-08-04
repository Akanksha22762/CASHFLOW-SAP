#!/usr/bin/env python3
"""
Simple Test - Check basic functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_revenue_ai_system import AdvancedRevenueAISystem

# Create simple test data
dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
test_data = []

# Mix of revenue and expenses
for i in range(30):
    if i % 3 == 0:  # Every third is an expense
        test_data.append({
            'Date': dates[i],
            'Description': f'Expense {i+1}',
            'Amount': -np.random.uniform(1000, 10000),
            'Type': 'Debit'
        })
    else:
        test_data.append({
            'Date': dates[i],
            'Description': f'Revenue {i+1}',
            'Amount': np.random.uniform(5000, 50000),
            'Type': 'Credit'
        })

df = pd.DataFrame(test_data)

# Test
ai_system = AdvancedRevenueAISystem()

print("Testing A6 - Operating Expenses")
try:
    result6 = ai_system.analyze_operating_expenses(df)
    print(f"Result: {result6}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting A7 - Accounts Payable")
try:
    result7 = ai_system.analyze_accounts_payable_terms(df)
    print(f"Result: {result7}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting A8 - Inventory Turnover")
try:
    result8 = ai_system.analyze_inventory_turnover(df)
    print(f"Result: {result8}")
except Exception as e:
    print(f"Error: {e}") 