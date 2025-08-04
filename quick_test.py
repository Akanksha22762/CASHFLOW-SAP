#!/usr/bin/env python3
"""
Quick Test - Check what's actually working
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_revenue_ai_system import AdvancedRevenueAISystem

def create_simple_test_data():
    """Create simple test data"""
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
    
    return pd.DataFrame(test_data)

# Test
ai_system = AdvancedRevenueAISystem()
test_data = create_simple_test_data()

print("Testing A1 - Historical Revenue Trends")
result1 = ai_system.analyze_historical_revenue_trends(test_data)
print(f"Result: {result1}")

print("\nTesting A6 - Operating Expenses")
result6 = ai_system.analyze_operating_expenses(test_data)
print(f"Result: {result6}")

print("\nTesting A9 - Loan Repayments")
result9 = ai_system.analyze_loan_repayments(test_data)
print(f"Result: {result9}") 