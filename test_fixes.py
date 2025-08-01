#!/usr/bin/env python3
"""
Test Fixes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_revenue_ai_system import AdvancedRevenueAISystem

def create_test_data():
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(50)]
    test_data = []
    
    for i, date in enumerate(dates):
        # Make some transactions negative (expenses)
        amount = np.random.uniform(1000, 50000)
        if i % 3 == 0:  # Every third transaction is an expense
            amount = -amount
        
        test_data.append({
            'Date': date,
            'Description': f'Transaction {i+1}',
            'Amount': amount,
            'Type': 'Credit' if amount > 0 else 'Debit'
        })
    
    return pd.DataFrame(test_data)

# Test the fixes
ai_system = AdvancedRevenueAISystem()
test_data = create_test_data()

print("Testing expense analysis...")
result = ai_system.analyze_operating_expenses(test_data)
print(f"Result: {result}")

print("\nTesting enhanced expense analysis...")
enhanced_result = ai_system.enhanced_analyze_operating_expenses(test_data)
print(f"Enhanced result: {enhanced_result}") 