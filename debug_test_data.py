#!/usr/bin/env python3
"""
Debug Test Data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

# Create and analyze test data
test_data = create_test_data()
print(f"Total transactions: {len(test_data)}")
print(f"Positive amounts: {len(test_data[test_data['Amount'] > 0])}")
print(f"Negative amounts: {len(test_data[test_data['Amount'] < 0])}")
print(f"Sample data:")
print(test_data.head(10))
print(f"\nAmount column: {test_data.columns.tolist()}")
print(f"Amount range: {test_data['Amount'].min()} to {test_data['Amount'].max()}") 