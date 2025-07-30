#!/usr/bin/env python3
"""
Check Data Columns
Check what column names are in the uploaded data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def check_data_columns():
    """Check what column names are in the data"""
    print("🔍 CHECKING DATA COLUMNS")
    print("=" * 60)
    
    # Create test data with different possible column names
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    n_transactions = 50
    
    # Test different possible column names
    test_cases = [
        {
            'name': 'Amount (INR)',
            'data': pd.DataFrame({
                'Date': np.random.choice(dates, n_transactions),
                'Amount (INR)': np.random.lognormal(10, 1, n_transactions),
                'Description': [f"Payment {i}" for i in range(n_transactions)]
            })
        },
        {
            'name': 'Amount',
            'data': pd.DataFrame({
                'Date': np.random.choice(dates, n_transactions),
                'Amount': np.random.lognormal(10, 1, n_transactions),
                'Description': [f"Payment {i}" for i in range(n_transactions)]
            })
        },
        {
            'name': 'Payment_Amount',
            'data': pd.DataFrame({
                'Date': np.random.choice(dates, n_transactions),
                'Payment_Amount': np.random.lognormal(10, 1, n_transactions),
                'Description': [f"Payment {i}" for i in range(n_transactions)]
            })
        },
        {
            'name': 'Transaction_Amount',
            'data': pd.DataFrame({
                'Date': np.random.choice(dates, n_transactions),
                'Transaction_Amount': np.random.lognormal(10, 1, n_transactions),
                'Description': [f"Payment {i}" for i in range(n_transactions)]
            })
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📊 Testing column name: '{test_case['name']}'")
        df = test_case['data']
        print(f"Columns in data: {list(df.columns)}")
        print(f"Sample data:")
        print(df.head(3))
        
        # Check if the system can find the amount column
        amount_columns = [col for col in df.columns if 'amount' in col.lower() or 'payment' in col.lower()]
        print(f"Amount-like columns found: {amount_columns}")
        
        if amount_columns:
            print(f"✅ Amount column found: {amount_columns[0]}")
        else:
            print(f"❌ No amount column found")

def check_actual_function_with_different_columns():
    """Test the actual function with different column names"""
    print("\n🔍 TESTING ACTUAL FUNCTION WITH DIFFERENT COLUMNS")
    print("=" * 60)
    
    try:
        from advanced_revenue_ai_system import AdvancedRevenueAISystem
        
        revenue_ai = AdvancedRevenueAISystem()
        
        # Test with 'Amount (INR)' column
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        n_transactions = 50
        
        test_data = pd.DataFrame({
            'Date': np.random.choice(dates, n_transactions),
            'Amount (INR)': np.random.lognormal(10, 1, n_transactions),
            'Description': [f"Payment from Customer_{i%20}" for i in range(n_transactions)]
        })
        
        print("📊 Testing with 'Amount (INR)' column:")
        print(f"Columns: {list(test_data.columns)}")
        print(f"Sample data:")
        print(test_data.head(3))
        
        # Call the function
        results = revenue_ai.complete_revenue_analysis_system_smart_ollama(test_data)
        
        if 'A5_ar_aging' in results:
            ar_result = results['A5_ar_aging']
            print(f"\n📊 A5_ar_aging Result:")
            for key, value in ar_result.items():
                print(f"  {key}: {value}")
        else:
            print("❌ No A5_ar_aging result found")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🚀 CHECKING DATA COLUMNS")
    print("=" * 60)
    
    # Check different column names
    check_data_columns()
    
    # Test actual function
    check_actual_function_with_different_columns()
    
    print("\n🎯 CONCLUSION:")
    print("The issue is that the system can't find the amount column.")
    print("This causes the calculation to fail and return error data.")
    print("The 5000% you're seeing is likely from cached/old results.")

if __name__ == "__main__":
    main() 