#!/usr/bin/env python3
"""
Test script that simulates the exact Flask app scenario
"""

import pandas as pd
import numpy as np
from advanced_revenue_ai_system import AdvancedRevenueAISystem

def test_flask_scenario():
    """Test the exact scenario that Flask app uses"""
    try:
        # Simulate the enhanced data that Flask app would have
        # Based on the logs, the Flask app has 16 columns
        data = {
            'Date': ['2025-01-02', '2025-07-09', '2025-03-15', '2025-06-20', '2025-02-10'],
            'Transaction_ID': ['BANK_000001', 'BANK_000002', 'BANK_000003', 'BANK_000004', 'BANK_000005'],
            'Type': ['Debit', 'Debit', 'Credit', 'Debit', 'Credit'],
            'Description': ['Infrastructure Development - Warehouse Construction - 4717 sq ft', 
                          'Plant Expansion - New Production Line - Capacity Increase - Festival Season',
                          'Customer Payment - Steel Supply Contract',
                          'Utility Payment - Electricity Bill',
                          'Scrap Sale - Excess Material'],
            'Amount': [709662.15, 669211.08, 500000.00, 150000.00, 75000.00],
            'Category': ['Investing - Capital Expenditure', 'Investing - Capital Expenditure', 
                        'Operating - Revenue', 'Operating - Expense', 'Operating - Revenue'],
            'Account_Number': ['ACC_9536', 'ACC_4672', 'ACC_1234', 'ACC_5678', 'ACC_9999'],
            'Reference_Number': ['REF_523926', 'REF_464199', 'REF_123456', 'REF_789012', 'REF_345678'],
            'Balance': [3709289.81, 3305684.48, 3809289.81, 3659289.81, 3734289.81],
            'Bank_Charges': [187.49, 81.9, 0.0, 25.0, 0.0],
            'Payment_Terms': ['Standard Terms', 'Standard Terms', 'Net 30', 'Immediate', 'Net 15'],
            'Customer_Vendor': ['Other', 'Other', 'Steel Corp', 'Power Utility', 'Scrap Buyer'],
            'Product_Type': ['Other', 'Other', 'Steel Products', 'Utilities', 'Scrap'],
            'Project_Reference': ['No Project', 'No Project', 'Steel Contract', 'No Project', 'No Project'],
            'Quantity': [4717.0, np.nan, 1000.0, np.nan, 500.0],
            'Seasonal_Indicator': ['No Seasonal Indicator', 'Festival Season', 'No Seasonal Indicator', 
                                 'No Seasonal Indicator', 'No Seasonal Indicator']
        }
        
        # Create DataFrame like Flask app would have
        df = pd.DataFrame(data)
        print(f"✅ Created enhanced data shape: {df.shape}")
        print(f"✅ Columns: {list(df.columns)}")
        
        # Test the enhanced operating expenses function
        ai_system = AdvancedRevenueAISystem()
        print("\n🎯 Testing enhanced Operating Expenses with Flask app data...")
        
        result = ai_system.enhanced_analyze_operating_expenses(df)
        
        print(f"\n📊 RESULT:")
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Success!")
            print(f"✅ Total expenses: {result.get('total_expenses', 'N/A')}")
            print(f"✅ Advanced features: {len(result.get('advanced_ai_features', {}))}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_flask_scenario() 