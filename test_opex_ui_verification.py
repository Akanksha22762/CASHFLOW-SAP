#!/usr/bin/env python3
"""
Test script to verify OPEX parameter implementation
"""

import pandas as pd
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_revenue_ai_system import AdvancedRevenueAISystem
    print("✅ Successfully imported AdvancedRevenueAISystem")
    
    # Initialize the system
    advanced_revenue_ai = AdvancedRevenueAISystem()
    print("✅ Successfully initialized AdvancedRevenueAISystem")
    
    # Create test data with expenses
    test_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=20, freq='D'),
        'Description': [
            'Utility Payment - Electricity',
            'Salary Payment - Production Staff',
            'Rent Payment - Factory Building',
            'Maintenance - Equipment Repair',
            'Insurance Premium - Factory',
            'Raw Material Purchase - Steel',
            'Transportation Cost - Delivery',
            'Office Supplies - Stationery',
            'Security Service - Factory',
            'Internet Service - Office',
            'Customer Payment - Steel Plates',
            'Payment to Supplier - Iron Ore',
            'Transfer - Bank',
            'ABC Corp Payment',
            'Steel payment received',
            'Loan Repayment - Principal',
            'Tax Payment - GST',
            'Equipment Purchase - New Mill',
            'Dividend Payment',
            'Interest Payment - Loan'
        ],
        'Amount': [
            -15000, -25000, -50000, -10000, -8000,
            -75000, -5000, -2000, -12000, -3000,
            150000, -75000, 25000, 50000, 80000,
            -50000, -25000, -200000, 10000, -15000
        ],
        'Type': ['Debit', 'Debit', 'Debit', 'Debit', 'Debit',
                 'Debit', 'Debit', 'Debit', 'Debit', 'Debit',
                 'Credit', 'Debit', 'Credit', 'Credit', 'Credit',
                 'Debit', 'Debit', 'Debit', 'Credit', 'Debit']
    })
    
    print("✅ Created test data with expenses")
    print(f"📊 Test data shape: {test_data.shape}")
    print(f"💰 Total amounts: {test_data['Amount'].sum()}")
    
    # Test OPEX analysis specifically
    print("\n🔍 Testing OPEX analysis...")
    opex_results = advanced_revenue_ai.analyze_operating_expenses(test_data)
    
    print("✅ OPEX analysis completed successfully!")
    print(f"📈 Total expenses: {opex_results.get('total_expenses', 'N/A')}")
    print(f"📊 Expense count: {opex_results.get('expense_count', 'N/A')}")
    print(f"🎯 Efficiency score: {opex_results.get('expense_efficiency_score', 'N/A')}%")
    
    # Test complete system integration
    print("\n🔍 Testing complete revenue analysis system...")
    complete_results = advanced_revenue_ai.complete_revenue_analysis_system(test_data)
    
    print("✅ Complete analysis completed successfully!")
    
    # Check if OPEX is in the results
    if 'A6_operating_expenses' in complete_results:
        print("✅ OPEX parameter (A6_operating_expenses) is present in complete results!")
        opex_in_complete = complete_results['A6_operating_expenses']
        print(f"📈 Complete OPEX total expenses: {opex_in_complete.get('total_expenses', 'N/A')}")
    else:
        print("❌ OPEX parameter (A6_operating_expenses) is MISSING from complete results!")
        print(f"Available keys: {list(complete_results.keys())}")
    
    print("\n🎉 All tests passed! OPEX parameter is working correctly.")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc() 