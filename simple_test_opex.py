#!/usr/bin/env python3
"""
Simple test for Operating Expenses Analysis
"""

import pandas as pd
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_revenue_ai_system import AdvancedRevenueAISystem
    print("✅ Successfully imported AdvancedRevenueAISystem")
    
    # Create simple test data
    test_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Description': ['Electricity Bill', 'Salary Payment', 'Raw Material Purchase'],
        'Amount': [-15000, -50000, -100000],
        'Type': ['Debit', 'Debit', 'Debit']
    })
    
    print(f"📊 Test data created: {len(test_data)} transactions")
    
    # Initialize AI system
    revenue_ai = AdvancedRevenueAISystem()
    print("✅ AI system initialized")
    
    # Test operating expenses analysis
    try:
        opex_result = revenue_ai.analyze_operating_expenses(test_data)
        print("✅ Operating expenses analysis completed!")
        print(f"   Total Expenses: {opex_result.get('total_expenses', 'N/A')}")
        print(f"   Expense Count: {opex_result.get('expense_count', 'N/A')}")
        
        # Test complete analysis
        complete_result = revenue_ai.complete_revenue_analysis_system(test_data)
        if 'revenue_analysis' in complete_result and 'A6_operating_expenses' in complete_result['revenue_analysis']:
            print("✅ A6_operating_expenses found in complete analysis!")
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ A6_operating_expenses NOT found in complete analysis!")
            
    except Exception as e:
        print(f"❌ Error in operating expenses analysis: {e}")
        
except Exception as e:
    print(f"❌ Error importing or initializing: {e}") 