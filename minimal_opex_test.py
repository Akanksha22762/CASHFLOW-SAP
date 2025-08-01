#!/usr/bin/env python3
"""
Minimal test for Operating Expenses Analysis
"""

import pandas as pd

# Create test data
test_data = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-02'],
    'Description': ['Electricity Bill', 'Salary Payment'],
    'Amount': [-15000, -50000],
    'Type': ['Debit', 'Debit']
})

print("📊 Test data created successfully")
print(f"   Transactions: {len(test_data)}")
print(f"   Total amount: ₹{test_data['Amount'].sum():,.2f}")

# Test basic functionality
try:
    # Import the system
    from advanced_revenue_ai_system import AdvancedRevenueAISystem
    print("✅ Import successful")
    
    # Initialize
    ai_system = AdvancedRevenueAISystem()
    print("✅ AI system initialized")
    
    # Test operating expenses analysis
    result = ai_system.analyze_operating_expenses(test_data)
    print("✅ Operating expenses analysis completed")
    print(f"   Total Expenses: {result.get('total_expenses', 'N/A')}")
    print(f"   Expense Count: {result.get('expense_count', 'N/A')}")
    
    # Test complete analysis
    complete_result = ai_system.complete_revenue_analysis_system(test_data)
    if 'revenue_analysis' in complete_result and 'A6_operating_expenses' in complete_result['revenue_analysis']:
        print("✅ A6_operating_expenses found in complete analysis")
        print("🎉 SUCCESS! Operating Expenses parameter is working!")
    else:
        print("❌ A6_operating_expenses NOT found in complete analysis")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 