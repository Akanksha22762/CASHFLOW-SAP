#!/usr/bin/env python3
"""
Debug script to test Operating Expenses function
"""

import pandas as pd
import numpy as np
from advanced_revenue_ai_system import AdvancedRevenueAISystem

def debug_opex():
    """Debug the Operating Expenses function"""
    try:
        # Load the real data
        df = pd.read_excel("uploads/bank_Bank_Statement_Combined.xlsx")
        print(f"✅ Loaded data shape: {df.shape}")
        print(f"✅ Columns: {list(df.columns)}")
        
        # Check data structure
        print(f"✅ Sample data:")
        print(df.head(3).to_string())
        
        # Check for negative amounts (expenses)
        amount_column = None
        for col in df.columns:
            if 'amount' in col.lower():
                amount_column = col
                break
        
        if amount_column:
            print(f"✅ Amount column: {amount_column}")
            
            # Check negative amounts
            negative_amounts = df[df[amount_column] < 0]
            print(f"✅ Negative amounts: {len(negative_amounts)}")
            
            if len(negative_amounts) > 0:
                print(f"✅ Sample negative amounts:")
                print(negative_amounts.head(3).to_string())
            else:
                print("❌ No negative amounts found!")
                
                # Check all amounts
                print(f"✅ All amounts range: {df[amount_column].min()} to {df[amount_column].max()}")
                print(f"✅ Amount distribution:")
                print(df[amount_column].value_counts().head())
        else:
            print("❌ No amount column found!")
        
        # Test the basic Operating Expenses function first
        ai_system = AdvancedRevenueAISystem()
        print("\n🎯 Testing basic Operating Expenses function...")
        
        basic_result = ai_system.analyze_operating_expenses(df)
        
        print(f"\n📊 BASIC OPEX RESULT:")
        if 'error' in basic_result:
            print(f"❌ Error: {basic_result['error']}")
        else:
            print(f"✅ Basic analysis successful!")
            print(f"✅ Total expenses: {basic_result.get('total_expenses', 'N/A')}")
            print(f"✅ Expense count: {basic_result.get('expense_count', 'N/A')}")
        
        # Now test the enhanced function
        print("\n🎯 Testing enhanced Operating Expenses function...")
        
        enhanced_result = ai_system.enhanced_analyze_operating_expenses(df)
        
        print(f"\n📊 ENHANCED OPEX RESULT:")
        if 'error' in enhanced_result:
            print(f"❌ Error: {enhanced_result['error']}")
        else:
            print(f"✅ Enhanced analysis successful!")
            print(f"✅ Total expenses: {enhanced_result.get('total_expenses', 'N/A')}")
            print(f"✅ Advanced features: {len(enhanced_result.get('advanced_ai_features', {}))}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_opex() 