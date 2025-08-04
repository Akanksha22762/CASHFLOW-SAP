#!/usr/bin/env python3
"""
Debug script to test AR Aging function
"""

import pandas as pd
import numpy as np
from advanced_revenue_ai_system import AdvancedRevenueAISystem

def debug_ar_aging():
    """Debug the AR Aging function"""
    try:
        # Load the real data
        df = pd.read_excel("uploads/bank_Bank_Statement_Combined.xlsx")
        print(f"✅ Loaded data shape: {df.shape}")
        print(f"✅ Columns: {list(df.columns)}")
        
        # Check data structure
        print(f"✅ Sample data:")
        print(df.head(3).to_string())
        
        # Check for positive amounts (receivables)
        amount_column = None
        for col in df.columns:
            if 'amount' in col.lower():
                amount_column = col
                break
        
        if amount_column:
            print(f"✅ Amount column: {amount_column}")
            
            # Check positive amounts
            positive_amounts = df[df[amount_column] > 0]
            print(f"✅ Positive amounts: {len(positive_amounts)}")
            
            if len(positive_amounts) > 0:
                print(f"✅ Sample positive amounts:")
                print(positive_amounts.head(3).to_string())
            else:
                print("❌ No positive amounts found!")
                
                # Check all amounts
                print(f"✅ All amounts range: {df[amount_column].min()} to {df[amount_column].max()}")
                print(f"✅ Amount distribution:")
                print(df[amount_column].value_counts().head())
        else:
            print("❌ No amount column found!")
        
        # Test the AR Aging function
        ai_system = AdvancedRevenueAISystem()
        print("\n🎯 Testing AR Aging function...")
        
        result = ai_system.calculate_dso_and_collection_probability(df)
        
        print(f"\n📊 AR AGING RESULT:")
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Success!")
            print(f"✅ Total receivables: {result.get('total_receivables', 'N/A')}")
            print(f"✅ DSO days: {result.get('dso_days', 'N/A')}")
            print(f"✅ Collection probability: {result.get('weighted_collection_probability', 'N/A')}")
            
            # Check aging buckets
            if 'aging_analysis' in result:
                aging = result['aging_analysis']
                print(f"✅ Aging buckets:")
                for bucket, data in aging.items():
                    print(f"   {bucket}: {data.get('count', 0)} transactions, ₹{data.get('amount', 0):,.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ar_aging() 