#!/usr/bin/env python3
"""
Debug why contract count is showing 1 instead of 2
"""

import pandas as pd
import numpy as np
from advanced_revenue_ai_system import AdvancedRevenueAISystem

def debug_contract_count():
    """Debug the contract counting logic"""
    try:
        # Load the real data
        df = pd.read_excel("uploads/bank_Bank_Statement_Combined.xlsx")
        print(f"✅ Loaded data shape: {df.shape}")
        
        # Filter revenue transactions
        revenue_transactions = df[df['Type'].str.contains('INWARD|CREDIT', case=False, na=False)]
        print(f"✅ Revenue transactions: {len(revenue_transactions)}")
        
        # Check descriptions
        descriptions = revenue_transactions['Description'].head(10).tolist()
        print(f"🔍 Sample revenue descriptions: {descriptions}")
        
        # Test customer extraction
        customer_extractions = revenue_transactions['Description'].str.extract(r'(Customer\s+[A-Z]|Client\s+[A-Z]|Customer\s+\w+|Client\s+\w+)')
        print(f"🔍 Customer extractions shape: {customer_extractions.shape}")
        
        if not customer_extractions.empty:
            unique_customers = customer_extractions.iloc[:, 0].dropna().unique()
            total_contracts = len(unique_customers)
            print(f"🔍 Unique customers found: {unique_customers}")
            print(f"🔍 Total contracts: {total_contracts}")
        else:
            total_contracts = max(1, len(revenue_transactions) // 4)
            print(f"🔍 Estimated contracts: {total_contracts}")
        
        # Test the full customer contracts analysis
        ai_system = AdvancedRevenueAISystem()
        result = ai_system.analyze_customer_contracts(df)
        
        print(f"\n🎯 FULL ANALYSIS RESULT:")
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Total contracts: {result.get('total_contracts', 'N/A')}")
            print(f"✅ Contract value: {result.get('total_contract_value', 'N/A')}")
            print(f"✅ Recurring revenue: {result.get('recurring_revenue', 'N/A')}")
            
            # Check customer segments
            if 'customer_segments' in result:
                segments = result['customer_segments']
                print(f"✅ Customer segments:")
                for segment_name, segment_data in segments.items():
                    print(f"   {segment_name}: {segment_data.get('count', 0)} customers")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_contract_count() 