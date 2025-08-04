#!/usr/bin/env python3
"""
Test script to debug customer contracts error
"""

import pandas as pd
import numpy as np
from advanced_revenue_ai_system import AdvancedRevenueAISystem

def test_customer_contracts():
    """Test the customer contracts analysis"""
    try:
        # Create sample data
        data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Description': ['Customer A Payment', 'Client B Invoice', 'Customer C Service'],
            'Amount': [1000, 2000, 1500],
            'Type': ['INWARD', 'INWARD', 'INWARD']
        }
        
        df = pd.DataFrame(data)
        print(f"✅ Sample data created: {df.shape}")
        print(f"✅ Columns: {list(df.columns)}")
        
        # Initialize the system
        ai_system = AdvancedRevenueAISystem()
        print("✅ AI system initialized")
        
        # Test customer contracts analysis
        result = ai_system.analyze_customer_contracts(df)
        print("✅ Customer contracts analysis completed")
        
        # Check if result has error
        if 'error' in result:
            print(f"❌ Analysis returned error: {result['error']}")
        else:
            print(f"✅ Analysis successful!")
            print(f"✅ Total contracts: {result.get('total_contracts', 'N/A')}")
            print(f"✅ Contract value: {result.get('total_contract_value', 'N/A')}")
            print(f"✅ Recurring revenue: {result.get('recurring_revenue', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_customer_contracts() 