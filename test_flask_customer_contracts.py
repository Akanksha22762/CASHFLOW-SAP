#!/usr/bin/env python3
"""
Test script to simulate Flask app calling customer contracts
"""

import pandas as pd
import numpy as np
from advanced_revenue_ai_system import AdvancedRevenueAISystem

def test_flask_simulation():
    """Simulate the Flask app data and call customer contracts"""
    try:
        # Simulate the data that Flask app would have
        data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'Description': ['Customer A Payment', 'Client B Invoice', 'Customer C Service', 'Vendor Payment', 'Customer D Order'],
            'Amount': [1000, 2000, 1500, -500, 3000],
            'Type': ['INWARD', 'INWARD', 'INWARD', 'OUTWARD', 'INWARD']
        }
        
        # Create DataFrame like Flask app would
        uploaded_bank_df = pd.DataFrame(data)
        print(f"✅ Simulated Flask data created: {uploaded_bank_df.shape}")
        print(f"✅ Columns: {list(uploaded_bank_df.columns)}")
        
        # Initialize AI system like Flask app would
        advanced_revenue_ai = AdvancedRevenueAISystem()
        print("✅ AI system initialized")
        
        # Call customer contracts like Flask app would
        print("🎯 Calling analyze_customer_contracts...")
        results = advanced_revenue_ai.analyze_customer_contracts(uploaded_bank_df)
        
        # Check results
        if 'error' in results:
            print(f"❌ Analysis returned error: {results['error']}")
        else:
            print(f"✅ Analysis successful!")
            print(f"✅ Total contracts: {results.get('total_contracts', 'N/A')}")
            print(f"✅ Contract value: {results.get('total_contract_value', 'N/A')}")
            print(f"✅ Recurring revenue: {results.get('recurring_revenue', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_flask_simulation() 