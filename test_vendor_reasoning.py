#!/usr/bin/env python3
"""
Test script to verify vendor analysis reasoning generation
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_vendor_reasoning():
    """Test vendor reasoning generation"""
    print("🧪 Testing vendor reasoning generation...")
    
    try:
        # Import the vendor analysis function
        from app1 import analyze_vendor_cash_flow
        
        # Create sample vendor data
        sample_data = {
            'Description': ['Vendor Payment 1', 'Vendor Payment 2', 'Vendor Payment 3'],
            'Amount': [1000000, 2000000, 1500000],
            'Date': ['2024-01-01', '2024-01-15', '2024-01-30']
        }
        
        sample_df = pd.DataFrame(sample_data)
        print(f"📊 Sample data created: {len(sample_df)} transactions")
        
        # Test the vendor analysis function
        result = analyze_vendor_cash_flow(sample_df, 'hybrid')
        
        print("\n🔍 Analysis Result Keys:", list(result.keys()) if isinstance(result, dict) else "Not a dict")
        
        if isinstance(result, dict):
            if 'simple_reasoning' in result:
                print("✅ SUCCESS: simple_reasoning field found!")
                print("📝 Simple Reasoning Content:")
                print(result['simple_reasoning'])
            else:
                print("❌ FAILURE: simple_reasoning field NOT found!")
                print("🔍 Available fields:", list(result.keys()))
                
                # Check if insights field exists
                if 'insights' in result:
                    print("📝 Insights field content (first 200 chars):")
                    print(str(result['insights'])[:200] + "...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing vendor reasoning: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_vendor_reasoning()
