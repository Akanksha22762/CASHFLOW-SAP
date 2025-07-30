#!/usr/bin/env python3
"""
Test Actual Function
Call the actual function being used to see what it returns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_actual_function():
    """Test the actual function being called"""
    print("🔍 TESTING ACTUAL FUNCTION")
    print("=" * 60)
    
    try:
        # Import the actual module
        from advanced_revenue_ai_system import AdvancedRevenueAISystem
        
        # Create the AI system
        revenue_ai = AdvancedRevenueAISystem()
        
        # Create test data similar to what would be uploaded
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        n_transactions = 50
        
        test_data = pd.DataFrame({
            'Date': np.random.choice(dates, n_transactions),
            'Amount (INR)': np.random.lognormal(10, 1, n_transactions),
            'Description': [
                f"Payment from Customer_{i%20}" if i % 3 == 0 else
                f"Service Fee {i%10}" if i % 3 == 1 else
                f"Subscription Renewal {i%5}" 
                for i in range(n_transactions)
            ]
        })
        
        print("📊 Test Data Created:")
        print(f"Total transactions: {len(test_data)}")
        print(f"Revenue range: ₹{test_data['Amount (INR)'].min():,.0f} to ₹{test_data['Amount (INR)'].max():,.0f}")
        
        # Call the actual function being used
        print("\n🧠 Calling complete_revenue_analysis_system_smart_ollama...")
        results = revenue_ai.complete_revenue_analysis_system_smart_ollama(test_data)
        
        print(f"\n📊 Results Keys: {list(results.keys())}")
        
        # Check the AR aging results specifically
        if 'A5_ar_aging' in results:
            ar_result = results['A5_ar_aging']
            print(f"\n📊 A5_ar_aging Result:")
            print(f"Type: {type(ar_result)}")
            print(f"Keys: {list(ar_result.keys()) if isinstance(ar_result, dict) else 'Not a dict'}")
            
            if isinstance(ar_result, dict):
                for key, value in ar_result.items():
                    print(f"  {key}: {value}")
                
                # Check collection probability specifically
                if 'collection_probability' in ar_result:
                    cp_value = ar_result['collection_probability']
                    print(f"\n🎯 Collection Probability: {cp_value}")
                    
                    if isinstance(cp_value, (int, float)):
                        if cp_value > 100:
                            print(f"❌ ISSUE: Collection probability is {cp_value}% (should be <= 100%)")
                        else:
                            print(f"✅ OK: Collection probability is {cp_value}%")
                    else:
                        print(f"⚠️ Collection probability is not a number: {cp_value}")
                else:
                    print("❌ No collection_probability key found")
            else:
                print(f"❌ A5_ar_aging result is not a dict: {ar_result}")
        else:
            print("❌ No A5_ar_aging key found in results")
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing actual function: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("🚀 TESTING ACTUAL FUNCTION")
    print("=" * 60)
    
    # Test the actual function
    results = test_actual_function()
    
    print("\n🎯 CONCLUSION:")
    if results:
        print("✅ Function executed successfully")
        print("📊 Check the collection probability value above")
    else:
        print("❌ Function failed to execute")

if __name__ == "__main__":
    main() 