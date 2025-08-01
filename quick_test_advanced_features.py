#!/usr/bin/env python3
"""
Quick Test for Advanced AI Features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create simple test data
def create_test_data():
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(50)]
    test_data = []
    
    for i, date in enumerate(dates):
        # Make some transactions negative (expenses)
        amount = np.random.uniform(1000, 50000)
        if i % 3 == 0:  # Every third transaction is an expense
            amount = -amount
        
        test_data.append({
            'Date': date,
            'Description': f'Transaction {i+1}',
            'Amount': amount,
            'Type': 'Credit' if amount > 0 else 'Debit'
        })
    
    return pd.DataFrame(test_data)

def test_import():
    """Test if modules can be imported"""
    try:
        from advanced_revenue_ai_system import AdvancedRevenueAISystem
        print("✅ AdvancedRevenueAISystem imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        from advanced_revenue_ai_system import AdvancedRevenueAISystem
        
        # Initialize system
        ai_system = AdvancedRevenueAISystem()
        print("✅ AI system initialized")
        
        # Create test data
        test_data = create_test_data()
        print(f"✅ Test data created: {len(test_data)} transactions")
        
        # Test one enhanced function
        result = ai_system.enhanced_analyze_operating_expenses(test_data)
        
        if 'error' not in result:
            print("✅ Enhanced analysis function works")
            if 'advanced_ai_features' in result:
                print("✅ Advanced AI features found in results")
                return True
            else:
                print("⚠️ No advanced AI features in results")
                return False
        else:
            print(f"❌ Analysis error: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Quick Test for Advanced AI Features")
    print("=" * 40)
    
    # Test import
    if test_import():
        # Test functionality
        if test_basic_functionality():
            print("\n🎉 All tests passed! Advanced AI features are working.")
        else:
            print("\n❌ Functionality test failed.")
    else:
        print("\n❌ Import test failed.") 