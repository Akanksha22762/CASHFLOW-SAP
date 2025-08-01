#!/usr/bin/env python3
"""
Test Flask App
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import pandas as pd
        print("✅ pandas imported")
        
        import numpy as np
        print("✅ numpy imported")
        
        # Test advanced AI system
        from advanced_revenue_ai_system import AdvancedRevenueAISystem
        print("✅ AdvancedRevenueAISystem imported")
        
        # Test integration
        from integrate_advanced_revenue_system import AdvancedRevenueIntegration
        print("✅ AdvancedRevenueIntegration imported")
        
        # Test Flask
        from flask import Flask
        print("✅ Flask imported")
        
        # Test app1
        import app1
        print("✅ app1 imported")
        
        print("\n🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_ai_system():
    """Test if AI system can be initialized"""
    try:
        print("\nTesting AI system...")
        
        from advanced_revenue_ai_system import AdvancedRevenueAISystem
        
        # Create AI system
        ai_system = AdvancedRevenueAISystem()
        print("✅ AI system initialized")
        
        # Test basic function
        import pandas as pd
        test_data = pd.DataFrame({
            'Amount': [1000, -500, 2000, -1000],
            'Description': ['Revenue 1', 'Expense 1', 'Revenue 2', 'Expense 2'],
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
        })
        
        result = ai_system.analyze_operating_expenses(test_data)
        print(f"✅ Basic function test: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI system error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Flask App Components")
    print("=" * 40)
    
    # Test imports
    if test_imports():
        # Test AI system
        if test_ai_system():
            print("\n🎉 All tests passed! Flask app should work.")
        else:
            print("\n❌ AI system test failed.")
    else:
        print("\n❌ Import test failed.") 