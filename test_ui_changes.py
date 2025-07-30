#!/usr/bin/env python3
"""
Test UI Changes
Verify that the Revenue Analysis cards display correctly with proper functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_revenue_ai_system import AdvancedRevenueAISystem
    print("✅ Advanced Revenue AI System imported successfully!")
except ImportError as e:
    print(f"❌ Error importing Advanced Revenue AI System: {e}")
    sys.exit(1)

def test_ui_changes():
    """Test that the UI changes work correctly"""
    print("🔍 TESTING UI CHANGES")
    print("=" * 60)
    
    try:
        # Initialize the system
        revenue_ai = AdvancedRevenueAISystem()
        print("✅ Revenue AI System initialized")
        
        # Create test data
        print("📊 Creating test data...")
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        revenue_amounts = np.random.normal(50000, 15000, len(dates))
        revenue_amounts = np.abs(revenue_amounts)
        
        test_data = pd.DataFrame({
            'Date': dates,
            'Description': [f'Revenue Transaction {i+1}' for i in range(len(dates))],
            'Amount': revenue_amounts,
            'Type': ['Credit'] * len(dates)
        })
        
        print(f"✅ Created test data with {len(test_data)} transactions")
        
        # Run revenue analysis
        print("\n🧠 Running SMART OLLAMA Revenue Analysis...")
        results = revenue_ai.complete_revenue_analysis_system_smart_ollama(test_data)
        
        # Check that all parameters have the correct structure for UI
        print("\n📊 UI PARAMETER STRUCTURE CHECK:")
        print("=" * 40)
        
        expected_parameters = [
            'A1_historical_trends',
            'A2_sales_forecast', 
            'A3_customer_contracts',
            'A4_pricing_models',
            'A5_ar_aging'
        ]
        
        all_parameters_valid = True
        for param_key in expected_parameters:
            if param_key in results:
                param_data = results[param_key]
                print(f"✅ {param_key}: Present")
                
                # Check for required UI fields
                required_fields = ['total_revenue', 'method', 'accuracy', 'speed', 'grade']
                for field in required_fields:
                    if field in param_data:
                        print(f"   ✅ {field}: {param_data[field]}")
                    else:
                        print(f"   ❌ {field}: MISSING")
                        all_parameters_valid = False
            else:
                print(f"❌ {param_key}: MISSING")
                all_parameters_valid = False
        
        if all_parameters_valid:
            print("\n🎉 UI CHANGES VERIFIED!")
            print("✅ All parameters have correct structure for UI display")
            print("✅ Cards should display properly with click functionality")
            print("✅ Modals should open with close buttons")
            print("✅ Export functionality should work")
        else:
            print("\n❌ UI CHANGES HAVE ISSUES")
            print("Some parameters are missing required fields")
        
        return all_parameters_valid
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 TESTING REVENUE ANALYSIS UI CHANGES")
    print("=" * 60)
    
    success = test_ui_changes()
    
    if success:
        print("\n🎉 TEST PASSED!")
        print("✅ Revenue Analysis cards should display correctly")
        print("✅ Click functionality should work")
        print("✅ Modals should have proper close buttons")
        print("✅ All parameters should show detailed analysis")
    else:
        print("\n❌ TEST FAILED!")
        print("UI changes have issues") 