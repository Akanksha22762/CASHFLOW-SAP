#!/usr/bin/env python3
"""
Test Sales Forecast Fix
Verify that the Sales Forecast now displays correctly
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

def test_sales_forecast_fix():
    """Test that Sales Forecast now works correctly"""
    print("🔍 TESTING SALES FORECAST FIX")
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
        
        # Check Sales Forecast specifically
        sales_forecast = results.get('A2_sales_forecast', {})
        
        print("\n📊 SALES FORECAST RESULTS:")
        print("=" * 40)
        
        # Check if all expected fields are present
        expected_fields = [
            'forecast_amount', 'confidence', 'growth_rate', 
            'total_revenue', 'monthly_average', 'trend_direction'
        ]
        
        all_fields_present = True
        for field in expected_fields:
            if field in sales_forecast:
                value = sales_forecast[field]
                print(f"✅ {field}: {value}")
            else:
                print(f"❌ {field}: MISSING")
                all_fields_present = False
        
        # Check for error field
        if 'error' in sales_forecast:
            print(f"❌ ERROR: {sales_forecast['error']}")
            all_fields_present = False
        
        if all_fields_present:
            print("\n🎉 SALES FORECAST FIX VERIFIED!")
            print("✅ All expected fields are present")
            print("✅ No 'format not recognized' error should appear")
            print("✅ UI should now display Sales Forecast correctly")
        else:
            print("\n❌ SALES FORECAST STILL HAS ISSUES")
            print("Some expected fields are missing")
        
        return all_fields_present
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 TESTING SALES FORECAST UI FIX")
    print("=" * 60)
    
    success = test_sales_forecast_fix()
    
    if success:
        print("\n🎉 TEST PASSED!")
        print("✅ Sales Forecast should now display correctly in the UI")
        print("✅ No more 'Data available but format not recognized' message")
    else:
        print("\n❌ TEST FAILED!")
        print("Sales Forecast still has issues") 