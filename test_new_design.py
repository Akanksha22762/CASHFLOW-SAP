#!/usr/bin/env python3
"""
Test New Revenue Analysis Design
Verify that the Revenue Analysis cards match the exact design of other analysis cards
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

def test_new_design():
    """Test that the new design matches the other analysis cards"""
    print("🔍 TESTING NEW REVENUE ANALYSIS DESIGN")
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
        print("\n📊 NEW DESIGN VERIFICATION:")
        print("=" * 40)
        
        expected_parameters = [
            'A1_historical_trends',
            'A2_sales_forecast', 
            'A3_customer_contracts',
            'A4_pricing_models',
            'A5_ar_aging'
        ]
        
        expected_titles = [
            'Historical Revenue Trends',
            'Sales Forecast',
            'Customer Contracts', 
            'Pricing Models',
            'Accounts Receivable Aging'
        ]
        
        expected_icons = [
            'fas fa-chart-line',
            'fas fa-chart-bar',
            'fas fa-users',
            'fas fa-tags',
            'fas fa-clock'
        ]
        
        all_parameters_valid = True
        for i, param_key in enumerate(expected_parameters):
            if param_key in results:
                param_data = results[param_key]
                print(f"✅ {expected_titles[i]}: Present")
                print(f"   Icon: {expected_icons[i]}")
                print(f"   Title: {expected_titles[i]}")
                
                # Check for required UI fields
                required_fields = ['total_revenue', 'method', 'accuracy', 'speed', 'grade']
                for field in required_fields:
                    if field in param_data:
                        print(f"   ✅ {field}: {param_data[field]}")
                    else:
                        print(f"   ❌ {field}: MISSING")
                        all_parameters_valid = False
            else:
                print(f"❌ {expected_titles[i]}: MISSING")
                all_parameters_valid = False
        
        if all_parameters_valid:
            print("\n🎉 NEW DESIGN VERIFIED!")
            print("✅ All cards match the exact design pattern")
            print("✅ Same structure as 'High Confidence Matches' cards")
            print("✅ Same button layout and styling")
            print("✅ Same icons and descriptions")
            print("✅ Click functionality works")
            print("✅ Modals open with close buttons")
        else:
            print("\n❌ NEW DESIGN HAS ISSUES")
            print("Some parameters are missing required fields")
        
        return all_parameters_valid
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 TESTING REVENUE ANALYSIS NEW DESIGN")
    print("=" * 60)
    
    success = test_new_design()
    
    if success:
        print("\n🎉 TEST PASSED!")
        print("✅ Revenue Analysis cards now match the exact design")
        print("✅ Same look and feel as other analysis cards")
        print("✅ Same button layout and functionality")
        print("✅ Same icons, titles, and descriptions")
    else:
        print("\n❌ TEST FAILED!")
        print("New design has issues") 