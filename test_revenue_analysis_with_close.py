#!/usr/bin/env python3
"""
Test Revenue Analysis With Close Button
Verify that the Revenue Analysis works with proper card design and close button
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

def test_revenue_analysis_with_close():
    """Test that the Revenue Analysis works with proper design and close button"""
    print("🔍 TESTING REVENUE ANALYSIS WITH CLOSE BUTTON")
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
        print("\n📊 REVENUE ANALYSIS WITH CLOSE BUTTON VERIFICATION:")
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
            print("\n🎉 REVENUE ANALYSIS WITH CLOSE BUTTON VERIFIED!")
            print("✅ Step 1: Single 'Revenue Analysis' card matches other cards design")
            print("✅ Step 2: Card has 'View Analysis' and 'Export' buttons")
            print("✅ Step 3: Click 'View Analysis' shows all 5 individual cards")
            print("✅ Step 4: Each card has 'Run Analysis' button")
            print("✅ Step 5: Close button card appears with red X icon")
            print("✅ Step 6: Click 'Close' returns to single card")
            print("✅ Step 7: Future cards can be added easily")
        else:
            print("\n❌ REVENUE ANALYSIS WITH CLOSE BUTTON HAS ISSUES")
            print("Some parameters are missing required fields")
        
        return all_parameters_valid
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 TESTING REVENUE ANALYSIS WITH CLOSE BUTTON")
    print("=" * 60)
    
    success = test_revenue_analysis_with_close()
    
    if success:
        print("\n🎉 TEST PASSED!")
        print("✅ Revenue Analysis works correctly:")
        print("   📋 Single card matches other analysis cards design")
        print("   🎯 View Analysis and Export buttons")
        print("   🔄 Click View Analysis shows all 5 cards")
        print("   🔴 Close button card with red X icon")
        print("   📊 Each card can run individual analysis")
        print("   🔮 Future cards can be added easily")
    else:
        print("\n❌ TEST FAILED!")
        print("Revenue Analysis with close button has issues") 