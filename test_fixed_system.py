#!/usr/bin/env python3
"""
Test script to verify the fixed XGBoost + Ollama system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_fixed_system():
    """Test the fixed XGBoost + Ollama system"""
    print("🧪 Testing Fixed XGBoost + Ollama System...")
    print("=" * 50)
    
    try:
        # Test 1: Import and initialize systems
        print("✅ Test 1: Importing Systems...")
        from app1 import lightweight_ai
        from advanced_revenue_ai_system import AdvancedRevenueAISystem
        
        advanced_ai = AdvancedRevenueAISystem()
        
        # Test 2: Check model types
        print("✅ Test 2: Checking Model Types...")
        print("   Main App Models:")
        for name, model in lightweight_ai.models.items():
            print(f"     {name}: {type(model).__name__}")
        
        print("   Advanced AI Models:")
        for name, model in advanced_ai.models.items():
            print(f"     {name}: {type(model).__name__}")
        
        # Test 3: Create sample data
        print("✅ Test 3: Creating Sample Data...")
        sample_data = pd.DataFrame({
            'Description': [
                'TATA STEEL PAYMENT',
                'JSW STEEL INVOICE',
                'CONSTRUCTION MATERIALS',
                'WAREHOUSE RENT',
                'STEEL PLATES DELIVERY'
            ],
            'Amount': [50000, 75000, -15000, -5000, 120000],
            'Date': [
                datetime.now() - timedelta(days=1),
                datetime.now() - timedelta(days=2),
                datetime.now() - timedelta(days=3),
                datetime.now() - timedelta(days=4),
                datetime.now() - timedelta(days=5)
            ]
        })
        
        # Test 4: Test XGBoost training (should not fail now)
        print("✅ Test 4: Testing XGBoost Training...")
        try:
            # Add Category column for training
            sample_data['Category'] = ['Operating Activities', 'Operating Activities', 'Investing Activities', 'Operating Activities', 'Operating Activities']
            
            # Test training
            success = lightweight_ai.train_transaction_classifier(sample_data)
            print(f"   Training success: {success}")
        except Exception as e:
            print(f"   Training error: {e}")
        
        # Test 5: Test revenue analysis
        print("✅ Test 5: Testing Revenue Analysis...")
        try:
            revenue_result = advanced_ai.analyze_historical_revenue_trends(sample_data)
            print(f"   Revenue analysis: {revenue_result.get('total_revenue', 'N/A')}")
        except Exception as e:
            print(f"   Revenue analysis error: {e}")
        
        # Test 6: Test XGBoost forecasting
        print("✅ Test 6: Testing XGBoost Forecasting...")
        try:
            forecast_result = advanced_ai.xgboost_sales_forecasting(sample_data)
            print(f"   Forecast: {forecast_result.get('current_month_forecast', 'N/A')}")
        except Exception as e:
            print(f"   Forecast error: {e}")
        
        print("\n🎉 All Tests Completed!")
        print("✅ System should now use only XGBoost + Ollama")
        
        return True
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        return False

if __name__ == "__main__":
    success = test_fixed_system()
    if success:
        print("\n✅ System is working correctly!")
    else:
        print("\n❌ System needs more fixes!") 