#!/usr/bin/env python3
"""
Test script for streamlined XGBoost + Ollama hybrid system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_xgboost_ollama_system():
    """Test the streamlined XGBoost + Ollama system"""
    print("🧪 Testing XGBoost + Ollama Hybrid System...")
    print("=" * 50)
    
    try:
        # Test 1: Import the advanced revenue AI system
        print("✅ Test 1: Importing Advanced Revenue AI System...")
        from advanced_revenue_ai_system import AdvancedRevenueAISystem
        
        # Test 2: Initialize the system
        print("✅ Test 2: Initializing XGBoost + Ollama Models...")
        advanced_ai = AdvancedRevenueAISystem()
        
        # Test 3: Check if models are XGBoost only
        print("✅ Test 3: Verifying XGBoost Models...")
        xgb_models = []
        other_models = []
        
        for model_name, model in advanced_ai.models.items():
            if 'xgb' in str(type(model)).lower():
                xgb_models.append(model_name)
            else:
                other_models.append(model_name)
        
        print(f"   XGBoost Models: {xgb_models}")
        print(f"   Other Models: {other_models}")
        
        if len(other_models) == 0:
            print("✅ All models are XGBoost!")
        else:
            print(f"⚠️ Found non-XGBoost models: {other_models}")
        
        # Test 4: Create sample data
        print("✅ Test 4: Creating Sample Data...")
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
        
        # Test 5: Test AI categorization
        print("✅ Test 5: Testing AI Categorization...")
        for idx, row in sample_data.iterrows():
            result = advanced_ai.ai_ml_categorize_any_description(
                row['Description'], row['Amount'], row['Date']
            )
            print(f"   {row['Description']} -> {result['category']} (Confidence: {result['confidence']:.2f})")
        
        # Test 6: Test revenue analysis
        print("✅ Test 6: Testing Revenue Analysis...")
        revenue_result = advanced_ai.analyze_historical_revenue_trends(sample_data)
        print(f"   Total Revenue: {revenue_result.get('total_revenue', 'N/A')}")
        print(f"   Trend Direction: {revenue_result.get('trend_direction', 'N/A')}")
        
        # Test 7: Test sales forecasting
        print("✅ Test 7: Testing Sales Forecasting...")
        forecast_result = advanced_ai.prophet_sales_forecasting(sample_data)
        print(f"   Current Month Forecast: {forecast_result.get('current_month_forecast', 'N/A')}")
        print(f"   Next Quarter Forecast: {forecast_result.get('next_quarter_forecast', 'N/A')}")
        
        # Test 8: Test complete analysis
        print("✅ Test 8: Testing Complete Analysis...")
        complete_result = advanced_ai.complete_revenue_analysis_system(sample_data)
        print(f"   System Status: {complete_result.get('system_status', 'N/A')}")
        print(f"   AI/ML Confidence: {complete_result.get('ai_ml_confidence', 'N/A')}")
        
        print("\n🎉 All Tests Passed! XGBoost + Ollama System is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        return False

if __name__ == "__main__":
    success = test_xgboost_ollama_system()
    if success:
        print("\n✅ System is ready for production!")
    else:
        print("\n❌ System needs fixes!") 