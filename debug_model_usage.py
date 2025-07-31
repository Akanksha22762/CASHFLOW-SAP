#!/usr/bin/env python3
"""
Debug script to check what models are currently being used
"""

import sys
import os

def check_model_usage():
    """Check what models are currently being used in the system"""
    print("🔍 DEBUGGING MODEL USAGE...")
    print("=" * 50)
    
    try:
        # Check main app models
        print("📊 MAIN APP (app1.py) MODELS:")
        from app1 import lightweight_ai
        print(f"   Available models: {list(lightweight_ai.models.keys())}")
        print(f"   Model types:")
        for name, model in lightweight_ai.models.items():
            print(f"     {name}: {type(model).__name__}")
        
        print("\n📊 ADVANCED REVENUE AI MODELS:")
        from advanced_revenue_ai_system import AdvancedRevenueAISystem
        advanced_ai = AdvancedRevenueAISystem()
        print(f"   Available models: {list(advanced_ai.models.keys())}")
        print(f"   Model types:")
        for name, model in advanced_ai.models.items():
            print(f"     {name}: {type(model).__name__}")
        
        print("\n📊 IMPORTS CHECK:")
        # Check what's imported in app1.py
        with open('app1.py', 'r') as f:
            content = f.read()
            
        # Check for old model imports
        old_models = ['RandomForest', 'LinearRegression', 'SVR', 'MLP', 'LightGBM', 'CatBoost', 'Prophet', 'ARIMA']
        found_old_models = []
        for model in old_models:
            if model in content:
                found_old_models.append(model)
        
        if found_old_models:
            print(f"   ⚠️ OLD MODELS STILL IMPORTED: {found_old_models}")
        else:
            print("   ✅ No old models imported")
        
        # Check for XGBoost
        if 'XGBClassifier' in content or 'XGBRegressor' in content:
            print("   ✅ XGBoost models found")
        else:
            print("   ❌ No XGBoost models found")
        
        print("\n📊 MODEL INITIALIZATION CHECK:")
        # Check model initialization in app1.py
        if 'xgb_classifier' in content:
            print("   ⚠️ 'xgb_classifier' still referenced (should be 'transaction_classifier')")
        else:
            print("   ✅ 'xgb_classifier' not found")
        
        if 'prophet_forecaster' in content:
            print("   ⚠️ 'prophet_forecaster' still referenced")
        else:
            print("   ✅ 'prophet_forecaster' not found")
        
        print("\n📊 ADVANCED REVENUE AI CHECK:")
        # Check advanced revenue AI system
        with open('advanced_revenue_ai_system.py', 'r') as f:
            adv_content = f.read()
        
        if 'Prophet' in adv_content:
            print("   ⚠️ Prophet still imported in advanced_revenue_ai_system.py")
        else:
            print("   ✅ Prophet not imported in advanced_revenue_ai_system.py")
        
        if 'RandomForest' in adv_content:
            print("   ⚠️ RandomForest still imported in advanced_revenue_ai_system.py")
        else:
            print("   ✅ RandomForest not imported in advanced_revenue_ai_system.py")
        
        print("\n🎯 SUMMARY:")
        print("   - Main app should use only XGBoost models")
        print("   - Advanced revenue AI should use only XGBoost + Ollama")
        print("   - No Prophet, RandomForest, or other old models should be used")
        
    except Exception as e:
        print(f"❌ Error checking models: {e}")

if __name__ == "__main__":
    check_model_usage() 