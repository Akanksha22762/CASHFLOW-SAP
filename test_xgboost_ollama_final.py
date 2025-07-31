#!/usr/bin/env python3
"""
Final test for XGBoost + Ollama Hybrid System
Verifies that all RandomForest references have been removed
"""

import pandas as pd
import numpy as np
from datetime import datetime

def test_xgboost_ollama_system():
    """Test that the system uses only XGBoost + Ollama"""
    print("🧪 Testing XGBoost + Ollama Hybrid System...")
    
    # Test 1: Check imports in app1.py
    print("\n📋 Test 1: Checking for RandomForest imports...")
    
    try:
        with open('app1.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for RandomForest imports
        rf_imports = content.count('RandomForest')
        prophet_imports = content.count('Prophet')
        linear_imports = content.count('LinearRegression')
        
        print(f"   RandomForest references: {rf_imports}")
        print(f"   Prophet references: {prophet_imports}")
        print(f"   LinearRegression references: {linear_imports}")
        
        if rf_imports == 0 and prophet_imports == 0:
            print("✅ SUCCESS: No RandomForest or Prophet imports found!")
        else:
            print("❌ FAILURE: Still found RandomForest or Prophet references")
            
    except Exception as e:
        print(f"❌ Error checking imports: {e}")
    
    # Test 2: Check advanced_revenue_ai_system.py
    print("\n📋 Test 2: Checking Advanced Revenue AI System...")
    
    try:
        with open('advanced_revenue_ai_system.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for XGBoost models
        xgb_models = content.count('XGBClassifier') + content.count('XGBRegressor')
        rf_models = content.count('RandomForest')
        
        print(f"   XGBoost models: {xgb_models}")
        print(f"   RandomForest models: {rf_models}")
        
        if xgb_models > 0 and rf_models == 0:
            print("✅ SUCCESS: Only XGBoost models found!")
        else:
            print("❌ FAILURE: Found RandomForest models or no XGBoost models")
            
    except Exception as e:
        print(f"❌ Error checking advanced system: {e}")
    
    # Test 3: Check console output messages
    print("\n📋 Test 3: Checking system messages...")
    
    try:
        with open('app1.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for proper system messages
        hybrid_message = "XGBoost + Ollama Hybrid System loaded successfully!" in content
        xgb_message = "XGBoost accuracy:" in content
        rf_message = "RandomForest accuracy:" in content
        
        print(f"   Hybrid system message: {'✅' if hybrid_message else '❌'}")
        print(f"   XGBoost accuracy message: {'✅' if xgb_message else '❌'}")
        print(f"   RandomForest accuracy message: {'❌' if not rf_message else '❌'}")
        
        if hybrid_message and xgb_message and not rf_message:
            print("✅ SUCCESS: Proper system messages found!")
        else:
            print("❌ FAILURE: Incorrect system messages")
            
    except Exception as e:
        print(f"❌ Error checking messages: {e}")
    
    # Test 4: Check statistics calculation
    print("\n📋 Test 4: Checking statistics calculation...")
    
    try:
        with open('app1.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for proper statistics
        xgb_stats = "ML Models (XGBoost):" in content
        rf_stats = "ML Models (RandomForest" in content
        
        print(f"   XGBoost statistics: {'✅' if xgb_stats else '❌'}")
        print(f"   RandomForest statistics: {'❌' if not rf_stats else '❌'}")
        
        if xgb_stats and not rf_stats:
            print("✅ SUCCESS: Proper statistics calculation!")
        else:
            print("❌ FAILURE: Incorrect statistics calculation")
            
    except Exception as e:
        print(f"❌ Error checking statistics: {e}")
    
    print("\n🎯 XGBoost + Ollama Hybrid System Test Complete!")
    print("The system should now use ONLY XGBoost + Ollama for all ML tasks.")

if __name__ == "__main__":
    test_xgboost_ollama_system() 