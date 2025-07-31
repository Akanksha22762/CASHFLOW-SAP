#!/usr/bin/env python3
"""
Clean test to verify system without hanging
"""

import sys
import time

def clean_test():
    """Clean test without hanging"""
    print("🧪 Clean System Test...")
    
    try:
        # Test 1: Check if basic imports work
        print("📋 Test 1: Basic imports...")
        import pandas as pd
        print("   ✅ Pandas imported")
        
        # Test 2: Check Ollama availability
        print("📋 Test 2: Ollama availability...")
        try:
            import httpx
            response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("   ✅ Ollama available")
            else:
                print("   ❌ Ollama not responding")
        except Exception as e:
            print(f"   ❌ Ollama error: {e}")
        
        # Test 3: Check XGBoost availability
        print("📋 Test 3: XGBoost availability...")
        try:
            import xgboost as xgb
            print("   ✅ XGBoost available")
        except Exception as e:
            print(f"   ❌ XGBoost error: {e}")
        
        # Test 4: Test hybrid function directly
        print("📋 Test 4: Hybrid function test...")
        try:
            # Import only the function we need
            sys.path.insert(0, '.')
            from app1 import hybrid_categorize_transaction
            
            result = hybrid_categorize_transaction("Infrastructure Development", 1000000)
            print(f"   Result: {result}")
            
            if '(XGBoost)' in result:
                print("   ✅ Using XGBoost")
            elif '(Ollama)' in result:
                print("   ✅ Using Ollama")
            elif '(Rules)' in result:
                print("   ⚠️ Using Rules")
            else:
                print("   ❌ Unknown method")
                
        except Exception as e:
            print(f"   ❌ Hybrid function error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n🎉 Clean test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    clean_test() 