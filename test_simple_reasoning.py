#!/usr/bin/env python3
"""
Simple Reasoning Engine Test
Tests the reasoning engine directly without Flask
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple_reasoning():
    """Test the reasoning engine directly"""
    print("🧠 TESTING REASONING ENGINE DIRECTLY")
    print("=" * 50)
    
    try:
        # Import the reasoning engine
        from app1 import reasoning_engine
        print("✅ Reasoning engine imported successfully!")
        
        # Test 1: Basic XGBoost reasoning
        print("\n🔍 Test 1: Basic XGBoost Reasoning")
        print("-" * 40)
        
        try:
            # Create simple dummy data
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor
            
            X = np.array([[1], [2], [3]])
            y = np.array([10, 20, 30])
            
            # Create a simple model
            model = RandomForestRegressor(n_estimators=5, random_state=42)
            model.fit(X, y)
            
            # Generate reasoning
            ml_reasoning = reasoning_engine.explain_xgboost_prediction(
                model, X, y[-1], feature_names=['simple_feature'], model_type='regressor'
            )
            
            print("✅ Basic ML reasoning generated successfully!")
            print(f"   📊 Keys: {list(ml_reasoning.keys())}")
            print(f"   🎯 Decision Logic: {ml_reasoning.get('decision_logic', 'N/A')[:100]}...")
            
        except Exception as e:
            print(f"❌ Basic ML reasoning failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: Basic Ollama reasoning
        print("\n🧠 Test 2: Basic Ollama Reasoning")
        print("-" * 40)
        
        try:
            ai_reasoning = reasoning_engine.explain_ollama_response(
                "Simple test prompt",
                "Simple test response",
                model_name='llama2:7b'
            )
            
            print("✅ Basic AI reasoning generated successfully!")
            print(f"   📊 Keys: {list(ai_reasoning.keys())}")
            print(f"   🎯 Decision Logic: {ai_reasoning.get('decision_logic', 'N/A')[:100]}...")
            
        except Exception as e:
            print(f"❌ Basic AI reasoning failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: Basic Hybrid reasoning
        print("\n🔗 Test 3: Basic Hybrid Reasoning")
        print("-" * 40)
        
        try:
            hybrid_reasoning = reasoning_engine.generate_hybrid_explanation(
                {'decision_logic': 'Test ML logic'},
                {'decision_logic': 'Test AI logic'},
                "Test combined result"
            )
            
            print("✅ Basic hybrid reasoning generated successfully!")
            print(f"   📊 Keys: {list(hybrid_reasoning.keys())}")
            print(f"   🎯 Decision Logic: {hybrid_reasoning.get('decision_logic', 'N/A')[:100]}...")
            
        except Exception as e:
            print(f"❌ Basic hybrid reasoning failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✅ All basic reasoning tests completed!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure app1.py is in the same directory")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Simple Reasoning Engine Test")
    print("=" * 60)
    
    # Test the reasoning capabilities
    test_simple_reasoning()
    
    print("\n🎉 Testing Complete!")
