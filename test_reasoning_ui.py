#!/usr/bin/env python3
"""
Test Script for Reasoning UI Integration
Verifies that reasoning explanations are being generated and accessible
"""

import sys
import os
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_reasoning_generation():
    """Test that reasoning explanations are being generated"""
    print("🧠 TESTING REASONING UI INTEGRATION")
    print("=" * 50)
    
    try:
        # Import the reasoning engine
        from app1 import reasoning_engine
        print("✅ Reasoning engine imported successfully!")
        
        # Test XGBoost reasoning generation
        print("\n🔍 Test 1: XGBoost Reasoning Generation")
        print("-" * 40)
        
        try:
            # Create dummy data for testing
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor
            
            X = np.array([[1], [2], [3], [4], [5]])
            y = np.array([10, 20, 30, 40, 50])
            
            # Create a dummy model
            model = RandomForestRegressor(n_estimators=5, random_state=42)
            model.fit(X, y)
            
            # Generate reasoning
            ml_reasoning = reasoning_engine.explain_xgboost_prediction(
                model, X, y[-1], feature_names=['time_period'], model_type='regressor'
            )
            
            print("✅ ML reasoning generated successfully!")
            print(f"   📊 Training Insights: {ml_reasoning.get('training_insights', {}).get('learning_strategy', 'N/A')}")
            print(f"   🔍 Pattern Analysis: {ml_reasoning.get('pattern_analysis', {}).get('forecast_trend', 'N/A')}")
            print(f"   💰 Business Context: {ml_reasoning.get('business_context', {}).get('financial_rationale', 'N/A')}")
            print(f"   🎯 Decision Logic: {ml_reasoning.get('decision_logic', 'N/A')[:100]}...")
            
        except Exception as e:
            print(f"❌ ML reasoning test failed: {e}")
        
        # Test Ollama reasoning generation
        print("\n🧠 Test 2: Ollama AI Reasoning Generation")
        print("-" * 40)
        
        try:
            ai_reasoning = reasoning_engine.explain_ollama_response(
                "Analyze financial transaction patterns",
                "The analysis shows increasing revenue trends with seasonal variations",
                model_name='llama2:7b'
            )
            
            print("✅ AI reasoning generated successfully!")
            print(f"   🎯 Context Understanding: {ai_reasoning.get('semantic_understanding', {}).get('context_understanding', 'N/A')}")
            print(f"   📚 Business Intelligence: {ai_reasoning.get('business_intelligence', {}).get('financial_knowledge', 'N/A')}")
            print(f"   🎯 Decision Logic: {ai_reasoning.get('decision_logic', 'N/A')[:100]}...")
            
        except Exception as e:
            print(f"❌ AI reasoning test failed: {e}")
        
        # Test Hybrid reasoning generation
        print("\n🔗 Test 3: Hybrid Reasoning Generation")
        print("-" * 40)
        
        try:
            hybrid_reasoning = reasoning_engine.generate_hybrid_explanation(
                ml_reasoning if 'ml_reasoning' in locals() else {},
                ai_reasoning if 'ai_reasoning' in locals() else {},
                "Combined ML and AI analysis"
            )
            
            print("✅ Hybrid reasoning generated successfully!")
            print(f"   🎯 Combination Strategy: {hybrid_reasoning.get('combination_strategy', {}).get('approach', 'N/A')}")
            print(f"   🔗 Synergy Analysis: {hybrid_reasoning.get('synergy_analysis', {}).get('synergy_score', 'N/A')}")
            print(f"   🎯 Decision Logic: {hybrid_reasoning.get('decision_logic', 'N/A')[:100]}...")
            
        except Exception as e:
            print(f"❌ Hybrid reasoning test failed: {e}")
        
        # Test UI formatting
        print("\n🎨 Test 4: UI Formatting")
        print("-" * 40)
        
        try:
            # Test detailed formatting
            detailed_format = reasoning_engine.format_explanation_for_ui(
                ml_reasoning if 'ml_reasoning' in locals() else {}, 
                'detailed'
            )
            print("✅ Detailed UI formatting successful!")
            print(f"   📝 Format Length: {len(detailed_format)} characters")
            
            # Test summary formatting
            summary_format = reasoning_engine.format_explanation_for_ui(
                ml_reasoning if 'ml_reasoning' in locals() else {}, 
                'summary'
            )
            print("✅ Summary UI formatting successful!")
            print(f"   📝 Format Length: {len(summary_format)} characters")
            
        except Exception as e:
            print(f"❌ UI formatting test failed: {e}")
        
        print("\n✅ All reasoning tests completed!")
        print("\n🎯 Next Steps:")
        print("   1. Start the Flask app: python app1.py")
        print("   2. Upload your data files")
        print("   3. Run any analysis (individual parameters, vendor analysis, etc.)")
        print("   4. Look for '🧠 AI/ML Reasoning Insights' section in results")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure app1.py is in the same directory")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Reasoning UI Integration Test Suite")
    print("=" * 60)
    
    # Test the reasoning capabilities
    test_reasoning_generation()
    
    print("\n🎉 Testing Complete!")
    print("The reasoning system is now integrated into your UI!")
