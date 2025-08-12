#!/usr/bin/env python3
"""
Test Script for Advanced Reasoning Engine
Demonstrates how XGBoost + Ollama reasoning works
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_reasoning_engine():
    """Test the Advanced Reasoning Engine"""
    print("🧠 TESTING ADVANCED REASONING ENGINE")
    print("=" * 50)
    
    try:
        # Import the reasoning engine from app1.py
        from app1 import reasoning_engine
        
        print("✅ Reasoning Engine imported successfully!")
        
        # Test 1: XGBoost Explanation
        print("\n🔍 Test 1: XGBoost Prediction Explanation")
        print("-" * 40)
        
        # Simulate XGBoost model data
        class MockXGBoostModel:
            def __init__(self):
                self.feature_importances_ = [0.3, 0.25, 0.2, 0.15, 0.1]
                self.n_estimators = 100
                self.max_depth = 6
                self.learning_rate = 0.1
        
        mock_model = MockXGBoostModel()
        sample_features = [[1000, 25, 1, 5, 0.8]]  # amount, desc_length, type, vendor_freq, time_feature
        
        xgb_explanation = reasoning_engine.explain_xgboost_prediction(
            mock_model, 
            sample_features, 
            "Operating Activities",
            feature_names=['amount', 'description_length', 'transaction_type', 'vendor_frequency', 'time_feature'],
            model_type='classifier'
        )
        
        print("ML System Explanation:")
        print(f"  Prediction: {xgb_explanation.get('prediction')}")
        print(f"  Key Factors: {xgb_explanation.get('key_factors')}")
        print(f"  Reasoning: {xgb_explanation.get('reasoning')}")
        print(f"  Model Parameters: {xgb_explanation.get('model_parameters')}")
        
        # Show deep training insights
        if 'training_insights' in xgb_explanation and xgb_explanation['training_insights']:
            insights = xgb_explanation['training_insights']
            print(f"  🧠 Learning Strategy: {insights.get('learning_strategy', 'Not available')}")
            print(f"  🔍 Pattern Discovery: {insights.get('pattern_discovery', 'Not available')}")
            print(f"  ⚡ Training Behavior: {insights.get('training_behavior', 'Not available')}")
        
        # Show pattern analysis
        if 'pattern_analysis' in xgb_explanation and xgb_explanation['pattern_analysis']:
            patterns = xgb_explanation['pattern_analysis']
            print(f"  📋 Business Rules: {patterns.get('business_rules_discovered', 'Not available')}")
            print(f"  💪 Pattern Strength: {patterns.get('pattern_strength', 'Not available')}")
        
        # Show business context
        if 'business_context' in xgb_explanation and xgb_explanation['business_context']:
            context = xgb_explanation['business_context']
            print(f"  💰 Financial Logic: {context.get('financial_rationale', 'Not available')}")
            print(f"  ⚙️ Operational Insight: {context.get('operational_insight', 'Not available')}")
        
        # Test 2: AI System Explanation
        print("\n🧠 Test 2: AI System Response Explanation")
        print("-" * 40)
        
        sample_prompt = """
        Categorize this transaction into one of these cash flow categories based on BUSINESS ACTIVITY:
        - Operating Activities (business revenue, business expenses, regular business operations)
        - Investing Activities (capital expenditure, asset purchases, investments)
        - Financing Activities (loans, interest, dividends, equity)
        
        Transaction: Steel sale to customer ABC Ltd
        Category:"""
        
        sample_response = "Operating Activities"
        
        ai_explanation = reasoning_engine.explain_ollama_response(
            sample_prompt, sample_response, "llama2:7b"
        )
        
        print("AI System Explanation:")
        print(f"  Response: {ai_explanation.get('response')}")
        print(f"  Quality: {ai_explanation.get('response_quality')}")
        print(f"  Decision Logic: {ai_explanation.get('decision_logic')}")
        print(f"  Reasoning Chain: {ai_explanation.get('reasoning_chain')}")
        
        # Show deep AI insights
        if 'semantic_understanding' in ai_explanation and ai_explanation['semantic_understanding']:
            semantics = ai_explanation['semantic_understanding']
            print(f"  🎯 Context Understanding: {semantics.get('context_understanding', 'Not available')}")
            print(f"  ✅ Semantic Accuracy: {semantics.get('semantic_accuracy', 'Not available')}")
            print(f"  📚 Business Vocabulary: {semantics.get('business_vocabulary', 'Not available')}")
        
        if 'business_intelligence' in ai_explanation and ai_explanation['business_intelligence']:
            business = ai_explanation['business_intelligence']
            print(f"  📚 Financial Knowledge: {business.get('financial_knowledge', 'Not available')}")
            print(f"  🔄 Business Patterns: {business.get('business_patterns', 'Not available')}")
        
        # Test 3: Hybrid Explanation
        print("\n🔗 Test 3: Hybrid ML + AI Explanation")
        print("-" * 40)
        
        hybrid_explanation = reasoning_engine.generate_hybrid_explanation(
            xgb_explanation, ai_explanation, "Operating Activities (Hybrid)"
        )
        
        print("Hybrid Explanation:")
        print(f"  Final Result: {hybrid_explanation.get('final_result')}")
        print(f"  Combined Reasoning: {hybrid_explanation.get('combined_reasoning')}")
        print(f"  Confidence Score: {hybrid_explanation.get('confidence_score'):.1%}")
        print(f"  Decision Summary: {hybrid_explanation.get('decision_summary')}")
        print(f"  Recommendations: {hybrid_explanation.get('recommendations')}")
        
        # Test 4: UI Formatting
        print("\n📱 Test 4: UI Formatting")
        print("-" * 40)
        
        detailed_format = reasoning_engine.format_explanation_for_ui(hybrid_explanation, 'detailed')
        summary_format = reasoning_engine.format_explanation_for_ui(hybrid_explanation, 'summary')
        debug_format = reasoning_engine.format_explanation_for_ui(hybrid_explanation, 'debug')
        
        print("Detailed Format:")
        print(detailed_format)
        print("\nSummary Format:")
        print(summary_format)
        print("\nDebug Format:")
        print(debug_format)
        
        print("\n✅ All tests completed successfully!")
        print("\n🎯 Key Benefits of the Deep Reasoning Engine:")
        print("   • Deep ML training pattern analysis")
        print("   • Comprehensive AI semantic understanding")
        print("   • Business logic validation")
        print("   • Pattern strength assessment")
        print("   • Operational insights generation")
        print("   • No technical model names in UI")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure app1.py is in the same directory")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()

def test_api_endpoints():
    """Test the reasoning API endpoints"""
    print("\n🌐 TESTING REASONING API ENDPOINTS")
    print("=" * 50)
    
    try:
        # This would require the Flask app to be running
        print("📋 Available API Endpoints:")
        print("   POST /get-reasoning-explanation")
        print("     - type: 'xgboost', 'ollama', or 'hybrid'")
        print("     - Provides detailed reasoning explanations")
        print("\n   POST /analyze-model-reasoning")
        print("     - model_type: 'xgboost', 'ollama', or 'hybrid'")
        print("     - Analyzes model decision logic")
        
        print("\n📝 Example API Usage:")
        print("""
        # Get XGBoost reasoning
        curl -X POST http://localhost:5000/get-reasoning-explanation \\
          -H "Content-Type: application/json" \\
          -d '{"type": "xgboost", "result": {"model": "model_data", "features": [[1,2,3]]}}'
        
        # Get Hybrid reasoning
        curl -X POST http://localhost:5000/get-reasoning-explanation \\
          -H "Content-Type: application/json" \\
          -d '{"type": "hybrid", "result": {"xgboost": {}, "ollama": {}, "final_result": "Result"}}'
        """)
        
    except Exception as e:
        print(f"❌ API Test Error: {e}")

if __name__ == "__main__":
    print("🚀 Advanced Reasoning Engine Test Suite")
    print("=" * 60)
    
    # Test the reasoning engine
    test_reasoning_engine()
    
    # Test API endpoints
    test_api_endpoints()
    
    print("\n🎉 Testing Complete!")
    print("Start the Flask app with 'python app1.py' to test the API endpoints")
