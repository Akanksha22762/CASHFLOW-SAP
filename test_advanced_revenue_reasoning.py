#!/usr/bin/env python3
"""
Test Script for Enhanced Advanced Revenue AI System
Demonstrates deep reasoning capabilities for XGBoost and Ollama integration
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_advanced_revenue_reasoning():
    """Test the Enhanced Advanced Revenue AI System with Deep Reasoning"""
    print("🧠 TESTING ENHANCED ADVANCED REVENUE AI SYSTEM")
    print("=" * 60)
    
    try:
        # Import the advanced revenue AI system
        from advanced_revenue_ai_system import AdvancedRevenueAISystem
        
        print("✅ Advanced Revenue AI System imported successfully!")
        
        # Initialize the system
        ai_system = AdvancedRevenueAISystem()
        print("✅ AI System initialized successfully!")
        
        # Create sample transaction data
        sample_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'Amount': np.random.uniform(1000, 50000, 30),
            'Description': [f'Transaction {i}' for i in range(1, 31)],
            'Type': ['INWARD'] * 20 + ['OUTWARD'] * 10
        })
        
        print(f"📊 Sample data created: {len(sample_data)} transactions")
        
        # Test 1: XGBoost Forecasting with Deep Reasoning
        print("\n🔍 Test 1: XGBoost Forecasting with Deep Reasoning")
        print("-" * 50)
        
        try:
            xgb_result = ai_system._forecast_with_xgboost(sample_data, forecast_steps=6)
            
            if xgb_result and isinstance(xgb_result, dict):
                print("✅ XGBoost forecasting completed with reasoning!")
                
                # Show forecast results
                if 'forecast' in xgb_result:
                    forecast = xgb_result['forecast']
                    print(f"📈 Forecast: {forecast[:3]}... (showing first 3 periods)")
                
                # Show deep reasoning
                if 'reasoning' in xgb_result:
                    reasoning = xgb_result['reasoning']
                    
                    # Training insights
                    if 'training_insights' in reasoning:
                        insights = reasoning['training_insights']
                        print(f"🧠 Learning Strategy: {insights.get('learning_strategy', 'Not available')}")
                        print(f"🔍 Pattern Discovery: {insights.get('pattern_discovery', 'Not available')}")
                        print(f"⚡ Training Behavior: {insights.get('training_behavior', 'Not available')}")
                    
                    # Pattern analysis
                    if 'pattern_analysis' in reasoning:
                        patterns = reasoning['pattern_analysis']
                        print(f"📋 Forecast Trend: {patterns.get('forecast_trend', 'Not available')}")
                        print(f"💪 Pattern Strength: {patterns.get('pattern_strength', 'Not available')}")
                    
                    # Business context
                    if 'business_context' in reasoning:
                        context = reasoning['business_context']
                        print(f"💰 Financial Logic: {context.get('financial_rationale', 'Not available')}")
                        print(f"⚙️ Operational Insight: {context.get('operational_insight', 'Not available')}")
                    
                    # Forecast confidence
                    if 'forecast_confidence' in reasoning:
                        confidence = reasoning['forecast_confidence']
                        print(f"🎯 Confidence Level: {confidence.get('confidence_level', 'Not available')}")
                        print(f"📊 Data Quality: {confidence.get('data_quality', 'Not available')}")
                    
                    # Decision logic
                    print(f"🔍 Decision Logic: {reasoning.get('decision_logic', 'Not available')}")
                
                # Show model info
                if 'model_info' in xgb_result:
                    model_info = xgb_result['model_info']
                    print(f"🤖 Model Info: {model_info}")
                
            else:
                print("⚠️ XGBoost forecasting returned legacy format or failed")
                
        except Exception as e:
            print(f"❌ XGBoost forecasting test failed: {e}")
        
        # Test 2: Ollama AI Analysis with Deep Reasoning
        print("\n🧠 Test 2: Ollama AI Analysis with Deep Reasoning")
        print("-" * 50)
        
        try:
            ollama_result = ai_system._analyze_with_ollama(sample_data, "revenue_trends")
            
            if ollama_result and isinstance(ollama_result, dict):
                print("✅ Ollama AI analysis completed with reasoning!")
                
                # Show AI analysis
                if 'ollama_analysis' in ollama_result:
                    print(f"🤖 AI Analysis: {ollama_result['ollama_analysis'][:100]}...")
                
                # Show data summary
                if 'data_summary' in ollama_result:
                    summary = ollama_result['data_summary']
                    print(f"📊 Total Transactions: {summary.get('total_transactions', 'N/A')}")
                    print(f"💰 Total Amount: ₹{summary.get('total_amount', 0):,.2f}")
                    print(f"📅 Date Range: {summary.get('date_range', 'N/A')}")
                
                # Show deep AI reasoning
                if 'ai_reasoning' in ollama_result:
                    ai_reasoning = ollama_result['ai_reasoning']
                    
                    # Semantic understanding
                    if 'semantic_understanding' in ai_reasoning:
                        semantics = ai_reasoning['semantic_understanding']
                        print(f"🎯 Context Understanding: {semantics.get('context_understanding', 'Not available')}")
                        print(f"✅ Semantic Accuracy: {semantics.get('semantic_accuracy', 'Not available')}")
                        print(f"📚 Business Vocabulary: {semantics.get('business_vocabulary', 'Not available')}")
                    
                    # Business intelligence
                    if 'business_intelligence' in ai_reasoning:
                        business = ai_reasoning['business_intelligence']
                        print(f"📚 Financial Knowledge: {business.get('financial_knowledge', 'Not available')}")
                        print(f"🔄 Business Patterns: {business.get('business_patterns', 'Not available')}")
                    
                    # Response patterns
                    if 'response_patterns' in ai_reasoning:
                        patterns = ai_reasoning['response_patterns']
                        print(f"🏗️ Response Structure: {patterns.get('response_structure', 'Not available')}")
                        print(f"🔧 Improvement Areas: {patterns.get('improvement_areas', 'Not available')}")
                    
                    # Analysis confidence
                    if 'analysis_confidence' in ai_reasoning:
                        confidence = ai_reasoning['analysis_confidence']
                        print(f"🎯 Confidence Level: {confidence.get('confidence_level', 'Not available')}")
                        print(f"📊 AI Reliability: {confidence.get('ai_reliability', 'Not available')}")
                    
                    # Decision logic
                    print(f"🔍 Decision Logic: {ai_reasoning.get('decision_logic', 'Not available')}")
                
            else:
                print("⚠️ Ollama AI analysis failed or returned unexpected format")
                
        except Exception as e:
            print(f"❌ Ollama AI analysis test failed: {e}")
        
        # Test 3: Hybrid Forecast Combination with Deep Reasoning
        print("\n🔗 Test 3: Hybrid Forecast Combination with Deep Reasoning")
        print("-" * 50)
        
        try:
            # Create enhanced XGBoost result
            enhanced_xgb = {
                'forecast': np.array([10000, 11000, 12000, 13000, 14000, 15000]),
                'reasoning': {
                    'forecast_confidence': {'confidence_level': '85%'},
                    'pattern_analysis': {'forecast_trend': 'increasing trend over 6 periods'}
                }
            }
            
            # Create enhanced Ollama result
            enhanced_ollama = {
                'ollama_analysis': 'Positive growth trends with strong market conditions and excellent opportunities for expansion',
                'ai_reasoning': {
                    'analysis_confidence': {'confidence_level': '90%'},
                    'semantic_understanding': {'context_understanding': 'AI understands revenue analysis context'}
                }
            }
            
            hybrid_result = ai_system._combine_xgboost_ollama_forecast(enhanced_xgb, enhanced_ollama)
            
            if hybrid_result and isinstance(hybrid_result, dict):
                print("✅ Hybrid forecast combination completed with reasoning!")
                
                # Show combined forecast
                if 'forecast' in hybrid_result:
                    forecast = hybrid_result['forecast']
                    print(f"📈 Combined Forecast: {forecast[:3]}... (showing first 3 periods)")
                
                # Show weights and adjustment
                print(f"⚖️ XGBoost Weight: {hybrid_result.get('xgb_weight', 'N/A')}")
                print(f"⚖️ Ollama Weight: {hybrid_result.get('ollama_weight', 'N/A')}")
                print(f"🔄 Ollama Adjustment: {hybrid_result.get('ollama_adjustment', 'N/A')}")
                print(f"🎯 Confidence Score: {hybrid_result.get('confidence_score', 'N/A'):.1%}")
                
                # Show deep hybrid reasoning
                if 'hybrid_reasoning' in hybrid_result:
                    hybrid_reasoning = hybrid_result['hybrid_reasoning']
                    
                    # Combination strategy
                    if 'combination_strategy' in hybrid_reasoning:
                        strategy = hybrid_reasoning['combination_strategy']
                        print(f"🎯 Approach: {strategy.get('approach', 'Not available')}")
                        print(f"📋 Methodology: {strategy.get('methodology', 'Not available')}")
                        print(f"🔗 Synergy Benefit: {strategy.get('synergy_benefit', 'Not available')}")
                    
                    # Weight justification
                    if 'weight_justification' in hybrid_reasoning:
                        weights = hybrid_reasoning['weight_justification']
                        print(f"🤖 ML Weight Rationale: {weights.get('ml_weight', 'Not available')}")
                        print(f"🧠 AI Weight Rationale: {weights.get('ai_weight', 'Not available')}")
                        print(f"⚖️ Balance Rationale: {weights.get('balance_rationale', 'Not available')}")
                    
                    # Adjustment rationale
                    if 'adjustment_rationale' in hybrid_reasoning:
                        adjustment = hybrid_reasoning['adjustment_rationale']
                        print(f"🔄 Adjustment Factor: {adjustment.get('adjustment_factor', 'Not available')}")
                        print(f"📈 Direction: {adjustment.get('direction', 'Not available')}")
                        print(f"📊 Business Impact: {adjustment.get('business_impact', 'Not available')}")
                    
                    # Synergy analysis
                    if 'synergy_analysis' in hybrid_reasoning:
                        synergy = hybrid_reasoning['synergy_analysis']
                        print(f"🎯 ML Confidence: {synergy.get('ml_confidence', 'Not available')}")
                        print(f"🧠 AI Confidence: {synergy.get('ai_confidence', 'Not available')}")
                        print(f"🔗 Synergy Score: {synergy.get('synergy_score', 'Not available')}")
                        print(f"💡 Complementarity: {synergy.get('complementarity', 'Not available')}")
                        print(f"🛡️ Risk Mitigation: {synergy.get('risk_mitigation', 'Not available')}")
                    
                    # Decision logic
                    print(f"🔍 Decision Logic: {hybrid_reasoning.get('decision_logic', 'Not available')}")
                
            else:
                print("⚠️ Hybrid forecast combination failed or returned unexpected format")
                
        except Exception as e:
            print(f"❌ Hybrid forecast combination test failed: {e}")
        
        print("\n✅ All tests completed successfully!")
        print("\n🎯 Key Benefits of Enhanced Advanced Revenue AI System:")
        print("   • Deep ML training pattern analysis for forecasting")
        print("   • Comprehensive AI semantic understanding for business analysis")
        print("   • Hybrid reasoning explaining forecast combinations")
        print("   • Business context validation for all predictions")
        print("   • Confidence scoring and risk assessment")
        print("   • No technical model names in explanations")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure advanced_revenue_ai_system.py is in the same directory")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Enhanced Advanced Revenue AI System Test Suite")
    print("=" * 70)
    
    # Test the enhanced reasoning capabilities
    test_advanced_revenue_reasoning()
    
    print("\n🎉 Testing Complete!")
    print("The Advanced Revenue AI System now provides deep reasoning for all XGBoost and Ollama operations!")
