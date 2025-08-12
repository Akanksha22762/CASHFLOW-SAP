#!/usr/bin/env python3
"""
Test API with Data
Simulates the actual API call with proper data to debug reasoning
"""

import sys
import os
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_api_with_data():
    """Test the API functions directly with data"""
    print("🧠 TESTING API WITH DATA")
    print("=" * 50)
    
    try:
        # Import the necessary components
        from app1 import run_parameter_analysis, uploaded_data
        import pandas as pd
        import numpy as np
        
        print("✅ Components imported successfully!")
        
        # Create test data
        print("\n🔍 Creating test data...")
        
        # Create a sample DataFrame
        test_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'Description': ['Test transaction ' + str(i) for i in range(1, 11)],
            'Amount': [100 + i * 10 for i in range(10)],
            'Category': ['Operating'] * 10
        })
        
        print(f"✅ Test data created: {test_data.shape}")
        print(f"   Columns: {list(test_data.columns)}")
        print(f"   Sample: {test_data.head(2).to_dict()}")
        
        # Set the global uploaded_data
        print("\n🔍 Setting global uploaded_data...")
        
        # Import the global variable
        import app1
        app1.uploaded_data = {'bank_df': test_data}
        
        print("✅ Global uploaded_data set successfully!")
        
        # Test the reasoning generation directly
        print("\n🔍 Testing reasoning generation directly...")
        
        try:
            # Import the reasoning engine
            from app1 import reasoning_engine
            
            # Generate reasoning explanations for the analysis
            reasoning_explanations = {}
            
            # Generate ML reasoning (XGBoost)
            try:
                print("🧠 Generating ML reasoning...")
                
                # Check if we have Amount column for ML reasoning
                if 'Amount' in test_data.columns and len(test_data) > 0:
                    # Create dummy model for reasoning using transaction amounts
                    from sklearn.ensemble import RandomForestRegressor
                    amounts = test_data['Amount'].values.reshape(-1, 1)
                    X = np.arange(len(amounts)).reshape(-1, 1)
                    y = amounts.flatten()
                    
                    if len(y) > 1:
                        dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
                        dummy_model.fit(X, y)
                        
                        # Generate ML reasoning
                        ml_reasoning = reasoning_engine.explain_xgboost_prediction(
                            dummy_model, X, y[-1] if len(y) > 0 else 0, 
                            feature_names=['transaction_sequence'], model_type='regressor'
                        )
                        reasoning_explanations['ml_analysis'] = ml_reasoning
                        print("✅ ML reasoning generated successfully!")
                        print(f"   📊 Keys: {list(ml_reasoning.keys())}")
                    else:
                        print("⚠️ Not enough data for ML reasoning")
                else:
                    print("⚠️ No Amount column or empty data")
                    
            except Exception as e:
                print(f"❌ ML reasoning generation failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Generate AI reasoning (Ollama)
            try:
                print("🧠 Generating AI reasoning...")
                
                # Create a sample prompt for AI reasoning
                sample_description = test_data['Description'].iloc[0] if len(test_data) > 0 else "Financial transaction"
                ai_prompt = f"Analyze this test data: {sample_description}"
                
                # Generate AI reasoning
                ai_reasoning = reasoning_engine.explain_ollama_response(
                    ai_prompt, 
                    f"Analysis of test data shows patterns in financial data",
                    model_name='llama2:7b'
                )
                reasoning_explanations['ai_analysis'] = ai_reasoning
                print("✅ AI reasoning generated successfully!")
                print(f"   📊 Keys: {list(ai_reasoning.keys())}")
                
            except Exception as e:
                print(f"❌ AI reasoning generation failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Generate hybrid reasoning
            try:
                print("🧠 Generating hybrid reasoning...")
                
                hybrid_reasoning = reasoning_engine.generate_hybrid_explanation(
                    reasoning_explanations.get('ml_analysis', {}),
                    reasoning_explanations.get('ai_analysis', {}),
                    f"Combined analysis of test data"
                )
                reasoning_explanations['hybrid_analysis'] = hybrid_reasoning
                print("✅ Hybrid reasoning generated successfully!")
                print(f"   📊 Keys: {list(hybrid_reasoning.keys())}")
                
            except Exception as e:
                print(f"❌ Hybrid reasoning generation failed: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n🧠 Final reasoning explanations: {list(reasoning_explanations.keys())}")
            
            # Test response creation
            print("\n🔍 Testing response creation...")
            
            response_data = {
                'status': 'success',
                'results': {'test': 'data'},
                'parameter_type': 'test_parameter',
                'processing_time': '1.0s',
                'ai_usage': '100% (XGBoost + Ollama)',
                'vendor_name': None,
                'transactions': [],
                'transaction_count': 0,
                'total_inflow': 0,
                'total_outflow': 0,
                'net_cash_flow': 0
            }
            
            # Add reasoning explanations if available
            if reasoning_explanations:
                response_data['reasoning_explanations'] = reasoning_explanations
                print(f"🧠 Added reasoning explanations to response: {list(reasoning_explanations.keys())}")
            else:
                print("⚠️ No reasoning explanations available to add to response")
            
            print(f"🔍 Final response keys: {list(response_data.keys())}")
            print(f"🔍 Reasoning explanations in response: {'reasoning_explanations' in response_data}")
            
            print("\n✅ All tests completed successfully!")
            
        except Exception as e:
            print(f"❌ Reasoning test failed: {e}")
            import traceback
            traceback.print_exc()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure app1.py is in the same directory")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 API with Data Test")
    print("=" * 60)
    
    # Test the API with data
    test_api_with_data()
    
    print("\n🎉 Testing Complete!")
