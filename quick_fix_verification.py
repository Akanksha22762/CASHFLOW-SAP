#!/usr/bin/env python3
"""
QUICK FIX VERIFICATION - Test if all systems are working
"""

def test_xgboost():
    """Test XGBoost functionality"""
    print("🧪 Testing XGBoost...")
    
    try:
        from app1 import lightweight_ai
        
        # Test if model is trained
        if lightweight_ai.is_trained:
            print("✅ XGBoost model is trained")
            
            # Test categorization
            result = lightweight_ai.categorize_transaction_ml("Infrastructure Development", 1000000)
            print(f"✅ XGBoost test result: {result}")
            
            if "Not-Trained" not in result:
                return True
            else:
                print("❌ XGBoost model not properly trained")
                return False
        else:
            print("❌ XGBoost model not trained")
            return False
            
    except Exception as e:
        print(f"❌ XGBoost error: {e}")
        return False

def test_ollama():
    """Test Ollama functionality"""
    print("🧪 Testing Ollama...")
    
    try:
        from ollama_simple_integration import OllamaSimpleIntegration
        
        ollama = OllamaSimpleIntegration()
        
        if ollama.is_available:
            print("✅ Ollama is available")
            print(f"✅ Available models: {ollama.available_models}")
            
            # Test with working model
            test_response = ollama.simple_ollama("Test prompt", model="llama2:7b", max_tokens=20)
            
            if test_response:
                print("✅ Ollama API working")
                return True
            else:
                print("❌ Ollama API not working")
                return False
        else:
            print("❌ Ollama not available")
            return False
            
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return False

def test_training():
    """Test training functionality"""
    print("🧪 Testing Training...")
    
    try:
        from app1 import lightweight_ai
        import pandas as pd
        from datetime import datetime
        
        # Create minimal training data
        training_data = pd.DataFrame({
            'Description': ['Test Infrastructure', 'Test Customer Payment', 'Test Investment'],
            'Amount': [1000000, 2000000, 500000],
            'Category': ['Investing Activities', 'Operating Activities', 'Financing Activities'],
            'Date': [datetime.now(), datetime.now(), datetime.now()],
            'Type': ['Credit', 'Credit', 'Credit']
        })
        
        # Try to train
        success = lightweight_ai.train_transaction_classifier(training_data)
        
        if success:
            print("✅ Training successful")
            return True
        else:
            print("❌ Training failed")
            return False
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 QUICK FIX VERIFICATION")
    print("=" * 40)
    
    results = {}
    
    # Test each system
    results['xgboost'] = test_xgboost()
    results['ollama'] = test_ollama()
    results['training'] = test_training()
    
    # Summary
    print("\n📊 VERIFICATION RESULTS:")
    print("=" * 30)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test.replace('_', ' ').title()}: {status}")
    
    print(f"\n📈 OVERALL: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL SYSTEMS WORKING PERFECTLY!")
        return True
    elif passed_tests >= total_tests * 0.7:
        print("\n✅ MOST SYSTEMS WORKING - MINOR ISSUES")
        return True
    else:
        print("\n❌ MULTIPLE SYSTEMS NEED FIXES")
        return False

if __name__ == "__main__":
    main() 