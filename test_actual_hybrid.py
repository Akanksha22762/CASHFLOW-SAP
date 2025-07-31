#!/usr/bin/env python3
"""
Test to see exactly what the hybrid system is returning
"""

def test_actual_hybrid():
    """Test the actual hybrid system step by step"""
    print("🔍 Testing Actual Hybrid System Step by Step...")
    
    try:
        from app1 import hybrid_categorize_transaction, lightweight_ai
        from ollama_simple_integration import check_ollama_availability, simple_ollama
        
        print("📋 System Status:")
        print(f"   Ollama Available: {check_ollama_availability()}")
        print(f"   XGBoost Trained: {lightweight_ai.is_trained}")
        
        # Test XGBoost directly
        print("\n🧪 Testing XGBoost Directly:")
        if lightweight_ai.is_trained:
            result = lightweight_ai.categorize_transaction_ml("Infrastructure Development", 1000000)
            print(f"   XGBoost Result: '{result}'")
            
            # Check what the hybrid function expects
            if "Error" not in result and "Not-Trained" not in result and "No-Prediction" not in result:
                print("   ✅ XGBoost result should be accepted by hybrid")
            else:
                print("   ❌ XGBoost result will be rejected by hybrid")
        else:
            print("   ⚠️ XGBoost not trained")
        
        # Test Ollama directly
        print("\n🧪 Testing Ollama Directly:")
        try:
            prompt = f"""
            Categorize this transaction into one of these cash flow categories:
            - Operating Activities (revenue, expenses, regular business operations)
            - Investing Activities (capital expenditure, asset purchases, investments)
            - Financing Activities (loans, interest, dividends, equity)
            
            Transaction: Infrastructure Development
            Category:"""
            
            result = simple_ollama(prompt, "llama2:7b", max_tokens=20)
            print(f"   Ollama Result: '{result}'")
            
            if result:
                category = result.strip().split('\n')[0].strip()
                print(f"   Parsed Category: '{category}'")
                if category in ["Operating Activities", "Investing Activities", "Financing Activities"]:
                    print("   ✅ Ollama result should be accepted by hybrid")
                else:
                    print("   ❌ Ollama result will be rejected by hybrid")
        except Exception as e:
            print(f"   ❌ Ollama Error: {e}")
        
        # Test hybrid function
        print("\n🧪 Testing Hybrid Function:")
        result = hybrid_categorize_transaction("Infrastructure Development", 1000000)
        print(f"   Hybrid Result: '{result}'")
        
        if '(XGBoost)' in result:
            print("   ✅ Using XGBoost")
        elif '(Ollama)' in result:
            print("   ✅ Using Ollama")
        elif '(Rules)' in result:
            print("   ⚠️ Using Rules")
        else:
            print("   ❌ Unknown method")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_actual_hybrid() 