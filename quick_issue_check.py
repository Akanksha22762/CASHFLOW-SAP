#!/usr/bin/env python3
"""
Quick check for specific issues
"""

def quick_issue_check():
    """Quick check for specific issues"""
    print("🔍 QUICK ISSUE CHECK...")
    
    try:
        from app1 import hybrid_categorize_transaction, lightweight_ai
        from ollama_simple_integration import check_ollama_availability
        
        print("📋 System Status:")
        print(f"   Ollama Available: {check_ollama_availability()}")
        print(f"   XGBoost Trained: {lightweight_ai.is_trained}")
        
        # Test 1: Check if system falls back to rules when XGBoost not trained
        print("\n🧪 Test 1: Untrained XGBoost (should fall back to Ollama/Rules)")
        lightweight_ai.is_trained = False
        
        result = hybrid_categorize_transaction("Infrastructure Development", 1000000)
        print(f"   Result: {result}")
        
        if '(XGBoost)' in result:
            print("   ✅ Using XGBoost (unexpected)")
        elif '(Ollama)' in result:
            print("   ✅ Using Ollama")
        elif '(Rules)' in result:
            print("   ⚠️ Using Rules (fallback)")
        else:
            print("   ❌ Unknown method")
        
        # Test 2: Check if Ollama is hanging
        print("\n🧪 Test 2: Ollama timeout check")
        try:
            from ollama_simple_integration import simple_ollama
            test_prompt = "Categorize this transaction: Infrastructure Development"
            result = simple_ollama(test_prompt, "llama2:7b", max_tokens=20)
            if result:
                print("   ✅ Ollama working")
            else:
                print("   ❌ Ollama returned None")
        except Exception as e:
            print(f"   ❌ Ollama Error: {e}")
        
        print("\n📊 Issue Summary:")
        print("   - If you see 'Using Rules', that's the fallback issue")
        print("   - If you see 'Ollama Error', that's the timeout issue")
        print("   - If you see 'Using XGBoost' when not trained, that's a logic issue")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_issue_check() 