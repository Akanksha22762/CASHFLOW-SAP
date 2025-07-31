#!/usr/bin/env python3
"""
Quick test for hybrid system
"""

def quick_hybrid_test():
    """Quick test of hybrid system"""
    print("🧪 Quick Hybrid System Test...")
    
    try:
        from app1 import hybrid_categorize_transaction
        
        # Test one transaction
        result = hybrid_categorize_transaction("Infrastructure Development", 1000000)
        print(f"Result: {result}")
        
        if '(XGBoost)' in result:
            print("✅ Using XGBoost")
        elif '(Ollama)' in result:
            print("✅ Using Ollama")
        elif '(Rules)' in result:
            print("⚠️ Using Rules")
        else:
            print("❌ Unknown method")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_hybrid_test() 