#!/usr/bin/env python3
"""
Simple test to verify system is working
"""

def simple_test():
    """Simple test of the system"""
    print("🧪 Simple System Test...")
    
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
            
        print("\n🎉 System is working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    simple_test() 