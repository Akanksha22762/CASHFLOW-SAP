#!/usr/bin/env python3
"""
Minimal test to check system
"""

print("🧪 Starting minimal test...")

try:
    print("✅ Importing modules...")
    from app1 import hybrid_categorize_transaction
    
    print("✅ Testing hybrid categorization...")
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
        
    print("🎉 Test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 