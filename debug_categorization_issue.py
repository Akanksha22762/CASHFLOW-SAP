#!/usr/bin/env python3
"""
Debug script to identify categorization inconsistency
"""

def debug_categorization_issue():
    """Debug why categorization is inconsistent"""
    print("🔍 DEBUGGING CATEGORIZATION INCONSISTENCY")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("Infrastructure Development - Warehouse Construction - 4717 sq ft", 3709289.81),
        ("Infrastructure Development - Warehouse Construction - 3356 sq ft - Tax Season", -3200742.66)
    ]
    
    for i, (desc, amount) in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {desc}")
        print(f"   Amount: {amount}")
        
        # Test 1: Rule-based categorization
        try:
            from app1 import categorize_transaction_perfect
            rule_result = categorize_transaction_perfect(desc, amount)
            print(f"   Rule-based: {rule_result}")
        except Exception as e:
            print(f"   Rule-based ERROR: {e}")
        
        # Test 2: XGBoost categorization
        try:
            from app1 import lightweight_ai
            if lightweight_ai.is_trained:
                xgb_result = lightweight_ai.categorize_transaction_ml(desc, amount)
                print(f"   XGBoost: {xgb_result}")
            else:
                print(f"   XGBoost: Not trained")
        except Exception as e:
            print(f"   XGBoost ERROR: {e}")
        
        # Test 3: Hybrid categorization
        try:
            from app1 import hybrid_categorize_transaction
            hybrid_result = hybrid_categorize_transaction(desc, amount)
            print(f"   Hybrid: {hybrid_result}")
        except Exception as e:
            print(f"   Hybrid ERROR: {e}")
        
        # Test 4: Check which method is being used in the main system
        try:
            # Simulate what happens in the main system
            from app1 import hybrid_categorize_transaction
            final_result = hybrid_categorize_transaction(desc, amount)
            
            # Extract the method used
            if " (XGBoost)" in final_result:
                method = "XGBoost"
            elif " (Ollama)" in final_result:
                method = "Ollama"
            elif " (Rules)" in final_result:
                method = "Rules"
            else:
                method = "Unknown"
            
            print(f"   Final Result: {final_result}")
            print(f"   Method Used: {method}")
            
        except Exception as e:
            print(f"   Final Result ERROR: {e}")

if __name__ == "__main__":
    debug_categorization_issue() 