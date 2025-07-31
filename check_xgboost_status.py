#!/usr/bin/env python3
"""
CHECK XGBOOST STATUS - Verify if XGBoost is actually trained
"""

def check_xgboost_status():
    """Check the actual XGBoost training status"""
    print("🔍 CHECKING XGBOOST STATUS...")
    print("=" * 50)
    
    try:
        from app1 import lightweight_ai
        
        print(f"📊 Model trained: {lightweight_ai.is_trained}")
        print(f"📊 Features available: {len(lightweight_ai.feature_names) if hasattr(lightweight_ai, 'feature_names') else 'None'}")
        
        # Test categorization
        test_cases = [
            ("Infrastructure Development", 1000000),
            ("Customer Payment", 2000000),
            ("Investment Liquidation", 500000)
        ]
        
        print("\n🧪 TESTING CATEGORIZATION:")
        for desc, amount in test_cases:
            result = lightweight_ai.categorize_transaction_ml(desc, amount)
            print(f"   {desc} → {result}")
            
            # Check if result contains "Not-Trained"
            if "Not-Trained" in result:
                print(f"   ❌ Model not properly trained")
            else:
                print(f"   ✅ Model working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking XGBoost: {e}")
        return False

if __name__ == "__main__":
    check_xgboost_status() 