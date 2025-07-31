#!/usr/bin/env python3
"""
Quick test for training system fixes
"""

import pandas as pd
import numpy as np

def quick_training_test():
    """Quick test of training system"""
    print("🧪 Quick Training System Test...")
    
    # Create simple test data
    test_data = pd.DataFrame({
        'Description': [
            'Infrastructure Development',
            'Customer Payment',
            'Equipment Purchase',
            'Bank Loan',
            'Salary Payment'
        ],
        'Amount': [1000000, 500000, 2000000, 1500000, 50000],
        'Category': [
            'Investing Activities',
            'Operating Activities', 
            'Investing Activities',
            'Financing Activities',
            'Operating Activities'
        ]
    })
    
    print(f"✅ Test data: {len(test_data)} samples")
    
    try:
        from app1 import lightweight_ai
        
        # Test training
        print("🤖 Testing training...")
        result = lightweight_ai.train_transaction_classifier(test_data)
        
        if result:
            print("✅ Training successful!")
            print(f"   Is trained: {lightweight_ai.is_trained}")
            
            # Test categorization
            if lightweight_ai.is_trained:
                test_result = lightweight_ai.categorize_transaction_ml("Infrastructure Development", 1000000)
                print(f"✅ Categorization: {test_result}")
                
                if '(XGBoost)' in test_result:
                    print("✅ SUCCESS: XGBoost model working!")
                else:
                    print("❌ FAILURE: Not using XGBoost")
            else:
                print("❌ Model not trained")
        else:
            print("❌ Training failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = quick_training_test()
    if success:
        print("\n🎉 Quick Training Test PASSED!")
    else:
        print("\n❌ Quick Training Test FAILED!") 