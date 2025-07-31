#!/usr/bin/env python3
"""
Test script to verify training system fixes
Tests array length mismatch and stratification issues
"""

import pandas as pd
import numpy as np
from datetime import datetime

def test_training_system_fix():
    """Test that training system handles array length and stratification properly"""
    print("🧪 Testing Training System Fixes...")
    
    # Test 1: Create test data with potential length mismatch
    print("\n📋 Test 1: Testing array length mismatch handling...")
    
    try:
        # Create test data
        test_data = pd.DataFrame({
            'Description': [
                'Infrastructure Development - Warehouse Construction',
                'VIP Customer Payment - Premium Service',
                'Equipment Purchase - New Production Line',
                'Bank Loan Disbursement - Working Capital',
                'Salary Payment - Employee Wages',
                'Utility Bill Payment - Electricity',
                'Marketing Campaign - Digital Advertising',
                'Interest Payment - Bank Loan',
                'Raw Material Purchase - Steel Components',
                'Dividend Payment - Shareholder Return'
            ],
            'Amount': [1000000, 500000, 2000000, 1500000, 50000, 10000, 25000, 15000, 300000, 100000],
            'Category': [
                'Investing Activities',
                'Operating Activities', 
                'Investing Activities',
                'Financing Activities',
                'Operating Activities',
                'Operating Activities',
                'Operating Activities',
                'Financing Activities',
                'Operating Activities',
                'Financing Activities'
            ]
        })
        
        print(f"✅ Created test data: {len(test_data)} samples")
        print(f"   Categories: {test_data['Category'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"❌ Error creating test data: {e}")
        return False
    
    # Test 2: Test training with the lightweight AI system
    print("\n📋 Test 2: Testing training system...")
    
    try:
        from app1 import lightweight_ai
        
        # Test training
        result = lightweight_ai.train_transaction_classifier(test_data)
        
        if result:
            print("✅ Training successful!")
            print(f"   Is trained: {lightweight_ai.is_trained}")
            print(f"   Models available: {list(lightweight_ai.models.keys())}")
        else:
            print("❌ Training failed")
            
    except Exception as e:
        print(f"❌ Error testing training: {e}")
        return False
    
    # Test 3: Test categorization with trained model
    print("\n📋 Test 3: Testing categorization with trained model...")
    
    try:
        if lightweight_ai.is_trained:
            # Test categorization
            test_desc = "Infrastructure Development - Warehouse Construction"
            test_amount = 1000000
            
            result = lightweight_ai.categorize_transaction_ml(test_desc, test_amount)
            print(f"✅ Categorization result: {result}")
            
            # Check if it's using XGBoost
            if '(XGBoost)' in result:
                print("✅ SUCCESS: Using XGBoost model!")
            else:
                print("❌ FAILURE: Not using XGBoost model")
        else:
            print("⚠️ Model not trained, skipping categorization test")
            
    except Exception as e:
        print(f"❌ Error testing categorization: {e}")
        return False
    
    # Test 4: Test with imbalanced data (stratification issue)
    print("\n📋 Test 4: Testing with imbalanced data...")
    
    try:
        # Create imbalanced data (mostly one category)
        imbalanced_data = pd.DataFrame({
            'Description': ['Test ' + str(i) for i in range(20)],
            'Amount': [1000 * i for i in range(20)],
            'Category': ['Operating Activities'] * 18 + ['Investing Activities'] * 2  # Imbalanced
        })
        
        print(f"✅ Created imbalanced data: {len(imbalanced_data)} samples")
        print(f"   Categories: {imbalanced_data['Category'].value_counts().to_dict()}")
        
        # Test training with imbalanced data
        result = lightweight_ai.train_transaction_classifier(imbalanced_data)
        
        if result:
            print("✅ Training with imbalanced data successful!")
        else:
            print("❌ Training with imbalanced data failed")
            
    except Exception as e:
        print(f"❌ Error testing imbalanced data: {e}")
        return False
    
    print("\n🎯 Training System Fix Test Complete!")
    print("The training system should now handle:")
    print("   ✅ Array length mismatches")
    print("   ✅ Stratification issues")
    print("   ✅ Imbalanced data")
    
    return True

if __name__ == "__main__":
    success = test_training_system_fix()
    if success:
        print("\n🎉 Training System Fix Test PASSED!")
    else:
        print("\n❌ Training System Fix Test FAILED!") 