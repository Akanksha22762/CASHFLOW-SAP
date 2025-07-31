#!/usr/bin/env python3
"""
FIX XGBOOST TRAINING PROPERLY - Actually train the XGBoost model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fix_xgboost_training_properly():
    """Actually train the XGBoost model properly"""
    print("🔧 FIXING XGBOOST TRAINING PROPERLY...")
    
    try:
        from app1 import lightweight_ai
        
        # Create simple, balanced training data
        training_data = pd.DataFrame({
            'Description': [
                # Investing Activities (5 samples)
                'Infrastructure Development - Warehouse Construction',
                'Equipment Purchase - Rolling Mill Upgrade',
                'Property Acquisition - Office Building',
                'Machinery Purchase - Production Line',
                'Facility Expansion - New Wing',
                
                # Operating Activities (5 samples)
                'VIP Customer Payment - Construction Company',
                'Salary Payment - Employee Payroll',
                'Utility Payment - Electricity Bill',
                'Tax Payment - Income Tax',
                'Insurance Premium - Business Insurance',
                
                # Financing Activities (5 samples)
                'Investment Liquidation - Mutual Fund Units',
                'Loan Repayment - Bank Loan',
                'Dividend Payment - Shareholder Return',
                'Bond Issuance - Corporate Bonds',
                'Equity Investment - Startup Funding'
            ],
            'Amount': [
                # Investing Activities
                3709289.81, 1500000, 2500000, 3000000, 1800000,
                # Operating Activities
                2000000, 800000, 50000, 200000, 75000,
                # Financing Activities
                500000, 300000, 100000, 1500000, 250000
            ],
            'Category': [
                # Investing Activities
                'Investing Activities', 'Investing Activities', 'Investing Activities',
                'Investing Activities', 'Investing Activities',
                # Operating Activities
                'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities',
                # Financing Activities
                'Financing Activities', 'Financing Activities', 'Financing Activities',
                'Financing Activities', 'Financing Activities'
            ],
            'Date': [datetime.now() - timedelta(days=i) for i in range(15)],
            'Type': ['Credit'] * 15
        })
        
        print(f"📊 Training data created: {len(training_data)} samples")
        print(f"📊 Categories: {training_data['Category'].value_counts().to_dict()}")
        
        # Verify we have enough samples per class
        category_counts = training_data['Category'].value_counts()
        min_samples = category_counts.min()
        print(f"📊 Minimum samples per class: {min_samples}")
        
        if min_samples < 2:
            print("❌ Not enough samples per class for stratification")
            return False
        
        # Train the model
        print("🤖 Training XGBoost model...")
        success = lightweight_ai.train_transaction_classifier(training_data)
        
        if success:
            print("✅ XGBoost training successful!")
            print(f"✅ Model trained: {lightweight_ai.is_trained}")
            print(f"✅ Features: {len(lightweight_ai.feature_names)}")
            
            # Test the trained model
            test_cases = [
                ("Infrastructure Development", 1000000),
                ("Customer Payment", 2000000),
                ("Investment Liquidation", 500000)
            ]
            
            print("\n🧪 TESTING TRAINED MODEL:")
            for desc, amount in test_cases:
                result = lightweight_ai.categorize_transaction_ml(desc, amount)
                print(f"   {desc} → {result}")
                
                # Check if result contains "Not-Trained"
                if "Not-Trained" in result:
                    print(f"   ❌ Model still not trained")
                else:
                    print(f"   ✅ Model working properly")
            
            return True
        else:
            print("❌ XGBoost training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing XGBoost: {e}")
        return False

def verify_fix():
    """Verify the fix worked"""
    print("\n🔍 VERIFYING FIX...")
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
        
        print("\n🧪 FINAL TESTING:")
        all_working = True
        for desc, amount in test_cases:
            result = lightweight_ai.categorize_transaction_ml(desc, amount)
            print(f"   {desc} → {result}")
            
            if "Not-Trained" in result:
                print(f"   ❌ Still not trained")
                all_working = False
            else:
                print(f"   ✅ Working properly")
        
        return all_working
        
    except Exception as e:
        print(f"❌ Error verifying fix: {e}")
        return False

if __name__ == "__main__":
    print("🚀 FIX XGBOOST TRAINING PROPERLY")
    print("=" * 50)
    
    # Apply the fix
    fix_successful = fix_xgboost_training_properly()
    
    if fix_successful:
        print("\n✅ Running verification...")
        verification_successful = verify_fix()
        
        if verification_successful:
            print("\n🎉 XGBOOST TRAINING FIXED SUCCESSFULLY!")
        else:
            print("\n⚠️ Fix applied but verification failed")
    else:
        print("\n❌ Fix failed") 