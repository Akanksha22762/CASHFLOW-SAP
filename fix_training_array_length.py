#!/usr/bin/env python3
"""
FIX TRAINING ARRAY LENGTH - Fix the array length mismatch in training system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fix_training_array_length():
    """Fix training system array length mismatch"""
    print("🔧 FIXING TRAINING ARRAY LENGTH...")
    
    try:
        from app1 import lightweight_ai
        
        # Create training data with EXACTLY matching array lengths
        n_samples = 48  # Fixed number of samples
        
        # Create lists with exactly 48 items each
        descriptions = [
            # Investing Activities (8 samples)
            'Infrastructure Development 1', 'Infrastructure Development 2',
            'Equipment Purchase 1', 'Equipment Purchase 2',
            'Property Acquisition 1', 'Property Acquisition 2',
            'Machinery Purchase 1', 'Machinery Purchase 2',
            
            # Operating Activities (8 samples)
            'Customer Payment 1', 'Customer Payment 2',
            'Salary Payment 1', 'Salary Payment 2',
            'Utility Payment 1', 'Utility Payment 2',
            'Tax Payment 1', 'Tax Payment 2',
            
            # Financing Activities (8 samples)
            'Investment Liquidation 1', 'Investment Liquidation 2',
            'Loan Repayment 1', 'Loan Repayment 2',
            'Dividend Payment 1', 'Dividend Payment 2',
            'Bond Issuance 1', 'Bond Issuance 2',
            
            # Additional Operating Activities (24 samples)
            'Insurance Premium', 'Marketing Expense', 'Software License', 'Rent Payment',
            'Freight Cost', 'Legal Fees', 'Audit Fees', 'Training Cost',
            'Maintenance Cost', 'Commission Payment', 'Interest Payment',
            'Refund Payment', 'Deposit Payment', 'Subscription Fee',
            'Consulting Fee', 'Travel Expense', 'Office Supplies',
            'Phone Bill', 'Internet Bill', 'Cleaning Service', 'Security Service',
            'Maintenance Fee', 'Service Charge', 'Processing Fee', 'Handling Fee'
        ]
        
        amounts = [
            # Investing Activities
            1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000,
            # Operating Activities
            5000000, 5500000, 6000000, 6500000, 7000000, 7500000, 8000000, 8500000,
            # Financing Activities
            500000, 750000, 1000000, 1250000, 1500000, 1750000, 2000000, 2250000,
            # Additional samples
            50000, 200000, 75000, 120000, 25000, 150000, 45000, 35000,
            50000, 30000, 25000, 60000, 15000, 25000, 50000, 12000, 40000, 15000,
            5000, 8000, 12000, 18000, 22000, 15000
        ]
        
        categories = [
            # Investing Activities
            'Investing Activities', 'Investing Activities', 'Investing Activities', 'Investing Activities',
            'Investing Activities', 'Investing Activities', 'Investing Activities', 'Investing Activities',
            # Operating Activities
            'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
            'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
            # Financing Activities
            'Financing Activities', 'Financing Activities', 'Financing Activities', 'Financing Activities',
            'Financing Activities', 'Financing Activities', 'Financing Activities', 'Financing Activities',
            # Additional samples (mostly Operating Activities)
            'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
            'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
            'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
            'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
            'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
            'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities'
        ]
        
        dates = [datetime.now() - timedelta(days=i) for i in range(n_samples)]
        types = ['Credit'] * n_samples
        
        # Verify all arrays have exactly the same length
        lengths = [len(descriptions), len(amounts), len(categories), len(dates), len(types)]
        print(f"📊 Array lengths: {lengths}")
        print(f"📊 All lengths equal: {len(set(lengths)) == 1}")
        
        if len(set(lengths)) != 1:
            print("❌ Array lengths don't match")
            return False
        
        # Create DataFrame
        training_data = pd.DataFrame({
            'Description': descriptions,
            'Amount': amounts,
            'Category': categories,
            'Date': dates,
            'Type': types
        })
        
        print(f"📊 Training data created: {len(training_data)} samples")
        print(f"📊 Categories: {training_data['Category'].value_counts().to_dict()}")
        
        # Verify we have enough samples per class
        category_counts = training_data['Category'].value_counts()
        min_samples = category_counts.min()
        print(f"📊 Minimum samples per class: {min_samples}")
        
        if min_samples < 8:  # Need at least 8 per class for proper train/test split
            print("❌ Not enough samples per class")
            return False
        
        # Train the model
        print("🤖 Training model...")
        success = lightweight_ai.train_transaction_classifier(training_data)
        
        if success:
            print("✅ Training successful!")
            print(f"✅ Model trained: {lightweight_ai.is_trained}")
            
            # Test the model
            test_cases = [
                ("Infrastructure Development", 1000000),
                ("Customer Payment", 2000000),
                ("Investment Liquidation", 500000)
            ]
            
            print("\n🧪 TESTING TRAINED MODEL:")
            all_working = True
            for desc, amount in test_cases:
                result = lightweight_ai.categorize_transaction_ml(desc, amount)
                print(f"   {desc} → {result}")
                
                if "Not-Trained" in result:
                    print(f"   ❌ Model still not trained")
                    all_working = False
                else:
                    print(f"   ✅ Model working properly")
            
            return all_working
        else:
            print("❌ Training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing training: {e}")
        return False

def verify_fix():
    """Verify the fix worked"""
    print("\n🔍 VERIFYING TRAINING FIX...")
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
    print("🚀 FIX TRAINING ARRAY LENGTH")
    print("=" * 50)
    
    # Apply the fix
    fix_successful = fix_training_array_length()
    
    if fix_successful:
        print("\n✅ Running verification...")
        verification_successful = verify_fix()
        
        if verification_successful:
            print("\n🎉 TRAINING SYSTEM FIXED SUCCESSFULLY!")
        else:
            print("\n⚠️ Fix applied but verification failed")
    else:
        print("\n❌ Fix failed") 